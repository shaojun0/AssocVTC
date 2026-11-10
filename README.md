# vlm8b — AssocVTC 448²/p32 阶段2编码器 → 8 预算并行 → Qwen3.5-4B 全参微调

> **状态（2026-11-09）：run1 / run2 / run3 已在 2×A800 上跑完**（run3 = 3.0 epoch /
> 11,463 步 / `mix_train_bal2.jsonl` 122,259 行，见 §7）。本分支现在包含
> **核心训练/评测代码（root）** + **`tools/`（评测、逐样本筛查、造数、启动与运维脚本，非训练必需）**；
> 文档/报告在主线 `main`，例如
> [`doc/2026-11-09/REPORT_budget_screen.md`](https://github.com/shaojun0/AssocVTC/blob/main/doc/2026-11-09/REPORT_budget_screen.md)。

把 AssocVTC 阶段2 的 `z_s (32×1744)` 投成 **8 个 2560 维 visual token**；文本 prompt
先编码（跨预算共享、**只前向一次**），答案段按预算 k 只看前 k 个 visual token
（「先看文字、后看图」），8 条答案段的 CE 取（加权）平均。
3 epoch：全量微调（encoder 不冻结）；DeepSpeed ZeRO-3。

设计文档：[`doc/2026-11-05/DESIGN_8budget_vlm_finetune.md`](https://github.com/shaojun0/AssocVTC/blob/main/doc/2026-11-05/DESIGN_8budget_vlm_finetune.md)
（§3 架构 / §4 8 行批 / §5 协议 / §9 评测；本分支只放代码，文档在主线仓库）。

## 1. 架构

```
image (B,3,448,448)          data.py: train RRC(448,0.35–1.0)+HFlip / val Resize(512)+CenterCrop
   │                                  Normalize = ImageNet mean/std
   ▼
AssocVTCEncoder              encoder.py（vendored）: 448²/p32, depth 13, K=32, enc_dim=1744
   │  forward(pixel_values) -> (z_cls, z_s)，只用 z_s
   ▼
z_s (B,32,1744)
   ▼  BudgetProjector: LayerNorm(1744) → Linear(1744,640)（逐 token）
tokens (B,32,640) → reshape (B,8,4×640=2560) → LayerNorm(2560)
   ▼
vis (B,8,2560)               与 Qwen LM hidden 同维
   ▼  文本 prompt 只前向一次（跨预算共享 KV cache）：prompt = text_embeds[:, :P-1]
prompt cache (B,P-1,·)       预算与 k 无关 ⇒ 预算只影响答案段
   ▼  每图 V 行（默认 k=1..8）：段 = [prompt 末 token（锚点）; vis[:, :k] ; answer]
rows.segment (B·V, 1+8+A, 2560)  段内 right-padding mask；labels 除 answer 外 -100
   ▼  Qwen3.5-4B 文本主干（只喂 inputs_embeds；不喂 pixel_values；vision tower 已删）
hidden (B·V, 1+8+A, 2560)    只 gather 段内**预测位置** `k+j` → lm_head（分块）
   ▼
loss = Σ_k w_k · CE_k        w_k = 1/V（equal）或 (1/k)/Σ(1/j)（inv-k）
```

### 张量形状

| 阶段 | 形状 | 说明 |
|---|---|---|
| `pixel_values` | `(B,3,448,448)` | train RRC+HFlip / val Resize(512)+CenterCrop |
| `z_s` | `(B,32,1744)` | `StandardAttentionEncoder` 的 specials 段（K=32） |
| `tokens` | `(B,32,640)` | `LayerNorm(1744) → Linear(1744,640)`，逐 token |
| `vis` | `(B,8,2560)` | `reshape(B,8,4*640) → LayerNorm(2560)` |
| `text_embeds` | `(B,T,2560)` | `lm.get_input_embeddings()(input_ids)` |
| `rows.prompt` | `(B,P−1,2560)` | 跨预算共享的前缀（末 token 进段做锚点），**只前向一次** |
| `rows.segment` | `(B·V, 1+8+A, 2560)` | 段 = `[prompt 末 token; vis[:,:k]; answer]`，右侧补零 |
| `rows.prompt_mask` / `rows.segment_mask` | `(B,P−1)` / `(B·V,1+8+A)` | 两段各自的 2D mask；prompt 右 pad、段尾 pad 均为 0 |
| `rows.position_ids` | `(B·V, 1+8+A)` | 段的位置，从**本样本** prompt 长度 − 1 开始续接 |
| `rows.ans_pos` | `(B·V, A)` | **段内预测位置** `k + j`（k = `row_ks`），无效位 `-1` |
| `rows.row_ks` | `(B·V,)` | 每行实际视觉 token 数（无图样本 0） |
| `loss` | 标量 | `Σ_k w_k·CE_k` |

`T` = 文本总长度（prompt+answer；nothink ≈ 112，think ≈ 578）；`P` = 批内最大 prompt 长度；
`A` = answer 长度（nothink 4）；`V` = 预算数（默认 8）；`A` 轴按批内最长 answer 对齐。

### 2. 三条关键裁决（§4）

- **8 行批，不打包**：Qwen3.5 是 hybrid（24 层 gated-delta 线性注意力 + 8 层全注意力）；
  线性层收到 4D 自定义 mask 时 `create_recurrent_attention_mask()` 返回 `None` ⇒ 打包会
  跨段泄露。本实现只用 **2D right-padding mask**：prompt 段一次 prefill（KV cache），
  再按预算展开成 `B·V` 条独立短段，行与行是独立 batch 行。
- **防泄露 = 截断 + padding mask**：段 k 只含 `vis[:, :k]`，**不用 mask 遮 visual**。
  `smoke.py` 验证：改 `z_s[4]`（第 2 个 visual token）不影响 k=1、影响 k=2。
- **`position_ids` 显式传**：prompt 段用 `arange(P)`；每条段从「本样本 prompt 长度 − 1」
  开始续接（锚点 + visual token + answer 各占一个位置），所以 `k=0`（文本任务）时答案
  0 的预测位置就是段首锚点。

**shift 公式一行**：取 hidden 的**段内**位置 = `k + j`（k = 该行视觉 token 数，j = 答案内
偏移），targets = `labels[answer 起始 + j]`（`answer 起始 >= 2` 断言，因为要留 prompt 末
token 做锚点）；**绝不能**用 label 自身位置取 hidden（hidden 已看见该 token 的 embedding
⇒ 目标泄露，loss 可被"复制输入"压到 0）。`smoke.py` 的 shift 检查把 prompt+段拼成一条
序列跑同一次前向，与 `forward_rows()["row_ce"]` 差 ≤ 1e-5；把 `ans_pos` 改成 label 位置
（+1）必须改变 CE。

### 3. 训练协议（§5）

| 项 | 值 | 位置 |
|---|---|---|
| 优化器 | AdamW β=0.9/0.95 eps=1e-8 | `VLM8BudgetTrainer.create_optimizer` |
| lr | LLM 1e-5 / proj 1e-4 / encoder 5e-6（E2+） | `--lr-llm/--lr-proj/--lr-encoder` |
| wd | LLM 0.1 / 其余 0.05；bias+LayerNorm 不衰减 | `build_param_groups` |
| 调度 | cosine，warmup 3%，grad clip 1.0 | `TrainingArguments` |
| 批次 | micro 2 img/GPU（16 行）× accum 8 × 2 GPU | `--micro-batch-size/--grad-accum` |
| 精度 / ckpt | bf16 + ZeRO-2；**不用 `--grad-ckpt`** | `ds_zero2.json`（ZeRO-3 见 §6.1：与本 encoder 不兼容） / `load_lm`（梯度检查点会强制 `use_cache=False`，与两段式前向不兼容） |
| 日志 | 每 20 步 per-budget CE → `per_budget_ce.jsonl` | `BudgetCELogCallback` |
| eval | **Trainer 内不做**（`eval_strategy="no"`，不传 eval_dataset） | 评测走 `eval.py` |

| epoch | encoder | 说明 |
|---|---|---|
| E1–E3 | 🔥 全量微调 | encoder 不冻结，lr 5e-6；投影头 1e-4 / LLM 1e-5 |

### 4. 跑法

```bash
# 本机（CPU）：只需 torch + transformers + torchvision + Pillow
python3 -m py_compile *.py
python3 smoke.py                        # 退出码 0 = PASS

# remote（2×A800，venv /root/venv_qwen）：先冒烟再长跑
torchrun --nproc_per_node=2 train.py --smoke \
    --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl
torchrun --nproc_per_node=2 train.py \
    --output-dir /root/autodl-tmp/vlm8b/output/run1 \
    --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl

# 评测：per-budget CE + greedy exact-match（输出 eval.json / eval.md）
python3 eval.py --checkpoint /root/autodl-tmp/vlm8b/output/run1/final \
    --jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/val_nothink_all.jsonl \
    --limit 1000

# 标准 HF 加载（在仓库根目录运行；ckpt 里只拷了 model.py，encoder 从本仓 import）
python3 -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('/root/autodl-tmp/vlm8b/output/run1/final', trust_remote_code=True)"
```

默认路径（可用 CLI 覆盖）：阶段2 ckpt `/root/autodl-tmp/exp448/B/ckpt_best.pt`、
Qwen `/root/qwen_models/Qwen3.5-4B`、图片根 `/root/autodl-tmp/datasets/imagenet100`。
单卡调试：`--deepspeed ""`。

### 5. 文件 / 设计 §13 映射

| 设计 §13 | 本实现 | 关键内容 |
|---|---|---|
| `model_wrap.py` | `model.py` | `AssocVTCEncoder` / `BudgetProjector` / `VLM8Budget` |
| `model_v2_linattn.py`（主线） | `encoder.py` | `StandardAttentionEncoder`（逐字 vendored，main@`1fc4920`） |
| `data_8budget.py` | `data.py` | `SelfDistillSFTDataset` / `EightBudgetCollator` / `build_image_transform` |
| `train_8budget.py` | `train.py` | `VLM8BudgetTrainer` / `BudgetCELogCallback` |
| `ds_zero3.json` | `ds_zero3.json` | ZeRO-3 + bf16 + gather-on-save |
| `smoke_mask.py` | `smoke.py` | 22 项自检（形状 / ans_pos / shift 等价 / 截断隔离×2 / 行独立 / pad 隔离 / 文本行零视觉 token / 纯文本 batch / prompt_lens / HF 加载契约 / jinja 模板 / 静态防回退） |
| `eval_8budget.py` | `eval.py` | per-k CE + greedy + json/md |
| `report_8budget.py` | 未做（v1） | `eval.md` 直接出表；画图留给报告脚本 |

最小可跑集合是 root 下的 11 个文件（`model.py` / `encoder.py` / `data.py` / `train.py` /
`eval.py` / `smoke.py` / `freeze.py` / `musa.py` / `ds_zero2.json` / `requirements.txt` /
`README.md`）；复制这些即可在 remote 跑训练。`MUSA_FREEZE.md` / `test_freeze.py` /
`musa_e2e.py` 是冻结机制与国产适配的说明与自检（非训练必需）。`tools/` 是后来补的
**非训练必需**脚本（见 §7.2），单独复制即可单独跑。

### 6. review 要点 / 已知限制

- **P0 shift**：`expand_budget_rows()` 里 `ans_pos = k + j`（段内预测位置 = 该行视觉
  token 数 + 答案内偏移）；改动此处必跑 `smoke.py`（shift 等价 + 防泄露）。临时改成
  label 自身位置（`+1`）会让防泄露检查 FAIL。
- **两段式前向与 `--grad-ckpt` 不兼容**：prompt 只前向一次靠的是 KV cache，梯度检查点
  会强制 `use_cache=False` ⇒ 第二段拿不到 cache。`train.py` 已 fail-fast；省显存请调
  micro-batch / ZeRO-2，别开检查点。
- **encoder 冻结**：见 §6.1；原来的 `EncoderFreezeCallback` / `_keep_encoder_eval()` 已被
  `freeze.py` 里的通用实现取代（`--freeze-encoder` / `--freeze-encoder-epochs`）。
- **`encoder.py` 是 vendored 副本**：主线 `model_v2_linattn.py`（commit `1fc4920`）逐字抽取；
  上游若改类名 / 子模块名，阶段2 ckpt 的 `strict=True` 加载会失败，需同步本文件。
- **未验证项**：`Qwen3_5ForConditionalGeneration.from_pretrained(dtype=bf16)`、`lm.model`
  取 hidden、`del lm.model.visual`、逐行 `generate(inputs_embeds=...)`、ZeRO-3 + 4B 显存
  —— 全部依赖 remote（transformers 5.22.1）；本机只保证 import / CLI / 张量逻辑。
- **评测权重**：`eval.py` 只读 safetensors 分片索引（Trainer 默认 `save_safetensors=True`）。
- **标准 HF 加载**：`AutoModel.from_pretrained(ckpt, trust_remote_code=True)` 可用——config 带
  `auto_map`，`save_pretrained` 会把 `model.py` 拷进 ckpt；v5 在 meta 设备上初始化模型，故
  `__init__` 只搭 Qwen 结构（权重由 ckpt 灌入），训练路径仍走 `load_lm` 读预训练权重。
  限制：需在仓库根目录运行（`encoder` 从本仓 import）、config 里 `encoder_ckpt`/`qwen_dir`
  指向的路径在本机有效（ckpt 不自包含这两个子模型）；加载按 `config.dtype=float32` 上浮。
  已在 transformers 5.22.2 + 微型 Qwen3.5/encoder 资产上端到端验证 loss/CE 与手工加载完全一致。
- **纯文本 chat / batch**：`forward(input_ids=..., pixel_values=None)` 等价于 `has_image`
  全 False（k=0），会跳过 encoder、答案段里只有锚点 + 答案；纯文本**生成**用
  `model.lm.generate(...)`（`tools/eval_mix/bench_general_vlm8b.py:run_batch` 是范例，
  不拼视觉 token）。参数顺序为 `(input_ids, pixel_values=None, ...)`，与 HF VLM 一致。
- **数据**：`train_nothink_all.jsonl` 的 `image` 是相对路径；`--image-root` 必须对。

### 6.1 encoder 冻结 × 国产适配（MUSA）

完整机制与实测数据见 [`MUSA_FREEZE.md`](MUSA_FREEZE.md)，这里只放操作口径：

| 目的 | 参数 | 语义 |
|---|---|---|
| 全程冻死 encoder | `--freeze-encoder all` | 冻结在 Trainer 之前生效；可训练参数才进 optimizer，**与 ckpt 逐位一致（Δ=0.0）** |
| 只训后段 | `--freeze-encoder last=2` | 冻 stem + blocks[0:-2]，只训最后 2 个 block + norm |
| 只训 blocks | `--freeze-encoder stem` | 冻 patch_embed/cls/special/pos |
| 冻前 N 层 | `--freeze-encoder blocks=0-8` | 含端点；逗号组合 `blocks=0-8,norm` |
| E1 冻、E2 起解冻 | `--freeze-encoder all --freeze-encoder-epochs 1` | **仅 DeepSpeed**；切换在 engine 建好后，解冻前有 ≤ lr/步 的动量残留 |

硬约束（都是实测，不是推测）：

- **ZeRO-3 不能用**：encoder 的 `cls_token`/`special_tokens`/`pos_embed` 是裸 Parameter，
  ZeRO-3 不为其 all-gather，**不冻结也会崩**；`train.py` 对 `stage>=3` fail-fast。
  用默认 `ds_zero2.json`。
- **按 epoch 冻结别提前冻**：初始化前整组冻结 → ZeRO-2 `IndexError`、ZeRO-3 则
  「解冻后静默不更新」；guard 会拦住。
- 切换冻结状态后要清梯度（callback 已做），否则 ZeRO 用上一轮 stale grad 再推一步。

国产适配收敛在 `musa.py`（`train.py:main()` 首行 `musa.bootstrap()`）：MCCL 通信后端、
fp64 范数探测降级、奇数 head_dim 的 SDPA 走 math、deepspeed 的 dist-info。自检：

```bash
python3 smoke.py            # 原 22 项张量逻辑
python3 test_freeze.py      # 冻结语义 27 项（CPU）
python3 test_freeze_compat.py  # 兼容性契约 18 项（不改 state_dict / 默认路径等价 / 先冻后载）
torchrun --nproc_per_node=2 musa_e2e.py --stage 2   # 真实 Trainer+DS 7 项
torchrun --nproc_per_node=2 musa_e2e.py --stage 3   # ZeRO-3 fail-fast 2 项
```

## 7. run1–run3 实测记录（2026-11-09）

### 7.1 三次续训

| run | 起点 | 训练集 | steps / epoch | 结果要点 |
|---|---|---|---|---|
| run1 | 阶段2 encoder + Qwen3.5-4B | ImageNet-100 自蒸馏 66k 左右 | — | in100 val k8 12.0% |
| run2 | run1/final | 同 run1（续训 2 epoch） | — | in100 val k8 28.8%（旧协议）；loss 平台期回升（续训 LR 重新 warmup 到 1e-5），后自行收敛 |
| run3 | run2/final | `mix_train_bal2.jsonl` 122,259（文本 82k + 图像 40.3k，in100 锚定 3k） | 3,821 / 11,463（3.0ep） | in100 val k8 **32.4%**（修正协议；旧协议 22.8%）；mix_heldout 分源见 main 的报告 |

三次都用了 **8 预算并行**（`k=1..8` 同一张图同一段文本，answer-only CE）与 ZeRO-2/3。

### 7.2 `tools/` 布局

| 文件 | 作用 |
|---|---|
| `tools/eval_mix/build_img_eval_mix.py`、`prepare_mix.py`、`rebalance_mix.py`、`rebuild_mix2.py`、`expand_*.py`、`preflight_mix.py` | 评测对齐混合集的造数/合并/重采样/预检（`build_img_eval_mix` 需要数据集，`preflight` 纯 CPU） |
| `tools/eval_mix/eval_mix_breakdown.py`、`bench_general_vlm8b.py`、`make_report.py` | 分任务表、通用文本能力 bench、结果拼报告 |
| `tools/eval_mix/eval_dump_per_sample.py` | **逐样本筛查**：每行 × 每档 k 的 CE（normal/null/shuffle/mean/repeat/pad1）+ k∈{1,8} greedy + 视觉 token 冗余度 |
| `tools/eval_mix/analyze_screen.py` | 读逐行 dump → 分源表、k 曲线、四象限、分箱、Spearman（纯 CPU，无需 GPU） |
| `tools/eval_mix/vis_probe.py` | 表征离散度 + full-attention 层「文本 query→视觉 token」注意力份额 |
| `tools/run_eval_mix.sh`、`run_main_vlm8b.sh`、`run_curriculum_vlm8b.sh`、`run_cont2.sh`、`run_cont3.sh`、`restart_lr.sh` | 训练启动/续训 |
| `tools/eval_after_run2.sh`、`eval_after_run3.sh`、`stop_after_epoch3.sh`、`screen_gpu0.sh`、`screen_gpu1.sh` | 自动评测链、按时收尾、筛查任务 |

小显存机器上做**代码测试**的最小路径：`python smoke.py`（纯 CPU，张量逻辑）+ `python eval.py
--checkpoint <小模型/假权重> --limit 8`；`eval_dump_per_sample.py` / `vis_probe.py` 需要真 GPU 与 4B 权重。

### 7.3 评测协议修复（P0，`eval.py`）

**旧实现（bug）**：`generate_budget()` 把 `input_ids`（含 answer）整段拼进 `inputs_embeds` 后调用
`generate`，模型看到答案后继续生成 ⇒ 大量空串，exact-match acc 被严重低估（300 行：22.3% vs 修正后 49.3%）。
**修正**：只喂到 answer 之前（`prompt_lens(labels)` = answer 首 token 位置 = prompt 长度）；
`--gen-protocol`/`full` 分支与 `model.generate_budget()` 已删除，全仓只剩 prompt-only 一条生成路径。
CE（teacher forcing）不受影响。

### 7.4 事故 A 修复（文本任务零视觉 token）

`data.py:is_text_only()` 用 `img/_blank/` 占位路径判定无图样本 ⇒ `has_image=False`；
`expand_budget_rows(..., has_image=)` 对无图样本的**所有预算行**都取 0 个视觉 token，
`eval.py`/`eval_dump_per_sample.py`/`eval_mix_breakdown.py` 的生成路径同样按 `has_image`
取 `k=0`（生成布局 `[prompt][vis[:k]]`，先文本后视觉、k 只影响 prompt 之后的视觉段）。
注意 `--visual-mode null` 只是把 8 个 emb 置零、**token 数不变，不等于没有图**。

### 7.5 预算机制筛查结论（代码侧要点）

- `tools/eval_mix/eval_dump_per_sample.py` 的 5 种视觉消融可直接判定「token 是否冗余」：
  `mean`（8 token 取均值）/ `repeat`（第 1 个 token 复制 8 份）与 `normal` 的 CE 差 ≈0 ⇒ 冗余。
- 结论：预算收益只在「图真被用上」的行成立，且 k 不增加信息（8 个 visual token = 32 个 z token）；
  细节任务（OCR/图表/文档）需要**编码器侧**扩容而不是加 k。详见 main 的
  `doc/2026-11-09/REPORT_budget_screen.md`。
