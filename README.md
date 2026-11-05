# vlm8b — AssocVTC 448²/p32 阶段2编码器 → 8 预算并行 → Qwen3.5-4B 全参微调

> **状态（2026-11-05）：仅本机逻辑自检通过** —— `python3 smoke.py` 在
> torch 2.8 CPU 上 **7 项 PASS / 0 FAIL**（含 shift 等价，tol 1e-5）。**尚未在 GPU /
> Qwen3.5-4B / ZeRO-3 上跑过**（本机无 transformers 5.22、无 4B 权重）；首次真机请先 `--smoke`。

把 AssocVTC 阶段2 的 `z_s (32×1744)` 投成 **8 个 2560 维 visual token**；同一张图一次
前向展开 **8 行**（行 k 只用前 k 个 token）去预测同一段文本，8 行 CE 取（加权）平均。
3 epoch：E1 冻结编码器，E2–E3 全解冻；DeepSpeed ZeRO-3。

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
   ▼  每图 V 行（默认 k=1..8）：行 k = [vis[:, :k] ; text_embeds]，right-pad 到 max_k+T
rows (B·V, 8+T, 2560)        2D right-padding mask；labels 除 answer 外 -100
   ▼  Qwen3.5-4B 文本主干（只喂 inputs_embeds；不喂 pixel_values；vision tower 已删）
hidden (B·V, 8+T, 2560)      只 gather answer 的**预测位置** → lm_head（分块）
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
| `rows.embeds` | `(B·V, 8+T, 2560)` | 行 k = `[vis[:,:k]; text]`，右侧补 `8-k` 个零 |
| `rows.ans_pos` | `(B·V, A)` | **预测位置** `k + (label_pos − 1)`，无效位 `-1` |
| `loss` | 标量 | `Σ_k w_k·CE_k` |

`T` = 文本长度（nothink ≈ 112，think ≈ 578）；`A` = answer 长度（nothink 4）；
`V` = 预算数（默认 8）；`A` 轴按批内最长 answer 对齐。

### 2. 三条关键裁决（§4）

- **8 行批，不打包**：Qwen3.5 是 hybrid（24 层 gated-delta 线性注意力 + 8 层全注意力）；
  线性层收到 4D 自定义 mask 时 `create_recurrent_attention_mask()` 返回 `None` ⇒ 打包会
  跨段泄露。本实现只用 **2D right-padding mask**，行与行是独立 batch 行。
- **防泄露 = 截断 + padding mask**：行 k 只含 `vis[:, :k]`，**不用 mask 遮 visual**。
  `smoke.py` 验证：改 `z_s[4]`（第 2 个 visual token）不影响 k=1、影响 k=2。
- **`position_ids` 不传**：Qwen3.5 文本 forward 默认 `arange(0..L-1)`，与 mask 无关。

**shift 公式一行**：取 hidden 的位置 = `k + (label_pos − 1)`，targets = `labels[label_pos]`，
`label_pos = answer 起始 + j`（`start >= 1` 断言）；**绝不能**用 label 自身位置取 hidden
（hidden 已看见该 token 的 embedding ⇒ 目标泄露，loss 可被"复制输入"压到 0）。
`smoke.py` 的 shift 检查用**非循环**参考：`F.cross_entropy(logits[:, :-1], labels[:, 1:])`
按行 mean，与 `forward_rows()["row_ce"]` 差 ≤ 1e-5。

### 3. 训练协议（§5）

| 项 | 值 | 位置 |
|---|---|---|
| 优化器 | AdamW β=0.9/0.95 eps=1e-8 | `VLM8BudgetTrainer.create_optimizer` |
| lr | LLM 1e-5 / proj 1e-4 / encoder 5e-6（E2+） | `--lr-llm/--lr-proj/--lr-encoder` |
| wd | LLM 0.1 / 其余 0.05；bias+LayerNorm 不衰减 | `build_param_groups` |
| 调度 | cosine，warmup 3%，grad clip 1.0 | `TrainingArguments` |
| 批次 | micro 2 img/GPU（16 行）× accum 8 × 2 GPU | `--micro-batch-size/--grad-accum` |
| 精度 / ckpt | bf16 + grad ckpt（`use_reentrant=False`） | `ds_zero3.json` / `load_lm` |
| 日志 | 每 20 步 per-budget CE → `per_budget_ce.jsonl` | `BudgetCELogCallback` |
| eval | **Trainer 内不做**（`eval_strategy="no"`，不传 eval_dataset） | 评测走 `eval.py` |

| epoch | encoder | 说明 |
|---|---|---|
| E1 | ❄️ `requires_grad=False` + `eval()` | 只训 LLM + projector；`forward` 入口幂等兜底 eval |
| E2–E3 | 🔥 全解冻 | encoder lr 5e-6，防表示崩塌 |

### 4. 跑法

```bash
# 本机（CPU）：只需 torch + transformers + torchvision + Pillow
python3 -m py_compile *.py
python3 smoke.py                        # 退出码 0 = PASS（7 项）

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
| `train_8budget.py` | `train.py` | `VLM8BudgetTrainer` / `EncoderFreezeCallback` / `BudgetCELogCallback` |
| `ds_zero3.json` | `ds_zero3.json` | ZeRO-3 + bf16 + gather-on-save |
| `smoke_mask.py` | `smoke.py` | 7 项自检（投影形状 / ans_pos 槽位 / shift 等价 / 截断隔离×2 / 行独立 / pad 隔离） |
| `eval_8budget.py` | `eval.py` | per-k CE + greedy + json/md |
| `report_8budget.py` | 未做（v1） | `eval.md` 直接出表；画图留给报告脚本 |

本分支就是一个独立最小项目：只有上面这 9 个文件（`model.py` / `encoder.py` / `data.py` /
`train.py` / `eval.py` / `smoke.py` / `ds_zero3.json` / `requirements.txt` / `README.md`），
不含 `doc/`、`tools/` 与历史实验脚本；复制整个目录即可在 remote 跑。

### 6. review 要点 / 已知限制

- **P0 shift**：`expand_budget_rows()` 里 `ans_pos = k + label_pos − 1`；改动此处必跑
  `smoke.py`（shift 等价 + 槽位覆盖双锁）。临时改回 no-shift 会让 shift 检查 FAIL。
- **冻结保护**：`VLM8Budget._keep_encoder_eval()` 在每个 forward 入口把已冻结 encoder 拉回
  `eval()`（Trainer 会 `model.train()`）；E2 起 `requires_grad=True` 后自动失效。
- **ZeRO-3 与 param groups**：E1 冻结期 encoder 组仍留在 optimizer（无梯度、不更新），
  否则 engine 初始化后无法把解冻参数加进去。
- **`encoder.py` 是 vendored 副本**：主线 `model_v2_linattn.py`（commit `1fc4920`）逐字抽取；
  上游若改类名 / 子模块名，阶段2 ckpt 的 `strict=True` 加载会失败，需同步本文件。
- **未验证项**：`Qwen3_5ForConditionalGeneration.from_pretrained(dtype=bf16)`、`lm.model`
  取 hidden、`del lm.model.visual`、逐行 `generate(inputs_embeds=...)`、ZeRO-3 + 4B 显存
  —— 全部依赖 remote（transformers 5.22.1）；本机只保证 import / CLI / 张量逻辑。
- **评测权重**：`eval.py` 只读 safetensors 分片索引（Trainer 默认 `save_safetensors=True`）。
- **数据**：`train_nothink_all.jsonl` 的 `image` 是相对路径；`--image-root` 必须对。
