# vlm8b — AssocVTC 448²/p32/960M 阶段2编码器 → 8 预算并行 → Qwen3.5-4B 全参微调

> **当前状态（2026-11-05，含 review 修复提交）：仅本地逻辑自检通过**——
> `cd vlm8b && python3 smoke.py` 在 torch 2.8 CPU 上 **66/66 PASS**；
> **尚未在 GPU / Qwen3.5-4B / ZeRO-3 上跑过**（本机无 transformers 5.22 / 无 4B
> 权重 / 无空闲 GPU）。所有 remote 命令按设计文档写就，第一次真机运行请先走 `--smoke`。
>
> 本次修复（详见 §11）：**P0 answer 位置差一位（目标泄露）** —— 预测位置改为
> 标准因果 shift `ans_pos = k + (label_pos − 1)`；**P1#1** Trainer 内不再 eval
> （`forward` 不返回 `logits`，`prediction_step` 无法工作）；**P1#2** 冻结 encoder
> 在 `forward` 入口的幂等 `eval()` 保护。

一句话：把 AssocVTC 阶段2 的 `z_s (32×1744)` 投成 **8 个 2560 维 visual token**，
同一张图一次前向展开 **8 行**（行 k 只用前 k 个 token）去预测**同一段文本**，
8 行 CE 取（加权）平均；3 epoch，E1 冻结编码器 / E2–E3 全解冻，DeepSpeed ZeRO-3。

对应设计：`doc/2026-11-05/DESIGN_8budget_vlm_finetune.md`（方案 v1）、
`doc/2026-11-05/DESIGN_parallel_budget_vlm.md`（8 行批 vs 打包的裁决）、
`doc/2026-09-22/ANALYSIS_attn_concat_vs_cross.md`（掩码语义）。

---

## 1. 架构

```
image (B,3,448,448)                 data.py: train=RRC(448,0.35–1.0)+HFlip / val=Resize(512)+CenterCrop(448)
   │                                Normalize = model_v2_linattn.IMAGENET_MEAN/STD
   ▼
AssocVTCEncoder                     StandardAttentionEncoder: 448²/p32, depth 13, heads 16, K=32
   │  forward(pixel_values) -> (z_cls, z_s)   ← 只用 z_s；decoder 完全不加载
   ▼
z_s (B,32,1744)
   │
   ▼  BudgetProjector
LayerNorm(1744) → Linear(1744,640)              逐 token
   │  (B,32,640) = 20,480 个数
   ▼  reshape (B,8,4×640=2560) → LayerNorm(2560)
vis (B,8,2560)                                  与 Qwen LM hidden 同维
   │
   ▼  每张图展开 V 行（默认 V=8，k=1..8）:  行 k = [vis[:, :k] ; text_embeds]
rows (B·V, 8+T, 2560)                           右侧补零到 max_k+T；2D right-padding mask
   │
   ▼  Qwen3.5-4B 文本主干（只喂 inputs_embeds，不喂 pixel_values，vision tower 已删）
hidden (B·V, 8+T, 2560)                         只 gather answer 的**预测位置**（label−1）
   │
   ▼  lm_head（分块）
logits (B·V, A_max, 248320)                     标准因果 shift：hidden[p] 预测 label[p+1]
   │
   ▼  CE（fp32 分块）→ 行 CE → 每预算 CE → loss = Σ_k w_k·CE_k
loss
```

### 张量形状表

| 阶段 | 形状 | 说明 |
|---|---|---|
| `pixel_values` | `(B,3,448,448)` | train RRC+HFlip / val Resize(512)+CenterCrop |
| `z_s` | `(B,32,1744)` | `StandardAttentionEncoder` 的 specials 段（K=32, enc_dim=1744） |
| `tokens` | `(B,32,640)` | `LayerNorm(1744) → Linear(1744,640)`，逐 token |
| `vis` | `(B,8,2560)` | `reshape(B,8,4*640) → LayerNorm(2560)` |
| `text_embeds` | `(B,T,2560)` | `lm.get_input_embeddings()(input_ids)` |
| `rows.embeds` | `(B·V, 8+T, 2560)` | 行 k = `[vis[:,:k]; text]`，右侧补 `8-k` 个零 |
| `rows.labels` | `(B·V, 8+T)` | visual 前缀 / padding / prompt 全 `-100` |
| `hidden` | `(B·V, 8+T, 2560)` | `lm.model(inputs_embeds=..., attention_mask=2D)` |
| `rows.ans_pos` | `(B·V, A_max)` | **预测位置** `k + (label_pos − 1)`；无效位 `-1` |
| `logits` | `(B·V, A_max, 248320)` | 只对 answer 的**预测位置**调 `lm_head`（分块 64 位置） |
| `loss` | 标量 | 8 行 CE 的 `Σ w_k·CE_k`；`equal` 或 `inv-k` |

`T` = prompt + answer 的文本长度（nothink 实测 ~112，think ~578）；`A` = answer 长度
（nothink 4，think 58，`logits` 的 A 轴按批内最长 answer 对齐）；`V` = 预算数（默认 8）。

---

## 2. 与设计文档的映射

### 2.1 §13 实现文件计划 → `vlm8b/`

| 设计 §13（`tools/vlm/*`） | 本实现 | 关键函数 / 类 |
|---|---|---|
| `model_wrap.py`（encoder + 投影 + 冻结/解冻） | `vlm8b/model.py` | `AssocVTCEncoder` / `BudgetProjector` / `VLM8Budget` |
| `data_8budget.py` | `vlm8b/data.py` | `SelfDistillSFTDataset` / `EightBudgetCollator` / `build_image_transform` |
| `train_8budget.py` | `vlm8b/train.py` | `VLM8BudgetTrainer` / `EncoderFreezeCallback` / `BudgetCELogCallback` |
| `ds_zero3.json` | `vlm8b/ds_zero3.json` | ZeRO-3 + bf16 + gather-on-save |
| `smoke_mask.py`（§7.2 四个自检） | `vlm8b/smoke.py` | `check_invariants()`（行独立/截断隔离/pad 隔离/预算单调） |
| `eval_8budget.py` | `vlm8b/eval.py` | `evaluate()`（per-k CE + greedy + exact match） |
| `report_8budget.py` | 未做（v1） | `eval.py` 直接输出 `eval.md` 表；画图留给报告脚本 |

### 2.2 关键裁决（§4）

- **8 行批，不打包成一条序列。** Qwen3.5 是 hybrid（24 层 gated-delta-rule 线性注意力 +
  8 层全注意力）；线性层收到 4D 自定义 mask 时 `create_recurrent_attention_mask()` 返回
  `None`，会跨段泄露。本实现只用 **2D right-padding mask**，行与行是独立 batch 行。
- **防泄露 = 截断（行 k 只含 `vis[:, :k]`）+ padding mask**，不靠 mask 遮 visual。
  `smoke.py` 的 `invariant 2b` 专门验证：改 `z_s` 第 5 个 token 不影响 k=1。
- **`position_ids` 不传**：Qwen3.5 文本 forward 默认 `arange(0..L-1)`，与 mask 无关。
- **answer-only logits**：`lm.model(...)` 出 hidden → gather answer 的**预测位置**
  → `lm_head` → 分块 CE（`CE_CHUNK_SIZE=64`），不 materialize 全序列 vocab logits。
- **标准因果 shift（P0 修复）**：`labels = input_ids`、prompt 全 `-100`，position `p`
  的 hidden 只能预测 `label[p+1]` ⇒ 取 hidden 的位置是
  `ans_pos = k + (label_pos − 1)`，targets 仍是 `labels[label_pos]`。用 answer 自身
  位置取 hidden 会让 hidden 看见该 token 的 embedding（目标泄露，loss 可被"复制输入"
  压到 0）。`smoke.py` 用**非循环**参考（`logits[:, :-1]` vs `labels[:, 1:]`）+
  leak 回归断言双向锁死这个语义。

### 2.3 训练协议（§5）

| 项 | 值 | 代码位置 |
|---|---|---|
| 优化器 | AdamW β=0.9/0.95 eps=1e-8 | `VLM8BudgetTrainer.create_optimizer` |
| lr | LLM 1e-5 / proj 1e-4 / enc 5e-6（E2–E3） | `--lr-llm/--lr-proj/--lr-encoder` |
| wd | LLM 0.1 / 其余 0.05；bias+LayerNorm 不衰减 | `build_param_groups` |
| 调度 | cosine → 0，warmup 3% | `TrainingArguments(lr_scheduler_type="cosine")` |
| 批次 | micro 2 img/GPU（16 行）× accum 8 × 2 GPU = 32 img/step | `--micro-batch-size/--grad-accum` |
| 精度 | bf16 参数/激活 + fp32 master（ZeRO 默认） | `ds_zero3.json` |
| grad ckpt | 开（`use_reentrant=False`） | `VLM8Budget.enable_gradient_checkpointing` |
| 冻结 | E1 encoder `requires_grad=False`；E2–E3 全解冻；`forward` 入口幂等 `eval()` 兜底 | `EncoderFreezeCallback` + `VLM8Budget._keep_frozen_encoder_in_eval` |
| 日志 | 每 20 步 loss / 8 个 per-budget CE / lr / gnorm → JSONL | `BudgetCELogCallback` |
| ckpt | 只存 best/last 的 bf16 权重（不存 optimizer） | `save_only_model=True` + `stage3_gather_16bit_weights_on_model_save` |

---

## 3. 数据与路径

| 用途 | remote 路径 |
|---|---|
| 阶段2 编码器 ckpt | `/root/autodl-tmp/exp448/B/ckpt_best.pt` |
| Qwen3.5-4B | `/root/qwen_models/Qwen3.5-4B`（remote venv `/root/venv_qwen`） |
| SFT 数据 | `/root/autodl-tmp/exp_qwen/in100_selfdistill/sft/{train,val}_{nothink,think}_{all,strict}.jsonl` |
| 图片根 | `/root/autodl-tmp/datasets/imagenet100/`（`row["image"]` 是相对路径） |
| 输出 | `/root/autodl-tmp/vlm8b/output/<run>/`（44G 盘，别写进仓库） |

jsonl 每条：`image` / `messages_short`（JSON 数组：system+user+assistant）/ `gt` /
`gt_hit` / `n_out_tokens`。

文本管线（`SelfDistillSFTDataset`）：

1. `prompt_ids = tokenizer.apply_chat_template(messages_short[:-1], add_generation_prompt=True)`
2. `full_ids = apply_chat_template(messages_short)`；校验前缀后取 `answer_ids = full_ids[len(prompt):]`
3. answer 截到**第一个 eos（含）**，丢掉模板尾部的换行 token；没有 eos 则补一个
4. `labels = [-100] * len(prompt) + answer_ids`；文本超 `--max-text-length` 直接报错（不静默截断）

> 训练/评测统一 `messages_short`（评测也用 short，口径一致）；`--messages-key messages`
> 可切到含 100 类清单的全量版做对照。

图像变换与阶段2 编码器一致：train `RandomResizedCrop(448, scale=(0.35,1.0))+HFlip`；
val `Resize(512, BICUBIC)+CenterCrop(448)`；`Normalize` 用
`model_v2_linattn.IMAGENET_MEAN/STD`（导入失败时有同值兜底并注释）。

---

## 4. 快速开始

### 4.1 本机自检（CPU，无需 Qwen/GPU）

```bash
cd /home/linaro/dsh/AssocVTC/vlm8b
python3 -m py_compile *.py
python3 smoke.py            # = --logic-only；66/66 PASS 时 rc=0
```

期望结尾：

```
PASS 66 / FAIL 0  (total 66)
RESULT: PASS
```

其中与本次修复直接相关的 4 行（应逐字出现）：

```
[PASS] row_ce == standard causal shift CE (logits[:-1] vs labels[1:])  (max|Δ|=...)
[PASS] structural: ans_pos == label_pos - 1 + k (per row/element)  (checked ... slots ...)
[PASS] leak-regression: no-shift CE differs from correct shift CE  (min|Δ|=...)
[PASS] trainer default path: eval_strategy='no' + no eval_dataset
```

`--full --encoder-ckpt <ckpt>` 会额外做一次真实编码器的 `z_s` 形状检查（本机无 ckpt
时打印 `[SKIP]`，不算失败）。

### 4.2 remote 冒烟（2×A800，DeepSpeed）

先确认 Step-0 门禁（设计 §7.1）：`deepspeed==0.19.9` 已装、`ninja/nvcc` 可用、
`Qwen3_5ForConditionalGeneration` 能加载。

```bash
cd /root/autodl-tmp/exp448/code/AssocVTC        # 仓库根（含 vlm8b/）

# 2 步冒烟：只走 8 张图，验证 ckpt/Qwen/ZeRO-3/保存链路
/root/venv_qwen/bin/torchrun --nproc_per_node=2 vlm8b/train.py \
  --smoke --train-limit 8 \
  --output-dir /root/autodl-tmp/vlm8b/output/smoke
```

### 4.3 主跑（D1 nothink，3 epoch）

```bash
/root/venv_qwen/bin/torchrun --nproc_per_node=2 vlm8b/train.py \
  --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl \
  --val-jsonl   /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/val_nothink_all.jsonl \
  --output-dir  /root/autodl-tmp/vlm8b/output/D1_nothink \
  --epochs 3 --freeze-encoder-epochs 1 \
  --budget-set 1,2,3,4,5,6,7,8 --loss-weighting equal
```

- 日志：`<output>/per_budget_ce.jsonl`（每 20 步一条，含 8 个 CE）
- 产物：`<output>/final/`（HF 目录：`config.json` + `model.safetensors`，≥10G 级）
- **训练中不做 eval**（P1#1）：`forward` 不返回 `logits`，HF `prediction_step` 无法工作，
  因此 `eval_strategy` 恒为 `"no"`、不传 `eval_dataset`；`--eval-limit` / `--val-jsonl`
  已标 deprecated（保留只为兼容旧启动脚本，**不生效**）。每 epoch 的 ckpt 由
  `--save-strategy epoch` 存下，训练结束后统一跑 §4.4 的 `eval.py`。
- 断点：只存模型、不存 optimizer（设计 §5），因此 `--resume-from-checkpoint` 只恢复权重语义

### 4.4 评测

```bash
/root/venv_qwen/bin/python vlm8b/eval.py \
  --checkpoint /root/autodl-tmp/vlm8b/output/D1_nothink/final \
  --jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/val_nothink_all.jsonl \
  --limit 1000 --budgets 1,2,3,4,5,6,7,8 \
  --output-dir /root/autodl-tmp/vlm8b/output/D1_nothink/eval
```

输出 `eval.json` + `eval.md`（per-k CE / exact-match acc / Δacc(8−1)）。
greedy 生成逐行做（right-padding 下整批 generate 会把 pad 当上下文）。

---

## 5. 阶段冻结表

| 阶段 | epoch | encoder | Qwen LLM | projector | 备注 |
|---|---|---|---|---|---|
| **E1** | 1 | ❄️ `requires_grad=False` + `eval()` | ✅ 1e-5 | ✅ 1e-4 | 只训 LLM+proj；wd 0.1/0.05 |
| **E2** | 2 | ✅ 5e-6 | ✅ 1e-5 | ✅ 1e-4 | `on_epoch_begin` 解冻，同一 optimizer |
| **E3** | 3 | ✅ 5e-6 | ✅ 1e-5 | ✅ 1e-4 | 同 E2 |

`--freeze-encoder-epochs 0` = 全程解冻；`2` = E1–E2 冻结。`VLM8Budget.train()`
会保证"已冻结的 encoder 不被 Trainer 切回 train 模式"。

---

## 6. 设计 §12 签字点 → CLI / 代码

| §12 签字点 | 默认 | 可配项 |
|---|---|---|
| 1. 数据臂 D1/D2/D3 | D1 nothink | `--train-jsonl` / `--val-jsonl`（D3 需先自行 merge 或换文件） |
| 2. nothink `_all` vs `_strict` | `_all` | `--train-jsonl ..._strict.jsonl` |
| 3. prompt `messages_short` vs `messages` | short | `--messages-key {messages_short,messages}` |
| 4. 训练图像变换 RRC vs 确定性 | RRC | `--train-transform {rrc,deterministic}` |
| 5. loss 权重 equal vs 1/k | equal | `--loss-weighting {equal,inv-k}`；模型层 `forward(..., budget_weights=)` |
| 6. 超参 | 设计 §5 | `--lr-llm --lr-proj --lr-encoder --wd-llm --wd-other --micro-batch-size --grad-accum --warmup-ratio --grad-clip` |
| 7. 总 epoch / 子采样 | 3 ep | `--epochs`、`--train-limit`（60k 子采样=60000） |
| 8. ZeRO-3 vs FSDP2 | ZeRO-3 | `--deepspeed <json>`；传 `""` 关闭（FSDP2 **v1 未做**） |
| 9. 评测频率/规模 | 训练后一次性全评 | **Trainer 内不 eval**（`--eval-limit` deprecated）；`eval.py --limit` |
| 10. 随机预算臂 E | 不做 | **v1 未做**（预留 `--budget-set` 做子集臂） |
| （§3 预算集合） | 1..8 | `--budget-set`，如 `2,4,8` |

---

## 7. 实现裁决 / 与"硬事实"的差异（审核重点）

1. **E1 的 encoder 参数组仍留在 optimizer 里**（但 `requires_grad=False` ⇒ 无梯度、不更新）。
   原因：ZeRO-3 在 engine 初始化后**无法**把新参数加进 optimizer；若 E1 严格"不进
   optimizer"，E2 解冻后 encoder 永远拿不到更新。HF Trainer 默认也把所有参数放进
   optimizer。→ 需 Lead 确认（见 §9 第 1 条）。副作用：encoder 组的 cosine 从 step 0
   就按 5e-6 起算，E2 解冻时实际 lr 已衰减（约 2–3e-6），不是干净的"E2 起 5e-6"。
2. **tied weights**：Qwen `tie_word_embeddings=True`，`save_pretrained` 的 shared-tensor
   检查要求显式声明 tied key，因此 `VLM8Budget._tied_weights_keys = ["lm.lm_head.weight"]`。
   存盘会省略 `lm.lm_head.weight`（与 `lm.model.embed_tokens.weight` 共享），
   `eval.py` 的 loader 允许这一个 key 缺失（其余 missing/unexpected 直接报错）。
3. **eval 不用 `from_pretrained` 加载权重**：transformers 4.57+ 的
   `low_cpu_mem_usage` 会把自定义模块建在 meta device 上，`AssocVTCEncoder` 在
   `__init__` 里手动 `load_state_dict` 会静默 no-op。`eval.py` 显式
   "构造 → to(dtype/device) → 读 safetensors/bin（支持分片）→ strict 校验"。
4. **`dtype` 字段改名 `load_dtype`**：`dtype` 是 transformers 的保留字段
   （`save_pretrained` 会把它改写成模型参数 dtype），不能拿来存"加载 Qwen 用什么精度"。
5. **generate 逐行**：right-padding 下整批 `generate` 会把 pad 当上下文（设计 §9 的
   Step-0 注意项），`generate_budget` 按 (图, k) 逐行调用，返回**只有新生成的 ids**。
6. **vision tower**：`load_lm` 后 `del lm.model.visual`；`pixel_values` 从不传给它。
7. **`position_ids` 不传**（按硬事实 4）；`attention_mask` 是 2D right-padding。
8. **`smoke.py` 的 4 个不变量**用极小替身 LM（真实 2560 维接口、内部 64 维）在 CPU 上
   验证；真实 Qwen 上的等价性仍需 Step-0 的 §7.2 数值门禁（见 §9）。
9. **预测位置 = `label_pos − 1`（P0 修复）**：`expand_budget_rows` 的 `ans_pos` 由
   "answer 自身位置"改成 "标准因果 shift 的预测位置"，并要求 `starts ≥ 1`（否则明确
   报错）。`forward_rows` 的 gather/CE 口径不变（仍是 fp32 分块），只是下标来源变了。
   `generate_budget` / `eval.py` **不依赖** `ans_pos`：生成走 prompt 长度，`eval.py`
   只用 `forward()` 的 CE 与 `generate_budget`，无需改动。
10. **Trainer 内不做 eval（P1#1 选型 A）**：`forward` 不返回 `logits`，HF
    `prediction_step` 取不到 logits。选"默认 `eval_strategy="no"` + 不传 `eval_dataset`"，
    评测交给 `eval.py`。已用微型替身模型跑过 1 步真实 `Trainer.train()`（`smoke.py::
    check_trainer_no_eval`），确认训练循环不碰 `prediction_step`；**未**在真实 4B /
    ZeRO-3 上验证（见 §11）。
11. **冻结 encoder 保护（P1#2）**：`VLM8Budget.train()` 覆写 + `forward` / `forward_rows`
    / `generate_budget` 入口的幂等 `_keep_frozen_encoder_in_eval()`。`AssocVTCEncoder`
    的 `is_frozen` / `freeze()` / `unfreeze()` 保留，`EncoderFreezeCallback` 不变。

---

## 8. 目录 / 文件

| 文件 | 内容 |
|---|---|
| `model.py` | 常量 / `VLM8BudgetConfig` / `BudgetRows` + `expand_budget_rows` / `BudgetProjector` / `AssocVTCEncoder` / `VLM8Budget`（forward / forward_rows / generate_budget / 冻结 / per-budget 统计） |
| `data.py` | `build_image_transform` / `SelfDistillSFTDataset` / `EightBudgetCollator` |
| `train.py` | argparse / tokenizer / dataset / `build_param_groups` / `VLM8BudgetTrainer` / 两个 callback / main |
| `eval.py` | checkpoint loader（safetensors/bin + 分片）/ per-k CE + greedy / `eval.json` + `eval.md` |
| `smoke.py` | CPU 逻辑自检（66 项，rc=0/1） |
| `ds_zero3.json` | ZeRO-3：`stage=3`、bf16、`overlap_comm`、`reduce_scatter`、`stage3_gather_16bit_weights_on_model_save=true`、batch="auto"；**CPU offload 默认关**（`offload_optimizer/offload_param: {"device": "none"}`，打开会显著变慢，44G 盘/80G 卡不需要） |
| `requirements.txt` | remote 实测版本（deepspeed 0.19.9 待装） |
| `.gitignore` | `__pycache__/ output/ cache/ *.pt *.jsonl` 等 |

---

## 9. 审核清单 / 需要 Lead 决策

开跑前（Step-0）：

- [ ] ckpt 的 `cfg` 满足 `enc_attn="std"`、`enc_dim=1744`、`K=32`、`patch=32`、`img=448`（不满足会**显式报错**）
- [ ] `Qwen3_5ForConditionalGeneration` 能从 `/root/qwen_models/Qwen3.5-4B` 加载，且 `lm.model.visual` 存在
- [ ] `apply_chat_template` 的默认模板（含 `enable_thinking` 等 kwargs）与生成 SFT 数据时**完全一致**
      —— 不一致会让 prompt/answer 切分点漂移（dataset 有前缀校验 + 兜底，但最好一致）
- [ ] `lm.model(inputs_embeds=..., attention_mask=2D)` 返回 `last_hidden_state`（transformers 5.22.1 源码）
- [ ] §7.2 四个数值门禁在真实 Qwen 上重跑（`smoke.py` 只证明了实现逻辑，不证明 Qwen 内核行为）
- [ ] 真实 Qwen（bf16）上 `row_ce` 与 `F.cross_entropy(logits[:, :-1], labels[:, 1:])`
      的等价性（本机 CPU fp32 是 `max|Δ|≈5e-7`；bf16 需实测，容差按 1e-5 量级看）
- [ ] Trainer 1 步 + 冻结 encoder 的冒烟在真实 4B / ZeRO-3 上跑通（本机只用微型替身验证过）
- [ ] E1→E2 解冻后 encoder 收到**非零 grad**（训练日志/探针；这是本实现唯一没在真机验证的机制点）
- [ ] 32 img/step 的峰值显存（设计估 45–50G/卡）
- [ ] 训练产物 `final/` 能被 `eval.py` 严格加载（tied `lm_head` 例外）

需要决策（≤5 条）：

1. **E1 encoder 是否必须"不进 optimizer"**？本实现为 ZeRO-3 的可解冻性把它保留在组里
   （冻结、无梯度）。若 Lead 坚持排除，E2 需要改成"重新起一个 Trainer/optimizer"的两段式训练。
2. **encoder 组 lr schedule**：当前与全局 cosine 同步（E2 起点已衰减）。若要"E2 起 5e-6 重新
   cosine"，需要在 callback 里重置 `initial_lr`（会偏离 HF scheduler，建议 Step-0 实测后再定）。
3. **nothink answer 只有 4 tok + eos**，预算增益可能饱和在 k=2；是否首跑就混入 think（或 D2 单跑）。
4. **`--budget-set` 起步用 1..8 还是 {2,4,8}**（设计 §5 风险 5 建议后者先跑）。
5. **推理/评测的 batch generate**：当前逐行（安全但慢，1000 图 × 8 预算 ≈ 8000 次 generate）；
   若 Step-0 验证 left-pad 可行，可改成按 k 批量。

---

## 10. v1 未做（明确不做，避免误读）

- **随机预算臂**（MQT 原式，每步随机抽 1 个 k）——设计 §8 的 E 臂。
- **FSDP2** 启动路径——只有 ZeRO-3 json；`--deepspeed ""` 仅用于单卡调试。
- **打包路径**（8 预算拼一条序列 + 4D 块对角 mask）——已按 §4.2 裁决否掉。
- 训练中实时 `acc`（Trainer 只记 CE；acc 走 `eval.py`）。
- optimizer 状态断点续训（`save_only_model=True`）。

---

## 11. 状态 / 未验证项（2026-11-05 修复提交）

### 11.1 本次修复（review 后追加）

| 编号 | 问题 | 改法（文件 / 函数） |
|---|---|---|
| **P0** | answer 位置差一位：用 answer token **自身**位置的 hidden 预测它 ⇒ 因果自注意力可见自己，loss 可被"复制输入"压到 0（目标泄露） | `model.py::expand_budget_rows` 的 `ans_pos` 改为 `k + (label_pos − 1)`，并断言 `starts.min() ≥ 1`；`BudgetRows.ans_pos` / docstring 同步；`forward_rows` 的 gather 仍只取该位置、CE 仍 fp32 分块 |
| **P1#1** | Trainer 内 eval 不可用：`forward` 不返回 `logits`，HF `prediction_step` 取不到 logits | `train.py`：`TRAINER_EVAL_STRATEGY="no"`、不构造/不传 `eval_dataset`、`--eval-limit`/`--val-jsonl` 标 deprecated；评测走 `eval.py`；`smoke.py::check_trainer_no_eval` 微型替身跑 1 步 `Trainer.train()` |
| **P1#2** | `model.train()` 把冻结的 encoder 拉回 train（dropout 打开） | `model.py`：`VLM8Budget.train()` 覆写 + `_encoder_is_frozen()` + `_keep_frozen_encoder_in_eval()`，在 `forward` / `forward_rows` / `generate_budget` 入口幂等调用；`AssocVTCEncoder.is_frozen/freeze/unfreeze` 与 `EncoderFreezeCallback` 保留 |

### 11.2 shift 语义（一句话公式）

> **取 hidden 的位置 = `k + (label_pos − 1)`，targets = `labels[label_pos]`**；
> 即文本位置 `p` 的 hidden 预测 `label[p + 1]`（标准因果 teacher forcing）。
> `label_pos` 是 answer token 在文本里的位置（`starts + j`，`starts ≥ 1`），
> `k` 是该行的 visual 前缀长度。行长度仍是 `k + T`，shape/显存口径不变。

### 11.3 已验证（本机，torch 2.8 CPU / transformers 4.57）

- `python3 smoke.py` → **66/66 PASS, rc=0**，包含：标准 shift 参考等价
  （`max|Δ|≈5e-7`，容差 1e-5）、逐行逐元素 `ans_pos == label_pos − 1 + k`、
  leak 回归（no-shift CE 与正确 CE `min|Δ| ≈ 1e-3 … 1e-1`）、冻结 encoder 5 项、
  Trainer 默认路径 3 项。
- `python3 -m py_compile vlm8b/*.py`、`python3 vlm8b/train.py --help`、
  `python3 vlm8b/eval.py --help` 均通过（Qwen 仍是 lazy import）。
- `generate_budget()` / `eval.py` 不依赖 `ans_pos`：生成用 prompt 长度
  （`attention_mask.sum()`）拼 `[vis[:k]; text[:n]]`；`eval.py` 只用 `forward()` 的
  `per_budget_ce_per_sample` 与 `generate_budget()`，因此**不受本次 shift 改动影响**
  （CE 的数值含义变成"标准 next-token CE"，这是修正而非破坏）。

### 11.4 仍未验证 / 需 Lead 决策

1. **真实 Qwen3.5-4B + bf16**：`row_ce` 与 `logits[:, :-1]/labels[:, 1:]` 的等价性只
   在 CPU fp32 替身模型上验证过；bf16 下的 Δ 量级未实测（容差按 1e-5 看）。
2. **ZeRO-3 / 2×A800**：`Trainer.train()` 只在 CPU 单进程 + 微型替身上跑过 1 步；
   ZeRO-3 下的冻结 encoder 模式保护、`--eval-limit` 废弃后的启动脚本兼容性未在真机跑。
3. **E1→E2 解冻后 encoder 收到非零 grad**（历史遗留，唯一没在真机验证的机制点）。
4. **`starts ≥ 1` 的硬约束**：若某条 SFT 数据的 chat template 把 answer 放在文本第 0
   位（prompt 为空），现在会**显式报错**而不是静默错位；是否需要在 dataset 侧断言/兜底
   由 Lead 定。
5. **性能**：`forward` 里 `_keep_frozen_encoder_in_eval()` 是纯属性读 + 条件 `eval()`，
   每步开销可忽略；但未在真机 profile。
