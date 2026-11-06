# vlm8b — AssocVTC 448²/p32 阶段2编码器 → 视觉读出（linear / loop）→ Qwen3.5-4B 全参微调

> **状态（2026-11-06）：仅本机 CPU 逻辑自检通过** —— `python3 smoke.py` 在 torch 2.8 CPU 上
> **38 项 PASS / 0 FAIL**（linear 基线 4 项 + loop 读出 5 组 × 2 种输出头 `mlp`/`transformer`
> + 阶段2 热启动加载器 3 项）。
> **loop 读出尚未在 GPU / Qwen3.5-4B / DeepSpeed 上跑过**（本机无 transformers 5.22、无 4B
> 权重）⇒ 首次真机请先 `--smoke`。
>
> 本分支 = `feat/vlm8b-v1`（tip `0340f40`）+ 视觉读出侧循环化：新增 `--readout loop`
> （默认 `linear`，与 `feat/vlm8b-v1` 逐位一致，可作 A/B 基线）。
> 设计文档：[`doc/2026-11-06/DESIGN_looped_readout_vlm8b.md`](https://github.com/shaojun0/AssocVTC/blob/main/doc/2026-11-06/DESIGN_looped_readout_vlm8b.md)
> （主线仓库；本分支只放代码）。

把 AssocVTC 阶段2 的 `z_s (32×1744)` 读成 **8 个 2560 维 visual token**；同一张图一次
前向展开 **8 行**（行 k 只用前 k 个 token）去预测同一段文本，8 行 CE 取（加权）平均。
3 epoch：E1 冻结编码器，E2–E3 全解冻；DeepSpeed ZeRO-3。

## 1. 两条视觉读出（`--readout`）

### 1.1 `linear`（默认 = 原基线，逐位不变）

```
z_s (B,32,1744) ──LN(1744)→Linear(1744→640)（逐 token）──▶ (B,32,640)
                ──reshape(B,8,4×640=2560)──▶ LayerNorm(2560) ──▶ vis (B,8,2560)
```
8 个 token 互不通信：第 j 个 token 只由 `z_s[4j:4j+4]` 线性压缩而来，预算 k = 前 4k 个 `z_s`。

### 1.2 `loop`（本分支新增：预算 = 循环步数）

```
z_cls(B,1,1744) ┐
z_s (B,32,1744) ┴▶ mem = LN(Linear([z_cls; z_s])) + mem_pos  (B,33,dim)
                    │  权重共享 decoder 栈（L 层）被施加 8 次：
          Q[1]      ─┴─▶ Stack(Q[1],       窗口 0..4 ) ─▶ V1  (B,1,2560)
          Q[2]+Y1   ────▶ Stack(Q[2]+Y1,   窗口 5..8 ) ─▶ V2
             …                                    …
          Q[8]+Y7   ────▶ Stack(Q[8]+Y7,   窗口 29..32) ─▶ V8
                    vis = cat(V1..V8) (B,8,2560) ─▶ LayerNorm(2560)
```
- 第 t 步只读第 t 块 4 个 register（`fixed block=4`，与 linear 的分组一一对应），
  carry **不 detach（BPTT）**，第 t 步产出第 t 个 token；
- 于是 **预算 k = 循环步数 k = 前 4k 个 register = 前 k 个 LM token**，三种截断对齐；
- 每步都有直接监督：token t 出现在第 t..8 行里 ⇒ 现有「8 行预算 CE」免费提供逐步监督；
- 每步的 identity 只由 `q_bank` 的独立一行提供，**不再加 step embedding**（两套会互相干扰）；
- **输出头（最后一层）可选**（`--loop-head`）：
  - `mlp`（**默认**，按设计稿的 MLP 投影）：逐 token `Linear(dim→dim) → GELU →
    Linear(dim→LM_HIDDEN)`，token 间独立；
  - `transformer`：**单层 Transformer（causal self-attn + FFN）+ 线性投影**。必须 causal：
    token t 只看 token ≤ t，否则 token 0 会看到后段 register，破坏「预算=前缀」契约
    （smoke ⑥ 会 FAIL）；causal 下补上了 V1 里缺失的 token 间通信。
- `--loop-init stage2` 可用阶段2 ckpt 的 `decoder.*` 热启动：复用 `mem_proj/mem_norm/
  mem_pos/stack.layers.{0..depth-1}/norm_out`，丢弃 `query_base/query_pos/pixel_head`
  （像素 3072 → LM 2560，形状不匹配），`q_bank` 与输出头新初始化。

### 1.3 张量形状

| 阶段 | linear | loop |
|---|---|---|
| `pixel_values` | `(B,3,448,448)` | 同左 |
| encoder 输出 | `z_s (B,32,1744)`（z_cls 丢弃） | `(z_cls (B,1,1744), z_s (B,32,1744))` |
| 读出 | `(B,32,640)` → `(B,8,2560)` | 8 步循环，每步 `(B,1,2560)` → `(B,8,2560)` |
| `vis` | `(B,8,2560)` | 同左 |
| 行展开 | `(B·V, 8+T, 2560)`，行 k = `[vis[:,:k]; text]`（right-pad） | 同左 |
| `ans_pos` | `k + (label_pos − 1)`（预测位置） | 同左 |

`T` = 文本长度（nothink ≈ 112，think ≈ 578）；`A` = answer 长度（nothink 4）；
`V` = 预算数（默认 8）；`A` 轴按批内最长 answer 对齐。

## 2. 三条关键裁决（`feat/vlm8b-v1` 原口径，loop 下不变）

- **8 行批，不打包**：Qwen3.5 是 hybrid（24 层 gated-delta 线性注意力 + 8 层全注意力）；
  线性层收到 4D 自定义 mask 时 `create_recurrent_attention_mask()` 返回 `None` ⇒ 打包会
  跨段泄露。本实现只用 **2D right-padding mask**，行与行是独立 batch 行。
- **防泄露 = 截断 + padding mask**：行 k 只含 `vis[:, :k]`，**不用 mask 遮 visual**。
  `smoke.py` 验证：改 `z_s[4]` 不影响 k=1、影响 k=2。
- **`position_ids` 不传**：Qwen3.5 文本 forward 默认 `arange(0..L-1)`，与 mask 无关。

**loop 的前缀因果（新增契约）**：token t 只依赖前 `t*block` 个 register ⇒ 行 k 的视觉前缀
`vis[:k]` 的依赖范围恰好是 `z_s[:k*block]`。`smoke.py` ⑥ 用 `z_s[4]`（→ 影响 vis[1:]）与
`z_s[20]`（→ 影响 vis[5:]）两处探针锁死；截断不会"偷看"后面的 register。

**输出头的因果约束（新增）**：`--loop-head transformer` 的 self-attn 必须是 causal——
若改成双向，`z_s[20]` 会经 token 8 反传到 token 0，`smoke.py` ⑥ 的
「`z_s[20] → vis[:5] unchanged`」立刻 FAIL，预算前缀语义失效。

**shift 公式一行**：取 hidden 的位置 = `k + (label_pos − 1)`，targets = `labels[label_pos]`，
`label_pos = answer 起始 + j`（`start >= 1` 断言）；**绝不能**用 label 自身位置取 hidden
（hidden 已看见该 token 的 embedding ⇒ 目标泄露）。`smoke.py` 的 shift 检查用**非循环**
参考：`F.cross_entropy(logits[:, :-1], labels[:, 1:])` 按行 mean，linear/loop 两条读出都跑。

## 3. 训练协议（§5）

| 项 | 值 | 位置 |
|---|---|---|
| 优化器 | AdamW β=0.9/0.95 eps=1e-8 | `VLM8BudgetTrainer.create_optimizer` |
| lr | LLM 1e-5 / proj 1e-4 / encoder 5e-6（E2+） | `--lr-llm/--lr-proj/--lr-encoder` |
| **loop lr** | **建议 proj 1e-5~3e-5（热启动）/ 1e-4（从零）** | `--lr-proj`；warm start 用大 lr 会冲掉阶段2 读出权重（train.py 会告警） |
| wd | LLM 0.1 / 其余 0.05；bias+LayerNorm 不衰减 | `build_param_groups` |
| 调度 | cosine，warmup 3%，grad clip 1.0 | `TrainingArguments` |
| 批次 | micro 2 img/GPU（16 行）× accum 8 × 2 GPU | `--micro-batch-size/--grad-accum` |
| 精度 / ckpt | bf16 + grad ckpt（`use_reentrant=False`） | `ds_zero3.json` / `load_lm` |
| 日志 | 每 20 步 per-budget CE → `per_budget_ce.jsonl` | `BudgetCELogCallback` |
| eval | **Trainer 内不做**（`eval_strategy="no"`） | 评测走 `eval.py` |

| epoch | encoder | loop 读出 |
|---|---|---|
| E1 | ❄️ `requires_grad=False` + `eval()` | 🔥 可训（挂在 `model.projector` 下，proj 组） |
| E2–E3 | 🔥 全解冻（lr 5e-6） | 🔥 继续训 |

## 4. 跑法

```bash
# 本机（CPU）：只需 torch + transformers + torchvision
python3 -m py_compile *.py
python3 smoke.py                        # 退出码 0 = PASS（23 项）

# remote（2×A800，venv /root/venv_qwen）：先冒烟再长跑
torchrun --nproc_per_node=2 train.py --smoke \
    --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl

# 基线（linear，与 feat/vlm8b-v1 同口径）
torchrun --nproc_per_node=2 train.py \
    --output-dir /root/autodl-tmp/vlm8b/output/loop_linear \
    --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl

# loop（从零，小栈）
torchrun --nproc_per_node=2 train.py --readout loop --loop-init none \
    --loop-dim 1024 --loop-depth 2 --lr-proj 1e-4 \
    --output-dir /root/autodl-tmp/vlm8b/output/loop_scratch \
    --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl

# loop（阶段2 ckpt 热启动；推荐首跑。dim/heads 由 ckpt 决定，depth=min(--loop-depth, ckpt深度)）
torchrun --nproc_per_node=2 train.py --readout loop --loop-init stage2 --loop-depth 2 \
    --lr-proj 2e-5 \
    --output-dir /root/autodl-tmp/vlm8b/output/loop_warm \
    --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl

# loop（热启动 + 单层 Transformer 输出头；与上面 MLP 版做 A/B）
torchrun --nproc_per_node=2 train.py --readout loop --loop-init stage2 --loop-depth 2 \
    --loop-head transformer --lr-proj 2e-5 \
    --output-dir /root/autodl-tmp/vlm8b/output/loop_warm_tf \
    --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl

# 评测：per-budget CE + greedy exact-match（输出 eval.json / eval.md）
python3 eval.py --checkpoint /root/autodl-tmp/vlm8b/output/loop_warm/final \
    --jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/val_nothink_all.jsonl \
    --limit 1000
```

默认路径（可用 CLI 覆盖）：阶段2 ckpt `/root/autodl-tmp/exp448/B/ckpt_best.pt`、
Qwen `/root/qwen_models/Qwen3.5-4B`、图片根 `/root/autodl-tmp/datasets/imagenet100`。
单卡调试：`--deepspeed ""`。

**loop 相关开关**：

| 开关 | 默认 | 说明 |
|---|---|---|
| `--readout` | `linear` | `linear` 基线 / `loop` 循环读出 |
| `--loop-dim` | 1024 | 循环宽度（`--loop-init stage2` 时由 ckpt `dec_dim` 覆盖） |
| `--loop-depth` | 2 | 共享栈层数（stage2 时取 `min(该值, ckpt dec_depth)`） |
| `--loop-heads` | 8 | 注意力头数（stage2 时由 ckpt 覆盖） |
| `--loop-init` | `none` | `none` 从零 / `stage2` 载入 ckpt `decoder.*` 浅层 |
| `--loop-block` | 4 | 每步读的 register 数（8×4=32=K） |
| `--loop-dropout` | 0.0 | 循环栈 dropout（阶段2 ckpt 训练时 dec dropout=0） |
| `--loop-head` | `mlp` | 输出头（最后一层）：`mlp`=逐 token 2 层 MLP / `transformer`=单层 Transformer（causal） |

## 5. 文件 / 实现映射

| 设计 §13 | 本实现 | 关键内容 |
|---|---|---|
| `model_wrap.py` | `model.py` | `AssocVTCEncoder` / `BudgetProjector` / **`LoopVisualReadout`** / `VLM8Budget` |
| `model_v2_linattn.py`（主线） | `encoder.py` | `StandardAttentionEncoder`（逐字 vendored，main@`1fc4920`） |
| `data_8budget.py` | `data.py` | `SelfDistillSFTDataset` / `EightBudgetCollator` / `build_image_transform` |
| `train_8budget.py` | `train.py` | `VLM8BudgetTrainer` / `EncoderFreezeCallback` / `BudgetCELogCallback` + loop CLI |
| `ds_zero3.json` | `ds_zero3.json` | ZeRO-3 + bf16 + gather-on-save |
| `smoke_mask.py` | `smoke.py` | linear 7 项 + **loop ⑥–⑨**（前缀因果/BPTT/热启动加载器） |
| `eval_8budget.py` | `eval.py` | per-k CE + greedy + json/md（按 config.json 自动重建 linear/loop，无需改） |

实现要点：

- **参数量（实测，含输出头）**：

  | 配置 | `--loop-head mlp` | `--loop-head transformer` |
  |---|---|---|
  | `--loop-init none --loop-dim 1024 --loop-depth 2` | **39.1M** | **50.7M** |
  | `--loop-init stage2 --loop-depth 2`（dim=2048） | **147.4M** | **193.6M** |
  | `--loop-init stage2 --loop-depth 7`（全量热启动） | **483.1M** | **529.3M** |

  算力（dim=2048×2 层，8 步 × 1 query）：MLP 头 ≈1.07 GMACs/图；Transformer 头再加一层
  ~0.4 GMACs/图。相对 8 行 LLM 前向（≈3.84 TMACs/图）均 **<0.1%**。
- **显存（ZeRO-2 口径，估算，≈1 GB / 100M）**：+0.4 GB/卡（39M）/ +1.5 GB/卡（147M）/
  +1.9 GB/卡（194M）/ +4.8 GB/卡（483M）/ +5.3 GB/卡（529M）。
  当前实测 72.7 GB/卡 ⇒ **全量热启动偏紧，首跑用 `--loop-depth 2` 或从零小栈**。
- **二次读 ckpt**：`--loop-init stage2` 会在建 encoder 之外再 `torch.load` 一次阶段2 ckpt
  （≈3.8 GB，瞬时）。若带宽紧张可先 `--loop-init none` 跑通链路。
- **param group**：loop 模块必须挂在 `model.projector` 下（已如此）⇒ 自动进 proj 组；
  否则 `build_param_groups` 只认 lm/projector/encoder，会**静默不更新**。
- **续训/评测**：`--init-from` 与 `eval.py` 的「只允许缺 `lm.lm_head.weight`」断言对
  linear/loop 同构产物均成立（只要 config.json 的 readout/loop_* 一致）。

## 6. review 要点 / 已知限制

- **P0 shift**：`expand_budget_rows()` 里 `ans_pos = k + label_pos − 1`；改动此处必跑
  `smoke.py`（shift 等价 + 槽位覆盖双锁）。临时改回 no-shift 会让 shift 检查 FAIL。
- **冻结保护**：`VLM8Budget._keep_encoder_eval()` 在每个 forward 入口把已冻结 encoder 拉回
  `eval()`（Trainer 会 `model.train()`）；E2 起 `requires_grad=True` 后自动失效。
- **ZeRO-3 与 param groups**：E1 冻结期 encoder 组仍留在 optimizer（无梯度、不更新），
  否则 engine 初始化后无法把解冻参数加进去。
- **`encoder.py` 是 vendored 副本**：主线 `model_v2_linattn.py`（commit `1fc4920`）逐字抽取；
  上游若改类名 / 子模块名，阶段2 ckpt 的 `strict=True` 加载会失败，需同步本文件。
- **loop 读出的未验证项**：GPU 上的 DeepSpeed 组合、bf16 下 8 步 carry 数值、以及
  「循环是否真让 per-budget 曲线站起来」——**全部待真机实验**（判定标准见设计文档 §9）。
- **输出头**：默认 `mlp`；`transformer` 头内部 **必须 causal**（改成双向会破坏预算前缀语义，
  `smoke.py` ⑥ 直接 FAIL）。两种头对同一配置的参数量差见 §5 表。
- **评测权重**：`eval.py` 只读 safetensors 分片索引（Trainer 默认 `save_safetensors=True`）。
- **数据**：`train_nothink_all.jsonl` 的 `image` 是相对路径；`--image-root` 必须对。
