# vlm8b 首跑报告：448²/p32 × 8 预算并行 × Qwen3.5-4B（1 epoch + 消融 + 通用能力 + 续训）

**日期**：2026-11-06 · 首跑 09:17 完成（9.44 h，2×A800 80G，ZeRO-2，3803 步 × 8.93 s/step）
**代码**：`feat/vlm8b-v1`（tip `8d40979`）· **产物**：remote `/root/autodl-tmp/vlm8b/output/run1/`（final 10G）+ 本地 `/home/linaro/dsh/vlm8b_eval/`

## 1. 评测口径修正（重要）

val jsonl **按类排序**，原 `--limit 300` 只评到前 3 个类（chance≈33%）⇒ 首版 acc 26–31% 不可信。
已改为**固定种子随机抽样**（`--sample-seed 42`，`eval.py` commit `8d40979`），以下均为 n=250 随机子集。
CE 不受子集偏置影响，旧表 CE 与随机抽样基本一致。

## 2. 主结果（n=250 随机子集，seed 42）

| k | CE | acc |
|---:|---:|---:|
| 1 | 0.4989 | 12.40% |
| 2 | 0.4684 | 14.00% |
| 3 | 0.4683 | 13.60% |
| 4 | 0.4678 | 11.60% |
| 5 | 0.4683 | 10.40% |
| 6 | 0.4669 | 12.80% |
| 7 | 0.4668 | 11.60% |
| 8 | **0.4666** | **12.00%** |

- ΔCE(k8−k1) = **−0.032**（k≥2 即饱和）；Δacc = −0.4%（噪声内）
- 训练内 per-budget CE（末 10 条均值）：k=1..8 = 0.512/0.519/0.511/0.507/0.507/0.510/0.509/0.511（**各预算几乎无差**）

## 3. 视觉消融（eval-only，n=250，同随机子集）

| 视觉模式 | CE k=1 / k=8 | acc k=1 / k=8 |
|---|---|---|
| normal | 0.499 / 0.467 | 12.40% / 12.00% |
| **null**（vis 置零） | 0.770 / 0.769 | 1.20% / 1.60% |
| **shuffle**（batch 内错位） | 1.007 / 1.047 | 0.80% / 0.00% |

判读：
1. **视觉通路是真的**：normal 12% ≫ chance 1%；null 掉到 chance；shuffle 更差（比"没图"还糟，说明模型会按错误图自信作答）。
2. **但读得浅**：acc 12%（相对 label 上限 88.7% 很低），8 预算曲线几乎平 ⇒ 1 epoch 下模型只学到粗粒度"看图取信息"，预算机制未激活。
3. k=1 与 k=8 的 CE 差只有 0.032，与"整条 8 预算曲线在 k≥2 饱和"一致。

## 4. 通用能力（文本通道，n=60：弱智吧式 20 / 算术 20 / 常识 20）

| 模型 | 弱智吧 | 算术 | 常识 | overall |
|---|---:|---:|---:|---:|
| base Qwen3.5-4B（thinking 关） | 45.0% | 85.0% | 95.0% | **75.0%** |
| vlm8b 微调后（纯文本） | 0.0% | 5.0% | 0.0% | **1.7%** |
| vlm8b 微调后（空白图 + 8 vis token，训练同格式） | 0.0% | 0.0% | 0.0% | **0.0%** |

结论：**1 epoch 窄域全参 SFT 已造成严重通用能力遗忘**（75% → ~0–2%）。对"只做 OCR/分类"的任务无妨，但若目标是通用 VLM 预算机制，必须混入 replay/通用数据或降 lr/改 LoRA。

## 5. 首跑配置偏差（相对方案 v1）

| 方案 v1 | 实际 | 原因 |
|---|---|---|
| ZeRO-3 | **ZeRO-2** | torch 2.12 + deepspeed 0.19.9 首步 backward 集合通信死锁 |
| E1 冻结 encoder | **全程可训（lr 5e-6）** | DeepSpeed Stage1/2 对全冻结参数组报 `IndexError` |
| grad ckpt 开 | **关** | ZeRO-3 下组合过卡死；关掉显存 72.7G 仍够 |
| 3 epoch | **1 epoch** | 8.93 s/step ⇒ 3 epoch 需 28h，超预算 |

## 6. 真机暴露并修复的兼容问题

1. `--seed` 缺失；2. transformers 5.22 无 `warmup_ratio`（→`warmup_steps`）；3. ZeRO-3 `auto` bucket 需 `config.hidden_size`；4. deepspeed 0.19.9 不接受 `exclude_frozen_parameters`；5. `apply_chat_template(tokenize=True)` 返回 BatchEncoding；6. `_tied_weights_keys` 5.x 要 dict；7. eval 只认分片 index.json；8. eval acc 口径（gt dict + 'The answer is' 前缀）；9. **eval 子集按类排序偏置**（随机抽样）；10. 通用 bench 的 thinking 模式截断（base 需 `enable_thinking=False`）。

## 7. 续训（进行中）

- 已从 `run1/final` 续训 **2 epoch**（`--init-from`，同一配置，`--save-total-limit 1` 省磁盘）
- 日志 `logs/run2.log`，产物 `output/run2/`，7606 步 × ~8.9 s ⇒ **ETA ~11-07 06:00**
- 完成后用同一随机子集（seed 42）重跑 normal/null/shuffle 三组 + 8 预算表

## 8. 判读与下一步

- 当前证据：视觉被用但浅、预算机制未激活、通用能力显著遗忘、明显欠训。
- 若 run2 后 acc 仍低 / 预算曲线仍平，候选手段：提高 LM lr（1e-5→3e-5/1e-4）、proj lr 提高、更长训练、混 replay 数据、或直接换 OCR 任务（真正依赖细节）。
