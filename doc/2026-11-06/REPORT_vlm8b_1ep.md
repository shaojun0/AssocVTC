# vlm8b 首跑报告：448²/p32 × 8 预算并行 × Qwen3.5-4B（1 epoch）

**日期**：2026-11-06 09:17 完成 · 耗时 **9.44 h**（2×A800 80G，ZeRO-2）· 训练 3803 步 × 8.93 s/step
**代码**：`feat/vlm8b-v1`（tip `1230b71`）· **产物**：remote `/root/autodl-tmp/vlm8b/output/run1/`（final 10G）+ 本地 `/home/linaro/dsh/vlm8b_eval/`

## 1. 结果（val_nothink_all，n=300）

| k | CE | exact-match acc |
|---:|---:|---:|
| 1 | 0.3354 | 26.00% |
| 2 | 0.2833 | 29.67% |
| 3 | 0.2798 | 31.00% |
| 4 | 0.2744 | 27.67% |
| 5 | 0.2749 | 26.00% |
| 6 | 0.2683 | 26.33% |
| 7 | 0.2685 | 28.00% |
| 8 | **0.2682** | **28.67%** |

- ΔCE(k=8 − k=1) = **−0.067**（单调性基本成立，k≥3 后饱和）
- Δacc(k=8 − k=1) = **+2.67%**（在 n=300 噪声内不显著，±2.6%/95% CI）
- 训练内 per-budget CE（每 20 步均值）：step≤40 ≈ 5.4 → step 900–960 ≈ 0.53 → 末 10 条 ≈ 0.51；loss 111 → 7.2

## 2. 判读

1. **视觉通路成立**：acc 28% ≫ 随机 1%（100 类），CE 随 k 单调下降（k=1 最差）。
2. **8 预算机制只"部分激活"**：CE 差异小（0.067），k≥3 后曲线压平，acc 无单调趋势 ⇒ 在 IN100 这个任务上，>2–3 个 visual token 带来的增益很小。
3. **明显欠训**：绝对 acc 28% 远低于该 val 集标签上限（GT hit 88.68%）；1 epoch × lr 1e-5 不足，3 epoch 预计显著更高。
4. 与 minival 先验（joint 优于 single）不矛盾：这里只跑了 1 epoch，且 k 曲线尚未拉开；需要更长训练或更依赖视觉细节的任务（OCR）来判别。

## 3. 今晚的配置偏差（相对方案 v1）

| 方案 v1 | 实际 | 原因 |
|---|---|---|
| ZeRO-3 | **ZeRO-2** | torch 2.12 + deepspeed 0.19.9 首步 backward 集合通信死锁（同 seq 号两侧 op 类型不同） |
| E1 冻结 encoder | **全程可训（lr 5e-6）** | DeepSpeed Stage1/2 对"全冻结参数组"报 `IndexError: list index out of range` |
| grad ckpt 开 | **关** | ZeRO-3 下组合有过卡死；关后可省 30% 时间，显存 72.7G 仍够 |
| 3 epoch | **1 epoch** | 8.93 s/step ⇒ 3 epoch 需 28h，超预算 |

其余按 v1：448²/p32、8 预算、equal 权重、nothink `_all`、lr 1e-5/1e-4/5e-6、warmup 3%、bf16、micro 2×accum 8×2GPU。

## 4. 真机暴露并修复的兼容问题（全部已 commit、push）

1. `args.seed` 未定义（argparse 缺 `--seed`）
2. transformers 5.22 无 `warmup_ratio` → 换算 `warmup_steps`
3. ZeRO-3 `auto` bucket 需 `config.hidden_size` → 外层 config 补 2560
4. deepspeed 0.19.9 不接受 `exclude_frozen_parameters`
5. `apply_chat_template(tokenize=True)` 在 5.x 返回 BatchEncoding → `_chat_ids()`
6. `_tied_weights_keys` 5.x 要求 dict → 外层空 dict（Qwen 自带 tie）
7. eval 只认分片 index.json → 支持单文件 `model.safetensors`
8. **eval acc 恒 0 的口径 bug**：`gt` 是 `{'name': 'robin'}` 的 dict 字符串、生成是 `The answer is robin` → 新增 `gt_name()` 归一化 + `extract_answer()` + `_norm()`（训练权重不受影响，仅评测口径）

## 5. 下一步（待定）

- A. **续训 2 epoch**（~19h）：需给 train.py 加 `--init-from`（从 final/ 载入后继续），预期 acc 大幅提升、k 曲线拉开；
- B. **OCR 实验**（真正的预算判别）：合成 OCR（字符串长度可控）+ 真实 benchmark（TextVQA/OCR-VQA 子集），1–2 epoch，指标 CER/EM + per-k 曲线；
- C. **视觉消融快测（1h，eval-only）**：text-only / batch 内打乱图像，量化"视觉被用得多深"（区分“没学足够”与“没在看图”），建议在 A/B 之前先做。

## 6. 待补

- S3-zs448 对照表（t'=1..16：21.21→26.30 dB；同 token 数比 zs224 +0.78→+1.43 dB；同值数 7,168 处 +0.99 dB）→ 写入 `doc/2026-11-04/` 并 commit。
