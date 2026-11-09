# vlm8b run3（评测对齐 + 按任务等量混合集，3 epoch）评测报告

生成：2026-11-09 12:20（Asia/Shanghai）
产物：`/root/autodl-tmp/vlm8b/output/run3/checkpoint-11463`（`final` 软链到此），训练 3.0 epoch / 11,463 步。

> ⚠️ **2026-11-09 13:45 修正（重要）**：本报告 §0.2、§2 的 acc 数字与「视觉 token 有害」结论来自
> **有 bug 的 greedy 协议**（`eval.py` 原实现把含答案的整段文本喂给 `generate`，大量输出空串、acc 被严重低估）。
> 修正为「只喂 prompt」后，同样 300 行：normal 48.7%→49.3%、null 48.3%、shuffle 48.7%，CE normal 0.585 < null 0.604
> ⇒ **视觉是略有益的，不是有害的**；真正的问题是「预算 k 曲线全平」。
> 逐样本筛查与机制分析见 `doc/2026-11-09/REPORT_budget_screen.md`（结论：预算=条件性有效 + 编码器侧容量封顶）。
> §4 的 in100 绝对 acc 也应按新协议重报（run3：29.2%→32.4%，250 行）。

## 0. 结论（TL;DR）

1. **视觉 token 预算机制仍不成立**：mix_heldout 上 normal 模式 acc 从 k1 24.0% → k8 ~22%（分任务表 27.1%→24.3%），
   CE 也基本平（0.582→0.581）。8 个预算行一起训练没有换来"多 token 更好"，run1/run2 的旧观察在广任务上复现且更明确。
2. **文本行用空白图占位是有害的**：300 样本混评里 normal 22.7% < **null（视觉置零）28.0%** > shuffle 23.3%。
   mix_heldout 60% 是纯文本题（MMLU 等），空白图经 encoder 产生的 visual token 是纯噪声，拉低文本题准确率。
   ⇒ 下一版必须让文本行用 **零视觉 token（null）**，而不是 blank.jpg。
3. **评测对齐数据确实提升了任务能力**：per-source 25 样本下，mathvista 60%、mmmu 60%、slake 44%、mmstar 44%、
   chartqa 28-32%、realworldqa 28-32%、mmlu_pro 24-28%，明显高于随机（4 选 1 = 25%）。
4. **通用文本能力大幅恢复**：bench `ours_text` 从 run1/run2 的 0/5/0%（overall 1.7%）提升到
   弱智吧 5% / 算术 **70%** / 常识 **85%**（overall ≈53%）；`ours_vis8` 0/25/45%。这是 run3 最大的正向收益。
5. **ImageNet 能力被稀释**：in100 val k8 28.8%（run2）→ **22.8%**（CE 0.372→0.475）。原因是 ImageNet 锚定从 66k 砍到 3k。
   若要保住 in100 精度，需要更多锚定数据或分阶段训练。
6. 训练按计划在周一 10:56 主动收尾（3.0 epoch，LR 已退火到峰值 3%），评测链 11:03–11:34 完成，结果 11:35 出。

## 1. 训练概要

| 项 | 值 |
|---|---|
| 起点 | `run2/final`（in100 val k8 acc 28.8%） |
| 数据 | `mix_train_bal2.jsonl` 122,259 行（文本 82,000 + 图像 40,259；ImageNet-100 3,000） |
| 步数 | 3,821 步/epoch × 3.0 = **11,463 步**（计划 3.36 epoch，按周一 12:00 截止与实测步速 16.47 s/step 主动收尾） |
| 超参 | grad-ckpt；micro2×accum8×2GPU（全局 batch 32）；LR 5e-6 / 2.5e-6 / 5e-5，warmup 2%，cosine |
| 显存/时长 | 65 GiB/卡；共 52.4 h（06:29 周六 → 10:56 周一） |
| 收尾 loss | Trainer loss ~2.1；同口径 per_budget CE ~0.09–0.12 |

## 2. mix_heldout 三组视觉消融（n=300 随机混采，seed42）

| 视觉模式 | acc k1 | acc k2 | acc k4 | acc k8 | CE k1 | CE k8 |
|---|---:|---:|---:|---:|---:|---:|
| **normal** | 22.7% | 20.7% | 21.3% | 22.3% | 0.584 | 0.585 |
| **null（视觉置零）** | **28.0%** | **28.7%** | **28.3%** | **28.0%** | 0.605 | 0.604 |
| shuffle（错图） | 23.3% | 23.3% | 22.7% | 21.7% | 0.615 | 0.619 |

- normal−null = **−5.3 ~ −8.7pp**（视觉反而有害）；normal−shuffle ≈ 0（视觉信息未被有效利用）。
- 预算曲线在三组里都基本平；k1 与 k8 无显著差异。

## 3. 分任务结果（normal，每源 25 样本；k1/k2/k4/k8 acc）

| source | 类型 | k1 | k2 | k4 | k8 |
|---|---|---:|---:|---:|---:|
| mathvista | img | 60.0% | 56.0% | 56.0% | 52.0% |
| mmmu | img | 60.0% | 48.0% | 52.0% | 52.0% |
| slake | img | 44.0% | 44.0% | 44.0% | 44.0% |
| mmstar | img | 44.0% | 40.0% | 36.0% | 40.0% |
| realworldqa | img | 32.0% | 28.0% | 24.0% | 28.0% |
| chartqa | img | 28.0% | 20.0% | 24.0% | 32.0% |
| ocrbench | img | 28.0% | 20.0% | 20.0% | 16.0% |
| mmlu_pro | text | 28.0% | 28.0% | 28.0% | 24.0% |
| mmlu | text | 20.0% | 20.0% | 20.0% | 24.0% |
| mmlu_redux | text | 20.0% | 16.0% | 16.0% | 20.0% |
| scienceqa | img | 20.0% | 16.0% | 20.0% | 24.0% |
| ai2d | img | 20.0% | 20.0% | 20.0% | 16.0% |
| docvqa | img | 16.0% | 16.0% | 20.0% | 16.0% |
| mathvision | img | 16.0% | 16.0% | 16.0% | 16.0% |
| countbench | img | 8.0% | 4.0% | 4.0% | 4.0% |
| ceval | text | 4.0% | 4.0% | 4.0% | 4.0% |
| cmmlu | text | 4.0% | 4.0% | 4.0% | 4.0% |
| textvqa | img | 4.0% | 4.0% | 0.0% | 0.0% |
| we_math | img | 0.0% | 0.0% | 0.0% | 0.0% |
| **图像源均值** | | **27.1%** | 23.7% | 24.0% | 24.3% |
| **文本源均值** | | 15.2% | 14.4% | 14.4% | 15.2% |

- 强项：mmmu/mathvista（选择题，60%）、slake/mmstar（44%）；弱项：textvqa 4%、we_math 0%（问答/多选答案格式难）、docvqa 16%。
- 文本题（MMLU 系）20–28%，但 C-Eval/CMMLU 只有 4%（中文知识迁移不足，训练量仅 1k/10k 且 CMMLU 用 test）。

## 4. ImageNet-100 主任务对比（val n=250, seed42）

| 模型 | acc k1 | acc k8 | CE k1 | CE k8 |
|---|---:|---:|---:|---:|
| run1（in100 自蒸馏 1 epoch） | 12.4% | 12.0% | 0.499 | 0.467 |
| run2（+1 epoch 续训） | 21.2% | **28.8%** | 0.402 | **0.372** |
| run3（+3 epoch 混合集） | 14.8% | **22.8%** | 0.474 | 0.475 |

⇒ 混合集训练（ImageNet 锚定仅 3,000 行）把 in100 能力从 28.8% 拉回到 22.8%（仍高于 run1）。

## 5. 通用文本能力 bench（文本通道，thinking off，每类 n=20）

| 模型 | 弱智吧 | 算术 | 常识 |
|---|---:|---:|---:|
| base Qwen3.5-4B | 45% | 85% | 95% |
| run1/run2 ours_text | 0% | 5% | 0% |
| run1/run2 ours_vis8 | 0% | 0% | 0% |
| **run3 ours_text** | **5%** | **70%** | **85%** |
| **run3 ours_vis8** | 0% | 25% | 45% |

⇒ 评测对齐混合集（82k 文本 + 40k 图像）显著修复了纯文本能力（run1/run2 的 1.7% → ~53%），
但仍低于 base（75%），且 8 预算视觉前缀下（vis8）仍有明显损失。

## 6. 下一步建议（按优先级）

1. **文本行改用零视觉 token（null）**：数据侧给 `has_image=false` 的行在 model 里把 vis 置零（或 collator 打标），
   预期直接抬升 60% 的 heldout 文本题准确率，并减少视觉噪声。改动小、收益明确。
2. **预算机制重做**：三组消融里预算曲线都平，说明"同一批图 × 8 个预算行一起训"没学到 token 数与精度关系。
   建议：(a) 只训 k∈{1,2,4,8} 减少冗余；(b) 或改成"预算条件 token"（把 k 作为输入条件）而非展开行；
   (c) 检查 k>2 时 visual token 是否被 LM 忽略（可视化 attention/消融单 token 必要性）。
3. **评测放大**：mix_heldout 现在只抽 300 张整体 + 每源 25 张，波动大（25 张 = 4pp/题）。
   下轮建议每源 ≥100、总 ≥2,000，才能对"某任务提升 Xpp"下结论。
4. **ImageNet 锚定权衡**：3,000 行保不住 28.8%（retained 22.8%）。如果要 in100 作为主指标，
   需要把锚定提到 10–20k 或做课程（先广数据、后 in100 回升）。
5. 中文知识（C-Eval/CMMLU 4%）：训练量不足 + 格式/语言迁移问题，单独补数据再看。

## 7. 运维与产物

- 训练：`logs/run3.log`、`output/run3/`（checkpoint-11463 + final 软链）+ `per_budget_ce.jsonl`
- 收尾脚本：`tools/eval_mix/stop_after_epoch3.sh`（按实测步速 16.47s/step、预测 17:14 超过 14:00 截止 → 10:56 停训）
- 评测：`output/eval_run3_mix_{normal,null,shuffle}/eval.json`、`output/eval/eval.json`（in100）、
  `logs/bench_run3.json`、`logs/run3_breakdown_normal.json`、`logs/RUN3_EVAL_DONE`
- 改动记录 commit：`471821d`（数据 v2）、`226208d`（磁盘预算）、`8358b9e`（兜底收尾）、本报告另提交。
