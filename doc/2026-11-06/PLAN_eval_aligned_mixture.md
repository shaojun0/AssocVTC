# 评测对齐混合训练集（eval-aligned mixture）规格 v1

**目的**：run2 明早 06:00 结束后，若 8 预算消融有效，则用一份**与 Qwen3.5-4B 官方评测任务同分布**的混合数据继续微调，
既检验"预算机制在更广任务上是否成立"，又抑制 1 epoch 窄域 SFT 造成的通用能力遗忘（75%→1.7%）。
**截止**：数据须在 2026-11-07 06:00 前就绪；训练跑到 2026-11-09 12:00 为止。

## 1. 体量换算（决定每个源抽多少）

- 单步 8.93 s、每步 32 图（micro 2 × accum 8 × 2 GPU）→ 周六 06:00 至周一 12:00 = 54 h ≈ **21,770 步 ≈ 69.7 万图视图**
- 目标 mixture **≈ 22.5 万样本** → 3.1 epoch ≈ 22,000 步 ≈ 54.5 h（正好压到周一 11:30 左右）
- 组成：**图像类评测数据 45k + 文本类评测数据 90k + ImageNet-100 锚定 90k**

## 2. 样本格式（与现有 `sft/train_nothink_all.jsonl` 完全一致）

```json
{"id": "mmlu_000012", "image": "/root/autodl-tmp/vlm8b/data_eval/img/mmlu/000012.jpg",
 "source": "mmlu", "task": "mcq_text",
 "messages": [ {"role":"system","content":"<SYS>"},
               {"role":"user","content":[{"type":"image","image":"<同 image>"},{"type":"text","text":"<QUESTION>"}]},
               {"role":"assistant","content":"The answer is B"} ],
 "messages_short": "<与 messages 相同>", "gt": "B"}
```

硬性要求：
1. **assistant 内容 = `The answer is <ans>`**（无句号、无解释）；`gt` = `<ans>` 字符串。训练是 answer-only CE，答案长度即监督 token 数。
2. `image` 用**绝对路径**（`train.py` 走 `os.path.join(image_root, image)`，绝对路径会覆盖 image_root，因此 mixture 里可以混 ImageNet-100 的旧行）。
3. `messages_short` 与 `messages` 同值（训练读 `messages_short`）。
4. **prompt + 答案 ≤ 1024 token**（`data.py:MAX_TEXT_LENGTH` 会 assert）：转换时按字符数 ≤ 1400 先过滤，最终用 Qwen tokenizer 复核。
5. 图片落盘：`<root>/img/<source>/<id>.jpg`，JPEG q90，**最长边 ≤ 640**（训练 448² RandomResizedCrop，存大无益、还占磁盘）。
6. 答案必须是"短答案"：<=60 字符（OCR/文档 VQA 的整段答案直接丢弃或用第一行）。
7. 去重：同一 `id`/图像不重复入 train 与 heldout。

## 3. 文本类任务的图像占位

文本类（MMLU/C-Eval/GSM8K/IFEval…）没有图，用**固定空白图** `img/_blank/blank.jpg`（448×448 灰底，全 0.5），
system prompt 明确写"纯文本任务，图像为空"，让模型学会用 system prompt 区分"要不要读视觉 token"。

## 4. system prompt 模板（按 task 类型，固定文本，不要临场发挥）

- `mcq_text`（MMLU/C-Eval/CMMLU/MMLU-Pro）：`你是中文/英文考试助手。这是一道纯文本任务，图像为空。读题后只回复 "The answer is <选项字母>"，不要解释。`
- `mcq_img`（MMMU/AI2D/MMBench/MMStar/SLAKE/ChartQA-mcq）：`你是多模态答题助手。结合图像与题目，只回复 "The answer is <选项字母>"，不要解释。`
- `short_vqa`（DocVQA/TextVQA/OCRBench/CountBench/RealWorldQA/HallusionBench/MathVista/MathVision/We-Math）：`你是多模态问答助手。结合图像与题目，给出最简短的答案，只回复 "The answer is <答案>"，不要解释。`
- `math_text`（GSM8K/MATH）：`你是数学助手。逐步思考后，只回复 "The answer is <最终数值>"，不要解释。`（答案只留最终数值，不要 CoT）

## 5. 抽样配额（train / heldout；heldout 不进训练，留作评测）

### 文本轨道（teammate `text-eval-data` 负责）
| source | HF id | train 抽样 | heldout | 备注 |
|---|---|---:|---:|---|
| mmlu | `cais/mmlu` (all/auxiliary_train) | 45,000 | 1,500 (dev) | aux 有 99,842 条 |
| ceval | `ceval/ceval-exam` (val) | 9,000 | 1,000 (val 剩) | 评测用 test 分片 |
| cmmlu | `haonan-li/cmmlu` (train) | 10,000 | 1,000 (dev) | |
| mmlu_pro | `TIGER-Lab/MMLU-Pro` (test) | 8,000 | 4,000 | 字母 A–J |
| mmlu_redux | `edinburgh-dawg/mmlu-redux-2.0` | 5,000 | 500 | 找不到就跳过 |
| gsm8k | `openai/gsm8k` (main/train) | 7,000 | 0 | 答案取 `####` 后数值；test 留评测 |
| math | `EleutherAI/hendrycks_math` 或 `lighteval/MATH` (train) | 6,000 | 0 | 答案取 `\boxed{}` 内 |
| ifeval | `google/IFEval` (train) | 400 | 141 | 约束指令，答案控制 <=60 字符，超长丢弃 |
| **合计** | | **90,400** | 8,141 | |

### 图像轨道（Lead 负责）
| source | HF id | train 抽样 | 备注 |
|---|---|---:|---|
| mmmu | `MMMU/MMMU`（30 个 config） | 8,000 | 4 选项，答案字母 |
| docvqa | `lmms-lab/DocVQA` (DocVQA/validation) | 8,000 | |
| textvqa | `lmms-lab/TextVQA` (validation) | 4,000 | |
| chartqa | `lmms-lab/ChartQA` (test/train) | 3,500 | 数值/字母答案 |
| slake | `BoKelvin/SLAKE` | 6,000 | 中英医学 VQA |
| ai2d | `lmms-lab/ai2d` | 2,500 | |
| mmbench | `opencompass/mmbench` (dev) | 3,000 | |
| mmstar | `Lin-Chen/MMStar` (val) | 1,000 | |
| mathvista | `AI4Math/MathVista` (testmini) | 800 | 数值答案 |
| mathvision | `MathLLMs/MathVision` | 2,000 | 数值答案 |
| we_math | `We-Math/We-Math` (testmini) | 3,000 | |
| ocrbench | `echo840/OCRBench` | 600 | 答案 <=60 字符 |
| cc_ocr | `YuliangLiu/CC-OCR` | 2,000 | 找不到就跳过 |
| realworldqa | `xai-org/RealworldQA` | 500 | |
| hallusionbench | `lmms-lab/HallusionBench` | 600 | 是/否 型 |
| countbench | `nielsr/countbench` | 300 | |
| **合计** | | **45,800** | |

### 锚定
- ImageNet-100：从现有 `sft/train_nothink_all.jsonl` 抽 90,000 行（`image` 改绝对路径），保持视觉通路不退化。现成无需造。

## 6. 交付物

- `/root/autodl-tmp/vlm8b/data_eval/text_train.jsonl`、`text_heldout.jsonl`（teammate）
- `/root/autodl-tmp/vlm8b/data_eval/img_train.jsonl`、`img_heldout.jsonl`（Lead）
- `/root/autodl-tmp/vlm8b/data_eval/mix_train.jsonl`（Lead 合并：text 90k + img 45k + imagenet 90k，打乱）
- `/root/autodl-tmp/vlm8b/data_eval/mix_heldout.jsonl`（合并 heldout，供周末后预算消融评测）
- 每个 jsonl 的 stats（条数 / 平均答案 token / 丢弃原因计数）

## 7. 训练启动（run3）

```bash
cd /root/vlm8b && torchrun --nproc_per_node=2 --master_port=29511 train.py \
  --freeze-encoder-epochs 0 --epochs 3 --init-from /root/autodl-tmp/vlm8b/output/run2/final \
  --output-dir /root/autodl-tmp/vlm8b/output/run3 --save-total-limit 1 \
  --train-jsonl /root/autodl-tmp/vlm8b/data_eval/mix_train.jsonl \
  --image-root /root/autodl-tmp/vlm8b/data_eval
```
- 前置条件：明早 run2 三组消融 eval 通过（预算曲线非平 / acc 未崩）→ 启动；否则不启动，等 user 决定
- 磁盘：run2 结束后剩 ~25G；mix 图片 ~2GB；run3 检查点 10G × 1 → 够
- 周末无人：看门狗 + DONE 邮件；train3 结束（周一 11:30 左右）后跑 mix_heldout 预算消融

## 8. 风险

1. 文本类样本占比 39%、且都带空白图 → 可能教模型"忽略视觉 token"。缓解：system prompt 区分 + ImageNet/图像类合计 60%（135k/225k）仍占多数；训练中监控 per-budget CE 是否在文本样本上变平（应当变平）。
2. 部分任务 8 token 下几乎不可做（DocVQA/OCR）→ loss 高位，作为"难度梯度"保留，但占比控制 <20%。
3. HF 镜像 429 限流 / script-based 数据集加载失败 → 直连 parquet + pyarrow 兜底；单个源失败不阻塞整体。

---

## 9. 实施记录（2026-11-06 17:30 实构建完成）

### 9.1 实际产出（远端 `/root/autodl-tmp/vlm8b/data_eval/`）

| 文件 | 行数 | 说明 |
|---|---:|---|
| `text_train.jsonl` | 82,000 | mmlu 45k / cmmlu 10k / mmlu_pro 8k / gsm8k 7k / math 6k / mmlu_redux 5k / ceval 1k |
| `text_heldout.jsonl` | 6,612 | mmlu 1,446 / mmlu_pro 3,985 / mmlu_redux 500 / ceval 346 / cmmlu 335 |
| `img_train.jsonl` | 17,445 | docvqa 4,451 / textvqa 3,997+2,000 / chartqa 1,500 / mmstar 999 / mathvista 698 / mathvision 2,000 / we_math 1,200 / mmmu 600 |
| `img_extra_train.jsonl` | 14,400 | scienceqa 6,000 / slake 5,000 / ai2d 2,000 / ocrbench 600 / realworldqa 500 / countbench 300 |
| `img_heldout.jsonl` / `img_extra_heldout.jsonl` | 5,327 / 3,763 | 留作周日/周一预算消融 |
| **`mix_train.jsonl`** | **179,845** | text 82,000 + 图像 31,845 + ImageNet-100 锚定 66,000，seed42 打乱 |
| **`mix_heldout.jsonl`** | **15,702** | 13 个源 |
| 图片 | 31,845 + 18,163 = 50,008 张 | 1.8 GB（JPEG q90，最长边 640） |

### 9.2 预检（CPU，`_preflight_mix.py`）

- 全量 179,845 行：`noimg=0  empty_gt=0  over1024=0`（Qwen3.5 tokenizer 逐行 chat_template 复核）
- 抽样 64 张真实走 `SelfDistillSFTDataset + EightBudgetCollator` → batch (64, 567)，`expand_budget_rows` 得 (512, 575, 2560)、A=12，行序/`ans_pos` 移位/`ans_targets` 跨预算一致 → **PREFLIGHT OK**

### 9.3 与 v1 规格的偏差（已决定，不再回改）

1. **MMLU 字符上限 1400 → 4000**：MMLU 长题干 token p50=466，全量 token ≤900，硬约束（1024 token）满足；严格 1400 会让 MMLU 只剩 33.6k。
2. **C-Eval train 只 1,000**（val 全量仅 1,346，规格误以为 9k）：**不用 test 补**，C-Eval test（12,342）保持干净供周末评测。
3. **CMMLU 无 train split** → 用官方 test 10k 做训练，**CMMLU test 视为污染**，周末评测中文知识改用 C-Eval test。已在此登记。
4. **IFEval 跳过**：无标准化短答案，硬造会污染（4B 89.8 分的一项没有对应训练数据，评测时按"未见过的能力"读）。
5. mmlu_pro / mmlu_redux 用 test/dev 切分（8k+4k / 5k+500），train 与 heldout 互斥。
6. countbench 498/540 可用（49 行 image=None，42 条 URL 失效）；realworldqa 实际混 438 选择题 + 327 短答，已按题型分流 `mcq_img`/`short_vqa`。

### 9.4 run3 配置与启动链

- 脚本：`/root/vlm8b/run_eval_mix.sh`（epochs 3.8 × 5,621 步 ≈ 21,360 步 ≈ 53 h，`--init-from run2/final`，`--save-total-limit 1`）
- 训练前把 `train_fixed.py`（sha256 `a08c2d08…`，E1 冻结修复版）覆盖 `train.py`（当前远端仍是旧版 `61166d2a…`）
- 自动评测链：`/root/vlm8b/eval_after_run2.sh` 已在后台等待 `logs/run2.rc` → 自动跑 normal/null/shuffle 三组（n=250, seed 42）+ 通用 bench → `logs/RUN2_EVAL_DONE`、`output/eval_run2_*/eval.{json,md}`、`logs/bench_run2.json`
- 判定门（06:00 后由 Lead 拍板）：k8 acc 明显高于 chance（≥15%）、normal−null 有明显差距（≥8 个点）、预算曲线出现非平趋势 → 启动 run3；否则不启动，发邮件等指示

### 9.5 磁盘预算与周末运维（11-06 17:35 复核）

- HF `Trainer._save_checkpoint` 是**先存新 ckpt、后 rotate 删旧**（已核 transformers 5.22.1 源码）⇒ 每次存档峰值要 **2×10G**。
- 推演：run2 20:30 存 ckpt（10G）→ 05:50 存 final（峰值需 20G）→ 13G 余；run3 第一次存档（周日 ~20:00）后剩 3G；**第二次存档必然 ENOSPC**。
- 处置：`run_eval_mix.sh` 启动前若剩余 <20G，删除**已被 run2 取代的 run1/final**（10G，评测数字见 `REPORT_vlm8b_1ep.md`），删除动作写入 `logs/disk_notes.txt`；删完仍 <18G 则中止。→ run3 期间余量 13G，存档峰值 3G。
- 看门狗（`tools/vlm8b_watchdog.sh`，已重启）：日志候选加入 `run3.log`、磁盘阈值 `FREE_MIN=2G`、新增 `RUN3_DONE` 里程碑邮件、进程消失判定加入 run3 未完成条件（避免 run2 完成后误报）。
- 自动链：`eval_after_run3.sh` 由 `run_eval_mix.sh` 在训练 rc=0 后自动拉起 → mix_heldout 三组（n=300 seed42）+ ImageNet val（与 run1/run2 同口径）+ 通用 bench → `logs/RUN3_EVAL_DONE`。

### 9.6 run2 训练中期观察：loss 触底回升（11-06 17:45）

- loss min **4.07 @ step≈1900（epoch 0.50）** → step 2680 时 **6.04**；窗口 2280–2680 均值 5.50 vs 前一窗口 4.66（**+18.0%**）
- per-budget CE 窗口 0.292→0.389（+33%），但 k=8 单批最低 0.024（部分样本已近零误差）⇒ 不是整体崩，是"部分样本变差 + 部分变极好"的再分配
- grad_norm 间或 105（早段 77），Trainer 默认 clip=1.0 生效；`monitor_loss.py` 判 **DIVERGING**
- 判断：更可能是续训 LR 偏高（run1 结束时余弦≈0，run2 重新 warmup 到 1e-5 把模型推出原极小点），而非数据/实现问题
- 处置：**不干预**（遵循"不中途改超参重启"），跑完 2 epoch（05:50，余弦归零）；已发邮件知会
- 对 run3 的连动：**以 06:00 评测定起点** —— normal acc ≥ run1(12.4%) 且 CE 不高于 run1 → init=run2/final（原计划）；若明显更差 → init=**run1/final**，混合集与流程不变

### 9.7 交付物入库

`tools/eval_mix/`（`1b3e48e`）：`build_img_eval_mix.py`、`prepare_mix.py`、`preflight_mix.py`、`run_eval_mix.sh`、`eval_after_run2.sh`、`eval_after_run3.sh`、`eval_mix_breakdown.py`、`make_report.py` + README。其中 `make_report.py` 已用 run1 产物自测通过（预算表/三组消融/bench 表格逐项对上）。

### 9.8 run3 起点策略（11-06 17:40 定稿）

| run2 评测（06:00 出数） | run3 起点 | 理由 |
|---|---|---|
| k8 acc ≥15%、normal−null(k8) ≥8pp、预算曲线非平 | `run2/final` | 预算机制有效，接着续训到更广数据 |
| 不达门（含 k8 acc 10–15%） | **`run1/final`** | 混合集实验与门无关；1 epoch 模型未经历后期 loss 回升；闲置 54h GPU 更浪费 |
| run2/final 缺失 / eval 链异常 | 不启动，邮件等指示 | 无可用产物 |

已加脚本保护：`run_eval_mix.sh` 的磁盘清理**永不删除 INIT 目录**（INIT=run1/final 时不会删 run1）。
用户若在周末回信要求改为预算课程或换方案，随时可 `pkill -f train.py` 停掉 run3（run3 从初始化状态开始，停掉无损失）。

### 9.9 数据重平衡（11-06 18:10，按用户反馈：图像任务等量、ImageNet 降权）

**反馈**：「ImageNet-100 的数据降低点；图像里面不同任务数据量尽可能相等，不要头重脚轻。」
原 `mix_train.jsonl` 图像行 97,845 = ImageNet 锚定 66,000（67%）+ 各评测任务 300–6,000，确实头重脚轻。

**做法**：`tools/eval_mix/rebalance_mix.py`（新增，`a079530` 入库）从既有 `mix_train.jsonl`
**按 source 抽样**生成 `mix_train_bal.jsonl`——不重新生成 prompt、不重新下载图片（行内容原样拷贝，
含 `messages_short`）。配额：每个图像任务 ≤2,000；textvqa 与 textvqa_train 同一任务合并 2,000（2:1 拆）；
**ImageNet-100 锚定 66,000 → 2,000**（与单个任务同量级，占比 2.0%）；文本源 82,000 行全保留
（文本任务量按原规格与知识覆盖需求保留，如要等量另议）。

| source | 原量 | 保留 | 占全量 |
|---|---:|---:|---:|
| imagenet100 | 66,000 | **2,000** | 2.0% |
| scienceqa / slake / docvqa | 6,000 / 5,000 / 4,451 | 各 2,000 | 各 2.0% |
| textvqa + textvqa_train | 3,997 + 2,000 | 1,333 + 667 | 2.0% |
| ai2d / mathvision | 2,000 / 2,000 | 各 2,000 | 各 2.0% |
| chartqa / we_math | 1,500 / 1,200 | 1,500 / 1,200 | 1.5% / 1.2% |
| mmstar / mathvista / mmmu / ocrbench / realworldqa / countbench | 999 / 698 / 600 / 600 / 500 / 300 | 全保留 | 1.0%–0.3% |
| 文本 7 源 | 82,000 | 82,000 | 80.1% |

- **新训练集 102,396 行**（文本 82,000 + 图像 20,396），逐行校验：图片缺失 0、assistant 前缀违规 0、
  `gt` 空 0、重复 id 0；抽 300 行 token：p50 148 / p99 689 / max 745，**0 条超 1024**；
  64 行真实走 `Dataset+Collator+expand_budget_rows` → **PREFLIGHT OK**
  （batch (64,829)、512×837×2560、valid target 2904 = expected 2904）
- 小任务的量无法再拉平（数据集本身只有 300–999 条可用），**差异只由可得数据量导致**，不是抽样偏置。

**run3 配置随之更新（`run_eval_mix.sh`）**：训练集换 `mix_train_bal.jsonl`；3,200 步/epoch ≈ 7.95 h；
epochs 不再写死，由「现在 → `DEADLINE`（默认 2026-11-09 09:00 训完，给评测链留 ~2h）」按 9.0 s/步倒推
（预计启动时 ~6.2 epoch ≈ 19,800 步）；`EPOCHS=` 仍可覆盖；新增 `DRY=1` 只打印配置不动磁盘。
旧 `mix_train.jsonl` 保留不删（可回退/对比）。

### 9.10 续训 LR 取舍（与 9.6 连动，11-06 18:10）

- 复核 9.6 的 loss 回升（用与 eval 同口径的 `per_budget_ce.jsonl`）：run2 四分段 CE =
  0.471 → 0.406 → **0.233** → 0.339（step 1400–2080 最低，之后 +45%）；Trainer 日志 `loss` 与之同比
  （loss/CE ≈ 27.5 恒定）⇒ **两个独立指标同向，回升是真的，不是单点噪声**。
- 分预算看：k=1 CE 0.373→0.437（+17%），**k=8 CE 0.210→0.318（+51%）** ⇒ 视觉通路（多 token 预算）退化
  比单 token 更明显；grad_norm 稳定在 70–105（clip=1.0），数据/实现已排除 ⇒ 最可能是**续训把峰值 LR
  重新抬到 1e-5，把 run1 已收敛的视觉通路推松**。run2 末尾余弦归零，最终 CE 大概率回落，
  最终 checkpoint 仍有竞争力（评测见 9.8 的门）。
- **run3 对策**：峰值 LR 减半——LLM 5e-6 / encoder 2.5e-6 / projector 5e-5，warmup 2%
  （run2 是 1e-5 / 5e-6 / 1e-4、3%）。数据与 LR 同时变，属整包实验；若周一结果不理想，回退顺序：
  先只回 LR（数据保留），再考虑数据。
