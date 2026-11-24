# 证据包 — 「压缩视觉表示 → 大模型（VLM/MLLM）下游任务」doc 历史检索（2026-11-24）

> 用途：为论文 v2.0 §4.5「表示的下游可接入性」（表 4）提供逐数字溯源；对应论文仓库之外的检索报告，原名 `tok448_work/DOC_HUNT_vlm_downstream.md`，此处原样入库。

> 产出：2026-11-24 ｜ 执行：doc-hunt（只读检索）｜ 目标读者：论文作者 / Lead
> 任务：在 `AssocVTC/doc` 与 `sr-diffusion-v3/doc` 的历史日志里，找出所有"把压缩后的视觉表示接入**大模型**做**下游任务**"的实验，判断哪些能写进论文、哪些必须排除。
> **纪律**：全程只读，未修改任何仓库/产物文件；本文件是唯一写出的文件。所有数字都带 `文件:行号`，找不到原句的一律标「未核实」。

**检索范围与去重（已实测，非估计）**
- 两个 checkout 的 `doc/` 在**共同路径上逐文件 md5 完全相同**：`AssocVTC/doc` 共 1,779 个文件，全部是 `sr-diffusion-v3/doc` 的子集（1,779/1,779 identical，differ=0）；`sr-diffusion-v3` 多出 247 个文件，集中在 2026-11-14 之后。
  - ⇒ 下文一律给 `AssocVTC/doc/...` 路径，同路径在 `sr-diffusion-v3/doc/...` 内容逐字相同；只有 11-14 之后的文档（`REPORT_tok448_2stage.md` 等）只在 `sr-diffusion-v3` 下有。
- 日期覆盖：`AssocVTC/doc` 2026-08-26 → 2026-11-13；`sr-diffusion-v3/doc` 2026-08-26 → 2026-11-24。
- grep 关键词（`--include='*.md'`）：`VLM`、`LLaVA`、`Qwen`、`下游`、`downstream`、`caption`、`VQA`、`8B`、`vlm8b`、`budget`、`finetune`、`CE`、`loss`、`tok448`、`tok256`、`BLEU`、`CIDEr`、`judge`、`线性探针`。命中文件全表见附录 A。

**证据等级（下文表格用）**
- **A**：真实产物（eval.md/json、逐样本 jsonl、训练日志）在本地可查，数字可复核。
- **B**：有真实报告 + 列出产物路径，但产物已删/在远端不可达，只能信报告数字。
- **C**：只有报告文字，无独立产物。
- **D**：设计/计划（`DESIGN_*`/`PLAN_*`），**没有跑过，不能当实验结果**。

---

## 0. 结论先行（TL;DR）

1. **doc 历史里只有一个真正跑过的 VLM/MLLM 下游实验：vlm8b。**
   它是 `AssocVTC 448²/p32/960M 阶段-2 编码器`（z_s=32 token×1744 维）→ 投影成 **8 个 visual token** → 接 **Qwen3.5-4B 文本主干**（**不是** 8B，见 §2.1；Qwen 原生 vision tower 被删掉，不喂 `pixel_values`），全参 SFT。三次 run（run1/run2/run3）都有真实数字，run3 还有 11-09 的逐样本重评（2,400 heldout + 1,300 train）。**其余 30 余个 VLM/caption/VQA 命中文件全是设计、计划、数据盘点、基础设施报告或与下游无关的像素实验。**
2. **它只能支撑一个"弱版本"的主张**：
   - （成立）压缩后的 8 个 token 经一个线性投影就能被 4B LLM 消费并参与语言生成，**不是摆设**：ImageNet-100 100 类闭集 k8 acc **28.8%**（chance 1%），同模型 null（视觉置零）**8.4%**、shuffle（错图）**0.0%**，CE 0.3719 vs 0.770 vs 1.227（run2）。
   - （成立但很弱）在 15 个公开图像 QA 源上能答出高于随机的分数（slake 71.7%、scienceqa 74.2%、mmstar 51.7%、mmmu 32.5%、mathvista 33.3%、realworldqa 46.7%），但**多数源的 `vision_gain≈0`**——把图去掉分数不降，说明高分主要来自文本先验/记忆，不是"读图"；细粒度 OCR/图表/文档只有 1.7%–10.8%。
   - （不成立/无证据）"压缩表示足以支撑下游视觉-语言任务"这个强主张**没有证据**：没有 caption/BLEU/CIDEr/judge 任何生成指标；**没有**未压缩 patch token、原生 Qwen 视觉塔或其他压缩器的同 LLM 对照；编码器**没有被冻结**（lr 5e-6/2.5e-6 全程可训），与论文"仅用重建信号训练的编码器"口径有 gap；预算 k 曲线全平。
3. **对照组只有"视觉通道消融"和"token 冗余消融"，没有"压缩对照"。**
   normal/null/shuffle 测的是"模型有没有用图"；mean/repeat 测的是"8 个 token 彼此是否冗余"；**都不是**"压缩表示 vs 未压缩"的 baseline。所以**不能**写"压缩表示≈未压缩表示"。
4. **必须标注的撤回/降级数字（4 条）**：
   - run3 报告的 §0.2/§2 的 acc 与「null > normal ⇒ 视觉 token 有害」结论：**同日 13:45 撤回**，原因是 greedy 协议把含答案的整段文本喂给 `generate`（`REPORT_budget_screen.md:38-48`、`:281`）。
   - run3 §3「每源 25 样本」分任务表：旧协议口径，已被 11-09 逐样本筛查的「每源 120 样本、修正协议」表取代（`REPORT_budget_screen.md:56, :284`）。
   - run1 首版 acc 26–31%：val jsonl 按类排序、`--limit 300` 只评到前 3 类，**已撤回**（`REPORT_vlm8b_1ep.md:8`）。
   - run2 的 28.8% 也是**旧 greedy 协议**口径（run2 ckpt 已删，无法重算；`REPORT_budget_screen.md:48, :284`）；但**CE 不受该 bug 影响**，可作为跨 run 可比的量。
5. **论文当前确实缺这一节**：10-26 论文快照摘要只有 ImageNet-100 像素重建 + 复杂度（`paper_final/snapshot/00_摘要.md:10`）；项目目标文档明写 Phase 2 下游是**唯一验收**（`2026-08-28/GOAL_compression_for_nlp.md:35, :48`），而它只被 vlm8b 试过、且没有按冻结编码器口径跑，之后 11-10 起整个项目转回像素/视频方向，**run4 建议从未执行**（`REPORT_budget_screen.md:245-273`，无后续 run4 文档）。

---

## 1. 候选清单表

「真跑过?」= 有真实训练/评测执行；「有数值?」= 有具体指标数字；「有对照?」= 有同预算/同协议的对照臂；「被推翻?」= 有后续文档撤回或降级该数字。

| 日期 | 文件 | 大模型 | 任务/指标 | 一句话结论 | 真跑过? | 有数值? | 有对照? | 被推翻? |
|---|---|---|---|---|---|---|---|---|
| 08-28 | `2026-08-28/GOAL_compression_for_nlp.md` | Qwen（计划） | Phase2 中文描述（唯一验收） | 目标文档：下游生成是唯一验收，像素重建是信息保持探针 | ✗ | ✗ | — | — |
| 08-28 | `2026-08-28/DESIGN_blip2_v3.md`、`REPORT_v3_blip2_experiment.md` | 无 LLM（下游=pixel head） | K=64 压缩 + 像素还原 | BLIP-2 式 K 压缩，**下游是像素头不是 LLM** | ✓ | ✓ | 有 K 探针 | 与 LLM 下游无关 |
| 09-17 | `2026-09-17/ANALYSIS_mm_datasets_inventory.md` + `data/dataset_recon/groupA_qa.md`…`groupD_*` | — | 12 个多模态数据集盘点（含 LLaVA-NeXT-Data） | 只有数据清单/字段核对，**没有训练或评测** | ✗ | ✗ | — | — |
| 09-22 | `2026-09-22/PLAN_paper_experiments.md` | 计划（分类 head / caption head） | E8「同 bpp 下游语义任务」Acc/F1/mAP | 计划"最值钱的一张图"，**从未执行**（10-19 仍写作"下一步"） | ✗ | ✗ | 计划有 | — |
| 10-16 | `2026-10-16/REPORT_alit_adaptive.md`（E-C1） | **无 LLM**（线性探针） | ImageNet-100 100 类 top-1 | 冻结 z_s 前缀线性分类，**探针分辨力不足，判"数据不足"** | ✓ | ✓ | 等平均 token 预算 | 不是 LLM 下游 |
| 10-16 | `2026-10-16/PLAN_alit_adaptive_experiments.md` | Qwen（计划） | E-C2 Phase2 caption/幻觉 | 计划有，**从未做**（"本项目一个 caption 指标都没跑过"） | ✗ | ✗ | — | — |
| 10-30 | `2026-10-30/REPORT_qwen_vision_pca_probe.md` | 无 LLM（Qwen3.8-27B 视觉塔 + 重建探针） | 像素 PSNR；附带 ridge 线性探针 91.14% | 探"Qwen 视觉塔"可解码性，**不是本论文表示** | ✓ | ✓ | 有 r=20/234/754/5120 四臂 | 与本论文表示无关 |
| 11-05 | `2026-11-05/DESIGN_8budget_vlm_finetune.md` | Qwen3.5-4B（计划） | ImageNet-100 100 类，CE/acc(k) | 方案 v1，文件头自述**未开跑** | ✗ | ✗ | 设计了 A/B/C/D/E 臂 | — |
| 11-05 | `2026-11-05/DESIGN_parallel_budget_vlm.md` | Qwen3.5-4B（分析） | — | 纯分析，文件头自述**未开跑** | ✗ | ✗ | — | — |
| 11-06 | `2026-11-06/DESIGN_looped_readout_vlm8b.md` | Qwen3.5-4B（计划） | 预算=循环步数 | 方案 v1，自述**未实现** | ✗ | ✗ | 有候选臂设计 | — |
| 11-06 | `2026-11-06/PLAN_budget_curriculum_vlm8b.md` | Qwen3.5-4B（计划） | 5 阶段课程 | 条件触发计划（"若 13:00/15:00 判 PLATEAU 则弃 run2 重训"），**未执行**；实际走 run3 混合集路线 | ✗ | ✗ | — | — |
| 11-06 | `2026-11-06/PLAN_eval_aligned_mixture.md` | — | 数据规格 | run3 数据 v2 的前身（规格 v1 配比与最终 122,259 行不同），自身无结果 | 部分 | ✗ | — | — |
| **11-06** | **`2026-11-06/REPORT_vlm8b_1ep.md`** | **Qwen3.5-4B** | **ImageNet-100 100 类；CE/acc(k)** | **1 ep：k8 12.0% / CE 0.4666；视觉通路真实（null 1.6%）但严重欠训** | ✓ | ✓ | normal/null/shuffle | 首版 acc 26–31% 已撤回 |
| 11-06 | `2026-11-06/REPORT_ds_e1_freeze.md` | — | 无下游指标 | DeepSpeed 冻结生命周期修复（纯基础设施） | ✓ | ✓ | 有 stage1/2/3 对照 | — |
| **11-06** | **`2026-11-06/REPORT_vlm8b_2ep.md`** | **Qwen3.5-4B** | **ImageNet-100 100 类** | **3 ep：k8 28.8% / CE 0.3719；null 8.4%、shuffle 0.0%** | ✓ | ✓ | normal/null/shuffle | acc 旧协议；ckpt 已删 |
| **11-09** | **`2026-11-09/REPORT_vlm8b_run3.md`** | **Qwen3.5-4B** | **mix_heldout 15 图像源 + 5 文本源 + ImageNet-100** | **首个广任务下游：预算机制不成立、视觉"看似有害"** | ✓ | ✓ | normal/null/shuffle | **§0.2/§2 已撤回** |
| **11-09** | **`2026-11-09/REPORT_budget_screen.md`** | **Qwen3.5-4B** | **同上 + 逐样本 5 种视觉消融** | **修正协议重评：视觉略有益；预算条件性有效；编码器容量封顶** | ✓ | ✓（2400+1300 行） | normal/null/shuffle/mean/repeat | 本身是修正文档 |
| **11-09** | **`2026-11-09/screen/tables_heldout_in100_train.md`** | **Qwen3.5-4B** | **自动生成的权威数字表** | 分源 acc/CE/vision_gain/budget_gain 全表 | ✓ | ✓ | 同上 | 同 run3 修正口径 |
| 11-10 | `2026-11-10/REPORT_intermediate_state_base.md` | — | 视频重建 | 自述「未测与下游 VLM 任务（QA/OCR）的关系」 | ✓ | ✓ | — | 与本论文表示无关 |
| 11-24 | `sr-diffusion-v3/doc/2026-11-24/REPORT_encoder_decodability_vendors.md` | 四家厂商 ViT（无 LLM） | 像素 PSNR/SSIM | 厂商视觉塔可解码性；**明确声明判据不用下游** | ✓ | ✓ | 四臂同预算 | 与本论文表示无关 |

**补充（非实验但对论证有用）**
- `2026-11-24/PLAN_encoder_decodability.md:6`：「判据不用下游分类/检索（那些只反映语义，不反映像素细节）」——项目后期**主动选择**不做下游判据，说明 doc 里没有更晚的下游实验。
- `2026-10-19/REPORT_k8sweep.md:319, :335`、`REPORT_savoias_complexity.md:236, :307`：K-sweep/复杂度实验收尾时都把「任务保真（E8）」列为**下一步**，证明 E8 从未执行。
- `2026-11-10/REPORT_intermediate_state_base.md:224, :257`：vlm8b 之后同一分支的视频工作自述「未测下游 VLM」。

---

## 2. 排名前 3 的可用证据

排序依据：对「压缩表示能否被大模型用于下游任务」这一主张的**证据强度 × 可引用性**。

- **证据 A（主）**：vlm8b run3 + 11-09 逐样本筛查 —— 唯一覆盖"通用下游任务"的实验，有修正协议、有分源、有视觉消融。**用途：证明"能被 LLM 消费、能答出高于随机的分数"，并诚实暴露边界。**
- **证据 B（强数字）**：vlm8b run2 ImageNet-100 —— 8 个 token 的**类别可判别性**最硬的数字（28.8% vs null 8.4% vs shuffle 0，CE 0.3719 vs 0.770 vs 1.227）。**用途：主表里最能量化"token 里有信息且被用"的一行。**
- **证据 C（同口径补充 + 边界）**：run3 修正协议下的 ImageNet-100（29.2%→32.4%）+ k 曲线全平 + 容量论证。**用途：给出同协议可比数字，并解释为什么"预算/压缩"这条线不成立。**

> 三次实验的**模型与接口完全相同**（同一分支 `feat/vlm8b-v1`、同一编码器 ckpt、同一投影、同一 Qwen3.5-4B）；差别只在数据/epoch/LR。因此 A/B/C 可以放在同一节里。

---

### 2.1 证据 A（主证据）— vlm8b run3 + 11-09 逐样本筛查：8 个 token 接 Qwen3.5-4B 跑 15 个公开图像 QA 源

#### A.1 完整协议

**模型 / 表示如何接进去**（这一点必须先说清，因为任务描述里的"8B / Qwen2.5-VL-7B / LLaVA-NeXT"都不对）：
- LLM：**Qwen3.5-4B**（`Qwen3_5ForConditionalGeneration` 的**文本主干**）。文档标题即写 Qwen3.5-4B（`REPORT_vlm8b_1ep.md:1`）；权重路径 `/root/qwen_models/Qwen3.5-4B`（`vlm8b_eval/run_args.json:3`）；LM hidden=2560（`vlm8b_snapshot_20261109/code/model.py:47`）。"vlm8b"是**项目代号（vlm + 8 budget）**，不是"8B 参数"。
- 视觉塔：**不是** Qwen 的 vision tower。视觉塔是 AssocVTC 阶段-2 编码器 `exp448/B/ckpt_best.pt`（448²/p32，32 个 special token × 1744 维；`run_args.json:2`；设计文档自述"448²/p32/960M 阶段2 编码器"`DESIGN_8budget_vlm_finetune.md:1`）。Qwen 自带 vision tower 被显式删除、**不喂 `pixel_values`**（`model.py:13, :264-265`）。
- 压缩表示怎么接：`z_s(32×1744)` → 逐 token `LayerNorm+Linear(1744→640)` → **4 个 640 拼成 1 个 2560** → `8×2560 = 20,480` 值 → `LayerNorm(2560)` → 作为 `inputs_embeds` 的**前缀**拼在文本 embedding 前（`model.py:152-166` 投影；`model.py:366` `torch.cat([vis[b,:k], text_embeds[b,:n]])`；设计见 `DESIGN_8budget_vlm_finetune.md:76-83`）。预算 k = 前缀取前 k 个 token。
  - ⚠️ 换算：8 个 LM token **吃掉全部 32 个 z token**（4:1 重分组），**不是**"从 32 个里截断到 8 个去接 LLM"。k∈1..8 才是真正的截断维度（`REPORT_budget_screen.md:32, :236` 明确指出）。8×2560=20,480 值/图，是 448²×3=601,344 像素值的 3.4%（本行为推算，非原文）。
- 训练：**全参 SFT**（LLM + 投影 + **编码器也可训**）。`freeze_encoder_epochs=0`（`run_args.json:17`）；lr_llm=1e-5 / lr_proj=1e-4 / lr_encoder=5e-6（`run_args.json:11-13`；run3 减半为 5e-6 / 5e-5 / 2.5e-6，`REPORT_vlm8b_run3.md:35`）。⇒ **不是"冻结编码器 + 训 adapter"**，这一点是最大边界（见 A.4）。
- run3 训练设置：ZeRO-2，bf16，micro 2×accum 8×2 GPU（全局 batch 32），grad-ckpt；3,821 步/epoch × **3.0 epoch = 11,463 步**，52.4h；数据 `mix_train_bal2.jsonl` **122,259 行**（文本 82,000 + 图像 40,259，其中 ImageNet-100 仅 3,000）（`REPORT_vlm8b_run3.md:32-37`）。
- 评测数据：`mix_heldout.jsonl`：15 个图像源 + 5 个文本源（`REPORT_budget_screen.md:56`）；逐样本筛查每源 120 行 = 2,400 行（图像 1,800 + 文本 600），另有 300 行随机混采子集（`REPORT_budget_screen.md:56, :62`）。
- 指标定义（原文）：`vision_gain = CE(视觉置零) − CE(正常)`（越大越说明"图被用"）、`spec_gain = CE(错图) − CE(正常)`、`mean_cost = CE(8 token 取均值) − CE(正常)`（≈0 ⇒ token 冗余）、`repeat_cost`、`budget_gain = CE(k1) − CE(k8)`（全部定义成"CE 降低=正数更好"，`REPORT_budget_screen.md:22`）。另有 answer-only CE（teacher forcing）与 greedy exact-match acc。
- **没有**的指标：caption/BLEU/CIDEr/ROUGE、检索 R@k、LLM-as-judge、幻觉率 —— 全 doc 无此类实验（附录 A grep 证据）。

#### A.2 逐个数字（一律标来源）

**总表（分层抽样，每源 120）** — `screen/tables_heldout_in100_train.md:5-11`；报告版 `REPORT_budget_screen.md:14-20`

| dataset | 行类型 | n | CE k1 | CE k8 | budget_gain | vision_gain | mean_cost | acc k1 | acc k8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heldout | 图像行 | 1,800 | 0.741 | 0.742 | −0.002 | **+0.016** | −0.005 | 33.8% | 34.1% |
| heldout | 文本行 | 600 | 0.248 | 0.246 | +0.003 | +0.003 | +0.002 | 65.8% | 66.0% |
| in100 val | 图像 | 400 | 0.461 | 0.463 | −0.002 | **+0.343** | +0.100 | **30.0%** | **34.5%** |
| train | 图像行 | 950 | 0.179 | 0.167 | +0.012 | **+0.100** | +0.025 | 73.9% | 75.8% |
| train | 文本行 | 350 | 0.041 | 0.041 | +0.000 | +0.002 | −0.000 | 92.6% | 92.6% |

（来源：`tables_heldout_in100_train.md:7-11`；报告版同数字 `REPORT_budget_screen.md:14-20`。`Δacc` 共 0.2%/0.2%/4.5%/1.9%/0.0%。）

**分源 acc / vision_gain（图像源，每源 120 行；按 budget_gain 排序）** — `tables_heldout_in100_train.md:17-36`（报告版 `REPORT_budget_screen.md:66-82`）

| source | 任务形态 | acc k1 | acc k8 | CE k1→k8 | vision_gain | budget_gain | 读法 |
|---|---|---:|---:|---|---:|---:|---|
| mmlu_pro | 文本 MCQ | 59.2% | 60.0% | 0.344→0.336 | +0.004 | +0.008 | 文本源（本不该用图） |
| ocrbench | 短答 VQA | 5.8% | 5.8% | 2.018→2.011 | −0.001 | +0.007 | 近随机 |
| slake | 短答 VQA | 71.7% | 73.3% | 0.140→0.137 | **+0.221** | +0.003 | **唯一"图明显被用"的通用源** |
| ceval | 文本 MCQ | 60.0% | 60.0% | 0.285→0.283 | −0.006 | +0.003 | 文本源 |
| mathvista | 短答 VQA | 33.3% | 34.2% | 0.482→0.479 | +0.001 | +0.003 | 高于 4 选 1（25%），但图≈没用 |
| cmmlu | 文本 MCQ | 64.2% | 64.2% | 0.245→0.243 | −0.001 | +0.002 | 文本源 |
| countbench | 短答 VQA | 16.7% | 17.5% | 0.390→0.388 | +0.028 | +0.002 | 计数，弱 |
| textvqa | 短答 VQA | 10.8% | 10.8% | 2.191→2.190 | −0.010 | +0.001 | **近随机** |
| docvqa | 短答 VQA | 1.7% | 1.7% | 1.871→1.870 | +0.077 | +0.001 | 有点视觉信号但答不出 |
| realworldqa | 短答 VQA | 46.7% | 46.7% | 0.284→0.283 | +0.003 | +0.001 | 图≈没用 |
| mmlu_redux | 文本 MCQ | 76.7% | 76.7% | 0.168→0.167 | +0.007 | +0.001 | 文本源（最高） |
| mmlu | 文本 MCQ | 69.2% | 69.2% | 0.199→0.199 | +0.013 | +0.000 | 文本源 |
| scienceqa | 图 MCQ | 74.2% | 74.2% | 0.109→0.109 | +0.004 | −0.000 | **图≈没用（靠先验/记忆）** |
| mmstar | 图 MCQ | 51.7% | 50.8% | 0.261→0.261 | +0.001 | −0.000 | 同上 |
| chartqa | 短答 VQA | 1.7% | 0.8% | 1.199→1.201 | +0.001 | −0.002 | **近随机** |
| realworldqa | 图 MCQ | 45.0% | 48.3% | 0.274→0.277 | −0.012 | −0.002 | 图略有害 |
| we_math | 短答 VQA | 47.5% | 45.8% | 0.435→0.440 | +0.008 | −0.005 | — |
| mmmu | 图 MCQ | 32.5% | 33.3% | 0.407→0.414 | −0.017 | −0.007 | 高于 25%，图≈没用 |
| ai2d | 图 MCQ | 55.0% | 55.0% | 0.250→0.258 | −0.021 | −0.008 | 图略有害 |
| mathvision | 短答 VQA | 13.3% | 12.5% | 0.796→0.814 | −0.047 | −0.018 | 近随机、图有害 |

**同一 300 行随机混采子集（修正后 greedy 协议）** — 产物 `vlm8b_snapshot_20261109/out/eval_run3_fix_*/eval.md:7-14`；报告 `REPORT_budget_screen.md:38-48`
- normal：acc k1 **48.67%** → k8 **49.33%**；CE 0.5839 → 0.5847（`eval_run3_fix_normal/eval.md:7-14`）
- null（视觉置零）：acc 48.33%，CE 0.6054（`eval_run3_fix_null/eval.md:7-8`）
- shuffle（错图）：acc 48.67%，CE 0.6145（`eval_run3_fix_shuffle/eval.md:7`）
- ⇒ 修正后 **normal CE 比 null 低 0.021、比 shuffle 低 0.030**，视觉"略有益"（`REPORT_budget_screen.md:47`）。acc 三者几乎一样（48.3–48.7%），**acc 分辨率不足以分辨**。
- ⚠️ 这个 300 行混采子集与上面的 2,400 行分层表**不可换算**：分层抽样"每源等量，与 heldout 真实配比不同"（`REPORT_budget_screen.md:54`），且 300 行噪声大（±~6pp）。

**逐 bin（"图越有用，预算才越有用"，2,400 行）** — `REPORT_budget_screen.md:94-107`
- vision ∈ (−0.1,−0.02] 的 262 行：budget **−0.038**，acc 1.5%→0.8%
- vision ∈ (0.02,0.1] 的 346 行：budget +0.014，acc 32.1%→34.7%
- vision > 0.3 的 78 行：budget **+0.044**，acc **16.7%→20.5%**
- 四象限：vision 有效仅 730/2,400 行（30%），其中只有 44% 同时 budget 有效（`REPORT_budget_screen.md:84-92`）

#### A.3 能不能支撑「压缩表示可支撑下游任务」

**能支撑（可写）**：
- 8 个 2560 维 token 经**一个线性投影**即可被 4B LLM 读取并参与生成，无需改动 LLM 结构、无需原生视觉塔（`model.py:13, :152-166, :366`；`REPORT_vlm8b_1ep.md:1`）。
- 模型在**多个公开图像 QA 源**上给出高于随机的分数（MMMU/mathvista 类 4 选 1 ≈25% → 32–34%；slake 71.7%；mmstar 51.7%；scienceqa 74.2%；ai2d 55%）（`tables_heldout_in100_train.md:19-35`），说明"表示 → 语言输出"链路**通**。
- 视觉通道**被使用**（不是完全绕过图）：ImageNet-100 上 vision_gain **+0.343**、`mean_cost` **+0.100**（token 有分工）（`tables_heldout_in100_train.md:9, :140`）；slake vision_gain **+0.221**（`:19`）；混评 2,400 行 CE normal < null < shuffle（`REPORT_budget_screen.md:38-48`）。

**不能支撑（不可写）**：
- **不能说"压缩表示足以支撑下游视觉-语言任务"**作为一般结论。通用 VQA 源里只有 slake（+0.221）和 docvqa（+0.077）有可辨视觉增益，其余图像源 `vision_gain` 都在 −0.05~+0.03 之间（≈0，最小为 mathvision −0.047），意味着**模型不靠图也能答对**（`REPORT_budget_screen.md:26-32, :66-82`）。
- **不能说"压缩代价小"**：没有同 LLM 的未压缩对照（见 §2.4）。
- **不能说"冻结编码器"**：编码器全程可训（`run_args.json:13, :17`）。
- **不能说"预算/截断有效"**：k 曲线 20 个 (source,图/文) 组合 CE 从 k1→k8 变化全部 ≤0.02，多数 ≤0.005（`REPORT_budget_screen.md:26-27`）；in100 的 `argmin_k=2`（`tables_heldout_in100_train.md:146`）。
- **不能引用任何 caption/VQA 生成分数**：doc 里根本不存在 BLEU/CIDEr/judge 数据。

#### A.4 边界（能说什么 / 不能说什么）

1. **模型是 4B 文本主干 + 外接视觉编码器，不是端到端 VLM**；且是**全参 SFT**（4B + 480M 编码器），通用文本能力被严重破坏（run1：base 75% → 1.7%/0%，`REPORT_vlm8b_1ep.md:43-49`；run3 混合集修复到 ~53%，`REPORT_vlm8b_run3.md:96-100`）。这是"为了打通下游而牺牲通用能力"的代价，论文若引用需说明。
2. **编码器不是论文主表模型**：vlm8b 用 448²/p32 **960M** 阶段-2 编码器；论文主表是 224² **240M**（447M 档在 tok448 是 487M）。⚠️ 跨模型族，不能直接宣称"论文的 K=32 表示也能跑下游"。
3. **数据/标签噪声**：ImageNet-100 SFT 数据是自蒸馏（标签 ~11% 错，理论上限 ~89%；`DESIGN_8budget_vlm_finetune.md:200`）；`mix_heldout` 的来源/构造在 doc 内没有完整规格（只有 `PLAN_eval_aligned_mixture.md` 的数据格式），**数据泄漏未独立验证**（报告只说"val 独立于训练"，`REPORT_vlm8b_2ep.md:9`）。
4. **单 seed、小样本**：acc 来自 n=120/源 或 n=250/300，未见多 seed 方差；`REPORT_budget_screen.md:255` 自己承认"每源 25 行 ≈ 4pp 抖动，下轮建议 ≥100"（这一轮已做到 120，但仍单 seed）。
5. **run3 的训练集里 2/3 是文本行**（82,000/122,259），且文本行的视觉/预算贡献精确为 0（`REPORT_budget_screen.md:30, :125`）⇒ "图像能力"只由 ~8% 训练行监督（`REPORT_budget_screen.md:157`），这解释了为什么视觉增益普遍很弱。

#### A.5 撤回链（引用时必须知道）

| 被撤回的说法 | 原文位置 | 撤回/修正 | 来源 |
|---|---|---|---|
| run3「null(28.0%) > normal(22.7%) ⇒ 视觉 token 有害」 | `REPORT_vlm8b_run3.md:17-18, :43-47` | greedy 协议把含答案文本喂给 generate ⇒ 大量空串；修正后 normal 48.7→49.3、null 48.3、shuffle 48.7，CE normal 0.584 < null 0.604 ⇒ 视觉**略有益**，不是有害 | `REPORT_budget_screen.md:6-8, :38-48, :281-282` |
| run3 §3 分任务表（每源 25 样本） | `REPORT_vlm8b_run3.md:50-74` | 旧协议口径，全部作废；改用每源 120 样本修正表 | `REPORT_budget_screen.md:56, :284` |
| run3 in100 k8 22.8% | `REPORT_vlm8b_run3.md:24, :85` | 旧协议；修正后 29.2%→32.4%（n=250） | `REPORT_budget_screen.md:46, :284` |
| run1 首版 acc 26–31% | `REPORT_vlm8b_1ep.md:8` | val jsonl 按类排序，只评到前 3 类；改为 seed42 随机抽样 | `REPORT_vlm8b_1ep.md:8-9` |
| run2 acc 28.8% | `REPORT_vlm8b_2ep.md:8-9` | **未被正式撤回，但也是旧协议**；run2 ckpt 已删无法重算，跨 run 只能用 CE 比 | `REPORT_budget_screen.md:48, :284`；`REPORT_vlm8b_2ep.md:64, :90` |

> 另：run3 的通用能力 bench（弱智吧 5%/算术 70%/常识 85%）被 `REPORT_budget_screen.md:283` 提示"该 bench 脚本不是同一生成协议，数值只能同口径内部比"——**不要**与 base 75% 直接并列。

---

### 2.2 证据 B（强数字）— vlm8b run2：8 个 token 在 ImageNet-100 上的类别可判别性

#### B.1 协议

- 同 §2.1 的模型/接口；训练 = run1（in100 自蒸馏 `train_nothink_all.jsonl` **121,689 行**，1 epoch，3,803 步，8.93 s/step，9.44h，2×A800，ZeRO-2；`REPORT_vlm8b_1ep.md:3`）→ run2 **从 run1/final 续训 2 epoch**（共 7,606 步，~8.9 s/step；`REPORT_vlm8b_1ep.md:66-67`）。⇒ 合计 **3 epoch、in100-only**。
- 任务：ImageNet-100 **100 类闭集**，prompt 后让模型生成类别名，greedy exact-match 与 GT 类名比对（`DESIGN_8budget_vlm_finetune.md:183`）；chance = 1%（`REPORT_vlm8b_1ep.md:37`）。
- 评测：`val_nothink_all.jsonl`，**n=250 随机子集，seed 42**（`REPORT_vlm8b_1ep.md:9`）；teacher-forced answer CE + greedy acc；预算 k=1..8。
- 视觉消融：normal / null（vis 置零）/ shuffle（batch 内错位）（`REPORT_vlm8b_2ep.md:36-42`）。

#### B.2 逐个数字

**主表（n=250，seed42；旧 greedy 协议）** — `REPORT_vlm8b_2ep.md:23-32`；run1 同表 `REPORT_vlm8b_1ep.md:14-23`

| k | run1 CE | run1 acc | **run2 CE** | **run2 acc** |
|---:|---:|---:|---:|---:|
| 1 | 0.4989 | 12.40% | 0.4015 | 21.20% |
| 2 | 0.4684 | 14.00% | 0.3742 | 31.60% |
| 3 | 0.4683 | 13.60% | 0.3757 | 26.40% |
| 4 | 0.4678 | 11.60% | 0.3734 | 26.40% |
| 5 | 0.4683 | 10.40% | 0.3737 | 26.40% |
| 6 | 0.4669 | 12.80% | 0.3732 | 29.60% |
| 7 | 0.4673 | 11.60% | 0.3721 | 30.00% |
| 8 | **0.4666** | **12.00%** | **0.3719** | **28.80%** |

- ΔCE(k8−k1)：run1 **−0.0323**、run2 **−0.0296**；Δacc：run1 −0.4pp、run2 **+7.6pp**（`REPORT_vlm8b_2ep.md:34`）。
- 判定门：k8 acc 28.8% ≥15% ✓、normal−null(k8) 20.4pp ≥8pp ✓、曲线非平 ✓（`REPORT_vlm8b_2ep.md:13-14`）。**注意"曲线非平"后来被 11-09 修正协议推翻**（新协议下 k1→k8 48.67→49.33，基本平），当时是靠旧协议 + k2 异常点（31.6%，报告自己标 ±2.8pp 噪声、不宜过度解读，`REPORT_vlm8b_2ep.md:11`）。

**视觉消融（同一子集，CE 不受 greedy 协议 bug 影响）** — `REPORT_vlm8b_2ep.md:38-42`

| 视觉模式 | run1 CE k1/k8 | run1 acc k1/k8 | **run2 CE k1/k8** | **run2 acc k1/k8** |
|---|---|---|---|---|
| normal | 0.499 / **0.467** | 12.4% / 12.0% | 0.402 / **0.372** | 21.2% / **28.8%** |
| null（视觉置零） | 0.770 / 0.769 | 1.2% / 1.6% | 0.771 / **0.770** | 7.2% / **8.4%** |
| shuffle（错图） | 1.007 / 1.047 | 0.8% / 0.0% | 1.147 / **1.227** | 0.0% / **0.0%** |

- 报告判读：normal 28.8% vs null 8.4%（+20.4pp）vs shuffle 0.0% ⇒ 模型确实用视觉（`REPORT_vlm8b_2ep.md:10`）；"不是评测集污染（val 独立于训练）"（`:9`）。
- 对照标准答案"null 仍 8.4% > chance 1%"——说明无图时模型输出**类先验/常见类**；这既是消融的噪声底，也说明 8.4% 应作为该任务的"无语义底"而非 1%。

#### B.3 能不能支撑

**能**：这是全部 doc 里对"8 个 token 里确实装了类别信息、且 LLM 在用"最干净的一条证据——同模型、同数据、只有视觉输入三态不同，CE 差 0.398（normal vs null）与 0.855（normal vs shuffle），acc 差 20.4pp / 28.8pp。即使考虑旧协议，**CE 与方向都不受影响**。
**不能**：这是 **100 类闭集分类**，不是视觉-语言任务；不能外推到 caption/VQA/检索；不能说明"压缩代价小"（没有未压缩对照）；acc 是旧协议，不能与 run3 修正后的 32.4% 直接比大小。

#### B.4 边界

- run2 ckpt 已按磁盘预算删除（`REPORT_vlm8b_2ep.md:64, :90`），**无法重跑修正协议**；引用 acc 必须写"旧 greedy 协议口径"。
- 数据是自蒸馏标签（~11% 错，上限 ~89%，`DESIGN_8budget_vlm_finetune.md:200`），所以 28.8% 的分母/天花板都不干净；28.8% 相对 88.7% 上限仍很低。
- run2 的 "28.8%" 与 run3 的 "32.4%" 是**同一任务的两次不同训练**（run2 = 66k in100 锚定、run3 = 3k 锚定），run3 report 自述 run3 因锚定从 66k 砍到 3k 而 in100 能力被稀释（`REPORT_vlm8b_run3.md:24`；`REPORT_budget_screen.md:191` 说 run2 是 66k 时 CE 增益 +0.03）。⇒ 别把 28.8% 当成"run3 更差"，两个口径不可交叉。

---

### 2.3 证据 C（同口径补充 + 边界）— run3 修正协议 ImageNet-100 + k 曲线全平 + 容量论证

#### C.1 协议

- 模型/接口同 §2.1；评测 `val_nothink_all.jsonl` n=250 seed42 / 400 行随机；**修正后 greedy 协议**（只喂 prompt 到 answer 前，`--gen-protocol prompt`，`REPORT_budget_screen.md:38`）。
- 这条证据的价值在于：**它是唯一"修正协议 + 同模型 + 同任务"可引用的绝对 acc**。

#### C.2 逐个数字

- n=250：acc k1 **29.20%** → k8 **32.40%**（Δacc +3.2pp）；CE 0.4740 → 0.4749（`vlm8b_snapshot_20261109/out/eval_run3_in100_fix/eval.md:7-14`；报告 `REPORT_budget_screen.md:46, :284`）。
- n=400（随机）：acc k1 **30.0%** → k8 **34.5%**（k8 修好 36 行/弄坏 18 行）；CE 0.461→0.463（平）；`vision_gain +0.343`、`mean_cost +0.100`、`spec_gain +0.288`（`screen/tables_heldout_in100_train.md:9, :140, :146`；`REPORT_budget_screen.md:179`）。
- "图有用"行：`vision>0.4` 的 46% 行 acc **51.1%→62.0%（+10.9pp）**；`vision>0.6` 的 118 行 **67.8%→81.4%**；`vision<−0.05` 的 19% 行 budget **−0.183**（`REPORT_budget_screen.md:183-190`）。
- 训练行（n=50）：CE **0.228→0.108**、acc **52%→72%**、mean_cost **+0.311**、budget **+0.120**（`tables_heldout_in100_train.md:201`；`REPORT_budget_screen.md:191`）；120/组复核 n=87：CE 0.170→0.075、acc 65.5%→79.3%（`REPORT_budget_screen.md:20`）。
- k 曲线：in100 `argmin_k=2`（`tables_heldout_in100_train.md:146`）；heldout 20 个 (source,图/文) 组合 k1→k8 CE 变化 ≤0.02、多数 ≤0.005（`REPORT_budget_screen.md:26`）。
- 容量论证（结论的根）：8 个 visual token = 32 个 z token 的 4:1 重分组，k∈1..8 只决定"暴露多少 z token"，**不改变 z_s 的信息上限（32 token）**（`REPORT_budget_screen.md:32, :236-241`）；细粒度任务（chartqa/docvqa/textvqa/ocrbench/mathvision）细节"根本不在"这 32 token 里 ⇒ acc 1%–11%（`REPORT_budget_screen.md:240`）。

#### C.3 能不能支撑 / 边界

**能**：给出"压缩表示 → LLM"在 100 类判别上的**同协议绝对成绩 32.4–34.5%**（chance 1%、无语义底 8.4%），并量化"图有用行还能再涨 10.9pp"。
**不能**：k 曲线平 ⇒ **不能用这组数据论证"多给 token 更好/预算可截断"**；run3 的 in100 能力被混合集稀释（22.8% 旧→32.4% 新，仍低于 run2 旧协议 28.8% 所暗示的水平），所以**in100 主数字应优先用 run2（CE 0.3719）或 run3 修正（32.4%）并注明训练数据不同**。

---

### 2.4 横向：三次 run 的视觉消融（"对照组是否成立"的唯一依据）

| run | 评测集 | n | normal acc k8 | null acc k8 | shuffle acc k8 | normal CE k8 | null CE k8 | shuffle CE k8 | 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| run1 | in100 val | 250 | 12.0% | 1.6% | 0.0% | 0.4666 | 0.769 | 1.047 | `REPORT_vlm8b_1ep.md:32-34` |
| run2 | in100 val | 250 | 28.8% | 8.4% | 0.0% | 0.3719 | 0.770 | 1.227 | `REPORT_vlm8b_2ep.md:40-42` |
| run3 | mix 300 混采 | 300 | 49.33% | 48.33% | 48.67% | 0.5847 | 0.6042 | 0.6192 | `out/eval_run3_fix_*/eval.md:7-14` |
| run3 | in100 val（修正） | 250 | 32.4% | — | — | 0.4749 | — | — | `out/eval_run3_in100_fix/eval.md:14` |
| run3 筛查 | heldout 图像 2,400 分层 | 1,800 | 34.1%（k8） | fold-in 在 vision_gain | — | 0.742 | — | — | `tables_heldout_in100_train.md:7` |
| run3 筛查 | in100 400 | 400 | 34.5% | — | — | 0.463 | — | — | `tables_heldout_in100_train.md:9, :140` |

**对照组成立性判定（重要）**：

| 对照类型 | 是否存在 | 覆盖什么 | 缺什么 |
|---|---|---|---|
| 视觉通道消融（null/shuffle） | ✅ 三次 run 都有 | "模型有没有用图" | 不是压缩对照 |
| token 冗余消融（mean / repeat，8 token 均值广播 / 首 token 复制 8 份） | ✅ 仅 11-09 筛查（heldout/in100/train） | "8 个 token 彼此是否分工"（in100 mean_cost +0.100 / train +0.311 / 混评 −0.005，`REPORT_budget_screen.md:206-208`） | 不是"压缩 vs 未压缩" |
| 预算档位 k=1..8 | ✅ 三次 run | 前缀截断（同一 8 token 的嵌套前缀） | 不是独立压缩率；且 k 全平 ⇒ 无信息 |
| **未压缩 patch token（196/196）同 LLM** | ❌ 没有 | — | **无法量化压缩代价** |
| **Qwen 原生 vision tower 同 LLM** | ❌ 没有（tower 被删） | — | 无法说"只靠我们编码器就够" |
| **其他压缩器（平均池化/随机 token/PCA）同 LLM** | ❌ 没有 | — | 无法说"我们的压缩更好" |
| **更大 token 数（32 个 z token 1:1 喂进 LLM）** | ❌ 没有 | — | **无法判断"4:1 重分组"是否损失信息**（现有 8 token 已吃掉全部 32 token，见 §2.1 A.1 换算） |

> 结论：**对照足以支撑"视觉被使用"，不足以支撑"压缩表示与未压缩相当"**。论文若写"压缩不损失下游可用性"会被审稿人直接打回，因为缺上述 4 条 baseline 中的任意一条。

---

## 3. 明确排除的候选 + 理由

### 3.1 只有设计 / 计划，没有实验（不能当结果引用）
| 文件 | 自述状态（原文位置） |
|---|---|
| `2026-11-05/DESIGN_8budget_vlm_finetune.md` | 「状态: **方案 v1，待用户审核**（未开跑）」`:3` |
| `2026-11-05/DESIGN_parallel_budget_vlm.md` | 「状态: 分析（**未开跑**）」`:3` |
| `2026-11-06/DESIGN_looped_readout_vlm8b.md` | 「状态：**方案 v1，未实现，待实验验证**」`:3` |
| `2026-11-06/PLAN_budget_curriculum_vlm8b.md` | 条件触发计划（「若 13:00 与 15:00 两次巡检都判 PLATEAU，则放弃 run2…重训」`:3-5`），**从未执行**（run3 走的是混合集续训，`REPORT_vlm8b_run3.md:32`） |
| `2026-10-16/PLAN_alit_adaptive_experiments.md`（E-C2） | 「Phase 2 NLP 探针（**唯一验收，独立排期**）」`:276`；`REPORT_alit_adaptive.md:679` 自述「本项目一个 caption/幻觉指标都没跑过」 |
| `2026-09-22/PLAN_paper_experiments.md`（E8） | 「同 bpp 下游语义任务…**E8**」`:327-335`；10-19 仍列"下一步"（`REPORT_k8sweep.md:335`）⇒ never run |

### 3.2 基础设施，无下游指标
- `2026-11-06/REPORT_ds_e1_freeze.md`：DeepSpeed 冻结生命周期修复（IndexError 根因 + 验证），报告里唯一与 vlm8b 相关的是"E1 真冻结现在可用"（`:1-40`）与生产尺寸 encoder 显存/速度（`:246-271`）。**没有下游数字**。
- 但它有一条对论文有用的**事实**：vlm8b run2/run3 都显式 `--freeze-encoder-epochs 0`（`REPORT_ds_e1_freeze.md:277`）⇒ 证实编码器未冻结。

### 3.3 下游但是"非 LLM"或"非本论文表示"
| 文件 | 排除理由 |
|---|---|
| `2026-10-16/REPORT_alit_adaptive.md`（E-C1） | 下游是**线性分类探针**（冻结 z_s 前缀 mean-pool → IN-100 top-1），不是 LLM；且判「数据不足」：val_top1 **0.049–0.112**（chance 0.01）、train_top1 0.68–1.00、oracle 上界 0.2968、Δ 的 CI 全跨 0（`:35, :65, :660-680`）。其意义是"证明连线性探针都测不出 K 相关差异"，可作为**负面对照**引用，但不能当"下游可用"证据。 |
| `2026-10-30/REPORT_qwen_vision_pca_probe.md` | 对象是 **Qwen3.8-27B 视觉塔的 token**（不是本论文编码器），任务是 PCA 截断 + 像素还原；唯一下游类数字是"特征健全性"ridge 线性探针 IN-100 top-1 91.14%（`:74`），属**另一个模型的特征质量校验**，与本论文表示无关。报告自己写明边界「像素指标 ≠ 任务保真」`:166`。 |
| `2026-11-24/REPORT_encoder_decodability_vendors.md` | 四家厂商 ViT 的可解码性（像素 PSNR），**无 LLM、无下游任务**；`PLAN_encoder_decodability.md:6` 明确「判据不用下游分类/检索」。 |
| `2026-08-28/REPORT_v3_blip2_experiment.md` + `DESIGN_blip2_v3.md` | BLIP-2 式 K=64 压缩，但**下游是 pixel head**（"本项目下游暂为像素头"，`DESIGN_blip2_v3.md:22`），不接 LLM。 |

### 3.4 数据/目标文档，没有实验
- `2026-09-17/ANALYSIS_mm_datasets_inventory.md` + `data/dataset_recon/groupA_qa.md`…`groupD_render_traversability.md`：12 个数据集/202GB 的**只读盘点**（「只读抽样；未修改/删除任何数据」，`ANALYSIS_mm_datasets_inventory.md:108`），结论是"任务形态高度异质、只有 aswin 是 construction_site 直接来源"（`:9`）。**没有训练或评测**。其中 LLaVA-NeXT-Data（`groupC_llava_next.md`）只是"数据镜像有 146GB、格式是 LLaVA 多轮"，**从未用于训练**。
- `2026-08-28/GOAL_compression_for_nlp.md`：项目目标/验收定义（Phase 2 = 唯一验收，`:35, :48`）。是**论证素材**，不是实验。

### 3.5 已被撤回 / 口径不可用（引用即错）
- `REPORT_vlm8b_run3.md` §0.2/§2 的 acc 与「视觉有害」结论 —— 见 §2.1 A.5。
- `REPORT_vlm8b_run3.md` §3 每源 25 样本分任务表 —— 旧协议，被 120 样本/源修正表取代。
- `REPORT_vlm8b_run3.md` §4 的 run3 in100 14.8%/22.8% —— 新协议 29.2%/32.4%。
- `REPORT_vlm8b_1ep.md` 首版 acc 26–31% —— 已撤回。
- run1/run2 的全部 **acc** 数字 —— 旧协议口径，只能标注引用；**CE 可用**。
- `REPORT_budget_screen.md:283` 自述 run3 通用能力 bench「不是同一生成协议，数值只能同口径内部比」⇒ 不要与 base 75% 并列。

### 3.6 与本论文表示无关（像素/视频/厂商/论文快照）
- `2026-11-18/REPORT_tok448_2stage.md` + `artifacts/analysis_s{1,2}/tables.md`：448²→16×448 图像分词器，**纯像素重建**（27.324 dB / SAVOIAS 26.451 dB / 复杂度相关），无 LLM、无下游。
- `2026-11-10/REPORT_intermediate_state_base.md`、`REPORT_video_loop_adaptive_budget.md`：视频中间态/循环重建，无下游；前者自述「未测与下游 VLM 任务（QA/OCR）的关系」`:224, :257`。
- `2026-11-14`…`2026-11-23`（仅 sr-diffusion-v3）：v9/v10 并行解码、MUSA 稳定性、rfid/kd/l2 等，全部为像素/训练稳定性。
- `2026-10-26/paper_final/`：论文快照（摘要+tex），当前实验章为像素重建+复杂度（`snapshot/00_摘要.md:10`），可用于确认"论文确实缺下游节"。

---

## 4. 给论文作者的建议

### 4.1 这一节该怎么定位（最重要）

**不要**把 vlm8b 写成"我们的压缩表示足以支撑下游视觉-语言任务"——现有证据不支持，会被"无未压缩对照 + 无 caption 指标 + 编码器未冻结 + 单 seed"四连击打穿。

**建议定位为「下游可接入性 / 可行性验证（proof of concept）」**，主张限定为：
> "以仅重建信号训练的 K-token 表示，经一个线性投影即可被现成 4B 级 LLM 读取并用于语言化任务；在 100 类闭集判别上显著高于随机与去视觉底，在部分公开 QA 源上高于随机；但 32-token 的容量上限使细粒度 OCR/图表任务不可达，且本文未做与未压缩表示的等算力对照。"

这样写：(a) 有真实数字；(b) 不越界；(c) 把"容量瓶颈"变成论文"信息保持"主张的**补强**（不是打脸）——正好呼应「信息保有量是压缩有效性的判据」：表示能重建像素 ⇒ 粗语义可读；不能重建文字细节 ⇒ 细粒度下游不可达。

### 4.2 建议进表的数字（两个表）

**表 1（主表）：8 token 接 Qwen3.5-4B 的 ImageNet-100（100 类闭集，chance 1%）**

| 训练 | epochs | CE(k8) | acc(k8) 旧协议 | acc(k8) 修正协议 | null acc | shuffle acc | 来源 |
|---|---:|---:|---:|---:|---:|---:|---|
| run1 | 1 | 0.4666 | 12.0% | —（未重算） | 1.6% | 0.0% | `REPORT_vlm8b_1ep.md:23, :32-34` |
| run2 | 3 | **0.3719** | 28.8% | —（ckpt 删） | 8.4% | 0.0% | `REPORT_vlm8b_2ep.md:32, :40-42` |
| run3 | 3（混合集） | 0.4749 | 22.8%→作废 | **32.4%** | — | — | `out/eval_run3_in100_fix/eval.md:14`；`REPORT_budget_screen.md:284` |

- **CE 列可跨 run 比，acc 列必须分协议标注**（脚注写"run1/run2 acc 为旧 greedy 协议，与 run3 修正协议不可比；run2 ckpt 已删无法重算"）。
- 若嫌乱，**只放 run2 一行**（CE 0.3719 + 旧协议 acc 28.8% + null 8.4% + shuffle 0.0%），脚注说明；这是信息量最大的一行。

**表 2（可选，放附录或正文小表）：公开图像 QA 分源（修正协议，每源 120 行）**

只放 5–6 个最有代表性的源，**必须带 vision_gain 列**（这是诚实的关键）：

| 源 | acc k1→k8 | vision_gain | 读法 |
|---|---|---|---|
| slake | 71.7→73.3% | **+0.221** | 图被真正使用 |
| scienceqa | 74.2→74.2% | +0.004 | 高但靠文本先验 |
| mmmu | 32.5→33.3% | −0.017 | 高于 25%，图≈没用 |
| chartqa | 1.7→0.8% | +0.001 | 细粒度不可达 |
| docvqa | 1.7→1.7% | +0.077 | 有视觉信号但答不出 |
| （文本源参照）mmlu | 69.2→69.2% | +0.013 | 文本能力仍在，但被 SFT 损伤过 |

来源：`screen/tables_heldout_in100_train.md:18-28`；`REPORT_budget_screen.md:66-82`。

> 若期刊篇幅紧，**表 2 可以整表删掉**，只在正文写一句"在 15 个公开图像 QA 源上…多数源去图后 CE 几乎不变（|vision_gain| ≲0.05；唯一 ≥0.05 的正向是 slake +0.22、docvqa +0.08）"——但**不要**只报 acc 不报 vision_gain，那等于把"文本先验"包装成"读图能力"。

### 4.3 只能进 caveat（正文文字/脚注）的

1. **编码器可训**：lr 5e-6（run1/2）/ 2.5e-6（run3），`freeze_encoder_epochs=0`（`run_args.json:13, :17`；`REPORT_ds_e1_freeze.md:277`）⇒ 不是"冻结编码器"。
2. **模型族不同**：448²/p32 阶段-2 编码器（文档称 960M 档，见附录 B.9）接 4B LLM，不是论文主表 240M/224²；跨模型族外推需声明。
3. **无未压缩 / 无原生视觉塔 / 无其他压缩器 baseline**（§2.4 表）⇒ 不能声称压缩代价小或更优。
4. **无 caption/BLEU/CIDEr/judge**：doc 里根本不存在这类实验，别写成"生成质量"。
5. **预算 k 全平**（`REPORT_budget_screen.md:26-27`）：不要把这节写成"可截断预算在下游也成立"。
6. **单 seed、acc 样本 120–400**；300 行混采与 2,400 行分层不可换算（`REPORT_budget_screen.md:54, :255`）。
7. **通用能力代价**：1 epoch in100 SFT 就把通用文本从 75% 打到 1.7%（`REPORT_vlm8b_1ep.md:45-49`），run3 混合集仅恢复到 ~53%（`REPORT_vlm8b_run3.md:96-100`）；且该 bench 口径不能与 base 直接并列（`REPORT_budget_screen.md:283`）。
8. **标签噪声**：自蒸馏 ~11% 错、上限 ~89%（`DESIGN_8budget_vlm_finetune.md:200`）。
9. **诚实边界原文**：`REPORT_qwen_vision_pca_probe.md:166`「像素指标 ≠ 任务保真」；`REPORT_budget_screen.md:6`「真正的瓶颈在编码器侧的信息容量（32 个 z token），不在 LM 侧的 k」。

### 4.4 还缺什么实验（按性价比排序）

1. **冻结编码器复跑评测（最高性价比，~1–2h GPU）**：用 run3/final（若远端还在）或 run2 的 ckpt，把 `encoder.requires_grad=False`（现有 `--freeze-encoder-epochs` 已修好，`REPORT_ds_e1_freeze.md:1-40`）只重跑 eval，最好再短训 1–2 epoch adapter-only。**这一步直接对上论文主张"仅重建训练的编码器本身"**；否则所有数字都混入了 encoder 适配。
   - 若 run3 的 ckpt 已删（只存 `final` 软链 → checkpoint-11463，`REPORT_vlm8b_run3.md:4`），跑一次纯 eval（不训练）仍然可行，不需要 ckpt 额外保存。
2. **同 LLM 的"未压缩/更大 token 数"对照（必补，半天–1 天）**：把 z_s **1:1** 投影成 32 个 LM token（而不是 4:1 压成 8 个）喂进同一 Qwen3.5-4B，冻结其余条件。这能直接回答"压缩到 8 个 token 损失了多少下游可用性"——目前**完全没有**这个数（现有 8 token 已吃掉全部 32 z token，见 §2.1 A.1）。若 32 token 显著更好，论文的"压缩"叙事要改写成"以信息保持为约束的压缩"。
3. **caption 类生成指标（缺整块，1–2 天）**：COCO caption / 工地场景中文描述，BLEU-4/CIDEr/ROUGE-L 或 LLM-judge。这是"视觉-语言下游"最直接的证据，doc 里**一个数字都没有**。
4. **更大样本 / 多 seed（低成本补强）**：每源 ≥120 已有，但需 seed 方差（至少 2–3 seed 跑 eval）才能说"vision_gain +0.22 显著"。
5. **（若要写进正文主表）用论文主模型重跑 vlm8b**：240M/224² 编码器 + Qwen，成本 ≈ 1 次 3-ep SFT（A800 数十小时）。这是把证据"接到论文主表"的唯一干净方式，但成本最高。

### 4.5 若一条新实验都不做——最小可行写法（可直接改写）

> 「为验证压缩表示是否可被现成 LLM 消费，我们把 448²/p32 阶段-2 编码器的 32 个 special token 经线性投影压成 8 个 2560 维 token，作为前缀接入 Qwen3.5-4B（删除其原生视觉塔、仅喂 `inputs_embeds`），全参微调 1–3 epoch。在 ImageNet-100 100 类闭集上，8 token 的准确率为 28.8%（旧 greedy 协议口径；随机 1%，视觉置零 8.4%，错图 0.0%），教师强制 CE 0.372（置零 0.770，错图 1.227），表明表示中的类别信息可被 LLM 读取。在 15 个公开图像 QA 源上模型能给出高于随机的答案（slake 71.7%、scienceqa 74.2%、mmstar 51.7%），但多数源移除图像后 CE 几乎不变（仅 slake +0.22、docvqa +0.08），且图表/文档/OCR 类仅 1.7–10.8%，说明 32-token 表示的容量不足以支撑细粒度视觉-语言任务。受限于无未压缩表示对照与单次训练，本结果仅作为下游可接入性的初步验证。」

配套 caveat 段（放该节末尾或脚注）：
> 「本节所有数字中，run1/run2 的准确率为旧 greedy 协议口径，run3 为修正协议；两组不可直接比较。编码器在本实验中可训练（lr 5e-6），与本论文"仅重建信号训练、冻结后使用"的设定不同，故不应解读为冻结表示的充分性证据。此外，全参 SFT 会显著损伤通用文本能力（1 epoch 后 75%→1.7%），本文未进一步优化该权衡。」

---

## 附录 A — 检索与去重日志

**A.1 checkout 去重（md5 实测）**
```
A=AssocVTC/doc  文件数 1779
B=sr-diffusion-v3/doc  文件数 2026
共同路径 1779；md5 相同 1779；不同 0
只在 A：0
只在 B：247（集中在 2026-11-14 之后；另有 3 个 2026-09-02 补充）
```
⇒ 共同路径上的 vlm8b 文档两份逐字相同；引用任一 checkout 等价。

**A.2 grep 命中（`--include='*.md'`，关键词 `VLM|LLaVA|LLaVA-NeXT|Qwen2.5-VL|Qwen3|MLLM|下游|downstream`）**
- `AssocVTC/doc` 命中 54 个 md；`sr-diffusion-v3/doc` 命中 54 个（多出的 3 个：`2026-11-16/DESIGN_v10_video_whole_clip.md`、`2026-11-17/FEATURE_v10_stream_embodied.md`、`2026-11-24/PLAN_encoder_decodability.md`）。
- 其中**真正含 LLM 下游训练/评测**的只有：`REPORT_vlm8b_1ep.md`、`REPORT_vlm8b_2ep.md`、`REPORT_vlm8b_run3.md`、`REPORT_budget_screen.md`、`screen/tables_heldout_in100_train.md`（+ 产物 eval.md/json）。
- `BLEU|CIDEr|ROUGE`：**0 命中**（全 doc 无生成质量指标）。
- `judge`：命中均为 **数据标注/翻译质检**的 DeepSeek 评委（`2026-09-18/REPORT_llm_judge_calibration.md` 等），与下游任务评测无关。
- `VQA|caption` 的命中绝大多数是**数据集字段盘点**（`2026-09-17/data/dataset_recon/*`）或**计划**（`PLAN_alit_adaptive_experiments.md`、`PLAN_paper_experiments.md`），无实验结果。

**A.3 旁证文件（`/home/linaro/dsh/`，以 doc 报告为准）**
- `vlm8b_eval/`：run1 的 `eval.json`/`eval.md`/`run_args.json`/`per_budget_ce.jsonl`/`train_run1.log`。⚠️ `eval.json` 的 n=300 / acc 26–31% 是**已被 run1 报告撤回的按类排序口径**（`REPORT_vlm8b_1ep.md:8`），**不要用**；`run_args.json` 是确认模型/接口/超参的干净证据。
- `vlm8b_snapshot_20261109/`：11-09 13:45 修正后的评测产物 `out/eval_run3_fix_{normal,null,shuffle}/`、`out/eval_run3_in100_fix/`、`logs/bench_run3.json`、`logs/run3_breakdown_normal.json`、`out/run3/per_budget_ce.jsonl`。**这是 §2.1/§2.3 数字的一手产物**（本地 5.3MB，可复核）。
- `vlm8b_watchdog.log/.pause/.state`：运维状态（`run2_done`/`run3_done`/`stale_2026110913`），无科学数字。
- `vlm8b_snapshot_20261109/out/run3/final/` 是空目录（模型权重未随快照保存）。

**A.4 时间线（证明没有更晚的 vlm8b 下游实验）**
- run1 训练 11-05→11-06 09:17；run2 续训→11-07 05:47；run3 11-07 06:29→11-09 10:56；修正评测 11-09 13:45。
- 11-10 起项目转视频/中间态；11-13 起转 448 分词器/并行解码；11-24 转厂商编码器可解码性。**11-09 之后没有任何 vlm8b 文档**（唯一提及是 11-10 `REPORT_intermediate_state_base.md` 把 `feat/vlm8b-v1` 当分支背景）。
- `REPORT_budget_screen.md:245-273` 的 run4 数据配方与最小实验建议**从未落地**（无 run4 报告）。

---

## 附录 B — 未核实 / 有矛盾的点（引用前请复核）

1. **"8B" 不是参数量**：doc 一律 Qwen3.5-4B，但 `run1` 的 4B 是设计文档估算的文本参数（`DESIGN_8budget_vlm_finetune.md:137` 估 3.86B 文本 + 0.482B encoder），**没有实测参数量**；也没有独立加载日志证实 hidden=2560 以外的一切。标"未核实"。
2. **`mix_heldout.jsonl` 的来源/去泄漏**：doc 只给了构造规格 `PLAN_eval_aligned_mixture.md`，但最终 `mix_train_bal2` 的配比（文 82k/图 40.3k/in100 3k）与规格 v1（图 45k+文 90k+in100 90k）**不同**，中间重排脚本未入库。数据是否与训练集有重叠**未核实**（报告只说"val 独立于训练"，`REPORT_vlm8b_2ep.md:9`，那是 run2 的 in100 val，不是 mix_heldout）。
3. **300 行混采 vs 2,400 行分层 acc 不自洽**：300 行 normal 48.7%，分层表图像 33.8%/文本 65.8%。因分层是"每源等量"、与真实配比不同（`REPORT_budget_screen.md:54`），**理论上不可比**，但两套数字相差 6pp 左右；我**未能核实**是否有第三种抽样口径在起作用。建议论文只用其中一套（推荐 2,400 分层表 + vision_gain）。
4. **旧协议与修正协议的偏差方向不一致**：同一源 chartqa 旧 28%→新 1.7%（下降），slake 旧 44%→新 71.7%（上升）。机制未在 doc 中解释（只知是"把答案喂给 generate"导致空串），**不要尝试在旧/新数字间换算**。
5. **run2 vs run3 的 ImageNet-100 谁更好**：CE 上 run2（0.3719）明显好于 run3（0.4749）；acc 上 run2=28.8%（旧协议，不可比）vs run3=32.4%（新协议）。**跨协议不可比**，不能下"run3 比 run2 好/差"的结论；只能说 run3 因锚定数据 66k→3k 而 in100 能力被稀释（`REPORT_vlm8b_run3.md:24`，这是报告自己的判断，非独立复现）。
6. **`mean_cost`/`repeat_cost` 的语义**：报告用它论证"token 冗余是任务决定的、不是投影头塌缩"（`REPORT_budget_screen.md:206-208`），但这两个量只测"8 token 互换后 CE 变化"，**不等于**"压缩损失"；论文若引用需重新定义为"token 冗余度"，不要写成 compression loss。
7. **`REPORT_ds_e1_freeze.md` 的验证数字**（E1 bit-exact、Δw 1.53e-5）来自极小随机 LM + 476M 生产 encoder 的本机验证，**不是 4B 真模型**（报告自己写 "4B LM 的数值行为、显存总量未在本机复测"，`:326`）；只能当"冻结机制可用"的证据。
8. **产物可达性**：一手训练产物在远端 `/root/autodl-tmp/vlm8b/…`，本机只有 **run1 eval 子集** 与 **11-09 snapshot（5.3MB，无权重）**。若要复算 run2 的 CE 或任一 acc，需要远端 ckpt/日志；run2 ckpt 已删（`REPORT_vlm8b_2ep.md:64, :90`）。
9. **vlm8b 的 encoder 参数量**：doc 说 960M（`DESIGN_8budget_vlm_finetune.md:1`），但 `REPORT_ds_e1_freeze.md:263` 说 "encoder 480M vs LM 4B ≈10%"——两者指不同对象（前者 = 448²/p32/960M 档整编码器？后者 = "生产同结构同尺寸 encoder 476.2M"）。**960M 与 480M 哪个是 vlm8b 实际用的 encoder 参数量，未核实**；引用参数量请先核实 ckpt。
