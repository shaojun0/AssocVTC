# AssocVTC（main 分支: v2 最小可运行代码 + 全部实验文档）

> **名字**：AssocVTC = **Assoc**iation-trained **V**isual **T**oken **C**ompression（靠 token 压缩训练编码器的"联想能力"）。
> **曾用名 `SR-Diffusion-v3`**（2026-10-16 更名，GitHub 旧链接自动跳转）。更早的 `SR-Diffusion` 是**超分扩散**线
> （SVD + DINOv2-giant + SD 2.1 U-Net，见首个提交 `524872f`）；2026-08-27 起主线已转为**视觉 token 压缩**
> （DINOv2 + register specials + 循环读出，全程无 diffusion 组件），旧名与现状不符，故更名。
> ⚠️ **更名范围**：只改"项目自称"。**以下一律保留原样、不要批量替换** —— `doc/` 按日期归档的历史文档/日志/证据里的旧名、
> 远端机器目录（`/root/autodl-tmp/srdiff_*`、`/data/srdiff-20260926/…`）、模块名 `srdiff_runtime.py`、环境变量 `SRDIFF_*`、
> 看板变量 `SRDASH_ROOT`（改它们会断代码与在跑的实验，且会篡改历史证据）。

## 0. 项目目标与核心思路

**项目目标（权威版）**: 通过 **token 压缩**训练编码器的**联想能力**（把图像信息压进少量 special token z_s），训练完成后冻结编码器做 **Phase 2 NLP**（Qwen 生成工地描述/隐患）。像素重建是 Phase 1 的训练脚手架 + "信息保持"直接探针，**不是最终目标**；唯一验收 = Phase 2 文字生成质量。目标的完整定义见 **[`doc/2026-08-28/GOAL_compression_for_nlp.md`](doc/2026-08-28/GOAL_compression_for_nlp.md)**（讨论架构/评估/下一步前**必须先读**）。

### 0.1 指导性判断：信息密度 ↑ ⇒ 幻觉率 ↓；token 数随场景精度自适应

> 这是本项目的**想法与动机**（用户口径，2026-10-13）。原文照录：
> 「目前多模态大模型对一张图片进行编码时，短则将其编码成几百个token，长则将其编码成上千个token，而具身智能大模型往往受限于最大上下文及现实场景的实时性的限制，上下文很快就会被大量的token流撑爆。按照行业内标准的做法，可以将图片压缩到固定大小的token，然后输入，但是这同样存在一个问题：对于高精密的场景，这样压缩会极大的丢失细节信息。为此，我们提出了一种自适应图像压缩的方案，当场景对精度要求较低时使用少量token进行输入，以节约了上下文长度，当场景对精度要求较高时则可以使用尽可能多的token进行输入，保证高精度场景下信息不失真。」

**表述为两句可操作的话**：

1. 让 z_s 的**每一个 token 都携带判决性的图像信息**（而不是把同一份全局摘要复制 K 遍），Phase 2 的 caption 才会更准、更少编；
2. **token 数随场景精度自适应**——精度要求低的场景给少量 token（省上下文），精度要求高的场景给尽可能多的 token（细节不失真），而不是把所有图都压成同一个固定大小。⚠️ 第 2 条目前**只有动机、没有实现**，与 `GOAL` §4.1 的口径差异见本节末「动态 token 数」一段。

注意这条有两个层面，它们是同一个道理：

| 层面 | 密度低的表现 | 提高密度 |
|---|---|---|
| **编码侧**（本项目在做的） | K 个 z_s 是"全局摘要的副本"（[GOAL](doc/2026-08-28/GOAL_compression_for_nlp.md) §2 推论 3：576 个 z_s 对 Qwen 是"同一句话重复 576 遍"） | 压掉冗余，让少量 token 承载布局/物体/边界等**活信息** |
| **解码侧**（Phase 2 消费时） | 上下文被冗余 token 灌满 ⇒ 模型对"该写什么"没有明确依据 ⇒ 用高频先验**填缝** = 幻觉 | 每 token 判决性信息更密 ⇒ 生成由证据而非先验驱动 |

**一个必须避开的误读：token 变少 ≠ 密度变高。** 密度是"**单位 token 承载的活信息**"；用更少 token 同时丢掉更多信息，那只是**信息损失**，不是压缩率提升。判据是**信息保持**（`GOAL` §2 的"必要条件"）：压完仍能还原布局/物体/边界。本仓库踩过这个坑——监督目标换回像素之前，特征目标会退化成"学质心即低 L1"的**假收敛**（`GOAL` §3）。

**四个可度量的代理**（密度本身不可直接观测，用这组代理逼近）：

1. **压缩率** K/N（K = special 数，N = patch 数）——当前最强杠杆，`doc/2026-09-16/REPORT_ksweep_bptt.md`；
2. **信息保持**：K 压缩下的像素 L1（越低越好，且必须显著优于 per-patch DC 基线）。⚠️ **不要用 PSNR 当验收指标**：高分辨率臂曾长期"停在预测均值"，2200+ 步后 PSNR 仍**低于**均值基线（`doc/2026-09-23/ANALYSIS_res_sweep_rootcause.md` §2）；
3. **冗余度**：z_s 之间/内部的相似度与使用率 → 目标是"每个 token 都不可替代"，而不是"全部相同"；
4. **可寻址性**：解码器能否**按内容指名**某一位（P2 的 `Linear(patch_feat)` 注入，见 §4）。这是资源问题，**不是密度问题**——解码器找不到那一位，等价于它不存在。

**动态 token 数是目标设想，尚未实现。** 这正是上面 2026-10-13 口径要的形态：**精度要求高的场景给尽可能多的 token、要求低的场景给少量 token**；本项目里可测量的一个代理是——**信息量大的图给更多 K、信息量小的图给更少 K**。⚠️ 诚实标注：当前代码里 K 是由采样步集推导的**固定值**（§1），而 `GOAL` §4.1 明确写了「固定 K 即可，不必复活预算机制（YAGNI）」——**本节与该条存在口径差异**：GOAL §4.1 是当期实现层面的决策（基于当时 k-sweep "k=1..32 已够还原"的结论），本节是目标层面的设想。两者要统一，需要一轮"按图自适应 K"的实验来判定**图间信息量差异是否真的存在且可利用**（在那之前不落地，避免复活已判 YAGNI 的机制）。

**依据与诚实边界**：

- **支持**：压缩↔幻觉是本领域在做的方向，例如 [SeeMe: Mitigating Hallucinations in LVLMs through Effective Visual Token Engineering](https://arxiv.org/pdf/2607.04163)、[RVSD: Retrieval Vision Sparse Decoding for Mitigating Visual Hallucinations](https://export.arxiv.org/pdf/2609.02731)。**但这些论文的"压缩"是推理期 token 裁剪，本项目的"压缩"是训练编码器**；上面的侧重点与它们**部分重合、不能直接引用为已证事实**。
- **本项目内的证据**：① K 压缩确实是"练联想"的主要杠杆（`doc/2026-09-16/REPORT_ksweep_bptt.md`，K=15→120 像素 L1 15.02→12.51）；② 真实且已被采纳的对立判据是**任务保真**（E8），而非 PSNR-RD（`doc/2026-09-22/REPORT_batch0_rd_earlystop.md`：224×126 对 JPEG −0.4~−14.2 dB ⇒ 确定性 RD 这条线应放弃）。
- ⚠️ **尚未验证的环节**：**"密度 ↑ ⇒ 幻觉 ↓"在本项目里还没有直接测量**——项目目前一个 caption/幻觉指标都没跑过；Phase 2（冻结编码器 → MLP → Qwen）尚未搭起来。**本节是方向性判断，不是已验证结论**，验收口径以 `GOAL` §2 为准。

---

## 1. 当前实现（v2 最小可运行代码）

register 式（2026-08-28 起唯一路径，`model_v2.py` 头部 docstring 是权威说明）：

- specials 作为额外 token 直接拼进 DINOv2-large 输入序列 `[cls; specials(K); patches(N)]`，DINO 24 层全双向算出 z_s（register token 式）。
- register 数 K（`num_specials`）与 patch 数 N 解耦：**K 由"最终生效采样步集"自动推导**（`K = min(max_t((⌊√t⌋+1)²−1), N)`，无"花瓶 register"）。
- 解码器 = OutputQueryDecoder（**循环读出，唯一路径**）：step1 查询 = `query_base`，step t≥2 查询 = `query_base + 上一步输出`（裸加，无投影/LayerNorm）；读侧按块切片 `A[:, lo:hi+1]`，每步只读自己那块 z_s，**循环 carry 走 BPTT**（唯一路径）+ 2 层 PixelHead MLP → 像素重建。循环版**不新增任何参数**。
- **读窗口 = 平方块（默认口径）**：步 t 读 `z_s[k²:(k+1)²−1]`（`k=⌊√t⌋`；首步 `lo=0` 另含 `z_cls`），z_s 全部来自 DINOv2 **末层**。⚠️ 另有 **opt-in 的 `--step_plan fixed --block W`**（步值 = 自然数，每步固定读 W 个 z_s）；**默认仍是 `square`，默认路径逐位不变**。实测：**固定宽度在高 N 上训不起来**（336² 卡在平凡解，比平方块差 −4.03 dB 且慢 4.06×）⇒ [`doc/2026-09-23/REPORT_fixw4_plan.md`](doc/2026-09-23/REPORT_fixw4_plan.md)。
- **A 相 warm start（`--warm_steps`，默认 0=关）**：前 N 个优化步把 `decoder.steps` 设成 `[K]`（**只跑 1 步、一次读完全部 z_s + z_cls**），到点由 `WarmStartSwitchCallback` 自动切回真采样步集跑多步循环。动机：多步循环早期步只读到 `2k+1` 个 token，条件最优解接近"预测均值"，从零直接训多步会停在该平凡解。224² 实测 A 相要 **1500~2000 步**才逃逸，`--warm_steps 1000→3000` 使 16 步曲线由平变单调降（最优 PSNR 14.26→**18.69**）⇒ [`doc/2026-09-23/REPORT_224_A3000_verdict.md`](doc/2026-09-23/REPORT_224_A3000_verdict.md)。
- 损失 = **直接预测口径**（2026-09-15 起）：每个采样步的输出 `Y_t` **各自直接**过 PixelHead 预测整图（无累加/集成），`loss = mean_t L1(PixelHead(Y_t), target)`（全轨迹深监督、各步平权）；`F_hat = Y_pix[:, -1]`（最后一步的直接预测），监控量 `recon` = 它的 L1。K 压缩（如 K=63）是练联想的主要杠杆。
- **已整块删除（传即 `TypeError`/argparse 报错，不静默忽略；复现请从 git 取回）**：`layer_tap` 逐层特征金字塔、`carry_detach` 开关（⇒ **carry = BPTT 唯一路径**）、并行解码路径与 `--query_mask_mode`、`--recurrent*` 全套开关与 `rec_*` 参数、`--loss_mode`/`--loss_decouple`。
- ⚠️ **checkpoint 兼容性（"静默算错"型风险）**：循环版**不新增参数**，与并行时代默认配置的 `state_dict` 逐 key 逐形状完全相同（实测 44 keys 全等）⇒ **旧 `final_model.pt` 用当前代码 `strict load` 不会报错**，会按新语义**静默算错**。消费方不能依赖 load 报错来区分代际；需要旧产物就用 git 取回当时的 `model_v2.py`（`git log -- model_v2.py`；最后一个含并行路径的提交 = `043ef2a`）。
- 历史设计文档（**只作设计过程参考，不代表当前行为**）：`doc/2026-09-15/DESIGN_v2_recurrent.md`（带开关那一版）、`doc/2026-09-22/{DESIGN,REPORT}_layer_tap_pyramid*.md`（已归档）。

## 2. 快速开始

```bash
# 训练（2 GPU DDP, 全 fp32; 单卡冒烟加 --smoke --limit 32 --max_steps 3）
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2 --epochs 40

# 全量 test 推理（slice / step_plan 参数必须与训练一致）
python infer_v2_test.py --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2/final_model.pt \
    --output output/phase1_v2/infer_test.json
```

自检: `python model_v2.py`（形状 / 块掩码 / 梯度 / PixelHead / eval 同路径，应输出 `ALL CHECKS PASSED`）。

环境：torch≥2.0 / transformers / accelerate（见 `requirements.txt`）；DINOv2-large 离线权重 + `HF_HUB_OFFLINE=1`（服务器路径 `/root/autodl-tmp/models/dinov2-large`）。

---

## 3. 结论索引（doc/ 按日期归档，读最新在前）

> 实验细节（口径、曲线、边界）全在 `doc/` 里；本节**只保留结论与指针**。
> 判据口径：**RD/PSNR 这条线应放弃**（[`doc/2026-09-22/REPORT_batch0_rd_earlystop.md`](doc/2026-09-22/REPORT_batch0_rd_earlystop.md)：224×126 对 JPEG −0.4~−14.2 dB、对 WebP −5.3~−15.9 dB，且 0.3 bpp 后曲线就平），剩下的牌是**极低码率独占区**与 **E8 任务保真**（⇔ 本 README §0.1 的「密度 + 精度自适应」口径）。

### 3.1 进行中

- **★ ALIT 参照的自适应长度 tokenization 实验（2026-10-16，已执行·进行中）** —— 对着 [ALIT（arXiv:2411.02393）](https://arxiv.org/abs/2411.02393) 把三条可移植机制（**逐轮新增记忆 / 动态停机 / 逐迭代损失权重**）与两条分析口径（**TSC–TOI、IID/OOD 熟悉度**）映射成本仓库可执行的 **12 个实验**（A 零训练 6 个 + B 重训 4 个 + C 验收 2 个，≈30 GPU·h 单卡 4080S）。**结构对照结论**：ALIT 的**解码器并行且无状态、循环在编码器（逐轮 +32 latent）**；本仓库**解码器是顺序循环 + 画布 carry（BPTT）、编码器单次前向**——两者是镜像关系，故只借机制与实验设计、**不借解码器**。
  - **已执行进展（截至 2026-10-16，A 阶段 11 步全 rc=0）**：① **四臂基线逐位复现 + 回归门 PASS**（PSNR 26.602 / 31.219 / 25.304 / 28.582 dB，\|ΔPSNR\| ≤ 0.0005 dB；新代码默认路径与发布版逐位一致，`gate_diff_stdattn_rec.json` 的 `PASS`/`max_abs_diff=0.0`）；② **E-A1 否证（两臂）**——std 臂 5,000 张里 **96.98%（4,849 张）** 最优停步 = step7、停步熵归一化 **0.065**，**oracle 上界只值 +0.00016 px**（63.0302 vs 固定 63 tokens）；lin 臂众数占比 60.92% 未命中字面条件，但 **oracle 上界也只有 +0.0029 px** ⇒ 实质同判否证；③ **E-A2 否证（信息保持口径结案）**——等平均预算下自适应**更差**：std **+0.2687 / +0.3272 / +0.0363 px**（固定 step3/5/7；ΔPSNR −0.65 / −0.92 / −0.15 dB，预算 gap ≤4.37%），lin 更差（+1.2060 px），coco/ade20k 同向；④ **E-A3 否证**——三级平均停步 7.0302 / 7.0304 / 7.0055，max\|Δμstep\| = **0.0247 步**（阈值 0.5），且**远距 ADE20K 反而更易重建**（step7 L1 **5.9722 < IID 6.5973**，共同 τ 下始终少 0.17~0.26 步）；⑤ **E-A5 成立（低冗余）**——K=64/D=448：有效秩 **47.077 = 0.736×K**、两两 cos 均值 **0.2569**、frac(cos>0.9) = **0.0030**；⑥ **E-A6 否证**——Δ₁ **27.56%**、前缀 step4 **65.09%**（step6 才 91.51%）、末步过冲率 **96.98%**、Δ_t **非几何衰减**（t2 1.5912 < t3 1.7348）；⑦ **E-A4 成立但极弱**——只在 `needed_step@median τ` 目标上显著为正（最强 edge_density ρ=**+0.9058**），`best_step` 目标因退化无信息量；⑧ **E-B1 成立：只有「给后步加权」有效，四点单调剂量-响应；ALIT 递减配方未复现且有害** —— 剂量（按后移程度）mean → increasing → **tail27** → lastonly：PSNR **28.4094 → 28.8575 → 30.0382 → 30.5220**、末步 s8 **6.7072 → 6.3620 → 5.5534 → 5.4448（单调改善）**、Δ₁ **25.43% → 23.35% → 7.10% → 6.46%**；**中间档 step7 存在最优点**：`b1_tail27`（`[0,0,0,0,0,0,1,3]`）s7=**5.5904** 是权重臂最好（比基线好 1.10 px），只用 **0.109 px** 末步代价换回相对 lastonly **1.45 px** 的中间档可用性 ⇒ **纯损失侧最实际的多档工作点**；`b1_decay`（ALIT 原版递减 `[2.5,2,1.5,1.25,1,…]`）**27.4845（−0.9250）**/末步 **7.4054（−0.6982 更差）**、前 3 步却更好（step1 −1.089 px）⇒ **质量前移、牺牲交付步**。**可交付性 trade-off**：`b3_grow`（**唯一全档可用**，7.51/6.34/5.51/5.51）→ `b1_increasing`（后两档反超）→ `b1_tail27`（t=7 最好、**t≤5 报废** 47.68/28.94）→ `b1_lastonly`（仅 t=8）；强后移臂 t=3/t=5 截断均逼近常量灰 50.892。⇒ **PLAN §1 第 5 条（ALIT 的递减配方）在本设定未复现，方向相反**；基线 `b1_mean` 28.4094（vs 历史 28.5823 = **−0.1729 dB**，重跑漂移+`dec_grad_ckpt` 开关不可分离）。⚠️ 剂量臂均**单种子**（`b1_increasing`/`b1_tail27`/`b1_lastonly` 的第二种子属主动牺牲）。　**复现性收尾（`dec_grad_ckpt` 归因探针）**：`b1_mean_nockpt`（`--dec_grad_ckpt 0`）= **28.0685 dB**，比 ON 的 `b1_mean`（28.4094）**差 0.3410 dB** ⇒ **开关不能解释历史的 +0.173 dB**（方向相反、量级 2×）⇒ 历史 `28.4094 vs 28.5823` 归因为**单种子/实例级重跑漂移**，报告维持「不宣称复现 28.582」；另记 **「数学等价 ≠ 数值等价」**——grad-ckpt 注释称反向重算数学等价，但 bf16 重算在 11,406 步后产生 **0.341 dB** 终值差异 ⇒ **配对实验必须固定该开关**（本轮 10 个正式臂全 ON，内部配对有效）。⑨ **E-B3 = 分裂（recon 增益成立 / ALIT 后步激活机制否证）**：`b3_grow`（GrowthBlock 读侧生长）**30.5910 dB / SSIM 0.8937 / L1_norm 0.09566**，同预算 vs `b1_mean` **+2.1815 dB**、末步 s8 **−1.1940 px**，距单步上界仅 −0.6276 dB；**截断全预算区间最好**（t=3/5/7/8 = 7.506 / 6.338 / 5.506 / 5.513）。**但机制假说被否证**：Δ₁ 占比 **25.43%→45.66%（升）**、后 4 步 Δt 均值 0.772→0.322（降）⇒ 增益来自「**每步都有新信息**」（生长 token 从 step1 就可读，读出集合 3→19，step1 即好 3.4431 px），**不是**「后续步被激活」；(c) 注意力熵**数据不足（未测）**。⚠️ **PLAN §8 三态表无法表达这种分裂**，报告显式记「分裂」并在 §11.3 按轴分别给三态判决。**最大未控制变量 = 参数量**（refiner +1,049,600，总参数 63.055 M→64.104 M，**+1.7%，无参数匹配 placebo**）。⇒ **E-B4「一条流多档」的两个可行候选 = `b3_grow`（全区间最好）与 `b1_increasing`（不新增参数）**；`b1_lastonly` 只在 t=8 可用（t≤6 逼近/劣于常量灰）⇒ 被截断代价排除。⑩ **E-B2 = 否证**（README §4.1 三判据未同时满足 ⇒ **「任务同构」是主因**）：判据(1) 后步非平凡**满足**（尺度无关比值 ≥1.027；`b1_lastonly` 4.77、`b2_dh1` 2.31 例外）；判据(2) recon 不劣化**不满足**（`b2_dh1` **−7.8556 dB**、`b2_dh3` **−1.2424 dB**；`b2_dh2` +0.2766 dB 但末步仅 −0.0152 px = 噪声内）；判据(3) 区域专精**不满足**（除 dh1 外 `diag/offdiag` 均 **0.97~1.00**，各步仍在做同一全局精修；dh1 唯一 0.283 却付 −7.86 dB）。⇒ **改监督口径会伤、改损失权重能提，两者不矛盾**。探针自洽：`prog_curve_L1_255` vs `stepwise L1_255` 最大差 **3.5e-06**。⑪ **E-B4 = 否证**（随机子集反传）：`b4_sg`（`--sample_grad_iters 3`）**28.0990 dB（ΔPSNR −0.3104）**、末步 **+0.3337 px 更差**、截断档 **+0.1867/+0.2211/+0.3334/+0.3337 px 全档更差** ⇒ 不追平全步反传；**真正「一条流多档」可用 = `b3_grow`（全档 −1.19~−3.91 px，最好）与 `b1_increasing`（后两档 −0.34 px，不新增参数）**；专训档 K'=16/32/48 成本裁剪 ⇒ 该子项数据不足。⑫ **E-C1 = 数据不足（探针分辨力不足）**——三臂 json 已归集：主臂 primary Δ = **+0.2723 pp**、CI **[−0.3713, +0.9776]**（跨 0），mean_k **37.9014 = 0.592×K**；9 档 τ 扫描 Δpp ∈ **[−0.3239, +0.2761]**，27 行**无一行**达 ±0.5 pp 判据；oracle 上界 `any_k_correct_rate` = **0.2968**；探针 `train_top1` **0.45–1.00**（`_cls` 臂 =1.0000）vs `val_top1` **0.049–0.112**（chance 0.01）、val 曲线非单调 ⇒ **只能说「该探针分辨力下测不到差异」，不能说「自适应 = 固定 K」**。
  - **★ E-A5 对 §0.1 的反证（必须连边界一起读）**：`GOAL` §2 推论 3「z_s 是全局摘要的 K 份副本」**在 K=64 这个从零 std 编码器上不成立**（有效秩 0.736×K、cos 0.257、cos>0.9 仅 0.30%）⇒ 「压冗余」在该操作点不是主要瓶颈；**但**（i）该主张原本描述 DINOv2 主线 **K=576**（两两 cos 0.963–0.974），两点**不同构**、不能互相外推，主线 K=576 上的同一组 `zs_stats` **未做**；（ii）`token_mean_var ≈ 1.9e-7`（「均值同置」）是编码器末层 **LayerNorm** 的下游效应，**不是**独立证据。⇒ 按 PLAN §8 的同一逻辑，**E-B3（新增记忆）的动机方向被削弱**（不是增强）；仍按判决实验执行。
  - **执行口径与预算裁剪（2026-10-16 Lead 确认，必须照录）**：**全部 B 臂同一台新实例串行**执行（同机、同后端、单一训练机器）⇒ **无跨机混淆**，配对性来自同进程/同批次评测；**E-B4 专训档 K'=16/32/48 不做**（从零训练 3×~5 h 超预算）⇒ PLAN §8 的「截断档 vs 专训档 <0.20 px」记「**数据不足 + 成本理由**」，**不是待补、不留空**；口径换算 `l1_norm × 57.63 → L1_255`（像素口径）已在报告 §1.3 写死。
  - **继续 B 阶段的决策记录（Lead 2026-10-16）**：A 阶段四项否证本可止损（PLAN §3 前言），但 **A 只覆盖信息保持/自监督口径**，唯一验收 E-C1 的分类探针**分辨力不足**（`train_top1` 0.45–1.00 vs `val_top1` 0.049–0.112 ⇒ 只能说「该探针测不到差异」，**不能说「自适应 = 固定 K」**），问题**尚无答案**；**继续跑完 B 全矩阵是主动决策、不是漏读 PLAN §3**——依据 = **B1/B2/B3 判决的是 README §4.1「让后步真干活」这个独立未决问题**（P3 损失加权 / P1 区域损失 / ALIT 记忆增长），A 阶段结论不能替它们判决，且成本可承受；且 E-B3 改信息通路是零训练原理上无法替代的判决实验。**已写明停止条件**（B1 四臂差 <0.10 px ⇒ 关闭权重方向；B3 三项全不成立 ⇒ 关闭记忆增长方向；B4 只做截断代价量化）。⇒ 报告 §14.0。
  - **结论与边界**：信息保持口径下「按图自适应 K」被**系统性否证**（E-A1/A2/A3/A6），但**最终判决权在 E-C1（任务保真）**，本实验**不改** §0.1 正文、**不推翻** GOAL §4.1；⚠️ 全部结果**单种子无方差**（<0.10 px / 0.1 dB 不作结论）、**PSNR 不是验收**、**与 ALIT 不同构**、E-A4 代理量是弱代理、E-A5 只做了冗余度子项（注意力熵**未做**）、E-B2 输入侧停机是**架构裁剪**、E-B4 专训档按成本裁剪、E-C1 线性探针 **≠ Phase 2**。
  - ⇒ 报告 [`doc/2026-10-16/REPORT_alit_adaptive.md`](doc/2026-10-16/REPORT_alit_adaptive.md)（逐实验「方法-结果-判决-诚实边界」+ 三态判决）；聚合表 [`report_tables.md`](doc/2026-10-16/artifacts/alit/report_tables.md)（`python3 tools/alit_report.py` 自动生成，每个数字带来源文件名、缺产物列 missing）；规划与判据 [`PLAN_alit_adaptive_experiments.md`](doc/2026-10-16/PLAN_alit_adaptive_experiments.md)；机制映射/可移植性/架构约束 [`ANALYSIS_alit_vs_assocvtc.md`](doc/2026-10-16/ANALYSIS_alit_vs_assocvtc.md)（输入侧动态停机在现结构装不下）。

- **★ 固定 8 步 × token 预算 K 消融（K=8…128，2026-10-19/20）** —— 按 [ALIT](https://arxiv.org/abs/2411.02393) 的「8 次 rollout」口径，在本仓库自建模型上**把 K 与步数解耦**：新增 `--step_plan cumulative`（`scaled_square_cumsum`：square 块尺寸形状按总预算缩放；默认 square 路径逐位不变，自检 §8e + 历史 K=64 回归 28.582 dB / cum [3,8,15,24,35,48,63,64] 逐位一致）。2×A800 上 **9 臂**同协议（A 相 T=1 45 ep → B 相 T=8 6 ep 热启动；首轮 6 臂 9.2 h + 追加 3 臂 ≈5.8 h）。**主结果**：末步 PSNR 在 **K=8→64 单调 21.78 → 27.76 dB（+5.98）**、同预算单步臂 **24.37 → 31.03 dB**（单步臂 ≈ **+2.2 dB/doubling**，到 K=64 仍无饱和），**K=48 落在曲线上（26.86 / 29.61 dB）**；但 **K>64 在统一协议下反而退化**（K=96 **26.24** / K=128 **25.09** dB；T=1 26.81 / 25.65）——且证据指向**优化受限而非容量到顶**：K=96 收尾 train L1 **0.136 vs K=64 0.082**、`z_s` 绝对有效秩 **36 < 42**；K=128 在 lr=2e-4 下 A 相停在平凡解（best 20.67 dB），被迫 lr=1e-4（仍有 73 次 grad-skip）⇒ **K=64 是本协议下的实际最优操作点**。另：**固定 8 步对 K≤12 严重过剩**（L1 第 3–4 步后走平），**K≥24 才 8 步都有正 Δ**；**零训练对照** K=64 原生模型 early-stop 在同等累计 token 下仍好 **+0.15~1.1 dB** ⇒ K 同时改变预算与编码器 memory 容量。⚠️ 单种子、读窗口计划是构造（K<T² 无原生计划）、**K=96/128 不是干净容量点（K=128 换了 lr）**、T=1/T=8 预算不同不可比、像素口径非任务保真 ⇒ 报告 [`doc/2026-10-19/REPORT_k8sweep.md`](doc/2026-10-19/REPORT_k8sweep.md)（产物 `doc/2026-10-19/artifacts/k8sweep/`，脚本 `tools/{run_k8sweep_arm,launch_k8sweep,eval_k8sweep}.sh` + `tools/plot_k8sweep.py` + `tools/diag_zs_rank.py`）。

- **★ 固定 8 步 × 每步**均匀** n 个 z_s 消融（K=64…256，2026-10-21）** —— 按用户口径把 2026-10-19 的「递增取」换成**每步固定取 n=K/8 个**（K=64/96/128/192/256 ⇒ n=8/12/16/24/32；`--step_plan fixed --block n`，首步另含 `z_cls`）。新机器 2×A800（torch 2.12.1+cu130）**5 臂**同协议 A+B；另跑 **3 个配对 cumulative B 相**（A 相 T=1 与读窗口计划无关 ⇒ 直接用 fixed 臂的 A 相 ckpt 热启动、只换 B 相窗口 ⇒ **同初始化、唯一自变量 = 窗口形状**；省 ~7 h GPU）。**主结果**：① **均匀分配 > 递增分配**：配对末步 PSNR **+1.26 dB（K=64）/ +1.12 dB（K=96）**，末步 L1_255 **6.51 vs 7.48 px / 6.44 vs 7.27 px**（K=128 两臂同处差 A 相，仅 +0.18 dB）；② 均匀分配下 **K=64/96/192 → 28.74 / 28.96 / 29.17 dB**（饱和但**不塌陷**，K≈64 已接近饱和）；③ **K=128/256 的 B 相低（25.52 / 23.96）是 A 相优化失败、不是分配形状或容量**（K=128 在 lr=1e-4 与 2e-4 下都进差盆地且旧机器复现；K=256 额外是 K>N=196 的超预算点）⇒ **2026-10-19 的「K>64 退化」需要修正：主因是 A 相盆地**（旧机 K=96/128 的 A 相 26.81/25.65 vs 新机 32.31/32.51）。⚠️ 单种子、**A 相 lr 在 K≥128 为 1e-4**（与 K≤96 的 2e-4 不同）、K=128/256 **不是干净容量点**、T=1/T=8 预算不同不可比、像素口径非任务保真、跨机噪声 0.3–0.4 dB ⇒ 报告 [`doc/2026-10-21/REPORT_fixedk_sweep.md`](doc/2026-10-21/REPORT_fixedk_sweep.md)（产物 `doc/2026-10-21/artifacts/fixedk/`，脚本 `tools/{run_fixedk_arm,launch_fixedk,eval_fixedk,launch_cumB_paired,eval_cum_paired,launch_k128_lr2e-4}.sh` + `tools/{plot_fixedk_compare,plot_fixedk_val_curves,plot_lr_probe,summarize_fixedk}.py`）。**2026-10-22 后续（重要）**：K=128/256 差盆地根因已定位 = **std（softmax）编码器缺 QK-Norm**（`enc_std_qk_norm` 默认关 = "普通 transformer 原貌"，而线性编码器默认开；同 lr/seed 的 16-ep 受控探针：关 20.8 dB vs 开 **28.8 dB**）。加回 QK-Norm 重训：K=128 **A 34.01 / B 31.23 dB**、K=256 **A 36.77 / B 33.82 dB**（较原协议 +7.8/+5.7 与 +10.4/+9.9 dB）——**B 相已高于本 bullet 的 K=64/96/192（28.74–29.17）⇒「K≈64 已接近饱和 / K=64 是最优操作点」需按 10-22 报告修正**（真实 K 曲线须 5 个 K 全部用 QK-Norm 重训）；配对 fixed-vs-cumulative 在 K=128 修好后为 **+0.82 dB**（原 +0.18 是差 A 相压扁的）。⚠️ K=256 仍是 K>N=196 的非压缩点；单种子。⇒ [`doc/2026-10-22/REPORT_k128_k256_fix.md`](doc/2026-10-22/REPORT_k128_k256_fix.md)（产物 `doc/2026-10-22/artifacts/k128k256_fix/`，脚本 `tools/{run_probe,launch_probes,launch_retrain,eval_retrain,diag_attn_entropy}.*`）。**2026-10-23 低 n 延伸**：补 n=2/4/6（K=16/32/48，8 步）——8 步末步 **24.10 / 25.82 / 27.68 dB**，相对 n=8 平台（28.74，L1_norm 0.113）为 **−4.64 / −2.91 / −1.05 dB**；而 n≥8 只再 +0.2~0.4 dB ⇒ **平台从 n=8（K=64，3.06×）起、拐点 n≈6–8**，n<4 陡降；且 n≤4 时 8 步「用不满」（最优步第 6–7 步）⇒ 高压缩档应同步减步数。同 K 的 cumulative 对照：均匀 vs 递增在 K=16/32/48 为 **+1.28 / +0.78 / +0.82 dB**（低 K 端优势更大）。⇒ [`doc/2026-10-23/REPORT_lown_sweep.md`](doc/2026-10-23/REPORT_lown_sweep.md)（产物 `doc/2026-10-23/artifacts/lown/`，脚本 `tools/{launch_lown,eval_lown,plot_lown_compare,fetch_lown_artifacts}.*`）。

- **★ 固定每步 n=2 × 解码步数 T 消融（T=16/24/32）+ SVD 还原基线（2026-10-23）** —— 用户口径「n/step=2，step=16/24/32」+「这次实验完就和奇异值对比一下，毕竟都是每次使用 2 个向量进行还原」。A 相**不重跑**（A 相与读窗口计划无关；复用 10-19 `srdiff_k8sweep` 的 ckpt，三臂 T=1 复评 **28.7283 / 29.6077 / 31.0271 dB** 与 10-19 `eval_val.json` **逐位一致**），只跑 B 相（`--step_plan fixed --block 2`、6 ep、等效 bs64、lr 2e-4 热启动）。**T=32 在 bs64 直接反传 OOM**（74.4 G 已分配再要 1.15 G）⇒ 新增 `train_linattn.py --accum_steps`（micro 32×2 累积；**等效批量/优化步数 11,406/lr 曲线/val 张数全不变**，vram 41 G），三臂统一用累积。**主结果**：① 同一 n=2 下 T=8→16→24→32（K=16→64，预算 ×4）末步 **24.098 / 24.148 / 24.722 / 24.738 dB**（L1_norm 0.1875/0.1877/0.1772/0.1767）——**预算翻两番只 +0.64 dB**，且 T=24→32 只 +0.015 dB；逐步曲线在**读满 32–44 个 z_s（第 16–22 步）后饱和**，之后 10–16 步无收益（L1 甚至微升）。② **同 K 下「T=8 × n=K/8」比「T=K/2 × n=2」高 +1.68 / +2.96 / +4.00 dB（K=32/48/64）** ⇒ 8 步不是瓶颈，**每步只读 2 个 z_s 的循环累积**才是（23 步后模型锁死在 L1_255≈10.2 / 24.7 dB，与预算无关）。③ **与 SVD 对比**（per-image 最优秩-r 还原，每图 r×(196+768+1) 个数；模型每步 2×448=896 个数）：**t≤13 模型更好**（t=8 +1.44 dB），**t≈14 起 SVD 反超并拉开**，t=32 时 **SVD rank-30（28,950 个数）28.57 dB vs 模型（28,672 个数）24.74 dB（+3.83）**；而 T=8 宽窗口臂在同预算下仍胜 SVD **+0.17~+0.76 dB** ⇒ n=2 档的短板是「窄窗口信息注入/累积效率」，**不是信息预算**（同预算线性码已达 28.6 dB）。附：per-patch PCA（训练集共享基，rank 143/28,028 个数）**32.84 dB**、模型 A 相（单步读满 64 token、45 ep）**31.03 dB** —— 像素重建口径下**全局 token 瓶颈才是主要损失源**（Phase 2 验收另论；SVD/PCA 是线性 oracle 码，不构成「模型无用」）。⚠️ 单种子、B 相固定 6 ep（val 已收敛：末两 epoch +0.08 dB）、K=64 的 A 相与 10-21 那台机器同协议有 0.33 dB 差（跨机噪声）、像素指标 ≠ 任务保真。⇒ 报告 [`doc/2026-10-23/REPORT_n2bigstep.md`](doc/2026-10-23/REPORT_n2bigstep.md)（产物 `doc/2026-10-23/artifacts/n2bigstep/`：9 评测 json + 2 基线 json + 6 图 + 表 + 日志 + args；脚本 `tools/{smoke_bigstep,launch_n2bigstep,eval_n2bigstep,fetch_n2bigstep_artifacts}.sh` + `tools/{summarize_n2bigstep,plot_n2bigstep,baseline_svd_stepwise,plot_model_vs_svd}.py`）。

- **★ 容量阶梯：enc/dec 各 60M → 120M → 240M → 360M（n=2, T=16, 2026-10-24/26）** —— 用户口径「只跑 n/step=2，step=16，但编码和解码器参数量逐步增加：先各 60M，再各 120M，再各 240M，最后 360M；若非平坦再补一臂 n=2 × T=24（选最好容量档），跑完关机」。**scaling 方案 = 深度钉死基线（enc 13 / dec 7）、只加宽 dim**（单一容量轴；BPTT 层数不变 ⇒ 显存随宽度线性），用 `tools/scale_ladder.py` 精确计参校准 4 档：enc/dec **61.5/57.9M（119.3M 总）· 117.5/120.7M（238.2M）· 244.4/237.1M（481.5M）· 354.6/366.0M（720.6M）**；协议与 10-23 的 63M n=2×T=16 臂逐项一致（A 相 T=1 45ep/bs128/lr2e-4 → B 相 T=16 `fixed` n=2 6ep/等效bs64/lr2e-4 热启动），唯一自变量 = 宽度。**主结果（5,000 张 val，T=16 末步）**：**24.148（63M 基线）→ 25.747 → 27.308 → 27.936（各 240M，481.5M 总）→ 27.314（各 360M，720.6M）dB**（L1_norm 0.1877→0.1570→0.1320→0.1215→0.1305；SSIM 0.6638→0.7389→0.8039→0.8288→0.8015）⇒ **相邻边际 +1.60 / +1.56 / +0.63 / −0.62 dB**：容量到 240M（481.5M 总）饱和、**360M 反转 −0.62 dB**（三指标同向、> 跨机噪声 0.3–0.4 dB）⇒ 本协议（lr 2e-4 / 45 ep A 相）下**甜点 = 240M 档**；A 相（T=1）同构：28.728→30.242→31.205→**32.345**→31.977，且 **B−A 落差 4.10–4.66 dB 与容量无关 ⇒ 「n=2 窄窗口 T=16」的 B 相结构性损失不能靠加参数填平**（与 10-23 步数/宽度维度的结论互为独立证据）。参数效率：63M→238M 用 175M 参数买 +3.16 dB，238M→482M 用 243M 再买 +0.63 dB，482M→721M 用 239M **倒亏 −0.62 dB**。按「非平坦」判据（最大档 vs 次大档 −0.62 dB）**触发条件臂 n=2 × T=24 @ 240M（K=48 ⇒ A 相需重训，完整 A+B）**：**条件臂（n=2 × T=24 @ 240M，K=48，2026-10-25/26 跑完）**：A 相（T=1，K=48）32.063 dB（8.2 h），B 相（T=24、n=2、accum=4）6 ep 13.8 h → 末步 **PSNR 27.386 dB / L1_norm 0.1303 / SSIM 0.8051**，比同容量 T=16（**27.936 / 0.1215 / 0.8288**）**低 0.550 dB**；扣掉 K=48 的 A 相起点差（−0.282 dB）后**净效应 ≈ −0.27 dB**，且与 63M 档「T=16→24 = +0.57 dB」相反 ⇒ **容量越大加步数越不划算**（大模型 16 步已把窗口用满：最优步 15/16；T=24 臂最低点在第 20 步、最后 4 步无收益）。⇒ 本协议操作点仍是 240M × T=16；要加预算应加宽每步窗口而非加步数。。训练成本：4 档 A+B 合计 **53.9 GPU·h**（两卡并行墙钟 28.2 h）。⚠️ 单种子；**只测了宽度这一条容量轴**（深度钉死）；**lr/日程未随容量调**（大模型 A 相前期收敛更慢，360M 反转很可能是优化问题而非容量上限）；像素指标 ≠ 任务保真。⇒ 报告 [`doc/2026-10-24/REPORT_scale_ladder.md`](doc/2026-10-24/REPORT_scale_ladder.md)（产物 `doc/2026-10-24/artifacts/scale/`；脚本 `tools/{scale_ladder.py,scale_arms.sh,bench_scale.sh,run_scale_arm.sh,launch_scale.sh,eval_scale.sh,launch_scale_t24.sh,summarize_scale.py,plot_scale.py,fetch_scale_artifacts.sh}`；`train_linattn.py` 新增 `--save_opt`）。

- **★ 容量阶梯 × SAVOIAS 人评复杂度集 OOD 泛化（2026-10-26）** —— 把 10-24 容量阶梯的 5 个 ckpt（63M / 119M / 238M / 482M / 721M 总参数，均 n=2 token/步、T=16、K=32）拿到**全新评测机 `ssh -p 34314`（RTX 4080 SUPER 32G）**上，用 **SAVOIAS 人评复杂度集**（Art 420 + Scenes 200 + Objects 200 = 820 张，带 0–100 复杂度分）做 OOD 重建评测（canonical `eval_stepwise_linattn.py --data_dir_ood --domain savoias`，ckpt 从 23556 直连 34314 流式传输、5/5 md5 逐字节校验，23556 全程只读未干扰其 T=24 作业）。**主结果**：① **OOD 排序与 ID 完全同序、甜点仍是 482M**：OOD 末步 **L1 11.450 / 9.482 / 7.899 / 7.231 / 7.787 px**、PSNR **23.525 / 25.161 / 26.761 / 27.406 / 26.779 dB**、SSIM 0.6502/0.7298/0.7990/0.8257/0.7979，**721M 在 OOD 上同样反转**（−0.627 dB，幅度与 ID 的 −0.519 px 相当）⇒ 容量曲线的「单调→饱和→反转」不是 ID 特异的过拟合；② **OOD−ID 泛化差距随容量单调收窄**：ΔL1 **+0.631 → +0.436 → +0.289 → +0.227 → +0.264 px**（相对差 5.83%→3.24%）、ΔPSNR −0.62→−0.53 dB、ΔSSIM −0.0135→−0.0031（**收窄 77%**），且 63M→482M 的 OOD 收益（−4.22 px）**略大于** ID（−3.82 px）⇒ 本轮加容量买到的是更好泛化而非仅更好拟合；③ **人评复杂度效应稳健**：ρ(复杂度, 末步 L1) = **+0.33 ~ +0.54**（5 档 × 3 子集共 15 格**全正**），随容量仅从 0.489 降到 0.441；**绝对复杂度惩罚缩 −41%**（Art 5.66→3.63 px、Places −43%、MSCOCO −46%）但**相对惩罚基本不变**（Art ~65–68%）⇒ 容量把整条复杂度–L1 曲线按比例下压，不会让复杂图变得和简单图一样容易；④ **OOD 最优停步 11/13/13/15/16 与 ID 逐档一致** ⇒ 收敛步数是读出协议属性（n=2/T=16）而非数据属性；⑤ 跨协议衔接：本轮 482M（32 token）OOD 质量（7.23 px）已接近 10-19 旧 63M 臂 48–63 token 段（7.63/6.79 px），「容量可部分替代 token 预算」在 OOD 上成立（**绝对数跨协议不可直接比**）。⚠️ 单种子；OOD 域差混杂分辨率/构图差异；SSIM 来自补充 pass（`tools/ssim_pass_ood.py`，**fp32 计算**、同 `metrics_linattn.ssim` 口径、非 canonical 一次前向；首版误放 bf16 autocast 内会使 SSIM 偏低 ~0.05，已修正）。⇒ 报告 [`doc/2026-10-26/REPORT_ood_savoias_scale.md`](doc/2026-10-26/REPORT_ood_savoias_scale.md)（产物 `doc/2026-10-26/artifacts/ood_savoias/`：5 个 per_image + 5 个 stepwise json + summary.json + per_image_joined.csv + 4 图 + setup_log.md；脚本 `tools/plot_savoias_scale.py`）。

- **★ 240M 档 rFID 实测 + 同信息量逐块 PCA 基线（2026-10-26，为论文改写提供数据）** —— 用户口径「论文以 240M 档为基准改写、对比指标 rFID、对比方法用 PCA、其他深度学习方法只在同信息量单元格标方法名」。在 23556（仅 GPU0；GPU1 的 T=24 训练全程未动）用 `tools/eval_rfid.py`（pytorch-fid 0.3.0 / InceptionV3 pool3 / dims=2048 / resize 299；5,000 张 IN-100 val；GT 复用 `rfid_k64/gt`）实测：**本文 240M 档（enc 244.43M + dec 237.11M = 481.54M；K=32；T=16；n=2/步；fixed）逐 step rFID 115.30 → 49.30(8tok) → 26.83(16tok) → 19.97(32tok 末步，第 15 步 19.90 最优)**，PSNR 22.07→27.94 dB；**单步读出（A 相 T=1 读满 K=32）rFID 6.13 / PSNR 32.35 dB**；**同信息量 per-patch PCA（训练集共享基、非逐图 oracle；`tools/eval_rfid_pca.py`）rank 25/51/102/204 → rFID 45.50/23.31/9.02/2.12**。口径修正：240M 档 z_s 为 **32×1248 = 39,936 个连续值/图**（不是旧 63M 的 448 维），故同值数 PCA 对齐 rank 204（39,984，+0.12%）。**结论（如实）**：同存储值数下逐块 PCA 的 rFID/PSNR 明显更低（2.12 / 35.75 dB vs 本文 19.97 / 27.94 dB），原因是 PCA 保留 196 个 patch 位置的码（序列长度不降）、本文把整图路由进 32 个 token（6.12×、可截断）；论文据此把定位写成「单一模型 + 可截断 32-token 码」，不主张同信息量像素最优。⚠️ 单步 6.13 优于 16 步 19.97 主要来自 A/B 相训练预算不对齐（45ep 从零 vs 6ep 热启动）；rFID 只与 PCA 同协议可比，TiTok/ALIT 数值不并列（口径不同）。⇒ 报告 [`doc/2026-10-26/REPORT_rfid_m240_pca.md`](doc/2026-10-26/REPORT_rfid_m240_pca.md)（产物 `doc/2026-10-26/artifacts/rfid_m240_pca/`；脚本 `tools/eval_rfid_pca.py`；论文源 `~/paper/pp_zh/` 同步改写，含 8 个 `.bak_20261026`）。

- **★ K=64 的 rFID（ALIT 指标口径对齐，2026-10-21）** —— ALIT 用 **rFID**（pytorch-fid 0.3.0 / InceptionV3 pool3 / `dims=2048`，`recon_dir vs gt_dir`）报告重建质量，本仓库 K-sweep 只有像素指标 ⇒ 补测 **K=64 三臂**：`stdattn_k64_rec`（native square，递增，ALIT 同构）**15.89**、`stdattn_k64_t8`（cumulative）**19.86**、`stdattn_k64_t1` **8.32**（全量 5,000 张 IN-100 val、224²；与既有 PSNR 逐位交叉校验 28.582 / 27.760 / 31.027 dB）。K-sweep 上 rFID 与 PSNR 同向：**K=64 最低**、K=96/128 回升（35.10 / 48.80，比 K=64 高 +76.7% / +145.7%），且 rFID 对低 token 惩罚更重（K=8→64 **−84.5%**）。与 ALIT IN-100 @64 token 的 Ours-S **16.17** / Ours-B **15.88** **同量级**（⚠️ 数据划分、分辨率 224 vs 256、重建器连续 z_s vs VQGAN+GAN 全不同，**不可宣称追平**；低 token 段本仓库差 ≈1.6×，与「无 GAN 阶段」一致）。新增 `tools/eval_rfid.py`（GT 落盘 / 逐步重建 / pytorch-fid / scipy shim / PSNR 交叉校验）与 `tools/plot_rfid_k64.py`；逐 step rFID（3→64 token）与 K-sweep rFID 曲线见 [`doc/2026-10-21/REPORT_rfid_k64.md`](doc/2026-10-21/REPORT_rfid_k64.md)（产物 `doc/2026-10-21/artifacts/rfid_k64/`）。

- **★ 八臂架构矩阵消融（2×2×2：编码器注意力 × 解码器注意力 × 读出方式，2026-10-17 已跑完）** —— 旧实例 2026-10-17 **07:11:42** 全部完成落 `QUEUE_DONE` 后关机；**8 臂同进程同批次全量 5,000 张**评测（[`artifacts/eval_ablation8.json`](doc/2026-10-17/artifacts/eval_ablation8.json)）。**主结果**：`std enc+std dec 单步` **31.2186 dB / 0.9037 / 0.09013**（最好）→ `std+std BPTT` 28.5823 → `lin+std 单步` 26.6020 → `lin+std BPTT` 25.3043 → `std+lin 单步` 24.7138 → `std+lin BPTT` 24.1658 → `lin+lin BPTT` 22.0424 → `lin+lin 单步` **21.8844 / 0.5535 / 0.24241**（最差）。**三个配对结论**：① **编码器 std 注意力是最大单项**（同格内 **+2.12 ~ +4.62 dB**，均值 **+3.21**，4/4 全正；最大 +4.62 dB 在 std dec+单步格）；② **解码器线性化代价更大**（**−3.26 ~ −6.50 dB**，均值 **−4.73**，4/4 全负）⇒ **解码器注意力形式比编码器更敏感**；③ **BPTT 多步精修只在 std 解码器下成立**——std/std 的 BPTT 有 **8.82 px** 持续精修（Δ₁ 27.0%），两个线性解码器臂 Δ₁ 占比 **84%~97%**、总落差仅 **0.158~3.40 px**（链式循环几乎不动）。⚠️ **边界（必须同读）**：(a) **只有同进程同批次评测构成配对**，且**「单步」与「BPTT」两列不可直接比**（45 ep 从零 vs 6 ep 热启动，预算 7.5×）——「BPTT−单步」的差**不是** BPTT 的收益/代价；(b) 4 个 rec 臂有 **`dec_grad_ckpt` 混淆**（历史 `linattn_rec`/`stdattn_rec` 的 args.json 无该键=默认 False，新 `stdlin_rec`/`alllinear_rec`=True）；(c) `k64`/`k64_rec` 与其余六臂**可能换机训练**（仅评测同机）；(d) **单种子**（<0.10 dB 不作结论）；(e) 结论仅适用 **224/K=64/31.9M+31.2M**。⇒ **唯一权威报告** [`doc/2026-10-17/REPORT_attention_matrix.md`](doc/2026-10-17/REPORT_attention_matrix.md)（叙事 + 定性对比 + 核函数/梯度检查点等价验证 + 参数量/预算表 + 11 条诚实边界）；**自动聚合表** [`artifacts/report_tables_ablation8.md`](doc/2026-10-17/artifacts/report_tables_ablation8.md)（`python3 tools/alit_report.py --ablation8` 可复跑，4 位小数逐位可复核，同进程配对规则写死在表头）；四臂版前置消融 [`REPORT_stdattn_ablation.md`](doc/2026-10-16/REPORT_stdattn_ablation.md)、[`artifacts/ablation_table.md`](doc/2026-10-16/artifacts/ablation_table.md)。

- **★ 从零注意力实验线：linattn → stdattn（2026-10-15/16，已跑完两轮）** —— 按用户口径新增的**并列实验线**（`model_v2.py` 主线逐位未改）：不用预训练编码器，自建 13 层编码器（**31.9 M**，线性注意力 → 10-16 换**普通 softmax 注意力**）+ 7 层普通注意力解码器（**31.2 M**），224×224→224×224，ImageNet-100（121,689/5,000），**K=64**（N=196，3.06×）。10-16 配对消融（两臂训练内唯一自变量 = 编码器注意力形式，参数量差 **0.19%**；四臂**同进程**全量 5,000 张评测）：单步读出 **26.602 → 31.219 dB（+4.617）**，BPTT/T=8 读出 **25.304 → 28.582 dB（+3.278）**。⚠️ 单步 vs BPTT 的 31.219 / 28.582 **不可比**（训练预算差 7.5×：45 ep 从零 vs 6 ep 热启动），该报告 §9 已立此边界。⇒ [`doc/2026-10-15/REPORT_linattn_fromscratch.md`](doc/2026-10-15/REPORT_linattn_fromscratch.md)、[`doc/2026-10-16/REPORT_stdattn_ablation.md`](doc/2026-10-16/REPORT_stdattn_ablation.md)。**注**：本节 §1 描述的仍是 DINOv2 register 主线口径，未含本实验线。

- **★ MultiResSR 解码器 ×10 重跑 + 代码通用化（2026-10-14）** —— 新实例 **2×A800 80GB**：把 09-26 六源实验**解码器参数量整体提升 10 倍**（`depth 2→2`、`width 1024→3072`、`heads 8→24`、`mlp_ratio 4.0→5.12` ⇒ 解码器 **140.9 M → 1409.3 M（×10.001）**，全模型 445.3 M → 1713.6 M；参数量用 `tools/count_params.py` 实测，不是估的）。同时按用户要求把 multi 代码**去 CUDA 化 + 去手写循环**：新增 `srdiff_runtime.py`（NVIDIA CUDA / 海光 DCU(ROCm) / 昇腾 NPU / XPU / CPU 的设备与后端解析）、`tools/prep_data.py`（任意机器一条命令重建六源数据布局），训练循环/DDP/AMP/梯度累积/裁剪/调度器/续训全部交给 **HuggingFace Accelerate**（手写 `DDP`/`init_process_group`/`autocast("cuda")`/`torch.cuda.*`/lr 曲线已删除）。入口 `run_multi_10x.sh`（路径/设备/启动器/batch/精度/SDPA 后端全可环境变量覆盖）。**现状（2026-10-14 收工）**：外网 2×A800 臂跑到 **step 4240** 后按用户要求停止并关机（产 `ckpt_2000/4000`，各 20.56 GB）；**内网 DCU 8 卡臂在跑**（已跑通 → 调 batch OOM → 定策：不死磕 batch，改 `math+efficient`，显存给「关解码器重算」省 ~25% 墙钟）。**没有跑完的 ×10 臂** ⇒ 「×10 是否改变逐步精炼结论」留待内网跑完 + 同机补 ×1 臂。新增工具：`tools/{check_platform,diag_sdpa,plan_batch,prep_data,count_params}.py`。⚠️ 数据偏差：本实例无 `/root/autodl-pub`，HF 镜像的 vimeo90k triplet 只有 **73,191** clip（09-26 是 91,701）⇒ 索引 259,641 而非 278,151。⇒ 报告 [`doc/2026-10-14/REPORT_multi_portable.md`](doc/2026-10-14/REPORT_multi_portable.md)、**交接** [`doc/2026-10-14/HANDOFF_multi_10x_dcu.md`](doc/2026-10-14/HANDOFF_multi_10x_dcu.md)

- **（当前无其它在跑实验）** 上一轮（2026-09-26/27 六源多分辨率微调）已收工，见下。

- **★ 多分辨率版 MultiResSR：六源自然图像微调（2026-09-26/27，已跑完）** —— 新实例 2×RTX 4080S 32G：编码器**不设最大尺寸**（原生尺寸 + 长边 1280 + 面积分桶）、**解码 token 固定 144（square 12 步 0~11）**、**4 个解码器** 448×252 / 252×448 / 224×224 / 448×448、L1 权重按「原图最适配分辨率」给 **2/5 + 其余各 1/5**；数据 = ImageNet val 50k + COCO2017 train 118k + Vimeo-90k 91,701 + DIV2K 800 + Flickr2K 2,650 + OpenImages 15,000 = **278,151**。1 epoch / **40,242 步 / 12h33m** / bf16 + 编码器&解码器双梯度检查点 / 峰值 **23.52 GiB**（显存够；不用解码器检查点时最大桶 bs=4 OOM）。**逐 step 精炼 = 有但极度前置**：COCO2017 **val** 留出 5,000 张上 best-fit 像素 L1 **31.25 → 26.10 px**（单调、嵌套码），但**前 6 步已拿 91.7% 降幅**、第 7→11 步多花 **80 token（+127% 码率）只买到 0.20 px**、第 12 步 **+0.01 px 过冲**、早停 oracle 仅 **0.050 px** ⇒ [`doc/2026-09-26/RUN_multires_natural.md`](doc/2026-09-26/RUN_multires_natural.md)。⚠️ **边界**：留出集与训练**同域**（COCO train vs val），与历史 `construction_site` 的 16.42 px **不同域、不同 K、不同分辨率集合，不可直接比**。

- **★ 448×252 主线 `warm_steps` + 「固定4」采样计划（2026-09-24，**已完结：负结果**）** —— 在**工地数据集**上跑两相配方：`--step_plan fixed --block 4`（\|T\|=144 步、K=N=576）+ `--warm_steps 3504`（A 相 16 epoch 单步全读）+ B 相 40 epoch，纯 fp32 / 全局 bs32 / 2×RTX PRO 6000 96GB。**结论（2026-09-25 定稿）**：A 相健康（eval 0.377→0.367）但 **step 3504 硬切到 144 步固定窗口时崩溃** —— loss 尖峰 11.36 / grad_norm 132，随后**梯度消失**（~3e-3）、eval 冻结 1.138、推理 **144 步输出逐 step 相同**（L1 恒 65.59）⇒ [`doc/2026-09-24/REPORT_448_warm_fixw4.md`](doc/2026-09-24/REPORT_448_warm_fixw4.md)；机制见 [`doc/2026-09-23/REPORT_fixw4_plan.md`](doc/2026-09-23/REPORT_fixw4_plan.md)（"窗口不增长"）。

### 3.2 已判决的主线结论（按主题）

- **★ 结果总账（先读这份）**: [`doc/2026-09-16/RESULTS_v2_bptt.md`](doc/2026-09-16/RESULTS_v2_bptt.md) —— 实验矩阵 / 主结果 / 逐步曲线 / 可写进论文的四个点 / 一个被静默作废的判据 / 诚实边界。
- **循环 carry：detach → BPTT 是全部轮次里最大的单笔收益** —— 全量 test 像素 L1 23.72 → **16.42（−30.8%）**，24 步曲线由"平"转"降"，显存/步速**逐位不变**（不是拿算力换的）⇒ [`doc/2026-09-16/REPORT_v2_bptt_vs_detach.md`](doc/2026-09-16/REPORT_v2_bptt_vs_detach.md)。CPU 侧独立验证与机制拆解（`out_t = t·query_base + ΣΔ`、`∂L_t/∂query_base = 0`、−30.8% = −19.2% 整体 × −14.0% 轨迹精修）⇒ [`doc/2026-09-16/ANALYSIS_v2_bptt_cpu_verify.md`](doc/2026-09-16/ANALYSIS_v2_bptt_cpu_verify.md)。
- **BPTT 送到编码器的信号是"变强"而非"变弱"** —— `‖∂L/∂z_s‖` **×5.0**、扇出 1→9.49，且逐 step 梯度**相干**（cos 均值 +0.493）；真正弱的是① `mean_t` 平权使交付的末步只占 `1/|T|`、② 编码器每参数梯度仍只有解码器 ≈1/3 ⇒ 补信号应靠**损失加权**或给 z_s 一条不经交叉注意力的直接监督 ⇒ [`doc/2026-09-22/REPORT_crossattn_encoder_grad.md`](doc/2026-09-22/REPORT_crossattn_encoder_grad.md)。
- **K-sweep（BPTT 口径 8 点）** —— K=15→120 像素 L1 **15.02 → 12.51**；单卡↔DDP 口径差量化为 **21.62% / 3.9030 px**，据此**推翻**"单卡 K=99/120 反超 DDP K=576"；Phase-1 判定：K≈128 最优且干净、K≈32 可用、**K≈64 存疑须重跑** ⇒ [`doc/2026-09-16/REPORT_ksweep_bptt.md`](doc/2026-09-16/REPORT_ksweep_bptt.md)。
- **逐步细化形态** —— 有真细化，但落差 **≥88% 在前 3–4 步**、增量按 ~3–4 倍几何衰减、**末 2–3 步转负（过冲）**；仅 K≥48 明显 ⇒ **K 的有效容量远小于名义 K** ⇒ [`doc/2026-09-16/ANALYSIS_ksweep_stepwise_refinement.md`](doc/2026-09-16/ANALYSIS_ksweep_stepwise_refinement.md)。
- **★ 分辨率扫描「平线」的根因复核 + 224² 判决（推翻旧主结论）** —— 旧报告 `doc/2026-09-22/REPORT_resolution_sweep.md` 的主结论「多步递归训不起来」**被推翻**：像素数不是自变量（336² 与主线 448×252 的像素数/N/\|T\| 完全相同，却一个平线一个达标）；高分辨率臂其实**停在平凡解**（PSNR **低于** per-patch 均值基线）；近因是 A 相预算恒 1000 步而 224² 要 1500–2000 步。修正后 224² 曲线由平变**单调降**，最优 PSNR 14.26→**18.69（+4.43 dB）** ⇒ [`doc/2026-09-23/ANALYSIS_res_sweep_rootcause.md`](doc/2026-09-23/ANALYSIS_res_sweep_rootcause.md) + [`doc/2026-09-23/REPORT_224_A3000_verdict.md`](doc/2026-09-23/REPORT_224_A3000_verdict.md)。
- **★ 修正配方全分辨率重跑（A=3000 + B=7000）+「逃逸步数 vs N」** —— 推广到 28→336 后 **5/5 分辨率由平变降**（相对旧预算 **+3.39/+5.38/+6.82/+5.33/+4.26 dB**），28²/56² **反超 JPEG +5.2 dB**；**A 相逃逸步数随 N 单调增长**（N=4 → ≤500；N=576 → 2000–2500）⇒ 旧配方恒 1000 步对 224²/336² 根本不够；⚠️ **336² 是唯一没跨过平凡解线的点**（17.62 < DC 基线 17.83，且仍在涨、未收敛）⇒ [`doc/2026-09-23/REPORT_rerun_A3000_10k.md`](doc/2026-09-23/REPORT_rerun_A3000_10k.md)。
- **★ 采样计划对照：平方块 vs 固定 4** —— **固定宽度在高 N 上直接训不起来**（不是略差，是卡在平凡解）：相对平方块 28² **+0.79**、56² −0.09、112² −2.86、224² −3.51、336² **−4.03 dB**，**差距随 N 单调增大**，且代价 4.06×。**机制**：平方块窗口宽 `2k+1` 递增、末尾几步读 31–47 token 提供"锐"梯度，固定 4 下 144 步**全部只有 4 个 token** ⇒ 被 144 个"只能猜均值"的损失项主导 ⇒ **"窗口不增长"才是问题，不是"自然数步值"**。⚠️ **方法学发现**：这套训练**不是逐位可复现的**（同配置重跑 A 相仍有分叉，逃逸分叉处最大差 2.8 dB；但终点漂移仅 0.10/0.30 dB ≪ 干预效应 2.86 dB）⇒ **分叉附近的单点数字不可当精确值用** ⇒ [`doc/2026-09-23/REPORT_fixw4_plan.md`](doc/2026-09-23/REPORT_fixw4_plan.md)。
- **切片 `[0:12]` → `[1:12]`（去掉 t=1）** —— **−0.38 dB 且每个 t 都输**，还丢掉 RD 曲线最左端的操作点（2 token / 0.073 bpp）。**机制（最有价值）**：t≥9 各步读窗口两臂**逐位相同**却处处差 0.69–0.99 px，且"冷启动首步"用**更大**窗口反而更差 ⇒ **BPTT 让采样步构成串联草稿链，删一步的代价会摊到所有后续步**；`decoder_steps` **不是**"可自由裁剪的 token 预算表" ⇒ [`doc/2026-09-22/REPORT_slice1_224.md`](doc/2026-09-22/REPORT_slice1_224.md)。
- **★ 896×504 高分辨率臂 + 448×252 精度/优化器对照（fp16+8bit）** —— **本轮最重要的结果**：fp16+8bit 把 BPTT 的精修收益**整块抹掉**，跨度 −2.674 px（fp32）→ **−0.071 px**，与"关掉 BPTT"的 detach 对照臂（+0.072 px）**在数值上已无法区分**（训练快 3.26×；但 fp16 与 8-bit 同时换，**不能单独归因**）。896 臂是"在动且降误差"（−19.1%）但 12 步的 token 预算对 45 万像素**量级不够**（同 bpp 落后 WebP 最多 −11.25 dB）。**下一步最省的一步** = 448 上补 `fp16+torch-fused` 与 `fp32+8bit` 两臂分离变量 ⇒ [`doc/2026-09-22/REPORT_res896_448_fp16_8bit.md`](doc/2026-09-22/REPORT_res896_448_fp16_8bit.md)。
- **批 0 RD/早停（零训练）** —— **C2a 成立**（一个 checkpoint 出完整 `bpp–PSNR / bpp–MS-SSIM` + 逐图逐 step 曲线）；**C2b 不成立**（逐图自适应天花板仅 +0.13 dB——⚠️ 这是**逐图像素 RD 分配**的天花板，回答不了 §0.1「按场景精度给不同 K」是否提升高精度场景保真，那要等 Phase 2 文字指标）；**传统 codec 对照 PSNR 轴完败** ⇒ 见本节的判据口径。
- **曲线为何平坦：「纹理偏浅层」vs「解码器路由」** —— 用户的"浅层"方向有道理，**但根因排序不在这里**：8-27 诊断把第一根因判给**解码器的逐 patch 信息路由缺陷**（P1-1 `q_patch` / P1-2 special 输入绑定 patch 特征，可回收 12.9 L1），这两条**至今未落地**；把浅层接进来（layer_tap）画质**反而 −0.80 dB** ⇒ 支持"**先修路由、再谈 tap**"的排序 ⇒ [`doc/2026-09-22/ANALYSIS_texture_vs_decoder_routing.md`](doc/2026-09-22/ANALYSIS_texture_vs_decoder_routing.md)。
- **架构设计（循环为唯一路径）** —— 旧并行架构在 `blockdiag` 下"每步可用信息只有一块键 + 静态 `query_base`"，最优解退化为"交给 step-1、自己输出≈0"；解法 = 顺序循环让"当前画到哪"显式携带。**只作设计过程参考**（当前实现是其中"无开关"的子集）⇒ [`doc/2026-09-15/DESIGN_v2_recurrent.md`](doc/2026-09-15/DESIGN_v2_recurrent.md)。
- **论文实验方案 / 文献与数据集 / 拼接式自注意力对照** ⇒ [`doc/2026-09-22/PLAN_paper_experiments.md`](doc/2026-09-22/PLAN_paper_experiments.md)（实验矩阵、§0.1 早停零成本分析、E3 真 bpp、**E8 任务保真判别树**）、[`doc/2026-09-22/LITREVIEW_codec_papers_datasets.md`](doc/2026-09-22/LITREVIEW_codec_papers_datasets.md)、[`doc/2026-09-22/ANALYSIS_attn_concat_vs_cross.md`](doc/2026-09-22/ANALYSIS_attn_concat_vs_cross.md)（拼接式自注意力 vs 交叉注意力：**掩码拼接逐位等价**，无掩码不等价）。
- **新服务器验收：AutoDL bjb1 / 1× A800-80GB** —— 网络/pip/镜像源与吞吐实测（28² 31.7 → 336² 1.08 it/s），只有 1 张卡 ⇒ `NUM_GPUS=1` ⇒ [`doc/2026-09-23/SERVER_bjb1_a800_test.md`](doc/2026-09-23/SERVER_bjb1_a800_test.md)。
- **★ 官方 ALIT 单卡复现（旁证，非主线）：实测预算 + 5 处兼容 patch** —— 官方代码 `ShivamDuggal4/adaptive-length-tokenizer@8c0b4ec` 在本机跑通（`alit_tiny`=论文 Ours-S，**52.8M 可训**，256²、IN-100、stage-1 8 rollout 全监督）：**0.478 s/iter、1.05 h/epoch**（bs16、91% GPU、20.3 GB），⇒ 论文 stage-1 200 epoch 单卡 **8.6 天**（16 GB 卡 **13.8 天**；`sample_grad_iters=2` **5.7 天**）。本次按 30 epoch 起步、**epoch 0 完成后按用户要求关机**（epoch-0 阶梯 62.6±0.3/255 是 warmup 启动态基线，**不能**当"阶梯不存在"的证据）。⚠️ base tokenizer 用 HF `Nehc/vqgan_imagenet_f16_1024` 替代 Mage 版（PSNR 20.2 dB，偏糊）⇒ **绝对 FID 不可与论文比，只比阶梯形状** ⇒ [`doc/2026-10-17/REPORT_alit_official_repro.md`](doc/2026-10-17/REPORT_alit_official_repro.md)（产物：`doc/2026-10-17/artifacts/alit_official_repro/`）。
- **★ SAVOIAS 人评复杂度 × BPTT token 预算（ALIT Fig.4/11/12 复刻，2026-10-19）** —— 补上「我们没有 Image Complexity Human Score」的缺口：不在自己的 val 上标复杂度，而是用公开 **SAVOIAS** 的 3 个 OOD 子集（People-Art 420 / Places 200 / MSCOCO 200，人评 0–100），让 ImageNet-100 训好的 `stdattn_k64_rec`（与 `k64_rec` 对照）在 step1–8（token 3→64）上重建。**ALIT 的定性结论复现**：复杂度与像素 L1 在**全部 8 个预算**上正相关（std 臂 ρ=**+0.33~+0.58**、线性臂 +0.37~+0.60）；复杂度最高/最低三分位的 L1 差距 **6.97 px（3 tok）→ 3.01 px（63 tok）**（People-Art），即**复杂图从加 token 中获益更大**；`ρ(complexity, needed_step)=+0.33~+0.54`；定性网格显示简单静物 3 token 即饱和、密集古典画 63 token 仍偏软。⚠️ 边界：OOD（ALIT 同样承认）、小样本单种子、只做 3/7 子集、token 范围 3–64（ALIT 32–256）、像素口径非任务保真。⇒ [`doc/2026-10-19/REPORT_savoias_complexity.md`](doc/2026-10-19/REPORT_savoias_complexity.md)（脚本 `tools/{fetch_savoias.sh,run_savoias_eval.sh,plot_savoias_complexity.py,plot_savoias_heatmap.py,render_savoias_grid.py}`；产物 `doc/2026-10-19/artifacts/savoias_complexity/`，含**论文版式热力图**——x=Image Complexity Human Score（15–85）、y=Number of Tokens、颜色条取色自 ALIT Fig.4 且**归一到 0–1**（绝对 pixel L1 版本另存 `_abs`）——及逐图热力图、原始分箱对照版）。

### 3.3 已归档 / 已从主线移除（细节见 doc，勿当当前行为）

| 主题 | 判决 | 文档 |
|---|---|---|
| **layer_tap 逐层金字塔** | 能买到**曲线斜率**（PSNR 跨度 0.30→1.69 dB）但**买不到画质**（输 0.80 dB），且自适应天花板仍仅 +0.269 dB ⇒ **2026-09-23 整块从 main 移除** | `doc/2026-09-22/DESIGN_layer_tap_pyramid.md`、`REPORT_layer_tap_pyramid_224.md` |
| **块级共享位置编码 `3151bab` 塌缩** | 三路独立取证 + 对抗裁决；该变体下"塌缩一旦发生就没有免费的恢复梯度"（实测 `0.00e+00`） | `doc/2026-09-10/ANALYSIS_k3_posenc_failure.md` |
| **`query_mask_mode` 掩码开关** | 开关**已删除**；块级共享位置编码塌缩归因的配套设计与实测留档 | `doc/2026-09-10/DESIGN_query_mask_mode.md` |
| **前端数据工作（CoT / 翻译 / 数据集盘点，2026-09-17→20）** | 全量 CoT **75,717 条**（99.995% 解析成功，42h20m）；**翻译质检：major 21.09%**（推翻旧"3–5%"，低估 5–8 倍）；Qwen3.8-27B **不能当文本评委**（误报率 0.60–1.00，会编造证据）；多模态数据集盘点 **202GB / 12 repo** | `doc/2026-09-18/REPORT_cot_full.md`、`REPORT_translation_audit_deepseek.md`、`REPORT_llm_judge_calibration.md`、`doc/2026-09-17/ANALYSIS_mm_datasets_inventory.md` |
| **历史版本（v1/v3/v4/v5）代码** | 已迁出 main | 见 **test** 分支 |

---

## 4. 阶段性目标 / 未决问题

> ⚠️ **本节历史包袱较重**：早期"后步输出≈0 / 零增量自锁"的结论是在 **detach 口径**下得到的，在 **BPTT + 足够 A 相预算**下**已不是当前现象**（§3.2 的 224² 判决与全分辨率重跑）。下面保留框架与判据，但**现状以 §3.2 为准**。

### 4.1 待决：让"后步真干活"

**现象与机制裁决**（详见 `doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`）：
旧损失把每个采样步的累加结果都监督成整图 + 按步解耦 ⇒ step-1 的子任务 = 完整重建，后步的子任务 = 预测 step-1 没做出来的残差 ⇒ 残差从后步可读的键里不可预测 ⇒ L1 最优预测 = 零（条件中位数）⇒**"零增量自锁"**。这是自洽均衡，不是某一层单独的病。

**两条可干预的耦合**：

| 耦合 | 表现 | 干预手段（候选，**均未落地**） |
|---|---|---|
| **步间任务同构** | 每步监督在函数形式上等价（现为 `L1(PixelHead(Y_t), 整图)`、仍平权、各收 1 份梯度）——"哪一步干活"对损失中性 | **P1 分区域掩码损失**：step t 只监督自己对应的 patch 行区，制造 step-1 抢不走的**私有目标** |
| **键的内容可寻址性不足** | 后区键的读出内容高度相似（slice27_v2 位置 16..63 两两 cos 0.963–0.974），解码器难以**按内容指名**某一位 | **P2 编码侧注入**：special 输入拼 `Linear(patch_feat)`，让键有"各自携带逐 patch 内容"的可能 |

要点：① 两者需**同时**处理（只给私有目标则键仍无可读内容，只改键则损失仍奖励后步归零）；② 手段不是外部正则"推一把"，而是改损失/改输入通路，让均衡点本身失稳；③ 已证伪的单侧手段：去解耦/后步加权（17.46 vs 18.77）、课程 fade-in、残差目标显式化。P1 曾实现并通过冒烟，但**代码已还原**（HEAD `a8eeabc`）；若走渐进路线按 `doc/2026-09-07/DESIGN_v2_region_loss.md` 重新落地即可（~20 行核心改动）。

**判据修正：「键相似度」不能当目标或观测特征**：高块内相似度在**健康**模型里普遍存在（09-02 健康 K=576 模型块内两两 cos = 0.915），且 `pos_embed` 是加性偏移 ⇒ 该判据**既不必要也不充分**。仍可用的判据（一律用**匹配步**，历史对比用 `eval_recon` 而非 `eval_loss`）：① 后步 `\|W·Y_t\| ≫ 0.015`；② 渐进曲线出真阶梯（覆盖区单调降）且 `eval_recon` 不劣化；③ 各步**隔离** L1 在自己区域上相近。

**掩码侧已排除**：`blockdiag` 复跑把 `eval_recon` 从 0.3584 改善到 0.3318，但同尺探针实测 `step_px_scale` 后几步仍只有 3.3–3.8% ⇒ 增益来自**单发通路收敛更好**，不是后步分工被激活。

**战略上下文**：渐进阶梯**不是** GOAL 验收项（Phase 2 一次性消费全部 K token）；v4 单发（8.26）已是仓库最佳。⇒ 除非"token 增量性"叙事本身成为目标，**本条目可长期冻结，不挡主路线**。

> 相关文档：`doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`（主分析）、`doc/2026-09-07/ANALYSIS_k3_slice_infodiff.md`、`doc/2026-09-07/REPORT_v2_E1_probe.md`（信息侧证据）、`doc/2026-09-07/DESIGN_v2_region_loss.md`（P1 设计）、`doc/2026-09-15/DESIGN_v2_recurrent.md`（循环架构）

### 4.2 下一步（与 §0.1 的信息密度 / 精度自适应目标对齐）

1. **P0｜K 压缩 × 重建质量**（Phase 1 核心验证，`GOAL` §5）：576 → 32/64/128 重训，标定"多少 token 够还原活信息"——这是"信息密度"设想最直接的一步；
2. **P1｜NLP 就绪探针**（**唯一验收**）：冻结编码器 → MLP → Qwen，小批量训文字，并**首次引入 caption/幻觉指标**（当前项目一个都没跑过，见 §0.1 边界）；
3. **P2｜先修解码器路由**（P1-1 `q_patch` / P1-2 special 绑定 patch 特征），再谈其它读出增强；
4. **P3｜损失加权**（`mean_t` 平权使交付的末步只占 `1/|T|`），给编码器补信号。

---

## 附：仓库结构（2026-09-04 整理，2026-09-24 精简说明）

| 分支 | 内容 |
|---|---|
| **main**（本分支） | ① v2 最小可运行代码 ② `doc/` 全部实验/分析文档 |
| **test** | 除文档外的历史内容归档：v1/v3/v4/v5 代码、boundary 实验、deepspeed 配置、旧 README（25 个根文件，无 doc/） |
| dev | **已删除**；本地安全 tag `pre_reorg_dev` / `pre_reorg_test` 可找回旧 tip |

- v2 最小可运行集 = `model_v2.py`、`train_v2.py`、`infer_v2_test.py`、`data_v2.py`、`visualize_recon_pixel.py`、`run_v2_train.sh`、`requirements.txt`、`.gitignore`。
- **复现历史实验**：`doc/` 里 2026-09-02 及更早的报告会引用 `train_v3..5.py` 等文件，它们**只在 test 分支** ⇒ `git checkout test` 后再跑，或 `git show test:<path>` 取单个文件。
- **文档头部标注的代码 commit 与 main HEAD 可能不一致**：机制分析/复现要**以文档头部的 commit 为准**（`git checkout <该 commit>`），不要假设 main HEAD == 当时代码。
- 新报告请提交回 **main** 的 `doc/<日期>/`，避免文档在两边分裂。
