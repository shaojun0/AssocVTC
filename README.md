# SR-Diffusion-v3（main 分支: v2 最小可运行代码 + 全部实验文档）

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
