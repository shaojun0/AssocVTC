# PLAN — 参照 ALIT 的自适应长度 tokenization 实验规划（AssocVTC）

> 日期：**2026-10-16**
> 上游依据：**ALIT**（[arXiv:2411.02393](https://arxiv.org/abs/2411.02393)，MIT CSAIL）
> 结构对照（**先读**）：[`ANALYSIS_alit_vs_assocvtc.md`](ANALYSIS_alit_vs_assocvtc.md)
> 现状依据：[`doc/2026-10-15/REPORT_linattn_fromscratch.md`](../2026-10-15/REPORT_linattn_fromscratch.md)、
> [`doc/2026-10-16/REPORT_stdattn_ablation.md`](REPORT_stdattn_ablation.md)、
> [`doc/2026-08-28/GOAL_compression_for_nlp.md`](../2026-08-28/GOAL_compression_for_nlp.md)、`README.md`
> 状态：**规划（未执行）**。本文件不含任何实测结果；写代码前请先复核 §6 的改动清单。
> 适用机器：单卡 **RTX 4080 SUPER 32 GB**（10-15/10-16 两台实例均可，吞吐见 §7）

---

## 0. 一页速览

**总目标**：把 ALIT 的三条可移植机制（**逐轮新增记忆** / **动态停机** / **逐迭代损失权重**）与
它的两条分析口径（**TSC–TOI**、**熟悉度/熵**）落到 AssocVTC，回答一个对项目主线有判决意义的问题：

> **"token 数随场景精度自适应"在本项目数据与架构上，到底值多少？**

| 阶段 | ID | 实验 | 自变量 | 需要训练？ | 成本 | 判决判据 |
|---|---|---|---|---|---|---|
| **A 零训练** | E-A1 | 每图逐步曲线 + oracle 停步 | — | 否（复用 ckpt） | 1 h | 产出 per-image L1[t] |
| | E-A2 | 自监督 TSC（重建阈值选 token 数）| 阈值 τ | 否 | 1 h | 等平均 token 数下 TOI 是否 ≥ 固定 K |
| | E-A3 | IID / OOD 熟悉度阶梯 | 数据集距离 | 否 | 2 h | OOD 图所需步数是否显著更多 |
| | E-A4 | 复杂度相关性（弱代理） | — | 否 | 1 h | Spearman ρ 及其显著性 |
| | E-A5 | z_s 冗余度 / 注意力稀疏度 | — | 否 | 2 h | 有效秩、两两 cos、注意力熵随迭代变化 |
| | E-A6 | 边际信息分解（Δ_t 与类别/停步） | — | 否 | 1 h | Δ_t 是否 ≥88% 前置（复现旧结论） |
| **B 重训** | E-B1 | 逐迭代损失权重（ALIT 配方） | 4 种权重 | 4×1.6 h | 6.4 h | 末步 L1 改善 > 0.10 px 且曲线更单调 |
| | E-B2 | 损失侧动态停机（P1） | 4 臂 | 4×1.6 h | 6.4 h | README §4.1 三条判据（见 §8） |
| | E-B3 | **ALIT 式自适应记忆**（核心） | 3 臂 | 2×2 h + 代码 | 4 h | 同参数预算下 PSNR/保真 + Δ_t 形态 |
| | E-B4 | 随机预算采样 / 一条流多档 | 1 臂 + 3 基线 | 4×2.5 h | 10 h | 截断档是否追平专训档 |
| **C 验收** | E-C1 | 任务保真下的自适应 K（E8） | TSC × TOI | 复用 B | 3 h | 等预算下自适应 ≥ 固定 K |
| | E-C2 | Phase 2 NLP 探针（唯一验收） | — | 独立排期 | 另计 | caption/幻觉指标 |

**推荐执行顺序**：E-A1 → E-A2/A3 → （评审）→ E-B1 → E-B2 → E-B3 → E-B4 → E-C1。
**约 30 GPU·h（单卡 4080S ≈ 1.5–2 天墙钟 + 评审）**，其中零训练部分占 8 h 且**信息量最大**。

**本轮坚持的三条口径**（违反则实验作废）：
1. **目的不是 PSNR**：RD/PSNR 线已判放弃（README §3）；PSNR 只作信息保持探针，且必须给出
   per-patch DC 基线（`metrics_linattn.py` 已有）；
2. **一次只动一个自变量**，配对臂同进程评测（`eval_linattn.py` 已具备四臂同进程能力，可复现到 1e-7 dB）；
3. **零训练能回答的，绝不先花 GPU 训练**。

---

## 1. 设计原则（从 ALIT 的哪条结论推出来的）

| # | ALIT 的结论 | 本项目的对应问题 | 原则 |
|---|---|---|---|
| 1 | 不同图的表示容量不同（熵/熟悉度） | §0.1"精度自适应"只有动机没有实现 | 先用**同一个 ckpt 的逐步曲线**测上界，再决定要不要训 |
| 2 | 重建损失可当**自监督 TSC**，≈60% 预算即可接近满预算任务性能 | batch-0 早停只测了 PSNR 口径（+0.13 dB） | TSC 实验必须用**任务保真**口径重做（E-A2/E-C1） |
| 3 | 每轮**新增** token 让后步有活干（记忆增长） | README §4.1"后步真干活"未决 | E-B3 是唯一"改信息通路"的手段，优先级高于加权 |
| 4 | 动态停机把算力集中到未重建好的区域 | P1 区域损失设计稿 | 现有单体编码器**装不下输入侧掩码** ⇒ 先做损失侧 |
| 5 | 逐迭代权重 `[2.5,2,1.5,1.25,1,…]`（递减） | P3（`mean_t` 平权使末步只占 1/|T|） | E-B1 是**最便宜**的一臂，先跑 |
| 6 | 解码器无 carry ⇒ 每档独立、截断合法 | 我们 BPTT 是"串联草稿链"，删一步伤全部后续步 | 若"单流多档"是目标，E-B4 必须裁定这个 trade-off |

---

## 2. 现状盘点（截至 2026-10-16，全部为**已有**资产）

### 2.1 数据与口径

| 项 | 值 |
|---|---|
| 数据集 | ImageNet-100，`/root/autodl-tmp/datasets/imagenet100`（100 类目录） |
| 划分 | 确定性等距：train **121,689** / val **5,000**（`val_per_class=50`） |
| 分辨率 / patch | 224×224，patch=16 ⇒ **N=196** |
| K | **64**（压缩率 196→64 = **3.06×**） |
| 归一化 | `IMAGENET_MEAN/STD`（[0,1] 标度；换算 0-255 必须乘 255，见 `metrics_linattn.py` §4 坑） |

### 2.2 可用 ckpt（四臂，配对可复现）

| 臂 | 路径 | 全量留出集（5,000 张） |
|---|---|---|
| 线性编码器 · 单步 | `/root/autodl-tmp/srdiff_linattn/k64/ckpt_best.pt` | PSNR 26.602 · SSIM 0.7654 · L1(norm) 0.1463 |
| **std 编码器 · 单步**（当前最佳） | `/root/autodl-tmp/srdiff_stdattn/stdattn_k64/ckpt_best.pt` | **31.219 · 0.9037 · 0.0901** |
| 线性编码器 · BPTT T=8 | `/root/autodl-tmp/srdiff_linattn/k64_rec/ckpt_best.pt` | 25.304 · 0.7156 · 0.1666 |
| **std 编码器 · BPTT T=8** | `/root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt` | **28.582 · 0.8505 · 0.1147** |

### 2.3 已有工具（E-A 阶段几乎零代码）

| 脚本 | 能力 |
|---|---|
| `eval_stepwise_linattn.py` | 逐 step 直接预测 → `L1_255[t]` / `PSNR[t]` / `Δ_t` / 总落差 / Δ₁ 占比 / **每图最优停步分布**（5,000 张 23 s） |
| `eval_linattn.py` | 多 ckpt **同进程**全量评测（5,000 张 56 s），PSNR/SSIM/L1 |
| `probe_linattn.py` | 吞吐/显存探针（bs/K/T/是否 BPTT） |
| `viz_ablation.py` / `ablate_stdattn.py` | 可视化与对照表 |
| `tools/prep_data.py` | 六源数据布局重建（含 COCO2017 / OpenImages / DIV2K 等，供 OOD 用） |
| `metrics_linattn.py` | `psnr` / `ssim` / `per_patch_dc_baseline` / `mean_image_baseline` |

### 2.4 已判决、本 PLAN 不重复的结论

- `mean_t` 平权下**逐步细化 ≥88% 在前 3–4 步**、增量按 3–4 倍几何衰减、末 2–3 步过冲
  （`doc/2026-09-16/ANALYSIS_ksweep_stepwise_refinement.md`）；
- std 编码器下 Δ₁ 只占 **27.0%**（线性臂 37.7%），**97% 的图在 step 7 最优**，step 8 的 ΔL1 = **−0.016 px**
  （`REPORT_stdattn_ablation.md` §4）⇒ **K=64 的有效容量 ≈ 63**；
- 删采样步会伤全部后续步（`doc/2026-09-22/REPORT_slice1_224.md`）⇒ 不自由的裁剪；
- `--step_plan fixed` 在高 N 上训不起来（`doc/2026-09-23/REPORT_fixw4_plan.md`）⇒ 读窗口必须递增。

---

## 3. 阶段 A — 零训练实验（复用现有 ckpt，1 天）

> 阶段 A 的全部目的：**在不训练的前提下，量化"自适应 token 数"的天花板与判据口径**。
> 若 A 阶段就给出否证，B/C 阶段直接停，节省 ~30 GPU·h。

### E-A1 每图逐步曲线 + oracle 停步（基础数据）

- **假设**：不同图的最优停步分布足够宽（方差大）⇒ 存在可利用的图间差异。
- **做法**：`eval_stepwise_linattn.py` 增加 `--dump_per_image`，对 `stdattn_k64_rec`（T=8）
  输出每图每步的 `L1_255`、`PSNR`、最优停步、类别索引、文件名。
- **产出**：`doc/2026-10-16/artifacts/alit/per_image_stdattn_rec.json`。
- **要看的量**：停步分布的方差/熵、oracle 总 token 数 vs 固定 step7 的 token 数之比。
- **代码**：`eval_stepwise_linattn.py` +约 30 行（`--dump_per_image`）。

### E-A2 自监督 TSC：重建阈值选 token 数（ALIT Fig.6 的复刻，换口径）

- **假设（ALIT）**：用**重建损失**当自监督 token 选择准则，可在**显著少于满预算**的平均 token 数下
  保住任务性能（ALIT 报 ≈60%）。
- **做法**：对每图取**最小**满足 `L1_255[t] ≤ τ` 的 t（读窗口 hi+1 即该图实际 token 数）；
  扫 τ 得到"数据集平均 token 数 ↔ 平均 L1/PSNR"曲线；再与**固定步**（同等平均 token 数）对照。
- **对照臂**：固定 step ∈ {3,5,7,8}（对应不同固定 token 数）——**等平均预算配对**是本实验的核心。
- **判决**：若在等平均 token 数下，自适应 TSC 的 L1 显著优于固定步（>0.10 px 且方向一致），
  则 §0.1 的"按图自适应 K"在**信息保持**口径下成立；否则记为"PSNR 口径不成立"，
  把判决权交给 E-C1（任务保真）。
- **诚实提醒**：batch-0 已测"逐图像素 RD 分配天花板仅 +0.13 dB"（`REPORT_batch0_rd_earlystop.md`）。
  本实验**预期会复现该结论**；它的价值在于**把它换成 ALIT 的 TSC 语言并给出可视化**，
  为 E-C1 提供"若换任务保真口径还有没有戏"的对照基线。
- **代码**：新增 `tools/alit_adaptive_k.py`（纯分析，读 E-A1 的 json，~120 行）。

### E-A3 IID / OOD 熟悉度阶梯（ALIT §4 第二条）

- **假设（ALIT）**：越 OOD 的图需要越多 token；IID 图能用更少 token 重建好。
- **做法**：同一个 `stdattn_k64_rec` ckpt，在同一进程跑三级：
  1. **IID**：IN-100 val（5,000，已有）；
  2. **中距离 OOD**：COCO2017 val 2017（`tools/prep_data.py` 布局，取 5,000 或全量 5k）；
  3. **远距离 OOD**：`construction_site`（项目自有工地数据，对 IN-100 模型完全 OOD）
     或 OpenImages 子集（与 IN-100 类别分布不同）。
- **度量**：每级 (a) 逐步 L1_255 曲线；(b) 达到同级固定阈值 τ 所需的平均步数；
  (c) "第 1 步 L1" 与 "step7 L1" 的差距。**预测**：OOD 的 Δ(所需步数) > 0 且单调。
- **判决**：若三级平均停步差 <0.5 步 ⇒ 熟悉度假设在本项目数据上不成立，E-C1 的动机削弱。
- **代码**：`data_linattn.py` 新增 `build_ood_dataset(dir)`（无标签 ImageFolder + val transform，
  ~40 行）；`eval_stepwise_linattn.py` 加 `--data_dir_ood`。

### E-A4 复杂度相关性（弱代理，诚实标注）

- **假设（ALIT §4 第一条）**：图越复杂（熵越高）越需要更多 token。
- **限制**：ALIT 用 SAVOIAS 的**人标注复杂度**（0–100），本项目没有人工标注。
  **只能用弱代理**：JPEG 文件大小 / 边缘密度（Sobel 或 Canny） / 目标尺寸下的残差
  空间方差 / 每图最优停步。
- **做法**：算每图代理量与"最优停步/所需 token 数"的 **Spearman ρ** 与 bootstrap 置信区间。
- **诚实边界**：ρ 显著**不等于**证明了 ALIT 的熵假设（代理量 ≠ 感知复杂度）；报告必须写明。
- **代码**：并入 `tools/alit_adaptive_k.py`。

### E-A5 z_s 冗余度 / 注意力稀疏度（对应 §0.1 代理 #3"冗余度"）

- **假设（ALIT）**：随预算/迭代增加，token 从"全局摘要副本"转向"稀疏局部绑定"；
  我们的 z_s 是**一次前向固定产物**，因此**预期不会有这种随迭代的专精**——
  这正好是该实验要证伪/证实的点。
- **度量**：
  1. z_s 两两 cosine 相似度（均值/分位）、有效秩（奇异值谱）、逐 token 方差；
  2. 解码器**每步交叉注意力的熵**（对 z_s 前缀）随 t 的变化：是否出现"注意力集中到少数 token"；
  3. token 使用率：每步 Δ_t 与所读 token 的对应（哪几个 token 真正带来降幅）。
- **代码**：`metrics_linattn.py` 加 `zs_stats()`；注意力需 forward hook 或重写交叉注意力打分
  （`nn.TransformerDecoderLayer` 不直接返回权重）——**本项是 A 阶段唯一中等成本项**，可后置。
- **判决**：若 z_s 高度冗余（有效秩 ≪ K 且两两 cos > 0.9），则 README §0.1 的"576 个副本"问题
  在 K=64 下仍未解决，E-B3（新增记忆）的动机变强。

### E-A6 边际信息分解（复现 + 细化）

- **做法**：把 `Δ_t` 与类别、图像复杂度代理、（E-A3 的）域标签交叉；检验"末步过冲"是否集中在某类图。
- **代码**：并入 `tools/alit_adaptive_k.py`（并复用 `REPORT_stdattn_ablation.md` §4 的表格式）。

---

## 4. 阶段 B — 低成本重训（每臂 1.6–2.5 h，单卡 4080S）

**统一协议**（除自变量外逐项固定，便于配对）：
- 数据 IN-100 224×224，K=64，`--enc_attn std`（13 层/448/8 头，31.848 M），
  解码器 `--dec_recurrent 1`（T=8，7 层/512/31.207 M）；
- **从 `stdattn_k64/ckpt_best.pt` 热启动**（阶段 1 单步 ckpt），阶段 2 预算 **6 epoch / 11,406 步 / bs64 / bf16**，
  AdamW lr 2e-4 cosine、warmup 8%、wd 0.05、grad_clip 1.0、`--skip_grad 50`；
- 评测：`eval_linattn.py`（全量 5,000，同进程）+ `eval_stepwise_linattn.py`；
- **基线行**：`stdattn_k64_rec`（28.582 dB / L1_255 6.613 / Δ₁ 27.0%）与
  `stdattn_k64`（31.219 dB，单步上界参考）。

### E-B1 逐迭代损失权重（P3；ALIT `lambda_loss_weight`）

- **假设**：`mean_t` 平权让交付的末步只占 1/|T|；改为**递减权重**（ALIT 原版）或**末步加权**
  能改善末步重建且不破坏逐步单调性。
- **4 臂**：
  1. `mean`（现状基线，已有）；
  2. **ALIT 原版递减** `[2.5,2.0,1.5,1.25,1.0,1.0,1.0,1.0]`；
  3. **递增** `[1,1,1,1,1.25,1.5,2.0,2.5]`（"让后步干活"的直觉版）；
  4. **last-only**（只在最后一步算 loss，对齐 ALIT stage2 的 `sample_grad_iters`）。
- **验收**：末步 L1_255 相对基线改善 **>0.10 px**（超过单种子噪声口径）且
  step7→step8 的过冲不放大；逐步曲线不出现新的平坦段。
- **判决**：若 4 臂差异 <0.10 px ⇒ 按 P3 关闭该方向（写进 README §4.1）。
- **代码**：`model_v2_linattn.py` 加 `--loss_weights`（空 = 平权，默认逐位不变），`train_linattn.py` 透传。

### E-B2 损失侧动态停机 / 区域损失（P1，ALIT dynamic halting 的可落地子集）

- **假设**：让每步只对"自己负责且尚未重建好"的 patch 负责，能激活后步分工。
- **4 臂**：
  1. baseline（平权全图）；
  2. **硬区域掩码**：step t 只监督第 t 块对应的 patch 行区（P1 原设计，`DESIGN_v2_region_loss.md`）；
  3. **软停机**：按上一步**逐 patch 重建误差**的每图分位阈值（如保留误差最大的 50% patch 参与后续监督），
     ≈ ALIT 的 `pred_gt_index_prob < 0.55` 换成像素残差口径；
  4. 3 + 对已"完成" patch 的损失权重置 0 并做 **stop-gradient**（防止把梯度花在已解决区域）。
- **验收（README §4.1 三条，必须同时看）**：
  1. 后步 `‖W·Y_t‖ ≫ 0.015`（后步不在输出零增量）；
  2. 渐进曲线出**真阶梯**（覆盖区单调降）且 **eval_recon 不劣化**；
  3. 各步**隔离** L1 在自己区域上相近。
- **⚠️ 架构边界（来自 ANALYSIS §3.2）**：ALIT 原版的动态停机作用在**编码器输入**上，
  而我们 BPTT 的循环在解码器、编码器是单次前向 ⇒ **输入侧掩码在本结构里装不下**。
  本实验只做**损失侧**；输入侧停机归入 E-B3。
- **代码**：`model_v2_linattn.py` 加 `--dh_mode {0,1,2,3}` + `--dh_quantile`；`train_linattn.py` 透传。

### E-B3 ALIT 式自适应记忆（**核心实验**）

- **假设（ALIT 最关键的一条）**：让后续步有**新键**可读（而不是读同一批固定键的更多前缀），
  是激活"后步真干活"的正交解法，且优于纯损失加权。
- **方案 a（推荐，最小改动，读侧生长）**：
  在编码器末层输出上接一个**共享的 GrowthBlock**（1–2 层 cross-attention，d=256，约 1–2 M）：
  `L_it = GrowthBlock([L_{i(t-1)} ; patch_feat])`，每步新增 **16** 个生长 token，
  循环解码器的读集合变为 `[z_s ; L_1..L_it]`（累计 16/32/48/64）。
  **代价**：不改编码器、不改解码器本体、不动 ckpt 兼容性（新增 key 只在 GrowthBlock）。
- **方案 b（ALIT 忠实版，写侧生长，可选）**：
  在编码器之上加小型共享 `RefEnc/RefDec`（2–4 层，d=256，约 4–6 M），
  逐轮把 patch 特征蒸馏成新 latent（+16/轮 × 4 轮），每轮独立解码 + 直接预测监督
  （ALIT 口径：无 carry，每档独立可截断）；此时**动态停机的输入侧版本**才可一并实现。
- **3 臂配对**：
  1. `stdattn_k64_rec`（现状 BPTT，28.582 dB）——**必须复跑同预算对照**，不能直接引用旧数；
  2. 方案 a + 同预算（6 ep 热启动）；
  3. 方案 a + TSC 训练（随机预算，见 E-B4）——若预算不够，E-B3 只跑臂 2，臂 3 并入 E-B4。
- **验收**：同参数/同步数下 (a) `eval_recon` 不劣于臂 1 且 **Δ₁ 占比下降**（收益后移）、
  (b) 后步 ΔL1 绝对值上升（后步不再是 ±0.02 px 级）、(c) E-A5 的注意力熵随迭代下降。
- **判决**：若 (a)(b) 均不成立 ⇒ **ALIT 的"记忆增长"在 224/K=64/从零编码器设定下无收益**，
  明确写入报告，关闭该方向（这条否证本身对项目有价值：说明"后步真干活"不是键不足，而是任务同构）。
- **代码**：`model_v2_linattn.py` 新增 `GrowthBlock`（~60 行）或 `LatentRefiner`（~150 行，
  含编码器返回全序列的改动：现在 `encode()` 只回 `z_cls, z_s`，需回 `seq`）。

### E-B4 随机预算采样 / "一条流多档"（ALIT `sample_grad_iters` + 层级独立性）

- **假设（ALIT）**：一个模型可在所有预算档都可用（训练时随机采样迭代算梯度），
  不必为每个 K 训练一个模型；且解码器无 carry 时截断天然合法。
- **4 臂**：
  1. 现状（全步反传，固定 T=8）；
  2. **随机子集反传**（每步随机抽 2–4 个迭代算 loss，其余只前向——显存/时间省，对齐 ALIT）；
  3. 臂 2 + 推理时**截断到 t=3/5/7**（用同 ckpt）；
  4. **专训固定档** 基线：K'=16 / 32 / 48 各训一个（**只需 3 个短臂**，或复用历史 k-sweep 数据）。
- **验收**：截断档在同等平均 token 数下与专训档的差距 **<0.20 px**（L1_255）⇒ "一条流多档"成立；
  否则必须保留"每档微调"，并在报告里写明变长主张的代价。
- **⚠️ 已知风险（必读）**：`REPORT_slice1_224.md` 已证"删一步的代价摊到所有后续步"
  ⇒ 本实验**预期臂 3 会输**。它的价值是**量化**这个损失并判定"层级独立性"是否值得用
  E-B3 方案 b（无 carry 的独立档解码）来换。
- **代码**：`model_v2_linattn.py` 加 `--sample_grad_iters`（默认 0 = 现状全步）。

---

## 5. 阶段 C — 验收（E8 任务保真 / Phase 2）

> 阶段 C 才是**唯一验收口径**（`GOAL` §2）。A/B 阶段的一切都是中间探针。

### E-C1 任务保真下的自适应 K

- **做法**：把 E-A2 的 per-image 停步接到两个 TOI 上：
  1. **分类保真**：冻结编码器 + 线性探针（`z_s` 前缀 mean-pool）在 IN-100 val 上的 top-1；
     对比"自适应 TSC 选 token"vs"固定 K' 同预算"；
  2. **重建保真（E8 判别树）**：沿用 `PLAN_paper_experiments.md` 的 E8 口径。
- **对照口径**：**等平均 token 数**（不是等最大 token 数），这是 ALIT Fig.6 的正确读法。
- **判决**：等平均 token 数下自适应 ≥ 固定 K 且差距 > 噪声 ⇒ "按图自适应 K"在任务保真口径成立，
  可写进 GOAL §4.1 推翻"固定 K 即可（YAGNI）"；否则维持 GOAL §4.1 并结案。
- **备注**：ALIT 的 TSC=TOI 结论提示"TSC 与 TOI 对齐时最优"；
  我们的 TSC 是重建损失（自监督、免费），若它与分类 TOI 不对齐，需报告这一 gap。

### E-C2 Phase 2 NLP 探针（唯一验收，独立排期）

- 冻结 Phase 1 编码器 → MLP → Qwen，小批量训中文描述/隐患；**首次引入 caption/幻觉指标**
  （README §0.1 明确标注：当前一个都没跑过）。
- 与自适应 K 的耦合点：**在同等平均 token 数下**比较自适应 vs 固定 K 的 caption 质量；
  这是"密度 ↑ ⇒ 幻觉 ↓"的第一个直接测量。
- 本 PLAN 只登记接口与判据，不估成本（依赖 Phase 2 搭建进度）。

---

## 6. 代码改动清单（实施时按此顺序，全部**新增/默认关**，不动既有默认路径）

| # | 文件 | 改动 | 依赖 | 约行数 |
|---|---|---|---|---|
| 1 | `eval_stepwise_linattn.py` | `--dump_per_image`（每图每步 L1/PSNR/最优停步/类别/文件名）；`--data_dir_ood` | — | 40 |
| 2 | `data_linattn.py` | `build_ood_dataset(dir)`（无标签 ImageFolder + val transform） | 1 | 40 |
| 3 | `tools/alit_adaptive_k.py`（新） | TSC 曲线、等预算对照、oracle 上界、IID/OOD 停步差、Spearman、Δ_t 分解 | 1,2 | 150 |
| 4 | `metrics_linattn.py` | `zs_stats()`（两两 cos / 有效秩 / 逐 token 方差）；`spearman_ci()` | — | 60 |
| 5 | `model_v2_linattn.py` | `--loss_weights`（B1）；`--dh_mode/--dh_quantile`（B2）；`GrowthBlock` + `--refiner_*`（B3）；`--sample_grad_iters`（B4）；`encode()` 可选返回全序列 | — | 250 |
| 6 | `train_linattn.py` | 透传 5 的新参数；`--ood_eval_dir`（训练中 OOD 探针，可选） | 5 | 40 |
| 7 | `run_alit_a.sh` / `run_alit_b.sh` / `run_alit_c.sh`（新） | 三段串联（零训练 / B1+B2 / B3+B4），含时间预算、断点续跑、收尾评测与画图（模板：`run_stdattn_chain.sh`） | 1–6 | 120 |
| 8 | `README.md` | §3.1 增索引 + 结论回填（**每个实验完成后必须回填**） | — | — |

**硬约束**：
- `model_v2.py`（DINOv2 主线）**逐位不改**；
- `LinAttnConfig` 新字段全部带默认值 ⇒ 旧 ckpt `strict` 载入不受影响（新增 key 只在 GrowthBlock/Refiner，
  旧 ckpt 载入时 missing 需显式白名单，沿用 `eval_stepwise_linattn.py` 现有的 missing 白名单写法）；
- 每臂训练完立即跑 `eval_linattn.py` 四臂同进程 + `eval_stepwise_linattn.py`，结果 json 直接进 `doc/2026-10-16/artifacts/alit/`。

---

## 7. 成本与排期（单卡 4080S 32GB，按 10-15/10-16 实测吞吐线性外推）

| 阶段 | 项 | 步数/epoch | 墙钟/臂 | 臂数 | 小计 |
|---|---|---|---|---|---|
| A | 逐步评测（5,000 张） | — | 23 s | ~6（含 OOD 三级） | **~0.3 h** |
| A | 复杂度/冗余分析（CPU） | — | — | — | **~1 h** |
| B1 | 6 ep 热启动 BPTT bs64 | 11,406 / 6 | ~1.6 h | 4 | **6.4 h** |
| B2 | 同上 | 11,406 / 6 | ~1.6 h | 4 | **6.4 h** |
| B3 | 方案 a（含 GrowthBlock，显存/步速略升） | 11,406 / 6 | ~2.0 h | 2（臂1复跑+臂2） | **4.0 h** |
| B4 | 随机预算 + 3 个专训档（K'=16/32/48，可少 epoch） | 11,406 / 6（档位可 4 ep） | ~2.5 h | 4 | **10.0 h** |
| C1 | 线性探针 + E8 判别（复用 B 的 ckpt） | — | — | — | **~3 h** |
| | **合计** | | | | **≈30 GPU·h（≈1.5–2 天）** |

**排期建议**：
- **Day 1 上午**：E-A1/E-A2/E-A3（零训练）→ 评审 → 决定 B 阶段是否全跑；
- **Day 1 下午**：E-B1（4 臂并联串行跑，~6.4 h）→ 回填；
- **Day 2**：E-B2 → E-B3（依 E-B2 结果决定是否加输入侧停机）→ E-B4；
- **Day 3**：E-C1 + 文档定稿。

**若机器换成内网 DCU 8 卡**：B 阶段可并行 4 臂，墙钟压到 ~2 h；但**配对臂必须同机同后端**
（`srdiff_runtime.py` 已支持设备/后端解析，见 `doc/2026-10-14/REPORT_multi_portable.md`）。

---

## 8. 判决 / 证伪总表（先把"什么算输"写下来）

| ID | 成立 | 否证（明确写进报告并关闭方向） |
|---|---|---|
| E-A1 | 停步分布熵高、oracle 明显优于固定 step7 | 停步高度集中（≥90% 同一档）⇒ 图间差异不可利用 |
| E-A2 | 等平均 token 数下自适应显著优于固定步（>0.10 px） | 复现 batch-0 的 +0.13 dB ⇒ PSNR 口径结案，判决权交 E-C1 |
| E-A3 | OOD 所需步数单调更多（≥1 步） | 三级停步差 <0.5 步 ⇒ 熟悉度假设不成立 |
| E-A4 | ρ 显著且方向为正 | ρ 不显著 ⇒ 只能记为"代理量不足"，**不能**说 ALIT 的熵假设错 |
| E-A5 | z_s 低冗余（有效秩高、cos 低） | 有效秩 ≪ K 且 cos > 0.9 ⇒ 冗余问题仍在，E-B3 动机增强 |
| E-B1 | 末步 L1 改善 >0.10 px 且过冲不放大 | 四臂差 <0.10 px ⇒ 按 P3 关闭 |
| E-B2 | README §4.1 三条判据**同时**满足 | 后步仍不出增量 ⇒ "任务同构"是主因，键/损失都是次要 |
| E-B3 | Δ₁ 占比下降 + 后步 ΔL1 上升 + recon 不劣 | 三项均不成立 ⇒ ALIT 记忆增长在本设定下无收益 |
| E-B4 | 截断档与专训档差 <0.20 px | 差距大 ⇒ 变长必须"每档微调"，报告须写明代价 |
| E-C1 | 等平均 token 数下自适应 ≥ 固定 K（任务保真） | 劣于固定 K ⇒ **推翻 §0.1 的实现必要性**，GOAL §4.1 维持 |

---

## 9. 风险与诚实边界

1. **单种子、无方差**：全部结果 seed 0。任何 <0.10 px / 0.1 dB 的差异**不作结论**
   （`REPORT_stdattn_ablation.md` §9.2 已立此口径）。若某臂处于判决边界，必须补第二个种子。
2. **与 ALIT 不同构**：256×256 + VQGAN 离散底座 + 量化 1D codebook ≠ 224×224 + 连续 z_s。
   **只借机制与实验设计，不引用其数字做对比**。
3. **PSNR 不是验收**：本 PLAN 中 PSNR 仅作探针；最终判据是 E8/Phase 2。
4. **E-B2 的架构裁剪**：输入侧动态停机在当前结构装不下，属**诚实降级**，不是等价复现。
5. **E-B4 预期为负**：已有 `REPORT_slice1_224.md` 证据倾向截断会输；其价值是量化代价。
6. **OOD 数据依赖**：COCO/construction_site 需在目标机器就位（`tools/prep_data.py`）；
   若不可得，E-A3 降级为"IN-100 内部难/易子集"（按 E-A4 的复杂度代理分层），并标注降级。
7. **注意力可视化成本**：E-A5 的第 2/3 项需要 hook 或重写交叉注意力打分，是本 PLAN 里
   唯一"可能比预估更贵"的地方；可先交付第 1 项（冗余度）单独出结论。
8. **不改主线**：`model_v2.py`、`GOAL` 验收口径、README 判据口径均不动；
   本 PLAN 若与它们冲突，以它们为准。

---

## 10. 复现命令（模板，路径按实际机器替换）

```bash
# ── 自检（含旧 ckpt strict 载入回归）─────────────────────────────
LINATTN_CKPT=/root/autodl-tmp/srdiff_linattn/k64/ckpt_best.pt python model_v2_linattn.py

# ── 阶段 A：零训练（1 h 级）─────────────────────────────────────
# E-A1/E-A2/A6: 每图逐步 dump
python eval_stepwise_linattn.py \
  --ckpt /root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt \
  --num_images 0 --dump_per_image \
  --out doc/2026-10-16/artifacts/alit/stepwise_stdattn_rec.json
python tools/alit_adaptive_k.py \
  --stepwise doc/2026-10-16/artifacts/alit/stepwise_stdattn_rec.json \
  --out doc/2026-10-16/artifacts/alit/tsc_curves.json

# E-A3: IID vs OOD
python eval_stepwise_linattn.py --ckpt .../stdattn_k64_rec/ckpt_best.pt --num_images 0 \
  --out .../ood_in100.json
python eval_stepwise_linattn.py --ckpt .../stdattn_k64_rec/ckpt_best.pt \
  --data_dir_ood /root/autodl-tmp/datasets/coco2017/val2017 --num_images 0 \
  --out doc/2026-10-16/artifacts/alit/ood_coco.json

# ── 阶段 B：重训（模板；B1 为例）───────────────────────────────
bash run_alit_b.sh --arm b1_decay      # ALIT 原版递减权重
bash run_alit_b.sh --arm b1_lastonly   # 末步 only
# 每臂跑完立即四臂同进程评测 + 逐步
python eval_linattn.py --ckpt <new_ckpt> .../stdattn_k64_rec/ckpt_best.pt \
  --num_images 0 --batch_size 64 --out doc/2026-10-16/artifacts/alit/eval_b1.json
```

---

## 11. 本轮**不做**的事（防 YAGNI 复发 / 防自变量污染）

- ❌ 不引入量化 codebook / 1D 离散化（ALIT 的 4096×12）——那是 Phase 2 之后的话题；
- ❌ 不改 K=64 主线、不重跑 k-sweep、不动 `model_v2.py`；
- ❌ 不做解码器 ×10 参数量臂（另有 09-26/10-14 多分辨率线在跑）；
- ❌ 不用 PSNR-RD 当验收（README §3 判据）；
- ❌ 不同时改编码器注意力形式 + 解码读出方式（10-16 报告已确立单变量协议）；
- ❌ 不在 E-A 阶段之前写 B 阶段代码（先让零训练数据决定要不要训）。
