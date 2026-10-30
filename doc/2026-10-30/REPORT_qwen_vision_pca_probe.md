# Qwen3.8-27B 视觉塔 token 的池化 PCA 谱 + 「同参数量 AssocVTC 交叉注意力」单步还原（1σ/2σ/3σ/全部）

日期：2026-10-30 ｜ 机器：`ssh -p 31074 root@connect.weste.seetacloud.com`（AutoDL 单卡 RTX 4080 SUPER 32G，128 核，503 GB RAM）
｜ 状态：**完成**

**一句话**：Qwen3.8-27B 视觉塔输出的 49×5120 token 在 IN-100 训练集全池化后**方差极度集中（PC1 独占 51.7%、PR=3.70）**，累计方差 68.27/95.45/99.73% 只需 **r = 20 / 234 / 754** 维；用「同参数量 AssocVTC 解码器」单步还原探针实测：**3σ(754 维) 几乎无损（−0.021 dB）、2σ(234 维) 微损（−0.275 dB）、1σ(20 维) 实质损失（−1.239 dB）**，且 3σ 以下「零训练截断」就够、不必重训，只有 1σ 需要重训（+1.46 dB）但仍补不回损失 ⇒ **"能降维"成立，但"按方差占比选维"是错的选择规则**：一半方差装在一个离群幅度方向上。

---

## 0. 用户口径与实现映射

用户原话（2026-10-30）：*「下载 `Qwen3.8-27B` 这个模型的仅图像编码器，然后输入 imagenet-100 的训练集获得其图像 embedding，然后同张图片按 token 顺序拼接，然后整个训练集在一起做 PCA，然后分别取 1-sigma、2-sigma、3-sigma、全部向量利用一个同参数量的 AssocVTC 的交叉注意力还原探针进行还原看看效果（单步）。」*

| 用户要求 | 本报告实现 |
| --- | --- |
| 下载 `Qwen3.8-27B` 的**仅图像编码器** | HF `Qwen/Qwen3.8-27B`（`Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`）的视觉塔 `model.visual.*`：**333 张量 / 460.73M 参数**（BF16），全部在 shard1。用 **HTTP Range** 只取该 shard 里**连续的视觉段** `[3,045,270,360, 3,966,730,552)` = **921.5 MB**（整 shard 3.97 GB，文本张量未下载），落 `weights/vision_only.safetensors`，sha256 `ef7ccc6d74936c63ee52d99a437fd9870708da7b383edab27ab0419fc669b9a8` |
| 输入 ImageNet-100 **训练集**得图像 embedding | 训练集 = `/root/autodl-pub/ImageNet100/imagenet100.zip`（**126,689 张 / 100 synset**；注意与旧报告的 121,689 张是**不同份 IN-100 拷贝**）；图像 PIL bicubic 拉伸到 **224×224**，按官方 Qwen 预处理（`/255`、mean=std=0.5、patch 16、temporal 2、merge 2）→ 视觉塔 → **每图 49 token（7×7 合并网格）× 5120 维**（`out_hidden_size=5120`，merger 后）；GPU 自复刻 patchify 与 HF `Qwen2VLImageProcessor` 的 **max\|Δ\|=1.19e-7** |
| 同张图片按 token 顺序拼接 | 存 `(N, 49, 5120)`，token 顺序 = 合并网格 row-major（= 视觉塔输出顺序，供解码器逐 token 读） |
| 整个训练集一起做 PCA | **池化**（pooled）：全部训练图的 49 个位置合并为 **6,207,761 个 5120 维样本**、**不分位置**、全局中心化后做 PCA（与 2026-10-30 `REPORT_zs_pca_240m.md` 的**逐位置** PCA 口径不同，见 §7 边界） |
| 1σ / 2σ / 3σ / 全部 | **累计解释方差** 68.27% / 95.45% / 99.73% / 100% 对应的主成分个数 r（§3）；另记 90/95/99/99.9% 与备择口径（Kaiser 等） |
| 「同参数量 AssocVTC 的交叉注意力还原探针」（单步） | AssocVTC 240M 档解码器逐位同构：`OutputQueryDecoder(dim=1440, depth=7, heads=16, N=196, K=49, steps=[49])`——**单步**（读窗口 `A[0:50]`=z_cls+全部 49 memory token）+ `mem_proj(5120→1440)` + 学习 `z_cls` + `PixelHead(1440→768)` → 224² 重建。**探针 244.69M**（解码器主体 **237.31M**，对 AssocVTC 240M 档的 237.11M 为 **+0.09%**；+7.37M 输入适配） |
| 还原效果（单步） | 像素 PSNR / SSIM / L1(0–255)，**5,000 张 IN-100 val**（训练未见），并对比「训练集平均图」平凡基线（**11.821 dB / SSIM 0.315**） |

## 1. 数据与 token 提取

**编码器**：Qwen3.8-27B 视觉塔 = 27 层 ViT（hidden 1152、heads 16、intermediate 4304、patch 16、spatial merge 2、`out_hidden_size=5120`、learned pos-embed 2304 可插值、无 deepstack）。权重只取视觉张量；`load_state_dict` **strict 无 missing/unexpected**。输出取 `BaseModelOutputWithPooling.pooler_output`（merger 后 `(B·49, 5120)`；`last_hidden_state` 是 merger 前 `(B·196, 1152)`，不能用——这是本轮修掉的一个真实 bug）。

**数据**：train 126,689（zip 内 100 个 synset 目录，PIL bicubic→224）；val 5,000 = 从 `ILSVRC2012_img_val.tar` 按 devkit GT 筛出这 100 类（每类 50）。devkit GT 是 **1-based**（min=1/max=1000、无 0），`meta.mat` 的 synsets 是 1860 项 struct（分类类目占前 1000，synset 字符串在 `WNID` 字段）——两处都做了位置序交叉校验（50,000/50,000 一致）。

**提取吞吐**：GPU-patchify 链路 bs256/workers20/prefetch8，稳态 **182 img/s**（首轮 v1，11.6 min）；B2 重提（见下）**139 img/s**（15.2 min，含 refs 估计）。CPU 瓶颈的 `torchvision`/processor patchify（≈19 ms/img）被「worker 只 decode+resize uint8、主进程 GPU 复刻 patchify」替换（等价性 1.19e-7）。

**token 存储精度（重要，做过一次返工）**：
- v1 = per-token absmax int8 ⇒ 实测 6.2M token 上 `rel_rmse=0.1418`、token PSNR **16.97 dB**（不可用）。原因：Qwen 视觉 token 有巨大的**共享偏置/离群维**——per-token absmax 中位 68、p99 173，而逐元素 RMS 仅 1.28 ⇒ per-token 尺度把内容精度吃掉了。
- v2 = **B2**：位置中心化 + 逐通道归一 + 逐 token int8，存 `quant_ref.npz`（`mu_p (49,5120)`、`sd (5120,)`）⇒ `rel_rmse=0.004914`、token PSNR **46.17 dB**、max\|Δ\|=1.46；反量化 `x=(q·sc)·sd+mu_p`。
- `mu_p/sd` 由 **val 5,000 张的 fp16 token** 估计（只影响存储刻度，不进入 PCA 拟合/训练目标/标签；qwen-runner 另用 train fp16 子集做 refs 稳健性对照，见 §7）。val token 本身始终是 fp16 原值，不参与量化。

**token 规模**：train `(126,689, 49, 5120) int8` = 31,783,736,448 B + `scale (126,689,49) fp16`；val `(5,000,49,5120) fp16` = 2.51 GB；另存 5,000 张 train 图的 fp16 子集（保真度对照用）。

## 2. PCA 口径

先算全局均值 `μ (5120,)` 与协方差（6.2M×5120，`f32` 块内 XᵀX + `f64` 跨块累计，块 4096 token 防显存 OOM），`eigh` 在 GPU 上以 **float64** 求（5120 维 1.2 s），特征值降序。
- **主口径**：全局中心化 + 池化（所有位置混在一起）。
- **次要对口**：位置中心化池化（减每个位置的均值 μ_p 后再池化），用于分离「位置/偏置结构」与「图像内容」；以及逐位置 PCA（49 个位置各做一次，与旧报告口径对齐）。
- **σ 定义（权威）**：1σ=68.27%、2σ=95.45%、3σ=99.73% 的累计解释方差对应 r；「全部」= 5120 维不做截断。
- 备择口径（只作透明性参考）：λ_i ≥ k·mean(λ)（k=1/2/3）、逐维标准化后 Kaiser（eigenvalue ≥ 1/2/3）。

## 3. 池化 PCA 谱与 σ 维度

**规模**：6,207,761 个 token（126,689 图 × 49）× 5120 维；全局中心化后 `eigh`（GPU float64）。

**σ 维度（主口径，池化/全局中心化）**：

| 口径 | 目标累计方差 | **r** | 实际累计方差 | λ_r |
| --- | --- | --- | --- | --- |
| **1σ** | 0.6827 | **20** | 0.68362 | 13.703 |
| **2σ** | 0.9545 | **234** | 0.954684 | 0.93117 |
| **3σ** | 0.9973 | **754** | 0.997303 | 0.017764 |
| 全部 | 1.0 | 5120 | 1.0 | — |
| 参考 90% / 95% / 99% / 99.9% | — | 134 / 222 / 425 / 1201 | — | — |

**谱形状（极端各向异性）**：`top1 方差占比 = 51.7%`、`top4 = 56.5%`、`top16 = 66.2%`、**`top49 = 78.3%`**、`top64 = 81.5%`、`top128 = 89.5%`、`top256 = 96.2%`、`top448 = 99.15%`、`top1024 = 99.85%`；参与率 **PR = 3.70**、entropy rank = 26.13（≈ top128 的量级）。

**PC1 是什么（本轮最有信息量的诊断）**：PC1 载荷**极稀疏**——单维最大 |loading| **0.936**、top-10 维占 **92.4%**、而 5120 维均匀载荷只有 1.95e-4；PC1 分数与 **token L2 范数**的 Pearson 相关 = **−0.999**（Spearman −0.998）。而 token L2 范数本身按图变化极大（mean 78.9、**std 40.0**）。⇒ **池化总方差的一半以上不是"空间内容"，而是一个离群坐标上的"图像级幅度"变化**；`1σ = 20 维` 里装的主要是它。这一点直接决定了 §5 的解读：**PCA 方差序 ≠ 图像信息量序**。
补充：PC1 的「位置间方差占比」只有 6.9% ⇒ 它是**跨图共享方向、幅度随图变**，不是位置结构。

**位置结构不是主导**：把位置均值之间方差与位置内残差分开——**位置常量只占总方差 4.53%**（位置内残差 95.47%）。据此做的**位置中心化池化 PCA** 与主口径几乎一样：1σ/2σ/3σ = **21 / 239 / 776**（top1 51.9%、PR 3.68）。**逐位置 PCA**（49 个位置各做一次，每位置 126,689 样本）：1σ 中位 **17**（13–23）、2σ 中位 **224**（171–237）、3σ 中位 **732**（634–755）——**各位置之间几乎没有差别**，也说明"位置"不是这组 token 的主要结构。

**截断的 token 空间误差**（逐元素 RMS，token 值 RMS≈1.28）：1σ **0.400**、2σ **0.151**、3σ **0.0369**（相对 1.28 ⇒ 31% / 12% / 2.9%）。即：**按方差口径"恢复 68% 方差"的 20 维，在 token 空间仍剩 31% 的逐元素误差**——方差占比高不代表重建误差小（因为被方差放大的方向集中在离群维）。

**逐位置差异与异常**：49 个位置里 **位置 0（7×7 网格左上角）明显异常**——它的 r@2σ 只有 **171**（其余中位 224）、方差占比仅 **0.99%**；最宽的位置是 4/5/6（r@2σ = 237/235/235）。位置 0 是合并网格的角点，与 CLIP/ViT 类模型「角点/边界 token 信息量低」的已知现象一致。

**PC1–PC4 与位置均值子空间**：PC1–4 与「位置均值 top-4 子空间」的最大 |cos| = **0.920 / 0.564 / 0.292 / 0.773**，位置均值 top-4 方向有 **54.7%** 被 PC1–4 捕获——即 PC1/PC2 里确实混着「位置均值」分量，但**位置常量整体只占 4.53%**（见上），所以不是主导。

**特征健全性（证明 token 提取正确）**：用 mean-pooled（49 token 平均、5120 维）特征做**闭式 ridge 线性探针**（= 线性分类器）。为覆盖全部 100 类，用随机 20,000 张 train 图训练：**val 5,000 上 top-1 = 91.14%、top-5 = 98.82%**（随机猜 1%）。⇒ token 的顺序/归一化/分辨率/图片↔token 对齐都是对的（若 token 被旋转/打乱或图片错位，top-1 会接近 1%）。另一口径「train_index 顺序前 50,000 张」只覆盖 40 类，已见类归一 top-1 = 89.4%、top-5 = 99.0%（full-val top-1 = 35.8% 受 40 类上限约束）。

**备择 σ 口径（透明性）**：未标准化 `λ_i ≥ k·mean(λ)` ⇒ k=1/2/3 → **314 / 222 / 171**；逐维标准化后 Kaiser（eigenvalue ≥1/2/3）⇒ **407 / 327 / 271**。与主口径（20/234/754）量级差异很大，说明「σ」这个词必须显式定义——本报告一律用累计方差口径。

**量化噪声对谱的影响（隔离检验）**：同一批 245,000 token，fp16 vs B2-int8 重算阈值 r 的差 = 1σ **0/0**、2σ **+0/+1**、3σ **+5/+6**（左=同批图 fp16 vs int8，右=全量 vs 子集）；量化噪声方差占比 **8.5e-5** ⇒ 存储精度不影响 §3 结论。


## 4. 单步交叉注意力还原探针

**架构（= AssocVTC 240M 档的真实解码器，只把 memory 入口加宽）**：`model_v2_linattn.RecurrentOutputQueryDecoder(enc_dim=5120, dim=1440, depth=7, heads=16, mlp_ratio=4.0, patch=16, num_patches=196, num_specials=49, pixel_hidden=1024, steps=[49])`。
- **单步**：`steps=[49]` ⇒ 读窗口 `[0,49]`（z_cls + 全部 49 个 memory token），一次 Transformer 堆叠（自注意力 + 交叉注意力 + FFN ×7 层）→ `norm_out` → `pixel_head`（1024 隐层）→ 196 个 16×16 patch → 224² 图像。
- **参数量 242.73M**：其中 `stack` 232.44M + `query_base/query_pos/mem_pos/mem_norm/norm_out/pixel_head` ≈ 4.67M = **解码器主体 237.11M（与 AssocVTC 240M 档逐 key 同构）**；唯一变宽的是 `mem_proj`：`1248→1440`（1.80M）换成 `5120→1440`（7.37M），即 **+5.58M（+2.35%）**——这是 Qwen token 维度决定的、不可避免的输入适配。
- **z_cls**：Qwen 侧没有对应 token，用 49 个 token 的均值代替（无额外参数）。
- **热启动（关键）**：从 AssocVTC 240M 档 B 相 `stdattn_m240_t16/ckpt_best.pt` 的**解码器权重**初始化（`decoder.*` 全键加载；`mem_pos` 从 K=32 的 33 行线性插值到 50 行；`mem_proj` 因输入维度改变保持随机初始化）。首次会话中「从零训练 2 epoch」的对照臂（v1）只有 **14.79 dB**（1 epoch 子集），明显欠训练，故改用热启动协议——这也是「探针」而非「从零训一个生成器」的正常做法。
- **输入截断**：`x_r = μ + U_r U_rᵀ(x−μ)`（r=5120 即恒等，="全部"臂），r ∈ {20, 234, 754, 5120}。4 臂**同架构/同 init/同 seed/同数据顺序/同 epoch 预算**，唯一自变量 = r。
- **训练**：AdamW（betas 0.9/0.95、wd 0.05 且 norm/bias 不衰减）、warmup 100 步 + cosine、bf16 autocast、grad clip 1.0、**bs 128、全量 126,689 张**、lr 1e-4；每 epoch 在固定 512 张 val 子集上选 best，最终 5,000 val 全量评。
- **像素空间**：loss 在 **DINOv2 归一化空间**（与 ckpt 训练一致），评测时反归一化到 0–255 再算 PSNR/SSIM/L1。


## 5. 主结果

**协议**：全部 5,000 张 IN-100 val；PSNR/SSIM 在 [0,1] 像素空间（等价 0–255 口径），L1_255 = 像素 L1 ×255；探针 242.71M，AssocVTC 240M ckpt 热启动，4 臂同预算（E=2、全量 126,689、bs128、lr 2e-4，单臂训练 21.5 min）。

**A. 四个训练臂（唯一自变量 r）**

| 臂 | r | 累计方差 | **PSNR** | SSIM | L1_255 | ΔPSNR vs all |
| --- | --- | --- | --- | --- | --- | --- |
| **1σ** | 20 | 0.68362 | **15.2595** | 0.3673 | 33.818 | **−1.2393** |
| **2σ** | 234 | 0.954684 | **16.2240** | 0.3903 | 29.172 | **−0.2748** |
| **3σ** | 754 | 0.997303 | **16.4781** | 0.3956 | 28.146 | **−0.0207** |
| **全部** | 5120 | 1.0 | **16.4988** | 0.3961 | 28.038 | 0 |

**B. 零训练截断（r=5120 探针直接吃截断输入，不重训）与「重训收益」**

| r | 累计方差 | 零训练 PSNR | 训练臂 PSNR | **trained − zero-shot** | 解读 |
| --- | --- | --- | --- | --- | --- |
| 20 | 0.68362 | 13.7983 | 15.2595 | **+1.461 dB** | 必须重训，且仍比 all 低 1.24 dB |
| 234 | 0.954684 | 16.2058 | 16.2240 | **+0.018 dB** | 重训无收益（噪声内） |
| 754 | 0.997303 | 16.4861 | 16.4781 | **−0.008 dB** | 重训无收益 |
| 5120 | 1.0 | 16.4988 | 16.4988 | 0 | 同点 |

**C. 平凡基线 / 从零参照**

| 参照 | PSNR | SSIM | L1_255 |
| --- | --- | --- | --- |
| 训练集平均图（meanbase） | **11.821** | 0.3151 | 58.50 |
| v1 从零 2ep（244.69M，[-1,1] 像素空间）all | 15.728 | 0.361 | 31.09 |
| v1 从零 零训练截断 r754/r234/r20 | 15.718 / 15.494 / 13.671 | — | — |
| **v2 热启动 all（本文主口径）** | **16.499** | 0.396 | 28.04 |

⇒ 热启动相对从零 **+0.77 dB**（16.499 vs 15.728），两种协议的**相对趋势完全一致**（r=754 几乎无损、r=20 掉约 2.7 dB），说明结论不依赖探针初始化协议。

**D. 与 AssocVTC 240M「训练式降维」的对照（口径不同，只做量级比较）**

| 方案 | token | 压缩 | ΔPSNR |
| --- | --- | --- | --- |
| AssocVTC 训练式瓶颈 448 维（含评测图 oracle PCA 基） | 32×1248 | 1248→448 | −0.242 dB |
| AssocVTC 训练式瓶颈 224 维 | 32×1248 | 1248→224 | −1.904 dB |
| 本文数据侧 PCA 截断 234 维（训练集基） | 49×5120 | 5120→234 | −0.275 dB |
| 本文数据侧 PCA 截断 20 维 | 49×5120 | 5120→20 | −1.239 dB |

**E. 图**：[`fig_qwen_probe_v2_psnr_vs_r.png`](artifacts/qwen_probe/fig_qwen_probe_v2_psnr_vs_r.png)（4 训练臂 + 4 零训练截断 + meanbase 线）、[`fig_qwen_probe_v2_grid.png`](artifacts/qwen_probe/fig_qwen_probe_v2_grid.png)（8 张 val：GT / all / 3σ / 2σ / 1σ）。


## 6. 解读

**① 数据侧"可降维"成立，但「方差」不是信息的标尺（本轮主结论）。**
- 截断到 **r = 754（5120 的 14.7%）**：保留 **99.73%** 方差，重训后 **16.478 dB vs 全量 16.499 dB ⇒ −0.021 dB**（几乎无损），零训练截断也只 −0.013 dB。
- 截断到 **r = 234（4.6%）**：保留 95.47% 方差，掉 **−0.275 dB**（零训练截断 −0.293 dB）。
- 截断到 **r = 20（0.39%）**：仍保留 **68.3%** 方差，却掉 **−1.239 dB**（15.260 dB；零训练更差，13.798 dB）。
⇒ 46% 的维度（234<r<754）只值 0.28 dB，而 68% 的方差可以装进 20 维却几乎没有图像信息。这与 §3 的机制诊断一致：**池化 PCA 的 51.7% 方差集中在一个"离群幅度方向"上（PC1 与 token L2 范数相关 −0.999、载荷 top-10 维占 92.4%、token L2 范数 78.9±40.0）**，它随图像明暗/尺度变化，但不是空间内容。**"能降维"的直觉对，"按方差占比砍维度"的选择规则错。**

**② 与 AssocVTC 240M 的"训练式降维"对照（口径不同，只做量级比较）。**
上一轮（`REPORT_zs_pca_240m.md`）在 1248 维 token 上做**训练式**瓶颈：448 维 −0.242 dB、224 维 −1.904 dB。本轮在 5120 维 token 上做**零训练数据侧 PCA 截断**：234 维 −0.293 dB、20 维 −2.70 dB。**相对损失的形状相似**（压缩 ×2~×5 时轻微掉、压缩 ×25 时明显掉），说明这批 Qwen 视觉 token 在"信息可压缩性"上与自训练编码器的 z_s 同量级；但注意 token 数（49 vs 32）、维度（5120 vs 1248）、探针训练预算（2 epoch vs 45+6 epoch）、PCA 口径（池化 vs 逐位置、训练集拟合）全都不同，**不可宣称"谁更好"**。

**③ 热启动（AssocVTC 240M 解码器）增益有限：+0.77 dB。**
v2（真实解码器 + 240M ckpt 热启动 + DINOv2 像素空间）2 epoch 全量 val = **16.499 dB**；v1（同参数量但从零、[-1,1] 像素空间）2 epoch = **15.728 dB**。原因：`mem_proj` 7.37M 是新的随机层（ckpt 的 1248→1440 不能直接用），把 5120 维 Qwen token 映射进解码器可读的空间需要时间；我们另试了两种 **PCA 投影初始化**（把 5120 主成分投到 ckpt 的 1248 维，raw / 白化）——**都比随机更差**（10.44 / 10.49 vs 10.77 dB，32 张 smoke），说明"按 PCA 方向对齐 pretrained mem_proj"不成立，方向需要学。⇒ **热启动只能省一点点，探针的主要成本在输入适配层。**

**④ 截断后要不要重训：3σ 不需要，1σ 需要且补不回来。** `trained − zero-shot` = r754 **−0.008 dB**、r234 **+0.018 dB**、r20 **+1.461 dB**。⇒ 在 ≥95% 方差（r≥234）的截断上，「用全维探针直接吃截断输入」与「专门重训」等价（差异在噪声内）；只有 1σ 这种把内容砍到极限（980 值/图）的情况下，重训才值 +1.46 dB，且仍比全量低 1.24 dB——**信息损失不可用适配弥补**。

**⑤ 绝对 PSNR 低是预算造成的，只能读相对趋势。** 本探针只训 **2 epoch / 1,980 步**（4 臂同预算），而 AssocVTC A 相是 **45 epoch**；meanbase 11.82 dB、from-scratch 2ep 15.73 dB、warm-start 2ep 16.50 dB 都说明还远没到"这个信息量能还原到什么程度"的上限。**不要**把 16.5 dB 与 AssocVTC 的 27.9/32.3 dB 并列比较。

**⑥ 另一个冗余轴：token 数（本轮未做实验，但可从数据推）。** 每图 49 token × 5120 维 = 250,880 值；即使 r=234，每图仍 49×234 = 11,466 值。配合 2026-10-30 报告里"32 token 同图能量 rank@95≈20/32、位置均值近似正交"的结论，**下一步更值得做的是"位置/ token 数剪枝 + 每 token 宽度截断"的联合压缩**，而不是继续往 5120 维里砍。数据侧线索：位置 0（7×7 左上角）方差占比仅 0.99%、r@2σ 只有 171（其余中位 224），是个天然的"低信息位置"候选。


## 7. 诚实边界

1. **单模型单种子**：Qwen3.8-27B 一个版本、探针 seed=0 一跑；<0.1 dB 不作结论。
2. **PCA 只在训练集上拟合**（126,689 张 / 6.2M token），评测集未参与基与 retention ⇒ 属**可泛化口径**，不是上一轮那种「含评测图的 transductive oracle」。也正因如此，1σ/2σ/3σ 的 r 是**训练集谱**给出的，不是最优 r。
3. **σ 的定义是可争议的选择**：采用「累计解释方差 68.27/95.45/99.73%」，与「高斯 1σ/2σ/3σ」的字面类比；报告同时给出 90/95/99/99.9% 与 Kaiser/λ≥k·mean(λ) 备择口径，便于换口径复核。
4. **token 存储精度**：train token 是 **int8 B2（46.17 dB、rel_rmse 0.49%）**，val 评测输入是 **fp16 原值**，训练/评测输入精度不完全对称；B2 的 `mu_p/sd` 由 val 估计（只影响存储刻度；refs 稳健性对照见 `tokens/verify_b2.json`）。
5. **池化 PCA ≠ 逐位置 PCA**：本轮把 49 个位置混在一起，主成分里有相当一部分在描述「位置/共享偏置结构」（Qwen token 跨图同位置余弦高达 0.87），因此 1σ/2σ/3σ 的 r 不能直接与旧报告的逐位置 1248 维数字相比。位置中心化口径作为次要对照给出。
6. **像素指标 ≠ 任务保真**：仓库既有结论已说明 PSNR/SSIM 与下游任务保真不同向；本报告只给像素重建。
7. **探针是「热启动 + 2 epoch」的短训探针**：4 臂同预算（2 epoch / 1,980 步，单臂 21.5 min），不是超参搜索结果；绝对 PSNR（16.5 dB）远低于 AssocVTC 在自身 token 上的 27.9 dB（A 相 45 epoch），**只能读相对趋势**。「同参数量」指解码器主体与 AssocVTC 240M 档逐 key 同构（237.11M），额外 +5.58M 是 `mem_proj` 由 1248→1440 变宽为 5120→1440 的输入适配（不可避免）。
7b. **热启动增益有限（+0.77 dB over 从零）**，且 `mem_proj` 的两种 **PCA 投影初始化实测更差**（raw 10.44 / 白化 10.49 vs 随机 10.77 dB，32 张 smoke）⇒ 「按 PCA 方向对齐 pretrained mem_proj」不成立，方向必须学；这一条可作为后续「换更强的输入适配器 / 冻结解码器只训适配层」的动机。
7c. **像素目标空间换了**：v1（从零）用 [-1,1] 空间，v2（热启动）用 ckpt 训练时的 DINOv2 归一化空间以匹配 pretrained pixel_head；两协议都在 §5 报告，趋势一致（r754 无损、r20 掉约 2.7 dB）。
8. **数据量/带宽**：IN-100 本机 zip 为 126,689 张（旧报告 121,689 张是另一份拷贝）；Qwen 输入固定 224²（未用原生动态分辨率），因此丢掉了高分辨率细节；224² 下每图仅 49 token。
9. **单步限制**：探针单步、无迭代精修；与 AssocVTC 的多步 BPTT 读出不可直接比；与 AssocVTC 的 32×1248 = 39,936 值/图相比，本实验是 49×5120 = 250,880 值/图（6.3×），信息预算不同。

## 8. 复现命令与产物清单

全部在远端 `/root/autodl-tmp/qwen_vtc/`（AutoDL 单卡 4080 SUPER）。

**权重（仅图像编码器）**
```bash
mkdir -p weights && cd weights
for f in config.json preprocessor_config.json model.safetensors.index.json; do
  curl -sL -o $f https://hf-mirror.com/Qwen/Qwen3.8-27B/resolve/main/$f; done
# shard1 里视觉张量连续，只取该段（921.5MB / 整 shard 3.97GB）
python /root/fetch_vision_range.py   # HTTP Range [3045270360,3966730552) → vision_only.safetensors
# 或整 shard: curl -L -o model-00001-of-00018.safetensors https://hf-mirror.com/.../model-00001-of-00018.safetensors
```

**数据 / token 提取**（`tools/qwen_extract_vision_tokens.py` + `tools/qwen_reextract_b2.py`）
```bash
cd /root/autodl-tmp/qwen_vtc
python code/qwen_extract_vision_tokens.py --mode smoke
python code/qwen_extract_vision_tokens.py --mode prep_val      # ILSVRC2012 val → 5,000 张（每类 50）
python code/qwen_extract_vision_tokens.py --mode extract --split val
# B2 量化重提（位置中心化 + 逐通道归一 + 逐 token int8；~15 min @139 img/s）
QWEN_ZIP=/root/img100.zip python code/qwen_reextract_b2.py --bs 256 --workers 20 --prefetch 8
```

**PCA**（`tools/qwen_pca_tokens.py --per_position` → GPU f64 eigh，~20 min）
```bash
python code/qwen_pca_tokens.py --out_dir pca --per_position
# → mean.npy / components.npy(5120,5120) / eigvals.npy / cumvar.npy / pca_stats.json /
#   per_position_stats.json / pca_stats_poscentered.json / quant_fidelity.json / pca_mechanism_diag.json
```

**探针**（`tools/qwen_probe.py`）
```bash
QWEN_ZIP=/root/img100.zip python code/qwen_probe.py --mode meanbase --out probe/meanbase.json
for R in 5120 754 234 20; do
  QWEN_ZIP=/root/img100.zip python code/qwen_probe.py --mode train --r $R --epochs 2 --bs 128 --workers 16      --pca_dir pca --out probe/r$R
done
python code/qwen_probe.py --mode eval --ckpt probe/r5120/best.pt --r 5120 --out probe/eval_r5120.json
python code/qwen_probe.py --mode eval --ckpt probe/r5120/best.pt --r 754  --out probe/eval_zeroshot_r754.json   # 零训练截断
```

**产物清单**

| 类别 | 远端路径 | 本地归档 |
| --- | --- | --- |
| 视觉塔权重 | `weights/vision_only.safetensors`（921.5 MB，sha256 `ef7ccc6d…`）、`vision_meta.json` | — |
| token（供后续实验复用） | `tokens/train_tokens_int8.npy`(31.78 GB) + `train_scale.npy` + **`quant_ref.npz`** + `train_index.json`；`tokens/val_tokens_fp16.npy`(2.51 GB) + `val_images_224.npy` + `val_labels.npy` + `val_paths.json`；`train_fp16_subset.npy`；`meta.json` / `verify_b2.json` / `quant_check.json` | 小 json 见 artifacts |
| PCA 基与谱 | `pca/mean.npy`、`pca/components.npy`(5120×5120 f32)、`eigvals.npy`、`cumvar.npy`、`pca_stats.json`、`arms.json`、`pca_stats_poscentered.json`、`per_position_stats.json`、`quant_fidelity.json`、`pca_mechanism_diag.json`、`feature_sanity.json`、3 图 png+pdf | `doc/2026-10-30/artifacts/qwen_probe/` |
| 探针 ckpt / 评测 | `probe/r{20,234,754,5120}/best.pt`、`eval_r*.json`、`eval_zeroshot_r*.json`、`summary.json`、`fig_qwen_probe_psnr_vs_r.*`、8 图网格 | 同上 |
| 脚本 | `tools/qwen_extract_vision_tokens.py`、`tools/qwen_reextract_b2.py`、`tools/qwen_pca_tokens.py`、`tools/qwen_pca_figs.py`、`tools/qwen_feature_sanity.py`、`tools/qwen_probe.py`、`code/re_extract_b2.py` | 仓库 `tools/` |

⚠️ token 反量化（后续实验唯一需要记住的一条）：`x = (q.float()*sc[...,None])*sd[None,None,:] + mu_p[None]`，其中 `q`=`train_tokens_int8.npy`、`sc`=`train_scale.npy`、`(mu_p, sd)`=`quant_ref.npz`。
