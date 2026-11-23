# 备忘：要不要走"纯率失真压缩"定位？该跟谁比？

- 时间：2026-11-23
- 触发：用户问"纯率失真压缩该跟什么论文对比，BLIP-2 吗？"
- 结论：**BLIP-2 完全不对标；而且纯 R-D 这条路比 rFID 那条更难——不是更容易。**
  下面是被实测数据否掉的证据。

---

## 1. BLIP-2 为什么不能对标

BLIP-2（Li et al., 2023）是**图文预训练**方法：冻结视觉编码器 + Q-Former + LLM。
它的 32 个 query token 是**给语言模型用的语义接口**：没有像素解码器、不重建图像、
没有 bpp、不报 PSNR/SSIM/LPIPS。它和"图像压缩"不是同一件事。
（同理 CLIP / DINO / ViT 的 "token" 都是语义 token，都不能当压缩 baseline。）

拿它对比会被审稿人一句话指出范畴错误。

## 2. 实测锚点：同一套 IN-100 val 5000 张、256²，JPEG / WebP 的 bpp-PSNR

脚本 `rd_codec_baseline.py`（纯 PIL，Resize(292,BICUBIC)+CenterCrop(256)，bpp = bytes×8/65536）

| codec | bpp ≈ 0.234 | bpp ≈ 0.375 | bpp ≈ 0.5625 |
|---|---|---|---|
| JPEG | 23.13 | **24.35** | **27.04** |
| WebP | 27.16 | **27.50** | **29.16** |
| **我们（k256×12 L1 ep14）** | — | **22.67** | **22.67** |

我们的 rate 换算（256² 共 65536 像素）：

- 3072 values × 8 bit ÷ 65536 = **0.375 bpp** ⇒ 输 JPEG **1.7 dB**、输 WebP **4.8 dB**
- 3072 values × 12 bit ÷ 65536 = **0.5625 bpp** ⇒ 输 JPEG **4.4 dB**、输 WebP **6.5 dB**
- （ALIT 的口径：256 tok × 12 dim × 5 bit = 15360 bit = **0.234 bpp**）

**⇒ 纯 R-D 定位不成立。** 连 WebP 都打不过 3–6 dB，学习式编解码器（Ballé/ELIC/MLIC++）
还会再高 1–3 dB。在压缩 track 上无法立住。

### 为什么输：不是训练不足，是**比特分配方式**

我们是 **256 个 patch 各固定 12 维**（均匀分配速率），而 JPEG/WebP 是**空间自适应**的
（DCT 块 + 量化表 + 熵编码，把比特花在刀刃上；平坦区域几乎不花）。
均匀分配 vs 率分配，在自然图像上差 3–6 dB 是完全预期的。
这不是调参能补的，是码流结构问题。

## 3. 正确的对标集合（按圈层）

### A. tokenizer 圈（我们真正所在的圈，主表）
- **ALIT** — Adaptive Length Image Tokenization via Recurrent Allocation (Duggal et al., ICLR'25, arXiv:2411.02393) ← 当前对标
- **TiTok** — An Image is Worth 32 Tokens (Yu et al., NeurIPS'24, arXiv:2406.07550)
- **VQGAN** (Esser et al., CVPR'21, arXiv:2012.09841) / ViT-VQGAN
- **MaskGIT** tokenizer (arXiv:2202.04200)、**RQ-VAE** (arXiv:2203.01941)
- **SD-VAE**（连续 token 的"重建上界"参考；我们今天实测 rFID 3.44 @4096 values）
- 变长/自适应（与"单模型多速率"最直接竞争）：**FlexTok**、**eLASTIC / ElasticTok**
- 新的通用 tokenizer：**Cosmos Tokenizer**（NVIDIA, 2025）

### B. 感知感知压缩（要报 FID-vs-bpp 时的参考语境）
- **Blau & Michaeli, ICML 2019** — R-D-P 理论，必引（定义 R-D-P 函数）
- **HiFiC** — High-Fidelity Generative Image Compression (Mentzer et al., NeurIPS'20)
- **MS-ILLM** (Muckley et al., 2023)、**PerCo** (Careil et al., ICCV'23)
- **Theis et al. 2022** — Lossy compression with Gaussian diffusion
- 注意：这些是 codec，需要 bit-accurate rate 才能进同一张表。

### C. 学习式 R-D（若哪天要做纯压缩论文）
- **Ballé et al. 2018** scale hyperprior（标准基线）
- **Minnen et al. 2018** autoregressive + hierarchical priors
- **Cheng et al. 2020** GMM + attention（anchor）
- **ELIC** (He et al. 2022)、**MLIC / MLIC++**
- 变量率：**Choi et al. 2019**、**Agustsson et al. 2019**、**Yang et al. 2020**
- 经典：JPEG / JPEG2000 / WebP / BPG / VVC(VTM intra)
- 实现：CompressAI（含预训练权重）

### D. 不能对标的
BLIP-2 / CLIP / DINO / LLM 的 "visual tokens" —— 语义接口，无像素重建。

## 4. 推荐的定位（如果 KD 有效）

**"固定 rate 下的 R-D-P + 单模型 8 速率"**，主表结构：

| 方法 | rate (bpp, 明确换算) | PSNR | LPIPS | rFID | 单模型多速率 |
|---|---|---|---|---|---|
| ALIT-S/B | 0.234 | — | — | 8.25 | 部分（变长） |
| TiTok / VQGAN | — | — | — | 个位数 | 否 |
| SD-VAE（实测） | ~0.5 (8bit) | 25.93 | — | 3.44 | 否 |
| HiFiC / PerCo（codec，仅参考行） | 0.15–0.5 | 27–29 | — | 2–5 | 否 |
| **我们（KD 后）** | 0.375–0.5625 | ~23–25 | ? | 目标 20–40 | **是（8 个 rate 一个模型）** |

配一句明确定位："本文不是 codec，不比 bpp 绝对最优；比的是**单一模型在 8 个 rate 上的
R-D-P 前沿形状**与速率切换能力。" —— 这样把对标对象从 codec 换回 tokenizer，且不撒谎。

## 5. 若坚持纯 R-D 论文，缺两件事

1. **bit-accurate rate**：必须量化 + 熵编码（对 z 建模并算术编码）实测 bpp。
   我们现在报的 "3072 values" 在压缩场不被接受，且量化还会掉点。
2. **PSNR 要追 3–6 dB**：需要空间自适应比特分配（现在的均匀 patch rate 结构缺这个）。

在这两点都没有的前提下，纯 R-D 论文写出来只会得到"比 JPEG 差"的评审结论。
