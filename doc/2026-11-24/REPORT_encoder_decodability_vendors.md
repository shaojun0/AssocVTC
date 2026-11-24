# 四家厂商视觉编码器的「可解码性」实测（2026-11-24）

**问题**：DeepSeek-V4.1-Flash / GLM-5.3-Flash / Qwen3.8-Flash-Next / MiMo-V2.6-Flash
的视觉编码器，到底把多少**视觉信息**留在特征里？

**方法**：冻结编码器 → 取空间 token → 训练一个**完全相同的小解码器**还原 256×256 图像。
还原质量（PSNR/SSIM/L1）就是该表示"可解码信息量"的直接下界。整个过程编码器不更新。

## 1. 协议

| 项 | 设定 |
|---|---|
| 仓库 | `deepseek-ai/DeepSeek-V4.1-Flash-1017`、`zai-org/GLM-5.3-Flash`、`Qwen/Qwen3.8-Flash-Next`、`XiaomiMiMo/MiMo-V2.6-Flash-RL` |
| 权重 | 整仓 178–510 GB，**只抽 vision tower**（safetensors header + HTTP Range）：GLM 347 keys/1.13GB、Qwen 333/0.90GB、MiMo 364/1.46GB、DS 259/0.82GB |
| 载入 | GLM/Qwen 用 transformers 5.20.2 原生 `*VisionModel`（missing=0/unexpected=0）；MiMo 用仓库自带 `modeling_mimo_v2.py`（缺 3 个 merger bias⇒置 0）；DS 用 `inference/vision.py`（自包含 ViT，修了 RoPE 建表在 CPU 的 bug） |
| 输入 | **各自原生分辨率**：GLM 448²、Qwen 512²、MiMo 448²、DS 448²；只做 resize + 各家 mean/std，无增强 |
| 特征 | **不压缩空间、不降维**，原生网格 + 全部通道（见下表 `feat`） |
| 解码器 | 同一个：`Conv1x1(C→256)` + `log2(256/g)` × [Upsample×2, Conv3×3, GN, GELU] + `Conv3×3→3`；2.0–3.6M 参数 |
| 损失/优化 | L1 + AdamW(2e-4) + OneCycle；batch 32 |
| 数据 | ImageNet-100：每类 100 张训练（10k）、每类 10 张验证（1000）；**2 epoch = 624 步**（四组完全同预算） |
| 指标 | val 1000 张的 PSNR / SSIM / L1(×255) |

## 2. 核心结果：**同一个编码器，两个出口差 7–9 dB**

这些多模态模型的视觉链是 `图像 → ViT → token →（merger/投影）→ LLM`：

| 编码器 | ViT 出口（merger 之前） | LLM 入口（merger 之后） | Δ |
|---|---|---|---|
| Qwen3.8-Flash-Next | **23.78 dB** | 15.12 dB | **+8.66** |
| GLM-5.3-Flash | **22.91 dB** | 14.23 dB | **+8.68** |
| MiMo-V2.6-Flash | **22.07 dB** | 14.77 dB | **+7.29** |
| DeepSeek-V4.1-Flash | **23.73 dB** | n/a¹ | — |

¹ DeepSeek 的 aligner（3× 下采样 + 5120 维投影）权重抽出后为脏值（BF16 文件里出现 ~1e37，
疑似 FP8 量化残留），该格暂缺；它的 **ViT 出口**已测（23.73 dB）。

**结论 1：四家的视觉塔本身都编码了大量像素级视觉信息**（22–24 dB 同一档，差距 ≤1.7 dB）。
"哪家视觉编码器更强"在这把尺子上几乎没有区别。

**结论 2：差距发生在通向 LLM 的 adapter（merger/投影）**：Qwen/GLM 掉 ~8.7 dB、MiMo 掉 7.3 dB。
GLM 尤其彻底（14.2 dB，见 §3 的打乱对照），Qwen/MiMo 稍好（15 dB 左右）。

## 3. 因果对照：空间位置打乱后重建质量几乎不变（GLM）

诊断（1000 张、250 步、解码器宽 128）：把特征的空间排列分别按
row-major / 转置 / pixel-unshuffle / **随机打乱** 喂给解码器。**随机打乱**是判据：
若与 row-major 相当，说明特征里没有"哪个位置长什么样"的信息。

| 编码器 | 出口 | row-major | 随机打乱 | 判定 |
|---|---|---|---|---|
| DeepSeek | ViT | **22.05** | 12.76 | 空间信息强 |
| Qwen3.8 | ViT | **21.94** | 12.81 | 空间信息强 |
| MiMo | ViT | **21.58** | 12.87 | 空间信息强 |
| GLM-5.3 | ViT | **19.69** | 12.78 | 空间信息强 |
| GLM-5.3 | **LLM 入口** | 13.84 | 12.82 | **≈打乱下限 ⇒ 空间信息基本没了** |

## 4. 全预算补充（20k 训练 / 3 epoch，仅作参考）

| 口径 | GLM | Qwen | MiMo | DS |
|---|---|---|---|---|
| LLM 入口（post） | 14.35 | 15.88 | 15.05 | — |
| ViT 出口（pre） | — | — | — | **25.81** |

（与 §2 的同预算表趋势一致；DS 的 25.81 是全预算、其余为 10k/2ep，不能混用。）

## 5. 图像证据

- `artifacts/probe_vendor2/grids_pre/`：ViT 出口，GLM 22.91 dB 能清楚认出鸟
- `artifacts/probe_vendor2/grids_post/`：LLM 入口，GLM 14.23 dB 只剩色块
- `artifacts/probe_vendor/`：全预算 LLM 入口的 4 张对照图

## 6. 口径与 caveat

1. **解码器容量有限**（2–3.6M 参数、10k 图像/2 epoch）⇒ 表中数字是**保守下界**，
   真实可解码性只会更高；不改变"pre ≫ post"的结论（两组同解码器、同预算）。
2. **输入分辨率/merge 规则不同**：Qwen 512²、其余 448²；GLM/Qwen/MiMo 出口是 merger 后
   (16/14×16/14, 4096/4608 维)，DS 出口是 ViT 的 (32×32, 1024 维)。同 budget、同解码器，
   但"特征形状"本身是各家设计的一部分——这正是被测量的东西。
3. 空间网格：GLM/Qwen/DS 32×32（DS 为 ViT），MiMo 28×28。解码器对非 2 幂网格做一次
   邻近插值（28→32），影响可忽略。
4. DeepSeek 的 LLM 入口（aligner 后）缺数；若要补齐需从 FP8 组织里正确反量化 aligner。
5. 无增强、单 seed；PSNR 的绝对基准是 256² 原图。

## 7. 一句话总结

> 这四家的视觉编码器**都真的编码了视觉信息**（可解码到 22–24 dB）；
> 但这些模型为了让 LLM 好用，在 merger/投影处**主动丢掉了大部分像素信息**（掉 7–9 dB，
> GLM 甚至把空间结构抹平）。因此"拿 LLM-facing token 做像素重建"必然失败，
> 想做像素级重建必须直接监督 ViT/latent，而不是走 LLM 的视觉摘要。

## 8. 复现

```bash
# 抽 vision 权重（只下 ~1GB）
python3 tools/fetch_vision_weights.py --repo <org/repo> --shard <shard> \
    --prefix <model.visual.|visual.|vision.> --out vis_enc.safetensors
# 单个编码器
python3 tools/probe_enc_dec_vendor.py --vendor glm5_flash --stage pre \
    --train_per_class 100 --epochs 2 --out glm_pre.json --save_grid glm_pre.png
# 打乱对照
python3 tools/diag_layout.py glm5_flash post
```
