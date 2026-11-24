# 各家视觉编码器「可解码性」探针（2026-11-24）

## 问题

各家大模型供应商的**重量级视觉编码器**（作为多模态大模型 / 对比学习的视觉塔）到底
编码了多少**视觉信息**？判据不用下游分类/检索（那些只反映语义，不反映像素细节），
而是最直接的一条：**冻结编码器，只训一个小解码器把特征还原成 256×256 图像**，
还原质量 = 该表示里"可解码的视觉信息量"的直接下界。

动机来自本项目自己的困境：我们的 tokenizer 3072 值 rFID 86.5，而 SD-VAE 4096 值
rFID 3.44 —— 同为"值数相近的 latent"，可解码性差一个量级。所以需要一把尺子，
横向量一下"别家的编码器把多少图像信息留在特征里"。

## 协议（所有编码器完全一致，只有编码器本身不同）

| 项 | 设定 |
|---|---|
| 数据 | ImageNet-100，每类固定划分：val = 每类前 10 张（1000），train = 每类 200 张（20000） |
| 目标 | 短边缩到 256 + center crop 256（无增强） |
| 编码器输入 | 同一张图缩放到该编码器的 native 分辨率（CLIP/ViT/DINO/MAE 224, SigLIP 384, SAM 1024, Qwen2-VL 448） |
| 特征 | ViT 类取最后一层**空间 patch token**（丢 CLS）→ (C,h,w)；SAM 取 image encoder 输出；SD-VAE 取 latent；像素对照取 16×16×3 |
| 统一瓶颈 | 所有特征 **adaptive-avg-pool 到 16×16**（不足则 bilinear 放大）⇒ 每个编码器都是 16×16×C，只剩通道数（= 维度）差异 |
| 解码器 | 同一结构 `Conv1x1(C→256)` + 4×[Upsample×2, Conv3×3, GN, GELU] + `Conv3×3→3`，≈2.6M 参数，只有第一层输入通道随 C 变 |
| 优化 | L1 + AdamW(2e-4, wd 0.01) + OneCycle + 2 epoch（40k 图像通过）/ batch 32 |
| 指标 | val 1000 张的 PSNR / SSIM / L1 |

## 编码器清单（3 家厂商 + 生成式参考 + 对照）

| key | 厂商 | 模型 | 说明 |
|---|---|---|---|
| `clip_l14` | OpenAI | openai/clip-vit-large-patch14 | CLIP ViT-L/14（对比学习，语义塔） |
| `clip_b32` | OpenAI | openai/clip-vit-base-patch32 | CLIP ViT-B/32（轻量对照） |
| `siglip_so400m` | Google | google/siglip-so400m-patch14-384 | SigLIP SO400M（sigmoid 对比） |
| `vit_l16` | Google | google/vit-large-patch16-224-in21k | ViT-L 监督 IN-21k（无多模态） |
| `vit_h14` | Google | google/vit-huge-patch14-224-in21k | ViT-H 监督 IN-21k |
| `dinov2_l` / `dinov2_g` | Meta | facebook/dinov2-large / giant | DINOv2 自监督（像素/结构导向） |
| `mae_h` | Meta | facebook/vit-mae-huge | MAE 自监督（重建预训练） |
| `sam_h` | Meta | facebook/sam-vit-huge | SAM ViT-H image encoder（密集预测） |
| `qwen2vl_2b` | Alibaba | Qwen/Qwen2-VL-2B-Instruct | Qwen2-VL 的 ViT |
| `sdvae` | Stability | stabilityai/sd-vae-ft-mse | 生成式编码器（4096 值）参考 |
| `pixel16` | — | 无 | 16×16×3 下采样（768 值平凡码） |
| `randvit_l` | — | ViT-L 随机初始化 | 同结构但**没训练**（信息下限） |

## 预登记的预期（跑之前写死，避免事后解释）

1. **生成式 > 自监督/密集预测 > 对比学习**：`sdvae`、`sam_h`、`mae_h`、`dinov2` 应显著
   高于 `clip`/`siglip`（对比学习目标是语义不变性，会主动丢弃像素级细节）。
2. `randvit_l` ≈ `pixel16`（都是"没有可用信息"的下限），且明显低于所有真实编码器。
3. 同一模型家族内更大/更深略好（`dinov2_g` > `dinov2_l`，`vit_h14` > `vit_l16`），但差距
   应远小于"训练目标"造成的差距。
4. `clip_l14` 在**最后一层**可解码性差，取中间层（`--layer 0.5`）会明显回升
   —— 这是 CLIP 已知的"低层保留细节、高层语义化"现象，作为协议有效性检查。

## 已知口径缺陷（要写进结论的 caveat）

- **输入分辨率不同**：CLIP/ViT 224 vs SigLIP 384 vs SAM 1024。高分辨率编码器看到更多
  像素，但也只是把 256 crop 放大（没有额外细节），所以优势应有限；仍不是严格等输入。
- **16×16 网格是统一瓶颈**：对 SAM（64×64 token）和 SD-VAE（32×32）是**降采样**，会低估
  它们的真实可解码性 ⇒ 另跑 `--grid 32/64` 的原生网格版作补充。
- **解码器容量**：2.6M 参数、40k 图像通过可能欠拟合 ⇒ 用 `--width 128/512` 做容量敏感性
  检查；若所有编码器都随宽度单调上升，说明表格里的差异是"可解码性"的保守下界。
- 特征池化用 avg（非 max/attention）⇒ 也是一种固定口径的简化。

## 运行

```bash
# 远端（4 GPU 并行, 先顺序下载权重）
bash run_probe_enc_dec.sh
# 单个 / 补充实验
python3 probe_enc_dec.py --enc clip_l14 --layer 0.5 --out .../clip_l14_Lmid.json
python3 probe_enc_dec.py --enc sdvae --grid 32 --out .../sdvae_g32.json
python3 probe_enc_dec.py --enc clip_l14 --width 512 --out .../clip_l14_w512.json
```

结果 json 落在 `/root/autodl-tmp/tok256x12_t8/probe/*.json`，本地拉回到
`tok448_work/artifacts/probe/`。
