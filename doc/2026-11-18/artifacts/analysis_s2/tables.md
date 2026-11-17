# SAVOIAS OOD complexity analysis — tok448-s2 16x448

Generated: 2026-11-18T06:54:07+08:00  
L1_255 / PSNR 取 **step 1/1**（单步模型 T=1：唯一一步）；L1_255 = 0–255 像素 L1（不 clamp），PSNR = [0,1] clamp 空间逐图 PSNR 再平均。

Inputs:
- per-image: `/tmp/tok448_eval/per_image_savoias_s2.json`
- gt_dir: `/tmp/savoias_gt`

## 1. Subset overview

| subset | n | complexity mean ± std | L1_255 ↓ | PSNR ↑ | Spearman ρ(complexity, L1) | p | Spearman ρ(complexity, PSNR) | p |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| People-Art | 420 | 46.1 ± 23.3 | 7.870 | 27.402 | +0.474 | p<1e-4 | -0.408 | p<1e-4 |
| Places | 200 | 43.0 ± 21.7 | 9.026 | 25.651 | +0.375 | p<1e-4 | -0.374 | p<1e-4 |
| MSCOCO | 200 | 47.5 ± 21.0 | 9.027 | 25.257 | +0.297 | p<1e-4 | -0.317 | p<1e-4 |
| **all joined** | 820 | — | 8.434 | 26.451 | — | — | — | — |

## 2. Complexity terciles (q1/q2 = 33.3/66.7 percentile; low: S≤q1, mid: q1<S≤q2, high: S>q2)

| subset | n (low/mid/high) | low L1 | mid L1 | high L1 | high−low (px) | relative (%) | low PSNR | mid PSNR | high PSNR | high−low (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| People-Art | 143/139/138 | 5.937 | 7.918 | 9.825 | 3.888 | 65.5 | 29.393 | 27.195 | 25.546 | -3.847 |
| Places | 71/63/66 | 7.476 | 9.194 | 10.533 | 3.057 | 40.9 | 27.313 | 25.478 | 24.028 | -3.285 |
| MSCOCO | 70/67/63 | 8.394 | 8.681 | 10.098 | 1.704 | 20.3 | 26.101 | 25.460 | 24.103 | -1.998 |

## 3. Quantile bins (k≤6) — complexity vs L1_255

| subset | bin | n | complexity mean | L1_255 mean | L1 SEM | PSNR mean | complexity range |
|---|---:|---:|---:|---:|---:|---:|---|
| People-Art | 1 | 69 | 12.2 | 5.364 | 0.374 | 30.130 | [0.0, 22.0] |
| People-Art | 2 | 65 | 27.3 | 6.283 | 0.368 | 29.009 | [22.0, 34.0] |
| People-Art | 3 | 72 | 38.9 | 7.404 | 0.372 | 27.347 | [34.0, 45.0] |
| People-Art | 4 | 73 | 50.0 | 8.367 | 0.532 | 26.997 | [45.0, 57.0] |
| People-Art | 5 | 66 | 63.9 | 8.684 | 0.405 | 26.398 | [57.0, 72.0] |
| People-Art | 6 | 75 | 80.9 | 10.797 | 0.504 | 24.827 | [72.0, 100.0] |
| Places | 1 | 34 | 12.6 | 7.517 | 0.846 | 27.590 | [0.0, 21.2] |
| Places | 2 | 32 | 26.8 | 7.429 | 0.803 | 27.084 | [21.2, 33.0] |
| Places | 3 | 34 | 37.0 | 8.530 | 0.755 | 25.988 | [33.0, 40.5] |
| Places | 4 | 32 | 45.4 | 9.436 | 0.848 | 25.352 | [40.5, 51.0] |
| Places | 5 | 33 | 58.3 | 9.369 | 0.583 | 24.798 | [51.0, 67.0] |
| Places | 6 | 35 | 76.8 | 11.737 | 0.640 | 23.206 | [67.0, 100.0] |
| MSCOCO | 1 | 34 | 17.8 | 8.370 | 0.828 | 26.404 | [0.0, 26.2] |
| MSCOCO | 2 | 31 | 31.1 | 8.825 | 0.724 | 25.644 | [26.2, 36.0] |
| MSCOCO | 3 | 33 | 40.5 | 7.723 | 0.536 | 25.904 | [36.0, 46.0] |
| MSCOCO | 4 | 33 | 51.2 | 8.926 | 0.861 | 25.442 | [46.0, 58.0] |
| MSCOCO | 5 | 32 | 62.9 | 9.540 | 0.678 | 24.531 | [58.0, 71.0] |
| MSCOCO | 6 | 37 | 78.5 | 10.612 | 0.562 | 23.764 | [71.0, 100.0] |

## Notes

- Join: SAVOIAS gt `{index: [filename, score 0–100]}` 按 **文件名 stem（整数）** 与 per-image `file_name` 对齐（gt 可能是 `.png`、图像是 `.jpg`，因此不能按全名 join）。
- 判据：ρ > 0 且 high−low 三分位差 > 0 ⇒ "越复杂越难重建"；relative % = (high−low)/low×100。
- 图：`fig_complexity_l1_panels.png/pdf`、`fig_complexity_l1_<sub>.png/pdf`、`fig_rho_bar.png/pdf`、`fig_tercile_bar.png/pdf`。
