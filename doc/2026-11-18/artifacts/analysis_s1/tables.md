# SAVOIAS OOD complexity analysis — tok448-s1 16x1248

Generated: 2026-11-18T06:54:13+08:00  
L1_255 / PSNR 取 **step 1/1**（单步模型 T=1：唯一一步）；L1_255 = 0–255 像素 L1（不 clamp），PSNR = [0,1] clamp 空间逐图 PSNR 再平均。

Inputs:
- per-image: `/tmp/tok448_eval/per_image_savoias_s1.json`
- gt_dir: `/tmp/savoias_gt`

## 1. Subset overview

| subset | n | complexity mean ± std | L1_255 ↓ | PSNR ↑ | Spearman ρ(complexity, L1) | p | Spearman ρ(complexity, PSNR) | p |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| People-Art | 420 | 46.1 ± 23.3 | 7.297 | 28.144 | +0.457 | p<1e-4 | -0.390 | p<1e-4 |
| Places | 200 | 43.0 ± 21.7 | 8.504 | 26.230 | +0.359 | p<1e-4 | -0.352 | p<1e-4 |
| MSCOCO | 200 | 47.5 ± 21.0 | 8.584 | 25.746 | +0.283 | p<1e-4 | -0.302 | p<1e-4 |
| **all joined** | 820 | — | 7.905 | 27.092 | — | — | — | — |

## 2. Complexity terciles (q1/q2 = 33.3/66.7 percentile; low: S≤q1, mid: q1<S≤q2, high: S>q2)

| subset | n (low/mid/high) | low L1 | mid L1 | high L1 | high−low (px) | relative (%) | low PSNR | mid PSNR | high PSNR | high−low (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| People-Art | 143/139/138 | 5.551 | 7.347 | 9.057 | 3.505 | 63.1 | 30.081 | 27.955 | 26.327 | -3.755 |
| Places | 71/63/66 | 7.088 | 8.682 | 9.857 | 2.769 | 39.1 | 27.859 | 26.039 | 24.660 | -3.198 |
| MSCOCO | 70/67/63 | 8.023 | 8.257 | 9.554 | 1.531 | 19.1 | 26.558 | 25.952 | 24.625 | -1.933 |

## 3. Quantile bins (k≤6) — complexity vs L1_255

| subset | bin | n | complexity mean | L1_255 mean | L1 SEM | PSNR mean | complexity range |
|---|---:|---:|---:|---:|---:|---:|---|
| People-Art | 1 | 69 | 12.2 | 5.051 | 0.357 | 30.767 | [0.0, 22.0] |
| People-Art | 2 | 65 | 27.3 | 5.842 | 0.349 | 29.745 | [22.0, 34.0] |
| People-Art | 3 | 72 | 38.9 | 6.877 | 0.351 | 28.097 | [34.0, 45.0] |
| People-Art | 4 | 73 | 50.0 | 7.758 | 0.505 | 27.768 | [45.0, 57.0] |
| People-Art | 5 | 66 | 63.9 | 8.025 | 0.381 | 27.159 | [57.0, 72.0] |
| People-Art | 6 | 75 | 80.9 | 9.940 | 0.476 | 25.623 | [72.0, 100.0] |
| Places | 1 | 34 | 12.6 | 7.169 | 0.827 | 28.102 | [0.0, 21.2] |
| Places | 2 | 32 | 26.8 | 6.994 | 0.772 | 27.683 | [21.2, 33.0] |
| Places | 3 | 34 | 37.0 | 8.092 | 0.726 | 26.496 | [33.0, 40.5] |
| Places | 4 | 32 | 45.4 | 8.867 | 0.813 | 25.962 | [40.5, 51.0] |
| Places | 5 | 33 | 58.3 | 8.818 | 0.567 | 25.393 | [51.0, 67.0] |
| Places | 6 | 35 | 76.8 | 10.954 | 0.621 | 23.860 | [67.0, 100.0] |
| MSCOCO | 1 | 34 | 17.8 | 8.064 | 0.813 | 26.806 | [0.0, 26.2] |
| MSCOCO | 2 | 31 | 31.1 | 8.369 | 0.693 | 26.153 | [26.2, 36.0] |
| MSCOCO | 3 | 33 | 40.5 | 7.352 | 0.512 | 26.386 | [36.0, 46.0] |
| MSCOCO | 4 | 33 | 51.2 | 8.475 | 0.835 | 25.953 | [46.0, 58.0] |
| MSCOCO | 5 | 32 | 62.9 | 9.082 | 0.654 | 25.004 | [58.0, 71.0] |
| MSCOCO | 6 | 37 | 78.5 | 10.006 | 0.536 | 24.319 | [71.0, 100.0] |

## Notes

- Join: SAVOIAS gt `{index: [filename, score 0–100]}` 按 **文件名 stem（整数）** 与 per-image `file_name` 对齐（gt 可能是 `.png`、图像是 `.jpg`，因此不能按全名 join）。
- 判据：ρ > 0 且 high−low 三分位差 > 0 ⇒ "越复杂越难重建"；relative % = (high−low)/low×100。
- 图：`fig_complexity_l1_panels.png/pdf`、`fig_complexity_l1_<sub>.png/pdf`、`fig_rho_bar.png/pdf`、`fig_tercile_bar.png/pdf`。
