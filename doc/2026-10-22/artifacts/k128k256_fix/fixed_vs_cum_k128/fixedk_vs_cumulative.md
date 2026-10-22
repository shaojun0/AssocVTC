# fixed（每步均匀）vs cumulative（square 形状缩放）—— 同 K, 各 8 步

| K | plan | T=1 PSNR | T=8 PSNR | T=8 L1_norm | T=8 SSIM | ΔT8 PSNR (fixed−cum) | ΔT8 L1_norm | ΔT8 SSIM |
|---|---|---|---|---|---|---|---|---|
| 128 | fixed | 34.005 | 31.234 | 0.087 | 0.9096 | +0.817 | -0.0095 | +0.0155 |
| 128 | cumulative | — | 30.417 | 0.096 | 0.8941 | — | — | — |
