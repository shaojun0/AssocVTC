# 八臂架构矩阵消融 — 聚合表（自动生成）

> 由 `tools/alit_report.py --ablation8` 生成；来源 `doc/2026-10-17/artifacts/eval_ablation8.json`（`eval_linattn.py` 多臂**同进程同批次**评测，全量 5,000 张）。
> **配对规则：只有同进程同批次评测构成配对**；跨预算（6 ep 热启动 vs 45 ep 从零）、跨实例/代际的格**不可直接比**。

## 1. 八臂主表

| 臂（目录） | enc attn | dec attn | 读出 | PSNR ↑ | SSIM ↑ | L1_norm ↓ | enc M | dec M | total M | step / epoch | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stdattn_k64` | std | std | 单步 | 31.219 | 0.9037 | 0.09013 | 31.848 | 31.173 | 63.021 | 42750 / 45 | `eval_ablation8.json` |
| `stdattn_k64_rec` | std | std | BPTT | 28.582 | 0.8505 | 0.11474 | 31.848 | 31.207 | 63.055 | 11406 / 6 | `eval_ablation8.json` |
| `k64` | lin | std | 单步 | 26.602 | 0.7654 | 0.14632 | 31.909 | 31.173 | 63.082 | 42750 / 45 | `eval_ablation8.json` |
| `k64_rec` | lin | std | BPTT | 25.304 | 0.7156 | 0.16662 | 31.909 | 31.207 | 63.116 | 11406 / 6 | `eval_ablation8.json` |
| `stdlin_k64` | std | lin | 单步 | 24.714 | 0.6882 | 0.17658 | 31.848 | 31.177 | 63.025 | 42750 / 45 | `eval_ablation8.json` |
| `stdlin_k64_rec` | std | lin | BPTT | 24.166 | 0.6655 | 0.18697 | 31.848 | 31.210 | 63.058 | 11406 / 6 | `eval_ablation8.json` |
| `alllinear_k64_rec` | lin | lin | BPTT | 22.042 | 0.5622 | 0.23759 | 31.909 | 31.210 | 63.119 | 11406 / 6 | `eval_ablation8.json` |
| `alllinear_k64` | lin | lin | 单步 | 21.884 | 0.5535 | 0.24241 | 31.909 | 31.177 | 63.086 | 42750 / 45 | `eval_ablation8.json` |

## 2. 配对差（同一格内只动一个因素）

### 2.1 编码器注意力：std − lin（同解码器、同读出；正 = std 更好）

| 解码器 | 读出 | std PSNR | lin PSNR | **ΔPSNR** | ΔSSIM |
|---|---|---|---|---|---|
| std | 单步 | 31.219 (`stdattn_k64`) | 26.602 (`k64`) | **+4.6166** | +0.1382 |
| std | BPTT | 28.582 (`stdattn_k64_rec`) | 25.304 (`k64_rec`) | **+3.2780** | +0.1349 |
| lin | 单步 | 24.714 (`stdlin_k64`) | 21.884 (`alllinear_k64`) | **+2.8294** | +0.1347 |
| lin | BPTT | 24.166 (`stdlin_k64_rec`) | 22.042 (`alllinear_k64_rec`) | **+2.1233** | +0.1033 |

### 2.2 解码器注意力：lin − std（同编码器、同读出；负 = 线性解码器代价）

| 编码器 | 读出 | std dec PSNR | lin dec PSNR | **ΔPSNR** | ΔSSIM |
|---|---|---|---|---|---|
| std | 单步 | 31.219 (`stdattn_k64`) | 24.714 (`stdlin_k64`) | **-6.5048** | -0.2155 |
| std | BPTT | 28.582 (`stdattn_k64_rec`) | 24.166 (`stdlin_k64_rec`) | **-4.4166** | -0.1850 |
| lin | 单步 | 26.602 (`k64`) | 21.884 (`alllinear_k64`) | **-4.7176** | -0.2120 |
| lin | BPTT | 25.304 (`k64_rec`) | 22.042 (`alllinear_k64_rec`) | **-3.2619** | -0.1534 |

### 2.3 读出方式：BPTT − 单步——⚠️ **两列预算不同，不可直接比**

> BPTT 臂 = 从各自单步版**热启动 6 ep / 11,406 步**；单步臂 = **45 ep / 42,750 步从零**。该表的差**不是**「BPTT 的收益」，只能读作「不同预算/协议下的两个交付点」。BPTT 是否有效要看 §3 的**同臂内逐步形态**。

| enc | dec | 单步 PSNR | BPTT PSNR | 差(BPTT−单步) |
|---|---|---|---|---|
| std | std | 31.219 (`stdattn_k64`) | 28.582 (`stdattn_k64_rec`) | -2.6363 |
| std | lin | 24.714 (`stdlin_k64`) | 24.166 (`stdlin_k64_rec`) | -0.5480 |
| lin | std | 26.602 (`k64`) | 25.304 (`k64_rec`) | -1.2977 |
| lin | lin | 21.884 (`alllinear_k64`) | 22.042 (`alllinear_k64_rec`) | +0.1580 |

## 3. 逐步形态（BPTT 臂；同臂内比较，与训练预算无关）

| BPTT 臂（来源） | L1_255 step1 → 最优 → step8 | 总落差 | Δ₁ 占比 | 正增量步 | 均值曲线最优步 | 每图众数档（占比） |
|---|---|---|---|---|---|---|
| `stdlin_k64_rec`（`stdlin/stepwise.json`） | 14.161 → 10.762 → 10.776 | 3.399 px | 97.16% | 4/7 | step 5 | step 5（39.34%） |
| `alllinear_k64_rec`（`alllinear/stepwise.json`） | 13.841 → 13.683 → 13.693 | 0.158 px | 84.03% | 4/7 | step 5 | step 4（29.44%） |
| `stdattn_k64_rec`（`doc/2026-10-16/artifacts/alit/stepwise_stdattn_k64_rec.json`） | 15.415 → 6.597 → 6.613 | 8.817 px | 27.04% | 6/7 | step 7 | step 7（96.98%） |
| `k64_rec`（`doc/2026-10-16/artifacts/alit/stepwise_linattn_k64_rec.json`） | 16.944 → 9.600 → 9.603 | 7.344 px | 37.70% | 6/7 | step 7 | step 7（60.92%） |

**读出**：只有 **std 解码器**的 BPTT 臂出现「多步持续精修」；线性解码器两臂的 Δ₁ 占比 ≥84% ⇒ 链式循环**基本不动**（详见报告 §4）。
