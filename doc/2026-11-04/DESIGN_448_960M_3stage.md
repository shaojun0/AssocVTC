# DESIGN — 448×448 / patch 32 / 编码器·解码器各 ~480M / 三阶段（AssocVTC）

> 日期 2026-11-04 · 用户口径 · 未设 goal，仅定时任务监控
> 论文依据：`/userdata/paper/pp_zh/cjc-article-v2.0.pdf`（v2.0，2026-11-03）
> 仓库：`/home/linaro/dsh/AssocVTC`（= GitHub `shaojun0/AssocVTC`）

## 0. 用户口径
1. 继续 AssocVTC 实验，分辨率 224² → **448²**；
2. 因分辨率变化，**分块面积 ×4**（patch 16 → **32**，已与用户确认；N 保持 196 不变）；
3. **编码器、解码器参数量各 480M**（合计 ~960M，是论文 240M+240M 的 2×）；
4. 沿用论文的 **3 阶段**训练协议；
5. 需要**定时任务监控**；**不设 goal**。

## 1. 架构：相对论文基线只变 3 项
| 项 | 论文基线（224） | 本实验（448） |
|---|---|---|
| 输入 | 224×224 | **448×448** |
| patch | 16×16 | **32×32**（面积 4×） |
| N = (H/p)² | 196 | **196（不变）** |
| K / T / 每步 | 32 / 16 / 2 | 32 / 16 / 2（不变） |
| 编码器 | 1248 宽 ×13 层, 16 头 → 244.43M | **1744 宽 ×13 层, 16 头 → 480.35M** |
| 解码器 | 1440 宽 ×7 层, 16 头 → 237.11M | **2048 宽 ×7 层, 16 头 → 478.72M** |
| 像素头 | hidden 1024 → 3·16² = 768 | hidden 1024 → **3·32² = 3072** |
| 阶段3 每 token 维 | 224（每图 7168 值） | 224（**每图 7168 值，与论文同值数口径**） |

- 计参口径：`SRPhase1LinAttn.count_params()`，与论文 244.43/237.11 逐位复现（见 `count_448_960m.py` 输出）。
- **scaling 轴 = 只加宽、深度钉死**（enc 13 / dec 7），与仓库 2026-10-24 容量阶梯同一方法（单一自变量）。
- 448² + patch32 ⇒ N=196，序列长度、注意力复杂度、K、T、压缩比全部与论文一致；**唯一变化 = 每 patch 覆盖的像素数（4×）与参数量（2×）**。

## 2. 三阶段协议（严格照论文 §3.2 / 表 1）
| 阶段 | 内容 | 初始化 | epoch | 等效 batch | 步数 | lr |
|---|---|---|---|---|---|---|
| 1 | 单步读满（T=1，窗口 [0,K]） | 从零 | 45 | 128 | 42,795 | 2e-4 |
| 2 | 16 档循环读出（fixed block 2，每步 2 token） | 阶段1 `ckpt_best` | 6 | 64 | 11,406 | 2e-4 |
| 3 | 输出投影到 224 维（`--zs_dim 224`，SVD 秩-224 热启动） | 阶段2 `ckpt_best` | 3 | 64 | 5,703 | 2e-4 |

- 优化器：AdamW，cosine + 8% warmup，wd 0.05，梯度裁剪 1.0，bf16（`--amp 1`）。
- 数据：ImageNet-100，train 121,689 / val 5,000（每类等距抽 50）。
- 步数：bs128 → 951 步/ep；bs64 → 1,901 步/ep（与论文 42,750/11,406/5,703 同量级，差 ≤0.2%）。
- ⚠️ **阶段 3 口径说明**：论文正文描述「宽度 D→d 线性退火」，但仓库既有实现（`--zs_dim` + `--init_from` SVD 秩近似，2026-10-26 `launch_zs224_e3.log`）是**一次性降到 224 维 + 3 epoch 全量微调**，论文表 2 的数字即出自该实现。本实验**沿用仓库既有实现**以保持与论文数字同口径；若需要真正的逐步退火，作为后续可选变体（不改本主线）。
- ⚠️ **`--dec_grad_ckpt` 必须全阶段固定同一值**（README 记录：bf16 重算在 11,406 步后产生 0.341 dB 终值差异，属实例级漂移，不能配对比较）。

## 3. 数据与评测
- 数据：`/root/autodl-tmp/datasets/imagenet100`（100 个 synset 目录、原始 JPEG）。
  train: `RandomResizedCrop(448, scale 0.35–1.0)` + flip；val: `Resize(512, BICUBIC)` + `CenterCrop(448)`（`data_linattn.py` 的 `--img_size` 自动处理）。
- 评测入口：`eval_stepwise_linattn.py`（cfg 从 `ckpt["cfg"]` 读，**自动**得到 448/patch32/N196；唯一硬编码是打印串里的 "224×224"，纯装饰）。
- 指标：PSNR（[0,1] 逐图再平均，dB）、SSIM、L1_255；逐 step 表（16 档）。
- 每阶段各评测一次：A 相单步（T=1）、B 相 16 步逐步、S3 相 224 维 16 步逐步。
- 与 224 基线对照时的口径：**同值数**（448 值 = 2 token×224；7168 值 = 32 token×224）——S3 臂直接可比；A/B 相为 1248/2048 维/token 的完整宽度臂，只作训练过程参考。

## 4. 远端环境
- `ssh -p 23556 root@connect.nma1.seetacloud.com`（AutoDL；2×A800 80GB PCIe，112 核，/root/autodl-tmp 80G 剩 ~35G）
- 代码：`/root/autodl-tmp/exp448/code/AssocVTC`（从本地 rsync，commit 见 `exp448_bench/sync_info.txt`）
- 产物：`/root/autodl-tmp/exp448/{A,B,S3,logs,eval}`
- python：`/root/miniconda3/bin/python -u`；`PYTHONPATH=<repo>`
- **GPU0 = 训练，GPU1 = 评测**（评测不抢训练卡）；每阶段训练结束后立即在 GPU1 起评测。

## 5. 时间预估（bench 实测，2026-11-04 12:09）
| 阶段 | 配置（micro×accum） | s/step | 峰值显存 | 步数 | 预估墙钟 |
|---|---|---|---|---|---|
| A 45ep（T=1） | 128×1 | 1.09 | 59.5 G | 42,795 | **13.0 h** |
| B 6ep（T=16） | 16×4 | 4.41 | 60.7 G | 11,406 | **14.0 h** |
| S3 3ep（zs 224） | 16×4 | 4.38 | 60.6 G | 5,703 | **6.9 h** |
| eval ×3 | bs 16–64 | 0.027 s/img | — | 5,000/次 | 0.12 h |
| **合计** | | | | | **≈34.0 h**（单卡 A800） |

- `--dec_grad_ckpt` 对 std 解码器 **no-op**（bench 实证）⇒ 全程 0，且与论文 240M 臂同口径。
- B/S3 的 micro 32 必 OOM（78.08 GiB 已分配）；60.7 G ≈ 77% 显存，是唯一可行且最快的协议内配置。
- 第二张 A800（GPU1）只做阶段评测（每阶段 ~2.5 min）+ PCA 基线，不占用训练。

## 6. 风险与对策
| 风险 | 对策 |
|---|---|
| B 相 960M + T=16 反传显存（240M 档 bs32 已 69.6G/80G） | 先 bench 扫 micro-batch / accum；必要时 `--dec_grad_ckpt 1` 并在全阶段固定 |
| 448 RandomResizedCrop 数据吞吐（JPEG 解码 + 4× 像素） | bench 量 img/s；`--num_workers` 16→24 |
| ckpt 体积（960M bf16 ~1.9G，fp32 ~3.8G） | `--save_opt 0`，只留 `ckpt_best`，阶段跑完删 `ckpt_last` |
| 磁盘 35G | 每阶段产物及时 `fetch` 回本地后清理 |
| 单种子 | 与仓库历史一致；结论只作方向性 |

## 7. 产物与监控
- 本地：`~/dsh/exp448_bench/{BENCH_448_960M.md,DESIGN_448_960M_3stage.md,SYNC.md}`
- 远端：`/root/autodl-tmp/exp448/logs/*.log`、`status.json`、`/root/autodl-tmp/exp448/eval/*.json`
- 监控：远端 watchdog（20 min 轮询写 `status.json`/`status.log`）+ 本地 poller + DSH 定时任务（每 2 h 汇总一次，异常即报）
- 最终：报告 `doc/2026-11-04/REPORT_448_960M_3stage.md` + 结果 json + 曲线
