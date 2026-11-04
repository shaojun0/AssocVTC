# RUN — AssocVTC 448² / patch32 / N196 / 960M · 论文三阶段（2026-11-04）

> 远端：`ssh -p 23556 root@connect.nma1.seetacloud.com`（AutoDL `autodl-container-119b47bef5-ac805d73`，
> 2×A800 80GB PCIe，112 核，`/root/autodl-tmp` 80G 剩 ~35G）
> 代码：`/root/autodl-tmp/exp448/code/AssocVTC`（本地 `/home/linaro/dsh/AssocVTC` @ `34638ed`，rsync 副本无 `.git`）
> 产物根：`/root/autodl-tmp/exp448/{A,B,S3,logs,eval}`
> 设计/协议全表：`/home/linaro/dsh/exp448_bench/DESIGN_448_960M_3stage.md`
> 脚本：`tools/run_448_960m_3stage.sh`、`tools/launch_448_960m.sh`、`tools/eval_448_960m.sh`、`tools/fetch_448_960m.sh`

## 0. 一句话

分辨率 224²→**448²**、patch 16→**32**（N 保持 **196**）、编码器/解码器各 **~480M**（论文 2×，合计 **960.27M**），
按论文三阶段（A: T=1 45ep / B: T=16 6ep / S3: zs_dim=224 3ep）串行训练，训练占 **GPU0**，
每阶段结束立刻在 **GPU1** 后台逐步评测（与下一阶段训练并行）。

## 1. 配置表

### 1.1 架构（唯一自变量 = 宽度 ×2；深度钉死 13/7）

| 项 | 论文基线（224） | **本实验（448）** |
|---|---|---|
| 输入 | 224×224 | **448×448** |
| patch | 16×16 | **32×32**（覆盖面积 4×） |
| N = (H/p)² | 196 | **196（不变）** |
| K / T / 每步 | 32 / 16 / 2 | 32 / 16 / 2（不变） |
| 编码器 | 1248 宽×13 层, 16 头 → 244.43M | **1744 宽×13 层, 16 头 → 480.537M** |
| 解码器 | 1440 宽×7 层, 16 头 → 237.11M | **2048 宽×7 层, 16 头 → 479.734M** |
| 合计 | 481.54M | **960.270M** |
| 像素头 | hidden 1024 → 3·16² = 768 | hidden 1024 → **3·32² = 3072** |
| S3 每 token 维 | 224（每图 7168 值） | 224（**每图 32×224 = 7168 值，同值数口径**） |

> 参数量为本机 `SRPhase1LinAttn.count_params()` 在 **meta device** 实测（`tools/launch_448_960m.sh`
> 启动时也会实测一遍并写进 `run_info.json`）。DESIGN 表里写的 480.35/478.72M 是**宽度扫描时
> enc 固定 1248** 的口径（解码器含 `mem_proj(enc_dim→dec_dim)`，enc 加宽到 1744 会给解码器多
> +496×2048 = 1.016M）⇒ 组合口径正确值是 A/B 相 **480.537 / 479.734M（合计 960.270M）**，差 0.2%，
> 不影响任何结论。S3 相引入 `zs_dim=224` 瓶颈后**总参数 957.939M**（编码器 481.318M：+zs_proj/zcls_proj
> 0.78M；解码器 476.621M：`mem_proj` 从 1744×2048 收缩到 224×2048，−3.11M）——这是同一模型换了
> 输出投影，不是另一个模型，属预期。

### 1.2 三阶段协议（严格照论文 §3.2 / 表 1）

| 阶段 | 内容 | 初始化 | epoch | 等效 batch | 步数/ep | lr | 产物 |
|---|---|---|---|---|---|---|---|
| **1 / A** | 单步读满（T=1，窗口 [0,K]） | 从零 | 45 | 128 | 951 | 2e-4 | `$OUT/A` |
| **2 / B** | 16 档循环读出（`fixed block 2`，每步 2 token） | A `ckpt_best` | 6 | 64 | 1,901 | 2e-4 | `$OUT/B` |
| **3 / S3** | 输出投影降到 224 维（`--zs_dim 224`，SVD 秩-224 热启动） | B `ckpt_best` | 3 | 64 | 1,901 | 2e-4 | `$OUT/S3` |

公共：AdamW · cosine · 8% warmup · wd 0.05 · clip 1.0 · bf16 `--amp 1` · `--seed 0` ·
`--save_opt 0`（不为 resume 存 AdamW 状态，ckpt 从 ~11.6G 降到 ~3.8G）· `--num_workers 16` ·
`--time_budget_h 0`（不设预算）。

> S3 口径说明：沿用仓库既有实现（`--zs_dim` + `--init_from` 的 SVD 秩近似 + 3 epoch 全量微调），
> 论文表 2 的数字即出自该实现；「宽度 D→d 线性退火」是论文正文描述，本主线不采用（见 DESIGN §2）。
> `--dec_grad_ckpt` **必须全阶段同一值**（bf16 重算在长跑后会造成实例级终值漂移 0.341 dB，不可配对比较）。

## 2. micro-batch / 累积 / 梯度检查点（bench 实测已定，2026-11-04 12:09）

`train_linattn.py` 的口径是 **`--batch_size` = 等效批量**，micro = `batch_size / accum_steps`。
脚本用 `MICRO_BS_* × ACCUM_*` 表达「micro-batch × 累积」，内部折算成
`--batch_size (micro×accum) --accum_steps (accum)`。

**bench 结论**（详见 `/home/linaro/dsh/exp448_bench/BENCH_448_960M.md`，A800 80GB）：

| 阶段 | micro × accum | 等效 batch | s/step | 峰值显存 | 步数 | 预计墙钟 |
|---|---|---|---|---|---|---|
| A（T=1） | **128 × 1** | 128 | **1.09 s** | 59.5 G | 42,795 | **13.0 h** |
| B（T=16） | **16 × 4** | 64 | **4.41 s** | 60.7 G | 11,406 | **14.0 h** |
| S3（zs 224） | **16 × 4** | 64 | **4.38 s** | 60.6 G | 5,703 | **6.9 h** |
| eval ×3 | bs 16–64 | — | 0.027 s/img | — | 5,000/次 | 0.12 h |

⇒ 训练合计 **≈33.9 h**、总计 **≈34.0 h**（单卡 A800；第二张 A800 只做阶段评测与 PCA 基线）。

| 变量 | 实取值 | 说明 |
|---|---|---|
| `MICRO_BS_A` / `ACCUM_A` | **128 / 1** | A 相单步，59.5 G，实测 1.06–1.09 s/step；micro 64×2 同速但显存更省（40.4 G），需更保守时可用 |
| `MICRO_BS_B` / `ACCUM_B` | **16 / 4** | B/S3 相；**micro 32 必 OOM**（78.08 GiB 已分配后再要 784 MiB 失败），micro 8×8 可行但慢 ~10%（4.86 s/step, 40.2 G） |
| `DEC_GRAD_CKPT` | **0** | ⚠️ bench 实测：该开关对 `--dec_attn std` 的 `RecurrentOutputQueryDecoder` **完全 no-op**（代码只接到全线性解码器）⇒ micro32 下 ck0/ck1 的 OOM 文本逐字相同、micro16 下显存/耗时全一致。**设 0 即可**，同时与论文 240M 臂（默认 0）同口径 |
| `FORCE_EFF_BS` | 0 | 等效批量未偏离（A=128 / B=64 / S3=64），与论文逐步数一致 |

> ⚠️ `MICRO_BS_A×ACCUM_A ≠ 128` 或 `MICRO_BS_B×ACCUM_B ≠ 64` 时脚本**默认直接报错退出**（防手误），
> 确认要偏离才 `FORCE_EFF_BS=1`。
> ⚠️ 训练内 val 为 `torch.no_grad()`，bench 未单独实测其显存峰值；首个 epoch 末 val 出现时盯一眼 `vram`。
> ℹ️ 另一条 bench 发现：`--limit 2000` 的短跑会被 epoch 边界停顿抬高计时 ~15%，A 相全量实测更接近 1.09 s/step。

## 3. 一键启动

### 3.1 干跑（先看命令，不执行任何东西）

```bash
cd /root/autodl-tmp/exp448/code/AssocVTC

DRY=1 bash tools/launch_448_960m.sh        # 启动器: 预检 + run_info 预览 + 三阶段命令
DRY=1 bash tools/run_448_960m_3stage.sh    # 训练链: 三阶段 + 三段评测的完整命令
DRY=1 CKPT=/root/autodl-tmp/exp448/S3/ckpt_best.pt bash tools/eval_448_960m.sh   # 评测
DRY=1 bash tools/fetch_448_960m.sh         # 拉取（只打印 ssh/scp）
```

### 3.2 正式启动（**已于 2026-11-04 12:10 执行**）

```bash
cd /root/autodl-tmp/exp448/code/AssocVTC
mkdir -p /root/autodl-tmp/exp448/logs

# 实际启动命令（bench 实测值已填入；提交时 commit=34638ed 由 env 显式指定，
# 因为 rsync 副本不含 .git）
COMMIT=34638ed MICRO_BS_A=128 ACCUM_A=1 MICRO_BS_B=16 ACCUM_B=4 DEC_GRAD_CKPT=0 \
  nohup setsid bash tools/launch_448_960m.sh \
  > /root/autodl-tmp/exp448/logs/launch.out 2>&1 &

# 启动器会打印：PID、tail 命令、完成标志、中断命令
```

**启动记录**：2026-11-04 12:10:05 +08:00，训练链 PID=15113，`run_info.json` 已落；
实测参数量 enc **480.537M** / dec **479.734M** / total **960.270M**（N=196）；
A 相 step 200/42,750 时 113 img/s、vram 59.5 G、ETA 13.4 h。
监控：远端 watchdog（20 min 轮询）+ 本地 poller + DSH 定时任务每 2 h。

> 重跑前先删 `$OUT/ALL_DONE`，或用 `SKIP_STAGE1/2/3=1` 从某阶段续跑；`FORCE=1` 可越过占用检查。

等价的「不经过启动器、直接前台跑」：

```bash
bash tools/run_448_960m_3stage.sh     # 前台；SSH 断了会死，正式跑请用上面的 setsid nohup
```

### 3.3 监控

```bash
cd /root/autodl-tmp/exp448
tail -f logs/run.log              # 链路汇总（阶段开始/rc/评测启动）
tail -f logs/stage1.log           # 当前阶段训练日志（首行即完整命令；含 [cfg] [params] [val]）
cat A/status.json                 # step / epoch / loss / val_psnr / eta_h / vram_gb
grep -c '^\[val\]' logs/stage1.log
ls STAGE1_DONE ALL_DONE 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

### 3.4 中断

```bash
kill "$(cat /root/autodl-tmp/exp448/run.pid)"     # 训练链（setsid 已脱离终端）
pkill -P "$(cat /root/autodl-tmp/exp448/run.pid)" # 连带子进程（评测）
```

## 4. 产物路径

| 远端 | 内容 |
|---|---|
| `$OUT/A/ckpt_best.pt` | A 相最优（T=1），**B 相热启动源**；~3.8G |
| `$OUT/B/ckpt_best.pt` | B 相最优（T=16），**S3 热启动源**；~3.8G |
| `$OUT/S3/ckpt_best.pt` | S3 相最优（zs_dim=224）；~3.8G |
| `$OUT/{A,B,S3}/args.json` | 该阶段完整 argparse（复现依据） |
| `$OUT/{A,B,S3}/status.json` | 末次监控快照（step/epoch/val_psnr/eta_h/vram_gb） |
| `$OUT/{A,B,S3}/train_log.jsonl` | 逐步曲线（step/loss/recon/lr/img_per_s/eta_h/vram_gb） |
| `$OUT/{A,B,S3}/eval_val.json` | 训练内验证指标（l1/psnr/ssim + 两条基线） |
| `$OUT/eval/stage{1,2,3}_stepwise.json` | **逐步评测结果**（16 档；A 相 1 行） |
| `$OUT/logs/stage{1,2,3}.log` | 训练日志（首行 = 完整命令） |
| `$OUT/logs/eval_stage{1,2,3}.log` | 评测日志 |
| `$OUT/logs/run.log`、`launch.out` | 链路汇总 / 启动器输出 |
| `$OUT/run_info.json` | commit、实测参数量、三阶段完整命令、GPU 分配、knobs |
| `$OUT/STAGE{1,2,3}_DONE`、`ALL_DONE` | 成功标记（`echo $?` 级别的判据） |
| `$OUT/FAILED_stage{N}`、`FAILED_eval_stage{N}` | 失败标记（文件内容 = 时间 + 原因） |
| `$OUT/BUDGET_HIT_stage{N}` | `TIME_BUDGET_H>0` 且到点触发（协议未跑满） |

本地（`PORT=23556 bash tools/fetch_448_960m.sh` 之后）：

| 本地 `/home/linaro/dsh/exp448_bench/artifacts/` | 内容 |
|---|---|
| `eval/stage{1,2,3}_stepwise.json` | 逐步评测 json |
| `logs/stage{1,2,3}.log`、`logs/eval_stage{1,2,3}.log` | 日志（已剥 `\r` 进度条） |
| `logs/run.log`、`logs/launch.out` | 链路日志 |
| `args/stage{N}_args.json` | argparse 全文 |
| `args/stage{N}_cfg.txt` | 从训练日志摘出的 `[cfg]/[params]/[alit]` 行（cfg 本体在 ckpt 内，未单独落盘） |
| `args/stage{N}_status.json`、`args/stage{N}_eval_val.json` | 状态快照 / 训练内验证 |
| `train_logs/stage{N}.jsonl` | 逐步曲线 |
| `run_info.json`、`markers/*`、`fetch_summary.txt` | 元数据 / 状态标记 / 拉取汇总 |

## 5. 评测单独重跑

评测是独立脚本，**任何 ckpt 都能单独评**（不需要训练在跑）：

```bash
cd /root/autodl-tmp/exp448/code/AssocVTC

# 全量 5,000 张（默认 NUM_IMAGES=0），GPU1
CKPT=/root/autodl-tmp/exp448/S3/ckpt_best.pt OUT=/root/autodl-tmp/exp448/eval \
  GPU=1 NUM_IMAGES=0 NAME=stage3_stepwise bash tools/eval_448_960m.sh

# 冒烟（前 200 张）+ 每图每步落盘（E-A1 分析用）
CKPT=.../S3/ckpt_best.pt NUM_IMAGES=200 DUMP_PER_IMAGE=1 \
  bash tools/eval_448_960m.sh
```

- `--num_images 0` = 全量；`>0` 时按等距抽样（不改 val transform，指标口径不变）。
- `BS` 只影响速度/显存，不影响指标；显存不够调小 `BS`（默认 16）。
- 脚本先读 `ckpt["cfg"]` 做 448/p32 守卫（不符则退出码 4；确认要评别的口径 `STRICT=0`）。
- 退出码：0 成功 / 2 参数或 ckpt 问题 / 3 评测失败或 json 未产出 / 4 cfg 不符。

## 6. 拉回本地

```bash
cd /home/linaro/dsh/AssocVTC
PORT=23556 bash tools/fetch_448_960m.sh        # → /home/linaro/dsh/exp448_bench/artifacts/
DRY=1 PORT=23556 bash tools/fetch_448_960m.sh  # 只看会拉什么
```

拉完看 `artifacts/fetch_summary.txt`（pulled/missing + 状态标记）。
磁盘策略：A/B/S3 三个 `ckpt_best` 同时留存 ≈11.4G（`ckpt_last` 每阶段成功后自动删）；
`/root/autodl-tmp` 只剩 ~35G，阶段产物 fetch 回本地后可删远端 `ckpt_last`/`samples/`。

## 7. 预估（**待 bench 实测替换**）

| 阶段 | 240M/224 历史实测 | 本实验 est（待填） | 来源 |
|---|---|---|---|
| A 45ep | ~8.2 h | `TODO(bench)` | 951 步/ep × 45 |
| B 6ep（T=16） | ~8–14 h | `TODO(bench)` | 1,901 步/ep × 6 |
| S3 3ep | ~4 h | `TODO(bench)` | 1,901 步/ep × 3 |
| **合计** | ~20–26 h | **`TODO(bench)`** | 含每 epoch 末 val（A 5,120 张 / B,S3 2,560 张前向） |

拿到 bench 的 s/step 或 img/s 后，ETA 直接读：`grep eta_h logs/stage1.log | tail -1`（训练本体已自报 ETA）。
总墙钟 ≈ A + B + S3 的训练时间（三段评测在 GPU1 并行，通常 <30 min/段，不占墙钟主导项）。

## 8. 失败语义与恢复

| 情况 | 行为 |
|---|---|
| 某阶段 `train_linattn.py` rc≠0 | 写 `FAILED_stage{N}`（含原因），立即退出（码 1/2/3） |
| 某阶段 `ckpt_best.pt` 缺失 | 同上（不会带着空产物进下一阶段） |
| `TIME_BUDGET_H>0` 到点 | 写 `BUDGET_HIT_stage{N}` + `FAILED_stage{N}` 并中止（**不静默续跑半训 ckpt**） |
| 评测失败 / json 未产出 | 写 `FAILED_eval_stage{N}`，退出码 5；训练产物保留，评测可单独重跑 |
| 已完成（`ALL_DONE` 存在） | 启动器/训练链直接跳过（重跑 `FORCE=1` 或删标记） |
| 已有存活 `run.pid` | 启动器拒绝重复启动（先 kill） |
| 训练卡已占用 >5G（如 bencher 还在跑） | 启动器拒绝启动（退出码 6；确认可抢占 `FORCE=1`） |

恢复（断点续跑）——训练链支持按阶段跳过，**不重跑已完成阶段**：

```bash
# A 相已完成、B 相失败 → 修好后只跑 B + S3（评测仍会自动起）
SKIP_STAGE1=1 bash tools/launch_448_960m.sh
# 只想重训 S3（A/B 都在）
SKIP_STAGE1=1 SKIP_STAGE2=1 bash tools/launch_448_960m.sh
```

注意：`SKIP_STAGE*` 只跳过**训练**，前置 ckpt 必须已在（缺失会立刻报错退出）。

## 9. 复现步骤（照抄即可）

```bash
# ── 本地: 同步脚本到远端并校验语法 ───────────────────────────────────────────
cd /home/linaro/dsh/AssocVTC
rsync -av -e 'ssh -p 23556' \
  tools/run_448_960m_3stage.sh tools/launch_448_960m.sh \
  tools/eval_448_960m.sh tools/fetch_448_960m.sh \
  root@connect.nma1.seetacloud.com:/root/autodl-tmp/exp448/code/AssocVTC/tools/

# ── 远端: 语法 + 干跑 ─────────────────────────────────────────────────────
cd /root/autodl-tmp/exp448/code/AssocVTC
bash -n tools/run_448_960m_3stage.sh tools/launch_448_960m.sh tools/eval_448_960m.sh tools/fetch_448_960m.sh
DRY=1 bash tools/launch_448_960m.sh

# ── 远端: 正式启动（bench 结论填进 env）────────────────────────────────────
nohup setsid bash tools/launch_448_960m.sh \
  > /root/autodl-tmp/exp448/logs/launch.out 2>&1 &

# ── 监控 ──────────────────────────────────────────────────────────────
tail -f /root/autodl-tmp/exp448/logs/run.log

# ── 本地: 拉回结果 + 写报告 ──────────────────────────────────────────────
cd /home/linaro/dsh/AssocVTC
PORT=23556 bash tools/fetch_448_960m.sh
# 报告: doc/2026-11-04/REPORT_448_960M_3stage.md
```

## 10. 注意事项（写给 Lead / 复现者）

1. **bench 结论是唯一阻塞项**：脚本默认值（A 16×8 / B 16×4 / ckpt 1）是「保守能跑」，
   不是「最快」；把 bench 实测的 micro-bs/accum/dec_grad_ckpt 填进 env 即可，无需改脚本。
2. **等效批量不能随手改**：A 128 / B 64 决定 951 / 1,901 步每 epoch，与论文 42,795 / 11,406 / 5,703
   同量级（差 ≤0.2%）；偏离会破坏与论文的逐项可比性（脚本默认拒绝，`FORCE_EFF_BS=1` 才放行）。
3. **`--dec_grad_ckpt` 全阶段固定**：换值后新旧 run 不可配对比较（README 记录 0.341 dB 实例级漂移）。
4. **A/B 相不直接与 224 数字比**：A/B 是 1248/2048 维/token 的完整宽度臂；与 224 基线同「值数」
   口径的只有 **S3 相**（7168 值/图 ⇒ 448 值 ≈ 2 token×224、7168 值 ≈ 32 token×224）。
5. **ckpt 体积**：960M ×(fp32 权重 3.8G + AdamW 状态 ≈7.6G)；`--save_opt 0` 不存优化器，
   每阶段只留 `ckpt_best`（3.8G）+ 成功后删 `ckpt_last`；盘紧张时先 fetch 再清远端。
6. **评测与训练并行**：评测在 GPU1 后台跑，**不会**阻塞 GPU0 上的下一阶段；全部评测在
   `ALL_DONE` 之前 `wait`，任何一段评测失败都不会写 `ALL_DONE`。
7. **不 commit**：脚本与本文档由 Lead 统一提交；远端 rsync 副本没有 `.git`，
   `run_info.json` 的 commit 会退化为 `src:<sha256前12位>`（或用 `COMMIT=34638ed` 显式指定）。
