# DDP 改造 + A 结束后自动接力 — 实现与验证报告

> 日期 2026-11-04 · 负责人 ddp（task-5）· 目标机：AutoDL `autodl-container-119b47bef5-ac805d73`（2×A800 80GB PCIe）
> 仓库：本地 `/home/linaro/dsh/AssocVTC`（= 远程 `/root/autodl-tmp/exp448/code/AssocVTC`）
> 产出根：`/root/autodl-tmp/exp448`；协议见 `DESIGN_448_960M_3stage.md` / `BENCH_448_960M.md` / `RUN_448_960M_3stage.md`

## 0. 一句话结论

**B/S3 两相已改成 2×A800 DDP，A 相（单卡 GPU0 跑到 42,750 步）零改动。** 单卡路径经 20 步对照**逐位不变**
（Δloss 恒等于 0，A 相形态与 B 相形态各一组）；DDP 分支用「同一张卡 2 个 rank」的极小模型冒烟跑通全部
plumbing（`train_linattn.py` 539~738 行）；`tools/run_448_960m_ddp_afterA.sh` 负责在 A 结束后自动接力：
等 A → 真实尺寸 DDP 冒烟测速（<1.3× 回退单卡）→ A 评测 → B DDP → B 评测 → S3 DDP → S3 评测 → `ALL_DONE`。
接力脚本 7 个场景 / 41 项断言在沙箱里全绿（含等待、回退、失败、幂等、A 崩溃拒绝接力）。

**未 commit、未启动接力（按分工由 Lead 启动）、未 kill/干扰 A 相。**

---

## 1. 代码改动（`train_linattn.py`，739 行；仅 DDP 分支，单卡路径逐位不变）

| 位置（行） | 改动 | 说明 |
|---|---|---|
| 34–40 | 导入 `torch.distributed` / `DDP` / `DistributedSampler` / `contextlib` | — |
| 53–58 | `_IS_MAIN` + `_p()`：rank0-only 打印 | 单卡时与 `print` 逐位等价 |
| 61–71 | `ddp_env()`：读 `WORLD_SIZE/RANK/LOCAL_RANK` | 未设置 ⇒ `(1,0,0)`，`ddp=False`，所有 DDP 分支不执行 |
| 428, 440–460 | `main()` 开头建进程组 | `world>1` ⇒ `torch.cuda.set_device(local_rank)`（同卡调试时 `local_rank % device_count`）+ `dist.init_process_group(DDP_BACKEND, 默认 nccl)`；每个 rank 打一行 `[ddp] rank=i/N ... device=cuda:j` |
| 463–471 | 目录/`args.json`/日志只由 rank0 创建/写 | 其余 rank 不落盘 |
| 480–499 | 所有 metadata 打印改 `_p` | 日志不再被两个 rank 交错污染 |
| 501–504 | `raw_model` + `DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True)` | 裸模型用于 `state_dict`/val/采样图（见 §1.1） |
| 506–519 | **等效批量语义**：`micro_bs = batch_size // (accum_steps × world)`；断言可整除 | `--batch_size` 仍是**全局**等效批量；单卡 world=1 时与原式相同 |
| 520–522 | val 张数口径：`val_batches × accum × world` | 保持「val 图数 = batch_size×val_batches」不变（B/S3 = 2560 张，与单卡同） |
| 526–536 | `DistributedSampler(train_ds, shuffle=True, seed=args.seed)` + 重建等价 DataLoader | 每卡只吃自己那份；**不改** `data_linattn.py` |
| 542–555 | **steps/epoch 守卫**：`steps_per_epoch == 单卡参考` 硬断言 + 每 rank 打印 | 等价批量被破坏（步数漂移）会立刻抛错；B/S3 = **1901 步/ep**（见 §4） |
| 561, 578–580, 613–616 | 每 epoch `tr_sampler.set_epoch(epoch)`；`model.no_sync()` 只对累积中间 micro 生效，最后一次 micro 才 allreduce；clip/opt 不变 | 通信量 = 1 次/优化步，而不是 accum 次 |
| 640 | `run_imgs += x.shape[0] * accum * world` | DDP 下 img/s 是全局吞吐（单卡恒等），ETA 正确 |
| 663–668 | 采样图：`_IS_MAIN` + **`raw_model`** 前向 | ⚠️ DDP 包装后的 forward 含 buffer 广播集合通信，单 rank 调用会 hang 死另一个 rank |
| 681–697 | epoch 末 val：**rank0 独占** + 前后 `dist.barrier()`，用 `raw_model` | rank≠0 在 barrier 等；val 不会和 DDP 集合通信互相死等 |
| 712–735 | ckpt/status/`[done]` 只由 rank0 落盘；收尾 `dist.barrier()` + `destroy_process_group()` | 非 rank0 不覆盖 ckpt |

### 1.1 三处「DDP 特有陷阱」及处理（都在代码里注释了）

1. **ckpt 不能被包成 `module.` 前缀**：`save_ckpt` 用 `raw_model.state_dict()`；只有 DDP 时才在 ckpt 里
   加一个 `"ddp": {world_size, rank, backend, micro_bs, accum}` 键 ⇒ **单卡 ckpt 逐字节不变**。
2. **卡在 rank0 独占的 forward 会 hang**：val / 采样图用 `raw_model`（无集合通信），前后 `dist.barrier()`。
3. **日志/status 不能两个 rank 同时写**：每步 `steps/ep` 日志块整体 `_IS_MAIN` 门控
   （否则两个 rank 抢 `status.json.tmp`，`os.replace` 会 ENOENT —— 这是单卡 2-rank 冒烟第一轮真实抓到的 bug）。

### 1.2 单卡路径不变性论证

单卡时 `world=1/rank=0/ddp=False`：DDP 分支全部跳过；`micro_bs`、`val_batches_n`、`run_imgs`、
`last_micro`、`sync_ctx=contextlib.nullcontext()`、`raw_model is model` 全部退化为原表达式；
`_p == print`。⇒ 数据顺序、RNG 消耗、梯度、优化器步数、val 采样、落盘内容均与改造前相同（下面 §2 实测证实）。

---

## 2. 单卡回归证据：**逐位不变（Δ = 0.0）**

方法：远端 GPU1（A 相在 GPU0 不动）用**极小模型**（enc 64×1 / dec 64×1 / patch32 / img448 / K32 /
pixel_hidden 128 / limit 2000 / 20 步 / `--log_interval 1` / `--val_batches 2` / seed 0），
「改造前 = 远端原始文件 sha256 `f536e4a…`」vs「改造后」各跑一遍，逐 step 比对 `train_log.jsonl`：

| 形态 | 配置 | 记录数 | max\|Δloss\| | max\|Δrecon\| | val psnr | 判定 |
|---|---|---|---|---|---|---|
| A 相形态 | micro128 × accum1（T=1） | 20 步 + 2 次 val | **0.000e+00** | **0.000e+00** | 11.647767066955566 两版逐位同 | **bitwise 一致** |
| B 相形态 | micro16 × accum4（T=16, fixed block2） | 20 步 + 1 次 val | **0.000e+00** | **0.000e+00** | 11.757665991783142 两版逐位同 | **bitwise 一致** |

另外：改造前版本自己跑两遍（base vs base2）也是 Δ=0 ⇒ 噪声地板为 0，上面的「Δ=0」是真结论而非运气。
原始文件已备份到远端 `exp448/ddp/reg/train_linattn.orig.py`；证据 `exp448_bench/ddp/regression/`。

---

## 3. 2-rank plumbing 冒烟（**同一张卡**，GPU1，不碰 A）

### 3.1 关键环境发现：同卡多 rank **不能用 NCCL**

实测（torch 2.12.1 / NCCL 2.29.7）：

```
NCCL WARN Duplicate GPU detected : rank 0 and rank 1 both on CUDA device ...
ncclInvalidUsage → 进程组初始化直接失败
```

`NCCL_IGNORE_DUPLICATE_GPU=1` 无效；`CUDA_VISIBLE_DEVICES=1,1` 枚举出 0 张卡。
⇒ **同卡多 rank 只能走 gloo**；实测 gloo 在本机支持 CUDA tensor 的 allreduce（`allreduce=[3.0,3.0]` ✓）。
因此 `train_linattn.py` 加了 `DDP_BACKEND` 环境变量（**默认 nccl**，正式 2 卡训练不受影响），
`tools/ddp_smoke_local2rank.sh` 用它把两个 rank 都放在 GPU1 上跑 plumbing。
（正式 2 卡：两个 rank 各占一张卡，NCCL 正常，不存在 duplicate 问题。）

### 3.2 冒烟结果（`tools/ddp_smoke_local2rank.sh`，17s，KEEP=1 一次 + 清理一次，日志 `ddp_smoke2rank.log`）

配置：enc 64×1 / dec 64×1 / patch32/448 / K32 / T=16 fixed block2 / `--batch_size 64 --accum_steps 2`
⇒ 每卡 micro16 × 每卡 accum2 × 2 卡 = 全局 64（与单卡 micro16×accum4 等价）/ limit 2000 / 20 步 / 带 val + 采样图。

| 检查项 | 结果 |
|---|---|
| 两个 rank 都起进程组 | ✓ 日志两条 `[ddp] rank={0,1}/2 local_rank=... backend=gloo device=cuda:0` |
| DistributedSampler 划分 | ✓ rank0/rank1 各 `num_samples=1000`、`loader_batches=62`、前 5 个索引不同 |
| 覆盖性 | ✓ 两卡并集 = 2000/2000（100.00%），无样本整 epoch 漏看 |
| steps/epoch | ✓ `31 == 单卡参考 31`（代码内硬断言 + 日志） |
| no_sync 梯度累积 | ✓ 20 步成功（每卡 micro16×accum2，最后一次 micro 才 allreduce） |
| rank0 独占落盘 | ✓ `args.json / train_log.jsonl(21 行=20 步+1 val) / status.json(done=true) / eval_val.json / ckpt_best / ckpt_last` 全在；rank1 不写 |
| val 不 hang | ✓ rank0 跑 val（11.75dB），rank1 在 barrier 等，20 步走完 |
| 采样图不 hang | ✓ 4+1 张 png（rank0，裸模型前向） |
| 全程 | ✓ rc=0、无 hang（17s）、退出后无残留进程 |

### 3.3 真实协议步数（纯 CPU 校验，不必等 A 结束）

`data_linattn.py` 全量数据（train 121,689 / val 5,000）上按实际 DataLoader 口径：

| 阶段 | 配置 | 每卡样本 | 每卡 micro-batch | steps/epoch | 协议 |
|---|---|---|---|---|---|
| B/S3 **DDP** | bs64 accum2 world2 | 60,845 | 3,802 | **1,901** | 1,901 ✓ |
| B/S3 单卡回退 | bs64 accum4 world1 | 121,689 | 7,605 | **1,901** | 1,901 ✓ |
| A 单卡（已跑） | bs128 accum1 | 121,689 | 950 | 950（×45 = **42,750**） | 42,795（drop_last 差 0.1%） |

⇒ DDP 与单卡回退**步数完全相同**（11,406 = 1901×6、5,703 = 1901×3），切换不改变协议；
样本覆盖差别仅在「每 epoch 121,689 vs 每卡 60,845×2=121,690（DistributedSampler 垫 1 个重复样本）」，可忽略。

### 3.4 B 相 warm start 预检（纯 CPU，用**真实的 A ckpt**，不等 A 结束都能做）

`A/ckpt_best.pt`（当前 epoch 11 的 best）→ B 相配置（T=16 循环解码）在 CPU 上建 960.27M 模型并
`load_init_from`：

```
missing=['decoder.mem_pos']  unexpected=[]  shape_mismatch=[]
B 相参数: total=960.270M enc=480.537M dec=479.734M values/图=55808
✓ 只缺 decoder.mem_pos（B 相新增的 1×33×2048 = 67,584 参数，零初始化）
```

⇒ B 相热启动链路在接力前已验证（跑失败的最大隐患之一排除）。S3 的 `--zs_dim 224` 热启动是仓库
2026-10-26 已跑通的既有路径（`_init_bottleneck_warm_start` 对旧 `mem_proj` 做 SVD 秩-224 近似），
需等到 B ckpt 产出后才能实测，属接力内部步骤。

---

## 4. 接力驱动 `tools/run_448_960m_ddp_afterA.sh`（532 行）

### 4.1 等待判据（绝不用半训 A 起 B）

每 `WAIT_POLL=30s` 检查三条**同时**成立：
1. `$OUT/A/ckpt_best.pt` 存在；
2. **A 训练进程已退出**（`ps` 匹配 `train_linattn.py` 且 `--output_dir $OUT/A`，精确到路径，不误伤别的训练/冒烟）；
3. `logs/stage1.log` 有 `^\[done\] steps=42750`（42,750 = 950×45）。

- 进程消失但**没有 `[done]`**（崩溃/被 kill）⇒ 写 `FAILED_stage1` 并退出 4，**拒绝接力**。
- `MAX_WAIT_H=30` 超时也写 `FAILED_stage1`。
- ⚠️ 沙箱自测抓到并修掉一个致命坑：原来用 `ps | awk -v pat=…` 传匹配串，**awk 自己的命令行就含
  `train_linattn.py --output_dir …/A`** ⇒ 永远「A 还在跑」，接力永不启动。现改为用 `ENVIRON` 传串 + 排除自身 PID。

### 4.2 真实尺寸 DDP 冒烟 + 安全阀（只有 A 结束后才能做）

- 用 **B 相配置 + 全量数据** 跑 `SMOKE_STEPS=30` 步（`--log_interval 5 --val_batches 0`），输出到
  `$OUT/ddp_smoke_B`（跑完即删，避免 7.7G ckpt 占盘）；顺带验证真实数据下 `steps/epoch=1901`。
- steady `s/step` = `(elapsed_min[30] − elapsed_min[10]) × 60 / 20`（丢掉前 10 步的启动/warmup 影响）；
  `speedup = 4.41 ÷ s/step`（4.41 = bench 单卡 B/S3 实测）。
- **等卡策略**：冒烟前等两张卡都空出了 `SMOKE_WAIT_GPU_MIN=30`（本机是共享实验机，GPU1 可能被别的
  作业占用，例如 15:19 就有别的 agent 的 `gen_qwen.py` 在用 GPU1 21.6G）；B/S3 DDP 开跑前
  `STAGE_WAIT_GPU_MIN=10`；单卡回退 `WAIT_GPU_MIN=5`。等不到空卡才降级。
  ⚠️ 等卡期间没有 train/eval 进程，watchdog 可能在 10 min 后误报一次 `CRASHED`（用 `afterA.heartbeat`
  与 `afterA.out` 的 `[gpu] ... 等显存释放` 行分辨，不是真失败）。
- 判定：`rc≠0` / 两个 rank banner 缺失 / `steps/epoch≠1901` / 拿不到 s/step / speedup `< MIN_SPEEDUP=1.3`
  ⇒ 写 `ALERT_ddp_slow.txt`（或 `ALERT_ddp_smoke.txt`）**回退单卡**跑 B/S3，并写 `DDP_FALLBACK` 标记。
- `FORCE_DDP=1`：冒烟失败时写 `FAILED_ddp_smoke` 并中止（不静默降级）；只是「慢」时仍强制 DDP。
- 决策落 `$OUT/ddp_afterA_decision.txt`（mode / s_step / speedup / 外推 B、S3 小时数 / 失败原因），
  并打印预计墙钟（B = 11406×s/step、S3 = 5703×s/step）。

### 4.3 三阶段命令（协议逐字对齐 run card）

| 阶段 | 命令要点 | 日志 | 标记 |
|---|---|---|---|
| A 训练 | **不重跑**（已单卡跑完），只补 `STAGE1_DONE`、删 `A/ckpt_last`(3.8G) | 只读 `logs/stage1.log` | `STAGE1_DONE` |
| A 评测 | `CKPT=$OUT/A/ckpt_best.pt OUT=$OUT/eval GPU=1 NAME=stage1` | `logs/eval_stage1.log` | → `eval/stage1_stepwise.json` |
| B | `torchrun --nproc_per_node=2 --master_port=<空闲>` `--dec_recurrent 1 --step_plan fixed --block 2 --dec_steps 1..16 --init_from A/ckpt_best --epochs 6 --batch_size 64 --accum_steps 2` | **`logs/stage2.log`** | `STAGE2_DONE`，删 `B/ckpt_last` |
| B 评测 | `NAME=stage2`（GPU1，**串行**：DDP 训练要占两张卡） | `logs/eval_stage2.log` | → `eval/stage2_stepwise.json` |
| S3 | 同上 + `--zs_dim 224 --init_from B/ckpt_best --epochs 3` | **`logs/stage3.log`** | `STAGE3_DONE`，删 `S3/ckpt_last` |
| S3 评测 | `NAME=stage3` | `logs/eval_stage3.log` | → `eval/stage3_stepwise.json` |
| 收尾 | — | — | **`ALL_DONE`** |

注意：`eval_448_960m.sh` 的产物名是 `$OUT/${NAME}_stepwise.json`，所以脚本传 `NAME=stage{N}`
（传 `stage{N}_stepwise` 会得到 `stage{N}_stepwise_stepwise.json` —— 已按 run card 约定修正并由自测覆盖）。

### 4.4 失败语义

| 情况 | 标记 | 退出码 |
|---|---|---|
| A 未正常跑完（无 `[done]` / 无 ckpt / 等待超时） | `FAILED_stage1` | 4 |
| DDP 冒烟失败且 `FORCE_DDP=1` | `FAILED_ddp_smoke` | 5 |
| A 评测失败 | `FAILED_eval_stage1` | 6 |
| B 训练 rc≠0 / 无 ckpt_best / `[done] steps≠11406` / `steps/epoch≠1901` | `FAILED_stage2` | 7 |
| B 评测失败 | `FAILED_eval_stage2` | 8 |
| S3 训练同类失败 | `FAILED_stage3` | 9 |
| S3 评测失败 | `FAILED_eval_stage3` | 10 |

任一失败即中止（不再往下跑），watchdog 的 `FAILED_*` 会报 error。

### 4.5 幂等

- `ALL_DONE` 已存在且 `FORCE≠1` ⇒ 立即退出 0；
- 某阶段「`ckpt_best.pt` 存在 **且** 该阶段日志有 `[done] steps=<协议步数>`」⇒ 跳过训练并补标记；
- 评测 json 已存在 ⇒ 跳过评测（`FORCE=1` 重跑，重跑会在训练前删旧 ckpt_best，避免「失败但旧 ckpt 骗过检查」）；
- `flock $OUT/.ddp_afterA.lock` 单实例（重复启动直接退出 3）；
- 覆盖被 kill 老链路的 stale `run.pid`（写入本驱动 PID）。

### 4.6 与 watchdog 的兼容（不重启 watchdog）

- 日志文件名保持 `logs/stage2.log` / `logs/stage3.log`，标记保持 `STAGE{1,2,3}_DONE` / `ALL_DONE` / `FAILED_*`
  （watchdog `active_stage()` = 第一个没有 `STAGE{n}_DONE` 的阶段；`resolve_log()` = `logs/stage{n}.log`）。
- 老链路被 kill 后没人写过 `STAGE1_DONE` ⇒ 本脚本**在确认 A 跑完的第一时间补写**，watchdog 才会切到 stage2；
  否则它会一直盯着早已静止的 `stage1.log` 误报 STALLED。
- ⚠️ watchdog 的 `CHAIN_PATTERN` 硬编码 `run_448_960m_3stage.sh`，本驱动名字不同 ⇒ watchdog 看不到"链路进程"。
  实际影响很小：本驱动在阶段之间几乎总有 train/eval 进程（间隔秒级），不会触发它的 `NO_PROC_GRACE=600s`；
  但**若驱动本身挂掉**，watchdog 不会报「链路没了」—— 兜底是 `$OUT/afterA.heartbeat`（每 30s 更新）
  + 缺 `STAGE*_DONE`/`ALL_DONE`。**建议 Lead 的定时检查加一条：heartbeat 超过 5 min 不更新即报警。**

---

## 5. 沙箱自测（`exp448_bench/ddp/relay_selftest.sh`）：7 场景 / 41 断言全绿

不占 GPU、不碰真 A：用假 PY（拦截 `-m torch.distributed.run` 与单卡两种调用，写假 `train_log.jsonl`/
`ckpt_best.pt`/`[done]` 行）、假 `eval_448_960m.sh`（按真实命名规则产出 json）、假 A 产物。
结果日志：`exp448_bench/ddp/relay_selftest_result.log`。

| 场景 | 断言 |
|---|---|
| T1 正常 DDP 全流程（假 2.4 s/step ⇒ speedup 1.84×） | 13 项 ✓：rc0 / ALL_DONE / 三 STAGE_DONE / 三 json / B·S3 命令含 torchrun / 决策 mode=ddp / 冒烟产物清理 / ckpt_last 删除 / watchdog 可解析 steps/epoch / `[done] steps=11406` 校验 / heartbeat |
| T2 幂等重跑 | 3 项 ✓：rc0、无重复训练、打印跳过 |
| T3 DDP 太慢（4.0 s/step ⇒ 1.10× < 1.3） | 7 项 ✓：回退单卡（日志无 torchrun、`--accum_steps 4`）、`DDP_FALLBACK`、`ALERT_ddp_slow.txt`、决策 mode=single、仍跑完 ALL_DONE |
| T4 冒烟失败 + FORCE_DDP=1 | 3 项 ✓：rc5、`FAILED_ddp_smoke`、不继续 B/S3 |
| T5 B 训练失败 | 3 项 ✓：rc7、`FAILED_stage2`、不进入 S3 |
| T6 B 评测失败 | 3 项 ✓：rc8、`FAILED_eval_stage2`、不进入 S3 |
| T7a/T7b 等待判据 | 9 项 ✓：A 在跑时真的等（不抢跑）；A 退出但无 `[done]` ⇒ rc4 + `FAILED_stage1` 拒绝接力；A 退出且有 `[done]` ⇒ 正常跑完 |

自测过程中修掉的 3 个真实 bug（都会让长跑崩/永不启动）：
1. `a_train_pids` 用 `awk -v` 传匹配串 ⇒ **awk 自身被匹配**，接力永不启动；
2. 决策文件 `printf` 少续行反斜杠 ⇒ 下一行被当命令执行（`command not found`）；
3. `${!arr[@]}` 不是「数组元素间接展开」⇒ 训练命令丢了全部参数（rc=3）。

---

## 6. 耗时预期与决策

冒烟测到的 s/step 会写进 `$OUT/ddp_afterA_decision.txt`，并外推：
`B_h = 11406 × s/step / 3600`、`S3_h = 5703 × s/step / 3600`。

理论估算（每卡 micro16 约 1.10 s，2 卡/步 = 2 micro + 1 次 allreduce；960M fp32 梯度 allreduce ≈ 3.84 GB/步，
PCIe 4.0 x16 约 0.3–0.4 s）⇒ **约 2.4–2.6 s/step，speedup ≈ 1.7–1.85×**：
- B：≈ **7.6–8.2 h**（单卡 14.0 h）
- S3：≈ **3.8–4.1 h**（单卡 6.9 h）
- 加上 3 次评测（各 ~2.5 min，串行）+ 冒烟 ~3 min ⇒ A 结束后约 **11.8–12.8 h**（若回退单卡则 +9 h）。

> 真实尺寸 s/step 只有 A 结束后才能测（GPU0 现在被占）；`<1.3× ⇒ 自动回退单卡` 就是这个不确定性的安全阀。

---

## 7. 启动方式（**由 Lead 执行；我没有启动**）

```bash
ssh -p 23556 root@connect.nma1.seetacloud.com
cd /root/autodl-tmp/exp448/code/AssocVTC
mkdir -p /root/autodl-tmp/exp448/logs
nohup setsid bash tools/run_448_960m_ddp_afterA.sh \
  > /root/autodl-tmp/exp448/logs/afterA.out 2>&1 < /dev/null &
# 立即自检（不等待）:
DRY=1 bash tools/run_448_960m_ddp_afterA.sh | head -6   # 干跑: 只打印命令
tail -5 /root/autodl-tmp/exp448/logs/afterA.out          # 应看到 driver PID + [wait] 等 A
cat /root/autodl-tmp/exp448/run.pid                      # = 驱动 PID
```

驱动启动后**只读** A 的日志/进程，**不会** kill/重启 A，也不会重启 watchdog。
停止方式：`kill "$(cat /root/autodl-tmp/exp448/run.pid)"`（setsid 已脱离终端；flock 随之释放）。

### A 结束后怎么确认「自动接力」在跑（检查点）

```bash
cd /root/autodl-tmp/exp448
tail -f logs/afterA.out                # 驱动主日志: [wait] ✓ A 相结束 → [smoke] ... → [stage2] ...
cat ddp_afterA_decision.txt            # 冒烟测速 + 策略（ddp / single）
ls logs/stage2.log logs/stage3.log 2>/dev/null
ls STAGE1_DONE STAGE2_DONE STAGE3_DONE ALL_DONE 2>/dev/null
cat status.json                        # watchdog 快照（stage 应变成 2/3）
tail -3 logs/stage2.log                # 每 20 步一行 + 每 epoch 一次 [val]
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # 应两张卡都 ~60G/100%
cat afterA.heartbeat                   # 30s 级心跳（`date +%s`）
```

失败时：`ls FAILED_*` + 对应 `logs/stage{N}.log` / `logs/eval_stage{N}.log` 尾部。

---

## 8. 风险 / 限制（需要 Lead 知道）

1. **同卡 2-rank 不能用 NCCL**（NCCL 2.29.7 duplicate-GPU 硬错误）⇒ 预演 plumbing 只能用 gloo；
   真实 2 卡 DDP 用 NCCL，需等 A 结束后的真实冒烟才第一次验证（这就是冒烟放在接力第 1 步的原因）。
   *若真实冒烟失败/太慢，脚本会自动回退单卡并写 ALERT（除非 FORCE_DDP=1）。*
2. **val 只在 rank0** ⇒ rank≠0 在 barrier 空转；B/S3 每 epoch 一次 val（2560 张 ≈ 70 s），6+3 个 epoch
   共 ~10 min 卡利用率损失，可接受（换来单卡路径逐位不变 + 少一套 val 代码路径）。
3. **评测必须与 DDP 训练串行**（DDP 要占两张卡）：3 次评测 ≈ +8 min，占 A 结束后总时长的 ~1%。
4. **watchdog 看不到本驱动的"链路进程"**（`CHAIN_PATTERN` 硬编码旧脚本名，不能重启 watchdog）⇒
   用 `afterA.heartbeat` 兜底监控（见 §4.6）；其余日志名/标记全部兼容。
5. **ckpt 写入不是原子操作**（沿用原行为：`torch.save` 直接写 `ckpt_best.pt`）⇒ 若在写 3.84G ckpt 时断电/被杀，
   该 ckpt 可能损坏。阶段完成后脚本会校验 `ckpt_best.pt` 存在 + `[done]` 行，但**不校验 ckpt 可读**（读一遍 3.8G 代价不小）。
   若追求更强鲁棒性，可给 train_linattn.py 加 `ckpt_best.pt.tmp → os.replace`（未做，不改动既有行为）。
6. **磁盘**：当前 `/root/autodl-tmp` 剩 50G（今天别人清过旧实验）。接力各阶段峰值占用 ≈ 15.2G（A/B/S3 best 各 3.8G + 当前阶段 last 3.8G），
   冒烟临时 +7.7G（跑完立即删）⇒ 安全。若磁盘被别的实验吃掉，阶段末落 ckpt 会失败并写 `FAILED_stageN`。
7. **`find_unused_parameters=True`**（按任务书要求）在冒烟日志里触发 PyTorch 警告「模型没有 unused 参数」，
   会多一次 autograd 图遍历（约 1–3% 开销）。本轮不优化（保持任务书指定行为）；若真实冒烟 speedup 明显低于理论，
   可再评估改成 `False`。
8. **A 相完成后 `run.pid` 会被本驱动覆盖**（原值 15113 早已死的）。若有其它工具靠 `run.pid` 找老链路，注意语义变化。
9. 单卡回退时 B/S3 的每 epoch 样本数与 DDP 略有差异（121,680 vs 121,689），等效批量/步数完全一致。
10. **GPU1 是共享资源**：验证期间实测到别的 agent 的任务（`gen_qwen.py`）占用 GPU1 21.6G/95%。
    接力脚本对此：DDP 冒烟前最多等 30 min、DDP 阶段前最多等 10 min 空卡，等不到才降级；
    等卡期间 watchdog 可能误报一次 CRASHED（heartbeat 可分辨）。A 结束（≈01:30）时若 GPU1 仍被长作业
    占着，会安全降级为单卡（慢 9h 但不会失败）。

---

## 9. 附：本次改动文件清单（未 commit）

| 文件 | 状态 | 说明 |
|---|---|---|
| `AssocVTC/train_linattn.py` | M（+172/−51 行） | DDP 支持，单卡逐位不变 |
| `AssocVTC/tools/run_448_960m_ddp_afterA.sh` | 新增 | A 结束后自动接力驱动 |
| `AssocVTC/tools/ddp_smoke_local2rank.sh` | 新增 | 同卡 2 rank 极小模型 plumbing 冒烟 |
| `exp448_bench/ddp/relay_selftest.sh` | 新增 | 接力脚本沙箱自测（7 场景 41 断言） |
| `exp448_bench/ddp/relay_selftest_result.log` | 新增 | 自测结果（PASS=41 FAIL=0） |
| `exp448_bench/ddp/ddp_smoke2rank.log` | 新增 | 2 rank 极小模型冒烟完整日志 |
| `exp448_bench/ddp/regression/` | 新增 | 单卡回归 6 份 train_log.jsonl + stdout |
| `exp448_bench/ddp/DDP_448_960M_VERIFICATION.md` | 新增 | 本文件 |

远端同步（sha256 已核对一致，只同步了这三个文件，未覆盖/删除 A 的产物与日志）：
`train_linattn.py`、`tools/run_448_960m_ddp_afterA.sh`、`tools/ddp_smoke_local2rank.sh`。
