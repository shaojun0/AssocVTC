# vlm8b 预算课程训练计划（2026-11-06 11:55 用户确认）

> 触发条件：**若 13:00 与 15:00 两次巡检都判 PLATEAU**，则放弃 run2（从 run1/final 续训的那次）
> 的结果，从 0 开始按本课程重训。若任一判定点非 PLATEAU，则继续 run2 原计划。
> 主训练机：`ssh -p 23556 root@connect.nma1.seetacloud.com`（2×A800 80G，ZeRO-2）。

## 1. 课程（5 阶段；前 4 阶段冻结 encoder）

「分 N 步输入」= 预算档位：每步读出看 k 个视觉 token（8 个 token 全给 = k=8）。

| 阶段 | 预算 `--budget-set` | epochs | encoder | 含义 |
|---|---|---|---|---|
| P1 | `8` | 2 | 冻结 | 8 个 token 一次全给（单步读满） |
| P2 | `4,8` | 1 | 冻结 | 分 2 步读出（4 → 8） |
| P3 | `2,4,6,8` | 1 | 冻结 | 分 4 步读出（2 → 8） |
| P4 | `1,2,3,4,5,6,7,8` | 1 | 冻结 | 分 8 步读出（1 → 8） |
| P5 | `1,2,3,4,5,6,7,8` | 2 | **解冻** | 分 8 步 + 适配 encoder |

- 阶段间 **init-from 上一阶段 `final`**（链式续训），每阶段独立进程（换 budget-set 必须换进程）。
- **LR（用户确认「大胆点」）**：LM `3e-5` / proj `3e-4` / encoder `1e-5`（run2 是 1e-5/1e-4/5e-6），
  AdamW(0.9,0.95)、cosine + 3% warmup、bf16、grad accum 8、micro 2 img/GPU、8.93 s/step。
- 总步数 7 epoch × 3803 = **26,621 步 ≈ 65 h**（约 11-09 上午收）。
- 前 4 阶段复用本次修好的 E1 冻结（DeepSpeed 0.19.9 下可用，见
  `doc/2026-11-06/REPORT_ds_e1_freeze.md`，代码在 `feat/vlm8b-v1` tip `0340f40`）。

## 2. 执行

脚本 `tools/run_curriculum_vlm8b.sh`（已同步到远端 `/root/vlm8b/`，`bash -n` + `DRY=1` 均通过）：

```bash
# 0) 必须先同步修复版 train.py（旧版冻结 encoder 会 IndexError）
scp -P 23556 /home/linaro/dsh/dsfix/train.py root@connect.nma1.seetacloud.com:/root/vlm8b/train.py
# 1) 停 run2，确认 GPU 释放
ssh -p 23556 root@connect.nma1.seetacloud.com 'pkill -f "torch.distributed.run"; sleep 20; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
# 2) 启动课程（后台，自带 5 阶段链）
ssh -p 23556 root@connect.nma1.seetacloud.com 'cd /root/vlm8b && nohup bash run_curriculum_vlm8b.sh > logs/cur_main.log 2>&1 &'
```

- 产物：`/root/autodl-tmp/vlm8b/output/cur/P{1..5}/final`；每阶段日志 `logs/cur_P{N}.log`；
  阶段标记 `P{N}/VLM8B_DONE`；总标记 `cur/CURRICULUM_DONE`；失败 `cur/FAILED_P{N}`。
- 每阶段结束自动做 **quick eval**（8 预算全评、`--limit 100 --sample-seed 42`）→ `cur/eval_P{N}/`，
  日志 `logs/cur_eval_P{N}.log`（`EVAL=0` 可关）。
- 断点续跑：重跑同一脚本会自动跳过已有 `VLM8B_DONE` 的阶段；`SKIP_TO=P3` 可指定起点。
- 前置校验：train.py 必须是修复版、无其它训练进程、`/root/autodl-tmp` ≥12G。

## 3. 磁盘策略（/root/autodl-tmp 现 25G 余量）

- 每阶段 final ≈10G；脚本在**下一阶段确认载入权重后删除上一阶段目录**，并在每阶段结束时删 `checkpoint-*`。
- 峰值余量 ≈5G（够，但不要同时写其它大文件）；巡检时 free <4G 要报警。
- `run1/final`（10G）保留不动（用于对照/复评）。

## 4. 判定与回退

- 13:00 / 15:00 两个判定点用 `monitor_loss.py --window 300 --min-steps 600`（run2 在 13:00 约 840 步、
  15:00 约 1,650 步），基线为 run1 终点窗口 CE≈0.511。
- 两次都 PLATEAU → 按 §2 切换；任一非 PLATEAU → 不切换，run2 继续到 11-07 06:00。
- 课程若前 1–2 个阶段 loss 不降或发散（如 P1 CE 明显高于 run1 同期），先汇报再决定是否调 lr/停。
