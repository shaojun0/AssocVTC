# alit-runner 运行笔记（执行侧记录）— 2026-10-16

> 本文件记录**执行过程中发现的工装坑、口径边界与决策留痕**，供 alit-report 引用。
> 判决（成立/否证/数据不足）不在本文件裁定，见 alit-report 的 `report_tables.md`。

## 1. 两个命令级坑（已实测，建议写入报告"执行注意"）

### 1.1 `train_linattn.py --batch_size` 默认是 **128**，不是 B 协议要求的 64
- 位置：`train_linattn.py` 的 `--batch_size` default=128。
- 实测后果：漏写 `--batch_size 64` 时冒烟实际跑 **bs128 + BPTT T=8** → `torch.OutOfMemoryError`
  （31.4 GiB used / 32 GiB card）。
- 正确做法：所有 B 臂显式 `--batch_size 64`（`run_alit_b.sh` 的 BASE 已固定为 64）。

### 1.2 `--smoke --limit < batch_size` ⇒ 空训练 loader ⇒ epoch 计数器空转
- 成因：训练 loader `drop_last=True`，当 `limit < batch_size` 时一个 batch 都没有。
- 实测：`--smoke --limit 32`（当时 bs=128）跑到 `epoch=453, step=0`，**不报错**，
  约 20 分钟持续空转（每个 epoch 只做一次 val，纯烧 GPU）。
- 规避：`--limit ≥ 2 × batch_size`；建议 train 侧加 `assert steps_per_epoch > 0`（非本 agent 的 write scope）。

### 1.3 修正后的 VRAM 事实（用于后续探针可行性）
- **bs64 + BPTT T=8**：`--dec_grad_ckpt 1` 峰值 **23.3–24.0 GB**；`--dec_grad_ckpt 0` 峰值 **24.0 GB**
  → 两者都能在 32 GB 卡上跑；**此前"关 grad-ckpt 会 OOM"的结论是 bs128 造成的，不成立**。
- b1_mean（bs64/T=8/ckpt=1）训练全过程 VRAM 23.8 GB，126–129 img/s，11406 步 94.9 min。

## 2. 口径边界

### 2.1 历史 rec 臂的 `dec_grad_ckpt` 不可分离
- 历史 `stdattn_k64_rec/args.json`、其 `ckpt_best.pt` 的 `cfg` **都没有** `dec_grad_ckpt`/`dec_attn`/
  `dec_linear_qk_norm` 三个键；git 历史显示 `dec_grad_ckpt` 是**最近一个 commit `da95749` 才引入**的
  （此前模型没有 checkpoint 代码）。
- 因此历史 rec 臂是"无 checkpointing"跑的；新臂统一 `--dec_grad_ckpt 1`。
- b1_mean（同机同协议复跑）**28.409 dB** vs 历史 `stdattn_k64_rec` **28.582 dB**（−0.173 dB）。
  **本报告不宣称复现 28.582**；候选归因 = 单种子重跑漂移 + 上述开关差异（不可分离）。
  参考项目已知重跑漂移口径：`doc/2026-09-23/REPORT_fixw4_plan.md`（同配置重跑终点漂移 0.10/0.30 dB）。
- 收尾探针：`--dec_grad_ckpt 0` 复跑同一臂（bs64，已验证 VRAM 够）用于尝试分离开关效应 vs 重跑漂移，
  排在所有 B 臂之后。

### 2.2 E-C1 探针的可分辨力受限（来自 `task_fidelity_*.json` 的 `honesty`）
- 各固定档 `train_top1 = 0.68–0.77` 而 `val_top1` 仅 **0.049–0.103**（100 类 chance=0.01）⇒ 线性探针严重过拟合；
- val 曲线**非单调**（工具自述："曲线非单调时先看各档 train top-1（欠拟合/过拟合）再谈结论"）；
- 单种子；工具判据要求 `|Δpp| ≥ 0.5 且 CI 不含 0` 才可谈方向。故三臂的 `indistinguishable`
  应表述为"**在该探针可分辨力下测不到差异**"，不能表述为"自适应 = 固定"。
- 分类线性探针 ≠ Phase 2 生成任务（E-C2），不能替代 GOAL §2 口径。

## 3. E-B3 读出集合核对（b3_grow 开跑前确认）
代码路径：`SRPhase1LinAttn.decode()` → `_grow_tokens(patch_feats, n_steps)`，
`n_steps = refiner_steps or T_dec = 8`，`extra_per_step = refiner_grow = 16`，
`mem_extra` 总长度 `M = 8 × 16 = 128`；`_step_mem_with_extra` 第 si 步累计读
`min(16·(si+1), M)` 个生长 token，并接在该步 `z_s` 读窗口之后。

| 步 si | z_s 读窗口累计 | 生长 token 累计 | 该步读出集合 |
|---|---|---|---|
| 0 | 3 | 16 | 19 |
| 1 | 8 | 32 | 40 |
| 2 | 15 | 48 | 63 |
| 3 | 24 | 64 | 88 |
| 4 | 35 | 80 | 115 |
| 5 | 48 | 96 | 144 |
| 6 | 63 | 112 | 175 |
| 7 | 64 | 128 | **192** |

末步读出 **192**（与 Lead 给的端点一致；若把 `z_cls` 计入则为 193）。
S3 冒烟核对：`refiner=GrowthBlock (layers=1 dim=256 grow=16 steps=T)`，params
`refiner=1,049,600`、total 64,104,128（1.05 M）——与 `run_alit_b.sh --arms b3_grow` 的
`--refiner_layers 1 --refiner_grow 16` 完全一致。

## 4. 决策留痕：B 阶段在 E-A 否证侧之后仍继续（**主动决策，非遗漏**）
- E-A 三个关键量都落在 PLAN §8 的否证侧：E-A1 众数 step7 占 **96.98%**（阈值 90%）；
  E-A2 等平均预算下自适应**更差**（+0.2687 / +0.3272 / +0.0363 px，无一行 adaptive_better）；
  E-A3 域间平均停步差 max **0.0247 步**（阈值 0.5 步，且方向非单调）。
- 严格按 PLAN §3「A 阶段给出否证则 B/C 直接停」，B 阶段本可全停。
- Lead 裁定继续，理由（照录）：PLAN §3 的止损条款**只对自适应-K 线成立**（B4/C1 已按其裁剪：
  K′=16/32/48 专训档不做、E-C1 判据按等平均 token 数）；B1/B2/B3 回答的是 README §4.1
  **"让后步真干活"这一独立未决问题**，分别对应 P3 损失加权 / P1 区域损失 / ALIT 记忆增长，
  A 阶段结论不能替它们判决；且用户要求把实验完成，边际成本小。
- 即：继续跑完 B 全矩阵是**主动决策**，不是漏读了 PLAN §3。

## 5. 关键路径与产物
- 服务器：`/root/AssocVTC`（代码与本地控制机 md5 一致）、产物 `/root/AssocVTC/doc/2026-10-16/artifacts/alit/`。
- 控制机镜像：`/home/linaro/dsh/AssocVTC/doc/2026-10-16/artifacts/alit/`（rsync）。
- 运行脚本（本 agent write scope）：`run_alit_a.sh`（E-A，含回归门硬停）、`run_alit_b.sh`
  （E-B，含断点续跑/每臂硬超时/失败隔离/缺开关显式 SKIP）。
