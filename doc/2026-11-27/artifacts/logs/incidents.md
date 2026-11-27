# 本实验过程中踩到并修掉的问题（MUSA 2.11.4 / torch 2.11）

按影响从大到小排列；每条都给了**现象 → 根因 → 处理**，方便复现时避坑。

## 1. 阶段1 梯度爆炸（100% 梯度被裁剪 / skip 计数暴涨）

- 现象：s1 训到 ~step 2200 时 grad_norm 0.65 → 7.9 → 68 → 252，之后每步 skip；
  val PSNR 从 13.24 掉到 11.16 再掉到 10.77（越训越差）。
- 根因：**MUSA 默认的 fused SDPA backward 在 bf16 下数值不稳**，不是模型/超参问题。
  同机 A/B 诊断（224²，同一 seed）：
  - 默认 backend：skip 76→175，val 13.24→11.16→10.77
  - `SDPA_BACKEND=math`：skip=0，val 13.27→13.97→14.56
- 处理：`export SDPA_BACKEND=math`（只换数值后端，架构/超参与 v2.0 完全一致，wall +6%）。
  中间试过的 `--enc_std_qk_norm 1`（QK-Norm）在 ~step 1700 仍炸（gn 2.37→96.8），已回退，
  **最终配置里没有 QK-Norm**。

## 2. epoch 末 val 触发 MUSA `launch timed out`（11-27 连崩 4 次）

- 现象：s2 每到 epoch 末（step 1901/3802/5703…）必崩，4 个 rank 同时 SIGABRT；
  core 里是 `MUSA_ERROR_LAUNCH_TIMEOUT` / `ContextSwitchTimeout`，
  traceback 全部停在 `train_linattn.py` 的 `dist.barrier()`（epoch 末 val 之后那一行），
  且日志里**从来没有打印过一行 `[val]`**。
- 根因：val 原先只在 rank0 跑（T=16、5000 张 ≈ 3–5 min），其余 3 个 rank 在
  `dist.barrier()` 里干等；**阻塞中的 MCCL barrier kernel 被 MUSA 驱动看门狗判成
  "the launch timed out and was terminated"**。s1（T=1，val 只要 ~10 s）45 个 epoch
  从未触发，正好对上「只有 s2/s3 崩」。
- 处理：val 改成**4 卡同步都跑同一份 5000 张**（冗余但对称，wall 时间不变），
  只有 rank0 打日志/存 ckpt/选 best ⇒ 指标口径与数值逐位不变。修复后 s2/s3 的
  epoch 末 val 全部正常。
- 附带修复：wrapper 重启时若已有 `ckpt_last` 就 `--resume`，避免 chain 一重启
  就把整个阶段从头再训。

## 3. MCCL 偶发崩溃（与 #2 无关）

- 现象：4-rank DDP 训到一半四个 rank 同时 abort，报
  `mcclKernel_AllReduce_RING_LL_Sum_float` + `MUSA_ERROR_LAUNCH_TIMEOUT`。
- 已知触发条件：**同机并发跑别的 GPU 任务**时必现（11-26 我一边跑 s1 一边起单卡
  stage3 smoke，s1 在 12:39 被炸掉）。教训：训练期间绝不在同一台机器上并发跑 GPU 任务。
- 处理：`--save_every_steps 300` 周期存 ckpt_last + wrapper 自动 `--resume`，
  单次崩溃最多丢 15 min。

## 4. 阶段3-CE 的 VQ 目标 id 在 bf16 autocast 下算错

- 现象：训练侧用 `vq.encode_ids(x)` 取离散码，但这段跑在 bf16 autocast 里；
  与 fp32 的 val/P0 口径对比，1568 张里有 91 张（5.8%）id 不一致。
- 处理：取目标码时 `with torch.no_grad(), torch.autocast("cuda", enabled=False)`
  强制 fp32。修复后训练侧与 val 侧目标逐位一致。

## 5. 其它小坑

- `--dec_grad_ckpt 1` 对 `--dec_attn std` 是**空操作**（grad checkpoint 只在线性
  decoder 里实现），s2/s3 显存靠 `--accum_steps` 调，不靠它。
- 续跑（`--resume`）会重置 Adam 一阶/二阶矩（`--save_opt 0` 不存优化器状态），
  且 epoch 末 val 的步点会平移（例如 s2 续跑后 val 落在 7001/8902/10803/11406
  而不是 5703/7604/9505）；总步数、LR 调度、数据量都不变，最终 val 仍是训练真正
  结束那一步的值，与 v2.0 口径可比。
