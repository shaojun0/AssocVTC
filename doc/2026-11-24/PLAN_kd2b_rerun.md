# kd2b：修掉 latent 蒸馏的两个 bug 后重跑（2026-11-24）

承接 `doc/2026-11-23/REPORT_kd2_latent_scale_bug.md`。kd2-#1 失败（step8 rFID 113.30，
比 kd1 的 83.63 还差）的原因是两个**我自己的** bug，不是"latent 路线不行"：

| bug | 现象 | 根因 | 状态 |
|---|---|---|---|
| 1 渲染多除 scale | 同 ckpt：13.58 dB/134.27 → 20.84 dB/113.30 | `MSE(z_hat·s, z_t·s)` 的最优解是 `z_hat = z_t`，渲染却又 `/s` ⇒ latent 放大 5.49× | 已修（`8472266`） |
| 2 有效 λ 被暗乘 s² | `kd_mse` 停在 0.31（= 未缩放 MSE 9.3、RMSE 3.05，latent std≈5.5 ⇒ 相对误差 ~55%），z_hat 从没接近 z_t | 两边乘 `0.18215` ⇒ 损失/梯度整体 × `s²=0.0332`，`--kd_weight 1.0` 实际只有 **0.033** | 本次修（`56334f5`） |

**概念更正**：两边同乘一个常数**不会**改变各通道的相对权重（这是 kd2-#1 里我写下的错误理由），
它只把损失/梯度整体缩放 `s²`，效果等价于暗改 λ。ALIT 的 `mul_(0.2325)` 之所以无害，
只因为其 stage-1 **没有别的损失**（`adaptive_vae.py:247-250`），λ 由它自己定。

## 本次改了什麼（`56334f5`）

1. `kd_loss.LATENT_SCALE`、`train_linattn --kd_latent_scale`、`tools/fast_rfid.py`
   默认值 **0.18215 → 1.0**（原始 latent MSE）；`scale≠1` 时训练启动即打印
   `⚠️ 有效 λ = kd_weight·s²`。
2. 回归测试：`tools/test_kd_latent_cpu.py` 第 4 项现在断言
   `MSE(a·s, b·s) = s²·MSE(a, b)`（防再犯），默认值断言改为 1.0。
3. **逐步诊断**（新）：`train_log.jsonl` 里每条日志新增
   `kd_mse_steps`（8 步各自的 MSE）与 `kd_rel_steps`（各步 `√MSE / rms(z_t)`，可读的
   相对误差）。kd2-#1 只有一个跨步平均值，分不清是早期步（384 值）拖累还是末步也压不下去。
4. `tools/run_kd2b_latent.sh`：一条全自动链
   **等 GPU → P0-d 门 → 60 步 smoke（硬门槛）→ 2 epoch → 逐 epoch+best 的 sdvae rFID**。

## 与 kd2-#1 的唯一变量

只有 `--kd_latent_scale 1.0`（其余超参、起点 `s2/ckpt_best.pt`、2 epoch、lr 2e-5、
8 步累积解码、逐步权重 `0.2 0.3 0.5 0.7 0.9 1.0 1.0 1.2`、`--render sdvae` 全部相同）。
等价于把 KD 项相对 kd2-#1 **加强 30.1×**。

## 三重门（跑之前就固定，不做事后解释）

1. **P0-d 门**（不训练，~10 min）：基线 8 步重建图过一遍 `D(E(·))`。
   若 `rFID(real, D(E(ours))) ≪ 86.50`，说明"把输出放回 SD-VAE 流形"本身有大收益，
   latent 路线有天花板支撑；若 ≈86.5，则 latent 路线收益上限很低，kd2 不值得继续。
2. **Smoke 门**（60 步，~6 min）：日志无 NaN、有 `kd_mse` 记录、逐步相对误差可读；
   不通过则 ABORT，不浪费 3h。
3. **预登记（kp2b 主判据）**：`--render sdvae` 的 step8 rFID **60–90**（kd2-#1 为 113.30，
   kd1 为 83.63）。同时看 `kd_rel_steps`：末步相对误差应从 ~55% 掉到 **<20%**。

## 结果怎么解读（事前写清）

- **rFID 60–90 且末步 rel<20%** ⇒ λ bug 是主因，latent 路线成立，下一步查 ratio/步数；
- **rFID 仍 ~110 但 rel 掉到 <20%** ⇒ **z_head 是瓶颈**（它从"像素预测"读 z，
  容量/信息路径不够），不是 λ；下一步把 readout 移到 decoder memory / 让 trunk 直接出
  `(B,4,32,32)`（I2 正式版）；
- **rel 下不去（>40%）** ⇒ 监督本身无法通过 z_head 传导（梯度被 8× 下采样抹平），
  需要改 z_head 结构（更大感受野/上采样回归）；
- **rFID 回到 ~86 且 rel<20%** ⇒ 蒸馏梯度对了但**教师 latent 目标本身**对 FID 帮助有限
  —— 回到"教师解码图 L1 与 FID 解耦"的结论（NOTES §2），该换的是目标（感知/rFID），不是空间。

## 运行记录（自动更新）

- 09:12:59 启动（远端 `/root/autodl-tmp/tok256x12_t8/kd2b`，`driver.log`）；
- P0-d / smoke / 正式训练的时间与结果见 `artifacts/kd2b/status`。
