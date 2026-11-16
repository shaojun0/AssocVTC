# REPORT — ImageNet-100 上「并行读出 vs 循环读出」对照（2026-11-16）

> 问题：在同一优化步数、同数据序、同 lr 调度下，把**循环读出**（论文口径，逐档读、
> 带 carry）换成 **并行读出**（v9 口径，16 步折进 batch、无 carry、前缀 mask），
> 重建质量与吞吐各变多少？
> 代码：`model_v2_linattn.py`（`ParallelOutputQueryDecoder`，commit `311c49a`）、
> `train_linattn.py`（`--dec_parallel` / `--stop_at_step`）。
> 原始产物：`doc/2026-11-16/artifacts/in100_par/{baseline,par}/`。

## 0. 一句话

同优化步数 16509 步下，并行读出 **25.5929 dB** vs 循环读出
**24.1217 dB**（**Δ = +1.4712 dB**，L1 -0.0284，SSIM
+0.0716），训练吞吐 **1.077×**。并行读出更好（Δ=+1.471 dB）。

## 1. 协议（唯一自变量 = 读出方式）

| 项 | 基线（循环读出） | 并行读出臂 |
|---|---|---|
| 数据 | ImageNet-100（train 121,689 / val 5,000） | 同 |
| 输入 / patch / N | 224² / 16 / 196 | 同 |
| K（specials）/ 宽度 | 64 / enc 448×13 | 同 |
| 解码器 | 循环 T 步（前缀窗口） / 512×7 | **并行 16 步折 batch** |
| 总步数 / lr 调度 | total_steps = 28,500（cosine，8% warmup） | 同 |
| 停止步 | 16509 | 16509 |
| batch / seed | 128 / 0 | 同 |
| 参数量 | 63.055M | 63.059M（+`step_emb` 4,096） |

## 2. 结果

| 指标 | 循环读出 | 并行读出 | Δ |
|---|---|---|---|
| **val PSNR (dB)** | 24.1217 | 25.5929 | **+1.4712** |
| val L1（归一化） | 0.1907 | 0.1623 | -0.0284 |
| val SSIM | 0.6575 | 0.7291 | +0.0716 |
| 训练内 best_psnr | 24.1217 | 25.5929 | +1.4712 |
| 墙钟（同 16509 步） | 180.24 min | 167.34 min | **1.077×** |
| 吞吐 | 196 img/s | 211 img/s | +7.7% |

参照基线（同 val 集）：mean 基线 11.81 dB、
patch-DC 基线 18.45 dB；两臂的 patch-DC 增益
分别为 5.67 / 7.14 dB。

## 3. 怎么读这个差

- **并行读出更好（Δ=+1.471 dB）**。差异超过 0.3 dB，值得复跑一个 seed 再下结论。
- 并行读出的收益是**吞吐与工程**：16 步折 batch 后没有逐步 carry 依赖，单步大 kernel、
  无串行；在本机实测 196 → 211 img/s（+7.7%），且显存不变（48.7 G）。
- 循环读出的独有价值不是 PSNR，而是**任意时刻截断**（第 t 档 = 前 κ_t 个 token 的可用
  表示）。本对照只比"读满"的质量，不能用来判断预算阶梯的优劣。
- ⚠️ 边界：单 seed、单数据集（ImageNet-100）、224²/patch16 小模型（63M）；PSNR 是像素
  指标，不等于下游任务保真；两臂都停在 16509 步（35% 的调度），**不是**
  收敛后的最终比较。

## 4. 对 v10 的意义

v10（[`doc/2026-11-16/DESIGN_v10_video_whole_clip.md`](DESIGN_v10_video_whole_clip.md)）
选择"每步重出整段 120 帧"的循环读出，**理由不是质量**（本实验说明读出方式本身对
读满质量几乎中性），而是两点结构性需求：

1. 需要**时间前缀阶梯**（停在第 t 档 = 只用了前 4(t+1) 帧的 token）；
2. 需要**后文回改前文**（每步重出全片 ⇒ z_t 可以修正之前的帧）。

如果后续证明这两点换不来收益，那么 v10 可以直接退化成并行读出（更快、更简单），
本实验就是那个退路的实测依据。

## 5. 原始数值

```json
// baseline eval_val.json
{
 "n_images": 5000,
 "l1_norm": 0.1907342192173004,
 "psnr": 24.121747454833983,
 "ssim": 0.6575094149589539,
 "psnr_mean_image_baseline": 11.805153381347656,
 "psnr_patch_dc_baseline": 18.45110532836914,
 "pred_clamped_to_01": true,
 "psnr_gain_over_patch_dc": 5.670642126464841,
 "step": 16509,
 "epoch": 18,
 "split": "val"
}
```

```json
// par eval_val.json
{
 "n_images": 5000,
 "l1_norm": 0.16229217281341551,
 "psnr": 25.59294885864258,
 "ssim": 0.7290959792137146,
 "psnr_mean_image_baseline": 11.805153381347656,
 "psnr_patch_dc_baseline": 18.45110532836914,
 "pred_clamped_to_01": true,
 "psnr_gain_over_patch_dc": 7.141843530273437,
 "step": 16509,
 "epoch": 18,
 "split": "val"
}
```

---

生成脚本：`tools/make_report_in100_par.py`（数据来自两臂的 eval_val/status/args）。
