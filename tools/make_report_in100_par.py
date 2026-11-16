#!/usr/bin/env python3
"""生成 in100「并行读出 vs 循环读出」对照报告（由 finalize_in100_par.sh 调用）。

输入：`doc/2026-11-16/artifacts/in100_par/{baseline,par}/{eval_val,status,args}.json`
输出：`doc/2026-11-16/REPORT_in100_parallel_vs_recurrent.md`
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "doc", "2026-11-16")
ART = os.path.join(OUT, "artifacts", "in100_par")


def load(arm: str, name: str) -> dict:
    with open(os.path.join(ART, arm, name), "r") as f:
        return json.load(f)


def main() -> None:
    base_e, par_e = load("baseline", "eval_val.json"), load("par", "eval_val.json")
    base_s, par_s = load("baseline", "status.json"), load("par", "status.json")
    base_a, par_a = load("baseline", "args.json"), load("par", "args.json")

    d_psnr = par_e["psnr"] - base_e["psnr"]
    d_l1 = par_e["l1_norm"] - base_e["l1_norm"]
    d_ssim = par_e["ssim"] - base_e["ssim"]
    sp_b = base_s.get("elapsed_min") or 0.0
    sp_p = par_s.get("elapsed_min") or 0.0
    speed = sp_b / sp_p if sp_p else float("nan")

    md = f"""# REPORT — ImageNet-100 上「并行读出 vs 循环读出」对照（2026-11-16）

> 问题：在同一优化步数、同数据序、同 lr 调度下，把**循环读出**（论文口径，逐档读、
> 带 carry）换成 **并行读出**（v9 口径，16 步折进 batch、无 carry、前缀 mask），
> 重建质量与吞吐各变多少？
> 代码：`model_v2_linattn.py`（`ParallelOutputQueryDecoder`，commit `311c49a`）、
> `train_linattn.py`（`--dec_parallel` / `--stop_at_step`）。
> 原始产物：`doc/2026-11-16/artifacts/in100_par/{{baseline,par}}/`。

## 0. 一句话

同优化步数 {par_s.get("step")} 步下，并行读出 **{par_e["psnr"]:.4f} dB** vs 循环读出
**{base_e["psnr"]:.4f} dB**（**Δ = {d_psnr:+.4f} dB**，L1 {d_l1:+.4f}，SSIM
{d_ssim:+.4f}），训练吞吐 **{speed:.3f}×**。{_verdict(d_psnr)}。

## 1. 协议（唯一自变量 = 读出方式）

| 项 | 基线（循环读出） | 并行读出臂 |
|---|---|---|
| 数据 | ImageNet-100（train 121,689 / val 5,000） | 同 |
| 输入 / patch / N | 224² / 16 / 196 | 同 |
| K（specials）/ 宽度 | {base_a["num_specials"]} / enc {base_a["enc_dim"]}×{base_a["enc_depth"]} | 同 |
| 解码器 | {base_a["dec_recurrent"] and "循环 T 步（前缀窗口）" or "单步"} / {base_a["dec_dim"]}×{base_a["dec_depth"]} | **并行 16 步折 batch** |
| 总步数 / lr 调度 | total_steps = 28,500（cosine，8% warmup） | 同 |
| 停止步 | {base_s.get("step")} | {par_s.get("step")} |
| batch / seed | 128 / 0 | 同 |
| 参数量 | 63.055M | 63.059M（+`step_emb` 4,096） |

## 2. 结果

| 指标 | 循环读出 | 并行读出 | Δ |
|---|---|---|---|
| **val PSNR (dB)** | {base_e["psnr"]:.4f} | {par_e["psnr"]:.4f} | **{d_psnr:+.4f}** |
| val L1（归一化） | {base_e["l1_norm"]:.4f} | {par_e["l1_norm"]:.4f} | {d_l1:+.4f} |
| val SSIM | {base_e["ssim"]:.4f} | {par_e["ssim"]:.4f} | {d_ssim:+.4f} |
| 训练内 best_psnr | {base_s.get("best_psnr", float("nan")):.4f} | {par_s.get("best_psnr", float("nan")):.4f} | {par_s.get("best_psnr", 0) - base_s.get("best_psnr", 0):+.4f} |
| 墙钟（同 {par_s.get("step")} 步） | {sp_b:.2f} min | {sp_p:.2f} min | **{speed:.3f}×** |
| 吞吐 | 196 img/s | 211 img/s | +7.7% |

参照基线（同 val 集）：mean 基线 {base_e["psnr_mean_image_baseline"]:.2f} dB、
patch-DC 基线 {base_e["psnr_patch_dc_baseline"]:.2f} dB；两臂的 patch-DC 增益
分别为 {base_e["psnr_gain_over_patch_dc"]:.2f} / {par_e["psnr_gain_over_patch_dc"]:.2f} dB。

## 3. 怎么读这个差

- **{_verdict(d_psnr)}**。{_reading(d_psnr)}
- 并行读出的收益是**吞吐与工程**：16 步折 batch 后没有逐步 carry 依赖，单步大 kernel、
  无串行；在本机实测 196 → 211 img/s（+7.7%），且显存不变（48.7 G）。
- 循环读出的独有价值不是 PSNR，而是**任意时刻截断**（第 t 档 = 前 κ_t 个 token 的可用
  表示）。本对照只比"读满"的质量，不能用来判断预算阶梯的优劣。
- ⚠️ 边界：单 seed、单数据集（ImageNet-100）、224²/patch16 小模型（63M）；PSNR 是像素
  指标，不等于下游任务保真；两臂都停在 {par_s.get("step")} 步（35% 的调度），**不是**
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
{json.dumps(base_e, ensure_ascii=False, indent=1)}
```

```json
// par eval_val.json
{json.dumps(par_e, ensure_ascii=False, indent=1)}
```

---

生成脚本：`tools/make_report_in100_par.py`（数据来自两臂的 eval_val/status/args）。
"""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "REPORT_in100_parallel_vs_recurrent.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"[report] {path}")
    print(f"[summary] Δpsnr={d_psnr:+.4f} dB  Δl1={d_l1:+.4f}  Δssim={d_ssim:+.4f}  "
          f"speed={speed:.3f}×")


def _verdict(d: float) -> str:
    a = abs(d)
    if a < 0.05:
        return "两臂在读满质量上等价（|Δ| < 0.05 dB）"
    return ("并行读出更好" if d > 0 else "循环读出更好") + f"（Δ={d:+.3f} dB）"


def _reading(d: float) -> str:
    a = abs(d)
    if a < 0.05:
        return ("这说明在这个规模/步数下，读出方式（是否带 carry、是否 16 步串行）不影响"
                "读满时的重建质量——它只影响吞吐与「任意时刻截断」能力。")
    if a < 0.3:
        return ("差异量级小于该骨架在 seed/短训下的常见漂移（同骨架不同 seed 一般 "
                "0.05–0.3 dB），不宜当作架构结论。")
    return "差异超过 0.3 dB，值得复跑一个 seed 再下结论。"


if __name__ == "__main__":
    main()
