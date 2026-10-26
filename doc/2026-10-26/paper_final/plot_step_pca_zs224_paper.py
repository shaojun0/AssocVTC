#!/usr/bin/env python
"""论文用图：按读出步数比较 L1 / PSNR（本文 224 维臂 vs 同值数逐块 PCA）。

数据来源（不改动）：
  /home/linaro/dsh/AssocVTC/doc/2026-10-26/artifacts/rfid_m240_pca/zs224_perstep.json
  /home/linaro/dsh/AssocVTC/doc/2026-10-26/artifacts/rfid_m240_pca/pca_allranks.json
与 artifacts 版（fig_step_pca_zs224.pdf）的区别：改为论文排版尺寸（6.6×2.75 in）
并放大字号，使 0.95\\textwidth 嵌入后字号不小于约 7.5 pt。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ART = "/home/linaro/dsh/AssocVTC/doc/2026-10-26/artifacts/rfid_m240_pca"
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 8.0, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

ours = json.load(open(f"{ART}/zs224_perstep.json"))["per_step"]
pca = json.load(open(f"{ART}/pca_allranks.json"))["per_rank"]
pca_r = np.array(sorted(int(k) for k in pca))
pca_l1 = np.array([pca[str(r)]["l1"] for r in pca_r])
pca_ps = np.array([pca[str(r)]["psnr"] for r in pca_r])

steps = np.array([d["step"] for d in ours])
vals = np.array([d["values"] for d in ours])
r_frac = vals / 196.0
pca_l1_t = np.interp(r_frac, pca_r, pca_l1)
pca_ps_t = np.interp(r_frac, pca_r, pca_ps)
our_l1 = np.array([d["l1"] for d in ours])
our_ps = np.array([d["psnr"] for d in ours])

def crossover(x, a, b):
    d = a - b
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            w = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]))
            return x[i] + w * (x[i + 1] - x[i])
    return None

xl1, xps = crossover(steps, our_l1, pca_l1_t), crossover(steps, our_ps, pca_ps_t)
print("L1 cross:", xl1, "PSNR cross:", xps)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))
specs = [("l1", axes[0], "（a）像素 L1（0--255，越低越好）", "L1$_{255}$ / px", our_l1, pca_l1_t,
          "逐块 PCA（同值数，$196r$）", "本文（32 token × 224 维）", xl1, None),
         ("psnr", axes[1], "（b）PSNR（dB，越高越好）", "PSNR / dB", our_ps, pca_ps_t,
          "逐块 PCA（同值数，$196r$）", "本文（32 token × 224 维）", xps, (0.03, 0.97))]
for key, ax, title, ylab, y_our, y_pca, lab_pca, lab_our, xc, apos in specs:
    ax.plot(steps, y_pca, "s--", color="#6f6f6f", ms=3.0, lw=1.1, label=lab_pca)
    ax.plot(steps, y_our, "o-", color="#1f5fa9", ms=3.0, lw=1.3, label=lab_our)
    if xc and steps[0] < xc < steps[-1] and apos is not None:
        ax.axvline(xc, color="#c0392b", ls=":", lw=0.9)
        ax.text(*apos, f"交叉 ≈ 第 {xc:.1f} 步", transform=ax.transAxes,
                color="#c0392b", fontsize=7.0, ha="left", va="top")
    ax.set_title(title, fontsize=8.8, pad=3)
    ax.set_xlabel("读出步数 $t$（第 $t$ 步累计 $448t$ 个值）", fontsize=7.8)
    ax.set_ylabel(ylab, fontsize=7.8)
    ax.set_xticks(range(1, 17))
    ax.tick_params(labelsize=7.0, length=2.5, pad=1.5)
    ax.grid(alpha=0.22, lw=0.4)
    ax.legend(fontsize=6.6, loc="best", frameon=False, handlelength=2.2,
              borderaxespad=0.2, labelspacing=0.3)
fig.tight_layout(pad=0.35)
for ext in ("pdf", "png"):
    fig.savefig(f"fig_step_pca_zs224.{ext}", dpi=300, bbox_inches="tight")
print("saved fig_step_pca_zs224.pdf/.png")
