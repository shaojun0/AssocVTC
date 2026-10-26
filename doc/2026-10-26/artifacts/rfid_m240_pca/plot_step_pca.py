#!/usr/bin/env python
"""按读出步数画 L1 / PSNR：本文 224 维臂 vs 逐块 PCA（同值数插值）。

数据
----
· 本文：zs224_perstep.json（5,000 张 val，第 t 步累计 2t 个 224 维 token = 448t 个值）
· PCA ：pca_allranks.json（rank 1..40 实测 L1_255/PSNR，每图 196r 个值）
        按 r = 448t/196（非整数）在相邻两个整数 rank 之间线性插值 ⇒ 与本文严格同值数。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

here = __file__.rsplit("/", 1)[0]
ours = json.load(open(f"{here}/zs224_perstep.json"))["per_step"]
pca = json.load(open(f"{here}/pca_allranks.json"))["per_rank"]
pca_r = np.array(sorted(int(k) for k in pca))
pca_l1 = np.array([pca[str(r)]["l1"] for r in pca_r])
pca_ps = np.array([pca[str(r)]["psnr"] for r in pca_r])

steps = np.array([d["step"] for d in ours])
vals = np.array([d["values"] for d in ours])          # 448t
r_frac = vals / 196.0                                  # 与本文同值数的 PCA rank
pca_l1_t = np.interp(r_frac, pca_r, pca_l1)
pca_ps_t = np.interp(r_frac, pca_r, pca_ps)
our_l1 = np.array([d["l1"] for d in ours])
our_ps = np.array([d["psnr"] for d in ours])

print("t  值数  本文L1  PCA_L1   差    本文PSNR PCA_PSNR  差")
for i, t in enumerate(steps):
    print(f"{t:2d} {vals[i]:6d} {our_l1[i]:7.3f} {pca_l1_t[i]:7.3f} "
          f"{our_l1[i]-pca_l1_t[i]:+7.3f}  {our_ps[i]:7.2f} {pca_ps_t[i]:8.2f} "
          f"{our_ps[i]-pca_ps_t[i]:+7.2f}")

def crossover(x, a, b):
    d = a - b
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            w = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]))
            return x[i] + w * (x[i + 1] - x[i])
    return None

xl1 = crossover(steps, our_l1, pca_l1_t)
xps = crossover(steps, our_ps, pca_ps_t)
print("L1 交叉步 ≈", xl1, " PSNR 交叉步 ≈", xps)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
panels = [("l1", axes[0], "（a）像素 L1（0--255，越低越好）", "L1$_{255}$ / px", True),
          ("psnr", axes[1], "（b）PSNR（dB，越高越好）", "PSNR / dB", False)]
for key, ax, title, ylab, lower_better in panels:
    ours_y = our_l1 if key == "l1" else our_ps
    pca_y = pca_l1_t if key == "l1" else pca_ps_t
    ax.plot(steps, pca_y, "s--", color="#777777", ms=4, lw=1.4,
            label="逐块 PCA（同值数，$196r$）")
    ax.plot(steps, ours_y, "o-", color="#1f5fa9", ms=4, lw=1.6,
            label="本文（32 token × 224 维）")
    xc = xl1 if key == "l1" else xps
    if xc and steps[0] < xc < steps[-1]:
        ax.axvline(xc, color="#c0392b", ls=":", lw=1.2)
        ax.annotate(f"交叉 ≈ 第 {xc:.1f} 步", xy=(xc, ax.get_ylim()[1]),
                    xytext=(-52, -14), textcoords="offset points",
                    color="#c0392b", fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("读出步数 $t$（第 $t$ 步累计 $448t$ 个值）", fontsize=8.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_xticks(range(1, 17))
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="best", frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{here}/fig_step_pca_zs224.{ext}", dpi=220, bbox_inches="tight")
print("saved fig_step_pca_zs224.png/.pdf")
