#!/usr/bin/env python3
"""SAVOIAS 复杂度 × token 预算 热力图 —— 240M 档版（2026-10-26 补图）

与旧图 `图/fig_savoias_heatmap.png`（63M/K=64、8 档预算）同版式，但换成
**240M 档（K=32、T=16、每步 2 token）** 的 16 档预算，数据来自
`per_image_joined.csv` 的 `tag=m240` 行（820 张 SAVOIAS 逐图逐步 L1_255）。

版式：
  · x 轴 = 人评复杂度（15, 25, …, 85，8 档中心，10 分宽）
  · y 轴 = token 预算（2, 4, …, 32，第 t 步累计 2t 个特殊 token）
  · 颜色 = 像素 L1（0–255，不 clamp；沿复杂度轴做 σ=5 的高斯核平滑）
  · 3 面板 = People-Art / Places / MSCOCO；右侧共享绝对色标

用法（仓库根目录）：
    python3 doc/2026-10-26/artifacts/ood_savoias/plot_m240_complexity_heatmap.py
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[4]
CSV = REPO / "doc/2026-10-26/artifacts/ood_savoias/per_image_joined.csv"
OUT = REPO / "doc/2026-10-26/artifacts/ood_savoias/fig_m240_complexity_heatmap"
TAG = "m240"
STEPS = list(range(1, 17))            # 16 档预算，token = 2*step
CENTERS = np.arange(15, 95, 10)       # 15..85
SIGMA = 5.0
SUBSETS = [("Art", "People-Art"), ("Scenes", "Places"), ("Objects", "MSCOCO")]


def set_cn_font():
    for name in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load():
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["tag"] == TAG]
    assert len(rows) == 820, len(rows)
    return rows


def matrix(rows, subset):
    rs = [r for r in rows if r["subset"] == subset]
    S = np.array([float(r["complexity"]) for r in rs])
    L = np.array([[float(r[f"L1_255_t{t}"]) for t in STEPS] for r in rs])  # (n,16)
    M = np.full((len(STEPS), len(CENTERS)), np.nan)
    for j, c in enumerate(CENTERS):
        w = np.exp(-0.5 * ((S - c) / SIGMA) ** 2)
        if w.sum() < 1e-9:
            continue
        w = w / w.sum()
        M[:, j] = (L * w[:, None]).sum(0)
    return M, len(rs)


def main():
    set_cn_font()
    rows = load()
    mats = {}
    for sub, _ in SUBSETS:
        mats[sub], n = matrix(rows, sub)
        print(f"{sub}: n={n}")
    allv = np.concatenate([m.ravel() for m in mats.values()])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    print("color range", round(lo, 2), round(hi, 2))

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 5.0), constrained_layout=True)
    im = None
    for ax, (sub, pretty) in zip(axes, SUBSETS):
        M = mats[sub]
        im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=lo, vmax=hi,
                       origin="lower", interpolation="nearest")
        ax.set_xticks(range(len(CENTERS)))
        ax.set_xticklabels([str(int(c)) for c in CENTERS], fontsize=10)
        ax.set_yticks(range(len(STEPS)))
        ax.set_yticklabels([str(2 * t) for t in STEPS], fontsize=7.6)
        ax.set_title(pretty, fontsize=13)
        ax.set_xlabel("人评复杂度", fontsize=11)
        ax.tick_params(length=3)
    axes[0].set_ylabel("token 预算", fontsize=11)
    cb = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.015)
    cb.set_label("像素 L1", fontsize=11)
    cb.ax.tick_params(labelsize=9)
    fig.savefig(str(OUT.with_suffix(".png")), dpi=300, bbox_inches="tight")
    fig.savefig(str(OUT.with_suffix(".pdf")), bbox_inches="tight")
    print("saved", OUT.with_suffix(".png"), OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
