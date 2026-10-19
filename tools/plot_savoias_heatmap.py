#!/usr/bin/env python3
"""SAVOIAS 复杂度 × token 数 热力图 —— 与 ALIT Fig.4/11/12 的版式对齐
=====================================================================

对齐点（按原论文）：
  · x 轴 = **Image Complexity Human Score**，刻度 15, 25, …, 85（8 档，10 分宽）
  · y 轴 = **Number of Tokens**，用**本仓库自己的** token 预算 [3,8,15,24,35,48,63,64]
  · 颜色条 = **从 ALIT Fig.4 的 colorbar 上采样**的 YlOrRd 锚点（见 ALIT_YlOrRd）
  · 无格内数字（原图无）；颜色条竖直放在右侧

输入：per_image_*.json（`eval_stepwise_linattn.py --dump_per_image`）+ gt_*.npy
输出：
  savoias_complexity_heatmap.png/pdf            2 臂 × 3 子集（主图，平滑版）
  savoias_complexity_heatmap_raw.png/pdf        同上，但用原始分箱均值（不光滑）
  savoias_complexity_heatmap_per_image.png/pdf  逐图热力图（按复杂度排序）

用法：
    python3 tools/plot_savoias_heatmap.py --data-dir <dir> --out-dir <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "doc/2026-10-19/artifacts/savoias_complexity"

SUBSETS = [("Art", "People-Art", "gt_art.npy"),
           ("Scenes", "Places", "gt_scenes.npy"),
           ("Objects", "MSCOCO", "gt_objects.npy")]
ARMS = [("stdattn_k64_rec", "Standard Transformer (BPTT)"),
        ("linattn_k64_rec", "Linear-encoder Transformer (BPTT)")]
TOKENS = [3, 8, 15, 24, 35, 48, 63, 64]          # y 轴（我们的读出预算）
CENTERS = np.arange(15, 95, 10)                   # x 轴 15..85（8 档）

# 从 ALIT Fig.4 (arXiv:2411.02393) 的 colorbar 上直接采样的 17 个锚点（bottom→top）。
# 与 matplotlib 的 "YlOrRd" 几乎逐位一致（mean|err| ≈ 0.6/255），这里用采样值以严格“取色”。
ALIT_YlOrRd = [
    (255, 254, 203), (255, 245, 181), (255, 236, 159), (254, 227, 139),
    (254, 217, 118), (254, 197, 96),  (254, 178, 76),  (253, 159, 68),
    (253, 142, 60),  (253, 110, 51),  (252, 79, 42),   (240, 53, 35),
    (227, 26, 28),   (208, 13, 33),   (190, 1, 38),    (161, 0, 38),
    (130, 0, 38),
]


def alit_cmap():
    return LinearSegmentedColormap.from_list(
        "alit_ylorrd", [tuple(c / 255 for c in rgb) for rgb in ALIT_YlOrRd])


def load_scores(data_dir, fn):
    o = np.load(data_dir / fn, allow_pickle=True).item()
    return {int(k): int(o[k][1]) for k in o}


def join(data_dir, arm, subset, scores):
    aj = json.load(open(data_dir / f"per_image_{arm}.json"))
    S, L, IDX = [], [], []
    for r in aj["images"]:
        if r["category"] != subset:
            continue
        stem = int(r["file_name"].split(".")[0])
        if stem not in scores:
            continue
        S.append(scores[stem]); L.append(r["L1_255"]); IDX.append(stem)
    return np.asarray(S, float), np.asarray(L, float), np.asarray(IDX, int)


def matrix(S, L, h):
    """返回 (len(TOKENS), len(CENTERS)): 每个 (token, complexity 中心) 的均值。

    h>0: 沿复杂度轴做 Gaussian 核平滑（原论文的曲面是光滑的；σ 以分数为单位）；
    h=0: 10 分宽固定分箱的普通均值。
    """
    M = np.full((len(TOKENS), len(CENTERS)), np.nan)
    for j, c in enumerate(CENTERS):
        if h > 0:
            w = np.exp(-0.5 * ((S - c) / h) ** 2)
        else:
            w = ((S >= c - 5) & (S < c + 5)).astype(float)
        if w.sum() < 1e-9:
            continue
        w = w / w.sum()
        M[:, j] = (L * w[:, None]).sum(0)
    return M


def make_grid_figure(mats, out, h, cmap, mode="relative"):
    """mode='relative': 全局 min-max 归一化到 0–1（对齐 ALIT 的 0–1 colorbar）；
    mode='absolute': 直接用 pixel L1（0–255）色标。"""
    allv = np.concatenate([m.ravel() for m in mats.values()])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    if mode == "relative":
        def transform(m):
            return (m - lo) / (hi - lo + 1e-12)
        norm = Normalize(0.0, 1.0)
    else:
        def transform(m):
            return m
        norm = Normalize(lo, hi)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.6))
    for ri, (arm, arm_label) in enumerate(ARMS):
        for ci, (sub, pretty, _) in enumerate(SUBSETS):
            ax = axes[ri, ci]
            M = transform(mats[(arm, sub)])
            ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, origin="lower",
                      interpolation="nearest")
            ax.set_xticks(range(len(CENTERS)))
            ax.set_xticklabels([str(int(c)) for c in CENTERS], fontsize=9)
            ax.set_yticks(range(len(TOKENS)))
            ax.set_yticklabels([str(t) for t in TOKENS], fontsize=8.5)
            ax.tick_params(length=3)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
            if ri == 0:
                ax.set_title(pretty, fontsize=11.5, pad=6)
            if ri == 1:
                ax.set_xlabel("Image Complexity Human Score", fontsize=10.5)
            if ci == 0:
                ax.set_ylabel(f"{arm_label}\nNumber of Tokens", fontsize=9.5)
    # 竖直共享 colorbar（版式同原论文）
    fig.subplots_adjust(left=0.115, right=0.885, bottom=0.14, top=0.90,
                        wspace=0.14, hspace=0.22)
    cax = fig.add_axes([0.905, 0.16, 0.016, 0.70])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    if mode == "relative":
        cbar.set_ticks(np.linspace(0, 1, 6))
        cbar.set_label("L1 Reconstruction Loss  (normalized 0–1)  →  worse",
                       fontsize=9.5)
        sub = (f"normalized per figure to [0,1]; absolute pixel L1 range "
               f"{lo:.1f}–{hi:.1f} / 255")
    else:
        cbar.set_label("L1 Reconstruction Loss  (pixel L1, 0–255)  →  worse",
                       fontsize=9.5)
        sub = "absolute pixel L1 (0–255)"
    kind = (f"Gaussian-smoothed, σ={h:g}" if h > 0 else "raw 10-point bins")
    fig.suptitle("Image complexity vs. number of tokens — SAVOIAS human scores, "
                 f"BPTT reconstruction ({kind})", fontsize=10.5, y=0.975)
    fig.text(0.01, 0.012,
             f"{sub}. Colormap sampled from ALIT Fig. 4 (YlOrRd). "
             "Model trained on ImageNet-100; SAVOIAS is out-of-distribution. "
             "Lower is better.",
             fontsize=7.2, color="#555")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[plot]", out)


def make_per_image_figure(data_dir, scores, out, cmap):
    # 仍用 std 臂；行 = 图（按复杂度排序），列 = token
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2))
    allv = []
    data = []
    for sub, pretty, _ in SUBSETS:
        S, L, _ = join(data_dir, "stdattn_k64_rec", sub, scores[sub])
        order = np.argsort(S, kind="stable")
        data.append((sub, pretty, S, L[order]))
        allv.append(L.ravel())
    allv = np.concatenate(allv)
    # 稳健色标（2%–98% 分位），避免少数极端图把色阶拉爆；超出范围的颜色截断
    vmin, vmax = np.percentile(allv, [2, 98])
    norm = Normalize(float(vmin), float(vmax), clip=True)
    for ax, (sub, pretty, S, M) in zip(axes, data):
        ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(TOKENS)))
        ax.set_xticklabels([str(t) for t in TOKENS], fontsize=8)
        ax.set_title(f"{pretty}  (n={len(S)})", fontsize=10.5)
        ax.set_xlabel("Number of Tokens", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("image index  (low → high complexity)", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.14, top=0.84, wspace=0.14)
    cax = fig.add_axes([0.905, 0.17, 0.012, 0.66])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("L1 Reconstruction Loss (pixel L1, 0–255)", fontsize=9.5)
    fig.suptitle("Per-image L1 vs. number of tokens, sorted by SAVOIAS human complexity "
                 "(standard Transformer BPTT)", fontsize=10.5, y=0.97)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[plot]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smooth-h", type=float, default=5.0,
                    help="复杂度轴 Gaussian 平滑带宽（分数单位）；0 = 原始 10 分箱")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    out = Path(a.out_dir) if a.out_dir else data_dir
    out.mkdir(parents=True, exist_ok=True)

    scores = {sub: load_scores(data_dir, fn) for sub, _, fn in SUBSETS}
    cmap = alit_cmap()

    mats = {}
    for arm, _ in ARMS:
        for sub, _, _ in SUBSETS:
            S, L, _ = join(data_dir, arm, sub, scores[sub])
            mats[(arm, sub)] = matrix(S, L, a.smooth_h)
    # 主图：0–1 归一化 colorbar（对齐 ALIT Fig.4 的 0.0–1.0 色标）
    make_grid_figure(mats, out / "savoias_complexity_heatmap.png",
                     a.smooth_h, cmap, mode="relative")
    # 绝对 pixel L1（0–255）版本
    make_grid_figure(mats, out / "savoias_complexity_heatmap_abs.png",
                     a.smooth_h, cmap, mode="absolute")

    # 原始分箱版（透明性对照；不做平滑）
    mats_raw = {}
    for arm, _ in ARMS:
        for sub, _, _ in SUBSETS:
            S, L, _ = join(data_dir, arm, sub, scores[sub])
            mats_raw[(arm, sub)] = matrix(S, L, 0.0)
    make_grid_figure(mats_raw, out / "savoias_complexity_heatmap_raw.png",
                     0.0, cmap, mode="absolute")

    make_per_image_figure(data_dir, scores,
                          out / "savoias_complexity_heatmap_per_image.png", cmap)


if __name__ == "__main__":
    main()
