#!/usr/bin/env python3
"""SAVOIAS 复杂度 × BPTT token 预算 热力图（ALIT Fig.4 的 2D 汇总版）
=====================================================================

主图：2 臂 × 3 子集，行 = human complexity 分箱（0–20 / 20–40 / …），
      列 = token 预算（3→64），颜色 = 平均 pixel L1 (0–255)。
附图：逐图热力图（每子集按复杂度排序，行 = 图，列 = 预算）。

用法：
    python3 tools/plot_savoias_heatmap.py \
        --data-dir /tmp/savoias_eval \
        --out-dir  doc/2026-10-19/artifacts/savoias_complexity
输入同 plot_savoias_complexity.py：per_image_*.json + gt_*.npy。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "doc/2026-10-19/artifacts/savoias_complexity"

SUBSETS = [("Art", "People-Art", "gt_art.npy"),
           ("Scenes", "Places", "gt_scenes.npy"),
           ("Objects", "MSCOCO", "gt_objects.npy")]
ARMS = [("stdattn_k64_rec", "Standard Transformer (BPTT)"),
        ("linattn_k64_rec", "Linear-encoder Transformer (BPTT)")]
TOKENS = [3, 8, 15, 24, 35, 48, 63, 64]
EDGES = np.array([0, 25, 50, 75, 100], float)


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


def binned_matrix(S, L, nbins=4):
    """按固定复杂度区间切箱（跨子集/跨臂标签一致）；返回 (均值矩阵, 箱边界, n)。"""
    edges = np.linspace(0, 100, nbins + 1)
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(S, edges) - 1, 0, nbins - 1)
    M = np.full((nbins, L.shape[1]), np.nan)
    N = np.zeros(nbins, int)
    for b in range(nbins):
        m = idx == b
        N[b] = int(m.sum())
        if m.any():
            M[b] = L[m].mean(0)
    return M, edges, N


def bin_labels(edges, N):
    return [f"{edges[b]:.0f}–{edges[b+1]:.0f}" for b in range(len(N))]


def annotate(ax, M, norm, cmap, fmt="{:.1f}", fontsize=7.6):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            r, g, b, _ = cmap(norm(v))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color = "white" if lum < 0.45 else "#1a1a1a"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    out = Path(a.out_dir) if a.out_dir else data_dir
    out.mkdir(parents=True, exist_ok=True)

    scores = {sub: load_scores(data_dir, fn) for sub, _, fn in SUBSETS}
    mats, qss, ns = {}, {}, {}
    for arm, _ in ARMS:
        for sub, _, _ in SUBSETS:
            S, L, _ = join(data_dir, arm, sub, scores[sub])
            M, qs, N = binned_matrix(S, L)
            mats[(arm, sub)], qss[(arm, sub)], ns[(arm, sub)] = M, qs, N

    # 统一色标（跨臂可比）
    allv = np.concatenate([m.ravel() for m in mats.values()])
    vmin, vmax = np.nanmin(allv), np.nanmax(allv)

    # ── 主图：2 臂 × 3 子集 ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3), sharex=True, sharey=True)
    for ri, (arm, arm_label) in enumerate(ARMS):
        for ci, (sub, pretty, _) in enumerate(SUBSETS):
            ax = axes[ri, ci]
            M = mats[(arm, sub)]
            im = ax.imshow(M, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                           origin="upper")
            annotate(ax, M, im.norm, im.cmap)
            ax.set_xticks(range(len(TOKENS)))
            ax.set_xticklabels([str(t) for t in TOKENS], fontsize=8)
            ax.set_yticks(range(M.shape[0]))
            ax.set_yticklabels(bin_labels(qss[(arm, sub)], ns[(arm, sub)]),
                               fontsize=7.4)
            if ri == 0:
                ax.set_title(pretty, fontsize=10.5)
            if ri == 1:
                ax.set_xlabel("cumulative memory tokens read", fontsize=9.5)
            if ci == 0:
                ax.set_ylabel(f"{arm_label}\nhuman complexity", fontsize=9)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("mean pixel L1 (0–255)  →  worse", fontsize=9.5)
    fig.suptitle("More complex images stay worse at every token budget — "
                 "SAVOIAS human complexity × BPTT read budget (OOD for an ImageNet-100 model)",
                 fontsize=10.5, y=0.985)
    fig.savefig(out / "savoias_complexity_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "savoias_complexity_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[plot]", out / "savoias_complexity_heatmap.png")

    # ── 附图：逐图热力图（每子集按复杂度排序）──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharey=False)
    norm = Normalize(vmin=vmin, vmax=vmax)
    for ax, (sub, pretty, _) in zip(axes, SUBSETS):
        S, L, _ = join(data_dir, "stdattn_k64_rec", sub, scores[sub])
        order = np.argsort(S, kind="stable")
        M = L[order]
        im = ax.imshow(M, aspect="auto", cmap="magma", norm=norm,
                       interpolation="nearest")
        ax.set_xticks(range(len(TOKENS)))
        ax.set_xticklabels([str(t) for t in TOKENS], fontsize=8)
        ax.set_title(f"{pretty}  (n={len(S)})", fontsize=10)
        ax.set_xlabel("cumulative memory tokens read", fontsize=9.5)
        if ax is axes[0]:
            ax.set_ylabel("image index (low → high complexity)", fontsize=9)
    cbar.set_label("pixel L1 (0–255)", fontsize=9.5)
    fig.suptitle("Per-image step-wise L1, images sorted by human complexity "
                 "(low → high within each panel; standard Transformer BPTT)",
                 fontsize=10.5, y=0.99)
    fig.subplots_adjust(left=0.06, right=0.885, bottom=0.14, top=0.84, wspace=0.14)
    cax = fig.add_axes([0.915, 0.17, 0.012, 0.66])
    cbar = fig.colorbar(im, cax=cax)
    fig.savefig(out / "savoias_complexity_heatmap_per_image.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(out / "savoias_complexity_heatmap_per_image.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[plot]", out / "savoias_complexity_heatmap_per_image.png")


if __name__ == "__main__":
    main()
