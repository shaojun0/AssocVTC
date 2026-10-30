#!/usr/bin/env python3
"""qwen_pca_figs.py — Qwen 视觉 token 池化 PCA 结果图（png+pdf，200 dpi，英文标签）。

输入（--out_dir，即 qwen_pca_tokens.py 的产物目录）:
  eigvals.npy / cumvar.npy / pca_stats.json / arms.json
  per_position_stats.json / per_position_eigvals.npz（--per_position 才有）
  poscentered_eigvals.npy / poscentered_cumvar.npy / pca_stats_poscentered.json（可选）

输出（--fig_dir，默认 = --out_dir）:
  fig_qwen_pca_spectrum.png / .pdf          谱 + 累计方差（标 1σ/2σ/3σ）
  fig_qwen_pca_perposition.png / .pdf       49 位置 r@1σ/2σ/3σ + 方差占比 + 7x7 heatmap
  fig_qwen_pca_spectrum_poscentered.png/.pdf  主口径 vs 位置中心化口径（可选）

用法:
  python tools/qwen_pca_figs.py --out_dir /root/autodl-tmp/qwen_vtc/pca
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUT = "/root/autodl-tmp/qwen_vtc/pca"
SIGMA_TARGETS = [("1sigma", 0.6827), ("2sigma", 0.9545), ("3sigma", 0.9973)]
COLORS = {"1sigma": "C1", "2sigma": "C3", "3sigma": "C5"}


def save(fig, fig_dir, name, dpi=200):
    out = []
    for ext in ("png", "pdf"):
        p = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    print("[fig] saved", ", ".join(out), flush=True)
    return out


def load(out_dir):
    d = {
        "ev": np.load(os.path.join(out_dir, "eigvals.npy")),
        "cum": np.load(os.path.join(out_dir, "cumvar.npy")),
        "stats": json.load(open(os.path.join(out_dir, "pca_stats.json"))),
    }
    ap = os.path.join(out_dir, "arms.json")
    d["arms"] = json.load(open(ap)) if os.path.exists(ap) else None
    p = os.path.join(out_dir, "per_position_stats.json")
    if os.path.exists(p):
        d["pp"] = json.load(open(p))
        npz = os.path.join(out_dir, "per_position_eigvals.npz")
        d["pp_ev"] = np.load(npz)["eigvals"] if os.path.exists(npz) else None
    p = os.path.join(out_dir, "poscentered_eigvals.npy")
    if os.path.exists(p):
        d["ev_pc"] = np.load(p)
        d["cum_pc"] = np.load(os.path.join(out_dir, "poscentered_cumvar.npy"))
        d["pc_stats"] = json.load(open(os.path.join(out_dir, "pca_stats_poscentered.json")))
    return d


def r_of(cum, t):
    return int(np.searchsorted(cum, t) + 1)


def fig_spectrum(d, fig_dir, dpi):
    ev, cum = d["ev"], d["cum"]
    n = len(ev)
    i = np.arange(1, n + 1)
    ev_plot = np.clip(ev, ev[0] * 1e-12, None)
    cum_at = {k: float(cum[r_of(cum, t) - 1]) for k, t in SIGMA_TARGETS}
    rs = {k: r_of(cum, t) for k, t in SIGMA_TARGETS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.1))
    ax1.loglog(i, ev_plot, lw=1.3, color="C0", label="eigenvalues $\\lambda_i$")
    ax1.axhline(float(ev.mean()), color="0.4", ls=":", lw=1.2,
                label=f"mean $\\lambda$ = {float(ev.mean()):.3g}")
    pr = d["stats"]["participation_ratio"]
    ax1.axvline(pr, color="0.6", ls="-.", lw=1.0, label=f"participation ratio = {pr:.1f}")
    for (k, t), c in zip(SIGMA_TARGETS, ("C1", "C3", "C5")):
        ax1.axvline(rs[k], color=c, ls="--", lw=1.2, label=f"{k} (r={rs[k]})")
    ax1.set_xlabel("principal component index $i$")
    ax1.set_ylabel("eigenvalue $\\lambda_i$")
    ax1.set_title("Qwen vision tokens: pooled PCA spectrum (IN-100 train)")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(fontsize=8, loc="lower left")

    ax2.plot(i, cum, lw=1.6, color="C0")
    ax2.axhline(1.0, color="0.7", lw=0.8)
    offs = {"1sigma": (10, -24), "2sigma": (8, -48), "3sigma": (-10, -30)}
    for (k, t), c in zip(SIGMA_TARGETS, ("C1", "C3", "C5")):
        ax2.axvline(rs[k], color=c, ls="--", lw=1.3)
        ax2.plot([rs[k]], [cum_at[k]], "o", color=c, ms=5)
        dx, dy = offs[k]
        ax2.annotate(f"{k}: r={rs[k]}\n(cum={cum_at[k]*100:.2f}%)",
                     xy=(rs[k], cum_at[k]), xytext=(dx, dy), textcoords="offset points",
                     color=c, fontsize=8.5, ha=("right" if dx < 0 else "left"))
    ax2.axhline(0.6827, color="C1", ls=":", lw=0.9)
    ax2.axhline(0.9545, color="C3", ls=":", lw=0.9)
    ax2.axhline(0.9973, color="C5", ls=":", lw=0.9)
    ax2.set_xlim(0, n)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("number of retained PCs  $r$")
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_title("Cumulative variance vs r (linear x)")
    ax2.grid(alpha=0.25)
    # 放大 [0, 2.5*r3]
    xmax = max(int(2.5 * rs["3sigma"]), 100)
    ax2.axvspan(0, min(xmax, n), color="C0", alpha=0.05)
    ins = ax2.inset_axes([0.42, 0.10, 0.55, 0.42])
    m = i <= xmax
    ins.plot(i[m], cum[m], lw=1.4, color="C0")
    for (k, t), c in zip(SIGMA_TARGETS, ("C1", "C3", "C5")):
        ins.axvline(rs[k], color=c, ls="--", lw=1.0)
    ins.set_xlim(0, min(xmax, n))
    ins.grid(alpha=0.25)
    ins.tick_params(labelsize=7)
    ins.set_title(f"zoom: r <= {min(xmax, n)}", fontsize=8)
    n_tok = int(d["stats"].get("n_tokens", 0))
    ntxt = f"{n_tok/1e6:.2f}M" if n_tok >= 1e6 else f"{n_tok/1e3:.0f}k"
    fig.suptitle(f"σ = Gaussian cumulative mass (1σ/2σ/3σ = 68.27%/95.45%/99.73% var); "
                 f"pooled over {ntxt} tokens · dim={d['stats'].get('dim', 5120)} · "
                 f"top1 share={d['stats']['top1_share']*100:.1f}%", fontsize=10, y=1.02)
    return save(fig, fig_dir, "fig_qwen_pca_spectrum", dpi)


def fig_perposition(d, fig_dir, dpi):
    pp = d["pp"]
    npos = int(pp["n_positions"])
    pos = np.arange(npos)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.7))

    ax = axes[0]
    for (k, t), c in zip(SIGMA_TARGETS, ("C1", "C3", "C5")):
        y = [pp["dims_at_thresholds"][str(p)][k] for p in range(npos)]
        ax.plot(pos, y, marker=".", ms=4, lw=1.2, color=c, label=k)
    ax.set_xlabel("merged token position (row-major 7x7 grid)")
    ax.set_ylabel("number of PCs  r")
    ax.set_title("Per-position r at 1σ/2σ/3σ")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    share = np.array([pp["variance_share_within_sum1"][str(p)] for p in range(npos)])
    ax.bar(pos, share * 100, color="C0", width=0.85)
    ax.axhline(100.0 / npos, color="C3", ls="--", lw=1.1,
               label=f"uniform = {100.0/npos:.2f}%")
    ax.set_xlabel("merged token position")
    ax.set_ylabel("share of within-position variance (%)")
    ax.set_title("Per-position variance share (Σ=100%)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    ax = axes[2]
    if d.get("pp_ev") is not None:
        r2 = np.array([pp["dims_at_thresholds"][str(p)]["2sigma"] for p in range(npos)])
    elif "dims_at_thresholds" in pp:
        r2 = np.array([pp["dims_at_thresholds"][str(p)]["2sigma"] for p in range(npos)])
    else:
        r2 = None
    if r2 is not None and int(round(np.sqrt(npos))) ** 2 == npos:
        g = int(round(np.sqrt(npos)))
        H = r2.reshape(g, g)
        im = ax.imshow(H, cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        thr = (H.min() + H.max()) / 2.0
        for r_ in range(g):
            for c_ in range(g):
                ax.text(c_, r_, f"{H[r_, c_]}", ha="center", va="center",
                        fontsize=6.5, color=("k" if H[r_, c_] > thr else "w"))
        ax.set_title("r@2σ heatmap (7x7 positions)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="r@2σ")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no per_position data", ha="center", va="center")
    fig.suptitle("Qwen vision tokens: per-position (49 positions) PCA, "
                 "same-token-position statistics across IN-100 train images", fontsize=10, y=1.02)
    return save(fig, fig_dir, "fig_qwen_pca_perposition", dpi)


def fig_poscentered(d, fig_dir, dpi):
    ev, cum = d["ev"], d["cum"]
    ev_pc, cum_pc = d["ev_pc"], d["cum_pc"]
    n = len(ev)
    i = np.arange(1, n + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.1))
    ax1.loglog(i, np.clip(ev, ev[0] * 1e-12, None), lw=1.3, color="C0",
               label="global-centred pooled (main)")
    ax1.loglog(i, np.clip(ev_pc, ev_pc[0] * 1e-12, None), lw=1.3, color="C2",
               label="position-mean subtracted (secondary)")
    for arr, cum_, col, tag in ((ev, cum, "C0", "main"), (ev_pc, cum_pc, "C2", "poscent")):
        for (k, t) in SIGMA_TARGETS:
            r = r_of(cum_, t)
            ax1.axvline(r, color=col, ls="--", lw=0.8, alpha=0.6)
    ax1.set_xlabel("principal component index $i$")
    ax1.set_ylabel("eigenvalue $\\lambda_i$")
    ax1.set_title("Spectrum: global-centred vs position-centred pooled PCA")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(fontsize=8, loc="lower left")

    ax2.plot(i, cum, lw=1.5, color="C0", label="main (global centred)")
    ax2.plot(i, cum_pc, lw=1.5, color="C2", label="position-mean subtracted")
    for (k, t) in SIGMA_TARGETS:
        ax2.axvline(r_of(cum, t), color="C0", ls=":", lw=0.9)
        ax2.axvline(r_of(cum_pc, t), color="C2", ls=":", lw=0.9)
    ax2.axhline(0.6827, color="0.6", ls=":", lw=0.8)
    ax2.axhline(0.9545, color="0.6", ls=":", lw=0.8)
    ax2.axhline(0.9973, color="0.6", ls=":", lw=0.8)
    ax2.set_xlim(0, n)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("number of retained PCs  $r$")
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_title("Cumulative variance: main vs position-centred")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="lower right")
    cs = d["pc_stats"].get("vs_main", {})
    txt = "\n".join(f"{k}: r {v['main_r']} → {v['poscentered_r']}" for k, v in cs.items()
                    if k in ("1sigma", "2sigma", "3sigma"))
    fig.suptitle("Position-centred pooled PCA (secondary): same tokens, per-position mean removed",
                 fontsize=10, y=1.02)
    ax2.text(0.02, 0.55, txt, transform=ax2.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.85))
    return save(fig, fig_dir, "fig_qwen_pca_spectrum_poscentered", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--fig_dir", default="")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    fig_dir = args.fig_dir or args.out_dir
    os.makedirs(fig_dir, exist_ok=True)
    d = load(args.out_dir)
    print(f"[fig] eigvals n={len(d['ev'])} top1={d['stats']['top1_share']:.4f} "
          f"PR={d['stats']['participation_ratio']:.2f}", flush=True)
    fig_spectrum(d, fig_dir, args.dpi)
    if "pp" in d:
        fig_perposition(d, fig_dir, args.dpi)
    if "ev_pc" in d:
        fig_poscentered(d, fig_dir, args.dpi)
    print("[fig] DONE →", fig_dir, flush=True)


if __name__ == "__main__":
    main()
