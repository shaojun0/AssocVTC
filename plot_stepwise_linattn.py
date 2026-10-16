"""
画"逐步重建"曲线 — AssocVTC 实验版 (linattn)
==================================================

    python plot_stepwise_linattn.py --json stepwise.json --out stepwise.png \
        [--single-json single_step.json]

左: 每步 0-255 像素 L1（含每图 oracle 最优步曲线与常量灰基线）
右: 质量 vs **累计读入的 z_s token 数**（"质量-传输量"曲线）
`--single-json` 传入单步版（T=1）的结果时, 画成水平参考线, 便于对照"循环 vs 单步"。
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--single-json", default="")
    ap.add_argument("--out", default="stepwise.png")
    ap.add_argument("--title", default="linattn K=64: per-step reconstruction")
    a = ap.parse_args()
    d = json.load(open(a.json))
    T = d["T"]
    L1 = d["L1_255"]
    PSNR = d["PSNR"]
    cum = d["cum_z_s"]
    steps = d["steps"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    xs = list(range(1, T + 1))
    axes[0].plot(xs, L1, "o-", color="tab:blue", lw=1.8, ms=5,
                 label="per-step prediction (0-255 L1)")
    bi = d["best_step"]
    axes[0].plot(xs[bi], L1[bi], "*", ms=18, color="gold", zorder=5,
                 label=f"best step {bi+1} (L1={L1[bi]:.3f})")
    axes[0].axhline(d["const_baseline_255"], ls=":", color="gray",
                    label=f"constant-gray baseline {d['const_baseline_255']:.1f}")
    if a.single_json and os.path.exists(a.single_json):
        s = json.load(open(a.single_json))
        axes[0].axhline(s["L1_255"][0], ls="--", color="tab:red",
                        label=f"single-step (T=1) {s['L1_255'][0]:.3f}")
    axes[0].set_xlabel("decode step")
    axes[0].set_ylabel("pixel L1 (0-255)")
    axes[0].set_xticks(xs)
    axes[0].set_title(f"per-step reconstruction  n={d['n_images']}")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)

    axes[1].plot(cum, L1, "o-", color="tab:green", lw=1.8, ms=5)
    for i, (c, v) in enumerate(zip(cum, L1)):
        axes[1].annotate(f"t{steps[i]}", (c, v), textcoords="offset points",
                         xytext=(4, 6), fontsize=7)
    if a.single_json and os.path.exists(a.single_json):
        s = json.load(open(a.single_json))
        axes[1].axhline(s["L1_255"][0], ls="--", color="tab:red",
                        label=f"single-step (T=1) {s['L1_255'][0]:.3f}")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("cumulative z_s tokens read")
    axes[1].set_ylabel("pixel L1 (0-255)")
    axes[1].set_title("quality vs transmitted tokens")
    axes[1].grid(alpha=.3)

    fig.suptitle(a.title, fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"[plot] {a.out}")


if __name__ == "__main__":
    main()
