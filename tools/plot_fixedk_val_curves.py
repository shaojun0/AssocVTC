#!/usr/bin/env python3
"""plot_fixedk_val_curves.py — 从训练日志画 A/B 相验证曲线（每 epoch 的 val PSNR）。

用途: 展示「固定 n 个 z_s」5 臂（以及同机 cumulative 锚臂）在各 token 预算 K 下的
      训练稳定性——尤其是 K≥128 时 A 相（T=1）收敛慢/不稳定的证据。

输入: 日志目录下 `k<K>_stage{1,2}.log`（train_linattn.py 的 stdout）
输出: <out_dir>/fixedk_val_curves.png/.pdf

用法:
    python tools/plot_fixedk_val_curves.py --logs_dir doc/2026-10-21/artifacts/fixedk/logs \
        --out_dir doc/2026-10-21/artifacts/fixedk --ks 64 96 128 192 256
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_vals(path):
    eps, psnrs = [], []
    if not os.path.isfile(path):
        return eps, psnrs
    pat = re.compile(r"\[val\] ep(\d+).*?psnr=([0-9.]+)dB")
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            eps.append(int(m.group(1)))
            psnrs.append(float(m.group(2)))
    return eps, psnrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=[64, 96, 128, 192, 256])
    ap.add_argument("--anchor_ks", type=int, nargs="*", default=[],
                    help="cumulative 锚臂的 K（日志前缀 cum_k<K>_stage*.log）")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, stage, title in ((axes[0], 1, "A phase (T=1, 45 ep)"),
                             (axes[1], 2, "B phase (T=8 BPTT, 6 ep)")):
        allk = list(args.ks) + list(args.anchor_ks)
        for i, k in enumerate(allk):
            pref = "cum_" if k in args.anchor_ks else ""
            p = os.path.join(args.logs_dir, f"{pref}k{k}_stage{stage}.log")
            eps, ps = parse_vals(p)
            if not eps:
                continue
            c = cmap(i / max(1, len(allk) - 1))
            style = "--" if (pref and k not in args.ks) else "-"
            ax.plot(eps, ps, style, marker="o", ms=2.5, lw=1.4, color=c,
                    label=f"K={k}" + (" (cumulative)" if pref else ""))
            j = max(range(len(ps)), key=lambda q: ps[q])
            ax.scatter([eps[j]], [ps[j]], s=26, color=c, zorder=5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val PSNR (dB, 5,000)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out_dir, f"fixedk_val_curves.{ext}"), dpi=200)
    plt.close()
    print(f"[out] {args.out_dir}/fixedk_val_curves.png/.pdf")


if __name__ == "__main__":
    main()
