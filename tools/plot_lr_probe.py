#!/usr/bin/env python3
"""plot_lr_probe.py — 训练 A 相 val PSNR 曲线的「学习率探针」对比图。

用法（2026-10-21 K=128 探针）:
    python tools/plot_lr_probe.py --out doc/2026-10-21/artifacts/fixedk/fixedk_k128_lr_probe.png \
        --series "K=128 lr=1e-4:doc/2026-10-21/artifacts/fixedk/logs/k128_stage1.log" \
        --series "K=128 lr=2e-4 (stalled):doc/2026-10-21/artifacts/fixedk/logs/k128_stage1_lr2e-4_STALLED.log" \
        --series "K=96 lr=2e-4 (好盆地):doc/2026-10-21/artifacts/fixedk/logs/k96_stage1.log"
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_vals(path):
    eps, ps = [], []
    pat = re.compile(r"\[val\] ep(\d+).*?psnr=([0-9.]+)dB")
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            eps.append(int(m.group(1)))
            ps.append(float(m.group(2)))
    return eps, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--series", action="append", required=True,
                    help='标签:日志路径（可多次）')
    ap.add_argument("--title", default="A phase (T=1): learning-rate probe")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#937860"]
    plt.figure(figsize=(6.6, 4.2))
    for i, spec in enumerate(args.series):
        label, path = spec.split(":", 1)
        eps, ps = parse_vals(path)
        if not eps:
            print(f"[warn] 无 val 行: {path}")
            continue
        plt.plot(eps, ps, "-o", ms=2.5, lw=1.5, color=colors[i % len(colors)],
                 label=f"{label}  (best {max(ps):.2f} dB)")
    plt.xlabel("epoch")
    plt.ylabel("val PSNR (dB, 5,000)")
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.splitext(args.out)[0] + f".{ext}", dpi=200)
    plt.close()
    print(f"[out] {os.path.splitext(args.out)[0]}.png/.pdf")


if __name__ == "__main__":
    main()
