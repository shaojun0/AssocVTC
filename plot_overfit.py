#!/usr/bin/env python
"""汇总/画图：v10 单片段过拟合的架构对照（4 个 metrics.jsonl）。

用法：
  python plot_overfit.py --root /root/autodl-tmp/tok448_v2/overfit \
      --out doc/2026-11-18/overfit_v10_arch.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402

NAMES = [
    ("prefix", "并行 prefix"),
    ("step_nocopy", "循环 step（纯递推）"),
    ("step_copy", "循环 step + 直抄"),
    ("step_warm", "循环 step + 直抄（并行预热）"),
]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]


def load(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/root/autodl-tmp/tok448_v2/overfit")
    ap.add_argument("--out", type=str, default="overfit_v10_arch.png")
    a = ap.parse_args()

    runs = []
    for name, label in NAMES:
        p = os.path.join(a.root, name, "metrics.jsonl")
        if os.path.isfile(p):
            runs.append((name, label, load(p)))
    if not runs:
        print("没有找到 metrics.jsonl")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for (name, label, rows), c in zip(runs, COLORS[:len(runs)]):
        it = [r["it"] for r in rows]
        ax.plot(it, [r["loss"] for r in rows], "-o", ms=3, color=c, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss（已读帧 masked L1）")
    ax.set_title("训练损失（单片段，越低越好）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for (name, label, rows), c in zip(runs, COLORS[:len(runs)]):
        it = [r["it"] for r in rows]
        ax.plot(it, [r["l1_all"] for r in rows], "-o", ms=3, color=c, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("末档全片段 latent L1")
    ax.set_title("过拟合质量（17 行全算，越低越好）")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ceil = None
    for (name, label, rows), c in zip(runs, COLORS[:len(runs)]):
        it = [r["it"] for r in rows]
        ps = [r.get("psnr", float("nan")) for r in rows]
        ax.plot(it, ps, "-o", ms=3, color=c, label=label)
        if ceil is None:
            ceil = rows[-1].get("tok_full")
    if ceil:
        ax.axhline(ceil, ls="--", c="k", lw=1,
                   label=f"分词器读全量上界 {ceil:.2f} dB")
    ax.set_xlabel("step")
    ax.set_ylabel("PSNR / dB")
    ax.set_title("末档像素 PSNR（冻结 R 解码）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for (name, label, rows), c in zip(runs, COLORS[:len(runs)]):
        cur = rows[-1]["l1_read"]
        ax.plot(range(1, len(cur) + 1), cur, "-o", ms=3, color=c,
                label=f"{label} @step{rows[-1]['it']}")
    ax.set_xlabel("读出步 t（第 t 步已读 4(t+1) 帧）")
    ax.set_ylabel("已读帧 latent L1")
    ax.set_title("最终评测的逐档曲线（已读帧；末点=全片）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("v10 单片段过拟合：循环(step) vs 并行(prefix) 读出架构", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"[plot] {a.out}")

    # ── 表格
    print("\n| 架构 | 最终 loss | 末档 latent L1 | PSNR | 上界 |")
    print("|---|---|---|---|---|")
    for name, label, rows in runs:
        r = rows[-1]
        print(f"| {label} | {r['loss']:.5f} | {r['l1_all']:.4f} | "
              f"{r.get('psnr', float('nan')):.2f} | {r.get('tok_full', float('nan')):.2f} |")


if __name__ == "__main__":
    main()
