#!/usr/bin/env python3
"""plot_lown_compare.py — 「固定 8 步 × 每步 n 个」的 n–质量曲线（含低 n 延伸，2026-10-23）

背景（用户口径）: 固定 8 步、每步均匀 n 个 z_s 时，K=64/96/192（n=8/12/24）的 T=8 L1 都差不多
（0.113/0.112/0.109）⇒ 想知道**下限在哪** ⇒ 补 n=6/4/2（K=48/32/16）。

本脚本把三个来源的评测 json 合到一张「n – 质量」图上：
  · base    : 2026-10-21 fixed-k 臂（无 QK-Norm）: K=64/96/192 → n=8/12/24
  · lown    : 2026-10-23 低 n 延伸臂（无 QK-Norm）: K=48/32/16 → n=6/4/2
  · retrain : 2026-10-22 QK-Norm 重训（另一配置，图上单独一条，不混进主曲线）

用法:
    python tools/plot_lown_compare.py \
        --base_eval    doc/2026-10-21/artifacts/fixedk/eval \
        --lown_eval    doc/2026-10-23/artifacts/lown/eval \
        --retrain_eval doc/2026-10-22/artifacts/k128k256_fix/eval \
        --out_dir      doc/2026-10-23/artifacts/lown
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_one(path):
    with open(path) as f:
        d = json.load(f)
    return next(iter(d.values())) if isinstance(d, dict) and d else None


def collect(ev, ks):
    rows = {}
    for k in ks:
        p = os.path.join(ev, f"linattn_k{k}_t8.json")
        if not os.path.isfile(p):
            continue
        d = load_one(p)
        rows[k] = dict(K=k, n=k // 8, psnr=d["psnr"], l1=d["l1_norm"], ssim=d["ssim"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_eval", required=True)
    ap.add_argument("--lown_eval", required=True)
    ap.add_argument("--retrain_eval", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_ks", type=int, nargs="*", default=[64, 96, 192])
    ap.add_argument("--lown_ks", type=int, nargs="*", default=[16, 32, 48])
    ap.add_argument("--retrain_ks", type=int, nargs="*", default=[128, 256])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = collect(args.base_eval, args.base_ks)
    lown = collect(args.lown_eval, args.lown_ks)
    ret = collect(args.retrain_eval, args.retrain_ks) if args.retrain_eval else {}
    main_rows = {**base, **lown}

    # ── 表 ──
    lines = ["| K (=8n) | n/step | T=8 PSNR | T=8 L1_norm | T=8 SSIM | 来源 |",
             "|---|---|---|---|---|---|"]
    for k in sorted(main_rows):
        r = main_rows[k]
        src = "2026-10-21（无 QK-Norm）" if k in base else "2026-10-23（低 n 延伸）"
        lines.append(f"| {k} | {r['n']} | {r['psnr']:.3f} | {r['l1']:.4f} | "
                     f"{r['ssim']:.4f} | {src} |")
    for k in sorted(ret):
        r = ret[k]
        lines.append(f"| {k} | {r['n']} | {r['psnr']:.3f} | {r['l1']:.4f} | "
                     f"{r['ssim']:.4f} | 2026-10-22（+QK-Norm 重训） |")
    tbl = "\n".join(lines)
    with open(os.path.join(args.out_dir, "lown_table.md"), "w") as f:
        f.write("# 固定 8 步 × 每步 n 个 z_s：n–质量曲线（ImageNet-100 val 5,000，末步）\n\n"
                + tbl + "\n")
    print(tbl)

    # ── 图 ──
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, key, ylab, ttl in (
            (axes[0], "l1", "L1_norm (normalized space; lower is better)", "reconstruction L1 vs n per step"),
            (axes[1], "psnr", "PSNR (dB)", "PSNR vs n per step")):
        xs = sorted(main_rows)
        ax.plot([main_rows[k]["n"] for k in xs],
                [main_rows[k][key] for k in xs], "o-", color="#C44E52",
                label="no QK-Norm (2026-10-21 + 10-23 low-n)")
        for k in xs:
            ax.annotate(f"K={k}", (main_rows[k]["n"], main_rows[k][key]),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        if ret:
            xr = sorted(ret)
            ax.plot([ret[k]["n"] for k in xr], [ret[k][key] for k in xr], "s--",
                    color="#55A868", label="+QK-Norm retrain (2026-10-22)")
            for k in xr:
                ax.annotate(f"K={k}", (ret[k]["n"], ret[k][key]),
                            textcoords="offset points", xytext=(0, -12), ha="center",
                            fontsize=7, color="#2F6B3C")
        ax.axvline(8, color="gray", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 6, 8, 12, 16, 24, 32], ["2", "4", "6", "8", "12", "16", "24", "32"])
        ax.set_xlabel("n = tokens read per decoding step (K = 8n)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out_dir, f"lown_n_vs_quality.{ext}"), dpi=200)
    plt.close()

    with open(os.path.join(args.out_dir, "lown_summary.json"), "w") as f:
        json.dump({"base": {str(k): base[k] for k in base},
                   "lown": {str(k): lown[k] for k in lown},
                   "retrain_qknorm": {str(k): ret[k] for k in ret},
                   "note": "fixed 8 步、每步均匀 n 个 z_s；base/lown 为无 QK-Norm 同协议，"
                           "retrain 为 2026-10-22 的 +QK-Norm 配置（仅作参照, 不混入主曲线）"},
                  f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out_dir}/lown_table.md, lown_summary.json, lown_n_vs_quality.png")


if __name__ == "__main__":
    main()
