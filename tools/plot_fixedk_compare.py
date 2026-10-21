#!/usr/bin/env python3
"""
plot_fixedk_compare.py — 「固定 8 步 × 每步均匀 n 个 z_s」消融汇总/作图（纯 json, 不载模型）

对照口径（2026-10-21 用户指定）:
    fixed     : 每步固定读 n 个 z_s ⇒ 总预算 K = 8·n ∈ {64,96,128,192,256};
    cumulative: 2026-10-19 的递增分配（square 形状按 K 缩放）: 同 K、同 8 步,
                唯一差别 = 读窗口形状（前少后多）。
    ⚠️ K=256 > N=196 ⇒ 不再是压缩（K/N=1.31）, 只作「预算超过 patch 数」的探针点。

输入（由 tools/eval_k8sweep.sh 产出, 文件名同 cumulative 臂）:
    <fixed_eval>/linattn_k<K>_t1.json       A 相 T=1 末步 PSNR/L1/SSIM
    <fixed_eval>/linattn_k<K>_t8.json       B 相 T=8（fixed plan）末步
    <fixed_eval>/stepwise_k<K>_t8.json      B 相逐步（1..8）L1_255/PSNR + 窗口
    <cum_eval>/...                          同 K 的 cumulative 臂（对照, 可缺）
产物（--out_dir）:
    fixedk_table.md / fixedk_vs_cumulative.md / fixedk_summary.json
    fixedk_psnr_vs_tokens.png/.pdf      T=8 / T=1 末步 PSNR vs K（fixed vs cumulative）
    fixedk_stepwise_l1.png/.pdf         fixed 臂逐 step L1_255（x=累计 z_s）+ Δ
    fixedk_stepwise_vs_cum.png/.pdf     同 K 的 fixed vs cumulative 逐步曲线

用法:
    python tools/plot_fixedk_compare.py \
        --fixed_eval doc/2026-10-21/artifacts/fixedk/eval \
        --cum_eval   doc/2026-10-19/artifacts/k8sweep/eval \
        --out_dir    doc/2026-10-21/artifacts/fixedk
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIXED_KS = [64, 96, 128, 192, 256]
CUM_KS = [64, 96, 128]
N_PATCHES = 196          # 224² / patch 16 ⇒ 14×14
T_STEPS = 8


def _load_one(path):
    """eval_linattn 的 json 形如 {"<dir>/ckpt_best.pt": {...}}。"""
    with open(path) as f:
        d = json.load(f)
    return next(iter(d.values())) if isinstance(d, dict) and d else None


def collect(ev, ks):
    rows = {}
    for k in ks:
        r = {"K": k, "n": k // T_STEPS, "K_over_N": k / N_PATCHES,
             "compression": N_PATCHES / k}
        p1 = os.path.join(ev, f"linattn_k{k}_t1.json")
        p8 = os.path.join(ev, f"linattn_k{k}_t8.json")
        if os.path.isfile(p1):
            d = _load_one(p1)
            r.update(t1_psnr=d["psnr"], t1_l1=d["l1_norm"], t1_ssim=d["ssim"])
        if os.path.isfile(p8):
            d = _load_one(p8)
            r.update(t8_psnr=d["psnr"], t8_l1=d["l1_norm"], t8_ssim=d["ssim"])
        ps = os.path.join(ev, f"stepwise_k{k}_t8.json")
        if os.path.isfile(ps):
            with open(ps) as f:
                d = json.load(f)
            r["plan"] = d.get("dec_step_plan")
            r["steps"] = d.get("steps")
            r["windows"] = d.get("windows")
            r["cum_z_s"] = d.get("cum_z_s")
            r["t8_step_l1"] = d.get("L1_255")
            r["t8_step_psnr"] = d.get("PSNR")
            if d.get("PSNR"):
                r["t8_best_psnr"] = float(max(d["PSNR"]))
                r["t8_best_step"] = int(np.argmax(d["PSNR"])) + 1
            r["t8_delta1_share"] = d.get("delta1_share")
            r["t8_total_drop"] = d.get("total_drop_255")
        rows[k] = r
    return rows


def _fmt(v, fmt=".3f"):
    return (("%" + fmt) % v) if isinstance(v, float) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed_eval", required=True)
    ap.add_argument("--cum_eval", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=FIXED_KS)
    ap.add_argument("--cum_ks", type=int, nargs="*", default=CUM_KS)
    args = ap.parse_args()
    od = args.out_dir
    os.makedirs(od, exist_ok=True)

    fx = collect(args.fixed_eval, args.ks)
    cum = collect(args.cum_eval, args.cum_ks) if args.cum_eval else {}

    # ── 1. fixed 主表 ──
    lines = [
        "| K (=8n) | n/step | K/N | 压缩比 N/K | T=1 PSNR | T=1 L1_norm | T=8 PSNR | "
        "T=8 L1_norm | T=8 SSIM | T=8 oracle PSNR | L1 最优步 | 备注 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in args.ks:
        r = fx[k]
        if "t8_psnr" not in r and "t1_psnr" not in r:
            continue
        note = "K>N: 非压缩（预算超过 patch 数）" if k > N_PATCHES else ""
        lines.append(
            f"| {k} | {r['n']} | {r['K_over_N']:.2f} | {r['compression']:.2f}× | "
            f"{_fmt(r.get('t1_psnr'))} | {_fmt(r.get('t1_l1'))} | "
            f"{_fmt(r.get('t8_psnr'))} | {_fmt(r.get('t8_l1'))} | "
            f"{_fmt(r.get('t8_ssim'), '.4f')} | {_fmt(r.get('t8_best_psnr'))} | "
            f"{r.get('t8_best_step', '—')} | {note} |")
    tbl = "\n".join(lines)
    with open(os.path.join(od, "fixedk_table.md"), "w") as f:
        f.write("# 固定 8 步 × 每步均匀 n 个 z_s（ImageNet-100 val 5,000, 末步）\n\n"
                "L1_norm = 归一化空间 L1（与主报告同口径; 换算 0–255 约 ×0.235×255 ≈ 60×）; "
                "逐步曲线用 0–255 口径 L1_255。\n\n"
                + tbl + "\n")
    print(tbl)

    # ── 2. fixed vs cumulative（同 K 同 8 步, 只差读窗口形状）──
    if cum:
        lines2 = [
            "| K | plan | T=1 PSNR | T=8 PSNR | T=8 L1_norm | T=8 SSIM | "
            "ΔT8 PSNR (fixed−cum) | ΔT8 L1_norm | ΔT8 SSIM |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for k in args.cum_ks:
            if k not in fx or k not in cum:
                continue
            a, b = fx[k], cum[k]
            if "t8_psnr" not in a or "t8_psnr" not in b:
                continue
            lines2.append(
                f"| {k} | fixed | {_fmt(a.get('t1_psnr'))} | {_fmt(a.get('t8_psnr'))} | "
                f"{_fmt(a.get('t8_l1'))} | {_fmt(a.get('t8_ssim'), '.4f')} | "
                f"{a['t8_psnr']-b['t8_psnr']:+.3f} | {a['t8_l1']-b['t8_l1']:+.4f} | "
                f"{a['t8_ssim']-b['t8_ssim']:+.4f} |")
            lines2.append(
                f"| {k} | cumulative | {_fmt(b.get('t1_psnr'))} | {_fmt(b.get('t8_psnr'))} | "
                f"{_fmt(b.get('t8_l1'))} | {_fmt(b.get('t8_ssim'), '.4f')} | — | — | — |")
        t2 = "\n".join(lines2)
        with open(os.path.join(od, "fixedk_vs_cumulative.md"), "w") as f:
            f.write("# fixed（每步均匀）vs cumulative（square 形状缩放）—— 同 K, 各 8 步\n\n"
                    + t2 + "\n")
        print("\n" + t2)

    # ── 3. 逐 step L1：fixed 各臂（x = 累计读入 z_s 数）──
    ks8 = [k for k in args.ks if "t8_step_l1" in fx[k]]
    if ks8:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        cmap = plt.get_cmap("viridis")
        for i, k in enumerate(ks8):
            r = fx[k]
            c = cmap(i / max(1, len(ks8) - 1))
            x = r["cum_z_s"]
            axes[0].plot(x, r["t8_step_l1"], "o-", ms=3.5, color=c, label=f"K={k} (n={r['n']})")
            d = np.asarray(r["t8_step_l1"][:-1]) - np.asarray(r["t8_step_l1"][1:])
            axes[1].plot(x[1:], d, "o-", ms=3.5, color=c, label=f"K={k}")
        axes[0].set_xlabel("cumulative z_s read at step")
        axes[0].set_ylabel("pixel L1 (0–255)")
        axes[0].set_title("fixed plan: per-step reconstruction L1")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        axes[1].axhline(0, color="gray", lw=0.8)
        axes[1].set_xlabel("cumulative z_s read at step")
        axes[1].set_ylabel("ΔL1 (positive = better)")
        axes[1].set_title("fixed plan: per-step improvement")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(od, f"fixedk_stepwise_l1.{ext}"), dpi=200)
        plt.close()

    # ── 4. 同 K 的 fixed vs cumulative 逐步对照 ──
    matched = [k for k in args.cum_ks if k in fx and k in cum
               and "t8_step_l1" in fx[k] and "t8_step_l1" in cum[k]]
    if matched:
        fig, axes = plt.subplots(1, len(matched), figsize=(4.6 * len(matched), 3.8),
                                 squeeze=False)
        for j, k in enumerate(matched):
            ax = axes[0][j]
            a, b = fx[k], cum[k]
            ax.plot(a["cum_z_s"], a["t8_step_l1"], "o-", color="#C44E52",
                    label="fixed (uniform n/step)")
            ax.plot(b["cum_z_s"], b["t8_step_l1"], "s--", color="#4C72B0",
                    label="cumulative (increasing, 2026-10-19)")
            ax.set_xlabel("cumulative z_s read")
            ax.set_ylabel("pixel L1 (0–255)")
            ax.set_title(f"K={k}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(od, f"fixedk_stepwise_vs_cum.{ext}"), dpi=200)
        plt.close()

    # ── 5. 末步 PSNR vs K ──
    ks_all = [k for k in args.ks if "t1_psnr" in fx[k] or "t8_psnr" in fx[k]]
    if ks_all:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        for ax, key, title in ((axes[0], "t8", "T=8 (BPTT, last step)"),
                               (axes[1], "t1", "T=1 (single step, A phase)")):
            xf = [k for k in ks_all if f"{key}_psnr" in fx[k]]
            ax.plot(xf, [fx[k][f"{key}_psnr"] for k in xf], "o-", color="#C44E52",
                    label="fixed (uniform n/step)")
            xc = [k for k in args.cum_ks
                  if k in cum and f"{key}_psnr" in cum[k]]
            if xc:
                ax.plot(xc, [cum[k][f"{key}_psnr"] for k in xc], "s--", color="#4C72B0",
                        label="cumulative (increasing)")
            for xx in xf:
                ax.annotate(f"{fx[xx][f'{key}_psnr']:.2f}", (xx, fx[xx][f"{key}_psnr"]),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7)
            ax.axvline(N_PATCHES, color="gray", ls=":", lw=1.0)
            ax.text(N_PATCHES * 1.02, ax.get_ylim()[0] + 0.5, "K=N=196\n(no compression)",
                    fontsize=7, color="gray")
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks_all, [str(k) for k in ks_all])
            ax.set_xlabel("total token budget K = 8·n")
            ax.set_ylabel("PSNR (dB, 224×224)")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(od, f"fixedk_psnr_vs_tokens.{ext}"), dpi=200)
        plt.close()

    out = {"fixed": {str(k): fx[k] for k in args.ks},
           "cumulative": {str(k): cum[k] for k in (args.cum_ks if cum else [])},
           "N_patches": N_PATCHES,
           "note": "fixed = 每步均匀读 n=K/8 个 z_s（首步另含 z_cls）; cumulative = 2026-10-19 "
                   "的递增分配（scaled_square_cumsum）; 两者同 8 步、同 K, 唯一差别 = 读窗口形状; "
                   "K=256>N=196 属超预算探针点（不再是压缩）"}
    with open(os.path.join(od, "fixedk_summary.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {od}/fixedk_table.md, fixedk_vs_cumulative.md, fixedk_summary.json, "
          f"fixedk_psnr_vs_tokens.png, fixedk_stepwise_l1.png, fixedk_stepwise_vs_cum.png")


if __name__ == "__main__":
    main()
