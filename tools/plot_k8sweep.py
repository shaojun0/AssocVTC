#!/usr/bin/env python3
"""
plot_k8sweep.py — 「固定 8 步 × 总 token 预算 K」K-sweep 汇总/作图（不加载模型，纯 json）

输入（由 tools/eval_k8sweep.sh 产出）:
    $EV/linattn_k<K>_t1.json       A 相 T=1（45 ep）末步 PSNR/L1/SSIM
    $EV/linattn_k<K>_t8.json       B 相 T=8（6 ep, cumulative plan）末步 PSNR/L1/SSIM
    $EV/stepwise_k<K>_t8.json      B 相逐步（1..8）L1_255/PSNR + 累计 token
另读历史原生 K=64 参照（不同读窗口计划, 仅作同预算参照）:
    doc/2026-10-17/artifacts/eval_ablation8.json  → stdattn_k64 / stdattn_k64_rec
    doc/2026-10-16/artifacts/alit/stepwise_stdattn_k64_rec.json

产物（--out_dir）:
    k8sweep_table.md / k8sweep_summary.json
    k8sweep_psnr_vs_tokens.png/.pdf      末步 PSNR vs 总 token 预算
    k8sweep_stepwise_l1.png/.pdf          逐 step L1_255（x=累计 z_s 数）+ Δ_t

用法:
    python tools/plot_k8sweep.py --eval_dir /root/autodl-tmp/srdiff_k8sweep/eval \
        --out_dir doc/2026-10-19/artifacts/k8sweep
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS_DEFAULT = [8, 12, 16, 24, 32, 64]
HIST_T1 = os.path.join("doc", "2026-10-17", "artifacts", "eval_ablation8.json")
HIST_STEPWISE = os.path.join("doc", "2026-10-16", "artifacts", "alit",
                             "stepwise_stdattn_k64_rec.json")


def _load_one(path):
    with open(path) as f:
        d = json.load(f)
    # eval_linattn 的 json: {"<dir>/ckpt_best.pt": {...}}
    return next(iter(d.values())) if isinstance(d, dict) and d else None


def collect(ev, ks):
    rows = {}
    for k in ks:
        r = {"K": k}
        p1 = os.path.join(ev, f"linattn_k{k}_t1.json")
        p8 = os.path.join(ev, f"linattn_k{k}_t8.json")
        ps = os.path.join(ev, f"stepwise_k{k}_t8.json")
        if os.path.isfile(p1):
            d = _load_one(p1)
            r.update(t1_psnr=d["psnr"], t1_l1=d["l1_norm"], t1_ssim=d["ssim"])
        if os.path.isfile(p8):
            d = _load_one(p8)
            r.update(t8_psnr=d["psnr"], t8_l1=d["l1_norm"], t8_ssim=d["ssim"])
        if os.path.isfile(ps):
            with open(ps) as f:
                d = json.load(f)
            r["cum_z_s"] = d["cum_z_s"]
            r["windows"] = d["windows"]
            r["t8_step_l1"] = d["L1_255"]
            r["t8_step_psnr"] = d["PSNR"]
            r["t8_delta1_share"] = d.get("delta1_share")
            r["t8_best_step"] = d.get("best_step")          # 0-based
            r["t8_total_drop"] = d.get("total_drop_255")
            ps = d["PSNR"]
            r["t8_best_psnr"] = float(max(ps))               # oracle 最优步 PSNR
            r["t8_best_psnr_step"] = int(np.argmax(ps)) + 1  # 1-based
        rows[k] = r
    return rows


def native64_reference(rows):
    """历史原生 square 计划 K=64 的两点（T=1 / T=8）+ 逐步曲线，仅作参照。"""
    ref = {}
    try:
        with open(HIST_T1) as f:
            d = json.load(f)
        if "stdattn_k64/ckpt_best.pt" in d:
            ref["t1_psnr"] = d["stdattn_k64/ckpt_best.pt"]["psnr"]
            ref["t1_l1"] = d["stdattn_k64/ckpt_best.pt"]["l1_norm"]
        if "stdattn_k64_rec/ckpt_best.pt" in d:
            ref["t8_psnr"] = d["stdattn_k64_rec/ckpt_best.pt"]["psnr"]
            ref["t8_l1"] = d["stdattn_k64_rec/ckpt_best.pt"]["l1_norm"]
    except FileNotFoundError:
        pass
    try:
        with open(HIST_STEPWISE) as f:
            d = json.load(f)
        ref["cum_z_s"] = d["cum_z_s"]
        ref["t8_step_l1"] = d["L1_255"]
        ref["t8_step_psnr"] = d["PSNR"]
    except FileNotFoundError:
        pass
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=KS_DEFAULT)
    args = ap.parse_args()
    ev, od = args.eval_dir, args.out_dir
    os.makedirs(od, exist_ok=True)
    rows = collect(ev, args.ks)
    hist = native64_reference(rows)

    # ── 1. 汇总表 ──
    lines = [
        "| K (token) | T=1 PSNR | T=1 L1_255 | T=1 SSIM | T=8 PSNR | T=8 L1_255 | "
        "T=8 SSIM | T=8 oracle PSNR | Δ₁占比 | L1 最优步(1-based) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in args.ks:
        r = rows[k]
        if "t8_psnr" not in r and "t1_psnr" not in r:
            continue
        f = lambda v, fmt=".3f": (("%" + fmt) % v) if isinstance(v, float) else "—"
        d1 = r.get("t8_delta1_share")
        lines.append(
            f"| {k} | {f(r.get('t1_psnr'))} | {f(r.get('t1_l1'))} | "
            f"{f(r.get('t1_ssim'), '.4f')} | {f(r.get('t8_psnr'))} | "
            f"{f(r.get('t8_l1'))} | {f(r.get('t8_ssim'), '.4f')} | "
            f"{f(r.get('t8_best_psnr'))} | "
            f"{(('%.1f%%' % (100 * d1)) if isinstance(d1, float) else '—')} | "
            f"{r.get('t8_best_psnr_step', '—')} |")
    tbl = "\n".join(lines)
    with open(os.path.join(od, "k8sweep_table.md"), "w") as f:
        f.write("# 固定 8 步 × K token（ImageNet-100 val 5,000）\n\n" + tbl + "\n")

    # ── 2. 末步 PSNR vs 总 token 预算 ──
    ks = [k for k in args.ks if "t1_psnr" in rows[k] or "t8_psnr" in rows[k]]
    plt.figure(figsize=(6.4, 4.2))
    x1 = [k for k in ks if "t1_psnr" in rows[k]]
    y1 = [rows[k]["t1_psnr"] for k in x1]
    x8 = [k for k in ks if "t8_psnr" in rows[k]]
    y8 = [rows[k]["t8_psnr"] for k in x8]
    plt.plot(x1, y1, "o-", color="#4C72B0", label="T=1 (single-step, 45 ep)")
    plt.plot(x8, y8, "s-", color="#C44E52", label="T=8 (BPTT, 6 ep warm start)")
    if "t1_psnr" in hist:
        plt.scatter([64], [hist["t1_psnr"]], marker="o", s=70, facecolors="none",
                    edgecolors="#4C72B0",
                    label="T=1 native square K=64 (historical)")
    if "t8_psnr" in hist:
        plt.scatter([64], [hist["t8_psnr"]], marker="s", s=70, facecolors="none",
                    edgecolors="#C44E52",
                    label="T=8 native square K=64 (historical)")
    if "t8_step_psnr" in hist:
        plt.plot(hist["cum_z_s"], hist["t8_step_psnr"], ":", color="#7F7F7F",
                 marker="x", ms=4, lw=1.2,
                 label="K=64 native, zero-training early stop")
    for xx, yy in list(zip(x1, y1)) + list(zip(x8, y8)):
        plt.annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7)
    plt.xscale("log", base=2)
    plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("total token budget K (z_s)")
    plt.ylabel("PSNR (dB, 224×224, last step)")
    plt.title("AssocVTC: 8 fixed decoding steps × token budget K")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(od, f"k8sweep_psnr_vs_tokens.{ext}"), dpi=200)
    plt.close()

    # ── 3. 逐 step L1（x = 累计 token 数）──
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(x8):
        r = rows[k]
        if "t8_step_l1" not in r:
            continue
        c = cmap(i / max(1, len(x8) - 1))
        x = r["cum_z_s"]
        axes[0].plot(x, r["t8_step_l1"], "o-", ms=3.5, color=c, label=f"K={k}")
        d = np.asarray(r["t8_step_l1"][:-1]) - np.asarray(r["t8_step_l1"][1:])
        axes[1].plot(x[1:], d, "o-", ms=3.5, color=c, label=f"K={k}")
    if "t8_step_l1" in hist:
        axes[0].plot(hist["cum_z_s"], hist["t8_step_l1"], "k--", lw=1.2,
                     label="K=64 native (historical)")
    axes[0].set_xlabel("cumulative z_s read at step")
    axes[0].set_ylabel("pixel L1 (0–255)")
    axes[0].set_title("per-step reconstruction L1")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("cumulative z_s read at step")
    axes[1].set_ylabel("ΔL1 (positive = better)")
    axes[1].set_title("per-step improvement")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(od, f"k8sweep_stepwise_l1.{ext}"), dpi=200)
    plt.close()

    out = {"rows": {str(k): rows[k] for k in args.ks},
           "native_square_k64_reference": hist,
           "note": "cumulative plan = square 形状按总预算缩放（scaled_square_cumsum）; "
                   "native K=64 为历史不同读窗口计划, 仅作同预算参照"}
    with open(os.path.join(od, "k8sweep_summary.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(tbl)
    print(f"\n[out] {od}/k8sweep_table.md, k8sweep_summary.json, "
          f"k8sweep_psnr_vs_tokens.png, k8sweep_stepwise_l1.png")


if __name__ == "__main__":
    main()
