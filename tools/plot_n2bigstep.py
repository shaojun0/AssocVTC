#!/usr/bin/env python3
"""plot_n2bigstep.py — 「固定每步 n=2 × T=16/24/32」消融的汇总 / 作图（2026-10-23）

用户口径（2026-10-23）: n/step=2, 分别 16 / 24 / 32 个解码步（K = 2·T = 32/48/64）。
与 10-23 的 lown 臂（固定 8 步 × n=2/4/6）合起来回答:
  (a) **同一 n=2** 下加步数（T=8→16→24→32, 总预算 K 也随之涨）质量怎么走;
  (b) **同 K** 下「少 token × 多步」vs 「多 token × 少步」:
      K=32: n=4/T=8 (10-23 lown) vs n=2/T=16
      K=48: n=6/T=8 (10-23 lown) vs n=2/T=24
      K=64: n=8/T=8 (10-21 fixedk) vs n=2/T=32

输入（全部 5000 张 val 的末步指标）:
  · n2     : 本次 `--n2_eval` 目录 linattn_k{K}_t{T}.json (T=16/24/32 → K=32/48/64) + t1
  · lown   : doc/2026-10-23/artifacts/lown/eval（K=16/32/48 的 T=8 臂）
  · fixedk : doc/2026-10-21/artifacts/fixedk/eval（K=64/96/192 的 T=8 臂）
输出:
  · n2bigstep_table.md        主表
  · n2bigstep_summary.json    机读汇总
  · n2_steps_vs_quality.png   同一 n=2 下 T–质量
  · n2_fixedK_compare.png     同 K 下「多步少 token」vs「少步多 token」
  · n2_stepwise_l1.png        逐步 L1_255（新臂实线, 同 K 的 T=8 臂虚线对照）
  · n2_val_curves.png         B 相逐 epoch val PSNR
用法:
    python tools/plot_n2bigstep.py \
        --n2_eval doc/2026-10-23/artifacts/n2bigstep/eval \
        --lown_eval doc/2026-10-23/artifacts/lown/eval \
        --fixedk_eval doc/2026-10-21/artifacts/fixedk/eval \
        --out_dir doc/2026-10-23/artifacts/n2bigstep \
        --train_logs doc/2026-10-23/artifacts/n2bigstep/train_logs
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_N2 = "#C44E52"
COL_T8 = "#4C72B0"


def load_one(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict) and "L1_255" in d:
        return d                       # eval_stepwise_linattn.py 的 json 是**扁平**的
    return next(iter(d.values())) if isinstance(d, dict) and d else None


def load_eval(path):
    """读 eval_linattn.py 的 json（顶层是 {ckpt: 指标}; K/T 从 cfg 反推）。"""
    if not os.path.isfile(path):
        return None
    d = load_one(path)
    if d is None:
        return None
    cfg = d.get("cfg", {}) or {}
    K = d.get("K") or d.get("num_specials") or cfg.get("num_specials")
    if K is None:
        return None
    steps = [x for x in str(cfg.get("dec_steps", "")).split(",") if x.strip()]
    T = int(d["T"]) if d.get("T") else (len(steps) if cfg.get("dec_recurrent") else 1)
    return dict(K=int(K), T=int(T),
                psnr=float(d["psnr"]), l1=float(d["l1_norm"]),
                ssim=float(d["ssim"]), ckpt=d.get("ckpt", ""))


def load_stepwise(path):
    if not os.path.isfile(path):
        return None
    d = load_one(path)
    if d is None:
        return None
    return d


def collect_fixedk(ev, ks):
    """T=8 fixed-k 臂（n = K/8）。"""
    out = {}
    for k in ks:
        d = load_eval(os.path.join(ev, f"linattn_k{k}_t8.json"))
        if d:
            d["n"] = k // 8
            d["T"] = 8
            out[k] = d
    return out


def collect_n2(ev, Ts):
    out = {}
    for T in Ts:
        k = 2 * T
        d = load_eval(os.path.join(ev, f"linattn_k{k}_t{T}.json"))
        if d:
            d["n"] = 2
            d["T"] = T
            out[k] = d
    return out


def val_curves(path):
    """从 train_log.jsonl 里取 [val] 记录（含 epoch/step/psnr/l1_norm）。"""
    rows = []
    if not os.path.isfile(path):
        return rows
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("split") == "val" and "psnr" in r:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n2_eval", required=True)
    ap.add_argument("--lown_eval", default="doc/2026-10-23/artifacts/lown/eval")
    ap.add_argument("--fixedk_eval", default="doc/2026-10-21/artifacts/fixedk/eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_logs", default="")
    ap.add_argument("--n2_Ts", type=int, nargs="*", default=[16, 24, 32])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n2 = collect_n2(args.n2_eval, args.n2_Ts)
    # T=8 的 n=2 点（K=16）来自 10-23 lown 臂
    t8 = {}
    t8.update(collect_fixedk(args.lown_eval, [16, 32, 48]))
    t8.update(collect_fixedk(args.fixedk_eval, [64, 96, 192]))

    # ── A 相（T=1）自洽 & 与 10-19 对拍 ──
    a_rows = {}
    for T in args.n2_Ts:
        k = 2 * T
        a = load_eval(os.path.join(args.n2_eval, f"linattn_k{k}_t1.json"))
        if a:
            a_rows[k] = a

    # ── 主表: 同一 n=2 下 T 扫描 ──
    sweep = []
    if 16 in t8:
        r = dict(t8[16]); r.update(T=8, K=16, n=2, src="2026-10-23 lown (T=8)")
        sweep.append(r)
    for k in sorted(n2):
        r = dict(n2[k]); r["src"] = "2026-10-23 本次"
        sweep.append(r)
    sweep.sort(key=lambda r: r["T"])

    lines = ["# 固定每步 n=2 × T=8/16/24/32（K=2T）—— ImageNet-100 val 5,000，末步", "",
             "| T | K (=2T) | 压缩比 N/K | PSNR dB | L1_norm | SSIM | 来源 |",
             "|---|---|---|---|---|---|---|"]
    for r in sweep:
        lines.append(f"| {r['T']} | {r['K']} | {196.0/r['K']:.2f}× | {r['psnr']:.3f} | "
                     f"{r['l1']:.4f} | {r['ssim']:.4f} | {r['src']} |")
    # 边际收益
    lines += ["", "**边际收益（相邻 T, 同一 n=2）**", "",
              "| 区间 | ΔT | ΔK | ΔPSNR dB | 每步 dB |", "|---|---|---|---|---|"]
    for a, b in zip(sweep[:-1], sweep[1:]):
        dT = b["T"] - a["T"]
        dp = b["psnr"] - a["psnr"]
        lines.append(f"| T={a['T']}→{b['T']} | +{dT} | +{b['K']-a['K']} | "
                     f"{dp:+.3f} | {dp/dT:+.4f} |")

    # ── 同 K 对照表 ──
    lines += ["", "**同一总预算 K 下：多步少 token vs 少步多 token**", "",
              "| K | T=8 臂 (n=K/8) | T=8 PSNR | n=2 臂 T=K/2 | n=2 PSNR | ΔPSNR (n=2 − T=8) |",
              "|---|---|---|---|---|---|"]
    for k in sorted(set(t8) & set(n2)):
        a, b = t8[k], n2[k]
        lines.append(f"| {k} | n={a['n']} | {a['psnr']:.3f} | n=2, T={b['T']} | "
                     f"{b['psnr']:.3f} | {b['psnr']-a['psnr']:+.3f} |")

    if a_rows:
        lines += ["", "**A 相（T=1）自洽核对**", "",
                  "| K | A 相 PSNR dB | L1_norm |", "|---|---|---|"]
        for k in sorted(a_rows):
            lines.append(f"| {k} | {a_rows[k]['psnr']:.3f} | {a_rows[k]['l1']:.4f} |")

    tbl = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "n2bigstep_table.md"), "w") as f:
        f.write(tbl)
    print(tbl)

    # ── 图 1: n=2 下 T–质量 ──
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, key, ylab, ttl in (
            (axes[0], "l1", "L1_norm (normalized space; lower is better)",
             "n=2: reconstruction L1 vs #decoding steps T"),
            (axes[1], "psnr", "PSNR (dB)", "n=2: PSNR vs #decoding steps T")):
        xs = [r["T"] for r in sweep]
        ys = [r[key] for r in sweep]
        ax.plot(xs, ys, "o-", color=COL_N2, label="n=2 per step (K=2T)")
        for r in sweep:
            ax.annotate(f"K={r['K']}", (r["T"], r[key]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xlabel("number of decoding steps T (K = 2·T)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out_dir, f"n2_steps_vs_quality.{ext}"), dpi=200)
    plt.close()

    # ── 图 2: 同 K 对照（分组柱）──
    ks = sorted(set(t8) & set(n2))
    if ks:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        xs = list(range(len(ks)))
        w = 0.36
        v8 = [t8[k]["psnr"] for k in ks]
        v2 = [n2[k]["psnr"] for k in ks]
        b1 = ax.bar([x - w / 2 for x in xs], v8, w, color=COL_T8,
                    label="T=8, n=K/8 tokens/step")
        b2 = ax.bar([x + w / 2 for x in xs], v2, w, color=COL_N2,
                    label="n=2 tokens/step, T=K/2")
        for bars, vals in ((b1, v8), (b2, v2)):
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
        for x, k in zip(xs, ks):
            ax.text(x, 1.0, f"K={k}\n(T={n2[k]['T']})", ha="center", va="bottom",
                    fontsize=7, color="gray")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"K={k}" for k in ks])
        lo = min(v8 + v2) - 1.0
        hi = max(v8 + v2)
        ax.set_ylim(lo, hi + 1.0)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("same token budget K: more steps vs more tokens per step")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(args.out_dir, f"n2_fixedK_compare.{ext}"), dpi=200)
        plt.close()

    # ── 图 3: 逐步 L1_255 ──
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {16: "#8C8C8C", 24: "#DD8452", 32: "#55A868"}
    for T in args.n2_Ts:
        k = 2 * T
        sw = load_stepwise(os.path.join(args.n2_eval, f"stepwise_k{k}_t{T}.json"))
        if not sw:
            continue
        L = sw["L1_255"]
        ax.plot(range(1, len(L) + 1), L, "-", color=colors.get(T, COL_N2),
                label=f"n=2, T={T} (K={k})  final={L[-1]:.2f}")
        bi = int(sw.get("best_step", len(L) - 1))
        ax.plot([bi + 1], [L[bi]], "*", color=colors.get(T, COL_N2), ms=10)
        # 同 K 的 T=8 臂作虚线对照
        old = load_stepwise(os.path.join(args.lown_eval, f"stepwise_k{k}_t8.json")) or \
            load_stepwise(os.path.join(args.fixedk_eval, f"stepwise_k{k}_t8.json"))
        if old:
            L8 = old["L1_255"]
            ax.plot(range(1, len(L8) + 1), L8, "--", color=colors.get(T, COL_N2),
                    alpha=0.55, label=f"n={k//8}, T=8 (K={k})  final={L8[-1]:.2f}")
    ax.set_xlabel("decoding step t")
    ax.set_ylabel("L1 (0-255 pixel scale; lower is better)")
    ax.set_title("stepwise reconstruction error (star = best step)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out_dir, f"n2_stepwise_l1.{ext}"), dpi=200)
    plt.close()

    # ── 图 4: B 相 val 曲线 ──
    if args.train_logs:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for T in args.n2_Ts:
            k = 2 * T
            rows = val_curves(os.path.join(args.train_logs, f"k{k}_t{T}.jsonl"))
            if not rows:
                continue
            ax.plot([r["step"] for r in rows], [r["psnr"] for r in rows], "o-",
                    color=colors.get(T, COL_N2), ms=4,
                    label=f"n=2, T={T} (K={k})")
        ax.set_xlabel("optimizer step (B phase, 6 epochs = 11,406 steps)")
        ax.set_ylabel("val PSNR (dB, 2,560-image epoch val)")
        ax.set_title("B-phase validation curves")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(args.out_dir, f"n2_val_curves.{ext}"), dpi=200)
        plt.close()

    summary = {
        "n2_sweep": [dict(T=r["T"], K=r["K"], psnr=r["psnr"], l1=r["l1"],
                          ssim=r["ssim"], src=r["src"]) for r in sweep],
        "T8_arms": {str(k): dict(n=t8[k]["n"], psnr=t8[k]["psnr"], l1=t8[k]["l1"],
                                 ssim=t8[k]["ssim"]) for k in sorted(t8)},
        "n2_arms": {str(k): dict(T=n2[k]["T"], psnr=n2[k]["psnr"], l1=n2[k]["l1"],
                                 ssim=n2[k]["ssim"]) for k in sorted(n2)},
        "A_phase": {str(k): dict(psnr=a_rows[k]["psnr"], l1=a_rows[k]["l1"])
                    for k in sorted(a_rows)},
        "note": "fixed plan, n=2 tokens/step, T=16/24/32 (K=32/48/64); "
                "B 相 6 ep / 等效 bs64 (micro32 × accum2) / lr2e-4 / 从 A 相 ckpt 热启动; "
                "A 相复用 10-19 k8sweep ckpt（K=32/48 与 10-23 重跑逐位一致）",
    }
    with open(os.path.join(args.out_dir, "n2bigstep_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out_dir}/n2bigstep_table.md, n2bigstep_summary.json, 4 张图")


if __name__ == "__main__":
    main()
