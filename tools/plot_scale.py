#!/usr/bin/env python3
"""plot_scale.py — 「容量阶梯」：编码器/解码器参数量 vs 还原质量（n=2, T=16）(2026-10-24)

用户口径（2026-10-24）: 固定 n/step=2、T=16（K=32），enc/dec 参数量各自
60M → 120M → 240M → 360M（深度钉死 enc 13 / dec 7，宽度 scaling）。
本脚本把 4 个新档 + 2026-10-23 的 63M 基线臂（enc 31.83 + dec 31.19，同协议）
画到同一张「容量–质量」图上，并输出主表与逐步曲线。

输入（相对仓库根目录, 默认值即本次产物位置）:
  --eval    doc/2026-10-24/artifacts/scale/eval      新 4 档 json
  --base    doc/2026-10-23/artifacts/n2bigstep/eval  63M 基线（K=32/T=16 与 T=1）
输出:
  scale_table.md / scale_summary.json
  scale_params_vs_quality.png|pdf   容量–质量（PSNR/L1/SSIM, T=16 末步 + T=1）
  scale_stepwise_l1.png|pdf         逐步 L1_255（含 63M 基线）
用法:
    python tools/plot_scale.py --eval doc/2026-10-24/artifacts/scale/eval \
        --out_dir doc/2026-10-24/artifacts/scale
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# M -> (enc_dim, enc_heads, dec_dim, dec_heads), 与 tools/scale_arms.sh 一致
ARMS = {
    60: (624, 12, 704, 8),
    120: (864, 16, 1024, 16),
    240: (1248, 16, 1440, 16),
    360: (1504, 16, 1792, 16),
}
BASE63 = dict(enc_M=31.83, dec_M=31.19, total_M=63.02,
              enc_dim=448, enc_heads=8, dec_dim=512, dec_heads=8)


def load_one(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict) and "L1_255" in d:
        return d                      # stepwise json 是扁平结构
    return next(iter(d.values())) if isinstance(d, dict) and d else None


def load_eval(path):
    if not os.path.isfile(path):
        return None
    d = load_one(path)
    if not d or "psnr" not in d:
        return None
    return dict(psnr=float(d["psnr"]), l1=float(d["l1_norm"]),
                ssim=float(d["ssim"]), n_images=int(d.get("n_images", 0)),
                ckpt=d.get("ckpt", ""))


def load_stepwise(path):
    if not os.path.isfile(path):
        return None
    d = load_one(path)
    if not d or "L1_255" not in d:
        return None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--base", default="doc/2026-10-23/artifacts/n2bigstep/eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--blocks", type=int, nargs="*", default=[60, 120, 240, 360])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    base16 = load_eval(os.path.join(args.base, "linattn_k32_t16.json"))
    base1 = load_eval(os.path.join(args.base, "linattn_k32_t1.json"))
    if base16:
        rows.append(dict(M=63, **BASE63, tag="63M 基线 (10-23)", **base16,
                         psnr_t1=base1["psnr"] if base1 else None))
    for M in args.blocks:
        ed, eh, dd, dh = ARMS[M]
        ev16 = load_eval(os.path.join(args.eval, f"linattn_m{M}_t16.json"))
        ev1 = load_eval(os.path.join(args.eval, f"linattn_m{M}_t1.json"))
        if not ev16:
            continue
        # 计参从 json 的 params 字段读（更准）
        pc = {}
        p = os.path.join(args.eval, f"linattn_m{M}_t16.json")
        d = load_one(p)
        if d and "params" in d:
            pc = d["params"]
        rows.append(dict(M=M, enc_M=pc.get("enc_M", 0), dec_M=pc.get("dec_M", 0),
                         total_M=pc.get("total_M", 0), enc_dim=ed, enc_heads=eh,
                         dec_dim=dd, dec_heads=dh, tag=f"{M}M 档",
                         **ev16, psnr_t1=(ev1["psnr"] if ev1 else None)))
    rows.sort(key=lambda r: r["total_M"])

    # ── 表 ──
    lines = ["# 容量阶梯：enc/dec 各 60/120/240/360M（n=2, T=16, K=32）", "",
             "> 协议: A 相 T=1 45ep/bs128/lr2e-4 → B 相 T=16 `fixed` n=2 6ep/等效bs64/lr2e-4; "
             "深度钉死 enc 13 / dec 7, 只加宽 dim; 全量 5,000 张 val; 单种子。", "",
             "| 档 | enc 参数 | dec 参数 | 合计 | T=16 PSNR | T=16 L1_norm | T=16 SSIM | "
             "T=1 PSNR(A 相) | ΔPSNR(T=16 − 基线) |",
             "|---|---|---|---|---|---|---|---|---|"]
    base_psnr = next((r["psnr"] for r in rows if r["M"] == 63), None)
    for r in rows:
        d = (f"{r['psnr'] - base_psnr:+.3f}" if base_psnr is not None else "—")
        t1 = f"{r['psnr_t1']:.3f}" if r.get("psnr_t1") else "—"
        lines.append(f"| {r['tag']} | {r['enc_M']:.2f}M | {r['dec_M']:.2f}M | "
                     f"{r['total_M']:.1f}M | {r['psnr']:.3f} | {r['l1']:.4f} | "
                     f"{r['ssim']:.4f} | {t1} | {d} |")
    tbl = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "scale_table.md"), "w") as f:
        f.write(tbl)
    print(tbl)

    # ── 图 1: 容量–质量 ──
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    xs = [r["total_M"] for r in rows]
    for ax, key, ylab, ttl in (
            (axes[0], "psnr", "PSNR (dB)", "T=16 (n=2) end-step PSNR"),
            (axes[1], "l1", "L1_norm (lower better)", "T=16 (n=2) end-step L1"),
            (axes[2], "ssim", "SSIM (higher better)", "T=16 (n=2) end-step SSIM")):
        ax.plot(xs, [r[key] for r in rows], "o-", color="#C44E52")
        for r in rows:
            ax.annotate(f"{r['total_M']:.0f}M", (r["total_M"], r[key]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.0f}" for x in xs], fontsize=8)
        ax.set_xlabel("total params (M, encoder+decoder; each ≈ half)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out_dir, f"scale_params_vs_quality.{ext}"), dpi=200)
    plt.close()

    # ── 图 2: 逐步 L1_255 ──
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    colors = ["#8C8C8C", "#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    steps = []
    for i, r in enumerate(rows):
        if r["M"] == 63:
            p = os.path.join(args.base, "stepwise_k32_t16.json")
        else:
            p = os.path.join(args.eval, f"stepwise_m{r['M']}_t16.json")
        sw = load_stepwise(p)
        if not sw:
            continue
        L = sw["L1_255"]
        steps.append((r, sw))
        ax.plot(range(1, len(L) + 1), L, "-o", ms=3, color=colors[i % len(colors)],
                label=f"{r['total_M']:.0f}M  final={L[-1]:.2f}")
    ax.set_xlabel("decoding step t (n=2 tokens/step)")
    ax.set_ylabel("L1 (0-255 pixel scale; lower better)")
    ax.set_title("stepwise reconstruction error vs capacity (n=2, T=16)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out_dir, f"scale_stepwise_l1.{ext}"), dpi=200)
    plt.close()

    summary = {"arms": [{k: v for k, v in r.items()} for r in rows],
               "note": "enc/dec 各 60/120/240/360M（深度钉死 13/7, 宽度 scaling）; "
                       "实际总参数 = 119.3/238.2/481.5/720.6M; "
                       "基线 63M 来自 2026-10-23 n2bigstep 的 K=32/T=16 臂（同协议）"}
    with open(os.path.join(args.out_dir, "scale_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print(f"[out] {args.out_dir}/scale_table.md, scale_summary.json, 2 张图")


if __name__ == "__main__":
    main()
