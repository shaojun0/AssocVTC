#!/usr/bin/env python3
"""plot_model_vs_svd.py — 模型「每步 2 个 z_s 还原」 vs SVD/PCA「每步 2 个奇异向量还原」（2026-10-23）

用户口径（2026-10-23）: "这次实验完就和奇异值对比一下，毕竟都是每次使用 2 个向量进行还原"。

两种「每步 2 个向量」的对齐方式（都画, 报告里讲清预算）:
  A. 预算对齐口径: 模型 step t 读 2t 个 z_s × 448 = **896·t** 个数;
     per-image SVD 取 rank r 存 r×(196+768+1) = **965·r** 个数
     ⇒ r = t 时 965·t ≈ 1.08× 模型预算（"1 个秩 1 分量/步 = u,v 2 个向量/步"）;
  B. 用户字面口径: 「每步 2 个向量」= SVD 每步加 2 个奇异向量 ⇒ rank r = 2t
     ⇒ 1930·t = 2.15× 模型预算（SVD 更贵, 对 SVD 有利）。
  per_patch_pca（训练集共享基, rank r/块）: 每图 196·r 个数 = 0.44× 模型（更便宜）。

图/表输出（--out_dir）:
  · model_vs_svd_stepwise.png  左: L1_255 vs 解码步 t（同样每步 2 向量）;
                               右: L1_255 vs 每图信息量（个数, 预算对齐）
  · model_vs_svd_psnr.png      同 PSNR
  · model_vs_svd_table.md      逐步对照表（模型 step t vs SVD rank t / rank 2t）

用法:
    python tools/plot_model_vs_svd.py \
        --n2_eval    doc/2026-10-23/artifacts/n2bigstep/eval \
        --lown_eval  doc/2026-10-23/artifacts/lown/eval \
        --fixedk_eval doc/2026-10-21/artifacts/fixedk/eval \
        --svd_json   doc/2026-10-23/artifacts/n2bigstep/eval/baseline_svd_per_image.json \
        --pca_json   doc/2026-10-23/artifacts/n2bigstep/eval/baseline_pca_per_patch.json \
        --out_dir    doc/2026-10-23/artifacts/n2bigstep
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 每图信息量（"个数"）口径
Z_DIM = 448          # z_s 维度
N_PATCH = 196        # 196 patch
PATCH_PX = 768       # 每 patch 3*16*16


def load_one(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict) and "L1_255" in d:
        return d                       # eval_stepwise_linattn.py 的 json 是**扁平**的
    return next(iter(d.values())) if isinstance(d, dict) and d else None


def load_stepwise(path, tag):
    if not os.path.isfile(path):
        return None
    d = load_one(path)
    if d is None:
        return None
    cum = d.get("cum_z_s") or []
    return dict(tag=tag, K=int(d["K"]), T=int(d["T"]),
                L1=[float(v) for v in d["L1_255"]],
                PSNR=[float(v) for v in d["PSNR"]],
                cum_z=[int(v) for v in cum])


def load_svd(path, tag):
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    ranks = sorted(int(k) for k in d["ranks"])
    return dict(tag=tag, mode=d["mode"], n_images=d.get("n_images"),
                rank=ranks,
                L1=[float(d["ranks"][str(r)]["L1_255"]) for r in ranks],
                PSNR=[float(d["ranks"][str(r)]["PSNR"]) for r in ranks])


def budget(model_name, r):
    """每图存储的"个数"：模型 448/向量；per_image_svd 965/秩; per_patch_pca 196/秩（+共享基）。"""
    if model_name == "svd":
        return r * (N_PATCH + PATCH_PX + 1)
    if model_name == "pca":
        return r * N_PATCH
    return r * Z_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n2_eval", required=True)
    ap.add_argument("--lown_eval", default="doc/2026-10-23/artifacts/lown/eval")
    ap.add_argument("--fixedk_eval", default="doc/2026-10-21/artifacts/fixedk/eval")
    ap.add_argument("--svd_json", default="")
    ap.add_argument("--pca_json", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    arms = []
    n2 = [
        (32, 16, "n=2, T=16 (K=32)"),
        (48, 24, "n=2, T=24 (K=48)"),
        (64, 32, "n=2, T=32 (K=64)"),
    ]
    for K, T, tag in n2:
        sw = load_stepwise(os.path.join(args.n2_eval, f"stepwise_k{K}_t{T}.json"), tag)
        if sw:
            arms.append(sw)
    t8 = [
        (args.lown_eval, 16, "n=2, T=8 (K=16)"),
        (args.lown_eval, 32, "n=4, T=8 (K=32)"),
        (args.lown_eval, 48, "n=6, T=8 (K=48)"),
        (args.fixedk_eval, 64, "n=8, T=8 (K=64)"),
    ]
    for ev, K, tag in t8:
        sw = load_stepwise(os.path.join(ev, f"stepwise_k{K}_t8.json"), tag)
        if sw:
            arms.append(sw)

    svd = load_svd(args.svd_json, "per-image SVD (oracle)")
    pca = load_svd(args.pca_json, "per-patch PCA (train basis)")

    # ── 表: 模型 step t vs SVD rank t / 2t ──
    lines = ["# 模型「每步 2 个 z_s」vs SVD/PCA「每步 2 个向量」还原（ImageNet-100 val 5,000）", "",
             "预算: 模型 step t = 2t 向量 × 448 = 896t 个数; per-image SVD rank r = "
             "r×(196+768+1) = 965r 个数（≈ 1.08×模型, r=t）; "
             "per-patch PCA rank r/块 = 196r 个数（0.44×模型, r=2t; +768×64 共享基）。", ""]
    if svd:
        rmax = max(svd["rank"])

        def cell(bl, r):
            return (f"{bl['L1'][bl['rank'].index(r)]:.3f} / "
                    f"{bl['PSNR'][bl['rank'].index(r)]:.2f}") if r in bl["rank"] else "-"

        lines += ["| 解码步 t | 模型 K=64/T=32 (896t 个数) L1/PSNR | "
                  "SVD 预算对齐 rank≈0.93t (965r≈896t) | SVD rank 2t (2.15× 预算) |",
                  "|---|---|---|---|"]
        k64 = next((a for a in arms if a["K"] == 64 and a["T"] == 32), None)
        for t in range(1, min(len(k64["L1"]) if k64 else 0, rmax // 2) + 1):
            m = f"{k64['L1'][t-1]:.3f} / {k64['PSNR'][t-1]:.2f}"
            rm = max(1, min(rmax, round(t * 896.0 / 965.0)))
            lines.append(f"| {t} | {m} | {cell(svd, rm)} (rank {rm}) | {cell(svd, 2*t)} |")

    # ── 图 1/2: 左=步数轴, 右=信息量轴 ──
    for key, ylab, fname in (("L1", "L1 (0-255 pixel scale; lower is better)",
                              "model_vs_svd_stepwise"),
                             ("PSNR", "PSNR (dB)", "model_vs_svd_psnr")):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
        # 左: vs 解码步 t（同一「每步 2 向量」口径）
        ax = axes[0]
        for a in arms:
            style = "-o" if a["T"] > 8 else "--"
            ax.plot(range(1, a["T"] + 1), a[key], style, ms=3, lw=1.4,
                    alpha=0.85, label=f"model {a['tag']}")
        if svd:
            for mult, ls in ((1, "-"), (2, ":")):
                rs = [r for r in svd["rank"] if r % mult == 0]
                xs = [r // mult for r in rs]
                ys = [svd[key][svd["rank"].index(r)] for r in rs]
                ax.plot(xs, ys, ls, color="black", lw=1.8,
                        label=f"SVD: {mult} rank(s)/step (rank={mult}t)")
        ax.set_xlabel("decoding step t  (model: 2 z_s per step;  SVD: 1-2 singular vectors per step)")
        ax.set_ylabel(ylab)
        ax.set_title("per-step curve (both add ~2 vectors per step)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.3, loc="best")
        # 右: vs 每图信息量（个数）——预算对齐
        ax = axes[1]
        for a in arms:
            xs = [c * Z_DIM for c in a["cum_z"]]
            ax.plot(xs, a[key], "-o", ms=3, lw=1.4, alpha=0.85,
                    label=f"model {a['tag']}")
        for bl, ls, col in ((svd, "-", "black"), (pca, "--", "#555555")):
            if not bl:
                continue
            xs = [budget("svd" if bl is svd else "pca", r) for r in bl["rank"]]
            ax.plot(xs, bl[key], ls, color=col, lw=1.8, label=bl["tag"])
        ax.set_xscale("log")
        ax.set_xlabel("per-image information budget (numbers stored; log scale)")
        ax.set_ylabel(ylab)
        ax.set_title("vs information budget")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.3, loc="best")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(args.out_dir, f"{fname}.{ext}"), dpi=200)
        plt.close()

    # ── PSNR 表（关键 t）──
    if svd:
        lines += ["", "**关键点 (t = 1/2/4/8/16/32)**", "",
                  "| t | 模型 K=64/T=32 PSNR | SVD rank t PSNR | SVD rank 2t PSNR |",
                  "|---|---|---|---|"]
        k64 = next((a for a in arms if a["K"] == 64 and a["T"] == 32), None)
        for t in (1, 2, 4, 8, 16, 32):
            if not k64 or t > len(k64["PSNR"]):
                continue
            g = lambda bl, r: (f"{bl['PSNR'][bl['rank'].index(r)]:.2f}"
                               if bl and r in bl["rank"] else "-")
            lines.append(f"| {t} | {k64['PSNR'][t-1]:.2f} | {g(svd, t)} | {g(svd, 2*t)} |")
    with open(os.path.join(args.out_dir, "model_vs_svd_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[out] {args.out_dir}/model_vs_svd_*.png, model_vs_svd_table.md")


if __name__ == "__main__":
    main()
