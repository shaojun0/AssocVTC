#!/usr/bin/env python3
"""SAVOIAS 人评复杂度 × BPTT token 预算 分析（ALIT Fig.4/11/12 复刻）
=====================================================================

输入（默认都在 `doc/2026-10-19/artifacts/savoias_complexity/`）
------------------------------------------------------------
  per_image_{stdattn,linattn}_k64_rec.json
      `eval_stepwise_linattn.py --dump_per_image` 在 SAVOIAS 上的产物（820 张，
      Art/Scenes/Objects 三个子集，step 1–8）。
  gt_{art,scenes,objects}.npy
      SAVOIAS 官方 ground truth：`{index: [filename, human_complexity_0_100]}`。
      **不随仓库分发**，用 `tools/fetch_savoias.sh` 下载（见该脚本）。

输出
----
  savoias_l1_vs_complexity.png/pdf   主图：复杂度 → L1（按 token 预算）
  savoias_l1_vs_tokens.png/pdf       复杂度三分位 → L1 vs token 预算
  savoias_needed_step.png/pdf        复杂度 → 达到本臂中位质量所需步数
  savoias_complexity_summary.json    全部数字（ρ / 分箱 / needed-step）
  savoias_per_image_joined.csv       逐图 join 后的原始数值（便于二次分析）

用法
----
    python3 tools/plot_savoias_complexity.py
    python3 tools/plot_savoias_complexity.py --data-dir /path/to/artifacts --no-csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "doc/2026-10-19/artifacts/savoias_complexity"

SUBSETS = [("Art", "People-Art", "gt_art.npy"),
           ("Scenes", "Places", "gt_scenes.npy"),
           ("Objects", "MSCOCO", "gt_objects.npy")]
ARMS = [("stdattn_k64_rec", "Standard Transformer (BPTT)"),
        ("linattn_k64_rec", "Linear-encoder Transformer (BPTT)")]
TOKENS = [3, 8, 15, 24, 35, 48, 63, 64]
TICKS = [3, 8, 15, 24, 35, 48, 64]
STEPS_SHOW = [0, 1, 3, 6]          # 3 / 8 / 24 / 63 tokens
STYLES = [dict(color="#c6dbef", marker="o"),
          dict(color="#6baed6", marker="s"),
          dict(color="#2171b5", marker="^"),
          dict(color="#08306b", marker="D")]


def load_scores(data_dir, fn):
    o = np.load(data_dir / fn, allow_pickle=True).item()
    return {int(k): int(o[k][1]) for k in o}


def load_arm(data_dir, arm):
    return json.load(open(data_dir / f"per_image_{arm}.json"))


def join(arm_json, subset, scores):
    S, L, P, B, IDX = [], [], [], [], []
    for r in arm_json["images"]:
        if r["category"] != subset:
            continue
        stem = int(r["file_name"].split(".")[0])
        if stem not in scores:
            continue
        S.append(scores[stem]); L.append(r["L1_255"])
        P.append(r["PSNR"]); B.append(r["best_step"]); IDX.append(stem)
    return (np.asarray(S, float), np.asarray(L, float),
            np.asarray(P, float), np.asarray(B, int), np.asarray(IDX, int))


def qbins(x, k=5):
    qs = np.quantile(x, np.linspace(0, 1, k + 1))
    qs[0] -= 1e-9
    return np.clip(np.digitize(x, qs) - 1, 0, k - 1)


def binned(x, y, k=5):
    idx = qbins(x, k)
    xs, ys, es = [], [], []
    for b in range(k):
        m = idx == b
        if not m.any():
            continue
        xs.append(x[m].mean())
        ys.append(y[m].mean())
        es.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
    return np.array(xs), np.array(ys), np.array(es)


def needed_step(L, tau):
    need = np.full(len(L), L.shape[1], float)
    for i, row in enumerate(L):
        hit = np.where(row <= tau)[0]
        if len(hit):
            need[i] = hit[0] + 1
    return need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-csv", action="store_true")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    out = Path(a.out_dir) if a.out_dir else data_dir
    out.mkdir(parents=True, exist_ok=True)

    scores = {sub: load_scores(data_dir, fn) for sub, _, fn in SUBSETS}
    summary = {"note": "SAVOIAS human complexity (0-100) vs BPTT token budget; "
                       "model trained on ImageNet-100, SAVOIAS is OOD.",
               "subsets": {sub: {"pretty": pretty, "n_scored": len(scores[sub])}
                           for sub, pretty, _ in SUBSETS},
               "arms": {}}
    csv_rows = []

    for arm, arm_label in ARMS:
        aj = load_arm(data_dir, arm)
        summary["arms"][arm] = {"label": arm_label, "subsets": {}}
        for sub, pretty, _ in SUBSETS:
            S, L, P, B, IDX = join(aj, sub, scores[sub])
            tau = float(np.median(L[:, -1]))
            need = needed_step(L, tau)
            rho_l1 = [float(spearmanr(S, L[:, t]).correlation) for t in range(8)]
            rho_psnr = [float(spearmanr(S, P[:, t]).correlation) for t in range(8)]
            rho_need = float(spearmanr(S, need).correlation)
            q1, q2 = np.percentile(S, [100 / 3, 200 / 3])
            ter = {}
            for name, m in [("low", S <= q1), ("mid", (S > q1) & (S <= q2)),
                            ("high", S > q2)]:
                gain = float(L[m, 0].mean() - L[m, 6].mean())
                ter[name] = {"n": int(m.sum()),
                             "l1": [float(L[m, t].mean()) for t in range(8)],
                             "gain_3_to_63_tokens": gain}
            summary["arms"][arm]["subsets"][sub] = {
                "pretty": pretty, "n": int(len(S)),
                "score_mean": float(S.mean()), "score_std": float(S.std()),
                "score_range": [int(S.min()), int(S.max())],
                "rho_l1_per_step": rho_l1, "rho_psnr_per_step": rho_psnr,
                "rho_score_vs_needed_step": rho_need,
                "needed_step_mean": float(need.mean()),
                "needed_step_std": float(need.std()),
                "terciles": ter,
                "mean_l1_per_step": [float(L[:, t].mean()) for t in range(8)],
                "median_final_l1": tau,
            }
            if not a.no_csv:
                for i in range(len(S)):
                    row = {"arm": arm, "subset": sub, "index": int(IDX[i]),
                           "complexity": int(S[i]), "best_step_1based": int(B[i]) + 1}
                    for t in range(8):
                        row[f"L1_255_t{t+1}"] = round(float(L[i, t]), 4)
                        row[f"PSNR_t{t+1}"] = round(float(P[i, t]), 4)
                    csv_rows.append(row)
            print(f"[{arm}] {pretty:11s} n={len(S):4d} "
                  f"ρ(L1@3tok)={rho_l1[0]:+.3f} ρ(L1@63tok)={rho_l1[6]:+.3f} "
                  f"ρ(needed)={rho_need:+.3f} need={need.mean():.2f}±{need.std():.2f}")

    # ── 图 1：复杂度 → L1（按 token 预算），std 臂 3 个子集 ─────────────────
    aj = load_arm(data_dir, "stdattn_k64_rec")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharey=True)
    for ax, (sub, pretty, _) in zip(axes, SUBSETS):
        S, L, _, _, _ = join(aj, sub, scores[sub])
        for si, st in enumerate(STEPS_SHOW):
            xs, ys, es = binned(S, L[:, st], k=6)
            ax.errorbar(xs, ys, yerr=es, fmt=f"-{STYLES[si]['marker']}",
                        color=STYLES[si]["color"], ms=4.5, lw=1.8, capsize=2,
                        label=f"{TOKENS[st]} tokens")
        rho0 = spearmanr(S, L[:, 0]).correlation
        rho6 = spearmanr(S, L[:, 6]).correlation
        ax.set_title(f"{pretty}\n(human complexity 0–100, n={len(S)})", fontsize=10)
        ax.set_xlabel("human-annotated image complexity", fontsize=10)
        ax.set_xlim(5, 92)
        ax.grid(alpha=.28)
        ax.text(0.03, 0.97, f"ρ@3tok={rho0:+.2f}\nρ@63tok={rho6:+.2f}",
                transform=ax.transAxes, va="top", fontsize=8, color="#333")
    axes[0].set_ylabel("pixel L1 (0–255)", fontsize=10.5)
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.80),
                   framealpha=.9)
    fig.suptitle("More complex images need more tokens — Standard Transformer (BPTT), "
                 "trained on ImageNet-100, evaluated on SAVOIAS (OOD)", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "savoias_l1_vs_complexity.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "savoias_l1_vs_complexity.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── 图 2：复杂度三分位 → L1 vs token 数 ────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.6), sharex=True, sharey="row")
    for ri, (arm, arm_label) in enumerate(ARMS):
        aj = load_arm(data_dir, arm)
        for ci, (sub, pretty, _) in enumerate(SUBSETS):
            ax = axes[ri, ci]
            S, L, _, _, _ = join(aj, sub, scores[sub])
            q1, q2 = np.percentile(S, [100 / 3, 200 / 3])
            for name, m, c in [("low", S <= q1, "#9ecae1"),
                               ("mid", (S > q1) & (S <= q2), "#4292c6"),
                               ("high", S > q2, "#08519c")]:
                mu = L[m].mean(0)
                se = L[m].std(0, ddof=1) / np.sqrt(m.sum())
                ax.plot(TOKENS, mu, "-o", color=c, ms=4, lw=1.8,
                        label=f"complexity {name} (n={m.sum()})")
                ax.fill_between(TOKENS, mu - se, mu + se, color=c, alpha=.18)
            ax.set_xscale("log")
            ax.set_xticks(TICKS)
            ax.set_xticklabels([str(t) for t in TICKS], fontsize=7.5)
            ax.grid(alpha=.28)
            if ri == 0:
                ax.set_title(pretty, fontsize=10)
            if ci == 0:
                short = "Standard (BPTT)" if arm.startswith("std") else "Linear enc (BPTT)"
                ax.set_ylabel(f"{short}\npixel L1 (0–255)", fontsize=9)
            if ri == 1:
                ax.set_xlabel("cumulative memory tokens read", fontsize=10)
            ax.legend(fontsize=7.2, loc="upper right", framealpha=.9)
    fig.suptitle("Quality vs. token budget, split by human-annotated complexity "
                 "(SAVOIAS, OOD for an ImageNet-100 model)", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "savoias_l1_vs_tokens.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "savoias_l1_vs_tokens.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── 图 3：复杂度 → 所需步数（阈值 = 各臂自己的最终中位 L1）─────────────
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharey=True)
    for ax, (sub, pretty, _) in zip(axes, SUBSETS):
        for arm, arm_label in ARMS:
            aj = load_arm(data_dir, arm)
            S, L, _, _, _ = join(aj, sub, scores[sub])
            need = needed_step(L, float(np.median(L[:, -1])))
            xs, ys, es = binned(S, need, k=6)
            c = "#1f4e9c" if arm.startswith("std") else "#c8553d"
            ax.errorbar(xs, ys, yerr=es, fmt="-o", color=c, ms=4, lw=1.6,
                        capsize=2, label=arm_label.split(" (")[0])
        ax.set_title(pretty, fontsize=10)
        ax.set_xlabel("human-annotated image complexity", fontsize=10)
        ax.set_xlim(5, 92)
        ax.grid(alpha=.28)
    axes[0].set_ylabel("steps needed to reach\nmedian final L1", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Complex images need more decoding steps (SAVOIAS, OOD)", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "savoias_needed_step.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "savoias_needed_step.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(out / "savoias_complexity_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    if csv_rows:
        fields = list(csv_rows[0].keys())
        with open(out / "savoias_per_image_joined.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(csv_rows)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
