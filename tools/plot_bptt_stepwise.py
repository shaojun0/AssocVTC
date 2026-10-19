#!/usr/bin/env python3
"""BPTT 逐步折线图：线性 vs 标准 Transformer + 单步（T=1）基线
================================================================

用途
----
把「循环读出 BPTT T=8」的**逐步重建质量**画成 step 1→8 的折线，
并用一条**水平虚线**标出某个臂的**单步（T=1）**结果作为基线。
默认画两条曲线（本仓库的两条主臂）：

  * 标准 Transformer  = softmax 编码器注意力 + softmax 解码器注意力
  * 线性 Transformer  = 线性核编码器注意力 + softmax 解码器注意力（仓库默认"线性方案"）

数据来源（全部是仓库内已落盘的评测产物，不重跑模型）
----------------------------------------------------
  * 逐步曲线：`eval_stepwise_linattn.py` 的 stepwise json（留出集 n=5,000，每步逐图
    `PSNR` = 逐图 PSNR 再平均，与 eval_linattn.py 主报告同口径；`L1_255` = 0-255 空间 L1）。
  * T=1 基线：`eval_linattn.py` 的 `eval_ablation8.json`（8 臂**同进程同批次**评测）。
    `L1_255` 由归一化空间换算：`l1_norm × mean(IMAGENET_STD) × 255 = l1_norm × 0.226 × 255`。

⚠️ 预算不对称（必须随图一起说）：BPTT 臂是「单步版 ckpt_best 热启动 6 epoch / 11,406 步」，
T=1 基线是「45 epoch / 42,750 步从零」。虚线是**参考线**，不是同一预算下的配对消融。

用法
----
    python3 tools/plot_bptt_stepwise.py                    # PSNR 主图 + L1 伴随图
    python3 tools/plot_bptt_stepwise.py --metric psnr
    python3 tools/plot_bptt_stepwise.py --arms std,linear,stdlin,alllinear
    python3 tools/plot_bptt_stepwise.py --show-linear-t1   # 额外画线性臂自己的 T=1 参考线
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[1]

# 归一化 L1 → 0-255 L1 的换算常数：mean(IMAGENET_STD) × 255 = 0.226 × 255
L1_NORM_TO_255 = 0.226 * 255.0

# 四个 BPTT 臂的逐步 json 与 eval_ablation8.json 里的 T=1 ckpt key
ARMS = {
    "std": {
        "label": "Standard Transformer (softmax enc + softmax dec)",
        "short": "Standard Transformer",
        "color": "#1f4e9c",
        "marker": "o",
        "stepwise": "doc/2026-10-16/artifacts/alit/stepwise_stdattn_k64_rec.json",
        "eval_key": "stdattn_k64/ckpt_best.pt",
    },
    "linear": {
        "label": "Linear Transformer (linear enc + softmax dec)",
        "short": "Linear Transformer",
        "color": "#c8553d",
        "marker": "s",
        "stepwise": "doc/2026-10-16/artifacts/alit/stepwise_linattn_k64_rec.json",
        "eval_key": "k64/ckpt_best.pt",
    },
    "stdlin": {
        "label": "Std enc + linear dec",
        "short": "Std enc + linear dec",
        "color": "#5b8c5a",
        "marker": "^",
        "stepwise": "doc/2026-10-17/artifacts/stdlin/stepwise.json",
        "eval_key": "stdlin_k64/ckpt_best.pt",
    },
    "alllinear": {
        "label": "All-linear (linear enc + linear dec)",
        "short": "All-linear",
        "color": "#8a6fa8",
        "marker": "D",
        "stepwise": "doc/2026-10-17/artifacts/alllinear/stepwise.json",
        "eval_key": "alllinear_k64/ckpt_best.pt",
    },
}

# 四臂同图时用的紧凑图例标签（避免图例过宽）
COMPACT = {
    "std": "std enc + std dec (standard)",
    "linear": "lin enc + std dec (linear)",
    "stdlin": "std enc + lin dec",
    "alllinear": "lin enc + lin dec (all-linear)",
}

EVAL_JSON = "doc/2026-10-17/artifacts/eval_ablation8.json"

METRICS = {
    "psnr": {
        "field": "PSNR",
        "ylabel": "PSNR (dB)",
        "better_high": True,
        "baseline": lambda ev: ev["psnr"],
        "fmt": "{:.2f}",
    },
    "l1": {
        "field": "L1_255",
        "ylabel": "Pixel L1 (0-255)",
        "better_high": False,
        "baseline": lambda ev: ev["l1_norm"] * L1_NORM_TO_255,
        "fmt": "{:.2f}",
    },
}


def load(repo, arm, eval_json):
    meta = ARMS[arm]
    d = json.load(open(repo / meta["stepwise"]))
    ev = json.load(open(repo / eval_json))[meta["eval_key"]]
    return d, ev


def make_figure(repo, out_dir, arms, baseline_arm, metric, show_linear_t1,
                point_labels, tag=None):
    spec = METRICS[metric]
    many = len(arms) > 2
    curves = {}
    for arm in arms:
        d, ev = load(repo, arm, EVAL_JSON)
        curves[arm] = {"d": d, "ev": ev}
    ref = curves[baseline_arm]
    base = spec["baseline"](ref["ev"])
    unit = "dB" if metric == "psnr" else "px"
    if many:
        base_label = f"standard single-step (T=1): {base:.2f} {unit}"
    else:
        base_label = (f'{ARMS[baseline_arm]["short"]}, single-step (T=1): '
                      f'{base:.2f} {unit}')

    steps = ref["d"]["steps"]
    cum = ref["d"]["cum_z_s"]
    T = ref["d"]["T"]
    xs = list(range(1, T + 1))

    fig, ax = plt.subplots(figsize=(7.2, 5.15) if many else (7.0, 5.05))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.215, top=0.795)

    # 先定 x 轴，再建顶轴，保证上下两套 tick 严格对齐
    ax.set_xlim(0.55, 8.55)
    ax.set_xticks(xs)

    # ── 曲线 ──────────────────────────────────────────────────────────────
    for arm in arms:
        m = ARMS[arm]
        y = curves[arm]["d"][spec["field"]]
        ax.plot(xs, y, "-", color=m["color"], marker=m["marker"],
                lw=1.8 if many else 2.0, ms=5.0 if many else 5.5,
                mec="white", mew=0.8,
                label=(COMPACT[arm] if many else m["label"]), zorder=4)

    # ── 数值标注 ──────────────────────────────────────────────────────────
    # PSNR（越大越好）: 上方曲线标上、下方曲线标下，避开两条曲线之间的空隙;
    # L1（越小越好）: 全部标在点上方，避免下方标注压到 T=1 基线（基线在低值侧）。
    if point_labels:
        for i, x in enumerate(xs):
            ranked = sorted(arms, key=lambda a: curves[a]["d"][spec["field"]][i],
                            reverse=True)          # 数值大 → 上方
            for rank, arm in enumerate(ranked):
                v = curves[arm]["d"][spec["field"]][i]
                if spec["better_high"]:
                    off = (0, 9) if rank == 0 else ((0, -14) if rank == len(ranked) - 1
                                                   else (0, 9))
                else:
                    off = (0, 9)
                ax.annotate(spec["fmt"].format(v), (x, v),
                            textcoords="offset points", xytext=off,
                            ha="center", va="bottom" if off[1] > 0 else "top",
                            fontsize=7.0, color=ARMS[arm]["color"], zorder=5)

    # ── 单步基线（虚线） ──────────────────────────────────────────────────
    ax.axhline(base, ls=(0, (6, 3)), color="#2b2b2b", lw=1.5, zorder=2,
               label=base_label)

    # ── 可选：线性臂自己的 T=1 参考线（点线，便于同族比较） ────────────────
    if show_linear_t1 and "linear" in curves and baseline_arm != "linear":
        lin_base = spec["baseline"](curves["linear"]["ev"])
        ax.axhline(lin_base, ls=(0, (2, 2)), color=ARMS["linear"]["color"],
                   lw=1.3, zorder=2,
                   label=f'Linear Transformer, single-step (T=1): '
                         f'{lin_base:.2f} {"dB" if metric == "psnr" else "px"}')

    # ── 顶轴：累计读入的 z_s token 数 ─────────────────────────────────────
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(xs)
    top.set_xticklabels([str(c) for c in cum])
    top.tick_params(axis="x", labelsize=8, length=3, colors="#444")
    top.set_xlabel("cumulative $z_s$ memory tokens read", fontsize=9, labelpad=3)

    ax.set_xlabel("BPTT decoding step", fontsize=10.5, labelpad=4)
    ax.set_ylabel(spec["ylabel"] + ("  (higher is better)" if spec["better_high"]
                                    else "  (lower is better)"), fontsize=10.5)
    ax.grid(True, which="major", axis="y", alpha=0.30)
    ax.grid(True, which="major", axis="x", alpha=0.12)

    # 基线在纵轴方向的留白；四臂时给图例额外留出空间
    lo = min(min(curves[a]["d"][spec["field"]]) for a in arms)
    hi = max(max(curves[a]["d"][spec["field"]]) for a in arms)
    baselines = [base]
    if show_linear_t1 and "linear" in curves and baseline_arm != "linear":
        baselines.append(spec["baseline"](curves["linear"]["ev"]))
    span = max(hi, *baselines) - min(lo, *baselines) + 1e-9
    pad = 0.08 * span
    lo_pad = hi_pad = 1.6 * pad
    if many and spec["better_high"]:            # 图例放右下 → 底部多留白
        lo_pad += 0.05 * span
    if many and not spec["better_high"]:        # 图例放右上 → 顶部多留白
        hi_pad += 0.02 * span
    ax.set_ylim(min(lo, *baselines) - lo_pad, max(hi, *baselines) + hi_pad)

    ax.legend(loc="lower right" if metric == "psnr" else "upper right",
              fontsize=7.6 if many else 8.0, framealpha=0.92,
              borderpad=0.55, handlelength=2.6,
              ncol=2 if many else 1)

    metric_name = "PSNR (dB)" if metric == "psnr" else "pixel L1 (0-255)"
    title = ("BPTT step-wise reconstruction: encoder × decoder attention forms"
             if many else
             "BPTT step-wise reconstruction: Linear vs. Standard Transformer")
    fig.suptitle(title, y=0.985, fontsize=12.0)
    fig.text(0.5, 0.932,
             f"{metric_name}  ·  K=64 (196→64 tokens)  ·  224×224  ·  "
             "ImageNet-100 val (n=5,000)  ·  seed 0",
             ha="center", fontsize=8.2, color="#444")

    fig.text(
        0.012, 0.018,
        "Caveat — training budgets differ: BPTT arms are 6-epoch warm starts (11,406 steps)\n"
        "from their T=1 checkpoints; the T=1 baselines are 45-epoch runs from scratch (42,750\n"
        "steps). The dashed line is a reference point, NOT a paired ablation. Single seed.",
        ha="left", va="bottom", fontsize=6.8, color="#555", linespacing=1.35)

    suffix = f'bptt_stepwise_{tag or "_".join(arms)}_{metric}'
    if show_linear_t1:
        suffix += "_with_linear_t1"
    png = out_dir / f"{suffix}.png"
    pdf = out_dir / f"{suffix}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[plot] {png}")
    print(f"[plot] {pdf}")
    return curves, base, suffix


def write_data(repo, out_dir, arms, curves):
    """写逐步数据 CSV + T=1 基线 JSON（与图完全同源）。"""
    ref = curves[arms[0]]["d"]
    steps, cum = ref["steps"], ref["cum_z_s"]

    rows = []
    for i, (s, c) in enumerate(zip(steps, cum)):
        row = {"step": i + 1, "last_z_s_index": s, "cum_z_s_tokens": c}
        for arm in arms:
            row[f"{arm}_psnr_db"] = round(curves[arm]["d"]["PSNR"][i], 4)
            row[f"{arm}_l1_255"] = round(curves[arm]["d"]["L1_255"][i], 4)
        rows.append(row)
    cols = list(rows[0].keys())
    csv_path = out_dir / "bptt_stepwise_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[data] {csv_path}")

    baselines = {
        "note": "T=1 (single-step) reference points from eval_ablation8.json "
                "(same-process, same-batch, 5,000 val images). L1_255 = l1_norm × 0.226 × 255.",
        "arms": {
            arm: {
                "label": ARMS[arm]["label"],
                "eval_key": ARMS[arm]["eval_key"],
                "config": curves[arm]["ev"]["cfg"],
                "params": curves[arm]["ev"]["params"],
                "n_images": curves[arm]["ev"]["n_images"],
                "psnr_db": round(curves[arm]["ev"]["psnr"], 6),
                "l1_255": round(curves[arm]["ev"]["l1_norm"] * L1_NORM_TO_255, 6),
                "ssim": round(curves[arm]["ev"]["ssim"], 6),
            }
            for arm in arms
        },
    }
    base_path = out_dir / "bptt_stepwise_baselines.json"
    with open(base_path, "w") as f:
        json.dump(baselines, f, ensure_ascii=False, indent=1)
    print(f"[data] {base_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out-dir",
                    default=str(REPO / "doc/2026-10-19/artifacts"))
    ap.add_argument("--arms", default="std,linear",
                    help="逗号分隔，默认 std,linear；可选 stdlin,alllinear")
    ap.add_argument("--baseline-arm", default="std",
                    help="哪条臂的 T=1 结果画成虚线基线（默认 std）")
    ap.add_argument("--metric", default="both", choices=["psnr", "l1", "both"])
    ap.add_argument("--show-linear-t1", action="store_true",
                    help="额外画线性臂自己的 T=1 参考线（点线）")
    ap.add_argument("--point-labels", dest="point_labels", action="store_true",
                    default=None, help="强制在点旁标数值")
    ap.add_argument("--no-point-labels", dest="point_labels", action="store_false",
                    help="不标数值（>2 臂时默认关闭）")
    ap.add_argument("--tag", default=None,
                    help="产物文件名里的标签（默认用臂名拼接）")
    a = ap.parse_args()

    repo = Path(a.repo)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        assert arm in ARMS, f"未知臂: {arm}（可选 {list(ARMS)}）"
    point_labels = (len(arms) <= 2) if a.point_labels is None else a.point_labels

    metrics = ["psnr", "l1"] if a.metric == "both" else [a.metric]
    rows = None
    for metric in metrics:
        curves, base, suffix = make_figure(
            repo, out_dir, arms, a.baseline_arm, metric,
            a.show_linear_t1, point_labels, a.tag)
        if rows is None:
            rows = write_data(repo, out_dir, arms, curves)

    # ── 终端依次打印精确值与简报用的 markdown 表 ──────────────────────────
    print("\n=== 逐步数据（%s）===" % ",".join(arms))
    hdr = ["step", "cum_z_s"] + [f"{ar}_psnr" for ar in arms] + [f"{ar}_l1_255" for ar in arms]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join([str(r["step"]), str(r["cum_z_s_tokens"])]
                         + [f'{r[f"{ar}_psnr_db"]:.4f}' for ar in arms]
                         + [f'{r[f"{ar}_l1_255"]:.4f}' for ar in arms]))
    ev = json.load(open(repo / EVAL_JSON))
    print("\n=== 单步（T=1）基线 ===")
    for arm in arms:
        e = ev[ARMS[arm]["eval_key"]]
        print(f'{arm:10s} PSNR={e["psnr"]:.4f} dB   '
              f'L1_255={e["l1_norm"] * L1_NORM_TO_255:.4f}   SSIM={e["ssim"]:.4f}')

    print("\n=== 图内 markdown 表（PSNR dB）===")
    print("| step | cum_z_s | " + " | ".join(ARMS[ar]["short"] for ar in arms) + " |")
    print("|---:|---:|" + "---:|" * len(arms))
    for r in rows:
        print(f'| {r["step"]} | {r["cum_z_s_tokens"]} | '
              + " | ".join(f'{r[f"{ar}_psnr_db"]:.2f}' for ar in arms) + " |")
    print("\n=== 图内 markdown 表（pixel L1 0-255）===")
    print("| step | cum_z_s | " + " | ".join(ARMS[ar]["short"] for ar in arms) + " |")
    print("|---:|---:|" + "---:|" * len(arms))
    for r in rows:
        print(f'| {r["step"]} | {r["cum_z_s_tokens"]} | '
              + " | ".join(f'{r[f"{ar}_l1_255"]:.2f}' for ar in arms) + " |")


if __name__ == "__main__":
    main()
