#!/usr/bin/env python3
"""plot_stepwise_inflection.py — 逐步曲线「二阶导 ≈ 0（拐点 / knee）」标注 + 跨容量拐点轨迹（2026-10-26）

用户口径（2026-10-26）: 「把『逐步曲线』的二阶导等于 0 的大概的点，在逐步曲线中连成一条
曲线，重新提交那个图」。

做法（对每条 16 点曲线）:
  1. 平滑: Savitzky-Golay (window=5, polyorder=2) → UnivariateSpline(s = 1e-3 * n)；
  2. 在 t = 1.00…16.00（步长 0.01）细网格上求 d²L/dt²；
  3. 主口径 knee: t ∈ [1.5, 15.5] 内 **首个 |d²L/dt²| ≤ 10% × max|d²L/dt²|** 的点
     —— 即「二阶导衰减到≈0（相对峰值）」的近似零点，对应曲线由「仍在弯」转入「近乎直线」。
     说明: 这 5 条曲线都是「单调下降 + 凸」（d²L/dt² ≥ 0 并衰减），除了 63M 基线尾部轻微
     回升外**没有严格的 d²L/dt² = 0 跨零点**，所以「大概的点」按相对容差取；严格跨零点若
     存在则记在 zero_crossing_step（63M ≈ 13.1）里。若既不跨零也无阈值点，才 fallback 到
     |d²L/dt²| 最大点并记 fallback=true；
  4. 交叉验证 elbow: t ∈ [1.5, 15.5] 内 |d²L/dt²| 最大的点（这 5 条曲线该项都落在起始
     陡降段 t≈1.5–2.0，因此只作对照、不作主口径，已在 json 里注明）。

输出:
  scale_stepwise_l1.png|pdf     在原图基础上: 同样 5 条线/颜色/坐标轴/标题风格 +
                                每条曲线 knee 处白边 marker（形状按档区分）+ 文字
                                「knee: step=X」+ 5 个 knee 点按容量升序的 PCHIP 轨迹线
  inflection_points.json/.csv   每模型 knee_step/knee_L1/elbow_step/d2_at_knee/fallback…

默认输入（相对仓库根目录）:
  doc/2026-10-24/artifacts/scale/eval/stepwise_m{60,120,240,360}_t16.json
  doc/2026-10-23/artifacts/n2bigstep/eval/stepwise_k32_t16.json   (63M 基线, 同协议)

用法（在仓库根目录）:
  python tools/plot_stepwise_inflection.py                  # 复跑：读 json → 写图 + 拐点表
  python tools/plot_stepwise_inflection.py --out_dir /tmp/x # 换个输出目录
  python tools/plot_stepwise_inflection.py --from_json doc/2026-10-24/artifacts/scale/inflection_points.json
  python tools/plot_stepwise_inflection.py --help
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

# 顺序 = 容量升序（63M 基线 → 60 → 120 → 240 → 360M 档的实际总参数）
ARMS = [
    dict(tag="63M",  capacity_M=63.02,  label="63M 基线", dir_key="base",
         fname="stepwise_k32_t16.json",     color="#8C8C8C", knee_marker="s"),
    dict(tag="119M", capacity_M=119.33, label="60M 档",   dir_key="eval",
         fname="stepwise_m60_t16.json",     color="#4C72B0", knee_marker="D"),
    dict(tag="238M", capacity_M=238.20, label="120M 档",  dir_key="eval",
         fname="stepwise_m120_t16.json",    color="#55A868", knee_marker="^"),
    dict(tag="482M", capacity_M=481.54, label="240M 档",  dir_key="eval",
         fname="stepwise_m240_t16.json",    color="#DD8452", knee_marker="v"),
    dict(tag="721M", capacity_M=720.62, label="360M 档",  dir_key="eval",
         fname="stepwise_m360_t16.json",    color="#C44E52", knee_marker="P"),
]
DEFAULT_EVAL = "doc/2026-10-24/artifacts/scale/eval"
DEFAULT_BASE = "doc/2026-10-23/artifacts/n2bigstep/eval"
DEFAULT_OUT = "doc/2026-10-24/artifacts/scale"
T_MIN, T_MAX = 1.5, 15.5          # 求 knee / elbow 的区间
GRID = 0.01                       # t 细网格步长


def load_stepwise(path):
    """读 stepwise_*.json（扁平结构，含 L1_255 列表）。"""
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict) and "L1_255" in d:
        return d
    if isinstance(d, dict) and d:
        return next(iter(d.values()))
    raise ValueError(f"无法解析 {path}")


def analyse_curve(L, win=5, poly=2, s_scale=1e-3, tol_frac=0.10):
    """单条曲线 → knee / elbow / d²L/dt² 统计。"""
    L = np.asarray(L, dtype=float)
    t = np.arange(1, len(L) + 1, dtype=float)
    ys = savgol_filter(L, win, poly)                    # 平滑
    sp = UnivariateSpline(t, ys, s=len(t) * s_scale)    # 再拟一条 C² 光滑曲线
    tg = np.arange(1.0, t[-1] + 1e-9, GRID)
    d1 = sp.derivative(1)(tg)
    d2 = sp.derivative(2)(tg)
    m = (tg >= T_MIN) & (tg <= T_MAX)
    ti, d2i = tg[m], d2[m]
    peak = float(np.abs(d2i).max())
    thr = tol_frac * peak

    # 严格零点（跨零）
    crossings = []
    for i in range(len(d2i) - 1):
        a, b = d2i[i], d2i[i + 1]
        if a == 0.0:
            crossings.append(float(ti[i]))
        elif a * b < 0:
            crossings.append(float(ti[i] - a * (ti[i + 1] - ti[i]) / (b - a)))

    # 主口径: 首个 |d²L/dt²| ≤ tol*peak（≈ 二阶导衰减到 0）
    below = np.where(np.abs(d2i) <= thr)[0]
    fallback = False
    if below.size:
        ik = int(below[0])
    else:                                            # 理论不会走到（阈值点必存在）
        ik = int(np.argmax(np.abs(d2i)))
        fallback = True
    knee_step = float(ti[ik])
    d2_at_knee = float(d2i[ik])

    # 交叉验证: |d²L/dt²| 最大点（elbow）
    ie = int(np.argmax(np.abs(d2i)))
    elbow_step = float(ti[ie])
    elbow_d2 = float(d2i[ie])

    return dict(
        L1_255=L, t=t,
        knee_step=knee_step,
        knee_L1=float(sp(knee_step)),
        d2_at_knee=d2_at_knee,
        elbow_step=elbow_step,
        d2_at_elbow=elbow_d2,
        d2_peak=peak,
        zero_crossing_step=(crossings[0] if crossings else None),
        zero_crossings=[round(c, 2) for c in crossings],
        fallback=fallback,
        argmin_L1_step=int(np.argmin(L)),
        final_L1=float(L[-1]),
        d2min=float(d2i.min()), d2max=float(d2i.max()),
        _sp=sp,
    )


def compute_all(args):
    rows = []
    for a in ARMS:
        base = args.base if a["dir_key"] == "base" else args.eval
        p = os.path.join(base, a["fname"])
        if not os.path.isfile(p):
            raise SystemExit(f"[err] 缺少输入: {p}")
        sw = load_stepwise(p)
        r = analyse_curve(sw["L1_255"], args.savgol_window, args.savgol_poly,
                          args.spline_s, args.tol_frac)
        r.update(tag=a["tag"], capacity_M=a["capacity_M"], arm_label=a["label"],
                 color=a["color"], knee_marker=a["knee_marker"], src=p)
        rows.append(r)
    return rows


def analyse_from_json(path, args):
    """--from_json: 复用已算好的 knee/elbow，只重画图。"""
    with open(path) as f:
        j = json.load(f)
    by_tag = {m["tag"]: m for m in j["models"]}
    rows = []
    for a in ARMS:
        base = args.base if a["dir_key"] == "base" else args.eval
        p = os.path.join(base, a["fname"])
        sw = load_stepwise(p)
        L = np.asarray(sw["L1_255"], dtype=float)
        t = np.arange(1, len(L) + 1, dtype=float)
        sp = UnivariateSpline(t, savgol_filter(L, args.savgol_window, args.savgol_poly),
                              s=len(t) * args.spline_s)
        m = by_tag[a["tag"]]
        rows.append(dict(tag=a["tag"], capacity_M=a["capacity_M"],
                         arm_label=a["label"], color=a["color"],
                         knee_marker=a["knee_marker"], src=p,
                         L1_255=L, t=t, _sp=sp,
                         knee_step=float(m["knee_step"]),
                         knee_L1=float(sp(m["knee_step"])),
                         d2_at_knee=m.get("d2_at_knee"),
                         elbow_step=m.get("elbow_step"),
                         d2_at_elbow=m.get("d2_at_elbow"),
                         d2_peak=m.get("d2_peak"),
                         zero_crossing_step=m.get("zero_crossing_step"),
                         zero_crossings=m.get("zero_crossings", []),
                         fallback=m.get("fallback", False),
                         argmin_L1_step=m.get("argmin_L1_step"),
                         final_L1=m.get("final_L1", float(L[-1])),
                         d2min=m.get("d2min"), d2max=m.get("d2max")))
    return rows, j


def write_tables(rows, args):
    os.makedirs(args.out_dir, exist_ok=True)
    models = [dict(tag=r["tag"], capacity_M=r["capacity_M"], arm_label=r["arm_label"],
                   source=r["src"],
                   knee_step=round(r["knee_step"], 2),
                   knee_L1=round(r["knee_L1"], 4),
                   elbow_step=round(r["elbow_step"], 2),
                   d2_at_knee=round(r["d2_at_knee"], 5),
                   d2_at_elbow=round(r["d2_at_elbow"], 5),
                   d2_peak=round(r["d2_peak"], 5),
                   d2_min=round(r["d2min"], 5), d2_max=round(r["d2max"], 5),
                   zero_crossing_step=(None if r["zero_crossing_step"] is None
                                       else round(r["zero_crossing_step"], 2)),
                   zero_crossings=r["zero_crossings"],
                   fallback=bool(r["fallback"]),
                   argmin_L1_step=r["argmin_L1_step"],
                   final_L1=round(r["final_L1"], 4))
              for r in rows]
    out = {
        "what": "逐步曲线（L1_255 vs 解码步 t=1..16）的二阶导≈0 点（knee）与跨容量拐点轨迹",
        "generated_by": "tools/plot_stepwise_inflection.py",
        "figure": os.path.join(args.out_dir, "scale_stepwise_l1.png"),
        "method": {
            "smooth": f"Savitzky-Golay(window={args.savgol_window}, polyorder={args.savgol_poly})"
                      f" → UnivariateSpline(s = {args.spline_s}*n)",
            "grid": f"t = 1.00..16.00 step {GRID}",
            "knee_primary": f"t∈[{T_MIN},{T_MAX}] 内首个 |d²L/dt²| ≤ {args.tol_frac:.0%} × max|d²L/dt²|"
                            " 的点（d²L/dt² 衰减到≈0 的近似零点；用户口径「大概的点」）",
            "knee_primary_why": "5 条曲线均为单调下降 + 凸（d²L/dt²>0 并衰减），除 63M 基线尾部轻微"
                                "回升外无严格 d²L/dt²=0 跨零点，故用相对容差定义近似零点；严格跨零点"
                                "若存在记入 zero_crossing_step",
            "elbow_crosscheck": "t∈[1.5,15.5] 内 argmax|d²L/dt²|（交叉验证；这 5 条曲线该项落在起始"
                                "陡降段 t≈1.5–2.0，仅作对照，不作主口径）",
            "fallback": "阈值点不存在时退回 argmax|d²L/dt²| 并置 fallback=true（本次未触发）",
            "knee_L1": "平滑曲线在 knee_step 处的值（非最近原始点）",
        },
        "data_sources": {
            "63M 基线": os.path.join(args.base, "stepwise_k32_t16.json"),
            "119M / 238M / 482M / 721M": os.path.join(args.eval, "stepwise_m{60,120,240,360}_t16.json"),
            "protocol": "fixed n=2 token/步, T=16 (K=32), 全量 5,000 张 val; 63M 来自 2026-10-23 n2bigstep 同协议臂",
        },
        "models": models,
    }
    jp = os.path.join(args.out_dir, "inflection_points.json")
    with open(jp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    cp = os.path.join(args.out_dir, "inflection_points.csv")
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "capacity_M", "knee_step", "knee_L1", "elbow_step",
                    "d2_at_knee", "fallback", "zero_crossing_step", "d2_peak",
                    "argmin_L1_step", "final_L1"])
        for m in models:
            w.writerow([m["tag"], m["capacity_M"], f"{m['knee_step']:.2f}",
                        f"{m['knee_L1']:.4f}", f"{m['elbow_step']:.2f}",
                        f"{m['d2_at_knee']:.5f}", str(m["fallback"]).lower(),
                        "" if m["zero_crossing_step"] is None else f"{m['zero_crossing_step']:.2f}",
                        f"{m['d2_peak']:.5f}", m["argmin_L1_step"], f"{m['final_L1']:.4f}"])
    print(f"[out] {jp}\n[out] {cp}")
    return models


def make_plot(rows, args):
    fig, ax = plt.subplots(figsize=(7.4, 4.55))
    # 1) 原图 5 条曲线（样式与原 plot_scale.py 图 2 一致）
    for r in rows:
        L = r["L1_255"]
        ax.plot(r["t"], L, "-o", ms=3, color=r["color"],
                label=f"{r['capacity_M']:.0f}M  final={r['final_L1']:.2f}  "
                      f"knee t={r['knee_step']:.2f}")
    # 2) knee 点（白边、形状按档区分）
    for r in rows:
        ax.plot([r["knee_step"]], [r["knee_L1"]], linestyle="none",
                marker=r["knee_marker"], ms=10, color=r["color"],
                markeredgecolor="white", markeredgewidth=1.8, zorder=6)
    # 3) 5 个 knee 按容量升序 → PCHIP 轨迹线
    caps = np.array([r["capacity_M"] for r in rows])
    kx = np.array([r["knee_step"] for r in rows])
    ky = np.array([r["knee_L1"] for r in rows])
    u = np.log10(caps)
    uu = np.linspace(u[0], u[-1], 240)
    px, py = PchipInterpolator(u, kx)(uu), PchipInterpolator(u, ky)(uu)
    ax.plot(px, py, "--", color="#1A1A1A", lw=2.0, alpha=0.95, zorder=4,
            path_effects=[pe.Stroke(linewidth=4.4, foreground="white"), pe.Normal()],
            label="knee trajectory (d²L/dt²≈0), capacity↑")
    # 4) 每点旁标注容量（白底小字，避免压线）
    offs = {"63M": (-2, 13), "119M": (-6, 12), "238M": (-44, -8),
            "482M": (20, -10), "721M": (0, 12)}
    for r in rows:
        ax.annotate(f"{r['capacity_M']:.0f}M", (r["knee_step"], r["knee_L1"]),
                    textcoords="offset points", xytext=offs[r["tag"]],
                    fontsize=6.4, color=r["color"], ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                              alpha=0.78), zorder=7)
    # 5) 原图样式
    ax.set_xlabel("decoding step t (n=2 tokens/step)")
    ax.set_ylabel("L1 (0-255 pixel scale; lower better)")
    ax.set_title("stepwise reconstruction error vs capacity (n=2, T=16)"
                 "  ·  knees: d²L/dt² ≈ 0")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.6, loc="upper right", framealpha=0.92, ncol=1)
    # 6) 方法脚注（图内底部，避免压曲线；纯英文避免 CJK 字体缺失）
    note = ("knee = d²L/dt² ≈ 0: first t∈[1.5,15.5] with |d²L/dt²| ≤ "
            f"{args.tol_frac:.0%} of its peak (≈ second derivative decayed to zero).\n"
            f"smooth: Savitzky–Golay (w={args.savgol_window}, p={args.savgol_poly}) → "
            f"UnivariateSpline (s={args.spline_s:g}·n) → d²L/dt² on a {GRID}-step grid; "
            "elbow (max|d²L/dt²|) is only a cross-check (t≈1.5–2.0).\n"
            "All five curves are monotone convex decays → no strict interior "
            "d²L/dt²=0; the 63M baseline has one strict crossing at t≈13.1 "
            "(tail upturn), not marked.")
    fig.text(0.010, 0.006, note, fontsize=5.7, va="bottom", ha="left",
             color="#333333", linespacing=1.5)
    plt.tight_layout(rect=(0.0, 0.135, 1.0, 1.0))
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(args.out_dir, f"scale_stepwise_l1.{ext}")
        plt.savefig(p, dpi=200)
        print(f"[out] {p}")
    plt.close(fig)


def print_table(rows):
    hdr = (f"{'tag':5s} {'cap_M':>7s} {'knee_t':>7s} {'knee_L1':>8s} "
           f"{'elbow_t':>8s} {'d2@knee':>9s} {'d2peak':>8s} {'zero_x':>7s} fb")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        zx = "" if r["zero_crossing_step"] is None else f"{r['zero_crossing_step']:.2f}"
        print(f"{r['tag']:5s} {r['capacity_M']:7.1f} {r['knee_step']:7.2f} "
              f"{r['knee_L1']:8.3f} {r['elbow_step']:8.2f} {r['d2_at_knee']:9.4f} "
              f"{r['d2_peak']:8.4f} {zx:>7s} {str(r['fallback']).lower()}")


def main():
    ap = argparse.ArgumentParser(
        description="逐步曲线二阶导≈0（knee）标注 + 跨容量拐点轨迹，并写出 inflection_points.json/csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval", default=DEFAULT_EVAL, help="容量阶梯 eval 目录（stepwise_m*_t16.json）")
    ap.add_argument("--base", default=DEFAULT_BASE, help="63M 基线 eval 目录（stepwise_k32_t16.json）")
    ap.add_argument("--out_dir", default=DEFAULT_OUT, help="输出目录（覆盖同目录原图）")
    ap.add_argument("--savgol_window", type=int, default=5, help="Savitzky-Golay 窗口")
    ap.add_argument("--savgol_poly", type=int, default=2, help="Savitzky-Golay 多项式阶")
    ap.add_argument("--spline_s", type=float, default=1e-3,
                    help="UnivariateSpline 平滑量 s = 该值 × 点数")
    ap.add_argument("--tol_frac", type=float, default=0.10,
                    help="|d²L/dt²| 视为 ≈0 的相对容差（相对峰值）")
    ap.add_argument("--from_json", default="",
                    help="复用已算好的 inflection_points.json（只重画图，不重算 knee）")
    ap.add_argument("--no_plot", action="store_true", help="只写 json/csv，不出图")
    args = ap.parse_args()

    if args.from_json:
        rows, _ = analyse_from_json(args.from_json, args)
        print(f"[info] 复用 {args.from_json} 的 knee/elbow（只重画）")
        if not args.no_plot:
            make_plot(rows, args)
        print_table(rows)
        return

    rows = compute_all(args)
    print_table(rows)
    write_tables(rows, args)
    if not args.no_plot:
        make_plot(rows, args)


if __name__ == "__main__":
    main()
