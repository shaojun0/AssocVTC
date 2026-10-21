#!/usr/bin/env python
"""汇总/作图：K=64 rFID（ALIT 口径）—— 纯 json, 不加载模型（2026-10-21）

输入：`<eval_dir>/<tag>/eval_rfid.json`（tools/eval_rfid.py 的产物）
输出：
  · rfid_k64_table.md      K=64 三臂主表 + ALIT 对齐列
  · rfid_vs_tokens.png/pdf  rFID（与 PSNR）vs token 预算曲线
  · rfid_k64_summary.json  全部数字（报告引用的唯一数据源）
  · rfid_step_table.md     原生 square K=64 模型逐步表（3…64 token, 与 ALIT 同构）
"""
import argparse
import json
import os

import numpy as np


def load_jsons(eval_dir: str) -> dict:
    arms = {}
    for name in sorted(os.listdir(eval_dir)):
        p = os.path.join(eval_dir, name, "eval_rfid.json")
        if os.path.isfile(p):
            with open(p) as f:
                arms[name] = json.load(f)
    assert arms, f"{eval_dir} 下没有 */eval_rfid.json"
    return arms


def arm_last(arm: dict):
    """取该臂 evaluable 的末步（K=64 主口径）。"""
    steps = arm["evaluated_steps"]
    return arm["per_step"][str(steps[-1])]


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k64_only", action="store_true",
                    help="只处理 K=64 三臂（缺 K-sweep 臂时不报错）")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    arms = load_jsons(args.eval_dir)
    summary = {"arms": {}}

    # ── 1) 汇总每臂 ────────────────────────────────────────────────────────
    for tag, arm in arms.items():
        last = arm_last(arm)
        summary["arms"][tag] = {
            "K": arm["K"], "T": arm["T"],
            "dec_step_plan": arm.get("dec_step_plan"),
            "cum_z_s": arm["cum_z_s"],
            "n_images": arm["n_images"],
            "psnr_last": last["psnr"],
            "l1_255_last": last["l1_255"],
            "l1_norm_last": last["l1_norm"],
            "rfid_last": last.get("rfid"),
            "rfid_by_step": {str(s): d.get("rfid") for s, d in arm["per_step"].items()},
            "psnr_by_step": {str(s): d["psnr"] for s, d in arm["per_step"].items()},
            "cum_z_by_step": {str(s): d["cum_z_s"] for s, d in arm["per_step"].items()},
        }

    # ── 2) K=64 主表 ───────────────────────────────────────────────────────
    rows = []
    k64_tags = ["native_k64", "k64_t8", "k64_t1"]
    plan_desc = {
        "native_k64": "原生 square 计划（递增; ALIT 同构）",
        "k64_t8": "cumulative（square 形状缩放, 前少后多）",
        "k64_t1": "cumulative 的 A 相（T=1 单步读全 64）",
    }
    for tag in k64_tags:
        if tag not in summary["arms"]:
            continue
        a = summary["arms"][tag]
        last_step = max(int(s) for s in a["rfid_by_step"])
        rows.append({
            "arm": tag, "plan": plan_desc.get(tag, a.get("dec_step_plan")),
            "T": a["T"], "tokens": a["cum_z_by_step"][str(last_step)],
            "psnr": a["psnr_last"], "l1_255": a["l1_255_last"],
            "rfid": a["rfid_last"],
        })
    with open(os.path.join(args.out_dir, "rfid_k64_table.md"), "w") as f:
        f.write("| 臂 | 读窗口计划 | T | token 预算 | PSNR (dB) ↑ | L1_255 ↓ | rFID ↓ |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| `{r['arm']}` | {r['plan']} | {r['T']} | {r['tokens']} | "
                    f"{fmt(r['psnr'], 3)} | {fmt(r['l1_255'], 3)} | {fmt(r['rfid'])} |\n")
    summary["k64_rows"] = rows

    # ── 3) 原生 square K=64 逐步表（ALIT 的 token 预算轴同构） ─────────────
    if "native_k64" in arms:
        nv = summary["arms"]["native_k64"]
        with open(os.path.join(args.out_dir, "rfid_step_table.md"), "w") as f:
            f.write("| 步 | 累计 token（cum_z_s） | PSNR (dB) ↑ | L1_255 ↓ | rFID ↓ |\n")
            f.write("|---|---|---|---|---|\n")
            for s in arms["native_k64"]["evaluated_steps"]:
                d = arms["native_k64"]["per_step"][str(s)]
                f.write(f"| {s} | {d['cum_z_s']} | {fmt(d['psnr'], 3)} | "
                        f"{fmt(d['l1_255'], 3)} | {fmt(d.get('rfid'))} |\n")
        summary["native_step_curve"] = {
            "tokens": [arms["native_k64"]["per_step"][str(s)]["cum_z_s"]
                       for s in arms["native_k64"]["evaluated_steps"]],
            "rfid": [arms["native_k64"]["per_step"][str(s)].get("rfid")
                     for s in arms["native_k64"]["evaluated_steps"]],
            "psnr": [arms["native_k64"]["per_step"][str(s)]["psnr"]
                     for s in arms["native_k64"]["evaluated_steps"]],
            "l1_255": [arms["native_k64"]["per_step"][str(s)]["l1_255"]
                       for s in arms["native_k64"]["evaluated_steps"]],
        }

    # ── 4) cumulative K-sweep 曲线（跨模型 K 对比; 可选） ──────────────────
    cum_pts = []
    for tag, a in summary["arms"].items():
        if tag.startswith("k") and tag.endswith("_t8") and tag[1:-3].isdigit():
            cum_pts.append((a["K"], a["psnr_last"], a["l1_255_last"], a["rfid_last"], tag))
    cum_pts.sort()
    summary["cum_sweep"] = [{"K": k, "psnr": p, "l1_255": l, "rfid": r, "arm": t}
                            for k, p, l, r, t in cum_pts]
    if cum_pts:
        with open(os.path.join(args.out_dir, "rfid_ksweep_table.md"), "w") as f:
            f.write("| K (token) | 臂 | PSNR (dB) ↑ | L1_255 ↓ | rFID ↓ |\n")
            f.write("|---|---|---|---|---|\n")
            for k, p, l, r, t in cum_pts:
                f.write(f"| {k} | `{t}` | {fmt(p, 3)} | {fmt(l, 3)} | {fmt(r)} |\n")

    # ── 5) 图：rFID（与 PSNR）vs token 预算 ──────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plot] 无 matplotlib, 跳作图: {e}")
        plt = None
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=200)
        ax = axes[0]
        if cum_pts:
            ks = [p[0] for p in cum_pts]
            fr = [p[3] for p in cum_pts]
            ax.plot(ks, fr, "o-", color="#1f77b4", label="cumulative sweep (T=8)")
        if "native_k64" in arms:
            sc = summary["native_step_curve"]
            ax.plot(sc["tokens"], sc["rfid"], "s--", color="#d62728",
                    label="native square K=64, early-stop")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("token budget K (cum_z_s)")
        ax.set_ylabel("rFID ↓")
        ax.set_title("rFID vs token budget (IN-100 val 5k, 224²)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1]
        if cum_pts:
            ax.plot([p[0] for p in cum_pts], [p[1] for p in cum_pts],
                    "o-", color="#1f77b4", label="cumulative sweep (T=8)")
        if "native_k64" in arms:
            sc = summary["native_step_curve"]
            ax.plot(sc["tokens"], sc["psnr"], "s--", color="#d62728",
                    label="native square K=64, early-stop")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("token budget K (cum_z_s)")
        ax.set_ylabel("PSNR (dB) ↑")
        ax.set_title("PSNR vs token budget")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = os.path.join(args.out_dir, f"rfid_vs_tokens.{ext}")
            fig.savefig(p, bbox_inches="tight")
            print(f"[plot] {p}")

    with open(os.path.join(args.out_dir, "rfid_k64_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"[summary] → {os.path.join(args.out_dir, 'rfid_k64_summary.json')}")
    print("\n".join(open(os.path.join(args.out_dir, "rfid_k64_table.md")).read().splitlines()))


if __name__ == "__main__":
    main()
