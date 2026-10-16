#!/usr/bin/env python3
"""B 阶段臂汇总表（从 eval_*.json + stepwise_*.json 重建，不依赖 alit_b_arms.tsv）
================================================================================
用法::

    python doc/2026-10-16/artifacts/alit/collect_b_arms.py \
      --dir doc/2026-10-16/artifacts/alit --baseline b1_mean

产出 `b_arms_table.json` + 打印 Lead 要求的固定格式行::

    臂名 | PSNR | L1_255 step7/step8 | Δ₁% | 最优停步众数 | vs b1_mean 末步 ΔL1 | 判决位置

说明:
- "末步" = T=8 协议下的**交付步 = step8**（`F_hat` 即该步），但 PLAN 语境里的"末步"
  也常指 step7（最优停步）；本表两个都列，ΔL1 以 **step8（交付步）** 为主、step7 附列。
- 判决文本仅供索引，最终三态判决见 alit-report 的 `report_tables.md`。
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="doc/2026-10-16/artifacts/alit")
    ap.add_argument("--baseline", default="b1_mean")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    A = args.dir

    rows = {}
    for p in sorted(glob.glob(os.path.join(A, "eval_*.json"))):
        arm = os.path.basename(p)[len("eval_"):-len(".json")]
        try:
            d = json.load(open(p))
        except Exception:
            continue
        own = next((v for k, v in d.items() if k.startswith(arm + "/")), None)
        if own is None:
            continue
        sw = None
        sp = os.path.join(A, f"stepwise_{arm}.json")
        if os.path.exists(sp):
            try:
                s = json.load(open(sp))
                sw = {"L1_255": s["L1_255"], "PSNR": s.get("PSNR"),
                      "hist": s.get("best_step_hist"),
                      "delta1_share": s.get("delta1_share")}
            except Exception:
                sw = None
        rows[arm] = {"psnr": own["psnr"], "ssim": own["ssim"],
                     "l1_norm": own["l1_norm"], "n_images": own["n_images"],
                     "step": own.get("step"), "epoch": own.get("epoch"),
                     "loss_weights": own.get("cfg", {}).get("loss_weights"),
                     "refiner_layers": own.get("cfg", {}).get("refiner_layers"),
                     "dh_mode": own.get("cfg", {}).get("dh_mode"),
                     "sample_grad_iters": own.get("cfg", {}).get("sample_grad_iters"),
                     "params_total": own.get("params", {}).get("total"),
                     "stepwise": sw}

    base = rows.get(args.baseline)
    # 第一遍: 为所有臂抽取 step7/step8/Δ₁/众数（**必须在算 delta 之前全做完**,
    # 否则基准臂按字母序排在最后时 delta 会取不到值）
    for arm, r in rows.items():
        sw = r.get("stepwise") or {}
        L = sw.get("L1_255") or []
        r["L1_255_step7"] = L[6] if len(L) > 6 else None
        r["L1_255_step8"] = L[7] if len(L) > 7 else None
        r["delta1_share"] = sw.get("delta1_share")
        h = sw.get("hist")
        if h:
            tot = sum(h) or 1
            r["mode_step"] = int(max(range(len(h)), key=lambda i: h[i])) + 1
            r["mode_share"] = h[r["mode_step"] - 1] / tot
    # 第二遍: 与基线比较
    if base:
        for arm, r in rows.items():
            r["vs_baseline_psnr"] = r["psnr"] - base["psnr"]
            if r["L1_255_step8"] is not None and base.get("L1_255_step8") is not None:
                r["vs_baseline_step8_dL1"] = r["L1_255_step8"] - base["L1_255_step8"]
            if r["L1_255_step7"] is not None and base.get("L1_255_step7") is not None:
                r["vs_baseline_step7_dL1"] = r["L1_255_step7"] - base["L1_255_step7"]

    order = ["b1_mean", "b1_decay", "b1_increasing", "b1_lastonly", "b3_grow",
             "b2_dh1", "b2_dh2", "b2_dh3", "b4_sg", "b1_increasing_seed1",
             "b1_mean_nockpt"]
    keys = [k for k in order if k in rows] + [k for k in sorted(rows) if k not in order]

    print(f"{'arm':<18} | {'PSNR':>8} | {'s7':>7}/{'s8':>7} | {'Δ1%':>6} | "
          f"{'mode':>9} | {'Δs8 vs base':>11} | {'Δs7 vs base':>11} | w/refiner/dh/sgi")
    for k in keys:
        r = rows[k]
        s7 = f"{r['L1_255_step7']:.4f}" if r.get("L1_255_step7") is not None else "   -   "
        s8 = f"{r['L1_255_step8']:.4f}" if r.get("L1_255_step8") is not None else "   -   "
        d1 = f"{r['delta1_share']*100:.2f}" if r.get("delta1_share") is not None else "  -  "
        mode = (f"step{r['mode_step']} {r['mode_share']*100:.1f}%"
                if r.get("mode_step") else "  -  ")
        ds8 = (f"{r['vs_baseline_step8_dL1']:+.4f}"
               if r.get("vs_baseline_step8_dL1") is not None else "  -  ")
        ds7 = (f"{r['vs_baseline_step7_dL1']:+.4f}"
               if r.get("vs_baseline_step7_dL1") is not None else "  -  ")
        cfg = f"{r.get('refiner_layers')}/{r.get('dh_mode')}/{r.get('sample_grad_iters')}"
        print(f"{k:<18} | {r['psnr']:>8.4f} | {s7}/{s8} | {d1:>6} | {mode:>9} | "
              f"{ds8:>11} | {ds7:>11} | {cfg}")

    out = {"baseline": args.baseline, "rows": rows,
           "note": "末步=交付步step8为主口径, step7 附列; 判决见 report_tables.md"}
    op = args.out or os.path.join(A, "b_arms_table.json")
    with open(op, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[table] {op}")


if __name__ == "__main__":
    main()
