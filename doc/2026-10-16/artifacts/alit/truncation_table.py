#!/usr/bin/env python3
"""E-B4 截断可用性表（**无需额外训练**）
================================================================================
链式因果下"截断到 t"的前缀计算与全链相同 ⇒ 各臂 `stepwise_*.json` 的逐 step
`L1_255` / `PSNR` 就是截断曲线。本脚本把它整理成"截断可用性 vs 末步质量"的对照表。

预算映射（K=64, cum_z_s=[3,8,15,24,35,48,63,64]）::

    t=3 → 15 token | t=5 → 35 token | t=7 → 63 token | t=8 → 64 token

用法::

    python doc/2026-10-16/artifacts/alit/truncation_table.py --dir doc/2026-10-16/artifacts/alit \
        --baseline b1_mean --arms b1_mean b1_increasing b1_tail27 b1_lastonly

产出 `truncation_table.json` + 打印 markdown 表。
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="doc/2026-10-16/artifacts/alit")
    ap.add_argument("--baseline", default="b1_mean")
    ap.add_argument("--arms", nargs="*",
                    default=["b1_mean", "b1_decay", "b1_increasing", "b1_tail27",
                             "b1_lastonly"])
    ap.add_argument("--ts", default="3,5,7,8")
    args = ap.parse_args()
    A, ts = args.dir, [int(x) for x in args.ts.split(",")]

    data = {}
    for arm in args.arms:
        p = os.path.join(A, f"stepwise_{arm}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cum = d.get("cum_z_s") or list(range(1, len(d["L1_255"]) + 1))
        data[arm] = {"L1_255": d["L1_255"], "PSNR": d.get("PSNR"), "cum_z_s": cum}

    if not data:
        raise SystemExit("没有可用的 stepwise json")
    cum = next(iter(data.values()))["cum_z_s"]
    base = data.get(args.baseline)

    print(f"预算映射: " + " | ".join(f"t={t}→{cum[t-1]} token" for t in ts if t <= len(cum)))
    print()
    hdr = f"{'arm':<16} | " + " | ".join(f"L1@t{t}({cum[t-1]})" for t in ts if t <= len(cum)) \
          + " | " + " | ".join(f"PSNR@t{t}" for t in ts if t <= len(cum))
    print(hdr)
    print("-" * len(hdr))
    for arm, r in data.items():
        cells = [f"{r['L1_255'][t-1]:.4f}" for t in ts if t <= len(r["L1_255"])]
        pcells = [f"{r['PSNR'][t-1]:.3f}" if r.get("PSNR") else "-"
                  for t in ts if t <= len(r["L1_255"])]
        print(f"{arm:<16} | " + " | ".join(cells) + " | " + " | ".join(pcells))
    if base:
        print()
        print("ΔL1_255 vs 基线（+ = 更差）:")
        print(f"{'arm':<16} | " + " | ".join(f"Δ@t{t}" for t in ts if t <= len(base["L1_255"])))
        print("-" * 60)
        for arm, r in data.items():
            cells = [f"{r['L1_255'][t-1] - base['L1_255'][t-1]:+.4f}"
                     for t in ts if t <= len(r["L1_255"])]
            print(f"{arm:<16} | " + " | ".join(cells))

    out = {"baseline": args.baseline, "ts": ts, "cum_z_s": cum,
           "budget_tokens": {f"t{t}": cum[t - 1] for t in ts if t <= len(cum)},
           "arms": data,
           "note": "截断档不需要额外训练: 链式因果下前缀与全链相同; 判决见 report_tables.md"}
    op = os.path.join(A, "truncation_table.json")
    with open(op, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[trunc] {op}")


if __name__ == "__main__":
    main()
