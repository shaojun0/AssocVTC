#!/usr/bin/env python3
"""Summarize the L2(MSE) continuation run vs the L1 ep14 baseline.

Reads:
  artifacts/k256x12_t8/t8_l1_ep14_fast.json        (L1 baseline, ep14)
  artifacts/l2ft/*_fast.json                      (L2 run: ckpt_ep001..003, ckpt_best)
  artifacts/l2ft/train_log.jsonl                  (val PSNR per epoch)
Writes:
  artifacts/l2ft/ANALYSIS.md, artifacts/l2ft/l2ft_curve.png
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/linaro/dsh/tok448_work/artifacts"
L1 = json.load(open(f"{BASE}/k256x12_t8/t8_l1_ep14_fast.json"))
L2DIR = f"{BASE}/l2ft"


def step8(d):
    return d["per_step"][str(max(int(k) for k in d["per_step"]))]


def load_epochs():
    rows = {}
    for p in sorted(glob.glob(f"{L2DIR}/*_fast.json")):
        name = os.path.basename(p).replace("_fast.json", "")
        d = json.load(open(p))
        m = re.search(r"ep(\d+)", name)
        ep = int(m.group(1)) if m else None
        rows[name] = {"json": d, "ep": ep}
    return rows


def val_psnr_by_epoch():
    out = {}
    p = f"{L2DIR}/train_log.jsonl"
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "psnr" in r and "epoch" in r:
            out[int(r["epoch"])] = (float(r["psnr"]), float(r.get("ssim", float("nan"))))
    return out


def main():
    rows = load_epochs()
    vp = val_psnr_by_epoch()
    print("L1 baseline ep14:", step8(L1)["values"], "values | PSNR %.3f | rFID %.2f"
          % (step8(L1)["psnr"], step8(L1)["rfid"]))
    table = []
    for name, r in sorted(rows.items(), key=lambda kv: (kv[1]["ep"] is None, kv[1]["ep"] or 0, kv[0])):
        s = step8(r["json"])
        ep = r["ep"]
        v = vp.get(ep) if ep else None
        table.append((name, ep, v, s, r["json"]))
        print(f"{name:18s} ep={ep} valPSNR={v[0] if v else float('nan'):.3f} "
              f"step8 PSNR={s['psnr']:.3f} rFID={s['rfid']:.2f} "
              f"dPSNR={s['psnr']-step8(L1)['psnr']:+.3f} drFID={s['rfid']-step8(L1)['rfid']:+.2f}")

    # per-step comparison for the best L2 ckpt (lowest rFID at step8)
    best = min(table, key=lambda t: t[3]["rfid"]) if table else None
    lines = ["# L2(MSE) 续训结果 vs L1 ep14 基线", ""]
    lines.append("| ckpt | ep | val PSNR(1024) | step8 PSNR | step8 rFID | ΔPSNR | ΔrFID |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| L1 基线 ep14 | 14 | 22.607 | {step8(L1)['psnr']:.3f} | {step8(L1)['rfid']:.2f} | — | — |")
    for name, ep, v, s, _j in table:
        vs = f"{v[0]:.3f}" if v else "—"
        epc = ep if ep is not None else "best"
        lines.append(f"| L2 {name} | {epc} | {vs} | {s['psnr']:.3f} | {s['rfid']:.2f} | "
                     f"{s['psnr']-step8(L1)['psnr']:+.3f} | {s['rfid']-step8(L1)['rfid']:+.2f} |")
    if best:
        lines += ["", "## 逐速率点对比（L1 ep14 vs L2 最优 rFID ckpt）", "",
                  "| values | step | L1 PSNR | L1 rFID | L2 PSNR | L2 rFID | ΔPSNR | ΔrFID |",
                  "|---|---|---|---|---|---|---|---|"]
        d = best[4]
        for k in sorted(L1["per_step"], key=lambda x: int(x)):
            a = L1["per_step"][k]; b = d["per_step"].get(k)
            if b is None:
                continue
            lines.append(f"| {a['values']} | {k} | {a['psnr']:.2f} | {a['rfid']:.2f} | "
                         f"{b['psnr']:.2f} | {b['rfid']:.2f} | "
                         f"{b['psnr']-a['psnr']:+.2f} | {b['rfid']-a['rfid']:+.2f} |")
    open(f"{L2DIR}/ANALYSIS.md", "w").write("\n".join(lines) + "\n")

    # plot
    if table:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        eps = [t[1] for t in table if t[1] is not None]
        fids = [t[3]["rfid"] for t in table if t[1] is not None]
        psnrs = [t[3]["psnr"] for t in table if t[1] is not None]
        ax[0].axhline(step8(L1)["rfid"], ls="--", color="gray", label="L1 ep14 baseline 86.50")
        ax[0].plot(eps, fids, "o-", color="tab:blue", label="L2 (MSE) continuation")
        ax[0].set_xlabel("L2 finetune epoch"); ax[0].set_ylabel("step8 rFID (lower better)")
        ax[0].set_title("rFID @3072 values"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
        ax[1].axhline(step8(L1)["psnr"], ls="--", color="gray", label="L1 ep14 baseline 22.67")
        ax[1].plot(eps, psnrs, "o-", color="tab:red", label="L2 (MSE) continuation")
        ax[1].set_xlabel("L2 finetune epoch"); ax[1].set_ylabel("step8 PSNR (dB)")
        ax[1].set_title("PSNR @3072 values"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"{L2DIR}/l2ft_curve.png", dpi=140)
    print("wrote", f"{L2DIR}/ANALYSIS.md")


if __name__ == "__main__":
    main()
