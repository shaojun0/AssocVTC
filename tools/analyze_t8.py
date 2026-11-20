#!/usr/bin/env python3
"""Turn fast_rfid.py JSONs into markdown tables + verdicts for the T=8 reports.

usage: analyze_t8.py --fast a.json b.json ... [--train_log s2/train_log.jsonl] [--patch_dc 18.07]
"""
import argparse, json

ALIT_TOK = [32, 64, 96, 128, 160, 192, 224, 256]
ALIT_S   = [22.57, 16.17, 13.30, 11.69, 10.22, 9.30, 8.55, 8.25]
ALIT_B   = [22.15, 15.88, 12.57, 10.44, 9.50, 8.96, 8.39, 8.19]
ALIT_DIM = 12


def _interp(t, ys):
    """Linear interpolation of ALIT Table-3 rFID at token count t (clamped at both ends)."""
    if t <= ALIT_TOK[0]:
        return ys[0]
    if t >= ALIT_TOK[-1]:
        return ys[-1]
    for i in range(len(ALIT_TOK) - 1):
        if ALIT_TOK[i] <= t <= ALIT_TOK[i + 1]:
            w = (t - ALIT_TOK[i]) / (ALIT_TOK[i + 1] - ALIT_TOK[i])
            return ys[i] * (1 - w) + ys[i + 1] * w
    return ys[-1]


def load(p):
    d = json.load(open(p))
    steps = sorted(d["per_step"].values(), key=lambda r: r["step"])
    return d, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", nargs="+", required=True)
    ap.add_argument("--train_log", default="")
    ap.add_argument("--patch_dc", type=float, default=18.07)
    a = ap.parse_args()

    rec, base = [], []
    for p in a.fast:
        d, steps = load(p)
        (rec if d["recurrent"] else base).append((d, steps))

    for d, steps in rec:
        print(f"\n### per-step — {d['tag']} (T={d['T']}, K={d['K']}, zs_dim={d['zs_dim']}, "
              f"{d['n_images']} imgs, {d['img_size']}²)\n")
        print("| step | cum_z_s | values | PSNR | L1_255 | rFID |")
        print("|---|---|---|---|---|---|")
        for r in steps:
            print(f"| {r['step']} | {r['cum_z_s']} | {r['values']} | {r['psnr']:.2f} | "
                  f"{r['l1_255']:.2f} | {r['rfid']:.2f} |")
        last, prev = steps[-1], steps[-2] if len(steps) > 1 else steps[-1]
        print(f"\nlast-step vs prev-step: PSNR {prev['psnr']:.2f}→{last['psnr']:.2f} "
              f"({last['psnr']-prev['psnr']:+.2f})  rFID {prev['rfid']:.2f}→{last['rfid']:.2f} "
              f"({last['rfid']-prev['rfid']:+.2f})")
        print(f"last-step vs patchDC {a.patch_dc:.2f}: {last['psnr']-a.patch_dc:+.2f} dB")
        mp = max(steps, key=lambda r: r["psnr"])
        print(f"best-PSNR step = {mp['step']} ({mp['psnr']:.2f} dB, values={mp['values']})")

    for d, steps in base:
        r = steps[-1]
        print(f"\n### baseline — {d['tag']} (T=1, values={r['values']}): "
              f"PSNR {r['psnr']:.2f}, L1_255 {r['l1_255']:.2f}, rFID {r['rfid']:.2f}")

    if rec and base:
        b = base[-1][1][-1]
        last = rec[-1][1][-1]
        print(f"\n### T=8 last step vs T=1 baseline\n")
        print(f"- PSNR {last['psnr']:.2f} vs {b['psnr']:.2f} ({last['psnr']-b['psnr']:+.2f} dB)")
        print(f"- L1_255 {last['l1_255']:.2f} vs {b['l1_255']:.2f}")
        print(f"- rFID {last['rfid']:.2f} vs {b['rfid']:.2f} ({last['rfid']/b['rfid']:.2f}×)")
        if last["values"] == b["values"]:
            print("- (same values/image ⇒ directly comparable)")

    print("\n### ALIT value-equivalence (ALIT 1D token dim = 12)\n")
    print("| values | ALIT-eq tokens | ALIT-S rFID | ALIT-B rFID |")
    print("|---|---|---|---|")
    seen = set()
    for d, steps in rec + base:
        for r in steps:
            t = r["values"] / ALIT_DIM
            if t < 32 or t in seen:
                continue
            seen.add(t)
            print(f"| {r['values']} | {t:.0f} | {_interp(t, ALIT_S):.2f} | "
                  f"{_interp(t, ALIT_B):.2f} |")

    if a.train_log:
        print(f"\n### training val ({a.train_log})\n")
        print("| epoch | PSNR | SSIM | L1_norm | gain vs patchDC |")
        print("|---|---|---|---|---|")
        for line in open(a.train_log):
            r = json.loads(line)
            if r.get("split") == "val":
                print(f"| {r['epoch']} | {r['psnr']:.2f} | {r['ssim']:.3f} | "
                      f"{r['l1_norm']:.3f} | {r.get('psnr_gain_over_patch_dc', 0):+.2f} |")


if __name__ == "__main__":
    main()
