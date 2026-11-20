"""Compare our per-step rFID (8-step recurrent, zs_dim=192) against ALIT Table 3.

Value-budget equivalence: our z_s token = zs_dim=192 values; ALIT continuous 1D
token = 12 values (ALIT paper: continuous latent is 32x12x32 bits, quantized 32x12 bits;
Table 4 column "潜变量维度 = 12").  So ALIT-equivalent tokens = cum_values / 12.
"""
import json
import sys

OUR = {
    "8step": "/root/autodl-tmp/rfid_zs192t8/zs192_t8rfid/eval_rfid.json",
    "t1": "/root/autodl-tmp/rfid_zs192t8/zs192_t1/eval_rfid.json",
}
ZS = 192
ALIT_TOK = [32, 64, 96, 128, 160, 192, 224, 256]
ALIT_S_S2 = [22.57, 16.17, 13.30, 11.69, 10.22, 9.30, 8.55, 8.25]   # IN-100 Stage2 small(8) dim1024
ALIT_B_S2 = [22.15, 15.88, 12.57, 10.44, 9.50, 8.96, 8.39, 8.19]   # IN-100 Stage2 base(12) dim768


def interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return float("nan")


def main():
    d8 = json.load(open(OUR["8step"]))
    dt1 = json.load(open(OUR["t1"]))
    print("## 8-step recurrent (zs_dim=192, K=16)")
    print(f"{'step':>4} {'cum_z_s':>7} {'values':>7} {'ALIT_eq_tok':>11} "
          f"{'PSNR':>7} {'L1_255':>7} {'rFID':>7} {'ALIT-S':>7} {'ALIT-B':>7}")
    for s in sorted(d8["per_step"], key=lambda k: int(k)):
        r = d8["per_step"][s]
        cum = r["cum_z_s"]
        vals = cum * ZS
        eq = vals / 12.0
        print(f"{r['step']:>4} {cum:>7} {vals:>7} {eq:>11.1f} "
              f"{r['psnr']:>7.2f} {r['l1_255']:>7.2f} {r['rfid']:>7.2f} "
              f"{interp(eq, ALIT_TOK, ALIT_S_S2):>7.2f} {interp(eq, ALIT_TOK, ALIT_B_S2):>7.2f}")
    print()
    print("## T=1 single-step baseline (zs_dim=192, K=16)")
    for s, r in dt1["per_step"].items():
        vals = r["cum_z_s"] * ZS
        print(f"  step{r['step']} values={vals} ALIT_eq={vals/12:.1f} "
              f"PSNR={r['psnr']:.2f} L1_255={r['l1_255']:.2f} rFID={r['rfid']:.2f}")
    print()
    print("## ALIT Table 3 (IN-100, Stage2) reference")
    print("tokens:", ALIT_TOK)
    print("Ours-S:", ALIT_S_S2)
    print("Ours-B:", ALIT_B_S2)


if __name__ == "__main__":
    main()
