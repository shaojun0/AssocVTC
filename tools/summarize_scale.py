#!/usr/bin/env python3
"""summarize_scale.py — 「容量阶梯」4 档的训练协议/结果汇总 → CSV（2026-10-24）

每档两行（stage1 = A 相 T=1 45ep；stage2 = B 相 T=16 6ep），列：
  M, stage, enc_dim, enc_depth, enc_heads, dec_dim, dec_depth, dec_heads,
  enc_M, dec_M, total_M, lr, epoch, batch, accum, wall_min, best_val_psnr, final_loss, skips

用法:
    python tools/summarize_scale.py --out_root /root/autodl-tmp/srdiff_scale \
        --csv /root/autodl-tmp/srdiff_scale/train_summary.csv
"""
import argparse
import json
import os
import re

ARMS = [  # M, enc_dim, enc_heads, dec_dim, dec_heads, enc_M, dec_M
    (60, 624, 12, 704, 8, 61.47, 57.85),
    (120, 864, 16, 1024, 16, 117.46, 120.74),
    (240, 1248, 16, 1440, 16, 244.43, 237.11),
    (360, 1504, 16, 1792, 16, 354.63, 365.99),
]


def parse(out_root, M, stage):
    T = 1 if stage == 1 else 16
    d = os.path.join(out_root, f"stdattn_m{M}_t{T}")
    log = os.path.join(out_root, f"m{M}_stage{stage}.log")
    r = dict(M=M, stage=stage, lr="?", epoch="?", batch="?", accum="?",
             wall_min="?", best_val_psnr="?", final_loss="?", skips=0)
    ap = os.path.join(d, "args.json")
    if os.path.isfile(ap):
        a = json.load(open(ap))
        r.update(lr=a.get("lr", "?"), epoch=a.get("epochs", "?"),
                 batch=a.get("batch_size", "?"), accum=a.get("accum_steps", "?"),
                 enc_dim=a.get("enc_dim"), enc_depth=a.get("enc_depth"),
                 enc_heads=a.get("enc_heads"), dec_dim=a.get("dec_dim"),
                 dec_depth=a.get("dec_depth"), dec_heads=a.get("dec_heads"))
    ev = os.path.join(d, "eval_val.json")
    if os.path.isfile(ev):
        e = json.load(open(ev))
        if isinstance(e, dict) and "psnr" in e:
            r["best_val_psnr"] = round(float(e["psnr"]), 2)
    if os.path.isfile(log):
        best, last, wall, skips = None, None, None, 0
        for line in open(log, errors="ignore"):
            m = re.search(r"\[val\].*psnr=([0-9.]+)dB", line)
            if m:
                best = float(m.group(1))
            m = re.search(r"loss=([0-9.]+)", line)
            if m and "step " in line:
                last = float(m.group(1))
            m = re.search(r"wall=([0-9.]+)min", line)
            if m:
                wall = float(m.group(1))
            if re.search(r"skip=1\b", line):
                skips += 1
        if best is not None:
            r["best_val_psnr"] = round(best, 2)
        if last is not None:
            r["final_loss"] = round(last, 4)
        if wall is not None:
            r["wall_min"] = round(wall, 1)
        r["skips"] = skips
    # 计参（A/B 相同架构; 从 scale_arms 表补, 避免再 build 模型）
    for (m_, ed, eh, dd, dh, em, dm) in ARMS:
        if m_ == M:
            r.update(enc_M=em, dec_M=dm, total_M=round(em + dm, 2),
                     enc_dim=r.get("enc_dim") or ed, enc_heads=r.get("enc_heads") or eh,
                     dec_dim=r.get("dec_dim") or dd, dec_heads=r.get("dec_heads") or dh)
    r.setdefault("enc_depth", 13)
    r.setdefault("dec_depth", 7)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--blocks", type=int, nargs="*", default=[60, 120, 240, 360])
    args = ap.parse_args()
    rows = []
    for M in args.blocks:
        for stage in (1, 2):
            rows.append(parse(args.out_root, M, stage))
    cols = ["M", "stage", "enc_dim", "enc_depth", "enc_heads", "dec_dim",
            "dec_depth", "dec_heads", "enc_M", "dec_M", "total_M", "lr",
            "epoch", "batch", "accum", "wall_min", "best_val_psnr",
            "final_loss", "skips"]
    with open(args.csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "?")) for c in cols) + "\n")
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r.get(c, "?")) for c in cols))


if __name__ == "__main__":
    main()
