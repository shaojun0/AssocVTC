#!/usr/bin/env python3
"""summarize_n2bigstep.py — 汇总「固定每步 n=2 × T=16/24/32」三臂的训练协议与结果。

与 tools/summarize_fixedk.py 同思路, 但 B 相目录是 `stdattn_k{K}_t{T}`（T 不是 8）,
且多了 `--accum_steps` 这一列（显存受限时的等效批量执行路径, 见 train docstring）。

用法（服务器仓库根目录或本地对已拉回的 logs/args）:
    python tools/summarize_n2bigstep.py --out_root /root/autodl-tmp/srdiff_n2bigstep \
        --csv /root/autodl-tmp/srdiff_n2bigstep/train_summary.csv
输出列: T,K,n,stage,lr,plan,block,accum,best_val_psnr,wall_min,final_train_loss,total_skips
"""
import argparse
import json
import os
import re


def parse_arm(out_root, T):
    k = 2 * T
    d = os.path.join(out_root, f"stdattn_k{k}_t{T}")
    log = os.path.join(out_root, f"k{k}_stage2.log")
    r = dict(T=T, K=k, n=2, stage=2, lr="?", plan="?", block="?", accum="?",
             best_val_psnr="?", wall_min="?", final_train_loss="?", total_skips=0)
    ap = os.path.join(d, "args.json")
    if os.path.isfile(ap):
        a = json.load(open(ap))
        r["lr"] = a.get("lr", "?")
        r["plan"] = a.get("step_plan", "?")
        r["block"] = a.get("block", "?")
        r["accum"] = a.get("accum_steps", "?")
    ev = os.path.join(d, "eval_val.json")
    if os.path.isfile(ev):
        e = json.load(open(ev))
        if isinstance(e, dict) and "psnr" in e:
            r["best_val_psnr"] = round(float(e["psnr"]), 2)
    if os.path.isfile(log):
        best = None
        last_loss = None
        skips = 0
        wall = None
        for line in open(log, errors="ignore"):
            m = re.search(r"\[val\].*psnr=([0-9.]+)dB", line)
            if m:
                best = float(m.group(1))
            m = re.search(r"loss=([0-9.]+)", line)
            if m and "step " in line:
                last_loss = float(m.group(1))
            if re.search(r"skip=1\b", line):
                skips += 1
            m = re.search(r"wall=([0-9.]+)min", line)
            if m:
                wall = float(m.group(1))
        if best is not None:
            r["best_val_psnr"] = round(best, 2)
        if last_loss is not None:
            r["final_train_loss"] = round(last_loss, 4)
        if wall is not None:
            r["wall_min"] = round(wall, 1)
        r["total_skips"] = skips
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--Ts", type=int, nargs="*", default=[16, 24, 32])
    args = ap.parse_args()
    rows = [parse_arm(args.out_root, T) for T in args.Ts]
    cols = list(rows[0].keys())
    with open(args.csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
