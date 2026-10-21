#!/usr/bin/env python3
"""summarize_fixedk.py — 从训练日志 + args.json 汇总每臂每相的协议/结果到 CSV。

用法:
    python tools/summarize_fixedk.py --out_root /root/autodl-tmp/srdiff_fixedk \
        --csv /root/autodl-tmp/srdiff_fixedk/train_summary.csv

输出列: K,stage,lr,best_val_psnr,wall_min,final_train_loss,total_skips,plan,block,allow_k_gt_n
· best_val_psnr 取日志最后一条 [val] 行（= ckpt_best 的选点口径, 与 eval_val.json 一致优先）;
· final_train_loss 取最后一条 step 行的 loss=;
· total_skips = 日志里 skip=1 的 step 行数（grad-skip 累计）;
· plan/block/allow_k_gt_n 取 args.json（协议证据）。
"""
import argparse
import json
import os
import re


def parse_arm(out_root: str, k: int, stage: int) -> dict:
    d = os.path.join(out_root, f"stdattn_k{k}_t{1 if stage == 1 else 8}")
    log = os.path.join(out_root, f"k{k}_stage{stage}.log")
    r = {"K": k, "stage": stage, "lr": "?", "best_val_psnr": "?",
         "wall_min": "?", "final_train_loss": "?", "total_skips": 0,
         "plan": "?", "block": "?", "allow_k_gt_n": "?"}
    ap = os.path.join(d, "args.json")
    if os.path.isfile(ap):
        a = json.load(open(ap))
        r["lr"] = a.get("lr", "?")
        r["plan"] = a.get("step_plan", "?")
        r["block"] = a.get("block", "?")
        r["allow_k_gt_n"] = a.get("allow_k_gt_n", "?")
    if os.path.isfile(os.path.join(d, "eval_val.json")):
        ev = json.load(open(os.path.join(d, "eval_val.json")))
        if isinstance(ev, dict) and "psnr" in ev:
            r["best_val_psnr"] = round(float(ev["psnr"]), 2)
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
        r["total_skips"] = skips
        if wall is not None:
            r["wall_min"] = round(wall, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=[64, 96, 128, 192, 256])
    args = ap.parse_args()
    rows = []
    for k in args.ks:
        for stage in (1, 2):
            if os.path.isfile(os.path.join(args.out_root, f"k{k}_stage{stage}.log")):
                rows.append(parse_arm(args.out_root, k, stage))
    cols = ["K", "stage", "lr", "best_val_psnr", "wall_min", "final_train_loss",
            "total_skips", "plan", "block", "allow_k_gt_n"]
    with open(args.csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(open(args.csv).read())


if __name__ == "__main__":
    main()
