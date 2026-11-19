"""诊断：v10 重构版（crossq middle / 无 copy_skip）阶段 2 的数值与表达塌缩。

1) middle 表达塌缩？比较 raw rate 与 middle 输出的跨片段/幅度统计（std）。
2) 30 步 carry 是否数值发散？逐 step 记录 max|h| / max|Y|，bf16 autocast vs fp32。
3) 读出的 30 步输出是否 NaN？
用法: SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0 python tools/diag_carry_norm.py \
        --trainable .../best_trainable.pt [--clips 0,1,2,3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_v10_stepwise import load_model, _set_sdpa_math  # noqa: E402
from data_v10 import ClipDataset  # noqa: E402


def run_loop(model, z, autocast: bool):
    rd = model.readout
    S = z.shape[1]
    Q = rd.Q.expand(z.shape[0], -1, -1)
    Y = torch.zeros_like(Q)
    rec = []
    for t in range(S):
        h = Q + Y + rd.step_emb[:, t]
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
               else torch.autocast("cuda", enabled=False))
        with ctx:
            Y = rd.decoder(tgt=h, memory=(z + rd.kv_pos)[:, t])
        rec.append(dict(step=t + 1, max_h=float(h.abs().max()),
                        max_Y=float(Y.abs().max()),
                        nan=bool(torch.isnan(Y).any())))
        if rec[-1]["nan"]:
            break
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainable", required=True)
    ap.add_argument("--clips", default="0,1,2,3")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    _set_sdpa_math()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    model = load_model(a.trainable).to(dev).eval()
    cfg = model.cfg
    ds = ClipDataset("/root/autodl-tmp/video_recon/data3", "val",
                     frames=cfg.frames, img=int(model.tokenizer.img))
    clips = [int(s) for s in a.clips.split(",") if s.strip()]
    enc_stats, loops = [], []
    with torch.no_grad():
        for ci in clips:
            x = ds[ci].unsqueeze(0).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zc, zs = model.tokenizer.encode(x)
                enc = model.encode_rate_latents(zc, zs)
            packed, mid, z = enc["packed"].float(), enc["mid"].float(), enc["z"]
            enc_stats.append(dict(clip=ci,
                                  packed_std=float(packed.std()),
                                  packed_mean_abs=float(packed.abs().mean()),
                                  mid_std=float(mid.std()),
                                  mid_mean_abs=float(mid.abs().mean()),
                                  per_clip_std_of_mean=float(mid.mean(1).std())))
            loops.append(dict(clip=ci,
                              bf16=run_loop(model, z, autocast=True),
                              fp32=run_loop(model, z, autocast=False)))
    res = dict(trainable=a.trainable, enc_stats=enc_stats, loops=loops)
    print(json.dumps(dict(enc_stats=enc_stats), ensure_ascii=False, indent=2))
    for lp in loops:
        b, f = lp["bf16"], lp["fp32"]
        print(f"[clip{lp['clip']}] bf16: steps_run={len(b)} "
              f"max|Y| last3={[round(s['max_Y'], 3) for s in b[-3:]]} "
              f"nan={b[-1]['nan']}")
        print(f"[clip{lp['clip']}] fp32: steps_run={len(f)} "
              f"max|Y| last3={[round(s['max_Y'], 3) for s in f[-3:]]} "
              f"nan={f[-1]['nan']}")
    if a.out:
        open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
