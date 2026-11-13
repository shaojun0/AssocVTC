"""v7 / v8(=v3_1) 单步时间与显存对比。

用法（在 video_recon_scale 目录下）:
  python bench_v8.py --cfg cfg_v32b.json --variant v7,v7b,v8 --bs 4 --F 32 --n-warm 24
其中 variant:
  v7  = v7 原样（recurrent 中间态 + 逐帧 decode）
  v7b = v7 + 帧维折 batch decode（recurrent 中间态）
  v8  = v3_1 移植（gdn 中间态 + 帧维折 batch decode）
只报 forward+backward。数据是随机张量（只测速度，不看 loss 数值）。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

from model_v7 import VideoStateScaleV7
from model_v8 import VideoStateScaleV8


def get_device():
    """CUDA / MUSA 自动选（MUSA 机器上 torch.cuda 不可用）。"""
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa:0")
    return torch.device("cuda:0")


def sync(dev):
    (torch.musa if dev.type == "musa" else torch.cuda).synchronize()


def peak_gb(dev):
    m = torch.musa if dev.type == "musa" else torch.cuda
    return m.max_memory_allocated() / 1e9


def reset_peak(dev):
    m = torch.musa if dev.type == "musa" else torch.cuda
    m.reset_peak_memory_stats()


def build(cfg: dict, variant: str):
    c = copy.deepcopy(cfg)
    if variant == "v7":
        c["mid"]["kind"] = "recurrent"
        c["dec"]["frame_batch"] = False
    elif variant == "v7b":
        c["mid"]["kind"] = "recurrent"
        c["dec"]["frame_batch"] = True
    elif variant == "v8":
        c["mid"]["kind"] = "gdn"
        c["dec"]["frame_batch"] = True
    elif variant == "v8c":
        c["mid"]["kind"] = "gdn"
        c["dec"]["frame_batch"] = True
    else:
        raise SystemExit(f"未知 variant={variant}")
    return VideoStateScaleV8(c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--variant", default="v7,v7b,v8")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--F", type=int, default=32)
    ap.add_argument("--n-warm", type=int, default=24)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--enc-chunk", type=int, default=0)
    ap.add_argument("--frame-chunk", type=int, default=0,
                    help="v8 decode 帧维分块（0=不分块）")
    ap.add_argument("--no-opt", action="store_true",
                    help="不建 optimizer（省 2×params 显存, 只测 fwd+bwd）")
    args = ap.parse_args()

    cfg = json.load(open(args.cfg))
    if args.frame_chunk:
        cfg["dec"]["frame_chunk"] = int(args.frame_chunk)
    img = int(cfg["img"])
    dev = get_device()
    print(f"device={dev}", flush=True)
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp_dtype)
    torch.manual_seed(0)
    variants = ["v8" if v == "v8c" else v for v in args.variant.split(",")]
    for variant in variants:
      try:
        model = build(cfg, variant).to(dev)
        model.train()
        n_par = model.param_groups()
        opt = None if args.no_opt else torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.rand(args.bs, args.F, 3, img, img, device=dev)
        grad_frames = args.F - args.n_warm
        reset_peak(dev)
        step_times = []
        for it in range(args.iters):
            t0 = time.time()
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
            if amp is None:
                out = model(x, n_warm=args.n_warm, enc_chunk=args.enc_chunk)
                out["loss"].backward()
            else:
                with torch.autocast(device_type=dev.type, dtype=amp):
                    out = model(x, n_warm=args.n_warm, enc_chunk=args.enc_chunk)
                out["loss"].backward()
            if opt is not None:
                opt.step()
            sync(dev)
            step_times.append(time.time() - t0)
        steady = sum(step_times[1:]) / max(1, len(step_times) - 1)
        peak = peak_gb(dev)
        print(f"[{variant}] params={n_par['total']/1e6:.1f}M "
              f"(enc {n_par['encoder']/1e6:.1f} / mid {n_par['middle']/1e6:.1f} "
              f"/ dec {n_par['decoder']/1e6:.1f}) "
              f"bs={args.bs} F={args.F} n_warm={args.n_warm} "
              f"grad_frames/step={args.bs * grad_frames} "
              f"| {steady:.2f} s/step | "
              f"{steady / (args.bs * grad_frames) * 1000:.1f} ms/梯度帧 | "
              f"peak {peak:.1f} GB | psnr {float(out['psnr']):.2f}", flush=True)
        del model, opt, x, out
        torch.cuda.empty_cache() if dev.type == "cuda" else torch.musa.empty_cache()
      except torch.OutOfMemoryError:
        print(f"[{variant}] OOM (bs={args.bs} F={args.F})", flush=True)
        torch.cuda.empty_cache() if dev.type == "cuda" else torch.musa.empty_cache()


if __name__ == "__main__":
    main()
