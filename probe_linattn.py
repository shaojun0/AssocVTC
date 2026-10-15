"""
吞吐/显存探针 — SR-Diffusion-v3 实验版 (linattn)
==============================================

    python probe_linattn.py --batch_sizes 64 128 256

用随机张量（不依赖数据集）测 224×224 前向+反向的 ms/step、img/s、峰值显存,
用于挑选 batch size 与估算总训练时长。
"""

import argparse
import time

import torch

from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn


def probe(bs: int, K: int, iters: int = 8):
    cfg = LinAttnConfig(num_specials=K)
    model = SRPhase1LinAttn(cfg).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(bs, 3, cfg.img_size, cfg.img_size, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    # warmup
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  bs={bs:>4d} K={K:>3d}  {dt*1000:7.1f} ms/step  "
          f"{bs/dt:7.1f} img/s  峰值显存 {peak:5.2f} GB  loss={loss.item():.4f}",
          flush=True)
    del model, opt, x
    torch.cuda.empty_cache()
    return bs / dt, peak


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args()
    print(f"[probe] {torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__} cuda={torch.version.cuda}")
    best = None
    for bs in a.batch_sizes:
        try:
            ips, peak = probe(bs, a.K, a.iters)
            if best is None or ips > best[1]:
                best = (bs, ips, peak)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  bs={bs:>4d} OOM: {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()
    if best:
        steps_per_epoch = 121689 // best[0]          # ImageNet-100 train 划分张数
        print(f"[best] bs={best[0]} {best[1]:.1f} img/s 峰值 {best[2]:.2f}GB "
              f"⇒ ~{steps_per_epoch} 步/epoch, "
              f"~{steps_per_epoch/best[1]*best[0]/60:.1f} min/epoch")
