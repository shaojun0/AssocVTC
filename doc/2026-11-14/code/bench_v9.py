"""v8(顺序+conv head) vs v9(并行+conv/mlp head) 的单步时间/显存对比。

用法: python bench_v9.py --cfg cfg_v9_32b.json --bs 8 --F 16 --n-warm 8 --iters 3
数据是随机张量（只测速度/显存）。
"""
from __future__ import annotations
import argparse, copy, json, time
import torch
from model_v8 import build_any


def dev():
    return torch.device("cuda:0")


def sync(d):
    torch.cuda.synchronize()


def variants(cfg):
    out = []
    c = copy.deepcopy(cfg); c["arch"] = "v8"; c["dec"]["parallel"] = False; c["dec"]["head_kind"] = "conv"
    out.append(("v8  seq +conv", c))
    c = copy.deepcopy(cfg); c["arch"] = "v9"; c["dec"]["parallel"] = True; c["dec"]["head_kind"] = "conv"
    out.append(("v9  par +conv", c))
    c = copy.deepcopy(cfg); c["arch"] = "v9"; c["dec"]["parallel"] = True; c["dec"]["head_kind"] = "mlp"
    out.append(("v9  par +mlp ", c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--F", type=int, default=16)
    ap.add_argument("--n-warm", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--frame-chunk", type=int, default=0)
    a = ap.parse_args()
    cfg0 = json.load(open(a.cfg))
    if a.frame_chunk:
        cfg0["dec"]["frame_chunk"] = a.frame_chunk
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}[a.amp]
    d = dev()
    img = int(cfg0["img"])
    G = a.F - a.n_warm
    for tag, cfg in variants(cfg0):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = build_any(cfg).to(d).train()
            pg = m.param_groups()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            x = torch.rand(a.bs, a.F, 3, img, img, device=d)
            ts = []
            for it in range(a.iters):
                t0 = time.time()
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp):
                    out = m(x, n_warm=a.n_warm, enc_chunk=4)
                out["loss"].backward(); opt.step(); sync(d)
                ts.append(time.time() - t0)
            st = sum(ts[1:]) / max(1, len(ts) - 1)
            frames = a.bs * G
            print(f"[{tag}] params {pg['total']/1e6:7.1f}M (dec {pg['decoder']/1e6:6.1f}M) "
                  f"bs={a.bs} F={a.F} nw={a.n_warm} | {st:7.2f} s/step | "
                  f"{st/frames*1000:7.1f} ms/监督帧 | {frames/st*86400/1e3:6.1f}k 帧/天 | "
                  f"peak {torch.cuda.max_memory_allocated()/1e9:5.1f} GB", flush=True)
            del m, opt, x, out
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"[{tag}] OOM bs={a.bs} F={a.F}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
