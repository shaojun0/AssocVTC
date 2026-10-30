#!/usr/bin/env python3
"""比较不同 init（mem_proj 随机 vs PCA 投影 raw/白化）在 8 张 val 上的初始 PSNR。"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/root/autodl-tmp/qwen_vtc/code")
import qwen_probe_v2 as V2
from metrics_linattn import psnr

device = "cuda"
vt = np.load("tokens/val_tokens_fp16.npy", mmap_mode="r")
vi = np.load("tokens/val_images_224.npy", mmap_mode="r")
idx = np.arange(32)
tok = torch.from_numpy(np.asarray(vt[idx]).astype(np.float32)).to(device)
u8 = torch.from_numpy(np.asarray(vi[idx]))
tgt = (u8.permute(0, 3, 1, 2).float() / 255.0).to(device)
res = {}
for tag, path in [("random", "probe/init_m240_decoder.pt"),
                  ("raw", "probe/init_m240_decoder_raw.pt"),
                  ("wht", "probe/init_m240_decoder_wht.pt")]:
    m = V2.build_model(device, path)
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pix = m(tok)
    pred = V2.patches_to_img01(pix.float()).permute(0, 3, 1, 2).clamp(0, 1)
    P = float(psnr(pred, tgt, max_val=1.0).mean())
    res[tag] = P
    print("[cmp] init=%-6s 8/32-img PSNR = %.3f dB" % (tag, P), flush=True)
    del m
    torch.cuda.empty_cache()
print("[cmp] RESULT", res)
