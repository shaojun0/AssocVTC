#!/usr/bin/env python3
"""v2 探针 smoke：init 后 8 张 val 的初始 PSNR（随机 mem_proj 下预期 12-18 dB）。"""
import sys, json
import numpy as np
import torch
sys.path.insert(0, "/root/autodl-tmp/qwen_vtc/code")
import qwen_probe_v2 as V2
from metrics_linattn import psnr, ssim
from PIL import Image

device = "cuda"
m = V2.build_model(device, "probe/init_m240_decoder.pt")
m.eval()
vt = np.load("tokens/val_tokens_fp16.npy", mmap_mode="r")
vi = np.load("tokens/val_images_224.npy", mmap_mode="r")
idx = np.arange(8)
tok = torch.from_numpy(np.asarray(vt[idx]).astype(np.float32)).to(device)
u8 = torch.from_numpy(np.asarray(vi[idx]))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    pix = m(tok)
pred = V2.patches_to_img01(pix.float()).permute(0, 3, 1, 2).clamp(0, 1)
tgt = (u8.permute(0, 3, 1, 2).float() / 255.0).to(pred.device)
P = psnr(pred, tgt, max_val=1.0); S = ssim(pred, tgt)
print("[smoke_v2] init PSNR %.3f dB  SSIM %.3f  (n=%d)" % (P.mean(), S.mean(), len(idx)))
print("[smoke_v2] per-image PSNR:", [round(float(x), 2) for x in P])
# 存 8 张重建图供人工检查
img = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
np.save("probe/smoke_v2_recon.npy", img)
json.dump({"psnr": float(P.mean()), "ssim": float(S.mean())}, open("probe/smoke_v2.json", "w"))
