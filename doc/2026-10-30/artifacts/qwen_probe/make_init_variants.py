#!/usr/bin/env python3
"""给 v2 探针做 mem_proj 的 PCA-投影初始化变体（decoder 其余键用 init_m240_decoder.pt）。

mem_proj 原始是 1248→1440（ckpt 的 W,b）；Qwen token 是 5120 维 ⇒ 用池化 PCA 的前 1248 个
主成分把 5120 投影到 1248 再进原 W：
    y = W @ U_1248^T @ (x - mu) + b        （U = components[:, :1248]）
变体:
  raw : 不白化（主成分按自身方差加权）
  wht : 白化（坐标除 std=sqrt(eigval)，各主成分等权）
因为 ckpt 里 mem_norm 在 mem_proj 之后（LayerNorm），整体尺度不影响，只看方向。
"""
import json
import numpy as np
import torch

R = "/root/autodl-tmp/qwen_vtc"
SRC = f"{R}/probe/init_m240_decoder.pt"
PCA = f"{R}/pca"

ck = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ck["model"]
info = ck.get("info", {})

# ckpt 原始 mem_proj（1248→1440）
ckpt = torch.load(f"{R}/probe/ckpt_m240_t16.pt", map_location="cpu", weights_only=False)
cs = ckpt["model"] if "model" in ckpt else ckpt
W = cs["decoder.mem_proj.weight"].float()      # (1440,1248)
b = cs["decoder.mem_proj.bias"].float()        # (1440,)
assert W.shape == (1440, 1248), W.shape

mean = np.load(f"{PCA}/mean.npy").astype(np.float32)                    # (5120,)
comps = np.load(f"{PCA}/components.npy", mmap_mode="r")                 # (5120,5120)
eig = np.load(f"{PCA}/eigvals.npy").astype(np.float32)                  # (5120,)
U = np.asarray(comps[:, :1248], dtype=np.float32)                       # (5120,1248)
mu = torch.from_numpy(mean)

for tag, scale in [("raw", np.ones_like(eig[:1248])),
                   ("wht", np.sqrt(np.maximum(eig[:1248], 1e-12)))]:
    Ut = torch.from_numpy(U / scale[None, :])             # (5120,1248)
    Wn = W @ Ut.t()                                       # (1440,5120)
    bn = b - Wn @ mu                                      # 中心化
    out = dict(sd)
    out["decoder.mem_proj.weight"] = Wn.contiguous()
    out["decoder.mem_proj.bias"] = bn.contiguous()
    dst = f"{R}/probe/init_m240_decoder_{tag}.pt"
    torch.save({"model": out, "info": info, "init_variant": tag}, dst)
    print("saved", dst, "W", tuple(Wn.shape))
