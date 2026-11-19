#!/usr/bin/env python
"""CPU 小规模形状自检：循环读出通路（不碰 GPU/MUSA）。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v10 import V10Config, WholeClipDecoder     # noqa: E402

K, C = 8, 16
cfg = V10Config(frames=8, group=4, rate_tokens=8, rate_dim=C, target_tokens=K,
                chunk_tokens=2, mid_dim=32, dec_dim=32, dec_heads=4, dec_depth=2,
                dec_ratio=2.0)
assert cfg.width == 32, cfg.width

dec = WholeClipDecoder(cfg, code_dim=C, num_specials=K)
z = torch.randn(2, cfg.n_steps, cfg.chunks_per_step, cfg.width)  # (2,2,16,32)
out = dec(z, return_step_preds=True)
assert out["pred"].shape == (2, 8, 1 + K, C), out["pred"].shape
assert out["preds"].shape == (2, cfg.n_steps, 8, 1 + K, C), out["preds"].shape
print(f"[ok] pred {tuple(out['pred'].shape)} preds {tuple(out['preds'].shape)}")

# 因果：扰动第 k 组 z 不影响更早步的输出
k = 1
zb = z.clone()
zb[:, k] += 0.5
with torch.no_grad():
    a = dec(z, return_step_preds=True)["preds"]
    b = dec(zb, return_step_preds=True)["preds"]
d = (a - b).abs().amax(dim=(0, 2, 3, 4))                 # (S,)
assert all(float(d[t]) == 0.0 for t in range(k)), d
assert float(d[k]) > 1e-6 and float(d[-1]) > 1e-6, d
print(f"[ok] causal per-step max|Δ| {[round(float(v), 4) for v in d]}")
print("ALL OK")
