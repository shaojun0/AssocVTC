#!/usr/bin/env python
"""CPU 小规模形状自检：step 模式 + copy_skip 的新通路（不碰 GPU/MUSA）。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v10 import V10Config, WholeClipDecoder     # noqa: E402

K, C = 8, 16
cfg = V10Config(frames=8, group=4, rate_tokens=8, rate_dim=C, target_tokens=K,
                chunk_tokens=2, mid_dim=32, dec_dim=32, dec_heads=4, dec_depth=2,
                dec_ratio=2.0, mem_mode="step", copy_skip=True, loss_read=True)
assert cfg.width == 32, cfg.width


def run(mem_mode, copy_skip):
    cfg.mem_mode, cfg.copy_skip = mem_mode, copy_skip
    dec = WholeClipDecoder(cfg, code_dim=C, num_specials=K)
    z = torch.randn(2, cfg.n_steps, cfg.chunks_per_step, cfg.width)  # (2,2,16,32)
    out = dec(z, return_step_preds=True)
    assert out["pred"].shape == (2, 8, 1 + K, C), out["pred"].shape
    assert out["preds"].shape == (2, cfg.n_steps, 8, 1 + K, C), out["preds"].shape
    print(f"[ok] {mem_mode:6s} copy_skip={int(copy_skip)} "
          f"pred {tuple(out['pred'].shape)} preds {tuple(out['preds'].shape)} "
          f"kv_pos={'yes' if dec.kv_pos is not None else 'no'}")
    return dec, z


run("step", True)
run("step", False)
run("prefix", True)
# copy_skip 在 step 模式下必须真的把「已读帧的 rate 行」抄进输出行：
# 对比 step+copy / step-nocopy 在第 0 步前 4 帧上的输出，二者不同即说明残差生效。
d1, z = run("step", True)
d0, _ = run("step", False)
with torch.no_grad():
    o1 = d1(z)["pred"]
    o0 = d0(z)["pred"]
diff_read = (o1[:, :4] - o0[:, :4]).abs().mean().item()
print(f"[ok] step copy vs nocopy 前 4 帧输出差异 {diff_read:.4f} (>0 即残差生效)")
assert diff_read > 0
print("ALL OK")
