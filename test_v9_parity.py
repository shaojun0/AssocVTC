#!/usr/bin/env python
"""v9 与 v10(prefix 模式) 的数值一致性自检（CPU 可跑，不需要 ckpt/GPU）。

v9 应当只是 v10 在 `mem_mode="prefix"` 下的**等价精简版**：
* 读出侧参数集合相同（v9 没有 kv_pos；v10 prefix 模式下 kv_pos=None）；
* 同一份权重、同一份 z → 输出 pred / preds / loss 完全一致。
这样「v9 训练出 27.98 dB」的结论才能直接与 v10 代码对齐。

    python test_v9_parity.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v10 import V10Config, VideoV10, WholeClipDecoder    # noqa: E402
from model_v9 import V9Config, VideoV9, ParallelReadout        # noqa: E402

K, C, W = 4, 8, 32            # 小号：4 个 special / token 宽 8 / 向量宽 32
COMMON = dict(frames=8, group=4, rate_tokens=4, rate_dim=C, target_tokens=K,
              chunk_tokens=4, mid_dim=W, dec_dim=W, mid_depth=2, mid_heads=2,
              mid_ratio=2.0, dec_depth=2, dec_heads=2, dec_ratio=2.0)


OLD = dict(step_gate=True, carry_ln=True, readout_mask=True)   # v9 历史口径
ALIGNED = dict(step_gate=False, carry_ln=False, readout_mask=False)  # v2 对齐口径


def check_readout() -> None:
    for tag, flags in (("old", OLD), ("aligned", ALIGNED)):
        c10 = V10Config(**COMMON, mem_mode="prefix", loss_read=True,
                        copy_skip=True, **flags)
        c9 = V9Config(**COMMON, loss_read=True, copy_skip=True, **flags)
        torch.manual_seed(0)
        r10 = WholeClipDecoder(c10, code_dim=C, num_specials=K)
        torch.manual_seed(0)
        r9 = ParallelReadout(c9, code_dim=C, num_specials=K)
        sd10, sd9 = r10.state_dict(), r9.state_dict()
        assert set(sd10) == set(sd9), (set(sd10) ^ set(sd9))
        r9.load_state_dict(sd10)                   # 直接搬权重
        z = torch.randn(2, c10.n_steps, c10.chunks_per_step, c10.width)
        o10 = r10(z, return_step_preds=True)
        o9 = r9(z, return_step_preds=True)
        for k in ("pred", "preds"):
            d = (o10[k] - o9[k]).abs().max().item()
            assert d == 0.0, (tag, k, d)
        print(f"[ok] {tag}: ParallelReadout == WholeClipDecoder(prefix)  "
              f"params={sum(p.numel() for p in r9.parameters())/1e3:.1f}K  max|Δ|=0")


def check_video() -> None:
    tok_cfg = dict(img_size=32, patch=16, num_specials=K, enc_dim=32, enc_depth=1,
                   enc_heads=2, enc_mlp_ratio=4.0, enc_attn="std",
                   dec_dim=32, dec_depth=1, dec_heads=2, dec_mlp_ratio=4.0,
                   pixel_hidden=32, zs_dim=C, dec_recurrent=False,
                   dec_steps="1", dec_step_plan="square")
    c10 = V10Config(tok_ckpt="", tok_cfg=tok_cfg, **COMMON,
                    mem_mode="prefix", loss_read=True, copy_skip=True, **OLD)
    c9 = V9Config(tok_ckpt="", tok_cfg=tok_cfg, **COMMON,
                  loss_read=True, copy_skip=True, **OLD)
    m9 = VideoV9(c9).eval()
    m10 = VideoV10(c10).eval()
    m10.load_state_dict(m9.state_dict())           # v9 → v10 权重
    torch.manual_seed(1)
    z_cls = torch.randn(2, c9.frames, 1, C)
    z_s = torch.randn(2, c9.frames, K, C)
    with torch.no_grad():
        o9 = m9(z_cls=z_cls, z_s=z_s)
        o10 = m10(z_cls=z_cls, z_s=z_s)
    d_loss = (o9["loss"] - o10["loss"]).abs().item()
    d_pred = (o9["pred"] - o10["pred"]).abs().max().item()
    assert d_loss == 0.0 and d_pred == 0.0, (d_loss, d_pred)
    print(f"[ok] VideoV9 == VideoV10(prefix)  loss={float(o9['loss']):.6f}  "
          f"pred max|Δ|={d_pred}")


if __name__ == "__main__":
    check_readout()
    check_video()
    print("ALL OK")
