"""v9 = v8 的**并行读出解码器**（encoder / middle 一行不改）。

设计（2026-11-13, 对话中定）:
    v8 的解码器是「顺序 16 步 × TransformerDecoder + carry」——每步一步前向、
    每步一次 ConvPixelHead（448² 上采样），实测 193 ms/监督帧（A800），其中
    16 次 conv head ≈ 120 ms、16 步串行 transformer ≈ 50 ms。
    v9 把它换成**一次性并行**：
      1. 16 个读步折进 batch 维：queries (B·T, N, d)，memory (B·T, S, d)，
         用 `memory_key_padding_mask` 实现「第 t 步只许看前缀 A[0 : 1+blk·t]」
         —— 等价于 v2/v7 的 prefix 读口径，但**没有 carry、没有顺序依赖**；
      2. 每步一个可学习 step_emb（否则并行后各步只有掩码不同，梯度信号弱）；
      3. head 换 v2 的**逐 patch 2 层 MLP**（dim→hidden→3·32·32），
         一次调用吃 (B·T·N, d) —— 单次 3.3 GFLOP，比 conv head 的 448²
         多级上采样便宜一个量级，且没有 NCHW↔NHWC 转换开销。
    深监督口径不变（mean_t L1，16 点平权），所以可直接和 v8 的曲线比。

开关（cfg["dec"]）:
    parallel:      false → 走 v8 原路（顺序解码器）; true → 本文件
    head_kind:     "conv"（默认, 沿用 ConvPixelHead）| "mlp"
    mlp_hidden:    MLP head 隐层宽（默认 2048）
    step_emb:      true（默认）
    read_plan:     "fixed"（默认, block 个/步）| "prefix"

自检: python model_v9.py
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model_scale import ConvPixelHead
from model_mini import psnr_from_l1
from model_v7 import (QueryDecoderScale, VideoStateScaleV7, fixed_steps,
                      step_windows, step_windows_fixed)
from model_v8 import VideoStateScaleV8


# ═══════════════════════════════════════════════════════ MLP 像素头（v2 口径）
class MLPPixelHead(nn.Module):
    """(B, N, dim) → (B, 3, H, W)，逐 patch 2 层 MLP（v2 的 PixelHead）。

    patch 向量布局 = (C, patch, patch) 通道优先，patch 顺序 row-major（先 y 后 x），
    是 ConvPixelHead 输出布局的严格同构（同一 target，可直接替换）。
    """

    def __init__(self, dim: int = 1024, grid: int = 14, patch: int = 32,
                 hidden: int = 2048):
        super().__init__()
        self.grid, self.patch = int(grid), int(patch)
        self.net = nn.Sequential(nn.Linear(dim, int(hidden)), nn.GELU(),
                                 nn.Linear(int(hidden), 3 * patch * patch))

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        g, p = self.grid, self.patch
        h = torch.sigmoid(self.net(x))                       # (B,N,3*p*p)
        h = h.reshape(B, g, g, 3, p, p)
        return h.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, g * p, g * p)


# ═══════════════════════════════════════════════════════ 并行读出解码器
class ParallelQueryDecoder(nn.Module):
    """16（T）个读步一次前向：无 carry、无顺序依赖，mem 用 padding mask 控前缀。

    A = [z_cls; z_s] (B, S, dim)，第 t 步允许的 key = A[0 : 1 + blk·t]，
    queries = Q + step_emb[t]（B·T, N, dim），一次 stack 前向 → (B·T, N, dim)，
    head 一次调用 → (B·T, 3, H, W) → (B, T, 3, H, W)。
    """

    def __init__(self, enc_dim: int = 1024, dim: int = 1024, num_patches: int = 196,
                 grid: int = 14, img: int = 448, heads: int = 16, depth: int = 14,
                 ratio: float = 4.0, num_specials: int = 32,
                 steps: Sequence[int] = (1, 4, 9, 16, 25), drop: float = 0.0,
                 head_cfg: Optional[dict] = None, head_kind: str = "conv",
                 mlp_hidden: int = 2048, step_emb: bool = True,
                 read_plan: str = "fixed", block: int = 0):
        super().__init__()
        self.num_patches = int(num_patches)
        self.num_specials = int(num_specials)
        self.grid, self.img = int(grid), int(img)
        self.dim = int(dim)
        self.head_kind = str(head_kind)
        self.steps = [int(s) for s in steps]
        self.T = len(self.steps)
        assert self.T >= 1
        if read_plan == "fixed":
            blk = int(block) if block else max(1, num_specials // self.T)
            self.block = blk
            # 允许的 key 数 = 1(z_cls) + blk·k（k = 第 k 步, 1-based）
            self.limits = [1 + blk * int(k) for k in self.steps]
        elif read_plan == "prefix":
            self.block = 0
            self.limits = [1 + hi for (_lo, hi) in step_windows(num_specials,
                                                                self.steps)]
        else:
            raise ValueError(f"未知 read_plan={read_plan}")
        self.S = num_specials + 1
        assert max(self.limits) <= self.S, (max(self.limits), self.S)

        self.mem_proj = nn.Identity() if enc_dim == dim else nn.Linear(enc_dim, dim)
        self.mem_norm = nn.LayerNorm(dim)
        self.res = nn.Parameter(torch.zeros(1, self.S, dim))
        self.step_emb = (nn.Parameter(torch.randn(1, self.T, 1, dim) * 0.02)
                         if step_emb else None)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        self.stack = nn.TransformerDecoder(layer, num_layers=depth,
                                           norm=nn.LayerNorm(dim))
        if self.head_kind == "mlp":
            self.pixel_head = MLPPixelHead(dim=dim, grid=grid,
                                           patch=max(1, img // grid),
                                           hidden=mlp_hidden)
        elif self.head_kind == "conv":
            self.pixel_head = ConvPixelHead(dim=dim, grid=grid, img=img,
                                            **(head_cfg or {}))
        else:
            raise ValueError(f"未知 head_kind={self.head_kind}")
        # 与 v7/v8 解码器同名属性（日志/评测脚本会读）：第 t 步的读窗口 = 前缀
        self.windows = [(0, lim - 1) for lim in self.limits]
        # 前缀 mask（T, S）: True = 屏蔽
        idx = torch.arange(self.S)
        self.register_buffer(
            "_key_pad", torch.stack([idx >= lim for lim in self.limits]), persistent=False)

    def forward(self, z_cls: Tensor, z_s: Tensor, query: Tensor,
                return_pix: bool = True) -> Dict[str, Tensor]:
        """z_cls (B,1,E), z_s (B,K,E), query (B,N,dim) → pix (B,T,3,H,W)."""
        B, K, _ = z_s.shape
        T = self.T
        A = self.mem_norm(self.mem_proj(torch.cat([z_cls, z_s], dim=1))) + self.res
        q = query.unsqueeze(1)                                # (B,1,N,dim)
        if self.step_emb is not None:
            q = q + self.step_emb                              # (B,T,N,dim)
        q = q.expand(B, T, -1, -1).reshape(B * T, self.num_patches, self.dim)
        mem = A.unsqueeze(1).expand(B, T, self.S, self.dim).reshape(B * T, self.S, self.dim)
        pad = self._key_pad.unsqueeze(0).expand(B, T, self.S).reshape(B * T, self.S)
        Y = self.stack(q, mem, memory_key_padding_mask=pad)   # (B·T, N, dim)
        out: Dict[str, Tensor] = dict(feats=Y)
        if return_pix:
            pix = self.pixel_head(Y.reshape(B * T, self.num_patches, self.dim))
            if pix.dim() == 4:                                # conv head → (B·T,3,H,W)
                out["pix"] = pix.reshape(B, T, *pix.shape[1:])
            else:                                             # 容错
                raise RuntimeError("unexpected pixel head output")
        return out


# ═══════════════════════════════════════════════════════ 总模型（v8 + 并行解码器）
class VideoStateScaleV9(VideoStateScaleV8):
    """encoder / middle 完全复用 v8；只把 decoder 换成 ParallelQueryDecoder。"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)                                # 建 v8 的 encoder/middle
        d = cfg["dec"]
        e = cfg["enc"]
        steps = [int(s) for s in cfg.get("steps", ())]
        if not steps:
            steps = fixed_steps(self.num_specials, int(d.get("block", 2)))
        self.decoder = ParallelQueryDecoder(
            enc_dim=e["dim"], dim=d["dim"], num_patches=self.num_patches,
            grid=self.grid, img=self.img, heads=d["heads"], depth=d["depth"],
            ratio=d.get("ratio", 4.0), num_specials=self.num_specials,
            steps=steps, drop=d.get("drop", 0.0), head_cfg=d.get("head_cfg"),
            head_kind=d.get("head_kind", "mlp"), mlp_hidden=d.get("mlp_hidden", 2048),
            step_emb=bool(d.get("step_emb", True)),
            read_plan=d.get("read_plan", "fixed"), block=int(d.get("block", 0)))
        self.steps = self.decoder.steps


def build_any(cfg: Optional[dict] = None):
    """arch="v9" → 本文件；其它 → 依次退回 v8/v7。"""
    cfg = cfg or {}
    arch = str(cfg.get("arch", "v9"))
    if arch == "v9":
        return VideoStateScaleV9(cfg)
    from model_v8 import build_any as _b8
    return _b8(cfg)


# ═══════════════════════════════════════════════════════ 自检（CPU 可跑）
if __name__ == "__main__":
    torch.manual_seed(0)
    grid, patch, img = 4, 32, 128
    N, K, T, blk = 16, 8, 4, 2
    base = dict(img=img, patch=patch, grid=grid, num_patches=N, num_specials=K,
                steps=list(range(1, T + 1)),
                enc=dict(dim=64, depth=2, heads=4, ratio=4.0),
                mid=dict(dim=64, depth=2, heads=4, ratio=4.0, kind="gdn"),
                dec=dict(dim=64, depth=2, heads=4, ratio=4.0, grad_ckpt=False,
                         plan="fixed", block=blk, parallel=True, head_kind="mlp",
                         mlp_hidden=64,
                         head_cfg=dict(ch0=16, mid=(16, 16, 16, 16),
                                       nblk=(1, 1, 1, 1, 1))))
    m = VideoStateScaleV9(base).eval()
    print("[params]", {k: round(v / 1e6, 3) for k, v in m.param_groups().items()})
    x = torch.rand(2, 3, 3, img, img)                       # (B,F,3,H,W) F=3
    out = m(x, n_warm=0, return_pix=True)
    print("[shape] loss", tuple(out["loss"].shape), "pix",
          tuple(out["pix"].shape), "psnr", float(out["psnr"]))
    assert out["pix"].shape == (2, 3, T, 3, img, img), out["pix"].shape
    assert 0.0 <= float(out["pix"].min()) and float(out["pix"].max()) <= 1.0
    out["loss"].backward()
    gn = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in m.parameters()
                       if p.grad is not None))
    print("[grad] norm", round(gn, 6))
    assert gn > 0, "梯度为 0"

    # ── 因果性：扰动最后一块 z_s，前 T-1 步的输出必须逐位不变
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        m.eval()
        z_cls = torch.randn(1, 1, 64)
        z_s = torch.randn(1, K, 64)
        Q = torch.randn(1, N, 64)
        p0 = m.decoder(z_cls, z_s, Q)["pix"]
        zs2 = z_s.clone()
        zs2[:, -blk:] += torch.randn_like(zs2[:, -blk:])
        p1 = m.decoder(z_cls, zs2, Q)["pix"]
    d = (p0 - p1).abs().amax(dim=(0, 2, 3, 4))              # (T,)
    print("[causal] per-step max|Δ|", [round(float(v), 3) for v in d])
    assert all(float(d[t]) < 1e-6 for t in range(T - 1)), d
    assert float(d[-1]) > 1e-6, "最后一步应看到扰动"

    # ── conv head 分支 + 并行/顺序参数量对比
    base["dec"]["head_kind"] = "conv"
    m2 = VideoStateScaleV9(base)
    print("[conv head] params", m2.param_groups()["total"] / 1e6, "M")
    base["dec"].update(parallel=False, head_kind="conv")
    base["arch"] = "v8"
    m3 = VideoStateScaleV8(base)
    print("[v8 顺序/conv] params", m3.param_groups()["total"] / 1e6, "M")
    print("ALL CHECKS PASSED")
