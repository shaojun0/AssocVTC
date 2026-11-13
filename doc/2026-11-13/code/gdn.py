"""自包含的门控 DeltaNet（GDN）实现 —— 从 transformers 5.22.2 的
`transformers/models/qwen3_5/modeling_qwen3_5.py`（Qwen3.5 的 `Qwen3_5GatedDeltaNet`，
Apache-2.0, Copyright The Qwen Team / HF）里摘出纯 PyTorch 那段（`fla` Triton kernel 缺席时
库自己走的回退路径），改成「只算 state、不算输出」的形态。

为什么自带一份而不是 pip install transformers>=5.22.2：
  · 那版 transformers 硬性要求 torch>=2.6，而 V100(sm_70) 这台机器上只有 torch 2.4.1 可用；
  · 我们只要 GDN 的 **recurrent state**（中间态读出用固定 query），不需要 token 输出
    （库每次都会多算 intra/inter-chunk 注意力输出，白花 2 个 matmul）；
  · 机器上装不到 fla / causal_conv1d 编译内核，库在 GPU 上跑的也正是这份纯 torch 回退。

与原实现逐位一致的部分：conv1d 因果卷积、A_log/dt_bias 门控、chunk 内 UT 变换、
chunk 间 scan、fp32 累加。**省掉**的只有 `core_attn_out`（等于不读历史输出的那部分）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _conv_step(mixed_qkv: Tensor, conv_state: Optional[Tensor],
               weight: Tensor, kernel: int) -> Tuple[Tensor, Optional[Tensor]]:
    """因果 depthwise conv1d（silu 在调用处加）。

    mixed_qkv: (B, conv_dim, L) 本帧新 token；conv_state: (B, conv_dim, k−1) 上帧末尾。
    返回 (out (B,conv_dim,L), new_conv_state)。
    """
    if conv_state is not None:
        inp = torch.cat([conv_state, mixed_qkv], dim=-1)
    else:
        inp = mixed_qkv
    out = F.conv1d(inp, weight.unsqueeze(1), None,
                   padding=kernel - 1, groups=weight.shape[0])
    out = out[:, :, :inp.shape[-1]]
    new_state = inp[..., -(kernel - 1):].contiguous() if kernel > 1 else None
    return out[:, :, -mixed_qkv.shape[-1]:], new_state


def chunk_gated_delta_rule_state(
        key: Tensor, value: Tensor, g: Tensor, beta: Tensor,
        initial_state: Optional[Tensor] = None,
        chunk_size: int = 64) -> Tensor:
    """gated delta rule 的 chunked scan，**只返回最终 recurrent state**。

    key:   (B,L,H,dk)（未归一化；函数内做 L2Norm）
    value: (B,L,H,dv)
    g:     (B,L,H) log 空间的每 token 衰减（≤0）
    beta:  (B,L,H) 写入强度（sigmoid 后）
    initial_state: (B,H,dk,dv) fp32 或 None
    返回 (B,H,dk,dv) fp32。

    算法与上游 `torch_chunk_gated_delta_rule` 相同（UT 变换 + chunk 间串行 scan），
    只是不计算注意力输出（我们不需要读历史输出）。
    """
    B, L, H, dk = key.shape
    dv = value.shape[-1]
    k, v, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (key, value, beta, g)]
    k = l2norm(k, dim=-1, eps=1e-6)
    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad:
        k, v = (F.pad(x, (0, 0, 0, pad)) for x in (k, v))
        beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    T = L + pad
    nc = T // chunk_size
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    k, k_beta, v_beta = [x.reshape(B, H, nc, chunk_size, x.shape[-1])
                         for x in (k, k_beta, v_beta)]
    decay = decay.reshape(B, H, nc, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device).triu(1)
    cum = decay.cumsum(dim=3)
    pairwise = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False,
                                               unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut, decayed_k_beta, upper=False,
                                               unitriangular=True)
    if initial_state is None:
        state = torch.zeros(B, H, dk, dv, dtype=v.dtype, device=v.device)
    else:
        state = initial_state.to(v.dtype)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    key_dec = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    for i in range(nc):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        state = state * chunk_decay[:, :, i] + key_dec[:, :, i].transpose(-1, -2) @ v_new
    return state


class RMSNormGated(nn.Module):
    """门控 RMSNorm（与 Qwen3_5RMSNormGated 同语义：先归一化再乘 silu(gate)）。"""

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        xf = self.weight * xf.to(dt)
        return (xf * F.silu(gate.float())).to(dt)


class GatedDeltaNetBlock(nn.Module):
    """Qwen3_5GatedDeltaNet 的参数化版（state-only）。

    参数名与库实现一一对应（in_proj_qkv / in_proj_z / in_proj_b / in_proj_a /
    conv1d / A_log / dt_bias / norm / out_proj），便于与上游 checkpoint 对齐。
    """

    def __init__(self, dim: int, heads: int, kernel: int = 4, eps: float = 1e-6,
                 decay_range: Tuple[float, float] = (5e-4, 5e-3)):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.hidden_size, self.heads = dim, heads
        self.head_dim = dim // heads
        self.key_dim = self.value_dim = dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.kernel_size = kernel
        self.in_proj_qkv = nn.Linear(dim, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(dim, heads, bias=False)
        self.in_proj_a = nn.Linear(dim, heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel,
                                groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(heads))
        # A = 每 token 遗忘率: γ_tok = exp(−A·softplus(a+dt_bias)), 初始 softplus(1)≈1.31
        # ⇒ 半衰期 ≈ 0.69/(A·1.31) 个 token。库默认 A∈[0.01,16] 半衰期 <1 token，
        # 历史跨不了帧；这里按上游 v3_1 的推荐区间重设。
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(*decay_range)))
        self.norm = RMSNormGated(self.head_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, dim, bias=False)

    def forward_state(self, x: Tensor, conv_state: Optional[Tensor] = None,
                      rec_state: Optional[Tensor] = None,
                      chunk_size: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """x: (B,L,dim) → (new_conv_state (B,conv_dim,k−1), new_state (B,H,hd,hd), z)。"""
        B, L, _ = x.shape
        mqkv = self.in_proj_qkv(x).transpose(1, 2)                 # (B,conv_dim,L)
        z = self.in_proj_z(x).reshape(B, L, self.heads, self.head_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        y, new_conv = _conv_step(mqkv, conv_state,
                                 self.conv1d.weight.squeeze(1),
                                 self.kernel_size)
        y = F.silu(y).transpose(1, 2)                              # (B,L,conv_dim)
        qkv = torch.split(y, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q, k, v = (t.reshape(B, L, self.heads, self.head_dim) for t in qkv)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        cs = int(chunk_size) if chunk_size and chunk_size > 0 else L
        state = chunk_gated_delta_rule_state(k, v, g, beta,
                                             initial_state=rec_state,
                                             chunk_size=cs)
        return new_conv, state, z


# ═══ 数值自检（python gdn.py）：chunked state ≡ 逐 token delta rule ═══
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, H, dk, dv = 2, 7, 4, 8, 8
    k = torch.randn(B, L, H, dk)
    v = torch.randn(B, L, H, dv)
    g = -torch.rand(B, L, H)
    beta = torch.rand(B, L, H)
    s_chunk = chunk_gated_delta_rule_state(k, v, g, beta, chunk_size=4)

    # 逐 token 参考实现（delta rule 的定义式）
    kn = l2norm(k.transpose(1, 2).float(), dim=-1)
    vn = v.transpose(1, 2).float()
    st = torch.zeros(B, H, dk, dv)
    for t in range(L):
        st = st * g[:, t].exp()[..., None, None]
        kv = (st * kn[:, :, t].unsqueeze(-1)).sum(dim=-2)
        delta = (vn[:, :, t] - kv) * beta[:, t].unsqueeze(-1)
        st = st + kn[:, :, t].unsqueeze(-1) * delta.unsqueeze(-2)
    assert torch.allclose(s_chunk, st, atol=1e-5, rtol=1e-5), \
        (s_chunk - st).abs().max().item()

    # 流式两段 ≡ 整段一次（中间态逐帧 carry 的正确性）
    s_a = chunk_gated_delta_rule_state(k[:, :3], v[:, :3], g[:, :3],
                                       beta[:, :3], chunk_size=3)
    s_b = chunk_gated_delta_rule_state(k[:, 3:], v[:, 3:], g[:, 3:],
                                       beta[:, 3:], initial_state=s_a, chunk_size=4)
    assert torch.allclose(s_b, s_chunk, atol=1e-5, rtol=1e-5), \
        (s_b - s_chunk).abs().max().item()

    # conv 流式 ≡ 整段（kernel=4, 逐帧 K=3）
    blk = GatedDeltaNetBlock(16, 4, kernel=4)
    x = torch.randn(B, L, 16)
    _, s_full, _ = blk.forward_state(x)
    cs = rs = None
    for i in range(0, L, 3):
        cs, rs, _ = blk.forward_state(x[:, i:i + 3], cs, rs)
    assert torch.allclose(rs, s_full, atol=1e-5, rtol=1e-5), \
        (rs - s_full).abs().max().item()
    print("[ok] GDN: chunked state ≡ 逐 token; 流式 carry ≡ 整段; conv 流式 ≡ 整段")
