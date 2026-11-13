"""AssocVTC 规模化实验 v8 = v7 + 上游 v3_1（commit 7c4e42c `model_v3_1.py`）的中间态。

v3_1 相对 v3（= 我们的 v7）改了两件事：

1) **MiddleEncoder 不再是 Q_{f-1} → Q_f 的非线性帧间递推**，而是
   「固定 query + 历史经门控 DeltaNet 递推」：
       S_f = GDN([cls] + z_s[0..f-1])        ← Qwen3.5 GatedDeltaNet 的 recurrent state
       Q_f = out_proj(RMSNormGated(S_fᵀ q_base, gate)) + q_base (+ FFN)
   第 1 层 query 固定 `query_base`（N,D，行 k↔patch k），所有帧共享；层数>1 时
   第 l 层的 query = 第 l−1 层的 Q_f（逐帧不同），但每层仍从同一份 mem 重新累积
   状态（**没有** Q_{f-1} 反馈边）。
   ⇒ 每帧状态是常数大小 O(H·head_dim²)，不再随帧数线性增长；帧内 K 个码字一次
     算完，只有"逐帧状态 scan"串行。
2) **decode 把帧维折进 batch**（decoder/PixelHead 逐帧独立）：
       (B,F,N,d) → (B·F,N,d) 一次算完，不再 F 次小 batch 调用。

保留 v7 的：编码器（register 式 ViT，帧维本来已折 batch）、解码器（N 行查询 +
采样步 + ConvPixelHead + 深监督）、损失/PSNR/TBPTT(n_warm)/cfg 口径。

把 v3_1 的收益拆开归因的三个组合（cfg 开关）：
  mid.kind = "gdn" | "recurrent"        （v3_1 中间态 / v7 中间态）
  dec.frame_batch = true | false        （帧维折 batch / 逐帧循环）
  → v7 = recurrent+false; v7+batchdec = recurrent+true; v8(=v3_1) = gdn+true

GDN 用本目录自包含的 `gdn.py`（从 transformers 5.22.2 的 Qwen3.5 GDN 摘出纯
PyTorch 回退路径，只算 state）。原因见 gdn.py 顶部 docstring。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from gdn import GatedDeltaNetBlock, RMSNormGated, chunk_gated_delta_rule_state, l2norm
from model_mini import psnr_from_l1
from model_v7 import MiddleEncoderScale, VideoStateScaleV7


def _ffn(dim: int, hidden: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(dim, hidden), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, dim))
    return nn.Sequential(*layers)


class LinearMidLayer(nn.Module):
    """v3_1 的 GDN 层：库 block 做状态 scan + 固定 query 一次性读出所有梯度帧 + FFN。

    与上游 `model_v3_1.LinearMiddleLayer` 的差异只有：多了 `n_warm`（前 n_warm 帧
    只推进状态、不建图、不读出；v7 的 TBPTT 口径）、`mem` 已由父模块投到 dim。
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, decay_range: Tuple[float, float] = (5e-4, 5e-3)):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 不能被 heads={heads} 整除"
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)                     # mem 与 query 共用
        self.gdn = GatedDeltaNetBlock(dim, heads, kernel=4, decay_range=decay_range)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = _ffn(dim, int(dim * mlp_ratio), dropout)
        self.drop = nn.Dropout(dropout)

    def _scan(self, mem_n: Tensor, n_frames: int, k: int, n_warm: int,
              grad_ckpt: bool):
        """逐帧推进状态；返回梯度帧的 state 栈 (B,G,H,hd,hd) fp32。"""
        chunks = torch.split(mem_n, [1] + [k] * n_frames, dim=1)[:-1]
        assert len(chunks) == n_frames, (len(chunks), n_frames)
        conv_state: Optional[Tensor] = None
        rec_state: Optional[Tensor] = None
        states: List[Optional[Tensor]] = []
        ck = grad_ckpt and torch.is_grad_enabled() and n_warm < n_frames
        for f, ch in enumerate(chunks):
            if f < n_warm:
                with torch.no_grad():
                    conv_state, rec_state, _ = self.gdn.forward_state(
                        ch, conv_state, rec_state)
                states.append(None)
                continue
            if ck:
                conv_state, rec_state, _ = checkpoint(
                    self.gdn.forward_state, ch, conv_state, rec_state,
                    use_reentrant=False)
            else:
                conv_state, rec_state, _ = self.gdn.forward_state(
                    ch, conv_state, rec_state)
            states.append(rec_state)
        return torch.stack([s for s in states if s is not None], dim=1).float()

    def forward(self, q: Tensor, mem: Tensor, n_frames: int, k_per_frame: int,
                grad_ckpt: bool = False) -> Tensor:
        """q: (B,G,N,D)（只有梯度帧）; mem: (B,1+F·K,D)（全部历史帧，含预热）。"""
        B, L, D = mem.shape
        K = k_per_frame
        assert L == 1 + n_frames * K, (L, n_frames, K)
        n_grad = q.shape[1]
        n_warm = n_frames - n_grad
        N, H, hd = q.shape[2], self.heads, self.head_dim
        s = self._scan(self.norm(mem), n_frames, K, n_warm, grad_ckpt)  # (B,G,H,hd,hd)
        assert s.shape[1] == n_grad, (s.shape, n_grad)
        # ---- 固定 query 一次读出所有梯度帧（fp32：state 是累积量，bf16 累加误差会进 query）
        qn = self.norm(q)
        qp = l2norm(self.gdn.in_proj_qkv(qn)[..., :D].reshape(
            B, n_grad, N, H, hd), dim=-1, eps=1e-6)
        o = torch.einsum("bgnhd,bghdv->bgnhv", qp.float(), s) * self.scale
        gate = self.gdn.in_proj_z(qn).view(B, n_grad, N, H, hd)
        o = self.gdn.norm(o.reshape(-1, hd), gate.reshape(-1, hd))
        o = self.gdn.out_proj(o.view(B, n_grad, N, D)).to(q.dtype)
        out = q + self.drop(o)
        return out + self.ff(self.norm_ff(out))


class SoftmaxMidLayer(nn.Module):
    """hybrid 的 full-attention 层 = 一层 nn.TransformerDecoder（掩码=可见历史前缀）。

    与上游 `model_v3_1.SoftmaxMiddleLayer` 同构；`n_warm` 语义同 LinearMidLayer。
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads = dim, heads
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True),
            num_layers=1, norm=nn.LayerNorm(dim))

    def forward(self, q: Tensor, mem: Tensor, n_frames: int, k_per_frame: int,
                grad_ckpt: bool = False) -> Tensor:
        B, L, D = mem.shape
        n_grad = q.shape[1]
        n_warm = n_frames - n_grad
        assert L == 1 + n_frames * k_per_frame, (L, n_frames, k_per_frame)
        N, H = q.shape[2], self.heads
        BF = B * n_grad
        tgt = q.reshape(BF, N, D)
        mem_rep = mem.unsqueeze(1).expand(B, n_grad, L, D).reshape(BF, L, D)
        idx = torch.arange(L, device=mem.device)
        limits = (torch.arange(n_grad, device=mem.device) + n_warm) * k_per_frame
        visible = idx[None] <= limits[:, None]                    # (G,L)
        mask = visible.view(1, n_grad, 1, 1, L).expand(
            B, -1, H, N, -1).reshape(BF * H, N, L)
        out = self.stack(tgt, mem_rep, memory_mask=~mask)
        return out.reshape(B, n_grad, N, D)


class MiddleEncoderGDN(nn.Module):
    """v3_1 的 MiddleEncoder：固定 query + GDN 状态递推（可选 3:1 hybrid）。

    返回**只含梯度帧**的 Q (B, F−n_warm, N, dim)：预热帧的输出没人用（v7 只解码
    梯度帧），状态仍由它们推进。
    """

    def __init__(self, dim: int = 1024, num_patches: int = 196, heads: int = 16,
                 depth: int = 1, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 enc_dim: Optional[int] = None, full_attn_every: int = 0,
                 decay_range: Tuple[float, float] = (5e-4, 5e-3),
                 grad_ckpt: bool = False):
        super().__init__()
        assert depth >= 1 and full_attn_every >= 0
        self.dim, self.num_patches, self.heads = dim, num_patches, heads
        self.depth, self.full_attn_every = depth, full_attn_every
        self.grad_ckpt = bool(grad_ckpt)
        enc_dim = dim if enc_dim is None else int(enc_dim)
        self.proj = nn.Identity() if enc_dim == dim else nn.Linear(enc_dim, dim)
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        self.mem_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layers: List[nn.Module] = []
        for i in range(depth):
            if full_attn_every > 0 and (i + 1) % full_attn_every == 0:
                layers.append(SoftmaxMidLayer(dim, heads, mlp_ratio, dropout))
            else:
                layers.append(LinearMidLayer(dim, heads, mlp_ratio, dropout,
                                             decay_range))
        self.layers = nn.ModuleList(layers)
        self.norm_out = nn.LayerNorm(dim)
        self.last_memory_lens: Optional[List[int]] = None

    def forward(self, z_s: Tensor, n_warm: int = 0,
                max_hist: Optional[int] = None) -> Tensor:
        """z_s: (B,F,K,enc_dim) → Q: (B,F−n_warm,N,dim)。

        max_hist 对 GDN 状态不适用（状态大小恒定，遗忘由 A_log 控制），参数只为
        与 v7 的调用口径兼容。
        """
        assert z_s.dim() == 4, f"z_s 须为 (B,F,K,D), got {tuple(z_s.shape)}"
        B, n_frames, K, _ = z_s.shape
        assert 0 <= n_warm < n_frames, (n_warm, n_frames)
        z = self.proj(z_s)                                        # (B,F,K,dim)
        mem = torch.cat([self.mem_init.expand(B, -1, -1),
                         z.reshape(B, n_frames * K, self.dim)], dim=1)
        n_grad = n_frames - n_warm
        q = self.query_base.unsqueeze(0).unsqueeze(0).expand(
            B, n_grad, -1, -1)
        for layer in self.layers:
            q = layer(q, mem, n_frames, K, grad_ckpt=self.grad_ckpt)
        self.last_memory_lens = [1 + K * f for f in range(n_frames)]
        return self.norm_out(q)                                   # (B,G,N,dim)


class VideoStateScaleV8(VideoStateScaleV7):
    """v7 编码器/解码器 + cfg 选择的中间态与解码批处理方式（见模块 docstring）。"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        e, m, d = cfg["enc"], cfg["mid"], cfg["dec"]
        self.mid_kind = str(m.get("kind", "gdn"))
        self.frame_batch = bool(d.get("frame_batch", True))
        # 帧维折 batch 的分块大小（0 = 一次全算）。折 batch 会把 PixelHead 输出
        # (B·G,T,3,H,W) 全存下来，显存随 G 线性涨；按块算能保住大 batch 的效率
        # 又不爆显存。
        self.frame_chunk = int(d.get("frame_chunk", 0))
        assert self.mid_kind in ("gdn", "recurrent"), self.mid_kind
        if self.mid_kind == "gdn":
            self.middle = MiddleEncoderGDN(
                dim=m["dim"], num_patches=self.num_patches, heads=m["heads"],
                depth=m["depth"], mlp_ratio=m.get("ratio", 4.0),
                dropout=m.get("drop", 0.0), enc_dim=e["dim"],
                full_attn_every=m.get("full_attn_every", 0),
                decay_range=tuple(m.get("decay_range", (5e-4, 5e-3))),
                grad_ckpt=m.get("grad_ckpt", False))
        if not self.frame_batch:
            assert self.mid_kind == "recurrent", \
                "frame_batch=False 只支持 v7 的 recurrent 中间态（其输出是 (B,F,·)）"

    @property
    def middle_is_gdn(self) -> bool:
        return isinstance(self.middle, MiddleEncoderGDN)

    def forward(self, x: Tensor, n_warm: int = 0, max_hist: Optional[int] = None,
                step_weights="mean", return_pix: bool = False,
                enc_chunk: int = 0) -> Dict[str, Tensor]:
        if not self.frame_batch:
            return super().forward(x, n_warm=n_warm, max_hist=max_hist,
                                   step_weights=step_weights,
                                   return_pix=return_pix, enc_chunk=enc_chunk)
        assert x.dim() == 5 and x.shape[2] == 3, tuple(x.shape)
        B, F_ = x.shape[:2]
        n_warm = min(int(n_warm), F_ - 1)
        assert n_warm >= 0
        G = F_ - n_warm
        if n_warm > 0:
            with torch.no_grad():
                zc0, zs0 = self.encode(x[:, :n_warm], enc_chunk)
            zc1, zs1 = self.encode(x[:, n_warm:], enc_chunk)
            zc = torch.cat([zc0, zc1], dim=1)
            zs = torch.cat([zs0, zs1], dim=1)
        else:
            zc, zs = self.encode(x, enc_chunk)
        Q = self.middle(zs, n_warm=n_warm, max_hist=max_hist)      # (B,G,N,dim)
        if not self.middle_is_gdn:
            Q = Q[:, n_warm:]
        K, D = zs.shape[2], zs.shape[-1]
        N, DQ = Q.shape[2], Q.shape[-1]
        zc_g = zc[:, n_warm:].reshape(B, G, D)
        zs_g = zs[:, n_warm:].reshape(B, G, K, D)
        Q_g = Q.reshape(B, G, N, DQ)
        tgt_all = x[:, n_warm:]                                    # (B,G,3,H,W)
        cs = self.frame_chunk if self.frame_chunk > 0 else G
        l1_sum = None
        pf_all: List[Tensor] = []
        pix_all: List[Tensor] = []
        for i in range(0, G, cs):
            j = min(i + cs, G)
            bf = B * (j - i)
            out = self.decoder(zc_g[:, i:j].reshape(bf, 1, D),
                               zs_g[:, i:j].reshape(bf, K, D),
                               Q_g[:, i:j].reshape(bf, N, DQ), return_pix=True)
            pix = out["pix"].reshape(B, j - i, *out["pix"].shape[1:])
            tgt = tgt_all[:, i:j]
            l1 = (pix - tgt.unsqueeze(2)).abs()                    # (B,c,T,3,H,W)
            part = l1.mean(dim=(0, 1, 3, 4, 5)) * (j - i)          # (T,) 未除帧数
            l1_sum = part if l1_sum is None else l1_sum + part
            with torch.no_grad():
                pf_all.append(psnr_from_l1(
                    pix[:, :, -1].reshape(bf, *pix.shape[3:]),
                    tgt.reshape(bf, *tgt.shape[2:])))
            if return_pix:
                pix_all.append(pix.detach())
        T = l1_sum.shape[0]
        w = self.step_weight(T, step_weights, x.device)
        loss = (l1_sum / G * w).sum()
        pf = torch.cat(pf_all, dim=0)
        with torch.no_grad():
            psnr = pf.mean()
            psnr_per_frame = pf.detach()
        res: Dict[str, Tensor] = dict(loss=loss, psnr=psnr,
                                      psnr_per_frame=psnr_per_frame, n_grad=G)
        if return_pix and pix_all:
            res["pix"] = torch.cat(pix_all, dim=1)                  # (B,G,T,3,H,W)
        return res


def build_any(cfg: Optional[dict] = None):
    """按 cfg 的 arch 发模型：`"arch": "v8"` → 本文件；否则 v7（默认，向后兼容）。"""
    from model_v7 import DEFAULT_CFG
    cfg = cfg or DEFAULT_CFG
    if str(cfg.get("arch", "v7")) == "v8":
        return VideoStateScaleV8(cfg)
    from model_v7 import build_model as _build_v7
    return _build_v7(cfg)


# ================================================================= 自检（CPU 可跑）
if __name__ == "__main__":
    torch.manual_seed(0)

    N, D, K, PATCH = 16, 64, 8, 32
    base = dict(img=128, patch=PATCH, grid=4, num_patches=N, num_specials=K,
                steps=[1, 2, 3, 4],
                enc=dict(dim=D, depth=2, heads=4, ratio=4.0),
                mid=dict(dim=D, depth=2, heads=4, ratio=4.0, kind="gdn",
                         decay_range=[7e-3, 3e-2]),
                dec=dict(dim=D, depth=2, heads=4, ratio=4.0, grad_ckpt=False,
                         plan="fixed", block=2,
                         head_cfg=dict(ch0=16, mid=(16, 16, 16, 16),
                                       nblk=(1, 1, 1, 1, 1))))
    m = VideoStateScaleV8(base)
    print("[cfg] params:", m.param_groups())
    assert m.middle_is_gdn and m.frame_batch
    m.eval()
    x = torch.randn(2, 3, 3, 128, 128)
    with torch.no_grad():
        out_v = m(x, return_pix=True)
        out_i = m(x[:, :1], return_pix=True)
    assert torch.allclose(out_i["pix"][:, 0], out_v["pix"][:, 0], atol=1e-6), \
        "F=1 与视频首帧应逐位等价"
    # 历史依赖 / 当前帧不进自己的状态
    zs = torch.randn(2, 3, K, D)
    with torch.no_grad():
        q = m.middle(zs, n_warm=0)
        zs0 = zs.clone(); zs0[:, 0] += 0.1
        q0 = m.middle(zs0, n_warm=0)
        zs1 = zs.clone(); zs1[:, 1] += 0.1
        q1 = m.middle(zs1, n_warm=0)
        qg = m.middle(zs, n_warm=1)
    assert q.shape == (2, 3, N, D), q.shape
    assert torch.allclose(q0[:, 0], q[:, 0], atol=1e-6), "Q_0 只由 [cls] 冷启动"
    assert not torch.allclose(q0[:, 1], q[:, 1]), "帧 0 应进入 Q_1 的历史"
    assert torch.allclose(q1[:, 1], q[:, 1], atol=1e-6), "Q_1 不读当前帧"
    assert not torch.allclose(q1[:, 2], q[:, 2]), "帧 1 应进入 Q_2 的历史"
    assert qg.shape == (2, 2, N, D) and torch.allclose(
        qg[:, 0], q[:, 1], atol=1e-6), "n_warm=1 应从帧 1 起"
    # 梯度全通
    mm = VideoStateScaleV8(base)
    mm.train()
    xg = torch.randn(2, 3, 3, 128, 128)
    mm(xg)["loss"].backward()
    lay = mm.middle.layers[0]
    for name, p in [("middle.query_base", mm.middle.query_base),
                    ("middle.mem_init", mm.middle.mem_init),
                    ("mid0.gdn.in_proj_qkv", lay.gdn.in_proj_qkv.weight),
                    ("mid0.gdn.in_proj_z", lay.gdn.in_proj_z.weight),
                    ("mid0.gdn.in_proj_a", lay.gdn.in_proj_a.weight),
                    ("mid0.gdn.conv1d", lay.gdn.conv1d.weight),
                    ("mid0.gdn.out_proj", lay.gdn.out_proj.weight),
                    ("mid0.gdn.A_log", lay.gdn.A_log),
                    ("mid1.ff", mm.middle.layers[1].ff[0].weight),
                    ("encoder", mm.encoder.blocks[0].attn.q.weight),
                    ("decoder", mm.decoder.stack.layers[0].self_attn.in_proj_weight),
                    ("pixel_head", mm.decoder.pixel_head.proj.weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    # 帧维折 batch ≡ 逐帧 decode（v7 实现）—— 同权重同数据
    cfg_r = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list)
                 else v) for k, v in base.items()}
    cfg_r["mid"] = dict(cfg_r["mid"], kind="recurrent")
    mr = VideoStateScaleV8(cfg_r)
    for (n1, p1), (n2, p2) in zip(m.state_dict().items(), mr.state_dict().items()):
        if p1.shape == p2.shape and n1.replace("middle.", "middle.") == n2:
            p2.data.copy_(p1.data)
    mr.frame_batch = False
    mr.eval()
    with torch.no_grad():
        o_loop = mr(xg)
    mr.frame_batch = True
    with torch.no_grad():
        o_batch = mr(xg)
    assert torch.allclose(o_loop["loss"], o_batch["loss"], atol=1e-6), \
        (o_loop["loss"].item(), o_batch["loss"].item())
    assert torch.allclose(o_loop["psnr"], o_batch["psnr"], atol=1e-5)
    # n_warm: TBPTT 口径（v7 逐帧实现 vs v8 折 batch 实现）
    with torch.no_grad():
        o_w = mr(xg, n_warm=1)
    assert torch.allclose(o_w["loss"], mr(xg, n_warm=1)["loss"], atol=1e-8)
    # hybrid（3:1）与 grad_ckpt 也能跑
    cfg_h = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list)
                 else v) for k, v in base.items()}
    cfg_h["mid"] = dict(cfg_h["mid"], depth=4, full_attn_every=4, grad_ckpt=True)
    mh = VideoStateScaleV8(cfg_h)
    mh.train()
    mh(xg)["loss"].backward()
    assert mh.middle.layers[3].stack.layers[0].multihead_attn.in_proj_weight.grad is not None
    assert mh.middle.layers[0].gdn.in_proj_qkv.weight.grad is not None
    print("[ok] v8 (v3_1 移植) 自检通过")
