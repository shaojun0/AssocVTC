"""v10 = **全片段循环读出**的视频模型（2026-11-16 设计定稿）。

和 v8/v9 的关系
--------------
v8/v9 是「逐帧读出 + 帧间状态」：解码器一次只解一帧（或一个窗口），帧间靠
middle 递推的状态传递。v10 反过来：**每一步都把整段 120 帧重出一次**，
每步多读一组 z（4 帧 = 0.5 s），读出逐步细化；状态 $Y_t$ 是 480 行×整段的
查询状态，30 步后 $Y_{30}$ 就是最终重建。这样一来

    第 t 档 = 只读前 t+1 组 z（= 时间前缀），其余档由"多读一组"决定

和论文「前 κ 个 token 即前缀」的读法是同一个东西，只是把"读多少 token"
换成了"读多少时间"。代价是每步都要算整段（30× 计算），收益是后文信息
可以回改前面帧（B 帧式双向细化）——这一步是 2026-11-16 明确选定的。

数据流（数字口径：448² / patch 32 / N=196 / K=32 / zs_dim=448）
--------------------------------------------------------------
::

    x            (B, 120, 3, 448, 448)                 视频片段 15 s @ 8 fps
    E（冻结）    每帧独立前向 → z_cls (B,120,1,448), z_s (B,120,32,448)
    middle 输入  拼 [z_cls; z_s] 共 33 token/帧 = (B, 3960, 448)
                 → 帧内 block 因果（block = 33，一帧一块）+ 6 层 Transformer
    compressor   30 组 × 每组 4 帧（132 个混合 token）→ 16 个 rate token
                 z = (B, 30, 64, 448)  ⇒ 64×448×30 = 860,160 值 = 7,168 值/帧
                 （= 用户口径的 "16 chunks × 1792"，同一张量换视图；见 DESIGN）
    读出（本文件核心）
                 Q（可学习，随机初始化）(1, 3960, d)；γ_0 = 0
                 for t in 0..29:
                     h  = LN(Q + γ_t·Y_{t-1}) + step_emb[t]
                     Y_t = Stack( 帧内自注意力 + 交叉注意力到 z_t + FFN )(h)
                     pred_t = head(Y_t) → (B, 120, 33, 448)   每步都出整段
                 loss = mean_t L1(pred_t, 冻结编码器给的目标（全 33 token）)
    R（冻结）    把 pred_30 还原成像素，用于 PSNR 监控/评测

为什么「率 < 目标」
------------------
如果 decoder 只需要复现"被截断的那 16 个 token"，最好情况就是逐位复现，
经 R 渲染 = 图像分词器第 8 档的质量 ⇒ 上限 = 逐帧基线，实验必然平局。
所以这里**率 = 前 16 个 token（7168 值/帧），目标 = 全部 33 个位置
（1+32 个 token，14,784 值/帧）**，跨帧混合才有地方兑现：
用邻帧预测被截断的细节（视频编码里 P/B 帧做的事）。

可改的旋钮（V10Config）
-----------------------
rate_tokens     率：每帧保留多少 special token（默认 16，= 前缀截断）
rate_dim        率宽度（= 分词器 zs_dim，448；0 = 原生 enc_dim）
target_tokens   目标 token 数（默认 32 = 全量；含 z_cls 另算）
mid_in_tokens   中间态读入 token 数/帧（默认 33 = 全量 [z_cls; z_s]）
group           z 步覆盖帧数（4 → 30 步，2 fps 潜在帧）
dec_depth/dim   读出栈深度/宽度
step_gate       是否用零初始化门控 γ（默认开：训练起点等价于单步读出）
step_weights    逐步损失权重（"mean" 平权 | "lin" 线性加权到末步）

自检：``python model_v10.py``（CPU 可跑，不需要 ckpt）
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn

__all__ = ["V10Config", "FrozenTokenizer", "TemporalMiddle", "StepCompressor",
           "WholeClipDecoder", "VideoV10", "patches_to_image", "psnr_from_l1"]


# ═══════════════════════════════════════════════════════════ 小工具
def patches_to_image(p: Tensor, H: int, W: int, patch: int) -> Tensor:
    """(B, N, 3·p·p) → (B, 3, H, W)。布局同 `image_to_patches`：通道优先、
    patch 顺序 row-major（先 y 后 x）。"""
    g = H // patch
    B = p.shape[0]
    h = p.reshape(B, g, g, 3, patch, patch)
    return h.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, g * patch, g * patch)


def psnr_from_l1(pred: Tensor, target: Tensor, eps: float = 1e-10) -> Tensor:
    """pred/target ∈ [0,1]，按 MSE 定义逐图 PSNR (B,)。"""
    mse = ((pred - target) ** 2).flatten(1).mean(1).clamp_min(eps)
    return -10.0 * torch.log10(mse)


def _fed(x: Tensor) -> int:
    return int(x.shape[-1])


# ═══════════════════════════════════════════════════════════ 配置
@dataclass
class V10Config:
    # ── 冻结图像分词器（E + R 同一个 SRPhase1LinAttn ckpt）
    tok_ckpt: str = ""
    tok_cfg: Dict = field(default_factory=dict)   # 覆盖 ckpt["cfg"] 里的字段
    # ── 视频
    frames: int = 120          # 15 s @ 8 fps
    group: int = 4             # z 步覆盖帧数（30 步）
    # ── 率 / 目标（值数口径见模块 docstring）
    rate_tokens: int = 16      # 每帧保留的 special token 数（前缀截断）
    rate_dim: int = 448        # 率宽度（= 分词器 zs_dim）
    target_tokens: int = 32    # 目标 special token 数（全量）
    # ── 中间态（帧内 block 因果）
    mid_in_tokens: int = 33    # 每帧读入 token 数（[z_cls; z_s] = 1 + num_specials）
    mid_dim: int = 1024
    mid_depth: int = 6
    mid_heads: int = 16
    mid_ratio: float = 4.0
    # ── 读出（整片段循环）
    dec_dim: int = 1024
    dec_depth: int = 4
    dec_heads: int = 16
    dec_ratio: float = 2.0
    dec_drop: float = 0.0
    step_gate: bool = True     # γ 零初始化门控
    step_weights: str = "mean"

    @property
    def n_steps(self) -> int:
        assert self.frames % self.group == 0, (self.frames, self.group)
        return self.frames // self.group

    @property
    def rate_tokens_per_step(self) -> int:
        """每个 z 步的 rate token 数（= rate_tokens × group）。"""
        return self.rate_tokens * self.group

    @property
    def positions_per_clip(self) -> int:
        """读出位置数 = frames × (1 + target_tokens)（含 z_cls 位）。"""
        return self.frames * (1 + self.target_tokens)

    def values_per_frame(self) -> int:
        return self.rate_tokens * self.rate_dim

    def values_per_clip(self) -> int:
        return self.values_per_frame() * self.frames


# ═══════════════════════════════════════════════════════════ 冻结分词器
class FrozenTokenizer(nn.Module):
    """包一个 `SRPhase1LinAttn`：E 与 R 全部冻结（eval + 常量输出）。

    接口：
      encode(x)           (B,F,3,H,W) → z_cls (B,F,1,c) / z_s (B,F,K,c)
      render(z_cls, z_s)  (B,F,1,c)/(B,F,K,c) → pix (B,F,3,H,W)（末档）
      code_dim / num_specials / patch / img 等属性
    """

    def __init__(self, ckpt: str = "", cfg_override: Optional[Dict] = None):
        super().__init__()
        ckpt = str(ckpt or "")
        self.ckpt_path = ckpt
        cfg_dict: Dict = {}
        sd = None
        if ckpt and os.path.isfile(ckpt):
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(blob, dict) and "model" in blob:
                sd, cfg_dict = blob["model"], dict(blob.get("cfg") or {})
            else:                                        # 纯 state_dict（v9 线）
                sd, cfg_dict = blob, dict(cfg_override or {})
        cfg_dict = {**cfg_dict, **dict(cfg_override or {})}
        fields = set(LinAttnConfig.__dataclass_fields__)
        cfg = LinAttnConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
        self.cfg = cfg
        self.tok = SRPhase1LinAttn(cfg)
        if sd is not None:
            missing, unexpected = self.tok.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"分词器权重不匹配: missing={len(missing)} unexpected={len(unexpected)}"
                    f"（前几个: {list(missing)[:4]} / {list(unexpected)[:4]}）")
        for p in self.tok.parameters():
            p.requires_grad_(False)
        self.tok.eval()
        self.img = int(cfg.img_size)
        self.patch = int(cfg.patch)
        self.num_specials = int(cfg.num_specials)
        self.code_dim = int(cfg.zs_dim) if int(cfg.zs_dim) > 0 else int(cfg.enc_dim)
        self.num_patches = int(cfg.num_patches)

    # ── E：逐帧前向（帧维折 batch）
    @torch.no_grad()
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """x (B,F,3,H,W) → z_cls (B,F,1,c), z_s (B,F,K,c)。"""
        assert x.dim() == 5, f"须 (B,F,3,H,W), got {tuple(x.shape)}"
        B, Fr = x.shape[:2]
        flat = x.reshape(B * Fr, *x.shape[2:])
        z_cls, z_s = self.tok.encode(flat)
        return (z_cls.view(B, Fr, 1, -1), z_s.view(B, Fr, self.num_specials, -1))

    # ── R：末档渲染成像素
    @torch.no_grad()
    def render(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        """(B,F,1,c)/(B,F,K,c) → pix (B,F,3,H,W)，取解码器末档。"""
        B, Fr = z_s.shape[:2]
        zc = z_cls.reshape(B * Fr, 1, -1)
        zs = z_s.reshape(B * Fr, self.num_specials, -1)
        out = self.tok.decoder(zc, zs)
        pix = out["pix"] if isinstance(out, dict) else out
        if pix.dim() == 3:                                # (M, N, pp) 单步读出
            pass
        elif pix.dim() == 4:                              # (M, T, N, pp) 逐步读出
            pix = pix[:, -1]                              # 末档 = 读全量
        else:
            raise RuntimeError(f"Unexpected decoder output {tuple(pix.shape)}")
        img = patches_to_image(pix, self.img, self.img, self.patch)
        # 分词器的像素头输出不加激活（训练用归一化像素 L1，见 PixelHead）；
        # 评测口径与 eval json 一致：clamp 到 [0,1] 再算 PSNR。
        return img.clamp(0.0, 1.0).view(B, Fr, 3, self.img, self.img)

    def train(self, mode: bool = True):                   # 永远 eval
        super().train(False)
        self.tok.eval()
        return self


# ═══════════════════════════════════════════════════════════ 基础块
class _Attn(nn.Module):
    def __init__(self, dim: int, heads: int, drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = drop

    def forward(self, x: Tensor, y: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        B, Nq, D = x.shape
        src = x if y is None else y
        q = self.q(x).reshape(B, Nq, self.h, self.dh).transpose(1, 2)
        k = self.k(src).reshape(B, src.shape[1], self.h, self.dh).transpose(1, 2)
        v = self.v(src).reshape(B, src.shape[1], self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.drop if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, Nq, D))


class _FFN(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        h = int(dim * ratio)
        self.a = nn.Linear(dim, h)
        self.b = nn.Linear(h, dim)
        self.drop = drop

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(self.b(F.gelu(self.a(x))), self.drop, self.training)


# ═══════════════════════════════════════════════════════════ 中间态
class TemporalMiddle(nn.Module):
    """帧内 block 因果的 Transformer：一帧之内的 token 互看，跨帧只看历史帧。

    掩码粒度和用户口径一致：**每帧 = 一个 block**，下三角按 block 推进
    （block = tokens_per_frame；在 "4 chunks × 1792" 打包视图下就是每 4 个
    token 解一次 mask，本实现用未打包的 33 token/帧，等价）。
    """

    def __init__(self, in_dim: int, dim: int, depth: int, heads: int,
                 ratio: float, tokens_per_frame: int, frames: int,
                 drop: float = 0.0):
        super().__init__()
        self.tpf = int(tokens_per_frame)
        self.frames = int(frames)
        L = self.tpf * self.frames
        self.in_proj = nn.Linear(in_dim, dim)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                n1=nn.LayerNorm(dim), attn=_Attn(dim, heads, drop),
                n2=nn.LayerNorm(dim), ffn=_FFN(dim, ratio, drop)))
            for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        fi = torch.arange(L) // self.tpf                    # token → 帧号
        mask = fi[None, :] <= fi[:, None]                   # True = 可看
        self.register_buffer("_mask", mask.view(1, 1, L, L), persistent=False)

    def forward(self, z_in: Tensor) -> Tensor:
        """z_in (B, F, T_f, c) → (B, F, T_f, dim)（同形，宽 dim）。"""
        B, Fr, Tf, _ = z_in.shape
        assert Fr == self.frames and Tf == self.tpf, (z_in.shape, self.frames, self.tpf)
        x = self.norm_in(self.in_proj(z_in.reshape(B, Fr * Tf, -1)))
        for blk in self.blocks:
            x = x + blk["attn"](blk["n1"](x), mask=self._mask)
            x = x + blk["ffn"](blk["n2"](x))
        return self.norm_out(x).view(B, Fr, Tf, -1)


class StepCompressor(nn.Module):
    """每步（group 帧 = group·T_f 个混合 token）压成 n_pool 个 rate token。

    Perceiver 式：n_pool 个可学习 latent 做 cross-attention 读这一步的全部
    混合 token，再线性投到 rate_dim。值数：n_pool × rate_dim = group ×
    rate_tokens × rate_dim = group × 7168（每帧 7168 ⇒ 率守恒）。
    """

    def __init__(self, dim: int, rate_dim: int, n_pool: int, heads: int,
                 depth: int = 1, ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_pool, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                nq=nn.LayerNorm(dim), cross=_Attn(dim, heads, drop),
                nf=nn.LayerNorm(dim), ffn=_FFN(dim, ratio, drop)))
            for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, rate_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, S, group·T_f, dim) → z (B, S, n_pool, rate_dim)。"""
        B, S = x.shape[:2]
        kv = x.reshape(B * S, x.shape[2], -1)
        q = self.q.expand(B * S, -1, -1)
        for blk in self.blocks:
            q = q + blk["cross"](blk["nq"](q), kv)
            q = q + blk["ffn"](blk["nf"](q))
        z = self.out(self.norm_out(q))
        return z.view(B, S, z.shape[1], z.shape[2])


# ═══════════════════════════════════════════════════════════ 整片段循环读出
class WholeClipDecoder(nn.Module):
    """Q（随机初始化、可学习）+ 30 步循环：每步读一组 z、重出整段。

    每步结构（读出侧）：
        h  = LN(Q + γ_t·Y_{t-1}) + step_emb[t]        γ 零初始化 ⇒ 起点=单步
        Y  = for layer: h += 帧内自注意力(h)
                        h += 交叉注意力(h, z_t)
                        h += FFN(h)
        pred_t = head(Y)                              (B, F, 1+T, code_dim)
    注意：自注意力限制在**帧内**（33 个位置）。跨帧信息全部来自 z_t（已做
    时间混合）与递推状态 Y_{t-1}——这是 30 步 × 3960 位置还跑得动的原因。
    """

    def __init__(self, cfg: V10Config, code_dim: int, num_specials: int):
        super().__init__()
        self.cfg = cfg
        self.code_dim = int(code_dim)
        self.T = 1 + int(num_specials)                     # 每帧位置数（含 z_cls）
        self.F = int(cfg.frames)
        self.S = int(cfg.n_steps)
        self.P = int(cfg.rate_tokens_per_step)
        d = int(cfg.dec_dim)
        self.d = d
        L = self.F * self.T
        # 查询与状态（整片段一排行）
        self.Q = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(1, self.S, 1, d) * 0.02)
        self.kv_proj = nn.Linear(int(cfg.rate_dim), d)
        self.kv_pos = nn.Parameter(torch.randn(1, self.P, d) * 0.02)
        self.gate = (nn.Parameter(torch.zeros(self.S)) if cfg.step_gate else None)
        self.n_in = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                n1=nn.LayerNorm(d), self_attn=_Attn(d, cfg.dec_heads, cfg.dec_drop),
                n2=nn.LayerNorm(d), cross=_Attn(d, cfg.dec_heads, cfg.dec_drop),
                n3=nn.LayerNorm(d), ffn=_FFN(d, cfg.dec_ratio, cfg.dec_drop)))
            for _ in range(int(cfg.dec_depth))])
        self.norm_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.code_dim)

    def forward(self, z: Tensor, n_steps: Optional[int] = None,
                return_step_preds: bool = False,
                perturb_step: Optional[int] = None) -> Dict[str, Tensor]:
        """z (B,S,P,rate_dim) → dict(pred 末档 (B,F,T,c), preds 可选 (S,...), feats)。

        `perturb_step` 只用于自检（在该步的 z 上加噪，验证因果性）。
        """
        B, S, P, _ = z.shape
        n = self.S if n_steps is None else int(n_steps)
        assert 1 <= n <= S, (n, S)
        kv_all = self.kv_proj(z) + self.kv_pos               # (B,S,P,d)
        Q = self.Q.expand(B, -1, -1)                         # 随机初始化、可学习的查询
        Y = torch.zeros_like(Q)                              # 状态初值 0（第 0 步只看 Q）
        outs: List[Tensor] = []
        for t in range(n):
            kv = kv_all[:, t]
            if perturb_step is not None and t == perturb_step:
                kv = kv + 1.0
            w = 1.0 if self.gate is None else self.gate[t]   # γ 零初始化 ⇒ 起点=单步
            h = self.n_in(Q + w * Y)
            h = h + self.step_emb[:, t]
            # 帧内自注意力：(B,F,T,d) → (B·F,T,d)
            hf = h.view(B * self.F, self.T, self.d)
            for blk in self.blocks:
                hf = hf + blk["self_attn"](blk["n1"](hf))
                h = hf.view(B, self.F * self.T, self.d)
                h = h + blk["cross"](blk["n2"](h), kv)
                h = h + blk["ffn"](blk["n3"](h))
                hf = h.view(B * self.F, self.T, self.d)
            Y = h
            pred = self.head(self.norm_out(Y))               # (B,L,c)
            outs.append(pred.view(B, self.F, self.T, self.code_dim))
        out = dict(pred=outs[-1], feats=Y.view(B, self.F, self.T, self.d), n_steps=n)
        if return_step_preds:
            out["preds"] = torch.stack(outs, dim=1)          # (B,S,F,T,c)
        return out


# ═══════════════════════════════════════════════════════════ 总模型
class VideoV10(nn.Module):
    """冻结分词器 + 中间态 + 整片段循环读出。训练只更新中间态与读出侧。"""

    def __init__(self, cfg: V10Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = FrozenTokenizer(cfg.tok_ckpt, cfg.tok_cfg)
        code = self.tokenizer.code_dim
        assert cfg.rate_dim in (0, code), (
            f"rate_dim={cfg.rate_dim} 与分词器 code_dim={code} 不一致；"
            "0 表示直接用分词器宽度")
        self.code_dim = code
        K = self.tokenizer.num_specials
        assert cfg.target_tokens <= K, (cfg.target_tokens, K)
        assert cfg.rate_tokens <= K, (cfg.rate_tokens, K)
        # 中间态固定读整帧的 [z_cls; z_s]（1 + num_specials）——由分词器决定，
        # 不让人工配置和它失配（2026-11-16 定）
        cfg.mid_in_tokens = 1 + K
        self.cfg = cfg
        self.middle = TemporalMiddle(
            in_dim=code, dim=cfg.mid_dim, depth=cfg.mid_depth,
            heads=cfg.mid_heads, ratio=cfg.mid_ratio,
            tokens_per_frame=cfg.mid_in_tokens, frames=cfg.frames)
        self.compressor = StepCompressor(
            dim=cfg.mid_dim, rate_dim=cfg.rate_dim or code,
            n_pool=cfg.rate_tokens_per_step, heads=cfg.mid_heads,
            depth=1, ratio=2.0)
        self.readout = WholeClipDecoder(cfg, code_dim=code, num_specials=K)

    # ── 目标：全量 token（冻结编码器给）
    def target_latents(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        """(B,F,1,c)/(B,F,K,c) → (B,F,1+target_tokens,c)。"""
        return torch.cat([z_cls, z_s[:, :, :self.cfg.target_tokens]], dim=2)

    # ── 率：前 rate_tokens 个 special token（前缀截断）；中间态读全量
    def rate_latents(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        return z_s[:, :, :self.cfg.rate_tokens]

    # ── 编码（冻结）+ middle + compressor
    def encode_rate(self, x: Tensor) -> Dict[str, Tensor]:
        z_cls, z_s = self.tokenizer.encode(x)                # (B,F,1,c),(B,F,K,c)
        full = torch.cat([z_cls, z_s], dim=2)                # (B,F,1+K,c) 中间态输入
        mid = self.middle(full)                              # (B,F,T_f,mid_dim)
        g = self.cfg.group
        B, Fr, Tf, D = mid.shape
        S = Fr // g
        m = mid.view(B, S, g * Tf, D)                        # 每步 group 帧
        z = self.compressor(m)                               # (B,S,P,rate_dim)
        return dict(z=z, z_cls=z_cls, z_s=z_s, full=full, mid=mid)

    @staticmethod
    def step_weights(n: int, mode: str, device) -> Tensor:
        if mode == "lin":
            w = torch.arange(1, n + 1, dtype=torch.float32, device=device)
        elif mode == "last":
            w = torch.zeros(n, device=device)
            w[-1] = 1.0
            return w
        else:
            w = torch.ones(n, device=device)
        return w / w.sum()

    # ── 前向 + 损失
    def forward(self, x: Tensor, n_steps: Optional[int] = None,
                return_pix: bool = False) -> Dict[str, Tensor]:
        """x (B,F,3,H,W) ∈[0,1] → dict(loss, pred, l1, step_l1, z, target)."""
        enc = self.encode_rate(x)
        target = self.target_latents(enc["z_cls"], enc["z_s"])   # (B,F,1+T,c) 冻结
        out = self.readout(enc["z"], n_steps=n_steps, return_step_preds=True)
        preds = out["preds"]                                     # (B,S',F,T,c)
        B, S1 = preds.shape[:2]
        per_step = (preds - target.unsqueeze(1)).abs().mean(dim=(2, 3, 4))  # (B,S1)
        w = self.step_weights(S1, self.cfg.step_weights, per_step.device)
        loss = (per_step.mean(dim=0) * w).sum()
        res = dict(loss=loss, l1=per_step.detach().mean(),
                   step_l1=per_step.detach().mean(dim=0), pred=out["pred"],
                   z=enc["z"], target=target, latents=dict(
                       z_cls=enc["z_cls"], z_s=enc["z_s"]))
        if return_pix:
            res["pix"] = self.tokenizer.render(
                res["pred"][:, :, :1].contiguous(), res["pred"][:, :, 1:].contiguous())
        return res

    # ── 评测：latent L1 + 经冻结 R 的像素 PSNR
    @torch.no_grad()
    def evaluate(self, x: Tensor, n_steps: Optional[int] = None,
                 pixel_psnr: bool = True) -> Dict[str, float]:
        res = self.forward(x, n_steps=n_steps, return_pix=False)
        out = {"latent_l1": float(res["l1"])}
        if pixel_psnr:
            pc = res["pred"][:, :, :1].contiguous()
            ps = res["pred"][:, :, 1:].contiguous()
            pred_pix = self.tokenizer.render(pc, ps)
            out["psnr"] = float(psnr_from_l1(pred_pix, x).mean())
            # 冻结分词器自身末档（= 每帧冷启动上界，逐帧独立）
            up = self.tokenizer.render(res["latents"]["z_cls"],
                                       res["latents"]["z_s"])
            out["psnr_tok_full"] = float(psnr_from_l1(up, x).mean())
        return out

    def trainable_param_groups(self) -> Dict[str, int]:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return dict(middle=n(self.middle), compressor=n(self.compressor),
                    readout=n(self.readout), total=n(self) )


# ═══════════════════════════════════════════════════════════ 自检
if __name__ == "__main__":
    torch.manual_seed(0)
    # 小号分词器（不需要 ckpt）：32×32 输入 / patch 16 / N=4 / K=4 / zs_dim=8
    tok_cfg = dict(img_size=32, patch=16, num_specials=4, enc_dim=32, enc_depth=1,
                   enc_heads=2, enc_mlp_ratio=4.0, enc_attn="std",
                   dec_dim=32, dec_depth=1, dec_heads=2, dec_mlp_ratio=4.0,
                   pixel_hidden=32, zs_dim=8, dec_recurrent=False,
                   dec_steps="1", dec_step_plan="square")
    cfg = V10Config(tok_ckpt="", tok_cfg=tok_cfg, frames=8, group=4,
                    rate_tokens=2, rate_dim=8, target_tokens=4,
                    mid_in_tokens=5, mid_dim=16, mid_depth=2, mid_heads=2,
                    mid_ratio=2.0, dec_dim=16, dec_depth=2, dec_heads=2,
                    dec_ratio=2.0)
    m = VideoV10(cfg).eval()
    print("[cfg] n_steps", cfg.n_steps, "rate/step", cfg.rate_tokens_per_step,
          "值/帧", cfg.values_per_frame(), "值/clip", cfg.values_per_clip(),
          "位置/clip", cfg.positions_per_clip)
    print("[params]", {k: round(v / 1e6, 4) for k, v in m.trainable_param_groups().items()})
    assert all(not p.requires_grad for p in m.tokenizer.parameters()), "分词器应冻结"

    B = 2
    x = torch.rand(B, cfg.frames, 3, 32, 32)
    out = m(x, return_pix=True)
    print("[shape] pred", tuple(out["pred"].shape), "pix", tuple(out["pix"].shape),
          "loss", round(float(out["loss"]), 4))
    assert out["pred"].shape == (B, cfg.frames, 1 + cfg.target_tokens, 8)
    assert out["pix"].shape == (B, cfg.frames, 3, 32, 32)
    assert 0.0 <= float(out["pix"].min()) and float(out["pix"].max()) <= 1.0

    # 率守恒：z 的值数 = 30 步 × P × rate_dim
    z = out["z"]
    assert z.shape == (B, cfg.n_steps, cfg.rate_tokens_per_step, cfg.rate_dim), z.shape
    print("[rate] z", tuple(z.shape), "=", int(z[0].numel()), "值/clip =",
          cfg.values_per_clip())

    # 梯度：只有中间态/压缩/读出有梯度，分词器为 None
    out["loss"].backward()
    gtok = sum(p.grad.abs().sum().item() for p in m.tokenizer.parameters()
               if p.grad is not None)
    gt = sum(p.grad.abs().sum().item() for p in m.middle.parameters()
             if p.grad is not None)
    print("[grad] tokenizer", round(gtok, 6), "middle", round(gt, 6))
    assert gtok == 0.0, "冻结分词器不该有梯度"
    assert gt > 0.0, "middle 梯度为 0"

    # 因果：第 t 步的输出只能依赖 z[0..t]（扰动 z[k] 不影响更早的步）
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        z = torch.randn(1, cfg.n_steps, cfg.rate_tokens_per_step, cfg.rate_dim)
        a = m.readout(z, return_step_preds=True)["preds"]
        k = 1
        zb = z.clone()
        zb[:, k] += 0.5
        b = m.readout(zb, return_step_preds=True)["preds"]
    d = (a - b).abs().amax(dim=(0, 2, 3, 4))               # (S,)
    print("[causal] per-step max|Δ|", [round(float(v), 4) for v in d])
    assert all(float(d[t]) == 0.0 for t in range(k)), d
    assert float(d[k]) > 1e-6 and float(d[-1]) > 1e-6, d

    # γ 零初始化 ⇒ 第 0 步的读出与 Y 无关（起点=单步读出）
    assert m.readout.gate is not None and float(m.readout.gate.abs().max()) == 0.0
    print("[gate] γ 零初始化 ✓")

    # 小步过拟合：20 步内 loss 应下降
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=3e-3, weight_decay=0.0)
    xs = torch.rand(2, cfg.frames, 3, 32, 32)
    first = last = None
    for i in range(20):
        opt.zero_grad(set_to_none=True)
        loss = m(xs)["loss"]
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    print(f"[overfit] loss {first:.4f} → {last:.4f}")
    assert last < first, (first, last)

    # 评测接口（像素 PSNR 需要 R：这里走单步分词器，末档即单步）
    m.eval()
    ev = m.evaluate(xs[:1])
    print("[eval]", {k: round(v, 3) for k, v in ev.items()})
    print("ALL CHECKS PASSED")
