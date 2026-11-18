"""v10 = **1792 原生宽度 + 全片段循环读出**的视频模型（2026-11-16 设计定稿）。

宽度只有一种：1792 = 4 token × 448
----------------------------------
`chunk_tokens=4` 个 token 拼成一个 1792 向量，于是「宽度」和「位置数」由同一
个常量决定，任何位置都不用投影、不用填充：

    码率侧   16 token/帧  → 4 个 1792/帧
    预测侧   32 special + 1 z_cls → 8 个 1792 + 1 行（冻结 R 出像素必须要 cls）

旧版把 token 直接当位置、宽 1024 时，每帧 33×1024 = 33,792 个数只承载
33×448 = 14,784 个值 ⇒ **填充 2.29×**；本版 4×1792 = 7,168 承载 7,168 ⇒ 填充 1.00×。

实现说明（2026-11-16 重构）
---------------------------
注意力、FFN 与堆叠全部使用 PyTorch 原生 `nn.TransformerEncoderLayer` /
`nn.TransformerDecoderLayer`；本文件只保留模型编排（打包、循环读出、门控），
不再手写 QKV 投影和两层 MLP。

- middle：`nn.TransformerEncoder`。每帧 4 个位置（不是 33 个 token），帧级
  因果 mask。等宽 ⇒ **不建 in_proj**。
- readout：`nn.TransformerDecoder`。121 帧 × 9 行 = 1089 个位置，帧级因果。
  等宽 ⇒ **不建 kv_proj**；special 侧**也不建 head**（8 行 1792 直接 reshape
  成 32×448），只有 z_cls 那一行需要 `cls_head`（1792→448）。
- 没有 compressor 这一级：middle 的输出本身就是码率（120×4×1792 与
  30×16×1792 值数相同），再加一级池化只是把同一堆数重新混合一遍。

和 v8/v9 的关系
--------------
v8/v9 是「逐帧读出 + 帧间状态」；v10 反过来：**每一步都把整段 120 帧重出一次**，
每步多读一组 z（4 帧 = 0.5 s），读出逐步细化。因为 self-attn 是帧级因果的，
整条链严格 streaming：第 t 档 = 只读前 t+1 组 z，等价于论文里「读前 κ 个 token」。

数据流（数字口径：448² / patch 32 / N=196 / K=32 / zs_dim=448）
--------------------------------------------------------------
::

    x            (B, 120, 3, 448, 448)                  视频片段 15 s @ 8 fps
    E（冻结）    每帧独立前向 → z_cls (B,120,1,448), z_s (B,120,32,448)
    rate         前缀截断前 16 个 special → reshape (B,120,4,1792)（纯 reshape）
    middle       480 个位置，帧级因果 + 6 层 Transformer → (B,120,4,1792)
                 reshape → z (B,30,16,1792)（4 帧 × 4 行 = 每步 16 行）
    readout      Q（可学习，随机初始化）(1,1089,1792)；γ_0 = 0
                 for t in 0..29:
                     h    = LN(Q + γ_t·Y_{t-1}) + step_emb[t]
                     Y_t  = Decoder(h, z_t)     帧级因果 self-attn + cross-attn(16×1792)
                     pred = LN(Y_t) → 后 120 帧 × 9 行
                          8 行 reshape 成 32×448；第 9 行过 cls_head 成 z_cls
                 loss = mean_t L1(pred_t, 冻结编码器给的全量 33 token)
    R（冻结）    用 pred_30 的 (z_cls, 32 special) 还原像素，用于 PSNR

为什么「率 < 目标」
------------------
率 = 前 16 个 token（7,168 值/帧），目标 = 全部 33 个位置（14,784 值/帧）。
如果目标也只有 16 个 token，恒等映射就是最优解，跨帧混合没有任何存在必要。

可改的旋钮（V10Config）
-----------------------
rate_tokens     率：每帧保留多少 special token（默认 16，= 前缀截断）
rate_dim        单个 token 宽度（= 分词器 zs_dim，448）
chunk_tokens    一个向量拼几个 token（默认 4 ⇒ 宽度 1792，不允许改宽度）
target_tokens   目标 special 数；本版必须等于 num_specials（冻结 R 要全量）
group           z 步覆盖帧数（4 → 30 步，2 fps 潜在帧）
mid_dim/dec_dim 只接受 0（自动）或等于 width，防止"投影到别的宽度"
step_gate       是否用零初始化门控 γ（默认开：训练起点等价于单步读出）
step_weights    逐步损失权重（"mean" 平权 | "lin" 线性加权到末步）

自检：``python model_v10.py``（CPU 可跑，不需要 ckpt）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn

__all__ = ["V10Config", "FrozenTokenizer", "TemporalMiddle",
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
    rate_dim: int = 448        # 单个 token 宽度（= 分词器 zs_dim）
    target_tokens: int = 32    # 目标 special token 数（全量）
    chunk_tokens: int = 4      # 一个可训练向量 = 几个 token 拼接（4×448 = 1792，不填充）
    # ── 中间态（帧内 block 因果）
    mid_dim: int = 0           # 0 = 自动取 width（1792），不允许填成别的宽度
    mid_depth: int = 6
    mid_heads: int = 16
    mid_ratio: float = 4.0
    # ── 读出（整片段循环）
    dec_dim: int = 0           # 0 = 自动取 width（1792）
    dec_depth: int = 4
    dec_heads: int = 16
    dec_ratio: float = 2.0
    dec_drop: float = 0.0
    step_gate: bool = True     # γ 零初始化门控
    step_weights: str = "mean"
    mem_mode: str = "step"     # "step" = 每步只看 z_t（设计原版）; "prefix" = 看已读前缀
    loss_read: bool = False    # True = 第 t 步只在「已读帧」上算 loss（去均值平凡解）
    copy_skip: bool = True     # 已读帧的 rate 行直抄到输出（残差捷径；prefix/step 都支持）


    @property
    def n_steps(self) -> int:
        assert self.frames % self.group == 0, (self.frames, self.group)
        return self.frames // self.group

    @property
    def rate_tokens_per_step(self) -> int:
        """每个 z 步的 rate token 数（= rate_tokens × group）。"""
        return self.rate_tokens * self.group

    # ── 1792 口径：宽度由 chunk_tokens 决定，位置数由它除出来
    @property
    def width(self) -> int:
        """一个可训练向量的宽度 = chunk_tokens × rate_dim（默认 4×448 = 1792）。

        这是本模型的唯一宽度：token 和它等宽时不需要任何投影，多出来的
        维度全是填充 —— 那正是要避免的冗余。
        """
        return self.chunk_tokens * self.rate_dim

    @property
    def rows_rate(self) -> int:
        """码率侧每帧的向量数（16 token / 4 = 4）。"""
        assert self.rate_tokens % self.chunk_tokens == 0, (
            f"rate_tokens={self.rate_tokens} 不能被 chunk_tokens={self.chunk_tokens} 整除")
        return self.rate_tokens // self.chunk_tokens

    @property
    def chunks_per_step(self) -> int:
        """每个 z 步的向量数（默认 64 token / 4 = 16）。"""
        assert self.rate_tokens_per_step % self.chunk_tokens == 0
        return self.rate_tokens_per_step // self.chunk_tokens

    @property
    def rows_pred(self) -> int:
        """预测侧每帧的向量数 = target_tokens/chunk_tokens 个 chunk + 1 行 z_cls。

        32/4 = 8 个 chunk 装 32 个 special，再加 1 行独立的 z_cls（冻结 R 出
        像素必须要它），所以是 9 行、不是 8 行。
        """
        assert self.target_tokens % self.chunk_tokens == 0, (
            f"target_tokens={self.target_tokens} 不能被 chunk_tokens={self.chunk_tokens} 整除")
        return self.target_tokens // self.chunk_tokens + 1

    @property
    def positions_per_clip(self) -> int:
        """读出位置数 = (frames + 1 个起始槽) × rows_pred。"""
        return (self.frames + 1) * self.rows_pred

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


# ═══════════════════════════════════════════════════════════ 中间态
class TemporalMiddle(nn.Module):
    """帧内 block 因果的 Transformer：一帧之内的向量互看，跨帧只看历史帧。

    打包视图就是设计口径本身：每帧 `chunk_tokens` 个 token 拼成一个 1792
    的向量，所以每帧 = 4 个位置，`tokens_per_frame` 直接等于 4，不需要在
    33 token 的视图里模拟 block 边界。

    当 `in_dim == dim`（打包后天然等宽）时**不建投影层**：硬加一个
    `Linear(1792, 1792)` 只是把同一个向量重新线性混合一遍，属于纯冗余。
    """

    def __init__(self, in_dim: int, dim: int, depth: int, heads: int,
                 ratio: float, tokens_per_frame: int, frames: int,
                 drop: float = 0.0):
        super().__init__()
        self.tpf = int(tokens_per_frame)
        self.frames = int(frames)
        L = self.tpf * self.frames
        assert in_dim == dim, (in_dim, dim)
        self.in_proj = None
        self.norm_in = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(depth), norm=None,
            enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(dim)
        fi = torch.arange(L) // self.tpf                    # token → 帧号
        allowed = fi[None, :] <= fi[:, None]                # True = 可看
        # nn.MultiheadAttention 的浮点 mask 是加性的：0 保留，-inf 屏蔽。
        mask = torch.zeros(L, L, dtype=torch.float32).masked_fill(
            ~allowed, float("-inf"))
        self.register_buffer("_attn_mask", mask, persistent=False)

    def forward(self, z_in: Tensor) -> Tensor:
        """z_in (B, F, T_f, d) → (B, F, T_f, d)（同形；打包后无需投影）。"""
        B, Fr, Tf, _ = z_in.shape
        assert Fr == self.frames and Tf == self.tpf, (z_in.shape, self.frames, self.tpf)
        x = self.norm_in(z_in.reshape(B, Fr * Tf, -1))
        x = self.encoder(x, mask=self._attn_mask)
        return self.norm_out(x).view(B, Fr, Tf, -1)


# ═══════════════════════════════════════════════════════════ 整片段循环读出
class WholeClipDecoder(nn.Module):
    """(frames+1) 行 × rows_pred 列的 1792 向量 + 30 步循环：每步读一组 z、重出整段。

    每步结构：
        h      = LN(Q + γ_t·Y_{t-1}) + step_emb[t]   γ 零初始化 ⇒ 起点=单步
        Y      = TransformerDecoder(h, z_t)          帧级因果 self-attn + cross-attn
        pred_t = LN(Y) 拆成 8 个 chunk（32 个 special）+ 1 行 z_cls

    位置数 = 121 × 9 = 1089：120 帧 + 1 个随机初始化起始槽，每帧 9 行 1792
    （8 行装 32 个 special，1 行装 z_cls —— 冻结 R 出像素必须要它）。
    self-attn 跨帧因果（帧 j 只看 0..j），所以整条链是 streaming 的，不需要
    任何帧内 mask。

    零冗余的三处：
      * 不建 kv_proj —— z 本身就是 1792，再投影一遍只是把同一向量重新线性
        混合一次；
      * 不建 special 的 head —— 8 行 1792 直接 reshape 成 32×448，只有
        z_cls 那一行需要 `cls_head`（1792→448）；
      * 不复制 memory —— 整段是一条 batch，cross-attn 直接对这一步的 16 个
        key，不需要 `repeat_interleave` 出 120 份副本。
    """

    def __init__(self, cfg: V10Config, code_dim: int, num_specials: int):
        super().__init__()
        self.cfg = cfg
        self.code_dim = int(code_dim)
        self.K = int(num_specials)
        self.F = int(cfg.frames)
        self.S = int(cfg.n_steps)
        self.P = int(cfg.chunks_per_step)
        self.R = int(cfg.rows_pred)                        # 9
        self.d = int(cfg.dec_dim)
        assert self.d == int(cfg.width), (self.d, cfg.width)
        L = (self.F + 1) * self.R                          # 1089
        # 查询与状态（整片段一条序列）
        self.Q = nn.Parameter(torch.randn(1, L, self.d) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(1, self.S, 1, self.d) * 0.02)
        self.mem_mode = str(getattr(cfg, "mem_mode", "step")).lower()
        # prefix 模式下 kv_pos 用不到：不建它，否则 DDP 会因为「有参数不参与
        # loss」在第二次迭代报 "Expected to have finished reduction"。
        self.kv_pos = (nn.Parameter(torch.randn(1, self.P, self.d) * 0.02)
                       if self.mem_mode != "prefix" else None)
        if self.mem_mode == "prefix":
            # 前缀读出（2026-11-18 诊断后新增）：第 t 步 cross-attn 的 key 是
            # **已读过的前 (t+1) 组 z**（每组 group 帧 × rows_rate 行），并给
            # key 与 query 都加「帧位置 + 行号」嵌入 ⇒ "把读过的帧抄到对应输出
            # 位置" 变成一步可学的映射，不再依赖 30 步递推去搬运整段信息。
            rr = int(self.P // int(cfg.group))                # 每帧 rows_rate 行
            self.frame_emb = nn.Parameter(torch.randn(1, self.F, 1, self.d) * 0.02)
            self.row_emb = nn.Parameter(torch.randn(1, 1, rr, self.d) * 0.02)
            self.start_emb = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)
            qf = torch.arange(L, dtype=torch.long) // self.R - 1     # -1 = 起始槽
            self.register_buffer("_qframe", qf, persistent=False)
        # 「直抄残差」：输出行 0..rr-1 += 已读帧的 rate 行。prefix 模式默认开
        # （没有它时实测训练会塌到「与输入无关的模板」：middle 输出跨片段 std
        # 从 0.29 崩到 0.015，pred 完全不吃 memory，因为均值解是局部极小）。
        # 2026-11-18 起 **step 模式也支持**（CLI --copy_skip），用于
        # 「纯循环 vs 循环+直抄」的单片段过拟合对照：两者信息量相同，唯一差别
        # 是未读帧由 30 步递推 Y 还是由前缀 cross-attn 预测。
        self.copy_skip = bool(getattr(cfg, "copy_skip", True))
        self.gate = (nn.Parameter(torch.zeros(self.S)) if cfg.step_gate else None)
        self.n_in = nn.LayerNorm(self.d)
        layer = nn.TransformerDecoderLayer(
            d_model=self.d, nhead=int(cfg.dec_heads),
            dim_feedforward=int(self.d * cfg.dec_ratio), dropout=cfg.dec_drop,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=int(cfg.dec_depth), norm=None)
        self.norm_out = nn.LayerNorm(self.d)
        self.cls_head = nn.Linear(self.d, self.code_dim)    # 只有 z_cls 需要投影
        fi = torch.arange(L) // self.R                      # 位置 → 帧号（0=起始槽）
        allowed = fi[None, :] <= fi[:, None]                # True = 可看
        mask = torch.zeros(L, L, dtype=torch.float32).masked_fill(
            ~allowed, float("-inf"))
        self.register_buffer("_mask", mask, persistent=False)

    def _split(self, Y: Tensor, resid: Optional[Tensor] = None
               ) -> Tuple[Tensor, Tensor]:
        """(B,L,d) → specials (B,F,K,c) / z_cls (B,F,1,c)。

        `resid` (B,F,R,d) 是 prefix 模式的**直抄残差**：把已经读到的 rate 行
        原样加到输出行上（第 r 行 = 该帧第 r 个 rate 行），写第一个 special
        不需要任何学习；未读帧的残差为 0。
        """
        B = Y.shape[0]
        Y = Y.view(B, self.F + 1, self.R, self.d)[:, 1:]    # 丢掉随机起始槽
        if resid is not None:
            Y = Y + resid
        spec = Y[:, :, :self.R - 1].reshape(B, self.F, self.K, self.code_dim)
        cls = self.cls_head(Y[:, :, self.R - 1]).unsqueeze(2)
        return spec, cls

    def forward(self, z: Tensor, n_steps: Optional[int] = None,
                return_step_preds: bool = False) -> Dict[str, Tensor]:
        """z (B,S,P,d) → dict(pred (B,F,1+K,c), feats, preds 可选 (B,S,F,1+K,c))。"""
        B, S, P, _ = z.shape
        assert P == self.P, (P, self.P)
        n = self.S if n_steps is None else int(n_steps)
        assert 1 <= n <= S, (n, S)
        # 加性 mask 必须与注意力权重同 dtype：bf16 下 muDNN/MUSA 的 SDPA 拒绝
        # "BF16 + FLOAT" 的加法（实测报 NOT_SUPPORTED in RunMath）。
        mask = self._mask.to(dtype=z.dtype)
        Q = self.Q.expand(B, -1, -1)                         # 随机初始化、可学习
        Y = torch.zeros_like(Q)                              # 状态初值 0（第 0 步只看 Q）
        prefix = (self.mem_mode == "prefix")
        grp = int(self.cfg.group)
        rr = int(self.P // grp)                          # 每帧 rows_rate 行
        if prefix or self.copy_skip:
            assert grp * rr == self.P and self.F * rr == self.S * self.P, (grp, rr)
            zr = z.reshape(B, self.F, rr, self.d)        # (B,F,rows_rate,d)
        if prefix:
            fem = self.frame_emb.reshape(self.F, self.d)
            qf = self._qframe
            qe = fem[qf.clamp(min=0)]                        # (L,d)
            qe = torch.where((qf >= 0).unsqueeze(-1), qe,
                             self.start_emb.reshape(1, self.d).expand_as(qe))
        else:
            kv_all = z + self.kv_pos                         # (B,S,P,d)，等宽投影
        specs: List[Tensor] = []
        clss: List[Tensor] = []
        for t in range(n):
            w = 1.0 if self.gate is None else self.gate[t]   # γ 零初始化 ⇒ 起点=单步
            h = self.n_in(Q + w * Y) + self.step_emb[:, t]
            nf = min(self.F, (t + 1) * grp)                  # 已读帧数
            if prefix:
                keys = (zr[:, :nf]
                        + self.frame_emb[:, :nf].reshape(1, nf, 1, self.d)
                        + self.row_emb)                       # (B,nf,rr,d)
                keys = keys.reshape(B, nf * rr, self.d)
                Y = self.decoder(tgt=h + qe, memory=keys, tgt_mask=mask)
            else:
                Y = self.decoder(tgt=h, memory=kv_all[:, t], tgt_mask=mask)
            # 直抄残差：已读帧的 rate 行原样回填到输出行（读到的帧立刻精确
            # 可出，不需要模型"学会抄"；未读帧为 0，留给模型预测）。
            # 两种 mem_mode 共用一套填充逻辑，对照实验里这是共同项。
            resid = None
            if self.copy_skip:
                resid = z.new_zeros(B, self.F, self.R, self.d)
                resid[:, :nf, :rr] = zr[:, :nf]
            spec, cls = self._split(self.norm_out(Y), resid)
            specs.append(spec)
            clss.append(cls)
        out = dict(pred=torch.cat([clss[-1], specs[-1]], dim=2),
                   feats=Y.view(B, self.F + 1, self.R, self.d)[:, 1:],
                   n_steps=n)
        if return_step_preds:
            out["preds"] = torch.cat([torch.stack(clss, 1),
                                      torch.stack(specs, 1)], dim=3)
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
        cfg.rate_dim = code
        self.code_dim = code
        K = self.tokenizer.num_specials
        assert cfg.rate_tokens <= K, (cfg.rate_tokens, K)
        # 冻结 R 必须拿到全部 K 个 special 才能出像素，目标因此只能是全量
        assert cfg.target_tokens == K, (
            f"target_tokens={cfg.target_tokens} 必须等于分词器 num_specials={K}；"
            "少预测会让冻结 R 的 render 直接 assert 失败")
        # ── 唯一宽度：chunk_tokens 个 token 拼成一个向量（4×448 = 1792）。
        # mid_dim / dec_dim 只接受 0（自动）或与它相等，杜绝"再投影到别的宽度"。
        cfg.mid_dim = int(cfg.mid_dim or cfg.width)
        cfg.dec_dim = int(cfg.dec_dim or cfg.width)
        assert cfg.mid_dim == cfg.dec_dim == cfg.width, (
            f"mid_dim={cfg.mid_dim} dec_dim={cfg.dec_dim} 必须都等于 "
            f"chunk_tokens×rate_dim={cfg.width}；换宽度只会引入填充冗余")
        self.cfg = cfg
        self.middle = TemporalMiddle(
            in_dim=cfg.width, dim=cfg.width, depth=cfg.mid_depth,
            heads=cfg.mid_heads, ratio=cfg.mid_ratio,
            tokens_per_frame=cfg.rows_rate, frames=cfg.frames)
        self.readout = WholeClipDecoder(cfg, code_dim=code, num_specials=K)

    # ── 目标：全量 special + z_cls（冻结编码器给）
    def target_latents(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        """(B,F,1,c)/(B,F,K,c) → (B,F,1+target_tokens,c)。"""
        return torch.cat([z_cls, z_s[:, :, :self.cfg.target_tokens]], dim=2)

    # ── 率：前 rate_tokens 个 special token（前缀截断），打包成 1792 向量
    def rate_latents(self, z_s: Tensor) -> Tensor:
        """(B,F,K,c) → (B,F,rows_rate,width)：纯 reshape，不引入任何参数。"""
        B, Fr = z_s.shape[:2]
        r = z_s[:, :, :self.cfg.rate_tokens]
        return r.reshape(B, Fr, self.cfg.rows_rate, self.cfg.width)

    # ── 编码（冻结）+ middle → 每步的 z（middle 输出本身就是码率，无压缩级）
    def encode_rate(self, x: Tensor) -> Dict[str, Tensor]:
        z_cls, z_s = self.tokenizer.encode(x)                # (B,F,1,c),(B,F,K,c)
        return self.encode_rate_latents(z_cls, z_s)

    def encode_rate_latents(self, z_cls: Tensor, z_s: Tensor) -> Dict[str, Tensor]:
        """已缓存的 (z_cls, z_s) → 每步的 z（跳过冻结编码器，训练时省 30%+）。"""
        packed = self.rate_latents(z_s)                      # (B,F,Rr,width)
        mid = self.middle(packed)                            # (B,F,Rr,width)
        B, Fr, R, d = mid.shape
        S = self.cfg.n_steps
        assert Fr // S * R == self.cfg.chunks_per_step, (Fr, S, R)
        # group 帧 × Rr 行 → 每步 P 行；纯 reshape（row-major），值数与 mid 相同
        z = mid.reshape(B, S, self.cfg.chunks_per_step, d)   # (B,30,P,width)
        return dict(z=z, z_cls=z_cls, z_s=z_s, packed=packed, mid=mid)

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
    def forward(self, x: Optional[Tensor] = None, n_steps: Optional[int] = None,
                return_pix: bool = False,
                z_cls: Optional[Tensor] = None,
                z_s: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """x (B,F,3,H,W) ∈[0,1]，或已缓存的 (z_cls (B,F,1,c), z_s (B,F,K,c))。

        → dict(loss, pred, l1, step_l1, z, target)；像素 PSNR 只在给了 x 时算。
        """
        if x is not None:
            enc = self.encode_rate(x)
        else:
            assert z_cls is not None and z_s is not None, "须给 x，或 (z_cls, z_s)"
            enc = self.encode_rate_latents(z_cls, z_s)
        target = self.target_latents(enc["z_cls"], enc["z_s"])   # (B,F,1+T,c) 冻结
        out = self.readout(enc["z"], n_steps=n_steps, return_step_preds=True)
        preds = out["preds"]                                     # (B,S',F,T,c)
        B, S1 = preds.shape[:2]
        err = (preds - target.unsqueeze(1)).abs()                # (B,S',F,T,c)
        if self.cfg.loss_read:
            # 只惩罚「第 t 步已经读到的帧」（前缀 t+1 组）。均值平凡解在已读帧
            # 上不再是解：读到的帧必须被抄出来，梯度不再被"不可知的未来帧"淹没。
            F = err.shape[2]
            grp = int(self.cfg.group)
            nf = (torch.arange(S1, device=err.device) + 1) * grp  # (S',) 已读帧数
            frame = torch.arange(F, device=err.device)
            m = (frame[None, :] < nf[:, None]).to(err.dtype)      # (S',F)
            per_frame = err.mean(dim=(3, 4))                      # (B,S',F)
            per_step = (per_frame * m).sum(-1) / m.sum(-1).clamp_min(1)
        else:
            per_step = err.mean(dim=(2, 3, 4))                   # (B,S1)
        w = self.step_weights(S1, self.cfg.step_weights, per_step.device)
        loss = (per_step.mean(dim=0) * w).sum()
        res = dict(loss=loss, l1=per_step.detach().mean(),
                   step_l1=per_step.detach().mean(dim=0), pred=out["pred"],
                   z=enc["z"], target=target, latents=dict(
                       z_cls=enc["z_cls"], z_s=enc["z_s"]))
        # HF Trainer 的 eval 走 forward；evaluate() 里设置 eval_pixel=True，
        # 这样 compute_metrics 能拿到 PSNR，而不必再写一套评估循环。
        pixel_eval = bool(return_pix or (
            not self.training and getattr(self, "eval_pixel", False)))
        if pixel_eval and x is not None:
            pred_pix = self.tokenizer.render(
                res["pred"][:, :, :1].contiguous(),
                res["pred"][:, :, 1:].contiguous())
            res["pix"] = pred_pix
            res["psnr"] = psnr_from_l1(pred_pix, x).mean()
            up = self.tokenizer.render(res["latents"]["z_cls"],
                                       res["latents"]["z_s"])
            res["psnr_tok_full"] = psnr_from_l1(up, x).mean()
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
        return dict(middle=n(self.middle), readout=n(self.readout), total=n(self))


# ═══════════════════════════════════════════════════════════ 自检
if __name__ == "__main__":
    torch.manual_seed(0)
    # 小号分词器（不需要 ckpt）：32×32 输入 / patch 16 / N=4 / K=4 / zs_dim=8
    tok_cfg = dict(img_size=32, patch=16, num_specials=4, enc_dim=32, enc_depth=1,
                   enc_heads=2, enc_mlp_ratio=4.0, enc_attn="std",
                   dec_dim=32, dec_depth=1, dec_heads=2, dec_mlp_ratio=4.0,
                   pixel_hidden=32, zs_dim=8, dec_recurrent=False,
                   dec_steps="1", dec_step_plan="square")
    # chunk_tokens=4 ⇒ 宽度 = 4×8 = 32（正式配置是 4×448 = 1792）
    cfg = V10Config(tok_ckpt="", tok_cfg=tok_cfg, frames=8, group=4,
                    rate_tokens=4, rate_dim=8, target_tokens=4, chunk_tokens=4,
                    mid_depth=2, mid_heads=2, mid_ratio=2.0,
                    dec_depth=2, dec_heads=2, dec_ratio=2.0)
    m = VideoV10(cfg).eval()
    print(f"[cfg] width={cfg.width} rows_rate={cfg.rows_rate} "
          f"rows_pred={cfg.rows_pred} chunks/step={cfg.chunks_per_step}")
    print("[cfg] n_steps", cfg.n_steps, "rate/step", cfg.rate_tokens_per_step,
          "值/帧", cfg.values_per_frame(), "值/clip", cfg.values_per_clip(),
          "位置/clip", cfg.positions_per_clip)
    print("[params]", {k: round(v / 1e6, 4) for k, v in m.trainable_param_groups().items()})
    assert all(not p.requires_grad for p in m.tokenizer.parameters()), "分词器应冻结"
    # 零填充：码率侧每帧的数值量 = 向量数 × 宽度，一点不多
    assert cfg.rows_rate * cfg.width == cfg.values_per_frame()

    B = 2
    x = torch.rand(B, cfg.frames, 3, 32, 32)
    out = m(x, return_pix=True)
    print("[shape] pred", tuple(out["pred"].shape), "pix", tuple(out["pix"].shape),
          "loss", round(float(out["loss"]), 4))
    assert out["pred"].shape == (B, cfg.frames, 1 + cfg.target_tokens, 8)
    assert out["pix"].shape == (B, cfg.frames, 3, 32, 32)
    assert 0.0 <= float(out["pix"].min()) and float(out["pix"].max()) <= 1.0

    # 率守恒：z 的值数 = 步数 × chunks/step × 宽度 = frames × rows_rate × 宽度
    z = out["z"]
    assert z.shape == (B, cfg.n_steps, cfg.chunks_per_step, cfg.width), z.shape
    print("[rate] z", tuple(z.shape), "=", int(z[0].numel()), "值/clip =",
          cfg.values_per_clip())
    assert int(z[0].numel()) == cfg.values_per_clip()

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
        z = torch.randn(1, cfg.n_steps, cfg.chunks_per_step, cfg.width)
        a = m.readout(z, return_step_preds=True)["preds"]
        assert a.shape == (1, cfg.n_steps, cfg.frames,
                           1 + cfg.target_tokens, 8), a.shape
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
