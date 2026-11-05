"""
标准 softmax 注意力编码器 — 从 AssocVTC 主仓库 vendored
======================================================

**来源**：AssocVTC `main` 分支 `model_v2_linattn.py`（2026-11-05, commit `1fc4920`）
的 `StandardAttention` / `StandardAttentionBlock` / `StandardAttentionEncoder` 三个类，
逐字抽取，未改语义。阶段2 ckpt `/root/autodl-tmp/exp448/B/ckpt_best.pt` 就是
`StandardAttentionEncoder(dim=1744, depth=13, heads=16, patch=32, img_size=448, K=32)`。

**为什么复制而不是 import**：本项目是独立最小项目（复制即可在 remote 跑，不依赖
AssocVTC 仓库其余 1900+ 文件）。上游 `model_v2_linattn.py` 里还有线性注意力、decoder、
像素头等与本实验无关的代码，未带过来。

**同步契约**：两处必须和上游一致，否则阶段2 权重 `strict=True` 加载会失败——
① `StandardAttentionEncoder` 构造参数名；② 子模块/参数名（`patch_embed` / `cls_token` /
`special_tokens` / `pos_embed` / `pos_drop` / `blocks.*` / `norm.*`）。
上游改这三点时，请重新从 `model_v2_linattn.py` 抽取本文件。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# ═══ 标准 softmax 注意力编码器（阶段2 用的就是这一支）═══
#   上游原文注释保留: 本类当初是作为「线性注意力 vs 普通注意力」消融臂写的,
#   block 与 LinearAttentionBlock **结构逐项对应**（pre-norm: 注意力 → 残差 →
#   FFN → 残差, GELU, mlp_ratio=4）, 唯一自变量 = 注意力形式;
#   dim=448/depth=13 时参数天然对齐（31.848M vs 31.909M）。阶段2 B 线最终选 std。

class StandardAttention(nn.Module):
    """标准多头缩放点积注意力（softmax, 双向, 无因果掩码）。

    与 `LinearAttention` 的接口差异: 不需要 `h/w/n_prefix`（没有 DWConv 局部
    混合, 因而也不存在"只卷 patch 段"的问题）。

    为什么**不**需要 `LinearAttention` 那套数值稳定性修复:
        softmax 天然把每行权重归一化到 [0,1] 且和为 1 ⇒ 输出是 value 的凸组合,
        量级被 value 钉住, 不会像 elu+1 核形式那样随 qkv 权重无界增长。
        因此这里用 `F.scaled_dot_product_attention`（PyTorch 自选 flash /
        mem-efficient 后端, 自带 1/√d 缩放）, 直接吃 bf16 autocast。
        `qk_norm` 默认关闭（普通 transformer 的原貌）; 置 True 可做 QK-Norm 变体。
    """

    def __init__(self, dim: int, heads: int = 8, qk_norm: bool = False):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 须被 heads={heads} 整除"
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        H, d = self.heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, H, d).transpose(1, 2)            # (B,H,N,d)
        k = k.reshape(B, N, H, d).transpose(1, 2)
        v = v.reshape(B, N, H, d).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)        # softmax(QKᵀ/√d)V
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class StandardAttentionBlock(nn.Module):
    """pre-norm: x + MHSA(LN(x)) → x + FFN(LN(x))（与 LinearAttentionBlock 同构）。

    forward 保留 `(x, h, w, n_prefix)` 签名（与线性块一致, 便于编码器统一循环）,
    但普通注意力不使用这些参数（无局部混合）。
    """

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0,
                 qk_norm: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = StandardAttention(dim, heads, qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x: Tensor, h: int = 0, w: int = 0,
                n_prefix: int = 0) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StandardAttentionEncoder(nn.Module):
    """`LinearAttentionEncoder` 的**普通 transformer** 对应体（消融臂）。

    token 化逐位相同: `[cls(1); specials(K); patches(N)]` + learnable pos,
    patch_embed 同 Conv2d(3→dim, k=patch, s=patch), 末尾 LayerNorm, 返回
    `(z_cls, z_s)`。**只把 13 层线性注意力块换成 13 层标准 softmax 注意力块。**
    """

    def __init__(self, dim: int = 448, depth: int = 13, heads: int = 8,
                 mlp_ratio: float = 4.0, patch: int = 16, img_size: int = 224,
                 num_specials: int = 64, qk_norm: bool = False,
                 drop_rate: float = 0.0, special_init_std: float = 0.02):
        super().__init__()
        assert img_size % patch == 0, f"img_size={img_size} 须被 patch={patch} 整除"
        self.patch = patch
        self.img_size = img_size
        self.num_patches = (img_size // patch) ** 2
        self.num_specials = num_specials
        self.dim = dim

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.special_tokens = nn.Parameter(
            torch.randn(1, num_specials, dim) * special_init_std)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.pos_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.blocks = nn.ModuleList([
            StandardAttentionBlock(dim, heads, mlp_ratio, qk_norm)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.special_tokens, std=special_init_std)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos(self, n: int) -> Tensor:
        """位置编码; n != num_patches 时线性插值（便于小分辨率冒烟）。"""
        if n == self.num_patches:
            return self.pos_embed
        pos = F.interpolate(self.pos_embed.transpose(1, 2), size=(n,),
                            mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, pixel_values: Tensor,
                return_seq: bool = False) -> Tuple[Tensor, ...]:
        x = pixel_values
        B, C, H, W = x.shape
        assert H % self.patch == 0 and W % self.patch == 0, \
            f"{H}x{W} 不是 patch={self.patch} 的整数倍"
        h, w = H // self.patch, W // self.patch
        n = h * w
        p = self.patch_embed(x).flatten(2).transpose(1, 2)          # (B,n,dim)
        p = self.pos_drop(p + self._pos(n))
        seq = torch.cat([self.cls_token.expand(B, -1, -1),
                         self.special_tokens.expand(B, -1, -1), p], dim=1)
        n_prefix = 1 + self.num_specials
        for blk in self.blocks:
            seq = blk(seq, h, w, n_prefix)
        seq = self.norm(seq)
        if return_seq:
            # E-B3: 额外回**末层 LayerNorm 后的 patch 段**（给 GrowthBlock 读; 默认关）
            return seq[:, :1], seq[:, 1:1 + self.num_specials], seq[:, 1 + self.num_specials:]
        return seq[:, :1], seq[:, 1:1 + self.num_specials]
