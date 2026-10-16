"""
SR-Diffusion Phase 1 v2 — **实验版 (linattn)**: 无预训练编码器
================================================================

本文件是 `model_v2.py` 的**实验性分支**, 与主线并存、互不影响
（主线 `model_v2.py` 逐位不变; 本文件不 import 它, 可独立自检）。

用户口径 (2026-10-15)
--------------------
> "把 model_v2 增加一个实验性的版本, 这个版本**不再使用预训练编码器**, 而是
>  自定义一个小的**线性注意力机制**的编码器, 参数量 **30 几 M** 就行, 解码器
>  是**同等体量的普通注意力机制**的 30 几 M 的 transformer。输入 224×224,
>  输出也是 224×224。"

=> 与主线 model_v2 的**唯一口径差异 = 编码器来源**:

| | 主线 `model_v2.py` | 本实验版 `model_v2_linattn.py` |
|---|---|---|
| 编码器 | DINOv2-large (304M, **预训练冻结/微调**) | `LinearAttentionEncoder` (**从零训练**, ~31M) |
| 编码器注意力 | softmax 全双向 (DINO 24 层) | **线性注意力** (elu+1 特征映射, O(N) 核形式) |
| 解码器 | `OutputQueryDecoder` (2 层, 循环读出, ~2M) | `StandardAttentionDecoder` (**普通 softmax 注意力**, 7 层, ~31M) |
| 分辨率 | 224×(126/252/…) | **224×224 → 224×224** (用户指定) |
| patch | 14 (DINO 原生) | **16** (从零, 无 DINO 约束) |

**保留的主线语义** (可与主线直接对照):
    · register 式 token 压缩: `[cls; specials(K); patches(N)]` 同一序列里做
      双向注意力, specials 与 patches 直接交互 ⇒ z_s 是"从像素里学出来的压缩"。
      K (`num_specials`) 与 N 解耦, K<N 即真压缩 (主线 doc 的核心杠杆)。
    · 输出查询式解码 (Perceiver IO / BLIP-2 口径): 查询基行 k ↔ patch k,
      行数恒为 N, 解码器对全部 patch 一次性给出直接预测。
    · 损失 = 像素 L1 (**直接预测**), 监控量 recon = L1(F_hat, target)。
    · 输出 dict 键与 `SRPhase1V2.forward` 完全一致
      (`loss/recon/F_hat/Y_pix/target_pix`) ⇒ 训练/推理脚本口径可复用。

**刻意偏离主线的一点 (诚实标注)**: 主线的循环读出 (BPTT carry) 在本实验版里
    改为**单步并行解码** (Y_pix 的 |T| 维 = 1)。理由: 本实验的自变量是
    "编码器是否预训练 + 注意力类型/体量", 循环读出是另一个自变量; 两者同时改
    会让结果无法归因。主线 `model_v2.py` 的循环路径不受影响, 需要时可按
    同一接口把 `StandardAttentionDecoder` 换成循环版。

架构 (默认配置, 见 `LinAttnConfig`)
----------------------------------
    输入 (B,3,224,224) --ImageNet 归一化-->
      patch_embed: Conv2d(3→448, k=16, s=16)  →  (B,196,448) + learnable pos
      序列 = [cls(1); specials(K); patches(196)] = 1+K+196 个 token
      13 × LinearAttentionBlock (LN → LinearAttn → 残差 → LN → FFN → 残差)
      末尾 LayerNorm → z_cls = seq[:,0], z_s = seq[:,1:1+K]
      ---------------------------------------------------------------
      解码器 memory = [z_cls; z_s] (K+1, 448) → Linear 448→512 → LN
      查询 = query_base(196,512) + query_pos(196,512)
      7 × nn.TransformerDecoderLayer (self-attn + cross-attn(memory) + FFN)
      末尾 LayerNorm → PixelHead(512→1024→768) → 每 patch 16×16×3
      重组 → 输出 (B,3,224,224)

参数量 (实测, 见 `python model_v2_linattn.py`):
    编码器 ≈ 31.8M (dim=448, depth=13) / 解码器 ≈ 31.1M (dim=512, depth=7)
    —— 两者"同等体量"; `count_params` 会打印精确值。

线性注意力 (Katharopoulos et al. 2020, "Transformers are RNNs")
-------------------------------------------------------------
    softmax(QK^T)V 的核形式: 把 softmax 换成特征映射 φ(x)=elu(x)+1, 则
        Attn(Q,K,V) = (φ(Q) (φ(K)^T V)) / (φ(Q) Σ_j φ(K_j))
    ⇒ 先算 (d×d) 的 KV 统计量, 复杂度 O(N d²) 而非 O(N² d); N 线性增长。
    本项目 N=196 不大, 用线性注意力的**动机不是省算力**, 而是:
      (a) 消融"预训练 + softmax"这一组合, 看从零方案的差距是否来自注意力形式;
      (b) 核形式无 softmax 归一化的强竞争, register token 的汇聚更"平均",
          与 token 压缩 (K≪N) 的聚合语义天然契合。
    块内另加**逐通道 3×3 卷积 (DWConv, 零初始化 ⇒ 初始为恒等)** 作为局部混合:
    纯线性注意力缺局部性, 这是线性注意力 ViT (MLLA / EfficientViT) 的通行做法,
    可由 `dwconv=False` 关掉做消融。

⚠️ **数值稳定性（2026-10-15 实测, 非可选）**: elu+1 核形式在 bf16 下会**间歇性
    梯度爆炸**（实测 grad_norm 0.6→26→338→1.2e6→3e8, 一步把 val PSNR 从
    19.6dB 打到 11.8dB 且不可恢复）。三处修复（详见 `LinearAttention` docstring）:
      ① 每 head 的 **QK LayerNorm**（`enc_qk_norm=True`, 默认开）;
      ② 线性注意力**聚合段强制 fp32**（`torch.autocast(enabled=False)`）;
      ③ train 侧**梯度爆炸跳步护栏**（`--skip_grad`, 默认 50）。
    `enc_qk_norm=False` 可复现旧的不稳定行为（用于消融）。

用法
----
    from model_v2_linattn import SRPhase1LinAttn, LinAttnConfig
    model = SRPhase1LinAttn(LinAttnConfig(num_specials=64))
    out = model(pixel_values)          # {"loss","recon","F_hat","Y_pix","target_pix"}
    out["loss"].backward()

    自检: python model_v2_linattn.py      # 形状/参数量/梯度/像素往返/eval 同路径

参考资料:
    [1] Katharopoulos et al. (ICML 2020), Transformers are RNNs: Fast Autoregressive
        Transformers with Linear Attention. https://arxiv.org/abs/2006.16236
    [2] Han et al. (ICCV 2023), FLatten Transformer (聚焦线性注意力).
        https://arxiv.org/abs/2308.00442
    [3] Cai et al. (ICCV 2023), EfficientViT (线性注意力 + 局部混合).
        https://arxiv.org/abs/2305.07027
    [4] Darcet et al. (ICLR 2024), Vision Transformers Need Registers
        (register token 式出处, 主线同款). https://arxiv.org/abs/2309.16588
    [5] Jaegle et al. (ICML 2022), Perceiver IO (输出查询式解码出处).
        https://arxiv.org/abs/2107.14795
"""

from dataclasses import dataclass, field, asdict
import json
import math
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ImageNet 统计量 (= 主线 DINO_MEAN/DINO_STD 同值); 输入归一化与像素目标同一口径
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ═══ 像素 patch ↔ 图像 的布局变换（patch=16 版, 主线的泛化）═══
# 布局权威定义: patch 向量 = (C, p, p) **通道优先**, patch 顺序 row-major (先 y 后 x)。
# 与 model_v2.patches_to_image 同一口径, 仅 patch 边长由 14 泛化为 p。

def image_to_patches(x: Tensor, patch: int = 16) -> Tensor:
    """(B,3,H,W) → (B,N,C*p*p)（row-major patch 序, 通道优先 patch 向量）。"""
    B, C, H, W = x.shape
    assert H % patch == 0 and W % patch == 0, f"{H}x{W} 不是 {patch} 的整数倍"
    gh, gw = H // patch, W // patch
    return x.reshape(B, C, gh, patch, gw, patch) \
            .permute(0, 2, 4, 1, 3, 5) \
            .reshape(B, gh * gw, C * patch * patch)


def patches_to_image(pixels: Tensor, H: int, W: int, patch: int = 16,
                     mean: Optional[Sequence[float]] = None,
                     std: Optional[Sequence[float]] = None,
                     clamp_255: bool = True) -> Tensor:
    """(B,N,C*p*p) → (B,3,H,W)；给定 mean/std 时反归一化, 并（默认）截断到 0-255。

    ⚠️ **单位契约（踩过坑, 见 metrics_linattn.py §4 回归测试）**:
    mean/std 的标度必须与**返回值**的标度一致。
      · 本文件顶层 `IMAGENET_MEAN/STD` 是 **[0,1] 标度**（0.485/0.229）
        ⇒ 反归一化结果就在 [0,1], 调用方**不要再 /255**; 需要 [0,1] 裁剪时
        传 `clamp_255=False` 后自己 `.clamp(0,1)`（0-255 截断对 [0,1] 是空操作,
        语义上会误导）。
      · 仓库主线 `model_v2.patches_to_image` 的 mean/std 是 **0-255 标度**
        ⇒ 那里才需要 /255。两处口径不同, 不要互相抄。
    2026-10-15 冒烟时正是把 [0,1] 标度的结果又除了 255, 导致"L1=0.85 却
    PSNR=61dB/SSIM=0.99"的自相矛盾指标（训练损失不受影响, 指标与可视化全错）。

    `image_to_patches` 的严格逆（自检 §4 有逐位往返断言）。
    """
    B, N, px = pixels.shape
    assert px == 3 * patch * patch, f"patch 向量长度应为 {3 * patch * patch}, got {px}"
    assert (H // patch) * (W // patch) == N, \
        f"N={N} 与 {H}x{W} (patch={patch}) 的 patch 数 {(H // patch) * (W // patch)} 不符"
    img = pixels.reshape(B, H // patch, W // patch, 3, patch, patch) \
                .permute(0, 3, 1, 4, 2, 5) \
                .reshape(B, 3, H, W)
    if mean is not None and std is not None:
        m = torch.as_tensor(mean, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        s = torch.as_tensor(std, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        img = img * s + m
    if clamp_255:
        img = img.clamp(0.0, 255.0)
    return img


# ═══ 线性注意力 (Katharopoulos et al. 2020) ═══

def phi(x: Tensor) -> Tensor:
    """特征映射 φ(x) = elu(x) + 1（保证非负 ⇒ 分母为正、无除零）。"""
    return F.elu(x) + 1.0


class DepthwiseConv(nn.Module):
    """逐通道 3×3 卷积 + 残差（局部混合）。

    **零初始化** ⇒ 初始时 `y + conv(y) == y`（恒等）, 不破坏训练初期的
    "纯线性注意力"行为, 之后按需学出局部性（MLLA 的做法）。
    """

    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        B, N, D = x.shape
        y = x.transpose(1, 2).reshape(B, D, h, w)
        y = y + self.conv(y)
        return y.reshape(B, D, N).transpose(1, 2)


class LinearAttention(nn.Module):
    """核形式线性注意力（双向, 无因果掩码）。

        out_i = Σ_j φ(q_i)·φ(k_j) v_j / (φ(q_i)·Σ_j φ(k_j))
              = (φ(Q) (φ(K)^T V)) / (φ(Q) Σ_j φ(K_j))

    先聚合 KV 统计量 (B,H,d,d) 再乘 Q ⇒ 对序列长度 N 线性。

    ⚠️ **数值稳定性（2026-10-15 实测踩坑, 这不是可选项）**
    elu+1 核形式没有 softmax 的归一化, `Σ_j φ(k_j)` 与 `Σ_j φ(k_j)v_jᵀ` 会随
    qkv 权重增大而**无界增长**; 在 bf16 下（尾数 8 位 ≈ 3 位十进制）分子分母
    都到 1e3 量级时相对误差 ~1e-3, 噪声经 13 层放大 ⇒ **梯度间歇性爆炸**。
    实测（K=64/bs128/lr3e-4）: 前 5000 步一切正常（val PSNR 19.58dB @ep5）,
    随后 grad_norm 从 0.6 跳到 26 → 338 → 1.2e6 → 3e8, 损失 0.32→0.96,
    PSNR 掉到 11.8dB 且**不可恢复**。三处修复（缺一不可）：

      1. **QK-Norm**（`qk_norm=True`, 默认开）: 在 φ 之前对每个 head 的 q/k 做
         LayerNorm(head_dim) ⇒ φ 的输出被钉在 [0, ~4], `Σφ(k)` 有界,
         分子分母不再随训练漂移（ViT-22B / FLatten 的同类做法）;
      2. **核内强制 fp32**: `.float()` 还不够 —— autocast 会把 fp32 的 matmul
         再降回 bf16, 必须用 `torch.autocast(enabled=False)` 包住聚合段
         （该段算力占比 <1%, 几乎不影响吞吐）;
      3. **梯度爆炸跳步护栏**（在 train 侧, `--skip_grad`）: 前向产出巨大
         activation 时参数本身还是好的, 只要**不做那一步优化器更新**就不会坏。

    分母 +1e-6 防除零（φ≥1 时理论不会发生, 但低精度下仍稳妥）。
    """

    def __init__(self, dim: int, heads: int = 8, dwconv: bool = True,
                 qk_norm: bool = True):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 须被 heads={heads} 整除"
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.local = DepthwiseConv(dim) if dwconv else None
        # 每 head 的 QK LayerNorm（参数极少: 13 层 × 2 × 2×56 ≈ 2.9k）
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None

    def forward(self, x: Tensor, h: int, w: int, n_prefix: int = 0) -> Tensor:
        """x: (B, n_prefix + h*w, D); 局部混合只作用在 patch 段（前 n_prefix 个
        token 是 cls/specials, 没有 2-D 网格语义, 不参与卷积）。"""
        if self.local is not None and n_prefix < x.shape[1]:
            pre = x[:, :n_prefix]
            pat = self.local(x[:, n_prefix:], h, w)
            x = torch.cat([pre, pat], dim=1) if n_prefix > 0 else pat
        B, N, D = x.shape
        H, d = self.heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, H, d).transpose(1, 2)           # (B,H,N,d)
        k = k.reshape(B, N, H, d).transpose(1, 2)
        v = v.reshape(B, N, H, d).transpose(1, 2)
        if self.q_norm is not None:                         # 稳定性修复 1
            q = self.q_norm(q)
            k = self.k_norm(k)
        out_dtype = q.dtype
        # 稳定性修复 2: 聚合段强制 fp32（autocast 会忽略 .float(), 必须关掉）
        with torch.autocast(device_type=x.device.type, enabled=False):
            qf = phi(q.float())
            kf = phi(k.float())
            vf = v.float()
            kv = kf.transpose(-1, -2) @ vf                  # (B,H,d,d)
            zs = kf.sum(dim=-2)                             # (B,H,d)  fp32 累加
            out = qf @ kv / (qf @ zs.unsqueeze(-1) + 1e-6)  # (B,H,N,d)
        out = out.to(out_dtype).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class LinearAttentionBlock(nn.Module):
    """pre-norm: x + LinearAttn(LN(x)) → x + FFN(LN(x))"""

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0,
                 dwconv: bool = True, qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads, dwconv, qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x: Tensor, h: int, w: int, n_prefix: int = 0) -> Tensor:
        x = x + self.attn(self.norm1(x), h, w, n_prefix)
        x = x + self.mlp(self.norm2(x))
        return x


# ═══ 线性注意力编码器（从零训练, 无预训练权重）═══

class LinearAttentionEncoder(nn.Module):
    """patch embed + register 式 specials + L 层线性注意力 → (z_cls, z_s)。

    序列 = `[cls(1); specials(K); patches(N)]`（与主线 register 式同口径）:
    specials 直接进主干、与 patches 双向交互 ⇒ z_s 由"内容路由"学出,
    而不是"无内容输入的可学习常量"。主干全双向（重建无因果需求）。
    序列总长 = 1 + K + N = 261 (K=64, N=196), 线性注意力对其为 O(N) 聚合。
    """

    def __init__(self, dim: int = 448, depth: int = 13, heads: int = 8,
                 mlp_ratio: float = 4.0, patch: int = 16, img_size: int = 224,
                 num_specials: int = 64, dwconv: bool = True,
                 qk_norm: bool = True, drop_rate: float = 0.0):
        super().__init__()
        assert img_size % patch == 0, f"img_size={img_size} 须被 patch={patch} 整除"
        self.patch = patch
        self.img_size = img_size
        self.num_patches = (img_size // patch) ** 2
        self.num_specials = num_specials
        self.dim = dim

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.special_tokens = nn.Parameter(torch.randn(1, num_specials, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.pos_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.blocks = nn.ModuleList([
            LinearAttentionBlock(dim, heads, mlp_ratio, dwconv, qk_norm)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.special_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos(self, n: int) -> Tensor:
        """位置编码; n != num_patches 时线性插值（便于小分辨率冒烟）。"""
        if n == self.num_patches:
            return self.pos_embed
        pos = F.interpolate(self.pos_embed.transpose(1, 2), size=(n,),
                            mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, pixel_values: Tensor) -> Tuple[Tensor, Tensor]:
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
        return seq[:, :1], seq[:, 1:1 + self.num_specials]


# ═══ 消融臂: 普通 (softmax) 注意力编码器 ═══
#   2026-10-16 用户要求: 「把编码器换成普通的 transformer, 跑 BPTT」。
#   本节的 block 与 LinearAttentionBlock **结构逐项对应**（pre-norm: 注意力 →
#   残差 → FFN → 残差, GELU, mlp_ratio=4）, **唯一自变量 = 注意力形式**:
#       线性核注意力 (φ=elu+1, O(N))  →  标准 softmax 缩放点积注意力 (O(N²))
#   参数量天然对齐: 普通块少了 DWConv(9D+9) 与 QK-Norm(4·head_dim) 两项,
#   dim=448/depth=13 时 **31.848M vs 31.909M（−0.19%）** ⇒ 编码器体量不变,
#   对比可归因到注意力形式本身（见 __main__ §9 实测断言）。

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
                 drop_rate: float = 0.0):
        super().__init__()
        assert img_size % patch == 0, f"img_size={img_size} 须被 patch={patch} 整除"
        self.patch = patch
        self.img_size = img_size
        self.num_patches = (img_size // patch) ** 2
        self.num_specials = num_specials
        self.dim = dim

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.special_tokens = nn.Parameter(torch.randn(1, num_specials, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.pos_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.blocks = nn.ModuleList([
            StandardAttentionBlock(dim, heads, mlp_ratio, qk_norm)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.special_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos(self, n: int) -> Tensor:
        """位置编码; n != num_patches 时线性插值（便于小分辨率冒烟）。"""
        if n == self.num_patches:
            return self.pos_embed
        pos = F.interpolate(self.pos_embed.transpose(1, 2), size=(n,),
                            mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, pixel_values: Tensor) -> Tuple[Tensor, Tensor]:
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
        return seq[:, :1], seq[:, 1:1 + self.num_specials]


# 编码器名 → 构造函数的注册表（`enc_attn` 的合法取值; 见 LinAttnConfig）
ENCODER_REGISTRY = {
    "linear": "LinearAttentionEncoder",
    "linattn": "LinearAttentionEncoder",
    "std": "StandardAttentionEncoder",
    "standard": "StandardAttentionEncoder",
    "softmax": "StandardAttentionEncoder",
}


# ═══ 像素头（与主线 PixelHead 同结构, patch_px 泛化到 16×16×3）═══

class PixelHead(nn.Module):
    """(...,N,dim) → (...,N,patch_px); 2 层 MLP + GELU, 输出不加激活。"""

    def __init__(self, dim: int, patch_px: int = 16 * 16 * 3, hidden: int = 1024):
        super().__init__()
        self.patch_px = patch_px
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, patch_px))

    def forward(self, feat: Tensor) -> Tensor:
        return self.net(feat)


# ═══ 标准（普通 softmax）注意力解码器 ═══

class StandardAttentionDecoder(nn.Module):
    """输出查询式 + 普通 softmax 注意力 Transformer 解码器（单步并行）。

    与主线 `OutputQueryDecoder` 的**接口**一致 (z_cls, z_s) → (B,N,patch_px),
    区别只在内部: 这里用 `nn.TransformerDecoder`（多头**softmax** 注意力:
    查询自注意力 + 对 memory 的交叉注意力 + FFN, pre-norm）堆叠 depth 层,
    memory = `[z_cls; z_s]` 一次算完, 不做循环读出（见文件头"刻意偏离"）。

    查询基行 k ↔ patch k（Perceiver IO 口径, 行数恒为 N）;
    `query_base` 静态 + `query_pos` 可学习位置; 两者相加后全 batch 共享。
    """

    def __init__(self, enc_dim: int = 448, dim: int = 512, depth: int = 7,
                 heads: int = 8, mlp_ratio: float = 4.0, patch: int = 16,
                 num_patches: int = 196, num_specials: int = 64,
                 pixel_hidden: int = 1024, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 须被 heads={heads} 整除"
        self.enc_dim = enc_dim
        self.dim = dim
        self.num_patches = num_patches
        self.num_specials = num_specials
        self.patch_px = 3 * patch * patch

        self.mem_proj = (nn.Identity() if enc_dim == dim
                         else nn.Linear(enc_dim, dim))
        self.mem_norm = nn.LayerNorm(dim)
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True),
            num_layers=depth)
        self.norm_out = nn.LayerNorm(dim)
        self.pixel_head = PixelHead(dim, self.patch_px, pixel_hidden)
        nn.init.trunc_normal_(self.query_base, std=0.02)
        nn.init.trunc_normal_(self.query_pos, std=0.02)

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        assert z_cls.dim() == 3 and z_s.dim() == 3, \
            f"z_cls/z_s 须为 (B,·,D), got {tuple(z_cls.shape)}/{tuple(z_s.shape)}"
        assert z_cls.shape[1] == 1, f"z_cls 应为 1 列, got {z_cls.shape[1]}"
        assert z_s.shape[1] == self.num_specials, \
            f"z_s 列数 {z_s.shape[1]} != num_specials={self.num_specials}"
        assert z_cls.shape[0] == z_s.shape[0], "batch 不一致"
        mem = torch.cat([z_cls, z_s], dim=1)                # (B,1+K,D_enc)
        mem = self.mem_norm(self.mem_proj(mem))             # (B,1+K,D_dec)
        B = mem.shape[0]
        tgt = (self.query_base + self.query_pos).unsqueeze(0).expand(B, -1, -1)
        y = self.stack(tgt, mem)                            # (B,N,D_dec)
        return self.pixel_head(self.norm_out(y))            # (B,N,patch_px)


# ═══ 逐步（循环）读出：平方块读窗口 + BPTT carry（与主线 model_v2 同口径）═══
#   本节的 square_block_starts / step_windows 是主线 model_v2 同名函数的本地副本,
#   语义**逐位一致**（自检 §8 直接与主线对拍, 防止两处口径漂移）。

def square_block_starts(num_patches: int) -> list:
    """步集 = {k² | 1 ≤ k² ≤ num_patches}（主线同式）。

    N=196 → [1,4,9,…,196]（14 步）; K=64 → [1,4,9,16,25,36,49,64]（**8 步**）。
    块 k = [k², min((k+1)²−1, N)] —— 步值即"该步读完前 k² 个 token"。
    """
    K = math.isqrt(int(num_patches))
    return [k * k for k in range(1, K + 1)]


def step_windows(num_specials: int, steps, plan: str = "square",
                 block: int = 0) -> list:
    """每个采样步在 A 坐标（0=z_cls, 1..K=z_s）下的读窗口 `[(lo, hi), …]`。

    与主线 `model_v2.step_windows` **逐位一致**（自检 §8 对拍）:
      · `plan="square"`（默认）: k=⌊√t⌋, hi=min((k+1)²−1, K);
        非首步 lo=max(k²,1); 首步 lo=0（额外含 z_cls）。
      · `plan="fixed"`: k=t, hi=min(k·block, K); 非首步 lo=(k−1)·block+1。
    """
    assert plan in ("square", "fixed"), f"未知 plan={plan}"
    if plan == "fixed":
        assert block >= 1, f"plan=fixed 须 block ≥1, got {block}"
    out = []
    for i, t in enumerate(steps):
        if plan == "fixed":
            k = int(t)
            hi = min(k * block, num_specials)
            lo = 0 if i == 0 else (k - 1) * block + 1
        else:
            k = math.isqrt(int(t))
            hi = min((k + 1) ** 2 - 1, num_specials)
            lo = 0 if i == 0 else max(k * k, 1)
        assert lo <= hi, \
            f"步 {t} 的读窗口 [{lo},{hi}] 为空 (K={num_specials}, plan={plan})"
        out.append((lo, hi))
    return out


class RecurrentOutputQueryDecoder(nn.Module):
    """主线口径的**循环读出**解码器（普通 softmax 注意力 stack 复用 T 次）。

    与 `StandardAttentionDecoder` **参数形状逐 key 一致**（同名子模块
    mem_proj/mem_norm/query_base/query_pos/stack/norm_out/pixel_head）
    ⇒ 可直接 `load_state_dict` 做 warm start; 额外只多一个**零初始化**的
    `mem_pos`（(1,K+1,dim)，给 memory 位置信息, 初始为恒等不改变行为）。

    机制（与 `model_v2.OutputQueryDecoder` 同构）:
        Y = query_base + query_pos                    ← step1 查询
        for 步 t（读窗口 lo..hi = 该步自己那块 z_s）:
            Y = stack(Y, mem[:, lo:hi+1])             ← 自注意力 + 交叉注意力到本块
            out_t = pixel_head(norm_out(Y))           ← 该步对**全部 N 个 patch** 的直接预测
            Y = query_base + Y                        ← carry（**BPTT, 不 detach**）
    返回 (B, T, N, patch_px)；训练损失 = 各步直接预测的**平权**像素 L1（主线同款）。

    与单步版的关系: T=1 时退化为单步版（只是多了一个零初始化 mem_pos）。
    """

    def __init__(self, enc_dim: int = 448, dim: int = 512, depth: int = 7,
                 heads: int = 8, mlp_ratio: float = 4.0, patch: int = 16,
                 num_patches: int = 196, num_specials: int = 64,
                 pixel_hidden: int = 1024, dropout: float = 0.0,
                 steps=None, step_plan: str = "square", block: int = 0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 须被 heads={heads} 整除"
        self.enc_dim, self.dim = enc_dim, dim
        self.num_patches, self.num_specials = num_patches, num_specials
        self.patch_px = 3 * patch * patch

        self.mem_proj = (nn.Identity() if enc_dim == dim
                         else nn.Linear(enc_dim, dim))
        self.mem_norm = nn.LayerNorm(dim)
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True),
            num_layers=depth)
        self.norm_out = nn.LayerNorm(dim)
        self.pixel_head = PixelHead(dim, self.patch_px, pixel_hidden)
        # memory 位置编码（零初始化 ⇒ 初始恒等, 不破坏 warm start）
        self.mem_pos = nn.Parameter(torch.zeros(1, num_specials + 1, dim))

        self.steps = ([int(x) for x in steps] if steps
                      else square_block_starts(num_specials))
        self.step_plan, self.block = step_plan, int(block)
        self.windows = step_windows(num_specials, self.steps, step_plan, self.block)
        self.last_windows = self.windows

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        assert z_cls.dim() == 3 and z_s.dim() == 3, "z_cls/z_s 须为 (B,·,D)"
        assert z_cls.shape[1] == 1 and z_s.shape[1] == self.num_specials, \
            f"z_cls 应 1 列 / z_s 应 {self.num_specials} 列"
        mem = torch.cat([z_cls, z_s], dim=1)                # (B,1+K,D_enc)
        mem = self.mem_norm(self.mem_proj(mem)) + self.mem_pos
        B = mem.shape[0]
        base = (self.query_base + self.query_pos).unsqueeze(0).expand(B, -1, -1)
        Y = base
        outs = []
        for lo, hi in self.windows:
            Y = self.stack(Y, mem[:, lo:hi + 1])            # 只读本步那块 memory
            outs.append(Y)
            Y = self.query_base.unsqueeze(0) + Y            # carry = BPTT（不 detach）
        Ys = torch.stack(outs, dim=1)                       # (B,T,N,dim)
        self.last_Y = Ys
        return self.pixel_head(self.norm_out(Ys))           # (B,T,N,patch_px)


# ═══ 配置 ═══

@dataclass
class LinAttnConfig:
    """默认值 = 实测编码器 31.8M / 解码器 31.1M 的配置（见 count_params）。"""
    img_size: int = 224
    patch: int = 16
    num_specials: int = 64          # K: z_s 压缩 token 数（N=196; K<N 即真压缩）
    # ── 编码器（线性注意力, 从零训练）──
    enc_dim: int = 448
    enc_depth: int = 13
    enc_heads: int = 8
    enc_mlp_ratio: float = 4.0
    enc_dwconv: bool = True
    enc_qk_norm: bool = True       # 线性注意力稳定性修复 1（见 LinearAttention）
    # ── 编码器注意力形式（消融开关; 2026-10-16）──
    #   "linear"（默认, 昨日臂）= 核形式线性注意力 + DWConv 局部混合;
    #   "std" = **普通 transformer** 的 softmax 缩放点积注意力（消融臂）。
    #   两者只差注意力形式, 参数量 31.909M vs 31.848M（−0.19%）。
    #   注意: enc_dwconv 只在 "linear" 下有意义; enc_qk_norm 只在 "linear" 下有意义。
    enc_attn: str = "linear"
    enc_std_qk_norm: bool = False  # "std" 编码器的 QK-Norm（默认关 = 普通 transformer 原貌）
    # ── 解码器（普通 softmax 注意力; dec_recurrent=True 时走循环读出）──
    dec_recurrent: bool = False   # 循环读出（主线口径; 单步版 = False）
    dec_steps: str = ""           # 逗号分隔采样步; 空 = square_block_starts(K)
    # ── 解码器（普通 softmax 注意力）──
    dec_dim: int = 512
    dec_depth: int = 7
    dec_heads: int = 8
    dec_mlp_ratio: float = 4.0
    pixel_hidden: int = 1024
    dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch) ** 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["num_patches"] = self.num_patches
        d["enc_patch_px"] = 3 * self.patch * self.patch
        return d


# ═══ 主模型 ═══

class SRPhase1LinAttn(nn.Module):
    """无预训练编码器的 SR-Diffusion Phase 1（实验版）。

    `forward(pixel_values)` → `{"loss","recon","F_hat","Y_pix","target_pix"}`
    · target_pix: (B,N,3*p*p) 归一化像素 patch（`image_to_patches` 布局）
    · Y_pix:      (B,1,N,3*p*p) 单步直接预测（|T|=1; 见文件头"刻意偏离"）
    · F_hat:      (B,N,3*p*p) = Y_pix[:, -1]（= 唯一那一步）
    · loss = recon = 像素 L1（归一化空间, 直接预测口径）
    """

    def __init__(self, config: Optional[LinAttnConfig] = None, **kwargs):
        super().__init__()
        cfg = config if config is not None else LinAttnConfig()
        if kwargs:
            # to_dict() 里含**派生键** num_patches / enc_patch_px（不是 dataclass
            # 字段）, 必须先过滤否则 LinAttnConfig(**...) 直接 TypeError
            # （2026-10-15 独立审查 finding 2）。
            merged = {**cfg.to_dict(), **kwargs}
            cfg = LinAttnConfig(**{k: v for k, v in merged.items()
                                   if k in LinAttnConfig.__dataclass_fields__})
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.patch = cfg.patch
        self.num_patches = cfg.num_patches
        self.num_specials = cfg.num_specials
        assert 1 <= cfg.num_specials <= self.num_patches, \
            f"num_specials(K)={cfg.num_specials} 越界 (须 1 ≤ K ≤ N={self.num_patches})"

        self.encoder = self._build_encoder(cfg)
        _dec_kw = dict(
            enc_dim=cfg.enc_dim, dim=cfg.dec_dim, depth=cfg.dec_depth,
            heads=cfg.dec_heads, mlp_ratio=cfg.dec_mlp_ratio, patch=cfg.patch,
            num_patches=self.num_patches, num_specials=cfg.num_specials,
            pixel_hidden=cfg.pixel_hidden, dropout=cfg.dropout)
        self.recurrent = bool(cfg.dec_recurrent)
        if self.recurrent:
            _steps = [int(x) for x in str(cfg.dec_steps).split(",") if x.strip()]
            self.decoder = RecurrentOutputQueryDecoder(steps=_steps or None, **_dec_kw)
        else:
            self.decoder = StandardAttentionDecoder(**_dec_kw)

    @staticmethod
    def _build_encoder(cfg: "LinAttnConfig") -> nn.Module:
        """按 `cfg.enc_attn` 选编码器（消融开关; 默认 "linear" = 昨日臂, 行为不变）。"""
        kind = str(cfg.enc_attn).lower()
        assert kind in ENCODER_REGISTRY, \
            f"未知 enc_attn={cfg.enc_attn!r}; 合法值: {sorted(ENCODER_REGISTRY)}"
        if kind == "linear":
            return LinearAttentionEncoder(
                dim=cfg.enc_dim, depth=cfg.enc_depth, heads=cfg.enc_heads,
                mlp_ratio=cfg.enc_mlp_ratio, patch=cfg.patch, img_size=cfg.img_size,
                num_specials=cfg.num_specials, dwconv=cfg.enc_dwconv,
                qk_norm=cfg.enc_qk_norm, drop_rate=cfg.dropout)
        if kind == "std":
            return StandardAttentionEncoder(
                dim=cfg.enc_dim, depth=cfg.enc_depth, heads=cfg.enc_heads,
                mlp_ratio=cfg.enc_mlp_ratio, patch=cfg.patch, img_size=cfg.img_size,
                num_specials=cfg.num_specials, qk_norm=cfg.enc_std_qk_norm,
                drop_rate=cfg.dropout)
        raise AssertionError(f"未处理的 enc_attn={kind!r}")

    # ── encode / decode（训练与推理同一路径, 与主线同名同义）──
    def encode(self, pixel_values: Tensor) -> Tuple[Tensor, Tensor]:
        return self.encoder(pixel_values)

    def decode(self, z_cls: Tensor, z_s: Tensor, pixel_values: Tensor) -> dict:
        """解码器 + 像素头 + 像素 L1（**各步直接预测平权**, 与主线同口径）。

        单步版（dec_recurrent=False）: `pix` = (B,N,pp) → 补 |T|=1 维,
        损失与改造前**逐位相同**（T=1 的平权 = 原来的单步 L1）。
        循环版: `pix` = (B,T,N,pp), 每步各自直接预测整图,
        `loss = mean_t L1(Y_t, target)`, `F_hat = Y_pix[:, -1]`（末步）。
        额外返回 `step_l1`(T,) = 每一步在**归一化空间**的 L1（逐步分析用）。
        """
        B, C, H, W = pixel_values.shape
        pix = self.decoder(z_cls, z_s)                       # (B,N,pp) 或 (B,T,N,pp)
        if pix.dim() == 3:
            pix = pix.unsqueeze(1)                           # (B,1,N,pp)
        target_pix = image_to_patches(pixel_values, self.patch)
        per_step = F.l1_loss(pix, target_pix.unsqueeze(1).expand_as(pix),
                             reduction="none").mean(dim=(0, 2, 3))   # (T,)
        F_pix = pix[:, -1]                                   # 末步 = F_hat
        recon = F.l1_loss(F_pix, target_pix)
        return {"loss": per_step.mean(), "recon": recon, "F_hat": F_pix,
                "Y_pix": pix, "target_pix": target_pix,
                "step_l1": per_step.detach()}

    def forward(self, pixel_values: Tensor) -> dict:
        x = pixel_values
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"本实验版固定 {self.img_size}x{self.img_size} 输入, got {H}x{W}"
        z_cls, z_s = self.encode(x)
        return self.decode(z_cls, z_s, x)

    # ── 参数统计（编码器/解码器分开, 用于"同等体量"核对）──
    def count_params(self) -> dict:
        def n(m):
            return sum(p.numel() for p in m.parameters())
        enc, dec = n(self.encoder), n(self.decoder)
        return {"encoder": enc, "decoder": dec, "total": enc + dec,
                "enc_M": round(enc / 1e6, 3), "dec_M": round(dec / 1e6, 3),
                "total_M": round((enc + dec) / 1e6, 3),
                "enc_dec_ratio": round(dec / enc, 4)}

    @torch.no_grad()
    def reconstruct(self, pixel_values: Tensor, mean=IMAGENET_MEAN,
                    std=IMAGENET_STD) -> Tensor:
        """推理便捷接口: (B,3,H,W) 归一化输入 → (B,3,H,W) **[0,1]** 重建图。

        ⚠️ 单位契约: mean/std 是 **[0,1] 标度**（0.485/0.229）⇒ 反归一化结果
        本身就是 [0,1]（见 `patches_to_image` 的说明）, 不是 0-255。预测可能
        越界, 这里显式 clamp(0,1)（与指标路径口径一致; 2026-10-15 独立审查
        finding 4: 旧实现 docstring 写 0-255、实际返回 [0,1] 且不裁剪）。
        """
        was_training = self.training
        self.eval()
        out = self(pixel_values)
        img = patches_to_image(out["F_hat"], self.img_size, self.img_size,
                               self.patch, mean, std, clamp_255=False)
        img = img.clamp(0.0, 1.0)
        if was_training:
            self.train()
        return img


# ═══ 自检（python model_v2_linattn.py）═══
#   1. 参数量（编码器/解码器各自 30 几 M, 打印精确值）
#   2. 形状正确性（默认配置 + 小配置; 含 K=N 与 K<N 两种压缩口径）
#   3. 梯度流向（编码器 + 解码器全部可训参数都有梯度）
#   4. 像素布局往返（image_to_patches → patches_to_image == 原图, 逐位）
#   5. eval 同路径（train/eval 输出一致, dropout=0）

if __name__ == "__main__":
    torch.manual_seed(0)

    def _fmt(d):
        return (f"enc={d['enc_M']}M dec={d['dec_M']}M total={d['total_M']}M "
                f"(dec/enc={d['enc_dec_ratio']})")

    print("=" * 78)
    print("[1] 参数量（默认配置 LinAttnConfig()）")
    cfg = LinAttnConfig()
    model = SRPhase1LinAttn(cfg)
    pc = model.count_params()
    print("    cfg:", json.dumps(cfg.to_dict(), ensure_ascii=False))
    print("    params:", _fmt(pc))
    assert abs(pc["enc_dec_ratio"] - 1.0) < 0.15, \
        f"编码器/解码器体量差 >15%: {pc['enc_dec_ratio']}"
    for k in ("enc_M", "dec_M"):
        assert 25.0 <= pc[k] <= 42.0, f"{k}={pc[k]}M 不在 30 几 M 量级"
    print("    OK: 编码器/解码器同为 30 几 M 且体量相当")

    print("[2] 形状正确性（默认配置, batch=2, 224×224, 无梯度）")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    N = cfg.num_patches
    pp = 3 * cfg.patch * cfg.patch
    assert out["F_hat"].shape == (2, N, pp), out["F_hat"].shape
    assert out["Y_pix"].shape == (2, 1, N, pp), out["Y_pix"].shape
    assert out["target_pix"].shape == (2, N, pp), out["target_pix"].shape
    assert out["loss"].shape == (), out["loss"].shape
    assert torch.isfinite(out["loss"]), "loss 非有限"
    print(f"    F_hat{tuple(out['F_hat'].shape)}  Y_pix{tuple(out['Y_pix'].shape)}"
          f"  target{tuple(out['target_pix'].shape)}  loss={out['loss'].item():.4f}")

    print("[3] 压缩口径 (K<N 与 K=N) + 小配置形状")
    for K in (cfg.num_specials, N):
        m = SRPhase1LinAttn(LinAttnConfig(num_specials=K))
        with torch.no_grad():
            o = m(x)
        assert o["F_hat"].shape == (2, N, pp)
        print(f"    K={K:>4d}: loss={o['loss'].item():.4f} "
              f"params={m.count_params()['total_M']}M")
    small = LinAttnConfig(img_size=64, patch=16, num_specials=8, enc_dim=64,
                          enc_depth=2, enc_heads=4, dec_dim=64, dec_depth=2,
                          dec_heads=4, pixel_hidden=128)
    ms = SRPhase1LinAttn(small)
    with torch.no_grad():
        os_ = ms(torch.randn(3, 3, 64, 64))
    assert os_["F_hat"].shape == (3, 16, 3 * 16 * 16)
    print(f"    小配置(64×64, N=16, K=8): F_hat{tuple(os_['F_hat'].shape)} OK")

    print("[4] 像素布局往返（image_to_patches ↔ patches_to_image 逐位相等）")
    xr = torch.rand(2, 3, 224, 224)
    rt = patches_to_image(image_to_patches(xr, 16), 224, 224, 16, clamp_255=False)
    md = (rt - xr).abs().max().item()
    assert md == 0.0, f"往返 max|diff|={md} (应为 0)"
    # ★ 用**数据管线真实产生的**归一化口径（[0,1] 图 + [0,1] 标度 mean/std）
    #   再验一次往返; 旧版这里用 (xr*255-mean)/std（管线永不产生的口径）,
    #   测试自洽但永远测不到真正的失败模式（2026-10-15 独立审查 finding 5）。
    m01 = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    s01 = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    xn = (xr - m01) / s01
    rt2 = patches_to_image(image_to_patches(xn, 16), 224, 224, 16,
                           IMAGENET_MEAN, IMAGENET_STD, clamp_255=False)
    md2 = (rt2 - xr).abs().max().item()
    assert md2 < 1e-5, f"真实归一化口径往返 max|diff|={md2}"
    print(f"    往返 max|diff|={md}（布局）; {md2:.2e}（管线真实归一化口径）")

    print("[5] 梯度流向（小配置, batch=2）")
    torch.manual_seed(0)
    mt = SRPhase1LinAttn(small)
    mt.train()
    ot = mt(torch.randn(2, 3, 64, 64))
    ot["loss"].backward()
    missing = [n_ for n_, p in mt.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"无梯度/非有限梯度的参数: {missing[:5]} ... 共 {len(missing)}"
    gn = sum(p.grad.norm().item() for p in mt.parameters() if p.grad is not None)
    n_enc_g = sum(1 for n_, p in mt.encoder.named_parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0)
    n_dec_g = sum(1 for n_, p in mt.decoder.named_parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"    全部 {sum(1 for _ in mt.parameters())} 个参数张量均有有限梯度; "
          f"梯度总范数≈{gn:.3f}; 编码器有梯度张量 {n_enc_g}, 解码器 {n_dec_g}")
    assert n_enc_g > 0 and n_dec_g > 0

    print("[6] eval/train 同路径（dropout=0 ⇒ 两次 forward 逐位一致）")
    xg = torch.randn(2, 3, 64, 64)
    mt.eval()
    with torch.no_grad():
        e1 = mt(xg)["F_hat"]
        e2 = mt(xg)["F_hat"]
    assert torch.equal(e1, e2), "eval 两次前向不一致"
    img = mt.reconstruct(xg[:1])
    assert img.shape == (1, 3, 64, 64), img.shape
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0, \
        f"reconstruct 应返回 [0,1], got [{img.min()}, {img.max()}]"
    print(f"    eval 两次前向逐位一致; reconstruct() → {tuple(img.shape)} "
          f"范围 [{img.min():.3f}, {img.max():.3f}] ∈ [0,1]")

    print("[7] ★数值稳定性（QK-Norm 必须把 φ 的量级钉死）")
    # 2026-10-15: 不设 QK-Norm 时 φ(k) 随 qkv 权重**无界增长** ⇒ 分子/分母都到
    # 1e3+ 量级, bf16 尾数只有 ~3 位十进制 ⇒ 相对噪声经 13 层放大 ⇒ 梯度间歇
    # 爆炸（实测 grad_norm 0.6→26→338→1.2e6→3e8, val PSNR 19.6→11.8dB 不可恢复）。
    # 这里直接量 **Σ_j φ(k_j) 的量级**随 qkv 权重放大的增长倍数:
    #   关掉 QK-Norm → 随权重线性放大（分母爆掉）; 打开 → 被 LayerNorm 钉住。
    def _den_mag(qk_norm: bool, scale: float) -> float:
        torch.manual_seed(0)
        att = LinearAttention(dim=32, heads=4, dwconv=False, qk_norm=qk_norm)
        with torch.no_grad():
            att.qkv.weight.mul_(scale)
            if att.qkv.bias is not None:
                att.qkv.bias.mul_(scale)
        x = torch.randn(2, 32, 32) * 10.0
        B, N, D = x.shape
        H, d = att.heads, att.head_dim
        with torch.no_grad():
            _, k, _ = att.qkv(x).chunk(3, dim=-1)
            k = k.reshape(B, N, H, d).transpose(1, 2)
            if att.k_norm is not None:
                k = att.k_norm(k)
            return float(phi(k.float()).sum(dim=-2).abs().max())

    base_on, base_off = _den_mag(True, 1.0), _den_mag(False, 1.0)
    big_on, big_off = _den_mag(True, 20.0), _den_mag(False, 20.0)
    grow_on = big_on / max(base_on, 1e-9)
    grow_off = big_off / max(base_off, 1e-9)
    assert all(map(lambda v: v == v and v < 1e12,
                   (base_on, base_off, big_on, big_off))), "出现 NaN/Inf"
    assert grow_off > 10.0, f"未复现未归一化的分母爆炸 (×{grow_off:.1f})"
    assert grow_on < 2.0, f"QK-Norm 后分母仍被放大 (×{grow_on:.2f})"
    print(f"    qkv 权重 ×20: Σφ(k) 无 QK-Norm 放大 ×{grow_off:,.1f}（分母爆炸）; "
          f"有 QK-Norm ×{grow_on:.2f}（钉住）")
    # 端到端: 大幅扰动权重后, 开了 QK-Norm 的整模型输出仍须有限
    torch.manual_seed(0)
    mstab = SRPhase1LinAttn(LinAttnConfig(img_size=64, patch=16, num_specials=8,
                                          enc_dim=64, enc_depth=2, enc_heads=4,
                                          dec_dim=64, dec_depth=2, dec_heads=4,
                                          pixel_hidden=128))
    with torch.no_grad():
        for n_, prm in mstab.named_parameters():
            if "qkv.weight" in n_:
                prm.mul_(20.0)
        o = mstab(torch.randn(2, 3, 64, 64) * 3.0)
    assert torch.isfinite(o["loss"]), f"扰动后损失非有限: {o['loss']}"
    print(f"    端到端（qkv 权重 ×20 + 输入 ×3）: loss={o['loss'].item():.4f} 有限 OK")

    print("[8] 循环读出（与主线 model_v2 口径对拍 + 与单步版参数兼容）")
    # 8a. 读窗口/步集必须与主线 model_v2 **逐位一致**（两处口径漂移是最危险的静默错误）
    import model_v2 as _mv2
    for K in (1, 4, 7, 15, 64, 196):
        assert square_block_starts(K) == _mv2.square_block_starts(K), K
        st = square_block_starts(K)
        assert step_windows(K, st, "square", 0) == _mv2.step_windows(K, st, "square", 0), K
        for blk in (1, 3, 8):
            fst = _mv2.fixed_block_starts(K, blk)
            assert step_windows(K, fst, "fixed", blk) == \
                _mv2.step_windows(K, fst, "fixed", blk), (K, blk)
    print(f"    step_windows/square_block_starts 与 model_v2 逐位一致 "
          f"(K=64 → steps={square_block_starts(64)})")

    # 8b. 循环解码器与单步解码器**参数形状逐 key 一致**（warm start 的前提）
    rdec = RecurrentOutputQueryDecoder(num_specials=64)
    sdec = StandardAttentionDecoder(num_specials=64)
    rs, ss_ = rdec.state_dict(), sdec.state_dict()
    shared = set(rs) & set(ss_)
    assert all(rs[k].shape == ss_[k].shape for k in shared), "共享 key 形状不一致"
    assert set(ss_) <= set(rs), f"单步版有循环版缺的 key: {set(ss_) - set(rs)}"
    only = set(rs) - set(ss_)
    assert only == {"mem_pos"}, f"循环版多出的 key 应为 mem_pos, got {only}"
    missing, unexpected = rdec.load_state_dict(ss_, strict=False)
    assert set(missing) == {"mem_pos"} and not unexpected
    assert float(rdec.mem_pos.abs().max()) == 0.0, "mem_pos 应零初始化"
    print(f"    循环/单步解码器共享 {len(ss_)} 个 key 且形状全同; "
          f"循环版仅多零初始化 mem_pos ⇒ 可 warm start")

    # 8c. 单步 == "只跑一步且窗口覆盖全部 memory" 的循环版（同一权重逐位一致）
    #     注意: steps=[1] 的窗口是 [0,3]（首块）, **不是**全量 —— 所以要用
    #     steps=[K]（k=⌊√K⌋ ⇒ hi=min((k+1)²−1,K)=K）才等价于单步的"读全部"。
    small_rec = LinAttnConfig(img_size=64, patch=16, num_specials=8, enc_dim=64,
                              enc_depth=2, enc_heads=4, dec_dim=64, dec_depth=2,
                              dec_heads=4, pixel_hidden=128,
                              dec_recurrent=True, dec_steps="8")
    mrec = SRPhase1LinAttn(small_rec)
    msingle = SRPhase1LinAttn(LinAttnConfig(img_size=64, patch=16, num_specials=8,
                                            enc_dim=64, enc_depth=2, enc_heads=4,
                                            dec_dim=64, dec_depth=2, dec_heads=4,
                                            pixel_hidden=128))
    msingle.load_state_dict(mrec.state_dict(), strict=False)
    xr2 = torch.randn(2, 3, 64, 64)
    mrec.eval(); msingle.eval()
    with torch.no_grad():
        o1, o2 = mrec(xr2), msingle(xr2)
    assert o1["Y_pix"].shape[1] == 1 and o2["Y_pix"].shape[1] == 1
    assert mrec.decoder.windows[0] == (0, 8), mrec.decoder.windows  # [K] ⇒ 读全量
    assert torch.allclose(o1["F_hat"], o2["F_hat"], atol=1e-6), \
        f"T=1 循环版与单步版不一致: {(o1['F_hat'] - o2['F_hat']).abs().max()}"
    print(f"    T=1 循环版与单步版前向逐位一致 (max|diff|={float((o1['F_hat']-o2['F_hat']).abs().max()):.2e})")

    # 8d. K=64 循环版形状 + 逐步 L1
    m64 = SRPhase1LinAttn(LinAttnConfig(num_specials=64, dec_recurrent=True)).eval()
    with torch.no_grad():
        o64 = m64(torch.randn(1, 3, 224, 224))
    assert o64["Y_pix"].shape == (1, 8, 196, 768), o64["Y_pix"].shape
    assert o64["step_l1"].shape == (8,)
    # A 坐标: 0=z_cls, 1..K=z_s ⇒ 末步 hi=min((8+1)²−1, 64)=64 (含 z_s[63])
    assert m64.decoder.windows[0] == (0, 3), m64.decoder.windows
    assert m64.decoder.windows[-1][1] == 64, m64.decoder.windows
    print(f"    K=64: Y_pix{tuple(o64['Y_pix'].shape)} (T=8), "
          f"读窗口 {m64.decoder.windows[0]}…{m64.decoder.windows[-1]}, "
          f"decoder 参数 {m64.count_params()['dec_M']}M")

    # 9e. 普通 transformer 编码器的 13 层参数体量必须与线性臂对齐（消融可归因的前提）
    print("[9] ★消融臂: 普通 (softmax) 注意力编码器（enc_attn='std'）")
    # 9a-9d 见下; 先验"注意力确实是标准 softmax 缩放点积"（数值对拍手工实现）
    torch.manual_seed(0)
    att_s = StandardAttention(dim=32, heads=4)
    xs = torch.randn(2, 7, 32)
    with torch.no_grad():
        got = att_s(xs)
        Bs, Ns, Ds = xs.shape
        Hs, ds = att_s.heads, att_s.head_dim
        q_, k_, v_ = att_s.qkv(xs).chunk(3, dim=-1)
        q_ = q_.reshape(Bs, Ns, Hs, ds).transpose(1, 2)
        k_ = k_.reshape(Bs, Ns, Hs, ds).transpose(1, 2)
        v_ = v_.reshape(Bs, Ns, Hs, ds).transpose(1, 2)
        w = torch.softmax(q_ @ k_.transpose(-1, -2) / math.sqrt(ds), dim=-1)
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5), \
            "softmax 注意力权重每行未归一（不是普通 transformer！）"
        want = att_s.proj((w @ v_).transpose(1, 2).reshape(Bs, Ns, Ds))
    dmax = float((got - want).abs().max())
    assert dmax < 1e-6, f"StandardAttention 与手工 softmax 实现不一致: {dmax}"
    print(f"    StandardAttention == softmax(QKᵀ/√d)V 手工实现 (max|diff|={dmax:.2e}); "
          f"注意力权重行和==1")

    # 9e. 普通 transformer 编码器的 13 层参数体量必须与线性臂对齐（消融可归因的前提）
    enc_lin = LinearAttentionEncoder(dim=448, depth=13, heads=8)
    enc_std = StandardAttentionEncoder(dim=448, depth=13, heads=8)
    n_lin = sum(p.numel() for p in enc_lin.parameters())
    n_std = sum(p.numel() for p in enc_std.parameters())
    rel = abs(n_std - n_lin) / n_lin
    assert rel < 0.01, f"编码器参数量差 {rel:.2%} 超过 1%（无法归因到注意力形式）"
    print(f"    编码器参数量: 线性 {n_lin:,} ({n_lin/1e6:.3f}M) vs "
          f"普通 {n_std:,} ({n_std/1e6:.3f}M)  相对差 {rel:.2%} OK")
    ms = SRPhase1LinAttn(LinAttnConfig(enc_attn="std"))
    ps = ms.count_params()
    assert type(ms.encoder).__name__ == "StandardAttentionEncoder", type(ms.encoder)
    assert 25.0 <= ps["enc_M"] <= 42.0 and 25.0 <= ps["dec_M"] <= 42.0
    print(f"    整机(enc_attn=std): {_fmt(ps)}")

    # 9f. 普通编码器端的形状/损失/梯度（单步 + BPTT 两种解码都要能跑通）
    xt = torch.randn(2, 3, 64, 64)
    for flag in (False, True):
        mc = SRPhase1LinAttn(LinAttnConfig(
            img_size=64, patch=16, num_specials=8, enc_dim=64, enc_depth=2,
            enc_heads=4, dec_dim=64, dec_depth=2, dec_heads=4, pixel_hidden=128,
            enc_attn="std", dec_recurrent=flag))
        T = len(mc.decoder.steps) if flag else 1
        mc.train()
        oc = mc(xt)
        assert oc["Y_pix"].shape[:2] == (2, T), oc["Y_pix"].shape
        assert torch.isfinite(oc["loss"])
        oc["loss"].backward()
        bad = [k for k, p in mc.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not bad, f"无梯度/非有限梯度: {bad[:5]}"
        nz = sum(1 for _, p in mc.named_parameters()
                 if p.grad is not None and p.grad.abs().sum() > 0)
        print(f"    enc_attn=std dec_recurrent={int(flag)} T={T}: "
              f"Y_pix{tuple(oc['Y_pix'].shape)} loss={oc['loss'].item():.4f} "
              f"有梯度张量 {nz}/{sum(1 for _ in mc.parameters())} OK")

    # 9g. 普通编码器在 bf16 autocast 下前向/反向必须有限（softmax 自带归一的护栏）
    mb = SRPhase1LinAttn(LinAttnConfig(
        img_size=64, patch=16, num_specials=8, enc_dim=64, enc_depth=2,
        enc_heads=4, dec_dim=64, dec_depth=2, dec_heads=4, pixel_hidden=128,
        enc_attn="std"))
    mb.train()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ob = mb(xt)
    assert torch.isfinite(ob["loss"]), f"bf16 下损失非有限: {ob['loss']}"
    ob["loss"].backward()
    gn = torch.nn.utils.clip_grad_norm_(list(mb.parameters()), 1e9)
    assert torch.isfinite(gn) and float(gn) < 1e4, f"bf16 梯度范数异常: {gn}"
    print(f"    enc_attn=std bf16 autocast: loss={ob['loss'].item():.4f} "
          f"grad_norm={float(gn):.3f} 有限 OK")

    # 9h. 默认路径不受影响（回归: 昨天的 linattn 臂必须逐位仍是原实现）
    cfg_def = LinAttnConfig()
    assert cfg_def.enc_attn == "linear" and cfg_def.enc_std_qk_norm is False
    mdl_def = SRPhase1LinAttn(cfg_def)
    assert type(mdl_def.encoder).__name__ == "LinearAttentionEncoder"
    assert mdl_def.count_params()["encoder"] == 31909024, \
        f"默认编码器参数量变了: {mdl_def.count_params()['encoder']} ≠ 31909024"
    assert type(mdl_def.decoder).__name__ == "StandardAttentionDecoder"
    print(f"    默认路径回归: enc_attn={cfg_def.enc_attn!r} → "
          f"{type(mdl_def.encoder).__name__} {mdl_def.count_params()['encoder']:,} 参数 不变 OK")

    # 9i. 昨天 ckpt 仍能 strict 载入（若本机有该文件; 远程/本地都可能没有 ⇒ 跳过）
    _ck = os.environ.get("LINATTN_CKPT", "")
    if _ck and os.path.exists(_ck):
        ck_old = torch.load(_ck, map_location="cpu", weights_only=False)
        cfg_old = LinAttnConfig(**{k: v for k, v in ck_old["cfg"].items()
                                   if k in LinAttnConfig.__dataclass_fields__})
        m_old = SRPhase1LinAttn(cfg_old)
        m_old.load_state_dict(ck_old["model"])       # strict=True
        print(f"    回归: 旧 ckpt strict 载入成功 ({os.path.basename(_ck)}) OK")
    else:
        print(f"    回归: 未提供 LINATTN_CKPT（跳过旧 ckpt strict 载入检查）")

    print("=" * 78)
    print("ALL CHECKS PASSED")
