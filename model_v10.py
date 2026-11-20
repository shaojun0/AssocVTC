"""v10 = **1792 原生宽度 + 全片段循环读出**的视频模型（2026-11-16 设计定稿）。

宽度只有一种：896 = 2 token × 448（正式配方；1792 = 4×448 是早期 K=32 设计口径）
----------------------------------
`chunk_tokens=2` 个 token 拼成一个 896 向量，于是「宽度」和「位置数」由同一
个常量决定，任何位置都不用投影、不用填充：

    码率侧   16 token/帧  → 8 个 896/帧
    预测侧   16 special + 1 z_cls → 8 个 896 + 1 行（冻结 R 出像素必须要 cls）

旧版把 token 直接当位置、宽 1024 时，每帧 33×1024 = 33,792 个数只承载
33×448 = 14,784 个值 ⇒ **填充 2.29×**；本版 8×896 = 7,168 承载 7,168 ⇒ 填充 1.00×。

实现说明（2026-11-16 重构）
---------------------------
注意力、FFN 与堆叠全部使用 PyTorch 原生 `nn.TransformerEncoderLayer` /
`nn.TransformerDecoderLayer`；本文件只保留模型编排（打包、循环读出、门控），
不再手写 QKV 投影和两层 MLP。

- middle：`V2EncoderMiddle`（2026-11-20 起默认；**v2 编码器同构**）：冻结 E 的
  120×16×448 先 reshape 成 480×1792，再拼一份同尺寸的可学 query（可加 cls），
  960 个 token 做**全双向自注意力**（`nn.TransformerEncoder`），取 query 半的
  输出 reshape 回 (B,120,8,896) 交给读出 ⇒ 读出侧一行不动。
  `mid_style="crossq"` 保留 2026-11-19 的 `CrossQueryMiddle`（可学 query 对 E
  做交叉注意力重采样）做对照；旧版帧级因果 `TemporalMiddle` 与 prefix 读出
  已于 2026-11-19 移除。
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
    rate         前缀截断前 16 个 special → reshape (B,120,8,896)（纯 reshape）
    middle       2026-11-20 起默认 v2enc：480×1792 的 E 行 + 480×1792 可学 query
                 → 960 token 全双向自注意力 → 取 query 半 → (B,120,8,896)
                 （mid_style="crossq" 时是 960 个 query 对 E 行的交叉注意力）
                 reshape → z (B,30,32,896)（4 帧 × 8 行 = 每步 32 行）
    readout      Q（可学习，随机初始化）(1,1089,896)；γ_0 = 0
                 for t in 0..29:
                     h    = Q + γ_t·Y_{t-1} + step_emb[t]   # carry 同 v2：无 LN
                     Y_t  = Decoder(h, z_t)     全双向 self-attn + cross-attn(32×896)
                     pred = LN(Y_t) → 后 120 帧 × 9 行
                          8 行 reshape 成 16×448；第 9 行过 cls_head（896→448）成 z_cls
                 loss = mean_t L1(pred_t, 冻结编码器给的全量 17 行)
    R（冻结）    用 pred_30 的 (z_cls, 32 special) 还原像素，用于 PSNR

2026-11-18 与 v2 对齐（远端 946226d 口径）
------------------------------------------
循环骨架与 `model_v2.OutputQueryDecoder` 保持一致：
  * carry = `Q + Y_{t-1}`（裸加，**无 LN**；默认也没有 γ 门控，即等权相加；
    Y 本身已被 decoder 末层 norm 归一化，状态有界）；
  * 读出侧 self-attn **全双向、无 tgt_mask**（streaming 由「第 t 步 memory 只有
    前 t+1 组 z」保证，不靠 self-attn mask）。

为什么「率 < 目标」
------------------
率 = 前 16 个 token（7,168 值/帧）；正式配方里分词器 K=16 ⇒ **率 = 目标**（没有压缩级，§7 已记录这个边界；本段「率<目标」的论证只对早期 K=32 设计成立）。
如果目标也只有 16 个 token，恒等映射就是最优解，跨帧混合没有任何存在必要。

可改的旋钮（V10Config）
-----------------------
rate_tokens     率：每帧保留多少 special token（默认 16，= 前缀截断）
rate_dim        单个 token 宽度（= 分词器 zs_dim，448）
chunk_tokens    一个向量拼几个 token（默认 2 ⇒ 宽度 896，不允许改宽度）
target_tokens   目标 special 数；本版必须等于 num_specials（冻结 R 要全量）
group           z 步覆盖帧数（4 → 30 步，2 fps 潜在帧）
mid_style       "v2enc"（默认，v2 编码器同构）| "crossq"（2026-11-19 交叉查询）
mid_hidden      v2enc 编码器宽度（0 = 2×width；正式 = 1792）
mid_depth/heads/ratio  middle 的层数/头数/FFN 比例（v2enc 建议 depth 2~3、ratio 2）
mid_cls         v2enc 是否加 1 个 cls（输出广播加到所有 rate 行；默认关）
mid_skip        v2enc 是否加 E 行残差（非 v2 口径的恒等保险；默认关）
mid_dim/dec_dim 只接受 0（自动）或等于 width，防止"读出再投影到别的宽度"
step_gate       是否用零初始化门控 γ（默认关 = w≡1，carry 与 v2 等权相加）
readout_mask     读出 self-attn 是否帧级因果（默认 0 = 全双向，同 v2）
readout_copy     已读帧 rate 行直抄残差（copy_skip；默认关。2026-11-20 判别用：
                 赢家配方里它给 middle 提供直达 loss 的恒等梯度，被 635e90c 删掉后
                 任何 middle 都塌，故以开关形式恢复，默认不影响其它路径）

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

__all__ = ["V10Config", "FrozenTokenizer",
           "V2EncoderMiddle", "IdentityMiddle", "CrossQueryMiddle",
           "WholeClipDecoder", "VideoV10",
           "patches_to_image", "psnr_from_l1"]


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
    target_tokens: int = 16    # 目标 special token 数（本分支配方 = num_specials = 16）
    chunk_tokens: int = 2      # 一个可训练向量 = 2 个 token 拼接（2×448 = 896，不填充）
    # ── 中间态
    # "v2enc"    = v2 编码器同构：E 行(480×1792) 拼同尺寸可学 query → 自注意力
    #              → 取 query 半为 rate（2026-11-20 用户定稿，默认）
    # "crossq"   = 2026-11-19 的交叉查询重采样（对照）
    mid_style: str = "v2enc"
    mid_dim: int = 0           # crossq 宽度；0 = 自动取 width（896）
    mid_hidden: int = 0        # v2enc 编码器宽度；0 = 2×width（正式 = 1792）
    mid_depth: int = 6
    mid_heads: int = 16
    mid_ratio: float = 4.0
    mid_cls: bool = False      # v2enc：拼 1 个可学 cls，输出广播加到所有 rate 行
    mid_skip: bool = False     # v2enc：rate 行额外加 E 行残差（非 v2 口径的恒等保险，默认关）
    # ── 读出（整片段循环）
    dec_dim: int = 0           # 0 = 自动取 width（896）
    dec_depth: int = 4
    dec_heads: int = 16
    dec_ratio: float = 2.0
    dec_drop: float = 0.0
    step_gate: bool = False    # γ 零初始化门控（默认关 = w≡1，同 v2 的 query_base + Y）
    readout_mask: bool = False # 读出 self-attn 是否用帧级因果 mask（默认关 = 全双向，同 v2）
    readout_copy: bool = False # 已读帧 rate 行直抄残差（copy_skip；默认关）
    # loss 侧无开关：每步 pred 对全部帧 latent L1 平权均值（v2 直接预测口径）


    @property
    def n_steps(self) -> int:
        assert self.frames % self.group == 0, (self.frames, self.group)
        return self.frames // self.group

    @property
    def rate_tokens_per_step(self) -> int:
        """每个 z 步的 rate token 数（= rate_tokens × group）。"""
        return self.rate_tokens * self.group

    # ── 宽度口径：宽度由 chunk_tokens 决定，位置数由它除出来
    @property
    def width(self) -> int:
        """一个可训练向量的宽度 = chunk_tokens × rate_dim（默认 2×448 = 896）。

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


# ═══════════════════════════════════════════════════════════ v2 同构编码器中间态
class V2EncoderMiddle(nn.Module):
    """(B,F,Rr,width) → 同形状：**v2 编码器同构**的 rate 变换（2026-11-20 定稿）。

    口径（用户 2026-11-20 指示，对齐 v2 的 `SRPhase1V2` 编码器）：
      · 冻结 E 的输出 120×16×448（每帧 16 个 special，448 宽）先纯 reshape 成
        480×1792（1792 = 4 token × 448 = 2 个打包行 896），**不做投影**；
      · 再拼一份与输入等大的**可学 query**（480×1792；可选加 1 个 cls token）；
      · 960 个 token 走 `nn.TransformerEncoder`：全双向自注意力，**无 mask**
        （v2 的编码器是在整帧 token 序列上做全局自注意力，不是因果的）；
      · 取 **query 半**的输出（480×1792）reshape 回 (B,120,8,896) 当 rate，
        读出侧（`WholeClipDecoder`）一行不改。

    与 v2 的差别必须记住：v2 的 E+R 是**联合训练**的，rate 空间怎么方便由 R
    反向决定；本模型 R 冻结且只用于评测渲染，middle 是在**冻结 E 的现有码空间**
    上再学一层重采样 —— 恒等/近恒等映射得靠训练自己学出来（不像 v2 天然
    同构）。`mid_skip=True` 给一条 E 行残差当保险，但那不是 v2 口径。

    位置信息：reshape 后帧号与块号都被打散，必须显式加回。输入行用可学
    `in_pos`（480×1792）编码 (帧, 块)，query 本身就是逐位置可学参数
    （等价 v2 的 special token bank = 共享 token + 位置嵌入）。

    代价：中间态是全片的全局混合 ⇒ **不再是流式/帧级因果的**；逐档读出的语义
    从"第 t 步只含前 4(t+1) 帧"变成 v2 那种"读多少行 token 的质量"。
    """

    def __init__(self, dim: int, depth: int, heads: int, ratio: float,
                 frames: int, rows_rate: int, width: int,
                 drop: float = 0.0, use_cls: bool = False, skip: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.frames = int(frames)
        self.rows_rate = int(rows_rate)
        self.width = int(width)
        n_in = (self.frames * self.rows_rate * self.width) // self.dim
        assert n_in * self.dim == self.frames * self.rows_rate * self.width, (
            f"mid_hidden={self.dim} 不能整除 rate 值总数 "
            f"{self.frames * self.rows_rate * self.width}")
        self.n_in = int(n_in)
        assert self.dim % int(heads) == 0, (self.dim, heads)
        self.in_pos = nn.Parameter(torch.randn(1, self.n_in, self.dim) * 0.02)
        self.q = nn.Parameter(torch.randn(1, self.n_in, self.dim) * 0.02)
        self.cls = (nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
                    if use_cls else None)
        self.skip = bool(skip)
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=int(heads), dim_feedforward=int(self.dim * ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        # 末层 norm 交给 TransformerEncoder（norm_first 的推荐用法）
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(depth), norm=nn.LayerNorm(self.dim),
            enable_nested_tensor=False)

    def forward(self, packed: Tensor) -> Tensor:
        """packed (B,F,Rr,width) → 同形状。内部纯 reshape，不引入投影。"""
        B = packed.shape[0]
        assert packed.shape[1:] == (self.frames, self.rows_rate, self.width), (
            tuple(packed.shape), (self.frames, self.rows_rate, self.width))
        x = packed.reshape(B, self.n_in, self.dim)          # 480×1792
        seq = torch.cat([x + self.in_pos, self.q.expand(B, -1, -1)], dim=1)
        if self.cls is not None:
            seq = torch.cat([seq, self.cls.expand(B, -1, -1)], dim=1)
        h = self.encoder(seq)
        out = h[:, self.n_in:2 * self.n_in]                 # 取 query 半
        if self.cls is not None:                            # 全局摘要广播到所有行
            out = out + h[:, -1:]
        if self.skip:                                       # 非 v2 口径的恒等保险
            out = out + x
        return out.reshape(B, self.frames, self.rows_rate, self.width)


# ═══════════════════════════════════════════════════════════ 交叉查询中间态
# ═══════════════════════════════════════════════════════════ 恒等中间态（对照）
class IdentityMiddle(nn.Module):
    """no-op：rate = E 的打包行，**零参数**（2026-11-20 判别实验的对照）。

    用途：把"middle 完全不动"这个下界跑出来 —— 在 rate==target 的 v10 设定里，
    它配合 `readout_copy=True` 就是"读到的帧直抄"的极限（上界 ≈ 24.76 dB，
    缺 z_cls）。它回答：塌缩到底来自 middle 的重编码，还是来自读出。
    """

    def forward(self, packed: Tensor) -> Tensor:
        return packed


class CrossQueryMiddle(nn.Module):
    """(B,F,T_f,d) → (B,F,T_f,d)：可学 query 对编码器输出做**全双向**重采样。

    为什么（2026-11-18 用户提出）：v2 的 readout memory 是 `[z_cls; z_s]`——这些
    token 都来自**全局双向**的 DINO（任意 token 都"看过"整张图，z_cls 更是全局
    摘要），所以 v2 的 decoder 每一步都拿得到"整图摘要 + 已读 token"。v10 的
    middle 却是**帧级因果**的 TransformerEncoder：第 t 步的 z 只含前 4(t+1) 帧，
    未读帧的信息根本不在。本模块把 middle 换成「E 输出做 K/V + 可学 query」的
    交叉注意力（**无 mask、全双向**）⇒ 每个 rate 行都是全片范围内的摘要，
    早期步也能拿到全片信息（等价于 v2 里 z_cls 的作用）。

    代价（必须记住）：rate 不再是「第 t 组只含前 4(t+1) 帧」的**渐进码流**；
    逐档曲线语义变成 v2 那种「读多少 token 的质量」，不能再解释为"读前 t 组帧"。

    结构：Q 随机初始化（1, F·T_f, d，行↔(帧,行)），KV = E 打包行 + 可学位置嵌入，
    `nn.TransformerDecoder`（self-attn 在 query 之间 + cross-attn 到 E）。
    """

    def __init__(self, dim: int, depth: int, heads: int, ratio: float,
                 frames: int, tokens_per_frame: int, drop: float = 0.0):
        super().__init__()
        self.frames = int(frames)
        self.tpf = int(tokens_per_frame)
        L = self.frames * self.tpf
        self.Q = nn.Parameter(torch.randn(1, L, dim) * 0.02)
        self.kv_pos = nn.Parameter(torch.randn(1, L, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        # 末层 norm 交给 TransformerDecoder（norm_first 的推荐用法）：
        # 输出已是 LN(Y_raw)，forward 里不用再写一遍。
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=int(depth), norm=nn.LayerNorm(dim))
        # 输出头：残差 MLP —— out = Y + MLP(Y)（Y 已被 decoder 末层 norm 归一化），
        # 与读出侧同构；残差保证 rate 行的尺度/语义锚在 decoder 摘要上。
        self.mlp_out = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, z_in: Tensor) -> Tensor:
        B, Fr, Tf, _ = z_in.shape
        assert Fr == self.frames and Tf == self.tpf, (z_in.shape, self.frames, self.tpf)
        mem = z_in.reshape(B, Fr * Tf, -1) + self.kv_pos
        Q = self.Q.expand(B, -1, -1)
        # 无 mask：960 个 query 全双向 attend 全片 960 行 ⇒ 每行都是全片摘要
        Y = self.decoder(tgt=Q, memory=mem)
        Y = Y + self.mlp_out(Y)
        return Y.view(B, Fr, Tf, -1)


# ═══════════════════════════════════════════════════════════ 整片段循环读出
class WholeClipDecoder(nn.Module):
    """(frames+1) 行 × rows_pred 列的 896 向量 + 30 步循环：每步读一组 z、重出整段。

    每步结构：
        h      = Q + γ_t·Y_{t-1} + step_emb[t]       carry 同 v2：无 LN、默认 w≡1
        Y      = TransformerDecoder(h, z_t)          全双向 self-attn（无 tgt_mask）+ cross-attn
        pred_t = Y + MLP(Y) 拆成 8 个 chunk（32 个 special）+ 1 行 z_cls

    位置数 = 121 × 9 = 1089：120 帧 + 1 个随机初始化起始槽，每帧 9 行 896
    （8 行装 32 个 special，1 行装 z_cls —— 冻结 R 出像素必须要它）。
    self-attn 跨帧因果（帧 j 只看 0..j），所以整条链是 streaming 的，不需要
    任何帧内 mask。

    零冗余的三处：
      * 不建 kv_proj —— z 本身就是 896，再投影一遍只是把同一向量重新线性
        混合一次；
      * 不建 special 的 head —— 8 行 896 直接 reshape 成 16×448，只有
        z_cls 那一行需要 `cls_head`（896→448）；
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
        # 直抄残差（copy_skip）：已读帧的 rate 行原样加回输出行；要求行按帧
        # 连续排布（rate 行 = 帧主序），所以 P = group × rows_rate。
        self.Rr = int(cfg.rows_rate)
        self.copy = bool(getattr(cfg, "readout_copy", False))
        assert int(cfg.group) * self.Rr == self.P, (cfg.group, self.Rr, self.P)
        L = (self.F + 1) * self.R                          # 1089
        # 查询与状态（整片段一条序列）
        self.Q = nn.Parameter(torch.randn(1, L, self.d) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(1, self.S, 1, self.d) * 0.02)
        self.kv_pos = nn.Parameter(torch.randn(1, self.P, self.d) * 0.02)
        self.gate = (nn.Parameter(torch.zeros(self.S)) if cfg.step_gate else None)
        # carry 对齐 v2（946226d）：裸加，h = Q + γ·Y + step_emb，无 LN；
        # Y 已被下方 decoder 的末层 norm 归一化，carry 状态有界。
        layer = nn.TransformerDecoderLayer(
            d_model=self.d, nhead=int(cfg.dec_heads),
            dim_feedforward=int(self.d * cfg.dec_ratio), dropout=cfg.dec_drop,
            activation="gelu", batch_first=True, norm_first=True)
        # 末层 norm 交给 TransformerDecoder（norm_first 的推荐用法）：每步输出
        # 已是 LN(Y_raw)，循环里不用再写一遍。副作用：carry 递推用的 Y 也是
        # 归一化后的（等价于 carry 处自带 LayerNorm，状态有界，利于 30 步 BPTT）。
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=int(cfg.dec_depth), norm=nn.LayerNorm(self.d))
        # 输出头：残差 MLP —— out = Y + MLP(Y)（Y 已被 decoder 末层 norm 归一化），
        # 保留恒等通路（替换掉原来的裸 LayerNorm）。
        self.mlp_out = nn.Sequential(
            nn.Linear(self.d, 4 * self.d),
            nn.GELU(),
            nn.Linear(4 * self.d, self.d),
        )
        self.cls_head = nn.Linear(self.d, self.code_dim)    # 只有 z_cls 需要投影
        # 读出 self-attn：默认**全双向、无 tgt_mask**（同 v2 的 OutputQueryDecoder，
        # 946226d 口径）；streaming 由「第 t 步 memory 只有前 t+1 组 z」保证，
        # 不靠 self-attn mask。cfg.readout_mask=True 时才加旧版帧级因果 mask。
        if cfg.readout_mask:
            fi = torch.arange(L) // self.R                  # 位置 → 帧号（0=起始槽）
            allowed = fi[None, :] <= fi[:, None]            # True = 可看
            mask = torch.zeros(L, L, dtype=torch.float32).masked_fill(
                ~allowed, float("-inf"))
            self.register_buffer("_mask", mask, persistent=False)
        else:
            self._mask = None
        # 读出窗口（v2 的 `self.windows` 口径）：阶段 2 = 第 t 步只读当步 P 行；
        # 阶段 1（v2 A 相 `decoder.steps=[K]`）= 1 步读完整段 S×P 行。
        self.windows = [(t * self.P, (t + 1) * self.P) for t in range(self.S)]
        self.full_windows = [(0, self.S * self.P)]

    def _split(self, Y: Tensor, resid: Optional[Tensor] = None
               ) -> Tuple[Tensor, Tensor]:
        """(B,L,d) → specials (B,F,K,c) / z_cls (B,F,1,c)。

        `resid` (B,F,R,d)：已读帧的 rate 行直抄残差（copy_skip）。行 r 原样加回，
        未读帧为 0 ⇒ 已读帧输出 = 模型预测 + rate 行，梯度可直达 middle 输出
        （把 rate 逼成 ≈ 目标），不需要读出"学会抄"。
        """
        B = Y.shape[0]
        Y = Y.view(B, self.F + 1, self.R, self.d)[:, 1:]    # 丢掉随机起始槽
        if resid is not None:
            Y = Y + resid
        spec = Y[:, :, :self.R - 1].reshape(B, self.F, self.K, self.code_dim)
        cls = self.cls_head(Y[:, :, self.R - 1]).unsqueeze(2)
        return spec, cls

    def forward(self, z: Tensor, n_steps: Optional[int] = None,
                return_step_preds: bool = False,
                windows: Optional[List[Tuple[int, int]]] = None,
                ) -> Dict[str, Tensor]:
        """z (B,S,P,d) → dict(pred (B,F,1+K,c), feats, preds 可选 (B,S',F,1+K,c))。

        与 v2 的 `OutputQueryDecoder` 同构：位置只加一次、循环体无分支，
        `windows` 决定第 t 步读哪段行（v2 的 `mem[:, lo:hi]`）。
        · 默认 `self.windows`：第 t 步只读当步 P 行（阶段 2 / 逐档评测）；
        · `self.full_windows`：1 步读完整段 S×P 行（阶段 1 = v2 A 相）。
        `n_steps` 只控制默认窗口的档数（评测逐档曲线用）；给 `windows` 时
        步数由它决定，二者互斥。
        """
        B, S, P, _ = z.shape
        assert P == self.P, (P, self.P)
        if windows is None:
            assert n_steps is None or 1 <= int(n_steps) <= S, (n_steps, S)
            ws = (self.windows if n_steps is None
                  else self.windows[:int(n_steps)])
        else:
            assert n_steps is None, "windows 与 n_steps 互斥"
            ws = list(windows)
        assert 1 <= len(ws) <= S, (len(ws), S)
        # 加性 mask 必须与注意力权重同 dtype：bf16 下 muDNN/MUSA 的 SDPA 拒绝
        # "BF16 + FLOAT" 的加法（实测报 NOT_SUPPORTED in RunMath）。
        mask = None if self._mask is None else self._mask.to(dtype=z.dtype)
        # 位置只加一次（v2 的 `mem = proj([z_cls; z_s]) + mem_pos` 口径）：
        # 行位置 kv_pos + z 步位置 step_emb；窗口只做切片。
        mem = (z + self.kv_pos.reshape(1, 1, self.P, self.d)
               + self.step_emb.reshape(1, S, 1, self.d)).reshape(B, S * P, self.d)
        # 直抄残差的源：z 按 (帧, rate 行) 重排；只有前缀窗口（行按帧主序）
        zr = (z.reshape(B, self.F, self.Rr, self.d) if self.copy else None)
        Q = self.Q.expand(B, -1, -1)                         # 随机初始化、可学习
        Y = torch.zeros_like(Q)                              # 状态初值 0（第 0 步只看 Q）
        specs: List[Tensor] = []
        clss: List[Tensor] = []
        for t, (lo, hi) in enumerate(ws):
            w = 1.0 if self.gate is None else self.gate[t]   # γ 零初始化 ⇒ 起点=单步
            carry = Q + w * Y                                # = v2 的 query_base + Y
            h = carry + self.step_emb[:, t]
            Y = self.decoder(tgt=h, memory=mem[:, lo:hi], tgt_mask=mask)
            resid = None
            if zr is not None:
                nf = min(self.F, int(hi) // self.Rr)         # 已读帧数
                resid = z.new_zeros(B, self.F, self.R, self.d)
                resid[:, :nf, :self.Rr] = zr[:, :nf]
            spec, cls = self._split(Y + self.mlp_out(Y), resid)
            specs.append(spec)
            clss.append(cls)
        out = dict(pred=torch.cat([clss[-1], specs[-1]], dim=2),
                   feats=Y.view(B, self.F + 1, self.R, self.d)[:, 1:],
                   n_steps=len(ws))
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
        # ── 宽度：chunk_tokens 个 token 拼成一个向量（默认 2×448 = 896）。
        # dec_dim 只接受 0（自动）或等于 width，杜绝"读出再投影到别的宽度"。
        cfg.mid_dim = int(cfg.mid_dim or cfg.width)
        cfg.dec_dim = int(cfg.dec_dim or cfg.width)
        assert cfg.dec_dim == cfg.width, (
            f"dec_dim={cfg.dec_dim} 必须等于 chunk_tokens×rate_dim={cfg.width}；"
            "换宽度只会引入填充冗余")
        style = str(cfg.mid_style).lower()
        if style == "v2enc":
            # v2 同构编码器：编码器宽度 mid_hidden 默认 = 2×width（正式 1792），
            # 与读出宽度解耦（输入/输出仍 reshape 回 (B,F,rows_rate,width)）。
            hidden = int(cfg.mid_hidden or 2 * cfg.width)
            self.middle = V2EncoderMiddle(
                dim=hidden, depth=cfg.mid_depth, heads=cfg.mid_heads,
                ratio=cfg.mid_ratio, frames=cfg.frames,
                rows_rate=cfg.rows_rate, width=cfg.width,
                use_cls=bool(cfg.mid_cls), skip=bool(cfg.mid_skip))
            cfg.mid_hidden = int(self.middle.dim)
        elif style == "crossq":
            assert cfg.mid_dim == cfg.width, (
                f"crossq 的 mid_dim={cfg.mid_dim} 必须等于 width={cfg.width}")
            self.middle = CrossQueryMiddle(
                dim=cfg.width, depth=cfg.mid_depth, heads=cfg.mid_heads,
                ratio=cfg.mid_ratio, frames=cfg.frames,
                tokens_per_frame=cfg.rows_rate)
            cfg.mid_style = "crossq"
        elif style == "identity":
            self.middle = IdentityMiddle()                   # 零参数对照（判别用）
            cfg.mid_style = "identity"
        else:
            raise ValueError(
                f"未知 mid_style={cfg.mid_style!r}（支持 v2enc/crossq/identity）")
        self.readout = WholeClipDecoder(cfg, code_dim=code, num_specials=K)
        # 两阶段训练：阶段 1（单步全读，v2 A 相）由 trainer 把窗口换成
        # `readout.full_windows`，训练态 forward 就走「1 步 + 整段 memory」。
        self.train_windows: Optional[List[Tuple[int, int]]] = None

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

    # ── 前向 + 损失（v2 口径：每步输出都监督全部帧，逐步平权，无开关）
    def forward(self, x: Optional[Tensor] = None, n_steps: Optional[int] = None,
                return_pix: bool = False,
                z_cls: Optional[Tensor] = None,
                z_s: Optional[Tensor] = None,
                windows: Optional[List[Tuple[int, int]]] = None,
                ) -> Dict[str, Tensor]:
        """x (B,F,3,H,W) ∈[0,1]，或已缓存的 (z_cls (B,F,1,c), z_s (B,F,K,c))。

        → dict(loss, pred, l1, step_l1, z, target)；像素 PSNR 只在给了 x 时算。
        `n_steps` 只用于评测逐档曲线（渐进读出）；训练恒走全 S 步。
        损失 = 每步 pred 对全部帧 latent L1 的平权均值（v2 的直接预测口径）。

        `windows=None` 时由训练态属性 `self.train_windows` 决定（两阶段训练
        的阶段 1 = `readout.full_windows`：一步读完整段 z、一步出全片）。
        """
        if windows is None and self.training:
            windows = getattr(self, "train_windows", None)
        if x is not None:
            enc = self.encode_rate(x)
        else:
            assert z_cls is not None and z_s is not None, "须给 x，或 (z_cls, z_s)"
            enc = self.encode_rate_latents(z_cls, z_s)
        target = self.target_latents(enc["z_cls"], enc["z_s"])   # (B,F,1+T,c) 冻结
        out = self.readout(enc["z"], n_steps=n_steps, return_step_preds=True,
                           windows=windows)
        preds = out["preds"]                                     # (B,S',F,T,c)
        # 与 v2 同口径：reduction="none" 保留步维，再按帧/token/通道求均值 ⇒ (B,S')
        per_step = F.l1_loss(preds, target.unsqueeze(1),
                             reduction="none").mean(dim=(2, 3, 4))
        res = dict(loss=per_step.mean(), l1=per_step.detach().mean(),
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
                 pixel_psnr: bool = True,
                 windows: Optional[List[Tuple[int, int]]] = None,
                 ) -> Dict[str, float]:
        res = self.forward(x, n_steps=n_steps, return_pix=False,
                           windows=windows)
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
    # 自检用 chunk_tokens=4 ⇒ 宽度 = 4×8 = 32（正式配置是 2×448 = 896）
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

    # ── middle 口径：默认 v2enc（960 token 拼接自注意力），输出形状与输入一致
    assert isinstance(m.middle, V2EncoderMiddle), type(m.middle)
    assert m.middle.dim == 2 * cfg.width == 64, m.middle.dim
    assert (m.middle.n_in == cfg.frames * cfg.rows_rate * cfg.width // m.middle.dim
            == 4), m.middle.n_in
    print("[middle]", type(m.middle).__name__, "dim", m.middle.dim,
          "n_in", m.middle.n_in, "seq", 2 * m.middle.n_in)
    with torch.no_grad():
        pk = torch.randn(2, cfg.frames, cfg.rows_rate, cfg.width)
        mo = m.middle(pk)
    assert mo.shape == pk.shape, (mo.shape, pk.shape)
    # v2enc 变体：+cls、+skip、crossq 对照，都必须保持输入/输出同形
    for extra in ({"mid_cls": True}, {"mid_skip": True}):
        mc = VideoV10(V10Config(**{**asdict(cfg), **extra})).eval()
        with torch.no_grad():
            assert mc.middle(pk).shape == pk.shape, extra
    # identity middle + readout_copy（判别对照）：零参数、输出=输入；copy 后
    # 已读帧输出行 = 模型预测 + rate 行，形状与梯度都要通
    cfg_id = V10Config(**{**asdict(cfg), "mid_style": "identity",
                          "readout_copy": True})
    mi = VideoV10(cfg_id).eval()
    assert isinstance(mi.middle, IdentityMiddle)
    assert mi.readout.copy and mi.readout.Rr == cfg.rows_rate
    with torch.no_grad():
        assert mi.middle(pk) is pk
        z0 = torch.randn(2, cfg.n_steps, cfg.chunks_per_step, cfg.width)
        r0 = mi.readout(z0, windows=mi.readout.full_windows)
        assert r0["pred"].shape == (2, cfg.frames, 1 + cfg.target_tokens, cfg.rate_dim)
    cfg_cq = V10Config(**{**asdict(cfg), "mid_style": "crossq"})
    mcq = VideoV10(cfg_cq).eval()
    assert isinstance(mcq.middle, CrossQueryMiddle)
    with torch.no_grad():
        assert mcq.middle(pk).shape == pk.shape
    print("[middle] v2enc(+cls/+skip) 与 crossq 对照均同形 ✓")

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

    # 默认 = v2 对齐口径：carry 无 LN、无 γ 门控、读出 self-attn 全双向无 mask
    assert not hasattr(m.readout, "n_in"), "carry_ln 已移除，不该再有 n_in"
    assert m.readout._mask is None, "默认读出不该有 tgt_mask（v2 口径）"
    assert m.readout.gate is None, "默认不该建 γ 门控（w≡1，= v2 的 query_base + Y）"
    print("[v2-aligned] carry 无 LN、无 γ、读出全双向（无 tgt_mask）✓")

    # 显式打开 step_gate 时，γ 零初始化 ⇒ 第 0 步的读出与 Y 无关（起点=单步读出）
    cfg_g = V10Config(**{**asdict(cfg), "step_gate": True})
    rd = WholeClipDecoder(cfg_g, code_dim=cfg.rate_dim, num_specials=cfg.target_tokens)
    assert rd.gate is not None and float(rd.gate.abs().max()) == 0.0
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
