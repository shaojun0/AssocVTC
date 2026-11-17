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
`nn.TransformerDecoderLayer`；本文件只保留模型编排（打包、循环读出），
不再手写 QKV 投影和两层 MLP。

- middle：`nn.TransformerEncoder`。每帧 4 个位置（不是 33 个 token），帧级
  因果 mask。等宽 ⇒ **不建 in_proj**。
- readout：`nn.TransformerDecoder`。121 帧 × 9 行 = 1089 个位置，**全双向、
  无 tgt_mask**（同 v2）。等宽 ⇒ **不建 kv_proj**；special 侧**也不建 head**
  （8 行 1792 直接 reshape 成 32×448），只有 z_cls 那一行需要 `cls_head`（1792→448）。
- 没有 compressor 这一级：middle 的输出本身就是码率（120×4×1792 与
  30×16×1792 值数相同），再加一级池化只是把同一堆数重新混合一遍。

和 v8/v9 的关系
--------------
v8/v9 是「逐帧读出 + 帧间状态」；v10 反过来：**每一步都把整段 120 帧重出一次**，
每步多读一组 z（4 帧 = 0.5 s），读出逐步细化。整条链严格 streaming，但**保证它
的是 memory 不是 mask**：第 t 档的 cross-attn 只喂 z[:t+1]，等价于论文里
「读前 κ 个 token」；self-attn 内部各帧全双向互通。

数据流（数字口径：448² / patch 32 / N=196 / K=32 / zs_dim=448）
--------------------------------------------------------------
::

    x            (B, 120, 3, 448, 448)                  视频片段 15 s @ 8 fps
    E（冻结）    每帧独立前向 → z_cls (B,120,1,448), z_s (B,120,32,448)
    rate         前缀截断前 16 个 special → reshape (B,120,4,1792)（纯 reshape）
    middle       480 个位置，帧级因果 + 6 层 Transformer → (B,120,4,1792)
                 reshape → z (B,30,16,1792)（4 帧 × 4 行 = 每步 16 行）
    readout      Q（可学习，随机初始化，= v2 的 query_base）(1,1089,1792)
                 for t in 0..29:
                     h    = Q + Y_{t-1} + step_emb[t]     # 等权 carry（同 v2）
                     Y_t  = Decoder(h, z_t)     全双向 self-attn + cross-attn(16×1792)
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
step_gate       是否用零初始化门控 γ（默认关 = w≡1，carry 与 v2 一样等权相加）
step_weights    逐步损失权重（"mean" 平权 | "lin" 线性加权到末步）

HF 口径（AutoModel + trust_remote_code）
----------------------------------------
`V10Config` 是 `PretrainedConfig`、`VideoV10` 是 `PreTrainedModel`，两者都
`register_for_auto_class`，`config.auto_map` 指向本文件，所以 checkpoint
是**自描述**的（`config.json` + `model.safetensors` + 代码）：

    m.save_pretrained(out)                 # 写 config.json / 权重 / 本文件+依赖
    m = AutoModel.from_pretrained(out, trust_remote_code=True)

`save_pretrained` 会把 `model_v10.py` 和它相对导入的 `model_v2_linattn.py`
一起拷进 out/，所以在没有本仓库源码的机器上也能加载；`tok_cfg` 里存了分词器
的完整结构，`tok_ckpt` 路径丢了也不影响重建（分词器权重本身就在权重文件里）。
`model_v10.py` 顶部对 `model_v2_linattn` 做了「相对导入 + sys.path 兜底」，
是为了让动态模块加载（文件被拷到 HF 缓存目录后独立执行）也能找到依赖。
`VideoV10.from_pretrained` 额外重新冻结分词器 —— transformers 的加载路径会把
所有浮点参数包成 requires_grad=True。

自检：``python model_v10.py``（CPU 可跑，不需要 ckpt；含一次 save→AutoModel 往返）
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel

# 分词器实现与本文件同级。两种加载路径都要能走通：
#   * 包内（`import pkg.model_v10`）/ transformers 的动态模块：相对导入；
#     顺带让 `custom_object_save` 把 model_v2_linattn.py 一起拷进 checkpoint。
#   * 平铺目录（`python model_v10.py`、以及文件被拷到 HF 缓存后独立执行）：
#     此时没有 package，相对导入会报 "no known parent package"，退回绝对导入，
#     并把本文件所在目录放进 sys.path（动态模块的目录不在默认搜索路径里）。
try:
    from .model_v2_linattn import LinAttnConfig, SRPhase1LinAttn
except ImportError as _e:                      # noqa: PERF203
    if "attempted relative import" not in str(_e) and "parent package" not in str(_e):
        raise                                  # 真的是依赖坏了，别吞掉
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
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
class V10Config(PretrainedConfig):
    """v10 的超参，同时是 `transformers` 的 `PretrainedConfig`。

    继承 `PretrainedConfig` 只为了一个目的：让 checkpoint 自描述 ——
    `save_pretrained` 会把 `to_dict()` 写进 config.json，`AutoConfig /
    AutoModel.from_pretrained(dir, trust_remote_code=True)` 据此重建。

    `auto_map` 指到本文件；文件名改了要同步改这里（HF 按「模块名.类名」
    在 checkpoint 目录里找 .py）。计算属性（width / rows_* / n_steps）不进
    config.json —— 它们由下面的基本字段推出来，写进去反而会随口径漂移。
    """

    model_type = "v10"
    auto_map = {
        "AutoConfig": "model_v10.V10Config",
        "AutoModel": "model_v10.VideoV10",
    }

    def __init__(
        self,
        tok_ckpt: str = "",
        tok_cfg: Optional[Dict] = None,   # 分词器结构（VideoV10 建好后会写全量进来）
        frames: int = 120,                # 15 s @ 8 fps
        group: int = 4,                   # z 步覆盖帧数（30 步）
        rate_tokens: int = 16,            # 每帧保留的 special token 数（前缀截断）
        rate_dim: int = 448,              # 单个 token 宽度（= 分词器 zs_dim）
        target_tokens: int = 32,          # 目标 special token 数（全量）
        chunk_tokens: int = 4,            # 一个可训练向量 = 几个 token（4×448=1792）
        mid_dim: int = 0,                 # 0 = 自动取 width，不允许填成别的宽度
        mid_depth: int = 6,
        mid_heads: int = 16,
        mid_ratio: float = 4.0,
        dec_dim: int = 0,                 # 0 = 自动取 width
        dec_depth: int = 4,
        dec_heads: int = 16,
        dec_ratio: float = 2.0,
        dec_drop: float = 0.0,
        step_gate: bool = False,          # γ 门控（默认关：w≡1，= v2 的 query_base + Y）
        step_weights: str = "mean",
        **kwargs,
    ):
        # config.json 里带的 auto_map / architectures / dtype / _name_or_path
        # 都走 kwargs 交给 PretrainedConfig；没带就补默认的 auto_map。
        kwargs.setdefault("auto_map", dict(self.auto_map))
        super().__init__(**kwargs)
        self.tok_ckpt = str(tok_ckpt or "")
        self.tok_cfg = dict(tok_cfg or {})
        self.frames = int(frames)
        self.group = int(group)
        self.rate_tokens = int(rate_tokens)
        self.rate_dim = int(rate_dim)
        self.target_tokens = int(target_tokens)
        self.chunk_tokens = int(chunk_tokens)
        self.mid_dim = int(mid_dim)
        self.mid_depth = int(mid_depth)
        self.mid_heads = int(mid_heads)
        self.mid_ratio = float(mid_ratio)
        self.dec_dim = int(dec_dim)
        self.dec_depth = int(dec_depth)
        self.dec_heads = int(dec_heads)
        self.dec_ratio = float(dec_ratio)
        self.dec_drop = float(dec_drop)
        self.step_gate = bool(step_gate)
        self.step_weights = str(step_weights)

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

    归一化全部走框架自带的槽：**出口**用 `nn.TransformerEncoder(norm=LayerNorm)`
    （逐位等价地替代了手写的 `norm_out`）；**入口**不建任何 LN —— 上游 z_s 已是
    分词器末尾 LayerNorm 的输出，层内 `norm1` 又负责每层分支输入的归一化，
    手写的 `norm_in` 只剩下「把已归一化的张量再归一次」。
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
        # 入口不再单独归一化（2026-11-17 删除原 norm_in）：上游 z_s 是分词器
        # 「末尾 LayerNorm」之后的输出（model_v2_linattn.py:47，zs_dim=0 时无
        # 额外投影），本身已是 per-token 标准化；而 norm_first=True 的层内 norm1
        # 又会归一化每层分支的输入。再建一个 norm_in 只是把已归一化的张量重归
        # 一次（LN∘LN = LN，恒等仿射下实测差 3e-4%）。代价是失去"对分词器仿射
        # 漂移免疫"：若 tokenizer 的 γ/β 漂出单位尺度，残差主干会跟着漂
        # （实测 γ/β ~ N(1,0.3)/N(0,0.3) 时输出差 4.8%）——换 ckpt 时留意。
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        # 出口归一化走 encoder 自带的 `norm=` 槽（就是为这件事准备的），
        # 不再手写 `self.norm_out` + forward 里再调一次 —— 逐位等价（误差 0.0）。
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(depth), norm=nn.LayerNorm(dim),
            enable_nested_tensor=False)
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
        # 入口直接进 encoder：层内 norm1 负责归一化分支输入，出口由 encoder.norm
        # 归一化。加性 mask 与 x 同 dtype（bf16 下 muDNN 的 SDPA 拒绝 BF16+FLOAT）。
        x = self.encoder(z_in.reshape(B, Fr * Tf, -1),
                         mask=self._attn_mask.to(dtype=z_in.dtype))
        return x.view(B, Fr, Tf, -1)


# ═══════════════════════════════════════════════════════════ 整片段循环读出
class WholeClipDecoder(nn.Module):
    """(frames+1) 行 × rows_pred 列的 1792 向量 + 30 步循环：每步读一组 z、重出整段。

    每步结构（与 model_v2.OutputQueryDecoder 同口径）：
        h      = Q + w_t·Y_{t-1} + step_emb[t]       w≡1（等权 carry，= query_base + Y）
        Y      = TransformerDecoder(h, z_t)          全双向 self-attn + cross-attn
        pred_t = LN(Y) 拆成 8 个 chunk（32 个 special）+ 1 行 z_cls

    位置数 = 121 × 9 = 1089：120 帧 + 1 个随机初始化起始槽，每帧 9 行 1792
    （8 行装 32 个 special，1 行装 z_cls —— 冻结 R 出像素必须要它）。
    self-attn **全双向、无任何 tgt_mask**（与 v2 的 OutputQueryDecoder 一致）；
    streaming 语义由 memory 保证、不靠 mask —— 第 t 步的 memory 只有 z[:, t]
    的 16 行，未来档的 z 根本不在场。

    零冗余的四处：
      * 不建 kv_proj —— z 本身就是 1792，再投影一遍只是把同一向量重新线性
        混合一次；
      * 不建 special 的 head —— 8 行 1792 直接 reshape 成 32×448，只有
        z_cls 那一行需要 `cls_head`（1792→448）；
      * 不复制 memory —— 整段是一条 batch，cross-attn 直接对这一步的 16 个
        key，不需要 `repeat_interleave` 出 120 份副本；
      * carry 处不建 LN、不建门控 —— 归一化由每层的 `norm_first=True` 负责
        （`nn.TransformerDecoderLayer` 自带），再手写一层 LN 只是重复造轮子。
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
        self.kv_pos = nn.Parameter(torch.randn(1, self.P, self.d) * 0.02)
        self.gate = (nn.Parameter(torch.zeros(self.S)) if cfg.step_gate else None)
        layer = nn.TransformerDecoderLayer(
            d_model=self.d, nhead=int(cfg.dec_heads),
            dim_feedforward=int(self.d * cfg.dec_ratio), dropout=cfg.dec_drop,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=int(cfg.dec_depth), norm=None)
        self.norm_out = nn.LayerNorm(self.d)
        self.cls_head = nn.Linear(self.d, self.code_dim)    # 只有 z_cls 需要投影
        # 无 tgt_mask：1089 个位置全双向（与 v2 的 OutputQueryDecoder 一致）。
        # 旧版这里有块因果 mask（fi = idx//R, allowed = fi_j <= fi_i）；2026-11-17
        # 删除，理由是实测它并不守 rate 前缀（那是 memory 的职责，见 forward 注释），
        # 只额外禁止"同一步内跨帧状态互看"，而这与"整片段细化"的目的相冲突。

    def _split(self, Y: Tensor) -> Tuple[Tensor, Tensor]:
        """(B,L,d) → specials (B,F,K,c) / z_cls (B,F,1,c)。"""
        B = Y.shape[0]
        Y = Y.view(B, self.F + 1, self.R, self.d)[:, 1:]    # 丢掉随机起始槽
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
        kv_all = z + self.kv_pos                             # (B,S,P,d)，等宽投影
        Q = self.Q.expand(B, -1, -1)                         # = v2 的 query_base
        Y = torch.zeros_like(Q)                              # 状态初值 0（第 0 步只喂 Q）
        specs: List[Tensor] = []
        clss: List[Tensor] = []
        for t in range(n):
            # carry = v2 的 `query_base + Y`：等权相加，不缩放、不额外加 LN
            # （归一化由 TransformerDecoderLayer(norm_first=True) 的入口 LN 负责）；
            # step_gate=True 时才退回零初始化 γ（w_0 = 0 ⇒ 起点等价于单步读出）。
            w = 1.0 if self.gate is None else self.gate[t]
            h = Q + w * Y + self.step_emb[:, t]
            # 无 tgt_mask ⇒ 整段 1089 行全双向。streaming 由 memory 保证而不是
            # mask：第 t 步的 cross-attn 只有 z[:, t] 这 16 行，未来档的 z 不在场
            # （实测：扰 z[1] 时 step 0 输出 max|Δ| 恒为 0）。
            Y = self.decoder(tgt=h, memory=kv_all[:, t])
            spec, cls = self._split(self.norm_out(Y))
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
class VideoV10(PreTrainedModel):
    """冻结分词器 + 中间态 + 整片段循环读出。训练只更新中间态与读出侧。

    `PreTrainedModel` 的意义只在存取：`save_pretrained` 会连 `config.json`
    （含 `auto_map`）和本文件代码一起落盘，于是
    ``AutoModel.from_pretrained(dir, trust_remote_code=True)`` 能重建整个模型。
    训练语义没变，`cfg` 仍是 `V10Config`（= `self.config` 的别名）。
    """

    config_class = V10Config
    main_input_name = "x"
    # device_map / FSDP 切分时不要把一层 Transformer 拆开
    _no_split_modules = ["TransformerEncoderLayer", "TransformerDecoderLayer"]

    def __init__(self, cfg: V10Config):
        super().__init__(cfg)                       # PreTrainedModel: 存 self.config
        self.cfg = cfg                              # 别名，老的调用点都读 .cfg
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
        # 把分词器的**完整结构**写回 config：换台机器 `tok_ckpt` 路径没了也能
        # 按同样的形状重建（权重本身存在 checkpoint 里）。这是 auto_map 能
        # 在别处加载的前提。
        cfg.tok_cfg = asdict(self.tokenizer.cfg)
        self.middle = TemporalMiddle(
            in_dim=cfg.width, dim=cfg.width, depth=cfg.mid_depth,
            heads=cfg.mid_heads, ratio=cfg.mid_ratio,
            tokens_per_frame=cfg.rows_rate, frames=cfg.frames)
        self.readout = WholeClipDecoder(cfg, code_dim=code, num_specials=K)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """什么都不做：每个子模块在自己的 `__init__` 里已经精确定义了初始化
        （readout 的 γ 零初始化、Q 随机、LN 默认等）。走默认的
        `normal_(0, initializer_range)` 会把它们全部覆盖掉。
        """
        return None

    def freeze_tokenizer(self) -> None:
        """冻结 E/R（幂等）。`__init__` 里已经冻过一次，这里再暴露成方法是因为
        `from_pretrained` 会把每个浮点参数包成 `nn.Parameter(..., requires_grad=True)`
        （transformers 的 `_load_parameter_into_model`，走 setattr 而不是
        `load_state_dict`，所以 post-hook 也拦不到），加载完必须重新冻一次。
        """
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        self.tokenizer.eval()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.freeze_tokenizer()            # 抵消加载路径把 requires_grad 归 True
        return model

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
        packed = self.rate_latents(z_s)                      # (B,F,4,1792)
        mid = self.middle(packed)                            # (B,F,4,1792)
        B, Fr, R, d = mid.shape
        S = self.cfg.n_steps
        assert Fr // S * R == self.cfg.chunks_per_step, (Fr, S, R)
        # 4 帧 × 4 行 → 每步 16 行；纯 reshape（row-major），值数与 mid 相同
        z = mid.reshape(B, S, self.cfg.chunks_per_step, d)   # (B,30,16,1792)
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
        # HF Trainer 的 eval 走 forward；evaluate() 里设置 eval_pixel=True，
        # 这样 compute_metrics 能拿到 PSNR，而不必再写一套评估循环。
        pixel_eval = bool(return_pix or (
            not self.training and getattr(self, "eval_pixel", False)))
        if pixel_eval:
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


# 注册到 auto classes：`save_pretrained` 见到 `_auto_class` 就会把本文件（以及
# 它相对导入的 model_v2_linattn.py）拷进 checkpoint 目录，并把 auto_map 写进
# config.json —— 这就是 `AutoModel.from_pretrained(dir, trust_remote_code=True)`
# 的入口。缺了这两行，权重能存但加载时找不到类。
V10Config.register_for_auto_class("AutoConfig")
VideoV10.register_for_auto_class("AutoModel")


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

    # 因果：第 t 步的输出只能依赖 z[0..t]（扰动 z[k] 不影响更早的步）。这条
    # 性质由 memory 保证（第 t 步只喂 z[:, t]），与 self-attn 有没有 mask 无关。
    assert not hasattr(m.readout, "_mask"), "已改全双向，不该再有 tgt_mask"
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

    # carry 与 v2 同口径：默认 w≡1（不建 γ），且不加额外 LN（层内 norm_first 负责）
    assert m.readout.gate is None, "默认 step_gate=False ⇒ 不该建 γ"
    assert not hasattr(m.readout, "n_in"), "carry 处不该再有手写 LN"
    print("[carry] w≡1（无门控、无额外 LN，= v2 的 query_base + Y）✓")

    # 显式打开 step_gate 时，γ 零初始化 ⇒ 第 0 步的读出与 Y 无关（起点=单步读出）
    # PretrainedConfig 不是 dataclass（不能 asdict/vars），用 deepcopy 派生
    cfg_g = copy.deepcopy(cfg)
    cfg_g.step_gate = True
    rd = WholeClipDecoder(cfg_g, code_dim=cfg.rate_dim,
                          num_specials=cfg.target_tokens)
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

    # ── HF 口径往返：save_pretrained 落盘 config.json + 代码 + 权重，然后在
    #    **两个全新进程**里验证 AutoModel.from_pretrained(trust_remote_code=True)。
    #    分两个进程是必须的：动态模块按路径缓存，同进程里会直接命中 sys.modules，
    #    测不出"文件被拷到 HF 缓存后能不能独立执行"这件事（依赖导入就死在这）。
    import json as _json
    import subprocess
    import tempfile

    proj = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "v10_tiny")
        save_py = os.path.join(td, "do_save.py")
        with open(save_py, "w", encoding="utf-8") as f:
            f.write("import sys\n"
                    f"sys.path.insert(0, {proj!r})\n"
                    "import model_v10 as mv\n"
                    f"cfg = mv.V10Config(tok_ckpt='', tok_cfg={cfg.tok_cfg!r},\n"
                    "                    frames=8, group=4, rate_tokens=4, rate_dim=8,\n"
                    "                    target_tokens=4, chunk_tokens=4,\n"
                    "                    mid_depth=2, mid_heads=2, mid_ratio=2.0,\n"
                    "                    dec_depth=2, dec_heads=2, dec_ratio=2.0)\n"
                    f"mv.VideoV10(cfg).save_pretrained({ckpt!r})\n")
        subprocess.run([sys.executable, save_py], check=True)

        files = sorted(os.listdir(ckpt))
        assert "config.json" in files, files
        assert "model_v10.py" in files, files            # 模型代码被一起拷走
        assert "model_v2_linattn.py" in files, files     # 相对导入的依赖也是
        with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
            saved = _json.load(f)
        assert saved["auto_map"]["AutoModel"] == "model_v10.VideoV10", saved
        assert saved["tok_cfg"]["num_specials"] == 4, saved["tok_cfg"]

        load_py = os.path.join(td, "do_load.py")
        with open(load_py, "w", encoding="utf-8") as f:
            f.write("import json\n"
                    "from transformers import AutoConfig, AutoModel\n"
                    f"c = AutoConfig.from_pretrained({ckpt!r}, trust_remote_code=True)\n"
                    f"m = AutoModel.from_pretrained({ckpt!r}, trust_remote_code=True)\n"
                    "print('@@' + json.dumps(dict(\n"
                    "    cls=type(m).__name__, q=list(m.readout.Q.shape),\n"
                    "    n=sum(p.numel() for p in m.parameters()),\n"
                    "    frames=c.frames, tok=c.tok_cfg['num_specials'],\n"
                    "    frozen=all(not p.requires_grad for p in m.tokenizer.parameters()))))\n")
        # 子进程的工作目录是临时目录，项目目录不在 sys.path 里 —— 模型代码只能
        # 靠 checkpoint 里那份拷贝跑起来，这正是要验的隔离性。
        res = subprocess.run([sys.executable, load_py], check=True,
                             capture_output=True, text=True)
        info = _json.loads([ln for ln in res.stdout.splitlines()
                            if ln.startswith("@@")][-1][2:])
        assert info["cls"] == "VideoV10", info
        assert info["q"] == list(m.readout.Q.shape), info
        assert info["n"] == sum(p.numel() for p in m.parameters()), (info, "参数量")
        assert info["frames"] == cfg.frames and info["tok"] == 4, info
        assert info["frozen"], f"加载后分词器必须仍然冻结: {info}"
        print(f"[hf] save_pretrained → AutoModel(trust_remote_code=True) ✓ "
              f"{info['cls']} {info['n'] / 1e3:.0f}K 参数，代码自包含")

    print("ALL CHECKS PASSED")
