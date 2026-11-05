"""
vlm8b — AssocVTC 448²/p32/960M 阶段2 编码器 → 8 预算并行 → Qwen3.5-4B 全参微调（v1 骨架）
==========================================================================================

数据流（形状表见 README §2）::

    image (B,3,448,448)
      → AssocVTCEncoder（StandardAttentionEncoder, 13 层, 448²/32）
      → z_s (B,32,1744)
      → BudgetProjector:  LayerNorm(1744) → Linear(1744,640)          [逐 token]
      → reshape (B,8,4*640=2560) → LayerNorm(2560)
      → vis (B,8,2560)                                                  [与 LM hidden 同维]
      → 每图展开 V 行: 行 k = [vis[:, :k] ; text_embeds]  (right-pad 到 max_k+T)
      → Qwen3.5-4B 文本主干（只喂 inputs_embeds, 不喂 pixel_values, 不走 vision tower）
      → 只在 answer 位置取 hidden → lm_head → CE（fp32 分块）

关键裁决（`doc/2026-11-05/DESIGN_8budget_vlm_finetune.md` §4）:

  · **8 行批，不打包成一条序列**。Qwen3.5 是 hybrid 模型（24 层 gated-delta-rule
    线性注意力 + 8 层全注意力）；线性层收到 4D 自定义 mask 时
    `create_recurrent_attention_mask()` 返回 None ⇒ 打包会跨段泄露。
    8 行批只用 **2D right-padding mask**，每行是独立 batch 行，天然互不可见。
  · 每行只含 `vis[:, :k]` + **同一段文本**：视觉侧靠 **截断** 隔离（k 个 token
    就是 k 个，不存在"看不见的 token"），不靠 mask 遮 visual。
  · `labels` 除 answer 外全 -100；loss = V 行 CE 的（可加权）平均。

模块结构::

    1. 常量
    2. VLM8BudgetConfig
    3. BudgetRows / expand_budget_rows()   ← 纯张量, 可在 CPU 上单独自检
    4. BudgetProjector
    5. AssocVTCEncoder
    6. VLM8Budget（PreTrainedModel）
    7. 延迟导入辅助（Qwen / AssocVTC encoder —— 本机 transformers 4.57 无 Qwen3_5*）

依赖约定：Qwen 相关 import 全部在函数内部延迟执行，因此本机可以直接
`python -m py_compile model.py` / `import vlm8b.model`（不加载 4B 模型）。
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel


# ═══════════════════════════════════════════════════════════════
# 常量（形状契约集中在此；改这里必须同步 README 的表格）
# ═══════════════════════════════════════════════════════════════

#: 阶段2 编码器（B/ckpt_best）实测维数：enc_dim=1744, K=32, depth=13, heads=16。
DEFAULT_ENC_DIM: int = 1744
DEFAULT_ENC_DEPTH: int = 13
DEFAULT_ENC_HEADS: int = 16
DEFAULT_NUM_SPECIALS: int = 32          # K：z_s 的 token 数
DEFAULT_IMAGE_SIZE: int = 448
DEFAULT_PATCH_SIZE: int = 32

#: 投影头：逐 token 1744→640，再把 4 个 token 拼成 2560。
DEFAULT_PROJ_DIM: int = 640
DEFAULT_VISUAL_TOKENS: int = 8
DEFAULT_LM_HIDDEN: int = 2560           # Qwen3.5-4B text_config.hidden_size
TOKENS_PER_VISUAL: int = 4              # 32 / 8：每个 visual token 吃 4 个 z_s token

#: 初始化 / loss 细节。
PROJ_INIT_STD: float = 0.02             # 设计 §3：投影头 N(0, 0.02)
IGNORE_INDEX: int = -100
CE_CHUNK_SIZE: int = 64                 # 分块算 CE 的 answer 位置数（控制 logits 峰值显存）

#: 预算默认集合 = 用户指定的"直接 8 个预算"。
DEFAULT_BUDGET_IDS: Tuple[int, ...] = tuple(range(1, DEFAULT_VISUAL_TOKENS + 1))

#: 支持的 loss 权重模式。
LOSS_WEIGHTINGS: Tuple[str, ...] = ("equal", "inv-k")


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class VLM8BudgetConfig(PretrainedConfig):
    """`VLM8Budget` 的 HF 风格配置。

    只放 **结构/契约** 与加载路径，不放训练超参（lr/batch 等在 `train.py` CLI）。
    所有路径都写进 `config.json`，因此 `from_pretrained(<ckpt>)` 能找回 encoder/Qwen。
    """

    model_type = "vlm8_budget"

    def __init__(
        self,
        encoder_ckpt: str = "",
        qwen_dir: str = "",
        assoctvc_repo: str = "",
        enc_dim: int = DEFAULT_ENC_DIM,
        enc_depth: int = DEFAULT_ENC_DEPTH,
        enc_heads: int = DEFAULT_ENC_HEADS,
        enc_mlp_ratio: float = 4.0,
        enc_qk_norm: bool = False,
        num_specials: int = DEFAULT_NUM_SPECIALS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        patch_size: int = DEFAULT_PATCH_SIZE,
        proj_dim: int = DEFAULT_PROJ_DIM,
        visual_tokens: int = DEFAULT_VISUAL_TOKENS,
        lm_hidden_size: int = DEFAULT_LM_HIDDEN,
        budget_ids: Optional[Sequence[int]] = None,
        loss_weighting: str = "equal",
        freeze_encoder: bool = True,
        use_gradient_checkpointing: bool = True,
        load_dtype: str = "bfloat16",
        local_files_only: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        auto_load_modules: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_ckpt = str(encoder_ckpt)
        self.qwen_dir = str(qwen_dir)
        self.assoctvc_repo = str(assoctvc_repo)

        self.enc_dim = int(enc_dim)
        self.enc_depth = int(enc_depth)
        self.enc_heads = int(enc_heads)
        self.enc_mlp_ratio = float(enc_mlp_ratio)
        self.enc_qk_norm = bool(enc_qk_norm)
        self.num_specials = int(num_specials)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)

        self.proj_dim = int(proj_dim)
        self.visual_tokens = int(visual_tokens)
        self.lm_hidden_size = int(lm_hidden_size)

        self.budget_ids = self._normalise_budget_ids(budget_ids)
        self.loss_weighting = str(loss_weighting).lower()
        self.freeze_encoder = bool(freeze_encoder)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        # 注意：不要占用 `dtype` 这个名字 —— 它是 transformers 的保留字段（
        # save_pretrained 会把它改写成模型的参数 dtype），这里用 load_dtype。
        self.load_dtype = str(load_dtype)
        self.local_files_only = bool(local_files_only)
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        self.auto_load_modules = bool(auto_load_modules)

        self._validate()

    # ─── 派生量与校验 ─────────────────────────────────────────
    @property
    def tokens_per_visual(self) -> int:
        """每个 visual token 由几个 z_s token 拼成（32/8 = 4）。"""
        return self.num_specials // self.visual_tokens

    @property
    def visual_dim(self) -> int:
        """一个 visual token 的宽度（4×640 = 2560）。"""
        return self.tokens_per_visual * self.proj_dim

    def _normalise_budget_ids(self, budget_ids: Optional[Sequence[int]]) -> Tuple[int, ...]:
        """把预算集合规整成升序去重的 tuple；None ⇒ 1..visual_tokens。"""
        if budget_ids is None:
            return tuple(range(1, self.visual_tokens + 1))
        if isinstance(budget_ids, str):
            budget_ids = [int(x) for x in budget_ids.split(",") if x.strip()]
        ks = tuple(sorted({int(k) for k in budget_ids}))
        return ks

    def _validate(self) -> None:
        """结构契约失败时尽早报错（不要等到 remote 训练第 1 步）。"""
        if self.enc_dim <= 0 or self.enc_depth <= 0 or self.enc_heads <= 0:
            raise ValueError("enc_dim / enc_depth / enc_heads 必须为正数")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} 必须被 patch_size={self.patch_size} 整除")
        if self.num_specials <= 0 or self.visual_tokens <= 0:
            raise ValueError("num_specials / visual_tokens 必须为正数")
        if self.num_specials % self.visual_tokens != 0:
            raise ValueError(
                f"num_specials={self.num_specials} 必须被 "
                f"visual_tokens={self.visual_tokens} 整除")
        if self.tokens_per_visual * self.proj_dim != self.lm_hidden_size:
            raise ValueError(
                "投影后每个 visual token 的宽度必须等于 LM hidden："
                f"{self.tokens_per_visual}×{self.proj_dim} != {self.lm_hidden_size}")
        if not self.budget_ids:
            raise ValueError("budget_ids 不能为空")
        bad = [k for k in self.budget_ids if not 1 <= k <= self.visual_tokens]
        if bad:
            raise ValueError(
                f"budget_ids 越界 {bad}；合法范围 1..{self.visual_tokens}")
        if self.loss_weighting not in LOSS_WEIGHTINGS:
            raise ValueError(
                f"loss_weighting={self.loss_weighting!r} 非法；"
                f"合法值: {list(LOSS_WEIGHTINGS)}")
        if self.load_dtype.lower() not in (
                "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"):
            raise ValueError(
                f"load_dtype={self.load_dtype!r} 非法；"
                "可选 bfloat16 / float16 / float32")


# ═══════════════════════════════════════════════════════════════
# 8 预算行展开（纯张量；smoke.py 直接单测这一层）
# ═══════════════════════════════════════════════════════════════

@dataclass
class BudgetRows:
    """一张图 × V 个预算展开后的批（行顺序 = ``b * V + vi``）。

    Attributes:
        embeds:         (B*V, L, H) —— 行 k 的 `[vis[:, :k]; text]`，右侧补零到
                        ``L = max_k + T``。
        attention_mask: (B*V, L) —— visual 前缀与真实文本为 1，padding 为 0。
        labels:         (B*V, L) —— visual 前缀 / padding / prompt 全 -100。
        ans_pos:        (B*V, A) —— 每行 answer token 在行内的绝对位置；无效位 -1。
        ans_targets:    (B*V, A) —— answer token id；无效位 -100。
        budget_ks:      (V,) —— 本批实际的预算值（升序）。
        num_images:     B。
        num_budgets:    V。
    """

    embeds: Tensor
    attention_mask: Tensor
    labels: Tensor
    ans_pos: Tensor
    ans_targets: Tensor
    budget_ks: Tensor
    num_images: int
    num_budgets: int


def _answer_spans(labels: Tensor) -> Tuple[Tensor, Tensor]:
    """从 (B,T) labels 里解析 answer 段：返回 (start (B,), count (B,))。

    约定：answer 是 labels 里**唯一一段连续非 -100**的位置（训练数据里 prompt /
    文本 padding 都是 -100）。文本是 right-pad 的，所以 answer 不一定在序列末尾。
    """
    if labels.dim() != 2:
        raise ValueError(f"labels 应为 (B,T)，得到 {tuple(labels.shape)}")
    valid = labels != IGNORE_INDEX
    counts = valid.sum(dim=1)
    if int(counts.min()) <= 0:
        raise ValueError("有样本没有任何 answer token（labels 全 -100）")
    starts = torch.full_like(counts, -1)
    for b in range(labels.size(0)):
        idx = valid[b].nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            raise ValueError(f"样本 {b} 的 labels 全为 -100")
        if int(idx[-1] - idx[0] + 1) != int(idx.numel()):
            raise ValueError(
                f"样本 {b} 的 answer 不是连续段（prompt/answer 交错）；"
                "dataset 应把 prompt 全置 -100、answer 连续放在文本末尾之前")
        starts[b] = idx[0]
    return starts, counts


def expand_budget_rows(
    vis_embeds: Tensor,
    text_embeds: Tensor,
    text_mask: Tensor,
    labels: Tensor,
    budget_ks: Sequence[int],
) -> BudgetRows:
    """把 ``vis (B,8,H)`` + ``text (B,T,H)`` 展开成 ``V`` 行/图的独立批。

    行 k 的结构::

        [ vis[:, :k] ][ text (T) ][ zero-pad (max_k - k) ]
          mask=1        mask=text_mask      mask=0
          label=-100    label=labels        label=-100

    Args:
        vis_embeds: (B, V_max, H)，`BudgetProjector` 的输出。
        text_embeds: (B, T, H)，文本（prompt+answer）embedding。
        text_mask: (B, T)，真实文本 token 为 1、文本 right-pad 为 0。
        labels: (B, T)，仅 answer 位置非 -100。
        budget_ks: 预算集合（1..V_max，升序、无重复）。

    Returns:
        `BudgetRows`；行顺序 `b * len(budget_ks) + vi`。

    Raises:
        ValueError: 形状/预算集合/answer 段不合法。
    """
    if vis_embeds.dim() != 3 or text_embeds.dim() != 3:
        raise ValueError("vis_embeds / text_embeds 必须是 3 维")
    B, T, H = text_embeds.shape
    if vis_embeds.size(0) != B:
        raise ValueError(f"vis/text batch 不一致: {vis_embeds.size(0)} vs {B}")
    if vis_embeds.size(2) != H:
        raise ValueError(f"vis/text hidden 不一致: {vis_embeds.size(2)} vs {H}")
    if text_mask.shape != (B, T) or labels.shape != (B, T):
        raise ValueError(
            f"text_mask/labels 应为 {(B, T)}，得到 {tuple(text_mask.shape)} / "
            f"{tuple(labels.shape)}")

    ks = [int(k) for k in budget_ks]
    if not ks or len(set(ks)) != len(ks):
        raise ValueError(f"budget_ks 为空或有重复: {ks}")
    if min(ks) < 1 or max(ks) > vis_embeds.size(1):
        raise ValueError(
            f"budget_ks 越界 {ks}；合法范围 1..{vis_embeds.size(1)}")

    max_k = max(ks)
    V = len(ks)
    L = max_k + T
    device, dtype = text_embeds.device, text_embeds.dtype

    embeds = text_embeds.new_zeros((B, V, L, H))
    masks = text_mask.new_zeros((B, V, L))
    row_labels = labels.new_full((B, V, L), IGNORE_INDEX)

    for vi, k in enumerate(ks):
        # 视觉前缀：只取前 k 个 token —— 这就是防泄露的"截断"。
        embeds[:, vi, :k] = vis_embeds[:, :k]
        # 同一段文本：8 行逐 token 相同。
        embeds[:, vi, k:k + T] = text_embeds
        masks[:, vi, :k] = 1
        masks[:, vi, k:k + T] = text_mask
        row_labels[:, vi, k:k + T] = labels

    starts, counts = _answer_spans(labels)
    A_max = int(counts.max())
    j = torch.arange(A_max, device=device).unsqueeze(0)                 # (1,A)
    ans_pos_b = starts.unsqueeze(1) + j                                # (B,A)
    ans_valid = j < counts.unsqueeze(1)                                # (B,A)
    ans_targets_b = labels.gather(1, ans_pos_b.clamp(min=0, max=T - 1))
    ans_targets_b = ans_targets_b.masked_fill(~ans_valid, IGNORE_INDEX)

    # (B,A) → (B,V,A)：预算 k 的 answer 整体后移 k 个 visual 前缀位置。
    ans_pos = ans_pos_b.unsqueeze(1) + torch.tensor(ks, device=device).view(1, V, 1)
    ans_pos = ans_pos.masked_fill(~ans_valid.unsqueeze(1), -1)
    ans_targets = ans_targets_b.unsqueeze(1).expand(B, V, A_max)

    return BudgetRows(
        embeds=embeds.reshape(B * V, L, H),
        attention_mask=masks.reshape(B * V, L),
        labels=row_labels.reshape(B * V, L),
        ans_pos=ans_pos.reshape(B * V, A_max),
        ans_targets=ans_targets.reshape(B * V, A_max),
        budget_ks=torch.tensor(ks, device=device),
        num_images=B,
        num_budgets=V,
    )


# ═══════════════════════════════════════════════════════════════
# BudgetProjector — z_s (B,32,1744) → vis (B,8,2560)
# ═══════════════════════════════════════════════════════════════

class BudgetProjector(nn.Module):
    """逐 token ``LayerNorm(1744) → Linear(1744, 640)``，再 4 拼 1 + ``LayerNorm(2560)``。

    预算 k 用 ``vis[:, :k]``（= ``z_s[:4k]``），所以 8 个预算天然嵌套（Matryoshka
    前缀性质）。初始化 N(0, 0.02)（Linear 权重），LayerNorm 用默认 gamma=1/beta=0。
    """

    def __init__(self, config: VLM8BudgetConfig):
        super().__init__()
        self.num_specials = config.num_specials
        self.visual_tokens = config.visual_tokens
        self.tokens_per_visual = config.tokens_per_visual
        self.proj_dim = config.proj_dim
        self.visual_dim = config.visual_dim

        self.token_norm = nn.LayerNorm(config.enc_dim)
        self.token_proj = nn.Linear(config.enc_dim, config.proj_dim)
        self.budget_norm = nn.LayerNorm(config.visual_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Linear: N(0, 0.02) + zero bias；LayerNorm: gamma=1, beta=0。"""
        for module in (self.token_norm, self.budget_norm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.token_proj.weight, mean=0.0, std=PROJ_INIT_STD)
        nn.init.zeros_(self.token_proj.bias)

    def forward(self, z_s: Tensor) -> Tensor:
        """Args: z_s (B,32,1744). Returns: vis (B,8,2560)."""
        B, K, D = z_s.shape
        if (K, D) != (self.num_specials, self.token_norm.normalized_shape[0]):
            raise ValueError(
                f"z_s 形状应为 (B,{self.num_specials},"
                f"{self.token_norm.normalized_shape[0]})，得到 {tuple(z_s.shape)}")
        tokens = self.token_proj(self.token_norm(z_s))            # (B,32,640)
        grouped = tokens.reshape(B, self.visual_tokens, self.visual_dim)
        return self.budget_norm(grouped)                          # (B,8,2560)


# ═══════════════════════════════════════════════════════════════
# AssocVTCEncoder — 阶段2 编码器（冻结/解冻；只加载 encoder 部分）
# ═══════════════════════════════════════════════════════════════

class AssocVTCEncoder(nn.Module):
    """从阶段2 ckpt 恢复 `StandardAttentionEncoder`，输出 ``z_s (B,32,1744)``。

    只加载 `model` 里的 ``encoder.*`` 权重（decoder / refiner 不用）。底层
    `StandardAttentionEncoder.forward` 返回 ``(z_cls, z_s)``，这里丢弃 z_cls。

    E1 用 :meth:`freeze`（requires_grad=False + eval），E2–E3 用 :meth:`unfreeze`。
    """

    def __init__(self, config: VLM8BudgetConfig):
        super().__init__()
        if not config.encoder_ckpt:
            raise ValueError("config.encoder_ckpt 为空：需要阶段2 ckpt 才能构造编码器")
        self.enc_dim = config.enc_dim
        self.num_specials = config.num_specials
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.ckpt_path = config.encoder_ckpt
        self.net = self._build_from_ckpt(config)

    # ─── 构建 / 权重加载 ──────────────────────────────────────
    @staticmethod
    def _build_from_ckpt(config: VLM8BudgetConfig) -> nn.Module:
        """读 ckpt 的 ``cfg`` 重建编码器并 strict 加载 ``encoder.*`` 权重。"""
        encoder_cls = _import_standard_attention_encoder(config.assoctvc_repo)
        path = os.path.expanduser(config.encoder_ckpt)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到阶段2 ckpt: {path}（remote: /root/autodl-tmp/exp448/B/ckpt_best.pt）")

        # 自产 ckpt（cfg/args 是纯 python 对象 + 权重），这里显式关闭 weights_only。
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "model" not in blob or "cfg" not in blob:
            raise ValueError(f"{path} 不是预期的阶段2 ckpt（缺 'model' / 'cfg'）")
        cfg: dict = blob["cfg"]

        enc_attn = cfg.get("enc_attn")
        if enc_attn is not None and str(enc_attn).lower() not in ("std", "standard"):
            raise ValueError(
                f"ckpt 的 enc_attn={enc_attn!r}，但本实验需要 standard softmax "
                "注意力编码器（enc_attn='std'）")

        net = encoder_cls(
            dim=int(cfg["enc_dim"]),
            depth=int(cfg["enc_depth"]),
            heads=int(cfg["enc_heads"]),
            mlp_ratio=float(cfg.get("enc_mlp_ratio", 4.0)),
            patch=int(cfg["patch"]),
            img_size=int(cfg["img_size"]),
            num_specials=int(cfg["num_specials"]),
            qk_norm=bool(cfg.get("enc_std_qk_norm", False)),
            drop_rate=float(cfg.get("dropout", 0.0)),
            special_init_std=float(cfg.get("special_init_std", 0.02)),
        )

        # 形状契约与 config 对齐（避免"ckpt 是别的档位"这类静默错误）。
        checks = {
            "enc_dim": (net.dim, config.enc_dim),
            "enc_depth": (len(net.blocks), config.enc_depth),
            "enc_heads": (net.blocks[0].attn.heads, config.enc_heads),
            "num_specials": (net.num_specials, config.num_specials),
            "img_size": (net.img_size, config.image_size),
            "patch": (net.patch, config.patch_size),
        }
        mismatch = {k: v for k, v in checks.items() if v[0] != v[1]}
        if mismatch:
            raise ValueError(
                f"ckpt 与 config 形状不一致（ckpt, config）: {mismatch}")

        state = AssocVTCEncoder._extract_encoder_state(blob["model"])
        net.load_state_dict(state, strict=True)
        return net

    @staticmethod
    def _extract_encoder_state(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """从完整 state_dict 里取出 ``encoder.*`` 并去掉前缀。"""
        out: Dict[str, Tensor] = {}
        for key, value in state_dict.items():
            name = key
            changed = True
            while changed:                              # 兼容 DDP / compile 前缀
                changed = False
                for prefix in ("module.", "_orig_mod."):
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        changed = True
            if name.startswith("encoder."):
                out[name[len("encoder."):]] = value
        if not out:
            raise ValueError("ckpt 里没有 'encoder.*' 权重（decoder-only ckpt？）")
        return out

    # ─── 前向 ─────────────────────────────────────────────────
    def forward(self, pixel_values: Tensor) -> Tensor:
        """Args: pixel_values (B,3,448,448). Returns: z_s (B,32,1744)."""
        _, z_s = self.net(pixel_values)
        return z_s

    # ─── 冻结 / 解冻 ──────────────────────────────────────────
    def freeze(self) -> None:
        """E1：requires_grad=False + eval()（不进梯度计算）。"""
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """E2–E3：全部 requires_grad=True（不改变 train/eval 模式）。"""
        for param in self.parameters():
            param.requires_grad_(True)

    @property
    def is_frozen(self) -> bool:
        """当前是否完全冻结（所有参数 requires_grad=False）。"""
        return not any(p.requires_grad for p in self.parameters())

    def num_parameters(self, trainable_only: bool = False) -> int:
        """参数量（便于启动日志）。"""
        params = [p for p in self.parameters()
                  if (p.requires_grad or not trainable_only)]
        return sum(p.numel() for p in params)


# ═══════════════════════════════════════════════════════════════
# VLM8Budget — PreTrainedModel 主体
# ═══════════════════════════════════════════════════════════════

class VLM8Budget(PreTrainedModel):
    """AssocVTC 编码器 + 8 预算投影头 + Qwen3.5-4B 文本主干。

    Qwen 与 encoder 都是 **延迟加载**：

    * 训练：`VLM8Budget(config)` 默认 `config.auto_load_modules=True`，构造时加载；
    * 单测：可传入 `encoder=` / `lm=` 的替身（smoke.py 的纯张量自检）；
    * 本机：`import` 本模块不会触发任何 Qwen import。

    Forward 返回 dict：``{"loss", "per_budget_ce", "per_budget_ce_per_sample",
    "row_ce", "budget_ks"}``（后 3 个是 detached / 无梯度，供日志与 eval 用）。
    """

    config_class = VLM8BudgetConfig
    base_model_prefix = "vlm8b"
    #: Qwen 的 lm_head 与 embed_tokens 共享存储（tie_word_embeddings=True）。
    #: 必须在这里声明，否则 `save_pretrained` 的 shared-tensor 检查会报错
    #: （transformers 用 `_get_tied_weight_keys` 递归收集，保存时丢掉重复份）。
    _tied_weights_keys: List[str] = ["lm.lm_head.weight"]

    def __init__(
        self,
        config: VLM8BudgetConfig,
        encoder: Optional[nn.Module] = None,
        lm: Optional[nn.Module] = None,
        **kwargs,
    ):
        """构造投影头；encoder / lm 未注入且 `auto_load_modules` 时立刻加载。

        Args:
            config: `VLM8BudgetConfig`。
            encoder: 可选注入（测试用），默认从 `config.encoder_ckpt` 加载。
            lm: 可选注入（测试用），默认从 `config.qwen_dir` 加载 Qwen。
            **kwargs: 兼容 `PreTrainedModel.from_pretrained` 传入的额外参数（忽略）。
        """
        super().__init__(config)
        self.projector = BudgetProjector(config)
        self.encoder = encoder
        self.lm = lm
        self.loss_weighting = config.loss_weighting
        self._ce_sum: Optional[Tensor] = None
        self._loss_sum: float = 0.0
        self._ce_count: int = 0
        self._ce_ks: Optional[List[int]] = None

        if self.encoder is None and config.auto_load_modules:
            self.load_encoder()
        if self.lm is None and config.auto_load_modules:
            self.load_lm()

        if config.freeze_encoder and self.encoder is not None:
            self.encoder.freeze()
        if config.use_gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ─── 延迟加载 ─────────────────────────────────────────────
    def load_encoder(self) -> AssocVTCEncoder:
        """构造并挂上 AssocVTC 编码器（remote 才需要 ckpt 文件）。"""
        self.encoder = AssocVTCEncoder(self.config)
        return self.encoder

    def load_lm(self) -> nn.Module:
        """加载 Qwen3.5-4B 并删除 vision tower（本函数内才 import Qwen 类）。"""
        qwen_cls = _import_qwen_class()
        if not self.config.qwen_dir:
            raise ValueError("config.qwen_dir 为空：需要 Qwen3.5-4B 目录")
        lm = qwen_cls.from_pretrained(
            os.path.expanduser(self.config.qwen_dir),
            dtype=_resolve_torch_dtype(self.config.load_dtype),
            attn_implementation="sdpa",
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        removed = self._drop_vision_tower(lm)
        if self.config.use_gradient_checkpointing:
            lm.config.use_cache = False
        self.lm = lm
        if not removed:
            # 不是致命错误：不同 transformers 版本属性名可能变化，只提示。
            print("[vlm8b] warning: 未找到 lm.model.visual，vision tower 未删除")
        return lm

    @staticmethod
    def _drop_vision_tower(lm: nn.Module) -> bool:
        """删除 `lm.model.visual` 以省显存（我们不喂 pixel_values）。"""
        inner = getattr(lm, "model", None)
        visual = getattr(inner, "visual", None) if inner is not None else None
        if visual is None:
            return False
        delattr(inner, "visual")
        return True

    def enable_gradient_checkpointing(self, use_reentrant: bool = False) -> None:
        """对 Qwen 主干开 grad ckpt（Hybrid 层 + 4B 全参必开）。"""
        self.config.use_gradient_checkpointing = True
        self.config.use_cache = False
        if self.lm is not None:
            self.lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
            self.lm.config.use_cache = False

    def _ensure_encoder(self) -> AssocVTCEncoder:
        """拿到可用的 encoder（未加载则按 config 加载）。"""
        if self.encoder is None:
            self.load_encoder()
        return self.encoder

    def _ensure_lm(self) -> nn.Module:
        """拿到可用的 LM（未加载则按 config 加载）。"""
        if self.lm is None:
            self.load_lm()
        return self.lm

    # ─── 预算 / loss 权重 ─────────────────────────────────────
    def _resolve_budget_ids(self, budget_ids: Optional[object]) -> List[int]:
        """把 None / list / Tensor / 逗号串统一成升序 int 列表。"""
        if budget_ids is None:
            ks = list(self.config.budget_ids)
        elif isinstance(budget_ids, Tensor):
            ks = [int(x) for x in budget_ids.flatten().tolist()]
        elif isinstance(budget_ids, str):
            ks = [int(x) for x in budget_ids.split(",") if x.strip()]
        else:
            ks = [int(x) for x in budget_ids]
        ks = sorted(set(ks))
        bad = [k for k in ks if not 1 <= k <= self.config.visual_tokens]
        if bad:
            raise ValueError(
                f"budget_ids 越界 {bad}；合法范围 1..{self.config.visual_tokens}")
        return ks

    def set_loss_weighting(self, mode: str) -> None:
        """切换 loss 权重：``equal``（等权 1/V）或 ``inv-k``（w ∝ 1/k）。"""
        mode = str(mode).lower()
        if mode not in LOSS_WEIGHTINGS:
            raise ValueError(f"loss_weighting={mode!r} 非法；合法值: {LOSS_WEIGHTINGS}")
        self.loss_weighting = mode
        self.config.loss_weighting = mode

    def budget_weights(self, ks: Tensor, device: torch.device) -> Tensor:
        """按当前模式返回归一化权重 (V,)。"""
        k = ks.to(device=device, dtype=torch.float32)
        if self.loss_weighting == "equal":
            w = torch.ones_like(k)
        elif self.loss_weighting == "inv-k":
            w = 1.0 / k
        else:                                        # 构造函数已校验，防御性兜底
            raise ValueError(f"未知 loss_weighting: {self.loss_weighting}")
        return w / w.sum()

    # ─── 前向 ─────────────────────────────────────────────────
    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        budget_ids: Optional[object] = None,
        budget_weights: Optional[Tensor] = None,
        gt_name: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """训练 / teacher-forcing 前向（不做生成；生成见 `generate_budget`）。

        Args:
            pixel_values: (B,3,448,448)，AssocVTC 输入变换后的图。
            input_ids: (B,T)，`[system][user][assistant header][answer]` 的文本 ids。
            attention_mask: (B,T)，文本 right-pad mask；None ⇒ 全 1。
            labels: (B,T)，仅 answer 位置非 -100（其余 -100）。
            budget_ids: 可选，本次训练/评测的预算集合；None ⇒ `config.budget_ids`。
            budget_weights: 可选，(V,) 权重（会归一化）；None ⇒ 按 `loss_weighting`。
            gt_name: 数据集附带的类名（本函数不使用，仅为 Trainer 透传）。
            **kwargs: 忽略的额外字段（HF Trainer 兼容）。

        Returns:
            dict: ``loss`` / ``per_budget_ce``(V,) / ``per_budget_ce_per_sample``(B,V)
            / ``row_ce``(B*V,) / ``budget_ks``(V,)。
        """
        if pixel_values is None or input_ids is None or labels is None:
            raise ValueError(
                "forward 需要 pixel_values / input_ids / labels；"
                "纯生成请用 generate_budget()")
        encoder = self._ensure_encoder()
        lm = self._ensure_lm()

        param_dtype = next(encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=param_dtype)

        z_s = encoder(pixel_values)                          # (B,32,1744)
        # 显式对齐三处 dtype（encoder 可能是 fp32、Qwen 是 bf16；不让 autocast 兜底）
        proj_dtype = self.projector.token_proj.weight.dtype
        vis = self.projector(z_s.to(dtype=proj_dtype))       # (B,8,2560)
        text_embeds = lm.get_input_embeddings()(input_ids)   # (B,T,2560)
        vis = vis.to(dtype=text_embeds.dtype)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        ks = self._resolve_budget_ids(budget_ids)
        rows = expand_budget_rows(vis, text_embeds, attention_mask, labels, ks)
        return self.forward_rows(rows, budget_weights=budget_weights)

    def forward_rows(
        self,
        rows: BudgetRows,
        budget_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """对已展开的 `BudgetRows` 跑 LM + answer-only CE（smoke 也走这条路径）。

        Args:
            rows: `expand_budget_rows` 的输出。
            budget_weights: 可选 (V,)；None ⇒ 按 `self.loss_weighting`。

        Returns:
            dict：同 `forward`。
        """
        lm = self._ensure_lm()
        outputs = lm.model(
            inputs_embeds=rows.embeds,
            attention_mask=rows.attention_mask,
            use_cache=False,          # 训练不需要 KV cache（hybrid 层也省显存）
        )
        hidden = outputs.last_hidden_state                    # (R,L,H)
        R, L, H = hidden.shape
        A = rows.ans_pos.size(1)

        # 只 gather answer 位置（不 materialize 全序列 vocab logits）。
        safe_pos = rows.ans_pos.clamp(min=0)
        row_index = torch.arange(R, device=hidden.device).unsqueeze(1)
        flat_index = (row_index * L + safe_pos).reshape(-1)
        gathered = hidden.reshape(R * L, H).index_select(0, flat_index)
        gathered = gathered.reshape(R * A, H)
        targets = rows.ans_targets.reshape(-1)

        # CE 在 fp32 下分块算（nothink A=4 / think A≈58，V=8）。
        ce = torch.zeros(R * A, device=hidden.device, dtype=torch.float32)
        for start in range(0, R * A, CE_CHUNK_SIZE):
            stop = min(start + CE_CHUNK_SIZE, R * A)
            logits = lm.lm_head(gathered[start:stop])
            ce[start:stop] = F.cross_entropy(
                logits.float(), targets[start:stop],
                reduction="none", ignore_index=IGNORE_INDEX)

        counts = (rows.ans_targets != IGNORE_INDEX).sum(dim=1).clamp(min=1)
        row_ce = ce.reshape(R, A).sum(dim=1) / counts.to(ce.dtype)   # (R,)
        per_sample = row_ce.reshape(rows.num_images, rows.num_budgets)
        per_budget = per_sample.mean(dim=0)                          # (V,)

        weights = (budget_weights if budget_weights is not None
                   else self.budget_weights(rows.budget_ks, per_budget.device))
        weights = weights.to(device=per_budget.device, dtype=torch.float32)
        if weights.numel() != per_budget.numel():
            raise ValueError(
                f"budget_weights 长度 {weights.numel()} != 预算数 "
                f"{per_budget.numel()}")
        weights = weights / weights.sum().clamp(min=1e-12)
        loss = (per_budget.float() * weights).sum()

        if self.training:
            self._accumulate_stats(per_budget, loss, rows.budget_ks)

        return {
            "loss": loss,
            "per_budget_ce": per_budget.detach(),
            "per_budget_ce_per_sample": per_sample.detach(),
            "row_ce": row_ce.detach(),
            "budget_ks": rows.budget_ks,
        }

    # ─── per-budget CE 累积（TrainerCallback 每个 logging step 取走） ──
    def _accumulate_stats(
        self, per_budget: Tensor, loss: Tensor, ks: Tensor) -> None:
        """累积 micro-batch 的 per-budget CE（CPU float32，避免占显存）。"""
        value = per_budget.detach().float().cpu()
        self._ce_sum = value if self._ce_sum is None else self._ce_sum + value
        self._loss_sum += float(loss.detach().float().cpu())
        self._ce_count += 1
        if self._ce_ks is None:
            self._ce_ks = [int(k) for k in ks.detach().cpu().tolist()]

    def pop_budget_ce_stats(self) -> Optional[Dict[str, object]]:
        """取走并清空累积的 per-budget CE；没有数据时返回 None。"""
        if self._ce_sum is None or self._ce_count == 0:
            return None
        stats = {
            "per_budget_ce": (self._ce_sum / self._ce_count).tolist(),
            "budget_ks": list(self._ce_ks or []),
            "loss": self._loss_sum / self._ce_count,
            "n_micro": self._ce_count,
        }
        self._ce_sum = None
        self._loss_sum = 0.0
        self._ce_count = 0
        self._ce_ks = None
        return stats

    # ─── 生成 ─────────────────────────────────────────────────
    @torch.no_grad()
    def generate_budget(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        budget_ids: Optional[object] = None,
        max_new_tokens: int = 16,
    ) -> Dict[int, List[List[int]]]:
        """每张图 × 每个预算做 greedy 生成，返回 ``{k: [[new_ids...], ...]}``。

        right-padding 下 batch 内逐行长度不同，批量 generate 会把 pad 当上下文
        （设计 §9 的 Step-0 注意项），因此这里 **逐行** 生成（eval 规模可控）。
        `generate(inputs_embeds=...)` 的返回值只有新生成的 token ids。
        """
        lm = self._ensure_lm()
        encoder = self._ensure_encoder()
        dtype = next(encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=dtype)

        proj_dtype = self.projector.token_proj.weight.dtype
        vis = self.projector(encoder(pixel_values).to(dtype=proj_dtype))
        text_embeds = lm.get_input_embeddings()(input_ids)   # (B,T,2560)
        vis = vis.to(dtype=text_embeds.dtype)
        ks = self._resolve_budget_ids(budget_ids)
        B = pixel_values.size(0)
        results: Dict[int, List[List[int]]] = {int(k): [] for k in ks}

        gen_kwargs: Dict[str, object] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        if self.config.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.config.pad_token_id
        if self.config.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.config.eos_token_id

        for k in ks:
            for b in range(B):
                n = int(attention_mask[b].sum())             # 去掉文本 right-pad
                embeds = torch.cat(
                    [vis[b, :k], text_embeds[b, :n]], dim=0).unsqueeze(0)
                mask = torch.ones(
                    (1, k + n), device=embeds.device, dtype=torch.long)
                generated = lm.generate(
                    inputs_embeds=embeds, attention_mask=mask, **gen_kwargs)
                results[int(k)].append(generated[0].tolist())
        return results

    # ─── 训练模式：冻结的 encoder 始终留在 eval ────────────────
    def train(self, mode: bool = True) -> "VLM8Budget":
        """`model.train()` 时不把已冻结的 encoder 切回 train（确定性）。"""
        super().train(mode)
        if self.encoder is not None and not any(
                p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self


# ═══════════════════════════════════════════════════════════════
# 延迟导入辅助（本机 transformers 4.57 没有 Qwen3_5*，不能顶层 import）
# ═══════════════════════════════════════════════════════════════

def _import_qwen_class():
    """延迟导入 `Qwen3_5ForConditionalGeneration`（remote transformers 5.22.1）。"""
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration
    except ImportError:
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5ForConditionalGeneration as QwenClass,
            )
            return QwenClass
        except ImportError as exc:
            raise ImportError(
                "当前 transformers 未提供 Qwen3_5ForConditionalGeneration。"
                "本机（4.57）只能跑 smoke.py 的纯张量自检；训练/评测请在 remote "
                "/root/venv_qwen（transformers==5.22.1）下执行。"
            ) from exc


def _import_standard_attention_encoder(assoctvc_repo: str = ""):
    """延迟导入仓库根目录 `model_v2_linattn.StandardAttentionEncoder`。

    会按需把 `assoctvc_repo`（remote 代码目录）与 **本文件上级目录**（本仓库根）
    加入 `sys.path`，这样 `cd vlm8b && python smoke.py` 也能用。
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (assoctvc_repo, repo_root):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        from model_v2_linattn import StandardAttentionEncoder
        return StandardAttentionEncoder
    except ImportError as exc:
        raise ImportError(
            "无法导入 model_v2_linattn.StandardAttentionEncoder；"
            f"请确认仓库根目录（{repo_root}）含该文件，或用 config.assoctvc_repo "
            "指定 AssocVTC 代码目录"
        ) from exc


def _resolve_torch_dtype(name: str) -> torch.dtype:
    """字符串 → torch.dtype（只支持训练会用到的几种）。"""
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"不支持的 dtype={name!r}；可选: {sorted(mapping)}")
    return mapping[key]
