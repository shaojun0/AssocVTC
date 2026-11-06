"""
vlm8b — AssocVTC 448²/p32 阶段2 编码器 → 视觉读出（linear / loop）→ Qwen3.5-4B
================================================================================

两条视觉读出（`--readout`）：

  · linear（默认，基线）：z_s (B,32,1744) → 逐 token LN+Linear(1744→640)
    → reshape(B,8,4*640=2560) → LN(2560) = vis (B,8,2560)；8 个 token 互不通信。

  · loop（本分支新增）：A=[z_cls; z_s] (B,33,1744) → mem 投影 → 权重共享循环栈
    被施加 8 次，第 t 步只读第 t 块 4 个 register、carry 走 BPTT，第 t 步产出
    第 t 个 token：vis = cat(V_1..V_8) (B,8,2560)。于是
        **预算 k = 循环步数 k = 前 4k 个 register = 前 k 个 LM token**（三种截断对齐）。
    可选 `--loop-init stage2` 用阶段2 ckpt 的 `decoder.*` 浅层热启动（见 README §5）。
    设计文档：main 仓库 `doc/2026-11-06/DESIGN_looped_readout_vlm8b.md`。

数据流（loop 时只多一次 z_cls 读取，其余与 linear 完全相同）::

    image (B,3,448,448)
      → AssocVTCEncoder（阶段2 ckpt，只读 encoder.*）→ (z_cls (B,1,1744), z_s (B,32,1744))
      → 读出（linear / loop）→ vis (B,8,2560)
      → 每图 V 行（默认 k=1..8）：行 k = [vis[:, :k]; text]（截断）
      → Qwen3.5-4B 文本主干（只喂 inputs_embeds）
      → hidden → 只 gather answer 的预测位置 → lm_head → CE

关键裁决（`doc/2026-11-05/DESIGN_8budget_vlm_finetune.md` §4）:

  · **8 行批，不打包成一条序列**：Qwen3.5 是 hybrid（24 层线性注意力 + 8 层全
    注意力），线性层收到 4D mask 时会忽略段边界 ⇒ 只用 2D right-padding mask。
  · **截断防泄露**：行 k 只含 `vis[:, :k]`，不靠 mask 遮 visual。
  · **标准因果 shift**：position p 的 hidden 只能预测 `label[p+1]`，故
    `ans_pos = k + (label_pos − 1)`、targets = `labels[label_pos]`，且 `label_pos ≥ 1`。
    用 label 自身位置取 hidden 会目标泄露（hidden 已看见该 token 的 embedding）。
  · **loop 读出的前缀因果**：token t 只依赖前 t*block 个 register ⇒ 行 k 的视觉
    前缀 `vis[:k]` 的依赖范围恰好是 `z_s[:k*block]`（smoke 自检 ⑥ 锁死）。

Qwen 相关 import 全部在函数内部（本机 transformers 4.x 没有 `Qwen3_5*`），所以
`import model` 不会加载 Qwen。
"""
import math
import os
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig

# ═══════════════════════════════════════════════════════════════
# 常量（形状契约；改这里必须同步 README 的表格）
# ═══════════════════════════════════════════════════════════════

ENC_DIM = 1744                  # 阶段2 编码器维度（B/ckpt_best 实测）
NUM_SPECIALS = 32               # z_s 的 token 数 K
IMAGE_SIZE = 448
PATCH_SIZE = 32
PROJ_DIM = 640                  # 逐 token 降维目标
VISUAL_TOKENS = 8               # visual token 数 = 预算上限
LM_HIDDEN = 2560                # Qwen3.5-4B text hidden
IGNORE_INDEX = -100
CE_CHUNK = 64                   # 分块 gather/lm_head 的位置数
DEFAULT_BUDGETS = tuple(range(1, VISUAL_TOKENS + 1))
LOOP_BLOCK = 4                   # 每步读的 register 数（8 步 × 4 = 32 = K）


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class VLM8BudgetConfig(PretrainedConfig):
    """`VLM8Budget` 的 HF 配置：只放结构与加载路径，训练超参在 `train.py` CLI。"""

    model_type = "vlm8_budget"

    def __init__(self, encoder_ckpt: str = "", qwen_dir: str = "",
                 budget_ids: Optional[Sequence[int]] = None,
                 loss_weighting: str = "equal",
                 readout: str = "linear",
                 loop_dim: int = 1024,
                 loop_depth: int = 2,
                 loop_heads: int = 8,
                 loop_mlp_ratio: float = 4.0,
                 loop_init: str = "none",
                 loop_block: int = LOOP_BLOCK,
                 loop_dropout: float = 0.0,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 hidden_size: int = LM_HIDDEN, **kwargs):
        super().__init__(**kwargs)
        self.encoder_ckpt = str(encoder_ckpt)
        self.qwen_dir = str(qwen_dir)
        self.budget_ids = [int(k) for k in (budget_ids or DEFAULT_BUDGETS)]
        self.loss_weighting = str(loss_weighting)
        assert readout in ("linear", "loop"), f"未知 readout={readout!r}"
        assert loop_init in ("none", "stage2"), f"未知 loop_init={loop_init!r}"
        self.readout = str(readout)
        self.loop_dim = int(loop_dim)
        self.loop_depth = int(loop_depth)
        self.loop_heads = int(loop_heads)
        self.loop_mlp_ratio = float(loop_mlp_ratio)
        self.loop_init = str(loop_init)
        self.loop_block = int(loop_block)
        self.loop_dropout = float(loop_dropout)
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        # DeepSpeed ZeRO-3 的 `"auto"` bucket 尺寸依赖 config.hidden_size（HF 在
        # trainer_config_finalize 里读它）；缺了会直接 ValueError。
        self.hidden_size = int(hidden_size)


# ═══════════════════════════════════════════════════════════════
# 8 预算行展开（纯张量；smoke.py 单测这一层）
# ═══════════════════════════════════════════════════════════════

class BudgetRows(NamedTuple):
    """行顺序 `b * V + vi`；`ans_pos` 是取 hidden 的**预测位置**（无效位 -1）。"""

    embeds: Tensor
    attention_mask: Tensor
    labels: Tensor
    ans_pos: Tensor
    ans_targets: Tensor
    budget_ks: Tensor
    num_images: int
    num_budgets: int


def expand_budget_rows(
    vis_embeds: Tensor,
    text_embeds: Tensor,
    text_mask: Tensor,
    labels: Tensor,
    budget_ks: Sequence[int],
) -> BudgetRows:
    """`vis (B,8,H)` + `text (B,T,H)` → 每图 V 行 `[vis[:, :k]; text]`（right-pad）。"""
    B, T, H = text_embeds.shape
    ks = [int(k) for k in budget_ks]
    assert ks and min(ks) >= 1 and max(ks) <= vis_embeds.size(1), f"预算越界: {ks}"
    V, L = len(ks), max(ks) + T

    embeds = text_embeds.new_zeros((B, V, L, H))
    masks = text_mask.new_zeros((B, V, L))
    row_labels = labels.new_full((B, V, L), IGNORE_INDEX)
    for vi, k in enumerate(ks):                 # 视觉前缀只取前 k 个 = 截断
        embeds[:, vi, :k] = vis_embeds[:, :k]
        embeds[:, vi, k:k + T] = text_embeds
        masks[:, vi, :k] = 1
        masks[:, vi, k:k + T] = text_mask
        row_labels[:, vi, k:k + T] = labels

    valid = labels != IGNORE_INDEX
    starts = valid.long().argmax(dim=1)         # 每行 answer 起始（连续段）
    assert int(starts.min()) >= 1, "answer 从文本第 0 位开始，没有 prompt 末 token 可预测"
    counts = valid.sum(dim=1)
    A = int(counts.max())
    j = torch.arange(A, device=labels.device).unsqueeze(0)
    label_pos = starts.unsqueeze(1) + j                          # (B,A)
    ans_valid = j < counts.unsqueeze(1)
    targets = labels.gather(
        1, label_pos.clamp(max=T - 1)).masked_fill(~ans_valid, IGNORE_INDEX)

    # 标准因果 shift：预测位置 = label 位置 − 1；再用 +k 换成行内绝对下标。
    # ⚠️ 不能写成 label 位置本身（hidden 已看见该 token ⇒ 目标泄露）。
    pos = (label_pos - 1).unsqueeze(1) + torch.tensor(ks, device=labels.device).view(1, V, 1)
    pos = pos.masked_fill(~ans_valid.unsqueeze(1), -1)

    return BudgetRows(
        embeds=embeds.reshape(B * V, L, H),
        attention_mask=masks.reshape(B * V, L),
        labels=row_labels.reshape(B * V, L),
        ans_pos=pos.reshape(B * V, A),
        ans_targets=targets.unsqueeze(1).expand(B, V, A).reshape(B * V, A),
        budget_ks=torch.tensor(ks, device=labels.device),
        num_images=B,
        num_budgets=V,
    )


# ═══════════════════════════════════════════════════════════════
# BudgetProjector — z_s (B,32,1744) → vis (B,8,2560)
# ═══════════════════════════════════════════════════════════════

class BudgetProjector(nn.Module):
    """逐 token `LayerNorm(1744)→Linear(1744,640)`，4 拼 1 后 `LayerNorm(2560)`。"""

    def __init__(self):
        super().__init__()
        self.token_norm = nn.LayerNorm(ENC_DIM)
        self.token_proj = nn.Linear(ENC_DIM, PROJ_DIM)
        self.budget_norm = nn.LayerNorm(LM_HIDDEN)
        nn.init.normal_(self.token_proj.weight, std=0.02)
        nn.init.zeros_(self.token_proj.bias)

    def forward(self, z_s: Tensor) -> Tensor:
        """z_s (B,32,1744) → vis (B,8,2560)。"""
        tokens = self.token_proj(self.token_norm(z_s))            # (B,32,640)
        return self.budget_norm(tokens.reshape(z_s.size(0), VISUAL_TOKENS, LM_HIDDEN))


# ═══════════════════════════════════════════════════════════════
# 循环读出（loop）：预算 k = 循环步 k = 前 k*block 个 register = k 个 LM token
# ═══════════════════════════════════════════════════════════════

def step_windows(num_specials: int, steps: Sequence[int], plan: str = "fixed",
                 block: int = 0) -> List[Tuple[int, int]]:
    """每步在 `A=[z_cls; z_s]`（列 0=z_cls，列 1..K=z_s）上的读窗口 `(lo, hi)`（闭区间）。

    与 main 分支 `model_v2_linattn.step_windows` 同式：
      · `plan="fixed"`（默认，阶段2 ckpt 的成功操作点）：第 k 步读第 k 块，块宽
        `block`；首步窗口额外含 z_cls（lo=0）。
      · `plan="square"`：k=isqrt(t)，窗口 [max(k²,1), min((k+1)²−1, K)]；首步 lo=0。
      · `plan="cumulative"`：`steps` 直接是每步累计读入的 z_s 数 C_1<…<C_T（首步 lo=0）。
    """
    assert plan in ("fixed", "square", "cumulative"), f"未知 plan={plan!r}"
    if plan == "fixed":
        assert block >= 1, f"plan=fixed 须 block≥1, got {block}"
    out: List[Tuple[int, int]] = []
    for i, t in enumerate(steps):
        if plan == "cumulative":
            cum = [int(x) for x in steps]
            assert all(cum[j] < cum[j + 1] for j in range(len(cum) - 1)), cum
            assert cum[0] >= 1 and cum[-1] <= num_specials, cum
            out = [(0 if j == 0 else cum[j - 1] + 1, c) for j, c in enumerate(cum)]
            return out
        if plan == "fixed":
            k = int(t)
            hi = min(k * block, num_specials)
            lo = 0 if i == 0 else (k - 1) * block + 1
        else:
            k = int(math.isqrt(int(t)))
            hi = min((k + 1) ** 2 - 1, num_specials)
            lo = 0 if i == 0 else max(k * k, 1)
        assert lo <= hi, f"步 {t} 的读窗口 [{lo},{hi}] 为空 (K={num_specials})"
        out.append((lo, hi))
    return out


class LoopVisualReadout(nn.Module):
    """权重共享循环读出：第 t 步产出第 t 个 LM token（预算=步数）。

    机制（与主线 `RecurrentOutputQueryDecoder` 同构，输出头从像素换成 LM hidden）::

        A = mem_norm(mem_proj([z_cls; z_s])) + mem_pos              (B, 1+K, dim)
        Y_0 = 0
        for t, (lo, hi) in enumerate(windows):                      # 步 t 只读本步窗口
            Y_t = Stack(q_bank[t] + Y_{t-1}, A[:, lo:hi+1])         # 栈权重跨步共享
            V_t = out_proj(norm_out(Y_t))                           # (B, 1, LM_HIDDEN)
            Y_{t} 作为下一步 carry（**BPTT：不 detach**）
        return cat(V_1..V_T)                                        # (B, T, LM_HIDDEN)

    设计要点（全来自主线实测教训，见设计文档 §4/§6）:
      · **BPTT 不 detach**：detach 会让 carry 变成纯前向接力，早步收不到末步梯度；
      · **每步都有直接监督**：由"8 行预算 CE"免费提供（token t 出现在第 t..8 行）；
      · **恰好一个 step identity**：`q_bank` 每行一个 token 身份，故**不再另加**
        step embedding（两套 identity 会互相干扰，主线 arm-4a 实测更差）。
    """

    def __init__(self, enc_dim: int = ENC_DIM, dim: int = 1024, depth: int = 2,
                 heads: int = 8, mlp_ratio: float = 4.0, n_tokens: int = VISUAL_TOKENS,
                 n_specials: int = NUM_SPECIALS, lm_hidden: int = LM_HIDDEN,
                 block: int = LOOP_BLOCK, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 须被 heads={heads} 整除"
        self.enc_dim, self.dim, self.depth = int(enc_dim), int(dim), int(depth)
        self.n_tokens, self.n_specials = int(n_tokens), int(n_specials)
        self.mem_proj = (nn.Identity() if self.enc_dim == self.dim
                         else nn.Linear(self.enc_dim, self.dim))
        self.mem_norm = nn.LayerNorm(self.dim)
        self.mem_pos = nn.Parameter(torch.zeros(1, 1 + self.n_specials, self.dim))
        self.q_bank = nn.Parameter(torch.randn(self.n_tokens, self.dim) * 0.02)
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.dim, nhead=heads,
                dim_feedforward=int(self.dim * float(mlp_ratio)),
                dropout=float(dropout), activation="gelu",
                batch_first=True, norm_first=True),
            num_layers=self.depth)
        self.norm_out = nn.LayerNorm(self.dim)
        self.out_proj = nn.Linear(self.dim, int(lm_hidden))
        self.steps = list(range(1, self.n_tokens + 1))
        self.windows = step_windows(self.n_specials, self.steps,
                                    plan="fixed", block=int(block))
        assert self.windows[-1][1] == self.n_specials, (
            f"读窗口未覆盖全部 {self.n_specials} 个 register: {self.windows}")

    def forward(self, z_cls: Tensor, z_s: Tensor,
                detach_carry: bool = False) -> Tensor:
        """`z_cls (B,1,enc_dim)` + `z_s (B,K,enc_dim)` → `vis (B,T,LM_HIDDEN)`。

        `detach_carry=True` 只用于自检（复现主线 detach 口径），训练永远走 BPTT。
        """
        assert z_cls.dim() == 3 and z_s.dim() == 3, "z_cls/z_s 须为 (B,·,D)"
        assert z_cls.shape[1] == 1 and z_s.shape[1] == self.n_specials, (
            f"z_cls 应 1 列 / z_s 应 {self.n_specials} 列, got "
            f"{tuple(z_cls.shape)} / {tuple(z_s.shape)}")
        mem = torch.cat([z_cls, z_s], dim=1)                    # (B,1+K,enc_dim)
        mem = self.mem_norm(self.mem_proj(mem)) + self.mem_pos  # (B,1+K,dim)
        outs: List[Tensor] = []
        carry: Optional[Tensor] = None
        for t, (lo, hi) in enumerate(self.windows):
            q = self.q_bank[t].view(1, 1, -1).expand(mem.size(0), -1, -1)
            q_in = q if carry is None else q + carry            # carry：BPTT，不 detach
            y = self.stack(q_in, mem[:, lo:hi + 1])             # (B,1,dim)
            outs.append(self.out_proj(self.norm_out(y)))        # (B,1,LM_HIDDEN)
            carry = y.detach() if detach_carry else y
        return torch.cat(outs, dim=1)                           # (B,T,LM_HIDDEN)


def load_stage2_decoder(readout: "LoopVisualReadout",
                        blob: Dict[str, object]) -> Dict[str, int]:
    """把阶段2 ckpt 的 `decoder.*` 灌进循环读出（浅/全量热启动）。

    复用：`mem_proj` / `mem_norm` / `mem_pos` / `stack.layers.{0..depth-1}` /
    `norm_out`；**丢弃**与输出目标绑定的 `query_base` / `query_pos` / `pixel_head`
    （像素 3072 维 → LM 2560 维，形状不再匹配）。返回载入/跳过的 key 数。
    """
    own = readout.state_dict()
    state: Dict[str, Tensor] = {}
    skipped = 0
    for key, value in blob["model"].items():
        name = str(key)
        for prefix in ("module.", "_orig_mod."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if not name.startswith("decoder."):
            continue
        sub = name[len("decoder."):]
        if sub.startswith(("query_base", "query_pos", "pixel_head")):
            skipped += 1
            continue
        if sub.startswith("stack.layers."):                 # 只载入前 depth 层
            if int(sub.split(".")[2]) >= readout.depth:
                skipped += 1
                continue
        if sub not in own or tuple(own[sub].shape) != tuple(value.shape):
            skipped += 1
            continue
        state[sub] = value
    readout.load_state_dict(state, strict=False)
    return {"loaded": len(state), "skipped": skipped}


# ═══════════════════════════════════════════════════════════════
# AssocVTCEncoder — 阶段2 编码器（E1 冻结 / E2+ 解冻）
# ═══════════════════════════════════════════════════════════════

class AssocVTCEncoder(nn.Module):
    """从阶段2 ckpt 恢复 `StandardAttentionEncoder`，输出 `z_s (B,32,1744)`。"""

    def __init__(self, ckpt_path: str):
        super().__init__()
        from encoder import StandardAttentionEncoder as encoder_cls
        blob = torch.load(os.path.expanduser(ckpt_path),
                          map_location="cpu", weights_only=False)
        cfg = blob["cfg"]
        self.net = encoder_cls(
            dim=int(cfg["enc_dim"]), depth=int(cfg["enc_depth"]),
            heads=int(cfg["enc_heads"]), mlp_ratio=float(cfg.get("enc_mlp_ratio", 4.0)),
            patch=int(cfg["patch"]), img_size=int(cfg["img_size"]),
            num_specials=int(cfg["num_specials"]),
            qk_norm=bool(cfg.get("enc_std_qk_norm", False)),
            drop_rate=float(cfg.get("dropout", 0.0)),
            special_init_std=float(cfg.get("special_init_std", 0.02)))
        contract = (self.net.dim, self.net.num_specials, self.net.img_size, self.net.patch)
        assert contract == (ENC_DIM, NUM_SPECIALS, IMAGE_SIZE, PATCH_SIZE), (
            f"阶段2 ckpt 形状契约不符: {contract}")

        state = {}
        for key, value in blob["model"].items():
            name = key
            for prefix in ("module.", "_orig_mod."):
                name = name[len(prefix):] if name.startswith(prefix) else name
            if name.startswith("encoder."):
                state[name[len("encoder."):]] = value
        assert state, "ckpt 里没有 encoder.* 权重"
        self.net.load_state_dict(state, strict=True)

    def forward(self, pixel_values: Tensor) -> Tuple[Tensor, Tensor]:
        """pixel_values (B,3,448,448) → (z_cls (B,1,1744), z_s (B,32,1744))。"""
        return self.net(pixel_values)

    def freeze(self) -> None:
        """E1：requires_grad=False + eval()。"""
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """E2–E3：全部 requires_grad=True。"""
        for param in self.parameters():
            param.requires_grad_(True)


# ═══════════════════════════════════════════════════════════════
# 视觉消融（评测用，不训练）
# ═══════════════════════════════════════════════════════════════

def apply_visual_mode(vis: Tensor, mode: str) -> Tensor:
    """`normal` / `null`（全零 ⇒ 等价没有图）/ `shuffle`（batch 内错位 ⇒ 看错图）。"""
    if mode == "normal":
        return vis
    if mode == "null":
        return vis * 0
    if mode == "shuffle":
        return vis * 0 if vis.size(0) <= 1 else torch.roll(vis, shifts=1, dims=0)
    raise ValueError(f"未知 visual_mode: {mode}")


# ═══════════════════════════════════════════════════════════════
# VLM8Budget — 编码器 + 投影头 + Qwen 文本主干
# ═══════════════════════════════════════════════════════════════

class VLM8Budget(PreTrainedModel):
    """AssocVTC encoder + 8 预算投影 + Qwen3.5-4B；`forward` 返回 loss 与 per-budget CE。"""

    config_class = VLM8BudgetConfig
    base_model_prefix = "vlm8b"
    # transformers 5.x: `_tied_weights_keys` 是 dict{key: 源 key}，且会递归遍历子模块；
    # `lm`（Qwen3_5ForConditionalGeneration）自己声明了 {'lm_head.weight': ...}，
    # 这里留空 dict 让递归生效即可（写成 list 会让 save_pretrained 直接 AttributeError）。
    _tied_weights_keys: Dict[str, str] = {}

    def __init__(self, config: VLM8BudgetConfig,
                 encoder: Optional[nn.Module] = None, lm: Optional[nn.Module] = None,
                 grad_ckpt: bool = False):
        super().__init__(config)
        self.readout = str(config.readout)
        self.projector = self._build_projector()
        self.encoder = encoder if encoder is not None else AssocVTCEncoder(
            config.encoder_ckpt)
        self.lm = lm if lm is not None else self.load_lm(grad_ckpt)
        self.loss_weighting, self.last_budget_ce = str(config.loss_weighting), None

    def _build_projector(self) -> nn.Module:
        """按 `config.readout` 建读出：linear=基线线性重分组；loop=权重共享循环。"""
        if self.config.readout == "linear":
            return BudgetProjector()
        dim, depth = int(self.config.loop_dim), int(self.config.loop_depth)
        heads, mlp = int(self.config.loop_heads), float(self.config.loop_mlp_ratio)
        if self.config.loop_init == "stage2":
            blob = torch.load(os.path.expanduser(self.config.encoder_ckpt),
                              map_location="cpu", weights_only=False)
            cfg = blob["cfg"]
            dim = int(cfg["dec_dim"])
            heads = int(cfg["dec_heads"])
            mlp = float(cfg.get("dec_mlp_ratio", 4.0))
            depth = min(depth, int(cfg["dec_depth"]))
            readout = LoopVisualReadout(
                enc_dim=ENC_DIM, dim=dim, depth=depth, heads=heads, mlp_ratio=mlp,
                block=self.config.loop_block, dropout=self.config.loop_dropout)
            stats = load_stage2_decoder(readout, blob)
            del blob
            print(f"[vlm8b] loop readout warm start from {self.config.encoder_ckpt}: "
                  f"dim={dim} depth={depth} heads={heads} "
                  f"loaded={stats['loaded']} skipped={stats['skipped']}")
            return readout
        return LoopVisualReadout(
            enc_dim=ENC_DIM, dim=dim, depth=depth, heads=heads, mlp_ratio=mlp,
            block=self.config.loop_block, dropout=self.config.loop_dropout)

    def visual_tokens(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        """(z_cls, z_s) → vis (B,8,LM_HIDDEN)；linear 只用 z_s（与基线逐位一致）。"""
        if self.readout == "loop":
            dtype = next(self.projector.parameters()).dtype
            return self.projector(z_cls.to(dtype), z_s.to(dtype))
        return self.projector(z_s)

    def load_lm(self, grad_ckpt: bool = False) -> nn.Module:
        """加载 Qwen3.5-4B（本函数内才 import Qwen），删 vision tower、开 grad ckpt。"""
        lm = _import_qwen().from_pretrained(
            os.path.expanduser(self.config.qwen_dir), dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True, local_files_only=True)
        if hasattr(lm.model, "visual"):
            del lm.model.visual                                  # 不喂 pixel_values
        if grad_ckpt:
            lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        lm.config.use_cache = False
        return lm

    def _budgets(self, budget_ids: Optional[Sequence[int]] = None) -> List[int]:
        """None ⇒ config.budget_ids；结果升序去重并检查越界。"""
        ks = sorted({int(k) for k in (self.config.budget_ids if budget_ids is None
                                      else budget_ids)})
        assert ks and 1 <= ks[0] and ks[-1] <= VISUAL_TOKENS, f"预算越界: {ks}"
        return ks

    def _keep_encoder_eval(self) -> None:
        """幂等保护：已冻结的 encoder 不能被 `model.train()` 拉回 train（dropout 必须关）。"""
        encoder = self.encoder
        if encoder.training and not any(p.requires_grad for p in encoder.parameters()):
            encoder.eval()

    def forward(self, pixel_values: Tensor, input_ids: Tensor,
                attention_mask: Optional[Tensor] = None,
                labels: Optional[Tensor] = None,
                budget_ids: Optional[Sequence[int]] = None,
                budget_weights: Optional[Tensor] = None,
                visual_mode: str = "normal",
                **kwargs) -> Dict[str, object]:
        """训练 / teacher-forcing 前向（生成见 `generate_budget`）。"""
        self._keep_encoder_eval()
        encoder_dtype = next(self.encoder.parameters()).dtype
        z_cls, z_s = self.encoder(pixel_values.to(dtype=encoder_dtype))  # (B,1,·)/(B,32,·)
        vis = apply_visual_mode(self.visual_tokens(z_cls, z_s), visual_mode)
        text_embeds = self.lm.get_input_embeddings()(input_ids)          # (B,T,2560)
        rows = expand_budget_rows(
            vis, text_embeds.to(vis.dtype), attention_mask, labels,
            self._budgets(budget_ids))
        return self.forward_rows(rows, budget_weights)

    def forward_rows(self, rows: BudgetRows,
                     budget_weights: Optional[Tensor] = None) -> Dict[str, object]:
        """对已展开的 `BudgetRows` 跑 LM + answer-only CE（smoke 也走这条路径）。"""
        self._keep_encoder_eval()
        hidden = self.lm.model(
            inputs_embeds=rows.embeds, attention_mask=rows.attention_mask,
            use_cache=False).last_hidden_state                     # (R,L,H)
        R, L, H = hidden.shape
        A = rows.ans_pos.size(1)

        # 只 gather 预测位置（不 materialize 全序列 vocab logits），CE 在 fp32 分块算。
        flat = (torch.arange(R, device=hidden.device).unsqueeze(1) * L + rows.ans_pos.clamp(min=0)).reshape(-1)
        gathered = hidden.reshape(R * L, H).index_select(0, flat)
        targets = rows.ans_targets.reshape(-1)
        ce = torch.empty(R * A, device=hidden.device, dtype=torch.float32)
        for start in range(0, ce.numel(), CE_CHUNK):
            stop = start + CE_CHUNK
            ce[start:stop] = F.cross_entropy(
                self.lm.lm_head(gathered[start:stop]).float(), targets[start:stop],
                reduction="none", ignore_index=IGNORE_INDEX)

        counts = (rows.ans_targets != IGNORE_INDEX).sum(dim=1).clamp(min=1)
        row_ce = ce.reshape(R, A).sum(dim=1) / counts.to(ce.dtype)
        per_budget = row_ce.reshape(rows.num_images, rows.num_budgets).mean(dim=0)
        k = rows.budget_ks.float()
        w = budget_weights if budget_weights is not None else (
            torch.ones_like(k) if self.loss_weighting == "equal" else 1.0 / k)
        loss = (per_budget.float() * (w / w.sum()).to(per_budget.device).float()).sum()

        if self.training:                       # callback 在 logging step 读走
            self.last_budget_ce = {
                "budget_ks": [int(v) for v in rows.budget_ks.tolist()],
                "per_budget_ce": [float(v) for v in per_budget.detach().cpu()],
            }
        return {
            "loss": loss,
            "per_budget_ce": per_budget.detach(),
            "per_budget_ce_per_sample": row_ce.reshape(
                rows.num_images, rows.num_budgets).detach(),
            "row_ce": row_ce.detach(),
            "budget_ks": rows.budget_ks,
        }

    @torch.no_grad()
    def generate_budget(self, pixel_values: Tensor, input_ids: Tensor,
                        attention_mask: Tensor,
                        budget_ids: Optional[Sequence[int]] = None,
                        max_new_tokens: int = 16,
                        visual_mode: str = "normal") -> Dict[int, List[List[int]]]:
        """每图 × 每预算 greedy 生成（right-pad 下逐行），返回 `{k: [[ids...], ...]}`。"""
        self._keep_encoder_eval()
        encoder_dtype = next(self.encoder.parameters()).dtype
        z_cls, z_s = self.encoder(pixel_values.to(dtype=encoder_dtype))
        vis = apply_visual_mode(self.visual_tokens(z_cls, z_s), visual_mode)
        text_embeds = self.lm.get_input_embeddings()(input_ids).to(vis.dtype)
        results: Dict[int, List[List[int]]] = {}
        for k in self._budgets(budget_ids):
            results[k] = []
            for b in range(pixel_values.size(0)):
                n = int(attention_mask[b].sum())                 # 去掉文本 right-pad
                embeds = torch.cat([vis[b, :k], text_embeds[b, :n]], dim=0).unsqueeze(0)
                mask = torch.ones(embeds.shape[:2], dtype=torch.long,
                                  device=embeds.device)
                generated = self.lm.generate(
                    inputs_embeds=embeds, attention_mask=mask,
                    max_new_tokens=int(max_new_tokens), do_sample=False,
                    pad_token_id=self.config.pad_token_id,
                    eos_token_id=self.config.eos_token_id)
                results[k].append(generated[0].tolist())
        return results


# ═══════════════════════════════════════════════════════════════
# 延迟导入（本机没有 Qwen3_5* / 编码器 ckpt，import 本模块不触发）
# ═══════════════════════════════════════════════════════════════

def _import_qwen():
    """延迟导入 `Qwen3_5ForConditionalGeneration`（remote transformers 5.22.1）。"""
    from transformers import Qwen3_5ForConditionalGeneration
    return Qwen3_5ForConditionalGeneration

