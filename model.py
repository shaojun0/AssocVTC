"""
vlm8b — AssocVTC 448²/p32 阶段2 编码器 → 8 预算并行 → Qwen3.5-4B 全参微调
========================================================================

数据流::

    image (B,3,448,448)
      → AssocVTCEncoder（从阶段2 ckpt 恢复 StandardAttentionEncoder，只读 encoder.*）
      → z_s (B,32,1744)
      → BudgetProjector: LayerNorm(1744) → Linear(1744,640)          [逐 token]
      → reshape (B,8,4*640=2560) → LayerNorm(2560) → vis (B,8,2560)
      → 每图 V 行（默认 k=1..8）：行 k = [vis[:, :k]; text]           [截断]
      → Qwen3.5-4B 文本主干（只喂 inputs_embeds，不喂 pixel_values，vision tower 已删）
      → hidden → 只 gather answer 的**预测位置** → lm_head → CE（fp32 分块）

关键裁决（`doc/2026-11-05/DESIGN_8budget_vlm_finetune.md` §4）:

  · **8 行批，不打包成一条序列**：Qwen3.5 是 hybrid（24 层线性注意力 + 8 层全
    注意力），线性层收到 4D mask 时会忽略段边界 ⇒ 只用 2D right-padding mask。
  · **截断防泄露**：行 k 只含 `vis[:, :k]`，不靠 mask 遮 visual。
  · **标准因果 shift**：position p 的 hidden 只能预测 `label[p+1]`，故
    `ans_pos = k + (label_pos − 1)`、targets = `labels[label_pos]`，且 `label_pos ≥ 1`。
    用 label 自身位置取 hidden 会目标泄露（hidden 已看见该 token 的 embedding）。

Qwen 相关 import 全部在函数内部（本机 transformers 4.57 没有 `Qwen3_5*`），所以
`import model` 不会加载 Qwen。
"""
import os
from typing import Dict, List, NamedTuple, Optional, Sequence

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


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class VLM8BudgetConfig(PretrainedConfig):
    """`VLM8Budget` 的 HF 配置：只放结构与加载路径，训练超参在 `train.py` CLI。"""

    model_type = "vlm8_budget"

    def __init__(self, encoder_ckpt: str = "", qwen_dir: str = "",
                 budget_ids: Optional[Sequence[int]] = None,
                 loss_weighting: str = "equal",
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 hidden_size: int = LM_HIDDEN, **kwargs):
        super().__init__(**kwargs)
        self.encoder_ckpt = str(encoder_ckpt)
        self.qwen_dir = str(qwen_dir)
        self.budget_ids = [int(k) for k in (budget_ids or DEFAULT_BUDGETS)]
        self.loss_weighting = str(loss_weighting)
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

    def forward(self, pixel_values: Tensor) -> Tensor:
        """pixel_values (B,3,448,448) → z_s (B,32,1744)。"""
        return self.net(pixel_values)[1]

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
        self.projector = BudgetProjector()
        self.encoder = encoder if encoder is not None else AssocVTCEncoder(
            config.encoder_ckpt)
        self.lm = lm if lm is not None else self.load_lm(grad_ckpt)
        self.loss_weighting, self.last_budget_ce = str(config.loss_weighting), None

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
                **kwargs) -> Dict[str, object]:
        """训练 / teacher-forcing 前向（生成见 `generate_budget`）。"""
        self._keep_encoder_eval()
        encoder_dtype = next(self.encoder.parameters()).dtype
        z_s = self.encoder(pixel_values.to(dtype=encoder_dtype))         # (B,32,1744)
        vis = self.projector(z_s.to(self.projector.token_proj.weight.dtype))
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
                        max_new_tokens: int = 16) -> Dict[int, List[List[int]]]:
        """每图 × 每预算 greedy 生成（right-pad 下逐行），返回 `{k: [[ids...], ...]}`。"""
        self._keep_encoder_eval()
        encoder_dtype = next(self.encoder.parameters()).dtype
        vis = self.projector(self.encoder(
            pixel_values.to(dtype=encoder_dtype)).to(
                self.projector.token_proj.weight.dtype))
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

