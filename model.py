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
        # 必须是实例属性（类属性不会被 to_dict 序列化）：保存时自定义代码随 ckpt 拷贝，
        # `AutoModel.from_pretrained(ckpt, trust_remote_code=True)` 靠它找到建模类。
        # 加载时 `encoder` 须在 sys.path 上（即从本仓根目录运行），`encoder_ckpt` /
        # `qwen_dir` 须指向有效路径——ckpt 本身不自包含这两个子模型。
        self.auto_map = {"AutoConfig": "model.VLM8BudgetConfig",
                         "AutoModel": "model.VLM8Budget"}


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
    has_image: Optional[Tensor] = None,
) -> BudgetRows:
    """`vis (B,8,H)` + `text (B,T,H)` → 每图 V 行 `[vis[:, :n]; text]`（right-pad）。

    `has_image[b]=False` ⇒ 该样本所有预算行都取 0 个视觉 token（文本任务零视觉前缀）。
    """
    B, T, H = text_embeds.shape
    ks = [int(k) for k in budget_ks]
    assert ks and min(ks) >= 0 and max(ks) <= vis_embeds.size(1), f"预算越界: {ks}"
    V, L = len(ks), max(ks) + T

    # 每行实际视觉 token 数 kb[b][vi]：文本样本（has_image=False）恒为 0。
    kb = torch.tensor(ks, device=text_embeds.device, dtype=torch.long).view(1, V).expand(B, V).clone()
    if has_image is not None:
        kb = kb * has_image.to(device=text_embeds.device, dtype=torch.long).view(B, 1)

    embeds = text_embeds.new_zeros((B, V, L, H))
    masks = text_mask.new_zeros((B, V, L))
    row_labels = labels.new_full((B, V, L), IGNORE_INDEX)
    kb_list = kb.tolist()                       # 只同步一次；B*V ≤ 32
    for vi in range(V):
        for b in range(B):
            n = kb_list[b][vi]                  # 文本样本 n=0 ⇒ 零视觉 token
            if n:
                embeds[b, vi, :n] = vis_embeds[b, :n]
                masks[b, vi, :n] = 1
            embeds[b, vi, n:n + T] = text_embeds[b]
            masks[b, vi, n:n + T] = text_mask[b]
            row_labels[b, vi, n:n + T] = labels[b]

    valid = labels != IGNORE_INDEX
    runs = (valid[:, 1:] & ~valid[:, :-1]).sum(dim=1) + valid[:, 0].long()
    assert int(runs.max()) <= 1, "answer 标签不是连续单段（多段会让 label_pos 跨过非答案 token）"
    starts = valid.long().argmax(dim=1)         # 每行 answer 起始（连续段）
    assert int(starts.min()) >= 1, "answer 从文本第 0 位开始，没有 prompt 末 token 可预测"
    counts = valid.sum(dim=1)
    A = int(counts.max())
    j = torch.arange(A, device=labels.device).unsqueeze(0)
    label_pos = starts.unsqueeze(1) + j                          # (B,A)
    ans_valid = j < counts.unsqueeze(1)
    targets = labels.gather(
        1, label_pos.clamp(max=T - 1)).masked_fill(~ans_valid, IGNORE_INDEX)

    # 标准因果 shift：预测位置 = label 位置 − 1；再用 +kb 换成行内绝对下标。
    # ⚠️ 不能写成 label 位置本身（hidden 已看见该 token ⇒ 目标泄露）。
    pos = (label_pos - 1).unsqueeze(1) + kb.unsqueeze(-1)
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
# AssocVTCEncoder — 阶段2 编码器（全量微调，不冻结）
# ═══════════════════════════════════════════════════════════════

class AssocVTCEncoder(nn.Module):
    """从阶段2 ckpt 恢复 `StandardAttentionEncoder`，输出 `z_s (B,32,1744)`。

    **冻结语义**（详见 `freeze.py` / `MUSA_FREEZE.md`）：
    - `freeze_encoder("all")` 后本模块 `_trainable=False`，`forward` 走 `no_grad`
      并 `detach` 输出——整段冻结时不再为 encoder 建图，backward 直接跳过它。
    - 部分冻结（只冻 stem / 前 N 个 block）时 `_trainable=True`：图必须穿过未冻层，
      冻结层由 `requires_grad=False` 自然截断梯度，不需要额外分支。

    **ZeRO-3 不适用**：本 encoder 的 `cls_token` / `special_tokens` / `pos_embed` 是
    挂在自己身上的**裸 Parameter**（不是子模块参数）。ZeRO-3 只在模块边界 all-gather，
    裸参数拿不到分区内容 ⇒ 前向报 `size of tensor a (0) ... b (1744)`。`GatheredParameters`
    手动 gather 也无效（DS 的 pre/post hook 会在子模块前后重新分区）。所以
    `train.py` 在「zeRO-3 + 冻结 encoder」时直接 fail-fast，请用 `ds_zero2.json`。
    """

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
        # v5 的 from_pretrained 在 meta 设备上建模型：此时 copy 到 meta 参数是 no-op
        # （还会 warning），权重由外层 checkpoint 统一灌入，跳过即可。
        if torch.empty(0).device.type != "meta":
            self.net.load_state_dict(state, strict=True)
        self._trainable = True

    def refresh_trainable(self) -> int:
        """按当前 requires_grad 重算 `_trainable`（`freeze.py` 改完冻结状态后调用）。"""
        numel = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self._trainable = numel > 0
        return numel

    def forward(self, pixel_values: Tensor) -> Tensor:
        """pixel_values (B,3,448,448) → z_s (B,32,1744)。

        整段冻结时的省图策略（两条路径都实测过，见 MUSA_FREEZE.md §2.4）：

        - **ZeRO-1/2 / 单卡**：`no_grad` + `detach`，encoder 不建 backward 图。
        - **ZeRO-3**：**不能** `no_grad`。ZeRO-3 在 module forward 前用 autograd
          Function 把分区参数 all-gather 回来，`no_grad` 下钩子失效 ⇒ 参数仍是空分区，
          前向直接 `size of tensor a (0) must match b (1744)`。此时照常建图
          （冻结参数 `requires_grad=False`，不会有梯度），只在输出上 `detach`。
        """
        if self._trainable:
            return self.net(pixel_values)[1]
        with torch.no_grad():
            z_s = self.net(pixel_values)[1]
        return z_s.detach()


# ═══════════════════════════════════════════════════════════════
# 视觉消融（评测用，不训练）
# ═══════════════════════════════════════════════════════════════

def apply_visual_mode(vis: Tensor, mode: str) -> Tensor:
    """`normal` / `null`（8 个 emb 置零；token 数不变、不等于没有图）/ `shuffle`（batch 内错位 ⇒ 看错图）。"""
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
    # 非空 ⇒ save_pretrained 调用 custom_object_save，把 model.py 拷进 ckpt；
    # `AutoModel.from_pretrained(ckpt, trust_remote_code=True)` 正是读这份代码。
    _auto_class = "AutoModel"
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
        if lm is not None:                                       # smoke / 调试用替身
            self.lm = lm
        elif torch.empty(0).device.type == "meta":               # transformers 5.x 的
            self.lm = self.build_lm()                            # from_pretrained 在
        else:                                                    # meta 设备上实例化模型
            self.lm = self.load_lm(grad_ckpt)
        self.loss_weighting, self.last_budget_ce = str(config.loss_weighting), None
        # transformers 5.x：`from_pretrained` 读 `all_tied_weights_keys`（由 post_init 登记），
        # 不调用会 AttributeError（5.22.2 实测）。但 **5.13 的 post_init 会调
        # init_weights() 把刚加载的 encoder/LM 权重全部重新随机初始化**（实测
        # 权重 Δ=0.12，等于白训），所以训练路径（权重已在子模块构造时加载）要屏蔽
        # 这次初始化；只有 meta 设备上的 from_pretrained 才让它照常跑。
        if hasattr(self, "get_expanded_tied_weights_keys"):
            if torch.empty(0).device.type == "meta":
                self.post_init()
            else:
                self._post_init_keep_weights()

    def _post_init_keep_weights(self) -> None:
        """跑 `post_init()` 但保住已加载的权重（transformers 5.13 会重初始化）。

        本项目权重在 `__init__` 里（encoder ckpt / Qwen `from_pretrained`）就已加载，
        `post_init()` 只需登记 `all_tied_weights_keys` 等元信息。把 `init_weights`
        临时置空即可——缺 key 的补齐由 `from_pretrained` / `load_state_dict` 负责。
        """
        def _noop(*args, **kwargs):
            return None

        self.init_weights = _noop
        try:
            self.post_init()
        finally:
            self.__dict__.pop("init_weights", None)     # 恢复类方法

    def load_lm(self, grad_ckpt: bool = False) -> nn.Module:
        """训练/手工加载：从 `qwen_dir` 读预训练权重（本函数内才 import Qwen）。"""
        lm = _import_qwen().from_pretrained(
            os.path.expanduser(self.config.qwen_dir), dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True, local_files_only=True)
        return self._finish_lm(lm, grad_ckpt)

    def build_lm(self) -> nn.Module:
        """只搭结构、不读权重：v5 在 meta 设备上初始化模型，嵌套 `from_pretrained`
        会 RuntimeError（5.22.2 实测），故权重留给外层 checkpoint 灌入。"""
        lm_cls = _import_qwen()
        lm = lm_cls(lm_cls.config_class.from_pretrained(
            os.path.expanduser(self.config.qwen_dir), local_files_only=True))
        return self._finish_lm(lm, grad_ckpt=False)

    def _finish_lm(self, lm: nn.Module, grad_ckpt: bool) -> nn.Module:
        """删 vision tower、按需开 grad ckpt、关 cache（两条构造路径共用）。"""
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
        assert ks and 0 <= ks[0] and ks[-1] <= VISUAL_TOKENS, f"预算越界: {ks}"
        return ks

    # ═══════════════════════════════════════════════════════════
    # encoder 冻结（策略/参数组在 freeze.py，机制说明在 MUSA_FREEZE.md）
    # ═══════════════════════════════════════════════════════════

    def freeze_encoder(self, spec: str = "none"):
        """按规格冻结 encoder（`all`/`stem`/`blocks=0-8`/`last=2`…），返回 FreezeReport。

        只改 `requires_grad` 与 `_trainable`；**不改 optimizer**。param group 必须在
        `freeze.build_param_groups()` 里按 `requires_grad` 现算，顺序不能反：
        先冻结 → 再建组（组里就没有冻结参数，DeepSpeed 不会因整组被剔空而崩）。
        """
        from freeze import freeze_encoder as _freeze
        return _freeze(self.encoder, spec)

    def train(self, mode: bool = True) -> "VLM8Budget":
        """`Trainer` 每 epoch 调 `model.train()`；冻结段必须保持 eval。

        nn.Module.train() 会把所有子模块（含冻结块里的 dropout）切回 train，
        而冻结对比实验要求「同一份权重、同一前向」，所以这里补一刀。
        """
        super().train(mode)
        if mode:
            from freeze import keep_frozen_eval
            keep_frozen_eval(self.encoder)
        return self

    def forward(self, input_ids: Tensor, pixel_values: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                labels: Optional[Tensor] = None,
                budget_ids: Optional[Sequence[int]] = None,
                budget_weights: Optional[Tensor] = None,
                visual_mode: str = "normal",
                has_image: Optional[Tensor] = None,
                **kwargs) -> Dict[str, object]:
        """训练 / teacher-forcing 前向（生成在 `eval.py`）。

        纯文本 batch / 纯文本 chat：`pixel_values=None`（或 `has_image` 全 False）⇒
        **不跑 encoder**，所有预算行 k=0，等价于不带视觉前缀的普通 LM 前向。
        `has_image[b]=False` ⇒ 该样本的预算行取 0 个视觉 token（文本任务）；混批时
        文本行可像 data.py 那样带一张零图占位，encoder 只对图行有贡献。

        纯文本**生成**不要走这里（forward 只算 CE）：直接 `model.lm.generate(...)`，
        见 `tools/eval_mix/bench_general_vlm8b.py:run_batch`。
        """
        B = input_ids.size(0)
        text_embeds = self.lm.get_input_embeddings()(input_ids)          # (B,T,2560)
        if pixel_values is None or (has_image is not None and not bool(has_image.any())):
            vis = text_embeds.new_zeros((B, VISUAL_TOKENS, text_embeds.size(-1)))
            has_image = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        else:
            encoder_dtype = next(self.encoder.parameters()).dtype
            z_s = self.encoder(pixel_values.to(dtype=encoder_dtype))     # (B,32,1744)
            vis = apply_visual_mode(
                self.projector(z_s.to(self.projector.token_proj.weight.dtype)),
                visual_mode)
        rows = expand_budget_rows(
            vis, text_embeds.to(vis.dtype), attention_mask, labels,
            self._budgets(budget_ids), has_image=has_image)
        return self.forward_rows(rows, budget_weights)

    def forward_rows(self, rows: BudgetRows,
                     budget_weights: Optional[Tensor] = None) -> Dict[str, object]:
        """对已展开的 `BudgetRows` 跑 LM + answer-only CE（smoke 也走这条路径）。"""
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


# ═══════════════════════════════════════════════════════════════
# 延迟导入（本机没有 Qwen3_5* / 编码器 ckpt，import 本模块不触发）
# ═══════════════════════════════════════════════════════════════

def _import_qwen():
    """延迟导入 `Qwen3_5ForConditionalGeneration`（remote transformers 5.22.1）。"""
    from transformers import Qwen3_5ForConditionalGeneration
    return Qwen3_5ForConditionalGeneration

