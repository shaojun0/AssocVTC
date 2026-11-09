"""
encoder 部分冻结：策略解析、参数组、DeepSpeed 生命周期
=====================================================

**为什么单独一个文件**：冻结不是「把 requires_grad 设 False」一句话。它同时牵动
三处语义，任何一处漏掉都会得到「看起来在跑、其实没按预期更新」的结果：

1. **优化器**：`requires_grad=False` 的参数必须从 param group 里去掉，否则
   DeepSpeed 在 engine 初始化时用 `is_optimized_parameter()`（= `requires_grad and
   numel > 0`）把它们剔出 ZeRO 分区；若某个 group 因此被整组剔空，stage 1/2 会在
   `stage_1_and_2.py` 取 `bit16_groups[i][0].dtype` 时 `IndexError`。本模块的
   `build_param_groups()` 直接不产生空组，从根上避开这个坑。
2. **前向图**：整段冻结时不要建图（`torch.no_grad()` + `detach()`），否则白算
   backward；部分冻结时图必须穿过未冻层，只能靠 `requires_grad` 自然截断。
3. **模块模式**：`Trainer` 每个 epoch 调 `model.train()`，会把冻结子模块里的
   dropout/BN 也切回 train。`keep_frozen_eval()` 在 `train()` 之后再把冻结部分
   按回 eval。

**DeepSpeed 的两条铁律**（本机 2×MTT S4000 + deepspeed 0.19.9 实测，见
`MUSA_FREEZE.md` §2）：

- 若冻结状态**永不改变**（静态冻结）→ 在 engine 初始化前冻结，并让 optimizer
  只拿到可训练参数。ZeRO-2 正常，optimizer state 也省了。
- 若冻结状态**中途会变**（E1 冻、E2 解）→ 必须在 engine 初始化**之后**再改
  `requires_grad`。初始化前冻掉整组，stage 2 直接 `IndexError`；stage 3 不报错，
  但该组不进优化器，解冻后**静默不更新**。

规格（CLI `--freeze-encoder`，逗号分隔组合）::

    none              不冻结（默认）
    all               整个 encoder
    stem              patch_embed + cls/special/pos embed（只训 blocks + norm）
    blocks=0-8        冻结第 0..8 个 block（含端点）
    blocks=0,1,2      点选
    last=2            只训最后 2 个 block + norm（= 冻结 stem 与 blocks[0:-2]）
    blocks=0-8,norm   组合：再冻末层 LayerNorm

`blocks` 下标是 `encoder.net.blocks` 的顺序下标，0 最靠输入。
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import re

import torch
import torch.nn as nn

# `blocks=0,2` 里的单个下标/区间片段（用于把逗号拆开的 token 合并回去）
_BLOCK_PIECE = re.compile(r"^\d+(-\d+)?$")

# encoder.net 里「词元化/位置」部分的参数名（AssocVTC 的 StandardAttentionEncoder 契约）
STEM_PARAMS = ("cls_token", "special_tokens", "pos_embed")
STEM_MODULES = ("patch_embed", "pos_drop")
BLOCKS_ATTR = "blocks"
NORM_ATTR = "norm"


# ═══════════════════════════════════════════════════════════════
# 策略
# ═══════════════════════════════════════════════════════════════

@dataclass
class FreezePlan:
    """要冻结的部分（按模块粒度，不按参数粒度）。"""

    all_: bool = False
    stem: bool = False
    blocks: Set[int] = field(default_factory=set)
    norm: bool = False
    depth: int = 0

    @property
    def active(self) -> bool:
        return self.all_ or self.stem or bool(self.blocks) or self.norm

    def describe(self) -> str:
        if self.all_:
            return f"all(encoder, {self.depth} blocks)"
        parts = []
        if self.stem:
            parts.append("stem")
        if self.blocks:
            ids = sorted(self.blocks)
            parts.append(f"blocks={','.join(str(i) for i in ids)}")
        if self.norm:
            parts.append("norm")
        return "+".join(parts) if parts else "none"


def _parse_block_ids(raw: str, depth: int) -> Set[int]:
    """`0-8` / `0,1,2` / `8` → {0..8}；越界直接报错（不静默截断）。"""
    ids: Set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_s, hi_s = piece.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"blocks 区间反了: {piece}")
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(piece))
    bad = [i for i in ids if not 0 <= i < depth]
    if bad:
        raise ValueError(f"blocks 下标越界 {sorted(bad)}（depth={depth}）")
    return ids


def parse_spec(spec: str, depth: int) -> FreezePlan:
    """解析冻结规格；未知 token / 越界给可执行的报错。"""
    spec = (spec or "none").strip().lower().replace(" ", "")
    plan = FreezePlan(depth=int(depth))
    if spec in ("", "none"):
        return plan
    # 顶层分隔符是 , 和 +；`blocks=0,2` 里的逗号属于同一 token，这里先合并回来
    raw = [t for t in spec.replace("+", ",").split(",") if t]
    tokens: List[str] = []
    for token in raw:
        if tokens and tokens[-1].startswith("blocks=") and _BLOCK_PIECE.match(token):
            tokens[-1] += "," + token
        else:
            tokens.append(token)
    for token in tokens:
        if token == "all":
            plan.all_ = True
        elif token == "stem":
            plan.stem = True
        elif token == "norm":
            plan.norm = True
        elif token.startswith("blocks="):
            plan.blocks |= _parse_block_ids(token[len("blocks="):], plan.depth)
        elif token.startswith("last="):
            keep = int(token[len("last="):])
            if not 0 <= keep <= plan.depth:
                raise ValueError(f"last={keep} 须在 0..{plan.depth}")
            plan.stem = True
            plan.blocks |= set(range(plan.depth - keep))
        else:
            raise ValueError(
                f"无法识别的冻结规格 {token!r}；可用: none/all/stem/norm/"
                f"blocks=a-b/blocks=a,b/last=N，逗号分隔组合")
    return plan


# ═══════════════════════════════════════════════════════════════
# 应用 / 报告
# ═══════════════════════════════════════════════════════════════

@dataclass
class FreezeReport:
    """冻结结果：给日志、单测和 `--freeze-encoder` 的回显用。"""

    spec: str
    frozen_params: int = 0
    trainable_params: int = 0
    frozen_numel: int = 0
    trainable_numel: int = 0
    frozen_names: List[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.frozen_numel > 0

    def summary(self) -> str:
        total = self.frozen_numel + self.trainable_numel
        pct = 100.0 * self.frozen_numel / max(total, 1)
        return (f"freeze={self.spec}: {self.frozen_params} params / "
                f"{self.frozen_numel/1e6:.2f}M frozen ({pct:.1f}%), "
                f"{self.trainable_params} params / {self.trainable_numel/1e6:.2f}M trainable")


def _iter_targets(root: nn.Module, plan: FreezePlan) -> Iterable[Tuple[str, object]]:
    """把 plan 展开成 (标签, 子模块或裸 Parameter) —— 后者的 requires_grad 要直接改。"""
    if plan.all_:
        yield "all", root
        return
    if plan.stem:
        for name in STEM_MODULES + STEM_PARAMS:
            if hasattr(root, name):
                yield name, getattr(root, name)
    blocks = getattr(root, BLOCKS_ATTR, None)
    if plan.blocks:
        if blocks is None:
            raise ValueError(f"{type(root).__name__} 没有 .{BLOCKS_ATTR}，无法按 block 冻结")
        for idx in sorted(plan.blocks):
            yield f"{BLOCKS_ATTR}.{idx}", blocks[idx]
    if plan.norm and hasattr(root, NORM_ATTR):
        yield NORM_ATTR, getattr(root, NORM_ATTR)


def encoder_net(encoder: nn.Module) -> nn.Module:
    """`AssocVTCEncoder` 包了一层 `.net`；直接传 StandardAttentionEncoder 也接受。"""
    return getattr(encoder, "net", encoder)


def _set_requires_grad(obj: object, flag: bool) -> None:
    """module → 递归；裸 Parameter（cls/special/pos embed）→ 直接改。"""
    if isinstance(obj, nn.Parameter):
        obj.requires_grad_(flag)
        return
    for param in obj.parameters(recurse=True):        # type: ignore[attr-defined]
        param.requires_grad_(flag)


def apply_plan(encoder: nn.Module, plan: FreezePlan, spec: str = "") -> FreezeReport:
    """先把 encoder 全部设为可训练，再按 plan 冻结——保证重复调用幂等。"""
    net = encoder_net(encoder)
    for param in net.parameters():
        param.requires_grad_(True)
    for _, obj in _iter_targets(net, plan):
        _set_requires_grad(obj, False)

    # 同步外层包装器的「是否还有可训练参数」（决定 forward 走不走 no_grad）
    if hasattr(encoder, "refresh_trainable"):
        encoder.refresh_trainable()

    report = FreezeReport(spec=spec or plan.describe())
    for name, param in net.named_parameters():
        if param.requires_grad:
            report.trainable_params += 1
            report.trainable_numel += param.numel()
        else:
            report.frozen_params += 1
            report.frozen_numel += param.numel()
            report.frozen_names.append(name)
    return report


def freeze_encoder(encoder: nn.Module, spec: str) -> FreezeReport:
    """按 CLI 规格冻结（`none` = 全部可训练）。"""
    net = encoder_net(encoder)
    depth = len(getattr(net, BLOCKS_ATTR, []))
    plan = parse_spec(spec, depth)
    return apply_plan(encoder, plan, spec=plan.describe())


def unfreeze_all(encoder: nn.Module) -> FreezeReport:
    """恢复全量可训练（中途解冻用）。"""
    return freeze_encoder(encoder, "none")


def keep_frozen_eval(encoder: nn.Module) -> int:
    """train() 之后把「参数全冻」的子模块再按回 eval；返回处理个数。

    `Trainer` 每 epoch 调 `model.train()` 会覆盖 `Module.eval()`，冻结段里的
    dropout（本仓 encoder 默认 drop_rate=0，但保留正确性）必须保持 eval 行为，
    否则同一份权重会给出两种前向结果，冻结对比实验就不可复现。
    """
    net = encoder_net(encoder)
    count = 0
    for module in net.modules():
        params = [p for p in module.parameters(recurse=False)]
        if params and not any(p.requires_grad for p in params):
            if module.training:
                module.eval()
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════
# param groups（DeepSpeed 安全）
# ═══════════════════════════════════════════════════════════════

def build_param_groups(
    groups_spec: Sequence[Tuple[nn.Module, float, float]],
    skip_frozen: bool = True,
) -> List[Dict[str, object]]:
    """`[(module, lr, wd), ...]` → AdamW param groups（每个模块拆成衰减/不衰减两组）。

    - 跳过 `requires_grad=False` 的参数（否则 DeepSpeed 会在 engine init 时剔除它们，
      整组被剔空就 `IndexError`）；
    - **丢弃空组**——这是「整段冻结 encoder」能直接在 ZeRO-1/2 上跑的关键；
    - 一个可训练参数都没有时直接报错（比 DS 的 `IndexError` 可读）。
    """
    groups: List[Dict[str, object]] = []
    for module, lr, wd in groups_spec:
        params = list(module.parameters())
        if skip_frozen:
            params = [p for p in params if p.requires_grad]
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": wd})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    if not groups:
        raise RuntimeError("没有任何可训练参数：冻结规格把所有参数都冻住了？")
    return groups


def trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]
