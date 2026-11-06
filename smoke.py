"""
vlm8b 本机自检（CPU，不加载 Qwen）
==================================

**linear（基线，原 4 项）**：
  ① 投影/展开形状  ② shift 等价（answer-only CE == 标准因果 CE，tol 1e-5）
  ③ 行独立/截断隔离  ④ pad 隔离

**loop（本分支新增，⑤–⑧）**：
  ⑤ 循环读出形状 (B,8,2560)（`--loop-head mlp` 与 `transformer` 都测）
  ⑥ 前缀因果性：`z_s[j]` 只影响 `vis[j//block ..]`（行 k 的依赖范围 = 前 k*block 个 register）
  ⑦ BPTT 信用分配：末 token loss 对 `q_bank[0]` / 共享栈有梯度；detach 版为 0
  ⑧ 循环读出端到端 forward（per-budget CE 有限），并复用 ② 的 shift 等价
  ⑨ 阶段2 ckpt 的 `decoder.*` 热启动加载器（不载入 head）

  注：⑤–⑦ 对两种输出头各跑一遍 —— `head=transformer` 内部是 causal self-attn，
  若改成双向，⑥ 会立刻 FAIL（token 0 会看到后段 register，破坏预算前缀语义）。

用法：`python3 smoke.py`（0=全 PASS，1=有 FAIL）；模型用极小替身（不加载 Qwen）。
"""
import math
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (IGNORE_INDEX, LoopVisualReadout, VLM8Budget, VLM8BudgetConfig,
                   expand_budget_rows, load_stage2_decoder)

ENC_DIM, NUM_SPECIALS, LM_HIDDEN = 1744, 32, 2560
VOCAB, TEXT_LEN, BATCH, HEADS = 64, 17, 2, 8
CLEAN_TOL, SHIFT_TOL = 1e-7, 1e-5
LOOP_DIM, LOOP_DEPTH, LOOP_BLOCK = 64, 2, 4


# ═══════════════════════════════════════════════════════════════
# 极小替身（CPU；不依赖 Qwen / 阶段2 ckpt）
# ═══════════════════════════════════════════════════════════════

class _FakeEncoder(nn.Module):
    """可控 `(z_cls, z_s)`：token 表 × 像素均值增益（不同图不同 z_s）。"""

    def __init__(self):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, NUM_SPECIALS, ENC_DIM) * 0.02)
        self.gain = nn.Linear(3, 1)

    def forward(self, pixel_values: torch.Tensor):
        z_s = self.tokens.expand(pixel_values.size(0), -1, -1) * (
            1.0 + self.gain(pixel_values.mean(dim=(2, 3)))).unsqueeze(-1)
        return z_s[:, :1], z_s


class _Backbone(nn.Module):
    """2560 维 1 层 causal self-attn（2D mask：causal + key padding），残差直连。"""

    def __init__(self, dim: int = LM_HIDDEN, heads: int = HEADS):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads

    def forward(self, inputs_embeds: torch.Tensor, attention_mask=None,
                use_cache: bool = False):
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)
        batch, length, dim = inputs_embeds.shape
        q, k, v = self.qkv(inputs_embeds).reshape(
            batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim // self.heads)
        causal = torch.ones(length, length, dtype=torch.bool,
                            device=inputs_embeds.device).tril()
        scores = scores.masked_fill(
            ~(causal & attention_mask.bool()[:, None, None, :]), -1e9)
        attn = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, dim)
        return SimpleNamespace(last_hidden_state=inputs_embeds + self.proj(attn))


class _TinyLM(nn.Module):
    """`get_input_embeddings` / `model` / `lm_head` 的最小替身。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, LM_HIDDEN)
        self.model = _Backbone()
        self.lm_head = nn.Linear(LM_HIDDEN, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.embed


# ═══════════════════════════════════════════════════════════════
# 自检框架 / 夹具
# ═══════════════════════════════════════════════════════════════

class Checker:
    """记录 PASS/FAIL，最后给出汇总与退出码。"""

    def __init__(self):
        self.passed = self.failed = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.passed, self.failed = self.passed + bool(condition), self.failed + (not condition)
        print(f"[{'PASS' if condition else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))

    def close(self, name, actual: torch.Tensor, expected: torch.Tensor, tol: float):
        delta = float((actual.detach() - expected.detach()).abs().max())
        self.check(name, delta <= tol, f"max|Δ|={delta:.2e} tol={tol:g}")

    def summary(self) -> int:
        print(f"\nRESULT: {'PASS' if not self.failed else 'FAIL'}  "
              f"(passed={self.passed}, failed={self.failed})")
        return 0 if not self.failed else 1


def make_batch() -> dict:
    """2 张图：图 0 answer 在文本 13..16；图 1 answer 10..12 + 文本 right-pad 4。"""
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (BATCH, TEXT_LEN), generator=generator)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, 13:17], labels[1, 10:13] = input_ids[0, 13:17], input_ids[1, 10:13]
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 13:17] = 0
    input_ids[1, 13:17] = 0
    return {"pixel_values": torch.randn(BATCH, 3, 448, 448, generator=generator),
            "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def fixed_z(batch_size: int = BATCH, seed_s: int = 3, seed_cls: int = 5):
    """固定种子的 (z_cls, z_s)：同一测试内多次调用必须逐位一致。"""
    z_s = torch.randn(batch_size, NUM_SPECIALS, ENC_DIM,
                      generator=torch.Generator().manual_seed(seed_s))
    z_cls = torch.randn(batch_size, 1, ENC_DIM,
                        generator=torch.Generator().manual_seed(seed_cls))
    return z_cls, z_s


def build_rows(model: VLM8Budget, batch: dict, z_cls=None, z_s=None):
    if z_cls is None or z_s is None:
        z_cls, z_s = fixed_z()
    return expand_budget_rows(
        model.visual_tokens(z_cls, z_s), model.lm.get_input_embeddings()(batch["input_ids"]),
        batch["attention_mask"], batch["labels"], range(1, 9))


# ═══════════════════════════════════════════════════════════════
# ① 形状   ② shift 等价（P0 防回退，linear/loop 都跑）
# ═══════════════════════════════════════════════════════════════

def check_shapes(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """投影 / 8 行展开的形状与 answer 位置结构。"""
    rows = build_rows(model, batch)
    checker.check("projector (2,32,1744) → rows (16, 8+T, 2560)",
                  tuple(rows.embeds.shape) == (BATCH * 8, 8 + TEXT_LEN, LM_HIDDEN))
    checker.check("ans_pos: image0 k=1 = [13,14,15,16]（= label−1+k）",
                  rows.ans_pos[0].tolist() == [13, 14, 15, 16],
                  str(rows.ans_pos[0].tolist()))


def check_shift(checker: Checker, model: VLM8Budget, batch: dict,
                tag: str = "") -> None:
    """answer-only CE == 标准因果 CE(logits[:, :-1], labels[:, 1:]) 按行 mean。"""
    rows = build_rows(model, batch)
    hidden = model.lm.model(inputs_embeds=rows.embeds,
                            attention_mask=rows.attention_mask).last_hidden_state
    logits = model.lm.lm_head(hidden).float()
    token_ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB), rows.labels[:, 1:].reshape(-1),
        reduction="none", ignore_index=IGNORE_INDEX).reshape(rows.embeds.size(0), -1)
    counts = (rows.labels[:, 1:] != IGNORE_INDEX).sum(dim=1).clamp(min=1)
    checker.close(f"shift equivalence{tag}: row_ce == CE(logits[:, :-1], labels[:, 1:])",
                  model.forward_rows(rows)["row_ce"], token_ce.sum(dim=1) / counts,
                  SHIFT_TOL)


# ═══════════════════════════════════════════════════════════════
# ③ 行独立 / 截断隔离   ④ pad 隔离
# ═══════════════════════════════════════════════════════════════

def check_isolation(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """只看前 k 个 visual token；行与行是独立 batch 行。"""
    z_cls, z_s = fixed_z()
    base = model.forward_rows(build_rows(model, batch, z_cls, z_s))
    changed = z_s.clone()
    changed[:, 4] += torch.randn(changed[:, 4].shape,
                                 generator=torch.Generator().manual_seed(7))
    after = model.forward_rows(build_rows(model, batch, z_cls, changed))
    checker.close("truncation: z_s[4] change → k=1 CE unchanged",
                  after["per_budget_ce"][:1], base["per_budget_ce"][:1], CLEAN_TOL)
    delta = float(after["per_budget_ce"][1] - base["per_budget_ce"][1])
    checker.check("truncation: z_s[4] change → k=2 CE changed", abs(delta) > 1e-4,
                  f"Δ={delta:.2e}")
    rows = build_rows(model, batch, z_cls, z_s)
    rows.embeds[0] = torch.randn_like(rows.embeds[0])
    checker.close("row independence: k=1 行被改 → k=8 行 CE unchanged",
                  model.forward_rows(rows)["row_ce"][7:8], base["row_ce"][7:8],
                  CLEAN_TOL)


def check_pad(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """padding 位置的 embedding 不影响任何行。"""
    rows = build_rows(model, batch)
    base = model.forward_rows(rows)["row_ce"]
    pad = rows.attention_mask == 0
    assert bool(pad.any()), "没有 padding 位置，pad 隔离检查无效"
    rows.embeds[pad] = torch.randn_like(rows.embeds[pad])
    checker.close("pad isolation: pad embedding change → all row CE unchanged",
                  model.forward_rows(rows)["row_ce"], base, CLEAN_TOL)


# ═══════════════════════════════════════════════════════════════
# ⑤ 循环读出形状   ⑥ 前缀因果性   ⑦ BPTT   ⑧ 端到端 forward
# ═══════════════════════════════════════════════════════════════

def make_loop_model(head: str = "mlp") -> VLM8Budget:
    """CPU 用的小循环读出（dim=64, depth=2, block=4；不读 ckpt）。"""
    return VLM8Budget(
        VLM8BudgetConfig(encoder_ckpt="", qwen_dir="", readout="loop",
                         loop_dim=LOOP_DIM, loop_depth=LOOP_DEPTH, loop_heads=HEADS,
                         loop_init="none", loop_block=LOOP_BLOCK, loop_head=head),
        encoder=_FakeEncoder(), lm=_TinyLM()).eval()


def check_loop_shapes(checker: Checker, model: VLM8Budget, tag: str = "") -> None:
    """(z_cls, z_s) → (B,8,2560)；读窗口覆盖全部 32 个 register。"""
    z_cls, z_s = fixed_z()
    vis = model.visual_tokens(z_cls, z_s)
    checker.check(f"loop readout{tag}: (B,1,·)+(B,32,·) → vis (2,8,2560)",
                  tuple(vis.shape) == (BATCH, 8, LM_HIDDEN))
    checker.check(f"loop head{tag}: kind == {model.projector.head_kind!r}",
                  model.projector.head_kind in ("mlp", "transformer"))
    windows = model.projector.windows
    checker.check(f"loop windows{tag}: 8 步覆盖 32 register（末窗口 hi=32）",
                  windows[-1][1] == NUM_SPECIALS and len(windows) == 8,
                  str(windows))


def check_loop_causality(checker: Checker, model: VLM8Budget, tag: str = "") -> None:
    """前缀因果性：z_s[j] 只影响 vis[j//block ..]（行 k 依赖前 k*block 个 register）。"""
    z_cls, z_s = fixed_z()
    base = model.visual_tokens(z_cls, z_s)
    for reg_index, first_affected in ((4, 1), (20, 5)):
        changed = z_s.clone()
        changed[:, reg_index] += 5.0
        after = model.visual_tokens(z_cls, changed)
        deltas = [float((after[:, t] - base[:, t]).abs().max()) for t in range(8)]
        head = max(deltas[:first_affected])
        tail = min(deltas[first_affected:])
        checker.check(f"loop causality{tag}: z_s[{reg_index}] → vis[:{first_affected}] unchanged",
                      head < CLEAN_TOL, f"max|Δ|={head:.2e}")
        checker.check(f"loop causality{tag}: z_s[{reg_index}] → vis[{first_affected}:] changed",
                      tail > 1e-4, f"min|Δ|={tail:.2e}")


def check_loop_grad(checker: Checker, model: VLM8Budget, tag: str = "") -> None:
    """末 token loss 对 step-1 的 query / 共享栈有梯度（BPTT）；detach 版必须为 0。

    探针用 **pre-head 状态**（临时把 head 换成 Identity）：`head=transformer` 的
    causal self-attn 会让 token 8 直接读到 token 0，掩盖"carry 是否被 detach"，
    故测信用分配时必须绕过 head，只看循环 carry 这条路径。
    """
    z_cls, z_s = fixed_z()
    head_backup = model.projector.head
    model.projector.head = nn.Identity()
    try:
        model.train()
        model.zero_grad()
        # 注意：不能用 .sum()（pre-head 状态最后一层是 LayerNorm，特征维求和恒为 0）
        model.visual_tokens(z_cls, z_s)[:, -1].pow(2).mean().backward()
        g_q = float(model.projector.q_bank.grad[0].norm())
        g_stack = float(model.projector.stack.layers[0].self_attn.in_proj_weight.grad.norm())
        checker.check(f"loop BPTT{tag}: 末 token loss → grad q_bank[0] > 0", g_q > 0,
                      f"{g_q:.3e}")
        checker.check(f"loop BPTT{tag}: 末 token loss → grad 共享栈 layer0 > 0",
                      g_stack > 0, f"{g_stack:.3e}")
        model.zero_grad()
        model.projector(z_cls, z_s, detach_carry=True)[:, -1].pow(2).mean().backward()
        g_detach = float(model.projector.q_bank.grad[0].norm())
        checker.check(f"loop detach{tag}: carry detach → grad q_bank[0] == 0",
                      g_detach == 0.0, f"{g_detach:.3e}")
    finally:
        model.projector.head = head_backup
    model.eval()


def check_loop_forward(checker: Checker, model: VLM8Budget, batch: dict,
                       tag: str = "") -> None:
    """端到端：循环读出 → 8 行展开 → LM → per-budget CE。"""
    out = model(**batch)
    per_budget = out["per_budget_ce"]
    checker.check(f"loop forward{tag}: per_budget_ce shape == (8,)",
                  tuple(per_budget.shape) == (8,), str(tuple(per_budget.shape)))
    checker.check(f"loop forward{tag}: loss finite",
                  bool(torch.isfinite(out["loss"])), f"loss={float(out['loss']):.4f}")
    checker.check(f"loop forward{tag}: row_ce shape == (16,)",
                  tuple(out["row_ce"].shape) == (BATCH * 8,))


# ═══════════════════════════════════════════════════════════════
# ⑨ 阶段2 ckpt 的 decoder.* 热启动加载器（不依赖真 ckpt：用合成 blob）
# ═══════════════════════════════════════════════════════════════

def check_loop_warmstart(checker: Checker) -> None:
    """只载入前 depth 层 + mem/norm；query_base / pixel_head / 第 3+ 层必须跳过。"""
    src = LoopVisualReadout(enc_dim=ENC_DIM, dim=LOOP_DIM, depth=4, heads=HEADS,
                            block=LOOP_BLOCK)
    src_state = src.state_dict()
    blob = {"model": {("decoder." + k): v for k, v in src_state.items()
                      if k.startswith(("mem_", "stack.", "norm_out."))}}
    blob["model"]["decoder.query_base"] = torch.zeros(1)        # 输出目标绑定 → 跳过
    blob["model"]["decoder.pixel_head.weight"] = torch.zeros(3)  # 像素头 → 跳过
    dst = LoopVisualReadout(enc_dim=ENC_DIM, dim=LOOP_DIM, depth=2, heads=HEADS,
                            block=LOOP_BLOCK)
    stats = load_stage2_decoder(dst, blob)
    same = all(torch.equal(dst.stack.layers[i].self_attn.in_proj_weight,
                           src.stack.layers[i].self_attn.in_proj_weight)
               for i in range(2))
    checker.check("loop warm start: 前 2 层与 ckpt 逐位一致", same)
    checker.check("loop warm start: 第 3+ 层 / query_base / pixel_head 被跳过",
                  stats["skipped"] >= 3, str(stats))
    checker.check("loop warm start: mem_proj/mem_pos 已载入",
                  torch.equal(dst.mem_pos, src.mem_pos))


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    """linear 4 项 + loop 4 组检查，返回 0/1。"""
    print(f"vlm8b smoke: CPU tensor logic  torch={torch.__version__}")
    torch.manual_seed(0)
    linear = VLM8Budget(VLM8BudgetConfig(encoder_ckpt="", qwen_dir=""),
                        encoder=_FakeEncoder(), lm=_TinyLM()).eval()
    checker, batch = Checker(), make_batch()
    with torch.no_grad():
        check_shapes(checker, linear, batch)
        check_shift(checker, linear, batch)
        check_isolation(checker, linear, batch)
        check_pad(checker, linear, batch)
        for head in ("mlp", "transformer"):
            loop = make_loop_model(head)
            tag = f" [{head}]"
            check_loop_shapes(checker, loop, tag)
            check_loop_causality(checker, loop, tag)
            check_loop_forward(checker, loop, batch, tag)
            check_shift(checker, loop, batch, tag=f" (loop/{head})")
    for head in ("mlp", "transformer"):
        check_loop_grad(checker, make_loop_model(head), f" [{head}]")
    check_loop_warmstart(checker)
    return checker.summary()


if __name__ == "__main__":
    sys.exit(main())
