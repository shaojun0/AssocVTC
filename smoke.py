"""
vlm8b 本机自检（CPU，不加载 Qwen）
==================================

四项：① 投影/展开形状 ② shift 等价（answer-only CE == 标准因果
`F.cross_entropy(logits[:, :-1], labels[:, 1:])` 按行 mean，tol 1e-5；把
`ans_pos` 改回 label 位置则必须 FAIL）③ 行独立/截断隔离 ④ pad 隔离。

用法：`python3 smoke.py`（0=全 PASS，1=有 FAIL）；模型用极小替身。
"""
import math, sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import IGNORE_INDEX, VLM8Budget, VLM8BudgetConfig, expand_budget_rows

ENC_DIM, NUM_SPECIALS, LM_HIDDEN = 1744, 32, 2560
VOCAB, TEXT_LEN, BATCH, HEADS = 64, 17, 2, 8
CLEAN_TOL, SHIFT_TOL = 1e-7, 1e-5

class _FakeEncoder(nn.Module):
    """可控 z_s：token 表 × 像素均值增益（不同图不同 z_s）。"""

    def __init__(self):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, NUM_SPECIALS, ENC_DIM) * 0.02)
        self.gain = nn.Linear(3, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.tokens.expand(pixel_values.size(0), -1, -1) * (
            1.0 + self.gain(pixel_values.mean(dim=(2, 3)))).unsqueeze(-1)

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

def build_rows(model: VLM8Budget, batch: dict, z_s: torch.Tensor = None):
    if z_s is None:
        z_s = torch.randn(BATCH, NUM_SPECIALS, ENC_DIM,
                          generator=torch.Generator().manual_seed(3))
    return expand_budget_rows(
        model.projector(z_s), model.lm.get_input_embeddings()(batch["input_ids"]),
        batch["attention_mask"], batch["labels"], range(1, 9))

# ═══════════════════════════════════════════════════════════════
# ① 形状   ② shift 等价（P0 防回退）
# ═══════════════════════════════════════════════════════════════

def check_shapes(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """投影 / 8 行展开的形状与 answer 位置结构。"""
    rows = build_rows(model, batch)
    checker.check("projector (2,32,1744) → rows (16, 8+T, 2560)",
                  tuple(rows.embeds.shape) == (BATCH * 8, 8 + TEXT_LEN, LM_HIDDEN))
    checker.check("ans_pos: image0 k=1 = [13,14,15,16]（= label−1+k）",
                  rows.ans_pos[0].tolist() == [13, 14, 15, 16],
                  str(rows.ans_pos[0].tolist()))

def check_shift(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """answer-only CE == 标准因果 CE(logits[:, :-1], labels[:, 1:]) 按行 mean。"""
    rows = build_rows(model, batch)
    hidden = model.lm.model(inputs_embeds=rows.embeds,
                            attention_mask=rows.attention_mask).last_hidden_state
    logits = model.lm.lm_head(hidden).float()
    token_ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB), rows.labels[:, 1:].reshape(-1),
        reduction="none", ignore_index=IGNORE_INDEX).reshape(rows.embeds.size(0), -1)
    counts = (rows.labels[:, 1:] != IGNORE_INDEX).sum(dim=1).clamp(min=1)
    checker.close("shift equivalence: row_ce == CE(logits[:, :-1], labels[:, 1:])",
                  model.forward_rows(rows)["row_ce"], token_ce.sum(dim=1) / counts,
                  SHIFT_TOL)

# ═══════════════════════════════════════════════════════════════
# ③ 行独立 / 截断隔离   ④ pad 隔离
# ═══════════════════════════════════════════════════════════════

def check_isolation(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """只看前 k 个 visual token；行与行是独立 batch 行。"""
    z_s = torch.randn(BATCH, NUM_SPECIALS, ENC_DIM,
                      generator=torch.Generator().manual_seed(3))
    base = model.forward_rows(build_rows(model, batch, z_s))
    changed = z_s.clone()
    changed[:, 4] += torch.randn(changed[:, 4].shape,
                                 generator=torch.Generator().manual_seed(7))
    after = model.forward_rows(build_rows(model, batch, changed))
    checker.close("truncation: z_s[4] change → k=1 CE unchanged",
                  after["per_budget_ce"][:1], base["per_budget_ce"][:1], CLEAN_TOL)
    delta = float(after["per_budget_ce"][1] - base["per_budget_ce"][1])
    checker.check("truncation: z_s[4] change → k=2 CE changed", abs(delta) > 1e-4, f"Δ={delta:.2e}")
    rows = build_rows(model, batch, z_s)
    rows.embeds[0] = torch.randn_like(rows.embeds[0])
    checker.close("row independence: k=1 行被改 → k=8 行 CE unchanged",
                  model.forward_rows(rows)["row_ce"][7:8], base["row_ce"][7:8], CLEAN_TOL)

def check_pad(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """padding 位置的 embedding 不影响任何行。"""
    rows = build_rows(model, batch)
    base = model.forward_rows(rows)["row_ce"]
    pad = rows.attention_mask == 0
    assert bool(pad.any()), "没有 padding 位置，pad 隔离检查无效"
    rows.embeds[pad] = torch.randn_like(rows.embeds[pad])
    checker.close("pad isolation: pad embedding change → all row CE unchanged",
                  model.forward_rows(rows)["row_ce"], base, CLEAN_TOL)

def main() -> int:
    """跑四项检查，返回 0/1。"""
    print(f"vlm8b smoke: CPU tensor logic  torch={torch.__version__}")
    model = VLM8Budget(VLM8BudgetConfig(encoder_ckpt="", qwen_dir=""),
                       encoder=_FakeEncoder(), lm=_TinyLM()).eval()
    checker, batch = Checker(), make_batch()
    with torch.no_grad():
        check_shapes(checker, model, batch)
        check_shift(checker, model, batch)
        check_isolation(checker, model, batch)
        check_pad(checker, model, batch)
    return checker.summary()

if __name__ == "__main__":
    sys.exit(main())
