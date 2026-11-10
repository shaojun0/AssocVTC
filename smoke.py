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

# ═══════════════════════════════════════════════════════════════
# ⑤ 文本任务零视觉 token   ⑥ prompt 生成边界   ⑦ 静态防回退
# ═══════════════════════════════════════════════════════════════

def check_text_rows(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """has_image=False ⇒ 该样本 8 个预算行都取 0 个视觉 token。"""
    z_s = torch.randn(BATCH, NUM_SPECIALS, ENC_DIM,
                      generator=torch.Generator().manual_seed(3))
    rows = expand_budget_rows(
        model.projector(z_s), model.lm.get_input_embeddings()(batch["input_ids"]),
        batch["attention_mask"], batch["labels"], range(1, 9),
        has_image=torch.tensor([True, False]))
    out = model.forward_rows(rows)
    text_pos = rows.ans_pos[BATCH * 8 - 8:]                  # 文本样本的 8 行
    checker.check("text row: 8 个预算的 ans_pos 相同（零视觉 token）",
                  bool((text_pos == text_pos[0]).all()), str(text_pos[0].tolist()))
    expected = int((batch["labels"][1] != IGNORE_INDEX).to(torch.int8).argmax())
    checker.check("text row: ans_pos[0] == label_pos-1",
                  bool(text_pos[0][0] == expected - 1), str(text_pos[0].tolist()))
    ps = out["per_budget_ce_per_sample"][1]
    checker.close("text row: 8 个预算的 CE 完全一致", ps, ps[0:1].expand_as(ps), 1e-6)
    ps_img = out["per_budget_ce_per_sample"][0]
    checker.check("image row: k=1 与 k=8 的 CE 不同",
                  abs(float(ps_img[0] - ps_img[-1])) > 1e-6,
                  f"Δ={float(ps_img[0] - ps_img[-1]):.2e}")


def check_text_only(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """纯文本 batch：`pixel_values=None` ⇒ 不跑 encoder，8 个预算行全 k=0。"""
    calls = []
    handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    try:
        out = model(input_ids=batch["input_ids"], pixel_values=None,
                    attention_mask=batch["attention_mask"], labels=batch["labels"])
    finally:
        handle.remove()
    checker.check("text-only: pixel_values=None 不跑 encoder", not calls)
    ref = model(input_ids=batch["input_ids"],
                pixel_values=torch.zeros(BATCH, 3, 448, 448),
                attention_mask=batch["attention_mask"], labels=batch["labels"],
                has_image=torch.zeros(BATCH, dtype=torch.bool))
    checker.close("text-only: forward(None) == 零图 + has_image=False",
                  out["loss"], ref["loss"], CLEAN_TOL)
    ps = out["per_budget_ce"]
    checker.close("text-only: 8 个预算 CE 完全一致",
                  ps, ps[0:1].expand_as(ps), CLEAN_TOL)


def check_row_dedup(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """无图样本 V 行逐位相同 ⇒ 只算 1 行；CE/loss/梯度与全量展开等价。"""
    z_s = torch.randn(BATCH, NUM_SPECIALS, ENC_DIM,
                      generator=torch.Generator().manual_seed(3))
    vis = model.projector(z_s).detach()
    has_image = torch.tensor([True, False])
    text = model.lm.get_input_embeddings()(batch["input_ids"]).detach().requires_grad_(True)

    def run(dedup: bool):
        rows = expand_budget_rows(
            vis, text, batch["attention_mask"], batch["labels"], range(1, 9),
            has_image=has_image)
        if not dedup:
            rows = rows._replace(unique_index=None, row_index=None)  # 全量行（旧路径）
        out = model.forward_rows(rows)
        out["loss"].backward()
        grad, text.grad = text.grad.clone(), None
        return rows, out, grad

    rows, dedup_out, dedup_grad = run(True)
    _, full_out, full_grad = run(False)
    checker.check("dedup: 16 行里只送 9 行（8 图行 + 1 文本行）",
                  rows.unique_index is not None and rows.unique_index.numel() == 9,
                  str(None if rows.unique_index is None else rows.unique_index.tolist()))
    checker.close("dedup: row_ce == 全量展开", dedup_out["row_ce"], full_out["row_ce"], 1e-6)
    checker.close("dedup: loss == 全量展开", dedup_out["loss"], full_out["loss"], 1e-6)
    checker.check("dedup: 对文本 embedding 的梯度 == 全量展开",
                  torch.allclose(dedup_grad, full_grad, atol=1e-6, rtol=1e-5),
                  f"max|Δ|={float((dedup_grad - full_grad).abs().max()):.2e}")


def check_prompt_lens(checker: Checker) -> None:
    """eval.prompt_lens：answer 首 token 位置 = prompt 长度；全 -100 必须报错。"""
    import eval as E
    labels = torch.full((2, 10), IGNORE_INDEX, dtype=torch.long)
    labels[0, 3:] = torch.arange(7)
    labels[1, 1:4] = torch.tensor([5, 6, 7])
    checker.check("prompt_lens = 首个非 -100 位置",
                  E.prompt_lens(labels).tolist() == [3, 1],
                  str(E.prompt_lens(labels).tolist()))
    try:
        E.prompt_lens(torch.full((1, 5), IGNORE_INDEX, dtype=torch.long))
        checker.check("prompt_lens: 无 answer 时 AssertionError", False)
    except AssertionError:
        checker.check("prompt_lens: 无 answer 时 AssertionError", True)


def check_static(checker: Checker) -> None:
    """A/B 两处入口删除后不得再出现（防回退）。"""
    import pathlib
    checker.check("model 不再有 generate_budget",
                  not hasattr(VLM8Budget, "generate_budget"))
    source = pathlib.Path(__file__).with_name("eval.py").read_text(encoding="utf-8")
    checker.check("eval.py 无 --gen-protocol", "--gen-protocol" not in source)
    checker.check("eval.py 产物写 gen_protocol", '"gen_protocol"' in source)
    cfg = VLM8BudgetConfig(encoder_ckpt="", qwen_dir="")
    checker.check("HF 加载契约: config.auto_map + _auto_class=AutoModel",
                  cfg.auto_map.get("AutoModel") == "model.VLM8Budget"
                  and VLM8Budget._auto_class == "AutoModel")
    checker.check("requirements.txt 声明 jinja2（apply_chat_template 依赖）",
                  "jinja2" in pathlib.Path(__file__).with_name(
                      "requirements.txt").read_text(encoding="utf-8"))
    data_source = pathlib.Path(__file__).with_name("data.py").read_text(encoding="utf-8")
    checker.check("data.py 走 apply_chat_template（不手搓 ChatML）",
                  "apply_chat_template" in data_source
                  and "<|im_start|>" not in data_source
                  and "<|im_end|>" not in data_source)


def main() -> int:
    """跑七项检查，返回 0/1。"""
    print(f"vlm8b smoke: CPU tensor logic  torch={torch.__version__}")
    torch.manual_seed(0)                    # 固定替身权重：否则 1/20 概率 truncation 阈值随机失败
    model = VLM8Budget(VLM8BudgetConfig(encoder_ckpt="", qwen_dir=""),
                       encoder=_FakeEncoder(), lm=_TinyLM()).eval()
    checker, batch = Checker(), make_batch()
    with torch.no_grad():
        check_shapes(checker, model, batch)
        check_shift(checker, model, batch)
        check_isolation(checker, model, batch)
        check_pad(checker, model, batch)
        check_text_rows(checker, model, batch)
        check_text_only(checker, model, batch)
    check_row_dedup(checker, model, batch)      # 需要梯度，放在 no_grad 之外
    check_prompt_lens(checker)
    check_static(checker)
    return checker.summary()

if __name__ == "__main__":
    sys.exit(main())
