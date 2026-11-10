"""
vlm8b 本机自检（CPU，不加载 Qwen）
==================================

四项：① 两段式布局形状（prompt 共享 + 段）② shift 等价（两段式 == 把
prompt+段拼成一条序列的单趟前向，且把 `ans_pos` 改成 label 位置必须改变 CE）
③ 行独立/截断隔离 ④ pad 隔离。

用法：`python3 smoke.py`（0=全 PASS，1=有 FAIL）；模型用极小替身。
"""
import math, sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

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
    """2560 维 1 层 causal self-attn（2D mask：causal + key padding，支持 KV cache 与位置）。

    位置用可学习 embedding 而不是无位置——这样 `position_ids` 算错会被 smoke 抓到
    （两段式最重要的坑就是段的位置必须接在各自 prompt 长度之后）。
    """

    def __init__(self, dim: int = LM_HIDDEN, heads: int = HEADS, max_pos: int = 4096):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pos = nn.Embedding(max_pos, dim)
        self.heads = heads

    def forward(self, inputs_embeds: torch.Tensor, attention_mask=None,
                position_ids=None, use_cache: bool = False,
                past_key_values=None):
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1],
                                        device=inputs_embeds.device).unsqueeze(0)
        batch, length, dim = inputs_embeds.shape
        hidden = inputs_embeds + self.pos(position_ids)
        if attention_mask is None:
            attention_mask = torch.ones(batch, length, dtype=torch.long,
                                        device=hidden.device)
        q, k, v = self.qkv(hidden).reshape(
            batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        if past_key_values is None and use_cache:
            past_key_values = DynamicCache()
        if past_key_values is not None:                 # 追加到 cache（HF 语义：原地更新）
            k, v = past_key_values.update(k, v, 0)
        total = k.size(2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim // self.heads)
        causal = torch.ones(length, total, dtype=torch.bool,
                            device=hidden.device).tril(diagonal=total - length)
        scores = scores.masked_fill(
            ~(causal & attention_mask.bool()[:, None, None, :]), -1e9)
        attn = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, dim)
        return SimpleNamespace(last_hidden_state=hidden + self.proj(attn),
                               past_key_values=past_key_values)


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
    """两段式布局：prompt 跨预算共享；段 = [prompt 末 token; vis[:k]; answer]。"""
    rows = build_rows(model, batch)
    starts = (batch["labels"] != IGNORE_INDEX).int().argmax(dim=1).tolist()
    checker.check("prompt 段 = (B, max_prompt_len-1, 2560)（末 token 进段做锚点）",
                  tuple(rows.prompt.shape) == (BATCH, max(starts) - 1, LM_HIDDEN),
                  f"{tuple(rows.prompt.shape)} starts={starts}")
    checker.check("segment = (B*V, 1+8+A, 2560)",
                  tuple(rows.segment.shape) == (BATCH * 8, 1 + 8 + 4, LM_HIDDEN),
                  str(tuple(rows.segment.shape)))
    checker.check("ans_pos: image0 k=1 = [1,2,3,4]（= k+j，标签位置的前一位）",
                  rows.ans_pos[0].tolist() == [1, 2, 3, 4],
                  str(rows.ans_pos[0].tolist()))
    checker.check("row_ks: k=1..8 与文本样本 0",
                  rows.row_ks[:8].tolist() == list(range(1, 9)),
                  str(rows.row_ks[:8].tolist()))


def check_shift(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """两段式 == 把 prompt+段拼成一条序列跑同一个 LM（cache/位置/mask 全对才成立）。"""
    rows = build_rows(model, batch)
    R, P, L = rows.segment.size(0), rows.prompt.size(1), rows.segment.size(1)
    rows_b = torch.arange(R) // rows.num_budgets
    full = torch.cat([rows.prompt[rows_b], rows.segment], dim=1)
    mask = torch.cat([rows.prompt_mask[rows_b], rows.segment_mask], dim=1)
    pos = torch.cat([torch.arange(P).unsqueeze(0).expand(R, -1), rows.position_ids], dim=1)
    hidden = model.lm.model(inputs_embeds=full, attention_mask=mask,
                            position_ids=pos).last_hidden_state
    logits = model.lm.lm_head(hidden).float()
    labels_full = torch.full((R, P + L), IGNORE_INDEX, dtype=torch.long)
    keep = rows.ans_pos >= 0
    row_idx = torch.arange(R).unsqueeze(1).expand_as(rows.ans_pos)
    labels_full[row_idx[keep], (P + rows.ans_pos + 1)[keep]] = rows.ans_targets[keep]
    token_ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB), labels_full[:, 1:].reshape(-1),
        reduction="none", ignore_index=IGNORE_INDEX).reshape(R, -1)
    counts = (labels_full[:, 1:] != IGNORE_INDEX).sum(dim=1).clamp(min=1)
    checker.close("shift equivalence: 两段式 row_ce == 单趟全序列 CE",
                  model.forward_rows(rows)["row_ce"], token_ce.sum(dim=1) / counts,
                  SHIFT_TOL)
    rows_bad = rows._replace(ans_pos=rows.ans_pos + 1)      # 用标签位置取 hidden → 目标泄露
    checker.check("shift: 用标签位置（+1）必须改变 CE（防泄露）",
                  bool((model.forward_rows(rows_bad)["row_ce"]
                        - model.forward_rows(rows)["row_ce"]).abs().max() > 1e-3))


def check_prompt_mask(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """prompt 段只覆盖本样本的 prompt：短 prompt 样本的答案 token 不得进共享 prefill。

    这是 layout 改动最容易踩的坑——错的话答案段会在 cache 里看到自己的答案（目标泄露），
    而「两段式 == 单趟」的等价性检查用的是同一份 mask，抓不到。
    """
    rows = build_rows(model, batch)
    starts = (batch["labels"] != IGNORE_INDEX).int().argmax(dim=1)
    limit = starts - 1                                        # prompt 段合法长度
    pos = torch.arange(rows.prompt.size(1)).unsqueeze(0)
    checker.check("prompt_mask 只覆盖 < starts-1（不含答案 token）",
                  bool((rows.prompt_mask == (pos < limit.unsqueeze(1)).long()).all()),
                  f"starts={starts.tolist()}")
    base = model.forward_rows(rows)["row_ce"]
    leak = (pos >= limit.unsqueeze(1)).expand_as(rows.prompt_mask)   # 越界的答案位
    rows.prompt[leak] = torch.randn_like(rows.prompt[leak])
    checker.close("改共享 prompt 里越界的答案 embedding → CE 不变（无泄露）",
                  model.forward_rows(rows)["row_ce"], base, CLEAN_TOL)


def check_two_stage(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """prompt 只前向一次：LM 应该被调用 2 次（prefill + 段），而不是 1+V 次。"""
    calls = []
    handle = model.lm.model.register_forward_hook(lambda *args: calls.append(1))
    try:
        model.forward_rows(build_rows(model, batch))
    finally:
        handle.remove()
    checker.check("两段式: LM 前向 2 次（prefill 一次 + 段一次）", len(calls) == 2,
                  f"calls={len(calls)}")


def check_isolation(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """只看前 k 个 visual token；行与行是独立 batch 行。"""
    z_s = torch.randn(BATCH, NUM_SPECIALS, ENC_DIM,
                      generator=torch.Generator().manual_seed(3))
    base = model.forward_rows(build_rows(model, batch, z_s))
    changed = z_s.clone()
    changed[:, 4] += torch.randn(changed[:, 4].shape,
                                 generator=torch.Generator().manual_seed(7))
    after = model.forward_rows(build_rows(model, batch, changed))
    # projector 是 4:1 拼接 ⇒ z_s[4] 落在 vis token 1 上，只有 k>=2 的预算看得到它
    checker.close("truncation: z_s[4] 变 → k=1 预算 CE 不变",
                  after["per_budget_ce"][:1], base["per_budget_ce"][:1], CLEAN_TOL)
    delta = float(after["per_budget_ce"][1] - base["per_budget_ce"][1])
    checker.check("truncation: z_s[4] 变 → k=2 预算 CE 改变", abs(delta) > 1e-4,
                  f"Δ={delta:.2e}")
    rows = build_rows(model, batch, z_s)
    rows.segment[0] = torch.randn_like(rows.segment[0])      # 改 k=1 行
    checker.close("row independence: k=1 行被改 → 其余行 CE 不变",
                  model.forward_rows(rows)["row_ce"][1:], base["row_ce"][1:], CLEAN_TOL)


def check_pad(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """padding 位置的 embedding（prompt 右 pad + 段尾 pad）不影响任何行。"""
    rows = build_rows(model, batch)
    base = model.forward_rows(rows)["row_ce"]
    prompt_pad, segment_pad = rows.prompt_mask == 0, rows.segment_mask == 0
    assert bool(segment_pad.any()), "段的 padding 位置都不存在，检查无效"
    rows.prompt[prompt_pad] = torch.randn_like(rows.prompt[prompt_pad])
    rows.segment[segment_pad] = torch.randn_like(rows.segment[segment_pad])
    checker.close("pad isolation: pad embedding change → all row CE unchanged",
                  model.forward_rows(rows)["row_ce"], base, CLEAN_TOL)


def check_text_rows(checker: Checker, model: VLM8Budget, batch: dict) -> None:
    """has_image=False ⇒ 该样本所有预算行 k=0（段里只有锚点 + 答案）。"""
    z_s = torch.randn(BATCH, NUM_SPECIALS, ENC_DIM,
                      generator=torch.Generator().manual_seed(3))
    rows = expand_budget_rows(
        model.projector(z_s), model.lm.get_input_embeddings()(batch["input_ids"]),
        batch["attention_mask"], batch["labels"], range(1, 9),
        has_image=torch.tensor([True, False]))
    out = model.forward_rows(rows)
    text_pos = rows.ans_pos[BATCH * 8 - 8:]                  # 文本样本的 8 行
    checker.check("text row: 8 个预算的 ans_pos 相同且 = [0,1,2,-1]（k=0，末位无效）",
                  bool((text_pos == text_pos[0]).all())
                  and text_pos[0].tolist() == [0, 1, 2, -1],
                  str(text_pos[0].tolist()))
    checker.check("text row: row_ks 全 0", rows.row_ks[BATCH * 8 - 8:].tolist() == [0] * 8)
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
        check_two_stage(checker, model, batch)
        check_prompt_mask(checker, model, batch)
        check_text_only(checker, model, batch)
    check_prompt_lens(checker)
    check_static(checker)
    return checker.summary()

if __name__ == "__main__":
    sys.exit(main())
