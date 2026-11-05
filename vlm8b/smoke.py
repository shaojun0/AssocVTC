"""
vlm8b 本机自检（纯 CPU / 无 Qwen / 无 GPU）
============================================

只验证 **张量逻辑**，不加载 4B 模型、不读远端数据：

  1. 投影头形状：z_s (B,32,1744) → vis (B,8,2560)；N(0,0.02) 初始化。
  2. 8 行展开：行 k = [vis[:, :k]; text]，right-pad 到 max_k+T；
     labels 只在 answer 非 -100；answer 位置随 k 平移。
  3. collator：图片 batch 的 right-padding / labels 填充 / gt_name 透传。
  4. 四个不变量（设计 §7.2）：
       ① 行独立：单图单跑 vs 混批，per-budget CE 差 < 1e-4；
       ② 截断隔离：改 z_s 第 5 个 token 不影响 k=1（只影响 k≥2）；
       ③ pad 隔离：把 padding 位置 embedding 换成随机值，loss 不变；
       ④ 预算单调结构：行长度随 k 严格 +1、前缀嵌套、answer 位置单调平移。
  5. CE 遮蔽：answer-only gather CE 与"整序列 CE + ignore_index"逐位一致；
     loss = Σ w_k·CE_k（equal / inv-k）；反向能到 projector 与 encoder。

用法::

    cd vlm8b && python3 smoke.py                 # 默认 --logic-only
    python3 smoke.py --full --encoder-ckpt <ckpt> # 额外做 ckpt 形状检查（可选）

退出码：0 = 全部 PASS；1 = 有 FAIL / 异常。
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:                                            # script / package 两种运行方式
    from data import (
        EightBudgetCollator,
        SelfDistillSFTDataset,
        build_image_transform,
    )
    from model import (
        IGNORE_INDEX,
        AssocVTCEncoder,
        VLM8Budget,
        VLM8BudgetConfig,
        expand_budget_rows,
    )
except ImportError:                             # pragma: no cover
    from .data import (
        EightBudgetCollator,
        SelfDistillSFTDataset,
        build_image_transform,
    )
    from .model import (
        IGNORE_INDEX,
        AssocVTCEncoder,
        VLM8Budget,
        VLM8BudgetConfig,
        expand_budget_rows,
    )


# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

ENC_DIM = 1744
NUM_SPECIALS = 32
LM_HIDDEN = 2560
IMAGE_SIZE = 448
VOCAB = 64
TEXT_LEN = 17
BATCH = 2
TINY_INNER = 64
TINY_HEADS = 4
TINY_BLOCKS = 2
ROW_TOL = 1e-4


# ═══════════════════════════════════════════════════════════════
# 极小的替身模型（形状真实，算力极小）
# ═══════════════════════════════════════════════════════════════

class _FakeEncoder(nn.Module):
    """可控的 z_s 替身：可学习 token × 像素均值增益（不同图不同 z_s）。"""

    def __init__(self, num_specials: int = NUM_SPECIALS, enc_dim: int = ENC_DIM):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_specials, enc_dim) * 0.02)
        self.pixel_gain = nn.Linear(3, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.size(0)
        gain = 1.0 + self.pixel_gain(pixel_values.mean(dim=(2, 3)))   # (B,1)
        return self.tokens.expand(batch, -1, -1) * gain.unsqueeze(-1)


class _TinyAttention(nn.Module):
    """带 causal + padding mask 的单头组注意力（每行独立，2D mask 语义）。"""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor):
        batch, length, dim = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch, length, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                     # (3,B,H,L,D)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        causal = torch.ones(
            (length, length), dtype=torch.bool, device=hidden.device).tril()
        key_ok = attention_mask.to(torch.bool)[:, None, None, :]
        scores = scores.masked_fill(~(causal & key_ok), -1e9)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
        return self.proj(out)


class _TinyBlock(nn.Module):
    """pre-norm 的 attention + MLP 残差块。"""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _TinyAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor):
        hidden = hidden + self.attn(self.norm1(hidden), attention_mask)
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden


class _TinyBackbone(nn.Module):
    """接受 2560 维 inputs_embeds，内部降到 64 维跑 2 层，再升回 2560。"""

    def __init__(self, hidden_size: int = LM_HIDDEN, inner: int = TINY_INNER):
        super().__init__()
        self.inp = nn.Linear(hidden_size, inner)
        self.blocks = nn.ModuleList(
            [_TinyBlock(inner, TINY_HEADS) for _ in range(TINY_BLOCKS)])
        self.norm = nn.LayerNorm(inner)
        self.out = nn.Linear(inner, hidden_size)

    def forward(self, inputs_embeds: torch.Tensor,
                attention_mask: torch.Tensor = None,
                use_cache: bool = False):
        """`use_cache` 只为对齐 Qwen 接口（替身不返回 cache）。"""
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long,
                device=inputs_embeds.device)
        hidden = self.inp(inputs_embeds)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return SimpleNamespace(last_hidden_state=self.out(self.norm(hidden)))


class _TinyLM(nn.Module):
    """`lm.get_input_embeddings()` / `lm.model(...)` / `lm.lm_head` 的最小替身。"""

    def __init__(self, hidden_size: int = LM_HIDDEN, vocab: int = VOCAB):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden_size)
        self.model = _TinyBackbone(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab, bias=False)
        self.config = SimpleNamespace(use_cache=True)

    def get_input_embeddings(self) -> nn.Embedding:
        """和 HF 模型同名的取 embedding 方法。"""
        return self.embed


class _FakeTokenizer:
    """collator 只需要 pad_token_id。"""

    pad_token_id = 0
    eos_token_id = 1


class _FakeChatTokenizer(_FakeTokenizer):
    """最小 chat template 替身：验证 dataset 的前缀切分 / eos 截断逻辑。

    模板语义与 Qwen 对齐：每条 message = ``<role 头> + content + <结束>``，
    assistant 的 role 头就是 generation prompt，因此
    ``full_ids[:len(prompt_ids)] == prompt_ids`` 成立。
    """

    eos_token_id = 2
    _ROLE_HEAD = {"system": [40], "user": [41], "assistant": [13, 14]}
    _END = 31

    @staticmethod
    def _content_ids(text: str) -> list:
        return [20 + (ord(char) % 50) for char in str(text)[:3]]

    def apply_chat_template(self, messages, tokenize: bool = True,
                            add_generation_prompt: bool = False) -> list:
        """按固定 token 规则渲染（不依赖真实 tokenizer）。"""
        ids: list = []
        for message in messages:
            ids.extend(self._ROLE_HEAD.get(message.get("role"), [41]))
            ids.extend(self._content_ids(message.get("content", "")))
            ids.append(self._END)
        if add_generation_prompt:
            ids.extend(self._ROLE_HEAD["assistant"])
        else:
            ids.extend([self.eos_token_id, 9])       # eos + 模板尾部换行
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        """chat template 前缀不一致时的兜底 tokenize。"""
        return {"input_ids": self._content_ids(text)}


# ═══════════════════════════════════════════════════════════════
# 自检框架
# ═══════════════════════════════════════════════════════════════

class Checker:
    """记录 PASS/FAIL，最后给出汇总与退出码。"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        """布尔断言。"""
        if condition:
            self.passed += 1
            suffix = f"  ({detail})" if detail else ""
            print(f"[PASS] {name}{suffix}")
        else:
            self.failed += 1
            print(f"[FAIL] {name}  {detail}")

    def close(self, name: str, actual, expected, tol: float = ROW_TOL) -> None:
        """张量/标量近似相等断言（max|Δ| <= tol）。"""
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().float()
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().float()
        if torch.is_tensor(actual) or torch.is_tensor(expected):
            delta = float((torch.as_tensor(actual) - torch.as_tensor(expected))
                          .abs().max())
        else:
            delta = abs(float(actual) - float(expected))
        self.check(name, delta <= tol, f"max|Δ|={delta:.3e} (tol={tol:g})")

    def summary(self) -> int:
        """打印汇总并返回进程退出码。"""
        total = self.passed + self.failed
        print("\n" + "-" * 62)
        print(f"PASS {self.passed} / FAIL {self.failed}  (total {total})")
        print("RESULT:", "PASS" if self.failed == 0 else "FAIL")
        return 0 if self.failed == 0 else 1


def make_fake_batch(batch_size: int = BATCH, text_len: int = TEXT_LEN):
    """构造 2 张图的 batch：图 0 answer=4 tok，图 1 answer=3 tok + 文本 right-pad。"""
    generator = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(
        batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    input_ids = torch.randint(
        0, VOCAB, (batch_size, text_len), generator=generator)
    attention_mask = torch.ones((batch_size, text_len), dtype=torch.long)
    labels = torch.full(
        (batch_size, text_len), IGNORE_INDEX, dtype=torch.long)

    labels[0, 13:17] = input_ids[0, 13:17]        # 图 0：answer 4 tokens
    labels[1, 10:13] = input_ids[1, 10:13]        # 图 1：answer 3 tokens
    attention_mask[1, 13:17] = 0                  # 图 1：文本 right-pad 4 个
    input_ids[1, 13:17] = 0
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "gt_name": ["class_a", "class_b"],
    }


def build_model() -> VLM8Budget:
    """真实形状 config + 极小替身 encoder/lm（不触发任何 Qwen import）。"""
    config = VLM8BudgetConfig(
        encoder_ckpt="<unused-in-smoke>",
        qwen_dir="<unused-in-smoke>",
        auto_load_modules=False,
        use_gradient_checkpointing=False,
        freeze_encoder=False,
    )
    model = VLM8Budget(config, encoder=_FakeEncoder(), lm=_TinyLM())
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# 各检查组
# ═══════════════════════════════════════════════════════════════

def check_projector(checker: Checker, model: VLM8Budget) -> None:
    """① 投影头：形状 / 初始化 / 数值有限。"""
    torch.manual_seed(1)
    z_s = torch.randn(2, NUM_SPECIALS, ENC_DIM)
    vis = model.projector(z_s)
    checker.check("projector shape (B,32,1744)→(B,8,2560)",
                  tuple(vis.shape) == (2, 8, LM_HIDDEN), str(tuple(vis.shape)))
    checker.check("projector output finite", bool(torch.isfinite(vis).all()))

    weight = model.projector.token_proj.weight.detach()
    std = float(weight.std())
    checker.check("projector Linear init N(0,0.02)",
                  0.018 < std < 0.022, f"std={std:.5f}")
    checker.check("projector LayerNorm gamma=1 / beta=0",
                  bool(torch.allclose(
                      model.projector.budget_norm.weight.detach(),
                      torch.ones_like(model.projector.budget_norm.weight))) and
                  bool(torch.allclose(
                      model.projector.budget_norm.bias.detach(),
                      torch.zeros_like(model.projector.budget_norm.bias))))


def check_rows(checker: Checker, model: VLM8Budget, batch: dict):
    """② 8 行展开 + labels/padding/answer 位置结构。"""
    torch.manual_seed(2)
    vis = model.projector(torch.randn(BATCH, NUM_SPECIALS, ENC_DIM))
    text_embeds = torch.randn(BATCH, TEXT_LEN, LM_HIDDEN)
    ks = list(range(1, 9))
    rows = expand_budget_rows(
        vis, text_embeds, batch["attention_mask"], batch["labels"], ks)

    rows_per_image = len(ks)
    total_rows = BATCH * rows_per_image
    max_k = max(ks)
    length = max_k + TEXT_LEN
    checker.check("rows embeds shape", tuple(rows.embeds.shape) ==
                  (total_rows, length, LM_HIDDEN), str(tuple(rows.embeds.shape)))
    checker.check("rows mask/labels shape", tuple(rows.attention_mask.shape) ==
                  (total_rows, length) and
                  tuple(rows.labels.shape) == (total_rows, length))

    # 行长度：图 0 = k+17；图 1 = k+13（文本 4 个 right-pad）。
    lengths = rows.attention_mask.sum(dim=1).reshape(BATCH, rows_per_image)
    expected = torch.tensor(
        [[k + TEXT_LEN, k + TEXT_LEN - 4] for k in ks]).T
    checker.check("row length = k + real_text_len", bool(torch.equal(
        lengths.cpu(), expected)), str(lengths.tolist()))

    # labels：visual 前缀 + pad 必须全 -100；answer 只在文本段里。
    labels_ok = True
    for image_index in range(BATCH):
        for k in ks:
            row = image_index * rows_per_image + (k - 1)
            if not bool((rows.labels[row, :k] == IGNORE_INDEX).all()):
                labels_ok = False
            pad_start = k + int(batch["attention_mask"][image_index].sum())
            if not bool((rows.labels[row, pad_start:] == IGNORE_INDEX).all()):
                labels_ok = False
    checker.check("labels: visual prefix / padding all -100", labels_ok)

    # 前缀嵌套：行 k 的 visual 前缀 = vis[:, :k]；文本段逐 token 相同、整体后移 k。
    row_k1 = 0 * rows_per_image + 0
    row_k2 = 0 * rows_per_image + 1
    checker.check("prefix nesting: vis prefix k=1 ⊂ k=2",
                  bool(torch.equal(rows.embeds[row_k1, :1],
                                   rows.embeds[row_k2, :1])) and
                  bool(torch.equal(rows.embeds[row_k2, 1:2], vis[0, 1:2])))
    checker.check("prefix nesting: text identical but shifted by k",
                  bool(torch.equal(rows.embeds[row_k1, 1:1 + TEXT_LEN],
                                   rows.embeds[row_k2, 2:2 + TEXT_LEN])))

    # answer 位置：图 0 的 k=1 → 14..17；图 1 的 k=8 → 18..20；无效位 -1。
    checker.check("ans_pos: image0 k=1 = [14,15,16,17]",
                  rows.ans_pos[row_k1].tolist() == [14, 15, 16, 17],
                  str(rows.ans_pos[row_k1].tolist()))
    image1_k8 = 1 * rows_per_image + 7
    checker.check("ans_pos: image1 k=8 has 3 valid + 1 invalid (-1)",
                  rows.ans_pos[image1_k8].tolist() == [18, 19, 20, -1],
                  str(rows.ans_pos[image1_k8].tolist()))
    checker.check("ans_targets: invalid slot is -100",
                  rows.ans_targets[image1_k8].tolist()[-1] == IGNORE_INDEX)
    checker.check("budget_ks stored", rows.budget_ks.tolist() == ks)
    return rows


def check_collator(checker: Checker) -> None:
    """③ collator：right-padding / labels / gt_name。"""
    features = [
        {
            "pixel_values": torch.zeros(3, 8, 8),
            "input_ids": torch.tensor([5, 6, 7]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 7]),
            "gt_name": "a",
        },
        {
            "pixel_values": torch.zeros(3, 8, 8),
            "input_ids": torch.tensor([8, 9]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "labels": torch.tensor([IGNORE_INDEX, 9]),
            "gt_name": "b",
        },
    ]
    batch = EightBudgetCollator(_FakeTokenizer())(features)
    checker.check("collator input_ids padded to (2,3) with pad_token_id=0",
                  batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]],
                  str(batch["input_ids"].tolist()))
    checker.check("collator attention_mask zero on padding",
                  batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]])
    checker.check("collator labels -100 on padding",
                  batch["labels"].tolist() ==
                  [[IGNORE_INDEX, IGNORE_INDEX, 7],
                   [IGNORE_INDEX, 9, IGNORE_INDEX]])
    checker.check("collator passes gt_name", batch["gt_name"] == ["a", "b"])
    checker.check("collator pixel_values stacked",
                  tuple(batch["pixel_values"].shape) == (2, 3, 8, 8))

    # 图像变换：val 确定性输出 (3,448,448)。
    image = Image.new("RGB", (640, 480), color=(120, 30, 200))
    transformed = build_image_transform(train=False)(image)
    checker.check("val transform → (3,448,448)",
                  tuple(transformed.shape) == (3, IMAGE_SIZE, IMAGE_SIZE),
                  str(tuple(transformed.shape)))


def check_dataset(checker: Checker) -> None:
    """③b dataset：jsonl + 图像 + chat template → ids/labels；collator 成批。"""
    tokenizer = _FakeChatTokenizer()
    root = tempfile.mkdtemp(prefix="vlm8b_smoke_")
    try:
        image_dir = os.path.join(root, "n01440764")
        os.makedirs(image_dir, exist_ok=True)
        for name, color in (("a.JPEG", (200, 30, 30)), ("b.JPEG", (30, 30, 200))):
            Image.new("RGB", (80, 60), color=color).save(
                os.path.join(image_dir, name))

        def _messages(answer: str) -> str:
            return json.dumps([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "what"},
                {"role": "assistant", "content": answer},
            ])

        jsonl_path = os.path.join(root, "sft.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for name, answer in (("a.JPEG", "cat"), ("b.JPEG", "dog")):
                handle.write(json.dumps({
                    "image": f"n01440764/{name}",
                    "messages_short": _messages(answer),
                    "gt": "cat" if answer == "cat" else "dog",
                }) + "\n")

        dataset = SelfDistillSFTDataset(
            jsonl_path=jsonl_path,
            tokenizer=tokenizer,
            image_root=root,
            transform=build_image_transform(train=False),
        )
        checker.check("dataset length", len(dataset) == 2, str(len(dataset)))
        item = dataset[0]
        checker.check("dataset pixel_values (3,448,448)",
                      tuple(item["pixel_values"].shape) == (3, IMAGE_SIZE, IMAGE_SIZE))
        expected_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "sys"},
             {"role": "user", "content": "what"}],
            add_generation_prompt=True)
        expected_answer = _FakeChatTokenizer._content_ids("cat") + [31, 2]
        checker.check("dataset input_ids = prompt + answer(eos-truncated)",
                      item["input_ids"].tolist()
                      == expected_prompt + expected_answer,
                      f"{item['input_ids'].tolist()}")
        n_prompt = len(expected_prompt)
        checker.check("dataset labels: prompt all -100",
                      bool((item["labels"][:n_prompt] == IGNORE_INDEX).all()))
        checker.check("dataset labels: answer ids + eos kept",
                      item["labels"][n_prompt:].tolist() == expected_answer)
        checker.check("dataset attention_mask all 1",
                      bool((item["attention_mask"] == 1).all()))

        batch = EightBudgetCollator(tokenizer)([dataset[0], dataset[1]])
        checker.check("dataset+collator batch shapes",
                      tuple(batch["input_ids"].shape) ==
                      (2, int(batch["input_ids"].shape[1])) and
                      tuple(batch["pixel_values"].shape) ==
                      (2, 3, IMAGE_SIZE, IMAGE_SIZE) and
                      batch["gt_name"] == ["cat", "dog"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_forward_and_loss(checker: Checker, model: VLM8Budget, batch: dict):
    """④ 前向 / loss 权重 / CE 遮蔽 / 反传路径。"""
    out = model(**batch)
    for key in ("loss", "per_budget_ce", "per_budget_ce_per_sample",
                "row_ce", "budget_ks"):
        checker.check(f"forward returns {key}", key in out)
    checker.check("per_budget_ce shape (8,)", tuple(out["per_budget_ce"].shape)
                  == (8,), str(tuple(out["per_budget_ce"].shape)))
    checker.check("per_budget_ce_per_sample shape (2,8)",
                  tuple(out["per_budget_ce_per_sample"].shape) == (2, 8),
                  str(tuple(out["per_budget_ce_per_sample"].shape)))
    checker.check("loss finite", bool(torch.isfinite(out["loss"])))
    checker.close("per_budget_ce = mean over images",
                  out["per_budget_ce"],
                  out["per_budget_ce_per_sample"].mean(dim=0), 1e-6)
    checker.close("equal weighting: loss = mean(per_budget_ce)",
                  out["loss"], out["per_budget_ce"].mean(), 1e-6)

    # inv-k 权重：loss = Σ (1/k)/Σ(1/j) · CE_k
    model.set_loss_weighting("inv-k")
    out_inv = model(**batch)
    ks = torch.arange(1, 9, dtype=torch.float32)
    weights = (1.0 / ks)
    weights = weights / weights.sum()
    checker.close("inv-k weighting matches formula", out_inv["loss"],
                  (out_inv["per_budget_ce"] * weights).sum(), 1e-6)
    # 自定义 budget_weights 也应生效
    custom = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out_custom = model(**batch, budget_weights=custom)
    checker.close("custom budget_weights: loss = CE(k=1)",
                  out_custom["loss"], out_custom["per_budget_ce"][0], 1e-6)
    model.set_loss_weighting("equal")

    # CE 遮蔽：gather 版 row CE == 整序列 CE + ignore_index。
    rows = expand_budget_rows(
        model.projector(torch.randn(BATCH, NUM_SPECIALS, ENC_DIM)),
        model.lm.get_input_embeddings()(batch["input_ids"]),
        batch["attention_mask"], batch["labels"], list(range(1, 9)))
    out_rows = model.forward_rows(rows)
    hidden = model.lm.model(
        inputs_embeds=rows.embeds,
        attention_mask=rows.attention_mask).last_hidden_state
    logits = model.lm.lm_head(hidden)                        # (R,L,V)
    per_token = F.cross_entropy(
        logits.reshape(-1, VOCAB), rows.labels.reshape(-1),
        reduction="none", ignore_index=IGNORE_INDEX).reshape(hidden.size(0), -1)
    counts = (rows.labels != IGNORE_INDEX).sum(dim=1).clamp(min=1)
    reference = per_token.sum(dim=1) / counts
    checker.close("answer-only gather CE == full-sequence CE (ignore -100)",
                  out_rows["row_ce"], reference, 1e-6)

    # 反传：梯度应到达 projector 与 encoder。
    model.zero_grad(set_to_none=True)
    model.train()
    out_train = model(**batch)
    out_train["loss"].backward()
    checker.check("backward reaches projector",
                  model.projector.token_proj.weight.grad is not None and
                  float(model.projector.token_proj.weight.grad.abs().sum()) > 0)
    checker.check("backward reaches encoder",
                  model.encoder.tokens.grad is not None and
                  float(model.encoder.tokens.grad.abs().sum()) > 0)
    stats = model.pop_budget_ce_stats()
    checker.check("per-budget CE stats accumulated in train mode",
                  stats is not None and len(stats["per_budget_ce"]) == 8 and
                  stats["n_micro"] == 1, str(stats and stats.get("n_micro")))
    checker.check("stats queue emptied after pop",
                  model.pop_budget_ce_stats() is None)
    model.zero_grad(set_to_none=True)
    model.eval()


def check_invariants(checker: Checker, model: VLM8Budget, batch: dict):
    """⑤ 四个不变量：行独立 / 截断隔离 / pad 隔离 / 预算单调结构。"""
    torch.manual_seed(3)
    z_s = torch.randn(BATCH, NUM_SPECIALS, ENC_DIM)
    vis = model.projector(z_s)
    text_embeds = model.lm.get_input_embeddings()(batch["input_ids"])
    ks = list(range(1, 9))

    # ① 行独立：图 0 单图单跑 vs 混在 2 图 batch 里。
    out_batch = model(**batch)["per_budget_ce_per_sample"]
    out_single = model(
        pixel_values=batch["pixel_values"][:1],
        input_ids=batch["input_ids"][:1],
        attention_mask=batch["attention_mask"][:1],
        labels=batch["labels"][:1],
    )["per_budget_ce_per_sample"]
    checker.close("invariant 1: row independence (single vs mixed batch)",
                  out_single[0], out_batch[0], 1e-4)

    rows = expand_budget_rows(
        vis, text_embeds, batch["attention_mask"], batch["labels"], ks)
    base = model.forward_rows(rows)["row_ce"]

    # ② 行隔离：改图 0 的 k=1 行 embedding，k=8 行 loss 不变。
    rows_changed = expand_budget_rows(
        vis, text_embeds, batch["attention_mask"], batch["labels"], ks)
    rows_changed.embeds[0] = torch.randn_like(rows_changed.embeds[0])
    after = model.forward_rows(rows_changed)["row_ce"]
    checker.close("invariant 2a: change k=1 row → k=8 row CE unchanged",
                  after[7], base[7], 1e-7)
    checker.check("invariant 2a: k=1 row CE actually changed",
                  abs(float(after[0] - base[0])) > 1e-4,
                  f"Δ={float(after[0] - base[0]):.3e}")

    # ② 截断隔离：z_s 第 5 个 token（属于第 2 个 visual token）只影响 k≥2。
    #    注意不能加常数（LayerNorm 平移不变 ⇒ 投影后逐位不变），必须加噪声。
    z_s_changed = z_s.clone()
    torch.manual_seed(7)
    z_s_changed[:, 4] += torch.randn_like(z_s_changed[:, 4])
    vis_changed = model.projector(z_s_changed)
    rows_changed2 = expand_budget_rows(
        vis_changed, text_embeds, batch["attention_mask"], batch["labels"], ks)
    after2 = model.forward_rows(rows_changed2)["per_budget_ce"]
    base_pb = model.forward_rows(rows)["per_budget_ce"]
    checker.close("invariant 2b: change z_s[4] → k=1 CE unchanged",
                  after2[0], base_pb[0], 1e-7)
    checker.check("invariant 2b: change z_s[4] → k=2 CE changed",
                  abs(float(after2[1] - base_pb[1])) > 1e-4,
                  f"Δ={float(after2[1] - base_pb[1]):.3e}")

    # ③ pad 隔离：padding 位置 embedding 换成随机值，所有行 loss 不变。
    rows_pad = expand_budget_rows(
        vis, text_embeds, batch["attention_mask"], batch["labels"], ks)
    pad_positions = rows_pad.attention_mask == 0
    checker.check("padding positions exist", bool(pad_positions.any()))
    rows_pad.embeds[pad_positions] = torch.randn_like(
        rows_pad.embeds[pad_positions])
    after_pad = model.forward_rows(rows_pad)["row_ce"]
    checker.close("invariant 3: pad embedding change → all row CE unchanged",
                  after_pad, base, 1e-7)

    # ④ 预算单调结构：行长度严格 +1、answer 位置随 k 平移、前缀嵌套。
    lengths = rows.attention_mask.sum(dim=1).reshape(BATCH, len(ks))
    strictly_increasing = all(
        int(lengths[b, i + 1] - lengths[b, i]) == 1
        for b in range(BATCH) for i in range(len(ks) - 1))
    checker.check("invariant 4: row length strictly +1 per budget",
                  strictly_increasing)
    shifted = rows.ans_pos.reshape(BATCH, len(ks), -1)
    valid = shifted >= 0
    checker.check("invariant 4: ans_pos shifts by +1 per budget",
                  bool(torch.equal(
                      (shifted[:, 1:] - shifted[:, :-1])[valid[:, 1:] & valid[:, :-1]],
                      torch.ones(int((valid[:, 1:] & valid[:, :-1]).sum()),
                                 dtype=torch.long))))

    # 行 k 的 visual 前缀 = vis[:, :k]；文本段逐 token 相同、整体后移 k。
    checker.check("invariant 4: vis prefix k=1 ⊂ k=2",
                  bool(torch.equal(rows.embeds[0, :1], rows.embeds[1, :1])) and
                  bool(torch.equal(rows.embeds[1, 1:2], vis[0, 1:2])))
    checker.check("invariant 4: text identical but shifted by k",
                  bool(torch.equal(rows.embeds[0, 1:1 + TEXT_LEN],
                                   rows.embeds[1, 2:2 + TEXT_LEN])))


def check_encoder_ckpt(path: str) -> bool:
    """（可选 --full）从阶段2 ckpt 构造编码器，验证 z_s 形状。"""
    config = VLM8BudgetConfig(
        encoder_ckpt=path, auto_load_modules=False,
        use_gradient_checkpointing=False)
    encoder = AssocVTCEncoder(config)
    encoder.eval()
    with torch.no_grad():
        z_s = encoder(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    ok = tuple(z_s.shape) == (1, NUM_SPECIALS, ENC_DIM)
    print(f"[{'PASS' if ok else 'FAIL'}] encoder ckpt → z_s "
          f"{tuple(z_s.shape)} (expected (1,32,1744))")
    return ok


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """`--logic-only` 是默认路径；`--full` 额外检查真实 ckpt（可选）。"""
    parser = argparse.ArgumentParser(description="vlm8b CPU logic smoke test")
    parser.add_argument("--logic-only", dest="logic_only", action="store_true",
                        default=True,
                        help="只跑纯张量逻辑（默认；不需要 Qwen/GPU）")
    parser.add_argument("--full", dest="logic_only", action="store_false",
                        help="额外用 --encoder-ckpt 做真实编码器形状检查")
    parser.add_argument("--encoder-ckpt", default="",
                        help="阶段2 ckpt 路径（仅 --full 用；不存在则 SKIP）")
    return parser.parse_args()


def main() -> int:
    """跑全部检查，返回 0/1。"""
    args = parse_args()
    print("=" * 62)
    print("vlm8b smoke: CPU tensor logic self-check")
    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    print("=" * 62)

    checker = Checker()
    try:
        model = build_model()
        batch = make_fake_batch()
        check_projector(checker, model)
        check_rows(checker, model, batch)
        check_collator(checker)
        check_dataset(checker)
        check_forward_and_loss(checker, model, batch)
        check_invariants(checker, model, batch)
    except Exception:                           # noqa: BLE001 - 自检需要打印完整栈
        traceback.print_exc()
        checker.check("unexpected exception", False)

    if not args.logic_only:
        if args.encoder_ckpt and os.path.isfile(args.encoder_ckpt):
            try:
                ok = check_encoder_ckpt(args.encoder_ckpt)
                checker.check("encoder ckpt shape check", ok)
            except Exception:                   # noqa: BLE001
                traceback.print_exc()
                checker.check("encoder ckpt shape check", False)
        else:
            print(f"[SKIP] encoder ckpt check (file not found: "
                  f"{args.encoder_ckpt or '<empty>'})")

    return checker.summary()


if __name__ == "__main__":
    sys.exit(main())
