#!/usr/bin/env python
"""run3 数据预检（CPU only，不占 GPU）：
1) 全量：图片存在、token 长度 <=1024、答案 span 非空
2) 抽样：真实走 data.py 的 Dataset+Collator，再用 model.expand_budget_rows 验证 8 预算行展开
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/vlm8b")
ROOT = "/root/autodl-tmp/vlm8b/data_eval"
PX = "/root/qwen_models/Qwen3.5-4B"

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="mix_train.jsonl")
ap.add_argument("--fast", action="store_true", help="跳过全量 token 复核（默认全量）")
ap.add_argument("--n", type=int, default=64)
args = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(PX, local_files_only=True, trust_remote_code=True)

path = os.path.join(ROOT, args.file)
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print(f"rows={len(rows)}", flush=True)

# ---------- 1) 全量静态校验 ----------
noimg = over = empty = 0
for i, r in enumerate(rows):
    if not os.path.exists(r["image"]):
        noimg += 1
        if noimg <= 5:
            print("  NOIMG", r["id"], r["image"], flush=True)
    if not r.get("gt"):
        empty += 1
    if not args.fast:
        ids = tok.apply_chat_template(r["messages"], tokenize=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if len(ids) > 1024:
            over += 1
            if over <= 10:
                print("  OVER", len(ids), r["id"], r["source"], flush=True)
    if i and i % 20000 == 0:
        print(f"  ...{i}", flush=True)
print(f"static: noimg={noimg} empty_gt={empty} over1024={over}", flush=True)

# ---------- 2) 抽样走真实 Dataset / Collator / budget 展开 ----------
from data import (EightBudgetCollator, SelfDistillSFTDataset,
                  build_image_transform, IGNORE_INDEX)  # noqa: E402
from model import expand_budget_rows  # noqa: E402

ds = SelfDistillSFTDataset(path, tok, ROOT, build_image_transform(train=True))
print(f"dataset len={len(ds)}", flush=True)
idx = list(range(0, len(ds), max(1, len(ds) // args.n)))[: args.n]
feats = [ds[i] for i in idx]
coll = EightBudgetCollator(tok)
batch = coll(feats)
B, T = batch["input_ids"].shape
print(f"batch B={B} T={T} pixel={tuple(batch['pixel_values'].shape)}", flush=True)

H = 2560
vis = torch.randn(B, 8, H)
text_embeds = torch.randn(B, T, H)
rows_b = expand_budget_rows(vis, text_embeds, batch["attention_mask"], batch["labels"],
                            [1, 2, 3, 4, 5, 6, 7, 8])
print("budget rows:", tuple(rows_b.embeds.shape), "num_images", rows_b.num_images,
      "num_budgets", rows_b.num_budgets, "A(max ans len)", rows_b.ans_pos.size(1), flush=True)
assert rows_b.embeds.shape == (B * 8, T + 8, H), rows_b.embeds.shape
valid = rows_b.ans_targets != IGNORE_INDEX
assert bool((rows_b.ans_pos[valid] >= 1).all()), rows_b.ans_pos[valid].min()
print("valid target count:", int(valid.sum()), "expected:", int((rows_b.ans_targets != IGNORE_INDEX).sum()),
      flush=True)
# k=1 行 与 k=8 行的 target 应完全一致
t1 = rows_b.ans_targets[0::8]
t8 = rows_b.ans_targets[7::8]
assert torch.equal(t1, t8)
# ans_pos 应满足 pos[k] = pos[1] + (k-1)
pos1 = rows_b.ans_pos[0::8]
v1 = rows_b.ans_targets[0::8] != IGNORE_INDEX
for k in range(8):
    pk = rows_b.ans_pos[k::8]
    assert torch.equal(pk[v1] - pos1[v1], torch.full_like(pos1[v1], k)), k
print("expand_budget_rows OK（行序/移位/target 一致）", flush=True)
print("PREFLIGHT OK", flush=True)
