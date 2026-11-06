#!/usr/bin/env python
"""合并/校验 vlm8b 评测对齐混合训练集。

子命令：
  merge   合并 text_train + img_train + imagenet 锚定 → mix_train.jsonl；heldout 同理 → mix_heldout.jsonl
  stats   打印各 source 条数 / 答案字符长度分布
  check   抽样用 Qwen tokenizer 复核长度（<=1024），并检查图片可读

远端路径约定见 doc/2026-11-06/PLAN_eval_aligned_mixture.md
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = "/root/autodl-tmp/vlm8b/data_eval"
IMAGENET_JSONL = "/root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl"
IMAGENET_IMG_ROOT = "/root/autodl-tmp/datasets/imagenet100"


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def cmd_merge(args):
    text = list(read_jsonl(os.path.join(ROOT, "text_train.jsonl")))
    img = list(read_jsonl(os.path.join(ROOT, "img_train.jsonl")))
    extra = os.path.join(ROOT, "img_extra_train.jsonl")
    if os.path.exists(extra):
        img += list(read_jsonl(extra))

    rng = random.Random(42)
    anchor = []
    n_anchor = args.imagenet_quota
    rows = list(read_jsonl(IMAGENET_JSONL))
    idx = rng.sample(range(len(rows)), min(n_anchor, len(rows)))
    for i in idx:
        row = dict(rows[i])
        rel = row["image"]
        row["image"] = rel if os.path.isabs(rel) else os.path.join(IMAGENET_IMG_ROOT, rel)
        row["source"] = "imagenet100"
        # 消息里的 image 字段也改绝对路径（保持与原结构一致）
        try:
            for m in row["messages"]:
                if isinstance(m.get("content"), list):
                    for c in m["content"]:
                        if c.get("type") == "image":
                            c["image"] = row["image"]
            for m in row["messages_short"]:
                if isinstance(m.get("content"), list):
                    for c in m["content"]:
                        if c.get("type") == "image":
                            c["image"] = row["image"]
        except Exception:
            pass
        anchor.append(row)

    mix = text + img + anchor
    rng.shuffle(mix)
    out = os.path.join(ROOT, "mix_train.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for row in mix:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    held = list(read_jsonl(os.path.join(ROOT, "text_heldout.jsonl")))
    for name in ("img_heldout.jsonl", "img_extra_heldout.jsonl"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            held += list(read_jsonl(p))
    rng.shuffle(held)
    with open(os.path.join(ROOT, "mix_heldout.jsonl"), "w", encoding="utf-8") as fh:
        for row in held:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"mix_train: {len(mix)} (text {len(text)} + img {len(img)} + anchor {len(anchor)})")
    print(f"mix_heldout: {len(held)}")
    cnt = Counter(r.get("source", "?") for r in mix)
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v}")


def cmd_stats(args):
    for name in ("mix_train.jsonl", "mix_heldout.jsonl"):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        cnt = Counter()
        lens = defaultdict(list)
        for row in read_jsonl(path):
            src = row.get("source", "?")
            cnt[src] += 1
            lens[src].append(len(row["gt"]))
        print(f"== {name}: {sum(cnt.values())}")
        for src, n in sorted(cnt.items(), key=lambda kv: -kv[1]):
            L = sorted(lens[src])
            print(f"  {src:16s} {n:7d}  ans_len p50={L[len(L)//2]:3d} p90={L[int(len(L)*0.9)]:3d} max={L[-1]:3d}")


def cmd_check(args):
    """抽样 tokenizer 复核 + 图片可读性。"""
    sys.path.insert(0, "/root/vlm8b")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/root/qwen_models/Qwen3.5-4B", local_files_only=True,
                                        trust_remote_code=True)
    path = os.path.join(ROOT, args.file)
    rows = list(read_jsonl(path))
    rng = random.Random(0)
    sample = rng.sample(rows, min(args.n, len(rows)))
    over, bad_img, bad_fmt = 0, 0, 0
    for row in sample:
        msgs = row["messages"]
        try:
            ids = tok.apply_chat_template(msgs, tokenize=True)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            if len(ids) > 1024:
                over += 1
                print("  OVER", len(ids), row["id"], row["source"])
        except Exception as e:
            bad_fmt += 1
            print("  BADFMT", row["id"], type(e).__name__, str(e)[:80])
        if not os.path.exists(row["image"]):
            bad_img += 1
            print("  NOIMG", row["id"], row["image"])
    print(f"checked {len(sample)} / {len(rows)}: over1024={over} bad_fmt={bad_fmt} missing_img={bad_img}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--imagenet-quota", type=int, default=90000)
    m.set_defaults(fn=cmd_merge)
    s = sub.add_parser("stats")
    s.set_defaults(fn=cmd_stats)
    c = sub.add_parser("check")
    c.add_argument("--file", default="mix_train.jsonl")
    c.add_argument("--n", type=int, default=500)
    c.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
