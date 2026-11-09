#!/usr/bin/env python
"""run3 训练集第二版：图像任务「能加就加、大任务封顶 6,000」，ImageNet 锚定 3,000。

用户 2026-11-06 18:10 反馈：不要只减；其他任务加数据、ImageNet 减数据，
并保证数据量够训到周一 12:00。

输入（远端 /root/autodl-tmp/vlm8b/data_eval）：
  text_train.jsonl              文本 82,000（原样全保留）
  img_train.jsonl               主构建图像源（docvqa 4,451 / textvqa / chartqa / ...）
  img_extra_train.jsonl         兜底构建源（scienceqa / slake 5,000 / ai2d / ...）
  expand_docvqa_train.jsonl     扩容 +1,549 → docvqa 6,000
  expand_slake_train.jsonl      扩容 +1,000 → slake 6,000
  mix_train.jsonl               旧混合集，只取 imagenet100 行抽样 3,000

输出：mix_train_bal2.jsonl + mix_train_bal2_report.json
安全：只读旧文件；输出前校验「id 不与 mix_heldout 重叠 / 图片存在 / 答案前缀 / gt 非空」。
"""
import argparse
import collections
import json
import os
import random
import sys

DATA = "/root/autodl-tmp/vlm8b/data_eval"

# source -> train 行数（None = 全取）
QUOTA = {
    "docvqa": 4500, "docvqa_train": 865, "textvqa": 3997, "textvqa_train": 2000,
    "chartqa": 1500, "chartqa_train": 4500, "mmstar": 999, "mathvista": 698,
    "mathvision": 2000, "we_math": 1200, "mmmu": 600, "scienceqa": 6000,
    "slake": 5000, "ai2d": 2000, "ocrbench": 600, "realworldqa": 500,
    "countbench": 300, "imagenet100": 3000,
}
MAIN = ["img_train.jsonl"]
EXTRA = ["img_extra_train.jsonl"]
EXPAND = ["expand_docvqa_train.jsonl", "expand_docvqa_train_split.jsonl",
          "expand_chartqa_train.jsonl"]
ANCHOR_FROM_MIX = "mix_train.jsonl"


def read_lines(path):
    with open(os.path.join(DATA, path), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def log(msg):
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "mix_train_bal2.jsonl"))
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()

    pools = collections.defaultdict(list)     # source -> [line]
    for name in MAIN + EXTRA + EXPAND:
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            log(f"[mix2] FATAL 缺少输入 {path}")
            return 2
        for line in read_lines(name):
            record = json.loads(line)
            pools[record["source"]].append(line)

    # 文本行：全部保留
    text_lines = list(read_lines("text_train.jsonl"))
    text_sources = collections.Counter(json.loads(line)["source"] for line in text_lines)

    # ImageNet 锚定：从旧混合集里抽
    rng = random.Random(args.seed)
    anchor = [line for line in read_lines(ANCHOR_FROM_MIX)
              if json.loads(line).get("source") == "imagenet100"]
    if len(anchor) < QUOTA["imagenet100"]:
        log(f"[mix2] FATAL imagenet 行数不足 {len(anchor)}")
        return 2
    anchor = [anchor[i] for i in sorted(rng.sample(range(len(anchor)), QUOTA["imagenet100"]))]

    out_lines = list(text_lines)
    kept = collections.Counter(text_sources)
    for source, quota in QUOTA.items():
        if source == "imagenet100":
            chosen = anchor
        else:
            lines = pools.get(source, [])
            if quota >= len(lines):
                chosen = list(lines)
            else:
                chosen = [lines[i] for i in sorted(rng.sample(range(len(lines)), quota))]
        kept[source] = len(chosen)
        out_lines.extend(chosen)

    # ---------- 校验 ----------
    heldout_ids = set()
    for line in read_lines("mix_heldout.jsonl"):
        heldout_ids.add(json.loads(line)["id"])
    ids = set()
    missing = bad_prefix = bad_gt = 0
    for line in out_lines:
        record = json.loads(line)
        rid = record["id"]
        if rid in ids:
            log(f"[mix2] FATAL 重复 id {rid}")
            return 1
        ids.add(rid)
        if not os.path.exists(record["image"]):
            missing += 1
        answers = [m["content"] for m in record["messages_short"] if m.get("role") == "assistant"]
        if not answers or not all(a.startswith("The answer is ") for a in answers):
            bad_prefix += 1
        if not str(record.get("gt", "")).strip():
            bad_gt += 1
    overlap = ids & heldout_ids
    if missing or bad_prefix or bad_gt or overlap:
        log(f"[mix2] FATAL missing_img={missing} bad_prefix={bad_prefix} "
            f"bad_gt={bad_gt} heldout_overlap={len(overlap)}")
        return 1

    random.Random(args.seed).shuffle(out_lines)
    with open(args.out, "w", encoding="utf-8") as handle:
        for line in out_lines:
            handle.write(line + "\n")

    img_rows = sum(v for k, v in kept.items() if k in QUOTA)
    text_rows = len(out_lines) - img_rows
    steps = (len(out_lines) + 31) // 32
    log(f"\n[mix2] 总行数 {len(out_lines)}（文本 {text_rows} + 图像 {img_rows}）"
        f" → {args.out}")
    log(f"[mix2] {steps} 步/epoch ≈ {steps * 8.95 / 3600:.2f} h/epoch")
    log("\n[mix2] source          行数")
    for source in sorted(kept, key=lambda s: -kept[s]):
        log(f"  {source:<16} {kept[source]:>7}")
    report = {"out": args.out, "rows": len(out_lines), "text_rows": text_rows,
              "img_rows": img_rows, "steps_per_epoch": steps, "kept": dict(kept),
              "quota": QUOTA, "seed": args.seed}
    with open(os.path.join(DATA, "mix_train_bal2_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
