#!/usr/bin/env python3
"""对 run3 训练混合数据做「按任务等量」重采样（不改图片、不联网、只挑行）。

背景（2026-11-06 用户反馈）：初版 `mix_train.jsonl` 里 ImageNet-100 锚点 66,000 行
占图像部分的 67%，而各评测图像任务 300–6,000 不等，头重脚轻。

本脚本从既有 `mix_train.jsonl` 里按 source 抽样，生成新的训练 jsonl：
  * 文本源（mmlu/cmmlu/... 7 个）原样全保留；
  * 图像源给统一目标配额 `--img-quota 2000`：数据量 ≥ 配额的任务抽 2000，
    数据量不够的用全量（mmstar 999 / mathvista 698 / mmmu 600 / ocrbench 600 /
    realworldqa 500 / countbench 300 —— 差异只由可得数据量造成）；
  * textvqa 与 textvqa_train 是同一任务的两份数据，合并后共 2000（按 2:1 拆）；
  * ImageNet-100 锚点从 66,000 降到 2,000（与单个图像任务同量级，占比 ~10%）；
  * 行内容（含 `messages_short`）原样拷贝，绝不重新生成 prompt。

用法:
  python rebalance_mix.py --train mix_train.jsonl --out mix_train_bal.jsonl \
      [--img-quota 2000] [--seed 20261106] [--report balance.json]
"""
import argparse
import collections
import json
import os
import random
import sys

TEXT_SOURCES = {"mmlu", "cmmlu", "mmlu_pro", "gsm8k", "math", "mmlu_redux", "ceval"}

# 同一任务的多份数据：合并成一个任务看总配额
TASK_ALIAS = {"textvqa_train": "textvqa"}
TASK_SPLIT = {"textvqa": {"textvqa": 2, "textvqa_train": 1}}  # 2:1 保持原始比例

IMAGE_SOURCES = [
    "scienceqa", "slake", "docvqa", "textvqa", "textvqa_train", "ai2d", "mathvision",
    "chartqa", "we_math", "mmstar", "mathvista", "mmmu", "ocrbench", "realworldqa",
    "countbench",
]
ANCHOR_SOURCES = ["imagenet100"]


def log(msg: str) -> None:
    print(msg, flush=True)


def pick(lines, quota: int, rng: random.Random):
    """无放回抽 quota 行；quota 超过行数则全取。"""
    if quota is None or quota >= len(lines):
        return list(lines)
    idx = sorted(rng.sample(range(len(lines)), quota))
    return [lines[i] for i in idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="原始 mix_train.jsonl")
    parser.add_argument("--out", required=True, help="输出 jsonl")
    parser.add_argument("--img-quota", type=int, default=2000,
                        help="每个图像任务的统一行数上限（默认 2000）")
    parser.add_argument("--anchor-quota", type=int, default=2000,
                        help="ImageNet-100 锚点行数（默认 2000，原 66000）")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--report", default=None, help="把统计写到该 json")
    args = parser.parse_args()

    by_source = collections.defaultdict(list)
    total_in = 0
    for line in open(args.train, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        record = json.loads(line)
        by_source[record.get("source", "?")].append(line)
        total_in += 1
    log(f"[balance] 输入 {args.train}: {total_in} 行 / {len(by_source)} 源")

    # 每个图像“任务”的目标配额：先按任务合并（textvqa 两份合并成 2000）
    task_quota = {}
    for src in IMAGE_SOURCES:
        task = TASK_ALIAS.get(src, src)
        if task in task_quota:
            continue
        available = sum(len(by_source.get(s, [])) for s in IMAGE_SOURCES
                        if TASK_ALIAS.get(s, s) == task)
        task_quota[task] = min(args.img_quota, available)
    for src in ANCHOR_SOURCES:
        task_quota[src] = min(args.anchor_quota, len(by_source.get(src, [])))

    rng = random.Random(args.seed)
    kept = {}
    lines_out = []
    known = set(IMAGE_SOURCES + ANCHOR_SOURCES) | TEXT_SOURCES
    unknown = [s for s in by_source if s not in known]
    if unknown:
        log(f"[balance] WARN 未知 source（原样保留）: {unknown}")

    for source in sorted(by_source):
        lines = by_source[source]
        if source in IMAGE_SOURCES:
            task = TASK_ALIAS.get(source, source)
            quota = task_quota[task]
            splits = TASK_SPLIT.get(task)
            if splits:  # 同一任务多份数据按固定比例分
                parts = sorted(s for s in IMAGE_SOURCES if TASK_ALIAS.get(s, s) == task)
                denom = sum(splits[s] for s in parts)
                quota_src = quota * splits[source] // denom
            else:
                quota_src = quota
            chosen = pick(lines, quota_src, rng)
        elif source in ANCHOR_SOURCES:
            chosen = pick(lines, task_quota[source], rng)
        else:  # 文本源 / 未知源：全保留
            chosen = list(lines)
        kept[source] = len(chosen)
        lines_out.extend(chosen)

    # 逐行校验：图片存在 + assistant 前缀（问题在训练早期暴露比训练崩掉便宜）
    missing = bad_prefix = bad_gt = 0
    for line in lines_out:
        record = json.loads(line)
        if not os.path.exists(record["image"]):
            missing += 1
        answers = [m["content"] for m in record["messages_short"] if m.get("role") == "assistant"]
        if not answers or not all(a.startswith("The answer is ") for a in answers):
            bad_prefix += 1
        if not str(record.get("gt", "")).strip():
            bad_gt += 1
    if missing or bad_prefix or bad_gt:
        log(f"[balance] FATAL 校验失败 missing_img={missing} bad_prefix={bad_prefix} bad_gt={bad_gt}")
        return 1

    with open(args.out, "w", encoding="utf-8") as handle:
        for line in lines_out:
            handle.write(line + "\n")

    # ---- 报告 ----
    img_rows = sum(kept[s] for s in IMAGE_SOURCES + ANCHOR_SOURCES)
    text_rows = sum(kept[s] for s in kept if s not in IMAGE_SOURCES + ANCHOR_SOURCES)
    log("\n[balance] source            原量     保留    占比(全量)")
    for source in sorted(kept, key=lambda s: -kept[s]):
        log(f"  {source:<16} {len(by_source[source]):>7} {kept[source]:>7}   "
            f"{kept[source] / len(lines_out) * 100:5.1f}%")
    steps_per_epoch = len(lines_out) // 32
    log(f"\n[balance] 总行数 {len(lines_out)}（图像 {img_rows} + 文本 {text_rows}）；"
        f"图像每任务配额 {args.img_quota}，ImageNet 锚点 {args.anchor_quota}")
    log(f"[balance] 每 epoch ≈ {steps_per_epoch} 步（全局 batch 32）；"
        f"1 epoch ≈ {steps_per_epoch * 8.95 / 3600:.2f} h")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump({"input": args.train, "output": args.out,
                       "rows_in": total_in, "rows_out": len(lines_out),
                       "img_quota": args.img_quota, "anchor_quota": args.anchor_quota,
                       "seed": args.seed, "kept": kept,
                       "steps_per_epoch": steps_per_epoch}, handle,
                      ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
