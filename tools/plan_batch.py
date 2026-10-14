#!/usr/bin/env python3
"""plan_batch.py — 上调 batch 前先算清楚：BS / patch_budget / 卡数 三者如何换算

**为什么需要它**：train_multi 里的 `--batch_size` 只是**上限**，实际每批样本数是

    bs = max(1, min(batch_size, patch_budget // N))      N = (bh//14)*(bw//14)

N 随桶（编码器输入尺寸）变化，最大到 4096 ⇒ 大 N 的桶被 `patch_budget` 卡住，
**只调 `--batch_size` 对这些桶毫无作用**（2026-09-26 那条 run 设了 `--batch_size 6`，
实测每卡平均 batch 只有 **3.35**）。反过来，`patch_budget` 才是「一批塞多少 patch」的
总预算，直接对应显存。

本脚本读索引缓存，把候选 (BS, BUDGET) 的后果全部算给你（纯 CPU，秒级）：

  · 每卡优化步/epoch、2 epoch 的 max_steps
  · 全局平均 batch（= bs × 卡数，随桶变化，取实际加权平均）
  · 最大 N 桶能拿到几个样本（**显存风险就在这一行**）
  · 被 drop_last 丢掉的样本占比（bs 调太大时浪费会变高）

用法::

    python tools/plan_batch.py --cache <data_root>/index_cache.json --world 8
    python tools/plan_batch.py --world 8 --bs 6 12 18 --budget 6144 12288 24576
    python tools/plan_batch.py --world 8 --target_global 54      # 高亮接近该全局 batch 的行
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_multi import DEFAULT_DATA_ROOT  # noqa: E402


def analyse(shapes: list, bs: int, budget: int, world: int):
    """完全按 BucketBatchSampler 的口径算（drop_last=True + 批次级 round-robin）。"""
    cnt = Counter(shapes)
    n_used, nb = 0, 0
    worst_n, worst_bs = 0, 0
    for (bh, bw), c in cnt.items():
        n = (bh // 14) * (bw // 14)
        b = max(1, min(bs, (budget // n) if budget else bs))
        k = c // b                      # drop_last=True
        nb += k
        n_used += k * b
        if n > worst_n:
            worst_n, worst_bs = n, b
    steps = nb // world                 # 截断到世界大小的整数倍
    total = sum(cnt.values())
    return {
        "bs": bs, "budget": budget,
        "steps_per_rank": steps,
        "max_steps_2ep": steps * 2,
        "global_avg": n_used / max(1, steps),
        "per_rank_avg": n_used / max(1, steps * world),
        "worst_n": worst_n, "worst_bs": worst_bs,
        "kept_pct": 100.0 * n_used / max(1, total),
        "dumped": total - n_used,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(DEFAULT_DATA_ROOT, "index_cache.json"))
    ap.add_argument("--world", type=int, default=8, help="卡数")
    ap.add_argument("--bs", type=int, nargs="+", default=[6, 12, 18, 24],
                    help="候选 --batch_size（每卡上限）")
    ap.add_argument("--budget", type=int, nargs="+",
                    default=[6144, 12288, 24576, 49152], help="候选 --patch_budget")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--target_global", type=float, default=0.0,
                    help="高亮与目标全局 batch 最接近的几行")
    args = ap.parse_args()

    idx = json.load(open(args.cache))
    shapes = [(bh, bw) for (_, bw, bh, _) in idx]
    print(f"索引 {args.cache}: {len(idx):,} 条 | {len(set(shapes))} 种桶 | 卡数 {args.world}")
    mx = max((bh // 14) * (bw // 14) for bh, bw in shapes)
    print(f"最大 N = {mx}（patch_budget < N×BS 时，这个桶的 batch 就被压到 budget//N）\n")

    rows = [analyse(shapes, bs, b, args.world) for bs in args.bs for b in args.budget]
    if args.target_global:
        rows.sort(key=lambda r: abs(r["global_avg"] - args.target_global))
    else:
        rows.sort(key=lambda r: (r["bs"], r["budget"]))

    print(f"{'BS':>4}{'BUDGET':>8} | {'每卡步/epoch':>13}{'2ep max_steps':>15} | "
          f"{'全局平均batch':>13}{'每卡平均':>10} | {'最大N桶bs':>10} | {'保留样本':>9}")
    print("-" * 100)
    for r in rows:
        mark = ""
        if args.target_global and abs(r["global_avg"] - args.target_global) / args.target_global < 0.05:
            mark = "  ← 接近目标"
        print(f"{r['bs']:>4}{r['budget']:>8} | {r['steps_per_rank']:>13,}"
              f"{r['max_steps_2ep']:>15,} | {r['global_avg']:>13.2f}"
              f"{r['per_rank_avg']:>10.2f} | {r['worst_bs']:>10} | "
              f"{r['kept_pct']:>8.2f}%{mark}")
    print("-" * 100)
    print("""
怎么用这张表：
  1) **先定全局 batch 目标**（要和 09-26 / ×1 臂对照就先对齐它；否则至少记下来）。
  2) 在表里挑一行：`全局平均 batch ≈ bs × 卡数`。要「显存安全」就看「最大 N 桶 bs」
     —— 这一列 ×1 通常是峰值显存的主要来源；它越大越吃显存。
  3) 设 `--patch_budget ≥ BS × N_max`（本机 N_max 见上）才能让所有桶都用满 `--batch_size`；
     否则大 N 桶仍被压小、你只是白改了一个上限。
  4) 记下这一行的 `每卡步/epoch` ⇒ 启动器会自动算 `max_steps = 每epoch × EPOCHS`，
     不用手填；换 BS/BUDGET/卡数后一定重新算（步数会变）。
  5) 改大全局 batch 后 **lr 要跟着调**（线性规则: lr × 新全局/旧全局；WARMUP_RATIO 保持），
     否则「batch 变大」会变成一次没记录的配方变更。
  6) `grad_accum` 是另一条路：不动显存、只乘有效 batch，代价是步数不变（省不了时间）。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
