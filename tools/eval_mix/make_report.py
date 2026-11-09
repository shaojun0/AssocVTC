#!/usr/bin/env python
"""把 eval.json / bench 结果拼成报告用的 markdown 表格（run2 / run3 通用）。

用法示例（run1 自测）：
  python make_report.py --title "run1" \
    --normal output/eval_r_normal/eval.json --null output/eval_r_null/eval.json \
    --shuffle output/eval_r_shuffle/eval.json --bench output/bench_general.json
run2：
  ... --normal output/eval_run2_normal/eval.json --null output/eval_run2_null/eval.json \
      --shuffle output/eval_run2_shuffle/eval.json --bench logs/bench_run2.json
run3：再加 --in100 output/eval_run3_in100/eval.json --breakdown logs/run3_breakdown_normal.json
"""
import argparse
import json
import os

BASE = "/root/autodl-tmp/vlm8b"


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def budget_table(ev):
    rows = ev["budgets"]
    lines = ["| k | CE | acc | n |", "|---:|---:|---:|---:|"]
    lines += [f"| {r['k']} | {r['ce']:.4f} | {r['acc'] * 100:.2f}% | {r['n']} |" for r in rows]
    k1, kl = rows[0], rows[-1]
    lines.append("")
    lines.append(f"- ΔCE(k{kl['k']}−k{k1['k']}) = **{kl['ce'] - k1['ce']:+.4f}**；"
                 f"Δacc = **{(kl['acc'] - k1['acc']) * 100:+.2f}%**（n={kl['n']}，seed {ev.get('sample_seed')}）")
    return "\n".join(lines)


def ablation_table(evs):
    ks = [r["k"] for r in evs["normal"]["budgets"]]
    lines = ["| 视觉模式 | " + " | ".join(f"CE k{k}" for k in (ks[0], ks[-1]))
             + " | " + " | ".join(f"acc k{k}" for k in (ks[0], ks[-1])) + " |",
             "|---|" + "---|" * 4]
    for mode in ("normal", "null", "shuffle"):
        ev = evs.get(mode)
        if not ev:
            continue
        b = {r["k"]: r for r in ev["budgets"]}
        lines.append(f"| {mode} | {b[ks[0]]['ce']:.3f} | {b[ks[-1]]['ce']:.3f} | "
                     f"{b[ks[0]]['acc'] * 100:.2f}% | {b[ks[-1]]['acc'] * 100:.2f}% |")
    n = evs["normal"].get("n_images")
    lines.append("")
    lines.append(f"- 同一子集 n={n}（seed {evs['normal'].get('sample_seed')}）；")
    lines.append("- null=视觉 emb 置零、token 数不变（**不等于没有图**）；shuffle=batch 内错位")
    return "\n".join(lines)


def bench_table(js):
    if not js:
        return None
    cats = list(js["base"]["by_category"].keys())
    lines = ["| 模型 | " + " | ".join(cats) + " | overall |",
             "|---|" + "---|" * (len(cats) + 1)]
    for key, label in (("base", "base Qwen3.5-4B"), ("ours_text", "vlm8b 微调后")):
        d = js.get(key)
        if not d:
            continue
        accs = [d["by_category"][c]["acc"] for c in cats]
        n = d["by_category"][cats[0]]["n"]
        lines.append("| " + label + " | " + " | ".join(f"{a * 100:.0f}%" for a in accs)
                     + f" | {sum(accs) / len(accs) * 100:.1f}% |")
    return "\n".join(lines) + f"\n\n- 每题类 n={n}（文本通道，thinking off）"


def breakdown_table(js):
    if not js:
        return None
    ks = js["budgets"]
    lines = ["| source | " + " | ".join(f"k{k}" for k in ks) + " | n/源 |",
             "|---|" + "---|" * (len(ks) + 1)]
    for src, d in js["per_source"].items():
        lines.append(f"| {src} | " + " | ".join(f"{d[str(k)]['acc'] * 100:.0f}%" for k in ks)
                     + f" | {d[str(ks[0])]['n']} |")
    ov = js["overall"]
    lines.append("| **overall** | " + " | ".join(f"{ov[str(k)]['acc'] * 100:.1f}%" for k in ks) + " | |")
    return "\n".join(lines) + f"\n\n- visual-mode={js.get('visual_mode')}，每源 {js.get('per_source_samples')} 条"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="run")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--normal", required=True)
    ap.add_argument("--null", default="")
    ap.add_argument("--shuffle", default="")
    ap.add_argument("--in100", default="")
    ap.add_argument("--bench", default="")
    ap.add_argument("--breakdown", default="")
    args = ap.parse_args()

    evs = {m: load(getattr(args, m)) for m in ("normal", "null", "shuffle")}
    print(f"## {args.title} 评测结果\n")
    print(f"- checkpoint：`{args.checkpoint or evs['normal']['checkpoint']}`")
    print(f"- 评测集：`{evs['normal']['jsonl']}`（n={evs['normal']['n_images']}, seed={evs['normal']['sample_seed']}）\n")
    print("### 1. 8 预算主表（normal）\n")
    print(budget_table(evs["normal"]), "\n")
    if evs["null"] or evs["shuffle"]:
        print("### 2. 视觉消融\n")
        print(ablation_table(evs), "\n")
    b = bench_table(load(args.bench))
    if b:
        print("### 3. 通用文本能力\n")
        print(b, "\n")
    i100 = load(args.in100)
    if i100:
        print("### 4. ImageNet-100（主任务，混合集稀释检查）\n")
        print(budget_table(i100), "\n")
    bt = breakdown_table(load(args.breakdown))
    if bt:
        print("### 5. 分任务 × 预算（mix_heldout）\n")
        print(bt, "\n")


if __name__ == "__main__":
    main()
