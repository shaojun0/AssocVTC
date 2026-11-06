#!/usr/bin/env python
"""mix_heldout 的**分任务** 8 预算消融评测（run3 结束后用）。

eval.py 只给总体 CE/acc；这份脚本按 source 出「预算 × 任务」表，回答
「预算机制在哪类任务上成立、在哪类上是浪费」。

用法（需 GPU，run3 结束后跑）：
  python eval_mix_breakdown.py --checkpoint output/run3/final \
      --jsonl data_eval/mix_heldout.jsonl --per-source 25 --budgets 1,2,3,4,5,6,7,8 \
      --visual-mode normal --out logs/run3_breakdown_normal.json
  --dry  只做数据/索引校验（CPU、不加载模型）

输出：{visual_mode, per_source:{src:{k:{ce,acc,n}}}, overall:{...}, samples}
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, "/root/vlm8b")

DATA = "/root/autodl-tmp/vlm8b/data_eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", default=os.path.join(DATA, "mix_heldout.jsonl"))
    ap.add_argument("--image-root", default=DATA)
    ap.add_argument("--per-source", type=int, default=25)
    ap.add_argument("--budgets", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--visual-mode", default="normal", choices=["normal", "null", "shuffle"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    by_src = defaultdict(list)
    for i, r in enumerate(rows):
        by_src[r.get("source", "?")].append(i)
    rng = random.Random(args.seed)
    sample = []
    for src, idxs in sorted(by_src.items()):
        take = min(args.per_source, len(idxs))
        sample += [(src, i) for i in rng.sample(idxs, take)]
    print(f"rows={len(rows)} sources={len(by_src)} sampled={len(sample)}", flush=True)
    for src, idxs in sorted(by_src.items()):
        print(f"  {src:16s} pool={len(idxs):5d} take={min(args.per_source, len(idxs))}", flush=True)
    if args.dry:
        print("DRY OK（数据/抽样校验通过，未加载模型）")
        return

    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoTokenizer
    import eval as E  # 复用 eval.py：config/加载/答案抽取
    from data import EightBudgetCollator, SelfDistillSFTDataset, build_image_transform

    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    config = E.VLM8BudgetConfig.from_pretrained(args.checkpoint)
    config.encoder_ckpt = config.encoder_ckpt
    config.qwen_dir = config.qwen_dir
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    full = SelfDistillSFTDataset(args.jsonl, tokenizer, args.image_root,
                                 build_image_transform(train=False))
    order = [i for _, i in sample]
    ds = Subset(full, order)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                        collate_fn=EightBudgetCollator(tokenizer))
    model = E.load_trained_model(args.checkpoint, config)
    device = next(model.parameters()).device

    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0]))   # src -> k -> [ce, ok, n]
    pos = 0
    with torch.no_grad():
        for batch in loader:
            inputs = {key: batch[key].to(device) for key in
                      ("pixel_values", "input_ids", "attention_mask", "labels")}
            outputs = model(budget_ids=budgets, visual_mode=args.visual_mode, **inputs)
            generated = model.generate_budget(
                inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"],
                budget_ids=budgets, max_new_tokens=E.MAX_NEW_TOKENS,
                visual_mode=args.visual_mode)
            per_sample = outputs["per_budget_ce_per_sample"].float().cpu()
            for i in range(inputs["pixel_values"].size(0)):
                src = sample[pos][0]
                pos += 1
                gt = str(batch["gt_name"][i]).strip()
                for bi, k in enumerate(budgets):
                    ids = generated[k][i]
                    eos = tokenizer.eos_token_id
                    if eos is not None and eos in ids:
                        ids = ids[:ids.index(eos)]
                    pred = E.extract_answer(tokenizer.decode(ids, skip_special_tokens=True))
                    ok = int(E._norm(pred) == E._norm(gt))
                    a = agg[src][k]
                    a[0] += float(per_sample[i, bi])
                    a[1] += ok
                    a[2] += 1

    out = {"checkpoint": args.checkpoint, "jsonl": args.jsonl,
           "visual_mode": args.visual_mode, "per_source_samples": args.per_source,
           "budgets": budgets, "per_source": {}, "overall": {}}
    total = defaultdict(lambda: [0.0, 0, 0])
    for src in sorted(agg):
        out["per_source"][src] = {}
        for k in budgets:
            ce, ok, n = agg[src][k]
            out["per_source"][src][str(k)] = {"ce": ce / max(n, 1), "acc": ok / max(n, 1), "n": n}
            t = total[k]
            t[0] += ce
            t[1] += ok
            t[2] += n
    out["overall"] = {str(k): {"ce": total[k][0] / max(total[k][2], 1),
                               "acc": total[k][1] / max(total[k][2], 1),
                               "n": total[k][2]} for k in budgets}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
    print("| source | " + " | ".join(f"k{k} acc" for k in budgets) + " |")
    print("|---|" + "---|" * len(budgets))
    for src, d in out["per_source"].items():
        print(f"| {src} | " + " | ".join(f"{d[str(k)]['acc'] * 100:.1f}%" for k in budgets) + " |")
    d = out["overall"]
    print("| **overall** | " + " | ".join(f"{d[str(k)]['acc'] * 100:.1f}%" for k in budgets) + " |")


if __name__ == "__main__":
    main()
