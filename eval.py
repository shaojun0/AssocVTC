"""
vlm8b 评测 — per-budget (k=1..V) teacher-forcing CE + greedy exact-match
=======================================================================

`CE(k)` = teacher forcing；`acc(k)` = greedy 生成与 GT 类名完全匹配；
`Δacc = acc(V) − acc(1)`。输出 `<output-dir>/eval.json` 与 `eval.md`。用法::

    python eval.py --checkpoint <run>/final --jsonl <val.jsonl> --limit 1000

训练产物是 HF 目录（ZeRO-3 gather 后的 bf16 权重）；按 config.json 里的
encoder_ckpt / qwen_dir 重建结构再灌权重（只允许 tied `lm_head` 缺失）。
greedy 逐行生成（right-padding 下整批 generate 会把 pad 当上下文）。
"""
import argparse
import glob
import json
import os
import random
import re
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from data import EightBudgetCollator, SelfDistillSFTDataset, build_image_transform
from model import IGNORE_INDEX, VLM8Budget, VLM8BudgetConfig

# ═══════════════════════════════════════════════════════════════
# 默认值 / CLI
# ═══════════════════════════════════════════════════════════════

DEFAULT_JSONL = ("/root/autodl-tmp/exp_qwen/in100_selfdistill/sft/"
                 "val_nothink_all.jsonl")
DEFAULT_IMAGE_ROOT = "/root/autodl-tmp/datasets/imagenet100"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/vlm8b/output/eval"
DEFAULT_BUDGETS = "1,2,3,4,5,6,7,8"
MAX_NEW_TOKENS, NUM_WORKERS, NUM_EXAMPLES = 16, 2, 20


def parse_args() -> argparse.Namespace:
    """解析评测参数。"""
    p = argparse.ArgumentParser(
        description="vlm8b per-budget evaluation (greedy + CE + exact match)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="训练产物目录（含 config.json）")
    p.add_argument("--jsonl", default=DEFAULT_JSONL)
    p.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--qwen-dir", default="", help="覆盖 config.json 的 Qwen 路径")
    p.add_argument("--encoder-ckpt", default="", help="覆盖 config.json 的阶段2 ckpt")
    p.add_argument("--budgets", default=DEFAULT_BUDGETS, help="逗号分隔，如 1,2,4,8")
    p.add_argument("--limit", type=int, default=0, help="只评前 N 张（0=全部）")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sample-seed", type=int, default=42,
                   help="--limit>0 时随机抽样的种子（避免按类排序的子集偏置）")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--visual-mode", default="normal",
                   choices=["normal", "null", "shuffle"],
                   help="视觉消融：null=vis 置零（token 数不变，不等于无图）/ shuffle=batch 内错位（看错图）")
    return p.parse_args()


def _norm(text: str) -> str:
    """类名归一化：小写、下划线→空格、压空白（'chocolate_sauce' ≡ 'Chocolate Sauce'）。"""
    return re.sub(r"\s+", " ", str(text).strip().lower().replace("_", " "))


def extract_answer(text: str) -> str:
    """从生成文本里取答案：去掉 'The answer is' 前缀与尾部句号。"""
    text = re.sub(r"^(?:the answer is|answer)\s*[:：]?\s*", "",
                  str(text).strip(), flags=re.IGNORECASE)
    return text.strip().strip(".")


def _safetensors_files(checkpoint: str) -> List[str]:
    """列 checkpoint 里的 safetensors 文件：优先分片索引，否则单文件 `model.safetensors`。"""
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as handle:
            names = sorted(set(json.load(handle)["weight_map"].values()))
        return [os.path.join(checkpoint, name) for name in names]
    files = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    assert files, f"checkpoint 里没有 safetensors: {checkpoint}"
    return files


def load_trained_model(checkpoint: str, config: VLM8BudgetConfig) -> VLM8Budget:
    """重建结构（encoder + Qwen + projector）并灌入训练后的权重（safetensors 分片）。"""
    from safetensors.torch import load_file
    handler = _safetensors_files(checkpoint)
    state: Dict[str, torch.Tensor] = {}
    for path in handler:
        state.update(load_file(path))
    model = VLM8Budget(config, grad_ckpt=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"lm.lm_head.weight"}, (
        f"权重不匹配: missing={missing[:5]} unexpected={unexpected[:5]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device=device, dtype=torch.bfloat16).eval()


def prompt_lens(labels):
    """answer 首 token 位置 = prompt 长度；只喂 [.., prompt) 给 generate。"""
    valid = labels != IGNORE_INDEX
    assert bool(valid.any(dim=1).all()), "有样本没有 answer 标签，生成边界不可确定"
    lens = valid.to(torch.int8).argmax(dim=1)
    assert int(lens.min()) >= 1, "存在 prompt 为空的样本（answer 从第 0 位开始）"
    return lens


@torch.no_grad()
def generate_prompt_only(model, pixel_values, input_ids, labels, budgets,
                         visual_mode: str, has_image=None):
    """只喂 prompt 的 greedy 生成：`[prompt][vis[:k]]`（先文字后图），逐行解码。"""
    from model import apply_visual_mode
    prompt_lens_ = prompt_lens(labels)
    encoder_dtype = next(model.encoder.parameters()).dtype
    vis = apply_visual_mode(model.projector(model.encoder(
        pixel_values.to(dtype=encoder_dtype)).to(
            model.projector.token_proj.weight.dtype)), visual_mode)
    text_embeds = model.lm.get_input_embeddings()(input_ids).to(vis.dtype)
    results = {}
    for k in budgets:
        results[k] = []
        for b in range(pixel_values.size(0)):
            n = int(prompt_lens_[b])
            k_eff = k if has_image is None or bool(has_image[b]) else 0
            embeds = torch.cat([text_embeds[b, :n], vis[b, :k_eff]], dim=0).unsqueeze(0)
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
            generated = model.lm.generate(
                inputs_embeds=embeds, attention_mask=mask,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id)
            results[k].append(generated[0].tolist())
    return results


# ═══════════════════════════════════════════════════════════════
# 评测主循环 / 输出
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    """跑 CE + greedy 生成，返回可 json.dump 的结果 dict。"""
    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    config = VLM8BudgetConfig.from_pretrained(args.checkpoint)
    config.encoder_ckpt = args.encoder_ckpt or config.encoder_ckpt
    config.qwen_dir = args.qwen_dir or config.qwen_dir
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SelfDistillSFTDataset(
        args.jsonl, tokenizer, args.image_root, build_image_transform(train=False))
    if args.limit > 0:
        # val jsonl 按类排序；取前 N 只会评到前几个类（chance = 1/类数）。
        # 这里固定种子随机抽样，避免子集偏置。
        rng = random.Random(args.sample_seed)
        indices = rng.sample(range(len(dataset)), min(args.limit, len(dataset)))
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS,
                        collate_fn=EightBudgetCollator(tokenizer))
    model = load_trained_model(args.checkpoint, config)
    device = next(model.parameters()).device

    stats = {k: [0.0, 0, 0] for k in budgets}          # [ce_sum, correct, n]
    examples: List[Dict[str, object]] = []
    n_no_image = 0
    for batch in loader:
        inputs = {key: batch[key].to(device) for key in
                  ("pixel_values", "input_ids", "attention_mask", "labels",
                   "has_image")}
        outputs = model(budget_ids=budgets, visual_mode=args.visual_mode, **inputs)
        n_no_image += int((~inputs["has_image"]).sum())
        generated = generate_prompt_only(
            model, inputs["pixel_values"], inputs["input_ids"], inputs["labels"],
            budgets, args.visual_mode, has_image=inputs["has_image"])
        per_sample = outputs["per_budget_ce_per_sample"].float().cpu()
        for i in range(inputs["pixel_values"].size(0)):
            gt_name = str(batch["gt_name"][i]).strip()
            for budget_index, k in enumerate(budgets):
                ids = generated[k][i]
                eos = tokenizer.eos_token_id
                if eos is not None and eos in ids:
                    ids = ids[:ids.index(eos)]
                prediction = extract_answer(
                    tokenizer.decode(ids, skip_special_tokens=True))
                correct = _norm(prediction) == _norm(gt_name)
                stats[k][0] += float(per_sample[i, budget_index])
                stats[k][1] += int(correct)
                stats[k][2] += 1
                if len(examples) < NUM_EXAMPLES:
                    examples.append({"k": k, "gt": gt_name, "pred": prediction,
                                     "correct": correct})

    rows = [{"k": k, "ce": s[0] / max(s[2], 1), "acc": s[1] / max(s[2], 1), "n": s[2]}
            for k, s in stats.items()]
    return {"checkpoint": args.checkpoint, "jsonl": args.jsonl,
            "visual_mode": args.visual_mode, "sample_seed": args.sample_seed,
            "n_images": rows[0]["n"], "budgets": rows,
            "delta_acc": rows[-1]["acc"] - rows[0]["acc"], "examples": examples,
            "gen_protocol": "prompt_only", "max_new_tokens": MAX_NEW_TOKENS,
            "limit": args.limit, "n_no_image": n_no_image}


def write_outputs(result: Dict[str, object], output_dir: str) -> None:
    """写 `eval.json` + `eval.md`。"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    lines = ["# vlm8b per-budget evaluation", "",
             f"- checkpoint: `{result['checkpoint']}`  n_images: {result['n_images']}",
             "", "| k | CE | exact-match acc | n |", "|---:|---:|---:|---:|"]
    lines += [f"| {row['k']} | {row['ce']:.4f} | {row['acc'] * 100:.2f}% | {row['n']} |"
              for row in result["budgets"]]
    lines += ["", f"Δacc(V−1) = **{result['delta_acc'] * 100:+.2f}%**", ""]
    with open(os.path.join(output_dir, "eval.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"[vlm8b] wrote {output_dir}/eval.json and eval.md")


def main() -> None:
    """入口。"""
    args = parse_args()
    result = evaluate(args)
    write_outputs(result, args.output_dir)
    for row in result["budgets"]:
        print(f"[vlm8b] k={row['k']} CE={row['ce']:.4f} "
              f"acc={row['acc'] * 100:.2f}% n={row['n']}")


if __name__ == "__main__":
    main()
