"""
vlm8b 评测 — per-budget (k=1..8) greedy 生成 + teacher-forcing CE + exact-match
===============================================================================

指标（与设计 §9 一致）::

    CE(k)   teacher forcing（同一 `messages_short` 口径）
    acc(k)  greedy 生成后与 GT 类名 **完全匹配**（strip 后逐字符比较）
    Δacc    acc(8) − acc(1)（预算收益；k=1/8 都在预算集合里时才输出）

用法::

    python vlm8b/eval.py \
        --checkpoint /root/autodl-tmp/vlm8b/output/run1/final \
        --jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/val_nothink_all.jsonl \
        --limit 1000

输出::

    <output-dir>/eval.json   完整数字 + 少量样例
    <output-dir>/eval.md     人看的 per-budget 表

实现说明::

  · 训练产物是 HF 目录（`Trainer.save_model`，ZeRO-3 gather 后的 bf16 权重）；
    `VLM8Budget.from_pretrained` 会用 config.json 里的 encoder_ckpt/qwen_dir
    重建结构再灌权重（因此 remote 上这两个路径必须仍然存在）。
  · greedy 生成逐行做（right-padding 下整批 generate 会把 pad 当上下文）。
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

try:                                            # script / package 两种运行方式
    from .data import EightBudgetCollator, SelfDistillSFTDataset, build_image_transform
    from .model import VLM8Budget, VLM8BudgetConfig
except ImportError:                             # pragma: no cover - 脚本入口
    from data import EightBudgetCollator, SelfDistillSFTDataset, build_image_transform
    from model import VLM8Budget, VLM8BudgetConfig


# ═══════════════════════════════════════════════════════════════
# 默认值
# ═══════════════════════════════════════════════════════════════

DEFAULT_QWEN_DIR = "/root/qwen_models/Qwen3.5-4B"
DEFAULT_IMAGE_ROOT = "/root/autodl-tmp/datasets/imagenet100"
DEFAULT_JSONL = ("/root/autodl-tmp/exp_qwen/in100_selfdistill/sft/"
                 "val_nothink_all.jsonl")
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/vlm8b/output/eval"
DEFAULT_BUDGETS = "1,2,3,4,5,6,7,8"
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_EXAMPLES = 20


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析评测参数。"""
    parser = argparse.ArgumentParser(
        description="vlm8b per-budget evaluation (greedy + CE + exact match)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="训练产物目录（含 config.json + 权重）")
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--qwen-dir", default="",
                        help="覆盖 config.json 里的 Qwen 路径")
    parser.add_argument("--encoder-ckpt", default="",
                        help="覆盖 config.json 里的阶段2 ckpt 路径")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS,
                        help="逗号分隔，如 1,2,4,8")
    parser.add_argument("--limit", type=int, default=0,
                        help="只评前 N 张（0=全部）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="每批图片数（行数 = 图片数 × 预算数）")
    parser.add_argument("--max-new-tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES,
                        help="json 里保留多少条生成样例")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="",
                        help="cuda / cpu；默认 cuda（不可用则 cpu）")
    return parser.parse_args()


def parse_budgets(raw: str) -> List[int]:
    """`"1,2,4,8"` → [1,2,4,8]（升序去重）。"""
    budgets = sorted({int(part) for part in str(raw).split(",") if part.strip()})
    if not budgets:
        raise ValueError(f"--budgets 为空: {raw!r}")
    return budgets


# ═══════════════════════════════════════════════════════════════
# 模型 / 数据
# ═══════════════════════════════════════════════════════════════

def load_trained_model(args: argparse.Namespace, device: torch.device) -> VLM8Budget:
    """重建结构（encoder + Qwen + projector）并灌入训练后的权重。

    不用 `from_pretrained`：transformers 4.57+ 默认 `low_cpu_mem_usage` 会把
    自定义模块建在 meta device 上，而我们的 encoder 是在 `__init__` 里手动
    `load_state_dict` 的（meta 上会静默 no-op）。这里显式：构造 → 转 dtype/device
    → 读训练产物（safetensors / bin，含分片）→ strict 校验（ties 除外）。
    """
    config = VLM8BudgetConfig.from_pretrained(args.checkpoint)
    if args.encoder_ckpt:
        config.encoder_ckpt = args.encoder_ckpt
    if args.qwen_dir:
        config.qwen_dir = args.qwen_dir
    config.auto_load_modules = True
    config.use_gradient_checkpointing = False
    config.freeze_encoder = False            # 推理不需要冻结（避免 eval 期误判）

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    model = VLM8Budget(config)
    model = model.to(device=device, dtype=dtype)
    missing = load_trained_weights(model, args.checkpoint, device)
    model.eval()
    if model.lm is not None:
        model.lm.config.use_cache = True     # 生成时打开 KV cache
    print(f"[vlm8b] loaded {args.checkpoint} on {device} ({args.dtype})"
          + (f"  (tied keys re-used: {missing})" if missing else ""))
    return model


def load_trained_weights(
    model: VLM8Budget, checkpoint: str, device: torch.device) -> List[str]:
    """把 HF 目录里的权重灌进已构造好的 model；返回允许缺失的 tied keys。

    Raises:
        FileNotFoundError: 目录里没有 model.safetensors / pytorch_model.bin。
        RuntimeError: 有非 tied 的缺失键或意外键（说明结构或路径不对）。
    """
    files, fmt = _resolve_weight_files(checkpoint)
    state: Dict[str, torch.Tensor] = {}
    if fmt == "safetensors":
        from safetensors.torch import load_file
        for path in files:
            state.update(load_file(path, device=str(device)))
    else:
        for path in files:
            state.update(torch.load(
                path, map_location=device, weights_only=True))

    missing, unexpected = model.load_state_dict(state, strict=False)
    tied_missing = [key for key in missing if key == "lm.lm_head.weight"]
    real_missing = [key for key in missing if key not in tied_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            "训练产物与模型结构不匹配：\n"
            f"  missing={real_missing[:10]}\n  unexpected={unexpected[:10]}")
    return tied_missing


def _resolve_weight_files(checkpoint: str) -> tuple:
    """返回 ([权重文件...], 'safetensors'|'bin')；支持 HF 分片索引。"""
    safetensors_index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.isfile(safetensors_index):
        with open(safetensors_index, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        return ([os.path.join(checkpoint, name)
                 for name in sorted(set(weight_map.values()))], "safetensors")

    safetensors_file = os.path.join(checkpoint, "model.safetensors")
    if os.path.isfile(safetensors_file):
        return [safetensors_file], "safetensors"

    bin_index = os.path.join(checkpoint, "pytorch_model.bin.index.json")
    if os.path.isfile(bin_index):
        with open(bin_index, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        return ([os.path.join(checkpoint, name)
                 for name in sorted(set(weight_map.values()))], "bin")

    bin_file = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.isfile(bin_file):
        return [bin_file], "bin"

    raise FileNotFoundError(
        f"{checkpoint} 下没有 model.safetensors / pytorch_model.bin（HF 权重）")


def build_loader(args: argparse.Namespace, tokenizer, dataset) -> DataLoader:
    """验证集 DataLoader（val 确定性变换 + right-pad collator）。"""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=EightBudgetCollator(tokenizer),
        pin_memory=False,
    )


# ═══════════════════════════════════════════════════════════════
# 评测主循环
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    """跑 CE + greedy 生成，返回可直接 json.dump 的结果 dict。"""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    budgets = parse_budgets(args.budgets)

    tokenizer = _load_tokenizer(args.qwen_dir or _config_qwen_dir(args.checkpoint))
    dataset = SelfDistillSFTDataset(
        jsonl_path=args.jsonl,
        tokenizer=tokenizer,
        image_root=args.image_root,
        transform=build_image_transform(train=False),
    )
    if args.limit and args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = build_loader(args, tokenizer, dataset)

    model = load_trained_model(args, device)
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    ce_sum = {k: 0.0 for k in budgets}
    correct = {k: 0 for k in budgets}
    total = {k: 0 for k in budgets}
    examples: List[Dict[str, object]] = []
    n_images = 0

    for step, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        gt_names = batch["gt_name"]

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            budget_ids=budgets,
        )
        per_sample = outputs["per_budget_ce_per_sample"].float().cpu()   # (B,V)

        generated = model.generate_budget(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            budget_ids=budgets,
            max_new_tokens=args.max_new_tokens,
        )

        for image_index in range(pixel_values.size(0)):
            n_images += 1
            gt_name = str(gt_names[image_index]).strip()
            for budget_index, k in enumerate(budgets):
                new_ids = generated[int(k)][image_index]
                # generate(inputs_embeds=...) 只返回新 token；防御性截断到首个 eos。
                for pos, token_id in enumerate(new_ids):
                    if token_id in eos_ids:
                        new_ids = new_ids[:pos]
                        break
                prediction = tokenizer.decode(
                    new_ids, skip_special_tokens=True).strip()
                ce_sum[k] += float(per_sample[image_index, budget_index])
                correct[k] += int(prediction == gt_name)
                total[k] += 1
                if len(examples) < args.num_examples:
                    examples.append({
                        "image_index": n_images - 1,
                        "k": k,
                        "gt": gt_name,
                        "pred": prediction,
                        "correct": prediction == gt_name,
                    })
        print(f"[vlm8b] batch {step + 1}/{len(loader)}  images={n_images}")

    rows = []
    for k in budgets:
        n = max(total[k], 1)
        rows.append({
            "k": k,
            "n": total[k],
            "ce": ce_sum[k] / n,
            "acc": correct[k] / n,
        })

    delta_acc: Optional[float] = None
    by_k = {row["k"]: row for row in rows}
    if 1 in by_k and 8 in by_k:
        delta_acc = by_k[8]["acc"] - by_k[1]["acc"]

    return {
        "checkpoint": args.checkpoint,
        "jsonl": args.jsonl,
        "image_root": args.image_root,
        "n_images": n_images,
        "budgets": rows,
        "delta_acc_k8_minus_k1": delta_acc,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "examples": examples,
    }


def _config_qwen_dir(checkpoint: str) -> str:
    """读训练产物 config.json 里的 qwen_dir（tokenizer 用）。"""
    path = os.path.join(checkpoint, "config.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("qwen_dir", DEFAULT_QWEN_DIR)


def _load_tokenizer(qwen_dir: str):
    """加载 Qwen tokenizer（推理用）。"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        qwen_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ═══════════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════════

def write_outputs(result: Dict[str, object], output_dir: str) -> None:
    """写 `eval.json` + `eval.md`。"""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "eval.json")
    md_path = os.path.join(output_dir, "eval.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(build_markdown(result))
    print(f"[vlm8b] wrote {json_path} and {md_path}")


def build_markdown(result: Dict[str, object]) -> str:
    """per-budget 结果 → markdown 表。"""
    lines = [
        "# vlm8b per-budget evaluation",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- data: `{result['jsonl']}`",
        f"- n_images: {result['n_images']}",
        f"- generation: greedy, max_new_tokens="
        f"{result['generation']['max_new_tokens']}",
        "",
        "| k | CE | exact-match acc | n |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["budgets"]:
        lines.append(
            f"| {row['k']} | {row['ce']:.4f} | {row['acc'] * 100:.2f}% | {row['n']} |")
    delta = result.get("delta_acc_k8_minus_k1")
    lines.append("")
    if delta is not None:
        lines.append(f"Δacc(8−1) = **{delta * 100:+.2f}%**")
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """入口。"""
    args = parse_args()
    result = evaluate(args)
    write_outputs(result, args.output_dir)
    for row in result["budgets"]:
        print(f"[vlm8b] k={row['k']}  CE={row['ce']:.4f}  "
              f"acc={row['acc'] * 100:.2f}%  n={row['n']}")


if __name__ == "__main__":
    main()
