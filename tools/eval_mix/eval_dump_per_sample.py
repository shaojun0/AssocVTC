"""
vlm8b 逐样本筛查 — 每行 × 每预算 CE（多视觉消融）+ 少量 greedy + 视觉 token 诊断
==============================================================================

目的：回答「预算机制（k 越大越好）在**哪些样本**上成立/失效、为什么」。与
`eval.py`（只报均值）不同，本脚本把每行 × 每 k 的 CE、预测、以及视觉 token 的
冗余度统计落盘，供 `analyze_screen.py` 做分源/分象限分析。

消融模式（便宜，一次前向算完所有 k）：

  · normal   原始 vis
  · null     vis 置零（等价没有图）
  · shuffle  batch 内错位（看错图）
  · mean     8 个 token 取均值后广播（测「一个 token 是否已含全部信息」）
  · repeat   只用第 1 个 token 复制 8 份（测冗余）

⏱ 成本：一次 forward_rows 覆盖 8 个预算；5 个模式 ≈ 22× 单序列前向；greedy 只跑
`--gen-k` 指定的预算（默认 1,8）。

用法::

    python eval_dump_per_sample.py --checkpoint /root/autodl-tmp/vlm8b/output/run3/final \
        --jsonl .../mix_heldout.jsonl --image-root .../data_eval \
        --per-group 120 --seed 7 --gen-k 1,8 \
        --modes normal,null,shuffle,mean,repeat --out logs/screen_heldout.jsonl
"""
import argparse
import json
import os
import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from data import (EightBudgetCollator, SelfDistillSFTDataset,
                  build_image_transform, gt_name)
from model import VLM8Budget, VLM8BudgetConfig, expand_budget_rows
from eval import extract_answer, load_trained_model, _norm

IGNORE_INDEX = -100


# ═══════════════════════════════════════════════════════════════
# 带元信息的数据集（source / task / blank / 问题字符数）
# ═══════════════════════════════════════════════════════════════

class MetaDataset(SelfDistillSFTDataset):
    """`SelfDistillSFTDataset` + 原始 jsonl 行元信息（按 __len__ 对齐）。"""

    def __init__(self, jsonl_path: str, tokenizer, image_root: str, transform):
        super().__init__(jsonl_path, tokenizer, image_root, transform)
        self.meta: List[Dict[str, object]] = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.meta.append(_row_meta(row))

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = super().__getitem__(index)
        item["meta_index"] = int(index)
        return item


def _user_text(messages) -> str:
    """取最后一条 user 消息的文本（list content 拼 text 段）。"""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
        return " ".join(parts)
    return ""


def _row_meta(row: Dict[str, object]) -> Dict[str, object]:
    """jsonl 行 → 分析用元信息（不读图）。"""
    messages = row.get("messages_short") or row.get("messages") or []
    question = _user_text(messages)
    image = str(row.get("image", ""))
    return {
        "id": str(row.get("id", "")),
        "source": str(row.get("source") or "unknown"),
        "task": str(row.get("task") or "unknown"),
        "blank": "_blank" in image,
        "gt": gt_name(row.get("gt", "")),
        "question_chars": len(question),
        "has_digit": any(ch.isdigit() for ch in question),
    }


# ═══════════════════════════════════════════════════════════════
# 视觉消融
# ═══════════════════════════════════════════════════════════════

def apply_mode(vis: torch.Tensor, mode: str) -> torch.Tensor:
    """normal / null / shuffle / mean / repeat。"""
    if mode == "normal":
        return vis
    if mode == "null":
        return vis * 0
    if mode == "shuffle":
        return vis * 0 if vis.size(0) <= 1 else torch.roll(vis, shifts=1, dims=0)
    if mode == "mean":
        return vis.mean(dim=1, keepdim=True).expand_as(vis).contiguous()
    if mode == "repeat":
        return vis[:, :1].expand_as(vis).contiguous()
    if mode == "pad1":
        # 只有第 1 个 token 有内容，其余槽位填零：分离「槽位长度」与「内容」
        out = vis * 0
        out[:, :1] = vis[:, :1]
        return out
    raise ValueError(f"未知 mode: {mode}")


def vis_diag(vis: torch.Tensor) -> Dict[str, object]:
    """视觉 token 冗余度：两两余弦、有效秩（参与率）、范数。"""
    v = vis.float()
    unit = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    sim = torch.bmm(unit, unit.transpose(1, 2))                    # (B,8,8)
    n = sim.size(1)
    eye = torch.eye(n, device=sim.device).unsqueeze(0)
    off = (sim * (1 - eye)).sum(dim=(1, 2)) / (n * (n - 1))
    centered = v - v.mean(dim=1, keepdim=True)
    sv = torch.linalg.svdvals(centered)                            # (B,8)
    energy = sv.pow(2)
    pr = energy.sum(dim=1).pow(2) / energy.pow(2).sum(dim=1).clamp(min=1e-12)
    return {
        "vis_cos_offdiag": off.cpu().tolist(),
        "vis_eff_rank": pr.cpu().tolist(),
        "vis_norm": v.norm(dim=-1).mean(dim=1).cpu().tolist(),
        "z_eff_rank": None,                                        # 由调用方填
    }


# ═══════════════════════════════════════════════════════════════
# 采样：按 (source, task, blank) 分层，每组最多 N 条
# ═══════════════════════════════════════════════════════════════

def stratified_indices(meta: List[Dict[str, object]], per_group: int,
                       seed: int) -> List[int]:
    """每组取 min(per_group, 组大小) 条（固定种子随机）。"""
    groups: Dict[tuple, List[int]] = {}
    for index, row in enumerate(meta):
        groups.setdefault((row["source"], row["task"], row["blank"]), []).append(index)
    rng = random.Random(seed)
    picked: List[int] = []
    for key in sorted(groups):
        pool = groups[key]
        picked += rng.sample(pool, min(per_group, len(pool)))
    return sorted(picked)


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析参数。"""
    p = argparse.ArgumentParser(description="per-sample budget screening dump")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out", required=True, help="逐行 jsonl 输出路径")
    p.add_argument("--per-group", type=int, default=0, help="0=全部样本")
    p.add_argument("--random-n", type=int, default=0,
                   help="per-group=0 时随机抽 N 条（0=全部）")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--budgets", default="1,2,3,4,5,6,7,8")
    p.add_argument("--modes", default="normal,null,shuffle,mean,repeat")
    p.add_argument("--gen-k", default="1,8", help="要 greedy 生成的预算（空=不做）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--limit-batches", type=int, default=0, help=">0 只跑前 N 个 batch（冒烟）")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    """跑完写逐行 jsonl。"""
    args = parse_args()
    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    gen_ks = sorted({int(x) for x in args.gen_k.split(",") if x.strip()})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    config = VLM8BudgetConfig.from_pretrained(args.checkpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 生成用的 pad/eos 必须沿用 config.json（248044/248046）：若用 tokenizer 的
    # pad=eos，generate 会第一个 token 就停 ⇒ 解码空串。

    dataset = MetaDataset(args.jsonl, tokenizer, args.image_root,
                          build_image_transform(train=False))
    if args.per_group > 0:
        indices = stratified_indices(dataset.meta, args.per_group, args.seed)
    elif args.random_n > 0:
        indices = sorted(random.Random(args.seed).sample(
            range(len(dataset)), min(args.random_n, len(dataset))))
    else:
        indices = list(range(len(dataset)))
    print(f"[dump] rows={len(indices)} / {len(dataset)} modes={modes} budgets={budgets}",
          flush=True)
    base_collate = EightBudgetCollator(tokenizer)

    def _collate(features):
        """基础 collate（丢掉额外键）+ 保住 meta_index。"""
        batch = base_collate(features)
        batch["meta_index"] = torch.tensor(
            [int(f["meta_index"]) for f in features], dtype=torch.long)
        return batch

    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=_collate)
    model = load_trained_model(args.checkpoint, config)
    model.eval()
    lm = model.lm
    text_embed = lm.get_input_embeddings()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    done = 0
    with open(args.out, "w", encoding="utf-8") as sink:
        for batch_no, batch in enumerate(loader):
            if args.limit_batches and batch_no >= args.limit_batches:
                break
            pixel = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            enc_dtype = next(model.encoder.parameters()).dtype
            z_s = model.encoder(pixel.to(dtype=enc_dtype))
            vis = model.projector(z_s.to(model.projector.token_proj.weight.dtype))
            text_embeds = text_embed(ids)
            ce_by_mode: Dict[str, torch.Tensor] = {}
            for mode in modes:
                rows = expand_budget_rows(
                    apply_mode(vis, mode), text_embeds.to(vis.dtype), mask, labels, budgets)
                out = model.forward_rows(rows)
                ce_by_mode[mode] = out["per_budget_ce_per_sample"].float().cpu()  # (B,V)

            diag = vis_diag(vis)
            zc = z_s.float() - z_s.float().mean(dim=1, keepdim=True)
            zsv = torch.linalg.svdvals(zc)
            zpr = (zsv.pow(2).sum(dim=1).pow(2) / zsv.pow(4).sum(dim=1).clamp(min=1e-12))

            preds: Dict[int, List[str]] = {}
            if gen_ks:
                for k in gen_ks:
                    preds[k] = []
                    for b in range(pixel.size(0)):
                        # ⚠️ 只喂 prompt（到 answer 之前）：喂全文的话模型会看到答案，
                        # 之后继续生成的内容与答案无关（实测大量空串，acc 被严重低估）。
                        n_prompt = int(mask[b].sum()) - int(
                            (labels[b] != IGNORE_INDEX).sum())
                        embeds = torch.cat(
                            [vis[b, :k], text_embeds[b, :n_prompt]], dim=0).unsqueeze(0)
                        row_mask = torch.ones(embeds.shape[:2], dtype=torch.long,
                                              device=embeds.device)
                        gen = lm.generate(inputs_embeds=embeds, attention_mask=row_mask,
                                          max_new_tokens=args.max_new_tokens,
                                          do_sample=False,
                                          pad_token_id=config.pad_token_id,
                                          eos_token_id=config.eos_token_id)
                        token_ids = gen[0].tolist()
                        eos = config.eos_token_id
                        if eos is not None and eos in token_ids:
                            token_ids = token_ids[:token_ids.index(eos)]
                        preds[k].append(extract_answer(
                            tokenizer.decode(token_ids, skip_special_tokens=True)))

            for i in range(pixel.size(0)):
                meta = dataset.meta[int(batch["meta_index"][i])]
                text_len = int(mask[i].sum())
                answer_len = int((labels[i] != IGNORE_INDEX).sum())
                record = dict(meta)
                record.update({
                    "prompt_len": text_len - answer_len,
                    "text_len": text_len,
                    "answer_len": answer_len,
                    "ce": {mode: [round(float(v), 5) for v in ce_by_mode[mode][i]]
                           for mode in modes},
                    "vis_cos_offdiag": round(float(diag["vis_cos_offdiag"][i]), 5),
                    "vis_eff_rank": round(float(diag["vis_eff_rank"][i]), 5),
                    "vis_norm": round(float(diag["vis_norm"][i]), 5),
                    "z_eff_rank": round(float(zpr[i]), 5),
                    "pred": {str(k): preds[k][i] for k in gen_ks} if gen_ks else {},
                    "correct": {str(k): int(_norm(preds[k][i]) == _norm(str(meta["gt"])))
                                for k in gen_ks} if gen_ks else {},
                })
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            done += pixel.size(0)
            if batch_no % 10 == 0:
                print(f"[dump] batch={batch_no} rows={done}", flush=True)
    print(f"[dump] DONE rows={done} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
