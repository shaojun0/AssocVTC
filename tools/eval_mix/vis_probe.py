"""
视觉信息探针 —— 8 个 visual token 里到底有多少（跨图可区分的）信息？
====================================================================

两个互补探针：

1) **dispersion（离散度）**：对每个 source 取 N 张图，算
   · `within`  同一图内 8 个 token 的两两余弦（越大越冗余）
   · `across`  不同图的 token 之间余弦（越大 ⇒ 编码器/投影头把不同图映射到
               几乎同一点 ⇒ 下游没有可用的细粒度信息）
   · `across_centered` 去均值后的跨图余弦（扣掉"所有图都有的公共分量"）
   · `pr`      跨图矩阵的参与率（有效维数；远小于 N ⇒ 图都挤在一个低维子空间）

2) **attention（注意力份额）**：full-attention 层里，文本 query 对视觉位置的
   注意力占自身总注意力的比例，并与"均匀分布下的份额" k/L 对比。
   ⚠️ Qwen3.5 是 hybrid（线性注意力层不产生 attention 矩阵），只统计 full 层。

用法::

    python vis_probe.py --checkpoint .../run3/final --jsonl .../mix_heldout.jsonl \
        --image-root .../data_eval --per-source 60 --attention-rows 16 \
        --out logs/vis_probe_heldout.json
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import (EightBudgetCollator, SelfDistillSFTDataset,
                  build_image_transform)
from model import VLM8BudgetConfig
from eval import load_trained_model
from eval_dump_per_sample import MetaDataset, stratified_indices

IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    """解析参数。"""
    p = argparse.ArgumentParser(description="visual token information probe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--per-source", type=int, default=60)
    p.add_argument("--attention-rows", type=int, default=16,
                   help="做注意力探针的样本数（0=跳过）")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    return p.parse_args()


def cos_matrix_mean(features: np.ndarray) -> float:
    """(n,d) → 两两余弦均值（去对角）。"""
    unit = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
    sim = unit @ unit.T
    n = sim.shape[0]
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def participation_ratio(features: np.ndarray) -> float:
    """(n,d) 的参与率 PR = (Σλ)²/Σλ²（λ = 奇异值²），用中心化后的矩阵。"""
    centered = features - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular.astype(np.float64) ** 2
    return float(energy.sum() ** 2 / max((energy ** 2).sum(), 1e-12))


@torch.no_grad()
def run(args: argparse.Namespace) -> Dict[str, object]:
    """前向取 vis / z_s，做离散度统计；可选做注意力份额。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = VLM8BudgetConfig.from_pretrained(args.checkpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = MetaDataset(args.jsonl, tokenizer, args.image_root,
                          build_image_transform(train=False))
    indices = stratified_indices(dataset.meta, args.per_source, args.seed)
    base_collate = EightBudgetCollator(tokenizer)

    def _collate(features):
        batch = base_collate(features)
        batch["meta_index"] = torch.tensor(
            [int(f["meta_index"]) for f in features], dtype=torch.long)
        return batch

    loader = DataLoader(Subset(dataset, indices), batch_size=4, num_workers=2,
                        collate_fn=_collate)
    model = load_trained_model(args.checkpoint, config).eval()
    if args.attention_rows > 0:
        # sdpa 不返回 attention 矩阵；临时切 eager（只影响本进程）
        try:
            model.lm.set_attn_implementation("eager")
        except Exception as error:                      # noqa: BLE001
            print(f"[probe] eager 切换失败：{error}")
    encoder_dtype = next(model.encoder.parameters()).dtype

    store: Dict[str, Dict[str, List[np.ndarray]]] = {}
    attention_rows = 0
    attention_stats: List[Dict[str, object]] = []
    for batch in loader:
        pixel = batch["pixel_values"].to(device)
        z_s = model.encoder(pixel.to(dtype=encoder_dtype))
        vis = model.projector(z_s.to(model.projector.token_proj.weight.dtype))
        vis_np = vis.float().cpu().numpy()
        z_np = z_s.float().cpu().numpy()
        metas = [dataset.meta[int(i)] for i in batch["meta_index"]]
        for row, meta in enumerate(metas):
            key = f"{meta['source']}|{'txt' if meta['blank'] else 'img'}"
            bucket = store.setdefault(key, {"vis": [], "z": []})
            if len(bucket["vis"]) < args.per_source:
                bucket["vis"].append(vis_np[row])
                bucket["z"].append(z_np[row])

        # 注意力探针：只统计 full-attention 层里文本 query 对视觉位置的注意力
        if attention_rows < args.attention_rows and batch["input_ids"].size(0) > 0:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            text_embeds = model.lm.get_input_embeddings()(ids).to(vis.dtype)
            k = 8
            prompt_lens = (mask.sum(dim=1) - (labels != IGNORE_INDEX).sum(dim=1))
            for b in range(ids.size(0)):
                if attention_rows >= args.attention_rows:
                    break
                n = int(prompt_lens[b])
                embeds = torch.cat([vis[b, :k], text_embeds[b, :n]], dim=0).unsqueeze(0)
                row_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
                out = model.lm.model(inputs_embeds=embeds, attention_mask=row_mask,
                                     use_cache=False, output_attentions=True)
                shares = []
                for attn in (out.attentions or []):
                    if attn is None:
                        continue
                    # attn: (1, heads, L, L)；视觉 key 位置 0..k-1，文本 query 位置 k..
                    weights = attn[0].float()                      # (heads,L,L)
                    text_queries = weights[:, k:, :]               # 文本 query 行
                    total = text_queries.sum(dim=-1)               # (heads,T)
                    to_vis = text_queries[:, :, :k].sum(dim=-1)    # (heads,T)
                    shares.append(float((to_vis.sum() / total.sum()).item()))
                length = int(embeds.shape[1])
                attention_stats.append({
                    "source": metas[b]["source"], "blank": metas[b]["blank"],
                    "k": k, "length": length, "uniform_share": k / length,
                    "attn_share": float(np.mean(shares)) if shares else float("nan"),
                    "n_full_layers": len(shares),
                })
                attention_rows += 1
    del model
    torch.cuda.empty_cache()

    result: Dict[str, object] = {"checkpoint": args.checkpoint,
                                 "jsonl": args.jsonl, "groups": {},
                                 "attention": attention_stats}
    for key, bucket in store.items():
        vis = np.stack(bucket["vis"])                # (n,8,2560)
        z = np.stack(bucket["z"])                    # (n,32,1744)
        n = vis.shape[0]
        vis_flat = vis.reshape(n, -1)
        z_flat = z.reshape(n, -1)
        within = cos_matrix_mean(vis.reshape(n * 8, -1))
        centered = vis_flat - vis_flat.mean(axis=0, keepdims=True)
        result["groups"][key] = {
            "n": n,
            "within": within,
            "across": cos_matrix_mean(vis_flat),
            "across_centered": cos_matrix_mean(centered),
            "pr_vis": participation_ratio(vis_flat),
            "pr_z": participation_ratio(z_flat),
            "vis_norm": float(np.linalg.norm(vis_flat, axis=1).mean()),
        }
        print(f"[probe] {key:18s} n={n:3d} within={within:.3f} "
              f"across={result['groups'][key]['across']:.3f} "
              f"centered={result['groups'][key]['across_centered']:.3f} "
              f"pr_vis={result['groups'][key]['pr_vis']:.1f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"[probe] wrote {args.out} (attention rows={len(attention_stats)})")


def main() -> None:
    """入口。"""
    run(parse_args())


if __name__ == "__main__":
    main()
