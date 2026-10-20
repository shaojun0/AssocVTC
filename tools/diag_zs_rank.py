#!/usr/bin/env python3
"""diag_zs_rank.py — 诊断 z_s 冗余度（K-sweep 用；2026-10-20）

对若干 ckpt 的前向 z_s 统计（留出集固定子集, 默认 128 张）:
  · pairwise cos（两两余弦, 取上三角均值）与 frac(cos>0.9)
  · effective rank: 每图 z_s 去均值后 SVD, 奇异值归一化后的熵指数, 再对图平均
    （E-A5 口径的变体; 0=完全坍缩, K=完全正交）
  · token variance（每个 token 维度的方差再平均）

用法:
  PYTHONPATH=$PWD python tools/diag_zs_rank.py --out /path/diag_zs_rank.json \
      --ckpts /path/a.pt /path/b.pt ...
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from data_linattn import build_datasets
from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    m = SRPhase1LinAttn(cfg).cuda()
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, va = build_datasets(args.data_dir, 224, 50)
    stride = max(1, len(va) // args.n)
    idx = list(range(0, len(va), stride))[:args.n]
    dl = DataLoader(torch.utils.data.Subset(va, idx), batch_size=64,
                    shuffle=False, num_workers=8, pin_memory=True)
    xs = torch.cat([x for x, _ in dl], 0)[:args.n].cuda()

    res = {}
    for p in args.ckpts:
        if not os.path.isfile(p):
            print(f"[skip] {p} 不存在")
            continue
        m, cfg, ck = load_model(p)
        with torch.no_grad():
            _, z_s = m.encode(xs)
        z = z_s.float()
        K = z.shape[1]
        zm = z - z.mean(1, keepdim=True)
        s = torch.linalg.svdvals(zm)
        p_ = s / s.sum(1, keepdim=True).clamp_min(1e-9)
        eff = float((-(p_ * p_.clamp_min(1e-9).log()).sum(1)).exp().mean())
        zn = torch.nn.functional.normalize(z, dim=-1)
        cos = torch.einsum("bik,bjk->bij", zn, zn)
        tri = torch.triu(torch.ones(K, K, dtype=torch.bool, device=z.device),
                         diagonal=1)
        c = cos[:, tri]
        d = {"ckpt": p, "K": K, "epoch": ck.get("epoch"),
             "tag": ck.get("tag", ""), "best_psnr": ck.get("best_psnr"),
             "n_images": int(z.shape[0]),
             "pairwise_cos": float(c.mean()),
             "frac_cos_gt_0.9": float((c > 0.9).float().mean()),
             "eff_rank": eff, "eff_rank_over_K": eff / K,
             "token_var_mean": float(z.var(0).mean())}
        res[os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)] = d
        print(f"{os.path.basename(os.path.dirname(p)):<28} K={K:>3} ep={d['epoch']} "
              f"val_psnr={d['best_psnr'] if d['best_psnr'] is not None else float('nan'):.2f} "
              f"cos={d['pairwise_cos']:.3f} frac>0.9={d['frac_cos_gt_0.9']:.4f} "
              f"eff_rank={eff:.2f} ({eff / K:.2f}xK) var={d['token_var_mean']:.2e}")
        del m
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
