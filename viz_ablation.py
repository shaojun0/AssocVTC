#!/usr/bin/env python3
"""四臂定性对比图 — 编码器: 线性注意力 vs 普通 transformer（单步 / BPTT）
================================================================================

把同一批留出图的【原图 / 四臂重建】排成一张图, 直观看"换编码器"带来的差异。

    python viz_ablation.py \
      --ckpt linear_T1=/root/autodl-tmp/srdiff_linattn/k64/ckpt_best.pt \
             std_T1=/root/autodl-tmp/srdiff_stdattn/stdattn_k64/ckpt_best.pt \
             linear_BPTT=/root/autodl-tmp/srdiff_linattn/k64_rec/ckpt_best.pt \
             std_BPTT=/root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt \
      --num_images 6 --out doc/2026-10-16/artifacts/ablation_visual.png

行序: 原图 → 线性·单步 → 普通·单步 → 线性·BPTT → 普通·BPTT（可 --order 覆盖）。
每行下方标注该臂的全量留出集 PSNR/SSIM（从 --metrics 指定的 eval json 读取, 可选）。
"""

import argparse
import json
import os

import torch
from torchvision.utils import make_grid, save_image

from data_linattn import build_datasets
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, LinAttnConfig,
                              SRPhase1LinAttn, image_to_patches, patches_to_image)


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    model = SRPhase1LinAttn(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="标签=ckpt路径, 例如 std_T1=/path/ckpt_best.pt")
    ap.add_argument("--order", default="",
                    help="出图行序（逗号分隔标签）; 空 = 按 --ckpt 顺序")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--num_images", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--metrics", default="", help="eval json, 用于行标注 PSNR/SSIM")
    ap.add_argument("--out", default="ablation_visual.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = []
    for s in args.ckpt:
        assert "=" in s, f"--ckpt 需要 标签=路径 形式, got {s!r}"
        # rsplit: 标签里允许出现 '='（例如 "单步(T=1)"），路径里不允许
        lab, path = s.rsplit("=", 1)
        specs.append((lab.strip(), path))
    if args.order:
        want = [x.strip() for x in args.order.split(",") if x.strip()]
        by = dict(specs)
        specs = [(w, by[w]) for w in want]

    metrics = {}
    if args.metrics and os.path.exists(args.metrics):
        for k, v in json.load(open(args.metrics)).items():
            metrics[os.path.basename(os.path.dirname(k))] = v

    _, va_ds = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    idx = range(min(args.num_images, len(va_ds)))
    xs = torch.stack([va_ds[i][0] for i in idx]).to(device)      # (n,3,H,W) 归一化
    print(f"[data] 取留出集前 {xs.shape[0]} 张（确定性顺序, 与评测同一划分）")

    rows, labels = [], []
    # 原图（走与重建**同一**反归一化口径, 保证上下可比）
    _, cfg0, _ = load_model(specs[0][1], device)
    p = cfg0.patch
    tgt = patches_to_image(image_to_patches(xs, p), args.img_size, args.img_size,
                           p, IMAGENET_MEAN, IMAGENET_STD,
                           clamp_255=False).clamp(0, 1)
    rows.append(tgt.cpu())
    labels.append(f"original  (n={xs.shape[0]})")

    for lab, path in specs:
        model, cfg, ck = load_model(path, device)
        out = model(xs)
        rec = patches_to_image(out["F_hat"].float(), args.img_size, args.img_size,
                               cfg.patch, IMAGENET_MEAN, IMAGENET_STD,
                               clamp_255=False).clamp(0, 1)
        rows.append(rec.cpu())
        m = metrics.get(os.path.basename(os.path.dirname(path)), {})
        tag = (f"PSNR {m['psnr']:.2f}dB / SSIM {m['ssim']:.4f}" if m else "")
        labels.append(f"{lab}  [{type(model.encoder).__name__}] {tag}")
        print(f"[recon] {lab:14s} {tag}  (step={ck.get('step')})")
        del model
        torch.cuda.empty_cache()

    # 每行: n 张横排 → 竖向堆叠
    strip = [make_grid(r, nrow=r.shape[0], padding=2, value_range=(0, 1)) for r in rows]
    sep = torch.ones(3, 6, strip[0].shape[2])
    canvas = strip[0]
    for s in strip[1:]:
        canvas = torch.cat([canvas, sep, s], dim=1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(canvas, args.out)
    print(f"[out] {args.out}  行序(上→下): {labels}")


if __name__ == "__main__":
    main()
