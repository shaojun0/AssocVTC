#!/usr/bin/env python3
"""baseline_svd_stepwise.py — 「每步 2 个向量还原」的非学习基线：SVD / PCA 逐步还原（2026-10-23）

动机（用户 2026-10-23 口径）:
    n=2 的 fixed 计划下，模型**每步读 2 个 z_s 向量**做一次全图还原；用户要求拿
    「同样每步用 2 个向量还原」的**奇异值分解**做对比。两个口径都算：
      · per_image_svd : 每张图自己的 patch 矩阵 P(N×768) 做 SVD，累计取 rank r = k·t
                        （k = 每步新增奇异向量数）还原 ⇒ **per-image 最优**（oracle,
                        无训练），对应「每步 2 个奇异向量」;
      · per_patch_pca : 在**训练集** patch 上学一个共享 PCA 基（768 维），每块取前
                        r = k·t 个主成分还原（系数 196×r 个数/图）⇒ 有学习基的廉价基线。
    逐步指标与 `eval_stepwise_linattn.py` **同口径**：
      L1_255 = 0-255 空间、不 clamp 的平均绝对误差（可直接与模型 stepwise 的 L1_255 比）;
      PSNR   = [0,1] 空间 clamp 后的**逐图 PSNR 再平均**。
    单位契约: dataset 的 x 是 (img01−MEAN)/STD ⇒ img255 = x·STD_255 + MEAN_255（不 clamp）。

信息预算（报告里必须写清，别让对比失真）:
    · 模型 step t: 2t 个 z_s × 448 维 = 896·t 个数（z_s 由编码器从**整图**算出,
      编码器/解码器参数算共享"模型"不算每图预算）;
    · per_image_svd rank r: r×(196+768+1) = 965·r 个数（左右奇异向量 + 奇异值）;
      ⇒ 用户口径「每步 2 向量」= r=2t ⇒ 1930·t 个数 = 模型的 **2.15×**;
      **预算对齐**口径 = r=t ⇒ 965·t ≈ 模型的 1.08×（所以两个都输出，plot 里都画）;
    · per_patch_pca rank r/块: 196·r 个数 + 768·64 共享基 ⇒ r=2t 时 392·t 个数
      = 模型的 **0.44×**（比模型便宜得多的码率）。

用法（服务器仓库根目录, 纯 CPU 可跑）:
    python tools/baseline_svd_stepwise.py --data_dir /root/autodl-tmp/datasets/imagenet100 \
        --out /root/autodl-tmp/srdiff_n2bigstep/eval/baseline_svd_stepwise.json \
        --mode per_image_svd --per_step_vectors 1 2 --max_rank 64
    python tools/baseline_svd_stepwise.py ... --mode per_patch_pca --max_rank 64 \
        --out .../baseline_perpatch_pca_stepwise.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_linattn import build_datasets
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, image_to_patches,
                              patches_to_image)

MEAN_255 = tuple(v * 255.0 for v in IMAGENET_MEAN)
STD_255 = tuple(v * 255.0 for v in IMAGENET_STD)


def psnr01(p01: torch.Tensor, t01: torch.Tensor) -> torch.Tensor:
    """逐图 PSNR（[0,1] 空间, 与 metrics_linattn.psnr 同口径）→ (B,)。"""
    mse = ((p01 - t01) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def metrics_from_patches(P_rec: torch.Tensor, P_tgt: torch.Tensor,
                         img: int, patch: int) -> tuple:
    """(B,N,3pp) 0-255 空间 patch → (L1_255 标量, 逐图 PSNR (B,))。"""
    B = P_rec.shape[0]
    # 注意: P_rec/P_tgt 已是 **0-255 空间**的 patch 向量 ⇒ 这里只做 layout 还原,
    # 不能再套 mean/std（那是"归一化空间 → 0-255"的仿射, 会二次放大 255 倍）
    r255 = patches_to_image(P_rec, img, img, patch, clamp_255=False)
    t255 = patches_to_image(P_tgt, img, img, patch, clamp_255=False)
    l1 = float((r255 - t255).abs().mean())
    p01 = (r255 / 255.0).clamp(0, 1)
    t01 = (t255 / 255.0).clamp(0, 1)
    return l1, psnr01(p01.view(B, 3, img, img), t01.view(B, 3, img, img))


def load_val(args):
    _, va = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    if args.max_images > 0:
        va = torch.utils.data.Subset(va, range(min(args.max_images, len(va))))
    dl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False)
    return dl, len(va)


def to_patches255(x: torch.Tensor, img: int, patch: int) -> torch.Tensor:
    """dataset 的归一化 x → (B,N,3pp) 的 0-255 patch 向量（不 clamp）。"""
    m = torch.tensor(MEAN_255).view(1, 3, 1, 1)
    s = torch.tensor(STD_255).view(1, 3, 1, 1)
    x255 = x * s + m
    return image_to_patches(x255, patch)


def run_per_image_svd(args, dl, n_img, ranks):
    sums = {r: {"l1": 0.0, "psnr": 0.0} for r in ranks}
    t0 = time.time()
    seen = 0
    for bi, (x, _) in enumerate(dl):
        P = to_patches255(x, args.img_size, args.patch)      # (B,N,3pp) 0-255
        B = P.shape[0]
        # 逐图 SVD: (B,N,Cpp) → U (B,N,N), S (B,min(N,Cpp)), Vh
        U, S, Vh = torch.linalg.svd(P, full_matrices=False)
        cur = torch.zeros_like(P)
        rmax = max(ranks)
        for r in range(1, rmax + 1):
            # 第 r 个 rank-1 分量累加进近似（P_r = P_{r-1} + s_r u_r v_r^T）
            cur = cur + (S[:, r - 1].view(B, 1, 1)
                         * U[:, :, r - 1].unsqueeze(2)
                         * Vh[:, r - 1, :].unsqueeze(1))
            if r in sums:
                l1, p = metrics_from_patches(cur, P, args.img_size, args.patch)
                sums[r]["l1"] += l1 * B
                sums[r]["psnr"] += float(p.sum())
        seen += B
        if (bi + 1) % 10 == 0:
            print(f"    [svd] {seen}/{n_img} 图  {time.time()-t0:.1f}s", flush=True)
    assert seen == n_img, (seen, n_img)
    return {r: {"L1_255": sums[r]["l1"] / n_img, "PSNR": sums[r]["psnr"] / n_img}
            for r in ranks}


def fit_patch_pca(args) -> torch.Tensor:
    """在训练集 patch 上拟合 PCA 基（768 维, 返回 (768, q) 的正交基 + 均值）。"""
    tr, _ = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    idx = list(range(0, len(tr), max(1, len(tr) // args.basis_images)))[:args.basis_images]
    sub = torch.utils.data.Subset(tr, idx)
    dl = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    chunks = []
    for x, _ in dl:
        chunks.append(to_patches255(x, args.img_size, args.patch).reshape(-1, 3 * args.patch ** 2))
    A = torch.cat(chunks, 0).float()                         # (M,768) 0-255
    mean = A.mean(0, keepdim=True)
    Ac = A - mean
    # torch.pca_lowrank 返回 (U,S,V): 主成分在 V 的列
    q = min(args.max_rank + 4, min(Ac.shape) - 1)
    U, S, V = torch.pca_lowrank(Ac, q=q, center=False, niter=4)
    basis = V[:, :args.max_rank]                              # (768,q)
    print(f"    [pca] 训练 patch {A.shape[0]} 个 → 基 {tuple(basis.shape)}, "
          f"解释方差前 5 比例 "
          f"{[round(float(s), 1) for s in (S[:5] ** 2 / (S ** 2).sum())]}", flush=True)
    return mean, basis


def run_per_patch_pca(args, dl, n_img, ranks, mean, basis):
    sums = {r: {"l1": 0.0, "psnr": 0.0} for r in ranks}
    t0 = time.time()
    seen = 0
    for bi, (x, _) in enumerate(dl):
        P = to_patches255(x, args.img_size, args.patch)       # (B,N,3pp) 0-255
        B = P.shape[0]
        Pc = P.reshape(B, -1, 3 * args.patch ** 2) - mean       # (B,N,768)
        C = Pc @ basis                                          # (B,N,q) 系数
        for r in ranks:
            P_rec = (C[:, :, :r] @ basis[:, :r].t()).reshape(
                B, -1, 3 * args.patch ** 2) + mean
            l1, p = metrics_from_patches(P_rec, P, args.img_size, args.patch)
            sums[r]["l1"] += l1 * B
            sums[r]["psnr"] += float(p.sum())
        seen += B
        if (bi + 1) % 10 == 0:
            print(f"    [pca] {seen}/{n_img} 图  {time.time()-t0:.1f}s", flush=True)
    assert seen == n_img, (seen, n_img)
    return {r: {"L1_255": sums[r]["l1"] / n_img, "PSNR": sums[r]["psnr"] / n_img}
            for r in ranks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["per_image_svd", "per_patch_pca"],
                    default="per_image_svd")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=24)
    ap.add_argument("--max_images", type=int, default=0, help="0 = 全量 5000 张 val")
    ap.add_argument("--max_rank", type=int, default=64)
    ap.add_argument("--basis_images", type=int, default=400,
                    help="per_patch_pca 拟合基用的训练图片数")
    args = ap.parse_args()

    dl, n_img = load_val(args)
    # 需要的 rank: 「每步 1/2/3 个向量」× 步数 1..T 的并集（T 上限 = max_rank//k）
    ranks = sorted(set(r for k in (1, 2, 3) for r in range(k, args.max_rank + 1, k)))
    print(f"[baseline] mode={args.mode} val={n_img} 张, ranks={ranks}", flush=True)

    if args.mode == "per_image_svd":
        res = run_per_image_svd(args, dl, n_img, ranks)
    else:
        mean, basis = fit_patch_pca(args)
        res = run_per_patch_pca(args, dl, n_img, ranks, mean, basis)

    out = {
        "mode": args.mode,
        "data_dir": args.data_dir,
        "n_images": n_img,
        "img_size": args.img_size, "patch": args.patch,
        "n_patches": (args.img_size // args.patch) ** 2,
        "budget": (
            "per_image_svd: rank r 每图 r×(N+3pp+1) 个数（含左右奇异向量+奇异值）; "
            "per_patch_pca: rank r/块 每图 N×r 个数 + 768×max_rank 共享基"
        ),
        "ranks": {str(r): res[r] for r in ranks},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out}")
    for r in ranks:
        if r in (1, 2, 4, 8, 16, 32, 64):
            print(f"    rank {r:>2}: L1_255={res[r]['L1_255']:.3f}  PSNR={res[r]['PSNR']:.3f}")


if __name__ == "__main__":
    main()
