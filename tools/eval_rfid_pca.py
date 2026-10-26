#!/usr/bin/env python
"""per-patch PCA 基线的 rFID（同信息量对照, 2026-10-26）

背景
----
用户在 2026-10-26 定了新口径：论文以 **240M 档**（enc 244.43M + dec 237.11M =
481.54M, K=32, n=2 token/步, T=16, fixed 计划）为基准，对比指标改为 **rFID**，
对比方法用 **PCA**。

本文方法的每图信息量（连续标量个数）：
    K × d = 32 × 1248 = **39,936** 个数/图
（注: 旧 63M 档 enc_dim=448 ⇒ K×d=28,672；240M 档 enc_dim=1248, 见
 `model_v2_linattn.py` 中 `specials` 的形状 (1,K,enc_dim)。2026-10-26 的
 任务书里写的 32×448=14,336 是**旧 63M 架构**的数，240M 档不适用，故本脚本
 默认额外算 rank 204 = 196×204 = 39,984 个数（与 39,936 差 +0.12%）。）

per-patch PCA 基线（与 `tools/baseline_svd_stepwise.py --mode per_patch_pca` 同族）：
  · 在 **ImageNet-100 训练集** patch 上拟合一个 **共享** PCA 基（768 维，
    768 = 3×16×16），**不逐图 oracle**；基的存储量算"模型参数"，不计入每图预算；
  · 每张图切成 N=196 个 patch，每块用前 r 个主成分重建 ⇒ 每图 **196×r** 个数；
  · 同信息量 = 196×r ≈ K×d ⇒ r = round(39936/196) = 204。
  另算 r=37 / 73 / 146 作参照（分别≈16/32/64 个旧 448 维 token 的信息量，
  便于与旧 63M/K=64 协议对照）。

口径（与 tools/eval_rfid.py 逐项一致）
  · 像素空间：0–255（dataset 归一化 x ⇒ x·STD_255 + MEAN_255，不 clamp）；
  · L1_255：0–255 空间、**不 clamp** 的平均绝对误差（与模型 stepwise 的
    l1_255 同口径，可直接比）；
  · PSNR：[0,1] clamp 空间、**逐图 PSNR 再平均**（与 eval_linattn 同口径）；
  · 存 PNG：clamp 到 [0,1] 再 round×255（与 eval_rfid.save_pngs 同做法）；
  · rFID：`pytorch_fid.fid_score.calculate_fid_given_paths([recon, gt_ref],
    batch_size=fid_batch, device=fid_device, dims=2048)` = InceptionV3 pool3、
    dims=2048、resize 299，与 `python -m pytorch_fid` / ALIT 官方口径一致。
    gt_ref 优先用预计算的 `gt_stats.npz`（省一半时间），否则用 GT 图目录。

PCA 基的确定性
  · `torch.manual_seed(--seed)`（+ cuda 手动种子）；
  · 用 **协方差精确特征分解**（`torch.linalg.eigh(768×768)`，float64 累加）
    取前 q 个特征向量，而不是 `torch.pca_lowrank` 的随机化近似 ⇒ 同一份训练
    数据 + 同一 seed 必然得到同一组基（特征向量符号不影响重建，重建对符号不变）。

用法（服务器 /root/AssocVTC 根目录; 纯 CPU 拟合 + GPU 只跑 FID）
----
python tools/eval_rfid_pca.py \
    --out_dir /root/autodl-tmp/rfid_pca --ranks 37,73,146,204 \
    --gt_dir /root/autodl-tmp/rfid_k64/gt \
    --gt_stats /root/autodl-tmp/rfid_k64/gt_stats.npz \
    --data_dir /root/autodl-tmp/datasets/imagenet100

产物
----
  out_dir/rank{r}/recon/000000.png …   第 r 阶 PCA 重建（5000 张, 224×224）
  out_dir/pca_basis.pt                 mean (768,) + basis (768, max_rank)
  out_dir/eval_rfid_pca.json           cfg / 信息预算 / 每 rank 的 PSNR、L1、rFID
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_linattn import build_datasets  # noqa: E402
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, image_to_patches,
                              patches_to_image)  # noqa: E402

MEAN_255 = tuple(v * 255.0 for v in IMAGENET_MEAN)
STD_255 = tuple(v * 255.0 for v in IMAGENET_STD)


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具（与 eval_rfid.py 同口径）
# ─────────────────────────────────────────────────────────────────────────────
def save_pngs(t01: torch.Tensor, out_dir: str, offset: int):
    """t01: (B,3,H,W) [0,1] → out_dir/{offset+j:06d}.png（与 ALIT save_imgs 同做法）"""
    os.makedirs(out_dir, exist_ok=True)
    arrs = t01.detach().float().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    arrs = np.round(arrs * 255.0).astype(np.uint8)
    for j, a in enumerate(arrs):
        Image.fromarray(a).save(os.path.join(out_dir, f"{offset + j:06d}.png"))


def psnr01(p01: torch.Tensor, t01: torch.Tensor) -> torch.Tensor:
    """逐图 PSNR（[0,1] 空间, 与 metrics_linattn.psnr 同口径）→ (B,)。"""
    mse = ((p01 - t01) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def to_patches255(x: torch.Tensor, img: int, patch: int) -> torch.Tensor:
    """dataset 的归一化 x → (B,N,3pp) 的 0–255 patch 向量（不 clamp）。"""
    m = torch.tensor(MEAN_255).view(1, 3, 1, 1)
    s = torch.tensor(STD_255).view(1, 3, 1, 1)
    return image_to_patches(x * s + m, patch)


def _patch_scipy_sqrtm():
    """pytorch-fid 0.3.0 与 scipy≥1.16 的兼容 shim（同 eval_rfid.py）。"""
    import scipy.linalg as _sl
    if getattr(_sl.sqrtm, "_fid_compat", False):
        return
    _orig = _sl.sqrtm

    def _sqrtm(A, disp=True, **kw):
        r = _orig(A, **kw)
        return (r, 0.0) if disp is False else r

    _sqrtm._fid_compat = True
    _sl.sqrtm = _sqrtm


def compute_fid(recon_dir: str, gt_ref: str, args) -> float:
    """gt_ref: GT 图目录或预计算 .npz；口径 = pytorch_fid dims=2048 / resize 299。"""
    _patch_scipy_sqrtm()
    from pytorch_fid.fid_score import calculate_fid_given_paths
    t0 = time.time()
    fid = calculate_fid_given_paths(
        [recon_dir, gt_ref],
        batch_size=args.fid_batch,
        device=args.fid_device,
        dims=args.fid_dims,
        num_workers=args.fid_workers,
    )
    print(f"[fid] {os.path.basename(recon_dir)} vs {os.path.basename(gt_ref)} → "
          f"{fid:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    return float(fid)


# ─────────────────────────────────────────────────────────────────────────────
# PCA 基拟合（训练集 patch, 共享, 非逐图 oracle）
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def fit_patch_pca(args):
    """在训练集 patch 上拟合共享 PCA 基。

    返回 (mean (768,), basis (768, q))；q = min(max_rank, 767)。
    基 = 协方差矩阵的前 q 个特征向量（按特征值降序）。
    """
    tr, _ = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    idx = list(range(0, len(tr), max(1, len(tr) // args.basis_images)))[:args.basis_images]
    sub = torch.utils.data.Subset(tr, idx)
    dl = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    dim = 3 * args.patch ** 2
    n = 0
    s1 = torch.zeros(dim, dtype=torch.float64)
    s2 = torch.zeros(dim, dim, dtype=torch.float64)
    for x, _ in dl:
        P = to_patches255(x, args.img_size, args.patch).reshape(-1, dim).double()
        s1 += P.sum(0)
        s2 += P.t() @ P
        n += P.shape[0]
    mean = (s1 / n).float()
    cov = (s2 / n - (s1 / n).unsqueeze(1) * (s1 / n).unsqueeze(0)).float()
    evals, evecs = torch.linalg.eigh(cov)                  # 升序
    basis = evecs[:, torch.argsort(evals, descending=True)]
    q = min(args.max_rank, dim - 1)
    basis = basis[:, :q].contiguous()
    ev = torch.argsort(evals, descending=True)
    frac = (evals[ev] / evals.sum()).clamp_min(0)
    print(f"    [pca] 训练图 {len(idx)} 张 → patch {n} 个; 基 {tuple(basis.shape)}; "
          f"前 5 解释方差 {[round(float(v), 4) for v in frac[:5]]}; "
          f"前 {q} 累计 {float(frac[:q].sum()):.4f}", flush=True)
    return mean, basis


def build_val_loader(args):
    _, va = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    if args.max_images > 0:
        va = torch.utils.data.Subset(va, range(min(args.max_images, len(va))))
    dl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False)
    return dl, len(va)


# ─────────────────────────────────────────────────────────────────────────────
# 逐 rank 重建 + 指标 + 落盘
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run(args, mean, basis, ranks):
    dl, n_img = build_val_loader(args)
    recon_dirs = {r: os.path.join(args.out_dir, f"rank{r}", "recon") for r in ranks}
    for d in recon_dirs.values():
        if os.path.isdir(d) and len(os.listdir(d)) == n_img:
            print(f"[pca] 已存在 {len(os.listdir(d))} 张, 复用: {d}", flush=True)
    sums = {r: {"l1_255": 0.0, "psnr": 0.0} for r in ranks}
    t0 = time.time()
    seen = 0
    for bi, (x, _) in enumerate(dl):
        P = to_patches255(x, args.img_size, args.patch)          # (B,N,768) 0–255
        B = P.shape[0]
        Pc = P.reshape(B, -1, basis.shape[0]) - mean
        C = Pc @ basis                                            # (B,N,q) 系数
        for r in ranks:
            P_rec = (C[:, :, :r] @ basis[:, :r].t()).reshape(B, -1, basis.shape[0]) + mean
            # 0–255 空间（不 clamp, 与 eval_rfid 的 l1_255 同口径）
            r255 = patches_to_image(P_rec, args.img_size, args.img_size, args.patch,
                                    clamp_255=False)
            t255 = patches_to_image(P, args.img_size, args.img_size, args.patch,
                                    clamp_255=False)
            l1 = float((r255 - t255).abs().mean())
            p01 = (r255 / 255.0).clamp(0, 1)
            t01 = (t255 / 255.0).clamp(0, 1)
            p = psnr01(p01, t01)
            sums[r]["l1_255"] += l1 * B
            sums[r]["psnr"] += float(p.sum())
            save_pngs(p01.view(B, 3, args.img_size, args.img_size),
                      recon_dirs[r], seen)
        seen += B
        if (bi + 1) % 10 == 0:
            print(f"    [pca] {seen}/{n_img} 图  {time.time() - t0:.1f}s", flush=True)
    assert seen == n_img, (seen, n_img)

    res = {}
    for r in ranks:
        d = recon_dirs[r]
        have = len(os.listdir(d)) if os.path.isdir(d) else 0
        assert have == n_img, f"{d} 只有 {have}/{n_img} 张"
        res[r] = {
            "rank": r,
            "numbers_per_image": int(196 * r),
            "l1_255": sums[r]["l1_255"] / n_img,
            "psnr": sums[r]["psnr"] / n_img,
            "recon_dir": d,
        }
        if not args.no_fid:
            res[r]["rfid"] = compute_fid(d, args.gt_ref, args)
    return res, n_img


def main():
    ap = argparse.ArgumentParser(description="per-patch PCA 基线 rFID（同信息量对照）")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/rfid_pca")
    ap.add_argument("--ranks", default="37,73,146,204",
                    help="per-patch 主成分数 r（每图 196×r 个数）")
    ap.add_argument("--max_rank", type=int, default=256,
                    help="PCA 基保留的主成分数上限（须 ≥ max(ranks)）")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--max_images", type=int, default=0, help="0 = 全量 5000 张 val")
    ap.add_argument("--basis_images", type=int, default=400,
                    help="拟合 PCA 基用的训练图片数（与 baseline_svd_stepwise 默认一致）")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    # 参考模型的信息量（只写进 json, 不参与计算）
    ap.add_argument("--model_K", type=int, default=32)
    ap.add_argument("--model_enc_dim", type=int, default=1248)
    # FID
    ap.add_argument("--gt_dir", default="/root/autodl-tmp/rfid_k64/gt")
    ap.add_argument("--gt_stats", default="/root/autodl-tmp/rfid_k64/gt_stats.npz")
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--fid_batch", type=int, default=50)
    ap.add_argument("--fid_dims", type=int, default=2048)
    ap.add_argument("--fid_device", default="cuda")
    ap.add_argument("--fid_workers", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ranks = sorted({int(s) for s in args.ranks.split(",") if s.strip()})
    os.makedirs(args.out_dir, exist_ok=True)
    model_numbers = args.model_K * args.model_enc_dim
    print(f"[pca] out_dir={args.out_dir} ranks={ranks}\n"
          f"  参考模型信息量 = K×d = {args.model_K}×{args.model_enc_dim} = "
          f"{model_numbers} 个数/图；同信息量 rank ≈ "
          f"{model_numbers / 196:.2f}", flush=True)

    args.gt_ref = args.gt_stats if (args.gt_stats and os.path.exists(args.gt_stats)) \
        else args.gt_dir
    have = len(os.listdir(args.gt_dir)) if os.path.isdir(args.gt_dir) else 0
    print(f"[pca] gt_dir={args.gt_dir} ({have} 张)  gt_ref={args.gt_ref}", flush=True)

    t0 = time.time()
    mean, basis = fit_patch_pca(args)
    q = min(max(ranks), basis.shape[1])
    assert q >= max(ranks), f"基只有 {basis.shape[1]} 维, 不足以算 rank {max(ranks)}"
    torch.save({"mean": mean, "basis": basis, "seed": args.seed,
                "basis_images": args.basis_images}, os.path.join(args.out_dir, "pca_basis.pt"))

    res, n_img = run(args, mean, basis, ranks)

    out = {
        "mode": "per_patch_pca",
        "data_dir": args.data_dir,
        "n_images": n_img,
        "img_size": args.img_size, "patch": args.patch,
        "n_patches": (args.img_size // args.patch) ** 2,
        "seed": args.seed,
        "basis": {
            "fit_on": "ImageNet-100 train patches (共享, 非逐图 oracle)",
            "basis_images": args.basis_images,
            "dim": int(basis.shape[0]), "max_rank": int(basis.shape[1]),
            "method": "covariance eigh (float64 累加, 前 q 特征向量)",
            "shared_basis_numbers": int(basis.shape[0] * basis.shape[1]),
            "note": "基的存储量算模型参数, 不计入每图预算（与 baseline_svd_stepwise 一致）",
        },
        "budget": {
            "model": f"K×d = {args.model_K}×{args.model_enc_dim} = "
                     f"{model_numbers} 个数/图（240M 档 z_s: 32×1248）",
            "pca": "196×r 个数/图（每 patch 前 r 个主成分系数）",
            "model_numbers_per_image": model_numbers,
            "matched_rank": round(model_numbers / 196),
        },
        "gt_dir": args.gt_dir,
        "gt_ref_used": args.gt_ref,
        "psnr_agg": "per-image PSNR then mean (与 eval_linattn 同口径)",
        "l1_255": "0–255 空间、不 clamp 的平均绝对误差（与 eval_rfid 同口径）",
        "rfid_lib": "pytorch-fid 0.3.0, InceptionV3 pool3, dims=2048, resize 299",
        "ranks": {str(r): res[r] for r in ranks},
    }
    with open(os.path.join(args.out_dir, "eval_rfid_pca.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[pca] → {os.path.join(args.out_dir, 'eval_rfid_pca.json')}", flush=True)
    for r in ranks:
        d = res[r]
        fid = d.get("rfid")
        print(f"[pca]   rank{r:>4} ({d['numbers_per_image']:>6d} 个数/图): "
              f"PSNR={d['psnr']:.3f} dB  L1_255={d['l1_255']:.3f}"
              + (f"  rFID={fid:.4f}" if fid is not None else ""), flush=True)
    print(f"[pca] 总耗时 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
