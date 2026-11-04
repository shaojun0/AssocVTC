#!/usr/bin/env python
"""pca_448_baseline.py — 「同值数」逐块 PCA 基线（224 自洽校验 + 448 正式）

背景
----
论文表 2（`/userdata/paper/pp_zh/cjc-article-v2.0.pdf`）的「同值数 PSNR：本文 vs PCA vs
ALIT」中，PCA 行 = **per-patch PCA、训练集共享基、非逐图 oracle**（与仓库 2026-10-26
`tools/eval_rfid_pca.py` / `tools/pca_strict_alignment.py` 同族口径）：

  · 基：在 **ImageNet-100 train** 的 patch 上拟合一个共享 PCA 基（224²/p16 时 768 维，
    448²/p32 时 3072 维）；基的存储量算「模型参数」，**不计入每图值数预算**；
  · 每张图切成 N 个 patch，每块用前 r 个主成分系数重建 ⇒ **每图 N×r 个连续值**；
    224²/p16 N=196，448²/p32 仍是 N=196（本扩展实验 patch 面积 ×4、N 不变）；
  · 值与模型的**同值数**对齐：模型 S3 臂第 t 步 = 2t 个 token × 224 维 = 448t 个值
    （t=1..16 ⇒ 448…7168），PCA 侧 r 取 196r 最接近 448t 的整数 rank，另给
    「精确同值数线性插值」列（论文表 2 的 PCA 行就是插值值）。

口径（与主线 eval **逐项一致**，务必不要改）
----
  · 数据：`--data_dir`（ImageNet-100），val = `build_datasets(data_dir, img_size)` 的
    等距 50/类 划分（与主线 `eval_stepwise_linattn.py` 同划分）；基**只用 train** 拟合；
  · patch 向量：`model_v2_linattn.image_to_patches(x, patch)`（通道优先、row-major），
    在 **0–255 patch 空间**拟合与重建（`x·STD_255 + MEAN_255`，不 clamp）；
  · L1_255：0–255 空间、**不 clamp**、逐图再对图平均（= `eval_stepwise_linattn` 的
    `L1_255` 口径）；
  · PSNR：`patches_to_image(..., IMAGENET_MEAN, IMAGENET_STD)` 反归一化后 clamp 到
    [0,1]，用 `metrics_linattn.psnr` **逐图算再平均**（= 主报告 `eval_linattn` 口径）；
  · SSIM：`metrics_linattn.ssim`（11×11 高斯窗、裁边界、逐通道均值），输入同上 [0,1]；
  · 基拟合：协方差精确特征分解（float64 累加 s1/s2，`torch.linalg.eigh` 取前 q 个
    特征向量），**不逐图 oracle**、无随机化近似；数据用 train 的随机裁剪增广
    （与 `eval_rfid_pca.py` 同协议；`--seed` 固定 ⇒ 可复现）。

224 自洽校验
------------
  论文表 2 PCA 行（448…7168 值，共 16 档）已内置为 `PAPER_224_PCA_PSNR`。
  `--img_size 224 --patch 16 --basis_images 400`（+ 可选 `--basis_pt` 用仓库里
  2026-10-26 产物基）可复现该行；脚本会把 16 档插值对照与最大偏差写进 json。

用法（远端示例）
----
  # (A) 224 自洽：用仓库已有基（精确复现 `pca_allranks.json` 那条曲线）
  ASSOCVTC_REPO=/root/autodl-tmp/exp448/code/AssocVTC \
  python pca_448_baseline.py --img_size 224 --patch 16 --basis_images 400 \
      --basis_pt pca_basis_224.pt --ranks 1-40 \
      --out pca_224_selfcheck_exactbasis.json

  # (B) 224 自洽：自己重拟合基（验证拟合链路），再对照论文表 2 PCA 行
  ASSOCVTC_REPO=... python pca_448_baseline.py --img_size 224 --patch 16 \
      --basis_images 400 --ranks 1-40 --out pca_224_selfcheck_refit.json

  # (C) 448 正式基线：train 5 万张拟合 3072 维基，r=1..40 全扫 5000 张 val
  CUDA_VISIBLE_DEVICES=1 ASSOCVTC_REPO=... python pca_448_baseline.py \
      --img_size 448 --patch 32 --basis_images 50000 --ranks 1-40 \
      --out pca_448_ranks.json

产物
----
  --out <json>：每 rank 的 PSNR/SSIM/L1_255、基拟合配置（样本/维数/耗时/解释方差）、
  16 行同值数对照（最近 rank + 精确插值）、可选的模型列（读 `--model_json`）、
  224 自洽对照（仅 224/p16 时）。
  --save_basis <pt>：mean (dim,) + basis (dim, max_rank)，供复用/复现。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# repo 解析（脚本远端放在 /root/autodl-tmp/exp448/pca/，repo 只读别处）
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo(explicit: str | None) -> str:
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("ASSOCVTC_REPO")
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.dirname(here), os.getcwd(),
              "/root/autodl-tmp/exp448/code/AssocVTC", "/root/AssocVTC"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "model_v2_linattn.py")):
            return c
    raise SystemExit("[pca] 找不到 AssocVTC repo（含 model_v2_linattn.py）；"
                     "请用 --repo 或 ASSOCVTC_REPO 指定")


# 论文表 2（v2.0，sec4_exp.tex `tab:psnr`）PCA 行：16 档 448t 个值
PAPER_224_PCA_PSNR = [18.40, 20.41, 21.45, 22.29, 22.88, 23.32, 23.72, 24.16,
                      24.51, 24.82, 25.12, 25.38, 25.61, 25.86, 26.07, 26.25]


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────────────────────────
def parse_ranks(spec: str):
    """'1-40' / '1,2,5-7' → 升序去重 list[int]。"""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    if not out:
        raise SystemExit("[pca] --ranks 为空")
    return sorted(out)


def to_patches255(x: torch.Tensor, patch: int, mean255, std255) -> torch.Tensor:
    """dataset 的归一化 x (B,3,H,W) → (B,N,3*p*p) 的 0–255 patch 向量（不 clamp）。"""
    m = torch.as_tensor(mean255, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    s = torch.as_tensor(std255, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return image_to_patches(x * s + m, patch)


def _fmt_secs(sec: float) -> str:
    return f"{sec:.1f}s" if sec < 90 else f"{sec / 60:.2f}min"


# ─────────────────────────────────────────────────────────────────────────────
# PCA 基拟合（train patch, 共享, 非逐图 oracle）
# ─────────────────────────────────────────────────────────────────────────────
def fit_patch_pca(args, dev, mean255, std255):
    tr, _ = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    n_take = len(tr) if args.basis_images <= 0 else min(args.basis_images, len(tr))
    step = max(1, len(tr) // n_take)
    idx = list(range(0, len(tr), step))[:n_take]
    sub = torch.utils.data.Subset(tr, idx)
    dl = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(dev.type == "cuda"), drop_last=False)
    dim = 3 * args.patch ** 2
    s1 = torch.zeros(dim, dtype=torch.float64, device=dev)
    s2 = torch.zeros(dim, dim, dtype=torch.float64, device=dev)
    n = 0
    t0 = time.time()
    for bi, (x, _) in enumerate(dl):
        P = to_patches255(x.to(dev, non_blocking=True), args.patch,
                          mean255, std255).reshape(-1, dim).double()
        s1 += P.sum(0)
        s2 += P.t() @ P
        n += P.shape[0]
        if (bi + 1) % max(1, len(dl) // 10) == 0 or bi == 0:
            el = time.time() - t0
            print(f"    [fit] {bi + 1}/{len(dl)} batch  {n} patches  "
                  f"{el:.1f}s ({n / max(el, 1e-9):.0f} patch/s)", flush=True)
    fit_secs = time.time() - t0
    mean64 = s1 / n
    cov = (s2 / n - mean64.unsqueeze(1) * mean64.unsqueeze(0)).float()
    print(f"    [fit] cov {tuple(cov.shape)} 用时 {_fmt_secs(fit_secs)}；eigh …",
          flush=True)
    t1 = time.time()
    evals, evecs = torch.linalg.eigh(cov)                    # 升序
    order = torch.argsort(evals, descending=True)
    basis = evecs[:, order].contiguous()
    ev_sorted = evals[order].clamp_min(0)
    q = min(args.max_rank, dim - 1)
    basis = basis[:, :q].contiguous()
    frac = (ev_sorted / ev_sorted.sum().clamp_min(1e-30)).tolist()
    eig_secs = time.time() - t1
    info = {
        "fit_on": "ImageNet-100 train patches（共享基, 非逐图 oracle）",
        "n_train_images_total": int(len(tr)),
        "n_basis_images": int(len(idx)),
        "n_patches": int(n),
        "dim": int(dim),
        "max_rank": int(q),
        "method": "covariance eigh（float64 累加 s1/s2, float32 特征分解, 前 q 特征向量）",
        "fit_seconds": round(fit_secs, 2),
        "eigh_seconds": round(eig_secs, 2),
        "explained_variance_ratio_top": [round(v, 8) for v in frac[:min(q, 40)]],
        "cumulative_explained_variance_top": [round(float(np.cumsum(frac[:min(q, 40)])[i]), 6)
                                              for i in range(min(q, 40))],
        "note": "基存储量算模型参数, 不计入每图值数预算（与 eval_rfid_pca/baseline_svd_stepwise 一致）",
    }
    print(f"    [fit] basis {tuple(basis.shape)}  今日累计解释方差 "
          f"top8={[round(v, 4) for v in frac[:8]]}  "
          f"top{q}={sum(frac[:q]):.4f}  eigh {_fmt_secs(eig_secs)}", flush=True)
    return mean64.float(), basis, info


# ─────────────────────────────────────────────────────────────────────────────
# 评测（r=1..max_rank 全扫, 逐图口径）
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_ranks(args, dev, mean, basis, ranks, mean255, std255):
    _, va = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    if args.max_images > 0:
        va = torch.utils.data.Subset(va, range(min(args.max_images, len(va))))
    dl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(dev.type == "cuda"))
    n_img = len(va)
    H = W = args.img_size
    patch = args.patch
    rank_set = set(ranks)
    max_rank = max(ranks)
    sums = {r: {"l1_255": 0.0, "psnr": 0.0, "ssim": 0.0} for r in ranks}
    seen = 0
    t0 = time.time()
    for bi, (x, _) in enumerate(dl):
        x = x.to(dev, non_blocking=True)
        P = to_patches255(x, patch, mean255, std255)              # (B,N,dim) 0–255
        B, N, dim = P.shape
        assert N == (H // patch) * (W // patch), (N, H, patch)
        t255 = patches_to_image(P, H, W, patch, clamp_255=False)  # 0–255, 不 clamp
        t01 = (t255 / 255.0).clamp(0, 1)
        C = (P.reshape(B, N, dim) - mean) @ basis                 # (B,N,max_rank)
        # R = mean + 已累计秩-r 重建（PCA 在中心化 patch 空间, 必须把 mean 加回来）
        R = mean.view(1, 1, dim).expand(B, N, dim).clone()
        for j in range(max_rank):
            r = j + 1
            R += C[:, :, j:j + 1] @ basis[:, j:j + 1].t()
            if r not in rank_set:
                continue
            r255 = patches_to_image(R, H, W, patch, clamp_255=False)
            sums[r]["l1_255"] += float((r255 - t255).abs().mean(dim=(1, 2, 3)).sum())
            p01 = (r255 / 255.0).clamp(0, 1)
            sums[r]["psnr"] += float(psnr(p01, t01).sum())
            sums[r]["ssim"] += float(ssim(p01, t01).sum())
        seen += B
        if (bi + 1) % max(1, len(dl) // 10) == 0:
            el = time.time() - t0
            print(f"    [eval] {seen}/{n_img} 图  {el:.1f}s "
                  f"({seen / max(el, 1e-9):.1f} img/s)", flush=True)
    assert seen == n_img, (seen, n_img)
    eval_secs = time.time() - t0
    per_rank = {}
    for r in ranks:
        per_rank[str(r)] = {
            "rank": int(r),
            "values_per_image": int(N * r),
            "l1_255": sums[r]["l1_255"] / n_img,
            "psnr": sums[r]["psnr"] / n_img,
            "ssim": sums[r]["ssim"] / n_img,
        }
    return per_rank, n_img, eval_secs


# ─────────────────────────────────────────────────────────────────────────────
# 同值数对照表（最近 rank + 精确插值）
# ─────────────────────────────────────────────────────────────────────────────
def same_value_table(args, per_rank, ranks, n_patches):
    """模型每图值数 = 2t × match_dim（t=1..T），PCA 每图值数 = N×r。"""
    rs = np.asarray(ranks, dtype=np.float64)
    ps = np.asarray([per_rank[str(r)]["psnr"] for r in ranks], dtype=np.float64)
    l1 = np.asarray([per_rank[str(r)]["l1_255"] for r in ranks], dtype=np.float64)
    ss = np.asarray([per_rank[str(r)]["ssim"] for r in ranks], dtype=np.float64)
    vv = rs * n_patches
    rows = []
    for t in range(1, args.T + 1):
        target = 2 * t * args.match_dim
        r_exact = target / n_patches
        nearest = int(rs[np.argmin(np.abs(vv - target))])
        # 精确同值数：在 rank 轴上线性插值（论文表 2 PCA 行的做法）
        interp = lambda arr: float(np.interp(r_exact, rs, arr))
        rows.append({
            "t": t,
            "model_values": int(target),
            "pca_rank_exact": round(float(r_exact), 4),
            "pca_rank_nearest": nearest,
            "pca_values_nearest": int(nearest * n_patches),
            "pca_values_rel_diff_pct": round(100.0 * (nearest * n_patches - target) / target, 3),
            "pca_psnr_nearest": round(float(per_rank[str(nearest)]["psnr"]), 4),
            "pca_l1_255_nearest": round(float(per_rank[str(nearest)]["l1_255"]), 4),
            "pca_ssim_nearest": round(float(per_rank[str(nearest)]["ssim"]), 4),
            "pca_psnr_interp": round(interp(ps), 4),
            "pca_l1_255_interp": round(interp(l1), 4),
            "pca_ssim_interp": round(interp(ss), 4),
        })
    return rows


def load_model_column(path, T):
    """读主线 eval_stepwise json 的 16 档 PSNR/L1_255（逐图再平均）。缺 → None。"""
    if not path or not os.path.isfile(path):
        return None, f"未找到模型列 json: {path}（TODO: 等主线 S3 eval 产出后自动读）"
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    p = d.get("PSNR") or d.get("psnr")
    l1 = d.get("L1_255") or d.get("l1_255")
    if p is None:
        return None, f"{path} 里没有 PSNR 字段（keys={list(d.keys())[:12]}）"
    p = [float(v) for v in p]
    if len(p) < T:
        return None, f"{path} 的 PSNR 只有 {len(p)} 档（期望 {T}）"
    out = {"source": path, "psnr": p[:T],
           "l1_255": ([float(v) for v in l1[:T]] if l1 is not None else None),
           "ckpt": d.get("ckpt"), "step": d.get("step"), "n_images": d.get("n_images"),
           "T": d.get("T"), "K": d.get("K")}
    return out, "ok"


def attach_model_column(rows, model):
    for row in rows:
        t = row["t"]
        if model is None:
            row["model_psnr"] = None
            row["model_l1_255"] = None
            row["delta_psnr_model_minus_pca_interp"] = None
            continue
        row["model_psnr"] = round(float(model["psnr"][t - 1]), 4)
        row["model_l1_255"] = (round(float(model["l1_255"][t - 1]), 4)
                               if model.get("l1_255") else None)
        row["delta_psnr_model_minus_pca_interp"] = round(
            row["model_psnr"] - row["pca_psnr_interp"], 4)
    return rows


def paper_224_check(rows):
    """我的 224 曲线（插值）vs 论文表 2 PCA 行。"""
    diffs = [row["pca_psnr_interp"] - paper
             for row, paper in zip(rows, PAPER_224_PCA_PSNR)]
    return {
        "paper_row_psnr": PAPER_224_PCA_PSNR,
        "mine_psnr_interp": [row["pca_psnr_interp"] for row in rows],
        "diff_mine_minus_paper": [round(v, 4) for v in diffs],
        "max_abs_diff": round(float(np.max(np.abs(diffs))), 4),
        "mean_abs_diff": round(float(np.mean(np.abs(diffs))), 4),
        "pass_0p5dB": bool(np.max(np.abs(diffs)) <= 0.5),
        "note": "论文 PCA 行 = rank 1..40 实测曲线在 r=448t/196 处线性插值（图 2 脚本口径）",
    }


def compare_per_rank(per_rank, ref_path):
    """与既有 per-rank json（如 pca_allranks.json）逐 rank 对照。"""
    if not ref_path or not os.path.isfile(ref_path):
        return None
    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    ref_ranks = ref.get("per_rank") or ref.get("ranks") or {}
    diffs_p, diffs_l = {}, {}
    for r, v in per_rank.items():
        if r not in ref_ranks:
            continue
        rv = ref_ranks[r]
        rp = rv.get("psnr")
        rl = rv.get("l1_255", rv.get("l1"))
        if rp is not None:
            diffs_p[r] = round(v["psnr"] - float(rp), 5)
        if rl is not None:
            diffs_l[r] = round(v["l1_255"] - float(rl), 5)
    if not diffs_p:
        return None
    return {
        "ref_json": ref_path,
        "n_ranks_compared": len(diffs_p),
        "psnr_diff_mine_minus_ref": diffs_p,
        "l1_255_diff_mine_minus_ref": diffs_l,
        "max_abs_psnr_diff": round(float(max(abs(v) for v in diffs_p.values())), 5),
        "max_abs_l1_diff": round(float(max(abs(v) for v in diffs_l.values())), 5) if diffs_l else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="同值数逐块 PCA 基线（224 自洽校验 + 448 正式），口径与主线 eval 一致",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 数据 / 维度
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--max_images", type=int, default=0, help="0 = 全量 val 5000 张")
    # PCA 基
    ap.add_argument("--basis_images", type=int, default=50000,
                    help="拟合基用的 train 图片数（等距抽样；<=0 表示全量 train）")
    ap.add_argument("--max_rank", type=int, default=40, help="保留/评测的主成分数上限")
    ap.add_argument("--basis_pt", default="", help="从 pt 加载 mean+basis（跳过拟合）")
    ap.add_argument("--save_basis", default="", help="把 mean+basis 存成 pt")
    # 同值数网格
    ap.add_argument("--T", type=int, default=16, help="模型读出步数档数")
    ap.add_argument("--match_dim", type=int, default=224, help="模型每 token 维数")
    # 运行
    ap.add_argument("--ranks", default="1-40", help="PCA rank 列表：'1-40' 或 '1,2,5-7'")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="评测设备；拟合跟随同设备（auto: 有 CUDA 就用）")
    ap.add_argument("--threads", type=int, default=0, help="CPU 线程数（0 = 默认）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", default="", help="AssocVTC repo 路径（默认自动探测）")
    # 产物 / 对照
    ap.add_argument("--out", default="pca_out.json", help="结果 json 路径")
    ap.add_argument("--model_json", default="/root/autodl-tmp/exp448/eval/stage3_stepwise.json",
                    help="主线 S3 逐步 eval json（填模型列；不存在则标 TODO）")
    ap.add_argument("--compare_per_rank_json", default="",
                    help="既有 per-rank json（如 224 的 pca_allranks.json）做逐 rank 对照")
    args = ap.parse_args()

    repo = _find_repo(args.repo)
    sys.path.insert(0, repo)
    global build_datasets, image_to_patches, patches_to_image, psnr, ssim, \
        IMAGENET_MEAN, IMAGENET_STD
    from data_linattn import build_datasets
    from metrics_linattn import psnr, ssim
    from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, image_to_patches,
                                  patches_to_image)
    MEAN_255 = tuple(v * 255.0 for v in IMAGENET_MEAN)
    STD_255 = tuple(v * 255.0 for v in IMAGENET_STD)

    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(f"[pca] device={dev}  {torch.cuda.get_device_name(dev)}  "
              f"mem_allocated={torch.cuda.memory_allocated(dev) / 2**20:.0f}MiB", flush=True)
    else:
        print(f"[pca] device=cpu  threads={torch.get_num_threads()}", flush=True)

    ranks = parse_ranks(args.ranks)
    if max(ranks) > args.max_rank:
        raise SystemExit(f"[pca] --ranks 最大 {max(ranks)} > --max_rank {args.max_rank}")
    N = (args.img_size // args.patch) ** 2
    assert args.img_size % args.patch == 0
    print(f"[pca] repo={repo}\n   {args.img_size}x{args.img_size}/patch{args.patch} "
          f"N={N} dim={3 * args.patch ** 2} ranks={ranks[0]}..{ranks[-1]}  "
          f"model 值数=2t×{args.match_dim}（t=1..{args.T} ⇒ "
          f"{2 * args.match_dim}…{2 * args.T * args.match_dim}）", flush=True)
    print(f"[pca] val={args.data_dir}  basis_images={args.basis_images}  "
          f"out={args.out}", flush=True)

    t_all = time.time()
    # ── 基 ──
    if args.basis_pt:
        ck = torch.load(args.basis_pt, map_location="cpu")
        mean = ck["mean"].float().to(dev)
        basis = ck["basis"].float().to(dev)
        assert basis.shape[0] == 3 * args.patch ** 2, basis.shape
        basis = basis[:, :args.max_rank].contiguous()
        info = {"fit_on": "外部加载（--basis_pt）", "basis_pt": os.path.abspath(args.basis_pt),
                "loaded_basis_shape": [int(basis.shape[0]), int(basis.shape[1])],
                "dim": int(basis.shape[0]), "max_rank": int(basis.shape[1]),
                "note": "基存储量算模型参数, 不计入每图值数预算"}
        print(f"[pca] 加载基 {args.basis_pt} → {tuple(basis.shape)}", flush=True)
        fit_secs = 0.0
    else:
        mean, basis, info = fit_patch_pca(args, dev, MEAN_255, STD_255)
        fit_secs = info["fit_seconds"] + info["eigh_seconds"]
        if args.save_basis:
            torch.save({"mean": mean.detach().cpu(),
                        "basis": basis.detach().cpu(),
                        "meta": {k: info[k] for k in
                                 ("n_basis_images", "n_patches", "dim", "max_rank")},
                        "seed": args.seed},
                       args.save_basis)
            print(f"[pca] 基已存 {args.save_basis}", flush=True)

    # ── 评测 ──
    print(f"[pca] 开始评测 {args.img_size}² rank {ranks[0]}..{ranks[-1]} …", flush=True)
    per_rank, n_img, eval_secs = eval_ranks(args, dev, mean, basis, ranks,
                                            MEAN_255, STD_255)
    rows = same_value_table(args, per_rank, ranks, N)
    model, model_note = load_model_column(args.model_json, args.T)
    rows = attach_model_column(rows, model)

    out = {
        "script": os.path.abspath(__file__),
        "repo": repo,
        "mode": "per_patch_pca",
        "config": {
            "data_dir": args.data_dir,
            "img_size": args.img_size, "patch": args.patch, "n_patches": int(N),
            "patch_dim": 3 * args.patch ** 2,
            "val_per_class": args.val_per_class, "n_val_images": int(n_img),
            "ranks": ranks, "max_rank": int(args.max_rank),
            "seed": args.seed, "device": str(dev),
            "batch_size": args.batch_size, "num_workers": args.num_workers,
            "basis_images": args.basis_images,
            "same_value_grid": {
                "model_values": [2 * t * args.match_dim for t in range(1, args.T + 1)],
                "formula": f"模型第 t 步 = 2t token × {args.match_dim} 维；PCA = N×r = "
                           f"{N}×r"},
        },
        "basis": info,
        "metrics": {
            "per_rank": per_rank,
            "psnr_agg": "逐图 PSNR 再平均（metrics_linattn.psnr, [0,1] clamp 口径）",
            "l1_255": "0–255 空间、不 clamp、逐图再平均（eval_stepwise_linattn 口径）",
            "ssim": "metrics_linattn.ssim（11×11 高斯窗, 裁边界, 通道均值, [0,1] clamp）",
            "pred_clamped_to_01": True,
        },
        "same_value_table": rows,
        "model_column": {
            "path": args.model_json,
            "status": model_note,
            "loaded": model is not None,
            "data": model,
            "note": "模型列 = 主线 S3 臂（448²/p32/N196/K32/T16/zs_dim224）"
                    "每步 PSNR/L1_255；不存在则 TODO，等 /root/autodl-tmp/exp448/eval/"
                    "stage3_stepwise.json 产出后重跑本脚本即可自动填",
        },
        "paper_224_check": (paper_224_check(rows)
                            if (args.img_size == 224 and args.patch == 16) else None),
        "compare_per_rank": compare_per_rank(per_rank, args.compare_per_rank_json),
        "timing": {"fit_seconds": round(fit_secs, 2), "eval_seconds": round(eval_secs, 2),
                   "total_seconds": round(time.time() - t_all, 2)},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[pca] WROTE {args.out}", flush=True)

    # ── 控制台摘要 ──
    print("\n=== 每图值数 vs 指标 ===")
    print(" rank  values   PSNR/dB   L1_255   SSIM")
    for r in ranks:
        d = per_rank[str(r)]
        print(f"{r:5d} {d['values_per_image']:7d}  {d['psnr']:8.3f}  "
              f"{d['l1_255']:7.3f}  {d['ssim']:.4f}")
    print("\n=== 16 行同值数对照（最近 rank / 精确插值）===")
    hdr = (f"{'t':>2} {'model值':>7} {'r_exact':>7} {'r_near':>6} {'PCA值':>7} "
           f"{'PSNR近':>7} {'PSNR插':>7} {'L1近':>6}")
    if model is not None:
        hdr += f" {'模型PSNR':>8} {'Δ':>7}"
    print(hdr)
    for row in rows:
        line = (f"{row['t']:>2} {row['model_values']:>7} {row['pca_rank_exact']:>7.3f} "
                f"{row['pca_rank_nearest']:>6} {row['pca_values_nearest']:>7} "
                f"{row['pca_psnr_nearest']:>7.3f} {row['pca_psnr_interp']:>7.3f} "
                f"{row['pca_l1_255_nearest']:>6.3f}")
        if model is not None:
            line += (f" {row['model_psnr']:>8.3f} "
                     f"{row['delta_psnr_model_minus_pca_interp']:>+7.3f}")
        print(line)
    if out["paper_224_check"] is not None:
        pc = out["paper_224_check"]
        print(f"\n[224 自洽] vs 论文表 2 PCA 行：max|Δ|={pc['max_abs_diff']:.4f} dB, "
              f"mean|Δ|={pc['mean_abs_diff']:.4f} dB, "
              f"{'PASS' if pc['pass_0p5dB'] else 'FAIL(>0.5dB)'}")
    if out["compare_per_rank"] is not None:
        cp = out["compare_per_rank"]
        print(f"[逐 rank 对照] {cp['ref_json']}：max|ΔPSNR|="
              f"{cp['max_abs_psnr_diff']:.5f} dB over {cp['n_ranks_compared']} ranks")
    print(f"[pca] 模型列：{model_note}")
    print(f"[pca] 总耗时 {_fmt_secs(time.time() - t_all)} "
          f"(fit {_fmt_secs(fit_secs)}, eval {_fmt_secs(eval_secs)})", flush=True)


if __name__ == "__main__":
    main()
