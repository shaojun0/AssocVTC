#!/usr/bin/env python
"""rFID（reconstruction FID）评测 — 对齐 ALIT 论文指标口径（2026-10-21）

为什么需要这个脚本
------------------
`doc/2026-10-19/REPORT_k8sweep.md` 的 K-sweep 主表只有像素指标
（L1 / PSNR / SSIM）。ALIT（arXiv:2411.02393, ICLR 2025）报告重建质量用的是
**rFID**（reconstruction FID），本脚本把本仓库 K=64 臂补上同一口径的 rFID，
方便与 ALIT 的 Table 3 / Fig.10 直接对齐。

ALIT 的 rFID 口径（官方仓库 `ShivamDuggal4/adaptive-length-tokenizer`）
---------------------------------------------------------------
`evaluate_rfid.py` 的流程：
  1. 把留出集逐图重建，**GT 与重建图分别写成 PNG**（uint8、RGB、同一分辨率）；
  2. 调 `python -m pytorch_fid {recon_dir} {gt_dir}` 算 FID（pytorch-fid 默认：
     InceptionV3 pool3，dims=2048，图 resize 到 299×299）。
本脚本用**同一个库**（`pytorch_fid.fid_score.calculate_fid_given_paths`）、同样
的 dir 结构、同样的 dims/resize，因此数值口径与 ALIT 一致。

与 ALIT 的**不可比之处**（必须同读）
---------------------------------
  · 数据：ALIT 在 ImageNet-100 val（及其 COCO/WIT OOD）上评；本仓库 val = IN-100
    的 5,000 张确定性留出（每类等距 50 张），也是 5,000 张、224×224。
  · 分辨率：ALIT 256×256；本仓库 224×224（patch16 ⇒ N=196）。
  · 模型：ALIT 是带 VQGAN base tokenizer + GAN loss 的量化 1D tokenizer；
    本仓库是从零训的连续 z_s + 线性注意力解码器。**绝对 rFID 不可跨模型比较**，
    只能做「同协议内 K 之间」的比较、以及把本仓库数字放进 ALIT 的指标列格式里。

用法
----
# 0) 先把 GT 落盘一次（所有臂共用, 避免重复写 5000 张）
python tools/eval_rfid.py --gt_only --out_dir /root/autodl-tmp/rfid_k64

# 1) 单臂：末步（K=64 常用口径）
python tools/eval_rfid.py --ckpt .../stdattn_k64_t8/ckpt_best.pt \
    --out_dir /root/autodl-tmp/rfid_k64 --tag k64_t8 --steps last

# 2) 原生 square 计划 K=64：逐步（3,8,15,24,35,48,63,64 token）都算 rFID
#    ⇒ 与 ALIT「同一模型、不同 token 预算」的 FID-vs-tokens 表同构
python tools/eval_rfid.py --ckpt .../_ref_k64_native/ckpt_best.pt \
    --out_dir /root/autodl-tmp/rfid_k64 --tag native_k64 --steps all

产物
----
  out_dir/gt/000000.png …                     GT（val transform 后, 224×224）
  out_dir/<tag>/recon_step{t:02d}/000000.png  第 t 步重建（t=1..T）
  out_dir/<tag>/eval_rfid.json                cfg / cum_z / 每步 PSNR、L1、rFID
"""
import argparse
import json
import os
import time

import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_linattn import build_datasets  # noqa: E402  (需先补仓库根到 sys.path)
from metrics_linattn import psnr
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, LinAttnConfig,
                              SRPhase1LinAttn, patches_to_image)


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device="cuda"):
    """与 eval_linattn.py 逐位同口径地重建 cfg / 载入权重。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    model = SRPhase1LinAttn(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def save_pngs(t01: torch.Tensor, out_dir: str, offset: int):
    """t01: (B,3,H,W) [0,1] → out_dir/{offset+j:06d}.png（与 ALIT save_imgs 同做法）"""
    os.makedirs(out_dir, exist_ok=True)
    arrs = t01.detach().float().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    arrs = np.round(arrs * 255.0).astype(np.uint8)
    for j, a in enumerate(arrs):
        Image.fromarray(a).save(os.path.join(out_dir, f"{offset + j:06d}.png"))


def build_val_loader(args):
    _, va_ds = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    n = min(len(va_ds), args.num_images) if args.num_images > 0 else len(va_ds)
    idx = list(range(n))
    return DataLoader(torch.utils.data.Subset(va_ds, idx),
                      batch_size=args.batch_size, shuffle=False,
                      drop_last=False, num_workers=args.num_workers,
                      pin_memory=True), n


# ─────────────────────────────────────────────────────────────────────────────
# GT 落盘（所有臂共用；只做一次）
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def dump_gt(args, gt_dir: str, n_expect: int):
    have = len(os.listdir(gt_dir)) if os.path.isdir(gt_dir) else 0
    if have == n_expect:
        print(f"[gt] 已存在 {have} 张, 跳过: {gt_dir}", flush=True)
        return
    assert not (0 < have < n_expect), f"[gt] {gt_dir} 只有 {have}/{n_expect} 张, 删除后重跑"
    t0 = time.time()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    va, n = build_val_loader(args)
    seen = 0
    for x, _ in va:
        # val transform 后的输入反归一化 = 模型 target 像素（与 eval_linattn 的
        # tgt01 同口径: clamp 到 [0,1] 后落 PNG）
        x01 = (x.float().cpu() * std + mean).clamp(0.0, 1.0)
        assert 0.2 < float(x01.mean()) < 0.9, f"反归一化口径异常 mean={float(x01.mean())}"
        save_pngs(x01, gt_dir, seen)
        seen += x.shape[0]
        if seen % 1000 == 0:
            print(f"[gt] {seen}/{n} …", flush=True)
    assert seen == n_expect, f"[gt] 实际 {seen} != 期望 {n_expect}"
    print(f"[gt] done {seen} 张, {time.time() - t0:.1f}s → {gt_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# FID
# ─────────────────────────────────────────────────────────────────────────────
def _patch_scipy_sqrtm():
    """pytorch-fid 0.3.0 调 `scipy.linalg.sqrtm(..., disp=False)`（旧 scipy API:
    disp=False 时返回 `(sqrtm, errest)` 二元组）；scipy ≥ 1.16 删除了 `disp` 参数，
    直接报 `TypeError: sqrtm() got an unexpected keyword argument \'disp\'`。
    这里装一个向后兼容 shim（语义与旧 scipy 一致），使 FID 数值与官方命令
    `python -m pytorch_fid` 完全一致；若环境里的 scipy 还支持 disp 则 shim 行为不变。"""
    import scipy.linalg as _sl
    if getattr(_sl.sqrtm, "_fid_compat", False):
        return
    _orig = _sl.sqrtm

    def _sqrtm(A, disp=True, **kw):
        r = _orig(A, **kw)                       # 新 API 返回单个矩阵
        return (r, 0.0) if disp is False else r  # 旧 API: disp=False → (矩阵, 误差估计)

    _sqrtm._fid_compat = True
    _sl.sqrtm = _sqrtm


def save_gt_stats(gt_dir: str, stats_path: str):
    """用官方 CLI 预计算 GT 的 Inception 统计量（.npz）:
    `python -m pytorch_fid --save-stats <gt_dir> <stats.npz>`，后续每个臂只需
    算重建图一侧的统计量 ⇒ FID 时间约减半（多臂/多步时差别很大）。"""
    import subprocess
    t0 = time.time()
    cmd = [sys.executable, "-m", "pytorch_fid", "--save-stats", gt_dir, stats_path]
    print(f"[gt_stats] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[gt_stats] → {stats_path}  ({time.time() - t0:.1f}s)", flush=True)


def compute_fid(recon_dir: str, gt_ref: str, args):
    """gt_ref: GT 图片目录或预计算的 .npz 统计量（pytorch-fid 两种都支持）。"""
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
# 单臂评测
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_arm(args, arm_dir: str, gt_dir: str, n_expect: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ck = load_model(args.ckpt, device)
    patch, img = cfg.patch, cfg.img_size
    recurrent = getattr(model, "recurrent", False)
    steps = list(model.decoder.steps) if recurrent else [None]
    windows = (model.decoder.windows if recurrent else [None])
    T = len(steps)
    cum_z = ([int(hi) for (lo, hi) in windows] if recurrent else [cfg.num_specials])
    all_steps = list(range(1, T + 1))
    if args.steps == "last":
        step_idx = [T]
    elif args.steps == "all":
        step_idx = all_steps
    else:
        want = {int(s) for s in args.steps.split(",") if s.strip()}
        step_idx = [s for s in all_steps if s in want]
        assert step_idx, f"--steps={args.steps} 与 T={T} 无交集"

    print(f"[rfid] {args.ckpt}\n  recurrent={recurrent} T={T} cfg.steps={steps}\n"
          f"  cum_z_s={cum_z}  K={cfg.num_specials}  N={cfg.num_patches}\n"
          f"  评测 step={step_idx}  数据={args.data_dir} val_per_class={args.val_per_class}",
          flush=True)

    recon_dirs = {s: os.path.join(arm_dir, f"recon_step{s:02d}") for s in step_idx}
    for d in recon_dirs.values():
        os.makedirs(d, exist_ok=True)

    va, n = build_val_loader(args)
    assert n == n_expect, f"本臂 {n} 张 != GT {n_expect} 张"

    sums = {s: {"l1_norm": 0.0, "psnr": 0.0, "l1_255": 0.0} for s in step_idx}
    mean_255 = torch.tensor([v * 255.0 for v in IMAGENET_MEAN], device=device).view(1, 3, 1, 1)
    std_255 = torch.tensor([v * 255.0 for v in IMAGENET_STD], device=device).view(1, 3, 1, 1)

    seen = 0
    for x, _ in va:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(device == "cuda" and args.amp)):
            out = model(x)
        pix = out["Y_pix"].float()                       # (B,T,N,3pp)
        tgt = out["target_pix"].float()                  # (B,N,3pp)
        B = x.shape[0]
        assert pix.shape[1] == T
        # 重建像素（[0,1] 空间, clamp; 与 eval_linattn 同口径）
        rec01 = patches_to_image(pix.reshape(B * T, cfg.num_patches, -1), img, img,
                                 patch, IMAGENET_MEAN, IMAGENET_STD,
                                 clamp_255=False).reshape(B, T, 3, img, img).clamp(0.0, 1.0)
        tgt01 = patches_to_image(tgt, img, img, patch, IMAGENET_MEAN, IMAGENET_STD,
                                 clamp_255=False).reshape(B, 3, img, img).clamp(0.0, 1.0)
        # 0-255 空间（不 clamp, 主线 L1_255 口径）
        rec255 = patches_to_image(pix.reshape(B * T, cfg.num_patches, -1), img, img,
                                  patch, mean_255, std_255,
                                  clamp_255=False).reshape(B, T, 3, img, img)
        tgt255 = patches_to_image(tgt, img, img, patch, mean_255, std_255,
                                  clamp_255=False).reshape(B, 3, img, img)
        assert 20.0 < float(tgt255.mean()) < 240.0, f"0-255 口径异常 {float(tgt255.mean())}"
        # 归一化空间 L1（与 eval_linattn 的 l1_norm 同口径, 逐步版）
        l1_norm = F.l1_loss(pix.reshape(B * T, -1),
                            tgt.unsqueeze(1).expand(B, T, tgt.shape[1], tgt.shape[2]).reshape(B * T, -1),
                            reduction="none").mean(dim=1).reshape(B, T)
        l1_255 = (rec255 - tgt255.unsqueeze(1)).abs().mean(dim=(2, 3, 4))     # (B,T)
        psnr_t = psnr(rec01.reshape(B * T, 3, img, img),
                      tgt01.unsqueeze(1).expand(B, T, 3, img, img).reshape(B * T, 3, img, img)
                      ).reshape(B, T)
        for s in step_idx:
            # ⚠️ 按解码步编号取列（s-1），不能按 enumerate 序号 si 取：--steps last
            # 时 si 恒为 0，会错取 step1（2026-10-21 冒烟时踩过）。
            col = s - 1
            sums[s]["l1_norm"] += float(l1_norm[:, col].sum())
            sums[s]["psnr"] += float(psnr_t[:, col].sum())
            sums[s]["l1_255"] += float(l1_255[:, col].sum())
            save_pngs(rec01[:, col], recon_dirs[s], seen)
        seen += B
        if seen % 1000 == 0:
            print(f"[rfid] {seen}/{n} …, 已落盘 step={step_idx}", flush=True)
    assert seen == n, f"实际 {seen} != {n}"

    per_step = {}
    for s in step_idx:
        per_step[s] = {
            "step": s,
            "cum_z_s": int(cum_z[s - 1]),
            "l1_norm": sums[s]["l1_norm"] / n,
            "psnr": sums[s]["psnr"] / n,
            "l1_255": sums[s]["l1_255"] / n,
        }
    gt_ref = args.gt_stats if (args.gt_stats and os.path.exists(args.gt_stats)) else gt_dir
    if not args.no_fid:
        for s in step_idx:
            per_step[s]["rfid"] = compute_fid(recon_dirs[s], gt_ref, args)

    res = {
        "ckpt": args.ckpt,
        "ckpt_step": ck.get("step"), "ckpt_epoch": ck.get("epoch"),
        "K": cfg.num_specials, "N": cfg.num_patches,
        "img_size": int(cfg.img_size), "patch": int(cfg.patch),
        "dec_step_plan": getattr(cfg, "dec_step_plan", None),
        "dec_recurrent": bool(recurrent),
        "T": T, "steps_cfg": steps if recurrent else [],
        "cum_z_s": [int(v) for v in cum_z],
        "evaluated_steps": step_idx,
        "arm_dir": arm_dir, "gt_dir": gt_dir,
        "gt_ref_used": (args.gt_stats if (args.gt_stats and os.path.exists(args.gt_stats)) else gt_dir),
        "data_dir": args.data_dir, "val_per_class": args.val_per_class,
        "n_images": n,
        "psnr_agg": "per-image PSNR then mean (与 eval_linattn 同口径)",
        "rfid_lib": "pytorch-fid 0.3.0, InceptionV3 pool3, dims=2048, resize 299",
        "per_step": {str(k): v for k, v in per_step.items()},
    }
    os.makedirs(arm_dir, exist_ok=True)
    with open(os.path.join(arm_dir, "eval_rfid.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"[rfid] {args.ckpt} → {os.path.join(arm_dir, 'eval_rfid.json')}", flush=True)
    for s in step_idx:
        d = per_step[s]
        fid = d.get("rfid")
        print(f"[rfid]   step{s} (cum_z={d['cum_z_s']:3d}): "
              f"PSNR={d['psnr']:.3f} dB  L1_255={d['l1_255']:.3f}"
              + (f"  rFID={fid:.4f}" if fid is not None else ""), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="rFID 评测（ALIT 口径）")
    ap.add_argument("--ckpt", default="", help="ckpt_best.pt 路径")
    ap.add_argument("--out_dir", required=True, help="产物根目录（gt/ 与 <tag>/ 之下）")
    ap.add_argument("--tag", default="", help="本臂子目录名; 默认取 ckpt 父目录名")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--num_images", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--steps", default="last", help="last | all | 1,2,... (1-based, 对应解码步)")
    ap.add_argument("--gt_only", action="store_true", help="只落 GT, 不加载模型")
    ap.add_argument("--gt_dir", default="", help="GT 目录; 默认 <out_dir>/gt")
    ap.add_argument("--gt_stats", default="",
                    help="GT 像素统计量 .npz（可选; 默认 <out_dir>/gt_stats.npz, 存在即用）")
    ap.add_argument("--gt_stats_only", action="store_true",
                    help="只预计算 GT 的 Inception 统计量 .npz（用官方 pytorch-fid CLI）")
    ap.add_argument("--no_fid", action="store_true", help="只落盘/算像素指标, 不算 FID")
    ap.add_argument("--fid_batch", type=int, default=50)
    ap.add_argument("--fid_dims", type=int, default=2048)
    ap.add_argument("--fid_device", default="cuda")
    ap.add_argument("--fid_workers", type=int, default=8)
    args = ap.parse_args()

    gt_dir = args.gt_dir or os.path.join(args.out_dir, "gt")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    _, n_expect = build_val_loader(args)
    print(f"[rfid] out_dir={args.out_dir} gt_dir={gt_dir} val_images={n_expect}", flush=True)

    gt_stats = args.gt_stats or os.path.join(args.out_dir, "gt_stats.npz")
    args.gt_stats = gt_stats
    if args.gt_only:
        dump_gt(args, gt_dir, n_expect)
        return
    if args.gt_stats_only:
        have = len(os.listdir(gt_dir)) if os.path.isdir(gt_dir) else 0
        assert have == n_expect, f"GT 目录 {gt_dir} 只有 {have} 张, 请先 --gt_only"
        save_gt_stats(gt_dir, gt_stats)
        return
    assert args.ckpt, "非 --gt_only 模式必须给 --ckpt"
    tag = args.tag or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    arm_dir = os.path.join(args.out_dir, tag)
    have = len(os.listdir(gt_dir)) if os.path.isdir(gt_dir) else 0
    assert have == n_expect, (
        f"GT 目录 {gt_dir} 只有 {have} 张, 请先跑 --gt_only 生成 {n_expect} 张"
        "（避免多进程同时写 GT 造成竞争）")
    run_arm(args, arm_dir, gt_dir, n_expect)


if __name__ == "__main__":
    main()
