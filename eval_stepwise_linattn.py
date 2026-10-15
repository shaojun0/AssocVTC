"""
逐 step 评测（循环读出）— SR-Diffusion-v3 实验版 (linattn)
==========================================================

对应主线 `eval_stepwise.py` + `doc/2026-09-16/ANALYSIS_ksweep_stepwise_refinement.md`
的 **"逐步增量表"** 口径:

    python eval_stepwise_linattn.py --ckpt <ckpt.pt> --out stepwise.json

对留出集逐图跑完整 |T| 步循环读出, 拿到**每一步的直接预测** `Y_pix[:, t]`,
然后逐步算:
    · `L1_255[t]`  **0-255 像素 L1（不做 clamp）= 主线口径**, 可直接与
      `ANALYSIS_ksweep_stepwise_refinement.md` 的表对照;
    · `PSNR[t]`    在 [0,1] 空间、预测 clamp 到 [0,1]（与主报告同口径）;
    · 逐步增量 `Δ_t = L1_255[t-1] − L1_255[t]`（**正 = 该步让图像变好**）;
    · 总落差 = L1_255[0] − min_t L1_255[t];Δ₁ 占比;正增量步数;
    · 每图最优停步（oracle）分布。

⚠️ **单位口径**（本项目踩过坑, 见 metrics_linattn.py §4）:
   `IMAGENET_MEAN/STD` 是 **[0,1] 标度**。要得到 0-255 的 L1, 必须把 mean/std
   **乘 255** 再反归一化（`MEAN_255/STD_255`）, 而**不是**把 [0,1] 结果再 /255。
   两种写法在本脚本里显式分开, 并各自断言量级。

单步模型（dec_recurrent=False, |T|=1）也能跑 —— 表里只有一行。
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_linattn import build_datasets
from metrics_linattn import psnr
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, LinAttnConfig,
                              SRPhase1LinAttn, patches_to_image)

MEAN_255 = tuple(v * 255.0 for v in IMAGENET_MEAN)
STD_255 = tuple(v * 255.0 for v in IMAGENET_STD)


def load_model(ckpt_path: str, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    model = SRPhase1LinAttn(cfg).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"ckpt 有多余 key: {unexpected[:5]}"
    assert set(missing) <= {"decoder.mem_pos"}, f"未加载参数: {list(missing)[:8]}"
    model.eval()
    return model, cfg, ck


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ck = load_model(args.ckpt, device)
    patch, img = cfg.patch, cfg.img_size
    recurrent = getattr(model, "recurrent", False)
    steps = list(model.decoder.steps) if recurrent else [None]
    windows = (model.decoder.windows if recurrent else [None])
    T = len(steps)
    # 累计读入的 z_s 数（square 计划: 第 k 步读完前 (k+1)²−1 个）
    cum_z = ([min((int(np.sqrt(int(t))) + 1) ** 2 - 1, cfg.num_specials)
              for t in steps] if recurrent else [cfg.num_specials])
    print(f"[stepwise] {args.ckpt}\n  recurrent={recurrent} T={T} steps={steps}\n"
          f"  windows={windows}\n  cum_z_s={cum_z}  (K={cfg.num_specials}, N={cfg.num_patches})",
          flush=True)

    _, va_ds = build_datasets(args.data_dir, img, args.val_per_class)
    n = min(len(va_ds), args.num_images) if args.num_images > 0 else len(va_ds)
    stride = max(1, len(va_ds) // max(1, n))
    idx = list(range(0, len(va_ds), stride))[:n]
    va = DataLoader(torch.utils.data.Subset(va_ds, idx), batch_size=args.batch_size,
                    shuffle=False, drop_last=False, num_workers=args.num_workers,
                    pin_memory=True)

    sum_l1 = np.zeros(T)        # 0-255 L1（不 clamp, 主线口径）
    sum_mse = np.zeros(T)       # [0,1] MSE（用于 PSNR）
    sum_l1_01 = np.zeros(T)
    sum_psnr_img = np.zeros(T)   # 逐图 PSNR 再平均（与主报告 eval_linattn 同口径）
    per_img = []                # (n, T) 每图逐步 0-255 L1（算 oracle 用）
    sum_const = 0.0             # 常量灰基线（逐图去均值）0-255 L1
    n_img = 0
    for bi, (x, _) in enumerate(va):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(device == "cuda" and args.amp)):
            out = model(x)
        pix = out["Y_pix"].float()                          # (B,T,N,3pp)
        assert pix.shape[1] == T, f"T 不符: {pix.shape[1]} vs {steps}"
        tgt = out["target_pix"].float()                     # (B,N,3pp)
        B = pix.shape[0]
        # ── 0-255 空间（mean/std ×255, **不做 clamp** = 主线 eval_stepwise 口径）
        p255 = patches_to_image(pix.reshape(B * T, cfg.num_patches, -1), img, img,
                                patch, MEAN_255, STD_255, clamp_255=False
                                ).reshape(B, T, 3, img, img)
        t255 = patches_to_image(tgt, img, img, patch, MEAN_255, STD_255,
                                clamp_255=False).reshape(B, 3, img, img)
        # 量级护栏: 0-255 图均值应 ≈ 115（若 ≈0.45 说明漏了 ×255; 若 ≈0.0018 说明多除了）
        assert 20.0 < float(t255.mean()) < 240.0, \
            f"0-255 口径异常: t255.mean()={float(t255.mean()):.4f}"
        sum_const += float((t255 - t255.mean(dim=(1, 2, 3), keepdim=True))
                           .abs().mean()) * B
        d = (p255 - t255.unsqueeze(1)).abs().mean(dim=(2, 3, 4))      # (B,T)
        sum_l1 += d.sum(0).cpu().numpy()
        per_img.append(d.cpu().numpy())
        # ── [0,1] 空间（clamp）用于 PSNR
        p01 = patches_to_image(pix.reshape(B * T, cfg.num_patches, -1), img, img,
                               patch, IMAGENET_MEAN, IMAGENET_STD, clamp_255=False
                               ).reshape(B, T, 3, img, img).clamp(0, 1)
        t01 = patches_to_image(tgt, img, img, patch, IMAGENET_MEAN, IMAGENET_STD,
                               clamp_255=False).reshape(B, 3, img, img).clamp(0, 1)
        sum_l1_01 += (p01 - t01.unsqueeze(1)).abs().mean(dim=(2, 3, 4)).sum(0).cpu().numpy()
        sum_mse += ((p01 - t01.unsqueeze(1)) ** 2).mean(dim=(2, 3, 4)).sum(0).cpu().numpy()
        t01e = t01.unsqueeze(1).expand(B, T, 3, img, img).reshape(B * T, 3, img, img)
        sum_psnr_img += psnr(p01.reshape(B * T, 3, img, img), t01e
                             ).reshape(B, T).sum(0).cpu().numpy()
        n_img += B
        if (bi + 1) % 10 == 0:
            print(f"    {n_img} 张…", flush=True)

    L1 = sum_l1 / n_img
    L1_01 = sum_l1_01 / n_img
    # ⚠️ 两种 PSNR 聚合口径（差的不是模型, 是聚合方式, 实测差 ~1.6dB）:
    #   psnr_img_mean = 逐图 PSNR 再平均  ← 与主报告 eval_linattn.py 同口径
    #   psnr_mean_mse = 先平均 MSE 再转 dB ← 主线 eval_stepwise.py 同口径
    psnr_img_mean = sum_psnr_img / n_img
    psnr_mean_mse = 10 * np.log10(1.0 / np.maximum(sum_mse / n_img, 1e-12))
    PSNR = psnr_img_mean
    curves = np.concatenate(per_img, 0)                 # (n,T)
    best_step = curves.argmin(1)
    # Δ_t = L1[t-1] − L1[t]（**正 = 该步让图像变好**, 主线 ANALYSIS 的约定）
    delta = L1[:-1] - L1[1:]
    drop = float(L1[0] - L1.min())
    best_t = int(L1.argmin())

    res = {"ckpt": args.ckpt, "step": ck.get("step"), "epoch": ck.get("epoch"),
           "recurrent": recurrent, "T": T, "steps": steps, "windows": windows,
           "cum_z_s": cum_z, "n_images": n_img,
           "L1_255": L1.tolist(), "L1_01": L1_01.tolist(),
           "PSNR": PSNR.tolist(),                    # = 逐图 PSNR 再平均
           "PSNR_mean_mse": psnr_mean_mse.tolist(),  # = 先平均 MSE 再转 dB（主线口径）
           "delta_L1_255": delta.tolist(),
           "total_drop_255": drop, "best_step": best_t,
           "delta1_share": (float(delta[0] / drop) if T > 1 and drop > 0 else None),
           "n_positive_steps": int((delta > 0).sum()),
           "best_step_hist": np.bincount(best_step, minlength=T).tolist(),
           "const_baseline_255": sum_const / max(1, n_img)}

    print(f"\n=== 逐步重建（留出集 n={n_img}, 224×224, K={cfg.num_specials}）===")
    if T > 1:
        print("step  t   z_s累计   L1_255(主线口径)   PSNR(dB)   ΔL1(正=变好)")
        for t in range(T):
            dl = f"{delta[t-1]:+8.3f}" if t > 0 else "       —"
            print(f"{t+1:4d} {steps[t]:4d} {cum_z[t]:9d}   {L1[t]:12.3f}   "
                  f"{PSNR[t]:8.3f}   {dl}")
        print(f"\nPSNR 逐图再平均(主报告口径) = {psnr_img_mean[best_t]:.3f}dB @最优步; "
              f"先平均MSE再转dB(主线 eval_stepwise 口径) = {psnr_mean_mse[best_t]:.3f}dB")
        print(f"总落差 L1_255(step1) − min = {L1[0]:.3f} − {L1.min():.3f} = "
              f"{drop:.3f} px  (最优 @step{best_t+1}/t={steps[best_t]})")
        if drop > 0:
            print(f"Δ₁ 占总落差 {float(delta[0]) / drop * 100:.1f}%  |  "
                  f"正增量步数 {int((delta > 0).sum())}/{T - 1}")
        print("前 k 步拿到的降幅占比: " + "  ".join(
            f"k={k}:{((L1[0]-L1[k-1])/drop*100 if drop>0 else float('nan')):.0f}%"
            for k in (1, 2, 3, 4, 6, T)))
        print("每图最优停步分布: " + " ".join(
            f"step{t+1}:{h}" for t, h in enumerate(res["best_step_hist"])))
    else:
        print(f"单步模型 (T=1): L1_255={L1[0]:.3f}  PSNR(逐图再平均)={PSNR[0]:.3f}dB  "
              f"PSNR(平均MSE)={psnr_mean_mse[0]:.3f}dB  L1_01={L1_01[0]:.4f}")
    print(f"常量灰基线（逐图均色）0-255 L1 ≈ {res['const_baseline_255']:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[out] {args.out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--num_images", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
