"""
训练曲线绘图 — SR-Diffusion-v3 实验版 (linattn)
==============================================

    python plot_linattn.py --log <output_dir>/train_log.jsonl --out curves.png

画三条: (1) 训练 loss(recon) vs step; (2) val PSNR vs epoch（含两条基线）;
(3) val SSIM vs epoch。JSONL 里的 val 记录带 "split":"val" 字段。
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path):
    tr, va = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("split") == "val":
                va.append(r)
            elif "loss" in r:
                tr.append(r)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="curves.png")
    ap.add_argument("--title", default="linattn (from-scratch, ImageNet-100)")
    a = ap.parse_args()
    tr, va = read_log(a.log)
    if not tr and not va:
        raise SystemExit(f"日志为空: {a.log}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    if tr:
        axes[0].plot([r["step"] for r in tr], [r["loss"] for r in tr],
                     lw=1.0, color="tab:blue")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("train L1 (normalized)")
        axes[0].set_title("训练损失")
        axes[0].grid(alpha=0.3)
    if va:
        ep = [r["epoch"] for r in va]
        ps = [r["psnr"] for r in va]
        axes[1].plot(ep, ps, "o-", color="tab:green", label="本实验版重建")
        if "psnr_patch_dc_baseline" in va[0]:
            b1 = va[0]["psnr_patch_dc_baseline"]
            axes[1].axhline(b1, ls="--", color="tab:red",
                            label=f"per-patch DC 基线 {b1:.2f}dB")
        if "psnr_mean_image_baseline" in va[0]:
            b0 = va[0]["psnr_mean_image_baseline"]
            axes[1].axhline(b0, ls=":", color="gray",
                            label=f"常量灰基线 {b0:.2f}dB")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val PSNR (dB)")
        axes[1].set_title("验证 PSNR（必须同时优于两条基线）")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[2].plot(ep, [r["ssim"] for r in va], "s-", color="tab:purple")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("val SSIM")
        axes[2].set_title("验证 SSIM")
        axes[2].grid(alpha=0.3)
    fig.suptitle(a.title, fontsize=13)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"[plot] {a.out}  (train 记录 {len(tr)}, val 记录 {len(va)})")
    if va:
        best = max(va, key=lambda r: r["psnr"])
        print(f"[best] epoch {best['epoch']} PSNR {best['psnr']:.3f}dB "
              f"SSIM {best['ssim']:.4f} L1 {best['l1_norm']:.4f}")


if __name__ == "__main__":
    main()
