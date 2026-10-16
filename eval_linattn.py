"""
评测脚本 — AssocVTC 实验版 (linattn)
==========================================

    python eval_linattn.py --ckpt output/linattn_k64/ckpt_best.pt \
        --data_dir /root/autodl-tmp/datasets/imagenet100 \
        --out output/linattn_k64/eval_full.json \
        --grid output/linattn_k64/samples/final_grid.png

可一次给多个 --ckpt 做对照; 全量 val（默认 5000 张）计算:
    L1(归一化空间) / PSNR / SSIM / per-patch-DC 基线 PSNR / 均值图基线 PSNR。
PSNR 口径警告（仓库主线结论）: **PSNR 单独不可信**, 必须同时优于两个基线。
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from data_linattn import build_datasets
from metrics_linattn import per_patch_dc_baseline, psnr, ssim
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, LinAttnConfig,
                              SRPhase1LinAttn, patches_to_image)
from torch.utils.data import DataLoader


def load_model(ckpt_path: str, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    model = SRPhase1LinAttn(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


@torch.no_grad()
def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, va_ds = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    n = min(len(va_ds), args.num_images) if args.num_images > 0 else len(va_ds)
    va = DataLoader(torch.utils.data.Subset(va_ds, range(n)),
                    batch_size=args.batch_size, shuffle=False, drop_last=False,
                    num_workers=args.num_workers, pin_memory=True)

    results = {}
    grid_in, grid_rec = [], []
    for ckpt in args.ckpt:
        model, cfg, ck = load_model(ckpt, device)
        patch = cfg.patch
        # 按**图数**加权累加（早期版本 append 每批的 mean 再等权平均 ⇒ 末批 8 张
        # 与 128 张等权, 系统性偏低 ~0.06dB; 改为 sum/count）
        tot_l1 = tot_p = tot_s = tot_dc = tot_c = 0.0
        const01 = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        seen = 0
        grid_done = False
        for x, _ in va:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=bool(device == "cuda" and args.amp)):
                out = model(x)
            Bb = x.shape[0]
            _l1 = F.l1_loss(out["F_hat"].float(), out["target_pix"].float()) * Bb
            tot_l1 += float(_l1)
            # ⚠️ IMAGENET_MEAN/STD 是 [0,1] 标度 ⇒ 反归一化后直接是 [0,1]，不再 /255
            rec01 = patches_to_image(out["F_hat"].float(), args.img_size,
                                     args.img_size, patch, IMAGENET_MEAN,
                                     IMAGENET_STD, clamp_255=False).clamp(0.0, 1.0)
            tgt01 = patches_to_image(out["target_pix"].float(), args.img_size,
                                     args.img_size, patch, IMAGENET_MEAN,
                                     IMAGENET_STD, clamp_255=False).clamp(0.0, 1.0)
            # ★运行时护栏: 反归一化单位口径写错会整体虚高 ~48dB（metrics_linattn §4）
            assert 0.02 < float(tgt01.mean()) < 0.98, (
                f"反归一化单位口径疑似错误: tgt01.mean()={float(tgt01.mean()):.6f}")
            tot_p += float(psnr(rec01, tgt01).sum())
            tot_s += float(ssim(rec01, tgt01).sum())
            tot_dc += float(per_patch_dc_baseline(tgt01, patch).sum())
            tot_c += float(psnr(const01.expand_as(tgt01), tgt01).sum())
            if args.grid and not grid_done and seen + x.shape[0] >= args.grid_n:
                grid_in.append(tgt01[:args.grid_n - seen].float().cpu())
                grid_rec.append(rec01[:args.grid_n - seen].float().cpu())
                grid_done = True
            seen += x.shape[0]
        res = {
            "ckpt": ckpt, "step": ck.get("step"), "epoch": ck.get("epoch"),
            "tag": ck.get("tag", ""),
            "cfg": ck["cfg"], "params": ck.get("params"),
            "n_images": seen,
            "l1_norm": tot_l1 / seen,
            "psnr": tot_p / seen,
            "ssim": tot_s / seen,
            "psnr_patch_dc_baseline": tot_dc / seen,
            "psnr_mean_image_baseline": tot_c / seen,
            "pred_clamped_to_01": True,
        }
        res["psnr_gain_over_patch_dc"] = res["psnr"] - res["psnr_patch_dc_baseline"]
        results[os.path.basename(os.path.dirname(ckpt)) + "/" + os.path.basename(ckpt)] = res
        print(f"[eval] {ckpt}\n        l1={res['l1_norm']:.4f} psnr={res['psnr']:.3f}dB "
              f"ssim={res['ssim']:.4f} | patchDC基线={res['psnr_patch_dc_baseline']:.2f}dB "
              f"({res['psnr_gain_over_patch_dc']:+.2f}dB) | "
              f"均值基线={res['psnr_mean_image_baseline']:.2f}dB", flush=True)
        if len(args.ckpt) > 1:
            del model
            torch.cuda.empty_cache()

    if args.grid and grid_in:
        a = make_grid(torch.cat(grid_in), nrow=args.grid_n, padding=2, value_range=(0, 1))
        b = make_grid(torch.cat(grid_rec), nrow=args.grid_n, padding=2, value_range=(0, 1))
        sep = torch.ones(3, 4, a.shape[2])
        canvas = torch.cat([a, sep, b], dim=1)
        os.makedirs(os.path.dirname(args.grid), exist_ok=True)
        save_image(canvas, args.grid)
        print(f"[grid] {args.grid} (上=原图, 下=重建)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"[out] {args.out}", flush=True)
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--num_images", type=int, default=0, help="0 = 全量 val")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--grid", default="")
    ap.add_argument("--grid_n", type=int, default=8)
    ap.add_argument("--out", default="")
    return ap.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
