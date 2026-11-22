"""Fast per-step rFID (in-memory, no PNG dump) — ALIT/FID protocol.

Same feature extractor as tools/eval_rfid.py: official FID InceptionV3 pool3
(2048-d, resize 299, [0,1]→[-1,1]).  Instead of dumping PNG dirs and calling
`calculate_fid_given_paths`, we accumulate the Inception feature mean/covariance
in memory for each decoding step (and for GT) and call pytorch-fid's exact
`calculate_frechet_distance`.  Reconstructions are quantized to the uint8 grid
(round(x*255)/255) to match the PNG path; sole deviation is the 448→299 resize
(F.interpolate vs PIL), which is sub-0.1 FID.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/torchvision/build/lib.linux-x86_64-cpython-310")
sys.path.insert(0, "/root/AssocVTC_tok448")

import scipy.linalg as _sl
if not getattr(_sl.sqrtm, "_fid_compat", False):
    _orig_sqrtm = _sl.sqrtm

    def _sqrtm(A, disp=True, **kw):
        r = _orig_sqrtm(A, **kw)
        return (r, 0.0) if disp is False else r

    _sqrtm._fid_compat = True
    _sl.sqrtm = _sqrtm

from pytorch_fid.fid_score import calculate_frechet_distance  # noqa: E402
from pytorch_fid.inception import InceptionV3                # noqa: E402

from data_linattn import build_datasets                      # noqa: E402
from metrics_linattn import psnr                             # noqa: E402
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD,   # noqa: E402
                              LinAttnConfig, SRPhase1LinAttn,
                              patches_to_image)

D = 2048


def add_feats(acc, f):
    acc["n"] += f.shape[0]
    acc["s"] += f.sum(0)
    acc["ss"] += f.T @ f


def stats(acc):
    n = acc["n"]
    mu = acc["s"] / n
    cov = (acc["ss"] - n * np.outer(mu, mu)) / (n - 1)
    return mu, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--num_images", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    model = SRPhase1LinAttn(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    recurrent = getattr(model, "recurrent", False)
    T = len(model.decoder.steps) if recurrent else 1
    windows = model.decoder.windows if recurrent else [(0, cfg.num_specials)]
    cum_z = [int(hi) for (lo, hi) in windows]
    print(f"[fast] ckpt={args.ckpt}\n  recurrent={recurrent} T={T} cum_z_s={cum_z} "
          f"K={cfg.num_specials} zs_dim={getattr(cfg,'zs_dim',0)}", flush=True)

    _, va = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    if args.num_images > 0:
        va = torch.utils.data.Subset(va, list(range(min(args.num_images, len(va)))))
    dl = torch.utils.data.DataLoader(va, batch_size=args.batch_size, shuffle=False,
                                     drop_last=False, num_workers=args.num_workers,
                                     pin_memory=False)
    print(f"[fast] val images = {len(va)}", flush=True)

    inc = InceptionV3([3], resize_input=True, normalize_input=True).to(dev).eval()
    for p in inc.parameters():
        p.requires_grad_(False)

    rec_acc = [{"n": 0, "s": np.zeros(D, np.float64),
                "ss": np.zeros((D, D), np.float64)} for _ in range(T)]
    gt_acc = {"n": 0, "s": np.zeros(D, np.float64),
              "ss": np.zeros((D, D), np.float64)}
    psnr_sum = [0.0] * T
    l1_sum = [0.0] * T
    seen = 0

    with torch.no_grad():
        for x, _ in dl:
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                out = model(x)
            B = x.shape[0]
            pix = out["Y_pix"].float()
            tgt = out["target_pix"].float()
            rec01 = patches_to_image(
                pix.reshape(B * T, cfg.num_patches, -1), cfg.img_size, cfg.img_size,
                cfg.patch, IMAGENET_MEAN, IMAGENET_STD, clamp_255=False
            ).reshape(B, T, 3, cfg.img_size, cfg.img_size).clamp(0.0, 1.0)
            tgt01 = patches_to_image(
                tgt, cfg.img_size, cfg.img_size, cfg.patch,
                IMAGENET_MEAN, IMAGENET_STD, clamp_255=False
            ).reshape(B, 3, cfg.img_size, cfg.img_size).clamp(0.0, 1.0)
            recq = (rec01 * 255.0).round() / 255.0
            tgtq = (tgt01 * 255.0).round() / 255.0
            # GT features (once)
            fg = inc(tgtq)[0].flatten(1).float().cpu().numpy().astype(np.float64)
            add_feats(gt_acc, fg)
            for t in range(T):
                fr = inc(recq[:, t])[0].flatten(1).float().cpu().numpy().astype(np.float64)
                add_feats(rec_acc[t], fr)
                ps = psnr(rec01[:, t], tgt01)
                psnr_sum[t] += float(ps.sum())
                l1_sum[t] += float((rec01[:, t] - tgt01).abs().mean(dim=(1, 2, 3)).sum())
            seen += B
            if seen % 640 == 0:
                print(f"[fast] {seen}/{len(va)} …", flush=True)

    assert seen == len(va), (seen, len(va))
    n = seen
    mu_gt, sig_gt = stats(gt_acc)
    per_step = {}
    for t in range(T):
        mu_r, sig_r = stats(rec_acc[t])
        fid = float(calculate_frechet_distance(mu_r, sig_r, mu_gt, sig_gt, eps=1e-6))
        per_step[t + 1] = {
            "step": t + 1,
            "cum_z_s": cum_z[t],
            "values": cum_z[t] * int(getattr(cfg, "zs_dim", 0) or cfg.enc_dim),
            "psnr": psnr_sum[t] / n,
            "l1_255": l1_sum[t] / n * 255.0,
            "rfid": fid,
        }
        print(f"[fast] step{t+1} cum_z={cum_z[t]:3d} "
              f"PSNR={psnr_sum[t]/n:.3f} L1_255={l1_sum[t]/n*255:.3f} rFID={fid:.3f}",
              flush=True)

    res = {
        "ckpt": args.ckpt, "tag": args.tag, "n_images": n,
        "K": cfg.num_specials, "zs_dim": int(getattr(cfg, "zs_dim", 0) or cfg.enc_dim),
        "recurrent": bool(recurrent), "T": T, "cum_z_s": cum_z,
        "img_size": int(cfg.img_size), "patch": int(cfg.patch),
        "rfid_protocol": "FID-InceptionV3 pool3 2048d, resize299, [0,1]->[-1,1], "
                         "uint8-quantized in-memory (pytorch-fid equivalent)",
        "per_step": {str(k): v for k, v in per_step.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"[fast] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
