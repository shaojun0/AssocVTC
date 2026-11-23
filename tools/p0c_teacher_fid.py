"""P0-c 决策门：量 teacher（SD-VAE）自己的重建 FID。

问题：蒸馏能不能救 rFID，取决于"同量级 rate 的 teacher 自己能把 rFID 压到多少"。
做法：对 IN-100 val 5000 张图，t = SD-VAE(x)，然后按**与 fast_rfid.py 完全相同的口径**算
      FID(real, t) + PSNR/L1(t, x)。

口径（逐条对齐 fast_rfid.py）：
  * 数据：build_datasets(data_dir, 256, 50) 的 val split（Resize(292,BICUBIC) + CenterCrop(256) +
    Normalize(IMAGENET_MEAN/STD)），与训练/评测逐位一致；
  * 特征：官方 FID InceptionV3 pool3 2048d，resize 299，[0,1]->[-1,1]；
  * 图在算特征前量化到 uint8 网格 round(x*255)/255；
  * Fréchet 用 pytorch-fid 的 calculate_frechet_distance（带 scipy.sqrtm 兼容补丁）。

额外参考点（--extra_refs 1，接近零成本）：
  * GT 做高斯模糊 sigma=0.5/1.0/2.0 后的 FID ⇒ 把"糊"换算成 FID 刻度，
    用来判断 86.5 里有多少是"糊"、teacher 的锐化能补多少。

用法：
  python3 p0c_teacher_fid.py --out /root/autodl-tmp/p0c/sdvae_fast.json
  python3 p0c_teacher_fid.py --num_images 64 --smoke 1        # 快速 smoke（含 identity 自检）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

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
from model_v2_linattn import IMAGENET_MEAN, IMAGENET_STD     # noqa: E402

D = 2048
MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def add_feats(acc, f):
    acc["n"] += f.shape[0]
    acc["s"] += f.sum(0)
    acc["ss"] += f.T @ f


def stats(acc):
    n = acc["n"]
    mu = acc["s"] / n
    cov = (acc["ss"] - n * np.outer(mu, mu)) / (n - 1)
    return mu, cov


def new_acc():
    return {"n": 0, "s": np.zeros(D, np.float64), "ss": np.zeros((D, D), np.float64)}


def gauss_kernel(sigma, device):
    r = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    return k, r


def blur(x01, sigma):
    """可分离高斯模糊，(B,3,H,W) 的 [0,1] 图。"""
    k, r = gauss_kernel(sigma, x01.device)
    B, C, H, W = x01.shape
    x = x01.reshape(B * C, 1, H, W)
    kh = k.view(1, 1, 1, -1).expand(1, 1, 1, 2 * r + 1)
    kv = k.view(1, 1, -1, 1).expand(1, 1, 2 * r + 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kh)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kv)
    return x.reshape(B, C, H, W).clamp(0.0, 1.0)


def quant(x01):
    return (x01 * 255.0).round() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--num_images", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--extra_refs", type=int, default=1)
    ap.add_argument("--smoke", type=int, default=0, help="1=含 identity 自检（应 ~0 FID）")
    args = ap.parse_args()
    dev = "cuda"

    # ── teacher ────────────────────────────────────────────────────────────
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.teacher).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vcfg = vae.config
    n_par = sum(p.numel() for p in vae.parameters())
    lat = (args.img_size // 2 ** (len(vcfg.block_out_channels) - 1))
    rate = vcfg.latent_channels * lat * lat
    print(f"[p0c] teacher={args.teacher} params={n_par/1e6:.1f}M "
          f"latent_ch={vcfg.latent_channels} down={2**(len(vcfg.block_out_channels)-1)} "
          f"latent={vcfg.latent_channels}x{lat}x{lat} => rate={rate} values/img", flush=True)

    # ── data ───────────────────────────────────────────────────────────────
    _, va = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    if args.num_images > 0:
        va = torch.utils.data.Subset(va, list(range(min(args.num_images, len(va)))))
    dl = torch.utils.data.DataLoader(va, batch_size=args.batch_size, shuffle=False,
                                     drop_last=False, num_workers=args.num_workers,
                                     pin_memory=False)
    print(f"[p0c] val images = {len(va)}", flush=True)

    inc = InceptionV3([3], resize_input=True, normalize_input=True).to(dev).eval()
    for p in inc.parameters():
        p.requires_grad_(False)

    KEYS = ["sdvae"] + (["identity"] if args.smoke else []) + \
           (["blur0.5", "blur1.0", "blur2.0"] if args.extra_refs else [])
    accs = {k: new_acc() for k in KEYS}
    gt_acc = new_acc()
    psnr_sum = {k: 0.0 for k in KEYS}
    l1_sum = {k: 0.0 for k in KEYS}
    seen = 0
    MEANd, STDd = MEAN.to(dev), STD.to(dev)

    with torch.no_grad():
        for x, _ in dl:
            x = x.to(dev, non_blocking=True)
            x01 = (x * STDd + MEANd).clamp(0.0, 1.0)          # 反归一化到 [0,1]
            B = x01.shape[0]

            outs = {}
            with torch.autocast("cuda", enabled=False):
                z = vae.encode(x01.float() * 2.0 - 1.0).latent_dist.mode()
                y = vae.decode(z).sample
            outs["sdvae"] = ((y.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            if args.smoke:
                outs["identity"] = x01.clone()
            if args.extra_refs:
                for s in (0.5, 1.0, 2.0):
                    outs[f"blur{s}"] = blur(x01, s)

            fg = inc(quant(x01))[0].flatten(1).float().cpu().numpy().astype(np.float64)
            add_feats(gt_acc, fg)
            for k, v in outs.items():
                fr = inc(quant(v))[0].flatten(1).float().cpu().numpy().astype(np.float64)
                add_feats(accs[k], fr)
                psnr_sum[k] += float(psnr(v, x01).sum())
                l1_sum[k] += float((v - x01).abs().mean(dim=(1, 2, 3)).sum())
            seen += B
            if seen % 640 == 0:
                print(f"[p0c] {seen}/{len(va)} …", flush=True)

    assert seen == len(va), (seen, len(va))
    n = seen
    mu_gt, sig_gt = stats(gt_acc)
    res = {
        "teacher": args.teacher, "n_images": n, "img_size": args.img_size,
        "teacher_params_M": round(n_par / 1e6, 1),
        "teacher_rate_values": int(rate),
        "rfid_protocol": "FID-InceptionV3 pool3 2048d, resize299, [0,1]->[-1,1], "
                         "uint8-quantized in-memory (pytorch-fid equivalent)",
        "results": {},
    }
    print(f"\n[p0c] n={n}  (FID 口径与 fast_rfid.py 完全一致)", flush=True)
    print(f"{'arm':<12} {'PSNR':>7} {'L1_255':>8} {'FID':>8}", flush=True)
    for k in KEYS:
        mu, sig = stats(accs[k])
        fid = float(calculate_frechet_distance(mu, sig, mu_gt, sig_gt, eps=1e-6))
        rec = {"psnr": psnr_sum[k] / n, "l1_255": l1_sum[k] / n * 255.0, "rfid": fid}
        res["results"][k] = rec
        print(f"{k:<12} {rec['psnr']:>7.3f} {rec['l1_255']:>8.3f} {fid:>8.3f}", flush=True)

    res["ref_ours_l1"] = {"psnr": 22.666, "rfid": 86.50,
                          "note": "k256x12_t8 L1 14ep, 同 5000 张评测（报告口径）"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"[p0c] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
