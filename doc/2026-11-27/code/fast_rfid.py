"""Fast per-step rFID (in-memory, no PNG dump) — ALIT/FID protocol.

Same feature extractor as tools/eval_rfid.py: official FID InceptionV3 pool3
(2048-d, resize 299, [0,1]→[-1,1]).  Instead of dumping PNG dirs and calling
`calculate_fid_given_paths`, we accumulate the Inception feature mean/covariance
in memory for each decoding step (and for GT) and call pytorch-fid's exact
`calculate_frechet_distance`.  Reconstructions are quantized to the uint8 grid
(round(x*255)/255) to match the PNG path; sole deviation is the 448→299 resize
(F.interpolate vs PIL), which is sub-0.1 FID.

2026-11-23 I2: `--render sdvae` 把每步的图换成**冻结 teacher decoder 渲染**的
`decode(z_hat[:,t])`（需要 ckpt 里 `cfg.z_head=1`）; 其余口径
（uint8 量化 / Inception 特征 / per-step 统计 / JSON 字段）不变。默认 `pixel` ⇒
与提交前逐位一致（含 `--vae_roundtrip` 那一臂）。

2026-11-24 修: `z_hat` 是**未缩放** latent（`MSE(z_hat·s, z_t·s)` 的最优解是
`z_hat = z_t`，`s` 只改损失尺度）⇒ 渲染**不能**再 `/ kd_latent_scale`。
`--kd_latent_scale` 默认改为 1.0（旧 ckpt 仍需显式传 0.18215，但注意那只影响
"训练时 λ 被暗乘 s²"这一历史事实，不影响本脚本渲染）。
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
    # P0-d 决策门（2026-11-23）：把重建图塞回 SD-VAE 流形 D(E(recon)) 再算一遍
    # ⇒ 量"latent 路线"的天花板代理（见 doc/2026-11-23/NOTES_kd_vs_alit_code_20261123.md §3-P0-d）。
    # 默认 0 ⇒ 本脚本行为与提交前逐位相同。
    ap.add_argument("--vae_roundtrip", type=int, default=0,
                    help="1=额外评测 D(E(recon)) 这一臂（慢 ~1.6×，建议配 --num_images 2000）")
    ap.add_argument("--teacher_vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--teacher_local_only", type=int, default=1)
    # I2 投放路径（2026-11-23）: 出图 = 冻结 teacher decoder 渲染 z_hat。
    # 默认 pixel ⇒ 本脚本与提交前逐位一致（含 --vae_roundtrip 那一臂）。
    ap.add_argument("--render", default="pixel",
                    choices=["pixel", "sdvae", "vq"],
                    help="pixel=自身像素头（默认, 逐位不变）| sdvae=z_hat→teacher.decode(z_hat), "
                         "需 ckpt 里 cfg.z_head=1 | vq=阶段3-CE: 每步 argmax 离散码 → "
                         "图像 tokenizer decoder 解码, 需 ckpt 里 cfg.token_ce=1")
    ap.add_argument("--vq_ckpt", default="/root/autodl-tmp/vq/llamagen_vq16_c2i.pt",
                    help="--render vq 用的图像 tokenizer 权重（LlamaGen VQ-16）")
    ap.add_argument("--vq_codebook_size", type=int, default=16384)
    ap.add_argument("--vq_embed_dim", type=int, default=8)
    ap.add_argument("--kd_latent_scale", type=float, default=1.0,
                    help="**已不再用于渲染**（z_hat 就是未缩放 latent）。默认 1.0; 仅回填进 "
                         "JSON 以便与训练 args 对齐（0.18215 = kd2-#1 的历史值）")
    args = ap.parse_args()
    assert args.render in ("pixel", "sdvae", "vq")
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    model = SRPhase1LinAttn(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    if args.render == "sdvae":       # 早失败: 没有潜变量头就没有 z_hat 可渲染
        assert int(getattr(cfg, "z_head", 0)) == 1, \
            "--render sdvae 需要 cfg.z_head=1 的 ckpt（当前 cfg.z_head=" \
            f"{getattr(cfg, 'z_head', 0)}）"
    vq = None
    if args.render == "vq":          # 阶段3-CE: 离散码 → tokenizer decoder 解码
        assert int(getattr(cfg, "token_ce", 0)) == 1, \
            "--render vq 需要 cfg.token_ce=1 的 ckpt（当前 cfg.token_ce=" \
            f"{getattr(cfg, 'token_ce', 0)}）"
        from vq_tokenizer import LlamaGenVQ16
        vq = LlamaGenVQ16(ckpt=str(args.vq_ckpt), device=dev)
        assert int(vq.n_embed) == int(getattr(cfg, "token_vocab",
                                               args.vq_codebook_size)), \
            f"tokenizer 码本 {vq.n_embed} ≠ ckpt cfg.token_vocab " \
            f"{getattr(cfg, 'token_vocab', None)}"
    _mean01 = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(dev)
    _std01 = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(dev)
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
    vae = None
    if args.vae_roundtrip or args.render == "sdvae":
        from diffusers import AutoencoderKL     # 惰性 import: pixel 默认路径不碰 diffusers
        vae = AutoencoderKL.from_pretrained(
            args.teacher_vae,
            local_files_only=bool(args.teacher_local_only)).to(dev).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
    if args.vae_roundtrip:
        rt_acc = [{"n": 0, "s": np.zeros(D, np.float64),
                   "ss": np.zeros((D, D), np.float64)} for _ in range(T)]
        rt_psnr, rt_l1 = [0.0] * T, [0.0] * T
        print(f"[fast] P0-d vae_roundtrip=on teacher={args.teacher_vae}", flush=True)
    if args.render == "sdvae":
        print(f"[fast] render=sdvae: 每步图 = teacher.decode(z_hat[:,t])（z_hat 未缩放, "
              f"不除 scale）teacher={args.teacher_vae}", flush=True)
    seen = 0

    with torch.no_grad():
        for x, _ in dl:
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                out = model(x)
            B = x.shape[0]
            if args.render == "vq":
                # ── 阶段3-CE 投放路径: 每步 argmax 出离散码 → tokenizer 解码 ──
                ids = out["tok_ids"]                     # (B,T,N) long
                tgt01 = (x.float() * _std01 + _mean01).clamp(0.0, 1.0)
                rec01 = None
            else:
                pix = out["Y_pix"].float()
                tgt = out["target_pix"].float()
                if args.render == "sdvae":
                    # ── I2 投放路径: z_hat → 冻结 teacher decoder（像素头不参与）──
                    # ⚠️ 2026-11-24 修: z_hat 是**未缩放** latent（MSE(z_hat·s, z_t·s) 的
                    #    最优解是 z_hat=z_t, s 只改损失尺度）⇒ 直接 decode, 不除 scale
                    #    （除 scale 会把 latent 放大 1/s≈5.5 倍 ⇒ 出图趋近均值图）。
                    zs = out["z_hat"].float()
                    rec01 = torch.stack([
                        ((vae.decode(zs[:, t]).sample.float() + 1.0) * 0.5
                         ).clamp(0.0, 1.0) for t in range(T)], dim=1)
                else:
                    rec01 = patches_to_image(
                        pix.reshape(B * T, cfg.num_patches, -1), cfg.img_size, cfg.img_size,
                        cfg.patch, IMAGENET_MEAN, IMAGENET_STD, clamp_255=False
                    ).reshape(B, T, 3, cfg.img_size, cfg.img_size).clamp(0.0, 1.0)
                tgt01 = patches_to_image(
                    tgt, cfg.img_size, cfg.img_size, cfg.patch,
                    IMAGENET_MEAN, IMAGENET_STD, clamp_255=False
                ).reshape(B, 3, cfg.img_size, cfg.img_size).clamp(0.0, 1.0)
            tgtq = (tgt01 * 255.0).round() / 255.0
            # GT features (once)
            fg = inc(tgtq)[0].flatten(1).float().cpu().numpy().astype(np.float64)
            add_feats(gt_acc, fg)
            for t in range(T):
                if args.render == "vq":
                    rec_t = vq.decode_ids(ids[:, t]).to(dev).float()  # (B,3,H,W) ∈[0,1]
                else:
                    rec_t = rec01[:, t]
                rq_t = (rec_t * 255.0).round() / 255.0
                fr = inc(rq_t)[0].flatten(1).float().cpu().numpy().astype(np.float64)
                add_feats(rec_acc[t], fr)
                ps = psnr(rec_t, tgt01)
                psnr_sum[t] += float(ps.sum())
                l1_sum[t] += float((rec_t - tgt01).abs().mean(dim=(1, 2, 3)).sum())
                if args.vae_roundtrip:
                    with torch.autocast("cuda", enabled=False):
                        z = vae.encode(rec_t.float() * 2.0 - 1.0).latent_dist.mode()
                        y = vae.decode(z).sample
                    r01 = ((y.float() + 1.0) * 0.5).clamp(0.0, 1.0)
                    rq = (r01 * 255.0).round() / 255.0
                    frt = inc(rq)[0].flatten(1).float().cpu().numpy().astype(np.float64)
                    add_feats(rt_acc[t], frt)
                    rt_psnr[t] += float(psnr(r01, tgt01).sum())
                    rt_l1[t] += float((r01 - tgt01).abs().mean(dim=(1, 2, 3)).sum())
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
        line = (f"[fast] step{t+1} cum_z={cum_z[t]:3d} "
                f"PSNR={psnr_sum[t]/n:.3f} L1_255={l1_sum[t]/n*255:.3f} rFID={fid:.3f}")
        if args.vae_roundtrip:
            mu_t, sig_t = stats(rt_acc[t])
            fidt = float(calculate_frechet_distance(mu_t, sig_t, mu_gt, sig_gt, eps=1e-6))
            per_step[t + 1].update({
                "psnr_vae_rt": rt_psnr[t] / n,
                "l1_255_vae_rt": rt_l1[t] / n * 255.0,
                "rfid_vae_rt": fidt,
            })
            line += (f" | D(E(·)): PSNR={rt_psnr[t]/n:.3f} "
                     f"L1_255={rt_l1[t]/n*255:.3f} rFID={fidt:.3f}")
        print(line, flush=True)

    res = {
        "ckpt": args.ckpt, "tag": args.tag, "n_images": n,
        "K": cfg.num_specials, "zs_dim": int(getattr(cfg, "zs_dim", 0) or cfg.enc_dim),
        "recurrent": bool(recurrent), "T": T, "cum_z_s": cum_z,
        "img_size": int(cfg.img_size), "patch": int(cfg.patch),
        "rfid_protocol": "FID-InceptionV3 pool3 2048d, resize299, [0,1]->[-1,1], "
                         "uint8-quantized in-memory (pytorch-fid equivalent)",
        "vae_roundtrip": bool(args.vae_roundtrip),
        "teacher_vae": (str(args.teacher_vae) if args.vae_roundtrip else None),
        "per_step": {str(k): v for k, v in per_step.items()},
    }
    if args.render == "sdvae":
        # 只在 sdvae 臂加这两个键 ⇒ 默认 pixel 臂的 JSON 结构与提交前**逐 key 相同**
        res["render"] = "sdvae"
        res["kd_latent_scale"] = float(args.kd_latent_scale)
        res["teacher_vae"] = str(args.teacher_vae)
    if args.render == "vq":
        # 阶段3-CE 臂: 出图 = tokenizer.decode(argmax(logits)) ⇒ 额外记录 tokenizer 信息
        res["render"] = "vq"
        res["vq_ckpt"] = str(args.vq_ckpt)
        res["vq_vocab"] = int(vq.n_embed)
        res["vq_grid"] = [int(v) for v in vq.grid(int(cfg.img_size))]
        res["token_ce"] = True
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"[fast] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
