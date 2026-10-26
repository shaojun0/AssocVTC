"""补充 SSIM pass（独立于 canonical eval_stepwise_linattn.py, 不改原脚本）。
复用 eval_stepwise_linattn.load_model 与 data_linattn.build_ood_dataset,
逐图逐步算 SSIM（metrics_linattn.ssim, [0,1] clamp 口径, 与主报告 eval_linattn 同口径）。
用法: python ssim_pass.py --ckpt ... --data_dir_ood ... --batch_size 32 --out per_image_ssim_<tag>.json --out_summary ssim_<tag>.json
"""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_linattn import build_ood_dataset
from metrics_linattn import ssim
from model_v2_linattn import IMAGENET_MEAN, IMAGENET_STD, patches_to_image
from eval_stepwise_linattn import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir_ood", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_images", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_summary", default="")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ck = load_model(a.ckpt, device)
    img, patch = cfg.img_size, cfg.patch
    va_ds = build_ood_dataset(a.data_dir_ood, img)
    rec_paths = list(va_ds.paths)
    n = min(len(va_ds), a.num_images) if a.num_images > 0 else len(va_ds)
    stride = max(1, len(va_ds) // max(1, n))
    idx = list(range(0, len(va_ds), stride))[:n]
    va = DataLoader(Subset(va_ds, idx), batch_size=a.batch_size, shuffle=False,
                    num_workers=a.num_workers, pin_memory=True)
    T = len(model.decoder.steps)
    sum_ssim = np.zeros(T)
    records = []
    n_img = 0
    for bi, (x, _) in enumerate(va):
        x = x.to(device, non_blocking=True)
        # 前向在 bf16 autocast 下（与 canonical eval 同口径）...
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=(device == "cuda")):
            out = model(x)
        B = x.shape[0]
        pix = out["Y_pix"].float()
        tgt = out["target_pix"].float()
        # ...但 SSIM 在 autocast **之外**用 fp32 算: eval_linattn.py（ID 口径）就是
        # 在 autocast 块外调 ssim(), 这样 OOD 与 ID 的 SSIM 口径一致。
        with torch.no_grad():
            p01 = patches_to_image(pix.reshape(B * T, cfg.num_patches, -1), img, img,
                                   patch, IMAGENET_MEAN, IMAGENET_STD,
                                   clamp_255=False).reshape(B, T, 3, img, img).clamp(0, 1)
            t01 = patches_to_image(tgt, img, img, patch, IMAGENET_MEAN, IMAGENET_STD,
                                   clamp_255=False).reshape(B, 3, img, img).clamp(0, 1)
            # 逐 step 算 SSIM（SSIM 的 conv 中间张量比 PSNR 大, 整批 B*T 展平会 OOM）
            s = torch.empty(B, T, device=p01.device)
            for t in range(T):
                s[:, t] = ssim(p01[:, t], t01)
        sum_ssim += s.sum(0).cpu().numpy()
        sn = s.cpu().numpy()
        off = bi * a.batch_size
        for j in range(B):
            gi = idx[off + j]
            records.append({
                "index": int(gi), "path": rec_paths[gi],
                "file_name": os.path.basename(rec_paths[gi]),
                "category": os.path.basename(os.path.dirname(rec_paths[gi])),
                "SSIM": [float(v) for v in sn[j]],
            })
        n_img += B
        if (bi + 1) % 10 == 0:
            print(f"    {n_img} 张…", flush=True)
    SSIM = sum_ssim / n_img
    cats = {}
    for r in records:
        cats.setdefault(r["category"], []).append(r["SSIM"][-1])
    summary = {"ckpt": a.ckpt, "T": T, "n_images": n_img,
               "SSIM": SSIM.tolist(), "SSIM_last": float(SSIM[-1]),
               "subset_n": {c: len(v) for c, v in cats.items()},
               "subset_SSIM_last": {c: float(np.mean(v)) for c, v in cats.items()}}
    with open(a.out, "w") as f:
        json.dump({"ckpt": a.ckpt, "T": T, "n_images": len(records),
                   "aggregate_SSIM": SSIM.tolist(), "images": records}, f)
    if a.out_summary:
        with open(a.out_summary, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:1500])
    print("[out]", a.out)


if __name__ == "__main__":
    main()
