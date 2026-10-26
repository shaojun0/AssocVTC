"""严格同值数对齐的 per-patch PCA 曲线（respond 用；不改仓库）"""
import os, sys, json, time
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

sys.path.insert(0, '/root/AssocVTC')
from data_linattn import build_datasets
from model_v2_linattn import (image_to_patches, patches_to_image,
                              IMAGENET_MEAN, IMAGENET_STD)

MEAN_255 = tuple(v * 255.0 for v in IMAGENET_MEAN)
STD_255 = tuple(v * 255.0 for v in IMAGENET_STD)
OUT = '/root/autodl-tmp/rfid_pca_strict'
GT_STATS = '/root/autodl-tmp/rfid_k64/gt_stats.npz'

# strict: 每图 2496*t 个值（2 token/步 × 1248）→ r = round(2496t/196) = round(12.7347t)
STRICT = [round(2496 * t / 196) for t in range(1, 17)]
# 旧图口径: r = round(896t/196) ≈ 4.57t
OLD = [round(896 * t / 196) for t in range(1, 17)]
RANKS = sorted(set(STRICT + OLD))
FID_RANKS = [5, 13, 38]

def psnr01(p, t):
    mse = ((p - t) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)

def to_patches255(x, img=224, patch=16):
    m = torch.tensor(MEAN_255).view(1, 3, 1, 1); s = torch.tensor(STD_255).view(1, 3, 1, 1)
    return image_to_patches(x * s + m, patch)

def save_pngs(t01, out_dir, off):
    os.makedirs(out_dir, exist_ok=True)
    a = t01.detach().float().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    a = np.round(a * 255.0).astype(np.uint8)
    for j, im in enumerate(a):
        Image.fromarray(im).save(os.path.join(out_dir, f"{off + j:06d}.png"))

def fid(recon_dir):
    import scipy.linalg as sl
    if not getattr(sl.sqrtm, '_fid_compat', False):
        o = sl.sqrtm
        def _sq(A, disp=True, **kw):
            r = o(A, **kw); return (r, 0.0) if disp is False else r
        _sq._fid_compat = True; sl.sqrtm = _sq
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return float(calculate_fid_given_paths([recon_dir, GT_STATS], batch_size=50,
                                           device='cuda', dims=2048, num_workers=8))

def main():
    torch.manual_seed(0); np.random.seed(0)
    tr, va = build_datasets('/root/autodl-tmp/datasets/imagenet100', 224, 50)
    idx = list(range(0, len(tr), max(1, len(tr) // 400)))[:400]
    sub = torch.utils.data.Subset(tr, idx)
    dl = DataLoader(sub, batch_size=32, shuffle=False, num_workers=16)
    dim = 768; n = 0
    s1 = torch.zeros(dim, dtype=torch.float64); s2 = torch.zeros(dim, dim, dtype=torch.float64)
    t0 = time.time()
    for x, _ in dl:
        P = to_patches255(x).reshape(-1, dim).double()
        s1 += P.sum(0); s2 += P.t() @ P; n += P.shape[0]
    mean = (s1 / n).float()
    cov = (s2 / n - (s1 / n).unsqueeze(1) * (s1 / n).unsqueeze(0)).float()
    ev, evec = torch.linalg.eigh(cov)
    order = torch.argsort(ev, descending=True)
    basis = evec[:, order][:, :256].contiguous()
    print(f"[fit] {len(idx)} imgs, {n} patches, basis {tuple(basis.shape)}, "
          f"{time.time() - t0:.1f}s", flush=True)

    dev = 'cuda'
    mean = mean.to(dev); basis = basis.to(dev)
    dlv = DataLoader(va, batch_size=32, shuffle=False, num_workers=16)
    acc = {r: {'l1': 0.0, 'psnr': 0.0} for r in RANKS}
    seen = 0; t0 = time.time()
    for bi, (x, _) in enumerate(dlv):
        P = to_patches255(x).to(dev)                    # (B,N,768) 0-255
        B = P.shape[0]
        Pc = P.reshape(B, -1, dim) - mean
        C = Pc @ basis
        t255 = patches_to_image(P, 224, 224, 16, clamp_255=False)
        t01 = (t255 / 255.0).clamp(0, 1)
        for r in RANKS:
            P_rec = (C[:, :, :r] @ basis[:, :r].t()).reshape(B, -1, dim) + mean
            r255 = patches_to_image(P_rec, 224, 224, 16, clamp_255=False)
            p01 = (r255 / 255.0).clamp(0, 1)
            acc[r]['l1'] += float((r255 - t255).abs().mean()) * B
            acc[r]['psnr'] += float(psnr01(p01, t01).sum())
            if r in FID_RANKS:
                save_pngs(p01.view(B, 3, 224, 224), f"{OUT}/rank{r}/recon", seen)
        seen += B
        if (bi + 1) % 20 == 0:
            print(f"  {seen} imgs {time.time() - t0:.1f}s", flush=True)
    res = {int(r): {'rank': int(r), 'values': 196 * r,
                    'l1_255': acc[r]['l1'] / seen, 'psnr': acc[r]['psnr'] / seen}
           for r in RANKS}
    for r in FID_RANKS:
        d = f"{OUT}/rank{r}/recon"
        print(f"[fid] rank{r} ...", flush=True)
        res[r]['rfid'] = fid(d)
    json.dump({'strict_alignment': {int(t): int(r) for t, r in zip(range(1, 17), STRICT)},
               'old_alignment': {int(t): int(r) for t, r in zip(range(1, 17), OLD)},
               'ranks': res}, open(f"{OUT}/strict.json", 'w'), indent=1)
    print(json.dumps({'strict': {int(t): int(r) for t, r in zip(range(1, 17), STRICT)},
                      'old': {int(t): int(r) for t, r in zip(range(1, 17), OLD)},
                      'ranks': res}, indent=1))

main()
