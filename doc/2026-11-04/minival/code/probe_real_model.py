#!/usr/bin/env python3
"""
Track 2 (lite): same question as Track 1, but on the REAL AssocVTC Phase-1 model
(model_v2_linattn.SRPhase1LinAttn, 62.85M params), shrunk config: 112x112, patch 16
(N=49 patches), K in {8,16,32}, batch 8, 300 steps/arm.

Arms (equal #steps per budget-loss = 300):
  single_K32 : one model trained only at K=32
  single_K8  : one model trained only at K=8
  joint      : ONE model (num_specials=32), each step encodes the image once, then
               decodes the 3 budget variants (K=8/16/32) as a padded memory batch with
               memory_key_padding_mask (per-variant isolation), loss = mean of 3 L1s.
               This is the minimal real-code implementation of the Lead's proposal:
               encoder once + decoder variant-batch + key-padding mask.

Also tests the enabler: the real decoder is key-length agnostic apart from the
one-line `z_s.shape[1]==num_specials` assert (patched via attribute swap here).
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_v2_linattn as M

IM, IS = M.IMAGENET_MEAN, M.IMAGENET_STD


def make_data(n, seed, S=112):
    rng = np.random.default_rng(seed)
    H = W = S
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    imgs = np.zeros((n, 3, H, W), np.float32)
    for i in range(n):
        bg = rng.uniform(0.0, 0.2, 3).astype(np.float32)
        img = np.broadcast_to(bg[:, None, None], (3, H, W)).copy()
        for _ in range(int(rng.integers(2, 5))):
            col = rng.uniform(0.35, 1.0, 3).astype(np.float32)
            cx, cy = rng.uniform(15, S - 15, 2)
            size = rng.uniform(12, 30)
            kind = int(rng.integers(0, 3))
            if kind == 0:
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= size * size
            elif kind == 1:
                m = (np.abs(xx - cx) <= size) & (np.abs(yy - cy) <= size)
            else:
                ang = rng.uniform(0, 2 * np.pi) + np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
                rad = size * 1.3
                vx, vy = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
                c = []
                for a in range(3):
                    b = (a + 1) % 3
                    c.append((vx[b] - vx[a]) * (yy - vy[a]) - (vy[b] - vy[a]) * (xx - vx[a]))
                c = np.stack(c)
                m = ((c > 0).all(0)) | ((c < 0).all(0))
            if m.sum() < 16:
                continue
            img[:, m] = col[:, None]
        img = img + rng.normal(0, 0.02, img.shape).astype(np.float32)
        imgs[i] = np.clip(img, 0, 1)
    return torch.from_numpy(imgs)


def normalize(x01):
    m = torch.tensor(IM, device=x01.device).view(1, 3, 1, 1)
    s = torch.tensor(IS, device=x01.device).view(1, 3, 1, 1)
    return (x01 - m) / s


def build_model(S, K, device):
    cfg = M.LinAttnConfig(img_size=S, patch=16, num_specials=K)
    return M.SRPhase1LinAttn(cfg).to(device)


def joint_loss(model, x, Ks):
    """encode once; decode all budget variants as a padded memory batch with per-variant mask."""
    dec = model.decoder
    z_cls, z_s = model.encode(x)                      # (B,1,D),(B,Kmax,D)
    mem = torch.cat([z_cls, z_s], dim=1)
    mem = dec.mem_norm(dec.mem_proj(mem))
    B, L, _ = mem.shape
    V = len(Ks)
    rows = mem.unsqueeze(1).expand(B, V, L, -1).reshape(B * V, L, -1)
    pad = torch.zeros(B * V, L, dtype=torch.bool, device=x.device)   # True = ignore
    for v, K in enumerate(Ks):
        pad[v::V, 1 + K:] = True
    tgt = (dec.query_base + dec.query_pos).unsqueeze(0).expand(B * V, -1, -1)
    y = dec.norm_out(dec.stack(tgt, rows, memory_key_padding_mask=pad))
    pix = dec.pixel_head(y).view(B, V, -1, dec.patch_px)
    target = M.image_to_patches(x, model.patch)
    return F.l1_loss(pix, target.unsqueeze(1).expand_as(pix))


def train(arm, steps, seed, x01, device, K=32, Ks=(8, 16, 32), bs=8, lr=2e-4,
          log_every=50):
    torch.manual_seed(seed)
    model = build_model(112, K, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    g = torch.Generator(device='cpu').manual_seed(seed + 7)
    N = x01.shape[0]
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    curve, t0 = [], time.time()
    for step in range(steps):
        idx = torch.randint(0, N, (bs,), device=device)
        xb = normalize(x01[idx])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            if arm == 'joint':
                loss = joint_loss(model, xb, Ks)
            else:
                out = model(xb)
                loss = out['loss']
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            curve.append((step, float(loss.item())))
    if device == 'cuda':
        torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1e6 if device == 'cuda' else 0.0
    return model, curve, time.time() - t0, peak


@torch.no_grad()
def evaluate(model, x01, device, budgets=(8, 16, 32), chunk=25):
    """Test L1 (normalized space, = training loss) and pixel L1 in [0,1] at each budget."""
    model.eval()
    old = model.decoder.num_specials
    res = {}
    S = model.img_size
    for K in budgets:
        if K > model.num_specials:          # cannot ask a model trained at K=8 for K=16
            res[K] = None
            continue
        model.decoder.num_specials = K
        l1n, l1p = [], []
        for i in range(0, x01.shape[0], chunk):
            xb01 = x01[i:i + chunk]
            xb = normalize(xb01)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                z_cls, z_s = model.encode(xb)
                out = model.decode(z_cls, z_s[:, :K], xb)
            l1n.append(float(out['recon']))
            img = M.patches_to_image(out['F_hat'].float(), S, S, model.patch, IM, IS,
                                     clamp_255=False).clamp(0, 1)
            l1p.append(float((img - xb01).abs().mean()))
        res[K] = {'l1_norm': float(np.mean(l1n)), 'l1_pixel01': float(np.mean(l1p))}
    model.decoder.num_specials = old
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results_track2.json')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.smoke:
        args.steps = 2
    dev = args.device
    torch.manual_seed(args.seed)
    x01 = make_data(800, 1234).to(dev)
    xt01 = make_data(100, 5678).to(dev)
    print(f'== device={dev} train={tuple(x01.shape)} test={tuple(xt01.shape)} steps={args.steps}',
          flush=True)
    out = {'meta': {'img_size': 112, 'patch': 16, 'num_patches': 49, 'batch': args.bs,
                    'steps': args.steps, 'seed': args.seed, 'budgets': [8, 16, 32],
                    'model': 'SRPhase1LinAttn (real AssocVTC model_v2_linattn), shrunk 112/p16',
                    'note': 'encoder once + decoder variant batch with memory_key_padding_mask'},
           'arms': {}}

    # 1) single-budget K=32 and K=8
    for name, K in [('single_K32', 32), ('single_K8', 8)]:
        model, curve, wall, peak = train('single', args.steps, args.seed, x01, dev, K=K)
        ev = evaluate(model, xt01, dev)
        out['arms'][name] = {'eval': ev, 'wall_s': wall, 'peak_mem_mb': peak,
                             'train_losses': curve, 'K_trained': K,
                             'params_M': sum(p.numel() for p in model.parameters()) / 1e6}
        print(f'[{name}] wall={wall:.1f}s peak={peak:.0f}MB eval={ev}', flush=True)

    # 2) joint
    model, curve, wall, peak = train('joint', args.steps, args.seed, x01, dev, K=32)
    ev = evaluate(model, xt01, dev)
    out['arms']['joint'] = {'eval': ev, 'wall_s': wall, 'peak_mem_mb': peak,
                            'train_losses': curve,
                            'params_M': sum(p.numel() for p in model.parameters()) / 1e6}
    print(f'[joint] wall={wall:.1f}s peak={peak:.0f}MB eval={ev}', flush=True)

    s32 = out['arms']['single_K32']['eval'][32]['l1_norm']
    j32 = out['arms']['joint']['eval'][32]['l1_norm']
    s8 = out['arms']['single_K8']['eval'][8]['l1_norm']
    j8 = out['arms']['joint']['eval'][8]['l1_norm']
    out['summary'] = {
        'alignment_tax_joint_vs_single_at_K32_pct': (j32 - s32) / s32 * 100,
        'joint_vs_single_at_K8_pct': (j8 - s8) / s8 * 100,
        'single_K32_truncation_ratio_K8': out['arms']['single_K32']['eval'][8]['l1_pixel01']
                                          / out['arms']['single_K32']['eval'][32]['l1_pixel01'],
        'single_K32_truncation_ratio_K16': out['arms']['single_K32']['eval'][16]['l1_pixel01']
                                           / out['arms']['single_K32']['eval'][32]['l1_pixel01'],
    }
    json.dump(out, open(args.out, 'w'), indent=1)
    print('\n== summary ==', flush=True)
    print(json.dumps(out['summary'], indent=1), flush=True)
    for a in ['single_K32', 'single_K8', 'joint']:
        print(a, {K: ({k: round(v, 4) for k, v in d.items()} if d else None)
                  for K, d in out['arms'][a]['eval'].items()}, flush=True)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
