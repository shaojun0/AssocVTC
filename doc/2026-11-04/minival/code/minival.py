#!/usr/bin/env python3
"""
Track 1: AssocVTC "train all token budgets simultaneously" minimal mechanism validation.

Self-contained toy: synthetic 32x32x3 images (2-4 colored shapes) -> encoder compressed
to N=32 tokens -> budget = first K tokens (K in {8,16,32}) -> decoder (64 pixel queries
cross-attend the K tokens + conv upsampling) -> reconstruct 32x32x3, L1 loss.

Arms (all share the same architecture / same data / same total step budget):
  A   : 3 independent models, each trained at a single fixed K  (reference)
  B   : ONE model, each step packs the 3 budget variants of the same image into one
        varlen sequence (keys = concat[z[:8], z[:16], z[:32]], 56 keys) with a
        block-diagonal additive mask, one forward/backward, mean of 3 L1 losses
  C   : same as B but NO mask (variants concatenated, full cross-variant attention)
  D1x : ONE model, each step samples a single random m in {8,16,32} (MQT style), steps
  D3x : same as D1x but 3x steps (compute-matched to A/B/C: same #variant-passes)

Metrics: test L1 at K=8/16/32, alignment tax of B vs A at K=32, C vs B, D variance,
peak VRAM, wall clock, packing equivalence check (masked-concat == separate forwards),
attention budget share under no-mask (link to 2026-09-22 analysis), serial-escalation
simulation using the B model.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

D_MODEL, N_TOK, NHEAD, GRID = 128, 32, 4, 8
PIXQ = GRID * GRID            # 64 decoder pixel queries
BATCH, LR, WD, WARM = 64, 1e-3, 1e-4, 50
BUDGETS = [8, 16, 32]


# ---------------------------------------------------------------- data
def make_data(n, seed, H=32, W=32):
    """Synthetic images: random dark background + 2-4 colored shapes (circle/square/triangle)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    imgs = np.zeros((n, 3, H, W), np.float32)
    for i in range(n):
        bg = rng.uniform(0.0, 0.2, 3).astype(np.float32)
        img = np.broadcast_to(bg[:, None, None], (3, H, W)).copy()
        for _ in range(int(rng.integers(2, 5))):
            col = rng.uniform(0.35, 1.0, 3).astype(np.float32)
            cx, cy = rng.uniform(5, 27, 2)
            size = rng.uniform(3.5, 8.0)
            kind = int(rng.integers(0, 3))
            if kind == 0:      # circle
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= size * size
            elif kind == 1:    # axis-aligned square
                m = (np.abs(xx - cx) <= size) & (np.abs(yy - cy) <= size)
            else:              # triangle
                ang = rng.uniform(0, 2 * np.pi) + np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
                rad = size * 1.5
                vx, vy = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
                c = []
                for a in range(3):
                    b = (a + 1) % 3
                    c.append((vx[b] - vx[a]) * (yy - vy[a]) - (vy[b] - vy[a]) * (xx - vx[a]))
                c = np.stack(c)
                m = ((c > 0).all(0)) | ((c < 0).all(0))
            if m.sum() < 4:
                continue
            img[:, m] = col[:, None]
        img = img + rng.normal(0, 0.02, img.shape).astype(np.float32)
        imgs[i] = np.clip(img, 0, 1)
    return torch.from_numpy(imgs)


# ---------------------------------------------------------------- model
class MHCA(nn.Module):
    """Multi-head cross-attention; mask is (Q,K) bool broadcast over batch (True = keep)."""
    def __init__(self, d=D_MODEL, h=NHEAD):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln_q, self.ln_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))
        self.store, self.attn = False, None

    def forward(self, x, kv, mask=None):
        B, Q, _ = x.shape
        K = kv.shape[1]
        q = self.wq(self.ln_q(x)).view(B, Q, self.h, self.dh)
        kvn = self.ln_kv(kv)
        k = self.wk(kvn).view(B, K, self.h, self.dh)
        v = self.wv(kvn).view(B, K, self.h, self.dh)
        s = torch.einsum('bqhd,bkhd->bhqk', q, k) * (self.dh ** -0.5)
        if mask is not None:
            s = s.masked_fill(~mask, float('-inf'))
        a = s.softmax(dim=-1)
        if self.store:
            self.attn = a.detach()
        o = torch.einsum('bhqk,bkhd->bqhd', a, v).reshape(B, Q, -1)
        return self.wo(o)


class MLP(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return self.f(x)


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.ln, self.mlp = MHCA(), nn.LayerNorm(D_MODEL), MLP()

    def forward(self, x, kv, mask=None):
        x = x + self.attn(x, kv, mask)
        return x + self.mlp(self.ln(x))


class Encoder(nn.Module):
    """32x32x3 -> conv (8x8x128=64 features) -> 32 learned-query tokens (MQT-style prefix bank)."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.c2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.c3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.q = nn.Parameter(torch.randn(N_TOK, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock() for _ in range(2)])
        self.lnf = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        h = F.gelu(self.c1(x))
        h = F.gelu(self.c2(h))
        h = F.gelu(self.c3(h))
        B = h.shape[0]
        f = h.flatten(2).transpose(1, 2)                      # B, 64, 128
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        for b in self.blocks:
            q = b(q, f)
        return self.lnf(q)


class Decoder(nn.Module):
    """64 pixel queries (8x8 grid) cross-attend the selected K tokens -> conv upsample -> 32x32x3."""
    def __init__(self):
        super().__init__()
        self.pix = nn.Parameter(torch.randn(PIXQ, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock() for _ in range(2)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.up1 = nn.ConvTranspose2d(D_MODEL, 64, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x, z, mask=None, V=1):
        for b in self.blocks:                                  # x: B, V*PIXQ, D
            x = b(x, z, mask)
        x = self.lnf(x)
        B = x.shape[0]
        h = x.reshape(B * V, GRID, GRID, D_MODEL).permute(0, 3, 1, 2)
        h = F.gelu(self.up1(h))
        h = F.gelu(self.up2(h))
        return torch.sigmoid(self.out(h)).reshape(B, V, 3, 32, 32)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = Encoder(), Decoder()

    def encode(self, x):
        return self.enc(x)

    def decode(self, z, Ks, use_mask, capture=False):
        """Ks = list of budgets. V=1 for single-variant inference, V=len(Ks) for packing."""
        B, V = z.shape[0], len(Ks)
        keys = torch.cat([z[:, :k] for k in Ks], dim=1)        # B, sum(Ks), D
        q = self.dec.pix.unsqueeze(0).expand(B, -1, -1).repeat(1, V, 1)
        mask = None
        if use_mask and V > 1:
            mask = torch.zeros(V * PIXQ, keys.shape[1], dtype=torch.bool, device=z.device)
            ok = oq = 0
            for k in Ks:
                mask[oq:oq + PIXQ, ok:ok + k] = True
                oq += PIXQ
                ok += k
        self.dec.blocks[0].attn.store = capture
        rec = self.dec(q, keys, mask, V)
        self.dec.blocks[0].attn.store = False
        return rec

    def attn_first_block(self):
        return self.dec.blocks[0].attn.attn


# ---------------------------------------------------------------- training
def lr_lam(step, steps):
    if step < WARM:
        return (step + 1) / WARM
    p = (step - WARM) / max(1, steps - WARM)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))


def train_single_fixed(K, steps, seed, train_imgs, device, log_every=50):
    """Mode A: one independent model trained at a single fixed budget K."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lam(s, steps))
    g = torch.Generator(device=device).manual_seed(seed + 9999)
    N = train_imgs.shape[0]
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    curve, t0 = [], time.time()
    for step in range(steps):
        idx = torch.randint(0, N, (BATCH,), device=device, generator=g)
        imgs = train_imgs[idx]
        z = model.encode(imgs)
        rec = model.decode(z, [K], False)
        loss = (rec - imgs.unsqueeze(1)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            curve.append((step, float(loss.item())))
    torch.cuda.synchronize(device)
    return model, curve, time.time() - t0, torch.cuda.max_memory_allocated(device) / 1e6


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, test_imgs, budgets=BUDGETS, bs=100):
    model.eval()
    mean, per = {}, {}
    for K in budgets:
        errs = []
        for i in range(0, test_imgs.shape[0], bs):
            imgs = test_imgs[i:i + bs]
            z = model.encode(imgs)
            rec = model.decode(z, [K], False).squeeze(1)
            errs.append((rec - imgs).abs().mean(dim=(1, 2, 3)))
        e = torch.cat(errs)
        mean[K] = float(e.mean())
        per[K] = [float(x) for x in e.cpu()]
    model.train()
    return mean, per


def packing_equivalence(seed=0):
    """Masked packed decode vs 3 separate single-budget decodes (the claim of the 09-22 note)."""
    dev = 'cuda'
    torch.manual_seed(seed)
    m = Model().to(dev).eval()
    imgs = make_data(8, 4242).to(dev)
    with torch.no_grad():
        z = m.encode(imgs)
        rec_pack = m.decode(z, BUDGETS, use_mask=True)
        rec_sep = torch.cat([m.decode(z, [k], False) for k in BUDGETS], dim=1)
        rec_nomask = m.decode(z, BUDGETS, use_mask=False)
        m.decode(z, BUDGETS, use_mask=False, capture=True)
        a = m.attn_first_block()                             # B,H,V*PIXQ,Ktot
    d_pack = (rec_pack - rec_sep).abs().max().item()
    d_nomask = (rec_nomask - rec_sep).abs().max().item()
    # attention mass on own variant's key block, for the no-mask case
    Ktot, share = sum(BUDGETS), {}
    ok = oq = 0
    for K in BUDGETS:
        rows = a[:, :, oq:oq + PIXQ, :]                       # B,H,PIXQ,Ktot
        own = rows[:, :, :, ok:ok + K].sum(-1).mean().item()
        share[K] = own
        oq += PIXQ
        ok += K
    return {'max_abs_diff_packed_masked_vs_separate': d_pack,
            'max_abs_diff_packed_nomask_vs_separate': d_nomask,
            'nomask_attn_mass_on_own_variant_block': share}


# ---------------------------------------------------------------- escalation
def escalation_analysis(per_sample, budgets=BUDGETS):
    """Oracle serial-escalation: run K=8, escalate on true error. Sweep escalation rate."""
    e = {k: np.asarray(per_sample[k]) for k in budgets}
    table = []
    for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        tau1 = np.quantile(e[8], 1 - f) if f < 1 else -1.0
        use16 = e[8] > tau1
        g = f * 0.5
        tau2 = np.quantile(e[16], 1 - g) if g > 0 else -1.0
        use32 = use16 & (e[16] > tau2)
        err = np.where(use32, e[32], np.where(use16, e[16], e[8]))
        budget = 8 + 8 * use16 + 16 * use32
        hard = e[32] > np.quantile(e[32], 0.75)               # samples that genuinely need more budget
        table.append({
            'escalate_frac': f,
            'mean_l1': float(err.mean()),
            'mean_budget': float(budget.mean()),
            'coverage_of_hard_quartile': float(use16[hard].mean()) if hard.any() else 0.0,
            'precision_benefited': float((e[16][use16] < e[8][use16]).mean()) if use16.any() else 0.0,
        })
    base = {'K8': float(e[8].mean()), 'K16': float(e[16].mean()), 'K32': float(e[32].mean())}
    # cheapest avg budget whose error is within 5% of always-K=32
    ok = [r for r in table if r['mean_l1'] <= 1.05 * base['K32']]
    within5 = min(ok, key=lambda r: r['mean_budget']) if ok else None
    return {'base': base, 'table': table, 'min_budget_within_5pct_of_K32': within5}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--out', default='results.json')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.seeds = 40, '0'
    dev = 'cuda'
    seeds = [int(s) for s in args.seeds.split(',')]
    print(f'== device={torch.cuda.get_device_name(0)} steps={args.steps} seeds={seeds}', flush=True)
    train_imgs = make_data(2000, 1234).to(dev)
    test_imgs = make_data(200, 5678).to(dev)
    print(f'data ready train={tuple(train_imgs.shape)} test={tuple(test_imgs.shape)}', flush=True)

    results = {'meta': {
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'steps_per_model': args.steps, 'batch': BATCH, 'lr': LR, 'wd': WD,
        'budgets': BUDGETS, 'seeds': seeds, 'd_model': D_MODEL, 'n_tokens': N_TOK,
        'arms': {'A': 'single fixed budget x3 models', 'B': 'packed varlen + block-diagonal mask',
                 'C': 'packed varlen, NO mask', 'D1x': 'random single m, steps',
                 'D3x': 'random single m, 3x steps (compute matched)'},
    }, 'arms': {}}

    print('== packing equivalence check', flush=True)
    results['packing_equivalence'] = packing_equivalence()
    print(json.dumps(results['packing_equivalence'], indent=1), flush=True)

    for seed in seeds:
        tag = f'seed{seed}'
        # ---- A: three independent single-budget models
        A = {}
        for K in BUDGETS:
            model, curve, wall, peak = train_single_fixed(K, args.steps, seed, train_imgs, dev)
            mean, per = evaluate(model, test_imgs)
            A[f'K{K}'] = {'eval': mean, 'per_sample': per, 'train_losses': curve,
                          'wall_s': wall, 'peak_mem_mb': peak,
                          'params': sum(p.numel() for p in model.parameters())}
            print(f'[A {tag} K={K}] test={ {k: round(v, 5) for k, v in mean.items()} } '
                  f'wall={wall:.1f}s peak={peak:.0f}MB', flush=True)
        results['arms'][f'A_{tag}'] = A

        # ---- B / C / D1x / D3x: one model each
        for arm, mode, nsteps in [('B', 'packed_mask', args.steps),
                                  ('C', 'packed_nomask', args.steps),
                                  ('D1x', 'random_single', args.steps),
                                  ('D3x', 'random_single', args.steps * 3)]:
            model = Model().to(dev)
            if mode == 'random_single':
                # train_model() handles the random-budget policy
                model, curve, wall, peak = train_random(nsteps, seed, train_imgs, dev)
            else:
                model, curve, wall, peak = train_packed(mode, nsteps, seed, train_imgs, dev)
            mean, per = evaluate(model, test_imgs)
            results['arms'][f'{arm}_{tag}'] = {
                'eval': mean, 'per_sample': per, 'train_losses': curve,
                'wall_s': wall, 'peak_mem_mb': peak, 'steps': nsteps,
                'params': sum(p.numel() for p in model.parameters())}
            print(f'[{arm} {tag} {nsteps}st] test={ {k: round(v, 5) for k, v in mean.items()} } '
                  f'wall={wall:.1f}s peak={peak:.0f}MB', flush=True)

    # ---- aggregate across seeds
    import statistics as st
    agg = {}
    for arm in ['A', 'B', 'C', 'D1x', 'D3x']:
        agg[arm] = {}
        for K in BUDGETS:
            if arm == 'A':
                vals = [results['arms'][f'A_seed{s}'][f'K{K}']['eval'][K] for s in seeds]
            else:
                vals = [results['arms'][f'{arm}_seed{s}']['eval'][K] for s in seeds]
            agg[arm][f'K{K}'] = {'mean': st.mean(vals), 'std': st.stdev(vals) if len(vals) > 1 else 0.0,
                                 'values': vals}
    results['aggregate'] = agg

    a32 = agg['A']['K32']['mean']
    b32 = agg['B']['K32']['mean']
    tax = (b32 - a32) / a32
    results['summary'] = {
        'alignment_tax_B_vs_A_at_K32': tax,
        'B_vs_A_all_budgets': {f'K{K}': (agg['B'][f'K{K}']['mean'] - agg['A'][f'K{K}']['mean'])
                               / agg['A'][f'K{K}']['mean'] for K in BUDGETS},
        'C_vs_B_all_budgets': {f'K{K}': (agg['C'][f'K{K}']['mean'] - agg['B'][f'K{K}']['mean'])
                               / agg['B'][f'K{K}']['mean'] for K in BUDGETS},
        'D3x_vs_A_all_budgets': {f'K{K}': (agg['D3x'][f'K{K}']['mean'] - agg['A'][f'K{K}']['mean'])
                                 / agg['A'][f'K{K}']['mean'] for K in BUDGETS},
        'D1x_vs_A_all_budgets': {f'K{K}': (agg['D1x'][f'K{K}']['mean'] - agg['A'][f'K{K}']['mean'])
                                 / agg['A'][f'K{K}']['mean'] for K in BUDGETS},
        'A_truncation_penalty': {f'K{K}': results['arms'][f'A_seed{seeds[0]}']['K32']['eval'][K] /
                                 results['arms'][f'A_seed{seeds[0]}']['K32']['eval'][32]
                                 for K in BUDGETS},
    }

    # ---- serial escalation on B seed0
    b0 = results['arms'][f'B_seed{seeds[0]}']
    results['escalation_B'] = escalation_analysis(b0['per_sample'])
    # attention-share diagnostic under arm C (no mask)
    try:
        m = results['arms'][f'C_seed{seeds[0]}']
        results['note'] = 'see packing_equivalence for no-mask attention share at init'
    except Exception:
        pass

    with open(args.out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n== AGGREGATE (mean +- std over seeds)', flush=True)
    print(f"{'arm':5s} " + ' '.join(f'{"K"+str(k):>16s}' for k in BUDGETS), flush=True)
    for arm in ['A', 'B', 'C', 'D1x', 'D3x']:
        row = ' '.join(f"{agg[arm]['K'+str(k)]['mean']:9.5f}+-{agg[arm]['K'+str(k)]['std']:6.5f}"
                       for k in BUDGETS)
        print(f'{arm:5s} {row}', flush=True)
    print(f"\nalignment tax B vs A @K32 = {tax * 100:+.2f}%  (|tax|<=5% required)", flush=True)
    print('B vs A %:', {k: round(v * 100, 2) for k, v in results['summary']['B_vs_A_all_budgets'].items()}, flush=True)
    print('C vs B %:', {k: round(v * 100, 2) for k, v in results['summary']['C_vs_B_all_budgets'].items()}, flush=True)
    print('D3x vs A %:', {k: round(v * 100, 2) for k, v in results['summary']['D3x_vs_A_all_budgets'].items()}, flush=True)
    print('D1x vs A %:', {k: round(v * 100, 2) for k, v in results['summary']['D1x_vs_A_all_budgets'].items()}, flush=True)
    print('A(K32 model) truncation ratio:', {k: round(v, 3) for k, v in
                                             results['summary']['A_truncation_penalty'].items()}, flush=True)
    print('\nescalation (B):', json.dumps(results['escalation_B']['base']), flush=True)
    for r in results['escalation_B']['table']:
        print(' ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    print('  min budget within 5% of K32:', results['escalation_B']['min_budget_within_5pct_of_K32'], flush=True)
    print(f'\nwrote {args.out}', flush=True)


# thin wrappers so main() stays readable
def train_packed(mode, steps, seed, train_imgs, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lam(s, steps))
    g = torch.Generator(device=dev).manual_seed(seed + 9999)
    N = train_imgs.shape[0]
    torch.cuda.reset_peak_memory_stats(dev)
    model.train()
    curve, t0 = [], time.time()
    for step in range(steps):
        idx = torch.randint(0, N, (BATCH,), device=dev, generator=g)
        imgs = train_imgs[idx]
        z = model.encode(imgs)
        rec = model.decode(z, BUDGETS, mode == 'packed_mask')
        tgt = imgs.unsqueeze(1).expand(-1, len(BUDGETS), -1, -1, -1)
        loss = (rec - tgt).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            curve.append((step, float(loss.item())))
    torch.cuda.synchronize(dev)
    return model, curve, time.time() - t0, torch.cuda.max_memory_allocated(dev) / 1e6


def train_random(steps, seed, train_imgs, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lam(s, steps))
    g = torch.Generator(device='cpu').manual_seed(seed + 9999)
    N = train_imgs.shape[0]
    torch.cuda.reset_peak_memory_stats(dev)
    model.train()
    curve, t0 = [], time.time()
    for step in range(steps):
        idx = torch.randint(0, N, (BATCH,), device=dev)
        imgs = train_imgs[idx]
        z = model.encode(imgs)
        k = BUDGETS[torch.randint(0, len(BUDGETS), (1,), generator=g).item()]
        rec = model.decode(z, [k], False)
        loss = (rec - imgs.unsqueeze(1)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            curve.append((step, float(loss.item())))
    torch.cuda.synchronize(dev)
    return model, curve, time.time() - t0, torch.cuda.max_memory_allocated(dev) / 1e6


if __name__ == '__main__':
    main()
