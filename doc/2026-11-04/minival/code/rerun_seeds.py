#!/usr/bin/env python3
"""Supplementary seeds for Track 1: A (single-budget) seeds 2,3 + B seeds 2,3 + A(K=32)@3000 steps."""
import json, torch, time
import minival as M

dev = 'cuda'
tr = M.make_data(2000, 1234).to(dev)
te = M.make_data(200, 5678).to(dev)
out = {}

for seed in [2, 3]:
    A = {}
    for K in [8, 16, 32]:
        model, curve, wall, peak = M.train_single_fixed(K, 1500, seed, tr, dev)
        mean, per = M.evaluate(model, te)
        A[f'K{K}'] = {'eval': mean, 'per_sample': per, 'train_losses': curve,
                      'wall_s': wall, 'peak_mem_mb': peak,
                      'params': sum(p.numel() for p in model.parameters())}
        print(f'[A seed{seed} K={K}]', {k: round(v, 4) for k, v in mean.items()},
              f'wall={wall:.0f}s', flush=True)
    out[f'A_seed{seed}'] = A
    model, curve, wall, peak = M.train_packed('packed_mask', 1500, seed, tr, dev)
    mean, per = M.evaluate(model, te)
    out[f'B_seed{seed}'] = {'eval': mean, 'per_sample': per, 'train_losses': curve,
                            'wall_s': wall, 'peak_mem_mb': peak, 'steps': 1500,
                            'params': sum(p.numel() for p in model.parameters())}
    print(f'[B seed{seed}]', {k: round(v, 4) for k, v in mean.items()}, flush=True)

model, curve, wall, peak = M.train_single_fixed(32, 3000, 0, tr, dev)
mean, per = M.evaluate(model, te)
out['A_seed0_K32_steps3000'] = {'eval': mean, 'per_sample': per, 'train_losses': curve,
                                'wall_s': wall, 'peak_mem_mb': peak, 'steps': 3000}
print('[A seed0 K=32 3000 steps]', {k: round(v, 4) for k, v in mean.items()}, flush=True)

json.dump(out, open('results_extra.json', 'w'), indent=1)
print('wrote results_extra.json', flush=True)
