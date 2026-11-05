#!/usr/bin/env python3
"""Track 1 — converged-regime rerun at 3000 steps (seed 0), to separate 'packing tax' from
'1500-step under-training / bad-basin seed noise'. Arms: A(K=8,16), B, C, D1x (A K=32 @3000
is already in results_extra.json from rerun_seeds.py)."""
import json
import torch
import minival as M

dev = 'cuda'
tr = M.make_data(2000, 1234).to(dev)
te = M.make_data(200, 5678).to(dev)
STEPS = 3000
out = {'meta': {'steps': STEPS, 'seed': 0, 'batch': 64, 'lr': 1e-3}}
A = {}
for K in [8, 16]:
    model, curve, wall, peak = M.train_single_fixed(K, STEPS, 0, tr, dev)
    mean, per = M.evaluate(model, te)
    A[f'K{K}'] = {'eval': mean, 'per_sample': per, 'train_losses': curve,
                  'wall_s': wall, 'peak_mem_mb': peak}
    print(f'[A K={K} {STEPS}st]', {k: round(v, 4) for k, v in mean.items()},
          f'wall={wall:.0f}s', flush=True)
out['A_seed0_steps3000'] = A

for arm, mode in [('B', 'packed_mask'), ('C', 'packed_nomask')]:
    model, curve, wall, peak = M.train_packed(mode, STEPS, 0, tr, dev)
    mean, per = M.evaluate(model, te)
    out[f'{arm}_seed0_steps3000'] = {'eval': mean, 'per_sample': per, 'train_losses': curve,
                                     'wall_s': wall, 'peak_mem_mb': peak}
    print(f'[{arm} {STEPS}st]', {k: round(v, 4) for k, v in mean.items()},
          f'wall={wall:.0f}s', flush=True)

model, curve, wall, peak = M.train_random(STEPS, 0, tr, dev)
mean, per = M.evaluate(model, te)
out['D1x_seed0_steps3000'] = {'eval': mean, 'per_sample': per, 'train_losses': curve,
                              'wall_s': wall, 'peak_mem_mb': peak}
print(f'[D1x {STEPS}st]', {k: round(v, 4) for k, v in mean.items()}, flush=True)

json.dump(out, open('results_long.json', 'w'), indent=1)
print('wrote results_long.json', flush=True)
