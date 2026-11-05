#!/usr/bin/env python3
"""Merge Track1 main run (seeds 0,1 all arms) + supplementary seeds -> merged_results.json + analysis.md"""
import json, statistics as st

main = json.load(open('results.json'))
extra = json.load(open('results_extra.json'))
try:
    longr = json.load(open('results_long.json'))
except FileNotFoundError:
    longr = None
arms = dict(main['arms'])
arms.update({k: v for k, v in extra.items() if k != 'A_seed0_K32_steps3000'})
A_extra = extra['A_seed0_K32_steps3000']

BUD = [8, 16, 32]
A_SEEDS = [0, 1, 2, 3]
B_SEEDS = [0, 1, 2, 3]
C_SEEDS = [0, 1]
D_SEEDS = [0, 1]

def a_eval(seed, K):
    return arms[f'A_seed{seed}'][f'K{K}']['eval'][str(K)]

def arm_eval(arm, seed, K):
    d = arms[f'{arm}_seed{seed}']
    return d['eval'][str(K)]

def agg(vals):
    return {'mean': st.mean(vals), 'std': st.stdev(vals) if len(vals) > 1 else 0.0,
            'min': min(vals), 'max': max(vals), 'n': len(vals), 'values': vals}

aggregate = {}
for K in BUD:
    aggregate.setdefault('A', {})[f'K{K}'] = agg([a_eval(s, K) for s in A_SEEDS])
    aggregate.setdefault('B', {})[f'K{K}'] = agg([arm_eval('B', s, K) for s in B_SEEDS])
    aggregate.setdefault('C', {})[f'K{K}'] = agg([arm_eval('C', s, K) for s in C_SEEDS])
    aggregate.setdefault('D1x', {})[f'K{K}'] = agg([arm_eval('D1x', s, K) for s in D_SEEDS])
    aggregate.setdefault('D3x', {})[f'K{K}'] = agg([arm_eval('D3x', s, K) for s in D_SEEDS])

# per-seed arm-A table (instability)
a_rows = []
for s in A_SEEDS:
    a_rows.append({'seed': s,
                   'trained_K8_eval_K8': a_eval(s, 8),
                   'trained_K16_eval_K16': a_eval(s, 16),
                   'trained_K32_eval_K32': a_eval(s, 32),
                   'K32model_at_K8': arms[f'A_seed{s}']['K32']['eval']['8'],
                   'K32model_at_K16': arms[f'A_seed{s}']['K32']['eval']['16']})

tax = {}
for K in BUD:
    a_mean = aggregate['A'][f'K{K}']['mean']
    a_best = aggregate['A'][f'K{K}']['min']
    b = aggregate['B'][f'K{K}']['mean']
    tax[f'K{K}'] = {'B_mean': b, 'A_mean_4seed': a_mean, 'A_best_seed': a_best,
                    'tax_vs_mean_pct': (b - a_mean) / a_mean * 100,
                    'tax_vs_best_seed_pct': (b - a_best) / a_best * 100}

c_vs_b = {f'K{K}': (aggregate['C'][f'K{K}']['mean'] - aggregate['B'][f'K{K}']['mean'])
          / aggregate['B'][f'K{K}']['mean'] * 100 for K in BUD}
d1x_vs_a = {f'K{K}': (aggregate['D1x'][f'K{K}']['mean'] - aggregate['A'][f'K{K}']['mean'])
            / aggregate['A'][f'K{K}']['mean'] * 100 for K in BUD}
d3x_vs_a = {f'K{K}': (aggregate['D3x'][f'K{K}']['mean'] - aggregate['A'][f'K{K}']['mean'])
            / aggregate['A'][f'K{K}']['mean'] * 100 for K in BUD}

# per-seed B numbers (stability across seeds)
b_rows = [{'seed': s, **{f'K{K}': arm_eval('B', s, K) for K in BUD}} for s in B_SEEDS]

merged = {'meta': main['meta'], 'packing_equivalence': main['packing_equivalence'],
          'arms': arms, 'aggregate': aggregate, 'a_per_seed': a_rows, 'b_per_seed': b_rows,
          'tax': tax, 'c_vs_b_pct': c_vs_b, 'd1x_vs_a_pct': d1x_vs_a,
          'd3x_vs_a_pct': d3x_vs_a,
          'escalation_B': main['escalation_B'],
          'A_seed0_K32_steps3000': {k: A_extra['eval'][k] for k in A_extra['eval']},
          'A_seed0_K32_steps3000_curve': A_extra['train_losses']}
# ---- markdown ----
L = []
L.append('### Table 1 — single-budget arms A (per seed); L1 at the budget each model was trained at\n')
L.append('| seed | trained K=8 | trained K=16 | trained K=32 | K=32 model @K=8 | K=32 model @K=16 |')
L.append('|---|---|---|---|---|---|')
for r in a_rows:
    L.append(f"| {r['seed']} | {r['trained_K8_eval_K8']:.4f} | {r['trained_K16_eval_K16']:.4f} | "
             f"{r['trained_K32_eval_K32']:.4f} | {r['K32model_at_K8']:.4f} | {r['K32model_at_K16']:.4f} |")
L.append('\n### Table 2 — aggregate test L1 (mean ± std over seeds)\n')
L.append('| arm | seeds | K=8 | K=16 | K=32 |')
L.append('|---|---|---|---|---|')
ns = {'A': len(A_SEEDS), 'B': len(B_SEEDS), 'C': len(C_SEEDS), 'D1x': len(D_SEEDS), 'D3x': len(D_SEEDS)}
for arm in ['A', 'B', 'C', 'D1x', 'D3x']:
    L.append(f"| {arm} | {ns[arm]} | " + ' | '.join(
        f"{aggregate[arm][f'K{K}']['mean']:.4f}±{aggregate[arm][f'K{K}']['std']:.4f}" for K in BUD) + ' |')
L.append('\n### Table 3 — per-seed B (stability)\n')
L.append('| seed | K=8 | K=16 | K=32 |')
L.append('|---|---|---|---|')
for r in b_rows:
    L.append(f"| {r['seed']} | {r['K8']:.4f} | {r['K16']:.4f} | {r['K32']:.4f} |")
L.append('\n### Table 4 — alignment tax of B vs A\n')
L.append('| budget | B mean | A mean (4 seeds) | A best seed | tax vs A mean | tax vs A best seed |')
L.append('|---|---|---|---|---|---|')
for K in BUD:
    t = tax[f'K{K}']
    L.append(f"| K={K} | {t['B_mean']:.4f} | {t['A_mean_4seed']:.4f} | {t['A_best_seed']:.4f} | "
             f"{t['tax_vs_mean_pct']:+.1f}% | {t['tax_vs_best_seed_pct']:+.1f}% |")
L.append('\n### Table 5 — C (no mask) vs B, D vs A [%]\n')
L.append('| budget | C vs B | D1x vs A | D3x vs A |')
L.append('|---|---|---|---|')
for K in BUD:
    L.append(f"| K={K} | {c_vs_b[f'K{K}']:+.1f}% | {d1x_vs_a[f'K{K}']:+.1f}% | {d3x_vs_a[f'K{K}']:+.1f}% |")
L.append('\n### Table 6 — serial escalation (B model, seed 0, oracle-L1 policy)\n')
e = main['escalation_B']
L.append(f"always-K8={e['base']['K8']:.4f}, always-K16={e['base']['K16']:.4f}, always-K32={e['base']['K32']:.4f}\n")
L.append('| escalate frac | mean L1 | mean budget | coverage of hard quartile | precision (benefited) |')
L.append('|---|---|---|---|---|')
for r in e['table']:
    L.append(f"| {r['escalate_frac']:.1f} | {r['mean_l1']:.4f} | {r['mean_budget']:.1f} | "
             f"{r['coverage_of_hard_quartile']:.2f} | {r['precision_benefited']:.2f} |")
L.append('\n### Packing equivalence / attention budget\n')
L.append('```json\n' + json.dumps(main['packing_equivalence'], indent=1) + '\n```')
L.append(f"\n### A(K=32) at 3000 steps (seed 0): " + json.dumps(
    {k: round(v, 4) for k, v in A_extra['eval'].items()}))
# ---- convergence classification at 1500 steps (last logged train loss) ----
def final_loss(curve):
    return curve[-1][1] if curve else float('nan')

conv = {}
for K in BUD:
    vals = [arms[f'A_seed{s}'][f'K{K}']['train_losses'] for s in A_SEEDS]
    conv[f'A_K{K}'] = [round(final_loss(v), 4) for v in vals]
for arm in ['B', 'C', 'D1x', 'D3x']:
    seeds = B_SEEDS if arm == 'B' else D_SEEDS
    conv[arm] = [round(final_loss(arms[f'{arm}_seed{s}']['train_losses']), 4) for s in seeds]
L.append('\n### Table 7 — 1500-step regime: final train loss per run (bimodality: >0.13 = stuck in bad basin)\n')
for k, v in conv.items():
    L.append(f"- {k}: {v}")
merged['final_train_losses'] = conv
merged['final_train_losses_long'] = ({k: round(final_loss(v['train_losses']), 4)
                                      for k, v in longr.items() if isinstance(v, dict) and 'train_losses' in v}
                                     if longr else None)

# ---- converged regime (3000 steps, seed 0) ----
if longr:
    T = {}
    A8 = longr['A_seed0_steps3000']['K8']['eval']
    A16 = longr['A_seed0_steps3000']['K16']['eval']
    A32 = extra['A_seed0_K32_steps3000']['eval']
    B3 = longr['B_seed0_steps3000']['eval']
    C3 = longr['C_seed0_steps3000']['eval']
    D3 = longr['D1x_seed0_steps3000']['eval']
    L.append('\n### Table 8 — converged regime (3000 steps, seed 0): test L1 at K=8/16/32\n')
    L.append('| arm | K=8 | K=16 | K=32 | wall | peak MB |')
    L.append('|---|---|---|---|---|---|')
    rows = [('A (3 models, trained at each K)', A8, A16, A32, None, None),
            ('B packed+mask', B3, B3, B3, longr['B_seed0_steps3000']['wall_s'],
             longr['B_seed0_steps3000']['peak_mem_mb']),
            ('C packed no-mask', C3, C3, C3, longr['C_seed0_steps3000']['wall_s'],
             longr['C_seed0_steps3000']['peak_mem_mb']),
            ('D1x random single', D3, D3, D3, longr['D1x_seed0_steps3000']['wall_s'],
             longr['D1x_seed0_steps3000']['peak_mem_mb'])]
    for name, e8, e16, e32, wall, peak in rows:
        L.append(f"| {name} | {e8['8']:.4f} | {e16['16']:.4f} | {e32['32']:.4f} | "
                 f"{'' if wall is None else f'{wall:.0f}s'} | {'' if peak is None else f'{peak:.0f}'} |")
    tax_conv = {f'K{K}': (B3[str(K)] - [A8, A16, A32][i][str(K)]) / [A8, A16, A32][i][str(K)] * 100
                for i, K in enumerate(BUD)}
    L.append(f"\nconverged-regime alignment tax B vs A: " + json.dumps(
        {k: round(v, 2) for k, v in tax_conv.items()}))
    L.append(f"\nA(K=32 @3000) truncation ratio: K8/K32={A32['8']/A32['32']:.2f}x, K16/K32={A32['16']/A32['32']:.2f}x")
    L.append(f"\nA(K=32 @3000) eval: {json.dumps({k: round(v, 4) for k, v in A32.items()})}")
    L.append(f"\nC@3000 vs B@3000 [%]: " + json.dumps(
        {K: round((C3[str(K)] - B3[str(K)]) / B3[str(K)] * 100, 1) for K in BUD}))
    L.append(f"\nD1x@3000 vs A@3000 [%]: " + json.dumps(
        {K: round((D3[str(K)] - [A8, A16, A32][i][str(K)]) / [A8, A16, A32][i][str(K)] * 100, 1)
         for i, K in enumerate(BUD)}))
    merged['converged_regime'] = {'A8': A8, 'A16': A16, 'A32': A32, 'B': B3, 'C': C3, 'D1x': D3,
                                  'tax_pct': tax_conv}

json.dump(merged, open('merged_results.json', 'w'), indent=1)
open('analysis.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
