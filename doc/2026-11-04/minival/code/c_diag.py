#!/usr/bin/env python3
"""C (no-mask) diagnostic: is its low-budget failure 'attention dilution' or 'inference mismatch'?
Train C again (seed 0) and compare:
  (a) no_mask_unpacked: inference with only z[:K] keys  -> deployment form (train/test mismatch)
  (b) no_mask_packed  : inference with the packed 56-key duplicating input it trained on
If (b) is much better at K=8, the no-mask model learned to *read tokens beyond its budget*
(it cheats), which is exactly why unpacked deployment collapses.
"""
import json
import torch
import minival as M

dev = 'cuda'
tr = M.make_data(2000, 1234).to(dev)
te = M.make_data(200, 5678).to(dev)
model, curve, wall, peak = M.train_packed('packed_nomask', 1500, 0, tr, dev)
mean, per = M.evaluate(model, te)
res = {'no_mask_unpacked_inference': mean, 'wall_s': wall, 'peak_mem_mb': peak,
       'train_losses': curve}
model.eval()
with torch.no_grad():
    z = model.encode(te)
    rec = model.decode(z, [8, 16, 32], use_mask=False)          # (B,3,3,32,32)
    res['no_mask_packed_inference'] = {K: float((rec[:, i] - te).abs().mean())
                                       for i, K in enumerate([8, 16, 32])}
json.dump(res, open('c_diag.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'train_losses'}, indent=1), flush=True)
