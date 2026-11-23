#!/bin/bash
# Remote-side: snapshot ckpt_best at epoch milestones so we can plot rFID vs training progress.
set -u
ST=/root/autodl-tmp/tok256x12_t8
mkdir -p "$ST/snaps"
for i in $(seq 1 300); do
  ep=$(python3 -c "
import json
ep=0
try:
    for l in open('$ST/s2/train_log.jsonl'):
        r=json.loads(l)
        if r.get('split')=='val': ep=max(ep,int(r['epoch']))
except Exception: pass
print(ep)" 2>/dev/null)
  ep=${ep:-0}
  for e in 4 8 12; do
    if [ "$ep" -ge "$e" ] 2>/dev/null && [ ! -f "$ST/snaps/ep$e.pt" ] && [ -f "$ST/s2/ckpt_best.pt" ]; then
      cp "$ST/s2/ckpt_best.pt" "$ST/snaps/ep$e.pt" && echo "[snap] ep$e $(date)" >> "$ST/snaps/snap.log"
    fi
  done
  if [ -f "$ST/status" ] && grep -q "phase2" "$ST/status" 2>/dev/null; then break; fi
  sleep 300
done
