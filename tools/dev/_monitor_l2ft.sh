#!/bin/bash
# Poll the remote L2-finetune run until DONE, then pull artifacts locally.
# Launched as a background job by the agent; prints one line per 5 min.
set -u
HOST='-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p 17677 root@connect.sha1.seetacloud.com'
ST=/root/autodl-tmp/tok256x12_t8/l2ft
LOCAL=/home/linaro/dsh/tok448_work/artifacts/l2ft
mkdir -p "$LOCAL"
for i in $(seq 1 110); do
  out=$(timeout 70 ssh $HOST "
      tail -1 $ST/status 2>/dev/null | cut -c1-110
      grep -E '\[val\] ep|step [0-9]+/2850' $ST/s2.log 2>/dev/null | tail -1 | cut -c1-150
    " 2>&1 | tr '\n' '|')
  echo "[$(date +%H:%M:%S)] $out"
  case "$out" in *"[l2ft] DONE"*) break;; esac
  sleep 300
done
echo "=== fetching artifacts ==="
timeout 120 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/*_fast.json" "$LOCAL/" 2>&1
timeout 120 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/s2/args.json" \
  "root@connect.sha1.seetacloud.com:$ST/s2/eval_val.json" \
  "root@connect.sha1.seetacloud.com:$ST/s2/train_log.jsonl" \
  "root@connect.sha1.seetacloud.com:$ST/status" "$LOCAL/" 2>&1
echo "=== local files ==="
ls -la "$LOCAL"
echo "=== per-ckpt step8 summary ==="
python3 - <<'PY'
import json, glob, os
for p in sorted(glob.glob('/home/linaro/dsh/tok448_work/artifacts/l2ft/*_fast.json')):
    try:
        d = json.load(open(p))
    except Exception as e:
        print(os.path.basename(p), 'ERR', e); continue
    s = d['per_step'][str(max(int(k) for k in d['per_step']))]
    print(f"{os.path.basename(p):28s} values={s['values']:5d} PSNR={s['psnr']:.3f} rFID={s['rfid']:.2f}")
PY
echo "=== status tail ==="
tail -20 "$LOCAL/status" 2>/dev/null
