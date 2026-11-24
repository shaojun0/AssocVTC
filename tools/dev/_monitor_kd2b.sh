#!/bin/bash
# Poll the remote kd2b run (latent KD, scale=1.0, teacher-decoder render) until DONE,
# then pull artifacts. 用法（由 agent 以后台 job 启动）:
#   nohup bash tools/dev/_monitor_kd2b.sh > /tmp/mon_kd2b.log 2>&1 &
set -u
HOST='-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p 17677 root@connect.sha1.seetacloud.com'
ST=/root/autodl-tmp/tok256x12_t8/kd2b
LOCAL=/home/linaro/dsh/tok448_work/artifacts/kd2b
mkdir -p "$LOCAL"
for i in $(seq 1 100); do
  out=$(timeout 70 ssh $HOST "
      tail -1 $ST/status 2>/dev/null | cut -c1-150
      grep -E '^\[p0d\] step' $ST/status 2>/dev/null | tail -2
      grep -E 'kd_rel=' $ST/s2.log 2>/dev/null | tail -1 | cut -c1-230
      grep -E '^\[ *[0-9.]+m\] step' $ST/smoke.log 2>/dev/null | tail -1 | cut -c1-150
      grep -E '\[val\] ep' $ST/s2.log 2>/dev/null | tail -1 | cut -c1-150
    " 2>&1 | tr '\n' '|')
  echo "[$(date +%H:%M:%S)] $out"
  case "$out" in *"[kd2b] DONE"*|*"ABORT"*) break;; esac
  sleep 240
done
echo "=== fetching artifacts ==="
timeout 240 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/*_fast.json" "$LOCAL/" 2>&1
timeout 240 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/status" \
  "root@connect.sha1.seetacloud.com:$ST/p0d.log" \
  "root@connect.sha1.seetacloud.com:$ST/smoke.log" \
  "root@connect.sha1.seetacloud.com:$ST/s2/args.json" \
  "root@connect.sha1.seetacloud.com:$ST/s2/eval_val.json" \
  "root@connect.sha1.seetacloud.com:$ST/s2/train_log.jsonl" "$LOCAL/" 2>&1
echo "=== local files ==="
ls -la "$LOCAL"
echo "=== p0d (baseline 8-step: render vs D(E(.))) ==="
grep -E '^\[p0d\]' "$LOCAL/status" 2>/dev/null
echo "=== per-ckpt summary ==="
python3 - "$LOCAL" <<'PY'
import json, glob, os, sys
for p in sorted(glob.glob(os.path.join(sys.argv[1], '*_fast.json'))):
    try:
        d = json.load(open(p))
    except Exception as e:
        print(os.path.basename(p), 'ERR', e); continue
    ps = d['per_step']
    s1, s8 = ps['1'], ps[str(max(int(k) for k in ps))]
    print(f"{os.path.basename(p):26s} step1 rFID={s1['rfid']:7.2f} | "
          f"step8 PSNR={s8['psnr']:.3f} rFID={s8['rfid']:7.2f} "
          f"L1={s8['l1_255']:.2f} (render={d.get('render','pixel')})")
PY
echo "=== status tail ==="
tail -25 "$LOCAL/status" 2>/dev/null
