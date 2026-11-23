#!/bin/bash
# Poll the remote kd2 run (latent KD + teacher-decoder render) until DONE, then pull artifacts.
# 用法（由 agent 以后台 job 启动）: nohup bash tools/dev/_monitor_kd2.sh > /tmp/mon_kd2.log 2>&1 &
set -u
HOST='-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p 17677 root@connect.sha1.seetacloud.com'
ST=/root/autodl-tmp/tok256x12_t8/kd2
LOCAL=/home/linaro/dsh/tok448_work/artifacts/kd2
mkdir -p "$LOCAL"
for i in $(seq 1 80); do
  out=$(timeout 70 ssh $HOST "
      tail -1 $ST/status 2>/dev/null | cut -c1-120
      grep -E 'step [0-9]+/1900' $ST/s2.log 2>/dev/null | tail -1 | cut -c1-170
    " 2>&1 | tr '\n' '|')
  echo "[$(date +%H:%M:%S)] $out"
  case "$out" in *"[kd2] DONE"*) break;; esac
  sleep 300
done
echo "=== fetching artifacts ==="
timeout 180 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/*_fast.json" "$LOCAL/" 2>&1
timeout 120 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/s2/args.json" \
  "root@connect.sha1.seetacloud.com:$ST/s2/eval_val.json" \
  "root@connect.sha1.seetacloud.com:$ST/s2/train_log.jsonl" \
  "root@connect.sha1.seetacloud.com:$ST/status" "$LOCAL/" 2>&1
echo "=== local files ==="
ls -la "$LOCAL"
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
          f"step8 PSNR={s8['psnr']:.3f} rFID={s8['rfid']:7.2f} (render={d.get('render','pixel')})")
PY
echo "=== status tail ==="
tail -20 "$LOCAL/status" 2>/dev/null
