#!/bin/bash
# Poll the remote encoder-decodability probe (tools/probe_enc_dec.py) until DONE,
# then pull the per-encoder JSONs and logs.
# 用法（agent 以后台 job 启动）: bash tools/dev/_monitor_probe.sh
set -u
HOST='-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p 17677 root@connect.sha1.seetacloud.com'
ST=/root/autodl-tmp/tok256x12_t8/probe_native
LOCAL=/home/linaro/dsh/tok448_work/artifacts/probe
mkdir -p "$LOCAL/logs"
for i in $(seq 1 90); do          # 90 × 120 s = 3 h
  out=$(timeout 70 ssh $HOST "
      tail -2 $ST/status.txt 2>/dev/null
      echo '--- jsons:'
      ls $ST/*.json 2>/dev/null | wc -l
      grep -h '\[probe\] RESULT' $ST/logs/*.log 2>/dev/null | tail -12
      echo '--- running:'
      pgrep -af 'probe_enc_dec[.]py' | sed -E 's/.*--enc ([a-z0-9_]+).*/\1/' | tr '\n' ' '
    " 2>&1 | tr '\n' '|')
  echo "[$(date +%H:%M:%S)] $out"
  case "$out" in *"[probe] DONE"*) break;; esac
  sleep 120
done
echo "=== fetching ==="
timeout 240 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/*.json" "$LOCAL/" 2>&1
timeout 240 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/status.txt" "$LOCAL/" 2>&1
mkdir -p "$LOCAL/grids"
timeout 300 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/grids/*.png" "$LOCAL/grids/" 2>&1
timeout 300 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/logs/*.log" "$LOCAL/logs/" 2>&1
echo "=== summary ==="
python3 - "$LOCAL" <<'PY'
import json, glob, os, sys
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    try:
        rows.append(json.load(open(p)))
    except Exception as e:
        print("ERR", os.path.basename(p), e)
rows.sort(key=lambda d: -d.get("psnr", 0))
print(f"{'enc':14s} {'vendor':9s} res  feat                    PSNR    SSIM    L1     "
      f"dec   ntrain steps")
for r in rows:
    print(f"{r['enc']:14s} {r['vendor']:9s} {r['enc_res']:4d} {str(r['feat_shape']):22s} "
          f"{r['psnr']:6.2f} {r['ssim']:.4f} {r['l1']:.4f} "
          f"{r['dec_params']/1e6:.1f}M {r['train_images']:6d} {r['steps']:5d}")
PY
echo "=== status tail ==="
tail -15 "$LOCAL/status.txt" 2>/dev/null
