#!/bin/bash
# Poll the four-vendor vision-tower decodability probe until DONE, then pull artifacts.
# 用法（agent 以后台 job 启动）: bash tools/dev/_monitor_probe_vendor.sh
set -u
HOST='-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p 17677 root@connect.sha1.seetacloud.com'
ST=/root/autodl-tmp/tok256x12_t8/probe_vendor
LOCAL=/home/linaro/dsh/tok448_work/artifacts/probe_vendor
mkdir -p "$LOCAL/logs" "$LOCAL/grids"
for i in $(seq 1 90); do          # 90 × 120 s = 3 h
  out=$(timeout 70 ssh $HOST "
      tail -3 $ST/status.txt 2>/dev/null
      echo '--- jsons:'
      ls $ST/*.json 2>/dev/null | wc -l
      grep -h 'RESULT' $ST/logs/*.log 2>/dev/null | tail -8
      echo '--- step:'
      grep -h 'step ' $ST/logs/*.log 2>/dev/null | tail -4
    " 2>&1 | tr '\n' '|')
  echo "[$(date +%H:%M:%S)] $out"
  case "$out" in *"[run] DONE"*) break;; esac
  sleep 120
done
echo "=== fetching ==="
timeout 240 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/*.json" "$LOCAL/" 2>&1
timeout 240 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/status.txt" "$LOCAL/" 2>&1
timeout 300 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/logs/*.log" "$LOCAL/logs/" 2>&1
timeout 300 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
  "root@connect.sha1.seetacloud.com:$ST/grids/*.png" "$LOCAL/grids/" 2>&1
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
print(f"{'vendor':22s} {'lab':14s} res  feat                     PSNR    SSIM    L1     dec")
for r in rows:
    print(f"{r.get('vendor',''):22s} {r.get('lab',''):14s} {r['enc_res']:4d} "
          f"{str(r['feat_shape']):24s} {r['psnr']:6.2f} {r['ssim']:.4f} {r['l1']:.4f} "
          f"{r['dec_params']/1e6:.1f}M")
PY
echo "=== status tail ==="
tail -15 "$LOCAL/status.txt" 2>/dev/null
