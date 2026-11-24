#!/bin/bash
# Wait for BOTH matched-budget vendor rounds (pre = ViT tokens, post = LLM input) to finish,
# then pull jsons/logs/grids from each.
# 用法: bash tools/dev/_monitor_probe_vendor2.sh
set -u
HOST='-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p 17677 root@connect.sha1.seetacloud.com'
PRE=/root/autodl-tmp/tok256x12_t8/probe_vendor_pre
POST=/root/autodl-tmp/tok256x12_t8/probe_vendor_post
LOCAL=/home/linaro/dsh/tok448_work/artifacts/probe_vendor2
mkdir -p "$LOCAL/pre" "$LOCAL/post" "$LOCAL/pre/logs" "$LOCAL/post/logs" \
         "$LOCAL/pre/grids" "$LOCAL/post/grids"
for i in $(seq 1 120); do          # 120 × 120 s = 4 h
  out=$(timeout 80 ssh $HOST "
      echo PRE: \$(tail -1 $PRE/status.txt 2>/dev/null | cut -c1-90)
      echo POST: \$(tail -1 $POST/status.txt 2>/dev/null | cut -c1-90)
      echo PRE_JSON: \$(ls $PRE/*.json 2>/dev/null | wc -l) POST_JSON: \$(ls $POST/*.json 2>/dev/null | wc -l)
      grep -h 'RESULT' $PRE/logs/*.log $POST/logs/*.log 2>/dev/null | tail -8
    " 2>&1 | tr '\n' '|')
  echo "[$(date +%H:%M:%S)] $out"
  case "$out" in
    *"PRE: [run] DONE"*|*"PRE: [run] start"*)
      pre_done=1;;
  esac
  pd=0; qd=0
  case "$out" in *"[run] DONE"*) :;; esac
  # 粗判: 两个 status 的最后一行都含 DONE
  echo "$out" | grep -q "PRE: \[run\] DONE" && pd=1
  echo "$out" | grep -q "POST: \[run\] DONE" && qd=1
  [ "$pd" = "1" ] && [ "$qd" = "1" ] && break
  sleep 120
done
echo "=== fetching ==="
for d in pre post; do
  S=$PRE; [ "$d" = "post" ] && S=$POST
  timeout 240 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
    "root@connect.sha1.seetacloud.com:$S/*.json" "$LOCAL/$d/" 2>&1
  timeout 240 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
    "root@connect.sha1.seetacloud.com:$S/status.txt" "$LOCAL/$d/" 2>&1
  timeout 300 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
    "root@connect.sha1.seetacloud.com:$S/logs/*.log" "$LOCAL/$d/logs/" 2>&1
  timeout 300 scp -P 17677 -o StrictHostKeyChecking=no -o BatchMode=yes \
    "root@connect.sha1.seetacloud.com:$S/grids/*.png" "$LOCAL/$d/grids/" 2>&1
done
echo "=== two-column summary ==="
python3 - "$LOCAL" <<'PY'
import json, glob, os, sys
root = sys.argv[1]
def load(sub):
    out = {}
    for p in glob.glob(os.path.join(root, sub, "*.json")):
        d = json.load(open(p))
        out[d["vendor"]] = d
    return out
pre, post = load("pre"), load("post")
print(f"{'vendor':22s} {'ViT token (pre)':>18s} {'LLM token (post)':>18s}   Δ")
for v in ("deepseek_v41_flash", "qwen38_flash", "glm5_flash", "mimo26_flash"):
    a, b = pre.get(v), post.get(v)
    fa = f"{a['psnr']:.2f} dB" if a else "—"
    fb = f"{b['psnr']:.2f} dB" if b else "—"
    dd = f"{a['psnr']-b['psnr']:+.2f} dB" if (a and b) else "—"
    print(f"{v:22s} {fa:>18s} {fb:>18s}   {dd}")
PY
