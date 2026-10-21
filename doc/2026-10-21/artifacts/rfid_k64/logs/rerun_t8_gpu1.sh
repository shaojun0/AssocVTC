#!/usr/bin/env bash
# 重跑 4 个 t8 臂: 旧进程在 step-index bug 修复前启动 ⇒ 错取 step1
set -uo pipefail
cd /root/AssocVTC
PY=/root/miniconda3/bin/python
OUT=/root/autodl-tmp/rfid_k64
K=/root/autodl-tmp/srdiff_k8sweep
for k in 8 16 32 48; do
  echo "=== [gpu1] k${k}_t8 rerun start $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=1 $PY tools/eval_rfid.py --ckpt $K/stdattn_k${k}_t8/ckpt_best.pt \
      --out_dir $OUT --tag k${k}_t8 --steps last > $OUT/log_k${k}_t8.out 2>&1
  echo "=== [gpu1] k${k}_t8 rerun rc=$? end $(date -Is) ==="
done
echo RERUN_DONE $(date -Is) > $OUT/RERUN_T8_DONE
