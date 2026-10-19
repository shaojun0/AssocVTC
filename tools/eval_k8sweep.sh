#!/usr/bin/env bash
# =============================================================================
# eval_k8sweep.sh — 「固定 8 步 × K token」K-sweep 评测（全量 5,000 留出集）
# -----------------------------------------------------------------------------
# 每臂两件事:
#   1) eval_linattn.py        → 末步（第 8 步）像素 L1 / PSNR / SSIM（与主报告同口径）
#   2) eval_stepwise_linattn.py → 逐步（1..8）L1/PSNR + 累计 token 数
# 另外把 A 相（T=1, 45ep）也评一遍, 作为「同 token 预算、单步」的参照。
#
# 用法: bash tools/eval_k8sweep.sh [K列表]     # 默认 8 12 16 24 32 64
# 产物: $OUT/eval/{linattn_k<K>_t8.json, stepwise_k<K>_t8.json, linattn_k<K>_t1.json}
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/srdiff_k8sweep}
EV=$OUT/eval
KS=${*:-"8 12 16 24 32 64"}
mkdir -p "$EV"
export PYTHONPATH=$PWD

for K in $KS; do
  CK1=$OUT/stdattn_k${K}_t1/ckpt_best.pt
  CK8=$OUT/stdattn_k${K}_t8/ckpt_best.pt
  echo "================ K=$K $(date -Is) ================"
  if [ -f "$CK8" ]; then
    $PY -u eval_linattn.py --ckpt "$CK8" --data_dir "$DATA" \
      --batch_size 128 --num_workers 12 --out "$EV/linattn_k${K}_t8.json" \
      > "$EV/linattn_k${K}_t8.log" 2>&1
    echo "[eval] K=$K T=8 final rc=$?"
    $PY -u eval_stepwise_linattn.py --ckpt "$CK8" --data_dir "$DATA" \
      --batch_size 64 --num_workers 10 --out "$EV/stepwise_k${K}_t8.json" \
      > "$EV/stepwise_k${K}_t8.log" 2>&1
    echo "[eval] K=$K T=8 stepwise rc=$?"
  else
    echo "[eval] K=$K 缺 $CK8，跳过"
  fi
  if [ -f "$CK1" ]; then
    $PY -u eval_linattn.py --ckpt "$CK1" --data_dir "$DATA" \
      --batch_size 128 --num_workers 12 --out "$EV/linattn_k${K}_t1.json" \
      > "$EV/linattn_k${K}_t1.log" 2>&1
    echo "[eval] K=$K T=1 final rc=$?"
  fi
done
echo "EVAL_K8SWEEP_DONE $(date -Is)" | tee "$EV/EVAL_DONE"
