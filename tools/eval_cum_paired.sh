#!/usr/bin/env bash
# =============================================================================
# eval_cum_paired.sh — 配对 cumulative B 相臂的评测（固定 8 步, 全量 5,000）
# -----------------------------------------------------------------------------
# 等 PAIRED_DONE → 双卡并行: GPU0 = 128 / GPU1 = 96 64
# 每臂只评 **T=8**（末步 + 逐步）——T=1 ckpt 是 fixed 臂的 A 相（已由主实验评过）。
# 产物: $OUT/eval/{linattn_k<K>_t8.json, stepwise_k<K>_t8.json}
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/eval_cum_paired.sh > $OUT/eval_launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
OUT=${OUT:-/root/autodl-tmp/srdiff_cum_paired}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU0_KS=${GPU0_KS:-"128"}
GPU1_KS=${GPU1_KS:-"96 64"}
export PYTHONPATH=$PWD

if [ -f "$OUT/PE_DONE" ]; then echo "[eval-paired] PE_DONE 已存在, 跳过"; exit 0; fi
echo "[eval-paired] 等待 PAIRED_DONE ... $(date -Is)"
while [ ! -f "$OUT/PAIRED_DONE" ]; do sleep 60; done
echo "[eval-paired] 开始评测 $(date -Is)"
EV=$OUT/eval; mkdir -p "$EV"

eval_ks() {
  local gpu=$1; shift
  local K
  for K in "$@"; do
    export CUDA_VISIBLE_DEVICES=$gpu
    local ck=$OUT/stdattn_k${K}_t8/ckpt_best.pt
    echo "== [paired eval] GPU$gpu K=$K $(date -Is)"
    $PY -u eval_linattn.py --ckpt "$ck" --data_dir "$DATA" \
      --batch_size 128 --num_workers 8 --out "$EV/linattn_k${K}_t8.json" \
      > "$EV/linattn_k${K}_t8.log" 2>&1
    $PY -u eval_stepwise_linattn.py --ckpt "$ck" --data_dir "$DATA" \
      --batch_size 64 --num_workers 8 --out "$EV/stepwise_k${K}_t8.json" \
      > "$EV/stepwise_k${K}_t8.log" 2>&1
    echo "== [paired eval] GPU$gpu K=$K rc=$?"
  done
}

eval_ks 0 $GPU0_KS > "$OUT/eval_gpu0.out" 2>&1 &
P0=$!
eval_ks 1 $GPU1_KS > "$OUT/eval_gpu1.out" 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "[eval-paired] 结束 rc=$R0/$R1 $(date -Is)"
$PY tools/summarize_fixedk.py --out_root "$OUT" --csv "$OUT/train_summary.csv" \
  --ks 64 96 128 > "$OUT/summarize.out" 2>&1
touch "$OUT/PE_DONE"
echo "[eval-paired] PE_DONE $(date -Is)"
