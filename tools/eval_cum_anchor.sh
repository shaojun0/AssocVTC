#!/usr/bin/env bash
# =============================================================================
# eval_cum_anchor.sh — 2026-10-21 新机器 cumulative 同机对照臂评测（K=64/96/128）
# -----------------------------------------------------------------------------
# 等锚臂训练 ANCHOR_DONE → 双卡并行 eval_k8sweep.sh（T=8 末步 + 逐步 + T=1）
# → summarize_fixedk.py 汇总 → touch AE_DONE
# 分派: GPU0 = 128 / GPU1 = 96 64
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/eval_cum_anchor.sh > /root/autodl-tmp/srdiff_cum_anchor/eval_launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
OUT=${OUT:-/root/autodl-tmp/srdiff_cum_anchor}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU0_KS=${GPU0_KS:-"128"}
GPU1_KS=${GPU1_KS:-"96 64"}
export DATA OUT

if [ -f "$OUT/AE_DONE" ]; then echo "[eval-anchor] AE_DONE 已存在, 跳过"; exit 0; fi
echo "[eval-anchor] 等待 ANCHOR_DONE ... $(date -Is)"
while [ ! -f "$OUT/ANCHOR_DONE" ]; do sleep 60; done
echo "[eval-anchor] 锚臂训练结束, 开始评测 $(date -Is)"

( CUDA_VISIBLE_DEVICES=0 bash tools/eval_k8sweep.sh $GPU0_KS ) > "$OUT/eval_gpu0.out" 2>&1 &
P0=$!
( CUDA_VISIBLE_DEVICES=1 bash tools/eval_k8sweep.sh $GPU1_KS ) > "$OUT/eval_gpu1.out" 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "[eval-anchor] 评测结束 rc=$R0/$R1 $(date -Is)"

$PY tools/summarize_fixedk.py --out_root "$OUT" --csv "$OUT/train_summary.csv" \
  --ks 64 96 128 > "$OUT/summarize.out" 2>&1
touch "$OUT/AE_DONE"
echo "[eval-anchor] AE_DONE $(date -Is)"
