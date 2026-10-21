#!/usr/bin/env bash
# =============================================================================
# eval_fixedk.sh — 「固定 8 步 × 每步均匀 n 个 z_s」5 臂全量评测（2026-10-21）
# -----------------------------------------------------------------------------
# 流程: 等训练 ALL_DONE → 双卡并行跑 eval_k8sweep.sh（每臂: T=8 末步 + 逐步 + T=1）
#       → 汇总 train_summary.csv → touch EVAL_ALL_DONE
# 分派: GPU0 = 64 128 256 / GPU1 = 96 192（每臂 ~8-10 min; 两卡均衡）
# 用法（服务器仓库根目录）:
#   nohup bash tools/eval_fixedk.sh > /root/autodl-tmp/srdiff_fixedk/eval_launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
OUT=${OUT:-/root/autodl-tmp/srdiff_fixedk}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU0_KS=${GPU0_KS:-"64 128 256"}
GPU1_KS=${GPU1_KS:-"96 192"}
export DATA OUT

if [ -f "$OUT/EVAL_ALL_DONE" ]; then
  echo "[eval] $OUT/EVAL_ALL_DONE 已存在, 跳过"; exit 0
fi
echo "[eval] 等待训练 ALL_DONE 与 K=256 重跑 K256_REDO_DONE ... $(date -Is)"
while [ ! -f "$OUT/ALL_DONE" ] || [ ! -f "$OUT/K256_REDO_DONE" ]; do sleep 60; done
echo "[eval] 训练已结束, 开始评测 $(date -Is)"

( CUDA_VISIBLE_DEVICES=0 bash tools/eval_k8sweep.sh $GPU0_KS ) > "$OUT/eval_gpu0.out" 2>&1 &
P0=$!
( CUDA_VISIBLE_DEVICES=1 bash tools/eval_k8sweep.sh $GPU1_KS ) > "$OUT/eval_gpu1.out" 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "[eval] 评测结束 rc=$R0/$R1 $(date -Is)"

$PY tools/summarize_fixedk.py --out_root "$OUT" --csv "$OUT/train_summary.csv" \
  > "$OUT/summarize.out" 2>&1
echo "[eval] train_summary.csv $(date -Is)"
touch "$OUT/EVAL_ALL_DONE"
echo "[eval] EVAL_ALL_DONE $(date -Is)"
