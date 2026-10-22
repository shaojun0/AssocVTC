#!/usr/bin/env bash
# =============================================================================
# run_probe.sh — A 相稳定性探针（2026-10-22，K=128/256「差盆地」诊断）
# -----------------------------------------------------------------------------
# 设计: 与正式 A 相**完全相同的超参和 lr 调度**（--epochs 45 ⇒ total_steps=42750、
#   warmup_frac 0.08），只用 --max_steps 截断到前 16 epoch（15200 步）⇒ 探针曲线在
#   前 16 ep 与正式运行「同分布」，可与 2026-10-21 的 K=128/K=256 失败曲线逐点对照
#   （失败曲线 ep16 ≈ 22.4/22.5 dB；好盆地 K=192 ep15 ≈ 27.45 dB）。
# 环境变量: NAME（必填）/ K / GPU / MAX_STEPS / PROBE_ARGS（附加 CLI 覆盖, 如 "--lr 5e-5"）
# 用法: NAME=k128_lr5e5 K=128 PROBE_ARGS="--lr 5e-5" bash tools/run_probe.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
NAME=${NAME:?请设置 NAME}
K=${K:-128}
GPU=${GPU:-0}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/probes}
MAX_STEPS=${MAX_STEPS:-15200}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

EXTRA=""
if [ "$K" -gt 196 ]; then EXTRA="--allow_k_gt_n 1"; fi

echo "[probe] $NAME K=$K max_steps=$MAX_STEPS args='${PROBE_ARGS:-}' start=$(date -Is)" >> "$OUT/probe_chain.log"
$PY -u train_linattn.py \
  --data_dir "$DATA" --output_dir "$OUT/$NAME" \
  --enc_attn std --num_specials "$K" --batch_size 128 --epochs 45 \
  --dec_recurrent 0 --max_steps "$MAX_STEPS" --num_workers 16 $EXTRA \
  ${PROBE_ARGS:-} > "$OUT/$NAME.log" 2>&1
RC=$?
rm -f "$OUT/$NAME/ckpt_last.pt"
LAST=$(grep -E "\[val\]|\[done\]" "$OUT/$NAME.log" | tail -1)
echo "[probe] $NAME rc=$RC | $LAST" >> "$OUT/probe_chain.log"
echo "[probe] $NAME rc=$RC end=$(date -Is)" >> "$OUT/probe_chain.log"
