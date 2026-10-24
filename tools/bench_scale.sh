#!/usr/bin/env bash
# =============================================================================
# bench_scale.sh — 「容量阶梯」跑前基准：A 相(T=1,bs128) / B 相(T=16,bs64,accum)
#   的 s/iter、img/s、vram 实测，用来定 accum 与估算整轮墙钟（2026-10-24）
# -----------------------------------------------------------------------------
# 用法（服务器仓库根目录）:
#   BLOCKS="60 120 240 360" GPU=0 bash tools/bench_scale.sh
# 环境变量: BLOCKS（档位列表）/ GPU / STEPS（默认 20）/ OUT（默认 /root/autodl-tmp/bench_scale）
# 产物: $OUT/bench_m{60,120,240,360}_{A,B}.log + 汇总 $OUT/bench_summary.txt
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
source tools/scale_arms.sh

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/bench_scale}
GPU=${GPU:-0}
STEPS=${STEPS:-20}
BLOCKS=${BLOCKS:-"60 120 240 360"}
mkdir -p "$OUT"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPU
SUMMARY=$OUT/bench_summary.txt
: > "$SUMMARY"

DEC_STEPS=$(seq -s, 1 "$SCALE_T")

for M in $BLOCKS; do
  scale_arm_cfg "$M"
  # ── A 相: 单步 T=1 / bs128 ──
  AOUT=$OUT/a_m${M}
  A_LOG=$OUT/bench_m${M}_A.log
  echo "[bench] m${M} A相 enc=${ENC_DIM}x13 h${ENC_HEADS} dec=${DEC_DIM}x7 h${DEC_HEADS} accum=$SCALE_ACCUM_A" | tee -a "$SUMMARY"
  $PY -u train_linattn.py \
    --data_dir "$DATA" --output_dir "$AOUT" \
    --enc_attn std --num_specials "$SCALE_K" --batch_size 128 --epochs 1 --lr "$SCALE_LR1" \
    --enc_dim "$ENC_DIM" --enc_heads "$ENC_HEADS" --dec_dim "$DEC_DIM" --dec_heads "$DEC_HEADS" \
    --dec_recurrent 0 --num_workers 8 --seed 0 \
    --max_steps "$STEPS" --log_interval 5 --sample_interval 0 --val_batches 2 \
    --accum_steps "$SCALE_ACCUM_A" --save_opt 0 >> "$A_LOG" 2>&1
  RC=$?
  echo "[bench] m${M} A相 rc=$RC" | tee -a "$SUMMARY"
  grep -E "^\[params\]|^\[ *[0-9.]+m\] step|OutOfMemory|Error" "$A_LOG" | tail -2 | tee -a "$SUMMARY"
  # ── B 相: 循环 T=16 / fixed n=2 / 等效 bs64 / accum ──
  BOUT=$OUT/b_m${M}
  B_LOG=$OUT/bench_m${M}_B.log
  echo "[bench] m${M} B相 T=$SCALE_T block=$SCALE_BLOCK accum=$SCALE_ACCUM_B" | tee -a "$SUMMARY"
  $PY -u train_linattn.py \
    --data_dir "$DATA" --output_dir "$BOUT" \
    --enc_attn std --num_specials "$SCALE_K" --batch_size 64 --epochs 1 --lr "$SCALE_LR2" \
    --enc_dim "$ENC_DIM" --enc_heads "$ENC_HEADS" --dec_dim "$DEC_DIM" --dec_heads "$DEC_HEADS" \
    --dec_recurrent 1 --step_plan fixed --block "$SCALE_BLOCK" --dec_steps "$DEC_STEPS" \
    --num_workers 8 --seed 0 --max_steps "$STEPS" --log_interval 5 --sample_interval 0 \
    --val_batches 2 --accum_steps "$SCALE_ACCUM_B" --save_opt 0 >> "$B_LOG" 2>&1
  RC=$?
  echo "[bench] m${M} B相 rc=$RC" | tee -a "$SUMMARY"
  grep -E "^\[params\]|^\[ *[0-9.]+m\] step|OutOfMemory|Error" "$B_LOG" | tail -2 | tee -a "$SUMMARY"
  echo "----" | tee -a "$SUMMARY"
done
echo "[bench] done $(date -Is) → $SUMMARY"
cat "$SUMMARY"
