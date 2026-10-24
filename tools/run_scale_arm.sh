#!/usr/bin/env bash
# =============================================================================
# run_scale_arm.sh — 「容量阶梯」单档全协议训练（A 相 T=1 45ep → B 相 T=16 6ep）(2026-10-24)
# -----------------------------------------------------------------------------
# 协议与 2026-10-23 的 63M n=2 × T=16 基线臂**逐项一致**（唯一自变量 = enc/dec 宽度）:
#   A 相: T=1（单步读满 K=32 个 z_s）, 45 ep, bs128, lr2e-4, 从零;
#   B 相: T=16, `--step_plan fixed --block 2`, 6 ep, 等效 bs64（accum=2 → micro32）,
#         lr2e-4, 从 A 相 ckpt_best 热启动。
# 变的是: enc_dim/enc_heads/dec_dim/dec_heads（见 tools/scale_arms.sh, 深度钉死 13/7）。
# 省盘: `--save_opt 0`（不为 resume 存 AdamW 状态）→ 360M 档 ckpt 从 11.6G 降到 ~2.9G;
#       跑完删 ckpt_last（评测/条件臂只用 ckpt_best）。
# 用法（服务器仓库根目录）:
#   M=240 GPU=0 bash tools/run_scale_arm.sh
# 环境变量: M（必填 60/120/240/360）/ GPU / DATA / OUT / SKIP_STAGE1 / SAVE_OPT
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
source tools/scale_arms.sh

PY=${PY:-/root/miniconda3/bin/python}
M=${M:?请设置 M（60/120/240/360）}
GPU=${GPU:-0}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/srdiff_scale}
SKIP_STAGE1=${SKIP_STAGE1:-0}
SAVE_OPT=${SAVE_OPT:-0}
scale_arm_cfg "$M"

A1=$OUT/stdattn_${SCALE_TAG}_t1
A2=$OUT/stdattn_${SCALE_TAG}_t${SCALE_T}
LOG1=$OUT/m${M}_stage1.log
LOG2=$OUT/m${M}_stage2.log
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

DEC_STEPS=$(seq -s, 1 "$SCALE_T")
echo "[scale] M=$M enc=${ENC_DIM}x13 h${ENC_HEADS} (${ENC_PARAM_M}M) dec=${DEC_DIM}x7 h${DEC_HEADS} (${DEC_PARAM_M}M) GPU=$GPU start=$(date -Is)"

if [ "$SKIP_STAGE1" != "1" ]; then
  echo "[scale] M=$M A相 T=1 45ep accum=$SCALE_ACCUM_A → $A1"
  $PY -u train_linattn.py \
    --data_dir "$DATA" --output_dir "$A1" \
    --enc_attn std --num_specials "$SCALE_K" --batch_size 128 --epochs "$SCALE_EP1" \
    --lr "$SCALE_LR1" --enc_dim "$ENC_DIM" --enc_heads "$ENC_HEADS" \
    --dec_dim "$DEC_DIM" --dec_heads "$DEC_HEADS" --dec_recurrent 0 \
    --time_budget_h 96 --num_workers 16 --seed 0 \
    --accum_steps "$SCALE_ACCUM_A" --save_opt "$SAVE_OPT" >> "$LOG1" 2>&1
  RC1=$?
  echo "[scale] M=$M A相 rc=$RC1 end=$(date -Is)"
  if [ $RC1 -ne 0 ] || [ ! -f "$A1/ckpt_best.pt" ]; then
    echo "[scale] M=$M FAILED(stage1)"; exit 1
  fi
  rm -f "$A1/ckpt_last.pt"        # 省盘（评测只用 best）
fi

echo "[scale] M=$M B相 T=$SCALE_T block=$SCALE_BLOCK accum=$SCALE_ACCUM_B → $A2"
$PY -u train_linattn.py \
  --data_dir "$DATA" --output_dir "$A2" \
  --enc_attn std --num_specials "$SCALE_K" --batch_size 64 --epochs "$SCALE_EP2" \
  --lr "$SCALE_LR2" --enc_dim "$ENC_DIM" --enc_heads "$ENC_HEADS" \
  --dec_dim "$DEC_DIM" --dec_heads "$DEC_HEADS" \
  --dec_recurrent 1 --step_plan fixed --block "$SCALE_BLOCK" --dec_steps "$DEC_STEPS" \
  --init_from "$A1/ckpt_best.pt" --time_budget_h 96 --num_workers 16 --seed 0 \
  --accum_steps "$SCALE_ACCUM_B" --save_opt "$SAVE_OPT" >> "$LOG2" 2>&1
RC2=$?
echo "[scale] M=$M B相 rc=$RC2 end=$(date -Is)"
if [ $RC2 -eq 0 ] && [ -f "$A2/ckpt_best.pt" ]; then
  rm -f "$A2/ckpt_last.pt"
  echo "[scale] M=$M DONE"
else
  echo "[scale] M=$M FAILED(stage2)"; exit 2
fi
