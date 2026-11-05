#!/usr/bin/env bash
# =============================================================================
# run_s3_zs448.sh — 448²/p32/960M 阶段3 变体：zs_dim 224 → 448（单卡 GPU0 微调）
# -----------------------------------------------------------------------------
# 目的：主线的 S3 相把 decoder 输入端投影到 224 维（每图 32×224 = 7,168 值）。
#       本变体只把 --zs_dim 改成 448（每图 32×448 = 14,336 值），其余完全沿用
#       S3 协议（3 epoch、等效 batch 64、lr 2e-4 cosine 8% warmup、从 B/ckpt_best
#       做 rank-448 SVD 热启动），用来回答：
#         - 每 token 448 维（值预算 ×2）能否把 448-S3 的 7,168/14,336 值档拉回
#           224 论文 26.08 dB 附近；
#         - 同值数下「模型 vs PCA」的优势在更高值数是否继续保持。
#
# 与主线隔离：
#   OUT 默认 /root/autodl-tmp/exp448/S3z448（不覆盖主线 S3 目录/日志）
#   日志  $OUT/logs/stage3z448.log；eval → /root/autodl-tmp/exp448/eval/stage3z448_stepwise.json
#   标记  /root/autodl-tmp/exp448/S3z448_DONE（训练+eval 都成功才写）
#
# 用法（远端）：
#   GPU=0 setsid nohup bash tools/run_s3_zs448.sh \
#       > /root/autodl-tmp/exp448/logs/s3z448_launch.out 2>&1 &
#   PLAN=1 bash tools/run_s3_zs448.sh        # 只打印将执行的命令
# 环境变量：
#   PY / DATA / EXP448（主线根目录）/ OUT / GPU / NUM_WORKERS / EP3 / ZS_DIM
#   MICRO=16 ACCUM=4（等效 batch 64；单卡 micro=16 是 960M/T=16 的实测可行值）
#   TIME_BUDGET_H=0（不设预算）；NO_EVAL=1 可跳过评测
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "[s3z448] ✗ 请在仓库内执行（tools/ 的上一级）" >&2; exit 2; }
ROOT=$PWD

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
EXP448=${EXP448:-/root/autodl-tmp/exp448}
OUT=${OUT:-$EXP448/S3z448}
GPU=${GPU:-0}
NUM_WORKERS=${NUM_WORKERS:-16}
EP3=${EP3:-3}
ZS_DIM=${ZS_DIM:-448}
MICRO=${MICRO:-16}
ACCUM=${ACCUM:-4}
LR=${LR:-2e-4}
MIN_LR=${MIN_LR:-1e-6}
WD=${WD:-0.05}
WARMUP=${WARMUP:-0.08}
CLIP=${CLIP:-1.0}
SEED=${SEED:-0}
AMP=${AMP:-1}
TIME_BUDGET_H=${TIME_BUDGET_H:-0}
NO_EVAL=${NO_EVAL:-0}
PLAN=${PLAN:-0}
INIT=${INIT:-$EXP448/B/ckpt_best.pt}

if [ ! -f "$INIT" ]; then echo "[s3z448] ✗ 缺少 B 相 ckpt: $INIT" >&2; exit 2; fi
mkdir -p "$OUT/logs"
LOG=$OUT/logs/stage3z448.log
EVAL_JSON=$EXP448/eval/stage3z448_stepwise.json
DEC_STEPS=$(seq -s, 1 16)
EFF=$((MICRO * ACCUM))

say() { echo "[s3z448] $*"; }

CMD=(
  "$PY" -u train_linattn.py
  --data_dir "$DATA" --output_dir "$OUT"
  --img_size 448 --patch 32 --num_specials 32 --special_init_std 0.02
  --enc_dim 1744 --enc_depth 13 --enc_heads 16 --enc_mlp_ratio 4.0
  --enc_dwconv 1 --enc_qk_norm 1 --enc_attn std --enc_std_qk_norm 0
  --dec_recurrent 1 --dec_steps "$DEC_STEPS" --step_plan fixed --block 2
  --dec_attn std --dec_linear_qk_norm 1 --dec_grad_ckpt 0 --freeze_encoder 0
  --dec_dim 2048 --dec_depth 7 --dec_heads 16 --dec_mlp_ratio 4.0 --pixel_hidden 1024
  --zs_dim "$ZS_DIM" --init_from "$INIT"
  --epochs "$EP3" --batch_size "$EFF" --accum_steps "$ACCUM"
  --lr "$LR" --min_lr "$MIN_LR" --weight_decay "$WD" --warmup_frac "$WARMUP"
  --grad_clip "$CLIP" --amp "$AMP" --seed "$SEED" --save_opt 0
  --num_workers "$NUM_WORKERS" --time_budget_h "$TIME_BUDGET_H"
)

if [ "$PLAN" = "1" ]; then
  say "OUT=$OUT  GPU=$GPU  zs_dim=$ZS_DIM  eff_batch=$EFF (micro $MICRO × accum $ACCUM)  epochs=$EP3"
  printf '[s3z448] CMD:'; printf ' %q' "${CMD[@]}"; printf '\n'
  say "eval: CKPT=$OUT/ckpt_best.pt OUT=$EXP448/eval GPU=$GPU NAME=stage3z448 bash tools/eval_448_960m.sh"
  exit 0
fi

say "================ S3-zs$ZS_DIM 变体（448²/p32/960M, T=16, 单卡 GPU$GPU）================"
say "start $(date -Is)  OUT=$OUT  init=$INIT  eff_batch=$EFF  epochs=$EP3"
say "log → $LOG"

export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
"${CMD[@]}" >> "$LOG" 2>&1
RC=$?
say "train rc=$RC end $(date -Is)"
if [ "$RC" -ne 0 ]; then say "✗ 训练失败（见 $LOG）"; exit "$RC"; fi

if [ "$NO_EVAL" != "1" ]; then
  say "eval 开始 $(date -Is)"
  CKPT=$OUT/ckpt_best.pt OUT=$EXP448/eval GPU=$GPU NUM_IMAGES=0 NAME=stage3z448 BS=16 NUM_WORKERS=8 \
    DATA=$DATA bash tools/eval_448_960m.sh >> "$OUT/logs/eval_stage3z448.log" 2>&1
  ERC=$?
  say "eval rc=$ERC end $(date -Is) → $EVAL_JSON"
  [ "$ERC" -ne 0 ] && { say "✗ eval 失败（见 $OUT/logs/eval_stage3z448.log）"; exit "$ERC"; }
fi

touch "$EXP448/S3z448_DONE"
say "✓ S3z448_DONE $(date -Is)"
