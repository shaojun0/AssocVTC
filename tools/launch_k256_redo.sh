#!/usr/bin/env bash
# =============================================================================
# launch_k256_redo.sh — K=256 固定臂 A 相 lr=1e-4 停滞后的重跑（2026-10-21）
# -----------------------------------------------------------------------------
# 背景: K=256 在 lr=1e-4 下 A 相（T=1）15 epoch 只到 22.5 dB（143 次 grad skip,
#   train loss 卡在 ~0.21）, 与 K=192 的 1e-4 成功（A 相 32.51 dB）形成对比。
#   本机 2e-4 在 K=64/96 上表现最好（A 相 31.35 / 31.82 dB 且零 skip）⇒ 用 2e-4 重跑,
#   1e-4 的停滞证据保留在 k256_stage1_lr1e-4_stalled.log + stdattn_k256_t1_lr1e-4_stalled/。
# 产物: 与主实验同目录（$OUT/stdattn_k256_t1|t8），结束时 touch $OUT/K256_REDO_DONE。
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/launch_k256_redo.sh > $OUT/k256_redo.log 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/srdiff_fixedk}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU=${GPU:-1}
LOG=$OUT/k256_redo.log
mkdir -p "$OUT"
LR1=${LR1:-1e-4}
echo "== K=256 redo (A lr=$LR1) start $(date -Is) GPU=$GPU" >> "$LOG"
K=256 GPU="$GPU" LR1="$LR1" LR2=2e-4 OUT="$OUT" DATA="$DATA" \
  bash tools/run_fixedk_arm.sh >> "$LOG" 2>&1
RC=$?
echo "== K=256 redo rc=$RC end $(date -Is)" >> "$LOG"
# 只有真产出 B 相 ckpt 才落完成标记（防止中途 kill 让评测提前放行）
if [ $RC -eq 0 ] && [ -f "$OUT/stdattn_k256_t8/ckpt_best.pt" ]; then
  touch "$OUT/K256_REDO_DONE"
else
  echo "== K=256 redo 未产出 ckpt, 不落完成标记" >> "$LOG"
fi
echo "K256 redo finished $(date -Is) rc=$RC" | tee -a "$LOG"
