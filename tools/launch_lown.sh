#!/usr/bin/env bash
# =============================================================================
# launch_lown.sh — 「固定 8 步 × 每步 n 个」的低 n 延伸：n=6/4/2（K=48/32/16）(2026-10-23)
# -----------------------------------------------------------------------------
# 动机（用户 2026-10-23 口径）: 10-21 固定 n 消融里 K=64/96/192 的 T=8 L1 都差不多
#   (0.113/0.112/0.109)，想知道**下限在哪** ⇒ 把每步 n 从 8 往小推: n=6/4/2（K=48/32/16）。
# 协议：与 10-21 的 fixed-k 臂**逐项同协议**（A 相 T=1 45ep/bs128/lr2e-4 从零 →
#   B 相 T=8 6ep/bs64/lr2e-4 热启动；`--step_plan fixed --block n`；无 QK-Norm，与
#   K=64/96/192 那批一致）。
# 分派：本机 2×A800 ⇒ 两条链并行（GPU0 = K=48；GPU1 = K=32 → K=16，约 6 h 墙钟）;
#   若只有 1 卡（NGPU=1）则顺序跑 48→32→16（约 9.1 h）。
# 产物: $OUT/stdattn_k{K}_t{1,8}; 标记 LOWN_DONE。
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/launch_lown.sh > $OUT/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/srdiff_lown}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPU0_ARMS=${GPU0_ARMS:-"48"}
GPU1_ARMS=${GPU1_ARMS:-"32 16"}
LOG=$OUT/lown.log
mkdir -p "$OUT"
if [ -f "$OUT/LOWN_DONE" ]; then echo "[lown] LOWN_DONE 已存在, 跳过"; exit 0; fi
: > "$LOG"
echo "[lown] ==== start $(date -Is) OUT=$OUT NGPU=$NGPU GPU0='$GPU0_ARMS' GPU1='$GPU1_ARMS'" >> "$LOG"

run_arm() {
  local gpu=$1 K=$2 n=$((K / 8))
  echo "[lown] GPU$gpu K=$K (n=$n) A+B start $(date -Is)" >> "$LOG"
  K="$K" GPU="$gpu" LR1=2e-4 LR2=2e-4 OUT="$OUT" DATA="$DATA" \
    bash tools/run_fixedk_arm.sh >> "$LOG" 2>&1
  echo "[lown] GPU$gpu K=$K (n=$n) rc=$? end $(date -Is)" >> "$LOG"
}

run_chain() {
  local gpu=$1; shift
  local K
  for K in "$@"; do run_arm "$gpu" "$K"; done
  echo "[lown] GPU$gpu chain done $(date -Is)" >> "$LOG"
}

if [ "$NGPU" -ge 2 ]; then
  run_chain 0 $GPU0_ARMS &
  P0=$!
  run_chain 1 $GPU1_ARMS &
  P1=$!
  wait $P0; R0=$?
  wait $P1; R1=$?
else
  run_chain 0 48 32 16
  R0=$?; R1=0
fi

touch "$OUT/LOWN_DONE"
echo "[lown] LOWN_DONE $(date -Is) rc=$R0/$R1" | tee -a "$LOG"
