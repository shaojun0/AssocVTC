#!/usr/bin/env bash
# =============================================================================
# launch_lown.sh — 「固定 8 步 × 每步 n 个」的低 n 延伸：n=6/4/2（K=48/32/16）(2026-10-23)
# -----------------------------------------------------------------------------
# 动机（用户 2026-10-23 口径）: 10-21 固定 n 消融里 K=64/96/192 的 T=8 L1 都差不多
#   (0.113/0.112/0.109)，想知道**下限在哪** ⇒ 把每步 n 从 8 往小推: n=6/4/2（K=48/32/16）。
# 协议：与 10-21 的 fixed-k 臂**逐项同协议**（A 相 T=1 45ep/bs128/lr2e-4 从零 →
#   B 相 T=8 6ep/bs64/lr2e-4 热启动；`--step_plan fixed --block n`；无 QK-Norm，与
#   K=64/96/192 那批一致）。每臂 A+B 串行，单卡顺序 K=48 → K=32 → K=16。
# 产物: $OUT/stdattn_k{K}_t{1,8}; 标记 LOWN_DONE。
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/launch_lown.sh > $OUT/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/srdiff_lown}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU=${GPU:-0}
LOG=$OUT/lown.log
mkdir -p "$OUT"
if [ -f "$OUT/LOWN_DONE" ]; then echo "[lown] LOWN_DONE 已存在, 跳过"; exit 0; fi
: > "$LOG"
echo "[lown] ==== start $(date -Is) OUT=$OUT" >> "$LOG"

run_arm() {
  local K=$1 n=$2
  echo "[lown] K=$K (n=$n) A+B start $(date -Is)" >> "$LOG"
  K="$K" GPU="$GPU" LR1=2e-4 LR2=2e-4 OUT="$OUT" DATA="$DATA" \
    bash tools/run_fixedk_arm.sh >> "$LOG" 2>&1
  echo "[lown] K=$K (n=$n) rc=$? end $(date -Is)" >> "$LOG"
}

run_arm 48 6
run_arm 32 4
run_arm 16 2

touch "$OUT/LOWN_DONE"
echo "[lown] LOWN_DONE $(date -Is)" | tee -a "$LOG"
