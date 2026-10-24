#!/usr/bin/env bash
# =============================================================================
# launch_scale.sh — 「容量阶梯」4 档分派到 2×A800（2026-10-24）
# -----------------------------------------------------------------------------
# 4 档（enc/dec 各 60/120/240/360M，深度钉死 13/7，宽度 scaling）：
#   GPU0 = 360M（最贵，先跑，尽早暴露大模型问题）→ 60M
#   GPU1 = 240M → 120M
# 两链并行，墙钟 ≈ max(360+60, 240+120) —— 360M 档占绝对主导（预估 ~60 h，
# 见 bench_scale.sh 的实测 s/iter；若与预估差太多会先报告再定）。
# 产物: $OUT/stdattn_m{M}_t{1,16}/；标记 $OUT/SCALE_DONE；日志 $OUT/scale.log
# 用法（服务器仓库根目录）:
#   mkdir -p $OUT
#   nohup setsid bash tools/launch_scale.sh > $OUT/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
source tools/scale_arms.sh

OUT=${OUT:-/root/autodl-tmp/srdiff_scale}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPU0_ARMS=${GPU0_ARMS:-"360 60"}
GPU1_ARMS=${GPU1_ARMS:-"240 120"}
LOG=$OUT/scale.log

mkdir -p "$OUT"
if [ -f "$OUT/SCALE_DONE" ]; then echo "[scale] SCALE_DONE 已存在, 跳过"; exit 0; fi
: > "$LOG"
echo "[scale] ==== start $(date -Is) OUT=$OUT NGPU=$NGPU" >> "$LOG"
echo "[scale] GPU0_ARMS='$GPU0_ARMS' GPU1_ARMS='$GPU1_ARMS'" >> "$LOG"

run_chain() {
  local gpu=$1; shift
  local m rc=0
  for m in "$@"; do
    echo "[scale] GPU$gpu M=$m chain start $(date -Is)" >> "$LOG"
    M="$m" GPU="$gpu" OUT="$OUT" DATA="$DATA" bash tools/run_scale_arm.sh >> "$LOG" 2>&1 || rc=$?
    echo "[scale] GPU$gpu M=$m chain rc=$rc end $(date -Is)" >> "$LOG"
  done
  echo "[scale] GPU$gpu chain done (rc=$rc) $(date -Is)" >> "$LOG"
  return 0
}

if [ "$NGPU" -ge 2 ]; then
  run_chain 0 $GPU0_ARMS & P0=$!
  run_chain 1 $GPU1_ARMS & P1=$!
  wait $P0; R0=$?
  wait $P1; R1=$?
else
  run_chain 0 360 240 120 60; R0=$?; R1=0
fi

touch "$OUT/SCALE_DONE"
echo "[scale] SCALE_DONE $(date -Is) chain_rc=$R0/$R1" | tee -a "$LOG"
