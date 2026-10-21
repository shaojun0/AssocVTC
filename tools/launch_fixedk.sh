#!/usr/bin/env bash
# =============================================================================
# launch_fixedk.sh — 「固定 8 步 × 每步均匀 n 个 z_s」5 臂双卡分派（2026-10-21）
# -----------------------------------------------------------------------------
# 臂: K=64/96/128/192/256 ⇒ n=K/8=8/12/16/24/32; 全部 A+B 两相。
# 分派依据（单臂 A+B 墙钟估算, 来自 2026-10-19 同协议实测 + K 的算力趋势）:
#   K=64≈3.3h 96≈3.5h 128≈3.8h 192≈4.2h 256≈4.5h
#   ⇒ GPU0: 64, 96, 128 (≈10.6h) / GPU1: 192, 256 (≈8.7h)
# A 相 lr 与同 K 的 cumulative 臂对齐: K≤96 → 2e-4; K≥128 → 1e-4.
# 用法（服务器仓库根目录）:
#   nohup bash tools/launch_fixedk.sh > /root/autodl-tmp/srdiff_fixedk/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/srdiff_fixedk}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU0_ARMS=${GPU0_ARMS:-"64 96 128"}
GPU1_ARMS=${GPU1_ARMS:-"192 256"}
mkdir -p "$OUT"
CHAIN_LOG=$OUT/launch_chain.log

if [ -f "$OUT/ALL_DONE" ]; then
  echo "[launch] $OUT/ALL_DONE 已存在, 不重复启动"; exit 0
fi
echo "[launch] ===== start $(date -Is) GPU0='$GPU0_ARMS' GPU1='$GPU1_ARMS' ====="

run_chain() {
  local gpu=$1; shift
  local K L1
  for K in "$@"; do
    L1=2e-4; [ "$K" -ge 128 ] && L1=1e-4
    echo "=== GPU$gpu start K=$K LR1=$L1 $(date -Is) ===" >> "$CHAIN_LOG"
    K="$K" GPU="$gpu" LR1="$L1" LR2=2e-4 bash tools/run_fixedk_arm.sh >> "$CHAIN_LOG" 2>&1
    echo "=== GPU$gpu end   K=$K rc=$? $(date -Is) ===" >> "$CHAIN_LOG"
  done
  echo "=== GPU$gpu chain done $(date -Is) ===" >> "$CHAIN_LOG"
}

run_chain 0 $GPU0_ARMS &
P0=$!
run_chain 1 $GPU1_ARMS &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?

touch "$OUT/ALL_DONE"
echo "[launch] ===== all chains done $(date -Is) rc=$R0/$R1 ====="
