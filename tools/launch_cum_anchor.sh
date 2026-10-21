#!/usr/bin/env bash
# =============================================================================
# launch_cum_anchor.sh — 2026-10-21 新机器上的 **cumulative 同机对照臂**（K=64/96/128）
# -----------------------------------------------------------------------------
# 动机: 2026-10-19 的 9 臂（cumulative）跑在旧机器/旧 torch 上; 本次 fixed 臂在新机器
#   (torch 2.12.1+cu130) 上。为把「fixed vs cumulative」的差值归因到**读窗口形状**而不是
#   机器/torch 数值差异, 必须在新机器上把同 K 的 cumulative 臂重跑一遍。
#   协议与旧 9 臂**逐项相同**: A 相 45ep/bs128（K=128 用 lr=1e-4, 其余 2e-4）,
#   B 相 6ep/bs64、lr 2e-4、cumulative 窗口（scaled_square_cumsum）。
# 分派: GPU0 = 128; GPU1 = 96 → 64; 各自等主实验占用该卡的臂退出后再开始
#   （GPU0 等 K=128 固定臂; GPU1 等 K=256 固定臂）。
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/launch_cum_anchor.sh > /root/autodl-tmp/srdiff_cum_anchor/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/srdiff_cum_anchor}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
mkdir -p "$OUT"
LOG=$OUT/anchor.log

if [ -f "$OUT/ANCHOR_DONE" ]; then
  echo "[anchor] ANCHOR_DONE 已存在, 跳过"; exit 0
fi
echo "[anchor] ===== start $(date -Is) OUT=$OUT =====" >> "$LOG"

wait_done() {
  local log=$1
  while ! grep -q "\[done\]" "$log" 2>/dev/null; do sleep 60; done
}

run_chain() {
  local gpu=$1 gate_log=$2; shift 2
  local item k l1
  wait_done "$gate_log"
  for item in "$@"; do
    k=${item%%:*}; l1=${item##*:}
    echo "== anchor GPU$gpu K=$k lr1=$l1 start $(date -Is)" >> "$LOG"
    K="$k" GPU="$gpu" LR1="$l1" LR2=2e-4 OUT="$OUT" DATA="$DATA" \
      bash tools/run_k8sweep_arm.sh >> "$LOG" 2>&1
    echo "== anchor GPU$gpu K=$k rc=$? end $(date -Is)" >> "$LOG"
  done
  echo "== anchor GPU$gpu chain done $(date -Is)" >> "$LOG"
}

run_chain 0 /root/autodl-tmp/srdiff_fixedk/k128_stage2.log 128:1e-4 &
P0=$!
run_chain 1 /root/autodl-tmp/srdiff_fixedk/k256_stage2.log 96:2e-4 64:2e-4 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
touch "$OUT/ANCHOR_DONE"
echo "ANCHOR_DONE $(date -Is) rc=$R0/$R1" | tee -a "$LOG"
