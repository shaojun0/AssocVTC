#!/usr/bin/env bash
# =============================================================================
# launch_k8sweep.sh — 2 卡并行跑「固定 8 步 × K token」K-sweep 全部臂
# -----------------------------------------------------------------------------
#   卡 0: K=32 → 16 → 8
#   卡 1: K=24 → 12 → 64（64 是同协议参照点, 对齐论文/历史的 64-token 预算）
# 每臂 = A 相 T=1 (45ep) + B 相 T=8 (6ep)，见 tools/run_k8sweep_arm.sh。
#
# 用法:  bash tools/launch_k8sweep.sh
#        tail -f /root/autodl-tmp/srdiff_k8sweep/launch.log
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-/root/autodl-tmp/srdiff_k8sweep}
mkdir -p "$OUT"
GPU0_ARMS=${GPU0_ARMS:-"32 16 8"}
GPU1_ARMS=${GPU1_ARMS:-"24 12 64"}

chain() {   # chain <gpu> <arm列表>
  local gpu=$1; shift
  for k in $@; do
    K=$k GPU=$gpu bash tools/run_k8sweep_arm.sh
    echo "[chain gpu$gpu] K=$k rc=$? $(date -Is)"
  done
  echo "[chain gpu$gpu] ALL DONE $(date -Is)"
}

echo "[launch] gpu0: $GPU0_ARMS | gpu1: $GPU1_ARMS | start=$(date -Is)"
chain 0 $GPU0_ARMS > "$OUT/chain_gpu0.log" 2>&1 &
PID0=$!
chain 1 $GPU1_ARMS > "$OUT/chain_gpu1.log" 2>&1 &
PID1=$!
wait $PID0; wait $PID1
echo "[launch] ALL DONE $(date -Is)" | tee "$OUT/QUEUE_DONE"
