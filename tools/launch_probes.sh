#!/usr/bin/env bash
# =============================================================================
# launch_probes.sh — K=128 A 相「差盆地」原因探针链（2026-10-22）
# -----------------------------------------------------------------------------
# 每个探针 16 epoch（~0.85 h），与正式 A 相 lr 调度同分布。候选稳定化杠杆:
#   1) qknorm   : std 编码器加 QK-Norm（注意力 logits 归一 → 抑制 token 同质化）
#   2) warmup25 : warmup 0.08→0.25（warmup 结束时的 grad 尖峰是差盆地的起点）
#   3) lr5e-5   : 峰值 lr 减半（差盆地 gn 持续偏高 → 降学习率噪声）
#   4) sinit0p1 : specials 初始化 std 0.02→0.1（K 个 special 无位置编码、初始化近乎
#                 相同 = 排列对称 ⇒ 加大初始化差异，直接检验「对称坍缩」假说）
#   5) seed1    : 换 seed（检验差盆地是随机/混沌还是系统性问题）
# 结束标记 PROBES_DONE。用法（服务器仓库根目录）:
#   nohup setsid bash tools/launch_probes.sh > /root/autodl-tmp/probes/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/probes}
K=${K:-128}
mkdir -p "$OUT"
if [ -f "$OUT/PROBES_DONE" ]; then echo "[probes] PROBES_DONE 已存在, 跳过"; exit 0; fi
: > "$OUT/probe_chain.log"

run() {
  local name=$1; shift
  NAME="$name" K="$K" OUT="$OUT" PROBE_ARGS="$*" bash tools/run_probe.sh
}

run k${K}_qknorm      --enc_std_qk_norm 1
run k${K}_warmup25    --warmup_frac 0.25
run k${K}_lr5e5       --lr 5e-5
run k${K}_sinit0p1    --special_init_std 0.1
run k${K}_seed1       --seed 1

touch "$OUT/PROBES_DONE"
echo "PROBES_DONE $(date -Is)" | tee -a "$OUT/probe_chain.log"
