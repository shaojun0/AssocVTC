#!/usr/bin/env bash
# =============================================================================
# launch_cumB_paired.sh — **配对** B 相：从 fixed 臂的同一个 A 相 ckpt 出发，只重跑
# cumulative 窗口的 B 相（2026-10-21）
# -----------------------------------------------------------------------------
# 为什么这样更干净: A 相（T=1）与读窗口计划**无关**（单步读全 K）⇒ fixed 臂与
# cumulative 臂的 A 相在配置上完全相同。因此把 A 相重跑一遍既贵又引入 basin 差异;
# 直接拿 fixed 臂的 A 相 ckpt 热启动、只换 B 相的读窗口计划, 就得到**同一初始化**下
# 的「均匀 vs 递增」配对比较（唯一自变量 = 读窗口形状）。
# 协议: 与 fixed 臂 B 相逐项相同（6 ep / bs64 / lr 2e-4 / 同 seed / cumulative 窗口）。
# 臂: K=64/96/128（每个 K 都从 $SRC/stdattn_k<K>_t1/ckpt_best.pt 出发）。
# 分派: GPU0 = 128（等 fixed K=128 B 相退出）; GPU1 = 96 → 64。
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/launch_cumB_paired.sh > /root/autodl-tmp/srdiff_cum_paired/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

SRC=${SRC:-/root/autodl-tmp/srdiff_fixedk}          # fixed 臂（提供 A 相 ckpt）
OUT=${OUT:-/root/autodl-tmp/srdiff_cum_paired}      # 配对 cumulative B 相产物
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
LOG=$OUT/paired.log
mkdir -p "$OUT"

if [ -f "$OUT/PAIRED_DONE" ]; then echo "[paired] PAIRED_DONE 已存在, 跳过"; exit 0; fi
: > "$LOG"
echo "[paired] ===== start $(date -Is) SRC=$SRC OUT=$OUT =====" >> "$LOG"

# 把 fixed 臂的 A 相 ckpt 以 run_k8sweep_arm.sh 期望的路径暴露（SKIP_STAGE1=1 只读它）
for K in 64 96 128; do
  mkdir -p "$OUT/stdattn_k${K}_t1"
  ln -sfn "$SRC/stdattn_k${K}_t1/ckpt_best.pt" "$OUT/stdattn_k${K}_t1/ckpt_best.pt"
done

wait_done() {
  local log=$1
  while ! grep -q "\[done\]" "$log" 2>/dev/null; do sleep 60; done
}

run_b() {
  local gpu=$1 k=$2
  echo "== paired GPU$gpu K=$k start $(date -Is)" >> "$LOG"
  K="$k" GPU="$gpu" SKIP_STAGE1=1 LR2=2e-4 OUT="$OUT" DATA="$DATA" \
    bash tools/run_k8sweep_arm.sh >> "$LOG" 2>&1
  echo "== paired GPU$gpu K=$k rc=$? end $(date -Is)" >> "$LOG"
}

( wait_done "$SRC/k128_stage2.log"; run_b 0 128; echo "== paired GPU0 chain done $(date -Is)" >> "$LOG" ) &
P0=$!
( run_b 1 96; run_b 1 64; echo "== paired GPU1 chain done $(date -Is)" >> "$LOG" ) &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
touch "$OUT/PAIRED_DONE"
echo "PAIRED_DONE $(date -Is) rc=$R0/$R1" | tee -a "$LOG"
