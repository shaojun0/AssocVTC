#!/usr/bin/env bash
# =============================================================================
# launch_n2bigstep.sh — 「固定每步 n=2 × T=16/24/32」（K=32/48/64）(2026-10-23)
# -----------------------------------------------------------------------------
# 用户口径（2026-10-23）: n/step=2, 分别跑 16 / 24 / 32 个解码步
#   ⇒ 总 token 预算 K = 2·T = 32 / 48 / 64。
# 与 10-23 的 lown 臂（固定 8 步 × n=2/4/6, K=16/32/48）合起来回答两个问题:
#   (a) 同一 n=2 下**加步数**（T=8→16→24→32）质量怎么走;
#   (b) 同 K 下「少 token × 多步」 vs 「多 token × 少步」（K=32: n=4/T=8 vs n=2/T=16;
#       K=48: n=6/T=8 vs n=2/T=24; K=64: n=8/T=8 vs n=2/T=32）。
#
# A 相**不重跑**: A 相与读窗口 plan 无关（单步读满全部 K 个 z_s）。本机
#   $A_DIR/stdattn_k{32,48,64}_t1 已有 10-19 的 A 相 ckpt_best; 其中 K=32/48 与
#   10-23 lown 重跑的 A 相**逐位一致**（28.728 / 29.608 dB, 见 REPORT_lown_sweep.md）
#   ⇒ 口径可信。K=64 的 A 相 31.03 dB（10-21 fixed-k 那台机器上同样协议跑到 31.35,
#   差 0.33 dB 属跨机噪声量级 —— 报告里如实标注, 不做跨机配对结论）。
#
# B 相: T 步 / `--step_plan fixed --block 2` / 6 ep / lr 2e-4 / 从 A 相 ckpt_best 热启动,
#   与 10-21/10-23 的 fixed 臂逐项同协议。
# 显存: T=32 在 bs64 直接反传 OOM（实测 max_allocated 74.4G / 80.7G 峰值 ⇒ backward
#   再要 1.15G 就炸）⇒ **三个臂统一** `--accum_steps 2`（micro 32 × 2 累积 =
#   等效批量 64; 优化步数 1901×6=11406、lr 曲线、val 张数都与原协议逐位相同,
#   见 train_linattn.py --accum_steps 的注释; 统一用累积 = 三臂执行路径也一致）。
#   实测 vram: T=16≈22G / T=24≈35G / T=32≈41G（accum=2）。
#
# 分派（2×A800 / A800-80G）: GPU0 = T=32（约 4.9 h）;
#                            GPU1 = T=16（约 2.5 h）→ T=24（约 3.4 h）。墙钟约 6 h。
# 产物: $OUT/stdattn_k{K}_t{T}/（B 相 ckpt + train_log.jsonl）;
#       $OUT/stdattn_k{K}_t1/ckpt_best.pt 是指向 A_DIR 的**软链**（只供评测/溯源自洽）。
# 标记: $OUT/N2BIGSTEP_DONE。日志: $OUT/n2bigstep.log, $OUT/k{K}_stage2.log
# 用法（服务器仓库根目录）:
#   mkdir -p $OUT
#   nohup setsid bash tools/launch_n2bigstep.sh > $OUT/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/srdiff_n2bigstep}
A_DIR=${A_DIR:-/root/autodl-tmp/srdiff_k8sweep}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
ACCUM=${ACCUM:-2}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPU0_ARMS=${GPU0_ARMS:-"32"}      # 只写 T; K=2T, block=2
GPU1_ARMS=${GPU1_ARMS:-"16 24"}
LOG=$OUT/n2bigstep.log

mkdir -p "$OUT"
if [ -f "$OUT/N2BIGSTEP_DONE" ]; then
  echo "[n2] N2BIGSTEP_DONE 已存在, 跳过"; exit 0
fi
: > "$LOG"
echo "[n2] ==== start $(date -Is) OUT=$OUT A_DIR=$A_DIR NGPU=$NGPU ACCUM=$ACCUM" >> "$LOG"
echo "[n2] GPU0_ARMS='$GPU0_ARMS' GPU1_ARMS='$GPU1_ARMS'" >> "$LOG"

run_arm() {
  local gpu=$1 T=$2 K=$((2 * T))
  local A1="$OUT/stdattn_k${K}_t1" A2="$OUT/stdattn_k${K}_t${T}"
  mkdir -p "$A1"
  ln -sfn "$A_DIR/stdattn_k${K}_t1/ckpt_best.pt" "$A1/ckpt_best.pt"
  if [ -f "$A2/ckpt_best.pt" ]; then
    echo "[n2] GPU$gpu K=$K T=$T 已有 B 相 ckpt_best, 跳过 $(date -Is)" >> "$LOG"
    return 0
  fi
  if [ ! -e "$A1/ckpt_best.pt" ]; then
    echo "[n2] K=$K 缺 A 相 ckpt: $A_DIR/stdattn_k${K}_t1/ckpt_best.pt" >> "$LOG"
    return 1
  fi
  echo "[n2] GPU$gpu K=$K T=$T (block=2) B相 start $(date -Is)" >> "$LOG"
  K="$K" T="$T" GPU="$gpu" SKIP_STAGE1=1 OUT="$OUT" DATA="$DATA" LR1=2e-4 LR2=2e-4 \
    EXTRA_ARGS="--accum_steps $ACCUM" bash tools/run_fixedk_arm.sh >> "$LOG" 2>&1
  local rc=$?
  echo "[n2] GPU$gpu K=$K T=$T B相 rc=$rc end $(date -Is)" >> "$LOG"
  return $rc
}

run_chain() {
  local gpu=$1; shift
  local T rc=0
  for T in "$@"; do run_arm "$gpu" "$T" || rc=$?; done
  echo "[n2] GPU$gpu chain done (rc=$rc) $(date -Is)" >> "$LOG"
  return 0
}

if [ "$NGPU" -ge 2 ]; then
  run_chain 0 $GPU0_ARMS & P0=$!
  run_chain 1 $GPU1_ARMS & P1=$!
  wait $P0; R0=$?
  wait $P1; R1=$?
else
  run_chain 0 32 16 24; R0=$?; R1=0
fi

touch "$OUT/N2BIGSTEP_DONE"
echo "[n2] N2BIGSTEP_DONE $(date -Is) chain_rc=$R0/$R1" | tee -a "$LOG"
