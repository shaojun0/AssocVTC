#!/usr/bin/env bash
# =============================================================================
# run_fixedk_arm.sh — 「固定 8 步 × 每步均匀读 n 个 z_s」消融单臂训练（2026-10-21）
# -----------------------------------------------------------------------------
# 消融口径（2026-10-21 用户指定）:
#   · 固定 T=8 个解码步（BPTT 循环读出）;
#   · 每步**固定读 n 个 z_s**（不再递增）: 第 i 步读 [(i−1)·n+1, i·n]
#     （首步窗口另含 z_cls, 与主线 square 计划同约定）;
#   · 总 token 预算 K = 8·n ∈ {64, 96, 128, 192, 256} ⇒ n = 8, 12, 16, 24, 32;
#   · 与 2026-10-19 的 cumulative 臂**同 K 唯一差别 = 读窗口的分配形状**
#     （cumulative 按 square 形状缩放 ⇒ 前少后多; fixed ⇒ 每步等量）。
#
# 训练协议与 k8sweep 臂逐项同协议（见 tools/run_k8sweep_arm.sh）:
#   A 相: 单步 T=1, 45 ep, bs128; B 相: 循环 T=8, 6 ep, bs64, 从 A 相 ckpt_best 热启动。
#   lr: A 相与**同 K 的 cumulative 臂对齐**（K≤96: 2e-4; K≥128: 1e-4 —— K=128 在
#   2e-4 下 A 相不收敛, 见 doc/2026-10-19/REPORT_k8sweep.md §3.4; K=192/256 无同 K
#   参照幅, 沿用 1e-4）; B 相统一 2e-4。
#   K=256 > N=196 ⇒ 不再是压缩（预算超过 patch 数）, 需 --allow_k_gt_n 显式开启。
#
# 用法（在服务器仓库根目录）:
#   K=64 GPU=0 bash tools/run_fixedk_arm.sh
# 环境变量: K（必填）/ GPU / T / DATA / OUT / PY / SKIP_STAGE1 / LR1 / LR2
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
K=${K:?请设置 K（64/96/128/192/256）}
GPU=${GPU:-0}
T=${T:-8}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/srdiff_fixedk}
SKIP_STAGE1=${SKIP_STAGE1:-0}
LR1=${LR1:-2e-4}
LR2=${LR2:-2e-4}

if [ $((K % T)) -ne 0 ]; then
  echo "[fixedk] K=$K 不能被 T=$T 整除（本消融要求每步 n=K/T 为整数）"; exit 3
fi
BLOCK=$((K / T))
A1=$OUT/stdattn_k${K}_t1
A2=$OUT/stdattn_k${K}_t${T}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

DEC_STEPS=$($PY -c "print(','.join(str(i) for i in range(1, ${T}+1)))")
EXTRA=""
if [ "$K" -gt 196 ]; then EXTRA="--allow_k_gt_n 1"; fi

echo "[fixedk] K=$K n=$BLOCK T=$T steps=$DEC_STEPS lr1=$LR1 lr2=$LR2 start=$(date -Is)"

if [ "$SKIP_STAGE1" != "1" ]; then
  # ── A 相：单步 T=1（45 ep / bs128）────────────────────────────────────────
  echo "[fixedk] K=$K A相 T=1 lr=$LR1 → $A1"
  $PY -u train_linattn.py \
    --data_dir "$DATA" --output_dir "$A1" \
    --enc_attn std --num_specials "$K" --batch_size 128 --epochs 45 --lr "$LR1" \
    --dec_recurrent 0 --time_budget_h 12 --num_workers 16 --seed 0 $EXTRA \
    >> "$OUT/k${K}_stage1.log" 2>&1
  RC1=$?
  echo "[fixedk] K=$K A相 rc=$RC1 end=$(date -Is)"
  if [ $RC1 -ne 0 ] || [ ! -f "$A1/ckpt_best.pt" ]; then
    echo "[fixedk] K=$K FAILED(stage1)"; exit 1
  fi
fi

# ── B 相：循环 T=8（6 ep / bs64 / fixed 读窗口 / 热启动）────────────────────
echo "[fixedk] K=$K B相 T=$T lr=$LR2 → $A2"
$PY -u train_linattn.py \
  --data_dir "$DATA" --output_dir "$A2" \
  --enc_attn std --num_specials "$K" --batch_size 64 --epochs 6 --lr "$LR2" \
  --dec_recurrent 1 --step_plan fixed --block "$BLOCK" --dec_steps "$DEC_STEPS" \
  --init_from "$A1/ckpt_best.pt" --time_budget_h 8 --num_workers 16 --seed 0 $EXTRA \
  >> "$OUT/k${K}_stage2.log" 2>&1
RC2=$?
echo "[fixedk] K=$K B相 rc=$RC2 end=$(date -Is)"
if [ $RC2 -eq 0 ] && [ -f "$A2/ckpt_best.pt" ]; then
  echo "[fixedk] K=$K DONE"
else
  echo "[fixedk] K=$K FAILED(stage2)"; exit 2
fi
