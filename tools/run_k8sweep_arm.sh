#!/usr/bin/env bash
# =============================================================================
# run_k8sweep_arm.sh — 「固定 8 步 × 总 token 预算 K」K-sweep 单臂训练
# -----------------------------------------------------------------------------
# 消融口径（2026-10-19 用户指定）:
#   · 固定 T=8 个解码步（BPTT 循环读出）;
#   · 总 token 预算 K ∈ {8, 12, 16, 24, 32}（另加 K=64 作同协议参照点）;
#   · K 与 T 在主线里是平方耦合（K=64 ↔ 8 步），这里用 `--step_plan cumulative`
#     把 square 计划的块尺寸形状（3,5,7,…）按比例缩放到 K，见
#     `model_v2_linattn.scaled_square_cumsum(K, T)` 的 docstring。
#
# 训练协议与该仓库既有 BPTT 臂**逐项同协议**（见 stdattn_k64 / stdattn_k64_rec）:
#   A 相: 单步 T=1, 45 ep, bs128, lr2e-4, enc_attn=std（从零）
#   B 相: 循环 T=8, 6 ep, bs64, lr2e-4, 从 A 相 ckpt_best 热启动（--init_from）
#
# 用法（在服务器仓库根目录）:
#   K=32 GPU=0 bash tools/run_k8sweep_arm.sh
# 环境变量: K（必填）/ GPU / T / DATA / OUT / PY / SKIP_STAGE1（1=只跑 B 相）
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
K=${K:?请设置 K（8/12/16/24/32/64）}
GPU=${GPU:-0}
T=${T:-8}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/srdiff_k8sweep}
SKIP_STAGE1=${SKIP_STAGE1:-0}
A1=$OUT/stdattn_k${K}_t1
A2=$OUT/stdattn_k${K}_t${T}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

DEC_STEPS=$($PY -c "from model_v2_linattn import scaled_square_cumsum as f; print(','.join(map(str, f($K, $T))))")
echo "[k8sweep] K=$K GPU=$GPU T=$T dec_steps=$DEC_STEPS start=$(date -Is)"

if [ "$SKIP_STAGE1" != "1" ]; then
  # ── A 相：单步 T=1（45 ep / bs128 / lr2e-4）────────────────────────────────
  echo "[k8sweep] K=$K A相 T=1 → $A1"
  $PY -u train_linattn.py \
    --data_dir "$DATA" --output_dir "$A1" \
    --enc_attn std --num_specials "$K" --batch_size 128 --epochs 45 --lr 2e-4 \
    --dec_recurrent 0 --time_budget_h 12 --num_workers 16 --seed 0 \
    >> "$OUT/k${K}_stage1.log" 2>&1
  RC1=$?
  echo "[k8sweep] K=$K A相 rc=$RC1 end=$(date -Is)"
  if [ $RC1 -ne 0 ] || [ ! -f "$A1/ckpt_best.pt" ]; then
    echo "[k8sweep] K=$K FAILED(stage1)"; exit 1
  fi
fi

# ── B 相：循环 T=8（6 ep / bs64 / lr2e-4 / cumulative 读窗口 / 热启动）──────
echo "[k8sweep] K=$K B相 T=$T → $A2"
$PY -u train_linattn.py \
  --data_dir "$DATA" --output_dir "$A2" \
  --enc_attn std --num_specials "$K" --batch_size 64 --epochs 6 --lr 2e-4 \
  --dec_recurrent 1 --step_plan cumulative --dec_steps "$DEC_STEPS" \
  --init_from "$A1/ckpt_best.pt" --time_budget_h 8 --num_workers 16 --seed 0 \
  >> "$OUT/k${K}_stage2.log" 2>&1
RC2=$?
echo "[k8sweep] K=$K B相 rc=$RC2 end=$(date -Is)"
if [ $RC2 -eq 0 ] && [ -f "$A2/ckpt_best.pt" ]; then
  echo "[k8sweep] K=$K DONE"
else
  echo "[k8sweep] K=$K FAILED(stage2)"; exit 2
fi
