#!/usr/bin/env bash
# =============================================================================
# smoke_bigstep.sh — 「固定每步 n=2 × T=16/24/32」臂的冒烟（2026-10-23）
# -----------------------------------------------------------------------------
# 目的（跑正式训练前 5 分钟成本换确定性）:
#   1) 验证 current 代码（fixed plan + T>8）能构建/前向/反传;
#   2) 验证 10-19 的 A 相 ckpt（K=32/48/64, 无 QK-Norm）能热启动到循环解码器
#      （只缺 decoder.mem_pos）;
#   3) 量 s/iter 与 vram_gb（--dec_steps 1..T ⇒ 线性增长; 80G 卡能否 bs64）;
#   4) 读窗口覆盖 0..K 自检（T 步 × n=2 恰好读满 K=2T 个 z_s）。
# 只跑 20 step（--max_steps 20），输出到 /root/autodl-tmp/smoke_n2bigstep（可随时删）。
# 用法（服务器仓库根目录）:
#   ARMS="64:32" GPU=0 bash tools/smoke_bigstep.sh
#   ARMS="32:16 48:24 64:32" GPU=0 bash tools/smoke_bigstep.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
A_DIR=${A_DIR:-/root/autodl-tmp/srdiff_k8sweep}
OUT=${OUT:-/root/autodl-tmp/smoke_n2bigstep}
GPU=${GPU:-0}
ARMS=${ARMS:-"32:16 48:24 64:32"}
BS=${BS:-64}
ACCUM=${ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-20}
mkdir -p "$OUT"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPU

for spec in $ARMS; do
  K=${spec%%:*}; T=${spec##*:}; BLOCK=$((K / T))
  A_CK=$A_DIR/stdattn_k${K}_t1/ckpt_best.pt
  LOG=$OUT/smoke_k${K}_t${T}.log
  echo "[smoke] K=$K T=$T block=$BLOCK A=$A_CK bs=$BS accum=$ACCUM start=$(date -Is)"
  if [ ! -f "$A_CK" ]; then echo "[smoke] K=$K 缺 A ckpt: $A_CK"; continue; fi
  STEPS=$($PY -c "print(','.join(str(i) for i in range(1, ${T}+1)))")
  $PY -u train_linattn.py \
    --data_dir "$DATA" --output_dir "$OUT/k${K}_t${T}" \
    --enc_attn std --num_specials "$K" --batch_size "$BS" --epochs 6 --lr 2e-4 \
    --dec_recurrent 1 --step_plan fixed --block "$BLOCK" --dec_steps "$STEPS" \
    --init_from "$A_CK" --num_workers 8 --seed 0 --accum_steps "$ACCUM" \
    --max_steps "$MAX_STEPS" --log_interval 5 --sample_interval 0 \
    --val_batches 8 >> "$LOG" 2>&1
  RC=$?
  echo "[smoke] K=$K T=$T rc=$RC end=$(date -Is)"
  grep -E "^\[init_from\]|windows=|^\[(data|cfg|params|done)\]" "$LOG" | head -6
  grep -E "^\[ *[0-9.]+m\] step" "$LOG" | tail -2
  if [ $RC -ne 0 ]; then
    echo "[smoke] K=$K T=$T FAILED, 尾部日志:"; tail -15 "$LOG"
  fi
done
echo "[smoke] all done $(date -Is)"
