#!/usr/bin/env bash
# SR-Diffusion-v3 实验版 (linattn) 训练启动脚本
#   K=64 BS=128 EPOCHS=30 TIME_BUDGET=6 ./run_linattn.sh
# 环境变量: K / BS / EPOCHS / TAG / TIME_BUDGET / LR
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/root/miniconda3/bin/python}
K=${K:-64}
BS=${BS:-128}
EPOCHS=${EPOCHS:-30}
LR=${LR:-3e-4}
TAG=${TAG:-linattn_k${K}}
TIME_BUDGET=${TIME_BUDGET:-6}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/SR-Diffusion-v3/output/${TAG}}
mkdir -p "$OUT"
echo "[run_linattn] K=$K bs=$BS epochs=$EPOCHS lr=$LR out=$OUT"
exec "$PY" -u train_linattn.py \
  --data_dir "$DATA" --output_dir "$OUT" \
  --num_specials "$K" --batch_size "$BS" --epochs "$EPOCHS" --lr "$LR" \
  --time_budget_h "$TIME_BUDGET" "$@"
