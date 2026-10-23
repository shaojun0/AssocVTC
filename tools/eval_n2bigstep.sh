#!/usr/bin/env bash
# =============================================================================
# eval_n2bigstep.sh — n=2 × T=16/24/32（K=32/48/64）臂的全量评测（2026-10-23）
# -----------------------------------------------------------------------------
# 每臂评三样（与 10-21/10-23 的 fixed 臂同口径）:
#   1) T 步末步: eval_linattn.py（PSNR / L1_norm / SSIM, 5000 张 val 全量）;
#   2) 逐步: eval_stepwise_linattn.py（每步 L1_255 / PSNR / Δ_t / 最优停步）;
#   3) T=1（A 相 ckpt, 软链）: 让同一份 json 自洽, 并与 10-19 的 A 相 eval 对拍。
# T=32 前向显存比 T=8 大 4 倍 ⇒ 统一 batch_size 64（前向, 不影响指标口径）。
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/eval_n2bigstep.sh > $OUT/eval.out 2>&1 &
# 环境变量: OUT / DATA / GPU / ARMS（T 列表, 默认 "32 16 24"）/ WAIT_H（等待上限, 默认 16）
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
OUT=${OUT:-/root/autodl-tmp/srdiff_n2bigstep}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU=${GPU:-0}
ARMS=${ARMS:-"32 16 24"}
WAIT_H=${WAIT_H:-16}
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPU
EV=$OUT/eval

if [ -f "$OUT/N2BIGSTEP_EVAL_DONE" ]; then
  echo "[eval-n2] N2BIGSTEP_EVAL_DONE 已存在, 跳过"; exit 0
fi
echo "[eval-n2] 等待 N2BIGSTEP_DONE ... $(date -Is)"
MAX_WAIT=$((WAIT_H * 60))
n=0
while [ ! -f "$OUT/N2BIGSTEP_DONE" ]; do
  sleep 60; n=$((n + 1))
  if [ "$n" -ge "$MAX_WAIT" ]; then
    echo "[eval-n2] 等 N2BIGSTEP_DONE 超时 ${WAIT_H}h, 退出"; exit 2
  fi
done
echo "[eval-n2] 开始评测 $(date -Is)"
mkdir -p "$EV"

for T in $ARMS; do
  K=$((2 * T))
  CK=$OUT/stdattn_k${K}_t${T}/ckpt_best.pt
  CK1=$OUT/stdattn_k${K}_t1/ckpt_best.pt
  if [ ! -e "$CK" ]; then echo "[eval-n2] K=$K T=$T 缺 B 相 ckpt, 跳过"; continue; fi
  $PY -u eval_linattn.py --ckpt "$CK" --data_dir "$DATA" \
    --batch_size 64 --num_workers 8 --out "$EV/linattn_k${K}_t${T}.json" \
    > "$EV/linattn_k${K}_t${T}.log" 2>&1
  echo "[eval-n2] K=$K T=$T 末步 rc=$? $(date -Is)"
  $PY -u eval_stepwise_linattn.py --ckpt "$CK" --data_dir "$DATA" \
    --batch_size 64 --num_workers 8 --out "$EV/stepwise_k${K}_t${T}.json" \
    > "$EV/stepwise_k${K}_t${T}.log" 2>&1
  echo "[eval-n2] K=$K T=$T 逐步 rc=$? $(date -Is)"
  if [ -e "$CK1" ]; then
    $PY -u eval_linattn.py --ckpt "$CK1" --data_dir "$DATA" \
      --batch_size 128 --num_workers 8 --out "$EV/linattn_k${K}_t1.json" \
      > "$EV/linattn_k${K}_t1.log" 2>&1
    echo "[eval-n2] K=$K T=1(A 相) rc=$? $(date -Is)"
  fi
done

touch "$OUT/N2BIGSTEP_EVAL_DONE"
echo "[eval-n2] N2BIGSTEP_EVAL_DONE $(date -Is)"
ls -la "$EV"
