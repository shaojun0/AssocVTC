#!/usr/bin/env bash
# =============================================================================
# eval_scale.sh — 「容量阶梯」4 档全量评测（2026-10-24）
# -----------------------------------------------------------------------------
# 每档评三样（与历史 fixed 臂同口径, 全量 5,000 张 val）:
#   1) B 相 T=16 末步: eval_linattn.py;
#   2) B 相逐步: eval_stepwise_linattn.py（每步 L1_255 / PSNR / Δ_t / 最优步）;
#   3) A 相 T=1: eval_linattn.py（对照「单步读满 K 个 z_s」的容量曲线）。
# 大模型前向显存大 ⇒ 240/360M 档用 batch_size 32（前向, 不影响指标口径）。
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/eval_scale.sh > $OUT/eval.out 2>&1 &
# 环境变量: OUT / DATA / GPU / BLOCKS / WAIT_H（等 SCALE_DONE 上限, 默认 120）
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
source tools/scale_arms.sh

PY=${PY:-/root/miniconda3/bin/python}
OUT=${OUT:-/root/autodl-tmp/srdiff_scale}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU=${GPU:-0}
BLOCKS=${BLOCKS:-"60 120 240 360"}
WAIT_H=${WAIT_H:-120}
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPU
EV=$OUT/eval

if [ -f "$OUT/SCALE_EVAL_DONE" ]; then echo "[eval-scale] 已存在, 跳过"; exit 0; fi
echo "[eval-scale] 等待 SCALE_DONE ... $(date -Is)"
n=0
while [ ! -f "$OUT/SCALE_DONE" ]; do
  sleep 120; n=$((n + 2))
  if [ "$n" -ge $((WAIT_H * 60)) ]; then echo "[eval-scale] 超时 ${WAIT_H}h"; exit 2; fi
done
echo "[eval-scale] 开始评测 $(date -Is)"
mkdir -p "$EV"

for M in $BLOCKS; do
  scale_arm_cfg "$M"
  CK=$OUT/stdattn_${SCALE_TAG}_t${SCALE_T}/ckpt_best.pt
  CK1=$OUT/stdattn_${SCALE_TAG}_t1/ckpt_best.pt
  BS=64
  if [ "$M" -ge 240 ]; then BS=32; fi
  if [ -f "$CK" ]; then
    $PY -u eval_linattn.py --ckpt "$CK" --data_dir "$DATA" \
      --batch_size "$BS" --num_workers 8 --out "$EV/linattn_m${M}_t${SCALE_T}.json" \
      > "$EV/linattn_m${M}_t${SCALE_T}.log" 2>&1
    echo "[eval-scale] m${M} T=${SCALE_T} 末步 rc=$? $(date -Is)"
    $PY -u eval_stepwise_linattn.py --ckpt "$CK" --data_dir "$DATA" \
      --batch_size "$BS" --num_workers 8 --out "$EV/stepwise_m${M}_t${SCALE_T}.json" \
      > "$EV/stepwise_m${M}_t${SCALE_T}.log" 2>&1
    echo "[eval-scale] m${M} 逐步 rc=$? $(date -Is)"
  else
    echo "[eval-scale] m${M} 缺 B 相 ckpt: $CK"
  fi
  if [ -f "$CK1" ]; then
    $PY -u eval_linattn.py --ckpt "$CK1" --data_dir "$DATA" \
      --batch_size "$BS" --num_workers 8 --out "$EV/linattn_m${M}_t1.json" \
      > "$EV/linattn_m${M}_t1.log" 2>&1
    echo "[eval-scale] m${M} A 相 rc=$? $(date -Is)"
  fi
done

touch "$OUT/SCALE_EVAL_DONE"
echo "[eval-scale] SCALE_EVAL_DONE $(date -Is)"
ls -la "$EV"
