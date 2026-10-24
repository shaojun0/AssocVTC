#!/usr/bin/env bash
# =============================================================================
# launch_scale_t24.sh — 条件臂：n=2 × T=24（K=48）@ 选定容量档（2026-10-24）
# -----------------------------------------------------------------------------
# 用户口径：「如果出现非平坦的情况下再实验一个 n/step=2，step=24，参数量选择
#   上一步实验里的最好的效果的那一个。」
# 说明：T=24 且 n=2 ⇒ K=48 个 z_s，与阶梯臂的 K=32 **不同** ⇒ A 相不能复用
#   （A 相是单步读满 K 个 z_s，K 变了就是另一个 A 相）⇒ 本臂是完整 A+B，
#   只是容量档由上一轮结果决定（M 由调用方给）。输出放独立目录避免覆盖阶梯臂。
# 协议与阶梯臂一致：A 相 T=1 45ep/bs128/lr2e-4 → B 相 T=24 `fixed` n=2 6ep/
#   等效 bs64/lr2e-4；容量档的 accum 见 tools/scale_arms.sh（T=24 显存更高，
#   必要时 FORCE_ACCUM_B=4）。
# 用法（服务器仓库根目录）:
#   M=240 nohup setsid bash tools/launch_scale_t24.sh > /root/autodl-tmp/srdiff_scale_t24/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
source tools/scale_arms.sh

PY=${PY:-/root/miniconda3/bin/python}
M=${M:?请设置 M（容量档 60/120/240/360，取上一轮效果最好的）}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/srdiff_scale_t24}
GPU=${GPU:-0}
WAIT_T16_DONE=${WAIT_T16_DONE:-1}   # 1 = 先等阶梯臂 SCALE_DONE 再跑（避免抢卡）
mkdir -p "$OUT"
EV=$OUT/eval

export SCALE_K=48
export SCALE_T=24
scale_arm_cfg "$M"

echo "[t24] M=$M K=$SCALE_K T=$SCALE_T accumA=$SCALE_ACCUM_A accumB=$SCALE_ACCUM_B start=$(date -Is)"
if [ "$WAIT_T16_DONE" = "1" ]; then
  echo "[t24] 等待阶梯臂 SCALE_DONE ... $(date -Is)"
  while [ ! -f /root/autodl-tmp/srdiff_scale/SCALE_DONE ]; do sleep 120; done
  echo "[t24] 阶梯臂已完成, 开跑 $(date -Is)"
fi

M="$M" GPU="$GPU" OUT="$OUT" DATA="$DATA" bash tools/run_scale_arm.sh
RC=$?
echo "[t24] M=$M 训练 rc=$RC end=$(date -Is)"

# ── 评测（末步 + 逐步 + A 相）────────────────────────────────────────────
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$EV"
CK=$OUT/stdattn_${SCALE_TAG}_t${SCALE_T}/ckpt_best.pt
CK1=$OUT/stdattn_${SCALE_TAG}_t1/ckpt_best.pt
BS=64; if [ "$M" -ge 240 ]; then BS=32; fi
if [ -f "$CK" ]; then
  $PY -u eval_linattn.py --ckpt "$CK" --data_dir "$DATA" --batch_size "$BS" \
    --num_workers 8 --out "$EV/linattn_m${M}_t${SCALE_T}.json" \
    > "$EV/linattn_m${M}_t${SCALE_T}.log" 2>&1
  echo "[t24] 末步 rc=$?"
  $PY -u eval_stepwise_linattn.py --ckpt "$CK" --data_dir "$DATA" --batch_size "$BS" \
    --num_workers 8 --out "$EV/stepwise_m${M}_t${SCALE_T}.json" \
    > "$EV/stepwise_m${M}_t${SCALE_T}.log" 2>&1
  echo "[t24] 逐步 rc=$?"
fi
if [ -f "$CK1" ]; then
  $PY -u eval_linattn.py --ckpt "$CK1" --data_dir "$DATA" --batch_size "$BS" \
    --num_workers 8 --out "$EV/linattn_m${M}_t1.json" > "$EV/linattn_m${M}_t1.log" 2>&1
  echo "[t24] A 相 rc=$?"
fi
touch "$OUT/T24_DONE"
echo "[t24] T24_DONE $(date -Is)"
