#!/usr/bin/env bash
# =============================================================================
# eval_lown.sh — n=6/4/2（K=48/32/16）固定 8 步臂的全量评测（2026-10-23）
# -----------------------------------------------------------------------------
# 等 LOWN_DONE → 每臂: T=8 末步 (eval_linattn) + 逐步 (eval_stepwise_linattn)
# （T=1 与新臂「同 A ckpt」, 但那是 10-19 cumulative 臂的 A 相、已评过; 这里也顺手重评 T=1
#   以便同一份 json 自洽。）
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/eval_lown.sh > $OUT/eval_launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
OUT=${OUT:-/root/autodl-tmp/srdiff_lown}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
EV=$OUT/eval

if [ -f "$OUT/LD_DONE" ]; then echo "[eval-lown] LD_DONE 已存在, 跳过"; exit 0; fi
echo "[eval-lown] 等待 LOWN_DONE ... $(date -Is)"
while [ ! -f "$OUT/LOWN_DONE" ]; do sleep 60; done
echo "[eval-lown] 开始评测 $(date -Is)"
mkdir -p "$EV"

for K in 16 32 48; do
  CK8=$OUT/stdattn_k${K}_t8/ckpt_best.pt
  CK1=$OUT/stdattn_k${K}_t1/ckpt_best.pt
  if [ ! -f "$CK8" ]; then echo "[eval-lown] K=$K 缺 B 相 ckpt, 跳过"; continue; fi
  $PY -u eval_linattn.py --ckpt "$CK8" --data_dir "$DATA" \
    --batch_size 128 --num_workers 8 --out "$EV/linattn_k${K}_t8.json" \
    > "$EV/linattn_k${K}_t8.log" 2>&1
  echo "[eval-lown] K=$K T=8 final rc=$?"
  $PY -u eval_stepwise_linattn.py --ckpt "$CK8" --data_dir "$DATA" \
    --batch_size 64 --num_workers 8 --out "$EV/stepwise_k${K}_t8.json" \
    > "$EV/stepwise_k${K}_t8.log" 2>&1
  echo "[eval-lown] K=$K stepwise rc=$?"
  # A 相（=旧 cumulative 臂的 A ckpt）也评一遍, 让同一份 json 自洽
  $PY -u eval_linattn.py --ckpt "$CK1" --data_dir "$DATA" \
    --batch_size 128 --num_workers 8 --out "$EV/linattn_k${K}_t1.json" \
    > "$EV/linattn_k${K}_t1.log" 2>&1
  echo "[eval-lown] K=$K T=1 final rc=$?"
done

$PY tools/summarize_fixedk.py --out_root "$OUT" --csv "$OUT/train_summary.csv" \
  --ks 16 32 48 > "$OUT/summarize.out" 2>&1
touch "$OUT/LD_DONE"
echo "[eval-lown] LD_DONE $(date -Is)"
