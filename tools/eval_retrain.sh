#!/usr/bin/env bash
# =============================================================================
# eval_retrain.sh — K=128/256 重训臂 + K=128 配对 cumulative B 的全量评测（2026-10-22）
# -----------------------------------------------------------------------------
# 等 RETRAIN_DONE → 依次（单卡）:
#   · stdattn_k128_t1 / _t8      末步 + 逐步
#   · stdattn_k256_t1 / _t8      末步 + 逐步
#   · paired stdattn_k128_t8     末步 + 逐步（T=1 与新 A 相同, 不再重复）
# → train_summary.csv → 标记 RE_DONE
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/eval_retrain.sh > $OUT/eval_launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/root/miniconda3/bin/python}
OUT=${OUT:-/root/autodl-tmp/srdiff_qknorm}
PAIRED=${PAIRED:-/root/autodl-tmp/srdiff_qknorm_paired}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0

if [ -f "$OUT/RE_DONE" ]; then echo "[eval-retrain] RE_DONE 已存在, 跳过"; exit 0; fi
echo "[eval-retrain] 等待 RETRAIN_DONE ... $(date -Is)"
while [ ! -f "$OUT/RETRAIN_DONE" ]; do sleep 60; done
echo "[eval-retrain] 开始评测 $(date -Is)"
mkdir -p "$OUT/eval"

ev_arm() {
  local ck=$1 tag=$2 evdir=$3
  mkdir -p "$evdir"
  $PY -u eval_linattn.py --ckpt "$ck" --data_dir "$DATA" \
    --batch_size 128 --num_workers 8 --out "$evdir/linattn_${tag}.json" \
    > "$evdir/linattn_${tag}.log" 2>&1
  echo "[eval-retrain] $tag t8 final rc=$?"
  $PY -u eval_stepwise_linattn.py --ckpt "$ck" --data_dir "$DATA" \
    --batch_size 64 --num_workers 8 --out "$evdir/stepwise_${tag}.json" \
    > "$evdir/stepwise_${tag}.log" 2>&1
  echo "[eval-retrain] $tag stepwise rc=$?"
}

# K=128/256 重训臂（A 相 T=1 + B 相 T=8）
for K in 128 256; do
  $PY -u eval_linattn.py --ckpt "$OUT/stdattn_k${K}_t1/ckpt_best.pt" --data_dir "$DATA" \
    --batch_size 128 --num_workers 8 --out "$OUT/eval/linattn_k${K}_t1.json" \
    > "$OUT/eval/linattn_k${K}_t1.log" 2>&1
  echo "[eval-retrain] k$K t1 rc=$?"
  ev_arm "$OUT/stdattn_k${K}_t8/ckpt_best.pt" "k${K}_t8" "$OUT/eval"
done

# K=128 配对 cumulative B
ev_arm "$PAIRED/stdattn_k128_t8/ckpt_best.pt" "k128_t8" "$PAIRED/eval"

$PY tools/summarize_fixedk.py --out_root "$OUT" --csv "$OUT/train_summary.csv" \
  --ks 128 256 > "$OUT/summarize.out" 2>&1
$PY tools/summarize_fixedk.py --out_root "$PAIRED" --csv "$PAIRED/train_summary.csv" \
  --ks 128 > "$PAIRED/summarize.out" 2>&1
touch "$OUT/RE_DONE"
echo "[eval-retrain] RE_DONE $(date -Is)"
