#!/usr/bin/env bash
# 筛查 GPU0：mix_heldout 逐样本（CE 多消融 + k1/k8 greedy）+ 修正协议复跑 300 行
set -uo pipefail
cd /root/vlm8b
export CUDA_VISIBLE_DEVICES=0
D=/root/autodl-tmp/vlm8b/data_eval
CK=/root/autodl-tmp/vlm8b/output/run3/final
PY=/root/venv_qwen/bin/python
echo "[gpu0] start $(date -Is)"
$PY -u eval_dump_per_sample.py --checkpoint $CK --jsonl $D/mix_heldout.jsonl \
  --image-root $D --per-group 120 --seed 7 --gen-k 1,8 \
  --out logs/screen_heldout.jsonl
echo "[gpu0] heldout dump rc=$? $(date -Is)"
for mode in normal null shuffle; do
  $PY -u eval.py --checkpoint $CK --jsonl $D/mix_heldout.jsonl --image-root $D \
    --limit 300 --sample-seed 42 --budgets 1,2,3,4,5,6,7,8 --visual-mode $mode \
    --output-dir /root/autodl-tmp/vlm8b/output/eval_run3_fix_$mode \
    --gen-protocol prompt
  echo "[gpu0] fix eval $mode rc=$? $(date -Is)"
done
touch logs/SCREEN_GPU0_DONE
echo "[gpu0] ALL DONE $(date -Is)"
