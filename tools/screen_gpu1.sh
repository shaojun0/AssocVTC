#!/usr/bin/env bash
# 筛查 GPU1：训练集逐样本 + ImageNet-100 val（修正协议 + 逐行）
set -uo pipefail
cd /root/vlm8b
export CUDA_VISIBLE_DEVICES=1
D=/root/autodl-tmp/vlm8b/data_eval
I=/root/autodl-tmp/exp_qwen/in100_selfdistill/sft/val_nothink_all.jsonl
IR=/root/autodl-tmp/datasets/imagenet100
CK=/root/autodl-tmp/vlm8b/output/run3/final
PY=/root/venv_qwen/bin/python
echo "[gpu1] start $(date -Is)"
$PY -u eval.py --checkpoint $CK --jsonl $I --image-root $IR \
  --limit 250 --sample-seed 42 --budgets 1,2,3,4,5,6,7,8 --visual-mode normal \
  --output-dir /root/autodl-tmp/vlm8b/output/eval_run3_in100_fix \
  --gen-protocol prompt
echo "[gpu1] in100 fix rc=$? $(date -Is)"
$PY -u eval_dump_per_sample.py --checkpoint $CK --jsonl $I --image-root $IR \
  --random-n 400 --seed 42 --gen-k 1,8 --out logs/screen_in100.jsonl
echo "[gpu1] in100 dump rc=$? $(date -Is)"
$PY -u eval_dump_per_sample.py --checkpoint $CK --jsonl $D/mix_train_bal2.jsonl \
  --image-root $D --per-group 120 --seed 7 --gen-k 1,8 \
  --out logs/screen_train.jsonl
echo "[gpu1] train dump rc=$? $(date -Is)"
touch logs/SCREEN_GPU1_DONE
echo "[gpu1] ALL DONE $(date -Is)"
