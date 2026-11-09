#!/usr/bin/env bash
# vlm8b 续训：从 run1/final 继续 2 epoch（448²/p32、8 预算、ZeRO-2、encoder 全可训）
set -uo pipefail
cd /root/vlm8b
mkdir -p logs
EPOCHS=${EPOCHS:-2}
OUT=${OUT:-/root/autodl-tmp/vlm8b/output/run2}
INIT=${INIT:-/root/autodl-tmp/vlm8b/output/run1/final}
SAVE=${SAVE:-1}
echo "[run2] start $(date -Is) epochs=$EPOCHS init=$INIT out=$OUT"
timeout 100000 /root/venv_qwen/bin/python -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29511 \
  train.py --epochs "$EPOCHS" --init-from "$INIT" \
  --output-dir "$OUT" --save-total-limit "$SAVE" \
  --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl
rc=$?
echo "[run2] rc=$rc $(date -Is)" | tee /root/vlm8b/logs/run2.rc
if [ "$rc" -eq 0 ]; then touch "$OUT/VLM8B_DONE"; fi
exit "$rc"
