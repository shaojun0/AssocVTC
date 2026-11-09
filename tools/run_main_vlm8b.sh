#!/usr/bin/env bash
# vlm8b 主跑：448²/p32、8 预算并行、Qwen3.5-4B 全参、ZeRO-2、encoder 全程可训
set -uo pipefail
cd /root/vlm8b
mkdir -p logs
EPOCHS=${EPOCHS:-1}
OUT=${OUT:-/root/autodl-tmp/vlm8b/output/run1}
echo "[run] start $(date -Is) epochs=$EPOCHS out=$OUT"
timeout 60000 /root/venv_qwen/bin/python -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29511 \
  train.py --epochs "$EPOCHS" \
  --output-dir "$OUT" \
  --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl
rc=$?
echo "[run] rc=$rc $(date -Is)" | tee /root/vlm8b/logs/run1.rc
if [ "$rc" -eq 0 ]; then
  touch "$OUT/VLM8B_DONE"
  echo "[run] VLM8B_DONE"
fi
exit "$rc"
