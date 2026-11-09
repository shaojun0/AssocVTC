#!/usr/bin/env bash
# vlm8b 平台期重启：加大 LR，从 INIT（默认 run1/final）续训
# 用法: LR_LLM=3e-5 EPOCHS=1.6 OUT=.../run3 INIT=.../run1/final bash run_cont3.sh
set -uo pipefail
cd /root/vlm8b
if pgrep -af "torch.distributed.run.*train.py" >/dev/null; then
  echo "[run3] 已有训练进程在跑，拒绝启动（先跑 restart_lr.sh 或手动 kill）" >&2
  exit 3
fi
LR_LLM=${LR_LLM:-3e-5}; LR_PROJ=${LR_PROJ:-3e-4}; LR_ENC=${LR_ENC:-1e-5}
EPOCHS=${EPOCHS:-2}; OUT=${OUT:-/root/autodl-tmp/vlm8b/output/run3}
INIT=${INIT:-/root/autodl-tmp/vlm8b/output/run1/final}
SAVE=${SAVE:-1}; PORT=${PORT:-29512}
mkdir -p logs
echo "[run3] start $(date -Is) epochs=$EPOCHS init=$INIT out=$OUT lr=$LR_LLM/$LR_PROJ/$LR_ENC"
timeout 100000 /root/venv_qwen/bin/python -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port "$PORT" \
  train.py --epochs "$EPOCHS" --init-from "$INIT" \
  --output-dir "$OUT" --save-total-limit "$SAVE" \
  --lr-llm "$LR_LLM" --lr-proj "$LR_PROJ" --lr-encoder "$LR_ENC" \
  --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl \
  > logs/run3.log 2>&1
rc=$?
echo "[run3] rc=$rc $(date -Is)" | tee logs/run3.rc
[ "$rc" -eq 0 ] && touch "$OUT/VLM8B_DONE"
exit "$rc"
