#!/usr/bin/env bash
# 等 run2 结束 → 自动跑 3 组视觉消融 + 通用能力 bench → 写 logs/RUN2_EVAL_DONE
# 产物：/root/autodl-tmp/vlm8b/output/eval_run2_{normal,null,shuffle}/eval.{json,md}
#       /root/vlm8b/logs/bench_run2.json、logs/eval_chain.log
set -uo pipefail
cd /root/vlm8b
LOG=logs/eval_chain.log
mkdir -p logs
echo "[chain] start, wait run2.rc $(date -Is)" >> "$LOG"
for _ in $(seq 1 1500); do
  [ -f logs/run2.rc ] && break
  sleep 60
done
sleep 20
OUT=/root/autodl-tmp/vlm8b/output/run2
if [ ! -d "$OUT/final" ]; then
  echo "[chain] NO $OUT/final -> abort eval ($(date -Is))" >> "$LOG"
  touch logs/RUN2_EVAL_DONE
  exit 0
fi
echo "[chain] run2 rc=$(cat logs/run2.rc) final=$OUT/final $(date -Is)" >> "$LOG"
for mode in normal null shuffle; do
  echo "[chain] eval $mode start $(date -Is)" >> "$LOG"
  /root/venv_qwen/bin/python -u eval.py \
    --checkpoint "$OUT/final" --limit 250 --sample-seed 42 --budgets 1,2,3,4,5,6,7,8 \
    --visual-mode "$mode" \
    --output-dir "/root/autodl-tmp/vlm8b/output/eval_run2_$mode" >> "$LOG" 2>&1
  echo "[chain] eval $mode rc=$? $(date -Is)" >> "$LOG"
done
echo "[chain] bench start $(date -Is)" >> "$LOG"
/root/venv_qwen/bin/python -u bench_general_vlm8b.py \
  --checkpoint "$OUT/final" --which both --out logs/bench_run2.json >> "$LOG" 2>&1
echo "[chain] bench rc=$? $(date -Is)" >> "$LOG"
touch logs/RUN2_EVAL_DONE
echo "[chain] ALL DONE $(date -Is)" >> "$LOG"
