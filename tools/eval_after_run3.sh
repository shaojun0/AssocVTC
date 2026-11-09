#!/usr/bin/env bash
# run3 结束 → 自动评测：mix_heldout（3 组视觉消融）+ ImageNet val + 通用 bench
# 产物：/root/autodl-tmp/vlm8b/output/eval_run3_mix_{normal,null,shuffle}/eval.{json,md}
#       /root/autodl-tmp/vlm8b/output/eval_run3_in100/eval.{json,md}
#       /root/vlm8b/logs/bench_run3.json、logs/eval_chain_run3.log、标记 logs/RUN3_EVAL_DONE
set -euo pipefail
cd /root/vlm8b
LOG=logs/eval_chain_run3.log
DATA=/root/autodl-tmp/vlm8b/data_eval
OUT=/root/autodl-tmp/vlm8b/output/run3
mkdir -p logs
echo "[chain3] start, wait run3.rc $(date -Is)" >> "$LOG"
for _ in $(seq 1 2400); do
  [ -f logs/run3.rc ] && break
  sleep 60
done
sleep 20
if [ ! -d "$OUT/final" ]; then
  echo "[chain3] NO $OUT/final -> abort ($(date -Is))" >> "$LOG"
  exit 1
fi
echo "[chain3] run3 rc=$(cat logs/run3.rc) $(date -Is)" >> "$LOG"

# 1) 混合 heldout（各任务预算消融）
for mode in normal null shuffle; do
  echo "[chain3] mix_heldout $mode start $(date -Is)" >> "$LOG"
  /root/venv_qwen/bin/python -u eval.py \
    --checkpoint "$OUT/final" --jsonl "$DATA/mix_heldout.jsonl" --image-root "$DATA" \
    --limit 300 --sample-seed 42 --budgets 1,2,3,4,5,6,7,8 --visual-mode "$mode" \
    --output-dir "/root/autodl-tmp/vlm8b/output/eval_run3_mix_$mode" >> "$LOG" 2>&1
  echo "[chain3] mix_heldout $mode rc=$? $(date -Is)" >> "$LOG"
done

# 2) ImageNet-100 val（与 run1/run2 同口径，看主任务有没有被混合集稀释）
echo "[chain3] imagenet100 start $(date -Is)" >> "$LOG"
/root/venv_qwen/bin/python -u eval.py \
  --checkpoint "$OUT/final" --limit 250 --sample-seed 42 --budgets 1,2,3,4,5,6,7,8 \
  --visual-mode normal \
  --output-dir /root/autodl-tmp/vlm8b/output/eval_run3_in100 >> "$LOG" 2>&1
echo "[chain3] imagenet100 rc=$? $(date -Is)" >> "$LOG"

# 3) 通用文本能力（弱智吧/算术/常识）
echo "[chain3] bench start $(date -Is)" >> "$LOG"
/root/venv_qwen/bin/python -u bench_general_vlm8b.py \
  --checkpoint "$OUT/final" --which both --out logs/bench_run3.json >> "$LOG" 2>&1
echo "[chain3] bench rc=$? $(date -Is)" >> "$LOG"

# 4) 分任务表（source × budget，回答「预算机制在哪类任务上成立」）
echo "[chain3] breakdown start $(date -Is)" >> "$LOG"
/root/venv_qwen/bin/python -u eval_mix_breakdown.py \
  --checkpoint "$OUT/final" --jsonl "$DATA/mix_heldout.jsonl" --image-root "$DATA" \
  --per-source 25 --visual-mode normal --out logs/run3_breakdown_normal.json >> "$LOG" 2>&1
echo "[chain3] breakdown rc=$? $(date -Is)" >> "$LOG"

touch logs/RUN3_EVAL_DONE
echo "[chain3] ALL DONE $(date -Is)" >> "$LOG"
