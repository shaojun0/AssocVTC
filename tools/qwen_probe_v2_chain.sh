#!/usr/bin/env bash
# 探针 v2 四臂链（warm start from AssocVTC 240M decoder; E=2; bs128; 全量）
# 顺序: 5120(上界) → 20(最狠截断) → 754 → 234；r=5120 训完先做 3 个零训练截断 eval
set -uo pipefail
cd /root/autodl-tmp/qwen_vtc
PY=/root/miniconda3/bin/python
export QWEN_ZIP=/root/img100.zip
export PYTHONPATH=/root/autodl-tmp/qwen_vtc/code
INIT=probe/init_m240_decoder.pt
for R in 5120 20 754 234; do
  OUT=probe/v2_r$R
  if [ -f "$OUT/best.pt" ] && [ -f "$OUT/train_state.json" ]; then echo "[chain] skip r=$R (done)"; else
    echo "[chain] ==== r=$R train start $(date -Is) ===="
    $PY -u code/qwen_probe_v2.py --mode train --r $R --epochs 2 --bs 128 --workers 16 \
        --lr 2e-4 --warmup 200 --init $INIT --out $OUT > probe/logs_v2_r$R.log 2>&1
    echo "[chain] r=$R train rc=$? end $(date -Is)"
    $PY -u code/qwen_probe_v2.py --mode eval --r $R --ckpt $OUT/best.pt \
        --out probe/eval_v2_r$R.json >> probe/logs_v2_r$R.log 2>&1
    rm -f $OUT/last.pt
    echo "[chain] r=$R eval done $(date -Is)"
  fi
  if [ "$R" = "5120" ]; then
    for RZ in 754 234 20; do
      $PY -u code/qwen_probe_v2.py --mode eval --r $RZ --ckpt $OUT/best.pt \
          --out probe/eval_v2_zeroshot_r$RZ.json >> probe/logs_v2_r5120.log 2>&1
    done
    echo "[chain] zeroshot done $(date -Is)"
  fi
done
echo "[chain] ALL_DONE $(date -Is)"
