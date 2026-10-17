#!/bin/bash
# 轮询 stage-1 checkpoint，每个新 ckpt 跑一次 32→256 token 重建阶梯
cd /root/autodl-tmp/alit
export PYTHONPATH=/root/autodl-tmp/alit WANDB_MODE=offline
OUT=/root/autodl-tmp/alit_out/s1_tiny_in100
LAD=/root/autodl-tmp/alit_out/ladder
mkdir -p "$LAD"
while [ -f /root/autodl-tmp/alit_out/.ladder_alive ]; do
  for f in $(ls $OUT/checkpoint-*.pth 2>/dev/null | grep -vE "last" | sort -V); do
    base=$(basename "$f" .pth)
    [ -f "$LAD/$base/ladder.json" ] && continue
    echo "[watch] eval $base $(date)"
    /root/miniconda3/bin/python eval_ladder.py --ckpt "$f" --which ema --n 128 --bs 8 --out "$LAD/$base" \
       >> "$LAD/watch.log" 2>&1
  done
  sleep 300
done
