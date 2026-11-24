#!/bin/bash
# 四家厂商 vision tower 的可解码性探针（4 GPU 并行）。协议见 tools/probe_enc_dec_vendor.py。
set -u
cd /root/AssocVTC_tok448
export HF_HOME=/root/autodl-tmp/hf2
export HF_ENDPOINT=https://hf-mirror.com
export SDPA_BACKEND=math
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-/root/autodl-tmp/tok256x12_t8/probe_vendor}
mkdir -p "$OUT/logs" "$OUT/grids"
TRAIN_PER_CLASS=${TRAIN_PER_CLASS:-200}
EPOCHS=${EPOCHS:-3}
echo "[run] start $(date) train_per_class=$TRAIN_PER_CLASS epochs=$EPOCHS" > "$OUT/status.txt"
encs=(glm5_flash qwen38_flash mimo26_flash deepseek_v41_flash)
i=0
for e in "${encs[@]}"; do
  (
    MUSA_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i \
    timeout 10800 python3 probe_enc_dec_vendor.py --vendor "$e" \
      --data_dir /root/autodl-tmp/datasets/imagenet100 \
      --train_per_class "$TRAIN_PER_CLASS" --val_per_class 10 --epochs "$EPOCHS" \
      --batch 32 --width 256 --workers 8 \
      --save_grid "$OUT/grids/$e.png" --out "$OUT/$e.json" --tag run1 \
      > "$OUT/logs/$e.log" 2>&1
    echo "[run] $e rc=$? $(date)" >> "$OUT/status.txt"
  ) &
  i=$((i + 1))
done
wait
echo "[run] DONE $(date)" >> "$OUT/status.txt"
grep -h "RESULT" "$OUT/logs/"*.log | tail -8
