#!/bin/bash
# 第三轮：**同预算 stage=post**（LLM 输入口径）——与 pre 轮同预算（10k/2ep=624 步）
set -u
cd /root/AssocVTC_tok448
export HF_HOME=/root/autodl-tmp/hf2 HF_ENDPOINT=https://hf-mirror.com SDPA_BACKEND=math
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-/root/autodl-tmp/tok256x12_t8/probe_vendor_post}
mkdir -p "$OUT/logs" "$OUT/grids"
STAGE=${STAGE:-post}
TPC=${TPC:-100}
EPOCHS=${EPOCHS:-2}
GPUS=${GPUS:-"1 2 3"}
echo "[run] start $(date) stage=$STAGE tpc=$TPC epochs=$EPOCHS gpus=$GPUS" > "$OUT/status.txt"
encs=(glm5_flash qwen38_flash mimo26_flash)
i=0
for e in "${encs[@]}"; do
  gpu=$(echo $GPUS | cut -d" " -f$((i+1)))
  (
    MUSA_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    timeout 10800 python3 probe_enc_dec_vendor.py --vendor "$e" --stage "$STAGE" \
      --data_dir /root/autodl-tmp/datasets/imagenet100 \
      --train_per_class "$TPC" --val_per_class 10 --epochs "$EPOCHS" \
      --batch 32 --width 256 --workers 8 \
      --save_grid "$OUT/grids/$e.png" --out "$OUT/$e.json" --tag "post_10k_2ep" \
      > "$OUT/logs/$e.log" 2>&1
    echo "[run] $e rc=$? $(date)" >> "$OUT/status.txt"
  ) &
  i=$((i + 1))
done
wait
echo "[run] DONE $(date)" >> "$OUT/status.txt"
