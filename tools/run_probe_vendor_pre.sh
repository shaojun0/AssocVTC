#!/bin/bash
# 第二轮：**同预算 stage-matched** 探针（4 GPU 并行）
#   stage=pre  = ViT 原始 token（GLM/Qwen/MiMo 的 merger 输入；DS 的 ViT 输出）
#   预算 10k 训练(每类100)/2 epoch = 625 步（与之前 20k/3epoch 的 post 表分开记录）
set -u
cd /root/AssocVTC_tok448
export HF_HOME=/root/autodl-tmp/hf2 HF_ENDPOINT=https://hf-mirror.com SDPA_BACKEND=math
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-/root/autodl-tmp/tok256x12_t8/probe_vendor_pre}
mkdir -p "$OUT/logs" "$OUT/grids"
STAGE=${STAGE:-pre}
TPC=${TPC:-100}
EPOCHS=${EPOCHS:-2}
echo "[run] start $(date) stage=$STAGE train_per_class=$TPC epochs=$EPOCHS" > "$OUT/status.txt"
encs=(glm5_flash qwen38_flash mimo26_flash deepseek_v41_flash)
i=0
for e in "${encs[@]}"; do
  (
    MUSA_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i \
    timeout 10800 python3 probe_enc_dec_vendor.py --vendor "$e" --stage "$STAGE" \
      --data_dir /root/autodl-tmp/datasets/imagenet100 \
      --train_per_class "$TPC" --val_per_class 10 --epochs "$EPOCHS" \
      --batch 32 --width 256 --workers 8 \
      --save_grid "$OUT/grids/$e.png" --out "$OUT/$e.json" --tag "pre_10k_2ep" \
      > "$OUT/logs/$e.log" 2>&1
    echo "[run] $e rc=$? $(date)" >> "$OUT/status.txt"
  ) &
  i=$((i + 1))
done
wait
echo "[run] DONE $(date)" >> "$OUT/status.txt"
