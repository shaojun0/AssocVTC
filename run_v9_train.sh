#!/bin/bash
# v9 正式训练（并行 / 前缀读出）：32k 片段 × 120 帧(15s@8fps) × 448²，
# 率 16×448 = 7,168 值/帧（860,160 值/片段）。4×MTT S4000，bf16，DDP(mccl)+SDPA math。
# 输入是离线缓存的 (z_cls, z_s)（build_latent_cache.py，与 v10 共用）。
# 2026-11-18 用这条配方训练出 eval_psnr = 27.98 dB（上界 = 冻结分词器满读 30.53）。
#   nohup bash run_v9_train.sh > logs/v9_train.log 2>&1 &
set -u
cd /root/AssocVTC_v10
export SDPA_BACKEND=math
export MUSA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
OUT=${OUT:-/root/autodl-tmp/tok448_v2/v9_120f}
RESUME=${RESUME:-}
exec torchrun --nproc_per_node=4 --master_port=29553 train_v9.py \
  --data_root /root/autodl-tmp/video_recon/data3 \
  --tok_ckpt /root/autodl-tmp/tok448_v2/s2/ckpt_best.pt \
  --latent_cache /root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train \
  --out "$OUT" ${RESUME:+--resume "$RESUME"} \
  --frames 120 --img 448 --group 4 \
  --rate_tokens 16 --rate_dim 448 --target_tokens 16 --chunk_tokens 2 \
  --mid_ratio 2 --mid_depth 6 --mid_heads 16 \
  --dec_ratio 2 --dec_depth 4 --dec_heads 16 \
  --step_gate 1 --carry_ln 1 --readout_mask 1 --step_weights mean \
  --loss_read 1 --copy_skip 1 \
  --batch 2 --accum_steps 2 \
  --lr 8e-4 --weight_decay 0.05 --warmup_frac 0.02 --grad_clip 1.0 \
  --max_steps 4000 --amp 1 --log_every 20 \
  --val_every 200 --val_batches 2 --psnr_eval 1 \
  --save_every 500 --save_total_limit 2 --num_workers 6
