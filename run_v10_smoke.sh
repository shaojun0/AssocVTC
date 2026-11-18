#!/bin/bash
# v10 4 卡 DDP 冒烟：验 MUSA DDP 补丁 / 缓存读取 / PSNR eval / ckpt 存取 / 续训
set -u
cd /root/AssocVTC_v10
export SDPA_BACKEND=math
export MUSA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=6
OUT=${OUT:-/root/autodl-tmp/tok448_v2/v10_smoke_ddp}
exec torchrun --nproc_per_node=4 --master_port=29551 train_v10.py \
  --data_root /root/autodl-tmp/video_recon/data3 \
  --tok_ckpt /root/autodl-tmp/tok448_v2/s2/ckpt_best.pt \
  --latent_cache /root/autodl-tmp/tok448_v2/latent_cache/smoke96 \
  --out "$OUT" \
  --frames 120 --img 448 --group 4 \
  --rate_tokens 16 --rate_dim 448 --target_tokens 16 --chunk_tokens 2 \
  --mid_ratio 2 --mid_depth 6 --mid_heads 16 \
  --dec_ratio 2 --dec_depth 4 --dec_heads 16 \
  --step_gate 1 --step_weights mean \
  --batch 2 --accum_steps 4 \
  --lr 2e-4 --weight_decay 0.05 --warmup_frac 0.05 --grad_clip 1.0 \
  --max_steps 6 --amp 1 --log_every 1 \
  --val_every 2 --val_batches 2 --psnr_eval 1 \
  --save_every 2 --save_total_limit 2 --num_workers 4 --limit 64
