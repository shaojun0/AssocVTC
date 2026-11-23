#!/bin/bash
# 8-step recurrent decode (K=16, T=8 cumulative) + batch rFID auxiliary loss, zs_dim=192.
# Reuses tok448_v2/s1 (zs_dim-independent full-width stage-1) + SVD bottleneck warm start.
# accum_steps=2 => micro 16/GPU (8-step BPTT OOMs at micro 32); global batch stays 128, 950 steps/epoch.
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
ST=/root/autodl-tmp/tok448_zs192_t8rfid
mkdir -p "$ST"
echo "[driver] start $(date)" > "$ST/status"
COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 448 --patch 32 --num_specials 16 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 2 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 50 \
 --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8"
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=29694 train_linattn.py $COMMON \
 --zs_dim 192 --epochs 3 --warmup_frac 0.02 \
 --init_from /root/autodl-tmp/tok448_v2/s1/ckpt_best.pt \
 --dec_recurrent 1 --step_plan cumulative --dec_steps "1,2,3,5,7,10,13,16" \
 --rfid_weight 1.0 --rfid_dim 2048 --rfid_every 1 \
 --rfid_start_frac 0.1 --rfid_ramp_frac 0.1 --rfid_norm_dim 1 \
 --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
rc=$?
echo "[driver] stage2 rc=$rc $(date)" >> "$ST/status"
echo "[driver] DONE rc=$rc $(date)" >> "$ST/status"
