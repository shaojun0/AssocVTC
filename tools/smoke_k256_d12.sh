#!/bin/bash
# Smoke: K=256 specials x zs_dim=12 (=3072 values), 256^2, patch 16 (N=256), 8-step cumulative
# decode (32 tokens/step => each step == one ALIT rate point 32,64,...,256 tokens).
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
ST=/root/autodl-tmp/tok256x12_smoke
mkdir -p "$ST"
echo "[smoke] start $(date)" > "$ST/status"
COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size ${IMG:-256} --patch ${PATCH:-16} --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps ${ACCUM:-4} --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 2 --val_batches 1 --log_interval 2 \
 --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8 \
 --zs_dim 12 --epochs 1 --warmup_frac 0.02 --max_steps ${MAXSTEPS:-24} \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256 \
 --allow_k_gt_n ${ALLOWKN:-1} --dec_grad_ckpt ${GC:-0}"
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=29701 train_linattn.py $COMMON \
 --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
rc=$?
echo "[smoke] rc=$rc $(date)" >> "$ST/status"
echo "[smoke] DONE $(date)" >> "$ST/status"
