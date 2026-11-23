#!/bin/bash
# PatchGAN 冒烟：18 步、4 卡、不存 ckpt、不跑 val，验证判别器前/反向 + 手动梯度同步 + 日志
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
ST=/root/autodl-tmp/gan_smoke
rm -rf "$ST"; mkdir -p "$ST"
COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 256 --patch 16 --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 4 --lr 2e-5 --min_lr 2e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 0 --log_interval 2 \
 --sample_interval 0 --save_every_epochs 0 --save_opt 0 --num_workers 4 \
 --zs_dim 12 --allow_k_gt_n 1 --limit 2400 \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256"
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 timeout 900 torchrun --nproc_per_node=4 --master_port=29709 train_linattn.py $COMMON \
 --epochs 3 --warmup_frac 0.02 --stop_at_step 18 \
 --init_from /root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt \
 --gan_weight 0.1 --gan_ndf 64 --gan_layers 3 --gan_d_lr 1e-4 \
 --gan_start_frac 0.0 --gan_ramp_frac 0.1 \
 --output_dir "$ST/s2" 2>&1 | tail -40
