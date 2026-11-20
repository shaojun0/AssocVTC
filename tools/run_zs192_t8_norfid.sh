#!/bin/bash
# 8-step recurrent decode (K=16, T=8 cumulative), zs_dim=192, NO rFID loss (λ=0 control).
# Purpose: decouple "8-step recurrent itself fails" from "rFID λ=1.0 hurts" (2026-11-20 run).
# Everything else identical to run_zs192_t8rfid.sh (same s1 init, same SVD rank-192 warm start,
# same 3ep / lr 2e-4 / bs 128 / accum 2 → micro 16/GPU).
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
ST=/root/autodl-tmp/tok448_zs192_t8_norfid
mkdir -p "$ST"
echo "[driver] start $(date)" > "$ST/status"
COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 448 --patch 32 --num_specials 16 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 2 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 50 \
 --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8"
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=29695 train_linattn.py $COMMON \
 --zs_dim 192 --epochs 3 --warmup_frac 0.02 \
 --init_from /root/autodl-tmp/tok448_v2/s1/ckpt_best.pt \
 --dec_recurrent 1 --step_plan cumulative --dec_steps "1,2,3,5,7,10,13,16" \
 --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
rc=$?
echo "[driver] stage2 rc=$rc $(date)" >> "$ST/status"
if [ -f "$ST/s2/ckpt_best.pt" ]; then
  python3 fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --out "$ST/zs192_t8_norfid_fast.json" \
    --tag zs192_t8_norfid --batch_size 64 --amp 1 > "$ST/fast8.log" 2>&1
  echo "[driver] fast_rfid rc=$? $(date)" >> "$ST/status"
else
  echo "[driver] no ckpt_best, skip eval $(date)" >> "$ST/status"
fi
echo "[driver] DONE $(date)" >> "$ST/status"
