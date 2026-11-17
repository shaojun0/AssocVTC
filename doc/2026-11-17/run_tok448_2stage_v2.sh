#!/bin/bash
# v2: SDPA_BACKEND=math（MUSA bf16 flash 反向数值坏 → 换 math 后端，见 doc/2026-11-17/REPORT_musa_instability.md）
set -u
cd /root/AssocVTC_tok448
ST=/root/autodl-tmp/tok448_v2
mkdir -p $ST
echo "[driver] start $(date)" > $ST/status
COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 448 --patch 32 --num_specials 16 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --dec_recurrent 0 --pixel_hidden 1024 \
 --batch_size 128 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 50 \
 --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8"
echo "[driver] stage1 $(date)" >> $ST/status
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched DIAG_GRAD=1 \
 torchrun --nproc_per_node=4 --master_port=29671 train_linattn.py $COMMON \
 --zs_dim 0 --epochs 45 --warmup_frac 0.08 --output_dir $ST/s1 > $ST/s1.log 2>&1
rc1=$?
echo "[driver] stage1 rc=$rc1 $(date)" >> $ST/status
if [ -f $ST/s1/ckpt_best.pt ]; then
  echo "[driver] stage2 $(date)" >> $ST/status
  SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched DIAG_GRAD=1 \
   torchrun --nproc_per_node=4 --master_port=29672 train_linattn.py $COMMON \
   --zs_dim 448 --epochs 3 --warmup_frac 0.02 --init_from $ST/s1/ckpt_best.pt \
   --output_dir $ST/s2 > $ST/s2.log 2>&1
  echo "[driver] stage2 rc=$? $(date)" >> $ST/status
else
  echo "[driver] stage1 ckpt missing, skip stage2" >> $ST/status
fi
echo "[driver] ALL_DONE $(date)" >> $ST/status
