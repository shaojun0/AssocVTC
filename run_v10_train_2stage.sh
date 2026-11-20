#!/bin/bash
# v10 两阶段训练（2026-11-19 用户版重构后重加）——
# 阶段 1：单步【全读】（对齐 v2 的 A 相 --warm_steps：一步读完 [z_cls; z_s]、
#         一步出整片、loss 覆盖全部帧）。
# 阶段 2：全 30 步渐进读出（每步只读当步 z_t，仍对全部帧算 loss，v2 直接预测口径）。
#
# 环境变量：S1E=2 S2E=2（epoch 数）、STEPGATE/MASK（默认 0=与上一轮 2stage 同口径）、
#           LR、OUT/NAME、PORT、INIT/RESUME
set -u
cd /root/AssocVTC_v10
export SDPA_BACKEND=math
export MUSA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
S1E=${S1E:-2}
S2E=${S2E:-2}
STEPGATE=${STEPGATE:-0}   # 0 = w≡1（v2 口径）；1 = 零初始化 γ 门控
MASK=${MASK:-0}           # 0 = 读出全双向（v2 口径）；1 = 帧级因果 mask
LR=${LR:-8e-4}
INIT=${INIT:-}
RESUME=${RESUME:-}
OUT=${OUT:-/root/autodl-tmp/tok448_v2/v10_120f_2stage_rf}
PORT=${PORT:-29555}
exec torchrun --nproc_per_node=4 --master_port=$PORT train_v10.py \
  --data_root /root/autodl-tmp/video_recon/data3 \
  --tok_ckpt /root/autodl-tmp/tok448_v2/s2/ckpt_best.pt \
  --latent_cache /root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train \
  --out "$OUT" ${RESUME:+--resume "$RESUME"} ${INIT:+--init_trainable "$INIT"} \
  --frames 120 --img 448 --group 4 \
  --rate_tokens 16 --rate_dim 448 --target_tokens 16 --chunk_tokens 2 \
  --mid_ratio 2 --mid_depth 6 --mid_heads 16 \
  --dec_ratio 2 --dec_depth 4 --dec_heads 16 \
  --step_gate "$STEPGATE" --readout_mask "$MASK" \
  --stage1_epochs "$S1E" --stage1_n_steps 1 --stage2_epochs "$S2E" \
  --batch 2 --accum_steps 2 \
  --lr "$LR" --weight_decay 0.05 --warmup_frac 0.02 --grad_clip 1.0 \
  --max_steps 8000 --amp 1 --log_every 20 \
  --val_every 200 --val_batches 2 --psnr_eval 1 \
  --save_every 500 --save_total_limit 2 --num_workers 6
