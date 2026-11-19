#!/bin/bash
# v10 正式训练 —— 循环读出（全量 32k 片段，4 卡）。
#
# 环境变量（都可覆盖）：
#   INIT=<best_trainable.pt>  课程学习：从已拟合权重初始化
#   LR / STEPS / OUT           学习率 / 步数 / 输出目录
#
# 例：OUT=.../v10_120f_step \
#       INIT=.../v10_120f/best_trainable.pt bash run_v10_train_step.sh
set -u
cd /root/AssocVTC_v10
export SDPA_BACKEND=math
export MUSA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
STEPGATE=${STEPGATE:-0}   # 0 = w≡1（v2 口径）; 1 = 零初始化 γ 门控（旧版）
MASK=${MASK:-0}           # 0 = 读出全双向（v2 口径）; 1 = 旧版帧级因果 mask
LR=${LR:-8e-4}
STEPS=${STEPS:-4000}
INIT=${INIT:-}
RESUME=${RESUME:-}
OUT=${OUT:-/root/autodl-tmp/tok448_v2/v10_120f_step}
PORT=${PORT:-29554}
exec torchrun --nproc_per_node=4 --master_port=$PORT train_v10.py \
  --data_root /root/autodl-tmp/video_recon/data3 \
  --tok_ckpt /root/autodl-tmp/tok448_v2/s2/ckpt_best.pt \
  --latent_cache /root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train \
  --out "$OUT" ${RESUME:+--resume "$RESUME"} ${INIT:+--init_trainable "$INIT"} \
  --frames 120 --img 448 --group 4 \
  --rate_tokens 16 --rate_dim 448 --target_tokens 16 --chunk_tokens 2 \
  --mid_ratio 2 --mid_depth 6 --mid_heads 16 \
  --dec_ratio 2 --dec_depth 4 --dec_heads 16 \
  --step_gate "$STEPGATE" \
  --readout_mask "$MASK" \
  --batch 2 --accum_steps 2 \
  --lr "$LR" --weight_decay 0.05 --warmup_frac 0.02 --grad_clip 1.0 \
  --max_steps "$STEPS" --amp 1 --log_every 20 \
  --val_every 200 --val_batches 2 --psnr_eval 1 \
  --save_every 500 --save_total_limit 2 --num_workers 6
