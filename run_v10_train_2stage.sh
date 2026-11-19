#!/bin/bash
# v10 两阶段训练（2026-11-19）—— 阶段1 单步预热（n_steps=1）→ 阶段2 全 30 步循环。
#
# 动机（对齐 v2 训练的 A 相）：v2 的 --warm_steps 把 decoder.steps 设成 [K]
# ——**一步读完全部 [z_cls; z_s]、一步出整图、loss 覆盖全部位置**。v10 对应
# 口径 = n_steps=1 + memory=整段 30 组 z（960 行）+ loss 覆盖全部 120 帧，
# 先把「整段 z → 整段 special/z_cls」的通路从均值平凡解里拽出来；随后切回
# 30 步渐进读出（每步只读当步 z_t、loss 只算已读帧）。阶段切换由
# train_v10.py 的 V10Trainer.compute_loss 按 global_step 完成，单次 run 内自动切换。
#
# 环境变量（都可覆盖）：
#   S1E=2      阶段1 epoch 数（单步全读；0=关）
#   S1MODE=full 阶段1 口径：full = 单步但一次读完整段 z、一步出全片（v2 A 相，默认）；
#               first = 只读第一组 z、只监督前 group 帧（旧口径，仅对照用）
#   S1NS=1     阶段1 的 n_steps（1 = 单步）
#   S2E=2      阶段2 epoch 数（全 30 步循环）
#   MEM=step|prefix  COPY=1|0  LOSSREAD=1|0  其余同 run_v10_train_step.sh
#   LR / OUT / NAME / PORT / INIT / RESUME
#
# 例：S1E=2 S2E=2 OUT=/root/autodl-tmp/tok448_v2/v10_120f_2stage \
#       bash run_v10_train_2stage.sh
set -u
cd /root/AssocVTC_v10
export SDPA_BACKEND=math
export MUSA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
S1E=${S1E:-2}
S1MODE=${S1MODE:-full}
S1NS=${S1NS:-1}
S2E=${S2E:-2}
MEM=${MEM:-step}
COPY=${COPY:-1}
LOSSREAD=${LOSSREAD:-1}
LFRAMENORM=${LFRAMENORM:-0} # 1 = loss_read 时按帧归一化（frame0 vs frame119 权重拉平）
STEPGATE=${STEPGATE:-0}   # 0 = w≡1（v2 口径）; 1 = 零初始化 γ 门控（旧版）
CARRYLN=${CARRYLN:-0}     # 0 = carry 无 LN（v2 口径）; 1 = 旧版手写 LN
MASK=${MASK:-0}           # 0 = 读出全双向（v2 口径）; 1 = 旧版帧级因果 mask
MIDMODE=${MIDMODE:-packed} # packed = 帧因果 TransformerEncoder; crossq = 双向交叉查询（实验 A）
LR=${LR:-8e-4}
INIT=${INIT:-}
RESUME=${RESUME:-}
OUT=${OUT:-/root/autodl-tmp/tok448_v2/v10_120f_2stage}
PORT=${PORT:-29561}
exec torchrun --nproc_per_node=4 --master_port=$PORT train_v10.py \
  --data_root /root/autodl-tmp/video_recon/data3 \
  --tok_ckpt /root/autodl-tmp/tok448_v2/s2/ckpt_best.pt \
  --latent_cache /root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train \
  --out "$OUT" ${RESUME:+--resume "$RESUME"} ${INIT:+--init_trainable "$INIT"} \
  --frames 120 --img 448 --group 4 \
  --rate_tokens 16 --rate_dim 448 --target_tokens 16 --chunk_tokens 2 \
  --mid_ratio 2 --mid_depth 6 --mid_heads 16 --mid_mode "$MIDMODE" \
  --dec_ratio 2 --dec_depth 4 --dec_heads 16 \
  --step_gate "$STEPGATE" --step_weights mean \
  --carry_ln "$CARRYLN" --readout_mask "$MASK" \
  --mem_mode "$MEM" --loss_read "$LOSSREAD" --loss_frame_norm "$LFRAMENORM" \
  --copy_skip "$COPY" \
  --stage1_epochs "$S1E" --stage1_mode "$S1MODE" \
  --stage1_n_steps "$S1NS" --stage2_epochs "$S2E" \
  --batch 2 --accum_steps 2 \
  --lr "$LR" --weight_decay 0.05 --warmup_frac 0.02 --grad_clip 1.0 \
  --max_steps 8000 --amp 1 --log_every 20 \
  --val_every 200 --val_batches 2 --psnr_eval 1 \
  --save_every 500 --save_total_limit 2 --num_workers 6
