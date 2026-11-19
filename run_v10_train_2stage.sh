#!/bin/bash
# v10 两阶段训练（2026-11-19）—— 阶段1 单步预热（n_steps=1）→ 阶段2 全 30 步循环。
#
# 动机：单步阶段只读第一组 z、只监督前 group 帧（copy_skip 下这条 loss 只训练
# middle 的「抄写」映射 + 前段 z_cls），把通路从均值平凡解里拽出来；随后切回
# 30 步深监督（全 30 档、只对已读帧算 loss）。阶段切换由 train_v10.py 的
# V10Trainer.compute_loss 按 global_step 完成，单次 run 内自动切换。
#
# 环境变量（都可覆盖）：
#   S1E=2      阶段1 epoch 数（n_steps=1 单步解码；0=关）
#   S1NS=1     阶段1 的 n_steps（1 = 只读第一组 z）
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
  --stage1_epochs "$S1E" --stage1_n_steps "$S1NS" --stage2_epochs "$S2E" \
  --batch 2 --accum_steps 2 \
  --lr "$LR" --weight_decay 0.05 --warmup_frac 0.02 --grad_clip 1.0 \
  --max_steps 8000 --amp 1 --log_every 20 \
  --val_every 200 --val_batches 2 --psnr_eval 1 \
  --save_every 500 --save_total_limit 2 --num_workers 6
