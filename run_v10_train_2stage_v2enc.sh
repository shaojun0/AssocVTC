#!/bin/bash
# v10 两阶段训练（2026-11-20 v2 同构编码器 middle）——
# 阶段 1：单步【全读】（对齐 v2 的 A 相：一步读完 [z_cls; z_s]、一步出整片、
#         loss 覆盖全部帧）。
# 阶段 2：全 30 步渐进读出（每步只读当步 z_t，仍对全部帧算 loss）。
#
# middle（v2enc，默认）：冻结 E 的 120×16×448 → reshape 480×1792，拼同尺寸
#         可学 query（可加 cls）→ 960 token 全双向自注意力 → 取 query 半
#         reshape 回 (B,120,8,896) 给读出；读出侧一行不改。
#
# 环境变量：S1E=2 S2E=2（epoch 数）、STEPGATE/MASK（默认 0=上一轮同口径）、
#           LR、MIDDEPTH=2 MIDRATIO=2 MIDCLS=0（middle 变体开关）、
#           OUT/NAME、PORT、INIT/RESUME、MAXSTEPS（0=按 epoch 反推）
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
MIDDEPTH=${MIDDEPTH:-2}   # v2enc 编码器层数（算力：2 层 ≈ 旧 crossq 6 层）
MIDRATIO=${MIDRATIO:-2}
MIDCLS=${MIDCLS:-0}       # 1 = middle 加 1 个 cls（输出广播加到所有 rate 行）
MAXSTEPS=${MAXSTEPS:-8000}
INIT=${INIT:-}
RESUME=${RESUME:-}
OUT=${OUT:-/root/autodl-tmp/tok448_v2/v10_120f_2stage_v2enc}
PORT=${PORT:-29556}
exec torchrun --nproc_per_node=4 --master_port=$PORT train_v10.py \
  --data_root /root/autodl-tmp/video_recon/data3 \
  --tok_ckpt /root/autodl-tmp/tok448_v2/s2/ckpt_best.pt \
  --latent_cache /root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train \
  --out "$OUT" ${RESUME:+--resume "$RESUME"} ${INIT:+--init_trainable "$INIT"} \
  --frames 120 --img 448 --group 4 \
  --rate_tokens 16 --rate_dim 448 --target_tokens 16 --chunk_tokens 2 \
  --mid_style v2enc --mid_hidden 1792 --mid_depth "$MIDDEPTH" \
  --mid_heads 16 --mid_ratio "$MIDRATIO" --mid_cls "$MIDCLS" \
  --dec_ratio 2 --dec_depth 4 --dec_heads 16 \
  --step_gate "$STEPGATE" --readout_mask "$MASK" \
  --stage1_epochs "$S1E" --stage2_epochs "$S2E" \
  --batch 2 --accum_steps 2 \
  --lr "$LR" --weight_decay 0.05 --warmup_frac 0.02 --grad_clip 1.0 \
  --max_steps "$MAXSTEPS" --amp 1 --log_every 20 \
  --val_every 200 --val_batches 2 --psnr_eval 1 \
  --save_every 500 --save_total_limit 2 --num_workers 6
