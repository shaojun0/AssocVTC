#!/bin/bash
# ── s3-CE: 阶段3 损失换成「对 LlamaGen VQ-16 给原图输出的 196 个离散码的逐步 CE」──
# 其余对齐 v2.0 阶段3：zs_dim=224、T=16、3ep、等效 bs64、s2 热启动。
# 2026-11-27 加固: 周期 ckpt(300 步) + 崩溃自动 --resume（最多 5 次）。
set -u
cd /root/AssocVTC_tok448
ST=${ST:-/root/autodl-tmp/v20_224}
mkdir -p "$ST/s3"
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export SDPA_BACKEND=math
echo "[s3] wrapper start $(date -Is)" > "$ST/status_s3"
[ -f "$ST/s3.log" ] && mv "$ST/s3.log" "$ST/s3_prev_$(date +%Y%m%d_%H%M).log"
COMMON=(--data_dir /root/autodl-tmp/datasets/imagenet100 --output_dir "$ST/s3"
  --img_size 224 --patch 16 --num_specials 32
  --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16
  --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024
  --dec_recurrent 1 --step_plan fixed --block 2
  --dec_steps "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" --dec_grad_ckpt 1
  --zs_dim 224 --token_ce 1 --vq_ckpt /root/autodl-tmp/vq/llamagen_vq16_c2i.pt
  --vq_codebook_size 16384 --vq_embed_dim 8 --token_ce_init_pixel 1
  --epochs 3 --batch_size 64 --accum_steps 2
  --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --warmup_frac 0.08 --grad_clip 1.0
  --amp 1 --seed 0 --val_per_class 50 --val_batches 80 --log_interval 20
  --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 0 --save_opt 0 --num_workers 16
  --save_every_steps 300)
ATTEMPT=0
while [ "$ATTEMPT" -lt 5 ]; do
  ATTEMPT=$((ATTEMPT+1))
  if [ "$ATTEMPT" -eq 1 ]; then
    # ★2026-11-27: 已有 ckpt_last 时直接续跑, 避免 wrapper 重启把 s3 从头再训
    if [ -f "$ST/s3/ckpt_last.pt" ]; then INIT=(--resume "$ST/s3/ckpt_last.pt");
    else INIT=(--init_from "$ST/s2/ckpt_best.pt"); fi
  else INIT=(--resume "$ST/s3/ckpt_last.pt"); fi
  echo "[s3] attempt $ATTEMPT $(date -Is) ${INIT[*]}" >> "$ST/status_s3"
  MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
    torchrun --nproc_per_node=4 --master_port=$((29751+ATTEMPT)) train_linattn.py \
      "${COMMON[@]}" "${INIT[@]}" >> "$ST/s3.log" 2>&1
  RC=$?
  echo "[s3] attempt $ATTEMPT rc=$RC $(date -Is)" >> "$ST/status_s3"
  if [ "$RC" -eq 0 ]; then echo "[s3] rc=0 $(date -Is)" >> "$ST/status_s3"; exit 0; fi
  if [ ! -f "$ST/s3/ckpt_last.pt" ]; then
    echo "[s3] rc=$RC (no ckpt_last; abort) $(date -Is)" >> "$ST/status_s3"; exit "$RC"
  fi
  echo "[s3] attempt $ATTEMPT failed (MUSA/MCCL?); resume from ckpt_last in 30s" >> "$ST/status_s3"
  sleep 30
done
echo "[s3] rc=1 (retries exhausted) $(date -Is)" >> "$ST/status_s3"; exit 1
