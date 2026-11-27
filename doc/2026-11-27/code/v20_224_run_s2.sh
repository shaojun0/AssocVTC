#!/bin/bash
# ── s2: T=16 循环读出（fixed/block2），像素 L1，6 ep，等效 bs64，从 s1 热启动 ──
# 2026-11-27 加固: ①--save_every_steps 300 周期存 ckpt_last；②崩溃自动 --resume 续跑（最多 5 次）；
#   ③accum 2（micro 8）降低显存压力。等效批量/步数不变（steps/epoch 仍 1,901）。
set -u
cd /root/AssocVTC_tok448
ST=${ST:-/root/autodl-tmp/v20_224}
mkdir -p "$ST/s2"
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export SDPA_BACKEND=math
echo "[s2] wrapper start $(date -Is)" > "$ST/status_s2"
[ -f "$ST/s2.log" ] && mv "$ST/s2.log" "$ST/s2_prev_$(date +%Y%m%d_%H%M).log"
COMMON=(--data_dir /root/autodl-tmp/datasets/imagenet100 --output_dir "$ST/s2"
  --img_size 224 --patch 16 --num_specials 32
  --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16
  --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024
  --dec_recurrent 1 --step_plan fixed --block 2
  --dec_steps "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" --dec_grad_ckpt 1
  --epochs 6 --batch_size 64 --accum_steps 2
  --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --warmup_frac 0.08 --grad_clip 1.0
  --amp 1 --seed 0 --val_per_class 50 --val_batches 80 --log_interval 50
  --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 0 --save_opt 0 --num_workers 16
  --save_every_steps 300)
ATTEMPT=0
while [ "$ATTEMPT" -lt 5 ]; do
  ATTEMPT=$((ATTEMPT+1))
  if [ "$ATTEMPT" -eq 1 ]; then
    # ★2026-11-27: 若已有 ckpt_last（崩溃后续跑/chain 重启）, 直接续跑而不是从 s1 重头开始
    if [ -f "$ST/s2/ckpt_last.pt" ]; then INIT=(--resume "$ST/s2/ckpt_last.pt");
    else INIT=(--init_from "$ST/s1/ckpt_best.pt"); fi
  else INIT=(--resume "$ST/s2/ckpt_last.pt"); fi
  echo "[s2] attempt $ATTEMPT $(date -Is) ${INIT[*]}" >> "$ST/status_s2"
  MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
    torchrun --nproc_per_node=4 --master_port=$((29741+ATTEMPT)) train_linattn.py \
      "${COMMON[@]}" "${INIT[@]}" >> "$ST/s2.log" 2>&1
  RC=$?
  echo "[s2] attempt $ATTEMPT rc=$RC $(date -Is)" >> "$ST/status_s2"
  if [ "$RC" -eq 0 ]; then echo "[s2] rc=0 $(date -Is)" >> "$ST/status_s2"; exit 0; fi
  if [ ! -f "$ST/s2/ckpt_last.pt" ]; then
    echo "[s2] rc=$RC (no ckpt_last; abort) $(date -Is)" >> "$ST/status_s2"; exit "$RC"
  fi
  sleep 30
done
echo "[s2] rc=1 (retries exhausted) $(date -Is)" >> "$ST/status_s2"; exit 1
