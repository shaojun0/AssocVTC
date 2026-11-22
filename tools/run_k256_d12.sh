#!/bin/bash
# ── 256 tokens × 12 dim (=3072 values/图) + 8 步循环解码（循环架构不变）───────────────
# 目的：对齐 ALIT 的 rate point 与分辨率（256²），看我们能否逼近/超过 ALIT 的 rFID。
# 8 步 cumulative = 32,64,...,256 token ⇒ 每一步恰是 ALIT Table 3 的一个点（32..256 token）。
# 阶段 1：从零训练，纯逐像素 L1（14 epoch ≈ 18.9h）
# 阶段 2：从阶段 1 的 best 继续，L1 + 小权重 batch rFID（λ=0.05/2048）2.4h，看能否压低 rFID
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
ST=/root/autodl-tmp/tok256x12_t8
mkdir -p "$ST"
echo "[driver] start $(date)" > "$ST/status"
COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 256 --patch 16 --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 4 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 25 \
 --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8 \
 --zs_dim 12 --allow_k_gt_n 1 \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256"

echo "[driver] === phase1: from scratch, L1, 14ep ===" >> "$ST/status"
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=29703 train_linattn.py $COMMON \
 --epochs 14 --warmup_frac 0.06 --time_budget_h 19.5 \
 --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
echo "[driver] phase1 rc=$? $(date)" >> "$ST/status"

echo "[driver] === eval phase1 ===" >> "$ST/status"
python3 fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --data_dir /root/autodl-tmp/datasets/imagenet100 \
  --img_size 256 --out "$ST/t8_l1_ep14_fast.json" --tag t8_l1_ep14 --batch_size 64 --amp 1 \
  > "$ST/fast_l1.log" 2>&1
echo "[driver] eval phase1 rc=$? $(date)" >> "$ST/status"

echo "[driver] === phase2: finetune L1 + rFID(l=0.05), 2.4h ===" >> "$ST/status"
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=29704 train_linattn.py $COMMON \
 --epochs 2 --warmup_frac 0.02 --time_budget_h 2.4 --lr 4e-5 --min_lr 4e-6 \
 --init_from "$ST/s2/ckpt_best.pt" \
 --rfid_weight 0.05 --rfid_dim 2048 --rfid_every 1 \
 --rfid_start_frac 0.0 --rfid_ramp_frac 0.05 --rfid_norm_dim 1 \
 --output_dir "$ST/s2ft" > "$ST/s2ft.log" 2>&1
echo "[driver] phase2 rc=$? $(date)" >> "$ST/status"

python3 fast_rfid.py --ckpt "$ST/s2ft/ckpt_best.pt" --data_dir /root/autodl-tmp/datasets/imagenet100 \
  --img_size 256 --out "$ST/t8_ft_ep2_fast.json" --tag t8_ft_ep2 --batch_size 64 --amp 1 \
  > "$ST/fast_ft.log" 2>&1
echo "[driver] eval phase2 rc=$? $(date)" >> "$ST/status"
echo "[driver] DONE $(date)" >> "$ST/status"
