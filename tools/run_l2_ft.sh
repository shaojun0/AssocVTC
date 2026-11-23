#!/bin/bash
# ── L2(MSE) 续训：256 token × 12 dim @256²，从 ep14 L1 ckpt 出发，续训 3 epoch ─────
# 目的：重建损失由逐像素 L1 改为 MSE（直接优化 PSNR），看 rFID / PSNR 怎么变。
# 每个 epoch 额外存 ckpt_ep{N}.pt 并逐 epoch 跑 8 步 fast_rfid（5000 张 IN-100 val）。
# 用法: bash run_l2_ft.sh   （EPOCHS/LR/SRC 可环境变量覆盖）
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
SRC=${SRC:-/root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt}
ST=${ST:-/root/autodl-tmp/tok256x12_t8/l2ft}
TAG=${TAG:-l2ft}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-5}
BUDGET=${BUDGET:-4.5}
PORT=${PORT:-29706}
mkdir -p "$ST/s2"
echo "[l2ft] start $(date) src=$SRC epochs=$EPOCHS lr=$LR loss=l2" > "$ST/status"

# 等 GPU 空出来（最多 10 分钟；仍占用则退出，避免两个训练抢卡）
for i in $(seq 1 20); do
  n=$(pgrep -f "train_linattn.py" 2>/dev/null | wc -l)
  [ "${n:-0}" -eq 0 ] && break
  sleep 30
done
n=$(pgrep -f "train_linattn.py" 2>/dev/null | wc -l)
if [ "${n:-0}" -ne 0 ]; then echo "[l2ft] ABORT: GPU busy ($n procs)" >> "$ST/status"; exit 1; fi

COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 256 --patch 16 --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 4 --lr $LR --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 25 \
 --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 1 --save_opt 0 --num_workers 8 \
 --zs_dim 12 --allow_k_gt_n 1 --recon_loss l2 \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256"

SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=$PORT train_linattn.py $COMMON \
 --epochs $EPOCHS --warmup_frac 0.02 --time_budget_h $BUDGET \
 --init_from "$SRC" --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
echo "[l2ft] train rc=$? $(date)" >> "$ST/status"
grep -E "\[val\] ep|\[done\]" "$ST/s2.log" | tail -6 >> "$ST/status" 2>/dev/null

# 逐 epoch + best 的 8 步 rFID
for ck in "$ST"/s2/ckpt_ep*.pt; do
  [ -f "$ck" ] || continue
  t=$(basename "$ck" .pt)
  python3 fast_rfid.py --ckpt "$ck" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --out "$ST/${t}_fast.json" --tag "$t" --batch_size 64 --amp 1 \
    > "$ST/fast_${t}.log" 2>&1
  echo "[l2ft] eval $t rc=$? $(date)" >> "$ST/status"
done
if [ -f "$ST/s2/ckpt_best.pt" ]; then
  python3 fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --out "$ST/ckpt_best_fast.json" --tag l2_best --batch_size 64 --amp 1 \
    > "$ST/fast_best.log" 2>&1
  echo "[l2ft] eval best rc=$? $(date)" >> "$ST/status"
fi
echo "[l2ft] DONE $(date)" >> "$ST/status"
