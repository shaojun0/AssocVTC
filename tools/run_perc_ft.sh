#!/bin/bash
# 阶段 3：从现有 ckpt 出发做 VGG16 感知损失微调（目标：压低 rFID，允许 PSNR 略降）。
# 用法: SRC=<ckpt> TAG=<name> PW=0.15 BUDGET=3.5 LR=6e-5 bash run_perc_ft.sh
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
SRC=${SRC:-/root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt}
TAG=${TAG:-perc}
ST=${ST:-/root/autodl-tmp/tok256x12_t8/$TAG}
PW=${PW:-0.15}
BUDGET=${BUDGET:-3.5}
LR=${LR:-6e-5}
RFID=${RFID:-0}
PORT=${PORT:-29705}
EXTRA=""
if [ "$RFID" != "0" ]; then
  EXTRA="--rfid_weight $RFID --rfid_dim 2048 --rfid_every 1 --rfid_start_frac 0.0 --rfid_ramp_frac 0.05 --rfid_norm_dim 1"
fi
mkdir -p "$ST"
echo "[percft] start $(date) src=$SRC w=$PW budget=${BUDGET}h lr=$LR" > "$ST/status"
# 等 GPU 空出来（最多 10 分钟；仍占用则退出，避免两个训练抢卡）
for i in $(seq 1 20); do
  n=$(pgrep -f "train_linattn.py" 2>/dev/null | wc -l)
  [ "${n:-0}" -eq 0 ] && break
  sleep 30
done
n=$(pgrep -f "train_linattn.py" 2>/dev/null | wc -l)
if [ "${n:-0}" -ne 0 ]; then echo "[percft] ABORT: GPU busy ($n procs)" >> "$ST/status"; exit 1; fi
COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 256 --patch 16 --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 4 --lr $LR --min_lr 6e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 25 \
 --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8 \
 --zs_dim 12 --allow_k_gt_n 1 \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256"
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=$PORT train_linattn.py $COMMON \
 --epochs 4 --warmup_frac 0.02 --time_budget_h $BUDGET \
 --init_from "$SRC" --perc_weight $PW --perc_layer 3,8,15 --perc_resize 256 $EXTRA \
 --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
echo "[percft] rc=$? $(date)" >> "$ST/status"
python3 fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --data_dir /root/autodl-tmp/datasets/imagenet100 \
  --img_size 256 --out "$ST/${TAG}_fast.json" --tag "$TAG" --batch_size 64 --amp 1 > "$ST/fast.log" 2>&1
echo "[percft] eval rc=$? $(date)" >> "$ST/status"
echo "[percft] DONE $(date)" >> "$ST/status"
