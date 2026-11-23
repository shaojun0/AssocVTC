#!/bin/bash
# ── SD-VAE 输出蒸馏续训：256 token × 12 dim @256²，从 ep14 L1 ckpt 出发 ────────────
# 目的（见 PLAN_distill_20261123.md / P0-c 决策门）：把 SD-VAE(sd-vae-ft-mse, 4096 values/img,
# PSNR 23.08 / rFID 15.7 @5000 张) 的重建当**监督目标**，把它的解码器先验装进我们的主解码器：
#     loss = L1(8 步, x) + λ·L1(末步 F_hat, teacher(x))
# 我们当前: PSNR 22.67 / rFID 86.50（同 5000 张口径）。
# 每个 epoch 额外存 ckpt_ep{N}.pt 并逐 epoch 跑 8 步 fast_rfid。
# 用法: bash run_kd_ft.sh                    # 正式 2 epoch
#       SMOKE=60 bash run_kd_ft.sh            # 只跑 60 步，验证 KD 分支通了没
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
SRC=${SRC:-/root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt}
ST=${ST:-/root/autodl-tmp/tok256x12_t8/kd1}
TAG=${TAG:-kd1}
EPOCHS=${EPOCHS:-2}
LR=${LR:-2e-5}
BUDGET=${BUDGET:-2.8}
PORT=${PORT:-29707}
KDW=${KDW:-1.0}
SMOKE=${SMOKE:-0}
mkdir -p "$ST/s2"
echo "[kd] start $(date) src=$SRC epochs=$EPOCHS lr=$LR kd_weight=$KDW smoke=$SMOKE" > "$ST/status"

if [ "$SMOKE" = "0" ]; then
  # 等 GPU 空出来（最多 10 分钟；仍占用则退出，避免两个训练抢卡）
  for i in $(seq 1 20); do
    n=$(pgrep -f "train_linattn.py" 2>/dev/null | wc -l)
    [ "${n:-0}" -eq 0 ] && break
    sleep 30
  done
  n=$(pgrep -f "train_linattn.py" 2>/dev/null | wc -l)
  if [ "${n:-0}" -ne 0 ]; then echo "[kd] ABORT: GPU busy ($n procs)" >> "$ST/status"; exit 1; fi
fi

COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 256 --patch 16 --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 4 --lr $LR --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 10 \
 --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 1 --save_opt 0 --num_workers 8 \
 --zs_dim 12 --allow_k_gt_n 1 \
 --kd_weight $KDW --kd_teacher stabilityai/sd-vae-ft-mse --kd_local_only 1 --kd_loss l1 \
 --kd_start_frac 0.02 --kd_ramp_frac 0.03 \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256"

EXTRA=""
if [ "$SMOKE" != "0" ]; then EXTRA="--stop_at_step $SMOKE"; fi

SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=$PORT train_linattn.py $COMMON $EXTRA \
 --epochs $EPOCHS --warmup_frac 0.02 --time_budget_h $BUDGET \
 --init_from "$SRC" --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
echo "[kd] train rc=$? $(date)" >> "$ST/status"

if [ "$SMOKE" != "0" ]; then
  grep -E "\[kd\]|kd=|Traceback|Error" "$ST/s2.log" | tail -12 >> "$ST/status"
  echo "[kd] SMOKE DONE $(date)" >> "$ST/status"
  exit 0
fi

grep -E "\[val\] ep|\[done\]" "$ST/s2.log" | tail -8 >> "$ST/status" 2>/dev/null

# 逐 epoch + best 的 8 步 rFID
for ck in "$ST"/s2/ckpt_ep*.pt; do
  [ -f "$ck" ] || continue
  t=$(basename "$ck" .pt)
  python3 fast_rfid.py --ckpt "$ck" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --out "$ST/${t}_fast.json" --tag "$t" --batch_size 64 --amp 1 \
    > "$ST/fast_${t}.log" 2>&1
  echo "[kd] eval $t rc=$? $(date)" >> "$ST/status"
done
if [ -f "$ST/s2/ckpt_best.pt" ]; then
  python3 fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --out "$ST/ckpt_best_fast.json" --tag kd_best --batch_size 64 --amp 1 \
    > "$ST/fast_best.log" 2>&1
  echo "[kd] eval best rc=$? $(date)" >> "$ST/status"
fi
echo "[kd] DONE $(date)" >> "$ST/status"
