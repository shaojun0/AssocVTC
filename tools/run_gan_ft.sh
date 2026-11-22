#!/bin/bash
# PatchGAN 对抗微调：从 L1 ckpt 出发，L1 + hinge PatchGAN（VQGAN 风格），目标压 rFID。
# 用法: SRC=<ckpt> TAG=gan1 GW=0.8 BUDGET=1.5 LR=3e-5 DLR=1e-4 bash run_gan_ft.sh
# 注意: GW 是**梯度比目标**（自适应 λ = GW·‖∂L_rec/∂w‖/‖∂L_adv/∂w‖，VQGAN 取 0.8），
#      不是损失权重；固定权重语（--gan_adaptive 0）下 GW 才当权重用。
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
SRC=${SRC:-/root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt}
TAG=${TAG:-gan1}
ST=${ST:-/root/autodl-tmp/tok256x12_t8/$TAG}
GW=${GW:-0.8}
BUDGET=${BUDGET:-1.5}
LR=${LR:-3e-5}
DLR=${DLR:-1e-4}
SNAP_MIN=${SNAP_MIN:-20}
PORT=${PORT:-29707}
mkdir -p "$ST/snaps"
echo "[ganft] start $(date) src=$SRC gw=$GW budget=${BUDGET}h lr=$LR d_lr=$DLR" > "$ST/status"
# 等 GPU 空出来（最多 10 分钟；仍占用则退出）
for i in $(seq 1 20); do
  n=$(pgrep -f "train_linattn.py" 2>/dev/null | wc -l)
  [ "${n:-0}" -eq 0 ] && break
  sleep 30
done
n=$(pgrep -f "train_linattn.py" 2>/dev/null | wc -l)
if [ "${n:-0}" -ne 0 ]; then echo "[ganft] ABORT: GPU busy ($n procs)" >> "$ST/status"; exit 1; fi

COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 256 --patch 16 --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 4 --lr $LR --min_lr 2e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 25 \
 --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8 \
 --zs_dim 12 --allow_k_gt_n 1 \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256"

# 快照：训练期间每 SNAP_MIN 分钟存一份 ckpt_last（用标志文件判断训练是否还在，
# 不能用 pgrep -f "train_linattn.py" —— 子 shell 自己的 cmdline 含该串 ⇒ 永久自匹配）
touch "$ST/running"
( while [ -f "$ST/running" ]; do
    sleep $((SNAP_MIN * 60))
    [ -f "$ST/running" ] && [ -f "$ST/s2/ckpt_last.pt" ] \
      && cp -f "$ST/s2/ckpt_last.pt" "$ST/snaps/gan_$(date +%H%M).pt"
  done ) &

SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=$PORT train_linattn.py $COMMON \
 --epochs 4 --warmup_frac 0.02 --time_budget_h $BUDGET \
 --init_from "$SRC" \
 --gan_weight "$GW" --gan_ndf 64 --gan_layers 3 --gan_d_lr "$DLR" \
 --gan_start_frac 0.0 --gan_ramp_frac 0.1 \
 --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
echo "[ganft] rc=$? $(date)" >> "$ST/status"
rm -f "$ST/running"       # 停止快照循环

# GAN 下 val PSNR 必然下降 ⇒ ckpt_best 无意义，评 ckpt_last
python3 fast_rfid.py --ckpt "$ST/s2/ckpt_last.pt" --data_dir /root/autodl-tmp/datasets/imagenet100 \
  --img_size 256 --out "$ST/${TAG}_fast.json" --tag "$TAG" --batch_size 64 --amp 1 > "$ST/fast.log" 2>&1
echo "[ganft] eval rc=$? $(date)" >> "$ST/status"
echo "[ganft] DONE $(date)" >> "$ST/status"
