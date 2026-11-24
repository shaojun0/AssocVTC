#!/bin/bash
# ── I1+I2 潜变量蒸馏探针（kd2）：payload → z_hat → 冻结 SD-VAE decoder ──────────────
# ⚠️ 这是 I1+I2 的**探针实验**（doc/2026-11-23/NOTES_kd_vs_alit_code.md §3-I1/I2/I4）：
#    · I1: KD 度量从"教师解码图的像素 L1"换成"教师 latent 的 MSE"（两边 ×0.18215）;
#    · I2: 评测/验证出图改走 `z_hat → freeze teacher.decode(z/0.18215)`（不再用自身像素头）;
#    · I4: 逐步监督 `--kd_steps all` + 随步递增权重（早期步 payload 少, 少压一点）。
#    · 权重按**加权平均**归一化（`L += kd_lam · Σ_t w_t·MSE_t / Σw`）⇒ `KDW` 始终是
#      总 λ（这里 =1.0）, 下面这组 w 只决定"各步的相对 Emphasis"（随步递增）;
#      若首个 epoch kd 项压过重建项, 先降 KDW 再谈别的。
#    **本脚本不自动启动，需人工批准后再跑**（一条 2 epoch 的 GPU 作业）。
# 起点: k256x12_t8 的 ep14 L1 ckpt（同 kd1）; 预算 2.8h / 2 epoch（先探针，别指望收敛）。
# 预登记（NOTES §6）: I1 臂 rFID 60–80; I2 若 z 预测误差主导则 >40。**先看 kd_mse 有没有降**。
# 用法（人工执行）:
#   bash run_kd2_latent.sh                 # 正式 2 epoch
#   SMOKE=60 bash run_kd2_latent.sh        # 只跑 60 步, 验证 latent KD 分支通了没
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
SRC=${SRC:-/root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt}
ST=${ST:-/root/autodl-tmp/tok256x12_t8/kd2}
TAG=${TAG:-kd2}
EPOCHS=${EPOCHS:-2}
LR=${LR:-2e-5}
BUDGET=${BUDGET:-2.8}
PORT=${PORT:-29708}
KDW=${KDW:-1.0}
SCALE=${SCALE:-0.18215}
SMOKE=${SMOKE:-0}
# 逐步 KD 权重（1-based 步序语义, T=8）: 随步递增 ⇒ 384 值的前几步不硬装 4096 值的细节。
KD_W=${KD_W:-"0.2 0.3 0.5 0.7 0.9 1.0 1.0 1.2"}
mkdir -p "$ST/s2"
echo "[kd2] start $(date) src=$SRC epochs=$EPOCHS lr=$LR kd_weight=$KDW scale=$SCALE smoke=$SMOKE" > "$ST/status"

if [ "$SMOKE" = "0" ]; then
  # 等 GPU 空出来（最多 60 分钟）。注意 kd1 的训练结束后还会串行跑 4 组
  # fast_rfid 评测（每组 ~8 min），所以 train_linattn **和** fast_rfid 都要等。
  for i in $(seq 1 120); do
    n=$(pgrep -f "train_linattn.py|fast_rfid.py" 2>/dev/null | wc -l)
    [ "${n:-0}" -eq 0 ] && break
    sleep 30
  done
  n=$(pgrep -f "train_linattn.py|fast_rfid.py" 2>/dev/null | wc -l)
  if [ "${n:-0}" -ne 0 ]; then
    echo "[kd2] ABORT: GPU busy ($n procs) $(date)" >> "$ST/status"; exit 1
  fi
  echo "[kd2] GPU free $(date)" >> "$ST/status"
fi

COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 256 --patch 16 --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 4 --lr $LR --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 10 \
 --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 1 --save_opt 0 --num_workers 8 \
 --zs_dim 12 --allow_k_gt_n 1 \
 --z_head 1 --render sdvae \
 --kd_space latent --kd_weight $KDW --kd_teacher stabilityai/sd-vae-ft-mse --kd_local_only 1 \
 --kd_loss l1 --kd_latent_scale $SCALE --kd_steps all --kd_step_weights $KD_W \
 --kd_start_frac 0.02 --kd_ramp_frac 0.03 \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256"

EXTRA=""
if [ "$SMOKE" != "0" ]; then EXTRA="--stop_at_step $SMOKE"; fi

SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=$PORT train_linattn.py $COMMON $EXTRA \
 --epochs $EPOCHS --warmup_frac 0.02 --time_budget_h $BUDGET \
 --init_from "$SRC" --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
echo "[kd2] train rc=$? $(date)" >> "$ST/status"

if [ "$SMOKE" != "0" ]; then
  grep -E "\[kd\]|kd_mse=|Traceback|Error" "$ST/s2.log" | tail -12 >> "$ST/status"
  echo "[kd2] SMOKE DONE $(date)" >> "$ST/status"
  exit 0
fi

grep -E "\[val\] ep|\[done\]" "$ST/s2.log" | tail -8 >> "$ST/status" 2>/dev/null

# 逐 epoch + best: 8 步 rFID 走 I2 投放路径（z_hat → teacher decoder）
for ck in "$ST"/s2/ckpt_ep*.pt; do
  [ -f "$ck" ] || continue
  t=$(basename "$ck" .pt)
  python3 fast_rfid.py --ckpt "$ck" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --out "$ST/${t}_fast.json" --tag "$t" --batch_size 64 --amp 1 \
    --render sdvae --kd_latent_scale "$SCALE" \
    > "$ST/fast_${t}.log" 2>&1
  echo "[kd2] eval $t rc=$? $(date)" >> "$ST/status"
done
if [ -f "$ST/s2/ckpt_best.pt" ]; then
  python3 fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --out "$ST/ckpt_best_fast.json" --tag kd2_best --batch_size 64 --amp 1 \
    --render sdvae --kd_latent_scale "$SCALE" \
    > "$ST/fast_best.log" 2>&1
  echo "[kd2] eval best rc=$? $(date)" >> "$ST/status"
fi
echo "[kd2] DONE $(date)" >> "$ST/status"
