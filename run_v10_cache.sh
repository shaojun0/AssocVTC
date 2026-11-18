#!/bin/bash
# 离线缓存冻结编码器输出：32,165 片段 × 120 帧 → (z_cls, z_s)，约 59 GB。
# 4 个**独立**单进程（各自 MUSA_VISIBLE_DEVICES=i，交错 20 s 启动），不用 MCCL：
# MUSA 上 4 进程同时初始化会把驱动卡死（实测 hang 在 flush_workqueue，GPU 0%）。
set -u
cd /root/AssocVTC_v10
export SDPA_BACKEND=math
export OMP_NUM_THREADS=6
LOG=/root/autodl-tmp/tok448_v2/logs
mkdir -p "$LOG"
for i in 0 1 2 3; do
  MUSA_VISIBLE_DEVICES=$i nohup python build_latent_cache.py \
    --data_root /root/autodl-tmp/video_recon/data3 --split train \
    --tok_ckpt /root/autodl-tmp/tok448_v2/s2/ckpt_best.pt \
    --out /root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train \
    --frames 120 --img 448 --batch 4 --num_workers 8 \
    --rank $i --world 4 > "$LOG/v10_cache_r$i.log" 2>&1 &
  sleep 20
done
wait
echo "[cache] all shards done"
