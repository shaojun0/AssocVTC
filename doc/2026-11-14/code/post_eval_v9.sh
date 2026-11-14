#!/bin/bash
# post_eval_v9.sh —— 等 v9 训练收尾后，用 val_long（4 条 120 帧、hist 1/4/8/16）出终判
set -u
S=/root/autodl-tmp/video_recon; CODE=/root/autodl-tmp/video_recon_scale; PY=/root/venv_v31/bin/python
L=/root/autodl-tmp/logs; mkdir -p "$L" /root/autodl-tmp/eval_v9_long
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L/post_eval_v9.log"; }
for i in $(seq 1 900); do [ -f $S/run_v9_vid/final_model.pt ] && break; sleep 60; done
log "检测到 final_model.pt，等 90 s 让存档落盘完"
sleep 90
cp -f $S/run_v9_vid/val.jsonl /root/autodl-tmp/evidence_v9_val.jsonl 2>/dev/null
cp -f $S/logs/v9_video.log $S/logs/v9_image.log /root/autodl-tmp/ 2>/dev/null
cd $CODE || exit 1
log "开始 val_long 评测"
$PY -u eval_v7.py --ckpt $S/run_v9_vid/final_model.pt \
  --root $S/data3/val_long --clips 4 --window 120 --hist-sweep 1,4,8,16 \
  --chunk 8 --enc-chunk 8 --amp bf16 --retr 1 --vis 1 \
  --out /root/autodl-tmp/eval_v9_long >> "$L/post_eval_v9.log" 2>&1
log "评测完成 rc=$?"
