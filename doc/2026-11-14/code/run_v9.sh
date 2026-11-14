#!/bin/bash
# v3_1 (v8 = GDN 中间态 + 帧维折 batch decode) 240/120/120 训练看护（A800 80G 单卡）
#   stage image: 图像预训练（F=1，全参数可训）
#   stage video: 视频 TBPTT（preheat 帧 + window 8，全参数可训）
# 与 run_v8.sh 的差别：单卡（--world 1）、bf16（A800 原生支持，不需 GradScaler）、
# 引脚 enc-chunk/keep-ckpt 控制显存与磁盘（487M 模型 1 个 ckpt 5.9 GB）。
set -u
PY=${PY:-/root/venv_v31/bin/python}
B=/root/autodl-tmp
CODE=$B/video_recon_scale
DATA=${DATA:-$B/video_recon/data3}
LOGS=$B/video_recon/logs
OUT_A=${OUT_A:-$B/video_recon/run_v9_img}
OUT_B=${OUT_B:-$B/video_recon/run_v9_vid}
CFG=${CFG:-cfg_v9_240_120_120.json}
IMG_HOURS=${IMG_HOURS:-2}
VID_HOURS=${VID_HOURS:-14}
IMG_BS=${IMG_BS:-16}
VID_BS=${VID_BS:-8}
WORKERS=${WORKERS:-16}
STALE=${STALE:-1800}
mkdir -p "$LOGS" "$OUT_A" "$OUT_B"
cd "$CODE" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%m-%d %H:%M:%S')] [v9] $*" | tee -a "$LOGS/v9_guard.log"; }
kill_all() { pkill -f "train_v7[.]py" >/dev/null 2>&1; for _ in $(seq 1 30); do pgrep -f "train_v7[.]py" >/dev/null || break; sleep 1; done; pkill -9 -f "train_v7[.]py" >/dev/null 2>&1; sleep 3; }

run_stage() {   # $1=stage $2=out $3=hours $4=init
  local stage=$1 out=$2 hours=$3 init=$4 fails=0 attempt=0 mark extra=() initarg resume
  if [ "$stage" = "image" ]; then
    extra=(--img-per 4 --batch "$IMG_BS" --lr 2e-4 --lr-enc 1.0 --lr-mid 1.0 --lr-dec 1.0
           --warmup 150 --val-every 500 --val-clips 1 --val-window 6 --val-hist 8
           --save-every 600 --keep-ckpt 1 --train-enc 1 --train-mid 1 --train-dec 1)
  else
    extra=(--window 8 --hist-choices 4,8,16,24 --p-cold 0.3 --batch "$VID_BS"
           --lr 1.5e-4 --lr-enc 0.3 --lr-mid 1.0 --lr-dec 1.0 --warmup 150
           --val-every 300 --val-clips 1 --val-window 6 --val-hist 8
           --save-every 600 --keep-ckpt 1 --train-enc 1 --train-mid 1 --train-dec 1)
  fi
  while :; do
    attempt=$((attempt+1)); fails=$((fails+1)); mark=$(date +%s)
    resume=""; initarg=""
    if [ -f "$out/last.pt" ]; then resume="--resume $out/last.pt"
    elif [ -n "$init" ] && [ -f "$init" ]; then initarg="--init $init"; fi
    log "stage=$stage attempt=$attempt hours=$hours $resume"
    setsid nohup $PY -u train_v7.py --stage "$stage" \
      --data-root "$DATA/train" --val-root "$DATA/val" \
      --out "$out" --cfg-json "$CFG" \
      --hours "$hours" --auto-steps-after 60 \
      --workers "$WORKERS" --enc-chunk 4 --grad-clip 1.0 --wd 0.05 \
      --amp 1 --amp-dtype bf16 --world 1 \
      --reload-every 0 --log-every 20 --print-cfg 1 \
      "${extra[@]}" $initarg $resume \
      >> "$LOGS/v9_${stage}.log" 2>&1 < /dev/null &
    sleep 90
    while :; do
      sleep 60
      pgrep -f "train_v7[.]py" >/dev/null || { log "stage=$stage 进程退出"; break; }
      if [ -f "$out/log.jsonl" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$out/log.jsonl") ))
        [ "$age" -gt "$STALE" ] && { log "stage=$stage 日志停滞 ${age}s"; kill_all; break; }
      elif [ $(( $(date +%s) - mark )) -gt 1800 ]; then
        log "stage=$stage 30min 无日志"; kill_all; break
      fi
    done
    kill_all
    if [ -f "$out/final_model.pt" ] && [ "$(stat -c %Y "$out/final_model.pt")" -ge "$mark" ]; then
      log "stage=$stage 完成 → $out/final_model.pt"; return 0
    fi
    log "stage=$stage attempt=$attempt 异常结束"
    [ "$fails" -ge 6 ] && { log "stage=$stage 连续失败 6 次，放弃"; return 1; }
    sleep 15
  done
}

log "v9 guard 启动 CFG=$CFG IMG_HOURS=$IMG_HOURS VID_HOURS=$VID_HOURS IMG_BS=$IMG_BS VID_BS=$VID_BS"
kill_all
run_stage image "$OUT_A" "$IMG_HOURS" "" || exit 1
rm -f "$OUT_A"/step*.pt "$OUT_A"/last.pt   # 留 final_model.pt 供 video 热启动，腾盘（last.pt 5.9G）
log "图像阶段清理完成，剩余空间: $(df -h $B | tail -1 | awk '{print $4}')"
run_stage video "$OUT_B" "$VID_HOURS" "$OUT_A/final_model.pt" || exit 1
rm -f "$OUT_A"/last.pt
log "全部完成 → $OUT_B/final_model.pt"
