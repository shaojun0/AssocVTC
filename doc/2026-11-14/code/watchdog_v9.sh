#!/bin/bash
# watchdog_v9.sh —— 每 10 min 记一行状态；训练/看护都没了且 marker 存在时重启看护（最多 3 次）
set -u
B=/root/autodl-tmp; L=$B/logs; MARK=$L/v9_launched; CNT=$L/v9_restart.count
mkdir -p "$L"; log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L/watchdog_v9.log"; }
: > "$L/v9_status.log"
while :; do
  STG=img; [ -d $B/video_recon/run_v9_vid ] && [ -f $B/video_recon/run_v9_vid/train.log ] && STG=vid
  D=$B/video_recon/run_v9_$([ "$STG" = vid ] && echo vid || echo img)
  STEP=$(tail -1 "$D/log.jsonl" 2>/dev/null | sed -n 's/.*"step": \([0-9]*\).*/\1/p')
  PSNR=$(tail -1 "$D/log.jsonl" 2>/dev/null | sed -n 's/.*"psnr": \([0-9.]*\).*/\1/p')
  VPSNR=$(tail -1 "$D/val.jsonl" 2>/dev/null | sed -n 's/.*"cold": \([0-9.]*\).*/\1/p')
  GPU=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')
  FREE=$(df -h $B | tail -1 | awk '{print $4}')
  RUN=$(pgrep -c -f "train_v7[.]py" || true); GRD=$(pgrep -c -f "run_v9[.]sh" || true)
  echo "$(date '+%m-%d %H:%M:%S') stage=$STG step=$STEP psnr=$PSNR val_cold=$VPSNR gpu=$GPU free=$FREE tv7=$RUN guard=$GRD" >> "$L/v9_status.log"
  if [ -f "$MARK" ] && [ "${RUN:-0}" -eq 0 ] && [ "${GRD:-0}" -eq 0 ]; then
    C=$(cat "$CNT" 2>/dev/null || echo 0)
    if [ "$C" -lt 3 ]; then
      echo $((C+1)) > "$CNT"; log "检测到训练/看护都不在，重启看护（第 $((C+1)) 次）"
      cd $B/video_recon_scale && IMG_HOURS=2 VID_HOURS=24 IMG_BS=4 VID_BS=4 STALE=1800 \
        setsid nohup bash run_v9.sh >> "$L/v9_driver.log" 2>&1 < /dev/null &
    else
      log "训练已停且重启 3 次用尽，不再重启"
    fi
  fi
  sleep 600
done
