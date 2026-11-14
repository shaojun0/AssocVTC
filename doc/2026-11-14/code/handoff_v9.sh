#!/bin/bash
# handoff_v9.sh —— 等 in100 那轮（3h 预算）结束后，自动：备份日志 → 487M v9 测速
# → 冒烟 → 起 26h 训练（图像 2h + 视频 24h）。全程 nohup，不依赖我的会话。
set -u
B=/root/autodl-tmp; CODE=$B/video_recon_scale; PY=/root/venv_v31/bin/python
L=$B/logs; mkdir -p "$L"
MARK=$L/v9_launched
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L/handoff_v9.log"; }
mkdir -p "$B/evidence_in100"
for i in $(seq 1 300); do
  pgrep -f "train_linattn[.]py" >/dev/null || break
  sleep 60
done
log "in100 结束 | gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
cp -f $B/run_v2in100/train.log $B/run_v2in100/log.jsonl $B/evidence_in100/ 2>/dev/null
sleep 90
cd $CODE || exit 1
if grep -q "v9  par +mlp" "$L/bench_v9_487.log" 2>/dev/null; then
  log "== 已有 487M 测速结果，跳过测速 =="
else
log "== 487M v9 测速开始 =="
for bs in 2 4 6 8; do
  timeout 900 $PY bench_v9.py --cfg cfg_v9_240_120_120.json --bs $bs --F 16 --n-warm 8 --iters 3 \
    >> "$L/bench_v9_487.log" 2>&1
  log "bench bs=$bs done"
done
fi
grep -E "^\[v9" "$L/bench_v9_487.log" | tail -12 | tee -a "$L/handoff_v9.log"
BEST=$($PY - "$L/bench_v9_487.log" <<'PYEOF'
import re, sys
best = (0.0, 4)
for line in open(sys.argv[1]):
    if "v9  par +mlp" not in line or "帧/天" not in line: continue
    m = re.search(r"bs=(\d+).*?\|\s*([0-9.]+)k 帧/天", line)
    if m and float(m.group(2)) > best[0]:
        best = (float(m.group(2)), int(m.group(1)))
print(best[1])
PYEOF
)
BEST=${BEST:-4}
log "选定 VID_BS=$BEST | ${BEST} 见 $L/bench_v9_487.log"
# ---- 冒烟（各 8-10 步），失败就不启动长跑并留状态
rm -rf $B/video_recon/smoke_v9_img $B/video_recon/smoke_v9_vid
log "== 冒烟 image =="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u train_v7.py --stage image \
  --data-root $B/video_recon/data3/train --val-root $B/video_recon/data3/val \
  --out $B/video_recon/smoke_v9_img --cfg-json cfg_v9_240_120_120.json \
  --img-per 4 --batch 4 --steps 8 --hours 999 --warmup 3 --val-every 8 --val-clips 1 \
  --val-window 4 --val-hist 4 --save-every 1000 --keep-ckpt 1 --workers 8 --enc-chunk 4 \
  --grad-clip 1.0 --wd 0.05 --amp 1 --amp-dtype bf16 --world 1 --log-every 2 \
  --print-cfg 0 --train-enc 1 --train-mid 1 --train-dec 1 >> "$L/smoke_v9_img.log" 2>&1
if ! grep -q "DONE step" "$L/smoke_v9_img.log"; then log "!! 图像冒烟失败，见 $L/smoke_v9_img.log"; exit 1; fi
log "图像冒烟 OK | $(grep -E 'step [0-9]+ loss' $L/smoke_v9_img.log | tail -1)"
log "== 冒烟 video =="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u train_v7.py --stage video \
  --data-root $B/video_recon/data3/train --val-root $B/video_recon/data3/val \
  --out $B/video_recon/smoke_v9_vid --cfg-json cfg_v9_240_120_120.json \
  --window 8 --hist-choices 4,8,16,24 --p-cold 0.3 --batch "$BEST" --steps 10 --hours 999 \
  --lr 1.5e-4 --lr-enc 0.3 --lr-mid 1.0 --lr-dec 1.0 --warmup 3 --val-every 10 --val-clips 1 \
  --val-window 6 --val-hist 8 --save-every 1000 --keep-ckpt 1 --workers 8 --enc-chunk 4 \
  --grad-clip 1.0 --wd 0.05 --amp 1 --amp-dtype bf16 --world 1 --log-every 2 \
  --print-cfg 0 --train-enc 1 --train-mid 1 --train-dec 1 >> "$L/smoke_v9_vid.log" 2>&1
if ! grep -q "DONE step" "$L/smoke_v9_vid.log"; then log "!! 视频冒烟失败，见 $L/smoke_v9_vid.log"; exit 1; fi
log "视频冒烟 OK | $(grep -E 'step [0-9]+ loss' $L/smoke_v9_vid.log | tail -1)"
rm -rf $B/video_recon/smoke_v9_img $B/video_recon/smoke_v9_vid
# ---- 正式 26h
log "== 启动正式训练：IMG_HOURS=2 VID_HOURS=24 VID_BS=$BEST IMG_BS=16 =="
cd $CODE && IMG_HOURS=2 VID_HOURS=24 IMG_BS=4 VID_BS=$BEST STALE=1800 \
  setsid nohup bash run_v9.sh > "$L/v9_driver.log" 2>&1 < /dev/null &
sleep 5; touch "$MARK"
log "已启动，marker=$MARK"
