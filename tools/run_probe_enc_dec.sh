#!/bin/bash
# ── 冻结视觉编码器「可解码性」探针：批量跑各家编码器（4 GPU 并行）────────────────
# 对应 `tools/probe_enc_dec.py`。协议见该文件头：所有编码器共用同一解码器/预算/目标 256。
# 依赖：HF_ENDPOINT=hf-mirror（本机直连 huggingface.co 不通）, SDPA_BACKEND=math。
#
# 用法：
#   bash run_probe_enc_dec.sh            # 全量（约 1.5–2 h, 含下载）
#   SMOKE=1 bash run_probe_enc_dec.sh    # 每个编码器只跑 200 张图, 验证能不能载入/前向
#   ONLY="clip_l14 sdvae" bash run_probe_enc_dec.sh
set -u
cd /root/AssocVTC_tok448
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache   # / 只有 30G, 缓存放数据盘
export SDPA_BACKEND=math
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-/root/autodl-tmp/tok256x12_t8/probe_native}
SMOKE=${SMOKE:-0}
[ "$SMOKE" != "0" ] && OUT="$OUT/smoke"
TAGSUF=${TAGSUF:-}
LOG=$OUT/logs
mkdir -p "$LOG"
ONLY=${ONLY:-}
TRAIN_PER_CLASS=${TRAIN_PER_CLASS:-200}
EPOCHS=${EPOCHS:-3}
GRID=${GRID:-0}   # 0=原生网格（不压缩）
WIDTH=${WIDTH:-256}
echo "[probe] start $(date) train_per_class=$TRAIN_PER_CLASS epochs=$EPOCHS grid=$GRID width=$WIDTH smoke=$SMOKE" \
  > "$OUT/status.txt"

# ── 0) 先顺序下载全部权重（并行下载会被镜像限速, 且失败要早发现）──
if [ "$SMOKE" = "0" ] && [ -z "$ONLY" ]; then
for m in openai/clip-vit-base-patch32 openai/clip-vit-large-patch14 \
         google/vit-large-patch16-224-in21k google/vit-huge-patch14-224-in21k \
         facebook/dinov2-large facebook/dinov2-giant facebook/vit-mae-huge \
         facebook/sam-vit-huge google/siglip-so400m-patch14-384 \
         Qwen/Qwen2-VL-2B-Instruct; do
  timeout 1200 python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
print('OK', '$m', p)
" >> "$OUT/status.txt" 2>&1
  echo "[probe] download $m rc=$? $(date)" >> "$OUT/status.txt"
done
fi
df -h /root/autodl-tmp | tail -1 >> "$OUT/status.txt"

run_one () {  # $1=enc  $2=gpu  $3=extra args（可空）
  local enc=$1 gpu=$2 extra=${3:-}
  local common="--data_dir /root/autodl-tmp/datasets/imagenet100 \
    --train_per_class $TRAIN_PER_CLASS --val_per_class 10 --epochs $EPOCHS \
    --batch 32 --grid $GRID --width $WIDTH --workers 8"
  [ "$SMOKE" != "0" ] && common="--data_dir /root/autodl-tmp/datasets/imagenet100 \
    --train_per_class 2 --val_per_class 2 --epochs 0.02 --batch 8 --grid $GRID \
    --width 64 --workers 2 --max_steps 3"
  MUSA_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    timeout 5400 python3 probe_enc_dec.py --enc "$enc" $common $extra \
    --save_grid "$OUT/grids/${enc}${TAGSUF:-}.png" \
    --out "$OUT/${enc}${TAGSUF:-}.json" --tag "run1" > "$LOG/${enc}${TAGSUF:-}.log" 2>&1
  echo "[probe] $enc gpu=$gpu rc=$? $(date)" >> "$OUT/status.txt"
}

# ── 1) 三批, 每批 4 个 GPUs ──
BATCH1="clip_l14 clip_b32 vit_l16 dinov2_l"
BATCH2="pixel16 siglip_so400m mae_h randvit_l"
BATCH3="sdvae sam_h dinov2_g vit_h14"
BATCH4="qwen2vl_2b"

want () {  # ONLY 过滤
  [ -z "$ONLY" ] && return 0
  case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac
}
launch_batch () {
  local i=0
  for enc in $1; do
    if want "$enc"; then run_one "$enc" "$i" & fi
    i=$((i + 1))
  done
  wait
}
for B in "$BATCH1" "$BATCH2" "$BATCH3" "$BATCH4"; do
  launch_batch "$B"
  echo "[probe] batch done: $B $(date)" >> "$OUT/status.txt"
done
echo "[probe] DONE $(date)" >> "$OUT/status.txt"
echo "=== summary ==="
python3 - "$OUT" <<'PY'
import json, glob, os, sys
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    try:
        rows.append(json.load(open(p)))
    except Exception as e:
        print("ERR", p, e)
for r in sorted(rows, key=lambda d: -d.get("psnr", 0)):
    print(f"{r['enc']:14s} {r['vendor']:9s} res={r['enc_res']:4d} feat={str(r['feat_shape']):18s} "
          f"PSNR={r['psnr']:6.2f} SSIM={r['ssim']:.4f} L1={r['l1']:.4f} ({r['img_per_s']}img/s)")
PY
