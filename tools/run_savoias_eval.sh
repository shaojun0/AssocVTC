#!/bin/bash
# 在 GPU 机器上跑 SAVOIAS 复杂度的逐步评测（两个 BPTT 臂）
# -----------------------------------------------------------------------------
# 前置：tools/fetch_savoias.sh 已把数据下到 $DATA_ROOT；
#       ckpt 路径按本机实际布局设置（下面默认值是 10-15/10-16 的 AutoDL 布局）。
#
# 用法：
#   bash tools/run_savoias_eval.sh [DATA_ROOT] [OUT_DIR]
# -----------------------------------------------------------------------------
set -euo pipefail

DATA_ROOT=${1:-/root/autodl-tmp/savoias}
OUT=${2:-/root/autodl-tmp/savoias_eval}
CKPT_STD=${CKPT_STD:-/root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt}
CKPT_LIN=${CKPT_LIN:-/root/autodl-tmp/srdiff_linattn/k64_rec/ckpt_best.pt}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"

run() {   # run <arm> <ckpt>
  local arm=$1 ckpt=$2
  python3 "$REPO/eval_stepwise_linattn.py" \
    --ckpt "$ckpt" --data_dir_ood "$DATA_ROOT" \
    --num_images 0 --batch_size 64 --num_workers 8 --domain savoias \
    --dump_per_image "$OUT/per_image_$arm.json" \
    --out "$OUT/stepwise_$arm.json"
}

run stdattn_k64_rec "$CKPT_STD"
run linattn_k64_rec "$CKPT_LIN"

# 提示：分析在控制机上跑（不需要 GPU）：
#   python3 tools/plot_savoias_complexity.py --data-dir <把 per_image_*.json + gt_*.npy 放一起的目录>
#   python3 tools/render_savoias_grid.py --images "$DATA_ROOT/Art" --scores "$DATA_ROOT/gt/gt_art.npy"
