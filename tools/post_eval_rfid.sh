#!/bin/bash
# Post-training rFID evaluation (ALIT protocol), auto-runs when the 8-step rFID training exits.
# Output: per-step rFID (8 steps) for our zs192 T=8 model + T=1 zs192 baseline.
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
ST=/root/autodl-tmp/tok448_zs192_t8rfid
OUT=/root/autodl-tmp/rfid_zs192t8
mkdir -p "$ST" "$OUT"
echo "[post] waiting for training to exit: $(date)" > "$ST/post_eval.status"

while pgrep -f "train_linattn.py.*tok448_zs192_t8rfid" >/dev/null 2>&1; do sleep 30; done
echo "[post] training exited: $(date)" >> "$ST/post_eval.status"

if [ ! -f "$ST/s2/ckpt_best.pt" ]; then
  echo "[post] ABORT: no ckpt_best.pt" >> "$ST/post_eval.status"
  exit 1
fi

IN=/root/autodl-tmp/datasets/imagenet100
COMMON="--data_dir $IN --img_size 448 --val_per_class 50"

# 1) GT PNG dump (5000 IN-100 val, 448x448, [0,1])
python3 tools/eval_rfid.py --gt_only --out_dir "$OUT" $COMMON \
  > "$ST/eval_gt.log" 2>&1
echo "[post] gt rc=$? $(date)" >> "$ST/post_eval.status"

# 2) Precompute GT Inception stats (npz) -> halves FID cost per arm
python3 -m pytorch_fid --save-stats "$OUT/gt" "$OUT/gt_stats.npz" \
  > "$ST/eval_gtstats.log" 2>&1
echo "[post] gt_stats rc=$? $(date)" >> "$ST/post_eval.status"

# 3) Our 8-step recurrent model: per-step rFID (steps 1..8)
python3 tools/eval_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --tag zs192_t8rfid \
  --out_dir "$OUT" $COMMON --steps all --gt_stats "$OUT/gt_stats.npz" \
  --batch_size 64 --amp 1 --fid_batch 50 --num_workers 8 \
  > "$ST/eval_main.log" 2>&1
echo "[post] main rc=$? $(date)" >> "$ST/post_eval.status"

# 4) Baseline: same zs_dim but T=1 single-step (tok448_zs192 arm)
python3 tools/eval_rfid.py --ckpt /root/autodl-tmp/tok448_zs192/s2/ckpt_best.pt \
  --tag zs192_t1 --out_dir "$OUT" $COMMON --steps last \
  --gt_stats "$OUT/gt_stats.npz" --batch_size 64 --amp 1 --fid_batch 50 --num_workers 8 \
  > "$ST/eval_t1.log" 2>&1
echo "[post] t1 rc=$? $(date)" >> "$ST/post_eval.status"
echo "[post] ALL_DONE $(date)" >> "$ST/post_eval.status"
