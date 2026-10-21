#!/usr/bin/env bash
# rFID 全量评测（K=64 主臂 + K-sweep 曲线; 2 卡分派）2026-10-21
set -uo pipefail
cd /root/AssocVTC
PY=/root/miniconda3/bin/python
OUT=/root/autodl-tmp/rfid_k64
K=/root/autodl-tmp/srdiff_k8sweep

run() {  # run <gpu> <tag> <ckpt> <steps> [extra...]
  local gpu=$1 tag=$2 ckpt=$3 steps=$4; shift 4
  echo "=== [gpu$gpu] $tag start $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/eval_rfid.py \
      --ckpt "$ckpt" --out_dir $OUT --tag "$tag" --steps "$steps" "$@" \
      > $OUT/log_$tag.out 2>&1
  local rc=$?
  echo "=== [gpu$gpu] $tag rc=$rc end $(date -Is) ==="
}

( run 0 native_k64  $K/_ref_k64_native/ckpt_best.pt  all
  run 0 k64_t8      $K/stdattn_k64_t8/ckpt_best.pt   last
  run 0 k96_t8      $K/stdattn_k96_t8/ckpt_best.pt   last
  run 0 k128_t8     $K/stdattn_k128_t8/ckpt_best.pt  last
  run 0 k24_t8      $K/stdattn_k24_t8/ckpt_best.pt   last
  run 0 k12_t8      $K/stdattn_k12_t8/ckpt_best.pt   last
  echo "GPU0_DONE $(date -Is)" ) > $OUT/chain_gpu0.log 2>&1 &
( run 1 k64_t1      $K/stdattn_k64_t1/ckpt_best.pt   last
  run 1 k32_t8      $K/stdattn_k32_t8/ckpt_best.pt   last
  run 1 k48_t8      $K/stdattn_k48_t8/ckpt_best.pt   last
  run 1 k16_t8      $K/stdattn_k16_t8/ckpt_best.pt   last
  run 1 k8_t8       $K/stdattn_k8_t8/ckpt_best.pt    last
  echo "GPU1_DONE $(date -Is)" ) > $OUT/chain_gpu1.log 2>&1 &
wait
echo "ALL_DONE $(date -Is)" > $OUT/ALL_DONE
