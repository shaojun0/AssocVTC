#!/bin/bash
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
ST=/root/autodl-tmp/tok448_zs192_t8rfid
OUT=/root/autodl-tmp/rfid_zs192t8
mkdir -p "$OUT"
echo "start $(date)" > "$ST/fast.status"
python3 fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --out "$OUT/zs192_t8rfid_fast.json" \
  --tag zs192_t8rfid --batch_size 64 --amp 1 > "$ST/fast8.log" 2>&1
echo "8step rc=$? $(date)" >> "$ST/fast.status"
python3 fast_rfid.py --ckpt /root/autodl-tmp/tok448_zs192/s2/ckpt_best.pt --out "$OUT/zs192_t1_fast.json" \
  --tag zs192_t1 --batch_size 64 --amp 1 > "$ST/fast_t1.log" 2>&1
echo "t1 rc=$? $(date)" >> "$ST/fast.status"
echo "ALL_DONE $(date)" >> "$ST/fast.status"
