#!/usr/bin/env bash
# run3：run2/final → 评测对齐混合数据（text 82k + img 31.8k + ImageNet-100 66k = 179,845 行）
#   epochs 3.8 × 5,621 步 ≈ 21,360 步 ≈ 53 h（周六 06:00 起 → 周一 11:00–12:00 止）
# 前置：run2 已结束；/root/vlm8b/train_fixed.py 就位（E1 冻结修复版）
# 磁盘策略：HF Trainer 是「先存新 checkpoint、再删旧的」⇒ 峰值要 2×10G。
#   故启动前若剩余 <20G，删除已被 run2 取代的 run1/final（10G，评测数字已在 doc 里）；
#   删完仍 <18G 则中止并留日志等人工。
set -uo pipefail
cd /root/vlm8b
mkdir -p logs
EPOCHS=${EPOCHS:-3.8}
OUT=${OUT:-/root/autodl-tmp/vlm8b/output/run3}
INIT=${INIT:-/root/autodl-tmp/vlm8b/output/run2/final}
DATA=${DATA:-/root/autodl-tmp/vlm8b/data_eval}
TRAIN_JSONL=${TRAIN_JSONL:-$DATA/mix_train.jsonl}

for f in "$INIT" "$TRAIN_JSONL" "$DATA/img"; do
  [ -e "$f" ] || { echo "[run3] FATAL missing $f"; exit 2; }
done

free_gb() { df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9'; }
FREE=$(free_gb)
echo "[run3] disk free=${FREE}G"
if [ "${FREE:-0}" -lt 20 ] && [ -d /root/autodl-tmp/vlm8b/output/run1/final ]; then
  echo "[run3] 磁盘紧张：删除已被 run2 取代的 run1/final（约 10G）——评测数字已在 doc/2026-11-06/REPORT_vlm8b_1ep.md"
  rm -rf /root/autodl-tmp/vlm8b/output/run1
  echo "run1/final 于 $(date -Is) 因磁盘预算被删除（run2 已取代；数字见 doc/2026-11-06/REPORT_vlm8b_1ep.md）" >> logs/disk_notes.txt
  FREE=$(free_gb)
  echo "[run3] disk free after cleanup=${FREE}G"
fi
if [ "${FREE:-0}" -lt 18 ]; then
  echo "[run3] FATAL 磁盘不足（${FREE}G < 18G），中止；请人工清理后再启动"
  exit 3
fi

# 用修复版 train.py（E1 冻结放在 DS engine 之后；本 run freeze-encoder-epochs=0，仅为代码对齐）
if [ -f train_fixed.py ]; then
  cp train.py "logs/train.py.bak.$(date +%s)"
  cp train_fixed.py train.py
  echo "[run3] train.py <= train_fixed.py ($(sha256sum train.py | cut -c1-12))"
fi

echo "[run3] start $(date -Is) epochs=$EPOCHS init=$INIT data=$TRAIN_JSONL"
timeout 260000 /root/venv_qwen/bin/python -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29513 \
  train.py --freeze-encoder-epochs 0 --epochs "$EPOCHS" --init-from "$INIT" \
  --output-dir "$OUT" --save-total-limit 1 \
  --train-jsonl "$TRAIN_JSONL" --image-root "$DATA"
rc=$?
echo "[run3] rc=$rc $(date -Is)" | tee logs/run3.rc
echo "[run3] disk free=$(free_gb)G"
if [ "$rc" -eq 0 ]; then
  touch "$OUT/VLM8B_DONE"
  echo "[run3] launching eval chain"
  nohup bash eval_after_run3.sh > logs/eval_chain_run3.nohup 2>&1 &
fi
exit "$rc"
