#!/usr/bin/env bash
# run3：run2/final →「评测对齐 + 按任务等量 + 能加就加」混合数据（v2，用户 18:10 反馈）
#   文本 82,000 + 图像 40,259（大任务封顶 6,000：scienceqa 6,000 / chartqa 6,000 /
#   textvqa 6,000 / docvqa 5,365 / slake 5,000；中小任务用到数据上限；
#   ImageNet-100 锚点 66,000→3,000）
#   = 122,259 行 ≈ 3,821 步/epoch ≈ 9.50 h/epoch；epochs 由「到周一截止时间」自动算，
#   默认目标 2026-11-09 12:00 训完（用户要求训到周一 12 点），之后自动接评测链。
# 前置：run2 已结束；/root/vlm8b/train_fixed.py 就位（E1 冻结修复版）
# 磁盘策略：HF Trainer 是「先存新 checkpoint、再删旧的」⇒ 峰值要 2×10G。
#   故启动前若剩余 <20G，删除已被 run2 取代的 run1/final（10G，评测数字已在 doc 里）；
#   删完仍 <18G 则中止并留日志等人工。
# 学习率：run2 的教训（续训 1e-5 重启余弦、loss 中段回升 +45%）⇒ run3 峰值减半：
#   LLM 5e-6 / encoder 2.5e-6 / projector 5e-5，warmup 2%（环境变量可覆盖）。
set -uo pipefail
cd /root/vlm8b
mkdir -p logs
OUT=${OUT:-/root/autodl-tmp/vlm8b/output/run3}
INIT=${INIT:-/root/autodl-tmp/vlm8b/output/run2/final}
DATA=${DATA:-/root/autodl-tmp/vlm8b/data_eval}
TRAIN_JSONL=${TRAIN_JSONL:-$DATA/mix_train_bal2.jsonl}
LR_LLM=${LR_LLM:-5e-6}
LR_ENCODER=${LR_ENCODER:-2.5e-6}
LR_PROJ=${LR_PROJ:-5e-5}
WARMUP=${WARMUP:-0.02}
DEADLINE=${DEADLINE:-2026-11-09 12:00:00}   # 训练截止（之后自动接评测链）
SEC_PER_STEP=${SEC_PER_STEP:-9.0}           # 实测 8.94–8.96 s/step，取 9.0 留余量
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-3821}    # 122,259 行 / 全局 batch 32

# epochs：优先用 EPOCHS 环境变量；否则按「现在 → DEADLINE」倒推（至少 1.0）
if [ -z "${EPOCHS:-}" ]; then
  TARGET_TS=$(date -d "$DEADLINE" +%s) || { echo "[run3] FATAL 无法解析 DEADLINE=$DEADLINE"; exit 4; }
  NOW_TS=$(date +%s)
  AVAIL_STEPS=$(( (TARGET_TS - NOW_TS) / ${SEC_PER_STEP%.*} ))
  if [ "$AVAIL_STEPS" -lt 1000 ]; then
    echo "[run3] FATAL 距 DEADLINE=$DEADLINE 太近（约 ${AVAIL_STEPS} 步），拒绝启动"
    exit 4
  fi
  EPOCHS=$(awk -v s="$AVAIL_STEPS" -v p="$STEPS_PER_EPOCH" 'BEGIN{printf "%.2f", s/p}')
fi

if [ "${DRY:-0}" != "1" ]; then
  for f in "$INIT" "$TRAIN_JSONL" "$DATA/img"; do
    [ -e "$f" ] || { echo "[run3] FATAL missing $f"; exit 2; }
  done
fi

free_gb() { df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9'; }
FREE=$(free_gb)
echo "[run3] config: init=$INIT data=$TRAIN_JSONL epochs=$EPOCHS lr=$LR_LLM/$LR_PROJ/$LR_ENCODER warmup=$WARMUP"
echo "[run3] deadline=$DEADLINE disk free=${FREE}G"
if [ "${DRY:-0}" = "1" ]; then
  echo "[run3] DRY=1：只打印配置，不动磁盘/不启动训练"
  exit 0
fi

RUN1=/root/autodl-tmp/vlm8b/output/run1/final
if [ "${FREE:-0}" -lt 20 ] && [ -d "$RUN1" ] && [ "$(readlink -f "$INIT")" != "$(readlink -f "$RUN1")" ]; then
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
timeout 300000 /root/venv_qwen/bin/python -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29513 \
  train.py --freeze-encoder-epochs 0 --epochs "$EPOCHS" --init-from "$INIT" \
  --output-dir "$OUT" --save-total-limit 1 \
  --lr-llm "$LR_LLM" --lr-encoder "$LR_ENCODER" --lr-proj "$LR_PROJ" \
  --warmup-ratio "$WARMUP" \
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
