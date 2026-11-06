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
SEC_PER_STEP=${SEC_PER_STEP:-15.0}          # run3 数据序列长（T 中位 121/均值 batch-max≈313）⇒ 实测明显慢于 run2 的 8.95 s/step，取 15.0 估算
DEL_INIT_AFTER=${DEL_INIT_AFTER:-1}         # 训练跑起来 15 分钟后删掉 INIT 权重目录（腾 10G；INIT 已在显存里）
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

RUN1_DIR=/root/autodl-tmp/vlm8b/output/run1
RUN2_DIR=/root/autodl-tmp/vlm8b/output/run2

# 1) run2 的 epoch 存档与 run2/final 是同一份权重（save_only_model=True），纯冗余 ⇒ 先删。
#    run3 不从中断点续训，删掉不影响任何评测/复现。
for d in "$RUN2_DIR"/checkpoint-*; do
  [ -d "$d" ] || continue
  echo "[run3] 删除冗余存档 $d（与 run2/final 同权重）"
  rm -rf "$d"
  echo "$d 于 $(date -Is) 删除（与 run2/final 同权重；run2 评测数字已存档）" >> logs/disk_notes.txt
done
FREE=$(free_gb)
echo "[run3] disk free after dup-checkpoint cleanup=${FREE}G"

# 2) 仍 <24G：删已被 run2 取代的 run1/final（10G，评测数字在 doc/2026-11-06/REPORT_vlm8b_1ep.md）
if [ "${FREE:-0}" -lt 24 ] && [ -d "$RUN1_DIR" ] && [ "$(readlink -f "$INIT")" != "$(readlink -f "$RUN1_DIR/final")" ]; then
  echo "[run3] 磁盘紧张：删除已被 run2 取代的 run1/final（约 10G）"
  rm -rf "$RUN1_DIR"
  echo "run1/final 于 $(date -Is) 因磁盘预算被删除（run2 已取代；数字见 doc/2026-11-06/REPORT_vlm8b_1ep.md）" >> logs/disk_notes.txt
  FREE=$(free_gb)
  echo "[run3] disk free after run1 cleanup=${FREE}G"
fi

# 3) 仍 <22G 且起点不是 run2/final（=评测门未过，run2 明显更差）：删 run2/final 腾 10G，
#    评测数字已在 REPORT_vlm8b_2ep.md。
if [ "${FREE:-0}" -lt 22 ] && [ "$(readlink -f "$INIT")" != "$(readlink -f "$RUN2_DIR/final")" ] && [ -d "$RUN2_DIR/final" ]; then
  echo "[run3] 磁盘紧张：删除评测更差、已被 run1/final 取代的 run2/final（约 10G）"
  rm -rf "$RUN2_DIR/final"
  echo "run2/final 于 $(date -Is) 因磁盘预算被删除（评测门未过，数字见 REPORT_vlm8b_2ep.md）" >> logs/disk_notes.txt
  FREE=$(free_gb)
fi

# 存档预算：稳态要同时容纳「当前 epoch 存档 10G + 正在写的新存档 10G」⇒ 启动时 ≥21G。
# 达标后：稳态剩余 ~11G，写新存档瞬间最低 ~1G（每次 epoch 存档几分钟）；这是刻意安排，
# 看门狗在 10 分钟采样下基本采不到瞬时低点。
if [ "${FREE:-0}" -lt 20 ]; then
  echo "[run3] FATAL 磁盘不足（${FREE}G < 20G），中止；请人工清理后再启动"
  exit 3
fi

# 用修复版 train.py（E1 冻结放在 DS engine 之后；本 run freeze-encoder-epochs=0，仅为代码对齐）
if [ -f train_fixed.py ]; then
  cp train.py "logs/train.py.bak.$(date +%s)"
  cp train_fixed.py train.py
  echo "[run3] train.py <= train_fixed.py ($(sha256sum train.py | cut -c1-12))"
fi

echo "[run3] start $(date -Is) epochs=$EPOCHS init=$INIT data=$TRAIN_JSONL"
# run3 的 122k 混合数据序列比 run2 的 in100 自蒸馏数据长得多（batch-max T≈313，最长 920）：
#   不开 grad-ckpt 时 micro2×accum8 在第一步就 OOM（79G 卡只剩 ~7G 给激活）。
#   ⇒ 打开 gradient checkpointing（+~30% 计算），全局 batch 不变（2×8×2=32）。
# 同时用 expandable_segments 降低碎片风险。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# INIT 权重在训练启动后已进显存；15 分钟后删掉腾 10G，给 epoch 存档留出安全余量
#（存档时要同时放「旧 checkpoint 10G + 新 checkpoint 10G」）。
if [ "$DEL_INIT_AFTER" = "1" ] && [ -d "$INIT" ]; then
  ( sleep 900; rm -rf "$INIT"; echo "INIT=$INIT 于 $(date -Is) 在 run3 启动 15 分钟后删除（权重已载入；腾 10G 给存档）" >> logs/disk_notes.txt ) &
  echo "[run3] 已排定：15 分钟后删除 INIT=$INIT 腾 10G"
fi

timeout 300000 /root/venv_qwen/bin/python -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29513 \
  train.py --freeze-encoder-epochs 0 --epochs "$EPOCHS" --init-from "$INIT" \
  --output-dir "$OUT" --save-total-limit 1 --grad-ckpt \
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
