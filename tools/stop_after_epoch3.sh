#!/usr/bin/env bash
# run3 兜底收尾：只有当「按当前步速会拖过周一 14:00」时，才在 epoch-3 存档（step 11463，
# 89%，余弦 LR 已退火到峰值 ~3%）处停训，用该存档当 final 并接评测链；否则让训练自然跑完。
#
# 背景：run3 步速随混合数据序列长度浮动（实测 14.8–16.5 s/step），跑完 12,839 步的落点在
# 周一 ~11:40 到 ~15:30 之间。若落点不超过 14:00，就让训练自然收尾（多 11% 步数）；
# 若会超过，则在 epoch-3 存档处收尾，把结果提前到 ~+1h 出。
# 判定：checkpoint-11463 出现 → 等大小稳定（>9G、含 *.safetensors 与 config.json）
# → 用 run3.log 里的 tqdm 已用时间算步速 → 预测跑完时间 > DEADLINE_TS 才停。
# 兜底：存档不完整 / 训练已自然结束（final 存在）→ 本脚本直接退出。
set -uo pipefail
cd /root/vlm8b
LOG=logs/stop_after_epoch3.log
OUT=/root/autodl-tmp/vlm8b/output/run3
CKPT="$OUT/checkpoint-11463"
TOTAL=12839
CKPT_STEP=11463
DEADLINE_TS=$(date -d "2026-11-09 14:00:00" +%s)
echo "[stop] watcher start $(date -Is) deadline=$(date -d @$DEADLINE_TS -Is)" >> "$LOG"
while true; do
  if [ -d "$CKPT" ]; then break; fi
  if [ -d "$OUT/final" ]; then echo "[stop] final 已存在（训练自然结束）→ watcher 退出 $(date -Is)" >> "$LOG"; exit 0; fi
  sleep 60
done
echo "[stop] 发现 $CKPT $(date -Is)" >> "$LOG"

prev=0; ok=0
for _ in 1 2 3 4 5 6 7 8; do
  sleep 30
  cur=$(du -sb "$CKPT" 2>/dev/null | cut -f1)
  echo "[stop] size=$cur prev=$prev $(date -Is)" >> "$LOG"
  if [ "${cur:-0}" -gt 9000000000 ] && [ "$cur" = "$prev" ]; then ok=1; break; fi
  prev=${cur:-0}
done
saf=$(ls "$CKPT"/*.safetensors 2>/dev/null | wc -l)
if [ "$ok" != "1" ] || [ "$saf" -lt 1 ] || [ ! -f "$CKPT/config.json" ]; then
  echo "[stop] 存档不完整（ok=$ok saf=$saf）→ 不动训练 $(date -Is)" >> "$LOG"
  exit 0
fi

# 从 run3.log 的 tqdm 读取已用时间，估算步速与跑完时间
eline=$(grep -oE '\[[0-9]+:[0-9]{2}:[0-9]{2}<' logs/run3.log | tail -1 | tr -d '[' | tr -d '<')
if [ -z "$eline" ]; then echo "[stop] 解析不到 tqdm 时间 → 保守停训 $(date -Is)" >> "$LOG"; STOP=1; else
  hh=$(echo "$eline" | cut -d: -f1); mm=$(echo "$eline" | cut -d: -f2); ss=$(echo "$eline" | cut -d: -f3)
  elapsed=$((10#$hh * 3600 + 10#$mm * 60 + 10#$ss))
  rate=$(awk -v e="$elapsed" -v s="$CKPT_STEP" 'BEGIN{printf "%.3f", e / s}')
  proj=$(( $(date +%s) + (TOTAL - CKPT_STEP) * elapsed / CKPT_STEP ))
  echo "[stop] elapsed=${elapsed}s rate=${rate}s/step proj_finish=$(date -d @$proj -Is)" >> "$LOG"
  if [ "$proj" -gt "$DEADLINE_TS" ]; then STOP=1; else STOP=0; fi
fi
if [ "$STOP" != "1" ]; then
  echo "[stop] 预测不会超过截止，让训练自然跑完 → watcher 退出 $(date -Is)" >> "$LOG"
  exit 0
fi

echo "[stop] 预计拖过截止 → 在 epoch-3 存档处收尾 $(date -Is)" >> "$LOG"
pkill -TERM -f "train.py --freeze-encoder-epochs" 2>/dev/null
sleep 20
pkill -KILL -f "train.py --freeze-encoder-epochs" 2>/dev/null
sleep 5
ln -sfn "$CKPT" "$OUT/final"
printf '[run3] rc=143（在 epoch-3 存档 step=11463 处主动收尾以对齐周一中午截止；LR 已退火到峰值 ~3%%）%s\n' "$(date -Is)" | tee logs/run3.rc
touch "$OUT/VLM8B_DONE"
nohup bash eval_after_run3.sh > logs/eval_chain_run3.nohup 2>&1 &
echo "[stop] 已停训并拉起评测链：$(readlink -f "$OUT/final") $(date -Is)" >> "$LOG"
