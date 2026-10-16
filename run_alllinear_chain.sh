#!/usr/bin/env bash
# =============================================================================
# 「全线性」消融臂链式驱动 — 2026-10-16
#   编码器 **线性注意力** + 解码器 **线性注意力**（自注意力与交叉注意力都是核形式）
#   + BPTT 循环读出。
#
#   两阶段协议与 stdattn / linattn 两臂**逐项相同**, 所以可直接与已有四臂配对:
#     阶段 1  全线性 · 单步 (bs128, 45ep)  ←→  k64（线性enc+softmax dec）26.602
#                                            ←→  stdattn_k64（softmax enc+dec）31.219
#     阶段 2  全线性 · BPTT (bs64, 6ep, 从阶段1热启动)
#                                          ←→  k64_rec 25.304 / stdattn_k64_rec 28.582
#
# 每个阶段带"时间预算 + 断点续跑"：AutoDL 掉线后重跑本脚本即可接着训。
# 用法: nohup setsid bash run_alllinear_chain.sh > /root/autodl-tmp/srdiff_alllinear/nohup.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/srdiff_alllinear
LL=$ROOT/alllinear_k64
LR=$ROOT/alllinear_k64_rec
L1=/root/autodl-tmp/srdiff_linattn/k64              # 线性enc + softmax dec, 单步
ST1=/root/autodl-tmp/srdiff_stdattn/stdattn_k64     # softmax enc + softmax dec, 单步
L2=/root/autodl-tmp/srdiff_linattn/k64_rec          # 线性enc + softmax dec, BPTT
ST2=/root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec # softmax enc + softmax dec, BPTT
CHAIN=$ROOT/chain.log
T1=$((950 * 45))      # bs128 → 950 步/epoch × 45 = 42750
T2=$((1901 * 6))      # bs64  → 1901 步/epoch × 6  = 11406
mkdir -p "$ROOT"

log() { echo "[chain $(date '+%F %T')] $*" | tee -a "$CHAIN"; }
step_of() {
  local d=$1 s=0
  if [ -f "$d/status.json" ]; then
    s=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('step',0))" \
        "$d/status.json" 2>/dev/null || echo 0)
  fi
  echo "${s:-0}"
}

log "=== 全线性（enc+dec 都线性）消融开始 (pid $$) ==="

# ─────────────────── 阶段 1: 全线性 · 单步 45ep (bs128) ───────────────────
for i in $(seq 1 12); do
  s=$(step_of "$LL")
  if [ "$s" -ge "$T1" ]; then log "阶段1 完成: $s/$T1 步"; break; fi
  r=(); [ -f "$LL/ckpt_last.pt" ] && r=(--resume "$LL/ckpt_last.pt")
  log "阶段1 第 $i 次启动 (当前 $s/$T1) ${r[*]:-冷启动}"
  "$PY" -u train_linattn.py --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --output_dir "$LL" --enc_attn linear --dec_attn linear \
    --num_specials 64 --batch_size 128 --epochs 45 --lr 2e-4 \
    --dec_recurrent 0 --time_budget_h 6.0 "${r[@]}" \
    >> "$ROOT/alllinear_k64_launch.log" 2>&1
  log "阶段1 本次退出码 $?; 进度 $(step_of "$LL")/$T1"
  sleep 15
done
s=$(step_of "$LL"); [ "$s" -ge "$T1" ] || { log "!! 阶段1 未跑完 ($s/$T1), 中止"; exit 1; }

# ───────────── 阶段 2: 全线性 · BPTT 6ep (bs64, 从阶段1热启动) ─────────────
INIT=$LL/ckpt_best.pt
[ -f "$INIT" ] || INIT=$LL/ckpt_last.pt
[ -f "$INIT" ] || { log "!! 找不到阶段1权重, 中止"; exit 1; }
log "阶段2 热启动来源: $INIT"

for i in $(seq 1 12); do
  s=$(step_of "$LR")
  if [ "$s" -ge "$T2" ]; then log "阶段2 完成: $s/$T2 步"; break; fi
  if [ -f "$LR/ckpt_last.pt" ]; then
    r=(--resume "$LR/ckpt_last.pt")
  else
    r=(--init_from "$INIT")
  fi
  log "阶段2 第 $i 次启动 (当前 $s/$T2) ${r[*]}"
  "$PY" -u train_linattn.py --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --output_dir "$LR" --enc_attn linear --dec_attn linear \
    --num_specials 64 --batch_size 64 --epochs 6 --lr 2e-4 \
    --dec_recurrent 1 --dec_grad_ckpt 1 --time_budget_h 3.0 "${r[@]}" \
    >> "$ROOT/alllinear_k64_rec_launch.log" 2>&1
  log "阶段2 本次退出码 $?; 进度 $(step_of "$LR")/$T2"
  sleep 15
done
s=$(step_of "$LR"); [ "$s" -ge "$T2" ] || { log "!! 阶段2 未跑完 ($s/$T2), 中止"; exit 1; }

# ─────────────────── 收尾: 六臂同条件全量评测 ───────────────────
log "=== 全量留出集评测（6 臂同进程同批次） ==="
"$PY" -u eval_linattn.py \
  --ckpt "$LL/ckpt_best.pt" "$LR/ckpt_best.pt" \
         "$L1/ckpt_best.pt" "$ST1/ckpt_best.pt" \
         "$L2/ckpt_best.pt" "$ST2/ckpt_best.pt" \
  --num_images 0 --batch_size 64 --num_workers 12 \
  --out "$ROOT/eval_ablation6.json" \
  --grid "$ROOT/FINAL_grid_alllinear.png" >> "$ROOT/eval_ablation.log" 2>&1 \
  && log "全量评测 OK → $ROOT/eval_ablation6.json" \
  || log "!! 全量评测失败, 见 eval_ablation.log"

log "=== 逐步(循环)重建评测（全线性 BPTT 臂） ==="
"$PY" -u eval_stepwise_linattn.py \
  --ckpt "$LR/ckpt_best.pt" --num_images 0 --batch_size 64 \
  --out "$ROOT/stepwise_alllinear_rec.json" >> "$ROOT/stepwise.log" 2>&1 \
  && log "逐步评测 OK" || log "!! 逐步评测失败, 见 stepwise.log"

log "=== 训练曲线 ==="
"$PY" -u plot_linattn.py --log "$LL/train_log.jsonl" --out "$LL/curves.png" \
  --title "all-linear (linear-attn encoder 31.9M + linear-attn decoder 31.2M): single-step, ImageNet-100 224x224" \
  >> "$ROOT/plot.log" 2>&1 || log "!! 阶段1 曲线失败"
"$PY" -u plot_linattn.py --log "$LR/train_log.jsonl" --out "$LR/curves.png" \
  --title "all-linear (linear-attn encoder + linear-attn decoder): recurrent readout (BPTT, T=8)" \
  >> "$ROOT/plot.log" 2>&1 || log "!! 阶段2 曲线失败"

log "=== 全部完成 ==="
