#!/usr/bin/env bash
# =============================================================================
# 两阶段「配对消融」链式驱动 — 2026-10-16
#   自变量: 编码器注意力形式 (线性核注意力 → 普通 softmax 缩放点积)
#   阶段 1: std 编码器 + 单步并行解码, bs128 / 45ep  ← 与昨日 k64     同协议
#   阶段 2: 从阶段 1 热启动 + 循环读出 (BPTT), bs64 / 6ep  ← 与昨日 k64_rec 同协议
#   收尾:   四臂同进程全量留出集评测 + 逐步重建 + 曲线
#
# 每个阶段都带"时间预算 + 断点续跑"：AutoDL 掉线/超时后重跑本脚本即可接着训。
# 用法:  nohup setsid bash run_stdattn_chain.sh > /root/autodl-tmp/srdiff_stdattn/nohup.log 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/srdiff_stdattn
S1=$ROOT/stdattn_k64
S2=$ROOT/stdattn_k64_rec
L1=/root/autodl-tmp/srdiff_linattn/k64            # 昨日臂 (线性注意力 + 单步)
L2=/root/autodl-tmp/srdiff_linattn/k64_rec        # 昨日臂 (线性注意力 + BPTT)
CHAIN=$ROOT/chain.log
T1=$((950 * 45))      # bs128, 121689 图 ⇒ 950 步/epoch × 45 = 42750
T2=$((1901 * 6))      # bs64,  121689 图 ⇒ 1901 步/epoch × 6  = 11406
mkdir -p "$ROOT"

log() { echo "[chain $(date '+%F %T')] $*" | tee -a "$CHAIN"; }

step_of() {  # $1=输出目录 → status.json 里的 step（缺失/坏文件 ⇒ 0）
  local d=$1 s=0
  if [ -f "$d/status.json" ]; then
    s=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('step',0))" \
        "$d/status.json" 2>/dev/null || echo 0)
  fi
  echo "${s:-0}"
}

log "=== 两阶段消融开始 (pid $$) ==="

# ─────────────────────────── 阶段 1: std 单步 45ep ───────────────────────────
for i in $(seq 1 12); do
  s=$(step_of "$S1")
  if [ "$s" -ge "$T1" ]; then log "阶段1 完成: $s/$T1 步"; break; fi
  r=(); [ -f "$S1/ckpt_last.pt" ] && r=(--resume "$S1/ckpt_last.pt")
  log "阶段1 第 $i 次启动 (当前 $s/$T1) ${r[*]:-冷启动}"
  TAG=stdattn_k64 BS=128 EPOCHS=45 TIME_BUDGET=5.5 OUT="$S1" \
    ./run_stdattn.sh "${r[@]}" >> "$ROOT/stdattn_k64_launch.log" 2>&1
  log "阶段1 本次退出码 $?; 进度 $(step_of "$S1")/$T1"
  sleep 15
done
s=$(step_of "$S1"); [ "$s" -ge "$T1" ] || { log "!! 阶段1 未跑完 ($s/$T1), 中止"; exit 1; }

# ───────────────────── 阶段 2: BPTT 6ep（从阶段 1 热启动）─────────────────────
INIT=$S1/ckpt_best.pt
[ -f "$INIT" ] || INIT=$S1/ckpt_last.pt
[ -f "$INIT" ] || { log "!! 找不到阶段1权重, 中止"; exit 1; }
log "阶段2 热启动来源: $INIT"

for i in $(seq 1 12); do
  s=$(step_of "$S2")
  if [ "$s" -ge "$T2" ]; then log "阶段2 完成: $s/$T2 步"; break; fi
  if [ -f "$S2/ckpt_last.pt" ]; then
    r=(--resume "$S2/ckpt_last.pt")
  else
    r=(--init_from "$INIT")
  fi
  log "阶段2 第 $i 次启动 (当前 $s/$T2) ${r[*]}"
  TAG=stdattn_k64_rec BS=64 EPOCHS=6 TIME_BUDGET=2.0 DEC_RECURRENT=1 OUT="$S2" \
    ./run_stdattn.sh "${r[@]}" >> "$ROOT/stdattn_k64_rec_launch.log" 2>&1
  log "阶段2 本次退出码 $?; 进度 $(step_of "$S2")/$T2"
  sleep 15
done
s=$(step_of "$S2"); [ "$s" -ge "$T2" ] || { log "!! 阶段2 未跑完 ($s/$T2), 中止"; exit 1; }

# ─────────────────────────── 收尾: 四臂同条件评测 ───────────────────────────
log "=== 全量留出集评测 (5000 张, 四臂同进程同批次) ==="
"$PY" -u eval_linattn.py \
  --ckpt "$S1/ckpt_best.pt" "$S2/ckpt_best.pt" "$L1/ckpt_best.pt" "$L2/ckpt_best.pt" \
  --num_images 0 --batch_size 64 --num_workers 12 \
  --out "$ROOT/eval_ablation_full.json" \
  --grid "$ROOT/FINAL_grid_ablation.png" >> "$ROOT/eval_ablation.log" 2>&1 \
  && log "全量评测 OK → $ROOT/eval_ablation_full.json" \
  || log "!! 全量评测失败, 见 eval_ablation.log"

log "=== 逐步(循环)重建评测 (std BPTT 臂) ==="
"$PY" -u eval_stepwise_linattn.py \
  --ckpt "$S2/ckpt_best.pt" --num_images 0 --batch_size 64 \
  --out "$ROOT/stepwise_stdattn_rec.json" >> "$ROOT/stepwise.log" 2>&1 \
  && log "逐步评测 OK → $ROOT/stepwise_stdattn_rec.json" \
  || log "!! 逐步评测失败, 见 stepwise.log"

log "=== 训练曲线 ==="
"$PY" -u plot_linattn.py --log "$S1/train_log.jsonl" --out "$S1/curves.png" \
  --title "stdattn (plain softmax-transformer encoder 31.8M + decoder 31.2M): single-step, ImageNet-100 224x224" \
  >> "$ROOT/plot.log" 2>&1 || log "!! 阶段1 曲线失败"
"$PY" -u plot_linattn.py --log "$S2/train_log.jsonl" --out "$S2/curves.png" \
  --title "stdattn (plain softmax-transformer encoder 31.8M + decoder 31.2M): recurrent readout (BPTT, T=8)" \
  >> "$ROOT/plot.log" 2>&1 || log "!! 阶段2 曲线失败"

log "=== 全部完成 ==="
