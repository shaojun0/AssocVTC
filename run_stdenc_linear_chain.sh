#!/usr/bin/env bash
# =============================================================================
# 最终臂「标准编码器 + 线性解码器」链式驱动 + 收尾关机 — 2026-10-16
#
#   编码器 = 普通 softmax 注意力 transformer（`--enc_attn std`）
#   解码器 = 线性注意力（自注意力 + 交叉注意力都是核形式 φ=elu+1, `--dec_attn linear`）
#   读出   = BPTT 循环（阶段 2）
#
#   本臂跑完后, 2×2×2 设计矩阵（编码器注意力 × 解码器注意力 × 读出方式）8 臂齐全。
#
#   行为:
#     0) 先等正在跑的「全线性」链 (run_alllinear_chain.sh) 完成 —— 不抢 GPU
#     1) 阶段 1: 本臂单步 bs128 × 45ep（与其它臂逐项同协议）
#     2) 阶段 2: 本臂 BPTT bs64 × 6ep（从阶段 1 热启动, 开梯度检查点放进显存）
#     3) 收尾: **8 臂同进程同批次**全量评测 + 逐步重建 + 训练曲线
#     4) 落一个 QUEUE_DONE 标记, 然后等 SHUTDOWN_NOW 哨兵文件（最多 6h）→ 关机
#        （哨兵机制: 让本地先把结果下载/提交/推送完成, 再安全关机;
#          即便本地不来, 最多 6h 后也会自动关机, 不会空烧机器）
#
# 用法: nohup setsid bash run_stdenc_linear_chain.sh > /root/autodl-tmp/srdiff_final/nohup.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/datasets/imagenet100
ALL=/root/autodl-tmp/srdiff_alllinear
QROOT=/root/autodl-tmp/srdiff_final
SL=$QROOT/stdlin_k64              # 本臂: 单步
SR=$QROOT/stdlin_k64_rec          # 本臂: BPTT
L1=/root/autodl-tmp/srdiff_linattn/k64
L2=/root/autodl-tmp/srdiff_linattn/k64_rec
ST1=/root/autodl-tmp/srdiff_stdattn/stdattn_k64
ST2=/root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec
AL=$ALL/alllinear_k64
AR=$ALL/alllinear_k64_rec
LOG=$QROOT/chain.log
T1=$((950 * 45))
T2=$((1901 * 6))
mkdir -p "$QROOT"

log() { echo "[chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
step_of() {
  local d=$1 s=0
  if [ -f "$d/status.json" ]; then
    s=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('step',0))" \
        "$d/status.json" 2>/dev/null || echo 0)
  fi
  echo "${s:-0}"
}

log "=== 最终臂(标准编码器 + 线性解码器)启动 (pid $$) ==="

# ── 0) 等「全线性」链完成（最多 ~10h）────────────────────────────────────
log "等待全线性链 (run_alllinear_chain.sh) 完成 ..."
for i in $(seq 1 600); do
  if grep -q "全部完成" "$ALL/chain.log" 2>/dev/null; then log "全线性链已完成"; break; fi
  if [ $((i % 30)) -eq 0 ]; then
    log "  ... 仍在等 (已等 $((i)) 分钟; 全线性进度 $(step_of "$ALL/alllinear_k64")/$T1)"
  fi
  sleep 60
done
if grep -q "全部完成" "$ALL/chain.log" 2>/dev/null; then
  log "全线性链 OK（若其评测失败, 本臂收尾的 8 臂评测会重新覆盖全部臂）"
else
  log "!! 全线性链 10h 内未报完成 —— 继续本臂（不阻塞）"
fi

# ── 1) 本臂 阶段 1: std 编码器 + 线性解码器, 单步, bs128 × 45ep ──────────
for i in $(seq 1 12); do
  s=$(step_of "$SL")
  if [ "$s" -ge "$T1" ]; then log "阶段1 完成: $s/$T1 步"; break; fi
  r=(); [ -f "$SL/ckpt_last.pt" ] && r=(--resume "$SL/ckpt_last.pt")
  log "阶段1 第 $i 次启动 (当前 $s/$T1) ${r[*]:-冷启动}"
  "$PY" -u train_linattn.py --data_dir "$DATA" --output_dir "$SL" \
    --enc_attn std --dec_attn linear --num_specials 64 \
    --batch_size 128 --epochs 45 --lr 2e-4 --dec_recurrent 0 \
    --time_budget_h 6.0 "${r[@]}" >> "$QROOT/stdlin_k64_launch.log" 2>&1
  log "阶段1 本次退出码 $?; 进度 $(step_of "$SL")/$T1"
  sleep 15
done
s=$(step_of "$SL"); [ "$s" -ge "$T1" ] || { log "!! 阶段1 未跑完 ($s/$T1)"; }

# ── 2) 本臂 阶段 2: BPTT, bs64 × 6ep, 从阶段 1 热启动 ────────────────────
INIT=$SL/ckpt_best.pt
[ -f "$INIT" ] || INIT=$SL/ckpt_last.pt
log "阶段2 热启动来源: $INIT"
for i in $(seq 1 12); do
  s=$(step_of "$SR")
  if [ "$s" -ge "$T2" ]; then log "阶段2 完成: $s/$T2 步"; break; fi
  if [ -f "$SR/ckpt_last.pt" ]; then r=(--resume "$SR/ckpt_last.pt")
  elif [ -f "$INIT" ]; then r=(--init_from "$INIT")
  else log "!! 找不到阶段1权重, 跳过阶段2"; break; fi
  log "阶段2 第 $i 次启动 (当前 $s/$T2) ${r[*]}"
  "$PY" -u train_linattn.py --data_dir "$DATA" --output_dir "$SR" \
    --enc_attn std --dec_attn linear --num_specials 64 \
    --batch_size 64 --epochs 6 --lr 2e-4 --dec_recurrent 1 --dec_grad_ckpt 1 \
    --time_budget_h 3.0 "${r[@]}" >> "$QROOT/stdlin_k64_rec_launch.log" 2>&1
  log "阶段2 本次退出码 $?; 进度 $(step_of "$SR")/$T2"
  sleep 15
done

# ── 3) 收尾: 8 臂同进程同批次全量评测 ────────────────────────────────────
log "=== 8 臂全量留出集评测（同进程同批次）==="
"$PY" -u eval_linattn.py \
  --ckpt "$SL/ckpt_best.pt" "$SR/ckpt_best.pt" \
         "$AL/ckpt_best.pt" "$AR/ckpt_best.pt" \
         "$L1/ckpt_best.pt" "$ST1/ckpt_best.pt" \
         "$L2/ckpt_best.pt" "$ST2/ckpt_best.pt" \
  --num_images 0 --batch_size 64 --num_workers 12 \
  --out "$QROOT/eval_ablation8.json" \
  --grid "$QROOT/FINAL_grid_8arms.png" >> "$QROOT/eval_ablation.log" 2>&1 \
  && log "8 臂评测 OK → $QROOT/eval_ablation8.json" \
  || log "!! 8 臂评测失败, 见 eval_ablation.log"

# 每臂缺失的 ckpt 会让上面报错; 若失败, 退化为"只评存在的臂"
if [ ! -s "$QROOT/eval_ablation8.json" ]; then
  log "退化: 仅评本轮两条新臂"
  EX=(); for c in "$SL/ckpt_best.pt" "$SR/ckpt_best.pt"; do [ -f "$c" ] && EX+=("$c"); done
  [ ${#EX[@]} -gt 0 ] && "$PY" -u eval_linattn.py --ckpt "${EX[@]}" \
    --num_images 0 --batch_size 64 --out "$QROOT/eval_ablation8.json" \
    >> "$QROOT/eval_ablation.log" 2>&1 && log "退化评测 OK"
fi

log "=== 逐步(循环)重建 —— 本臂 BPTT ==="
[ -f "$SR/ckpt_best.pt" ] && "$PY" -u eval_stepwise_linattn.py \
  --ckpt "$SR/ckpt_best.pt" --num_images 0 --batch_size 64 \
  --out "$QROOT/stepwise_stdlin_rec.json" >> "$QROOT/stepwise.log" 2>&1 \
  && log "逐步评测 OK" || log "!! 逐步评测失败/跳过"

log "=== 训练曲线 ==="
[ -f "$SL/train_log.jsonl" ] && "$PY" -u plot_linattn.py --log "$SL/train_log.jsonl" \
  --out "$SL/curves.png" --title "std-enc + linear-dec: single-step, ImageNet-100 224x224" \
  >> "$QROOT/plot.log" 2>&1 || true
[ -f "$SR/train_log.jsonl" ] && "$PY" -u plot_linattn.py --log "$SR/train_log.jsonl" \
  --out "$SR/curves.png" --title "std-enc + linear-dec: recurrent readout (BPTT, T=8)" \
  >> "$QROOT/plot.log" 2>&1 || true

# ── 4) 完成标记 + 关机握手 ───────────────────────────────────────────────
date '+%F %T' > "$QROOT/QUEUE_DONE"
log "=== 全部实验完成, 已落 QUEUE_DONE ==="

log "等待哨兵 $QROOT/SHUTDOWN_NOW 后关机（最多 6h; 本地需先下载/提交/推送）"
for i in $(seq 1 360); do
  [ -f "$QROOT/SHUTDOWN_NOW" ] && { log "收到 SHUTDOWN_NOW"; break; }
  [ $((i % 30)) -eq 0 ] && log "  ... 等待关机哨兵中 (已等 $((i)) 分钟)"
  sleep 60
done

log "=== 执行关机 (shutdown -h now) ==="
sync
if shutdown -h now 2>&1 | tee -a "$LOG"; then
  log "shutdown 已发出"
else
  log "!! shutdown 返回非零, 尝试 poweroff"
  poweroff -f 2>&1 | tee -a "$LOG" || log "!! poweroff 也失败 —— 请在 AutoDL 控制台手动关机"
fi
