#!/bin/bash
# =============================================================================
# run_alit_b.sh — PLAN §4 阶段 B（重训臂）执行脚本         [task-4 · owner: alit-runner]
# -----------------------------------------------------------------------------
# 目标机: 新实例 ssh -p 18540 root@connect.westc.seetacloud.com  (RTX 4080S, 独占)
# 在服务器上运行:
#   cd /root/AssocVTC && nohup setsid bash run_alit_b.sh \
#       --arms b1_mean,b1_decay,b1_increasing,b1_lastonly \
#       > doc/2026-10-16/artifacts/alit/logs/run_alit_b.log 2>&1 &
#
# 统一协议（PLAN §4）: IN-100 224×224, K=64, enc_attn=std, dec_recurrent=1(T=8),
#   从 stdattn_k64/ckpt_best.pt 热启动, 6 epoch / 11406 步 / bs64 / bf16 / lr2e-4。
#   每臂完成后**立即**同进程全量 5000 张评测 + 逐步评测。
#
# 特性: 断点续跑（status.json/ckpt_last.pt 判定步数, --resume 续跑）、每臂硬超时、
#       失败不阻塞后续臂、全部 launch log 落盘、缺 CLI 开关的臂显式 SKIP。
# 用法: bash run_alit_b.sh [--arms LIST] [--budget_h 3.0] [--timeout_h 5.0] [--force]
# =============================================================================
set -uo pipefail

PY=/root/miniconda3/bin/python
REPO=${REPO:-/root/AssocVTC}
A=$REPO/doc/2026-10-16/artifacts/alit
LOGS=$A/logs
P=/root/autodl-tmp

CKPT_INIT=$P/srdiff_stdattn/stdattn_k64/ckpt_best.pt
CKPT_STD_REC=$P/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt
CKPT_STD=$P/srdiff_stdattn/stdattn_k64/ckpt_best.pt
TOTAL_STEPS=11406
BUDGET_H=3.0
TIMEOUT_H=5.0
MAX_ATTEMPTS=4
ARMS=""
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --arms)      ARMS="$2"; shift 2 ;;
    --budget_h)  BUDGET_H="$2"; shift 2 ;;
    --timeout_h) TIMEOUT_H="$2"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO" || exit 3
mkdir -p "$A" "$LOGS" "$P/srdiff_alit"
BLOG=$LOGS/run_alit_b.progress.log
BTSV=$A/alit_b_arms.tsv
# 追加语义（历史不丢）: 只在文件不存在时写表头
if [ ! -f "$BTSV" ]; then printf "arm\tstatus\trc\tts\n" > "$BTSV"; fi

log()  { echo "[$(date -Is)] $*" | tee -a "$BLOG"; }
want() { [ -z "$ARMS" ] && return 0; case ",$ARMS," in *",$1,"*) return 0;; esac; return 1; }

# ── 臂表: 名字 | 额外训练参数 | 必需 CLI 开关(空=无条件) ─────────────────────
declare -A EXTRA REQ NOTE
EXTRA[b1_mean]="";              REQ[b1_mean]="";                 NOTE[b1_mean]="基线复跑 mean_t（=E-B3 臂1）"
EXTRA[b1_decay]="--loss_weights 2.5,2,1.5,1.25,1,1,1,1"; REQ[b1_decay]="--loss_weights"; NOTE[b1_decay]="ALIT 原版递减"
EXTRA[b1_increasing]="--loss_weights 1,1,1,1,1.25,1.5,2,2.5"; REQ[b1_increasing]="--loss_weights"; NOTE[b1_increasing]="递增（让后步干活）"
EXTRA[b1_lastonly]="--loss_weights 0,0,0,0,0,0,0,1"; REQ[b1_lastonly]="--loss_weights"; NOTE[b1_lastonly]="last-only"
EXTRA[b2_dh1]="--dh_mode 1";                 REQ[b2_dh1]="--dh_mode";        NOTE[b2_dh1]="硬区域掩码"
EXTRA[b2_dh2]="--dh_mode 2 --dh_quantile 0.5"; REQ[b2_dh2]="--dh_quantile";  NOTE[b2_dh2]="软停机 50%"
EXTRA[b2_dh3]="--dh_mode 3 --dh_quantile 0.5"; REQ[b2_dh3]="--dh_quantile";  NOTE[b2_dh3]="软停机+stop-grad"
EXTRA[b3_grow]="--refiner_layers 1 --refiner_grow 16"; REQ[b3_grow]="--refiner_layers"; NOTE[b3_grow]="E-B3 方案a GrowthBlock"
EXTRA[b4_sg]="--sample_grad_iters 3";        REQ[b4_sg]="--sample_grad_iters"; NOTE[b4_sg]="E-B4 随机预算反传"
# 专训档 K'=16/32/48 已按 Lead 决策**不做**（无法热启动 ⇒ 须从零, 3 臂 ≈15h, 超预算）

# 执行顺序（Lead 2026-10-16 决策: B 阶段全部在新实例串行, 不跨机并行）:
#   b1_mean → b1_decay → b1_increasing → b1_lastonly → b3_grow → b2_dh1/dh2/dh3 → b4_sg
# 不做 b4_k16/k32/k48 专训档（无法热启动, 3 臂从零 ≈15h 超预算 ⇒ 报告记"数据不足+成本理由"）。
# E-B4 的"截断档 t=3/5/7"**不需要额外训练**: b4_sg 的 stepwise json 已含 L1_255[t] 全档,
# 直接读 t=3/5/7 即截断档指标（与专训档对照则因上述理由不做）。
DEFAULT_ORDER="b1_mean b1_decay b1_increasing b1_lastonly b3_grow b2_dh1 b2_dh2 b2_dh3 b4_sg"

# ── preflight ────────────────────────────────────────────────────────────────
log "########## run_alit_b.sh START arms='${ARMS:-default}' budget=${BUDGET_H}h timeout=${TIMEOUT_H}h ##########"
for f in "$CKPT_INIT" "$CKPT_STD_REC" "$REPO/train_linattn.py" "$REPO/eval_linattn.py" \
         "$REPO/eval_stepwise_linattn.py"; do
  [ -e "$f" ] || { log "FATAL 缺文件: $f"; exit 4; }
done
TRAIN_HELP=$("$PY" "$REPO/train_linattn.py" --help 2>&1 || true)
flag_ok() { [ -z "$1" ] && return 0; echo "$TRAIN_HELP" | grep -q -- "$1"; }
BUSY=$(pgrep -af "train_linattn.py|eval_linattn.py|eval_stepwise_linattn.py" | grep -v run_alit_b | wc -l)
if [ "$BUSY" -gt 0 ] && [ "$FORCE" -eq 0 ]; then
  log "!!! 检测到 $BUSY 个并发 GPU 进程, GPU 必须独占 ⇒ 中止（确认无冲突后加 --force）:"
  pgrep -af 'train_linattn.py|eval_' | head -5 | tee -a "$BLOG"; exit 5
fi
log "preflight: GPU=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"

status_step() { "$PY" - "$1" <<'PYX' 2>/dev/null || echo 0
import json, sys, os
p = os.path.join(sys.argv[1], "status.json")
try: print(int(json.load(open(p)).get("step", 0)))
except Exception: print(0)
PYX
}

# ── 单臂: 训练(含续跑) → 同进程四臂评测 → 逐步评测 ───────────────────────────
run_arm() {
  local name="$1"
  local out=$P/srdiff_alit/$name
  local t_start; t_start=$(date +%s)

  if [ -n "${REQ[$name]}" ] && ! flag_ok "${REQ[$name]}"; then
    printf "%s\tSKIP\t%s\t-\n" "$name" "train_linattn.py 缺 ${REQ[$name]}" >> "$BTSV"
    log "SKIP $name : train_linattn.py 无 ${REQ[$name]}（需 task-2 rsync）"; return 0
  fi

  if [ "$name" = "b3_base" ]; then out=$P/srdiff_alit/b1_mean; log "$name 复用 b1_mean（PLAN: 臂1 必须同预算复跑）"; fi

  mkdir -p "$out"
  local -a EXTRA_ARR=(); [ -n "${EXTRA[$name]:-}" ] && read -r -a EXTRA_ARR <<< "${EXTRA[$name]}"

  # 训练（断点续跑循环）
  if [ "$name" != "b3_base" ]; then
    local attempt=0 last_step=-1
    while :; do
      local cur; cur=$(status_step "$out")
      if [ "$cur" -ge "$TOTAL_STEPS" ]; then log "$name 训练已满 $cur/$TOTAL_STEPS 步"; break; fi
      attempt=$((attempt + 1))
      if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then log "$name 达最大重试 $MAX_ATTEMPTS, 停止续跑（step=$cur）"; break; fi
      local -a R=(); [ -f "$out/ckpt_last.pt" ] && R=(--resume "$out/ckpt_last.pt")
      local tsec; tsec=$(awk -v h="$TIMEOUT_H" 'BEGIN{printf "%.0f", h*3600}')
      log "$name attempt=$attempt start_step=$cur resume=${R[*]:-none} hard_timeout=${tsec}s"
      timeout -k 60 "$tsec" \
        "$PY" -u train_linattn.py \
          --data_dir "$P/datasets/imagenet100" --output_dir "$out" \
          --enc_attn std --num_specials 64 --batch_size 64 --epochs 6 --lr 2e-4 \
          --dec_recurrent 1 --dec_grad_ckpt 1 --init_from "$CKPT_INIT" \
          --time_budget_h "$BUDGET_H" "${EXTRA_ARR[@]}" "${R[@]}" \
          > "$out/launch_$attempt.log" 2>&1
      local rc=$? new; new=$(status_step "$out")
      log "$name attempt=$attempt rc=$rc step=$cur→$new  (log: $out/launch_$attempt.log)"
      if [ "$new" -ge "$TOTAL_STEPS" ]; then break; fi
      if [ "$new" -le "$cur" ]; then log "$name 无进展（step 未增加）⇒ 终止该臂训练"; break; fi
    done
  fi

  # 评测（同进程: 新臂 + 对照 rec + 单步上界）
  if [ -f "$out/ckpt_best.pt" ]; then
    "$PY" eval_linattn.py --ckpt "$out/ckpt_best.pt" "$CKPT_STD_REC" "$CKPT_STD" \
      --num_images 0 --batch_size 128 --num_workers 16 \
      --out "$A/eval_$name.json" >> "$LOGS/eval_$name.log" 2>&1
    local rc1=$?
    "$PY" eval_stepwise_linattn.py --ckpt "$out/ckpt_best.pt" --num_images 0 \
      --batch_size 64 --num_workers 16 --out "$A/stepwise_$name.json" \
      >> "$LOGS/stepwise_$name.log" 2>&1
    local rc2=$?
    printf "%s\tOK\trc_eval=%s rc_stepwise=%s\t%s\n" "$name" "$rc1" "$rc2" "$(date -Is)" >> "$BTSV"
    log "$name 评测完成 rc_eval=$rc1 rc_stepwise=$rc2 → $A/eval_$name.json"
  else
    printf "%s\tFAIL\t无 ckpt_best.pt\t%s\n" "$name" "$(date -Is)" >> "$BTSV"
    log "!!! $name FAIL: $out/ckpt_best.pt 不存在（训练未产出）"
  fi
  log "$name 总耗时 $(( ($(date +%s) - t_start) / 60 ))min"
}

run_arm_safe() { run_arm "$1" || log "!!! $1 抛错（继续后续臂）"; }

# ── 主循环 ───────────────────────────────────────────────────────────────────
if [ -z "$ARMS" ]; then
  for n in $DEFAULT_ORDER; do run_arm_safe "$n"; done
else
  IFS=',' read -r -a LIST <<< "$ARMS"
  for n in "${LIST[@]}"; do n=$(echo "$n" | tr -d ' '); run_arm_safe "$n"; done
fi

log "########## run_alit_b.sh DONE ##########"
cat "$BTSV" | tee -a "$BLOG"
echo ">>> 控制机取回: rsync -rltzv -e 'ssh -p 18540' root@connect.westc.seetacloud.com:$A/ /home/linaro/dsh/AssocVTC/doc/2026-10-16/artifacts/alit/"
