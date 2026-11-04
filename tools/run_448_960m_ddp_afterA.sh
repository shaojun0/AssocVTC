#!/usr/bin/env bash
# =============================================================================
# run_448_960m_ddp_afterA.sh — A 相结束后**自动接力**: 真实尺寸 DDP 冒烟 → A 评测
#                              → B 相 2×A800 DDP → B 评测 → S3 相 DDP → S3 评测 → ALL_DONE
# -----------------------------------------------------------------------------
# 2026-11-04 · DDP 提速改造的自动接力驱动（原链路 tools/run_448_960m_3stage.sh 在 A 相
# 跑到一半时被人工 kill，A 训练进程本身仍在单卡 GPU0 上跑完；本脚本等它跑完后接力）。
#
# 为什么这样分工
#   · A 相（T=1，45ep，42,750 步）已经在 GPU0 单卡跑了 ~10h，**不能中断**（中断就白跑）；
#   · 但 B/S3 两相（各 11,406 / 5,703 步，单卡 14.0h / 6.9h）可以改成 2×A800 DDP，
#     预计 1.6–1.9× 提速 ⇒ 本脚本在 A 结束后自动把 GPU1 也用上。
#   · 协议不可变: 等效批量 A=128 / B/S3=64、epoch 45/6/3、步数 42,750 / 11,406 / 5,703、
#     lr 2e-4 cosine 8% warmup、wd 0.05、clip 1.0、bf16、seed 0。
#
# 执行顺序（严格；每步都幂等）
#   0) 等待 A 相结束: `$OUT/A/ckpt_best.pt` 存在 **且** A 训练进程（train_linattn.py 且
#      --output_dir $OUT/A）已退出 **且** stage1.log 有 `[done] steps=42750`
#      （进程消失但没有 [done] ⇒ A 是崩的，写 FAILED_stage1 并中止，绝不拿半训 ckpt 起 B）；
#      确认后写 STAGE1_DONE（老链路被 kill 后没人写过这个标记，watchdog 靠它切阶段）
#   1) 真实尺寸 DDP 冒烟: B 相配置跑 30 步（全量数据 ⇒ 同时验证 steps/epoch=1901），
#      由 train_log.jsonl 的 elapsed_min 差商算 steady s/step，对照单卡 4.41 s/step；
#      speedup < 1.3× ⇒ 写 ALERT + **回退单卡**（安全阀；FORCE_DDP=1 强制用 DDP）。
#      注意: 真实尺寸 2 卡冒烟**只能**在 A 结束后做（A 占着 GPU0）。
#   2) A 相评测（GPU1，约 2.5 min）→ $OUT/eval/stage1_stepwise.json
#   3) B 相: torchrun 2 rank DDP（等效 batch 64 = 每卡 micro16 × 每卡 accum2 × 2 卡）
#      → $OUT/logs/stage2.log（watchdog 依赖此文件名）→ STAGE2_DONE，删 ckpt_last
#   4) B 相评测（GPU1，串行: DDP 训练占两张卡, 不能并行）→ stage2_stepwise.json
#   5) S3 相 DDP（--zs_dim 224，从 B/ckpt_best 热启动）→ logs/stage3.log → STAGE3_DONE
#   6) S3 相评测 → stage3_stepwise.json
#   7) ALL_DONE
#
# 失败语义
#   · 训练 rc≠0 / ckpt_best 缺失 / 日志无 `[done] steps=<协议步数>` / steps/epoch≠1901
#     ⇒ `FAILED_stage{N}` 并中止（watchdog 会报 FAILED_MARKER）
#   · 评测 rc≠0 或 json 未产出 ⇒ `FAILED_eval_stage{N}` 并中止
#   · DDP 冒烟失败或太慢且 FORCE_DDP=1 ⇒ `FAILED_ddp_smoke` 并中止
#
# 幂等（断电/重启/重复执行都安全）
#   · 某阶段已有 ckpt_best.pt **且** 该阶段日志有 `[done] steps=<协议步数>` ⇒ 跳过训练
#   · 评测 json 已存在 ⇒ 跳过评测
#   · ALL_DONE 已存在 ⇒ 直接退出（FORCE=1 重跑）
#   · flock 单实例锁（$OUT/.ddp_afterA.lock），不会两个驱动同时跑
#
# 与 watchdog 的兼容（**不重启 watchdog**）
#   · 训练日志仍写 logs/stage2.log / logs/stage3.log；标记仍用 STAGE{1,2,3}_DONE /
#     ALL_DONE / FAILED_*；A 阶段日志 logs/stage1.log 只读不写。
#   · ⚠️ watchdog 的 CHAIN_PATTERN 硬编码 `run_448_960m_3stage.sh`，本脚本名不同 ⇒
#     watchdog 看不到"链路进程"。由于本脚本在阶段间几乎总有 train/eval 进程（间隔秒级），
#     实际不会误报；万一驱动本身死掉，`$OUT/afterA.heartbeat`（每 30s 更新）停止更新，
#     外加缺少后续 STAGE*_DONE/ALL_DONE，可据此发现（DSH 定时任务/人工查 heartbeat）。
#
# 用法（远端仓库根目录）
#   DRY=1 bash tools/run_448_960m_ddp_afterA.sh          # 干跑: 打印全部命令, 不等待/不执行
#   nohup setsid bash tools/run_448_960m_ddp_afterA.sh \
#       > /root/autodl-tmp/exp448/logs/afterA.out 2>&1 < /dev/null &   # 正式启动
#
# 环境变量
#   OUT/DATA/PY/GPU_TRAIN(0)/GPU_EVAL(1)/NPROC(2)/NUM_WORKERS(16)/EVAL_BS(16)/EVAL_WORKERS(8)
#   WAIT_POLL(30)          等 A 结束的轮询秒数
#   MAX_WAIT_H(30)         最多等 A 多少小时（超时写 FAILED_stage1 并退出；0=无限等）
#   WAIT_GPU_MIN(5)        单卡回退时等目标卡显存的分钟数
#   SMOKE_WAIT_GPU_MIN(30) 真实尺寸 DDP 冒烟前等**两张卡**都空出的分钟数（GPU1 可能被别的
#                          作业临时占用；等不到才降级。⚠️ 等 >10min 期间没有 train/eval 进程，
#                          watchdog 可能误报一次 CRASHED —— 用 afterA.heartbeat 分辨）
#   STAGE_WAIT_GPU_MIN(10) B/S3 DDP 开跑前等两张卡的分钟数
#   MIN_SPEEDUP(1.3)       DDP 冒烟 speedup 安全阀（< 该值回退单卡）
#   SINGLE_REF_S_STEP(4.41) 单卡 B/S3 实测 s/step（bench 4.41；仅用于算 speedup）
#   FORCE_DDP(0)           1 = 即使冒烟失败/太慢也强制 DDP（失败则 FAILED_ddp_smoke）
#   FORCE(0)               1 = 重跑（忽略已有 ckpt/marker/eval json）
#   KEEP_LAST(0)           1 = 保留 ckpt_last.pt（默认删，960M fp32 约 3.8G/个）
#   SMOKE_STEPS(30)        真实尺寸 DDP 冒烟步数（log_interval 5 ⇒ 用 10→30 段算 steady）
#   SMOKE_TIMEOUT(1800)    冒烟超时秒数（超时视为 DDP 有问题）
#   NO_EVAL(0)             1 = 跳过全部评测（只训）
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "[afterA] ✗ 请在仓库内执行（tools/ 的上一级）" >&2; exit 2; }
ROOT=$PWD

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/exp448}
GPU_TRAIN=${GPU_TRAIN:-0}
GPU_EVAL=${GPU_EVAL:-1}
NPROC=${NPROC:-2}
NUM_WORKERS=${NUM_WORKERS:-16}
EVAL_BS=${EVAL_BS:-16}
EVAL_WORKERS=${EVAL_WORKERS:-8}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
WAIT_POLL=${WAIT_POLL:-30}
MAX_WAIT_H=${MAX_WAIT_H:-30}
WAIT_GPU_MIN=${WAIT_GPU_MIN:-5}
SMOKE_WAIT_GPU_MIN=${SMOKE_WAIT_GPU_MIN:-30}
STAGE_WAIT_GPU_MIN=${STAGE_WAIT_GPU_MIN:-10}
MIN_SPEEDUP=${MIN_SPEEDUP:-1.3}
SINGLE_REF_S_STEP=${SINGLE_REF_S_STEP:-4.41}
FORCE_DDP=${FORCE_DDP:-0}
FORCE=${FORCE:-0}
KEEP_LAST=${KEEP_LAST:-0}
SMOKE_STEPS=${SMOKE_STEPS:-30}
SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-1800}
NO_EVAL=${NO_EVAL:-0}
DRY=${DRY:-0}

# ── 协议常量（不可变）──────────────────────────────────────────────────────
EP1=${EP1:-45}; EP2=${EP2:-6}; EP3=${EP3:-3}
STEPS_A=42750          # 950 步/ep × 45 ep（drop_last；协议写 42,795）
STEPS_EP=1901          # B/S3 每 epoch 步数（等效 batch 64）
STEPS_B=$((STEPS_EP * EP2))      # 11,406
STEPS_S3=$((STEPS_EP * EP3))     # 5,703
ZS_DIM=${ZS_DIM:-224}
LR=${LR:-2e-4}; MIN_LR=${MIN_LR:-1e-6}; WD=${WD:-0.05}
WARMUP=${WARMUP:-0.08}; CLIP=${CLIP:-1.0}; SEED=${SEED:-0}
IMG=${IMG:-448}; PATCH=${PATCH:-32}; K=${K:-32}
ENC_DIM=${ENC_DIM:-1744}; ENC_DEPTH=${ENC_DEPTH:-13}; ENC_HEADS=${ENC_HEADS:-16}
DEC_DIM=${DEC_DIM:-2048}; DEC_DEPTH=${DEC_DEPTH:-7}; DEC_HEADS=${DEC_HEADS:-16}
PIXEL_HIDDEN=${PIXEL_HIDDEN:-1024}; DEC_GRAD_CKPT=${DEC_GRAD_CKPT:-0}

ADIR=$OUT/A; BDIR=$OUT/B; SDIR=$OUT/S3
LOG1=$OUT/logs/stage1.log; LOG2=$OUT/logs/stage2.log; LOG3=$OUT/logs/stage3.log
DEC_STEPS=$(seq -s, 1 16)
SELF=run_448_960m_ddp_afterA.sh

say()  { echo "[afterA] $*"; }
warn() { echo "[afterA] ⚠️ $*" >&2; }
die()  { echo "[afterA] ✗ $*" >&2; exit "${2:-1}"; }
show_cmd() { local pfx=$1 a; shift; printf '%s' "$pfx"; for a in "$@"; do printf ' %q' "$a"; done; printf '\n'; }
py_ok() { case "$PY" in */*) [ -x "$PY" ] ;; *) command -v "$PY" >/dev/null 2>&1 ;; esac; }
touch_heartbeat() { [ "$DRY" = "1" ] || date +%s > "$OUT/afterA.heartbeat" 2>/dev/null || true; }

# ── 命令构建 ───────────────────────────────────────────────────────────────
ARCH=( --img_size "$IMG" --patch "$PATCH" --num_specials "$K"
       --pixel_hidden "$PIXEL_HIDDEN" --enc_attn std
       --enc_dim "$ENC_DIM" --enc_depth "$ENC_DEPTH" --enc_heads "$ENC_HEADS"
       --dec_dim "$DEC_DIM" --dec_depth "$DEC_DEPTH" --dec_heads "$DEC_HEADS" )
OPT=( --lr "$LR" --min_lr "$MIN_LR" --weight_decay "$WD" --warmup_frac "$WARMUP"
      --grad_clip "$CLIP" --amp 1 --seed "$SEED" --save_opt 0
      --num_workers "$NUM_WORKERS" --dec_grad_ckpt "$DEC_GRAD_CKPT" --time_budget_h 0 )
# DDP: 等效 batch 64 = 每卡 micro 16 × 每卡 accum 2 × 2 卡（与单卡 micro16×accum4 等价）
ARGS_B_DDP=( --data_dir "$DATA" --output_dir "$BDIR" "${ARCH[@]}"
             --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps "$DEC_STEPS"
             --init_from "$ADIR/ckpt_best.pt" --epochs "$EP2"
             --batch_size 64 --accum_steps 2 "${OPT[@]}" )
ARGS_S3_DDP=( --data_dir "$DATA" --output_dir "$SDIR" "${ARCH[@]}"
              --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps "$DEC_STEPS"
              --zs_dim "$ZS_DIM" --init_from "$BDIR/ckpt_best.pt" --epochs "$EP3"
              --batch_size 64 --accum_steps 2 "${OPT[@]}" )
# 回退单卡: micro16 × accum4（等效 batch 64，步数不变，与 bench 的单卡口径一致）
ARGS_B_1=( --data_dir "$DATA" --output_dir "$BDIR" "${ARCH[@]}"
           --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps "$DEC_STEPS"
           --init_from "$ADIR/ckpt_best.pt" --epochs "$EP2"
           --batch_size 64 --accum_steps 4 "${OPT[@]}" )
ARGS_S3_1=( --data_dir "$DATA" --output_dir "$SDIR" "${ARCH[@]}"
            --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps "$DEC_STEPS"
            --zs_dim "$ZS_DIM" --init_from "$BDIR/ckpt_best.pt" --epochs "$EP3"
            --batch_size 64 --accum_steps 4 "${OPT[@]}" )
# 真实尺寸冒烟（B 相配置 + 30 步 + 关 val/采样，全量数据 ⇒ 顺带验证 steps/epoch=1901）
ARGS_SMOKE=( --data_dir "$DATA" --output_dir "$OUT/ddp_smoke_B" "${ARCH[@]}"
             --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps "$DEC_STEPS"
             --init_from "$ADIR/ckpt_best.pt" --epochs 1
             --batch_size 64 --accum_steps 2 --max_steps "$SMOKE_STEPS"
             --log_interval 5 --val_batches 0 --sample_interval 0 "${OPT[@]}" )

# ── A 相进程识别（只按 `--output_dir $OUT/A` 精确匹配，绝不误伤别的训练/冒烟）──
a_train_pids() {
  # ⚠️ 坑（沙箱自测抓到）: 不能把匹配串当 awk 的 -v 参数 —— 那条 awk 进程自己的
  # cmdline 里就含 "train_linattn.py --output_dir <ADIR>"，会被 ps 列出来自匹配，
  # 导致"A 永远在跑"、接力永远不启动。用 ENVIRON 传匹配串，并按 PID 排除自己。
  ps -eo pid=,args= 2>/dev/null \
    | SELF_PID=$$ PAT="train_linattn.py" OD="--output_dir $ADIR" awk '
        $1 + 0 == ENVIRON["SELF_PID"] { next }
        index($0, ENVIRON["PAT"]) && index($0, ENVIRON["OD"]) { print $1 }'
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
pick_port() {
  "$PY" - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()
PY
}

# ── 预检 ───────────────────────────────────────────────────────────────────
[ -f "$ROOT/train_linattn.py" ] || die "缺少 $ROOT/train_linattn.py（先把本地改动 rsync 过来）" 2
[ -f "$ROOT/tools/eval_448_960m.sh" ] || die "缺少 tools/eval_448_960m.sh" 2
py_ok || die "python 不存在: $PY" 2
[ -d "$DATA" ] || die "数据目录不存在: $DATA" 2

say "================ 448²/p32/960M · A 结束后 DDP 自动接力 ================"
say "$(date -Is)  ROOT=$ROOT  OUT=$OUT"
say "A 相完成判据: $ADIR/ckpt_best.pt 存在 且 A 训练进程已退出 且 stage1.log 有 [done] steps=$STEPS_A"
say "协议: A ${EP1}ep/${STEPS_A}步(单卡已跑)  B ${EP2}ep/${STEPS_B}步  S3 ${EP3}ep/${STEPS_S3}步  zs_dim=$ZS_DIM"
say "DDP: nproc=$NPROC 每卡 micro16×accum2（等效 batch 64）  单卡回退: micro16×accum4"
say "安全阀: DDP 冒烟 speedup < ${MIN_SPEEDUP}× （对照单卡 ${SINGLE_REF_S_STEP}s/step）⇒ 回退单卡"

if [ "$DRY" = "1" ]; then
  say "(dry) A 等待: 轮询每 ${WAIT_POLL}s，判据同上（本模式不等待、不执行）"
  show_cmd "[dry][smoke] CUDA_VISIBLE_DEVICES=$GPU_TRAIN,$GPU_EVAL $PY -m torch.distributed.run --nproc_per_node=$NPROC --master_port=<free> --max_restarts=0 $ROOT/train_linattn.py" "${ARGS_SMOKE[@]}"
  show_cmd "[dry][eval1] CKPT=$ADIR/ckpt_best.pt OUT=$OUT/eval GPU=$GPU_EVAL NUM_IMAGES=0 NAME=stage1_stepwise BS=$EVAL_BS bash tools/eval_448_960m.sh"
  show_cmd "[dry][stage2 ddp] CUDA_VISIBLE_DEVICES=$GPU_TRAIN,$GPU_EVAL $PY -m torch.distributed.run --nproc_per_node=$NPROC --master_port=<free> --max_restarts=0 $ROOT/train_linattn.py" "${ARGS_B_DDP[@]}"
  show_cmd "[dry][stage2 single] CUDA_VISIBLE_DEVICES=$GPU_TRAIN $PY -u $ROOT/train_linattn.py" "${ARGS_B_1[@]}"
  show_cmd "[dry][eval2] CKPT=$BDIR/ckpt_best.pt ... NAME=stage2 bash tools/eval_448_960m.sh"
  show_cmd "[dry][stage3 ddp] CUDA_VISIBLE_DEVICES=$GPU_TRAIN,$GPU_EVAL $PY -m torch.distributed.run --nproc_per_node=$NPROC --master_port=<free> --max_restarts=0 $ROOT/train_linattn.py" "${ARGS_S3_DDP[@]}"
  show_cmd "[dry][eval3] CKPT=$SDIR/ckpt_best.pt ... NAME=stage3 bash tools/eval_448_960m.sh"
  say "(dry) 结束：未等待、未执行、未写文件"
  exit 0
fi

mkdir -p "$OUT/logs" "$OUT/eval" "$ADIR" "$BDIR" "$SDIR"
exec 9> "$OUT/.ddp_afterA.lock"
if ! flock -n 9; then
  die "已有一个 $SELF 在跑（flock $OUT/.ddp_afterA.lock 被占）→ 退出" 3
fi
echo $$ > "$OUT/run.pid"              # 权威 PID（覆盖被 kill 的老链路的 stale run.pid）
say "driver PID=$$  （$OUT/run.pid 已更新）"
touch_heartbeat

if [ -f "$OUT/ALL_DONE" ] && [ "$FORCE" != "1" ]; then
  say "$OUT/ALL_DONE 已存在 → 跳过（重跑: FORCE=1）"; exit 0
fi

# ── 0) 等 A 相跑完 ────────────────────────────────────────────────────────
wait_for_A_done() {
  local t0 now pids waited done_line
  t0=$(date +%s)
  say "[wait] 等 A 相结束（每 ${WAIT_POLL}s 检查; 绝不 kill/干扰 A 进程）..."
  while :; do
    pids=$(a_train_pids)
    if [ -z "$pids" ]; then
      # 进程没了: 必须 ckpt_best 在 + 日志有 [done]，否则是崩了 ⇒ 不接力
      if [ ! -f "$ADIR/ckpt_best.pt" ]; then
        printf '%s FAILED: A 训练进程已退出但 %s 不存在\n' "$(date -Is)" "$ADIR/ckpt_best.pt" \
          > "$OUT/FAILED_stage1"
        die "A 相没有 ckpt_best → 不接力（FAILED_stage1）" 4
      fi
      done_line=$(grep -E '^\[done\] steps=' "$LOG1" 2>/dev/null | tail -1)
      if [ -z "$done_line" ]; then
        printf '%s FAILED: A 训练进程已退出但 %s 没有 [done]（疑似崩溃/被 kill）\n' \
          "$(date -Is)" "$LOG1" > "$OUT/FAILED_stage1"
        die "A 相未正常跑完（无 [done] 行）→ 不接力（FAILED_stage1）" 4
      fi
      if ! printf '%s' "$done_line" | grep -q "steps=$STEPS_A\b"; then
        warn "A 的 [done] 行与协议步数 $STEPS_A 不一致: $done_line（继续，但请核对）"
        printf '%s WARNING: A [done] steps 非协议值: %s\n' "$(date -Is)" "$done_line" \
          > "$OUT/WARN_stage1_steps"
      fi
      say "[wait] ✓ A 相结束: $done_line"
      return 0
    fi
    now=$(date +%s)
    if [ "$MAX_WAIT_H" -gt 0 ] && [ $(( (now - t0) / 3600 )) -ge "$MAX_WAIT_H" ]; then
      printf '%s FAILED: 等待 A 相超过 %sh\n' "$(date -Is)" "$MAX_WAIT_H" > "$OUT/FAILED_stage1"
      die "等待 A 相超过 ${MAX_WAIT_H}h → 退出（FAILED_stage1）" 4
    fi
    # 进度日志（每 20 次轮询 ≈ 10 min 打一行，避免刷屏）
    if [ $(( (now - t0) / WAIT_POLL % 20 )) -eq 0 ] && [ $(( now - t0 )) -ge $WAIT_POLL ]; then
      local st; st=$(grep -oE 'step [0-9]+/[0-9]+' "$LOG1" 2>/dev/null | tail -1)
      say "[wait] A 仍在跑（pid $(echo "$pids" | tr '\n' ' '); $st）已等 $(( (now - t0) / 60 ))min"
    fi
    touch_heartbeat
    sleep "$WAIT_POLL"
  done
}

# ── 训练阶段执行 ───────────────────────────────────────────────────────────
stage_train() {   # n outdir log exp_steps <mode:ddp|single> ; 其余 argv 为命令
  local n=$1 od=$2 lg=$3 exp=$4 mode=$5; shift 5
  local rc
  rm -f "$OUT/FAILED_stage$n"
  mkdir -p "$od" "$(dirname "$lg")"
  if [ "$mode" = "single" ]; then
    wait_gpu_free "$GPU_TRAIN" || warn "GPU$GPU_TRAIN 仍有占用（继续，可能 OOM）"
  else
    wait_gpu_free "$GPU_TRAIN" "$STAGE_WAIT_GPU_MIN" || warn "GPU$GPU_TRAIN 仍有占用（继续，可能 OOM）"
    wait_gpu_free "$GPU_EVAL"  "$STAGE_WAIT_GPU_MIN" || warn "GPU$GPU_EVAL 仍有占用（继续，可能 OOM）"
  fi
  { printf '\n===== stage%s start %s (DDP 接力, mode=%s; rc 非 0 / 无 [done] / steps 不足即 FAILED) =====\n' \
      "$n" "$(date -Is)" "$mode"
    show_cmd "[stage$n] CMD" "$@"; } >> "$lg"
  say "[stage$n] start $(date -Is) mode=$mode → $od（日志 $lg）"
  touch_heartbeat
  "$@" >> "$lg" 2>&1
  rc=$?
  say "[stage$n] rc=$rc end=$(date -Is)"
  if [ "$rc" -ne 0 ]; then
    printf '%s FAILED: train rc=%s（日志 %s）\n' "$(date -Is)" "$rc" "$lg" > "$OUT/FAILED_stage$n"
    return 1
  fi
  if [ ! -f "$od/ckpt_best.pt" ]; then
    printf '%s FAILED: ckpt_best.pt 缺失（日志 %s）\n' "$(date -Is)" "$lg" > "$OUT/FAILED_stage$n"
    return 1
  fi
  if ! grep -qE "^\[done\] steps=$exp\b" "$lg"; then
    printf '%s FAILED: 日志无 [done] steps=%s（协议未跑满; 日志 %s）\n' \
      "$(date -Is)" "$exp" "$lg" > "$OUT/FAILED_stage$n"
    return 1
  fi
  if ! grep -q "steps/epoch=$STEPS_EP " "$lg"; then
    printf '%s FAILED: steps/epoch≠%s（等效批量语义被破坏; 日志 %s）\n' \
      "$(date -Is)" "$STEPS_EP" "$lg" > "$OUT/FAILED_stage$n"
    return 1
  fi
  touch "$OUT/STAGE${n}_DONE"
  if [ "$KEEP_LAST" != "1" ]; then rm -f "$od/ckpt_last.pt"; fi
  say "[stage$n] ✓ STAGE${n}_DONE$( [ "$KEEP_LAST" = 1 ] || echo "（已删 ckpt_last）")"
  return 0
}

run_stage_ddp() {  # n outdir log exp_steps args_array_name
  local n=$1 od=$2 lg=$3 exp=$4 arr=$5 port
  local -n A=$arr            # nameref: 不能用 ${!arr[@]}（那给的是索引, 不是元素）
  port=$(pick_port)
  local -a cmd=( env "CUDA_VISIBLE_DEVICES=$GPU_TRAIN,$GPU_EVAL" "PYTHONPATH=$ROOT"
                 "OMP_NUM_THREADS=$OMP_NUM_THREADS"
                 "$PY" -m torch.distributed.run --nproc_per_node="$NPROC"
                 --master_port="$port" --max_restarts=0
                 "$ROOT/train_linattn.py" )
  say "[stage$n] torchrun nproc=$NPROC master_port=$port"
  stage_train "$n" "$od" "$lg" "$exp" ddp "${cmd[@]}" "${A[@]}"
}

run_stage_single() {  # n outdir log exp_steps args_array_name
  local n=$1 od=$2 lg=$3 exp=$4 arr=$5
  local -n A=$arr
  local -a cmd=( env "CUDA_VISIBLE_DEVICES=$GPU_TRAIN" "PYTHONPATH=$ROOT"
                 "OMP_NUM_THREADS=$OMP_NUM_THREADS"
                 "$PY" -u "$ROOT/train_linattn.py" )
  say "[stage$n] 单卡（GPU$GPU_TRAIN, micro16×accum4）"
  stage_train "$n" "$od" "$lg" "$exp" single "${cmd[@]}" "${A[@]}"
}

stage_train_done() {  # n outdir log exp_steps
  local n=$1 od=$2 lg=$3 exp=$4
  [ "$FORCE" != "1" ] || return 1
  [ -f "$od/ckpt_best.pt" ] || return 1
  grep -qE "^\[done\] steps=$exp\b" "$lg" 2>/dev/null || return 1
  return 0
}

wait_gpu_free() {  # gpu [最多等几分钟, 默认 WAIT_GPU_MIN]
  local g=$1 mins=${2:-$WAIT_GPU_MIN} i=0 used
  while [ $i -lt $(( mins * 2 )) ]; do
    used=$(gpu_used "$g")
    [ -z "$used" ] && return 0
    [ "$used" -lt 3000 ] && return 0
    [ $i -eq 0 ] && say "[gpu] GPU$g 已用 ${used}MiB，等显存释放（最多 ${mins}min）..."
    i=$((i + 1)); touch_heartbeat; sleep 30
  done
  return 1
}

# ── 评测 ───────────────────────────────────────────────────────────────────
eval_stage() {  # n outdir
  local n=$1 od=$2 js="$OUT/eval/stage$1_stepwise.json" lg="$OUT/logs/eval_stage$1.log" rc
  if [ "$NO_EVAL" = "1" ]; then say "[eval$n] NO_EVAL=1 → 跳过"; return 0; fi
  if [ -f "$js" ] && [ "$FORCE" != "1" ]; then
    say "[eval$n] $js 已存在 → 跳过"; return 0
  fi
  rm -f "$OUT/FAILED_eval_stage$n"
  mkdir -p "$OUT/eval" "$(dirname "$lg")"
  say "[eval$n] start $(date -Is)（GPU$GPU_EVAL, 串行: DDP 训练占两张卡无法并行）"
  { printf '\n===== eval stage%s start %s =====\n' "$n" "$(date -Is)"
    show_cmd "[eval$n] CMD" env "CKPT=$od/ckpt_best.pt" "OUT=$OUT/eval" "GPU=$GPU_EVAL" \
      "NUM_IMAGES=0" "NAME=stage${n}" "BS=$EVAL_BS" "NUM_WORKERS=$EVAL_WORKERS" \
      "DATA=$DATA" bash tools/eval_448_960m.sh; } >> "$lg"
  # eval_448_960m.sh: JSON = $OUT/${NAME}_stepwise.json ⇒ NAME=stage{N} 才得到
  # $OUT/eval/stage{N}_stepwise.json（run card 约定的产物名）。
  ( CKPT="$od/ckpt_best.pt" OUT="$OUT/eval" GPU="$GPU_EVAL" NUM_IMAGES=0 \
    NAME="stage${n}" BS="$EVAL_BS" NUM_WORKERS="$EVAL_WORKERS" DATA="$DATA" \
    bash tools/eval_448_960m.sh >> "$lg" 2>&1 )
  rc=$?
  touch_heartbeat
  if [ "$rc" -ne 0 ] || [ ! -f "$js" ]; then
    printf '%s FAILED: eval stage%s rc=%s json=%s\n' "$(date -Is)" "$n" "$rc" "$js" \
      > "$OUT/FAILED_eval_stage$n"
    say "[eval$n] ✗ FAILED（rc=$rc, json $([ -f "$js" ] && echo 存在 || echo 缺失)）"
    return 1
  fi
  say "[eval$n] ✓ → $js"
  return 0
}

# ── 1) 真实尺寸 DDP 冒烟 + 决策 ───────────────────────────────────────────
run_ddp_smoke() {   # 输出: 设 MODE=ddp|single; 写 $OUT/ddp_afterA_decision.txt
  local lg=$OUT/logs/ddp_smoke_B.log rc port s_step speedup pred_b pred_s3
  rm -rf "$OUT/ddp_smoke_B"; mkdir -p "$OUT/ddp_smoke_B"
  # GPU1 可能被别的作业临时占用（本机是共享实验机）⇒ 冒烟前多等一会儿（默认 30min），
  # 等不到才降级；等的时候没有 train 进程，watchdog 可能误报一次 CRASHED（看 heartbeat 即可分辨）。
  wait_gpu_free "$GPU_TRAIN" "$SMOKE_WAIT_GPU_MIN" || warn "GPU$GPU_TRAIN 仍有占用（冒烟可能 OOM）"
  wait_gpu_free "$GPU_EVAL"  "$SMOKE_WAIT_GPU_MIN" || warn "GPU$GPU_EVAL 仍有占用（冒烟可能 OOM）"
  port=$(pick_port)
  { printf '\n===== 真实尺寸 DDP 冒烟 start %s（B 相配置 %s 步, 全量数据）=====\n' \
      "$(date -Is)" "$SMOKE_STEPS"
    show_cmd "[smoke] CMD" env "CUDA_VISIBLE_DEVICES=$GPU_TRAIN,$GPU_EVAL" "PYTHONPATH=$ROOT" \
      "OMP_NUM_THREADS=$OMP_NUM_THREADS" "$PY" -m torch.distributed.run \
      --nproc_per_node="$NPROC" --master_port="$port" --max_restarts=0 \
      "$ROOT/train_linattn.py" "${ARGS_SMOKE[@]}"; } > "$lg"
  say "[smoke] 真实尺寸 2 卡 DDP 冒烟（$SMOKE_STEPS 步，全量数据；日志 $lg）..."
  env "CUDA_VISIBLE_DEVICES=$GPU_TRAIN,$GPU_EVAL" "PYTHONPATH=$ROOT" \
      "OMP_NUM_THREADS=$OMP_NUM_THREADS" \
      timeout "$SMOKE_TIMEOUT" "$PY" -m torch.distributed.run \
      --nproc_per_node="$NPROC" --master_port="$port" --max_restarts=0 \
      "$ROOT/train_linattn.py" "${ARGS_SMOKE[@]}" >> "$lg" 2>&1
  rc=$?
  touch_heartbeat

  # 冒烟必须是"真 DDP 且链路正确": rc=0 + 两个 rank 都起来 + steps/epoch=1901 + 有 elapsed
  local ok=1 why=""
  [ "$rc" -eq 0 ] || { ok=0; why="rc=$rc$( [ "$rc" = 124 ] && echo '（超时=hang）')"; }
  grep -q '\[ddp\] rank=0/' "$lg" || { ok=0; why="${why:+$why; }无 rank0 [ddp] banner"; }
  grep -q '\[ddp\] rank=1/' "$lg" || { ok=0; why="${why:+$why; }无 rank1 [ddp] banner"; }
  grep -q "steps/epoch=$STEPS_EP == 单卡参考 $STEPS_EP" "$lg" \
    || { ok=0; why="${why:+$why; }steps/epoch 断言未通过（应=$STEPS_EP）"; }
  s_step=$("$PY" - "$OUT/ddp_smoke_B/train_log.jsonl" "$SMOKE_STEPS" <<'PY' 2>/dev/null
import json, sys
p, n = sys.argv[1], int(sys.argv[2])
try:
    recs = [json.loads(l) for l in open(p) if '"elapsed_min"' in l and '"step"' in l]
except Exception:
    print(""); raise SystemExit
d = {r["step"]: r["elapsed_min"] for r in recs if "step" in r}
lo, hi = 10, n            # 丢掉前 10 步（warmup/首轮 dataloader 填充）算 steady
if lo in d and hi in d and hi > lo:
    print(f'{(d[hi] - d[lo]) * 60.0 / (hi - lo):.4f}')
else:
    print("")
PY
)
  [ -n "$s_step" ] || { ok=0; why="${why:+$why; }拿不到 steady s/step（日志缺 step 10/30 的 elapsed_min）"; }

  speedup=""; pred_b=""; pred_s3=""
  if [ -n "$s_step" ]; then
    speedup=$("$PY" -c "print(f'{$SINGLE_REF_S_STEP/float(\"$s_step\"):.2f}')")
    pred_b=$("$PY" -c "print(f'{$STEPS_B*float(\"$s_step\")/3600:.2f}')")
    pred_s3=$("$PY" -c "print(f'{$STEPS_S3*float(\"$s_step\")/3600:.2f}')")
    say "[smoke] steady ${s_step} s/step  ⇒ speedup ${speedup}×（对照单卡 ${SINGLE_REF_S_STEP}）"
    say "[smoke] 外推: B ${STEPS_B} 步 ≈ ${pred_b}h（单卡 14.0h）; S3 ${STEPS_S3} 步 ≈ ${pred_s3}h（单卡 6.9h）"
  else
    say "[smoke] ✗ 未能测出 s/step"
  fi

  MODE="ddp"
  if [ "$ok" != "1" ]; then
    say "[smoke] ✗ DDP 冒烟失败: $why"
    printf '%s DDP 冒烟失败: %s（target s/step=%s speedup=%s）\n' \
      "$(date -Is)" "$why" "${s_step:-?}" "${speedup:-?}" > "$OUT/ALERT_ddp_smoke.txt"
    if [ "$FORCE_DDP" = "1" ]; then
      printf '%s FAILED: DDP 冒烟失败且 FORCE_DDP=1: %s\n' "$(date -Is)" "$why" > "$OUT/FAILED_ddp_smoke"
      rm -rf "$OUT/ddp_smoke_B"
      die "DDP 冒烟失败且 FORCE_DDP=1 → 不降级（FAILED_ddp_smoke）" 5
    fi
    MODE="single"
  elif [ -n "$speedup" ] && \
       ! "$PY" -c "import sys; sys.exit(0 if float('$speedup') >= float('$MIN_SPEEDUP') else 1)"; then
    say "[smoke] ⚠️ speedup ${speedup}× < 安全阀 ${MIN_SPEEDUP}× ⇒ DDP 收益不足"
    printf '%s DDP speedup=%s× < %s×（%s s/step）\n' \
      "$(date -Is)" "$speedup" "$MIN_SPEEDUP" "$s_step" > "$OUT/ALERT_ddp_slow.txt"
    if [ "$FORCE_DDP" = "1" ]; then
      say "[smoke] FORCE_DDP=1 ⇒ 仍然用 DDP"
      MODE="ddp"
    else
      MODE="single"
    fi
  fi
  if [ "$MODE" = "single" ]; then
    touch "$OUT/DDP_FALLBACK"
    say "[smoke] ⇒ 回退单卡跑 B/S3（安全阀）; 标记 $OUT/DDP_FALLBACK, 告警 $OUT/ALERT_ddp_*.txt"
  else
    rm -f "$OUT/DDP_FALLBACK"
    say "[smoke] ⇒ 采用 2×A800 DDP 跑 B/S3"
  fi
  { printf 'ts=%s\nmode=%s\ns_per_step=%s\nspeedup_vs_single=%s\nmin_speedup=%s\nsmoke_steps=%s\n' \
      "$(date -Is)" "$MODE" "${s_step:-?}" "${speedup:-?}" "$MIN_SPEEDUP" "$SMOKE_STEPS"
    printf 'pred_B_h=%s\npred_S3_h=%s\nsmoke_ok=%s\nsmoke_why=%s\n' \
      "${pred_b:-?}" "${pred_s3:-?}" "$ok" "${why:-none}"
  } > "$OUT/ddp_afterA_decision.txt"
  say "[smoke] 决策已写 $OUT/ddp_afterA_decision.txt"
  rm -rf "$OUT/ddp_smoke_B"     # 冒烟产物 ~7.7G ckpt, 立即清掉
}

# ── 主流程 ─────────────────────────────────────────────────────────────────
wait_for_A_done
# 老链路（run_448_960m_3stage.sh）被 kill 后没人写过 STAGE1_DONE；
# 补上它 watchdog 才会把 ACTIVE 切到 stage2（否则它会盯着早已静止的 stage1.log）。
touch "$OUT/STAGE1_DONE"
say "[stage1] ✓ A 相训练完成 ⇒ STAGE1_DONE（评测在下面做）"
rm -f "$OUT/FAILED_stage1" "$OUT/WARN_stage1_steps"
if [ -f "$ADIR/ckpt_last.pt" ]; then
  rm -f "$ADIR/ckpt_last.pt"; say "[stage1] 删 $ADIR/ckpt_last.pt（省 3.8G）"
fi

# 决定 B/S3 用 DDP 还是单卡（只有还需要训练时才跑冒烟）
NEED_TRAIN=0
stage_train_done 2 "$BDIR" "$LOG2" "$STEPS_B" || NEED_TRAIN=1
stage_train_done 3 "$SDIR" "$LOG3" "$STEPS_S3" || NEED_TRAIN=1
if [ "$NEED_TRAIN" = "1" ]; then
  run_ddp_smoke
else
  MODE="ddp"; say "[smoke] B/S3 都已完成 → 跳过真实尺寸 DDP 冒烟"
fi

# 2) A 相评测（GPU1；此时 GPU0/1 都空, 评测完再起 DDP 训练）
eval_stage 1 "$ADIR" || die "A 相评测失败（FAILED_eval_stage1）→ 中止" 6

# 3) B 相训练
if stage_train_done 2 "$BDIR" "$LOG2" "$STEPS_B"; then
  say "[stage2] ckpt_best + [done] steps=$STEPS_B 已存在 → 跳过训练"
  touch "$OUT/STAGE2_DONE"
else
  [ "$FORCE" = "1" ] && rm -f "$BDIR/ckpt_best.pt"
  if [ "$MODE" = "ddp" ]; then
    run_stage_ddp 2 "$BDIR" "$LOG2" "$STEPS_B" ARGS_B_DDP || die "B 相 DDP 训练失败（FAILED_stage2）→ 中止" 7
  else
    run_stage_single 2 "$BDIR" "$LOG2" "$STEPS_B" ARGS_B_1 || die "B 相单卡训练失败（FAILED_stage2）→ 中止" 7
  fi
fi

# 4) B 相评测
eval_stage 2 "$BDIR" || die "B 相评测失败（FAILED_eval_stage2）→ 中止" 8

# 5) S3 相训练
if stage_train_done 3 "$SDIR" "$LOG3" "$STEPS_S3"; then
  say "[stage3] ckpt_best + [done] steps=$STEPS_S3 已存在 → 跳过训练"
  touch "$OUT/STAGE3_DONE"
else
  [ "$FORCE" = "1" ] && rm -f "$SDIR/ckpt_best.pt"
  if [ "$MODE" = "ddp" ]; then
    run_stage_ddp 3 "$SDIR" "$LOG3" "$STEPS_S3" ARGS_S3_DDP || die "S3 相 DDP 训练失败（FAILED_stage3）→ 中止" 9
  else
    run_stage_single 3 "$SDIR" "$LOG3" "$STEPS_S3" ARGS_S3_1 || die "S3 相单卡训练失败（FAILED_stage3）→ 中止" 9
  fi
fi

# 6) S3 相评测
eval_stage 3 "$SDIR" || die "S3 相评测失败（FAILED_eval_stage3）→ 中止" 10

# 7) 收尾
touch "$OUT/ALL_DONE"
say "================ ALL_DONE $(date -Is) mode=$MODE ================"
say "产物: $OUT/{A,B,S3}/ckpt_best.pt, $OUT/eval/stage{1,2,3}_stepwise.json"
say "决策/耗时: $OUT/ddp_afterA_decision.txt；stage2/3 日志: $OUT/logs/stage{2,3}.log"
exit 0
