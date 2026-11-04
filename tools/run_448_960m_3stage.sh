#!/usr/bin/env bash
# =============================================================================
# run_448_960m_3stage.sh — AssocVTC 448²/patch32/960M：论文三阶段串行训练 + 分段评测
# -----------------------------------------------------------------------------
# 2026-11-04 · 协议见 doc/2026-11-04/RUN_448_960M_3stage.md 与
#   /home/linaro/dsh/exp448_bench/DESIGN_448_960M_3stage.md
#
# 架构（相对论文基线只改 3 项：分辨率 224→448、patch 16→32（N 仍 196）、宽度 ×2）:
#   enc 1744×13 h16 = 480.54M  +  dec 2048×7 h16 = 479.73M  ⇒ 960.27M
#   像素头 hidden 1024 → 3·32² = 3072（由 --patch 自动决定）；N = (448/32)² = 196
#
# 三阶段（严格照论文 §3.2 / 表 1）:
#   1) A 相 : --dec_recurrent 0（T=1 单步读满 K=32），45 ep，等效 bs128，lr 2e-4，从零
#   2) B 相 : --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps 1..16，
#            6 ep，等效 bs64，lr 2e-4，从 $OUT/A/ckpt_best.pt 热启动
#   3) S3 相: --zs_dim 224（每图 32×224 = 7168 值，与论文同值数口径），
#            3 ep，等效 bs64，lr 2e-4，从 $OUT/B/ckpt_best.pt 热启动
#   公共    : AdamW lr2e-4 / cosine / 8% warmup / wd0.05 / clip1.0 / bf16(--amp 1)
#             --seed 0 --save_opt 0 --num_workers 16
#
# 卡分配: 训练固定 GPU0；**每阶段训练成功后立刻**在 GPU1 后台起逐步评测，
#         与下一阶段训练并行（评测不抢训练卡）；全部阶段结束后 wait 汇总。
#
# 产物:
#   $OUT/{A,B,S3}/{ckpt_best.pt,args.json,status.json,train_log.jsonl,eval_val.json}
#   $OUT/logs/stage{1,2,3}.log        训练日志（开头即该阶段完整命令）
#   $OUT/logs/eval_stage{1,2,3}.log   评测日志
#   $OUT/eval/stage{1,2,3}_stepwise.json  逐步评测（16 档；A 相只有 1 行）
#   $OUT/STAGE{1,2,3}_DONE / ALL_DONE / FAILED_stage{N} / FAILED_eval_stage{N}
#
# 用法（远端仓库根目录；通常由 tools/launch_448_960m.sh 后台启动）:
#   DRY=1  bash tools/run_448_960m_3stage.sh   # 干跑: 只打印命令, 不执行/不建目录/不写文件
#   PLAN=1 bash tools/run_448_960m_3stage.sh   # 只输出 3 行 "STAGE{N} <shell 命令>"（给 launch 写 run_info.json）
#   bash tools/run_448_960m_3stage.sh          # 真正开跑（前台；launch 脚本用 setsid nohup 起它）
#
# 环境变量:
#   OUT / DATA / PY / GPU_TRAIN(0) / GPU_EVAL(1) / NUM_WORKERS(16) / EVAL_BS(16)
#   SKIP_STAGE1|SKIP_STAGE2|SKIP_STAGE3=1  跳过某阶段（续跑用；需上一阶段 ckpt 已在）
#   NO_EVAL=1        不跑评测（只训）
#   KEEP_LAST=1      保留 ckpt_last.pt（默认删：960M fp32 ckpt ≈3.8G/个，盘只有 35G）
#   FORCE=1          已有 ALL_DONE 也重跑
#   TIME_BUDGET_H=0  0=不设预算。>0 时该阶段到点保存退出 ⇒ 本脚本判为**协议未跑满**，
#                    写 BUDGET_HIT_stageN + FAILED_stageN 并中止（不会静默带着半训 ckpt 进下一阶段）
#   EP1/EP2/EP3 / LR / MIN_LR / WD / WARMUP / CLIP / SEED / AMP  协议默认值，仅冒烟时改
#   IMG_SIZE/PATCH/K/T/BLOCK/ENC_DIM/ENC_DEPTH/ENC_HEADS/DEC_DIM/DEC_DEPTH/DEC_HEADS/
#   PIXEL_HIDDEN/ENC_ATTN  架构（默认即本实验口径，不改协议就别动）
#
# ---- TODO(bench): 以下 5 个值等 bencher 实测显存/耗时后由 Lead 填最终值 ------------
#   本仓库口径（train_linattn.py）: --batch_size 是**等效批量**，
#   micro-batch = batch_size / accum_steps ⇒ 这里用 MICRO_BS_* × ACCUM_* 表达
#   「micro-batch × 累积」，脚本内部折算成 --batch_size(micro×accum) --accum_steps(accum)。
#   默认 = 保守可跑 + 守死论文等效批量（A: 16×8=128，B/S3: 16×4=64）。
#   ⚠️ 若 bench 后不得不牺牲等效批量，必须显式 FORCE_EFF_BS=1（脚本会告警并打印实际步数变化）。
#   ⚠️ --dec_grad_ckpt 必须**全阶段同一值**：bf16 重算在长跑后会造成实例级终值漂移
#      （README: 11,406 步后 0.341 dB），不可配对比较。默认 1 = 保守（省显存）。
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "[$(basename "$0")] ✗ 请在仓库内执行（tools/ 的上一级）" >&2; exit 2; }
ROOT=$PWD

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/exp448}
GPU_TRAIN=${GPU_TRAIN:-0}
GPU_EVAL=${GPU_EVAL:-1}
NUM_WORKERS=${NUM_WORKERS:-16}
EVAL_BS=${EVAL_BS:-16}
EVAL_WORKERS=${EVAL_WORKERS:-8}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

DRY=${DRY:-0}
PLAN=${PLAN:-0}
NO_EVAL=${NO_EVAL:-0}
KEEP_LAST=${KEEP_LAST:-0}
FORCE=${FORCE:-0}
SKIP_STAGE1=${SKIP_STAGE1:-0}
SKIP_STAGE2=${SKIP_STAGE2:-0}
SKIP_STAGE3=${SKIP_STAGE3:-0}

# ---- TODO(bench): micro-batch / 累积 / 梯度检查点（见文件头说明）-----------------
MICRO_BS_A=${MICRO_BS_A:-16}
ACCUM_A=${ACCUM_A:-8}
MICRO_BS_B=${MICRO_BS_B:-16}
ACCUM_B=${ACCUM_B:-4}
DEC_GRAD_CKPT=${DEC_GRAD_CKPT:-1}
FORCE_EFF_BS=${FORCE_EFF_BS:-0}

# ---- 协议常量（论文 §3.2；冒烟才改）--------------------------------------------
EP1=${EP1:-45}
EP2=${EP2:-6}
EP3=${EP3:-3}
LR=${LR:-2e-4}
MIN_LR=${MIN_LR:-1e-6}
WD=${WD:-0.05}
WARMUP=${WARMUP:-0.08}
CLIP=${CLIP:-1.0}
SEED=${SEED:-0}
AMP=${AMP:-1}
ZS_DIM=${ZS_DIM:-224}
TIME_BUDGET_H=${TIME_BUDGET_H:-0}

# ---- 架构（本实验唯一自变量；默认值即口径）-------------------------------------
IMG_SIZE=${IMG_SIZE:-448}
PATCH=${PATCH:-32}
K=${K:-32}
T=${T:-16}
BLOCK=${BLOCK:-2}
PIXEL_HIDDEN=${PIXEL_HIDDEN:-1024}
ENC_ATTN=${ENC_ATTN:-std}
ENC_DIM=${ENC_DIM:-1744}
ENC_DEPTH=${ENC_DEPTH:-13}
ENC_HEADS=${ENC_HEADS:-16}
DEC_DIM=${DEC_DIM:-2048}
DEC_DEPTH=${DEC_DEPTH:-7}
DEC_HEADS=${DEC_HEADS:-16}

ADIR=$OUT/A
BDIR=$OUT/B
SDIR=$OUT/S3
LOG1=$OUT/logs/stage1.log
LOG2=$OUT/logs/stage2.log
LOG3=$OUT/logs/stage3.log
DEC_STEPS=$(seq -s, 1 "$T")

if [ "$PLAN" = "1" ]; then DRY=1; fi

# ---- 输出小工具 ---------------------------------------
say() { if [ "$PLAN" != "1" ]; then echo "$@"; fi; }
show_cmd() { local pfx=$1 a; shift; printf '%s' "$pfx"; for a in "$@"; do printf ' %q' "$a"; done; printf '\n'; }
require_python() { if py_ok; then say "[run448] ✓ python: $PY"
  elif [ "$DRY" = "1" ]; then say "[run448] (dry) 未找到 python: $PY"
  else echo "[run448] ✗ 缺少 python: $PY" >&2; exit 2; fi; }
py_ok() { case "$PY" in */*) [ -x "$PY" ] ;; *) command -v "$PY" >/dev/null 2>&1 ;; esac; }
require_envfile() { if [ ! -e "$1" ]; then
    if [ "$DRY" = "1" ]; then say "[run448] (dry) 未找到 $2: $1"; else echo "[run448] ✗ 缺少 $2: $1" >&2; exit 2; fi
  else say "[run448] ✓ $2: $1"; fi; }

# ---- 0) 配置自检（真实等效批量守卫）---------------------------------------------
EFF_A=$((MICRO_BS_A * ACCUM_A))
EFF_B=$((MICRO_BS_B * ACCUM_B))
if [ "$MICRO_BS_A" -lt 1 ] || [ "$ACCUM_A" -lt 1 ] || [ "$MICRO_BS_B" -lt 1 ] || [ "$ACCUM_B" -lt 1 ]; then
  echo "[run448] ✗ micro-batch / accum 必须 ≥1" >&2; exit 2
fi
if [ "$EFF_A" -ne 128 ] || [ "$EFF_B" -ne 64 ]; then
  if [ "$FORCE_EFF_BS" != "1" ]; then
    echo "[run448] ✗ 等效批量偏离论文: A=$EFF_A (论文 128), B=$EFF_B (论文 64)" >&2
    echo "          A/B 相步数会跟着变（951/1901 步每 epoch → 128/64 之外即不可比）。" >&2
    echo "          调整 MICRO_BS_*/ACCUM_* 保持等效批量不变；确认要偏离请 FORCE_EFF_BS=1。" >&2
    exit 2
  fi
  say "[run448] ⚠️ FORCE_EFF_BS=1：等效批量 A=$EFF_A（论文128）B=$EFF_B（论文64）— 步数已偏离，仅供能跑通"
fi
if [ "$K" -ne $((T * BLOCK)) ]; then
  say "[run448] ⚠️ K=$K ≠ T×block=$((T * BLOCK))：每步读 $BLOCK 个 z_s，T 步共 $((T * BLOCK)) 个，K=$K 会读不满/读超（协议 K=32,T=16,block=2）"
fi

say "[run448] ================ AssocVTC 448²/p32/960M 三阶段 ================"
say "[run448] $(date -Is)  ROOT=$ROOT"
say "[run448] OUT=$OUT  DATA=$DATA"
say "[run448] GPU: train=$GPU_TRAIN  eval=$GPU_EVAL  (eval 与下一阶段训练并行)"
say "[run448] arch: enc ${ENC_DIM}×${ENC_DEPTH} h${ENC_HEADS} / dec ${DEC_DIM}×${DEC_DEPTH} h${DEC_HEADS} @ ${IMG_SIZE}²/p${PATCH}, K=$K T=$T block=$BLOCK"
say "[run448] micro/accum: A ${MICRO_BS_A}×${ACCUM_A}=${EFF_A}（951 步/ep） | B/S3 ${MICRO_BS_B}×${ACCUM_B}=${EFF_B}（1901 步/ep） | dec_grad_ckpt=$DEC_GRAD_CKPT"
say "[run448] epochs: ${EP1}/${EP2}/${EP3}  lr=$LR  time_budget_h=$TIME_BUDGET_H  zs_dim(S3)=$ZS_DIM"

if [ "$PLAN" != "1" ] && [ "$DRY" != "1" ] && [ -f "$OUT/ALL_DONE" ] && [ "$FORCE" != "1" ]; then
  echo "[run448] $OUT/ALL_DONE 已存在 → 跳过（重跑请 FORCE=1，或先删 ALL_DONE / 用 SKIP_STAGE*）"
  exit 0
fi

# ---- 1) 预检 -------------------------------------------------------------------
require_python
require_envfile "$ROOT/train_linattn.py" "train_linattn.py"
require_envfile "$ROOT/eval_stepwise_linattn.py" "eval_stepwise_linattn.py"
require_envfile "$ROOT/tools/eval_448_960m.sh" "tools/eval_448_960m.sh"
require_envfile "$DATA" "数据集目录"
if [ "$DRY" != "1" ]; then
  mkdir -p "$OUT/logs" "$OUT/eval" "$ADIR" "$BDIR" "$SDIR"
  echo $$ > "$OUT/run.pid"          # 权威 PID（launch_448_960m.sh 也读它）
  AVAIL_G=$(df -Pk "$OUT" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')
  say "[run448] 磁盘可用: ${AVAIL_G:-?}G（每阶段 ckpt_best≈3.8G，A+B+S3 同时留存 ≈11.4G；ckpt_last 已默认删）"
  if [ -n "${AVAIL_G:-}" ] && [ "$AVAIL_G" -lt 20 ]; then
    say "[run448] ⚠️ 可用空间 <20G：建议先 fetch 回本地再清理旧产物，否则阶段末尾落 ckpt 可能失败"
  fi
fi

# ---- 2) 命令构建（顺序 = 论文三阶段）--------------------------------------------
ENV_PREFIX=( env "CUDA_VISIBLE_DEVICES=$GPU_TRAIN" "PYTHONPATH=$ROOT" "OMP_NUM_THREADS=$OMP_NUM_THREADS" )
ARCH=( --img_size "$IMG_SIZE" --patch "$PATCH" --num_specials "$K" --pixel_hidden "$PIXEL_HIDDEN"
       --enc_attn "$ENC_ATTN" --enc_dim "$ENC_DIM" --enc_depth "$ENC_DEPTH" --enc_heads "$ENC_HEADS"
       --dec_dim "$DEC_DIM" --dec_depth "$DEC_DEPTH" --dec_heads "$DEC_HEADS" )
OPT=( --lr "$LR" --min_lr "$MIN_LR" --weight_decay "$WD" --warmup_frac "$WARMUP" --grad_clip "$CLIP"
      --amp "$AMP" --seed "$SEED" --save_opt 0 --num_workers "$NUM_WORKERS"
      --dec_grad_ckpt "$DEC_GRAD_CKPT" --time_budget_h "$TIME_BUDGET_H" )

CMD1=( "${ENV_PREFIX[@]}" "$PY" -u train_linattn.py
       --data_dir "$DATA" --output_dir "$ADIR" "${ARCH[@]}"
       --dec_recurrent 0 --epochs "$EP1" --batch_size "$EFF_A" --accum_steps "$ACCUM_A" "${OPT[@]}" )
CMD2=( "${ENV_PREFIX[@]}" "$PY" -u train_linattn.py
       --data_dir "$DATA" --output_dir "$BDIR" "${ARCH[@]}"
       --dec_recurrent 1 --step_plan fixed --block "$BLOCK" --dec_steps "$DEC_STEPS"
       --init_from "$ADIR/ckpt_best.pt" --epochs "$EP2" --batch_size "$EFF_B" --accum_steps "$ACCUM_B" "${OPT[@]}" )
CMD3=( "${ENV_PREFIX[@]}" "$PY" -u train_linattn.py
       --data_dir "$DATA" --output_dir "$SDIR" "${ARCH[@]}"
       --dec_recurrent 1 --step_plan fixed --block "$BLOCK" --dec_steps "$DEC_STEPS"
       --zs_dim "$ZS_DIM" --init_from "$BDIR/ckpt_best.pt"
       --epochs "$EP3" --batch_size "$EFF_B" --accum_steps "$ACCUM_B" "${OPT[@]}" )

# PLAN：只把三条训练命令吐给 launch_448_960m.sh 记 run_info.json（无任何副作用）
if [ "$PLAN" = "1" ]; then
  show_cmd "STAGE1" "${CMD1[@]}"
  show_cmd "STAGE2" "${CMD2[@]}"
  show_cmd "STAGE3" "${CMD3[@]}"
  exit 0
fi

# ---- 3) 阶段执行小工具 ----------------------------------------------------------
fail_stage() { # n reason
  local n=$1; shift
  printf '%s FAILED: %s\n' "$(date -Is)" "$*" > "$OUT/FAILED_stage$n"
  echo "[run448] ✗ stage$n FAILED — $*" >&2
}

stage_train() { # n log outdir argv...
  local n=$1 lg=$2 od=$3 rc=0; shift 3
  if [ "$DRY" = "1" ]; then
    show_cmd "[dry][stage${n}] mkdir -p" "$od" "$(dirname "$lg")"
    show_cmd "[dry][stage${n}] CMD" "$@"
    say "[dry][stage${n}] 日志 → $lg ；成功后写 $OUT/STAGE${n}_DONE"
    return 0
  fi
  rm -f "$OUT/FAILED_stage$n"
  mkdir -p "$od" "$(dirname "$lg")"
  { printf '\n===== stage%s start %s (rc 非 0 或 ckpt_best 缺失即中止) =====\n' "$n" "$(date -Is)"
    show_cmd "[stage$n] CMD" "$@"; } >> "$lg"
  say "[run448] ---- stage$n start $(date -Is) → $od"
  say "[run448]      log: $lg"
  "$@" >> "$lg" 2>&1
  rc=$?
  say "[run448] ---- stage$n rc=$rc end=$(date -Is)"
  return $rc
}

check_stage() { # n rc log outdir
  local n=$1 rc=$2 lg=$3 od=$4
  if [ "$DRY" = "1" ]; then say "[dry][stage${n}] 跳过 rc / ckpt_best / time_budget 检查"; return 0; fi
  if [ "$rc" -ne 0 ]; then fail_stage "$n" "train_linattn.py rc=$rc（日志 $lg）"; return 1; fi
  if [ ! -f "$od/ckpt_best.pt" ]; then fail_stage "$n" "ckpt_best.pt 缺失: $od/ckpt_best.pt（日志 $lg）"; return 1; fi
  if awk -v x="$TIME_BUDGET_H" 'BEGIN{exit !(x>0)}' && grep -q '\[stop\] 达到时间预算' "$lg" 2>/dev/null; then
    printf '%s 时间预算 %sh 触发，本阶段协议未跑满\n' "$(date -Is)" "$TIME_BUDGET_H" > "$OUT/BUDGET_HIT_stage$n"
    fail_stage "$n" "time_budget=${TIME_BUDGET_H}h 触发（协议未跑满，见 $lg 的 [stop] 行）"
    return 1
  fi
  touch "$OUT/STAGE${n}_DONE"
  say "[run448] ---- stage$n ✓ STAGE${n}_DONE"
  if [ "$KEEP_LAST" != "1" ]; then rm -f "$od/ckpt_last.pt"; fi
  return 0
}

EVAL_PIDS=()
EVAL_NS=()
# 训练链中途失败时, 已在 GPU1 上跑的评测不杀（数据仍然有用），但要在退出时提示
on_exit() {
  local i
  [ "${#EVAL_PIDS[@]}" -eq 0 ] && return 0
  for i in "${!EVAL_PIDS[@]}"; do
    if kill -0 "${EVAL_PIDS[$i]}" 2>/dev/null; then
      echo "[run448] 注：stage${EVAL_NS[$i]} 评测仍在 GPU$GPU_EVAL 上跑（pid ${EVAL_PIDS[$i]}），完成后落 $OUT/eval/stage${EVAL_NS[$i]}_stepwise.json"
    fi
  done
}
trap on_exit EXIT

launch_eval() { # n outdir
  local n=$1 od=$2
  local json="$OUT/eval/stage${n}_stepwise.json" lg="$OUT/logs/eval_stage${n}.log"
  if [ "$NO_EVAL" = "1" ]; then say "[run448] NO_EVAL=1 → 跳过 stage$n 评测"; return 0; fi
  if [ "$DRY" = "1" ]; then
    show_cmd "[dry][eval stage${n}]" CKPT="$od/ckpt_best.pt" OUT="$OUT/eval" GPU="$GPU_EVAL" \
      NUM_IMAGES=0 NAME="stage${n}_stepwise" BS="$EVAL_BS" NUM_WORKERS="$EVAL_WORKERS" DATA="$DATA" \
      bash tools/eval_448_960m.sh
    say "[dry][eval stage${n}] 背景运行，日志 → $lg ；结果 → $json"
    return 0
  fi
  mkdir -p "$OUT/eval" "$OUT/logs"
  ( CKPT="$od/ckpt_best.pt" OUT="$OUT/eval" GPU="$GPU_EVAL" NUM_IMAGES=0 \
    NAME="stage${n}_stepwise" BS="$EVAL_BS" NUM_WORKERS="$EVAL_WORKERS" DATA="$DATA" \
    bash tools/eval_448_960m.sh > "$lg" 2>&1 ) &
  EVAL_PIDS+=("$!")
  EVAL_NS+=("$n")
  say "[run448] ---- eval stage$n 后台启动 pid=$! (GPU$GPU_EVAL) → $json（日志 $lg）"
  return 0
}

wait_evals() {
  local i n pid rc bad=0 json
  if [ "$DRY" = "1" ] || [ "${#EVAL_PIDS[@]}" -eq 0 ]; then return 0; fi
  say "[run448] 等待 ${#EVAL_PIDS[@]} 个评测进程结束 ..."
  for i in "${!EVAL_PIDS[@]}"; do
    n=${EVAL_NS[$i]}; pid=${EVAL_PIDS[$i]}
    wait "$pid" 2>/dev/null; rc=$?
    json="$OUT/eval/stage${n}_stepwise.json"
    if [ "$rc" -ne 0 ] || [ ! -f "$json" ]; then
      printf '%s FAILED: eval stage%s rc=%s json=%s\n' "$(date -Is)" "$n" "$rc" "$json" > "$OUT/FAILED_eval_stage$n"
      echo "[run448] ✗ eval stage$n FAILED（rc=$rc, json $json $( [ -f "$json" ] && echo 存在 || echo 缺失)）" >&2
      bad=1
    else
      say "[run448] ---- eval stage$n ✓ → $json"
    fi
  done
  return $bad
}

# ---- 4) 三阶段串行 --------------------------------------------------------------
# 阶段 1：从零，T=1 单步读满（45 ep, 等效 bs128）
if [ "$SKIP_STAGE1" = "1" ]; then
  say "[run448] SKIP_STAGE1=1 → 跳过 A 相"
else
  stage_train 1 "$LOG1" "$ADIR" "${CMD1[@]}"; RC=$?
  if ! check_stage 1 "$RC" "$LOG1" "$ADIR"; then say "[run448] 中止：A 相失败"; exit 1; fi
  launch_eval 1 "$ADIR"
fi

# 阶段 2：16 档循环读出（fixed block 2, 每步 2 token），6 ep, 等效 bs64
if [ "$SKIP_STAGE2" = "1" ]; then
  say "[run448] SKIP_STAGE2=1 → 跳过 B 相"
else
  if [ "$DRY" != "1" ] && [ ! -f "$ADIR/ckpt_best.pt" ]; then
    fail_stage 2 "缺少 A 相 ckpt（--init_from 需要的 $ADIR/ckpt_best.pt）"; exit 2
  fi
  stage_train 2 "$LOG2" "$BDIR" "${CMD2[@]}"; RC=$?
  if ! check_stage 2 "$RC" "$LOG2" "$BDIR"; then say "[run448] 中止：B 相失败"; exit 2; fi
  launch_eval 2 "$BDIR"
fi

# 阶段 3：输出投影降到 224 维（zs_dim=224 ⇒ 每图 7168 值），3 ep, 等效 bs64
if [ "$SKIP_STAGE3" = "1" ]; then
  say "[run448] SKIP_STAGE3=1 → 跳过 S3 相"
else
  if [ "$DRY" != "1" ] && [ ! -f "$BDIR/ckpt_best.pt" ]; then
    fail_stage 3 "缺少 B 相 ckpt（--init_from 需要的 $BDIR/ckpt_best.pt）"; exit 3
  fi
  stage_train 3 "$LOG3" "$SDIR" "${CMD3[@]}"; RC=$?
  if ! check_stage 3 "$RC" "$LOG3" "$SDIR"; then say "[run448] 中止：S3 相失败"; exit 3; fi
  launch_eval 3 "$SDIR"
fi

# ---- 5) 收尾：等评测 + ALL_DONE ------------------------------------------------
if ! wait_evals; then
  say "[run448] ✗ 有评测失败：训练产物已在，评测可单独重跑（tools/eval_448_960m.sh）"
  exit 5
fi
if [ "$DRY" = "1" ]; then
  say "[dry] touch $OUT/ALL_DONE"
  say "[dry] DRY 干跑结束（未执行任何命令、未写任何文件）"
  exit 0
fi
touch "$OUT/ALL_DONE"
say "[run448] ================ ALL_DONE $(date -Is) ================"
say "[run448] 产物: $OUT/{A,B,S3}/ckpt_best.pt, $OUT/eval/stage{1,2,3}_stepwise.json"
say "[run448] 下次: PORT=23556 bash tools/fetch_448_960m.sh   # 拉回本地 artifacts"
