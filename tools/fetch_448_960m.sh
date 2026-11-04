#!/usr/bin/env bash
# =============================================================================
# fetch_448_960m.sh — 把 448²/p32/960M 三阶段产物拉回本地 artifacts
# -----------------------------------------------------------------------------
# 2026-11-04 · 远端 `/root/autodl-tmp/exp448/` → 本地
#   `/home/linaro/dsh/exp448_bench/artifacts/`
#
# 拉什么（存在才拉, 缺的不报错, 最后汇总）:
#   eval/stage{1,2,3}_stepwise.json     逐步评测结果（A 相 1 行 / B,S3 相 16 档）
#   logs/stage{1,2,3}.log               训练日志（\r 进度条已剥掉）
#   logs/run.log, logs/launch.out       启动器与链路汇总日志（同上处理）
#   logs/eval_stage{1,2,3}.log          各阶段评测日志
#   args/stage{N}_args.json             train_linattn.py 的完整 argparse
#   args/stage{N}_status.json           末次 status.json（step/epoch/val_psnr/eta）
#   args/stage{N}_cfg.txt               从训练日志摘出的 [cfg]/[params]/[alit] 行
#                                       （cfg 本体在 ckpt["cfg"] 里, 未单独落盘）
#   args/stage{N}_eval_val.json         训练内验证指标
#   train_logs/stage{N}.jsonl           train_log.jsonl（逐步曲线）
#   run_info.json                       commit / 参数量 / 三阶段命令 / GPU 分配
#   markers/{STAGE*_DONE,ALL_DONE,FAILED_*,BUDGET_HIT_*}   状态标记
#
# 用法（本地, 任意目录）:
#   PORT=23556 bash tools/fetch_448_960m.sh
#   DRY=1 PORT=23556 bash tools/fetch_448_960m.sh          # 只打印将执行的 ssh/scp
#   PORT=23556 SRC=/root/autodl-tmp/exp448 DST=/tmp/art bash tools/fetch_448_960m.sh
#
# 环境变量: PORT（默认 23556）/ HOST / SRC / DST / DRY
# 返回: 0 至少拉到 1 个文件；1 一个都没拉到（检查 ssh / 是否还没开始跑）
# =============================================================================
set -uo pipefail
HOST=${HOST:-root@connect.nma1.seetacloud.com}
PORT=${PORT:-23556}
SRC=${SRC:-/root/autodl-tmp/exp448}
DST=${DST:-/home/linaro/dsh/exp448_bench/artifacts}
DRY=${DRY:-0}

SSH=( ssh -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$HOST" )
SCP=( scp -P "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 )

PULLED=()
MISSING=()
say() { echo "[fetch448] $*"; }

[ "$DRY" = "1" ] || mkdir -p "$DST"/{eval,logs,args,train_logs,markers}
say "SRC=$HOST:$SRC  →  DST=$DST  ($(date -Is))"

# 先确认远端目录在（避免 30 个 scp 各自超时）
if [ "$DRY" != "1" ]; then
  if ! "${SSH[@]}" "test -d '$SRC'" 2>/dev/null; then
    echo "[fetch448] ✗ 远端目录不存在或 ssh 不通: $SRC" >&2
    echo "           先确认: ssh -p $PORT $HOST 'ls $SRC'" >&2
    exit 1
  fi
fi

# ── 文本文件：ssh cat + 去 \r 进度条 + 去空行 ────────────────────────────────
pull_text() { # $1=远端相对路径  $2=本地路径
  local rp="$SRC/$1" lp="$2" n
  if [ "$DRY" = "1" ]; then
    say "[dry] ssh -p $PORT $HOST \"cat '$rp'\" | tr '\\r' '\\n' | 去进度条 > $lp"
    return 0
  fi
  mkdir -p "$(dirname "$lp")"
  if "${SSH[@]}" "cat '$rp'" 2>/dev/null \
       | tr '\r' '\n' \
       | grep -vE '^[[:space:]]*[0-9]+%\|' \
       | grep -vE '^[[:space:]]*$' > "$lp.tmp"; then
    if [ -s "$lp.tmp" ]; then
      mv "$lp.tmp" "$lp"; n=$(wc -l < "$lp")
      PULLED+=("$1"); say "  ✓ $1  ($n 行 → ${lp#"$DST"/})"; return 0
    fi
  fi
  rm -f "$lp.tmp"; MISSING+=("$1"); say "  · 缺 $1"; return 1
}

# ── 二进制/JSON：scp 原样 ─────────────────────────────────────────────────
pull_file() { # $1=远端相对路径  $2=本地路径
  local rp="$SRC/$1" lp="$2"
  if [ "$DRY" = "1" ]; then
    say "[dry] scp -P $PORT $HOST:$rp $lp"
    return 0
  fi
  mkdir -p "$(dirname "$lp")"
  # 注意: 标记文件（STAGE*_DONE / ALL_DONE）是 touch 出来的 0 字节文件 ⇒ 不能按非空判断
  if "${SCP[@]}" -q "$HOST:$rp" "$lp" 2>/dev/null && [ -e "$lp" ]; then
    PULLED+=("$1"); say "  ✓ $1 → ${lp#"$DST"/}"; return 0
  fi
  rm -f "$lp"; MISSING+=("$1"); say "  · 缺 $1"; return 1
}

# ── 文本拉取时远端不存在可能产生空文件：已由 -s 判断兜住 ─────────────────────

say "— eval（逐步评测 json）"
for n in 1 2 3; do
  pull_file "eval/stage${n}_stepwise.json" "$DST/eval/stage${n}_stepwise.json"
done

say "— 训练/评测日志（已去 \\r 进度条）"
for n in 1 2 3; do
  pull_text "logs/stage${n}.log"       "$DST/logs/stage${n}.log"
  pull_text "logs/eval_stage${n}.log"  "$DST/logs/eval_stage${n}.log"
done
pull_text "logs/run.log"     "$DST/logs/run.log"
pull_text "logs/launch.out"  "$DST/logs/launch.out"

say "— args / status / eval_val（每阶段）"
for s in A B S3; do
  case "$s" in A) n=1 ;; B) n=2 ;; *) n=3 ;; esac
  pull_file "$s/args.json"      "$DST/args/stage${n}_args.json"
  pull_file "$s/status.json"    "$DST/args/stage${n}_status.json"
  pull_file "$s/eval_val.json"  "$DST/args/stage${n}_eval_val.json"
  pull_file "$s/train_log.jsonl" "$DST/train_logs/stage${n}.jsonl"
done

say "— run_info.json 与状态标记"
pull_file "run_info.json" "$DST/run_info.json"
for m in STAGE1_DONE STAGE2_DONE STAGE3_DONE ALL_DONE \
         FAILED_stage1 FAILED_stage2 FAILED_stage3 \
         FAILED_eval_stage1 FAILED_eval_stage2 FAILED_eval_stage3 \
         BUDGET_HIT_stage1 BUDGET_HIT_stage2 BUDGET_HIT_stage3; do
  pull_file "$m" "$DST/markers/$m"
done

# ── 从训练日志摘 cfg / params（cfg 本体在 ckpt["cfg"], 未单独落盘）─────────────
if [ "$DRY" != "1" ]; then
  for n in 1 2 3; do
    lg=$DST/logs/stage${n}.log
    [ -f "$lg" ] || continue
    { echo "# stage${n} cfg / params / alit（摘自 $lg 的 [cfg] [params] [alit] 行）"
      grep -hE '^\[(cfg|params|alit|data|accum|init_from)\]' "$lg" | head -20
    } > "$DST/args/stage${n}_cfg.txt"
  done
fi

# ── 汇总 ───────────────────────────────────────────────────────────────────
if [ "$DRY" != "1" ]; then
  SUM=$DST/fetch_summary.txt
  {
    echo "# fetch_448_960m.sh  $(date -Is)"
    echo "# SRC=$HOST:$SRC"
    echo "# DST=$DST"
    echo "pulled=${#PULLED[@]}  missing=${#MISSING[@]}"
    echo
    echo "## 状态标记"
    for m in STAGE1_DONE STAGE2_DONE STAGE3_DONE ALL_DONE; do
      [ -f "$DST/markers/$m" ] && echo "  $m  ✅"
    done
    for m in FAILED_stage1 FAILED_stage2 FAILED_stage3 FAILED_eval_stage1 FAILED_eval_stage2 \
             FAILED_eval_stage3 BUDGET_HIT_stage1 BUDGET_HIT_stage2 BUDGET_HIT_stage3; do
      [ -f "$DST/markers/$m" ] && echo "  $m  ❌ $(head -1 "$DST/markers/$m" 2>/dev/null)"
    done
    echo
    echo "## 已拉取"
    printf '  %s\n' "${PULLED[@]:-（无）}"
    echo
    echo "## 缺失"
    printf '  %s\n' "${MISSING[@]:-（无）}"
  } > "$SUM"
  say "汇总 → $SUM"
  say "拉了 ${#PULLED[@]} 个, 缺 ${#MISSING[@]} 个"
  if [ "${#PULLED[@]}" -eq 0 ]; then
    echo "[fetch448] ✗ 一个文件都没拉到：确认远端是否已开跑（$SRC/logs/stage1.log）" >&2
    exit 1
  fi
  ls -la "$DST" "$DST/eval" "$DST/logs" 2>/dev/null | head -40
else
  say "(dry) 结束：未建目录、未拉任何文件"
fi
exit 0
