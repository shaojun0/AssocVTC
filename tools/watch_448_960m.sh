#!/usr/bin/env bash
# =============================================================================
# watch_448_960m.sh — AssocVTC 448²/patch32/960M 三阶段长跑 watchdog（远端常驻）
# =============================================================================
# 用途
#   每 INTERVAL(=20min) 轮询 /root/autodl-tmp/exp448，解析 logs/stage{1,2,3}.log，
#   原子写 $OUT/status.json、追加 $OUT/status.log；异常写 $OUT/ALERT_<ts>.txt。
#   本脚本**不启动训练、不杀进程、不改仓库**，只读日志/进程/磁盘。
#
# 检测项
#   1) 进程消失但阶段未完成（崩溃），容忍阶段切换中的短暂无进程（NO_PROC_GRACE）
#   2) 进程活着但当前阶段日志 mtime > STALL_SEC(=40min)（僵死）
#   3) OOM / NaN/Inf loss / 梯度 skip 异常 / Traceback
#   4) 磁盘可用 < DISK_MIN_GB(=5G)
#   5) FAILED 标记、启动后迟迟无日志（LAUNCH_STUCK）
#   eval 期间（eval_stepwise_linattn.py / eval_448_960m.* 在跑）不误判僵死。
#
# 启动（远端，任选其一）
#   cd /root/autodl-tmp/exp448/code/AssocVTC
#   OUT=/root/autodl-tmp/exp448 setsid nohup bash tools/watch_448_960m.sh --daemon \
#       >/root/autodl-tmp/exp448/watch.out 2>&1 < /dev/null &
#   # 或让脚本自己 daemon 化（推荐，带 flock 单实例）：
#   bash tools/watch_448_960m.sh --daemon
#
# 单次（调试/演习，只写一次 status.json，不常驻）：
#   bash tools/watch_448_960m.sh --once
#
# 查看：cat /root/autodl-tmp/exp448/status.json ; tail -5 .../status.log
# 停止：pkill -f watch_448_960m.sh
#
# 可覆盖环境变量
#   OUT LOGS INTERVAL STALL_SEC NO_PROC_GRACE START_GRACE DISK_MIN_GB DISK_PATH
#   TRAIN_PATTERN CHAIN_PATTERN EVAL_PATTERN STAGE_EPOCHS ALERT_REPEAT_SEC
#   WATCH_NOW_TS（演习用：把"现在"固定为给定 epoch 秒）
#
# 退出码（--once）：0 无告警；10 有 error 级告警；20 有 warn 级告警
# =============================================================================
set -u
LC_ALL=C

VERSION="1.1"
OUT=${OUT:-/root/autodl-tmp/exp448}
LOGS=${LOGS:-$OUT/logs}
INTERVAL=${INTERVAL:-1200}            # 20 min
STALL_SEC=${STALL_SEC:-2400}          # 40 min 日志无更新 = 僵死
NO_PROC_GRACE=${NO_PROC_GRACE:-600}   # 无进程宽限 10 min（容忍阶段切换）
START_GRACE=${START_GRACE:-1800}      # launch 后 30 min 仍无日志 = LAUNCH_STUCK
DISK_MIN_GB=${DISK_MIN_GB:-5}
DISK_PATH=${DISK_PATH:-/root/autodl-tmp}
TRAIN_PATTERN=${TRAIN_PATTERN:-train_linattn\.py}
# 排除 bench/探针等旁路训练进程（cmdline 含该子串即忽略），避免把 bench 当正式训练
TRAIN_EXCLUDE=${TRAIN_EXCLUDE:-/exp448/bench}
CHAIN_PATTERN=${CHAIN_PATTERN:-run_448_960m_3stage\.sh}
EVAL_PATTERN=${EVAL_PATTERN:-eval_(stepwise_linattn\.py|448_960m)}
STAGE_EPOCHS=${STAGE_EPOCHS:-"45 6 3"}
ALERT_REPEAT_SEC=${ALERT_REPEAT_SEC:-7200}   # 同一告警最少 2h 才重复写 ALERT 文件

STATE="$OUT/.watch_state"
LOCK="$OUT/.watch.lock"
status_log_rotate=3000

mkdir -p "$OUT" "$LOGS" 2>/dev/null || true

now_epoch() {
  if [ -n "${WATCH_NOW_TS:-}" ]; then printf '%s' "$WATCH_NOW_TS"; else date +%s; fi
}
ts_iso() {
  if [ -n "${WATCH_NOW_TS:-}" ]; then
    date -d "@${WATCH_NOW_TS}" -Iseconds 2>/dev/null || date -Iseconds
  else
    date -Iseconds
  fi
}
fmtime() { stat -c %Y "$1" 2>/dev/null || echo 0; }

# ---- 状态文件（KEY=VALUE，只有本脚本写） -----------------------------------
state_get() { [ -f "$STATE" ] && awk -F= -v k="$1" '$1==k{print substr($0, index($0,"=")+1)}' "$STATE" 2>/dev/null | tail -n 1; }
state_set() {
  local k=$1 v=${2:-} tmp="$STATE.tmp.$$"
  { [ -f "$STATE" ] && grep -v "^${k}=" "$STATE" 2>/dev/null; printf '%s=%s\n' "$k" "$v"; } > "$tmp" 2>/dev/null && mv -f "$tmp" "$STATE"
}

# ---- 定位日志 --------------------------------------------------------------
resolve_log() {                        # $1 = 阶段号; 输出路径或返回 1
  local s=$1 f g
  f="$LOGS/stage${s}.log"
  [ -f "$f" ] && { printf '%s' "$f"; return 0; }
  for g in "$LOGS"/stage${s}_*.log "$LOGS"/*stage${s}*.log; do
    [ -f "$g" ] && { printf '%s' "$g"; return 0; }
  done
  return 1
}
latest_log_info() {                    # 输出 "stage|path|mtime"
  local s f m bs=0 bp="" bm=0
  for s in 1 2 3; do
    if f=$(resolve_log "$s"); then
      m=$(fmtime "$f")
      if [ "$m" -gt "$bm" ]; then bm=$m; bp=$f; bs=$s; fi
    fi
  done
  [ -n "$bp" ] && printf '%s|%s|%s' "$bs" "$bp" "$bm"
  return 0
}
done_flag() { [ -f "$OUT/STAGE${1}_DONE" ]; }

# 列出匹配进程；排除自身/祖先（cmdline 里含 watch_448_960m 的 shell 包装），
# 避免「调用行里恰好写了 train 模式串」导致误判进程存在。
list_pids() {                          # $1=pgrep 模式  $2=排除子串（可空）
  local p cl pat=$1 excl=${2:-}
  pgrep -f "$pat" 2>/dev/null | while read -r p; do
    [ -z "$p" ] && continue
    [ "$p" = "$$" ] && continue
    cl=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
    case "$cl" in *watch_448_960m*) continue ;; esac
    if [ -n "$excl" ]; then
      printf '%s' "$cl" | grep -q -- "$excl" && continue
    fi
    printf '%s\n' "$p"
  done
}
active_stage() {                       # 第一个没 DONE 的阶段
  local s
  for s in 1 2 3; do done_flag "$s" || { printf '%s' "$s"; return 0; }; done
  printf '3'
}
stage_epochs() {                       # $1=stage
  local s=$1 i=1
  for e in $STAGE_EPOCHS; do [ "$i" -eq "$s" ] && { printf '%s' "$e"; return 0; }; i=$((i+1)); done
  printf ''
}

# ---- 解析：最后一行训练进度 -------------------------------------------------
PROG_LINE=""; PROG_WALLM=""; PROG_STEP=""; PROG_TOTAL=""; PROG_EP=""
PROG_LOSS=""; PROG_RECON=""; PROG_LR=""; PROG_GN=""; PROG_SKIP=""
PROG_IMGS=""; PROG_VRAM=""; PROG_ETA=""
read_progress() {
  local f=$1 L
  PROG_LINE=""; PROG_WALLM=""; PROG_STEP=""; PROG_TOTAL=""; PROG_EP=""
  PROG_LOSS=""; PROG_RECON=""; PROG_LR=""; PROG_GN=""; PROG_SKIP=""
  PROG_IMGS=""; PROG_VRAM=""; PROG_ETA=""
  [ -f "$f" ] || return 0
  PROG_LINE=$(grep -E '^\[ *[0-9.]+m\] step [0-9]+/[0-9]+' "$f" 2>/dev/null | tail -n 1)
  [ -n "$PROG_LINE" ] || return 0
  L=$PROG_LINE
  PROG_WALLM=$(printf '%s' "$L" | sed -nE 's/^\[ *([0-9.]+)m\].*/\1/p')
  PROG_STEP=$(printf '%s' "$L"  | sed -nE 's/.* step ([0-9]+)\/([0-9]+).*/\1/p')
  PROG_TOTAL=$(printf '%s' "$L" | sed -nE 's/.* step ([0-9]+)\/([0-9]+).*/\2/p')
  PROG_EP=$(printf '%s' "$L"    | sed -nE 's/.* ep([0-9]+) .*/\1/p')
  PROG_LOSS=$(printf '%s' "$L"  | sed -nE 's/.* loss=([^ ]+).*/\1/p')
  PROG_RECON=$(printf '%s' "$L" | sed -nE 's/.* recon=([^ ]+).*/\1/p')
  PROG_LR=$(printf '%s' "$L"    | sed -nE 's/.* lr=([^ ]+).*/\1/p')
  PROG_GN=$(printf '%s' "$L"    | sed -nE 's/.* gn=([^ ]+).*/\1/p')
  PROG_SKIP=$(printf '%s' "$L"  | sed -nE 's/.* skip=([0-9]+).*/\1/p')
  PROG_IMGS=$(printf '%s' "$L"  | sed -nE 's/.* ([0-9]+)img\/s.*/\1/p')
  PROG_VRAM=$(printf '%s' "$L"  | sed -nE 's/.* vram=([0-9.]+)G.*/\1/p')
  PROG_ETA=$(printf '%s' "$L"   | sed -nE 's/.* eta=([0-9.]+)h.*/\1/p')
  return 0
}

# ---- 解析：最后一行 val -----------------------------------------------------
VAL_LINE=""; VAL_PSNR=""; VAL_SSIM=""; VAL_L1=""; VAL_EP=""
read_val() {
  local f=$1 L
  VAL_LINE=""; VAL_PSNR=""; VAL_SSIM=""; VAL_L1=""; VAL_EP=""
  [ -f "$f" ] || return 0
  VAL_LINE=$(grep -E '^\[val\]' "$f" 2>/dev/null | tail -n 1)
  [ -n "$VAL_LINE" ] || return 0
  L=$VAL_LINE
  VAL_PSNR=$(printf '%s' "$L" | sed -nE 's/.*psnr=([^ ]+)dB.*/\1/p')
  VAL_SSIM=$(printf '%s' "$L" | sed -nE 's/.*ssim=([^ ]+).*/\1/p')
  VAL_L1=$(printf '%s' "$L"   | sed -nE 's/.*l1=([^ ]+).*/\1/p')
  VAL_EP=$(printf '%s' "$L"   | sed -nE 's/.* ep([0-9]+) .*/\1/p')
  return 0
}

# ---- 异常扫描 ---------------------------------------------------------------
SKIP_N=0; SKIP_BAD=0; SKIP_MAX=0
read_skip_stats() {
  local f=$1 recent
  SKIP_N=0; SKIP_BAD=0; SKIP_MAX=0
  [ -f "$f" ] || return 0
  recent=$(grep -E '^\[ *[0-9.]+m\] step [0-9]+/[0-9]+' "$f" 2>/dev/null | tail -n 30 | sed -nE 's/.* skip=([0-9]+).*/\1/p')
  [ -n "$recent" ] || return 0
  SKIP_N=$(printf '%s\n' "$recent" | wc -l | tr -d ' ')
  SKIP_BAD=$(printf '%s\n' "$recent" | awk '$1>0{c++} END{print c+0}')
  SKIP_MAX=$(printf '%s\n' "$recent" | sort -n | tail -n 1)
  return 0
}
scan_anomaly() {                       # 输出 code|msg
  local f=$1
  [ -f "$f" ] || return 0
  if tail -n 500 "$f" 2>/dev/null | grep -qiE 'CUDA out of memory|torch\.cuda\.OutOfMemoryError|OutOfMemoryError|out of memory'; then
    printf 'OOM|日志出现 CUDA out of memory / OutOfMemoryError\n'
  fi
  if tail -n 400 "$f" 2>/dev/null | grep -qiE '(loss|recon|gn)=(nan|inf)|psnr=nan|l1=nan|non-finite'; then
    printf 'NAN|日志出现 nan/inf loss 或非有限数值\n'
  fi
  if tail -n 300 "$f" 2>/dev/null | grep -qE 'Traceback \(most recent call last\)'; then
    printf 'TRACEBACK|日志尾部出现 Traceback\n'
  fi
  return 0
}

# ---- JSON 片段 --------------------------------------------------------------
jnum() {                               # 数字或 null（nan/inf/空 → null）
  case "${1:-}" in
    ''|nan|NaN|inf|-inf|Inf) printf 'null' ;;
    *) printf '%s' "$1" | grep -qE '^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$' && printf '%s' "$1" || printf 'null' ;;
  esac
}
jstr() {
  printf '"%s"' "$(printf '%s' "${1:-}" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr '\n' ' ')"
}

# ---- ALERT 文件 -------------------------------------------------------------
write_alert_file() {
  local sig=$1 body ts f
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  f="$OUT/ALERT_${ts}.txt"
  [ -e "$f" ] && f="$OUT/ALERT_${ts}_$$.txt"
  body=$(
    printf 'ts: %s\n' "$(ts_iso)"
    printf 'host: %s\n' "$HOST_SHORT"
    printf 'alert_signature: %s\n' "$sig"
    printf 'stage: %s/%s  state: %s  step: %s/%s  ep: %s\n' "$ACTIVE" "3" "$STAGE_STATE" "$PROG_STEP" "$PROG_TOTAL" "$PROG_EP"
    printf 'alerts:\n'
    for a in "${ALERTS[@]:-}"; do
      [ -z "$a" ] && continue
      printf '  - [%s] %s: %s\n' "$(printf '%s' "$a" | cut -d'|' -f1)" "$(printf '%s' "$a" | cut -d'|' -f2)" "$(printf '%s' "$a" | cut -d'|' -f3)"
    done
    printf '\nlast_log: %s\n' "${LATEST_PATH:-none}"
    if [ -n "${LATEST_PATH:-}" ] && [ -f "$LATEST_PATH" ]; then
      printf 'last_log_lines:\n'; tail -n 6 "$LATEST_PATH" 2>/dev/null | sed 's/^/  /'
    fi
    printf '\nstatus.json: %s\nstatus.log: %s\n' "$OUT/status.json" "$OUT/status.log"
    printf '处理建议: 检查 %s 尾部与 FAILED 标记；确认 GPU/磁盘；重启命令见 tools/launch_448_960m.sh\n' "${LATEST_PATH:-$LOGS}"
  )
  printf '%s\n' "$body" > "$f" 2>/dev/null || true
  printf '%s' "$f"
}

# ---- 单次轮询 ---------------------------------------------------------------
poll_once() {
  local NOW TS HOST_SHORT
  NOW=$(now_epoch); TS=$(ts_iso)
  HOST_SHORT=$(hostname 2>/dev/null || echo unknown)

  local complete=0 failed=0
  [ -f "$OUT/ALL_DONE" ] && complete=1
  if [ -f "$OUT/FAILED" ] || ls "$OUT"/FAILED_* >/dev/null 2>&1; then failed=1; fi
  local run_info_mtime=0 run_info_age=-1
  [ -f "$OUT/run_info.json" ] && { run_info_mtime=$(fmtime "$OUT/run_info.json"); run_info_age=$((NOW - run_info_mtime)); }

  local TRAIN_PIDS CHAIN_PIDS EVAL_PIDS TRAIN_ALIVE=0 CHAIN_ALIVE=0 EVAL_ALIVE=0 pid_alive=0
  TRAIN_PIDS=$(list_pids "$TRAIN_PATTERN" "$TRAIN_EXCLUDE" | head -n 5 | tr '\n' ' ')
  CHAIN_PIDS=$(list_pids "$CHAIN_PATTERN" | head -n 5 | tr '\n' ' ')
  EVAL_PIDS=$(list_pids "$EVAL_PATTERN"   | head -n 5 | tr '\n' ' ')
  [ -n "${TRAIN_PIDS// /}" ] && { TRAIN_ALIVE=1; pid_alive=1; }
  [ -n "${CHAIN_PIDS// /}" ] && CHAIN_ALIVE=1
  [ -n "${EVAL_PIDS// /}" ] && EVAL_ALIVE=1

  local ACTIVE latest latest_stage latest_path latest_mtime latest_age=-1
  ACTIVE=$(active_stage)
  latest=$(latest_log_info)
  latest_stage=""; latest_path=""; latest_mtime=0
  if [ -n "$latest" ]; then
    latest_stage=$(printf '%s' "$latest" | cut -d'|' -f1)
    latest_path=$(printf '%s' "$latest"  | cut -d'|' -f2)
    latest_mtime=$(printf '%s' "$latest" | cut -d'|' -f3)
    latest_age=$((NOW - latest_mtime))
  fi

  # 当前阶段日志（active stage 的；没有则退到最新日志）
  local cur_log="" cur_log_mtime=0 cur_log_age=-1
  cur_log=$(resolve_log "$ACTIVE" 2>/dev/null || true)
  if [ -n "$cur_log" ]; then
    cur_log_mtime=$(fmtime "$cur_log"); cur_log_age=$((NOW - cur_log_mtime))
  fi

  local eval_json_n=0
  if [ -d "$OUT/eval" ]; then eval_json_n=$(ls -1 "$OUT"/eval/*.json 2>/dev/null | wc -l | tr -d ' '); fi

  # ---- 解析（用 active 日志；没有就用最新日志） ----
  local plog=$cur_log; [ -z "$plog" ] && plog=$latest_path
  read_progress "$plog"
  read_val "$plog"
  read_skip_stats "$plog"
  PROG_STEP=${PROG_STEP:-0}; PROG_TOTAL=${PROG_TOTAL:-0}; PROG_EP=${PROG_EP:-0}

  # ---- 状态机 + 告警 ----
  local ALERTS=() a sig
  STAGE_STATE="unknown"
  add_alert() { ALERTS+=("$1|$2|$3"); }

  local progress_pct=""
  if [ "${PROG_STEP:-0}" -gt 0 ] && [ "${PROG_TOTAL:-0}" -gt 0 ]; then
    progress_pct=$(awk -v s="$PROG_STEP" -v t="$PROG_TOTAL" 'BEGIN{printf "%.2f", 100*s/t}')
  fi

  if [ "$complete" = 1 ]; then
    STAGE_STATE="done"
    state_set first_noproc_ts ""
  elif [ "$failed" = 1 ]; then
    STAGE_STATE="failed"
    add_alert "error" "FAILED_MARKER" "检测到 $OUT/FAILED 标记：训练链路已失败退出"
  elif [ "$TRAIN_ALIVE" = 1 ]; then
    STAGE_STATE="running"
    state_set first_noproc_ts ""
    if [ -n "$cur_log" ] && [ "$cur_log_age" -gt "$STALL_SEC" ]; then
      add_alert "error" "STALLED" "训练进程在，但当前阶段日志已 ${cur_log_age}s 未更新（>${STALL_SEC}s）"
    fi
  elif [ "$CHAIN_ALIVE" = 1 ] || [ "$EVAL_ALIVE" = 1 ]; then
    # 阶段切换 / 评测中：无训练进程但链路或 eval 在跑
    if [ "$EVAL_ALIVE" = 1 ]; then STAGE_STATE="evaluating"; else STAGE_STATE="between_stages"; fi
    state_set first_noproc_ts ""
    if [ -n "$latest_path" ] && [ "$latest_age" -gt "$STALL_SEC" ] && [ "$EVAL_ALIVE" = 0 ]; then
      add_alert "error" "STALLED" "链路进程在，但日志已 ${latest_age}s 未更新（>${STALL_SEC}s）"
    fi
  else
    # 没有任何训练/链路/eval 进程
    local first_noproc grace_age
    first_noproc=$(state_get first_noproc_ts)
    if [ -z "$latest_path" ]; then
      # 从未有日志
      if [ "$run_info_age" -ge 0 ] && [ "$run_info_age" -gt "$START_GRACE" ]; then
        STAGE_STATE="launch_stuck"
        add_alert "error" "LAUNCH_STUCK" "run_info.json 已 ${run_info_age}s，但仍无任何阶段日志"
      else
        STAGE_STATE="not_started"
      fi
      state_set first_noproc_ts ""
    elif [ -z "$cur_log" ]; then
      # 前一阶段已完成、当前阶段日志未出现（阶段间空档）
      if [ -z "$first_noproc" ]; then first_noproc=$NOW; state_set first_noproc_ts "$NOW"; fi
      grace_age=$((NOW - first_noproc))
      if [ "$latest_age" -gt "$STALL_SEC" ] || [ "$grace_age" -gt "$NO_PROC_GRACE" ]; then
        STAGE_STATE="chain_stuck"
        add_alert "error" "NO_PROC_AFTER_DONE" "阶段${ACTIVE}无进程且无日志（前一阶段日志已 ${latest_age}s，空档已 ${grace_age}s）"
      else
        STAGE_STATE="transition"
      fi
    else
      if [ -z "$first_noproc" ]; then first_noproc=$NOW; state_set first_noproc_ts "$NOW"; fi
      grace_age=$((NOW - first_noproc))
      if [ "$cur_log_age" -gt "$STALL_SEC" ] || [ "$grace_age" -gt "$NO_PROC_GRACE" ]; then
        STAGE_STATE="crashed"
        add_alert "error" "CRASHED" "训练进程消失且阶段${ACTIVE}无 DONE 标记（日志已 ${cur_log_age}s 未更新）"
      else
        STAGE_STATE="no_proc_grace"
      fi
    fi
  fi

  # 日志异常扫描（只看当前/最新日志尾部）
  local anom
  if [ -n "$plog" ]; then
    while IFS='|' read -r code msg; do
      [ -n "${code:-}" ] && add_alert "error" "$code" "$msg"
    done < <(scan_anomaly "$plog")
  fi
  if [ "${SKIP_N:-0}" -ge 5 ] && [ "${SKIP_BAD:-0}" -gt 0 ]; then
    local ratio; ratio=$(awk -v b="$SKIP_BAD" -v n="$SKIP_N" 'BEGIN{printf "%.0f", 100*b/n}')
    if [ "$ratio" -ge 30 ] || [ "${SKIP_MAX:-0}" -ge 10 ]; then
      add_alert "warn" "SKIP_HIGH" "最近 ${SKIP_N} 步梯度 skip 占比 ${ratio}%（max=${SKIP_MAX}）"
    fi
  fi

  # 磁盘（DISK_PATH 不存在时退到 OUT 所在分区）
  local disk_free_kb="" disk_free_gb="" dpath="$DISK_PATH"
  [ -d "$dpath" ] || dpath="$OUT"
  if [ -d "$dpath" ]; then
    disk_free_kb=$(df -Pk "$dpath" 2>/dev/null | awk 'NR==2{print $4}')
    if [ -n "$disk_free_kb" ]; then
      disk_free_gb=$(awk -v k="$disk_free_kb" 'BEGIN{printf "%.1f", k/1048576}')
      if [ "$disk_free_kb" -lt $((DISK_MIN_GB * 1048576)) ]; then
        add_alert "error" "DISK_LOW" "磁盘 $dpath 可用 ${disk_free_gb}G (<${DISK_MIN_GB}G)"
      fi
    fi
  fi

  # ---- 写 status.json（原子） ----
  local tmp="$OUT/.status.json.tmp.$$" n_alerts=0 n_err=0 n_warn=0 alerts_json=""
  if [ "${#ALERTS[@]}" -gt 0 ]; then
    for a in "${ALERTS[@]}"; do
      local lvl code msg
      lvl=$(printf '%s' "$a" | cut -d'|' -f1)
      code=$(printf '%s' "$a" | cut -d'|' -f2)
      msg=$(printf '%s' "$a" | cut -d'|' -f3)
      [ -n "$alerts_json" ] && alerts_json="$alerts_json,"
      alerts_json="$alerts_json{\"level\":$(jstr "$lvl"),\"code\":$(jstr "$code"),\"msg\":$(jstr "$msg")}"
      n_alerts=$((n_alerts+1))
      [ "$lvl" = "error" ] && n_err=$((n_err+1))
      [ "$lvl" = "warn" ]  && n_warn=$((n_warn+1))
    done
  fi
  local stage_epochs_v; stage_epochs_v=$(stage_epochs "$ACTIVE")
  local ts_utc; ts_utc=$(date -u -d "@$NOW" -Iseconds 2>/dev/null || date -u -Iseconds)

  {
    printf '{\n'
    printf '  "schema": "assoc448.status.v1",\n'
    printf '  "ts": "%s",\n' "$TS"
    printf '  "ts_epoch": %s,\n' "$NOW"
    printf '  "ts_utc": "%s",\n' "$ts_utc"
    printf '  "host": "%s",\n' "$HOST_SHORT"
    printf '  "watchdog_version": "%s",\n' "$VERSION"
    printf '  "watchdog_pid": %s,\n' "$$"
    printf '  "out": "%s",\n' "$OUT"
    printf '  "stage": %s,\n' "$ACTIVE"
    printf '  "stage_state": "%s",\n' "$STAGE_STATE"
    printf '  "stage_done": %s,\n' "$(done_flag "$ACTIVE" && echo true || echo false)"
    printf '  "total_stages": 3,\n'
    printf '  "step": %s,\n' "$(jnum "${PROG_STEP:-}")"
    printf '  "total_steps": %s,\n' "$(jnum "${PROG_TOTAL:-}")"
    printf '  "progress_pct": %s,\n' "$(jnum "${progress_pct:-}")"
    printf '  "epoch": %s,\n' "$(jnum "${PROG_EP:-}")"
    printf '  "total_epochs": %s,\n' "$(jnum "${stage_epochs_v:-}")"
    printf '  "loss": %s,\n' "$(jnum "${PROG_LOSS:-}")"
    printf '  "recon": %s,\n' "$(jnum "${PROG_RECON:-}")"
    printf '  "psnr": %s,\n' "$(jnum "${VAL_PSNR:-}")"
    printf '  "ssim": %s,\n' "$(jnum "${VAL_SSIM:-}")"
    printf '  "val_l1": %s,\n' "$(jnum "${VAL_L1:-}")"
    printf '  "lr": %s,\n' "$(jnum "${PROG_LR:-}")"
    printf '  "grad_norm": %s,\n' "$(jnum "${PROG_GN:-}")"
    printf '  "skip": %s,\n' "$(jnum "${PROG_SKIP:-}")"
    printf '  "img_per_s": %s,\n' "$(jnum "${PROG_IMGS:-}")"
    printf '  "vram_gb": %s,\n' "$(jnum "${PROG_VRAM:-}")"
    printf '  "eta_h": %s,\n' "$(jnum "${PROG_ETA:-}")"
    printf '  "wall_min": %s,\n' "$(jnum "${PROG_WALLM:-}")"
    printf '  "pid_alive": %s,\n' "$([ "$pid_alive" = 1 ] && echo true || echo false)"
    printf '  "train_pid": %s,\n' "$(jnum "$(printf '%s' "$TRAIN_PIDS" | awk '{print $1}')")"
    printf '  "train_alive": %s,\n' "$([ "$TRAIN_ALIVE" = 1 ] && echo true || echo false)"
    printf '  "chain_alive": %s,\n' "$([ "$CHAIN_ALIVE" = 1 ] && echo true || echo false)"
    printf '  "eval_alive": %s,\n' "$([ "$EVAL_ALIVE" = 1 ] && echo true || echo false)"
    printf '  "eval_json_n": %s,\n' "$eval_json_n"
    printf '  "disk_free_gb": %s,\n' "$(jnum "${disk_free_gb:-}")"
    printf '  "disk_min_gb": %s,\n' "$DISK_MIN_GB"
    printf '  "last_log": %s,\n' "$(jstr "${latest_path:-}")"
    printf '  "last_log_stage": %s,\n' "$(jnum "${latest_stage:-}")"
    printf '  "last_log_age_s": %s,\n' "$(jnum "${latest_age:--1}")"
    printf '  "cur_log": %s,\n' "$(jstr "${cur_log:-}")"
    printf '  "cur_log_age_s": %s,\n' "$(jnum "${cur_log_age:--1}")"
    printf '  "skip_recent_n": %s,\n' "$(jnum "${SKIP_N:-0}")"
    printf '  "skip_recent_bad": %s,\n' "$(jnum "${SKIP_BAD:-0}")"
    printf '  "complete": %s,\n' "$([ "$complete" = 1 ] && echo true || echo false)"
    printf '  "failed": %s,\n' "$([ "$failed" = 1 ] && echo true || echo false)"
    printf '  "run_info_age_s": %s,\n' "$(jnum "${run_info_age:--1}")"
    printf '  "n_alerts": %s,\n' "$n_alerts"
    printf '  "n_errors": %s,\n' "$n_err"
    printf '  "n_warnings": %s,\n' "$n_warn"
    printf '  "alerts": [%s]\n' "$alerts_json"
    printf '}\n'
  } > "$tmp" 2>/dev/null && mv -f "$tmp" "$OUT/status.json"

  # ---- 告警文件（签名变化或超过重复间隔才写） ----
  sig=""
  if [ "$n_alerts" -gt 0 ]; then
    sig=$(for a in "${ALERTS[@]}"; do printf '%s\n' "$a" | cut -d'|' -f2; done | sort -u | paste -sd, -)
    local prev_sig last_alert_epoch
    prev_sig=$(state_get last_alert_sig)
    last_alert_epoch=$(state_get last_alert_epoch)
    if [ "$sig" != "${prev_sig:-}" ] || [ -z "${last_alert_epoch:-}" ] || [ $((NOW - last_alert_epoch)) -ge "$ALERT_REPEAT_SEC" ]; then
      local afile; afile=$(write_alert_file "$sig")
      state_set last_alert_epoch "$NOW"
      printf '%s [ALERT] %s -> %s\n' "$TS" "$sig" "$afile" >> "$OUT/status.log" 2>/dev/null || true
    fi
    state_set last_alert_sig "$sig"
  else
    state_set last_alert_sig ""
    state_set last_alert_epoch ""
  fi

  # ---- status.log 追加 + 轻量轮转 ----
  local alert_codes="-"
  [ "$n_alerts" -gt 0 ] && alert_codes=$(printf '%s\n' "${ALERTS[@]}" | cut -d'|' -f2 | paste -sd+ -)
  printf '%s state=%s stage=%s ep=%s step=%s/%s loss=%s lr=%s vram=%sG %simg/s eta=%sh alive=%s eval=%s disk=%sG alerts=%s\n' \
    "$TS" "$STAGE_STATE" "$ACTIVE" "${PROG_EP:-?}" "${PROG_STEP:-?}" "${PROG_TOTAL:-?}" \
    "${PROG_LOSS:-?}" "${PROG_LR:-?}" "${PROG_VRAM:-?}" "${PROG_IMGS:-?}" "${PROG_ETA:-?}" \
    "$pid_alive" "$EVAL_ALIVE" "${disk_free_gb:-?}" "$alert_codes" \
    >> "$OUT/status.log" 2>/dev/null || true
  if [ "$(wc -l < "$OUT/status.log" 2>/dev/null || echo 0)" -gt "$status_log_rotate" ]; then
    tail -n 2000 "$OUT/status.log" > "$OUT/.status.log.tmp" 2>/dev/null && mv -f "$OUT/.status.log.tmp" "$OUT/status.log"
  fi

  # ---- 人读一行（stdout / watch.out） ----
  printf '%s  stage=%s/%s %s  step=%s/%s ep=%s loss=%s psnr=%s lr=%s vram=%s img/s=%s eta=%sh alive=%s alerts=%s\n' \
    "$TS" "$ACTIVE" "3" "$STAGE_STATE" "${PROG_STEP:-?}" "${PROG_TOTAL:-?}" "${PROG_EP:-?}" \
    "${PROG_LOSS:-?}" "${VAL_PSNR:-?}" "${PROG_LR:-?}" "${PROG_VRAM:-?}" "${PROG_IMGS:-?}" \
    "${PROG_ETA:-?}" "$pid_alive" "$alert_codes"

  # 退出码（--once 用）：有 error→10，只有 warn→20
  if [ "$n_err" -gt 0 ]; then return 10; elif [ "$n_warn" -gt 0 ]; then return 20; else return 0; fi
}

# ---- 入口 -------------------------------------------------------------------
MODE="${1:---loop}"
case "$MODE" in
  --once|once)
    poll_once; rc=$?
    exit $rc
    ;;
  --daemon|daemon)
    # 用 export 传参（不写进子进程 cmdline，避免 pgrep -f 自匹配模式串）
    export OUT LOGS INTERVAL STALL_SEC NO_PROC_GRACE START_GRACE DISK_MIN_GB DISK_PATH \
           TRAIN_PATTERN TRAIN_EXCLUDE CHAIN_PATTERN EVAL_PATTERN STAGE_EPOCHS ALERT_REPEAT_SEC
    setsid nohup bash "$0" --loop >> "$OUT/watch.out" 2>&1 < /dev/null &
    printf 'watchdog started pid=%s log=%s\n' "$!" "$OUT/watch.out"
    exit 0
    ;;
  --loop|loop)
    # flock 保证单实例
    exec 9>"$LOCK" 2>/dev/null || true
    if command -v flock >/dev/null 2>&1; then
      flock -n 9 || { printf '[watch] another instance holds %s, exit\n' "$LOCK" >&2; exit 0; }
    fi
    printf '[watch] %s start version=%s OUT=%s interval=%ss stall=%ss\n' "$(ts_iso)" "$VERSION" "$OUT" "$INTERVAL" "$STALL_SEC"
    trap 'printf "[watch] %s stop\n" "$(ts_iso)"; exit 0' INT TERM
    while :; do
      poll_once || true
      for _ in $(seq 1 "$INTERVAL"); do sleep 1; done
    done
    ;;
  -h|--help|help)
    sed -n '2,50p' "$0"
    exit 0
    ;;
  *)
    printf 'usage: %s [--daemon|--loop|--once|--help]\n' "$0" >&2
    exit 2
    ;;
esac
