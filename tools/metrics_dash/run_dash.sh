#!/usr/bin/env bash
# =====================================================================================
# AssocVTC 训练指标看板 —— 一键在 GPU 服务器上拉起
#
#   6006 = TensorBoard   （事件文件由 emit_tensorboard.py 幂等重写）
#   6008 = Streamlit 交互式看板
#   后台刷新循环每 REFRESH_SECONDS 秒重跑 parse_metrics.py + emit_tensorboard.py
#
# 用法::
#
#   bash run_dash.sh start      # 启动 tb + 看板 + 刷新循环（默认）
#   bash run_dash.sh stop       # 停止三者
#   bash run_dash.sh restart
#   bash run_dash.sh status     # 进程 / 端口 / HTTP 自检
#   bash run_dash.sh refresh    # 只跑一次解析 + 写事件文件
#   bash run_dash.sh loop       # 前台跑刷新循环（由 start 在后台调用）
#   bash run_dash.sh logs       # tail 三个日志
#   bash run_dash.sh health     # 只做 HTTP 自检
#
# 安全约束（务必保持）：
#   * 本脚本只 **读** 训练日志与探针/推理 JSON；只 **写** 自己的 DATA_DIR、
#     TF_LOGDIR(/root/tf-logs-metrics) 与 LOG_DIR(/root/train_logs/dash_*.log)。
#   * 绝不触碰 /root/tf-logs（6007 端口既有 TensorBoard）、绝不触碰训练进程。
#   * 全部进程只用 CPU，不占用 GPU。
# =====================================================================================
set -uo pipefail

DASH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 可覆盖配置 -------------------------------------------------------------------
BASE_PY="${BASE_PY:-/root/miniconda3/bin/python}"          # 有 torch，用来写 TB 事件
VENV_PY="${VENV_PY:-/root/dashvenv/bin/python}"            # 有 streamlit/plotly
STREAMLIT_BIN="${STREAMLIT_BIN:-/root/dashvenv/bin/streamlit}"
DATA_DIR="${DATA_DIR:-$DASH_DIR/data}"
TF_LOGDIR="${TF_LOGDIR:-/root/tf-logs-metrics}"
RUN_DIR="${RUN_DIR:-$DASH_DIR/run}"
LOG_DIR="${LOG_DIR:-/root/train_logs}"
LOG_PREFIX="${LOG_PREFIX:-dash}"
TB_PORT="${TB_PORT:-6006}"
DASH_PORT="${DASH_PORT:-6008}"
REFRESH_SECONDS="${REFRESH_SECONDS:-45}"
TB_HOST="${TB_HOST:-0.0.0.0}"
DASH_HOST="${DASH_HOST:-0.0.0.0}"

export SRDASH_DATA="${SRDASH_DATA:-$DATA_DIR}"
export TF_LOGDIR

TB_PID="$RUN_DIR/tensorboard.pid"
DASH_PID="$RUN_DIR/streamlit.pid"
LOOP_PID="$RUN_DIR/loop.pid"
LOOP_TICK="$RUN_DIR/last_tick"
mkdir -p "$DATA_DIR" "$RUN_DIR" "$TF_LOGDIR" "$LOG_DIR"

TB_LOG="$LOG_DIR/${LOG_PREFIX}_tensorboard.log"
DASH_LOG="$LOG_DIR/${LOG_PREFIX}_streamlit.log"
LOOP_LOG="$LOG_DIR/${LOG_PREFIX}_loop.log"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- 小工具 -----------------------------------------------------------------------
pid_alive() { [ -n "${1:-}" ] && [ -f "${1:-}" ] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; }

port_listening() {  # port_listening <port>
    "$BASE_PY" - "$1" <<'PY' 2>/dev/null
import sys
port = int(sys.argv[1])
for f in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(f).read().splitlines()[1:]
    except OSError:
        continue
    for l in lines:
        p = l.split()
        if len(p) > 3 and p[3] == "0A" and int(p[1].split(":")[1], 16) == port:
            sys.exit(0)
sys.exit(1)
PY
}

http_code() { curl -s -o /dev/null -m 5 -w '%{http_code}' "$1" 2>/dev/null || echo "000"; }

wait_http() {  # wait_http <url> <label> <tries>
    local url="$1" label="$2" tries="${3:-30}" code
    for _ in $(seq 1 "$tries"); do
        code="$(http_code "$url")"
        if [ "$code" = "200" ]; then log "  ✅ $label → HTTP $code ($url)"; return 0; fi
        sleep 2
    done
    log "  ❌ $label → HTTP $code（超时）"
    return 1
}

stop_pidfile() {
    local f="$1" name="$2"
    if pid_alive "$f"; then
        local pid; pid="$(cat "$f")"
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null
        log "  停止 $name (pid=$pid)"
    fi
    rm -f "$f"
}

# ---- 核心：一次解析 + 写事件文件 ---------------------------------------------------
do_refresh() {
    local py="$VENV_PY"
    [ -x "$py" ] || py="$BASE_PY"
    [ -x "$py" ] || py="$(command -v python3 || command -v python)"

    "$py" "$DASH_DIR/parse_metrics.py" --out "$DATA_DIR" 2>&1 | sed 's/^/    /'
    local rc=${PIPESTATUS[0]}

    if [ -x "$BASE_PY" ] && "$BASE_PY" -c "import torch" 2>/dev/null; then
        "$BASE_PY" "$DASH_DIR/emit_tensorboard.py" \
            --metrics "$DATA_DIR/metrics.json" --logdir "$TF_LOGDIR" 2>&1 | sed 's/^/    /'
    else
        log "    (跳过 TensorBoard 事件写入：$BASE_PY 无 torch)"
    fi
    date +%s > "$LOOP_TICK"
    return "$rc"
}

# ---- 各服务 ------------------------------------------------------------------------
start_tensorboard() {
    if pid_alive "$TB_PID"; then log "TensorBoard 已在运行 (pid=$(cat "$TB_PID"))"; return 0; fi
    if port_listening "$TB_PORT"; then
        log "⚠️ 端口 $TB_PORT 已被占用，跳过 TensorBoard 启动（不动别人的进程）"; return 1
    fi
    log "启动 TensorBoard :$TB_PORT (logdir=$TF_LOGDIR)"
    setsid nohup "$BASE_PY" -m tensorboard.main \
        --host "$TB_HOST" --port "$TB_PORT" --logdir "$TF_LOGDIR" \
        --reload_interval 15 --samples_per_plugin scalars=1000000 \
        >> "$TB_LOG" 2>&1 < /dev/null &
    echo $! > "$TB_PID"
    log "  pid=$(cat "$TB_PID")  log=$TB_LOG"
}

start_dashboard() {
    if pid_alive "$DASH_PID"; then log "Streamlit 已在运行 (pid=$(cat "$DASH_PID"))"; return 0; fi
    if port_listening "$DASH_PORT"; then
        log "⚠️ 端口 $DASH_PORT 已被占用，跳过看板启动"; return 1
    fi
    if [ ! -x "$STREAMLIT_BIN" ]; then
        log "⚠️ 未找到 streamlit ($STREAMLIT_BIN) → 退化为静态 HTML + http.server"
        start_static_fallback; return $?
    fi
    log "启动 Streamlit :$DASH_PORT"
    setsid nohup "$STREAMLIT_BIN" run "$DASH_DIR/app_streamlit.py" \
        --server.address "$DASH_HOST" --server.port "$DASH_PORT" \
        --server.headless true --browser.gatherUsageStats false \
        --server.fileWatcherType none --server.runOnSave false \
        >> "$DASH_LOG" 2>&1 < /dev/null &
    echo $! > "$DASH_PID"
    log "  pid=$(cat "$DASH_PID")  log=$DASH_LOG"
}

# 兜底方案：生成静态看板并用 http.server 发布（streamlit 装不上时）
start_static_fallback() {
    local py="$VENV_PY"; [ -x "$py" ] || py="$BASE_PY"; [ -x "$py" ] || py="python3"
    "$py" "$DASH_DIR/make_static_html.py" --metrics "$DATA_DIR/metrics.json" \
        --out "$DATA_DIR/index.html" >> "$LOOP_LOG" 2>&1 || true
    setsid nohup "$py" -m http.server "$DASH_PORT" --bind "$DASH_HOST" \
        --directory "$DATA_DIR" >> "$DASH_LOG" 2>&1 < /dev/null &
    echo $! > "$DASH_PID"
    log "  静态兜底 pid=$(cat "$DASH_PID")  log=$DASH_LOG"
}

start_loop() {
    if pid_alive "$LOOP_PID"; then log "刷新循环已在运行 (pid=$(cat "$LOOP_PID"))"; return 0; fi
    log "启动刷新循环（每 ${REFRESH_SECONDS}s）"
    setsid nohup bash "$DASH_DIR/run_dash.sh" loop >> "$LOOP_LOG" 2>&1 < /dev/null &
    echo $! > "$LOOP_PID"
    log "  pid=$(cat "$LOOP_PID")  log=$LOOP_LOG"
}

run_loop() {
    log "=== 刷新循环启动 (pid=$$, 间隔 ${REFRESH_SECONDS}s) ==="
    while true; do
        log "--- refresh tick ---"
        do_refresh
        # 若正在使用静态兜底页面（index.html 存在），同步重新生成
        if [ -f "$DATA_DIR/index.html" ] && [ -f "$DASH_DIR/make_static_html.py" ]; then
            py="$VENV_PY"; [ -x "$py" ] || py="$BASE_PY"; [ -x "$py" ] || py="python3"
            "$py" "$DASH_DIR/make_static_html.py" \
                --metrics "$DATA_DIR/metrics.json" \
                --out "$DATA_DIR/index.html" >/dev/null 2>&1 \
                && log "    已重新生成静态 index.html"
        fi
        sleep "$REFRESH_SECONDS"
    done
}

# ---- 子命令 ------------------------------------------------------------------------
cmd_start() {
    log "===== 启动 AssocVTC 指标看板 ====="
    log "DASH_DIR=$DASH_DIR  DATA_DIR=$DATA_DIR  TF_LOGDIR=$TF_LOGDIR"
    log "[1/4] 首次解析 + 写 TensorBoard 事件…"
    do_refresh
    log "[2/4] TensorBoard :$TB_PORT"
    start_tensorboard
    log "[3/4] 交互看板 :$DASH_PORT"
    start_dashboard
    log "[4/4] 刷新循环"
    start_loop
    log "等待服务就绪…"
    wait_http "http://127.0.0.1:$TB_PORT" "TensorBoard" 30
    wait_http "http://127.0.0.1:$DASH_PORT" "Dashboard" 30
    cmd_status
}

cmd_stop() {
    log "===== 停止 ====="
    stop_pidfile "$LOOP_PID" "刷新循环"
    stop_pidfile "$DASH_PID" "看板"
    stop_pidfile "$TB_PID" "TensorBoard"
    log "完成"
}

cmd_status() {
    log "===== 状态 ====="
    for pair in "TensorBoard:$TB_PORT:$TB_PID" "Dashboard:$DASH_PORT:$DASH_PID" "Loop:-:$LOOP_PID"; do
        IFS=: read -r name port pidf <<< "$pair"
        if pid_alive "$pidf"; then
            log "  $name: RUNNING pid=$(cat "$pidf")${port:+ port=$port}"
        else
            log "  $name: stopped"
        fi
    done
    log "  监听检查: 6006=$(port_listening 6006 && echo yes || echo no) \
6008=$(port_listening 6008 && echo yes || echo no) \
6007=$(port_listening 6007 && echo yes || echo no)"
    log "  HTTP 自检:"
    log "    6006 → $(http_code http://127.0.0.1:$TB_PORT)"
    log "    6008 → $(http_code http://127.0.0.1:$DASH_PORT)"
    [ -f "$DATA_DIR/metrics.json" ] && \
        log "  metrics.json: $(stat -c '%s bytes, mtime %y' "$DATA_DIR/metrics.json" 2>/dev/null)"
    [ -f "$LOOP_TICK" ] && log "  上次刷新: $(date -d @"$(cat "$LOOP_TICK")" '+%F %T' 2>/dev/null)"
}

cmd_logs() {
    tail -n 40 "$LOOP_LOG" "$TB_LOG" "$DASH_LOG" 2>/dev/null
}

cmd_health() {
    wait_http "http://127.0.0.1:$TB_PORT" "TensorBoard" 5
    wait_http "http://127.0.0.1:$DASH_PORT" "Dashboard" 5
}

case "${1:-start}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; sleep 2; cmd_start ;;
    status)  cmd_status ;;
    refresh) do_refresh ;;
    loop)    run_loop ;;
    logs)    cmd_logs ;;
    health)  cmd_health ;;
    *) echo "用法: $0 {start|stop|restart|status|refresh|loop|logs|health}"; exit 2 ;;
esac
