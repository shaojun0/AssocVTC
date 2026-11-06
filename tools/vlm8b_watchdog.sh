#!/usr/bin/env bash
# vlm8b 周末无人值守看门狗（本机常驻，每 INTERVAL 秒 ssh 巡检主训练机）
# ---------------------------------------------------------------------------
# 检测：进程消失 / 日志僵死（>45min）/ NaN 或 OOM / 磁盘 <4G / FAILED_P{N}
# 通知：异常与里程碑（切换课程、阶段完成、run2 完成、课程完成）发邮件给自己
# 自动处置（先斩后奏，仅课程阶段）：进程无标记地消失 → 自动重启 run_curriculum_vlm8b.sh
#            （脚本本身断点续跑；同一阶段最多自动重启 2 次，之后只发邮件等人工）
# 运行时抑制：touch /home/linaro/dsh/vlm8b_watchdog.pause 可让看门狗静默（agent 手动干预时用）
# 启动：setsid nohup bash vlm8b_watchdog.sh > /home/linaro/dsh/vlm8b_watchdog.log 2>&1 &
# 单次：ONCE=1 bash vlm8b_watchdog.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NOTIFY="$HERE/vlm8b_notify.sh"
SSH_TARGET=${SSH_TARGET:-connect.nma1.seetacloud.com}
SSH_PORT=${SSH_PORT:-23556}
SSH="ssh -p $SSH_PORT -o ConnectTimeout=15 -o BatchMode=yes root@$SSH_TARGET"
REMOTE_ROOT=/root/autodl-tmp/vlm8b/output
REMOTE_HOME=/root/vlm8b
INTERVAL=${INTERVAL:-600}
STATE=${STATE:-/home/linaro/dsh/vlm8b_watchdog_state.txt}
PAUSE=${PAUSE:-/home/linaro/dsh/vlm8b_watchdog.pause}
LOG=${LOG:-/home/linaro/dsh/vlm8b_watchdog.log}
STALE_SEC=${STALE_SEC:-2700}
FREE_MIN=${FREE_MIN:-2}   # 周末 run3 峰值只剩 ~3G（HF 先存后轮换），阈值降到 2G

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }
sent() { grep -qxF "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; }
mail() { local tag=$1 subj=$2 body=$3; TAG=$tag bash "$NOTIFY" "$subj" "$body" >> "$LOG" 2>&1; }

check_once() {
    local info
    info=$($SSH 'bash -s' <<REMOTE 2>/dev/null
ROOT=$REMOTE_ROOT
echo "DATE=\$(date -Is)"
echo "TRAIN=\$(pgrep -f 'torch.distributed.run.*train.py' | head -1)"
echo "CURSH=\$(pgrep -f 'run_curriculum_vlm8b.sh' | head -1)"
echo "EVAL=\$(pgrep -f 'eval.py' | head -1)"
echo "FREE=\$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')"
echo "GPU=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
[ -f "\$ROOT/run2/VLM8B_DONE" ] && echo "RUN2_DONE=1"
[ -f "\$ROOT/run3/VLM8B_DONE" ] && echo "RUN3_DONE=1"
[ -f "\$ROOT/cur/CURRICULUM_DONE" ] && echo "CUR_DONE=1"
for f in "\$ROOT"/cur/FAILED_P*; do [ -f "\$f" ] && echo "FAILED=\$(basename "\$f")"; done
for d in "\$ROOT"/cur/P*/; do [ -d "\$d" ] && echo "PHASE=\$(basename "\$d")"; done
for f in "\$ROOT"/cur/P*/VLM8B_DONE; do [ -f "\$f" ] && echo "PHASEDONE=\$(basename "\$(dirname "\$f")")"; done
LOG=\$(ls -t /root/vlm8b/logs/cur_P*.log /root/vlm8b/logs/run2.log /root/vlm8b/logs/run3.log 2>/dev/null | head -1)
echo "LOG=\$LOG"
if [ -n "\$LOG" ]; then
  echo "LOGAGE=\$(( \$(date +%s) - \$(stat -c %Y "\$LOG") ))"
  echo "LOGTAIL=\$(tail -2 "\$LOG" | tr -d '\r' | cut -c1-150 | tr '\n' '|')"
  echo "BADNAN=\$(grep -ciE 'nan|out of memory' "\$LOG" 2>/dev/null || echo 0)"
fi
REMOTE
)
    if [ -z "$info" ]; then
        log "[WARN] ssh 失败（已忽略本轮）"
        if ! sent "sshfail_$(date +%Y%m%d%H)"; then
            mark "sshfail_$(date +%Y%m%d%H)"
            mail 异常 "主训练机 SSH 不可达" "看门狗 $(date '+%F %T') 连不上 $SSH_TARGET。若持续，训练可能仍在跑（网络问题），待恢复后按日志确认。"
        fi
        return 0
    fi

    local TRAIN="" CURSH="" EVAL="" FREE="" GPU="" DATE="" RLOG="" RLOGAGE="0" RLOGTAIL="" BADNAN="0"
    local RUN2_DONE="" CUR_DONE="" RUN3_DONE=""
    local -a FAILED=() PHASE=() PHASEDONE=()
    while IFS='=' read -r k v; do
        case "$k" in
            DATE) DATE=$v ;;
            TRAIN) TRAIN=$v ;;
            CURSH) CURSH=$v ;;
            EVAL) EVAL=$v ;;
            FREE) FREE=$v ;;
            GPU) GPU=$v ;;
            LOG) RLOG=$v ;;
            LOGAGE) RLOGAGE=$v ;;
            LOGTAIL) RLOGTAIL=$v ;;
            BADNAN) BADNAN=$v ;;
            RUN2_DONE) RUN2_DONE=1 ;;
            RUN3_DONE) RUN3_DONE=1 ;;
            CUR_DONE) CUR_DONE=1 ;;
            FAILED) FAILED+=("$v") ;;
            PHASE) PHASE+=("$v") ;;
            PHASEDONE) PHASEDONE+=("$v") ;;
        esac
    done <<< "$info"

    local alive=0
    { [ -n "$TRAIN" ] || [ -n "$CURSH" ] || [ -n "$EVAL" ]; } && alive=1
    local mode="run2"; [ "${#PHASE[@]}" -gt 0 ] && mode="curriculum"
    log "mode=$mode alive=$alive train=${TRAIN:+y} cur=${CURSH:+y} eval=${EVAL:+y} free=${FREE}G gpu=[${GPU}] log=${RLOG##*/} age=${RLOGAGE}s"

    # 里程碑：切换课程 / run2 完成 / 阶段完成 / 课程完成
    if [ "$mode" = "curriculum" ] && ! sent "cur_started"; then
        mark "cur_started"
        mail 计划 "已切换到预算课程（P1 启动）" "时间：$DATE
课程：P1 冻结 k=8 2ep → P2 k={4,8} 1ep → P3 k={2,4,6,8} 1ep → P4 k={1..8} 1ep → P5 k={1..8}+解冻 2ep（LR 3e-5/3e-4/1e-5）
进度：阶段=${PHASE[*]}
GPU: $GPU  磁盘剩余: ${FREE}G
日志: $RLOG"
    fi
    if [ "$RUN2_DONE" = "1" ] && ! sent "run2_done"; then
        mark "run2_done"
        mail 里程碑 "run2 训练完成（2 epoch）" "时间：$DATE
产物：$REMOTE_ROOT/run2/final
下一步：三组消融 eval（normal/null/shuffle，--limit 250 --sample-seed 42）+ 通用 bench + 报告。"
    fi
    if [ "$RUN3_DONE" = "1" ] && ! sent "run3_done"; then
        mark "run3_done"
        mail 里程碑 "run3 训练完成（评测对齐混合集 3.8 epoch）" "时间：$DATE
产物：$REMOTE_ROOT/run3/final
自动评测链应已启动（mix_heldout 三组 + ImageNet val + 通用 bench），标记 logs/RUN3_EVAL_DONE。
下一步：写 run3 报告（预算消融是否在广任务上成立）并推 main。"
    fi
    local p
    for p in "${PHASEDONE[@]}"; do
        if ! sent "phase_done_$p"; then
            mark "phase_done_$p"
            mail 里程碑 "课程 $p 训练完成" "时间：$DATE
已完成阶段：${PHASEDONE[*]}
当前活动阶段：${PHASE[*]}
日志：$RLOG
quick eval 输出：$REMOTE_ROOT/cur/eval_$p/（日志 /root/vlm8b/logs/cur_eval_$p.log）"
        fi
    done
    if [ "$CUR_DONE" = "1" ] && ! sent "cur_done"; then
        mark "cur_done"
        mail 里程碑 "预算课程全部完成（5 阶段 / 7 epoch）" "时间：$DATE
产物：$REMOTE_ROOT/cur/P5/final
下一步：全量评测（三组消融 + 通用 bench）+ 报告。"
    fi

    # 异常：失败标记
    local f
    for f in "${FAILED[@]}"; do
        if ! sent "failed_$f"; then
            mark "failed_$f"
            mail 异常 "课程阶段失败：$f" "时间：$DATE
失败标记：$REMOTE_ROOT/cur/$f
日志尾部：$RLOGTAIL
脚本会按断点续跑，但课程已因该失败中止；看门狗将尝试自动重启一次（见下一条邮件）。"
        fi
    done
    # 异常：进程消失（无 DONE 标记且无脚本/eval 在跑）；连续 2 次巡检都死才动作（避免阶段切换间隙误报）
    local deadf="${STATE}.deadcount" dc
    dc=$(cat "$deadf" 2>/dev/null || echo 0)
    if [ "$alive" = "0" ] && [ "$CUR_DONE" = "0" ] && [ "$RUN2_DONE" != "1" ] && [ "$RUN3_DONE" != "1" ]; then dc=$((dc + 1)); else dc=0; fi
    echo "$dc" > "$deadf"
    if [ "$dc" -ge 2 ] && [ "$CUR_DONE" = "0" ] && [ "$RUN2_DONE" != "1" ] && [ "$RUN3_DONE" != "1" ]; then
        local bucket="dead_$(date +%Y%m%d%H)"
        if ! sent "$bucket"; then
            mark "$bucket"
            if [ "$mode" = "curriculum" ] && [ "${#FAILED[@]}" -eq 0 ]; then
                local tries=0
                for key in $(grep -oE "restart_cur_[0-9]+" "$STATE" 2>/dev/null | tail -1); do tries=${key##*_}; done
                if [ "${tries:-0}" -lt 2 ]; then
                    log "课程进程消失，自动重启（第 $((tries + 1)) 次）"
                    mark "restart_cur_$((tries + 1))"
                    $SSH "cd $REMOTE_HOME && nohup bash run_curriculum_vlm8b.sh >> logs/cur_main.log 2>&1 &" >/dev/null 2>&1
                    sleep 20
                    local after
                    after=$($SSH "pgrep -f 'run_curriculum_vlm8b.sh' | head -1" 2>/dev/null)
                    mail 异常 "课程进程消失 → 已自动重启（先斩后奏）" "时间：$DATE
检测：无训练进程、无失败标记、课程未完成；日志尾部：$RLOGTAIL
已执行：nohup bash run_curriculum_vlm8b.sh（断点续跑；第 $((tries + 1))/2 次自动重启）
重启后进程：${after:-未检测到（可能又挂了）}
请稍后查看 /root/vlm8b/logs/cur_main.log 与当前阶段日志。"
                else
                    mail 异常 "课程进程反复消失（已自动重启 2 次，停手等人工）" "时间：$DATE
日志尾部：$RLOGTAIL
请人工检查 /root/vlm8b/logs/cur_*.log。"
                fi
            else
                mail 异常 "训练进程消失（未完成，且非课程自动重启场景）" "时间：$DATE
模式：$mode；训练=${TRAIN:-无}；脚本=${CURSH:-无}；eval=${EVAL:-无}
marker：run2_done=${RUN2_DONE:-0} cur_done=${CUR_DONE:-0} failed=${FAILED[*]:-无}
日志尾部：$RLOGTAIL
看门狗不会自动重启该模式，等待定时巡检（每 2h）处置。"
            fi
        fi
    fi
    # 异常：日志僵死
    if [ "$alive" = "1" ] && [ "${RLOGAGE:-0}" -gt "$STALE_SEC" ] && ! sent "stale_$(date +%Y%m%d%H)"; then
        mark "stale_$(date +%Y%m%d%H)"
        mail 异常 "训练日志 ${RLOGAGE}s 无更新（可能僵死）" "时间：$DATE
进程：train=${TRAIN:+有} script=${CURSH:+有} eval=${EVAL:+有}；日志：$RLOG
尾部：$RLOGTAIL"
    fi
    # 异常：磁盘
    if [ "${FREE:-999}" -lt "${FREE_MIN:-4}" ] && ! sent "disk_$(date +%Y%m%d%H)"; then
        mark "disk_$(date +%Y%m%d%H)"
        mail 异常 "磁盘剩余 ${FREE}G（<${FREE_MIN}G）" "时间：$DATE
请尽快清理；看门狗不会自动删除训练产物。"
    fi
    # 异常：NaN/OOM 关键字
    if [ "${BADNAN:-0}" -gt 0 ] && ! sent "badnan_${BADNAN}"; then
        mark "badnan_${BADNAN}"
        mail 异常 "日志出现 NaN/OOM 关键字（${BADNAN} 处）" "时间：$DATE
日志：$RLOG
尾部：$RLOGTAIL"
    fi
    return 0
}

mkdir -p "$(dirname "$LOG")" "$(dirname "$STATE")"
touch "$STATE"
log "watchdog start pid=$$ interval=${INTERVAL}s target=$SSH_TARGET:$SSH_PORT"
if [ "${ONCE:-0}" = "1" ]; then
    [ -f "$PAUSE" ] && { log "pause 文件存在，跳过本轮"; exit 0; }
    check_once
    exit 0
fi
while true; do
    if [ -f "$PAUSE" ]; then
        log "pause 文件存在，跳过本轮"
    else
        check_once || log "[WARN] check_once 异常（继续）"
    fi
    sleep "$INTERVAL"
done
