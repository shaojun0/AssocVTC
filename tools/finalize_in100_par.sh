#!/bin/bash
# in100「并行读出 vs 循环读出」对照的收尾：等跑完 → 拉结果 → 出报告 → 推 GitHub → 关机。
#
# 设计要点
#   * 幂等：done 标记存在且报告已推送时，只补做关机，不重复提交。
#   * 不在训练中途打扰：status.running=true 且日志仍在更新 → 打印 PENDING 退出 0。
#   * 只在报告**推送成功**之后才关机，避免"关机了但结果没入库"。
#
# 用法：bash tools/finalize_in100_par.sh            # 由定时监控调用
#       FORCE_SHUTDOWN=0 bash tools/finalize_in100_par.sh   # 只出报告不关机
set -u

HOST=root@connect.nma1.seetacloud.com
PORT=23556
SSH="ssh -p ${PORT} -o ConnectTimeout=10 -o BatchMode=yes ${HOST}"
SCP="scp -P ${PORT} -o ConnectTimeout=10 -o BatchMode=yes"
REPO=/home/linaro/dsh/sr-diffusion-v3
DAY=doc/2026-11-16
ART=${REPO}/${DAY}/artifacts/in100_par
MARK=${REPO}/${DAY}/.finalize_in100_par.done
LOG=${REPO}/${DAY}/.finalize_in100_par.log
FORCE_SHUTDOWN=${FORCE_SHUTDOWN:-1}

say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

mkdir -p "${ART}/baseline" "${ART}/par" "$(dirname "${LOG}")"

# ── 1. 读状态（连不上就退出，等下一次）
STATUS=$(timeout 25 ${SSH} 'cat /root/autodl-tmp/run_v2in100_par/status.json' 2>/dev/null) || {
  say "SSH 不可达（实例可能已关机/换端口）"; exit 3; }

READY=$(STATUS="${STATUS}" python3 - <<'PY'
import json, os
try:
    s = json.loads(os.environ["STATUS"])
except Exception:
    print("BAD"); raise SystemExit
print("DONE" if (s.get("done") or s.get("running") is False) else "PENDING")
PY
)
case "${READY}" in
  DONE) : ;;
  PENDING) say "训练仍在进行：$(echo "${STATUS}" | tr -d '\n' | head -c 200)"; exit 0 ;;
  *) say "status.json 解析失败：$(echo "${STATUS}" | head -c 200)"; exit 2 ;;
esac

# ── 2. 已在库里 → 只补关机
if [ -f "${MARK}" ] && git -C "${REPO}" log --oneline -1 -- "${DAY}/REPORT_in100_parallel_vs_recurrent.md" | grep -q .; then
  say "报告已入库，跳过出报告；$( [ "${FORCE_SHUTDOWN}" = "1" ] && echo '执行关机' || echo '不关机' )"
  if [ "${FORCE_SHUTDOWN}" = "1" ]; then
    timeout 20 ${SSH} 'sync; (setsid nohup shutdown >/dev/null 2>&1 </dev/null &) ; echo shutdown-issued' \
      && say "关机指令已发出（AutoDL 容器内 shutdown 会连带 kill sshd，连接断开属正常）" \
      || say "关机指令发送失败（需到控制台确认）"
  fi
  exit 0
fi

# ── 3. 拉结果
say "训练完成，开始拉取产物"
for arm in baseline par; do
  d=/root/autodl-tmp/run_v2in100
  [ "${arm}" = "par" ] && d=/root/autodl-tmp/run_v2in100_par
  for f in eval_val.json status.json args.json train_log.jsonl; do
    timeout 40 ${SCP} -q "${HOST}:${d}/${f}" "${ART}/${arm}/${f}" 2>/dev/null \
      || say "  缺 ${arm}/${f}（继续）"
  done
  say "  ${arm}: $(ls "${ART}/${arm}" | tr '\n' ' ')"
done

# ── 4. 出报告
python3 "${REPO}/tools/make_report_in100_par.py" 2>&1 | tee -a "${LOG}" || {
  say "报告生成失败"; exit 4; }

# ── 5. 提交 + 推送（push 成功才算数）
say "提交并推送报告"
cd "${REPO}" || exit 5
git add "${DAY}/REPORT_in100_parallel_vs_recurrent.md" "${DAY}/artifacts/in100_par" tools/make_report_in100_par.py tools/finalize_in100_par.sh
git -c user.name="assocvtc" -c user.email="assocvtc@local" \
    commit -q -m "doc(2026-11-16): in100 并行读出 vs 循环读出对照报告（同优化步数 16509）

- 两臂严格同数据/同 seed/同 lr 调度（total_steps=28500），并行臂 --stop_at_step 16509
- 报告含协议表、PSNR/L1/SSIM 对照、吞吐、边界，以及对 v10 读出选择的含义
- 原始产物 doc/2026-11-16/artifacts/in100_par/{baseline,par}/
- 收尾脚本 tools/finalize_in100_par.sh（幂等：报告推送成功后才关机）" || true

if ! git push -q origin main 2>&1 | tee -a "${LOG}"; then
  say "!! push 失败，不关机（报告留在本地，下次重试）"; exit 6
fi
say "已推送：$(git -C "${REPO}" log --oneline -1)"
touch "${MARK}"

# ── 6. 关机（AutoDL 容器内 shutdown 会 kill supervisord ⇒ sshd 一起没了，属正常）
if [ "${FORCE_SHUTDOWN}" = "1" ]; then
  timeout 20 ${SSH} 'sync; (setsid nohup shutdown >/dev/null 2>&1 </dev/null &) ; echo shutdown-issued' \
    && say "关机指令已发出" || say "关机指令发送失败（需到控制台确认）"
else
  say "FORCE_SHUTDOWN=0：跳过关机"
fi
exit 0
