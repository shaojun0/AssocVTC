#!/usr/bin/env bash
# vlm8b 监控邮件通知（本机 msmtp / QQ 邮箱，~/.msmtprc 已配好 qq 账号）
# 用法：
#   bash vlm8b_notify.sh "主题" "正文"
#   echo "正文" | bash vlm8b_notify.sh "主题"        # 正文走 stdin
#   TAG=异常 bash vlm8b_notify.sh "主题" "正文"      # 主题前缀 [vlm8b][异常]
#   TO=xxx@qq.com 可覆盖收件人（默认发给自己 2211057934@qq.com）
set -uo pipefail
TO=${TO:-2211057934@qq.com}
TAG=${TAG:-}
SUBJECT=${1:-"(无主题)"}
if [ "$#" -ge 2 ]; then BODY=$2; else BODY=$(cat); fi
SUBJ="[vlm8b]${TAG:+[$TAG]} $SUBJECT"
{
    printf 'From: %s\nTo: %s\nSubject: %s\nDate: %s\nMIME-Version: 1.0\n' "$TO" "$TO" "$SUBJ" "$(date -R)"
    printf 'Content-Type: text/plain; charset=UTF-8\nContent-Transfer-Encoding: 8bit\n\n'
    printf '%s\n\n' "$BODY"
    printf -- '---\nhost=%s time=%s\n' "$(hostname)" "$(date '+%F %T %Z')"
} | msmtp -a qq "$TO"
rc=$?
if [ "$rc" -eq 0 ]; then
    echo "[notify] OK: $SUBJ"
else
    echo "[notify] FAIL rc=$rc: $SUBJ" >&2
fi
exit "$rc"
