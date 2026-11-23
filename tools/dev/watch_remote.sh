#!/bin/bash
ST="${1:?run dir}"
INTERVAL="${2:-300}"
while true; do
  out=$(timeout 45 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 17677 \
        root@connect.sha1.seetacloud.com "
          grep -E '\] step [0-9]+/' $ST/s2.log 2>/dev/null | tail -1 | cut -c1-150
          grep -E 'epoch .*val|val ' $ST/s2.log 2>/dev/null | tail -1 | cut -c1-150
          grep -oE '\[(driver|percft|ganft)\].*' $ST/status 2>/dev/null | tail -1
        " 2>&1 | tr '\n' '|')
  echo "[$(date +%H:%M:%S)] $out"
  case "$out" in *"DONE"*|*" rc="*) echo "WATCH_DONE"; break;; esac
  sleep "$INTERVAL"
done
