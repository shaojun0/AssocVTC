#!/bin/bash
# Remote ssh with retry (the cloud box intermittently drops the TCP handshake).
# usage: bash _rssh.sh '<remote command>'
HOST="root@connect.sha1.seetacloud.com"
PORT=17677
CMD="$1"
for i in $(seq 1 12); do
  out=$(timeout 120 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
        -o BatchMode=yes -o ServerAliveInterval=15 -p $PORT $HOST "$CMD" 2>&1)
  rc=$?
  if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
    echo "$out"
    exit $rc
  fi
  if echo "$out" | grep -q "kex_exchange_identification\|Connection closed\|Connection refused\|Connection reset"; then
    sleep 6
    continue
  fi
  echo "$out"
  exit $rc
done
echo "[_rssh] gave up after 12 attempts"
exit 255
