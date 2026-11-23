#!/bin/bash
# Retry launcher: keeps trying to start the 256x12 run until it is confirmed running.
# Guards against (a) ssh relay throttling, (b) double launch.
set -u
for i in $(seq 1 40); do
  out=$(timeout 60 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p 17677 \
        root@connect.sha1.seetacloud.com '
set -u
mkdir -p /root/autodl-tmp/tok256x12_t8
if pgrep -f "run_k256_d12.sh" >/dev/null 2>&1; then echo "ALREADY_RUNNING"; exit 0; fi
if pgrep -f "train_linattn.py" >/dev/null 2>&1; then echo "OTHER_TRAIN_RUNNING"; exit 0; fi
chmod +x /root/autodl-tmp/run_k256_d12.sh
cd /root/autodl-tmp/tok256x12_t8
nohup setsid bash /root/autodl-tmp/run_k256_d12.sh > driver.out 2>&1 &
sleep 5
echo "LAUNCHED"; pgrep -c -f "run_k256_d12.sh"' 2>&1)
  echo "[$(date +%H:%M:%S)] try $i -> $out"
  case "$out" in
    *LAUNCHED*|*ALREADY_RUNNING*|*OTHER_TRAIN_RUNNING*) echo "LAUNCH_OK"; exit 0;;
  esac
  sleep 120
done
echo "GAVE_UP"
