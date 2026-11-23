#!/bin/bash
# Remote-side idempotent launcher. Kept in a file so that `pgrep -f` patterns
# cannot match the invoking ssh command line (self-match bug).
set -u
mkdir -p /root/autodl-tmp/tok256x12_t8
cd /root/autodl-tmp/tok256x12_t8
if pgrep -f "train_linattn.py" >/dev/null 2>&1; then
  echo "ALREADY_TRAINING"; exit 0
fi
if [ -f status ] && grep -q "DONE" status 2>/dev/null; then
  echo "ALREADY_DONE"; exit 0
fi
chmod +x /root/autodl-tmp/run_k256_d12.sh
nohup setsid bash /root/autodl-tmp/run_k256_d12.sh > driver.out 2>&1 &
sleep 8
if pgrep -f "train_linattn.py" >/dev/null 2>&1; then
  echo "LAUNCHED_OK procs=$(pgrep -c -f train_linattn.py)"
else
  echo "LAUNCH_ATTEMPTED_NO_PROC"; tail -5 driver.out 2>/dev/null
fi
