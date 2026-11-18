#!/bin/bash
# 等 v10 正式训练（4000 步）进程退出 → 让显存释放 → 自动启动单片段过拟合对照。
# 用 nohup 挂在远端，避免 ssh 断线影响。
#   nohup bash wait_and_overfit.sh > logs/v10_overfit_driver.log 2>&1 &
set -u
LOG=/root/autodl-tmp/tok448_v2/logs
mkdir -p "$LOG"
while pgrep -f "train_v10[.]py" >/dev/null 2>&1; do sleep 20; done
sleep 30                                    # 等 MCCL 进程/显存完全释放
echo "[wait] $(date) 正式训练已退出，启动 overfit 对照" >> "$LOG/v10_overfit_driver.log"
cd /root/AssocVTC_v10
bash run_v10_overfit.sh
