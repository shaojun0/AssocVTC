#!/usr/bin/env bash
# 平台期处置：停当前训练 → 清理空闲 → 以更大 lr 重启（nohup 后台）
# 用法: bash restart_lr.sh            # 默认 LM 1e-5→3e-5, 2 epoch
set -uo pipefail
cd /root/vlm8b
if pgrep -af "torch.distributed.run.*train.py" >/dev/null; then
  echo "[restart] 停掉当前训练: $(pgrep -f torch.distributed.run | tr n  )"
  pkill -f "torch.distributed.run"; sleep 25
  if pgrep -f "torch.distributed.run" >/dev/null; then
    echo "[restart] 残留进程，强杀"; pkill -9 -f "torch.distributed.run"; sleep 10
  fi
fi
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "[restart] 启动 run3（lr=$LR_LLM）$(date -Is)"
nohup bash run_cont3.sh >/dev/null 2>&1 &
echo "[restart] pid=$!"
