#!/bin/bash
# v10 单片段过拟合对照（2026-11-18 用户要求）：循环(step) vs 并行(prefix)。
# 4 个独立单卡进程（MUSA 上不要多进程绑同一张卡），交错 15 s 启动。
# 用法（正式训练跑完后）：nohup bash run_v10_overfit.sh > /dev/null 2>&1 &
set -u
cd /root/AssocVTC_v10
export SDPA_BACKEND=math
export OMP_NUM_THREADS=6
LOG=/root/autodl-tmp/tok448_v2/logs
OUTROOT=/root/autodl-tmp/tok448_v2/overfit
STEPS=${STEPS:-800}
CLIP=${CLIP:-0}
INIT=${INIT:-/root/autodl-tmp/tok448_v2/v10_120f/best_trainable.pt}
mkdir -p "$LOG" "$OUTROOT"

run() {  # $1=GPU $2=名字 其余=overfit_v10.py 参数
  local g=$1 name=$2; shift 2
  MUSA_VISIBLE_DEVICES=$g nohup python overfit_v10.py \
    --out "$OUTROOT/$name" --steps "$STEPS" --clip "$CLIP" \
    "$@" > "$LOG/v10_overfit_$name.log" 2>&1 &
  echo "[launch] gpu$g $name pid $!"
  sleep 15
}

run 0 prefix      --arch prefix --copy_skip 1 --loss_read 1
run 1 step_nocopy --arch step   --copy_skip 0 --loss_read 1
run 2 step_copy   --arch step   --copy_skip 1 --loss_read 1
run 3 step_warm   --arch step   --copy_skip 1 --loss_read 1 --init_ckpt "$INIT"
wait
echo "[overfit] all done"
