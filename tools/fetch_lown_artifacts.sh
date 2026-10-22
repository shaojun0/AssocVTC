#!/usr/bin/env bash
# =============================================================================
# fetch_lown_artifacts.sh — 把 n=2/4/6 实验（K=16/32/48）的产物拉回本地 artifacts
# -----------------------------------------------------------------------------
# 用法（仓库根目录）:
#   PORT=12345 bash tools/fetch_lown_artifacts.sh
# 环境变量: PORT（必填, 服务器新端口）/ HOST（默认 connect.bjb1.seetacloud.com）/
#           SRC（服务器侧输出目录, 默认 /root/autodl-tmp/srdiff_lown）
# 产物: doc/2026-10-23/artifacts/lown/{eval/*.json, logs/*.log, args/*.json, train_summary.csv}
# =============================================================================
set -euo pipefail
HOST=${HOST:-root@connect.bjb1.seetacloud.com}
PORT=${PORT:?请设置 PORT（服务器 SSH 端口）}
SRC=${SRC:-/root/autodl-tmp/srdiff_lown}
DST=doc/2026-10-23/artifacts/lown
mkdir -p "$DST/eval" "$DST/logs" "$DST/args"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/eval/*.json" "$DST/eval/"
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/k16_stage1.log" "$HOST:$SRC/k16_stage2.log" \
    "$HOST:$SRC/k32_stage1.log" "$HOST:$SRC/k32_stage2.log" \
    "$HOST:$SRC/k48_stage1.log" "$HOST:$SRC/k48_stage2.log" "$DST/logs/" || true
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/train_summary.csv" "$DST/" || true
for K in 16 32 48; do
  scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/stdattn_k${K}_t8/args.json" \
      "$DST/args/k${K}_t8_args.json" || true
done
echo "[fetch] done → $DST"
ls -la "$DST" "$DST/eval"
