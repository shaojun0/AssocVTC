#!/usr/bin/env bash
# =============================================================================
# fetch_n2bigstep_artifacts.sh — 把 n=2 × T=16/24/32 实验的产物拉回本地 artifacts
# -----------------------------------------------------------------------------
# 用法（仓库根目录）:
#   PORT=23556 bash tools/fetch_n2bigstep_artifacts.sh
# 环境变量: PORT（必填）/ HOST（默认 root@connect.nma1.seetacloud.com）/
#           SRC（服务器输出目录, 默认 /root/autodl-tmp/srdiff_n2bigstep）/
#           DST（默认 doc/2026-10-23/artifacts/n2bigstep）
# 产物: $DST/{eval/*.json, logs/*.log, args/*.json, train_logs/*.jsonl,
#             train_summary.csv, launch.out, n2bigstep.log}
# =============================================================================
set -euo pipefail
HOST=${HOST:-root@connect.nma1.seetacloud.com}
PORT=${PORT:?请设置 PORT（服务器 SSH 端口）}
SRC=${SRC:-/root/autodl-tmp/srdiff_n2bigstep}
DST=${DST:-doc/2026-10-23/artifacts/n2bigstep}
mkdir -p "$DST/eval" "$DST/logs" "$DST/args" "$DST/train_logs"

scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/eval/*.json" "$DST/eval/"
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/eval/*.log" "$DST/logs/" || true
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/k32_stage2.log" \
    "$HOST:$SRC/k48_stage2.log" "$HOST:$SRC/k64_stage2.log" \
    "$HOST:$SRC/n2bigstep.log" "$HOST:$SRC/launch.out" "$DST/logs/" || true
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/train_summary.csv" "$DST/" || true
for T in 16 24 32; do
  K=$((2 * T))
  scp -P "$PORT" -o StrictHostKeyChecking=no \
      "$HOST:$SRC/stdattn_k${K}_t${T}/args.json" "$DST/args/k${K}_t${T}_args.json" || true
  scp -P "$PORT" -o StrictHostKeyChecking=no \
      "$HOST:$SRC/stdattn_k${K}_t${T}/train_log.jsonl" "$DST/train_logs/k${K}_t${T}.jsonl" || true
  scp -P "$PORT" -o StrictHostKeyChecking=no \
      "$HOST:$SRC/stdattn_k${K}_t${T}/eval_val.json" "$DST/train_logs/k${K}_t${T}_eval_val.json" || true
done
echo "[fetch] done → $DST"
ls -la "$DST" "$DST/eval"
