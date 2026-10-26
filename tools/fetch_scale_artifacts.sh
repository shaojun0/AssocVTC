#!/usr/bin/env bash
# =============================================================================
# fetch_scale_artifacts.sh — 把「容量阶梯」实验产物拉回本地 artifacts（2026-10-24）
# -----------------------------------------------------------------------------
# 用法（仓库根目录）:
#   PORT=23556 bash tools/fetch_scale_artifacts.sh
# 环境变量: PORT（必填）/ HOST（默认 root@connect.nma1.seetacloud.com）/
#           SRC（服务器输出目录, 默认 /root/autodl-tmp/srdiff_scale）/
#           DST（默认 doc/2026-10-24/artifacts/scale）
# 产物: $DST/{eval/*.json, logs/*.log, args/*.json, train_logs/*.jsonl,
#             train_summary.csv, launch.out, scale.log}
# =============================================================================
set -euo pipefail
HOST=${HOST:-root@connect.nma1.seetacloud.com}
PORT=${PORT:?请设置 PORT（服务器 SSH 端口）}
SRC=${SRC:-/root/autodl-tmp/srdiff_scale}
DST=${DST:-doc/2026-10-24/artifacts/scale}
mkdir -p "$DST/eval" "$DST/logs" "$DST/args" "$DST/train_logs"

# ⚠️ 只取本实验的 4 档，别用 eval/*.json 通配：该目录会被其它并行作业写入（如 *_zs448）
for M in 60 120 240 360; do
  for T in 1 16; do
    scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/eval/linattn_m${M}_t${T}.json" "$DST/eval/" || true
    scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/eval/stepwise_m${M}_t16.json" "$DST/eval/" || true
    scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/eval/linattn_m${M}_t${T}.log" "$DST/logs/" || true
    scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/eval/stepwise_m${M}_t16.log" "$DST/logs/" || true
  done
done
for M in 60 120 240 360; do
  scp -P "$PORT" -o StrictHostKeyChecking=no \
      "$HOST:$SRC/m${M}_stage1.log" "$HOST:$SRC/m${M}_stage2.log" "$DST/logs/" || true
done
scp -P "$PORT" -o StrictHostKeyChecking=no \
    "$HOST:$SRC/scale.log" "$HOST:$SRC/launch.out" "$HOST:$SRC/bench_scale/bench_summary.txt" \
    "$DST/logs/" || true
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC/train_summary.csv" "$DST/" || true
for M in 60 120 240 360; do
  for T in 1 16; do
    scp -P "$PORT" -o StrictHostKeyChecking=no \
        "$HOST:$SRC/stdattn_m${M}_t${T}/args.json" "$DST/args/m${M}_t${T}_args.json" || true
    scp -P "$PORT" -o StrictHostKeyChecking=no \
        "$HOST:$SRC/stdattn_m${M}_t${T}/eval_val.json" "$DST/train_logs/m${M}_t${T}_eval_val.json" || true
    scp -P "$PORT" -o StrictHostKeyChecking=no \
        "$HOST:$SRC/stdattn_m${M}_t${T}/train_log.jsonl" "$DST/train_logs/m${M}_t${T}.jsonl" || true
  done
done
# ── 条件臂 n=2 × T=24 @ 最优容量档（独立 OUT 目录, 存在才拉）──────────────
SRC24=${SRC24:-/root/autodl-tmp/srdiff_scale_t24}
# 条件臂单独目录 eval_t24/：其 linattn_m240_t1.json 是 K=48 的 A 相，不能与阶梯臂的 K=32 混名
mkdir -p "$DST/eval_t24"
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC24/eval/*.json" "$DST/eval_t24/" || true
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC24/eval/*.log" "$DST/logs/" || true
scp -P "$PORT" -o StrictHostKeyChecking=no "$HOST:$SRC24"/m240_stage*.log \
    "$HOST:$SRC24/launch.out" "$DST/logs/" || true
for T in 1 24; do
  scp -P "$PORT" -o StrictHostKeyChecking=no \
      "$HOST:$SRC24/stdattn_m240_t${T}/args.json" "$DST/args/m240_t${T}_args.json" || true
  scp -P "$PORT" -o StrictHostKeyChecking=no \
      "$HOST:$SRC24/stdattn_m240_t${T}/train_log.jsonl" "$DST/train_logs/m240_t${T}.jsonl" || true
  scp -P "$PORT" -o StrictHostKeyChecking=no \
      "$HOST:$SRC24/stdattn_m240_t${T}/eval_val.json" "$DST/train_logs/m240_t${T}_eval_val.json" || true
done
echo "[fetch] done → $DST"
ls -la "$DST" "$DST/eval"
