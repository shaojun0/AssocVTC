#!/usr/bin/env bash
# AssocVTC 消融臂 (stdattn): 编码器 = **普通 transformer** (softmax 注意力)
# =============================================================================
# 2026-10-16 用户口径: 「把编码器换成普通的 transformer, 跑 BPTT」
#   相对昨日 linattn 臂 (run_linattn.sh) 的**唯一自变量 = 编码器注意力形式**:
#       线性核注意力 (elu+1, O(N), +DWConv, +QK-Norm)   →   标准 softmax 缩放点积
#   编码器参数量 31.909M → 31.848M (−0.19%), 解码器/K/数据/超参全部不变。
#
# 用法:
#   # 阶段 1: 单步并行解码, 与昨日 k64 臂逐项同协议 (bs128 / 45ep / lr2e-4)
#   TAG=stdattn_k64 EPOCHS=45 BS=128 TIME_BUDGET=5.5 ./run_stdattn.sh
#   # 阶段 2: 从阶段 1 热启动, 循环读出 (BPTT), 与昨日 k64_rec 逐项同协议 (bs64 / 6ep)
#   TAG=stdattn_k64_rec EPOCHS=6 BS=64 TIME_BUDGET=2.0 \
#       INIT_FROM=/root/autodl-tmp/srdiff_stdattn/stdattn_k64/ckpt_best.pt \
#       DEC_RECURRENT=1 ./run_stdattn.sh
#
# 环境变量: K BS EPOCHS TAG TIME_BUDGET LR INIT_FROM DEC_RECURRENT DATA OUT PY
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/root/miniconda3/bin/python}
K=${K:-64}
BS=${BS:-128}
EPOCHS=${EPOCHS:-45}
LR=${LR:-2e-4}
TAG=${TAG:-stdattn_k${K}}
TIME_BUDGET=${TIME_BUDGET:-5.5}
INIT_FROM=${INIT_FROM:-}
DEC_RECURRENT=${DEC_RECURRENT:-0}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/srdiff_stdattn/${TAG}}
mkdir -p "$OUT"
echo "[run_stdattn] K=$K bs=$BS epochs=$EPOCHS lr=$LR rec=$DEC_RECURRENT out=$OUT"
EXTRA=()
[ -n "$INIT_FROM" ] && EXTRA+=(--init_from "$INIT_FROM")
exec "$PY" -u train_linattn.py \
  --data_dir "$DATA" --output_dir "$OUT" \
  --enc_attn std --num_specials "$K" --batch_size "$BS" --epochs "$EPOCHS" \
  --lr "$LR" --dec_recurrent "$DEC_RECURRENT" \
  --time_budget_h "$TIME_BUDGET" "${EXTRA[@]}" "$@"
