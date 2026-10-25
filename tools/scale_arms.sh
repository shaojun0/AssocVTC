#!/usr/bin/env bash
# =============================================================================
# scale_arms.sh — 「容量阶梯」4 档的架构表（被 bench/launch/eval 脚本 source）
# -----------------------------------------------------------------------------
# 用户口径（2026-10-24）: 固定 n=2 / T=16（K=32），把 **编码器与解码器参数量各自**
#   抬到 60M → 120M → 240M → 360M。scaling 方案 = **深度钉死基线（enc 13 / dec 7）、
#   只加宽 dim**（单一容量轴；BPTT 层数不变 ⇒ 显存随宽度线性而非 depth×T 双乘）。
# 选型由 tools/scale_ladder.py 精确计参校准，基线 63M = enc 448x13 h8 (31.83M) +
#   dec 512x7 h8 (31.19M)。
# 用法:  source tools/scale_arms.sh ; scale_arm_cfg 240   # → 设置 ENC_DIM/…
# =============================================================================

# 每档: "<enc_dim> <enc_heads> <dec_dim> <dec_heads> <实测 enc M> <实测 dec M> <accumA> <accumB>"
#   accum 只改**执行路径**（micro-batch 累积），等效批量 A=128 / B=64 与优化步数都不变；
#   大档显存不够时才升 accum（实测见 doc/2026-10-24 报告 §bench）。
SCALE_ARMS="
60  624  12  704   8   61.47   57.85  1  2
120 864  16  1024  16  117.46  120.74 1  2
240 1248 16  1440  16  244.43  237.11 1  2
360 1504 16  1792  16  354.63  365.99 1  4
"

scale_arm_cfg() {
  local want=$1 found="" line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    set -- $line
    if [ "$1" = "$want" ]; then found="$line"; break; fi
  done <<< "$SCALE_ARMS"
  if [ -z "$found" ]; then echo "[scale] 未知档位 $want" >&2; return 1; fi
  set -- $found
  SCALE_M=$1; ENC_DIM=$2; ENC_HEADS=$3; DEC_DIM=$4; DEC_HEADS=$5
  ENC_PARAM_M=$6; DEC_PARAM_M=$7
  SCALE_ACCUM_A=$8; SCALE_ACCUM_B=$9
  # 全局强改（例如 T=24 条件臂显存更高, 需 FORCE_ACCUM_B=4）
  if [ -n "${FORCE_ACCUM_A:-}" ]; then SCALE_ACCUM_A=$FORCE_ACCUM_A; fi
  if [ -n "${FORCE_ACCUM_B:-}" ]; then SCALE_ACCUM_B=$FORCE_ACCUM_B; fi
  # 目录/日志标签（与历史 srdiff_* 一致风格）
  SCALE_TAG="m${SCALE_M}"
}

# 公共协议（与 2026-10-23 n=2 × T=16 的 63M 基线臂逐项一致）
SCALE_K=${SCALE_K:-32}        # K = 2·T
SCALE_T=${SCALE_T:-16}
SCALE_BLOCK=${SCALE_BLOCK:-2}
SCALE_LR1=${SCALE_LR1:-2e-4}
SCALE_LR2=${SCALE_LR2:-2e-4}
SCALE_EP1=${SCALE_EP1:-45}
SCALE_EP2=${SCALE_EP2:-6}
