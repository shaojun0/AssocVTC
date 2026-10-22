#!/usr/bin/env bash
# =============================================================================
# launch_retrain.sh — K=128/256 的「稳定性修复」重训（2026-10-22）
# -----------------------------------------------------------------------------
# 根因（§诊断）: K≥128 的 A 相落进「z_s 冗余坍缩」差盆地; 受控探针显示
#   **std 编码器加 QK-Norm（--enc_std_qk_norm 1）即可修复**:
#     同 lr=2e-4、同 16 ep 截断: 无 QK-Norm 停滞在 19–21 dB; 有 QK-Norm 前 14 ep 就到 28.7 dB。
#   （线性编码器默认就带 QK-Norm; std 编码器此前默认关 = "普通 transformer 原貌"。）
# 重训配置 = 原协议 + QK-Norm, lr=2e-4（与探针一致）; 其余（45ep/bs128 → 6ep/bs64、
#   fixed 8 步读窗口、seed 0）逐项不变。
# 顺序: K=128 A+B → K=256 A+B → K=128 配对 cumulative B（从新 A ckpt 出发, 只换读窗口）
# 产物: $OUT/stdattn_k{128,256}_t{1,8}; $PAIRED/stdattn_k128_t8; 标记 RETRAIN_DONE
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/launch_retrain.sh > $OUT/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/srdiff_qknorm}
PAIRED=${PAIRED:-/root/autodl-tmp/srdiff_qknorm_paired}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU=${GPU:-0}
QK_ARGS="--enc_std_qk_norm 1"
LOG=$OUT/retrain.log
mkdir -p "$OUT" "$PAIRED"
if [ -f "$OUT/RETRAIN_DONE" ]; then echo "[retrain] RETRAIN_DONE 已存在, 跳过"; exit 0; fi
: > "$LOG"
echo "[retrain] ==== start $(date -Is) OUT=$OUT PAIRED=$PAIRED" >> "$LOG"

echo "[retrain] K=128 A+B (qknorm, lr 2e-4) $(date -Is)" >> "$LOG"
K=128 GPU="$GPU" LR1=2e-4 LR2=2e-4 EXTRA_ARGS="$QK_ARGS" OUT="$OUT" DATA="$DATA" \
  bash tools/run_fixedk_arm.sh >> "$LOG" 2>&1
echo "[retrain] K=128 rc=$? $(date -Is)" >> "$LOG"

echo "[retrain] K=256 A+B (qknorm, lr 2e-4) $(date -Is)" >> "$LOG"
K=256 GPU="$GPU" LR1=2e-4 LR2=2e-4 EXTRA_ARGS="$QK_ARGS" OUT="$OUT" DATA="$DATA" \
  bash tools/run_fixedk_arm.sh >> "$LOG" 2>&1
echo "[retrain] K=256 rc=$? $(date -Is)" >> "$LOG"

# 配对 cumulative B 相: 从**新的** K=128 A 相 ckpt 出发, 只换读窗口（同 2026-10-21 配对设计）
mkdir -p "$PAIRED/stdattn_k128_t1"
ln -sfn "$OUT/stdattn_k128_t1/ckpt_best.pt" "$PAIRED/stdattn_k128_t1/ckpt_best.pt"
echo "[retrain] K=128 配对 cumulative B (qknorm) $(date -Is)" >> "$LOG"
K=128 GPU="$GPU" SKIP_STAGE1=1 LR2=2e-4 EXTRA_ARGS="$QK_ARGS" OUT="$PAIRED" DATA="$DATA" \
  bash tools/run_k8sweep_arm.sh >> "$LOG" 2>&1
echo "[retrain] K=128 paired rc=$? $(date -Is)" >> "$LOG"

touch "$OUT/RETRAIN_DONE"
echo "[retrain] RETRAIN_DONE $(date -Is)" | tee -a "$LOG"
