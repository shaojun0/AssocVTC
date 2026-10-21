#!/usr/bin/env bash
# =============================================================================
# launch_k128_lr2e-4.sh — K=128 固定臂的 **lr=2e-4 诊断重跑**（2026-10-21）
# -----------------------------------------------------------------------------
# 背景: K=128 在 A 相 lr=1e-4 下（协议沿用 2026-10-19）落在「差盆地」（新机 26.17 /
#   旧机 25.65 dB）, 而 K=96（lr=2e-4）与 K=192（lr=1e-4）都能到 32 dB 量级。
#   本重跑只为回答「K=128 的差是 lr/优化问题还是预算/分配形状问题」:
#     · 若 2e-4 也能到 ~32 dB（A）/ ~29 dB（B）⇒ K=128 的异常是 lr 选择, 不是容量;
#     · 若同样停在 ~26 / 更低 ⇒ K=128 点本身不稳定, 支持「高 K 段优化脆弱」。
# 协议除 lr 外与主实验逐项相同（45ep/bs128 从零 → 6ep/bs64 fixed 窗口热启动）。
# 与主实验分开目录（不覆盖主实验 K=128 的产物）。
# 用法（服务器仓库根目录）:
#   nohup setsid bash tools/launch_k128_lr2e-4.sh > /root/autodl-tmp/srdiff_k128_lr2e-4/launch.out 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/root/autodl-tmp/srdiff_k128_lr2e-4}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
LOG=$OUT/k128_lr2e-4.log
mkdir -p "$OUT"

echo "[k128lr2e-4] 等待 GPU0 上 paired K=128 B 相退出 ... $(date -Is)" >> "$LOG"
while pgrep -f "num_specials 128[ ]" >/dev/null 2>&1; do sleep 60; done
echo "[k128lr2e-4] GPU 空闲, 开始 K=128 A+B (lr1=2e-4) $(date -Is)" >> "$LOG"

K=128 GPU=0 LR1=2e-4 LR2=2e-4 OUT="$OUT" DATA="$DATA" \
  bash tools/run_fixedk_arm.sh >> "$LOG" 2>&1
RC=$?
echo "[k128lr2e-4] rc=$RC end $(date -Is)" >> "$LOG"
if [ $RC -eq 0 ] && [ -f "$OUT/stdattn_k128_t8/ckpt_best.pt" ]; then
  touch "$OUT/K128LR2E4_DONE"
fi
echo "K128LR2E4 finished $(date -Is) rc=$RC" | tee -a "$LOG"
