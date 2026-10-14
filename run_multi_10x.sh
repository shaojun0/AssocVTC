#!/usr/bin/env bash
# ============================================================================
# run_multi_10x.sh — MultiResSR「解码器参数量 ×10」训练入口（2026-10-14）
#
# 设计原则（用户 2026-10-14：要通用、支持 DCU/NPU 等其它机器）:
#   · 零硬编码路径：全部可被环境变量覆盖（DATA_ROOT / DINO_DIR / OUTPUT_DIR）
#   · 零设备假设：DEVICE_TYPE=auto 时由 srdiff_runtime 探测
#     （NVIDIA CUDA / 海光 DCU(ROCm) / 昇腾 NPU / Intel XPU / CPU）
#   · 启动器可选：LAUNCHER=accelerate（默认）| torchrun | python
#     —— NPU/DCU 集群常见 torchrun，CUDA 单机 accelerate 更省事
#
# 本轮唯一的实验变量 = 解码器宽度/FFN（depth 保持 2 不变）:
#     基线 2026-09-26 : depth=2 width=1024(=dim) heads=8  mlp_ratio=4.0
#                       → 解码器 140.9 M（全模型 445.3 M）
#     本轮            : depth=2 width=3072      heads=24 mlp_ratio=5.12
#                       → 解码器 1409.3 M = **×10.001**（全模型 1695.5 M）
#   验算（不需要 GPU）: python tools/count_params.py --depth 2 --width 3072 \
#                        --heads 24 --mlp_ratio 5.12
#   数据/K/步数/损失/优化器配方与 2026-09-26 六源 run 完全一致，保证可比。
#
# 用法:
#   bash run_multi_10x.sh                      # 2 卡 A800 全量 1 epoch
#   NUM_GPUS=1 bash run_multi_10x.sh --smoke   # 单卡冒烟
#   NUM_GPUS=8 LAUNCHER=torchrun DEVICE_TYPE=npu bash run_multi_10x.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS="${NUM_GPUS:-2}"
LAUNCHER="${LAUNCHER:-accelerate}"
DEVICE_TYPE="${DEVICE_TYPE:-auto}"
DATA_ROOT="${DATA_ROOT:-${SRDIFF_DATA_ROOT:-/root/autodl-tmp/data}}"
DINO_DIR="${DINO_DIR:-${SRDIFF_DINO_DIR:-/root/autodl-tmp/models/dinov2-large}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SRDIFF_OUTPUT_DIR:-/root/autodl-tmp/output/multires_10x}}"
CACHE="${CACHE:-$DATA_ROOT/index_cache.json}"
MASTER_PORT="${MASTER_PORT:-29550}"

# 解码器 ×10（唯一实验变量）
DEC_W="${DEC_W:-3072}"
DEC_D="${DEC_D:-2}"
DEC_H="${DEC_H:-24}"
DEC_MLP="${DEC_MLP:-5.12}"

# 与 2026-09-26 六源 run 逐位一致的训练配方
BS="${BS:-6}"                    # 每卡 batch 上限（**上限**，不是实际值！见 BUDGET）
BUDGET="${BUDGET:-6144}"         # 每批编码器 patch 数上限：实际 bs=min(BS, BUDGET/N)
GRAD_ACCUM="${GRAD_ACCUM:-1}"    # 梯度累积（不增显存，只增有效 batch）
PRECISION="${PRECISION:-bf16}"
SDPA_BACKEND="${SDPA_BACKEND:-auto}"   # DCU 上融合后端崩了就用 math / math+efficient
ATTN_IMPL="${ATTN_IMPL:-auto}"         # 只管 HF DINOv2 编码器
GRAD_CKPT_DECODER="${GRAD_CKPT_DECODER:-1}"  # 0 = 关掉解码器重算（省 ~25% 算力，吃显存）
GRAD_CKPT_ENCODER="${GRAD_CKPT_ENCODER:-1}"  # 0 = 关掉编码器梯度检查点（吃显存）
EPOCHS="${EPOCHS:-2}"          # 用户 2026-10-14 拍板：本轮训 2 个 epoch
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_NEW="${LR_NEW:-3e-4}"
LR_DINO="${LR_DINO:-1.5e-4}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
KEEP_CKPTS="${KEEP_CKPTS:-4}"     # 单 ckpt ≈20 GB（含优化器）⇒ 默认只留最近 4 个
LOG_EVERY="${LOG_EVERY:-20}"

# 解释器可配（有些集群只有 python3，没有 python）
PYTHON="${PYTHON:-$(command -v python || command -v python3 || echo python)}"

LOG="${LOG:-$OUTPUT_DIR/launch.log}"
mkdir -p "$OUTPUT_DIR"

# torchrun / accelerate 都需要可见卡的环境变量；CPU 或已被框架接管时留空
if [ "$DEVICE_TYPE" = "auto" ] || [ "$DEVICE_TYPE" = "cuda" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
elif [ "$DEVICE_TYPE" = "npu" ]; then
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
fi

# max_steps = 每 epoch 优化步数（每卡）× EPOCHS；可用 STEPS 直接覆盖
if [ "${STEPS:-0}" -eq 0 ] 2>/dev/null; then
  PER_EPOCH=$("$PYTHON" epoch_steps.py "$CACHE" "$BS" "$BUDGET" "$NUM_GPUS" 2>/dev/null | awk '{print $4}')
  [ -n "${PER_EPOCH:-}" ] || { echo "!! 算不出 max_steps（索引 $CACHE 还没建好？先跑 tools/prep_data.py）" >&2; exit 1; }
  STEPS=$((PER_EPOCH * EPOCHS))
  echo "[launch] 每 epoch $PER_EPOCH 步/卡 × $EPOCHS epoch = $STEPS 步/卡"
fi
[ -n "${STEPS:-}" ] || { echo "!! 算不出 max_steps" >&2; exit 1; }

ARGS=(train_multi.py
  --data_root "$DATA_ROOT" --dino_dir "$DINO_DIR" --output_dir "$OUTPUT_DIR"
  --data_cache "$CACHE" --device_type "$DEVICE_TYPE" --precision "$PRECISION"
  --sdpa_backend "$SDPA_BACKEND" --attn_implementation "$ATTN_IMPL"
  --batch_size "$BS" --patch_budget "$BUDGET" --grad_accum "$GRAD_ACCUM"
  --decoder_width "$DEC_W" --decoder_depth "$DEC_D" --heads "$DEC_H"
  --decoder_mlp_ratio "$DEC_MLP"
  --lr_new "$LR_NEW" --lr_dino "$LR_DINO" --weight_decay 0.01
  --warmup_ratio "$WARMUP_RATIO" --grad_clip 1.0
  --num_specials 144 --step_plan square --num_workers "$NUM_WORKERS"
  --log_every "$LOG_EVERY" --save_every "$SAVE_EVERY" --keep_ckpts "$KEEP_CKPTS"
  --epochs "$EPOCHS" --max_steps "$STEPS")

[ "$GRAD_CKPT_DECODER" = "1" ] && ARGS+=(--grad_ckpt_decoder)
[ "$GRAD_CKPT_ENCODER" != "1" ] && ARGS+=(--no_grad_ckpt)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

echo "[launch] $(date '+%F %T') launcher=$LAUNCHER gpus=$NUM_GPUS device=$DEVICE_TYPE" | tee -a "$LOG"
echo "[launch] 解码器 ×10: width=$DEC_W depth=$DEC_D heads=$DEC_H mlp=$DEC_MLP" | tee -a "$LOG"
echo "[launch] batch: 每卡上限 BS=$BS, patch_budget=$BUDGET ⇒ 实际 bs=min(BS, BUDGET/N)" | tee -a "$LOG"
echo "[launch]         grad_accum=$GRAD_ACCUM ⇒ 全局有效 batch ≈ bs×$NUM_GPUS×$GRAD_ACCUM（随桶变化）" | tee -a "$LOG"
echo "[launch] 重算: 解码器=$GRAD_CKPT_DECODER 编码器=$GRAD_CKPT_ENCODER | 精度=$PRECISION | SDPA=$SDPA_BACKEND" | tee -a "$LOG"
echo "[launch] steps_per_rank=$STEPS data=$DATA_ROOT out=$OUTPUT_DIR" | tee -a "$LOG"
echo "[launch] 想按目标全局 batch 选 BS/BUDGET: python tools/plan_batch.py --cache $CACHE --world $NUM_GPUS" | tee -a "$LOG"

case "$LAUNCHER" in
  accelerate)
    AMP_FLAG="no"; [ "$PRECISION" = "bf16" ] && AMP_FLAG="bf16"
    [ "$PRECISION" = "fp16" ] && AMP_FLAG="fp16"
    if [ "$NUM_GPUS" -eq 1 ]; then
      set -x; "$PYTHON" "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"
    else
      set -x; accelerate launch --multi_gpu --num_processes "$NUM_GPUS" \
        --num_machines 1 --same_network --main_process_port "$MASTER_PORT" \
        --mixed_precision "$AMP_FLAG" \
        "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"
    fi
    ;;
  torchrun)
    set -x; torchrun --nproc_per_node="$NUM_GPUS" --master_port "$MASTER_PORT" \
      "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"
    ;;
  python)
    set -x; "$PYTHON" "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"
    ;;
  *) echo "未知 LAUNCHER=$LAUNCHER（accelerate|torchrun|python）" >&2; exit 2 ;;
esac
echo "RUN_MULTI_10X_DONE"
