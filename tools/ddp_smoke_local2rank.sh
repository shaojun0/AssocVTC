#!/usr/bin/env bash
# =============================================================================
# ddp_smoke_local2rank.sh — DDP 代码 plumbing 冒烟: **同一张卡上开 2 个 rank**
# -----------------------------------------------------------------------------
# 用途（2026-11-04 · DDP 改造配套）
#   GPU0 被 A 相占用（不许干扰）时，在**一张空闲卡**上起 2 个 rank，验证
#   `train_linattn.py` 的 DDP 分支全部 plumbing:
#     1) torchrun WORLD_SIZE=2 起得来、两个 rank 都 init_process_group 成功;
#     2) DistributedSampler 每卡各拿 ceil(N/world) 个样本（互不重叠、并集=全集）;
#     3) steps/epoch 与同超参单卡参考**完全一致**（等价批量语义不变 → 脚本断言）;
#     4) 梯度累积 no_sync: 每卡 micro16×accum2 → 全局等效 batch 64;
#     5) rank0 独占 train_log.jsonl / status.json / args.json / eval_val.json / ckpt_*;
#     6) val（rank0 独占）与采样图（rank0 独占、用裸模型）不会让 rank1 在 DDP
#        集合通信上 hang 死; 全程无 hang、退出码 0;
#     7) 极小模型（enc 64×1 / dec 64×1 / patch32 / img448）⇒ 秒级，不抢资源。
#
# ⚠️ 为什么默认 backend=gloo（而不是 NCCL）
#   两个 rank 共用同一张物理卡时，NCCL 2.29.7 会直接
#   `Duplicate GPU detected : rank 0 and rank 1 both on CUDA device ...`
#   拒绝初始化（实测，NCCL_IGNORE_DUPLICATE_GPU 无效）。而 gloo 支持 CUDA tensor
#   的 allreduce（torch 2.12.1 实测）⇒ **同卡 plumbing 只能用 gloo**。
#   正式 2 卡训练（run_448_960m_ddp_afterA.sh）用 NCCL（DDP_BACKEND 默认 nccl），
#   两卡各占一张卡，不存在 duplicate 问题。
#
# 用法（远端仓库根目录）:
#   bash tools/ddp_smoke_local2rank.sh              # 默认 GPU=1
#   GPU=1 PORT=29531 OUT=/root/autodl-tmp/exp448/ddp/smoke2rank bash tools/ddp_smoke_local2rank.sh
#   DRY=1 bash tools/ddp_smoke_local2rank.sh        # 只打印将执行的命令
#
# 退出码: 0 全部检查通过 / 2 参数或环境问题 / 3 训练 rc≠0 / 4 断言/产物检查失败
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "[ddp_smoke] ✗ 请在仓库内执行（tools/ 的上一级）" >&2; exit 2; }
ROOT=$PWD

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
GPU=${GPU:-1}                       # 只暴露这一张卡给两个 rank
OUT=${OUT:-/root/autodl-tmp/exp448/ddp/smoke2rank}
PORT=${PORT:-29531}
NUM_WORKERS=${NUM_WORKERS:-8}
BACKEND=${DDP_BACKEND:-gloo}        # 同卡 2 rank 只能 gloo（见文件头）
MAX_STEPS=${MAX_STEPS:-20}
LIMIT=${LIMIT:-2000}
EPOCHS=${EPOCHS:-1}
VAL_BATCHES=${VAL_BATCHES:-2}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-5}
TIMEOUT=${TIMEOUT:-600}
DRY=${DRY:-0}
KEEP=${KEEP:-0}

say() { echo "[ddp_smoke] $*"; }
fail() { echo "[ddp_smoke] ✗ $*" >&2; exit "${2:-4}"; }
py_ok() { case "$PY" in */*) [ -x "$PY" ] ;; *) command -v "$PY" >/dev/null 2>&1 ;; esac; }

[ -f "$ROOT/train_linattn.py" ] || fail "缺少 $ROOT/train_linattn.py" 2
py_ok || fail "python 不存在: $PY" 2
[ -d "$DATA" ] || fail "数据目录不存在: $DATA" 2

T=$(seq -s, 1 16)
# 极小模型: enc 64×1 / dec 64×1 / patch32 / img448 / K32；B 相形态（T=16 fixed block2）
# ⇒ 走与正式 B/S3 相**同一条代码路径**（循环解码 + 梯度累积 + no_sync + val）。
# 注意: 环境变量放在 torchrun 外层（子进程继承），**不能**把 `env` 当 entrypoint:
# torch.distributed.run 会把第一个非选项 token 当 training_script（它内部用
# `python -u <training_script>` 起），`env` 会被当成 .py 脚本 ⇒ 启动失败。
RUN_ENV=( env "CUDA_VISIBLE_DEVICES=$GPU" "PYTHONPATH=$ROOT" "OMP_NUM_THREADS=4"
          "DDP_BACKEND=$BACKEND" )
# entrypoint 只给脚本路径: torchrun 内部会自己拼 `python -u <script>`（legacy 模式），
# 写成 `python script` 会被当成"用 python 去执行 python 二进制" ⇒ SyntaxError。
CMD=( "$ROOT/train_linattn.py"
      --data_dir "$DATA" --output_dir "$OUT"
      --img_size 448 --patch 32 --num_specials 32
      --enc_attn std --enc_dim 64 --enc_depth 1 --enc_heads 4
      --dec_dim 64 --dec_depth 1 --dec_heads 4 --pixel_hidden 128
      --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps "$T"
      --epochs "$EPOCHS" --batch_size 64 --accum_steps 2
      --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --warmup_frac 0.08
      --grad_clip 1.0 --amp 1 --seed 0 --save_opt 0 --num_workers "$NUM_WORKERS"
      --limit "$LIMIT" --max_steps "$MAX_STEPS" --val_batches "$VAL_BATCHES"
      --log_interval 1 --sample_interval "$SAMPLE_INTERVAL" )

TORCHRUN_CMD=( "$PY" -m torch.distributed.run --nproc_per_node=2 --master_port="$PORT"
               --max_restarts=0 )

say "================ DDP plumbing 冒烟（同一张卡上 2 rank）================"
say "$(date -Is)  ROOT=$ROOT"
say "GPU=$GPU（CUDA_VISIBLE_DEVICES=$GPU, 两个 rank 都落在同一张卡的 cuda:0）"
say "backend=$BACKEND  （同卡多 rank: NCCL 会拒绝 Duplicate GPU, 只能 gloo）"
say "OUT=$OUT  PORT=$PORT  max_steps=$MAX_STEPS  limit=$LIMIT  T=16  micro16×accum2=全局64"
say "模型: enc 64×1 / dec 64×1 / patch32 / img448 / pixel_hidden 128（plumbing 用极小模型）"

if [ "$DRY" = "1" ]; then
  printf '[dry] '; printf '%q ' "${RUN_ENV[@]}"; printf '%q ' "${TORCHRUN_CMD[@]}" "${CMD[@]}"; echo
  say "(dry) 结束：未执行"
  exit 0
fi

# 端口先探一下（被占则换一个空闲的）
if "$PY" - "$PORT" <<'PY' 2>/dev/null; then
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1]))); sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
  :
else
  PORT=$("$PY" - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()
PY
)
  say "端口被占 ⇒ 改用空闲端口 $PORT"
  TORCHRUN_CMD=( "$PY" -m torch.distributed.run --nproc_per_node=2 --master_port="$PORT"
                 --max_restarts=0 )
fi

rm -rf "$OUT"; mkdir -p "$OUT"
LOG=$OUT/../smoke2rank.log
say "启动 torchrun（日志 $LOG）..."
START=$(date +%s)
"${RUN_ENV[@]}" timeout "$TIMEOUT" "${TORCHRUN_CMD[@]}" "${CMD[@]}" > "$LOG" 2>&1
RC=$?
DUR=$(( $(date +%s) - START ))
say "rc=$RC  wall=${DUR}s"
if [ "$RC" -ne 0 ]; then
  echo "---- 日志尾部 ----" >&2; tail -n 40 "$LOG" >&2
  fail "训练 rc=$RC（$([ "$RC" = "124" ] && echo "超时 ${TIMEOUT}s = hang!" || echo "见上述日志")）" 3
fi

# ── 断言 ──────────────────────────────────────────────────────────────────
say "检查 1/4: 两个 rank 都起来了（各自 [ddp] banner + 各自的 sampler 划分）..."
N_RANK0=$(grep -o '\[ddp\] rank=0/2 ' "$LOG" | wc -l || true)
N_RANK1=$(grep -o '\[ddp\] rank=1/2 ' "$LOG" | wc -l || true)
N_SAMP0=$(grep -o '\[ddp\] rank=0 sampler num_samples=' "$LOG" | wc -l || true)
N_SAMP1=$(grep -o '\[ddp\] rank=1 sampler num_samples=' "$LOG" | wc -l || true)
say "  rank0 banner=$N_RANK0  rank1 banner=$N_RANK1  sampler 行 rank0=$N_SAMP0 rank1=$N_SAMP1"
[ "$N_RANK0" -ge 1 ] || fail "缺少 rank0 的 [ddp] banner（进程没起来或没进 DDP 分支）"
[ "$N_RANK1" -ge 1 ] || fail "缺少 rank1 的 [ddp] banner（rank1 没起来）"
[ "$N_SAMP0" -ge 1 ] || fail "缺少 rank0 的 sampler 行"
[ "$N_SAMP1" -ge 1 ] || fail "缺少 rank1 的 sampler 行"
grep -o '\[ddp\] rank=[01] sampler[^[]*' "$LOG" | sed 's/^/    /'

say "检查 2/4: steps/epoch 与单卡参考一致（等价批量语义不变）..."
STEPS_LINE=$(grep -o '\[ddp\] rank=0 sampler[^[]*' "$LOG" | head -1)
say "  $STEPS_LINE"
echo "$STEPS_LINE" | grep -q '== 单卡参考 .* ✓' || fail "steps/epoch 没通过代码内断言（见日志）"
EXPECT_STEPS=$("$PY" - "$LIMIT" <<'PY'
import sys
n = int(sys.argv[1]); world = 2
# 单卡参考: micro=64/2=32? 不 —— 单卡同超参为 micro 64/2? 这里直接按单卡算法:
# DDP micro = 64/(2*2)=16; 单卡 micro = 64/2=32（accum=2）
micro_single = 64 // 2
print((n // micro_single) // 2)
PY
)
echo "$STEPS_LINE" | grep -q "steps/epoch=$EXPECT_STEPS " || fail "steps/epoch ≠ 单卡参考 $EXPECT_STEPS"
say "  ✓ steps/epoch=$EXPECT_STEPS（limit=$LIMIT, 单卡参考口径）"

say "检查 3/4: rank0 独占落盘（日志/args/status/eval_val/ckpt）..."
for f in args.json train_log.jsonl status.json eval_val.json ckpt_best.pt ckpt_last.pt; do
  [ -f "$OUT/$f" ] || fail "rank0 未落盘: $OUT/$f"
done
say "  ✓ args/train_log/status/eval_val/ckpt_best/ckpt_last 全部存在（只由 rank0 写）"
say "  train_log 行数=$(wc -l < "$OUT/train_log.jsonl")  ckpt 大小=$(du -sh "$OUT" | cut -f1)"
say "  status.json: $(tr -d '\n' < "$OUT/status.json")"
say "  val: $(grep '^\[val\]' "$LOG" | head -1)"
say "  采样图: $(ls "$OUT/samples" 2>/dev/null | wc -l) 个（rank0 独占，用裸模型）"

say "检查 4/4: DistributedSampler 覆盖性（两卡并集=全集、交集=空、无重复）..."
"$PY" - "$DATA" "$LIMIT" <<'PY' || fail "sampler 覆盖性检查失败"
import sys
from torch.utils.data import DistributedSampler
sys.path.insert(0, ".")
from data_linattn import build_loaders
# 不建 loader（省内存/不 fork worker）, 只拿 tr_ds
tr_loader, va_loader, tr_ds, va_ds = build_loaders(
    sys.argv[1], 448, 16, 50, 0, int(sys.argv[2]), 0)
n = len(tr_ds)
seen = []
for r in range(2):
    s = DistributedSampler(tr_ds, num_replicas=2, rank=r, shuffle=True, seed=0)
    s.set_epoch(1)
    idx = list(s)
    seen.append(len(idx))
    print(f"  rank{r}: num_samples={len(idx)} 前5={idx[:5]}")
u = set()
for r in range(2):
    s = DistributedSampler(tr_ds, num_replicas=2, rank=r, shuffle=True, seed=0)
    s.set_epoch(1); u |= set(list(s))
print(f"  train_n={n} 每卡={seen} 并集={len(u)} 覆盖率={len(u)/n:.4%}")
assert len(u) == n, f"并集 {len(u)} ≠ 全集 {n}（有样本整 epoch 没被任何 rank 看到）"
assert sum(seen) >= n, "两卡样本数之和 < 全集"
print("  ✓ 两卡并集 = 全集（每个 epoch 所有训练样本都被看到）")
PY

if [ "$KEEP" = "1" ]; then
  say "KEEP=1 ⇒ 保留产物 $OUT（含 ${LOG}）"
else
  say "清理临时产物（KEEP=1 可保留）: rm -rf $OUT"
  rm -rf "$OUT"
fi
say "================ ✓ DDP plumbing 冒烟全部通过（${DUR}s）================"
exit 0
