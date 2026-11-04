#!/usr/bin/env bash
# =============================================================================
# eval_448_960m.sh — 448²/patch32/960M 单 ckpt 逐步评测（eval_stepwise_linattn.py 封装）
# -----------------------------------------------------------------------------
# 2026-11-04 · 用于三阶段产物 A / B / S3 的分段评测（run_448_960m_3stage.sh 每阶段结束调它），
#   也可单独重跑某个 ckpt。所有参数 env 化，默认**全量 5,000 张** val
#   （ImageNet-100 每类 50 张；--num_images 0 = 全量，不抽样）。
#
# 口径（与论文/仓库同口径, 不改）:
#   · cfg 从 ckpt["cfg"] 读 ⇒ 448/patch32/N196 自动生效（脚本里那行 "224×224" 打印串纯装饰）;
#   · 指标: L1_255（0-255 空间不 clamp）/ PSNR(/[0,1] clamp, 逐图再平均) / SSIM(训练侧) /
#     16 档逐步表 + Δ_t + 逐图最优停步; A 相（T=1）只有 1 行;
#   · S3 相（--zs_dim 224）每图 32×224 = 7168 值，与论文同「值数」口径，可直接与 224 基线比。
#
# 用法（远端仓库根目录）:
#   CKPT=/root/autodl-tmp/exp448/S3/ckpt_best.pt OUT=/root/autodl-tmp/exp448/eval \
#     GPU=1 NUM_IMAGES=0 bash tools/eval_448_960m.sh
#   DRY=1 ... bash tools/eval_448_960m.sh          # 干跑: 只打印命令
#
# 环境变量:
#   CKPT(必填)       ckpt 路径（通常是 <stage>/ckpt_best.pt）
#   OUT             json 输出目录（默认 /root/autodl-tmp/exp448/eval）
#   NAME            输出文件名前缀（默认: ckpt 所在目录名, 如 A/B/S3）
#                   ⇒ json = $OUT/${NAME}_stepwise.json
#   GPU             评测卡（默认 1；训练占 GPU0, 评测走 GPU1）
#   NUM_IMAGES      评测张数（默认 0 = 全量 5,000；>0 时按等距抽样, 冒烟用）
#   BS              评测 batch（默认 16; 前向, 显存不够就调小, 不影响指标口径）
#   DATA / NUM_WORKERS / AMP / VAL_PER_CLASS（默认 /dataset、8、1、50）
#   DUMP_PER_IMAGE  E-A1 每图每步落盘: 不设/0=关; 设 1=写到 $OUT/${NAME}_per_image.json;
#                   设路径=写到该路径
#   STRICT          1(默认)=ckpt 的 img_size/patch 不是 448/32 就报错退出; 0=只警告
#   LOG             额外落一份日志的路径（默认空 = 只走 stdout, 由调用方重定向）
#   DRY=1           只打印将执行的命令, 不加载 ckpt、不评测
#
# 退出码: 0 成功 / 2 参数或 ckpt 问题 / 3 评测进程失败或 json 未产出 / 4 ckpt cfg 不符
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "[$(basename "$0")] ✗ 请在仓库内执行（tools/ 的上一级）" >&2; exit 2; }
ROOT=$PWD

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/exp448/eval}
GPU=${GPU:-1}
NUM_IMAGES=${NUM_IMAGES:-0}
BS=${BS:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
AMP=${AMP:-1}
VAL_PER_CLASS=${VAL_PER_CLASS:-50}
STRICT=${STRICT:-1}
DRY=${DRY:-0}
CKPT=${CKPT:-}
NAME=${NAME:-}
DUMP_PER_IMAGE=${DUMP_PER_IMAGE:-0}
LOG=${LOG:-}

say() { echo "[eval448] $*"; }
py_ok() { case "$PY" in */*) [ -x "$PY" ] ;; *) command -v "$PY" >/dev/null 2>&1 ;; esac; }

[ -n "$CKPT" ] || { echo "[eval448] ✗ 请设置 CKPT（例如 CKPT=\$OUT/S3/ckpt_best.pt）" >&2; exit 2; }
[ -n "$NAME" ] || NAME=$(basename "$(dirname "$CKPT")")
JSON=$OUT/${NAME}_stepwise.json

# ── DUMP_PER_IMAGE: 0/空=关, 1=默认路径, 其它=当作路径 ─────────────────────────
DUMP=""
case "$DUMP_PER_IMAGE" in
  ""|0|"off") DUMP="" ;;
  1)          DUMP=$OUT/${NAME}_per_image.json ;;
  *)          DUMP=$DUMP_PER_IMAGE ;;
esac

say "ckpt = $CKPT"
say "out  = $JSON   (GPU$GPU, NUM_IMAGES=$NUM_IMAGES [0=全量 5000], BS=$BS)"
[ -n "$DUMP" ] && say "per-image dump = $DUMP"

if [ ! -f "$CKPT" ]; then
  if [ "$DRY" = "1" ]; then say "(dry) ckpt 不存在（干跑不评测）"; else echo "[eval448] ✗ ckpt 不存在: $CKPT" >&2; exit 2; fi
elif [ "$DRY" != "1" ] && ! py_ok; then
  echo "[eval448] ✗ python 不存在: $PY" >&2; exit 2
fi

# ── cfg 自检（448/p32 守卫；mmap 只读元数据, 不整块载入 3.8G 权重）─────────────
if [ "$DRY" != "1" ] && [ -f "$CKPT" ]; then
  CFG_INFO=$(env "PYTHONPATH=$ROOT" "$PY" - "$CKPT" <<'PY' 2>&1
import json, sys, torch
p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
cfg = ck.get("cfg") or {}
d = {k: cfg.get(k) for k in ("img_size", "patch", "num_specials", "dec_dim", "enc_dim",
                             "dec_recurrent", "dec_steps", "dec_step_plan", "zs_dim",
                             "num_patches", "pixel_hidden")}
d["step"] = ck.get("step"); d["epoch"] = ck.get("epoch")
print(json.dumps(d, ensure_ascii=False))
PY
) || true
  if [ -z "$CFG_INFO" ]; then
    say "⚠️ ckpt cfg 读取失败（继续评测, 由 eval_stepwise_linattn.py 自己校验）"
  else
    say "cfg: $CFG_INFO"
    IMG=$(printf '%s' "$CFG_INFO" | sed -n 's/.*"img_size": *\([0-9]*\).*/\1/p')
    PAT=$(printf '%s' "$CFG_INFO" | sed -n 's/.*"patch": *\([0-9]*\).*/\1/p')
    if [ "$STRICT" = "1" ] && { [ -n "$IMG" ] && [ "$IMG" != "448" ] || [ -n "$PAT" ] && [ "$PAT" != "32" ]; }; then
      echo "[eval448] ✗ cfg 不符本实验: img_size=$IMG patch=$PAT（期望 448/32）；确认要评别的口径请 STRICT=0" >&2
      exit 4
    fi
  fi
fi

if [ ! -d "$DATA" ] && [ "$DRY" != "1" ]; then echo "[eval448] ✗ 数据目录不存在: $DATA" >&2; exit 2; fi
[ "$DRY" = "1" ] || mkdir -p "$OUT"

CMD=( env "CUDA_VISIBLE_DEVICES=$GPU" "PYTHONPATH=$ROOT" "$PY" -u eval_stepwise_linattn.py
      --ckpt "$CKPT" --data_dir "$DATA" --batch_size "$BS" --num_workers "$NUM_WORKERS"
      --amp "$AMP" --val_per_class "$VAL_PER_CLASS" --num_images "$NUM_IMAGES" --out "$JSON" )
[ -n "$DUMP" ] && CMD+=( --dump_per_image "$DUMP" )

show_cmd() { local a; for a in "$@"; do printf '%q ' "$a"; done; printf '\n'; }

if [ "$DRY" = "1" ]; then
  printf '[dry][eval448] '
  show_cmd "${CMD[@]}"
  say "(dry) 结束：未加载 ckpt、未评测、未写文件"
  exit 0
fi

say "开始评测 $(date -Is)"
if [ -n "$LOG" ]; then
  mkdir -p "$(dirname "$LOG")"
  { "${CMD[@]}" 2>&1; } | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
else
  "${CMD[@]}"
  RC=$?
fi
say "rc=$RC end=$(date -Is)"
if [ "$RC" -ne 0 ]; then echo "[eval448] ✗ 评测进程失败 rc=$RC" >&2; exit 3; fi
if [ ! -f "$JSON" ]; then echo "[eval448] ✗ 未产出 json: $JSON" >&2; exit 3; fi

# ── 摘要（逐步表的关键行, 方便直接贴报告/回给 Lead）────────────────────────────
env "PYTHONPATH=$ROOT" "$PY" - "$JSON" <<'PY' 2>/dev/null || true
import json, sys
r = json.load(open(sys.argv[1]))
L1, P = r["L1_255"], r["PSNR"]
b = int(r.get("best_step") or 0)
T = len(L1)
print(f"[eval448] n={r['n_images']} T={T} K={r['K']} recurrent={r['recurrent']} "
      f"steps={r['steps']}")
if T == 1:
    print(f"[eval448] 单步(T=1): L1_255={L1[0]:.3f}  PSNR={P[0]:.3f} dB  "
          f"(常量基线 {r['const_baseline_255']:.2f})")
else:
    print(f"[eval448] 末步 step{T}: L1_255={L1[-1]:.3f}  PSNR={P[-1]:.3f} dB")
    print(f"[eval448] 最优步 step{b+1}/t={r['steps'][b]}: L1_255={L1[b]:.3f}  "
          f"PSNR={P[b]:.3f} dB  (总落差 {r['total_drop_255']:.3f} px, "
          f"常量基线 {r['const_baseline_255']:.2f})")
print(f"[eval448] json → {sys.argv[1]}")
PY
say "完成 $(date -Is)"
