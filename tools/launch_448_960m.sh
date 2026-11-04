#!/usr/bin/env bash
# =============================================================================
# launch_448_960m.sh — 后台启动 448²/p32/960M 三阶段（setsid nohup）+ 记录 run_info.json
# -----------------------------------------------------------------------------
# 2026-11-04 · 启动器只做 5 件事:
#   1) 预检（python / 代码 / 数据 / GPU 是否被占用）;
#   2) 记录 run_info.json（commit、enc/dec 参数量、三阶段完整命令、开始时间、GPU 分配）;
#   3) setsid nohup 后台跑 tools/run_448_960m_3stage.sh（训练 GPU_TRAIN=0, 评测 GPU_EVAL=1）;
#   4) 写 $OUT/run.pid;
#   5) 打印 PID 与 tail / 中断命令。
#
# 用法（远端仓库根目录）:
#   DRY=1 bash tools/launch_448_960m.sh              # 干跑: 只打印将执行的命令 + 预览 run_info（不启动作业）
#   bash tools/launch_448_960m.sh                    # 前台启动器（等待 3 秒确认存活后返回）
#   nohup setsid bash tools/launch_448_960m.sh > /root/autodl-tmp/exp448/logs/launch.out 2>&1 &
#
# 产物:
#   $OUT/run_info.json   本次 run 的元数据（见下）
#   $OUT/run.pid         训练链 PID（= run_448_960m_3stage.sh 的 $$）
#   $OUT/logs/run.log    启动器 + 三阶段的 stdout 汇总
#
# 环境变量（与 tools/run_448_960m_3stage.sh 同名同义, 由本脚本透传）:
#   OUT / DATA / PY / GPU_TRAIN(0) / GPU_EVAL(1) / NUM_WORKERS(16) / EVAL_BS(16)
#   MICRO_BS_A(16) / ACCUM_A(8) / MICRO_BS_B(16) / ACCUM_B(4) / DEC_GRAD_CKPT(1) / FORCE_EFF_BS(0)
#   TIME_BUDGET_H(0) / SEED(0) / KEEP_LAST(0) / NO_EVAL(0) / SKIP_STAGE1..3(0)
#   COMMIT      手动指定 commit 字符串（远端副本可能没有 .git; 未给则自动探测）
#   FORCE=1     已有 ALL_DONE / 已有存活 run / 训练卡被占用 时仍强行启动
#   DRY=1       干跑（只打印, 不 mkdir / 不写 run_info / 不启动作业）
#
# ---- TODO(bench): micro-bs / 累积 / dec_grad_ckpt 默认值见 run 脚本文件头 ------------
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "[$(basename "$0")] ✗ 请在仓库内执行（tools/ 的上一级）" >&2; exit 2; }
ROOT=$PWD

PY=${PY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
OUT=${OUT:-/root/autodl-tmp/exp448}
GPU_TRAIN=${GPU_TRAIN:-0}
GPU_EVAL_EXPLICIT=${GPU_EVAL:-}      # 空 = 未显式指定（单卡机器上会自动回落）
GPU_EVAL=${GPU_EVAL_EXPLICIT:-1}
DRY=${DRY:-0}
FORCE=${FORCE:-0}
COMMIT=${COMMIT:-}

# 透传给 run 脚本的训练/评测旋钮（默认值与 run_448_960m_3stage.sh 一致）
MICRO_BS_A=${MICRO_BS_A:-16}
ACCUM_A=${ACCUM_A:-8}
MICRO_BS_B=${MICRO_BS_B:-16}
ACCUM_B=${ACCUM_B:-4}
DEC_GRAD_CKPT=${DEC_GRAD_CKPT:-1}
FORCE_EFF_BS=${FORCE_EFF_BS:-0}
NUM_WORKERS=${NUM_WORKERS:-16}
EVAL_BS=${EVAL_BS:-16}
EVAL_WORKERS=${EVAL_WORKERS:-8}
TIME_BUDGET_H=${TIME_BUDGET_H:-0}
SEED=${SEED:-0}
KEEP_LAST=${KEEP_LAST:-0}
NO_EVAL=${NO_EVAL:-0}
SKIP_STAGE1=${SKIP_STAGE1:-0}
SKIP_STAGE2=${SKIP_STAGE2:-0}
SKIP_STAGE3=${SKIP_STAGE3:-0}

say() { echo "[launch448] $*"; }
die() { echo "[launch448] ✗ $1" >&2; exit "${2:-2}"; }

# 传给 run 脚本的环境（顺序固定, 便于 run_info 记录与手工复现）
RUN_ENV=( env "OUT=$OUT" "DATA=$DATA" "PY=$PY" "GPU_TRAIN=$GPU_TRAIN" "GPU_EVAL=$GPU_EVAL"
  "MICRO_BS_A=$MICRO_BS_A" "ACCUM_A=$ACCUM_A" "MICRO_BS_B=$MICRO_BS_B" "ACCUM_B=$ACCUM_B"
  "DEC_GRAD_CKPT=$DEC_GRAD_CKPT" "FORCE_EFF_BS=$FORCE_EFF_BS"
  "NUM_WORKERS=$NUM_WORKERS" "EVAL_BS=$EVAL_BS" "EVAL_WORKERS=$EVAL_WORKERS"
  "TIME_BUDGET_H=$TIME_BUDGET_H" "SEED=$SEED" "KEEP_LAST=$KEEP_LAST" "NO_EVAL=$NO_EVAL"
  "SKIP_STAGE1=$SKIP_STAGE1" "SKIP_STAGE2=$SKIP_STAGE2" "SKIP_STAGE3=$SKIP_STAGE3" )

say "================ AssocVTC 448²/p32/960M 三阶段 · 启动器 ================"
say "$(date -Is)  ROOT=$ROOT"
say "OUT=$OUT  DATA=$DATA"

# ── 1) 短路：已完成 / 已在跑 ─────────────────────────────────────────────────
if [ "$DRY" != "1" ]; then
  if [ -f "$OUT/ALL_DONE" ] && [ "$FORCE" != "1" ]; then
    say "$OUT/ALL_DONE 已存在 → 不重复启动（重跑: 删 ALL_DONE 或 FORCE=1）"; exit 0
  fi
  if [ -f "$OUT/run.pid" ]; then
    OLDPID=$(cat "$OUT/run.pid" 2>/dev/null || true)
    if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null; then
      say "已有训练链在跑: PID=$OLDPID（$OUT/run.pid）→ 不重复启动（要重启先 kill）"; exit 0
    fi
  fi
fi

# ── 2) 预检 ─────────────────────────────────────────────────────────────────
py_ok() { case "$PY" in */*) [ -x "$PY" ] ;; *) command -v "$PY" >/dev/null 2>&1 ;; esac; }
if ! py_ok; then
  [ "$DRY" = "1" ] && say "(dry) python 不存在: $PY" || die "python 不存在: $PY"
fi
[ -f "$ROOT/train_linattn.py" ] || die "缺少 train_linattn.py（请在仓库根目录的上一级执行 tools/ 下脚本）"
[ -f "$ROOT/eval_stepwise_linattn.py" ] || die "缺少 eval_stepwise_linattn.py"
[ -f "$ROOT/tools/run_448_960m_3stage.sh" ] || die "缺少 tools/run_448_960m_3stage.sh"
[ -f "$ROOT/tools/eval_448_960m.sh" ] || die "缺少 tools/eval_448_960m.sh"
[ -d "$DATA" ] || { [ "$DRY" = "1" ] && say "(dry) 数据目录不存在: $DATA" || die "数据目录不存在: $DATA"; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
NGPU=${NGPU:-0}
say "可见 GPU 数: $NGPU"
if [ "$NGPU" -lt 2 ]; then
  if [ -z "$GPU_EVAL_EXPLICIT" ]; then
    say "⚠️ 只有 $NGPU 张卡 → GPU_EVAL 回落为 $GPU_TRAIN（评测会和训练抢同一张卡）"
    GPU_EVAL=$GPU_TRAIN
    RUN_ENV=( env "OUT=$OUT" "DATA=$DATA" "PY=$PY" "GPU_TRAIN=$GPU_TRAIN" "GPU_EVAL=$GPU_EVAL"
      "MICRO_BS_A=$MICRO_BS_A" "ACCUM_A=$ACCUM_A" "MICRO_BS_B=$MICRO_BS_B" "ACCUM_B=$ACCUM_B"
      "DEC_GRAD_CKPT=$DEC_GRAD_CKPT" "FORCE_EFF_BS=$FORCE_EFF_BS"
      "NUM_WORKERS=$NUM_WORKERS" "EVAL_BS=$EVAL_BS" "EVAL_WORKERS=$EVAL_WORKERS"
      "TIME_BUDGET_H=$TIME_BUDGET_H" "SEED=$SEED" "KEEP_LAST=$KEEP_LAST" "NO_EVAL=$NO_EVAL"
      "SKIP_STAGE1=$SKIP_STAGE1" "SKIP_STAGE2=$SKIP_STAGE2" "SKIP_STAGE3=$SKIP_STAGE3" )
  fi
elif [ "$GPU_TRAIN" = "$GPU_EVAL" ]; then
  say "⚠️ GPU_TRAIN == GPU_EVAL == $GPU_TRAIN：评测会与训练抢卡（正常应为 0 / 1）"
fi

# 训练卡是否被别的作业占着（bencher 等）——被占时 OOM 概率高，默认拒绝启动
if [ "$DRY" != "1" ] && [ "$NGPU" -gt 0 ]; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_TRAIN" 2>/dev/null | tr -d ' ' | head -1)
  if [ -n "${USED:-}" ] && [ "$USED" -gt 5000 ]; then
    say "⚠️ GPU$GPU_TRAIN 已占用 ${USED}MiB（可能有别的作业, 如 bencher 还在跑）"
    if [ "$FORCE" != "1" ]; then
      die "训练卡被占用 → 中止。确认可抢占请加 FORCE=1（或等 bench 结束）" 6
    fi
    say "FORCE=1 → 继续启动（显存可能不够）"
  fi
  EVUSED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_EVAL" 2>/dev/null | tr -d ' ' | head -1)
  if [ -n "${EVUSED:-}" ] && [ "$EVUSED" -gt 5000 ]; then
    say "⚠️ 评测卡 GPU$GPU_EVAL 已占 ${EVUSED}MiB（评测可能变慢/OOM; 可用 GPU_EVAL 换卡）"
  fi
fi

# ── 3) commit 探测（远端 rsync 副本可能没有 .git）─────────────────────────────
if [ -n "$COMMIT" ]; then
  COMMIT_SRC="env"
elif git -C "$ROOT" rev-parse --short HEAD >/dev/null 2>&1; then
  COMMIT=$(git -C "$ROOT" rev-parse --short HEAD)
  COMMIT_SRC="git"
elif [ -f "$ROOT/.exp448_commit" ]; then
  COMMIT=$(head -1 "$ROOT/.exp448_commit")
  COMMIT_SRC="dotfile"
else
  COMMIT="src:$( { sha256sum "$ROOT/train_linattn.py" "$ROOT/model_v2_linattn.py" \
                        "$ROOT/eval_stepwise_linattn.py" 2>/dev/null; } | sha256sum | cut -c1-12)"
  COMMIT_SRC="srcsum"
fi
DIRTY=""
if git -C "$ROOT" rev-parse --short HEAD >/dev/null 2>&1; then
  DIRTY=$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l)
fi
say "commit=$COMMIT（来源 $COMMIT_SRC${DIRTY:+, 工作区改动 $DIRTY 个文件}）"

# ── 4) 抓三阶段完整命令（PLAN=1 干跑, 无任何副作用）────────────────────────────
say "生成三阶段命令（PLAN=1 干跑）..."
PLAN_ENV=( "${RUN_ENV[@]}" "PLAN=1" )
if PLAN_OUT=$( "${PLAN_ENV[@]}" bash "$ROOT/tools/run_448_960m_3stage.sh" 2>&1 ); then :; else
  echo "$PLAN_OUT" >&2
  die "run 脚本 PLAN 失败（配置有问题, 见上）" 2
fi
ST1=""; ST2=""; ST3=""
while IFS= read -r line; do
  case "$line" in
    "STAGE1 "*) ST1=${line#STAGE1 } ;;
    "STAGE2 "*) ST2=${line#STAGE2 } ;;
    "STAGE3 "*) ST3=${line#STAGE3 } ;;
  esac
done <<< "$PLAN_OUT"
[ -n "$ST1" ] && [ -n "$ST2" ] && [ -n "$ST3" ] || die "没能从 PLAN 输出解析出三阶段命令" 2

if [ "$DRY" = "1" ]; then
  say "(dry) 将执行:"
  printf '  OUT=%s DATA=%s GPU_TRAIN=%s GPU_EVAL=%s' "$OUT" "$DATA" "$GPU_TRAIN" "$GPU_EVAL"; echo
  printf '  setsid nohup bash %s/tools/run_448_960m_3stage.sh >> %s/logs/run.log 2>&1 &\n' "$ROOT" "$OUT"
  say "(dry) run_info.json 将写入 $OUT/run_info.json；run.pid 将写入 $OUT/run.pid"
  say "(dry) 三阶段命令预览:"
  printf '  stage1: %s\n  stage2: %s\n  stage3: %s\n' "$ST1" "$ST2" "$ST3"
  say "(dry) 结束：未创建目录、未写文件、未启动作业"
  exit 0
fi

mkdir -p "$OUT/A" "$OUT/B" "$OUT/S3" "$OUT/logs" "$OUT/eval"
: >> "$OUT/logs/run.log"

# ── 5) 后台启动（setsid nohup：SSH 断开不影响）───────────────────────────────
STARTED_AT=$(date -Is)
say "启动训练链 ..."
"${RUN_ENV[@]}" setsid nohup bash "$ROOT/tools/run_448_960m_3stage.sh" >> "$OUT/logs/run.log" 2>&1 < /dev/null &
LAUNCH_PID=$!
sleep 3
RUNPID=$LAUNCH_PID
if [ -f "$OUT/run.pid" ]; then RUNPID=$(cat "$OUT/run.pid" 2>/dev/null || echo "$LAUNCH_PID"); fi
if kill -0 "$RUNPID" 2>/dev/null; then
  say "训练链存活: PID=$RUNPID"
else
  say "⚠️ 启动 3 秒内进程已退出 → tail $OUT/logs/run.log 看原因"
fi
echo "$RUNPID" > "$OUT/run.pid"

# ── 6) 写 run_info.json ─────────────────────────────────────────────────────
export RI_PATH="$OUT/run_info.json" \
       RI_COMMIT="$COMMIT" RI_COMMIT_SRC="$COMMIT_SRC" RI_DIRTY="${DIRTY:-}" \
       RI_STARTED_AT="$STARTED_AT" RI_HOST="$(hostname 2>/dev/null || echo '?')" \
       RI_ROOT="$ROOT" RI_PY="$PY" RI_OUT="$OUT" RI_DATA="$DATA" \
       RI_GPU_TRAIN="$GPU_TRAIN" RI_GPU_EVAL="$GPU_EVAL" RI_NGPU="$NGPU" \
       RI_PID="$RUNPID" RI_LAUNCH_PID="$LAUNCH_PID" \
       RI_MICRO_BS_A="$MICRO_BS_A" RI_ACCUM_A="$ACCUM_A" \
       RI_MICRO_BS_B="$MICRO_BS_B" RI_ACCUM_B="$ACCUM_B" \
       RI_DEC_GRAD_CKPT="$DEC_GRAD_CKPT" RI_FORCE_EFF_BS="$FORCE_EFF_BS" \
       RI_NUM_WORKERS="$NUM_WORKERS" RI_EVAL_BS="$EVAL_BS" RI_TIME_BUDGET_H="$TIME_BUDGET_H" \
       RI_SEED="$SEED" RI_KEEP_LAST="$KEEP_LAST" RI_NO_EVAL="$NO_EVAL" \
       RI_SKIP="$SKIP_STAGE1/$SKIP_STAGE2/$SKIP_STAGE3" \
       RI_ST1="$ST1" RI_ST2="$ST2" RI_ST3="$ST3" \
       RI_GPULIST="$(nvidia-smi -L 2>/dev/null | tr '\n' '|' || true)"
if env "PYTHONPATH=$ROOT" "$PY" - <<'PY' 2>&1
import json, os, sys

E = os.environ
info = {
    "schema": "assocvtc-run-info/1",
    "experiment": "AssocVTC 448x448 / patch32 / N196 / enc~480M+dec~480M / 3-stage",
    "started_at": E["RI_STARTED_AT"],
    "hostname": E["RI_HOST"],
    "repo": {"root": E["RI_ROOT"], "commit": E["RI_COMMIT"],
             "commit_source": E["RI_COMMIT_SRC"],
             "dirty_files": int(E["RI_DIRTY"]) if E["RI_DIRTY"].strip().isdigit() else None},
    "python": E["RI_PY"],
    "design_doc": "/home/linaro/dsh/exp448_bench/DESIGN_448_960M_3stage.md",
    "run_card": "doc/2026-11-04/RUN_448_960M_3stage.md",
    "paths": {"out": E["RI_OUT"], "data": E["RI_DATA"],
              "logs": E["RI_OUT"] + "/logs", "run_log": E["RI_OUT"] + "/logs/run.log",
              "eval_dir": E["RI_OUT"] + "/eval",
              "ckpt": {"A": E["RI_OUT"] + "/A/ckpt_best.pt",
                       "B": E["RI_OUT"] + "/B/ckpt_best.pt",
                       "S3": E["RI_OUT"] + "/S3/ckpt_best.pt"}},
    "gpus": {"train": int(E["RI_GPU_TRAIN"]), "eval": int(E["RI_GPU_EVAL"]),
             "visible": [s for s in E["RI_GPULIST"].split("|") if s.strip()]},
    "pid": int(E["RI_PID"]), "launcher_pid": int(E["RI_LAUNCH_PID"]),
    "pid_file": E["RI_OUT"] + "/run.pid",
    "knobs": {"micro_bs_a": int(E["RI_MICRO_BS_A"]), "accum_a": int(E["RI_ACCUM_A"]),
              "eff_bs_a": int(E["RI_MICRO_BS_A"]) * int(E["RI_ACCUM_A"]),
              "micro_bs_b": int(E["RI_MICRO_BS_B"]), "accum_b": int(E["RI_ACCUM_B"]),
              "eff_bs_b": int(E["RI_MICRO_BS_B"]) * int(E["RI_ACCUM_B"]),
              "dec_grad_ckpt": int(E["RI_DEC_GRAD_CKPT"]),
              "force_eff_bs": int(E["RI_FORCE_EFF_BS"]),
              "num_workers": int(E["RI_NUM_WORKERS"]), "eval_bs": int(E["RI_EVAL_BS"]),
              "time_budget_h": float(E["RI_TIME_BUDGET_H"]), "seed": int(E["RI_SEED"]),
              "keep_last": int(E["RI_KEEP_LAST"]), "no_eval": int(E["RI_NO_EVAL"]),
              "skip_stage": E["RI_SKIP"]},
    "stage_cmds": {"stage1": E["RI_ST1"], "stage2": E["RI_ST2"], "stage3": E["RI_ST3"]},
    "eval_cmd_template": ("CKPT=<stage>/ckpt_best.pt OUT=" + E["RI_OUT"] + "/eval GPU="
                          + E["RI_GPU_EVAL"] + " NUM_IMAGES=0 NAME=stage<N>_stepwise "
                          "bash tools/eval_448_960m.sh"),
    "params": None,
    "notes": ["commit_source=srcsum 说明远端副本没有 .git，commit 是三个源文件的 sha256 前 12 位",
              "stage_cmds 由 tools/run_448_960m_3stage.sh 的 PLAN=1 干跑生成，与实际执行逐字一致",
              "K/T/每步=32/16/2；S3 相 zs_dim=224 ⇒ 每图 32×224=7168 值（与论文同值数口径）"],
}

# enc/dec 参数量：meta device 建模型, 不占显存（与 train/eval 同一份代码）
try:
    import torch
    from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn

    def build(rec, zs=0, T=16):
        steps = ",".join(str(i) for i in range(1, T + 1))
        with torch.device("meta"):
            cfg = LinAttnConfig(
                num_specials=32, enc_attn="std", enc_dim=1744, enc_depth=13, enc_heads=16,
                dec_dim=2048, dec_depth=7, dec_heads=16,
                dec_recurrent=rec, dec_steps=steps if rec else "",
                dec_step_plan="fixed", dec_block=2,
                img_size=448, patch=32, pixel_hidden=1024, zs_dim=zs)
            m = SRPhase1LinAttn(cfg)
            pc = m.count_params()
        return {"encoder_M": round(pc["encoder"] / 1e6, 3),
                "decoder_M": round(pc["decoder"] / 1e6, 3),
                "total_M": round(pc["total"] / 1e6, 3),
                "values_per_image": pc["values_per_image"],
                "N": cfg.num_patches, "zs_dim": zs}

    info["params"] = {"stage1": build(False), "stage2": build(True),
                      "stage3": build(True, zs=224)}
    info["params_source"] = "SRPhase1LinAttn.count_params() @meta（本机代码实测）"
except Exception as exc:                                  # 计参失败不能拖垮启动记录
    info["params"] = None
    info["params_error"] = f"{type(exc).__name__}: {exc}"

with open(E["RI_PATH"], "w") as f:
    json.dump(info, f, ensure_ascii=False, indent=1)

p = info["params"] or {}
s2 = p.get("stage2") or {}
s3 = p.get("stage3") or {}
print("[launch448] run_info.json → " + E["RI_PATH"])
if s2:
    print("[launch448] 参数量(实测): enc %.3fM  dec %.3fM  total %.3fM  (N=%s)"
          % (s2["encoder_M"], s2["decoder_M"], s2["total_M"], s2["N"]))
if s3:
    print("[launch448] S3 相: enc %.3fM（含 zs_dim=224 末端 FC） 每图 %s 值"
          % (s3["encoder_M"], s3["values_per_image"]))
else:
    print("[launch448] ⚠️ 计参失败: " + str(info.get("params_error")))
PY
then :; else
  say "⚠️ run_info.json 写入失败（训练已在跑, 不影响训练）"
fi

# ── 7) 提示 ─────────────────────────────────────────────────────────────────
say "======================= 已后台启动 ======================="
say "PID      : $RUNPID   （$OUT/run.pid）"
say "训练卡   : GPU$GPU_TRAIN    评测卡: GPU$GPU_EVAL（每阶段训练完立即后台评测）"
say "进度     : cat $OUT/A/status.json   # 阶段 1；B/S3 同理"
say "tail     : tail -f $OUT/logs/run.log"
say "           tail -f $OUT/logs/stage1.log   $OUT/logs/stage2.log   $OUT/logs/stage3.log"
say "评测     : tail -f $OUT/logs/eval_stage1.log  → $OUT/eval/stage1_stepwise.json"
say "完成标志 : $OUT/STAGE{1,2,3}_DONE  →  $OUT/ALL_DONE（失败: FAILED_stage{N}）"
say "中断     : kill $RUNPID   # setsid 已脱离终端；如需整链 kill 用 pkill -P $RUNPID"
say "拉回本地 : PORT=23556 bash tools/fetch_448_960m.sh"
