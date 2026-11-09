#!/usr/bin/env bash
# vlm8b 预算课程训练（5 阶段，从 0 开始；阶段间 init-from 上一阶段 final）
# ---------------------------------------------------------------------------
# 用户 2026-11-06 计划：若 run2 两次判定 PLATEAU → 放弃 run2 结果，按下面课程重新训：
#   P1: k=8（8 个视觉 token 一次全给）                      2 epoch
#   P2: k∈{4,8}（分 2 步读出）                              1 epoch
#   P3: k∈{2,4,6,8}（分 4 步）                              1 epoch
#   P4: k∈{1..8}（分 8 步）                                 1 epoch
#   P5: 解冻 encoder, k∈{1..8}（分 8 步）                   2 epoch
# 合计 7 epoch = 26,621 步（448²/p32、8 token、Qwen3.5-4B、ZeRO-2，~8.9 s/step ≈ 65h）。
# LR（用户确认「大胆点」）：LM 3e-5 / proj 3e-4 / encoder 1e-5（cosine + 3% warmup）。
#
# 用法:
#   bash run_curriculum_vlm8b.sh                # 全跑（已有 DONE 的阶段自动跳过，可断点续跑）
#   EVAL=0 bash run_curriculum_vlm8b.sh         # 阶段间不做 quick eval
#   SKIP_TO=P3 bash run_curriculum_vlm8b.sh     # 从 P3 开始（调试用；默认仍会接着前面的 final）
#   DRY=1 bash run_curriculum_vlm8b.sh          # 只打印将要执行的命令
#
# 产物: /root/autodl-tmp/vlm8b/output/cur/P{1..5}/{final,VLM8B_DONE}
#       每阶段日志 logs/cur_P{N}.log；总标记 output/cur/CURRICULUM_DONE；失败写 FAILED_P{N}
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-/root/venv_qwen/bin/python}
PORT=${PORT:-29511}
OUT=${OUT:-/root/autodl-tmp/vlm8b/output/cur}
LR_LLM=${LR_LLM:-3e-5}; LR_PROJ=${LR_PROJ:-3e-4}; LR_ENC=${LR_ENC:-1e-5}
EVAL=${EVAL:-1}; DRY=${DRY:-0}
TRAIN_JSONL=${TRAIN_JSONL:-/root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl}
VAL_JSONL=${VAL_JSONL:-/root/autodl-tmp/exp_qwen/in100_selfdistill/sft/val_nothink_all.jsonl}

mkdir -p logs "$OUT"

# ── 前置检查（DRY 时跳过，便于本地/在线演练）───────────────────────────────
free_gb=999
if [ "${DRY:-0}" != "1" ]; then
    if pgrep -f "torch.distributed.run.*train.py" >/dev/null; then
        echo "[cur] ERROR: 已有训练进程在跑，先停掉（pkill -f torch.distributed.run）再启动课程" >&2
        pgrep -af "torch.distributed.run" >&2
        exit 3
    fi
    free_gb=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
    if [ "${free_gb:-0}" -lt 12 ]; then
        echo "[cur] ERROR: /root/autodl-tmp 仅剩 ${free_gb}G（<12G），先清理再跑" >&2
        exit 4
    fi
fi

# ── 阶段表 ─────────────────────────────────────────────────────────────────
NAMES=(P1 P2 P3 P4 P5)
BUDGETS=( "8" "4,8" "2,4,6,8" "1,2,3,4,5,6,7,8" "1,2,3,4,5,6,7,8" )
EPOCHS=( 2 1 1 1 2 )
DESC=( "k=8 一次全给" "k={4,8} 分2步" "k={2,4,6,8} 分4步" "k=1..8 分8步" "k=1..8 分8步" )

echo "[cur] plan: total_epochs=7 lr=$LR_LLM/$LR_PROJ/$LR_ENC out=$OUT free=${free_gb}G $(date -Is)"

prev=""                       # 上一阶段 final 目录（P1 为空 ⇒ 从预训练权重开始）
if [ -n "${SKIP_TO:-}" ]; then
    for name in "${NAMES[@]}"; do
        [ "$name" = "$SKIP_TO" ] && break
        [ -d "$OUT/$name/final" ] && prev="$OUT/$name/final"
    done
fi

for idx in "${!NAMES[@]}"; do
    name=${NAMES[$idx]}; dir="$OUT/$name"; log="logs/cur_$name.log"
    if [ -n "${SKIP_TO:-}" ] && [ -z "${started:-}" ]; then
        [ "$name" = "$SKIP_TO" ] || { [ -d "$dir/final" ] && prev="$dir/final"; continue; }
        started=1
    fi
    if [ -f "$dir/VLM8B_DONE" ]; then
        echo "[cur] $name 已完成，跳过（prev=$dir/final）"
        prev="$dir/final"
        continue
    fi
    init_args=(); [ -n "$prev" ] && init_args=(--init-from "$prev")

    cmd=( $PY -u -m torch.distributed.run --nproc_per_node=2 --master_port "$PORT"
          train.py --epochs "${EPOCHS[$idx]}"
          --budget-set "${BUDGETS[$idx]}" "${init_args[@]}"
          --lr-llm "$LR_LLM" --lr-proj "$LR_PROJ" --lr-encoder "$LR_ENC"
          --output-dir "$dir" --save-total-limit 1 --train-jsonl "$TRAIN_JSONL" )
    echo "[cur] ===== $name start $(date -Is) budgets=${BUDGETS[$idx]} epochs=${EPOCHS[$idx]} init=${prev:-none} (${DESC[$idx]})"
    if [ "$DRY" = "1" ]; then printf '[cur] DRY: %q ' "${cmd[@]}"; echo; prev="$dir/final"; continue; fi

    "${cmd[@]}" > "$log" 2>&1 &
    pid=$!

    # 新阶段已把上一阶段权重读进显存后，删除上一阶段目录（磁盘：每阶段 final ~10G）
    if [ -n "$prev" ]; then
        for _ in $(seq 1 180); do
            grep -q "续训权重已载入" "$log" 2>/dev/null && break
            kill -0 "$pid" 2>/dev/null || break
            sleep 5
        done
        if grep -q "续训权重已载入" "$log" 2>/dev/null; then
            echo "[cur] $name 已载入 $prev ⇒ 删除上一阶段目录 $(dirname "$prev")"
            rm -rf "$(dirname "$prev")"
        else
            echo "[cur] WARN: 未等到『续训权重已载入』，保留 $prev 目录" >&2
        fi
    fi
    wait "$pid"; rc=$?
    echo "[cur] $name rc=$rc $(date -Is)" | tee "logs/cur_$name.rc"
    if [ "$rc" -ne 0 ] || [ ! -d "$dir/final" ]; then
        echo "[cur] ERROR: $name 失败（rc=$rc, final=$([ -d "$dir/final" ] && echo yes || echo no)）→ 中止" >&2
        touch "$OUT/FAILED_$name"
        exit 1
    fi
    touch "$dir/VLM8B_DONE"
    rm -rf "$dir"/checkpoint-*          # 只保留 final，省磁盘

    if [ "$EVAL" = "1" ]; then       # 阶段末 quick eval（8 预算全评，100 张随机固定 seed）
        echo "[cur] $name quick eval ..."
        $PY -u eval.py --checkpoint "$dir/final" --jsonl "$VAL_JSONL" \
            --limit 100 --sample-seed 42 --output-dir "$OUT/eval_$name" \
            >> "logs/cur_eval_$name.log" 2>&1 || echo "[cur] WARN: $name eval 失败（不影响训练）" >&2
    fi
    prev="$dir/final"
done

if [ "$DRY" = "1" ]; then
    echo "[cur] DRY 结束（未落任何标记/产物）"
    exit 0
fi
touch "$OUT/CURRICULUM_DONE"
echo "[cur] CURRICULUM DONE $(date -Is)"
