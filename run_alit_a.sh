#!/bin/bash
# =============================================================================
# run_alit_a.sh — PLAN §3 阶段 A（零训练）执行脚本        [task-3 · owner: alit-runner]
# -----------------------------------------------------------------------------
# 目标机: 新实例 ssh -p 18540 root@connect.westc.seetacloud.com  (RTX 4080S)
# 在服务器上运行:
#   cd /root/AssocVTC && nohup setsid bash run_alit_a.sh \
#       --ood_far /root/autodl-tmp/datasets/ade20k/val \
#       > doc/2026-10-16/artifacts/alit/logs/run_alit_a.log 2>&1 &
#
# 前置依赖（必须由 Lead rsync 到位; 缺依赖的步骤写 SKIP_*.json 后**继续**）:
#   · eval_stepwise_linattn.py : --dump_per_image / --data_dir_ood / --domain
#   · tools/alit_adaptive_k.py (E-A2/A4/A6)
#   · metrics_linattn.py       : zs_stats（E-A5 第 1 项, 可选）
#
# 用法: bash run_alit_a.sh [--ood_far DIR] [--only a0,a1,a2,a3,a5] [--no_rsync]
# 产物: doc/2026-10-16/artifacts/alit/*.json （+ 控制机同路径 rsync 回）
# =============================================================================
set -uo pipefail

PY=/root/miniconda3/bin/python
REPO=${REPO:-/root/AssocVTC}
A=$REPO/doc/2026-10-16/artifacts/alit
LOGS=$A/logs
P=/root/autodl-tmp

CKPT_STD=$P/srdiff_stdattn/stdattn_k64/ckpt_best.pt
CKPT_STD_REC=$P/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt
CKPT_LIN=$P/srdiff_linattn/k64/ckpt_best.pt
CKPT_LIN_REC=$P/srdiff_linattn/k64_rec/ckpt_best.pt
OOD_COCO=$P/datasets/coco2017/val2017
OOD_FAR=""
ONLY=""
NO_RSYNC=0
RESULT=$A/alit_a_steps.tsv

while [ $# -gt 0 ]; do
  case "$1" in
    --ood_far)  OOD_FAR="$2"; shift 2 ;;
    --only)     ONLY="$2"; shift 2 ;;
    --no_rsync) NO_RSYNC=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO" || exit 3
mkdir -p "$A" "$LOGS"
: > "$RESULT"

log()  { echo "[$(date -Is)] $*" | tee -a "$LOGS/run_alit_a.progress.log"; }
want() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }

run_step() {           # run_step <name> <cmd...>  —— 失败隔离, 记录 rc/耗时
  local name="$1"; shift
  if ! want "$name"; then log "SKIP(not selected) $name"; return 0; fi
  local t0; t0=$(date +%s)
  log "=== BEGIN $name : $* ==="
  "$@" >> "$LOGS/$name.log" 2>&1
  local rc=$? dt; dt=$(( $(date +%s) - t0 ))
  printf "%s\t%s\t%s\n" "$name" "$rc" "$dt" >> "$RESULT"
  if [ $rc -ne 0 ]; then log "!!! FAIL $name rc=$rc (${dt}s) — 继续后续步骤"
  else log "=== OK $name (${dt}s)"; fi
  return 0
}

skip_mark() {          # skip_mark <name> <reason> —— 显式 SKIP 标记, 不静默
  local name="$1" reason="$2"
  printf '{"id":"%s","status":"SKIPPED","reason":"%s","ts":"%s"}\n' \
    "$name" "$reason" "$(date -Is)" > "$A/SKIP_$name.json"
  printf "%s\t%s\t%s\n" "$name" "SKIP" "0" >> "$RESULT"
  log "SKIP $name : $reason"
}

# ── preflight ────────────────────────────────────────────────────────────────
log "########## run_alit_a.sh START (only='${ONLY:-all}') ##########"
for f in "$CKPT_STD" "$CKPT_STD_REC" "$CKPT_LIN" "$CKPT_LIN_REC" \
         "$REPO/eval_stepwise_linattn.py" "$REPO/eval_linattn.py"; do
  [ -e "$f" ] || { log "FATAL 缺文件: $f"; exit 4; }
done
HAS_DUMP=$(grep -c -- "--dump_per_image" "$REPO/eval_stepwise_linattn.py" || true)
HAS_OOD=$(grep -c -- "--data_dir_ood" "$REPO/eval_stepwise_linattn.py" || true)
HAS_AK=0; [ -f "$REPO/tools/alit_adaptive_k.py" ] && HAS_AK=1
HAS_ZS=$(grep -c "def zs_stats" "$REPO/metrics_linattn.py" || true)
N_COCO=$([ -d "$OOD_COCO" ] && find "$OOD_COCO" -type f | wc -l || echo 0)
N_FAR=$( [ -n "$OOD_FAR" ] && [ -d "$OOD_FAR" ] && find "$OOD_FAR" -type f | wc -l || echo 0)
log "preflight: dump_per_image=$HAS_DUMP data_dir_ood=$HAS_OOD alit_adaptive_k=$HAS_AK zs_stats=$HAS_ZS"
log "preflight: OOD_COCO=$OOD_COCO ($N_COCO imgs); OOD_FAR='$OOD_FAR' ($N_FAR imgs)"
log "preflight: GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"
BUSY=$(pgrep -af "train_linattn.py|eval_linattn.py|eval_stepwise_linattn.py" | grep -v run_alit_a | wc -l)
if [ "$BUSY" -gt 0 ]; then
  log "!!! WARNING: 检测到 $BUSY 个并发 GPU 进程（应独占）:"; pgrep -af 'train_linattn.py|eval_' | head -5 | tee -a "$LOGS/run_alit_a.progress.log"
fi

# 统一 stepwise 调用（自动按依赖决定是否加 --dump_per_image / --data_dir_ood）
sw_step() {            # sw_step <name> <ckpt> <out> [ood_dir] [domain] [dump_path]
  local name="$1" ck="$2" out="$3" ood="${4:-}" dom="${5:-}" dump="${6:-}"
  local -a args=(--ckpt "$ck" --num_images 0 --batch_size 64 --num_workers 16 --out "$out")
  if [ -n "$ood" ]; then
    if [ "$HAS_OOD" -eq 0 ]; then skip_mark "$name" "eval_stepwise_linattn.py 无 --data_dir_ood（需 rsync）"; return 0; fi
    args+=(--data_dir_ood "$ood")
  fi
  [ -n "$dom" ] && args+=(--domain "$dom")
  if [ -n "$dump" ]; then
    if [ "$HAS_DUMP" -gt 0 ]; then args+=(--dump_per_image "$dump")
    else skip_mark "$name" "eval_stepwise_linattn.py 无 --dump_per_image（需 rsync）"; return 0; fi
  fi
  run_step "$name" "$PY" eval_stepwise_linattn.py "${args[@]}"
}

# ── A0: 四臂同进程全量评测（已复现, 有则复用）────────────────────────────────
if [ ! -f "$A/baseline_four_arm_eval.json" ]; then
  run_step a0_eval4 "$PY" eval_linattn.py \
    --ckpt "$CKPT_LIN" "$CKPT_STD" "$CKPT_LIN_REC" "$CKPT_STD_REC" \
    --num_images 0 --batch_size 128 --num_workers 16 \
    --out "$A/baseline_four_arm_eval.json"
else
  log "A0 复用已有 $A/baseline_four_arm_eval.json（内容已逐位复现 26.602/31.219/25.304/28.582）"
fi

# ── 回归门（task-3 步骤 1）: 新代码**不加新开关**必须与发布版逐位一致 ────────
# 先冻结发布版基准（A1 会用新代码重写同名文件, 故基准另存 _published_code）
if [ ! -f "$A/stepwise_stdattn_k64_rec_published_code.json" ]; then
  cp -p "$A/stepwise_stdattn_k64_rec.json" "$A/stepwise_stdattn_k64_rec_published_code.json"
  log "回归门: 冻结发布版基准 → stepwise_stdattn_k64_rec_published_code.json"
fi
GATE_A=$A/stepwise_stdattn_k64_rec_published_code.json
GATE_B=$A/stepwise_gate_stdattn_k64_rec.json
run_step a0b_gate_eval "$PY" eval_stepwise_linattn.py --ckpt "$CKPT_STD_REC" \
  --num_images 0 --batch_size 64 --num_workers 16 --out "$GATE_B"
run_step a0b_gate_diff "$PY" "$A/gate_diff.py" --a "$GATE_A" --b "$GATE_B" \
  --out "$A/gate_diff_stdattn_rec.json"
GATE_V=$("$PY" -c "import json;print(json.load(open('$A/gate_diff_stdattn_rec.json')).get('verdict','NOFILE'))" 2>/dev/null || echo NOFILE)
log "回归门判决 = $GATE_V"
if [ "$GATE_V" = "FAIL" ]; then
  log "!!! 回归门 FAIL: 新代码默认路径与发布版不一致 ⇒ 中止 E-A, 先报 Lead 查清"
  exit 6
fi

# ── E-A1: 每图逐步曲线 + oracle 停步（两条 rec 臂）───────────────────────────
sw_step a1_dump_std_rec "$CKPT_STD_REC" "$A/stepwise_stdattn_k64_rec.json" \
        "" in100 "$A/per_image_stdattn_k64_rec.json"
sw_step a1_dump_lin_rec "$CKPT_LIN_REC" "$A/stepwise_linattn_k64_rec.json" \
        "" in100 "$A/per_image_linattn_k64_rec.json"

# ── E-A1/A2/A4/A6: 一次调用同时产出停步分布 / TSC / 等预算 / Spearman / Δ_t ──
if [ "$HAS_AK" -eq 1 ]; then
  run_step a2_analysis_std "$PY" tools/alit_adaptive_k.py \
    --stepwise "$A/per_image_stdattn_k64_rec.json" \
    --out "$A/tsc_stdattn_rec.json"
  run_step a2_analysis_lin "$PY" tools/alit_adaptive_k.py \
    --stepwise "$A/per_image_linattn_k64_rec.json" \
    --out "$A/tsc_linattn_rec.json"
else
  skip_mark a2_analysis_std "tools/alit_adaptive_k.py 不存在（task-1 alit-tools 交付）"
  skip_mark a2_analysis_lin "tools/alit_adaptive_k.py 不存在（task-1 alit-tools 交付）"
fi

# ── E-A3: IID / 中距 OOD(COCO) / 远距 OOD(ADE20K) ───────────────────────────
sw_step a3_iid_std_rec "$CKPT_STD_REC" "$A/ood_in100_stdattn_rec.json" \
        "" in100 "$A/per_image_in100_stdattn_rec.json"
sw_step a3_coco "$CKPT_STD_REC" "$A/ood_coco_stdattn_rec.json" \
        "$OOD_COCO" coco "$A/per_image_coco_stdattn_rec.json"
if [ -n "$OOD_FAR" ]; then
  sw_step a3_far "$CKPT_STD_REC" "$A/ood_far_stdattn_rec.json" \
          "$OOD_FAR" ade20k "$A/per_image_far_stdattn_rec.json"
else
  skip_mark a3_far "未提供 --ood_far（第三级远距 OOD: ADE20K/CelebA）"
fi
# 三级域阶梯: 一份 json 同时喂 IID/COCO/ADE20K（--domains 与 --stepwise 顺序对齐）
if [ "$HAS_AK" -eq 1 ] && [ -f "$A/per_image_in100_stdattn_rec.json" ] \
   && [ -f "$A/per_image_coco_stdattn_rec.json" ] && [ -f "$A/per_image_far_stdattn_rec.json" ]; then
  run_step a3_ladder "$PY" tools/alit_adaptive_k.py \
    --stepwise "$A/per_image_in100_stdattn_rec.json" \
               "$A/per_image_coco_stdattn_rec.json" \
               "$A/per_image_far_stdattn_rec.json" \
    --domains iid coco ade20k \
    --out "$A/domain_ladder.json"
else
  skip_mark a3_ladder "缺少任一 per_image dump 或 alit_adaptive_k.py"
fi

# ── E-A5: z_s 冗余度（第 1 项; 注意力熵可后置）──────────────────────────────
if [ "$HAS_ZS" -gt 0 ]; then
  run_step a5_zs_stats "$PY" "$A/zs_redundancy.py" \
    --ckpt "$CKPT_STD_REC" --num_images 2000 \
    --out "$A/zs_redundancy_stdattn_rec.json"
else
  skip_mark a5_zs_stats "metrics_linattn.py 无 zs_stats（可后置）"
fi

# ── 汇总 ─────────────────────────────────────────────────────────────────────
"$PY" - "$A" <<'PYEOF' >> "$LOGS/a_summary.log" 2>&1
import json, os, sys
A = sys.argv[1]
out = {"steps": [], "artifacts": {}}
tsv = os.path.join(A, "alit_a_steps.tsv")
if os.path.exists(tsv):
    out["steps"] = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
for name in ("baseline_four_arm_eval.json", "per_image_stdattn_k64_rec.json",
             "stepwise_stdattn_k64_rec.json", "tsc_stdattn_rec.json",
             "ood_in100_stdattn_rec.json", "ood_coco_stdattn_rec.json",
             "ood_far_stdattn_rec.json", "domain_ladder.json",
             "zs_redundancy_stdattn_rec.json", "gate_diff_stdattn_rec.json"):
    out["artifacts"][name] = "present" if os.path.exists(os.path.join(A, name)) else "MISSING"
json.dump(out, open(os.path.join(A, "alit_a_summary.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
PYEOF

log "########## run_alit_a.sh DONE ##########"
cat "$RESULT" | tee -a "$LOGS/run_alit_a.progress.log"
echo ">>> 控制机取回: rsync -rltzv -e 'ssh -p 18540' root@connect.westc.seetacloud.com:$A/ /home/linaro/dsh/AssocVTC/doc/2026-10-16/artifacts/alit/"
