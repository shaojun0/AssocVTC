#!/usr/bin/env bash
# =============================================================================
# relay_selftest.sh — run_448_960m_ddp_afterA.sh 的沙箱自测（不碰 GPU / 不碰真 A 相）
# -----------------------------------------------------------------------------
# 为什么需要: 接力脚本要在 A 相跑完（~10h 后）无人值守地自动执行，必须先把
#   「等待判据 / DDP 冒烟决策 / 回退单卡 / 失败语义 / 幂等 / 标记」全部分支在沙箱里
#   跑一遍（真实尺寸 DDP 冒烟要等 A 结束，GPU0 现在被占，不能在本机预跑）。
#
# 做法: 在 /root/autodl-tmp/exp448/ddp/relaytest 造一个假工作区:
#   · 假 PY: 拦截 train_linattn.py 调用（-m torch.distributed.run 与单卡两种形态），
#     写假的 train_log.jsonl / ckpt_best.pt / [done] 行；`-c`/`-` 透传真 python；
#   · 假 eval_448_960m.sh: 直接产出 stage{N}_stepwise.json（可按 env 让它失败）；
#   · 假 A 相产物: A/ckpt_best.pt + logs/stage1.log（含 [done] steps=42750）。
#   然后跑 7 个场景，断言退出码/marker/日志命令形态/产物。
#
# 覆盖场景
#   T1 正常 DDP 全流程（冒烟 2.4 s/step ⇒ speedup 1.84×）→ B/S3 都走 torchrun, ALL_DONE
#   T2 幂等重跑（ALL_DONE 已存在）→ 立即退出 0，不重复训练
#   T3 DDP 太慢（4.0 s/step ⇒ speedup 1.10× < 1.3）→ ALERT + 回退单卡 + DDP_FALLBACK
#   T4 DDP 冒烟直接失败（rank 缺失）且 FORCE_DDP=1 → FAILED_ddp_smoke + 中止
#   T5 B 相训练失败 → FAILED_stage2 + 中止（不进入 S3）
#   T6 B 相评测失败 → FAILED_eval_stage2 + 中止（不进入 S3）
#   T7 等待判据: A 进程还在 ⇒ 一直等；A 进程退出且无 [done] ⇒ FAILED_stage1 拒绝接力
#
# 用法（远端）: bash /root/autodl-tmp/exp448/ddp/relaytest/relay_selftest.sh
# =============================================================================
set -uo pipefail
SB=${SB:-/root/autodl-tmp/exp448/ddp/relaytest}
REALPY=${REALPY:-/root/miniconda3/bin/python}
DATA=${DATA:-/root/autodl-tmp/datasets/imagenet100}
SRC=${SRC:-/root/autodl-tmp/exp448/code/AssocVTC}
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "  ✓ $*"; }
bad()  { FAIL=$((FAIL+1)); echo "  ✗ $*"; }
chk()  { if eval "$1"; then ok "$2"; else bad "$2  [$1]"; fi; }

rm -rf "$SB"; mkdir -p "$SB/repo/tools"

# ── 沙箱 repo（真 relay 脚本 + 假 eval；train_linattn.py 只做存在性检查用）──
cp "$SRC/tools/run_448_960m_ddp_afterA.sh" "$SB/repo/tools/"
cp "$SRC/train_linattn.py" "$SB/repo/train_linattn.py"
cat > "$SB/fakepy.sh" <<'FAKEPY'
#!/usr/bin/env bash
# 假 python: 只模拟 relay 用到的两种训练调用; -c / - 透传真 python
REAL=${REALPY:-/root/miniconda3/bin/python}
if [ "${1:-}" = "-c" ] || [ "${1:-}" = "-" ]; then exec "$REAL" "$@"; fi
out=""; maxsteps=0; epochs=1; ddp=0
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  case "${args[$i]}" in
    -m) [ "${args[$((i+1))]:-}" = "torch.distributed.run" ] && ddp=1 ;;
    --output_dir) out="${args[$((i+1))]:-}" ;;
    --max_steps) maxsteps="${args[$((i+1))]:-0}" ;;
    --epochs) epochs="${args[$((i+1))]:-1}" ;;
  esac
done
[ -n "$out" ] || { echo "[fakepy] no --output_dir" >&2; exit 3; }
# 可注入失败: FAKE_FAIL_MATCH 是 output_dir 的子串
if [ -n "${FAKE_FAIL_MATCH:-}" ] && [[ "$out" == *"$FAKE_FAIL_MATCH"* ]]; then
  echo "[fakepy] ✗ 注入失败: out=$out" >&2; exit 1
fi
if [ -n "${FAKE_MISSING_RANK1:-}" ] && [ "$ddp" = "1" ]; then :; fi
mkdir -p "$out"
if [ "$maxsteps" -gt 0 ]; then
  # ===== 真实尺寸 DDP 冒烟（run_ddp_smoke 读 train_log.jsonl 的 elapsed_min 差商）=====
  [ "$ddp" = "1" ] && [ -z "${FAKE_NO_RANK1:-}" ] && \
    echo "[ddp] rank=1/2 local_rank=1 backend=nccl device=cuda:1"
  [ "$ddp" = "1" ] && echo "[ddp] rank=0/2 local_rank=0 backend=nccl device=cuda:0"
  if [ "$ddp" = "1" ]; then
    echo "[data] train=121689 val=5000 steps/epoch=1901 total_steps=$maxsteps warmup=2"
    echo "[ddp] rank=0 sampler num_samples=60845 loader_batches=3802 micro=16 accum=2 ⇒ steps/epoch=1901 == 单卡参考 1901 ✓"
    echo "[ddp] rank=1 sampler num_samples=60845 loader_batches=3802 micro=16 accum=2 ⇒ steps/epoch=1901 == 单卡参考 1901 ✓"
  fi
  REALPY="$REAL" "$REAL" - "$out" "$maxsteps" <<'PYEOF'
import json, os, sys
out, n = sys.argv[1], int(sys.argv[2])
s = float(os.environ.get("FAKE_S_STEP", "2.4"))
with open(os.path.join(out, "train_log.jsonl"), "w") as f:
    for step in range(1, n + 1):
        f.write(json.dumps({"step": step, "epoch": 1, "elapsed_min": (30.0 + step * s) / 60.0}) + "\n")
if os.environ.get("FAKE_SMOKE_EMPTY_LOG") == "1":
    open(os.path.join(out, "train_log.jsonl"), "w").close()
PYEOF
  echo "[fakepy] smoke done ddp=$ddp out=$out"
  exit 0
fi
# ===== 正式阶段 =====
echo "[data] train=121689 val=5000 steps/epoch=1901 total_steps=$((1901*epochs)) warmup=912"
echo "[fakepy] stage ddp=$ddp epochs=$epochs out=$out"
printf 'x' > "$out/ckpt_best.pt"; printf 'x' > "$out/ckpt_last.pt"
echo "[done] steps=$((1901*epochs)) epochs=$epochs best_val_psnr=22.00 wall=1.0min"
exit 0
FAKEPY
chmod +x "$SB/fakepy.sh"

cat > "$SB/repo/tools/eval_448_960m.sh" <<'FAKEEVAL'
#!/usr/bin/env bash
# 假评测: 直接写 stage{N}_stepwise.json（FAKE_EVAL_RC=1 或 FAKE_EVAL_FAIL_MATCH 可注入失败）
NAME=${NAME:-X}; OUT=${OUT:-.}
if [ -n "${FAKE_EVAL_FAIL_MATCH:-}" ] && [[ "$NAME" == *"$FAKE_EVAL_FAIL_MATCH"* ]]; then
  echo "[fakeeval] ✗ 注入评测失败 NAME=$NAME" >&2; exit 1
fi
if [ "${FAKE_EVAL_RC:-0}" != "0" ]; then echo "[fakeeval] ✗ rc=$FAKE_EVAL_RC" >&2; exit "$FAKE_EVAL_RC"; fi
mkdir -p "$OUT"
echo '{"fake": true, "psnr": 22.0}' > "$OUT/${NAME}_stepwise.json"
echo "[fakeeval] $CKPT → $OUT/${NAME}_stepwise.json"
FAKEEVAL
chmod +x "$SB/repo/tools/eval_448_960m.sh"

# ── 造"假 A 相已完成"的 OUT ────────────────────────────────────────────────
mk_out() {  # $1 = out dir
  rm -rf "$1"; mkdir -p "$1/logs" "$1/A" "$1/eval"
  printf 'x' > "$1/A/ckpt_best.pt"; printf 'x' > "$1/A/ckpt_last.pt"
  cat > "$1/logs/stage1.log" <<'L'
[cfg] fake A phase log
[ 162.17m] step 8600/42750 ep10 loss=0.2151 113img/s
[ 163.00m] step 42750/42750 ep45 loss=0.1000 113img/s
[done] steps=42750 epochs=45 best_val_psnr=24.50 wall=780.1min
L
}

run_relay() {  # outdir env... ; 其余参数当 env 前缀
  local od=$1; shift
  ( cd "$SB/repo" && env OUT="$od" DATA="$DATA" PY="$SB/fakepy.sh" REALPY="$REALPY" \
      WAIT_POLL=1 WAIT_GPU_MIN=0 SMOKE_WAIT_GPU_MIN=0 STAGE_WAIT_GPU_MIN=0 MAX_WAIT_H=1 "$@" \
      bash tools/run_448_960m_ddp_afterA.sh )
}

echo "================ relay selftest @ $SB ================"

# ── T1: 正常 DDP 全流程 ───────────────────────────────────────────────────
echo "[T1] 正常 DDP 全流程（烟雾 2.4 s/step ⇒ speedup 1.84×）"
OUT1=$SB/out1; mk_out "$OUT1"
run_relay "$OUT1" > "$SB/T1.out" 2>&1; RC1=$?
chk '[ "$RC1" = 0 ]' "退出码 0（实际 $RC1）"
chk '[ -f "$OUT1/ALL_DONE" ]' "ALL_DONE 已写"
chk '[ -f "$OUT1/STAGE1_DONE" ] && [ -f "$OUT1/STAGE2_DONE" ] && [ -f "$OUT1/STAGE3_DONE" ]' "STAGE{1,2,3}_DONE 齐全"
chk '[ -f "$OUT1/eval/stage1_stepwise.json" ] && [ -f "$OUT1/eval/stage2_stepwise.json" ] && [ -f "$OUT1/eval/stage3_stepwise.json" ]' "三份 stepwise json 齐全"
chk 'grep -q "torch.distributed.run" "$OUT1/logs/stage2.log" && grep -q "torch.distributed.run" "$OUT1/logs/stage3.log"' "B/S3 都走 torchrun DDP"
chk 'grep -q "mode=ddp" "$OUT1/ddp_afterA_decision.txt"' "决策文件 mode=ddp"
chk 'grep -q "speedup_vs_single=1.8" "$OUT1/ddp_afterA_decision.txt"' "speedup 记录 ≈1.84×"
chk '[ ! -f "$OUT1/ddp_smoke_B" ] && [ ! -d "$OUT1/ddp_smoke_B" ]' "冒烟临时产物已清理"
chk '[ ! -f "$OUT1/B/ckpt_last.pt" ] && [ ! -f "$OUT1/S3/ckpt_last.pt" ]' "阶段成功后已删 ckpt_last"
chk '[ ! -f "$OUT1/A/ckpt_last.pt" ]' "A 的 ckpt_last 已删（省盘）"
chk 'grep -q "steps/epoch=1901" "$OUT1/logs/stage2.log"' "B 日志 steps/epoch=1901（watchdog 可解析）"
chk 'grep -q "\[done\] steps=11406" "$OUT1/logs/stage2.log"' "B [done] steps=11406 被校验通过"
chk '[ -f "$OUT1/afterA.heartbeat" ]' "heartbeat 文件存在"

# ── T2: 幂等重跑 ───────────────────────────────────────────────────────────
echo "[T2] 幂等重跑（ALL_DONE 已存在）"
N_BEFORE=$(grep -c 'stage2 start' "$OUT1/logs/stage2.log" || true)
run_relay "$OUT1" > "$SB/T2.out" 2>&1; RC2=$?
N_AFTER=$(grep -c 'stage2 start' "$OUT1/logs/stage2.log" || true)
chk '[ "$RC2" = 0 ]' "退出码 0"
chk '[ "$N_BEFORE" = "$N_AFTER" ]' "没有重复训练（stage2 start 行数 $N_BEFORE→$N_AFTER）"
chk 'grep -q "ALL_DONE 已存在" "$SB/T2.out"' "打印'已存在 → 跳过'"

# ── T3: DDP 太慢 ⇒ 回退单卡 ────────────────────────────────────────────────
echo "[T3] DDP 太慢（4.0 s/step ⇒ speedup 1.10× < 1.3）⇒ 回退单卡"
OUT3=$SB/out3; mk_out "$OUT3"
run_relay "$OUT3" FAKE_S_STEP=4.0 > "$SB/T3.out" 2>&1; RC3=$?
chk '[ "$RC3" = 0 ]' "退出码 0（回退后仍跑完，实际 $RC3）"
chk '[ -f "$OUT3/ALL_DONE" ]' "ALL_DONE 已写"
chk '[ -f "$OUT3/DDP_FALLBACK" ]' "DDP_FALLBACK 标记"
chk '[ -f "$OUT3/ALERT_ddp_slow.txt" ]' "ALERT_ddp_slow.txt 已写"
chk 'grep -q "mode=single" "$OUT3/ddp_afterA_decision.txt"' "决策文件 mode=single"
chk '! grep -q "torch.distributed.run" "$OUT3/logs/stage2.log"' "B 相日志里没有 torchrun（走了单卡）"
chk 'grep -q -- "--accum_steps 4" "$OUT3/logs/stage2.log"' "回退用 micro16×accum4（等效 batch 64）"

# ── T4: DDP 冒烟失败 + FORCE_DDP=1 ⇒ 不降级 ────────────────────────────────
echo "[T4] DDP 冒烟失败（缺 rank1 banner）+ FORCE_DDP=1 ⇒ FAILED_ddp_smoke 中止"
OUT4=$SB/out4; mk_out "$OUT4"
run_relay "$OUT4" FAKE_NO_RANK1=1 FORCE_DDP=1 > "$SB/T4.out" 2>&1; RC4=$?
chk '[ "$RC4" = 5 ]' "退出码 5（实际 $RC4）"
chk '[ -f "$OUT4/FAILED_ddp_smoke" ]' "FAILED_ddp_smoke 已写"
chk '[ ! -f "$OUT4/STAGE2_DONE" ] && [ ! -f "$OUT4/ALL_DONE" ]' "没有继续跑 B/S3"

# ── T5: B 相训练失败 ───────────────────────────────────────────────────────
echo "[T5] B 相训练失败 ⇒ FAILED_stage2 中止"
OUT5=$SB/out5; mk_out "$OUT5"
run_relay "$OUT5" FAKE_FAIL_MATCH=/B > "$SB/T5.out" 2>&1; RC5=$?
chk '[ "$RC5" = 7 ]' "退出码 7（实际 $RC5）"
chk '[ -f "$OUT5/FAILED_stage2" ]' "FAILED_stage2 已写"
chk '[ ! -f "$OUT5/STAGE2_DONE" ] && [ ! -f "$OUT5/ALL_DONE" ] && [ ! -f "$OUT5/STAGE3_DONE" ]' "未进入 S3"

# ── T6: B 相评测失败 ───────────────────────────────────────────────────────
echo "[T6] B 相评测失败 ⇒ FAILED_eval_stage2 中止"
OUT6=$SB/out6; mk_out "$OUT6"
run_relay "$OUT6" FAKE_EVAL_FAIL_MATCH=stage2 > "$SB/T6.out" 2>&1; RC6=$?
chk '[ "$RC6" = 8 ]' "退出码 8（实际 $RC6）"
chk '[ -f "$OUT6/FAILED_eval_stage2" ]' "FAILED_eval_stage2 已写"
chk '[ ! -f "$OUT6/STAGE3_DONE" ] && [ ! -f "$OUT6/ALL_DONE" ]' "未进入 S3"

# ── T7a: A 进程还在 ⇒ 一直等 ───────────────────────────────────────────────
echo "[T7a] 等待判据: A 进程在跑 ⇒ 一直等（不抢跑）"
OUT7=$SB/out7; mk_out "$OUT7"
cat > "$OUT7/logs/stage1.log" <<'L'
[ 162.17m] step 8600/42750 ep10 loss=0.2151
L
( exec -a "train_linattn.py --output_dir $OUT7/A" sleep 120 ) &
DUMMY=$!
( cd "$SB/repo" && env OUT="$OUT7" DATA="$DATA" PY="$SB/fakepy.sh" REALPY="$REALPY" \
    WAIT_POLL=1 WAIT_GPU_MIN=0 SMOKE_WAIT_GPU_MIN=0 STAGE_WAIT_GPU_MIN=0 MAX_WAIT_H=1 bash tools/run_448_960m_ddp_afterA.sh ) \
    > "$SB/T7a.out" 2>&1 &
RELAY=$!
sleep 4
chk 'kill -0 "$RELAY" 2>/dev/null' "A 还在跑时 relay 仍在等待"
chk '[ ! -f "$OUT7/STAGE2_DONE" ]' "A 还在跑时没有开始 B 相"
chk 'grep -q "等 A 相结束" "$SB/T7a.out"' "打印了等待日志"
kill -0 "$DUMMY" 2>/dev/null && kill "$DUMMY" 2>/dev/null
# A 进程退出但日志仍无 [done] ⇒ relay 应判 A 崩溃并拒绝接力
wait "$RELAY" 2>/dev/null; RC7A=$?
chk '[ "$RC7A" = 4 ]' "A 无 [done] ⇒ 退出码 4（实际 $RC7A）"
chk '[ -f "$OUT7/FAILED_stage1" ]' "FAILED_stage1 已写"
chk '[ ! -f "$OUT7/ALL_DONE" ] && [ ! -f "$OUT7/STAGE2_DONE" ]' "拒绝用半训 ckpt 起 B"

# ── T7b: A 进程退出 + [done] 存在 ⇒ 正常接力 ───────────────────────────────
echo "[T7b] 等待判据: A 进程退出且 [done] 在 ⇒ 正常接力"
OUT7B=$SB/out7b; mk_out "$OUT7B"
cat >> "$OUT7B/logs/stage1.log" <<'L'
[done] steps=42750 epochs=45 best_val_psnr=24.50 wall=780.1min
L
( exec -a "train_linattn.py --output_dir $OUT7B/A" sleep 120 ) &
DUMMY2=$!
( cd "$SB/repo" && env OUT="$OUT7B" DATA="$DATA" PY="$SB/fakepy.sh" REALPY="$REALPY" \
    WAIT_POLL=1 WAIT_GPU_MIN=0 SMOKE_WAIT_GPU_MIN=0 STAGE_WAIT_GPU_MIN=0 MAX_WAIT_H=1 bash tools/run_448_960m_ddp_afterA.sh ) \
    > "$SB/T7b.out" 2>&1 &
RELAY2=$!
sleep 3
chk 'kill -0 "$RELAY2" 2>/dev/null' "A 在跑时 relay 等待"
kill "$DUMMY2" 2>/dev/null
wait "$RELAY2" 2>/dev/null; RC7B=$?
chk '[ "$RC7B" = 0 ]' "A 结束+[done] ⇒ 接力跑完退出 0（实际 $RC7B）"
chk '[ -f "$OUT7B/ALL_DONE" ]' "ALL_DONE 已写"

echo
echo "================ 结果: PASS=$PASS FAIL=$FAIL ================"
[ "$FAIL" = "0" ] || { echo "✗ 有失败场景，见 $SB/T*.out"; exit 1; }
echo "✓ 全部场景通过（沙箱 $SB）"
