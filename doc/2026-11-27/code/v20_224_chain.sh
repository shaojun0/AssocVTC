#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# chain_v20.sh — v2.0(224²) 全自动链:
#   [s1 已在跑] → 等 s1 完 → s2 smoke(20 步门) → s2 full(6ep) →
#   s3 smoke(8 步门) → s3 full(3ep, 阶段3-CE: 对齐 LlamaGen VQ-16 离散码) →
#   rFID 评测 (s1/s2/s3) → SUMMARY.md
# 任何一步失败即写 FAILED 并退出（不留半训产物进下一阶段）。
# ═══════════════════════════════════════════════════════════════════════════════
set -u
cd /root/AssocVTC_tok448
ST=/root/autodl-tmp/v20_224
LOG=$ST/chain_status
mkdir -p "$ST/eval"
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
DATA=/root/autodl-tmp/datasets/imagenet100
VQ=/root/autodl-tmp/vq/llamagen_vq16_c2i.pt
STEPS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
say(){ echo "[chain $(date -Is)] $*" >> "$LOG"; }
fail(){ say "FAILED: $*"; echo "$(date -Is) $*" > "$ST/FAILED"; exit 1; }
COMMON_ARCH="--img_size 224 --patch 16 --num_specials 32 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024"
COMMON_OPT="--lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --warmup_frac 0.08 --grad_clip 1.0 --amp 1 --seed 0 --val_per_class 50 --save_opt 0 --num_workers 8 --sample_interval 0"

say "chain start; waiting for s1 to finish"
for i in $(seq 1 1800); do
  grep -q "rc=" "$ST/status" 2>/dev/null && break
  sleep 60
done
grep -q "rc=" "$ST/status" 2>/dev/null || fail "等 s1 超过 30h"
RC1=$(grep -o "rc=[0-9]*" "$ST/status" | tail -1 | cut -d= -f2)
say "s1 done rc=$RC1"
[ "$RC1" = "0" ] || fail "s1 rc=$RC1"
[ -f "$ST/s1/ckpt_best.pt" ] || fail "缺 s1/ckpt_best.pt"
say "s1 best: $(python -c "import json;d=json.load(open('$ST/s1/eval_val.json'));print('psnr=%.3f l1=%.4f'%(d['psnr'],d['l1_norm']))" 2>/dev/null)"

# ── s2 smoke: 20 步真跑（T=16, bs64, 从 s1 热启动）────────────────────────────
say "s2 smoke (20 steps) ..."
SM=$ST/smoke_s2
run_s2_smoke() {   # $1 = accum（micro = 64/(accum*4)）
  rm -rf "$SM"; mkdir -p "$SM"
  SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched timeout 2400 torchrun --nproc_per_node=4 --master_port=29742 train_linattn.py \
    --data_dir $DATA --output_dir "$SM" $COMMON_ARCH $COMMON_OPT \
    --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps "$STEPS" --dec_grad_ckpt 1 \
    --init_from "$ST/s1/ckpt_best.pt" --epochs 1 --batch_size 64 --accum_steps "$1" \
    --val_batches 1 --log_interval 5 --save_every_epochs 0 --stop_at_step 20 \
    > "$ST/smoke_s2.log" 2>&1
}
run_s2_smoke 1
RC=$?
if [ $RC -ne 0 ] && grep -qi "out of memory" "$ST/smoke_s2.log"; then
  # OOM 自动降 micro：accum 1→2（micro 16→8），steps/epoch 仍 1901 ⇒ 协议不变
  say "s2 smoke OOM(accum=1) ⇒ 自动改 accum=2 重试（micro 8, steps/epoch 不变）"
  sed -i "s/--accum_steps 1 /--accum_steps 2 /" /root/autodl-tmp/v20_224_run_s2.sh
  grep -n "accum_steps" /root/autodl-tmp/v20_224_run_s2.sh >> "$LOG"
  run_s2_smoke 2
  RC=$?
fi
if [ $RC -ne 0 ] || grep -qiE "nan|out of memory|Traceback" "$ST/smoke_s2.log"; then
  tail -30 "$ST/smoke_s2.log" >> "$LOG"; rm -rf "$SM"; fail "s2 smoke rc=$RC"
fi
say "s2 smoke PASS: $(grep -oE 'step 20/[0-9]+ .*eta=[0-9.]+h' "$ST/smoke_s2.log" | tail -1)"
grep -q "img/s" "$ST/smoke_s2.log" || { rm -rf "$SM"; fail "s2 smoke 无吞吐行"; }
rm -rf "$SM"

# ── s2 正式 ────────────────────────────────────────────────────────────────────
say "s2 full start (~8h)"
bash /root/autodl-tmp/v20_224_run_s2.sh
RC=$(grep -o "rc=[0-9]*" "$ST/status_s2" | tail -1 | cut -d= -f2)
say "s2 rc=$RC"
[ "$RC" = "0" ] || fail "s2 rc=$RC"
[ -f "$ST/s2/ckpt_best.pt" ] || fail "缺 s2/ckpt_best.pt"
say "s2 best: $(python -c "import json;d=json.load(open('$ST/s2/eval_val.json'));print('psnr=%.3f l1=%.4f'%(d['psnr'],d['l1_norm']))" 2>/dev/null)"

# ── s3 smoke: 8 步真跑（阶段3-CE, zs_dim=224, 从 s2 热启动）──────────────────
say "s3-CE smoke (8 steps) ..."
SM=$ST/smoke_s3
run_s3_smoke() {   # $1 = accum（micro = 64/(accum*4)）
  rm -rf "$SM"; mkdir -p "$SM"
  SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched timeout 2400 torchrun --nproc_per_node=4 --master_port=29752 train_linattn.py \
    --data_dir $DATA --output_dir "$SM" $COMMON_ARCH $COMMON_OPT \
    --dec_recurrent 1 --step_plan fixed --block 2 --dec_steps "$STEPS" --dec_grad_ckpt 1 \
    --zs_dim 224 --init_from "$ST/s2/ckpt_best.pt" \
    --token_ce 1 --vq_ckpt $VQ --vq_codebook_size 16384 --vq_embed_dim 8 --token_ce_init_pixel 1 \
    --epochs 1 --batch_size 64 --accum_steps "$1" --val_batches 1 --log_interval 2 \
    --save_every_epochs 0 --stop_at_step 8 \
    > "$ST/smoke_s3.log" 2>&1
}
run_s3_smoke 2
RC=$?
if [ $RC -ne 0 ] && grep -qi "out of memory" "$ST/smoke_s3.log"; then
  # OOM 自动降 micro：accum 2→4（micro 8→4），steps/epoch 仍 1901 ⇒ 协议不变
  say "s3 smoke OOM(accum=2) ⇒ 自动改 accum=4 重试（micro 4, steps/epoch 不变）"
  sed -i "s/--accum_steps 2 /--accum_steps 4 /" /root/autodl-tmp/v20_224_run_s3_ce.sh
  grep -n "accum_steps" /root/autodl-tmp/v20_224_run_s3_ce.sh >> "$LOG"
  run_s3_smoke 4
  RC=$?
fi
if [ $RC -ne 0 ] || grep -qiE "nan|out of memory|Traceback" "$ST/smoke_s3.log"; then
  tail -40 "$ST/smoke_s3.log" >> "$LOG"; rm -rf "$SM"; fail "s3 smoke rc=$RC"
fi
grep -q " ce=" "$ST/smoke_s3.log" || { tail -30 "$ST/smoke_s3.log" >> "$LOG"; rm -rf "$SM"; fail "s3 smoke 无 ce 日志"; }
say "s3 smoke PASS: $(grep -oE 'step 8/[0-9]+ .*ce=[0-9.]+ acc=[0-9.]+' "$ST/smoke_s3.log" | tail -1)"
rm -rf "$SM"

# ── s3 正式 ────────────────────────────────────────────────────────────────────
say "s3-CE full start (~4h)"
bash /root/autodl-tmp/v20_224_run_s3_ce.sh
RC=$(grep -o "rc=[0-9]*" "$ST/status_s3" | tail -1 | cut -d= -f2)
say "s3 rc=$RC"
[ "$RC" = "0" ] || fail "s3 rc=$RC"
[ -f "$ST/s3/ckpt_best.pt" ] || fail "缺 s3/ckpt_best.pt"
say "s3 best: $(python -c "import json;d=json.load(open('$ST/s3/eval_val.json'));print('tok_ce=%.4f tok_acc=%.4f'%(d['tok_ce'],d['tok_acc']))" 2>/dev/null)"

# ── rFID 评测（5000 张 val, 与 fast_rfid 既有口径一致）────────────────────────
EV=$ST/eval
say "eval s1 (pixel, T=1) ..."
python fast_rfid.py --ckpt "$ST/s1/ckpt_best.pt" --data_dir $DATA --img_size 224 \
  --out "$EV/s1_rfid.json" --tag s1_stage1 --batch_size 64 --amp 1 --render pixel \
  --num_images 0 > "$EV/s1.log" 2>&1
say "eval s1 rc=$?"
say "eval s2 (pixel, T=16) ..."
python fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --data_dir $DATA --img_size 224 \
  --out "$EV/s2_rfid.json" --tag s2_stage2 --batch_size 64 --amp 1 --render pixel \
  --num_images 0 > "$EV/s2.log" 2>&1
say "eval s2 rc=$?"
say "eval s3-CE (vq, T=16) ..."
python fast_rfid.py --ckpt "$ST/s3/ckpt_best.pt" --data_dir $DATA --img_size 224 \
  --out "$EV/s3_rfid.json" --tag s3_ce --batch_size 64 --amp 1 --render vq \
  --vq_ckpt $VQ --vq_codebook_size 16384 --vq_embed_dim 8 \
  --num_images 0 > "$EV/s3.log" 2>&1
say "eval s3 rc=$?"
if [ -f "$ST/s3/ckpt_last.pt" ]; then
  python fast_rfid.py --ckpt "$ST/s3/ckpt_last.pt" --data_dir $DATA --img_size 224 \
    --out "$EV/s3_last_rfid.json" --tag s3_ce_last --batch_size 64 --amp 1 --render vq \
    --vq_ckpt $VQ --vq_codebook_size 16384 --vq_embed_dim 8 \
    --num_images 0 > "$EV/s3_last.log" 2>&1
  say "eval s3_last rc=$?"
fi

# ── 汇总 ──────────────────────────────────────────────────────────────────────
python /root/autodl-tmp/v20_224_summarize.py "$ST" >> "$LOG" 2>&1
say "ALL DONE"
echo "ALL DONE $(date -Is)" > "$ST/ALL_DONE"
