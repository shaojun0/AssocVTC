#!/bin/bash
# ── kd2b: latent 蒸馏的**第二次**尝试（修掉 kd2-#1 的两个 bug 后）──────────────────
# kd2-#1 失败根因（doc/2026-11-23/REPORT_kd2_latent_scale_bug.md）:
#   Bug 1 渲染多除 scale（z_hat 是未缩放 latent）⇒ 已修（decode(z_hat)）;
#   Bug 2 两边乘 0.18215 ⇒ 损失/梯度整体 × s²=0.0332, 有效 λ 只剩 0.033 ⇒ 本脚本
#         `SCALE=1.0`（原始 latent MSE）, λ 由 `KDW` 直接表达。
# 与 kd2-#1 的差异**只有这三样**：SCALE 1.0、输出目录 kd2b、外加逐步 kd_mse 日志
# （`kd_mse_steps`/`kd_rel_steps`，用来区分"早期步拖累" vs "末步也压不下去"）。
#
# 运行链（全自动, 每步都写 $ST/status）:
#   0) 等 GPU 空
#   1) **P0-d 决策门**（10–15 min, 不训练）: 基线 ckpt 的 8 步图重新过一遍
#      D(E(ours))，回答"把输出放回 SD-VAE 流形"本身值多少 rFID（NOTES §3-P0-d）
#   2) **SMOKE=60 步**: 确认 latent 分支在 λ=1.0 下不发散（NaN / kd_mse 下降）
#      不通过 ⇒ ABORT，不浪费 3h
#   3) 正式 2 epoch（预算 3h）
#   4) 逐 epoch + best 的 8 步 rFID（`--render sdvae` = z_hat→teacher decoder）
# 预登记: `--render sdvae` 的 rFID **60–90**；若仍 ~110 ⇒ 瓶颈是 z_head
#        （从像素预测读 z），不是 λ ⇒ 下一步把 readout 移到 decoder memory。
set -u
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
SRC=${SRC:-/root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt}
ST=${ST:-/root/autodl-tmp/tok256x12_t8/kd2b}
TAG=${TAG:-kd2b}
EPOCHS=${EPOCHS:-2}
LR=${LR:-2e-5}
BUDGET=${BUDGET:-3.0}
PORT=${PORT:-29709}
KDW=${KDW:-1.0}
SCALE=${SCALE:-1.0}
SMOKE=${SMOKE:-60}
SKIP_P0D=${SKIP_P0D:-0}
# 逐步 KD 权重（1-based, T=8）: 随步递增 ⇒ 384 值的前几步不硬装 4096 值的细节。
KD_W=${KD_W:-"0.2 0.3 0.5 0.7 0.9 1.0 1.0 1.2"}
mkdir -p "$ST/s2"
echo "[kd2b] start $(date) src=$SRC epochs=$EPOCHS lr=$LR kd_weight=$KDW scale=$SCALE smoke=$SMOKE" \
  > "$ST/status"

# ── 0) 等 GPU 空（最多 60 min）──
for i in $(seq 1 120); do
  n=$(pgrep -f "train_linattn.py|fast_rfid.py" 2>/dev/null | wc -l)
  [ "${n:-0}" -eq 0 ] && break
  sleep 30
done
n=$(pgrep -f "train_linattn.py|fast_rfid.py" 2>/dev/null | wc -l)
if [ "${n:-0}" -ne 0 ]; then
  echo "[kd2b] ABORT: GPU busy ($n procs) $(date)" >> "$ST/status"; exit 1
fi
echo "[kd2b] GPU free $(date)" >> "$ST/status"

# ── 1) P0-d 决策门：rFID(real, D(E(基线 8 步重建))) ──
if [ "$SKIP_P0D" = "0" ]; then
  python3 fast_rfid.py --ckpt "$SRC" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --num_images 2000 --out "$ST/p0d_baseline.json" --tag p0d_baseline \
    --batch_size 64 --amp 1 --render pixel --vae_roundtrip 1 \
    > "$ST/p0d.log" 2>&1
  echo "[kd2b] p0d rc=$? $(date)" >> "$ST/status"
  python3 - "$ST/p0d_baseline.json" >> "$ST/status" 2>&1 <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[p0d] n={d['n_images']} ckpt={d['ckpt']}")
for k in sorted(d["per_step"], key=int):
    v = d["per_step"][k]
    print(f"[p0d] step{k}: PSNR={v['psnr']:.3f} rFID={v['rfid']:.3f} "
          f"| D(E(.)) PSNR={v['psnr_vae_rt']:.3f} L1={v['l1_255_vae_rt']:.3f} "
          f"rFID={v['rfid_vae_rt']:.3f}")
PY
fi

COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 256 --patch 16 --num_specials 256 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --batch_size 128 --accum_steps 4 --lr $LR --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 10 \
 --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 1 --save_opt 0 --num_workers 8 \
 --zs_dim 12 --allow_k_gt_n 1 \
 --z_head 1 --render sdvae \
 --kd_space latent --kd_weight $KDW --kd_teacher stabilityai/sd-vae-ft-mse --kd_local_only 1 \
 --kd_loss l1 --kd_latent_scale $SCALE --kd_steps all --kd_step_weights $KD_W \
 --kd_start_frac 0.02 --kd_ramp_frac 0.03 \
 --dec_recurrent 1 --step_plan cumulative --dec_steps 32,64,96,128,160,192,224,256"

# ── 2) SMOKE：60 步, λ=1.0 下先确认不发散 ──
if [ "$SMOKE" != "0" ]; then
  rm -rf "$ST/smoke"; mkdir -p "$ST/smoke"
  SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
   torchrun --nproc_per_node=4 --master_port=$PORT train_linattn.py $COMMON \
   --epochs 1 --warmup_frac 0.02 --time_budget_h 0.5 --stop_at_step "$SMOKE" \
   --init_from "$SRC" --output_dir "$ST/smoke" > "$ST/smoke.log" 2>&1
  echo "[kd2b] smoke rc=$? $(date)" >> "$ST/status"
  # 硬门槛: (a) 无 NaN; (b) 有 kd_mse 记录; (c) 最后一条 kd_mse 的逐步均值 <= 首条
  python3 - "$ST/smoke/train_log.jsonl" >> "$ST/status" 2>&1 <<'PY'
import json, math, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
kd = [r for r in rows if r.get("kd_mse") is not None]
bad = [r["step"] for r in rows
       if any(isinstance(v, float) and not math.isfinite(v) for v in [r["loss"], r["recon"]])]
print(f"[smoke] rows={len(rows)} kd_rows={len(kd)} nan_steps={bad[:5]}")
if kd:
    print(f"[smoke] kd_mse first={kd[0]['kd_mse']:.5f} last={kd[-1]['kd_mse']:.5f}")
    print(f"[smoke] kd_rel first={kd[0].get('kd_rel_steps')}")
    print(f"[smoke] kd_rel last ={kd[-1].get('kd_rel_steps')}")
ok = (len(kd) >= 2 and not bad and any(isinstance(v, float) and math.isfinite(v)
      for r in kd for v in r.get("kd_rel_steps", {}).values()))
print("[smoke] GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  if [ $? -ne 0 ]; then
    echo "[kd2b] ABORT: smoke gate FAIL $(date)" >> "$ST/status"; exit 1
  fi
  echo "[kd2b] smoke gate PASS $(date)" >> "$ST/status"
fi

# ── 3) 正式训练 ──
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
 torchrun --nproc_per_node=4 --master_port=$PORT train_linattn.py $COMMON \
 --epochs $EPOCHS --warmup_frac 0.02 --time_budget_h $BUDGET \
 --init_from "$SRC" --output_dir "$ST/s2" > "$ST/s2.log" 2>&1
echo "[kd2b] train rc=$? $(date)" >> "$ST/status"
grep -E "\[val\] ep|\[done\]" "$ST/s2.log" | tail -8 >> "$ST/status" 2>/dev/null

# ── 4) 逐 epoch + best: 8 步 rFID 走 I2 投放路径（z_hat → teacher decoder）──
for ck in "$ST"/s2/ckpt_ep*.pt; do
  [ -f "$ck" ] || continue
  t=$(basename "$ck" .pt)
  python3 fast_rfid.py --ckpt "$ck" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --out "$ST/${t}_fast.json" --tag "$t" --batch_size 64 --amp 1 \
    --render sdvae --kd_latent_scale "$SCALE" \
    > "$ST/fast_${t}.log" 2>&1
  echo "[kd2b] eval $t rc=$? $(date)" >> "$ST/status"
done
if [ -f "$ST/s2/ckpt_best.pt" ]; then
  python3 fast_rfid.py --ckpt "$ST/s2/ckpt_best.pt" --data_dir /root/autodl-tmp/datasets/imagenet100 \
    --img_size 256 --out "$ST/ckpt_best_fast.json" --tag kd2b_best --batch_size 64 --amp 1 \
    --render sdvae --kd_latent_scale "$SCALE" \
    > "$ST/fast_best.log" 2>&1
  echo "[kd2b] eval best rc=$? $(date)" >> "$ST/status"
fi
echo "[kd2b] DONE $(date)" >> "$ST/status"
