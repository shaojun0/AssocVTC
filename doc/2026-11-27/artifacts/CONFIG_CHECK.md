# CONFIG_CHECK — v2.0(224²) 三阶段 + 阶段3-CE 上传脚本核对

2026-11-26 14:59 **更新**: QK-Norm 已撤销（被证伪，非根因）；改 env **`SDPA_BACKEND=math`**。步数/参数/init_from/白名单结论不变。只读/纯 CPU，未动 GPU/训练。依据 `train_linattn.py:1121/1147`: micro=batch/(accum·world)、steps_per_epoch=len(loader)//accum；DistributedSampler num_samples=ceil(121689/4)=30423、drop_last=True；val 三阶段同 5000 张。
## 1. 步数 —— PASS（精确命中 42,750 / 11,406 / 5,703）

| 阶段 | bs | accum | micro | loader_batches | steps/ep | ep | total | 期望 |
|---|---|---|---|---|---|---|---|---|
| s1 | 128 | 1 | 32 | 950 | 950 | 45 | **42,750** | 42,750 |
| s2 | 64 | 1 | 16 | 1,901 | 1,901 | 6 | **11,406** | 11,406 |
| s3 | 64 | 2 | 8 | 3,802 | 1,901 | 3 | **5,703** | 5,703 |

- 推导: 30423//32=950 / 30423//16=1901(1901.4) / 30423//8=3802(3802.9)→3802//2=1901；warmup=int(total·0.08)=3,420/912/456。实测: 运行中 s1.log `[data] ... steps/epoch=950 total_steps=42750 warmup=3420` ⇒ 与推算逐位一致。

## 2. 逐项参数 —— PASS（20 项 CLI 协议项 × 5 处 + `SDPA_BACKEND=math`(env) 全一致）

img224 · patch16 · K32 · enc 1248×13 h16 · dec 1440×7 h16 · pixel_hidden1024 · enc_attn std（**无 QK-Norm**）· dec_grad_ckpt1 ·
lr2e-4 · min_lr1e-6 · wd0.05 · warmup0.08 · clip1.0 · amp1 · seed0 · val_per_class50（s1/s2/s3 full + 2 smoke 共 5 处全命中）。
- **env**: `export SDPA_BACKEND=math` 在 s1:11 / s2:11 / s3:15；chain 两个 smoke 用 inline `SDPA_BACKEND=math`（chain:42/80）⇒ 5 处一致。
- s2/s3: dec_recurrent1 + fixed + block2 + dec_steps 1..16（16×2=32=K，末步读满）；s3: zs_dim224 + token_ce1 + vq(16384/8) + token_ce_init_pixel1。
- init_from: s2←`s1/ckpt_best.pt`、s3←`s2/ckpt_best.pt` ✓；ep 45/6/3；bs 128/64/64；accum 1/1/2；val_batches 40/80/80（40×128=80×64=5120≥5000 ⇒ 同一 5000 val）。
- **参数回到论文原样**: encoder **244,429,536** / total **481,495,840（481.496M）**；运行中 s1.log `[params]`、`[sdpa] backend=math`×4 已确认；
  s2 481,543,360（+mem_pos 47,520）/ s3 498,897,536（+token 头 18,269,184 +zs 559,552 +mem_pos −mem_proj 1,474,560）。

## 3. smoke vs full —— PASS（架构/损失/优化器/数值后端同配置）

差异仅 6 项: epochs 1 vs 6/3 · val_batches 1 vs 80 · num_workers 8 vs 16 · save_every_epochs 0 vs 1 · stop_at_step 20/8 · log_interval 5/2。
关键: **batch_size 64 与 accum 1/2 在 smoke 与 full 完全相同**（显存足迹一致）；smoke 同走 `$COMMON_ARCH $COMMON_OPT`、同 `$STEPS`、同 init_from、同 `SDPA_BACKEND=math` ⇒ 无"smoke 过、full 不同"隐患。

## 4. 加载兼容（无 RISK）

- s2←s1: cfg 差 dec_recurrent 0→1 ⇒ 仅多 `decoder.mem_pos`（白名单）；step_plan/block/dec_steps 只改 T/窗口、不动参数形状；两侧均**无 QK-Norm**（默认 0）⇒ encoder keys 一致。
- s3←s2: zs_dim 0→224 ⇒ `decoder.mem_proj` (1440,1248)→(1440,224) 进 shape_mismatch 白名单 + SVD 热启动；缺 `zs_proj/zcls_proj/token_head.*` 均在白名单；两侧均无 QK-Norm。
- 反例（现为 env 级）: 若某臂漏 `SDPA_BACKEND=math` ⇒ 走默认 flash/mem-efficient → 编码器梯度可大 30–230×、~step 2000 后梯度爆炸/全跳过（REPORT §2b）；已核对 4 脚本 + 2 smoke 全带 ✓

## 5. RISK / 提示

- **R1（文档）**: `run_s1.sh:2` 注释写 "45*951≈42,795" 错，实际 950×45=**42,750**（已实测）。仅注释。
- **R2（运行时）**: `--dec_grad_ckpt 1` 对 `dec_attn=std` 是 **no-op**（checkpoint 只在 `LinearAttentionDecoder._run`, model:1111）；若 s2 smoke OOM，兜底 `--batch_size 64 --accum_steps 2`（s2 micro 16→8；s3 已 accum2/micro8 ⇒ 需 accum 4），steps/epoch 仍 1,901。

**结论: ALL MATCH** — 42,750/11,406/5,703 精确命中；20 CLI 项 + `SDPA_BACKEND=math` 全一致；smoke≡full；无加载失败风险。架构已回到论文原样（244,429,536 / 481.496M）。链现状: s1 14:57:38 重启（`[sdpa] backend=math`、skip=0、vram 17.7G）。
