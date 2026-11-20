> ⚠️ **勘误（2026-11-21 补充）**：本报告 §2/§3 关于"8 步循环解码失败"的归因**已被后续实验推翻**。
> 在完全相同的配置下只把 rFID 权重从 1.0 改成 0（唯一变量），末步 PSNR 就从 18.11 回到 **21.87 dB**
> （patchDC +3.80），rFID 同时从 116.3 降到 **102.5**，末步掉点与棋盘格伪影都消失。
> ⇒ **18.04 dB 的塌点是 rFID λ=1.0 造成的，8 步循环解码本身可训**；该损失在本设置下纯有害
> （连它自己的 rFID 指标都没刷赢）。zs_dim=448 臂进一步到 **23.11 dB**（patchDC +5.04），
> 与同值 T=1 仍差 4.17 dB（差距随码率增大）⇒ 短板是循环读出的高码率 rate-scaling。
> 详见 `doc/2026-11-21/REPORT_t8_norfid.md`。

# REPORT — 8步循环解码 + 批次 rFID 损失（zs_dim=192, K=16），2026-11-20

> 一句话：**8 步循环解码（从 T=1 单步 ckpt 热启动、3 epoch）失败**（IN-100 val PSNR 18.04 dB，
> 低于 patchDC 18.07）；**批次 rFID 辅助损失 λ=1.0 出现 reward hacking**（末步 rFID 119→116，
> 但 PSNR 20.72→18.11 dB，棋盘格伪影）；与 ALIT 同值数量级对比差 5.8–14×。代码见
> `rfid_loss.py` + `train_linattn.py` 的 `--rfid_*` 参数。

## 0. 任务与背景

本次两条改动：

1. **8 步循环解码**：K=16 个 special token 分 8 步渐进读出（`--dec_recurrent 1
   --step_plan cumulative --dec_steps 1,2,3,5,7,10,13,16`），读出窗口
   `[0,1],[2,2],[3,3],[4,5],[6,7],[8,10],[11,13],[14,16]`（cum_z_s=[1,2,3,5,7,10,13,16]）。
2. **批次 rFID 损失**：对「末步重建图 vs 原图」的 batch 内 Inception 特征分布做 Fréchet 距离，
   作为逐像素 L1 的**辅助项**。

依据用户指定的会话结论（rFID 可当损失；FastFID, arXiv:2009.14075；但 rFID 是分布级统计量、
不惩罚逐图张冠李戴 ⇒ **只能当辅助项、小权重、并监控**）。本实现刻意采用：

- 冻结官方 FID-InceptionV3 pool3（2048-d、resize 299、[0,1]→[-1,1]，与 `tools/eval_rfid.py` 同权重）；
- **精确 Gram 恒等式**避免 2048×2048 的 `sqrtm`：
  `tr((Σx Σy)^{1/2}) = sum_sv(X Yᵀ)/(n-1)` ⇒ 只需 n×n SVD，且可反传；
- 用**每卡 micro-batch**（不做跨卡 all_gather，避免 DDP averaging 把梯度整体缩小 world 倍）；
- 全程 fp32（torch_musa 的 `linalg_svdvals` 没有 bf16 kernel）；
- 默认 `--rfid_weight 0`（旧路径逐位不变）；λ 用 10% 步数预热 + 10% 线性 ramp；raw rFID 逐日志上报。

## 1. 协议

| 项 | 值 |
|---|---|
| 初始化 | 复用 `tok448_v2/s1`（全宽 45ep, zs_dim=0），stage-2 SVD 瓶颈 warm start（秩-192，相对误差 0.781） |
| zs_dim / K | 192（3072 值/图） / 16 |
| 输入 | 448²，patch 32，N=196 |
| 解码 | `RecurrentOutputQueryDecoder` T=8，cumulative steps `[1,2,3,5,7,10,13,16]` |
| 损失 | 逐像素 L1（8 步平权）+ λ·rFID/2048（仅末步，start_frac=0.1, ramp_frac=0.1） |
| 优化 | AdamW lr 2e-4 cosine→1e-6, warmup 2%, wd 0.05, clip 1.0, bf16 autocast, SDPA_BACKEND=math |
| 批量 | 全局 128 = 4 卡 × micro 16 × accum 2（8 步 BPTT 在 micro 32 会 OOM，>42.8G） |
| 硬件/时长 | 4× MTT S4000 48G；3 epoch / 2850 步；**166.8 min**；VRAM 29.5–30.7G |

## 2. 结果

### 2.1 训练侧 val（1024 图，末步 F_hat）

| epoch | PSNR | SSIM | L1_norm |
|---|---|---|---|
| 1 | 17.57 | 0.351 | 0.4032 |
| 2 | 17.89 | 0.388 | 0.3754 |
| **3** | **18.04** | 0.318 | 0.3815 |

对照：同 zs_dim 的 **T=1 单步**（`tok448_zs192/s2`）val PSNR **25.20**（5k 评测 25.255）。
⇒ 8 步循环解码在这套「T=1 热启动 + 3 epoch」预算下**没有学到**，低于 per-patch-DC 基线 18.07。

### 2.2 逐步 rFID（5000 张 IN-100 val，448²，官方 FID-InceptionV3 pool3/2048）

| step | cum_z_s | values | PSNR | L1_255 | rFID |
|---|---|---|---|---|---|
| 1 | 1 | 192 | 19.41 | 18.81 | 173.30 |
| 2 | 2 | 384 | 20.13 | 17.20 | 149.84 |
| 3 | 3 | 576 | 20.29 | 16.86 | 144.06 |
| 4 | 5 | 960 | 20.45 | 16.52 | 135.36 |
| 5 | 7 | 1344 | 20.62 | 16.20 | 126.65 |
| 6 | 10 | 1920 | 20.74 | 16.00 | 120.32 |
| 7 | 13 | 2496 | 20.72 | 16.03 | 119.10 |
| **8** | **16** | **3072** | **18.11** | **21.50** | **116.31** |

**关键异常（reward hacking 证据）**：rFID 损失只加在**末步**。末步 rFID 从 step7 的 119.10 仅改善到
116.31（−2.3%），但 PSNR 从 20.72 **掉到 18.11（−2.6 dB）**、L1_255 16.03→21.50，并出现明显
**棋盘格/高频伪影**（见 `artifacts/rfid_t8/sample_grid_ep3.png`，上排原图/下排重建）。
即：λ=1.0 的 rFID 项让末步「往 Inception 特征分布上凑」，代价是逐像素保真度崩塌——正是参考结论
警告的「FID 可以刷、必须小权重 + 监控」的实例。前 7 步没有 rFID 项，PSNR 反而单调升到 20.7。

### 2.3 与 ALIT 对比（用户口径：ALIT 的 32 token ≈ 我们一步）

ALIT（arXiv:2411.02393, IN-100 Stage2 small-8, 256²）的 1D 连续 token 维度 = **12**
（Table 4「潜变量维度 12」）。故**同绝对值数**映射：ALIT-token 等价数 = 我们的 values/12。

| 我们 | values | ALIT 等价 token | 我们 rFID | ALIT-S rFID | ALIT-B rFID | 倍数 |
|---|---|---|---|---|---|---|
| step2 | 384 | **32** | 149.84 | 22.57 | 22.15 | 6.6× |
| step3 | 576 | 48 | 144.06 | 19.37 | 19.02 | 7.4× |
| step4 | 960 | 80 | 135.36 | 14.74 | 14.23 | 9.2× |
| step5 | 1344 | 112 | 126.65 | 12.50 | 11.50 | 10.1× |
| step6 | 1920 | 160 | 120.32 | 10.22 | 9.50 | 11.8× |
| step7 | 2496 | 208 | 119.10 | 8.93 | 8.68 | 13.3× |
| step8 | 3072 | 256 | 116.31 | 8.25 | 8.19 | 14.1× |
| **T=1 单步** | 3072 | 256 | **48.13** | 8.25 | 8.19 | **5.8×** |

**诚实边界（不可宣称追平/被追平）**：
- **分辨率不同**：我们 448²（每 value 覆盖 196 px），ALIT 256²（每 value 覆盖 64 px）⇒ 同绝对值数下
  我们的相对码率低 3.06×，任务更难；
- ALIT 有 **GAN stage**（Stage1 200ep + Stage2 200ep）、VQGAN 基础 tokenizer；我们是**从零连续 z_s**、
  stage-2 仅 3 epoch；
- rFID 绝对数值跨分辨率/跨模型不可直接比，本表只作**同值数量级**对照。
- 即便按最有利的 T=1 单步（48.13 vs 8.25）也差 5.8×，说明差距主要来自「重建器质量/训练预算」，
  而不是 token 数口径。

### 2.4 附：本会话前面的 zs_dim 二分（K=16，复用同一 s1，stage-2 3ep）

| zs_dim | values/图 | in100 PSNR | savoias PSNR | SSIM(1k) |
|---|---|---|---|---|
| 0（全宽 1248） | 19968 | 27.996 | 27.092 | 0.755 |
| 448 | 7168 | 27.324 | 26.451 | 0.732 |
| 192 | 3072 | 25.255 | 24.439 | 0.661 |
| 16 | 256 | 20.028 | 19.265 | 0.520 |

## 3. 结论与后续（建议）

1. **8 步循环解码不能靠 T=1 ckpt 热启动 + 3 epoch 救活**：主线循环解码在仓库里是**从零 45ep**
   训练的；本次同时叠加「SVD 瓶颈」+「单步→8 步 BPTT」两个分布漂移，3ep 远不够。下一步应
   ① 从零训练循环解码（长预算），或 ② 分阶段：先在 zs_dim=0 上把 8 步循环 decoder 训起来，再压瓶颈。
2. **rFID λ=1.0 明显过大 / 被刷**：末步「rFID −2.3%，PSNR −2.6dB + 伪影」。建议 λ∈[0.01,0.1]、
   更慢 ramp、或对 Σ̂ 做 EMA、或对 rFID 项也做逐步平权 + 与 L1 的比值裁剪；并以 PSNR/L1 作为
   reward-hacking 护栏（rFID 只作辅助，不作验收）。
3. **缺一个 no-rFID 对照**：本次无法把「8 步失败」与「rFID 有害」完全解耦（两者同时变了）。
   干净的下一步：**同一 8 步配置跑 λ=0** 作为 control，再单独加 rFID。
4. **与 ALIT 的公平比较**：应对齐分辨率（都 256²）或对齐 values/pixel；当前 448² 下只能作量级参考。

## 4. 复现

```bash
cd AssocVTC_tok448

# 8 步循环解码 + 批次 rFID（zs_dim=192, K=16, 4 卡）
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310   # MUSA 完整 torchvision
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=4 --master_port=29694 train_linattn.py \
  --data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 448 --patch 32 --num_specials 16 \
  --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
  --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
  --batch_size 128 --accum_steps 2 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
  --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 50 \
  --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8 \
  --zs_dim 192 --epochs 3 --warmup_frac 0.02 \
  --init_from /root/autodl-tmp/tok448_v2/s1/ckpt_best.pt \
  --dec_recurrent 1 --step_plan cumulative --dec_steps "1,2,3,5,7,10,13,16" \
  --rfid_weight 1.0 --rfid_dim 2048 --rfid_every 1 \
  --rfid_start_frac 0.1 --rfid_ramp_frac 0.1 --rfid_norm_dim 1 \
  --output_dir /root/autodl-tmp/tok448_zs192_t8rfid/s2

# 逐步 rFID（等价的 in-memory 版，避免 4 万张 PNG 落盘；见 fast_rfid.py）
python3 fast_rfid.py --ckpt <ckpt_best.pt> --out <out.json> --batch_size 64
```

产物：`doc/2026-11-20/artifacts/rfid_t8/`
- `zs192_t8rfid_fast.json` — 8 步逐步 PSNR/L1/rFID
- `zs192_t1_fast.json` — T=1 zs192 基线
- `train_log.jsonl` / `args.json` / `eval_val.json` — 训练日志与配置
- `sample_grid_ep3.png` — 定性（可见棋盘格伪影）

⚠️ rFID 评测口径说明：`fast_rfid.py` 与 `tools/eval_rfid.py` 用**同一 FID-InceptionV3 权重与公式**，
仅把「落 PNG → `calculate_fid_given_paths`」换成内存内累积特征统计；唯一差异是 448→299 的
resize 实现（F.interpolate vs PIL），对 FID 影响 <0.1。
