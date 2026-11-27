# v2.0 (224²) 阶段3 换成「对齐 LlamaGen VQ-16 离散码的交叉熵」— 实验报告

- **实验日期**：2026-11-26（s1）→ 11-27（s2、s3 开跑）→ 11-28 05:34（s3 结束）/ 06:55（评测+汇总完成）
- **远端机器**：4×MTT S4000 48GB，DDP4（MCCL），torch 2.11 + torch_musa 2.11.4，bf16 混合精度，`SDPA_BACKEND=math`
- **配套产物**：`artifacts/tables/SUMMARY.md`、`artifacts/tables/TABLE.md`、`artifacts/figures/fig_step_rfid.png`、`artifacts/figures/fig_step_psnr.png`、`artifacts/results/*.json`、`artifacts/logs/*`；坑与修复见 `artifacts/logs/incidents.md`

---

## 0. TL;DR

- **s3-CE（阶段3 = 预测 LlamaGen VQ-16 离散码，逐步 CE）末步 rFID = 21.117**（n=5,000），对照：
  - tokenizer 物理下界（P0，用**GT 码**解码）**8.766** ⇒ 与下界差 **12.35**；说明 3 个 epoch 的 CE 训练只把 latent 推到"码可预测"的很浅层（val top-1 仅 13.62%），离"码选对"还很远。
  - 本机 s2 像素臂 **19.742** ⇒ s3-CE 比像素臂**差 1.38**，但注意 s3 多了一层 VQ 渲染器（见下面的对照臂）。
- **训练期 token 指标**：s3 末次 val `tok_ce = 4.5295`、`tok_acc = 13.62%`（16384 类随机猜中率 0.006%）；逐步曲线单调改善：CE 6.86→4.07、acc 3.9%→16.0%（t=1→16）。
- **三阶段主结果**：s1 rFID **6.657** / PSNR 32.107；s2 rFID **19.742** / PSNR 28.058；s3-CE rFID **21.117** / PSNR 20.078。
- **与论文 v2.0 重跑对照**（同协议但跨机）：s1 论文 6.13 → 本机 6.657（+0.53）；s2 论文 19.97 → 本机 19.742（−0.23）。**s1/s2 与论文量级一致**，可作为 s3 的可靠对照。
- **决定性的中间对照（§4.1）**：把 s2 像素臂的输出再套一遍 VQ 码本，rFID 从 **19.742 劣化到 23.290（+3.55）**。s3-CE 的 **21.117 比这个朴素 VQ 往返好 2.17** ⇒ CE 目标**确实**把 latent 推得更贴码本（不是被渲染器白拖累），只是 3 ep 的码预测精度（top-1 13.6%）还不足以抵消量化的固有代价。

---

## 1. 方法与口径

### 1.1 统一配置（三阶段共用）

| 项 | 值 |
|---|---|
| 数据 | ImageNet-100：train 121,689 / 确定性 val 5,000（`val_per_class=50`，与训练同划分） |
| 输入 | img_size 224 / patch 16 → N=196；K=`num_specials`=32 |
| 编码器 | `enc_attn=std`（softmax）1248d × 13L × 16h（**与论文一致，无 QK-Norm**） |
| 解码器 | `dec_attn=std` 1440d × 7L × 16h；像素头 `pixel_hidden=1024`；输出查询式（查询行 k ↔ patch k，row-major） |
| 优化 | lr 2e-4 cosine → `min_lr` 1e-6；`warmup_frac` 0.08；`weight_decay` 0.05；`grad_clip` 1.0 |
| 精度/并行 | bf16 autocast + `amp=1`；DDP4（MCCL，`DDP_IMPL=patched`）在 4×MTT S4000 上 |
| **数值后端** | **`SDPA_BACKEND=math`**（MUSA 上 bf16 后向必需；SDPA 的数学等价实现路径，**不是架构/超参差异**，墙钟约 +6%，见 §2b） |
| 步数口径 | `micro_bs = batch/(accum·world)`，`steps_per_epoch = len(loader)//accum`；DistributedSampler `num_samples=ceil(121689/4)=30423`、`drop_last=True` |

### 1.2 三阶段配置与步数

| 阶段 | T / 读出方式 | 损失 | epochs | 等效 bs | accum | micro/卡 | steps/ep | **total_steps** | init_from | 参数量 |
|---|---|---|---|---|---|---|---|---|---|---|
| **s1（阶段1）** | T=1，单步读满 K=32 | 像素 L1 | 45 | 128 | 1 | 32 | 950 | **42,750** | 从零 | 481.496M |
| **s2（阶段2）** | T=16，fixed/block2（每步 2 个 z_s，`cum_z_s=2t`） | 像素 L1 | 6 | 64 | 2 | 8 | 1,901 | **11,406** | `s1/ckpt_best.pt` | 481.543M |
| **s3-CE（本实验）** | T=16，同 s2 | **VQ-16 离散码逐步 CE** | 3 | 64 | 2 | 8 | 1,901 | **5,703** | `s2/ckpt_best.pt` | 498.898M |

> s2/s3 的 `accum=2`（micro 8/卡）是**显存兜底**：等效批量 64 与 steps/epoch 1,901 都不变（数学等价，梯度 = 2 个 micro 的均值）。实测代价是吞吐 **21.0 → 18.3 img/s（−13.7%）**，见 `incidents.md`。

- s3 参数拆解（相对 s1）：`decoder.mem_pos` **+47,520**（循环解码器 memory 位置编码）+ token 头 **+18,269,184** + zs 瓶颈 `zs_proj/zcls_proj` **+559,552** − `decoder.mem_proj` 缩小 **−1,474,560** ⇒ **498,897,536（498.898M）**；s2 相对 s1 仅 `mem_pos` +47,520 ⇒ **481,543,360（481.543M）**。s3 相对 s2 净 **+17.354M**。

### 1.3 三阶段命令（抄自实跑的 `code/v20_224_run_s*.sh` / `chain.sh`；`ST=/root/autodl-tmp/v20_224`）

统一环境：`PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310`、`PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True`、`HF_HUB_OFFLINE=1`、**`SDPA_BACKEND=math`**；前缀 `MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched`。

```bash
# ── s1（--master_port 29779，45 ep）──
torchrun --nproc_per_node=4 --master_port=29779 train_linattn.py \
 --data_dir /root/autodl-tmp/datasets/imagenet100 --output_dir "$ST/s1" \
 --img_size 224 --patch 16 --num_specials 32 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --pixel_hidden 1024 \
 --dec_recurrent 0 --dec_grad_ckpt 1 \
 --epochs 45 --batch_size 128 --accum_steps 1 \
 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --warmup_frac 0.08 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 40 --log_interval 50 \
 --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 0 --save_opt 0 --num_workers 16

# ── s2（--master_port 29741，6 ep，从 s1 热启动）──
torchrun --nproc_per_node=4 --master_port=29741 train_linattn.py \
 ... --dec_recurrent 1 --step_plan fixed --block 2 \
 --dec_steps "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" --dec_grad_ckpt 1 \
 --init_from "$ST/s1/ckpt_best.pt" \
 --epochs 6 --batch_size 64 --accum_steps 2 \
 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --warmup_frac 0.08 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 80 --log_interval 50 \
 --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 0 --save_opt 0 --num_workers 16 \
 --save_every_steps 300

# ── s3-CE（--master_port 29751，3 ep，从 s2 热启动）──
torchrun --nproc_per_node=4 --master_port=29751 train_linattn.py \
 ... --dec_recurrent 1 --step_plan fixed --block 2 \
 --dec_steps "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" --dec_grad_ckpt 1 \
 --zs_dim 224 --init_from "$ST/s2/ckpt_best.pt" \
 --token_ce 1 --vq_ckpt /root/autodl-tmp/vq/llamagen_vq16_c2i.pt \
 --vq_codebook_size 16384 --vq_embed_dim 8 --token_ce_init_pixel 1 \
 --epochs 3 --batch_size 64 --accum_steps 2 \
 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --warmup_frac 0.08 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 80 --log_interval 20 \
 --sample_interval 0 --save_every_epochs 1 --save_epoch_ckpts 0 --save_opt 0 --num_workers 16 \
 --save_every_steps 300
```

评测（chain 末尾自动执行）：`fast_rfid.py --num_images 0`（全 5,000 val），s1/s2 `--render pixel`、s3 `--render vq`
（每步 argmax → `vq.decode_ids` 渲染）；指标 rFID（FID-InceptionV3 pool3 2048d、resize 299、uint8 量化）+ PSNR + L1_255。

---

## 2. 与论文 v2.0 的差异清单

> a) 是**实验变量**（阶段3 换损失）；b) 是**运行环境**（SDPA 数值后端，非架构差异）；c) 是**实现细节**（不影响协议）。
> **结论：除阶段3 的损失目标外，架构与超参与论文 v2.0 一致**（encoder 244,429,536；total 481.496M；运行日志确认）。

**a) 阶段3 的损失：像素 L1 → LlamaGen VQ-16 离散码的逐步交叉熵（本实验唯一的实验变量）。**
每个读出步把 `norm_out(Y_t)`（B,N,dec_dim=1440）经 token 头映射成 **V=16384** 类 logits，对 tokenizer 对**同一增广视图原图**输出的 (N=196) 个码算 CE：**每步 196 个位置平权 + T=16 步平权**（`mean_t mean_{B,N} CE`），CE 在 fp32 下算。目标码与模型位置严格同序（id 下标 k = 行主序 patch k）。
token 头 = `Linear(1440,1024)+GELU+Linear(1024,16384)`，**18,269,184 参数**；其**第一层（1,475,584 参数）从 s2 像素头 MLP 第一层热启动**，第二层随机初始化。像素头保留在 s3 中但**不参与损失**（仍参与前向，梯度为 None）。

**b) MUSA 数值后端：`SDPA_BACKEND=math`（不是架构差异）。**
默认 SDPA 后端（flash/mem-efficient）在 MUSA + bf16 下**反向数值不稳**：同等权重/批次下编码器梯度可大 30–230 倍。证据：
- 同机同配置 21-min 对照（`E1_224_ctl` vs `T1_224_math`，224²/K=32/240M/T=1/bs128/lr2e-4）：默认后端 skip 76→175、val PSNR 13.24→11.16→10.77（退化）；`math` 后端 skip 全程 0、val 13.27→13.97→14.56（持续变好）。
- 本次两次复现：(i) 默认后端在 ~step 2200 崩（gn 0.65→7.9→68→252→4.5e3→5.2e4，skip 100%，val PSNR 12.84→11.62）；(ii) 中途加 QK-Norm 的试验臂在 ~step 1700 仍崩（gn 2.37→96.8）⇒ **QK-Norm 不是根因，已撤销**，架构完全回到论文原样。
- 修复：改用 `SDPA_BACKEND=math`（数学等价实现路径，不改架构/超参），墙钟约 +6%。

**c) 实现细节（影响可复现性，不是协议差异）。**
- `--dec_grad_ckpt 1` 对 `dec_attn=std` **是 no-op**（梯度检查点只在 `LinearAttentionDecoder._run`）⇒ T=16 的 BPTT 实际无检查点；s2/s3 的显存兜底靠 `--accum_steps`（等效批量与步数不变）。
- 训练侧 VQ 目标码**必须在 `autocast(enabled=False)` 下算**：`torch.no_grad()` 不关 autocast，bf16 下 conv 走 bf16 会使 **~5.8%（实测 91/1568）id 与 fp32 不同**，而 val/P0 都用 fp32 码 ⇒ train/val 标签口径不一致。已显式关掉。
- **续跑带来的两个已知偏差**（本实验发生过 4 次 MUSA 崩溃，均自动续跑）：① `--save_opt 0` 不存优化器状态 ⇒ 续跑时 Adam 一/二阶矩重置；② 续跑后 epoch 末 val 的步点会平移（s2 实际 val 落在 7001/8902/10803/11406，本应是 5703/7604/9505/11406）。总步数、LR 调度、数据量与最终 val 口径都不受影响。

---

## 3. 判据

- **物理下界（P0 天花板）**：LlamaGen VQ-16 在 224²/IN-100 val（n=5,000）用 **GT 码**重建：**rFID = 8.766、PSNR = 20.477、L1_255 = 17.109**（与 `fast_rfid.py` 同协议）⇒ s3-CE 的 rFID 不可能低于 8.766。
- **判据**：s3-CE 末步 rFID 应落在 **[8.766, 本机 s2 末步 rFID]** 之间；显著 > s2 ⇒ CE 目标在 3 ep 内没换到端到端增益。
- PSNR **不可**在 s2 与 s3-CE 之间横比（渲染器不同：像素头 vs VQ decoder）。

---

## 4. 主结果（末步，n=5,000 val，同协议 rFID）

| 臂 | 末步 rFID ↓ | 末步 PSNR ↑ | 末步 L1_255 ↓ | 参数量 |
|---|---|---|---|---|
| s1 T=1（像素头，读满 K=32） | **6.657** | 32.107 | 4.756 | 481.496M |
| s2 T=16（像素头） | **19.742** | 28.058 | 7.052 | 481.543M |
| **s3-CE T=16（token 头→VQ decoder）** | **21.117** | 20.078 | 17.846 | 498.898M |
| s3-CE ckpt_last | 21.117 | 20.078 | 17.846 | 498.898M |
| **P0 天花板（VQ-16, GT 码, n=5000）** | **8.766** | **20.477** | **17.109** | — |
| 论文 v2.0 历史读数（跨机，仅参照） | s1 6.13 / s2 19.97 | s1 32.35 / s2 27.94 | — | — |

- s3 `ckpt_best` 与 `ckpt_last` 是同一个模型（best 出现在末步），两行数字相同属预期。
- **训练内 val（阶段3-CE 口径）**：s3 末次 `tok_ce = 4.5295`、`tok_acc = 13.62%`、`T = 16`。
- **VQ 往返对照臂**：见 §4.1。

### 4.1 对照臂：把 s2 的像素重建再套一遍 VQ 码本

目的：把「VQ 渲染器本身的代价」从「CE 对齐的效果」里拆出来 —— 若 s3-CE 的 rFID 明显好于 `rec_pixel → VQ roundtrip`，说明 CE 训练让 latent 更贴码本，而不只是被渲染器拖累。

| 臂 | 末步 rFID ↓ | 末步 PSNR ↑ |
|---|---|---|
| s2 像素重建（`rec_pixel`） | 19.742 | 28.058 |
| s2 像素重建 → VQ 二次量化（`rec_pixel_vq`） | **23.290** | 21.35 |
| s3-CE（预测码 → VQ 解码） | 21.117 | 20.078 |

**读法（本实验最关键的一条对照）**：`rec_pixel_vq` = 19.742 → **23.290**（+3.55，PSNR 28.06→21.35），即"什么也不做、只把像素臂的输出过一遍 VQ"要付 **+3.55 rFID** 的渲染代价。而 s3-CE 是 **21.117** —— **比这个朴素 VQ 往返好 2.17**，但比不过像素臂本身（19.742）。
⇒ 结论：**CE 目标确实把 z_s 推向了"码本友好"的区域（净收益 +2.17），但 3 ep 的码预测精度不足以抵消 VQ 渲染的固有代价（+3.55），所以端到端反而比像素臂差 1.38。** 若把 CE 训练拉长到码预测更准（val top-1 从 13.6% 往上），s3-CE 有机会穿过 19.742 追上像素臂。

---

## 5. 逐步曲线

`cum_z_s = 2t`（每步 2 个 z_s）：

| 步 t | 累计 z_s | s2 rFID | s3-CE rFID | s3-CE PSNR |
|---|---|---|---|---|
| 1 | 2 | 103.42 | 89.88 | 16.35 |
| 2 | 4 | 76.31 | 61.98 | 17.56 |
| 3 | 6 | 57.41 | 39.57 | 18.65 |
| 4 | 8 | 47.55 | 29.88 | 19.51 |
| 5 | 10 | 40.90 | 27.30 | 19.74 |
| 6 | 12 | 34.99 | 24.53 | 19.87 |
| 7 | 14 | 32.01 | 22.85 | 19.97 |
| 8 | 16 | 30.11 | 22.34 | 20.00 |
| 9 | 18 | 27.89 | 21.80 | 20.05 |
| 10 | 20 | 25.93 | 21.74 | 20.05 |
| 11 | 22 | 22.32 | 21.45 | 20.07 |
| 12 | 24 | 21.60 | **21.13** | 20.07 |
| 13 | 26 | 20.29 | 21.34 | 20.07 |
| 14 | 28 | 19.85 | 21.18 | 20.08 |
| 15 | 30 | 19.69 | 21.19 | 20.08 |
| 16 | 32 | **19.74** | 21.12 | 20.08 |

**观察**：
1. 两条曲线都**单调下降**，说明循环读出（每步再读 2 个 z_s）确实在持续改善重建——这与 v2.0 的结论一致。
2. **前 12 步 s3-CE 一直优于 s2**（t=1 时 89.9 vs 103.4，差距最大；t=12 首次反超前 s3 是 21.13 vs 21.60）。即：**离散码目标让早期读出更快收敛**（每次少读 z_s 就能表达更多语义）。
3. 但 s3-CE 在 t≈12 后**基本饱和到 21.1**（PSNR 也钉在 20.07–20.08），而 s2 继续降到 19.74 —— 因为 s3 的输出被 VQ 渲染器"锁"在 P0 附近的 PSNR（20.48 是 GT 码的上界，s3 只比它低 0.40 dB），**rFID 的剩余差距来自码预测错误（top-1 13.6%）而非渲染能力**。
4. 训练内逐步 CE / acc（val）：`[6.86, 5.91, 5.25, 4.65, 4.43, 4.31, 4.20, 4.17, 4.12, 4.11, 4.09, 4.08, 4.08, 4.07, 4.07, 4.07]` / `[3.9%, 7.3%, 9.7%, 12.3%, 13.4%, 14.3%, 15.0%, 15.2%, 15.6%, 15.6%, 15.8%, 15.9%, 16.0%, 16.0%, 16.0%, 16.0%]` ⇒ 第 12 步后同样饱和，与 rFID 的饱和点吻合。

图片见 `artifacts/figures/fig_step_rfid.png`、`fig_step_psnr.png`。

---

## 6. 结论与下一步

1. **verdict**：阶段3 从"还原像素"换成"对齐 VQ-16 离散码的 CE"，在**同样的 3 epoch 预算**下：
   - 早期读出（t≤12）比像素臂更好，收敛更快；
   - 末步 rFID **21.12**，比像素臂 19.74 差 1.38，比 tokenizer 下界 8.77 差 12.35；
   - 瓶颈明确是**码预测精度**（val top-1 13.6%，CE 4.53 vs 若完美预测应接近 0），不是渲染器（PSNR 已逼近 P0 的 20.48）。
1b. **量化 vs 对齐的拆分**：VQ 渲染本身的代价是 +3.55 rFID（§4.1），CE 训练回收了其中 2.17；剩下的 1.38 缺口来自码预测错误。若延长训练/改码预测头，缺口可以继续压。
2. **为什么 3 ep 不够**：CE 从 ln(16384)=9.70 起步，3 ep 只到 4.53；且 s3 是**全参微调**，encoder 同时要保留像素重建能力，任务更重。若要逼近下界，需要更长训练（或对 token 头用更大 lr、或冻 encoder 只训头、或加 196 位置的码间相关性建模）。
3. **可复现性**：代码快照在 `code/`；三阶段命令、步数、参数拆分、MUSA 坑与修复（`SDPA_BACKEND=math`、epoch 末 val 的 barrier 崩溃、续跑口径、VQ 目标的 fp32 计算）都写在 `artifacts/logs/incidents.md`。

---

## 7. 诚实边界

- **单种子**（seed=0）：每个阶段 1 条轨迹，无多种子方差/置信区间。
- **跨机差异**：本机 4×MTT S4000 + bf16 + DDP4 + `SDPA_BACKEND=math`，与论文 A800 单卡存在跨机/精度差异（s1 PSNR 32.107 vs 论文 32.35）。rFID 我只在**同机同协议**内做对照，不把绝对值与论文 FID 混比。
- **s3 只训 3 ep**（与 v2.0 阶段3 同），明显欠训；未做更长训练、CE 温度/标签平滑/类别重加权等超参搜索。
- **渲染器不同**：s2 用像素头输出连续像素，s3-CE 用 `argmax id → VQ decoder` 输出，**PSNR 不可横比**；rFID 也只能作为"整条投放路径的端到端质量"对照，§4.1 的 VQ 往返臂是为此设的中间对照。
- **rFID 是集合级配对度量**（InceptionV3 pool3, n=5,000、100 类、224²）：只在与同协议、同 n 时可比；n≈2,000 时同分布抽样偏差约 +22–27。
- **参数量**：s3 因 token 头 +18.3M、zs 瓶颈 +0.56M − mem_proj 缩小 1.47M、+mem_pos 0.05M ⇒ **498.898M（vs s2 481.543M，净 +17.354M）**；未做等参重配平。
- **续跑**：4 次 MUSA 崩溃后自动续跑（详见 `incidents.md`），Adam 矩重置 + val 步点平移，对最终指标影响有限但不可忽略不计。
