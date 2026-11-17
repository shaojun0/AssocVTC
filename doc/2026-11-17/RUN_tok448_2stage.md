# 448² 图像分词器：两阶段训练（全宽 16×1248 → SVD 降维 16×448）

日期：2026-11-17　机器：AutoDL `sha1`（4×MTT S4000 48 GiB，MUSA / torch_musa 2.11）

## 1. 目标（用户口径）

训练一个 **448×448 → 16 token** 的图像分词器（`SRPhase1LinAttn`, T=1 单步读满），
最终可用码率 **16 × 448 = 7,168 值/图**，编码器 ~240M；随后用 SAVOIAS 测跨域泛化。

## 2. 为什么从「一步到位窄瓶颈」改成两阶段

`2026-11-16` 直接训 `K=16, zs_dim=448`（7,168 值/图，从零初始化）在 ImageNet-100 上**崩**：

| 指标 | 一步到位窄瓶颈（`/root/autodl-tmp/tok448_k16`） | 全宽参照（`doc/2026-10-24/.../m240_t1.jsonl`） |
|---|---|---|
| grad_norm 中位 / p90 / 最大 | ≈0.3 → 51 → 254 → 603 → 1210 | 0.20 / 0.51 / **9.33** |
| `--skip_grad 50` 跳步 | 1,507 / 3,500（43%） | **0 / 42,750** |
| val PSNR | 13.31 → 12.77 → 12.32 dB（倒退） | 13.26 → … → 24+ dB（单调涨） |

诊断（`DIAG_GRAD=1`）：尖峰最先最大出现在 `dec_mem` / `dec_stack0_2`（解码器侧）,
而同配置下 `zs_dim=0`（全宽）梯度始终健康。

> ⚠️ **2026-11-17 更正**：当时的结论（「窄瓶颈信息压力」）是**错的**。后续在**同一台 MUSA 机器**
> 上跑全宽（`zs_dim=0`）对照、以及 224² CUDA 参照配置，**同样崩**；根因是 MUSA 上
> bf16 SDPA（flash/mem-efficient）反向数值坏，与宽度/K/分辨率无关。
> 详见 `REPORT_musa_instability.md`；修复 = `SDPA_BACKEND=math`。
> 论文三阶段（全宽→循环→降维）仍然是正确的路线，但不是因为"窄瓶颈必然崩"。

因此改为用户确认的两阶段：

1. **阶段 1（≈12 h）**：全宽 `K=16 × enc_dim 1248`（19,968 值/图），从零 45 epoch；
2. **阶段 2（≈50 min）**：`zs_dim=448`，用阶段 1 `ckpt_best` 的 `decoder.mem_proj.weight`
   做秩-448 SVD 热启动（Eckart–Young），全参数微调 3 epoch ⇒ 最终 16×448。

（宽度选择：`1744` 宽对应 `doc/2026-11-04` 的 960M 设计；本 run 按用户最初「240M 编码器」
口径取 `1248` 宽，两阶段合计 ~13 h，4×48 GiB 显存安全。）

## 3. 配置

| 项 | 阶段 1 | 阶段 2 |
|---|---|---|
| 数据 | ImageNet-100（100 类 / 126,689 张，train 121,689 / val 5,000） | 同左 |
| 输入 / patch | 448²，patch 32（N=196） | 同左 |
| token 数 K | 16 | 16 |
| zs_dim（每 token 宽度） | 0（= 1248，全宽） | **448** |
| 编码器 | 1248×13 层 / 16 头，247.28M | 同左 |
| 解码器 | 1440×7 层 / 16 头，239.43M（`StandardAttentionDecoder` T=1） | 同左 |
| 总参数 | 486.71M | 486.68M |
| 值/图 | 19,968 | **7,168** |
| 优化 | AdamW，lr 2e-4 → 1e-6 cosine，warmup 8%，wd 0.05，clip 1.0 | AdamW，lr 2e-4 → 1e-6，warmup 2%，其余同左 |
| 批量 | 128（4 卡 × micro 32，等效步骤不变） | 同左 |
| epoch | 45（42,750 步） | 3（2,850 步） |
| 精度 / 后端 | bf16 autocast；MUSA + `mccl`，`DDP_IMPL=patched` | 同左 |
| 初始化 | 随机（`--zs_dim 0`） | `--init_from s1/ckpt_best.pt`（SVD 秩-448 热启动） |

## 4. 命令（远端 `/root/AssocVTC_tok448`，驱动脚本见 `run_tok448_2stage.sh`）

```bash
COMMON="--data_dir /root/autodl-tmp/datasets/imagenet100 --img_size 448 --patch 32 --num_specials 16 \
 --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
 --dec_dim 1440 --dec_depth 7 --dec_heads 16 --dec_recurrent 0 --pixel_hidden 1024 \
 --batch_size 128 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.05 --grad_clip 1.0 \
 --amp 1 --seed 0 --val_per_class 50 --val_batches 8 --log_interval 50 \
 --sample_interval 0 --save_every_epochs 1 --save_opt 0 --num_workers 8"

# 阶段 1
MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched DIAG_GRAD=1 \
 torchrun --nproc_per_node=4 --master_port=29641 train_linattn.py $COMMON \
 --zs_dim 0 --epochs 45 --warmup_frac 0.08 --output_dir /root/autodl-tmp/tok448_2stage/s1

# 阶段 2
MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched DIAG_GRAD=1 \
 torchrun --nproc_per_node=4 --master_port=29642 train_linattn.py $COMMON \
 --zs_dim 448 --epochs 3 --warmup_frac 0.02 \
 --init_from /root/autodl-tmp/tok448_2stage/s1/ckpt_best.pt \
 --output_dir /root/autodl-tmp/tok448_2stage/s2
```

## 5. SVD 热启动的验证

`train_linattn._init_bottleneck_warm_start()`：对旧 `mem_proj.weight` W(dec,enc) 做 SVD，
`zs_proj/zcls_proj.weight = Vh[:448]`、`decoder.mem_proj.weight = U[:, :448]·diag(S[:448])`
⇒ 初始 `mem_proj(FC(z))` = W 的最优秩-448 近似。

远端迷你冒烟（0.96M 模型，CPU 口径）端到端跑通两阶段：阶段 1 收 `psnr=14.85 dB`，
阶段 2 热启动后首轮 `psnr=14.90 dB` ⇒ **投影几乎不掉点**，形状重初始化与 missing-key 白名单也正确。

## 6. 时间线与产物

- 阶段 1：14:46 起，实测 ~0.93 s/step（118 img/s）⇒ 950 步/epoch ≈ 15 min，45 epoch ≈ 11.2 h，预计 **~02:00 结束**；
- 阶段 2：~15 min/epoch × 3 ⇒ 约 02:00–02:50；
- 产物：`/root/autodl-tmp/tok448_2stage/{s1,s2}/{ckpt_best.pt,ckpt_last.pt}`, 日志 `s1.log`/`s2.log`, 状态 `status`。

## 7. 评测计划（阶段 2 完成后）

1. ImageNet-100 val：`eval_stepwise_linattn.py --data_dir … --dump_per_image`
   （报告 7,168 值/图的 PSNR/SSIM，以及逐图曲线）；
2. SAVOIAS 跨域：`--data_dir_ood /root/autodl-tmp/savoias --domain savoias --dump_per_image`
   （Art 420 / Scenes 200 / Objects 200，gt 在 `/root/autodl-tmp/savoias/gt/`）；
3. `tools/analyze_savoias_single.py` 复现论文的「人脸复杂度 vs 重建误差」相关性
   （People-Art ρ=+0.518/+3.629px、Places ρ=+0.477/+2.821px、MSCOCO ρ=+0.327/+1.504px 为已验证参照）；
4. SAVOIAS 与 ImageNet-100 的 PSNR 差 = 泛化 gap，报告落在 `doc/2026-11-17/`。
