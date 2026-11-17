# 448² → 16×448（7,168 值/图）图像分词器：两阶段训练 + SAVOIAS 泛化评测

日期：2026-11-18　机器：AutoDL `sha1`（4×MTT S4000 48 GiB，MUSA / torch_musa 2.11）
代码：`/home/linaro/dsh/sr-diffusion-v3`（远端 `/root/AssocVTC_tok448`）　协议：论文阶段 1 + 阶段 3 合并

## 0. 一句话 + 主结果

按用户口径「阶段 1 全宽 45ep → 阶段 2 SVD 降到 16×448、微调 3ep」训练完成：
**最终 7,168 值/图（16 token × 448 维）在 ImageNet-100 留出集 27.32 dB、SAVOIAS 跨域 26.45 dB**，
相对全宽 19,968 值只掉 **0.67 dB**（2.79× 压缩）；同时复现了论文的「越复杂越难重建」规律。

| 指标 | 阶段 1：16×1248（全宽） | 阶段 2：**16×448（目标）** |
|---|---|---|
| 值/图 | 19,968 | **7,168** |
| ImageNet-100 val PSNR（逐图再平均 / 平均MSE） | **27.996 / 25.631 dB** | **27.324 / 25.093 dB** |
| ImageNet-100 val L1_255 | 7.494 px | 7.983 px |
| SAVOIAS OOD PSNR（逐图再平均 / 平均MSE） | 27.092 / 25.431 dB | **26.451 / 24.863 dB** |
| SAVOIAS OOD L1_255 | 7.905 px | 8.434 px |
| 跨域 gap（PSNR，IID−OOD） | −0.90 dB | **−0.87 dB** |
| 训练 | 45 ep / 42,750 步，13.05 h，**0 跳步** | +3 ep / 2,850 步，52.4 min |

**结论**：2.79× 码率压缩只换来 0.67 dB（IID）/ 0.64 dB（OOD）损失，且 OOD 泛化几乎不掉（gap < 1 dB）；
在同样 7,168 值/图的紧预算下，比历史 448²/960M+SVD 版本（24.87 dB）高 **2.46 dB**、
比论文 224²/240M 阶段 3（26.08 dB）高 **1.24 dB**。

## 1. 训练配置

| 项 | 阶段 1（全宽） | 阶段 2（目标 16×448） |
|---|---|---|
| 数据 | ImageNet-100：100 类，train 121,689 / val 5,000 | 同左 |
| 输入 / patch / N | 448² / 32 / 196 | 同左 |
| token 数 K | 16 | 16 |
| zs_dim（每 token 宽） | 0（= 1248，19,968 值/图） | **448（7,168 值/图）** |
| 编码器 | 1248×13 层 / 16 头 → 248.404M | 同左（+zs_proj/zcls_proj 1.12M） |
| 解码器 | 1440×7 层 / 16 头 → 238.276M | 238.276M（`mem_proj` 缩为 448→1440） |
| 总参数 | **486.71M** | **486.68M** |
| 优化 | AdamW，lr 2e-4→1e-6 cosine，warmup 8%，wd 0.05，clip 1.0，bf16 | 同左，warmup 2% |
| 批量 | 128（4 卡 × micro 32） | 同左 |
| 初始化 | 从零 | `--init_from s1/ckpt_best.pt`：`mem_proj` 秩-448 SVD 热启动（相对误差 0.528），FC=前 448 个右奇异向量 |
| 其他 | `SDPA_BACKEND=math`（MUSA 必需，见 §6） | 同左 |

**为什么是「阶段 1 + 阶段 3」**：用户要求「一步到位 448²→16×448」；但 11-17 已验证从零直接训窄瓶颈
在 MUSA 上必崩（真实原因是 MUSA 的 flash SDPA 反向 bug，见 `doc/2026-11-17/`）。最终采用
「全宽阶段 1 → SVD 降维阶段 3」的两阶段折中：仍只用「单步读满」这一种解码方式（`T=1`，不做 16 档循环），
只是把宽度分两步收窄——与论文三阶段中的 1+3 完全一致。

## 2. 训练曲线

阶段 1（每 epoch 验证，验证集切片 1,024 张）：

| epoch | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|
| val PSNR | 13.17 | 17.55 | 23.32 | 25.24 | 26.36 | 27.03 | 27.44 | 27.73 | 27.87 | **27.91** |

- **0 跳步**（`--skip_grad 50` 从未触发），最大 enc 分组梯度 15.96，训练全程稳定；
- 最后 5 个 epoch 只涨 0.1 dB，45 epoch 已接近饱和。

阶段 2（SVD 热启动后 3 epoch）：26.80 → 27.16 → **27.24 dB**（验证切片）。
热启动即时恢复：SVD 投影后第一步就有 26.80 dB（阶段 1 末 27.91 → 瞬时 −1.11 dB，3 epoch 微调后收回 0.44 dB）。
完整曲线：`artifacts/val_curve_s1.tsv`、`artifacts/val_curve_s2.tsv`。

## 3. 码率-质量对照（历史数据）

| 实验 | 分辨率 | 模型 | 值/图 | val PSNR |
|---|---|---|---|---|
| 224 论文基线阶段 1（K=32 全宽，A800） | 224² | 240M | 39,936 | 32.35 dB |
| **本实验阶段 1（K=16 全宽）** | 448² | 487M | 19,968 | **27.996 dB** |
| exp448 CUDA 阶段 1（K=32 全宽，A800 2 卡） | 448² | 960M | 55,808 | 28.712 dB |
| 论文 224²/240M 阶段 3（zs=224，SVD+3ep） | 224² | 240M | 7,168 | 26.08 dB |
| exp448 CUDA 阶段 3（zs=224，SVD+3ep） | 448² | 960M | 7,168 | 24.869 dB |
| **本实验阶段 2（zs=448，SVD+3ep）** | 448² | **487M** | **7,168** | **27.324 dB** |

读法：同 448²、同 7,168 值预算下，487M/K=16 的本模型比 960M/K=32 的 exp448-S3 高 **2.46 dB**；
即使对比 224²（4× 少像素）的 240M 论文 S3，本模型仍高 **1.24 dB**。
阶段 1 全宽 27.996 dB 比 exp448 的 28.712 dB 低 0.72 dB，符合「模型减半（487M vs 960M）+ K 减半」的预期。

## 4. SAVOIAS 跨域泛化

| 子集 | n | 复杂度均值±std | L1_255 | PSNR | 备注 |
|---|---:|---|---:|---:|---|
| People-Art（艺术人脸） | 420 | 46.1 ± 23.3 | 7.870 | **27.402 dB** | 最高 |
| Places（场景） | 200 | 43.0 ± 21.7 | 9.026 | 25.651 dB | |
| MSCOCO（物体） | 200 | 47.5 ± 21.0 | 9.027 | 25.257 dB | 最低 |
| **全部 820** | 820 | — | 8.434 | **26.451 dB** | IID gap = −0.87 dB |

- 跨域几乎不掉点：ImageNet-100 27.324 → SAVOIAS 26.451 dB（**−0.87 dB**）；
- 三个域里 People-Art（线稿/绘画人脸）反而最好（27.40 dB），说明模型没有过拟合 ImageNet 的自然图像统计；
- 全部 5000 张 IN-100 留出集评测：`artifacts/summary_in100_s2.json`（27.324 dB；平均 MSE 口径 25.093 dB；
  两种 PSNR 聚合口径差 2.23 dB，引用数字时须注明口径）。

## 5. 人脸/图像复杂度依赖（复现论文规律）

用 SAVOIAS 人工复杂度分数（0–100）与逐图重建误差做 Spearman 相关（`tools/analyze_savoias_single.py`，
产物 `artifacts/analysis_s2/`）。**阶段 2（16×448）**：

| 子集 | ρ(复杂度, L1) | p | 三分位 low/mid/high L1 | high−low | 相对 |
|---|---|---:|---|---:|---:|
| People-Art | **+0.474** | <1e-4 | 5.937 / 7.918 / 9.825 | **+3.888 px** | +65.5% |
| Places | **+0.375** | <1e-4 | 7.476 / 9.194 / 10.533 | +3.057 px | +40.9% |
| MSCOCO | **+0.297** | <1e-4 | 8.394 / 8.681 / 10.098 | +1.704 px | +20.3% |

对照历史 240M@224²（T=1 读满，39,936 值；`doc/2026-10-26` 已验证可复现论文数值）：
People-Art ρ=+0.518 / +3.629 px、Places +0.477 / +2.821 px、MSCOCO +0.327 / +1.504 px。

- **规律被复现**：三个域 ρ 都显著为正、三分位单调上升、high−low 差为正，且域间排序（Art > Places > COCO）一致；
- 本模型的 high−low **更大**（+3.888 vs +3.629；+3.057 vs +2.821；+1.704 vs +1.504），
  说明压缩后复杂图像被「伤」得更重；ρ 略低（Art 0.474 vs 0.518，Places 0.375 vs 0.477），
  即排序一致性稍弱但退化幅度更大；
- **阶段 2 vs 阶段 1**：压缩把复杂度敏感度放大了——People-Art high−low 由 +3.505（全宽）升到 +3.888 px，
  Places +2.769 → +3.057，MSCOCO +1.531 → +1.704；且各子集平均 PSNR 掉幅为
  Art −0.74 / Places −0.58 / COCO −0.49 dB ⇒ **压缩的代价主要落在复杂图像（尤其人脸区域）上**。

## 6. 关键前提：MUSA flash-SDPA 反向 bug

本 run 全程 `SDPA_BACKEND=math`。若用默认 flash 后端，同一配置会在 ep2 起梯度爆到 1e4–1e5、val 崩到 12 dB
（已定位为 `head_dim % 8 != 0` 时 muDNN flash 反向 dQ/dK 错误；本模型 head_dim 78/90 全部命中），
详见 `doc/2026-11-17/REPORT_musa_sdpa_bwd.md` 与上游提交 torch_musa#162。
代价：端到端 **+8%**（1.019 → 1.098 s/step）。

## 7. 复现

```bash
# 阶段 1（4 卡，13 h）
SDPA_BACKEND=math torchrun --nproc_per_node=4 train_linattn.py \
  --data_dir /root/autodl-tmp/datasets/imagenet100 --output_dir $OUT/s1 \
  --img_size 448 --patch 32 --num_specials 16 --zs_dim 0 \
  --enc_attn std --enc_dim 1248 --enc_depth 13 --enc_heads 16 \
  --dec_dim 1440 --dec_depth 7 --dec_heads 16 --dec_recurrent 0 --pixel_hidden 1024 \
  --epochs 45 --batch_size 128 --accum_steps 1 --lr 2e-4 --min_lr 1e-6 \
  --weight_decay 0.05 --warmup_frac 0.08 --grad_clip 1.0 --amp 1 --seed 0 \
  --val_per_class 50 --val_batches 8 --save_every_epochs 1 --save_opt 0 --num_workers 8

# 阶段 2（SVD 热启动，52 min）
SDPA_BACKEND=math torchrun --nproc_per_node=4 train_linattn.py ... --zs_dim 448 \
  --epochs 3 --warmup_frac 0.02 --init_from $OUT/s1/ckpt_best.pt --output_dir $OUT/s2

# 评测（IID / OOD 各 1 条命令，见 artifacts/run_eval.sh）
SDPA_BACKEND=math python eval_stepwise_linattn.py --ckpt $OUT/s2/ckpt_best.pt \
  --data_dir /root/autodl-tmp/datasets/imagenet100 --val_per_class 50 \
  --out eval/summary_in100_s2.json --dump_per_image eval/per_image_in100_s2.json
SDPA_BACKEND=math python eval_stepwise_linattn.py --ckpt $OUT/s2/ckpt_best.pt \
  --data_dir_ood /root/autodl-tmp/savoias --domain savoias \
  --out eval/summary_savoias_s2.json --dump_per_image eval/per_image_savoias_s2.json

# 复杂度分析
python3 tools/analyze_savoias_single.py --per_image eval/per_image_savoias_s2.json \
  --gt_dir <gt 目录: gt_art.npy/gt_scenes.npy/gt_objects.npy> --out analysis_s2 --label "tok448-s2 16x448"
```

## 8. 产物

| 路径 | 内容 |
|---|---|
| `/root/autodl-tmp/tok448_v2/s1/`、`s2/` | ckpt_best.pt / ckpt_last.pt（各 1.95 GB）、train_log.jsonl |
| `/root/autodl-tmp/tok448_v2/eval/` | 4 份 summary + 4 份 per-image dump + 日志（远端） |
| `doc/2026-11-18/artifacts/` | summary×4、per_image_savoias×2、val_curve×2、run_eval.sh、analysis_{s1,s2}/（tables.md、summary.json、图） |
| `doc/2026-11-17/` | MUSA 崩溃定位报告、上游 issue 草稿与已提交评论存档 |

## 9. 局限

- 单 seed（seed 0）、单次 run；未做 seed 稳定性重复。
- 阶段 2 只训 3 epoch（论文口径）；更长的微调可能再收 0.1–0.3 dB。
- SAVOIAS 评测是 448² resize+center-crop（与 IN-100 val 同口径），非原始分辨率。
- 复杂度分析用的是 SAVOIAS 的人工 0–100 分数（非感知复杂度模型），结论限于该代理。
- MUSA 上 SDPA 必须走 math 后端；若换环境/换 torch_musa 版本需复核（上游 bug 修复后可关掉该开关）。
