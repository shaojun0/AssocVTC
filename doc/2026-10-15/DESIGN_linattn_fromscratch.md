# DESIGN — model_v2 实验版：无预训练编码器（线性注意力编码器 + 普通注意力解码器）

> 日期：**2026-10-15**
> 代码：`model_v2_linattn.py` / `data_linattn.py` / `metrics_linattn.py` / `train_linattn.py` / `eval_linattn.py` / `probe_linattn.py` / `plot_linattn.py` / `run_linattn.sh`
> 数据集：**ImageNet-100**（AutoDL 公共盘 `/root/autodl-pub/ImageNet100/imagenet100.zip`）
> 主线 `model_v2.py` **逐位未改**（本实验版是**并列新增**，不 import 它、不影响它）。

---

## 1. 用户口径（本实验版的验收标准）

> 「把 `https://github.com/shaojun0/SR-Diffusion-v3` 里的 `model_v2` 增加一个实验性的版本，
> 这个版本**不再使用预训练编码器**，而是自定义一个小的**线性注意力机制**的编码器，
> 参数量 **30 几 M** 就行，解码器是**同等体量的普通注意力机制**的 30 几 M 的 transformer。
> **输入是 224×224，输出也是 224×224**。」（2026-10-15，用户原话，训练数据用 ImageNet-100）

拆成可验收的四条：

| # | 要求 | 落地 | 实测 |
|---|---|---|---|
| A | 不用预训练编码器 | 编码器从零训练（随机初始化 Conv patch embed + 13 层线性注意力） | 无任何预训练权重加载 |
| B | 线性注意力编码器 ~30 几 M | `LinearAttentionEncoder(dim=448, depth=13)` | **31.909 M** |
| C | 解码器 = 同等体量普通（softmax）注意力 transformer | `StandardAttentionDecoder(dim=512, depth=7, nn.TransformerDecoder)` | **31.173 M**（dec/enc = 0.977） |
| D | 224×224 → 224×224 | patch=16 ⇒ N=196，PixelHead 出 16×16×3 像素 patch | 见 `python model_v2_linattn.py` 自检 |

---

## 2. 与主线 `model_v2` 的差异（唯一自变量 = 编码器来源 + 注意力形式/体量）

| | 主线 `model_v2.py` | 本实验版 `model_v2_linattn.py` |
|---|---|---|
| 编码器 | DINOv2-large（预训练，304M，register 式拼 specials 进 24 层） | **从零** 13 层**线性注意力**主干，31.9M |
| 编码器注意力 | softmax 全双向 | **线性注意力**（φ(x)=elu(x)+1 的核形式，O(N) 聚合） |
| 解码器 | `OutputQueryDecoder`（2 层，**循环读出** BPTT carry，~2M） | **`nn.TransformerDecoder` 7 层普通 softmax 注意力**，31.2M |
| 解码读出 | 循环多步（每步读一块 z_s，逐步细化） | **单步并行**（|T| = 1）—— 见 §5「刻意偏离」 |
| patch | 14（DINO 原生） | 16 |
| 分辨率 | 224 及其倍数 | **224 → 224**（用户指定） |
| 数据 | construction_site（工地图） | **ImageNet-100** |

**保留不变的主线语义**（保证两条线可直接对照）：

- **register 式 token 压缩**：序列 = `[cls(1); specials(K); patches(N)]`，specials 与 patches
  在同一主干里双向交互 ⇒ `z_s` 是"从像素里学出来的压缩"，不是无内容输入的可学习常量
  （Darcet et al., register token 式; 主线同款）。
- **K 与 N 解耦**：`num_specials=K` 是压缩 token 数，`N=196` 是输出 patch 数；**K<N 即真压缩**
  （主线文档的核心杠杆）。
- **输出查询式解码**（Perceiver IO / BLIP-2 口径）：查询基行 k ↔ patch k，行数恒为 N。
- **损失 = 像素 L1 直接预测**（归一化空间），输出 dict 键与 `SRPhase1V2.forward` **完全一致**
  （`loss/recon/F_hat/Y_pix/target_pix`）⇒ 训练/推理脚本口径可复用。

---

## 3. 架构细节

```
输入 (B,3,224,224) --ImageNet 归一化 (0.485/0.456/0.406, 0.229/0.224/0.225)-->
  ┌─ 编码器 LinearAttentionEncoder（31.906 M, 从零）────────────────────────┐
  │ patch_embed: Conv2d(3→448, k=16, s=16)  → (B,196,448) + learnable pos   │
  │ seq = [cls(1); specials(K); patches(196)]  (K=64 ⇒ 261 token)           │
  │  × 13  LinearAttentionBlock:                                            │
  │      x ← x + LinearAttn(LN(x))        ← 核形式线性注意力 + DWConv 局部混合 │
  │      x ← x + MLP(LN(x))               ← GELU, mlp_ratio 4               │
  │ LayerNorm → z_cls = seq[:,0], z_s = seq[:,1:1+K]                        │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ 解码器 StandardAttentionDecoder（31.173 M, 普通 softmax 注意力）───────┐
  │ memory = [z_cls; z_s] (K+1,448) → Linear 448→512 → LayerNorm            │
  │ 查询 = query_base(196,512) + query_pos(196,512)                        │
  │  × 7  nn.TransformerDecoderLayer(pre-norm, gelu):                       │
  │      自注意力(196) + 交叉注意力(memory) + FFN(4×)                        │
  │ LayerNorm → PixelHead MLP(512→1024→768) → 每 patch 16×16×3              │
  └─────────────────────────────────────────────────────────────────────────┘
输出 (B,3,224,224)；损失 = L1(预测 patch, 目标 patch)（归一化空间）
```

**参数量核算**（`python model_v2_linattn.py` §1 实测，非估算）：

| 模块 | 参数量 | 构成 |
|---|---|---|
| `LinearAttentionEncoder` | **31,909,024**（31.909 M） | patch_embed 0.34M + cls/specials/pos 0.12M + 13 层 × 2.42M（含 QK-Norm 2.9k） |
| `StandardAttentionDecoder` | **31,173,376**（31.173 M） | mem_proj 0.23M + query 0.20M + 7 层 × 4.19M + PixelHead 1.31M + LN |
| 合计 | **63,082,400**（63.082 M） | dec/enc = **0.977**（31.909/31.173）（"同等体量"） |

每层参数（`dim=D`）：线性注意力块 `12D² + 9D`（qkv 3D² + proj D² + DWConv 9D + 2×LN）;
TransformerDecoderLayer `16D²`（self-attn 4D² + cross-attn 4D² + FFN 8D²）。

---

## 4. 线性注意力：实现与动机

**核形式**（Katharopoulos et al., ICML 2020）：

```
Attn(Q,K,V)_i = φ(q_i)ᵀ Σ_j φ(k_j) v_jᵀ / (φ(q_i)ᵀ Σ_j φ(k_j))
```

先聚合 `(d×d)` 的 `Σ_j φ(k_j)v_jᵀ` 与 `Σ_j φ(k_j)`，再乘 `φ(Q)` ⇒ 对序列长度 **O(N)**，
而不是 `O(N²)`。特征映射 `φ(x)=elu(x)+1`（非负 ⇒ 分母恒正），分母另加 `1e-6`
（fp16/bf16 下的稳妥处理）。

**动机（本项目里 N=196 不大，省算力不是理由）**：

1. 消融"预训练 + softmax"这一**组合**：从零方案与主线的差距，是否来自注意力形式本身？
2. 核形式**没有 softmax 的强竞争归一化**，register token 的汇聚更"平均"，
   与 `K≪N` 的聚合语义天然契合（softmax 容易让少数 patch 独占 attention）。

**局部混合（DWConv）**：纯线性注意力缺局部性，块内加**逐通道 3×3 卷积 + 残差**
（线性注意力 ViT 的通行做法，EfficientViT / MLLA）。**零初始化** ⇒ 初始为恒等映射，
不破坏训练初期的纯线性注意力行为。`enc_dwconv=0` 可关掉做消融。

⚠️ 实现要点（独立审查逐条验证，见 §8）：DWConv **只作用在 patch 段**（`x[:, n_prefix:]`，
`n_prefix = 1+K`）—— cls/specials 没有 2-D 网格语义，早期版本把它套在整个序列上
（`(B,D,14,14)` reshape 失败 = 直接报错，不是静默算错）。

---

## 5. 刻意偏离主线的一点（诚实标注）

主线解码器是**循环读出**（每步读一块 z_s，carry 走 BPTT，query = query_base + 上一步输出）。
本实验版改为**单步并行解码**（`Y_pix` 的 |T| 维 = 1）。

**理由**：本实验的自变量是"编码器是否预训练 + 注意力类型/体量"，循环读出是**另一个自变量**；
两者同时改会让结果无法归因。需要逐步精炼语义时，可把 `StandardAttentionDecoder` 按同一
接口换成循环版（主线代码未动，仍然在）。

---

## 6. 数据管线：ImageNet-100

- 来源：AutoDL 公共盘 `/root/autodl-pub/ImageNet100/imagenet100.zip`（14.4 GB），
  解压到 `/root/autodl-tmp/datasets/imagenet100/`（zip 顶层是 `imagenet100/`）。
- 规模：**100 类 × ~1267 张 = 126,689 张**（与官方 ImageNet-100 train 划分张数一致）。
  **没有官方 val 划分**（只有 train 全部图像）。
- **确定性划分**：每类按文件名排序后**等距抽取** `val_per_class=50` 张作验证集 ⇒
  **train 121,689 / val 5,000**。与磁盘顺序、类别顺序无关; `data_linattn.split_index`
  已用暴力枚举（n=1…300×k=1…60 与 n=300…12689×k≤200）验证 **train/val 零重叠**、
  每类 val 数恰为 `min(k, n−1)`、并集 = 全集、无重复。
- 变换：train = `RandomResizedCrop(224, scale=(0.35,1.0))` + `RandomHorizontalFlip`；
  val = `Resize(256, BICUBIC)` + `CenterCrop(224)`；均 `Normalize(ImageNet)`。

---

## 7. 训练与评测配置

**训练**（`run_linattn.sh`，单卡 RTX 4080 SUPER 32 GB）：

| 项 | 值 |
|---|---|
| 模型 | K=64（N=196 ⇒ 压缩率 196/64 ≈ 3.06×）, enc 448×13, dec 512×7 |
| 批量 / 优化器 | 128 / AdamW (β=0.9,0.95, eps=1e-8), wd 0.05（norm/bias 不 decay） |
| lr / schedule | **2e-4**, cosine + warmup **8%**（3420 步）, min_lr 1e-6, grad_clip 1.0 |
| **稳定性** | `enc_qk_norm=1`（默认）+ 线性注意力聚合段 fp32 + `--skip_grad 50`（见 §8.1） |
| 精度 | bf16 autocast（参数 fp32; **线性注意力聚合段内部强制 fp32**） |
| 步数 | 950 步/epoch × **45 epoch** = 42,750 步 |
| 实测吞吐 | **~318 img/s**（bs=128 峰值显存 19.1 GB; fp32 聚合段比纯 bf16 慢约 20%）⇒ ~6.4 min/epoch, 总 ~5.0 h |

> lr 从 3e-4 降到 2e-4、warmup 从 5% 提到 8%, 是 §8.1 那次梯度爆炸之后的调整（不是调参偏好）。

**评测口径**（`metrics_linattn.py`，沿用主线"PSNR 单独不可信"的教训）：

- L1（归一化空间）+ PSNR + SSIM（**[0,1] 空间**, 11×11 高斯窗 σ=1.5,
  C1=0.01², C2=0.03², **边界按参考实现裁掉 pad** 后平均）。
- **两条参照基线**（都打在**真实 val 目标**上，不是黑图）：
  1. **per-patch DC 基线**：每张图自己的 16×16 patch 均值重建
     —— "不做任何跨 patch 推理"的朴素重建（强基线）；
  2. **常量灰基线**：ImageNet 均值灰图 vs 真实目标 —— "预测均值"型平凡解。
- 判据：重建只有**同时**显著优于两条基线，PSNR 才有意义
  （主线 `doc/2026-09-23/ANALYSIS_res_sweep_rootcause.md` 的结论）。

---

## 8. 踩坑记录（本轮真实发生，已加回归测试）

### 8.1 线性注意力在 bf16 下**间歇性梯度爆炸**（本轮最大的一次翻车）

第一次正式训练（K=64 / bs=128 / **lr 3e-4**）跑得非常好，然后**突然崩**：

```
step 4750 ep5  loss=0.3217  val PSNR=19.58dB   ← 一切正常
step 5300 ep6  loss=0.3934  gn=0.69
step 5320 ep6  loss=0.8197  gn=26.28           ← 开始异常
step 5400 ep6  loss=0.8878  gn=338.23
step 5420 ep6  loss=0.9338  gn=1,219,840.62    ← 爆炸
step 6020 ep7  loss=0.9624  gn=337,617,760.00  ← 3.4e8
[val] ep6 PSNR=11.76dB                          ← 不可恢复（此后一直 12dB 上下）
```

**根因**：`φ(x)=elu(x)+1` 的核形式**没有 softmax 的归一化**，
`Σ_j φ(k_j)` 与 `Σ_j φ(k_j)v_jᵀ` 会随 qkv 权重增大而**无界增长**。
分子分母都长到 1e3+ 时，bf16（尾数 8 位 ≈ 3 位十进制）的相对误差 ~1e-3，
噪声经 13 层放大 ⇒ 梯度间歇性爆炸; `clip_grad_norm_(1.0)` **挡不住**
（方向本身已被少数巨大分量支配, 一步更新就足以毁掉权重）。

**三处修复（缺一不可，已全部落地 + 回归测试）**：

| # | 修复 | 位置 | 代价 |
|---|---|---|---|
| ① | 每个 head 的 **QK LayerNorm**（φ 之前）⇒ φ 输出钉在 [0,~4], `Σφ(k)` 有界 | `LinearAttention.__init__` | +2.9k 参数, 可忽略 |
| ② | 线性注意力**聚合段强制 fp32**：`.float()` 不够, autocast 会把 fp32 matmul 再降回 bf16, 必须 `torch.autocast(enabled=False)` | `LinearAttention.forward` | **吞吐 −20%**（404→318 img/s） |
| ③ | train 侧**梯度爆炸跳步护栏**：clip 前 `grad_norm > --skip_grad(50)` 就跳过本步更新（前向爆 activation 时参数本身还好, 不做更新就不会坏） | `train_linattn.py` | 无 |

**量化验证**（`python model_v2_linattn.py` §7, 回归测试）：
qkv 权重 ×20 时 `Σ_j φ(k_j)` 的量级 —— **无 QK-Norm 放大 ×17.1（分母爆炸）**，
**有 QK-Norm ×1.00（钉住）**。另断言"权重 ×20 + 输入 ×3"的端到端 loss 仍有限。

**边界（诚实标注）**：本轮**没有**单独消融 ①②③ 各自的贡献 ——
只能确认"三件一起上"之后训练稳定（见 REPORT）。`enc_qk_norm=0` 可复现旧行为。

### 8.2 单位口径 bug（PSNR 虚高 48.1 dB）：`IMAGENET_MEAN/STD` 是 **[0,1] 标度**（0.485/0.229），
所以反归一化后**本来就是 [0,1]**；首版 `denorm01` 又除了 255 ⇒ 预测与目标**同时**缩到
1/255 ⇒ MSE 缩 1/255² ⇒ **PSNR 整体虚高 `20log10(255) = +48.13 dB`、SSIM 退化到 ≈0.99**。
症状是**自相矛盾**的数字："归一化空间 L1 = 0.85（≈预测均值水平），却报 PSNR 61 dB / SSIM 0.994"。

- **训练不受影响**（loss 两侧同口径）; 受害的是**指标、`eval_val.json`、所有可视化**。
- 修复：`clamp_255=False` + 显式 `.clamp(0,1)`。
- **回归测试**（`python metrics_linattn.py`）: §4 复现"+48.1 dB 虚高"机制;
  §5 **直接 import `train_linattn.denorm01` 跑真实调用点**（不是重写一遍）；
  另在 `train_linattn.evaluate` / `eval_linattn` 里加运行时护栏
  `assert 0.02 < tgt01.mean() < 0.98`（真实图像 ≈0.45，错误口径 ≈0.0018）。
- 同源的第二个坑：`model_v2_linattn.py` 自检 §4 原先用 `(x*255−mean)/std` 验往返
  —— 那是**数据管线永不产生**的口径, 测试自洽但永远测不到真正的失败模式; 已改为
  真实口径 `(x−mean)/std`。

**其它（独立审查后修复）**：`SRPhase1LinAttn(**kwargs)` 的派生键过滤、`reconstruct()`
返回值口径（旧 docstring 写 0-255 实返 [0,1] 且不裁剪）、时间预算续跑少跑 <1 epoch、
`--num_workers 0` 仍起 2 个 val worker、val 子集取"前缀"导致类别不均衡（改为等距）。

---

## 9. 复现命令

```bash
# 0) 数据（一次性）
cd /root/autodl-tmp/datasets && unzip -q -o /root/autodl-pub/ImageNet100/imagenet100.zip
# 1) 自检（形状/参数量/梯度/像素往返/指标口径）
python model_v2_linattn.py && python metrics_linattn.py && python data_linattn.py
# 2) 吞吐/显存探针
python probe_linattn.py --batch_sizes 64 128 256
# 3) 冒烟（1 分钟）
python -u train_linattn.py --smoke --output_dir /tmp/smoke
# 4) 正式训练（K=64）
K=64 BS=128 EPOCHS=60 TIME_BUDGET=6 bash run_linattn.sh
# 5) 全量 val 评测 + 曲线
python eval_linattn.py --ckpt <out>/ckpt_best.pt --out <out>/eval_full.json \
    --grid <out>/samples/final_grid.png
python plot_linattn.py --log <out>/train_log.jsonl --out <out>/curves.png
```

---

## 10. 尚未做 / 边界（诚实标注）

- **未做 K-sweep**：只跑了 K=64 一条。`K=196`（无压缩）与 `K=32/128` 的对照未跑,
  ⇒ 本实验版**尚不能回答**"压缩率对该架构的重建质量影响多大"。
- **未做 LR / 架构消融**：lr 只用了 3e-4（未扫）；`enc_dwconv` 关掉的效果未测；
  编码器/解码器宽度-深度配比未调。
- **未接 Phase 2**：本实验版只到像素重建（信息保持探针），
  与主线一样尚未接 Qwen 做文字生成验收。
- **未与主线同数据对照**：主线的 224 结果是在 construction_site 上得的,
  与 ImageNet-100 不可直接比; 同数据同 K 的主线臂未跑。
- 单步并行解码（§5）意味着**没有**主线的"逐步精炼"曲线可比。
