# ANALYSIS — ALIT（arXiv:2411.02393）结构与 AssocVTC 对照，重点：解码器是并行还是循环

> 日期：**2026-10-16**
> 对照对象：**ALIT**（Adaptive Length Image Tokenization via Recurrent Allocation,
> Shivam Duggal / Phillip Isola / Antonio Torralba / William T. Freeman, MIT CSAIL,
> [arXiv:2411.02393](https://arxiv.org/abs/2411.02393)，2024-11-04）
> 本仓库侧：**`model_v2.py`（DINOv2 register 主线）** 与 **`model_v2_linattn.py`（10-15 起的从零注意力实验线）**
> 上游实验报告：[`doc/2026-10-15/REPORT_linattn_fromscratch.md`](../2026-10-15/REPORT_linattn_fromscratch.md)、
> [`doc/2026-10-16/REPORT_stdattn_ablation.md`](REPORT_stdattn_ablation.md)
> 本文件的用途：为 [`PLAN_alit_adaptive_experiments.md`](PLAN_alit_adaptive_experiments.md) 提供
> **机制映射与差异清单**——哪些能抄、哪些抄了会砸掉我们自己的收益。
> 代码引用均为**已 clone 的两个仓库在 2026-10-16 的 HEAD**，行号可复核。

---

## 0. 结论速览（先看表）

| | ALIT | AssocVTC（主线 / linattn 线） |
|---|---|---|
| **循环位置** | **编码器 / 记忆分配**（每轮追加新 latent，旧 latent 残差更新） | **解码器 / 读出画布**（`Y = query_base + Y_{t-1}`，BPTT） |
| **解码器** | **并行、无状态**：每轮用同一份权重对 `latent_t` 一次性解码全部 2D token | **顺序循环**：`|T|` 步串行，每步只读自己那块 z_s |
| **递进载体** | **写侧**：可读键逐轮变多（32→256） | **读侧**：窗口逐轮变宽（z_s 固定 K 个） |
| **变长** | 已实现（按重建阈值选迭代） | 目标设想，**未实现**；`derive_num_specials` 保证 K 与步集解耦 |
| **离散化** | 有：共享 1D codebook（4096×12）量化 | 无：连续 z_s + PixelHead |
| **监督** | stage1 逐轮 2D-token CE + `lambda_loss_weight`；stage2 单轮像素 + GAN | `mean_t L1(PixelHead(Y_t), 整图)` 平权 |
| **空间维** | 非自回归（256 个 2D token 一次出） | 非自回归（每步 N 个 patch 一次出），**步间**串行 |
| **输出** | 1D token 流 → VQGAN 解码器出像素 | 直接像素 |

> **一句话**：ALIT 的解码器确实是**并行且无状态**的，它的"循环"在编码器；AssocVTC 恰好相反，
> 把循环放在解码器、用画布 carry 承担递进。这不是实现细节，而是两种世界观，**决定了哪些机制可以移植**。

---

## 1. ALIT 侧代码证据

### 1.1 解码器：一次拼接、一次前向、全部 2D token 并行出

`modules/latent_distillation_modules.py`（`Decoder.forward`，L48–68）：

```python
def forward(self, latent_1D_tokens, masked_2D_tokens):
    if self.factorize_latent: latent_1D_tokens = self.decoder_embed(latent_1D_tokens)
    latent_1D_tokens = latent_1D_tokens + self.latent_token_positional_embedding[:latent_1D_tokens.shape[1]]
    x = torch.cat([masked_2D_tokens, latent_1D_tokens], dim=1)      # 一次拼接
    x = self.ln_pre(x).permute(1, 0, 2)
    for i in range(self.num_layers): x = self.transformer[i](x)     # 层循环，不是 token/步循环
    x = x.permute(1, 0, 2)
    reconstructed_2D_tokens = x[:, 1:masked_2D_tokens.shape[1]]     # 256 个 2D token 一次出
    ...
```

无 carry、无逐步、无 query 回喂。**空间维 = 非自回归**（没有 AR over tokens）。

### 1.2 循环在编码器：追加新 latent + 旧 latent 残差更新

`adaptive_tokenizers/adaptive_vqgan.py`（`forward`，L185–298）：

```python
masked_2d_tokens = self.preprocess_decoder(img_tokens)   # L201: 循环外算一次 ⇒ 8 轮共用
x = torch.cat([img_tokens, init_latent_tokens + self.timestep_embedding[0]], dim=1)
for iter in range(self.max_rollout_iters):               # 8 轮
    x = self.encoder(x)                                  # L224: 循环体 = 编码器
    latent_tokens = x[:, img_tokens.shape[1]:]           # 累积的全部 latent
    ...
    decoded_logits = self.decoder(latent_tokens_quantized, masked_2d_tokens)   # L248
    ...
    if iter != self.max_rollout_iters - 1:
        x = torch.cat((x, init_latent_tokens + self.timestep_embedding[iter+1]), dim=1)  # L294 追加 32 个
```

可直接读出三条事实：

1. **解码器输入里唯一变化的是 latent**——`masked_2d_tokens` 在循环外构造、8 轮**逐位相同**；
   即解码器是一个纯函数 `D(latents_t)`，每轮用**共享权重**重新调用一次（amortized，不是 RNN）。
2. **状态全在编码器的 token 序列里**：旧 latent 每轮被重新编码（残差更新），新 latent 每轮追加
   32 个（`num_init_latent_tokens=32`，`max_rollout_iters=8` ⇒ 32/64/…/256）。
3. **动态停机只作用于编码器输入**，不动解码器：`perform_dynamic_halting` 把
   `pred_gt_index_prob >= 0.55` 的 patch 换成可学习 mask token（见 §1.3）。

### 1.3 ALIT 的超参（抄配方时的默认值）

| 项 | 值 | 位置 |
|---|---|---|
| 初始 latent / 迭代数 / 每轮新增 | 32 / 8 / 32（→ 256） | `adaptive_vqgan.py` L14 |
| 逐迭代损失权重 | `[2.5, 2.0, 1.5, 1.25, 1.0, 1.0, 1.0, 1.0]`（**递减**） | L68 |
| 动态停机阈值 | 0.55（GT-code 概率） | L15 / `adaptive_vqgan.yaml` |
| 量化 codebook | 4096 × 12 维，factorized | `adaptive_vqgan.yaml` |
| 梯度子集采样 | `sample_grad_iters`（stage2 只在一个迭代算 loss） | L207–218 |
| 底座 | 预训练 VQGAN（256×256 → 16×16 = 256 个 2D token） | L194 |

> ⚠️ 抄 `lambda_loss_weight` 时注意它是**递减**的：早期（token 少）权重更高。
> 这与"后步要干活"的直觉相反，是"先把 32-token 表示做好、再让它逐级细化"的口径。

---

## 2. AssocVTC 侧代码证据

### 2.1 `model_v2.py`（DINOv2 register 主线）：解码器 = 顺序循环 + 画布 carry

`OutputQueryDecoder.forward`（L602–612）：

```python
for i, t in enumerate(self.steps):                       # 顺序循环 |T| 步
    lo, hi = windows[i]
    Y = self.stack(self.stack_in(Y), A_in[:, lo:hi + 1])  # 交叉注意 memory = 自己那块 z_s 切片
    Y = self.stack_out(Y)
    Y_total.append(Y)
    Y = self.query_base + Y                               # carry，不 detach ⇒ BPTT
```

编码器 `_encode_register`（L713–734）是**一次 DINOv2 24 层前向**，`z_s` 只取末层 ⇒ **编码器无循环**。

### 2.2 `model_v2_linattn.py`（10-15 起的实验线）：同一分工

- 编码器 `StandardAttentionEncoder` / `LinearAttentionEncoder`：一次前向，输出 `z_cls, z_s`，
  `seq = [cls(1); specials(K); patches(196)]`，K=64。
- 解码器两条路：`StandardAttentionDecoder`（**单步并行** `|T|=1`）与
  `RecurrentOutputQueryDecoder`（**顺序循环**，`Y = query_base + Y`，BPTT）。
- 10-16 实测：`std` 编码器下 **单步 31.219 dB** vs **BPTT（T=8）28.582 dB**——但两者训练预算差 7.5×
  （45 ep 从零 vs 6 ep 热启动），**不能**据此说"BPTT 更差"（见该报告 §9.3）。

**结论：本仓库的"循环 = 解码器"在两条线上一致。** 这是与 ALIT 最大的结构分歧。

---

## 3. 四条真正影响实验设计的差异

### 3.1 递进载体：改"记忆"还是改"画布"

- **ALIT**：每轮**新增可写 token**（写侧增长）⇒ 后续迭代有"新键可读"，老 latent 因此可以收缩到
  稀疏局部区域（论文 Fig.8/9、Tab.2：40% 注意力阈值下 mIoU 56.7→57.8 随迭代上升）。
- **我们**：固定 K 个 z_s，把**读窗口**逐级变宽（读侧增长）⇒ 后续步读的是同一批键的更多前缀。

**对本仓库的直接含义**：README §4.1 诊断的"后步真干活"困境（`mean_t` 平权 + 后步只能读自己那块 +
键可寻址性不足），ALIT 给了一条**正交**解法——**给后步新键**，而不是在后步加权。
这落在我们的 **P2**（编码侧注入）上，且比 `Linear(patch_feat)` 更彻底。

### 3.2 动态停机 ≈ 我们的 P1，但作用面不同（且当前架构装不下原版）

- ALIT：把已重建好的 patch **从编码器输入里 mask 掉**——动的是**信息通路**；
- 我们的 P1 设计：step t **只监督自己对应的 patch 行区**——动的是**梯度**。

⚠️ **架构约束（本文件最有价值的一条）**：我们的编码器是**单次前向**（BPTT 的循环只在解码器侧）。
因此"把已完成的 patch 从编码器输入 mask 掉"在现有结构里**无法实现**——要 mask 就得重编码。
所以：

- 现在能落地的只有**损失侧区域掩码**（P1 原设计）；
- **输入侧停机**必须和"多轮编码"（§3.3 的 refiner）一起做。

### 3.3 层级独立性：我们的是"串联草稿链"，ALIT 的是"每档独立"

- ALIT 解码器无 carry ⇒ 第 1 轮的 32-token 重建与第 8 轮彼此独立，**截断天然合法**（代价：各档不复用像素级劳动）。
- 我们 carry 走 BPTT ⇒ 已有实验证明"删一步的代价摊到所有后续步"
  （`doc/2026-09-22/REPORT_slice1_224.md`，README §3.2）。

**含义**：如果主线目标包含"按图截断 K / 一条流多档"，**层级独立性是刚需**——否则不同 K 的表示互不一致。
这是 ALIT 最值得借鉴的**设计取舍**（不是模块），也是 PLAN 里 E-B4 要裁定的事。

### 3.4 前缀嵌套的语义：ALIT 连前缀都不共享

- 我们：**同一份 z_s**，读窗口变宽 ⇒ 位置上像真前缀嵌套。
- ALIT：第 k 轮的前 32 个 latent 与第 1 轮的 32 个**数值不同**（每轮被重新编码）⇒ 每档"为自己的预算专门优化过"，
  但失去"单流前缀截断即任意档"的性质。

两边各有代价，**不是谁更先进**：我们在"单流可截断"上更干净，ALIT 在"每档各自最优"上更强。
PLAN 的 E-B4 就是测这个 trade-off 在本项目数据上到底值多少。

---

## 4. ALIT 机制 → 本仓库映射表

| ALIT 机制 | ALIT 参数 | 本仓库现状 | 可落地形态 | PLAN 实验 |
|---|---|---|---|---|
| 自适应记忆（逐轮 +32 latent） | 32/8/32 | 读侧窗口递增（`step_windows`） | B3a 读侧"生长键"（τ 小改动）/ B3b 写侧 refiner（ALIT 忠实版） | **E-B3** |
| 随机预算采样 | `sample_grad_iters` | 无（BPTT 全步反传） | `--sample_grad_iters` | **E-B4** |
| 动态停机（输入侧） | `th=0.55` | 无（只有 P1 设计稿） | 需配合 B3 的多轮编码 | **E-B2**（损失侧先行） |
| 逐迭代损失权重 | `[2.5,2,1.5,1.25,1,1,1,1]` | `mean_t` 平权 | `--loss_weights`（现成改动） | **E-B1** |
| 量化共享 codebook | 4096×12 | 无（连续 z_s） | **不做**（YAGNI，Phase 2 之后再说） | — |
| TSC–TOI 曲线（重建阈值选 token） | Fig.6 ≈60% 预算 | batch-0 早停（**PSNR 口径**，+0.13 dB） | 换成任务保真口径重做 | **E-A2 / E-C1** |
| IID/OOD 熟悉度（FID gap） | 7.92/12.56/23.32 | 无 | 同一 ckpt 跑 IID vs OOD 逐步 | **E-A3** |
| 熵/复杂度相关性 | SAVOIAS 人标注 0–100 | 无 | 弱代理（JPEG 大小/边缘密度/残差方差）+ Spearman | **E-A4** |
| token 专精 / 对象发现 | 40% attn → 57.8 mIoU | 无 | z_s 注意力熵/稀疏度 + 冗余度 | **E-A5** |

---

## 5. 诚实边界（引用本文件时必须带上）

1. **不可直接比数字**：ALIT 是 256×256 + 预训练 VQGAN 离散底座 + 213k IN-100 训练 + 1D 量化 latent；
   我们是 224×224 + 从零 softmax 编码器（或 DINOv2 主线）+ 连续 z_s。FID/线性探针的数字**不同构**。
2. **单种子、无方差**：本仓库 10-15/10-16 的结果均为 seed 0，不能支撑 0.1 dB 级结论
   （`REPORT_stdattn_ablation.md` §9.2 已标注）。
3. **PSNR-RD 不是验收**：README §3 判据口径明确"RD/PSNR 这条线应放弃"，
   ALIT 的"熵/预算"主张在本仓库必须用**任务保真（E8）**复核，不能用 PSNR 直接证否或采纳。
4. **"密度 ↑ ⇒ 幻觉 ↓"仍未验证**：本仓库一个 caption/幻觉指标都没跑过（README §0.1 边界）；
   本文件与 PLAN 都是**方向性设计**，不是已验证结论。
5. 本文件只读了两个仓库的 **main HEAD 代码**与论文 HTML 版；ALIT 的 PDF 附录细节未逐条核。
