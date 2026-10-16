# DESIGN — 消融实验：把编码器换成普通 transformer（并跑 BPTT）

> 日期：**2026-10-16**
> 基线提交：`b3a5f3a`（= 昨日 2026-10-15 的 linattn 臂，`doc/2026-10-15/`）
> 机器：AutoDL 单卡 **RTX 4080 SUPER 32 GB**（`ssh -p 17094 root@connect.weste.seetacloud.com`，与昨日同一实例）
> 代码改动：`model_v2_linattn.py`（+274 行，**纯新增**）、`train_linattn.py`（+12/−4，**纯新增开关**）
> 新增：`run_stdattn.sh`（消融臂启动器）、`run_stdattn_chain.sh`（两阶段链式驱动 + 断点续跑）

---

## 1. 用户口径

> 「帮我做一个昨天实验的消融实验，那就是**把编码器换成普通的 transformer，跑 BPTT**。」
> （2026-10-16，用户原话；数据集/其余超参照旧沿用昨日实验）

拆成两条可验收要求：

| # | 要求 | 落地 |
|---|---|---|
| A | **把编码器换成普通的 transformer** | 编码器注意力由**线性核注意力**换成**标准 softmax 缩放点积注意力**（`enc_attn="std"`）；token 化/位置编码/patch embed/深度/宽度/解码器/K/数据**全部不动** |
| B | **跑 BPTT** | 解码器走主线的**循环读出**（`dec_recurrent=1`，每步只读自己那块 z_s，carry 不 detach ⇒ 梯度穿过 T=8 步） |

---

## 2. 自变量与对照组（这是本消融的全部意义所在）

**唯一自变量 = 编码器注意力形式。** 其余逐项冻结：

| | 昨日臂（线性注意力编码器） | 今日消融臂（普通 transformer 编码器） |
|---|---|---|
| 编码器注意力 | 核形式线性注意力 `φ(x)=elu(x)+1`，`O(N)` 聚合 | **标准 softmax 缩放点积** `softmax(QKᵀ/√d)V` |
| 局部混合 DWConv | 有（零初始化，可学） | **无**（普通 transformer 原貌） |
| QK-Norm | 有（线性注意力稳定性修复） | **无**（softmax 自带归一化，不需要） |
| 编码器 dim / depth / heads | 448 / 13 / 8 | **同** |
| token 序列 | `[cls(1); specials(K=64); patches(196)]` + learnable pos | **同** |
| patch embed | `Conv2d(3→448, k=16, s=16)` | **同** |
| 解码器 | `StandardAttentionDecoder` 512×7 / `RecurrentOutputQueryDecoder` 512×7 | **同** |
| K / N / 分辨率 / 数据 / 损失 | K=64, N=196, 224×224, ImageNet-100, 像素 L1 | **同** |

**参数量天然对齐**（关键：普通块与线性块的结构几乎同构，只差 DWConv 与 QK-Norm）：

| | 编码器 | 解码器 | 合计 | dec/enc |
|---|---|---|---|---|
| 昨日 linattn 臂 | **31,909,024**（31.909 M） | 31,173,376（31.173 M） | 63,082,400 | 0.9769 |
| 今日 stdattn 臂 | **31,847,872**（31.848 M） | 31,173,376（31.173 M） | 63,021,248 | 0.9788 |
| 差 | **−61,152（−0.19%）** | 0 | −0.10% | — |

⇒ 编码器体量几乎不变，**结果差异可归因到"注意力形式"本身**，而不是"换了更大的模型"。

每层差值的解析来源：普通块 `13D² = 2,414,272`，线性块 `12D² + 9D + 2·(2·head_dim) = 2,418,976`
（DWConv `9D+9` = 4,041 + QK-Norm `4·head_dim` = 224）；13 层合计差 61,152。

---

## 3. 两阶段训练（与昨日臂**逐项同协议**，才能配对比较）

昨日留下了两条臂，因此今日也跑**两阶段**，分别与它们一一对应：

| 阶段 | 今日消融臂 | 昨日对照臂 | 协议（两边完全相同） | 对照变量 |
|---|---|---|---|---|
| **1** | `stdattn_k64`：std 编码器 + 单步并行解码 | `k64`（PSNR **26.602**） | bs128 / 45 epoch / 42750 步 / lr 2e-4 / warmup 8% / cosine / wd 0.05 / grad_clip 1.0 / bf16 / 从零 | 编码器注意力形式 |
| **2** | `stdattn_k64_rec`：**从阶段 1 热启动** + 循环读出（**BPTT**, T=8） | `k64_rec`（PSNR **25.304**） | bs64 / 6 epoch / 11406 步 / lr 2e-4 / warmup 8% / bf16 / 编码器联合微调（不冻结） | 编码器注意力形式 |

```
阶段 1  stdattn_k64        (单步, 从零, 45ep)   ←→  k64      (单步, 从零, 45ep)
   │  ckpt_best.pt 热启动（只缺零初始化 decoder.mem_pos）
   ▼
阶段 2  stdattn_k64_rec    (BPTT, 6ep)          ←→  k64_rec  (BPTT, 6ep)
```

> 为什么这样设计：昨日已确认「单步并行 vs 循环读出」本身是**另一个自变量**
> （`doc/2026-10-15/DESIGN_linattn_fromscratch.md` §5）。本消融**不**再动解码读出方式，
> 而是在**两种读出方式下各做一次"只换编码器"的配对比较** —— 这样两条结论都干净可归因。
> 昨日 `REPORT` §3.4 已诚实标注"循环版只热启动 6 epoch、与单步版预算差 7.5×"；
> 本消融**不试图修复那个预算缺口**，而是让"新臂 vs 旧臂"在**同一预算**下比较。

---

## 4. 实现（`model_v2_linattn.py` 纯新增）

新增三个组件（不修改任何既有类）：

- `StandardAttention`：标准多头缩放点积注意力，`F.scaled_dot_product_attention`
  （PyTorch 自选 flash / mem-efficient 后端，自带 1/√d 缩放），双向无掩码。
  **不需要** `LinearAttention` 那套数值稳定性修复（QK-Norm / 核内 fp32 / 跳步护栏）——
  softmax 天然把每行权重归一化到和为 1，输出是 value 的凸组合，量级不会随 qkv 权重无界增长。
- `StandardAttentionBlock`：`x + MHSA(LN(x)) → x + FFN(LN(x))`，GELU，`mlp_ratio=4`，
  与 `LinearAttentionBlock` **逐项同构**；`forward(x, h, w, n_prefix)` 保留同签名便于统一循环。
- `StandardAttentionEncoder`：与 `LinearAttentionEncoder` token 化**逐位相同**
  （同 `patch_embed` / `cls_token` / `special_tokens` / `pos_embed` / `pos_drop` / 末尾 `LayerNorm`）。

开关：`LinAttnConfig.enc_attn ∈ {"linear"(默认), "std"}` + `enc_std_qk_norm`（默认 False），
由 `SRPhase1LinAttn._build_encoder()` 分发；CLI 新增 `--enc_attn` / `--enc_std_qk_norm`。
**默认值 = 昨日行为**，因此既有四件套（`eval_linattn.py` / `eval_stepwise_linattn.py` /
`plot_linattn.py` / `probe_linattn.py`）**无需改动**即可评今天的臂
（它们按 `ck["cfg"]` 重建模型，新字段自动生效）。

---

## 5. 验证证据（跑训练之前必须全绿）

### 5.1 自检 `python model_v2_linattn.py` — 新增 §9，本地 + 远程均 `ALL CHECKS PASSED`

```
[9] ★消融臂: 普通 (softmax) 注意力编码器（enc_attn='std'）
    StandardAttention == softmax(QKᵀ/√d)V 手工实现 (max|diff|=5.96e-08); 注意力权重行和==1
    编码器参数量: 线性 31,909,024 (31.909M) vs 普通 31,847,872 (31.848M)  相对差 0.19% OK
    整机(enc_attn=std): enc=31.848M dec=31.173M total=63.021M (dec/enc=0.9788)
    enc_attn=std dec_recurrent=0 T=1: Y_pix(2, 1, 16, 768) loss=0.8115 有梯度张量 77/77 OK
    enc_attn=std dec_recurrent=1 T=2: Y_pix(2, 2, 16, 768) loss=0.8123 有梯度张量 78/78 OK
    enc_attn=std bf16 autocast: loss=0.8149 grad_norm=0.300 有限 OK
    默认路径回归: enc_attn='linear' → LinearAttentionEncoder 31,909,024 参数 不变 OK
    回归: 旧 ckpt strict 载入成功 (ckpt_best.pt) OK
```

其中 §9a 是**"它真的是普通 transformer 吗"**的数值对拍（不是靠类名自证）：
手工写 `softmax(QKᵀ/√d)V` 与 `StandardAttention` 前向逐位一致（5.96e-08），
且注意力权重每行和 == 1（线性注意力的核形式**不满足**这一条）。

### 5.2 回归：昨日臂未被破坏

- `model_v2_linattn.py` 既有 §1–§8 全部通过；`LinAttnConfig()` 默认 `enc_attn="linear"`，
  编码器参数量与昨日**逐位相同**（31,909,024）。
- **强硬证据**：把昨日训练产物 `/root/autodl-tmp/srdiff_linattn/k64/ckpt_best.pt`（757 MB）
  用改后的代码 `load_state_dict(..., strict=True)` —— **载入成功**（state_dict key/形状零变化）。
- `train_linattn.py` 的改动是纯新增参数 + 一行日志；不传 `--enc_attn` 时行为与昨日一致。

### 5.3 GPU 冒烟 + 吞吐/显存实测（同机同 torch）

| 臂 | 配置 | 吞吐 | 峰值显存 | 每 epoch | 昨日同配置 |
|---|---|---|---|---|---|
| std 单步 | bs128, T=1 | ~290 img/s | **13.6 GB** | ~7.0 min | 线性臂 286–318 img/s / 19.16 GB |
| std BPTT | bs64, T=8 | ~105 img/s | **23.8 GB** | ~19.3 min | 线性臂 114 img/s / 26.6 GB |

- 冒烟 `--smoke`（30 步）+ 40 步真实配置探针**均跑通**，`grad_norm` 全程 0.1–0.3，无爆炸、无跳步。
- ⚠️ BPTT 在 **bs=128 会 CUDA OOM**（梯度要穿过 T=8 次解码，显存 ≈2× ）——
  与昨日 `k64_rec` 用 bs=64 是同一个原因，因此阶段 2 沿用 bs=64。
- std 编码器比线性编码器**更快、更省显存**：省掉了线性注意力"核内强制 fp32 聚合段"（昨日 −20% 吞吐）
  与 DWConv，且 softmax 走 `scaled_dot_product_attention` 的融合后端。

---

## 6. 复现命令

```bash
cd /root/SR-Diffusion-v3
# 0) 自检（形状/参数量/梯度/像素往返/数值稳定性/普通编码器对拍/旧 ckpt 回归）
LINATTN_CKPT=/root/autodl-tmp/srdiff_linattn/k64/ckpt_best.pt python model_v2_linattn.py

# 1) 一键两阶段（含时间预算 + 断点续跑 + 收尾评测）
nohup setsid bash run_stdattn_chain.sh > /root/autodl-tmp/srdiff_stdattn/nohup.out 2>&1 &

# —— 或手工分步 ——
TAG=stdattn_k64     BS=128 EPOCHS=45 TIME_BUDGET=5.5 ./run_stdattn.sh
TAG=stdattn_k64_rec BS=64  EPOCHS=6  TIME_BUDGET=2.0 DEC_RECURRENT=1 \
  INIT_FROM=/root/autodl-tmp/srdiff_stdattn/stdattn_k64/ckpt_best.pt ./run_stdattn.sh

# 2) 四臂同进程全量留出集评测（5000 张, 同批次同口径）+ 逐步重建 + 曲线
python eval_linattn.py --ckpt \
    /root/autodl-tmp/srdiff_stdattn/stdattn_k64/ckpt_best.pt \
    /root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt \
    /root/autodl-tmp/srdiff_linattn/k64/ckpt_best.pt \
    /root/autodl-tmp/srdiff_linattn/k64_rec/ckpt_best.pt \
  --num_images 0 --batch_size 64 --out /root/autodl-tmp/srdiff_stdattn/eval_ablation_full.json
python eval_stepwise_linattn.py --ckpt .../stdattn_k64_rec/ckpt_best.pt --num_images 0 \
  --out .../stepwise_stdattn_rec.json
```

产物目录：`/root/autodl-tmp/srdiff_stdattn/{stdattn_k64,stdattn_k64_rec}/`

---

## 7. 预期时间与产物

> 以下"预期"列写于训练之前（基于 40 步探针外推）；"实测"列为 2026-10-16 实跑结果。
> 实测比预期快，因为稳态吞吐高于短探针（4090 步后 ~407 img/s vs 探针 283 img/s）。

| 阶段 | 预期 | **实测** |
|---|---|---|
| 阶段 1（bs128, 45ep） | ≈ 5.3 h | **3.74 h**（224.3 min, ~407 img/s, 13.63 GB） |
| 阶段 2（bs64, 6ep, BPTT） | ≈ 1.9 h | **1.59 h**（95.3 min, ~128 img/s, 23.82 GB） |
| 收尾评测（4 臂 × 5000 + 逐步 + 曲线） | ≈ 0.4 h | **1.5 min**（评测 56 s + 逐步 23 s + 画图 3 s） |
| 合计 | ≈ 7.6 h | **5 h 22 min**（09:17:40 → 14:39:26） |

结果见 [`REPORT_stdattn_ablation.md`](REPORT_stdattn_ablation.md)：普通 transformer 编码器在两配对中
分别 **+4.617 dB**（单步）与 **+3.278 dB**（BPTT）。

---

## 8. 诚实边界（写在前面，免得结论被过度解读）

1. **只做"编码器注意力形式"这一个自变量**。DWConv 与 QK-Norm 随之一起去掉
   （它们本来就是线性注意力方案的配套件）——所以严格说消融对象是
   **"线性注意力编码器方案"整体 vs "普通 transformer 编码器"**，而不只是核函数本身。
   若要单独拆 DWConv / QK-Norm，可再跑 `enc_attn=std, enc_std_qk_norm=1` 与
   `enc_attn=linear, enc_dwconv=0`（本轮**没跑**）。
2. **阶段 2 沿用了昨日"6 epoch 热启动"的预算**（与 `k64_rec` 完全一致，故可配对），
   但该预算本身**不足以判断循环读出是否优于单步**（昨日 §3.4 已标注）。本消融
   **不**回答"单步 vs BPTT 哪个好"，只回答"在同样读出方式下，换编码器有什么影响"。
3. **单种子**（`--seed 0`）。无方差估计 ⇒ 小差距（<~0.2 dB）不宜当结论。
4. **不与主线 DINOv2 臂比**（数据/编码器来源/解码读出方式全不同，昨日 §6 已说明）。
5. 所有 PSNR 均为**逐图 PSNR 再平均**口径（与 2026-10-15 主报告同口径）；
   主线 `eval_stepwise.py` 的"先平均 MSE 再转 dB"口径差约 1.5 dB，**不可混用**。
