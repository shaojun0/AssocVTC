# FEATURE — v10 `TemporalMiddle` 流式化（具身智能组件）

> 状态：**设计稿，未实现**。本文只定义契约、边界与验收口径，不含代码改动。
> 前置：`doc/2026-11-16/DESIGN_v10_video_whole_clip.md`（v10 整片段循环读出）
> 相关代码：`model_v10.py` 的 `TemporalMiddle`（231.4M 参数，6 层 × d=1792）
> 核心结论：**P0 补丁零训练、与现有 `forward` 数值等价（实测 7.2e-07）**，
> 前提是 middle 本身已经是帧级因果的 —— 这是它相对双向视频编码器的结构性优势。

---

## 0. 一句话

把 `TemporalMiddle` 从「必须攒满 120 帧、整段算一次」改成「帧到即算、每 4 帧出一个
16 行的 z」，靠的是**帧级因果 ⇒ 历史 KV 可缓存 ⇒ 增量与全量精确等价**；模型权重、
训练图、损失函数一律不动，只加一条推理路径，外加几个互不依赖、可单独决定装不装的
轻量适配件。

---

## 1. 问题

`TemporalMiddle` 冻结后是一个合格的「视频理解」部件：

- 输入 480 个位置（120 帧 × 4 行），帧级因果，输出同形；
- 每帧 4 行 × 1792 = 7,168 值，正好等于码率预算；
- 输出可以直接 reshape 成 30 步 × 16 行喂给下游。

但它对具身智能（实时交互）**不可用**，原因不是精度，而是四条硬约束：

| # | 约束 | 代码位置 | 后果 |
|---|---|---|---|
| 1 | **长度写死** | `assert Fr == self.frames`（`TemporalMiddle.forward`） | episode 长度不等于 120 帧就报错 |
| 2 | **mask 是常量** | `_attn_mask` 为 480×480 buffer，`persistent=False` | 长度变了 mask 就不匹配 |
| 3 | **无增量接口** | `nn.TransformerEncoder` 不暴露 K/V | 每帧都要 O(L²) 重算全部历史 |
| 4 | **无时间度量** | 无位置编码、无 Δt 输入 | 分不清「上一帧是 0.125 s 前还是 10 s 前」 |

外加一条**感知层面的缺口**：middle 的训练目标是「够重建的码率」，与「够控制的信息」
不是一回事（见 §5）。这项不在本 feature 的范围内，但它决定了 `z` 不能直接当策略输入。

---

## 2. 现状审计

### 2.1 已经具备的结构性质

`TemporalMiddle` 的 mask 由 `fi = arange(L) // tpf; allowed = fi_j <= fi_i` 生成，
也就是**帧级块因果**：

- 同一帧的 4 行彼此**双向**可见（这 4 行是同一时刻的打包视图，没有先后）；
- 跨帧只能看历史帧（帧 j 看不到 j+1..N）。

由此得到一个关键推论，整个 feature 都建立在它上面：

> **第 j 帧的输出只由 0..j 帧决定，未来帧永远不会改写它。**
> 因此历史帧的 K/V 一旦算出就是终值，可以缓存；增量推理与全量推理**数学上等价**，
> 差值只来自浮点累加顺序。

对比一下：如果 middle 是双向的（像大多数视频 ViT），加新帧会改写全部历史输出，
既不能 cache，也不能增量 —— 只能重训成因果版本。**v10 不需要付这个代价。**

### 2.2 缺的四件事

见 §1 的表。注意 1 和 2 是「接口层的锁」，3 是「性能问题」，4 是「语义问题」。
难度依次递增：前面两个纯工程，第三个要写 cache 调度，第四个必须碰训练。

### 2.3 核心实测：增量 ≡ 全量

用 `F.scaled_dot_product_attention` 手动维护每层 `(k, v)`，逐 chunk 前向，
与 `TemporalMiddle.forward` 对比（fp32，`frames=32`，2~6 层，`tpf=4`）：

| 形式 | 与全量 `forward` 的 max\|Δ\| | 结论 |
|---|---|---|
| 逐帧流式（chunk = 1 帧 = 4 token） | **≤ 7.2e-07** | 等价（fp32 数值噪声） |
| 每 4 帧流式（chunk = 1 个 z 步） | **≤ 7.2e-07**（部分配置逐位相同） | 等价 |
| 每 4 帧，chunk 内不加 mask | 0.39 | **错误**，见 §2.4 |
| 每 4 帧，chunk 内加 mask 但漏掉历史列 | 形状直接报错 | **错误**，见 §2.4 |

输出量级参照：`|full|.mean() ≈ 0.80`，所以 7.2e-07 是 1e-6 相对量级，纯数值噪声。

### 2.4 一个必须写死的实现细节

增量时 mask 的**列**要覆盖「历史 + 当前 chunk」，而**行**只覆盖当前 chunk：

```
mask.shape = (n_new_tokens, n_history_tokens + n_new_tokens)
  历史列        → 全 0     （全可见：因果允许看所有历史）
  chunk 内列    → 块因果    （fi_j <= fi_i，同一帧的 4 行互看）
```

两种错误写法及其实测后果：

1. **后续 chunk 完全不给 mask**：同一 chunk 内帧 j 能看到帧 j+1（同一时刻的后续帧），
   误差 0.39。
2. **只给 chunk 内的方阵 mask**（漏掉历史列）：形状不匹配直接报错；若强行 pad，
   语义会退化成「只看当前 chunk」，历史被静默丢弃。

这条规则必须进自检断言，否则以后任何人重构 `forward_stream` 都可能踩回去。

### 2.5 变长与外推（宽松结论，需训练后复核）

去掉 `assert`、mask 按实际长度生成后，用**随机初始化**权重测 8 ~ 480 帧：

| 帧数 | 输出 per-token std | max\|y\| |
|---|---|---|
| 8 | 1.0079 | 3.04 |
| 120（训练长度） | 1.0079 | 4.15 |
| 480（4×） | 1.0079 | 4.81 |

输出尺度被 `encoder.norm` 钉死在 1.0079，**数值上不会爆**。但这只说明前向不崩，
不代表表示质量不退化 —— 见 §2.6。

### 2.6 长历史的注意力摊薄（机制性风险）

随机权重下测末帧注意力熵（去掉 1/√d 缩放以便看形状）：

| 帧数 | 可见 token | 注意力熵 | 上界 ln(L) |
|---|---|---|---|
| 32 | 128 | 4.704 | 4.852 |
| 120 | 480 | 6.049 | 6.174 |
| 480 | 1920 | 7.434 | 7.560 |

熵几乎贴着上界，且随长度线性增长。随机权重下这说明不了训练后的行为，但机制是清楚的：
**全历史可见 ⇒ 注意力跨度随 episode 线性增长 ⇒ 越长的历史越被摊薄**。
训练长度是 120 帧，推理若跑到 480 帧就是 4× 外推，这是 P1（时间偏置 / 窗口）
要解决的问题，不能靠 P0 假装不存在。

---

## 3. 补丁设计

### 3.1 P0 — 变长 + 增量 KV cache（零训练，主推）

**目标**：现有 checkpoint 不改一个字节，就能以「每 4 帧一次」的节奏推理，
且输出与全量 `forward` 数值一致。

**契约**：

```python
state = middle.init_stream()                    # 每层 (k, v) = None，n_frames = 0
y, state = middle.forward_stream(z_chunk, state)
#   z_chunk : (B, n_new_frames, 4, 1792)         n_new_frames 任意正整数
#   y       : (B, n_new_frames, 4, 1792)         与全量 forward 对应位置一致
#   state   : 可继续传入；也可 detach 后跨 episode 重置
```

**实现要点**：

1. 逐层复用现有权重（`layer.norm1/2`、`self_attn.in_proj_*`、`out_proj`、`_ff_block`），
   只用 `F.scaled_dot_product_attention` 替掉 `nn.MultiheadAttention` —— 不重写任何
   注意力数学，只重写「K/V 从哪来」。这是本 feature 唯一新增的「轮子」，约 40 行。
2. mask 按 §2.4 的规则生成，尺寸随 chunk 变化，不再用固定 buffer。
3. `forward`（训练路径）**完全不动**。训练仍是一次性全序列并行。
4. 两条路径的一致性由自检保证（§6 验收项 1）。

**代价**：cache 显存随帧数线性增长（§4.2）。对 ≤60 s 的 episode（480 帧）是 83 MB，
可以接受；这也是把 P2（有界记忆）降级为「长时运行才需要」的原因。

**语义**：与现有模型**完全等价**，不引入任何新的近似。

### 3.2 P1 — 时间度量（零参数优先）

**问题**：middle 没有任何位置编码，顺序信息完全由 mask 的「可见集合」隐式编码。
后果有两个：

- 同一个内容序列，按 8 fps 采样和按 30 fps 采样，走完全相同的计算，模型无法区分；
- 帧间真实间隔（控制回路的 jitter）无从表达。

**两个候选，按侵入性排序**：

| 方案 | 做法 | 参数 | 训练 | 侵入性 |
|---|---|---|---|---|
| P1a ALiBi 帧距离偏置 | 在现有加性 mask 上加 `-m·(fi - fj)`（同帧内为 0，只按帧距离衰减） | **0** | 需短微调（1k 步） | 只改 mask 生成 |
| P1b Δt 输入通道 | Δt 标量 → 傅里叶特征（16 维）→ `Linear(16, 1792)` → 广播到该帧 4 行，**零初始化** | ≈ 28.7K | 需短微调 | 只改输入侧 |

两者都不碰残差流语义：P1a 只影响注意力内积，P1b 零初始化后起点精确等于原模型。
P1a 顺带缓解 §2.6 的注意力摊薄（远处帧被压低），P1b 才真正给「绝对时间」。

**建议**：先做 P1a（零参数，可解释），若发现不同控制频率下行为不一致，再加 P1b。

### 3.3 P2 — 有界记忆（延后，非必须）

**只在上线场景是「连续运行 > 5 分钟」时才需要**，因为 P0 的 cache 在 1 小时
@8 fps 下会涨到 4.95 GB。

两种做法：

- **P2a 滑窗淘汰**：只保留最近 W 帧的 K/V，更早的直接丢弃。零参数，但丢历史。
- **P2b 摘要槽**：保留 M 个可学习 summary token，被踢出的帧用一次 cross-attn 写进摘要。
  需要新增写入模块（约 12.9M/层），必须训练。

**决策**：不出现在 v1 范围内。若将来要做，优先 P2b —— 因为 P2a 会让「长期上下文」
变成「最近 4 秒」，与 v10「整片段细化」的设计意图直接冲突。

### 3.4 P3 — 动作条件（真正的「具身组件」）

P0~P2 做完，middle 只是「实时的视觉特征提取器」，还缺最关键的一环：**action**。

**方案：adaLN（DiT 式调制），零初始化**

```
a (B, A) → MLP → 每层 (γ, β, α)，初始化为 0
x = x + (1 + γ)·Attn(norm1(x))        # 或对 norm 的仿射做调制
```

- **为什么不用 action token 拼接**：每帧从 4 行变 5 行会破坏 `fi = idx // tpf` 的
  帧块结构，mask 和 `rows_rate=4` 的等宽约束全部要改，等于重做结构。
- **为什么是零初始化**：`γ=β=α=0` 时整个调制是恒等，**起点精确等于原模型**，
  可以直接从 P0 的 ckpt 热启动，不会因为多了一个未训练的支路而崩。
- **代价**：共享一个 `Linear(d, 6d)` 产出每层的（shift, scale, gate）× 2 组
  （attn / ffn），`d×6d ≈ 19.3M`；最后一个 Linear 零初始化 ⇒ 起点恒等。
  只在有 action 的数据上训。

**前置依赖**：手上有 (video, action) 对齐数据，且 action 空间（关节角 / 末端位姿 /
离散动作）已定。当前仓库没有这类数据，所以 P3 只写方案，不排期。

---

## 4. 资源账

### 4.1 算力（d=1792，6 层，ratio 4；1 MAC = 2 FLOPs）

每 token 每层：线性项 `4d²`（qkv+o）+ `8d²`（FFN）= 38.5M MACs。

| 口径 | 线性项 | 注意力项 | 合计 | 相对 |
|---|---|---|---|---|
| 全量 F=120（480 token） | 110.96 GMACs | 4.95 GMACs | **115.9 GMACs** | 1× |
| 每 4 帧增量（16 token） | 3.70 GMACs | 0.165 GMACs | **3.86 GMACs** | 30× 便宜 |
| 每帧增量（4 token） | 0.93 GMACs | 0.041 GMACs | **0.97 GMACs** | 120× 便宜 |

参照：`WholeClipDecoder`（readout）是 **11.1 TFLOP/clip**，middle 全量只有
0.23 TFLOP/clip —— **readout 是 middle 的 48 倍**。所以对「理解」场景，
`WholeClipDecoder`（30 步全片段重建）根本不该跑，延迟大头自然消失。

### 4.2 显存（KV cache）

每帧的 cache 字节数（6 层 × K/V × 1792 维 × 4 token × 2 B）= **172 KB/帧**：

| 时长 | 帧数 @8 fps | cache 显存 | 结论 |
|---|---|---|---|
| 15 s | 120 | 20.6 MB | 无感 |
| 60 s | 480 | 83 MB | **P0 可直接覆盖** |
| 5 min | 2400 | 413 MB | 开始有压力 |
| 1 h | 28800 | 4.95 GB | 需要 P2 |

### 4.3 延迟的真实下限：带宽不是算力

3.86 GMACs 在算力上微不足道，但每步都要把 231.4M 参数读一遍：

```
231.4M × 2 B (bf16) = 463 MB  →  假设 1 TB/s 有效带宽时 ≈ 0.46 ms/步
```

（1 TB/s 只是量级假设，A800 的 HBM 是 2 TB/s、消费卡/边缘卡更低。）

且每步只有 16 个 token，GEMM 极瘦（shapes 接近 GEMV），算力利用率通常只有 10~20%。
所以**实测延迟大概率由内存带宽决定，而不是 FLOPs**。上表所有算力数字只能用于
「量级判断」，真实延迟必须在目标硬件上实测（§8 开放问题 1）。

---

## 5. 非目标

1. **不重训 middle**。P0/P1 都不改 `forward`，不引入新的训练目标。
2. **不做自回归视频预测**。middle 不是 world model，它不做「给定动作预测未来帧」。
   那需要 P3 + 一个解码器，是独立 feature。
3. **不改码率口径**。每帧 7,168 值、`rows_rate=4`、`width=1792` 全部不动。
4. **不解决「特征是否适合控制」**。middle 是按重建训练的，保的是可重建信息，
   未必是控制所需的（接触高频、速度、可操作性）。这是 P3 之后、甚至独立一轮的事，
   本文只标注风险，不给方案。
5. **不覆盖 readout 的流式化**。`WholeClipDecoder` 已删 mask、全双向，
   历史帧输出会随新帧改变 ⇒ 天然不可增量。若将来要「边看边重建」，
   那是恢复因果 mask 的另一套设计，与本文无关。

---

## 6. 验收标准

P0 完成的定义（全部可自动检查）：

| # | 检查项 | 通过口径 |
|---|---|---|
| 1 | 增量 ≡ 全量 | `forward_stream` 逐帧喂入与 `forward` 整段喂入，`max|Δ| < 1e-5`（fp32） |
| 2 | 前缀性质 | 只喂前 k 帧的输出，等于整段喂入后取前 k 帧（误差 < 1e-6，即数值上完全一致） |
| 3 | 变长 | 1 / 3 / 120 / 480 帧均能跑，输出 `(B, n, 4, 1792)`，无断言失败 |
| 4 | state 可复用 | 同一 state 连续喂 30 个 chunk，结果与一次性喂 120 帧一致 |
| 5 | state 可重置 | `init_stream()` 后与首次调用行为一致，无残留 |
| 6 | 训练路径未变 | `python model_v10.py` 全过，`forward` 输出与改动前逐位一致 |
| 7 | 反例固化 | 若把 chunk 内 mask 去掉，自检 1 **必须失败**（防止规则被误删） |

第 7 条是「负向测试」，容易被忽略但它才是防回归的关键 —— §2.4 那个 0.39 的坑
就是这么踩出来的。

---

## 7. 路线图

| 阶段 | 内容 | 训练 | 交付物 | 前置 |
|---|---|---|---|---|
| **P0** | 变长 + 增量 KV cache | 无 | `forward_stream` + 自检 7 项 | 无 |
| **P1a** | ALiBi 帧距离偏置 | 1k 步微调 | mask 生成函数 + 外推 A/B | P0 |
| **P1b** | Δt 输入通道 | 1k 步微调 | 输入侧适配层 | P1a（可选） |
| P2 | 有界记忆 | 2k 步 | 仅长时运行需要 | 上线场景确认 |
| **P3** | adaLN 动作条件 | 需 action 数据 | 调制层 + 训练脚本 | 数据就绪 |

**建议顺序：P0 → 用真实数据验证外推（P1a 的 A/B）→ 再决定 P1b / P2 是否必要。**
不要一开始就把 P1/P2/P3 一起做 —— P0 的等价性一旦成立，就已经能覆盖 60 s 内的 episode。

---

## 8. 开放问题

1. **真实延迟未知**。§4.1 全是理论 MACs，瘦 GEMM 的实际效率、kernel launch
   开销、分词器耗时都没测。需要在目标硬件（含 MUSA）上跑一次 P0 原型。
2. **外推质量未知**。§2.5/§2.6 用的是随机初始化权重，只能证明「不崩」，
   证明不了「4× 长度下表示还有效」。必须在训练后的 ckpt 上测：同一段视频裁成
   120 帧 vs 480 帧，比较重叠部分的 `z` 余弦相似度。
3. **缓存精度**。cache 用 bf16 还是 fp32？bf16 下 30 步累积误差未测。
   建议 P0 默认 bf16 + 每 8 步回算一次（可配置），实测后再定。
4. **多 episode 的状态隔离**。state 里带 `n_frames`，跨 episode 忘记重置会静默
   继承上一段历史。需要 API 层面强制（比如 state 不可跨 `reset()` 复用）。
5. **帧率与 tpf 的耦合**。`tpf=4` 是按 8 fps × 0.5 s 设计的。若控制回路要 30 Hz，
   要么每帧单独成组（`tpf=4` 不变，但每帧都跑一次 middle，0.97 GMACs/帧），
   要么改 `tpf`（等于改结构、重训）。前者更现实。
6. **是否需要 projection 头**。`z` 是 1792 维 × 每帧 4 行，直接喂策略网络可能过宽；
   但加投影头就引入新参数与新训练目标，属于 P3 的范畴。

---

## 附录 A — 复现脚本（贴在文档里便于复核，未作为产品代码入库）

以下是 §2.3 / §2.4 结论的最小复现，用于复核「增量 ≡ 全量」。仅依赖 `model_v10.py`：

```python
import torch, torch.nn.functional as F
from model_v10 import TemporalMiddle

m = TemporalMiddle(in_dim=64, dim=64, depth=2, heads=4, ratio=2.0,
                   tokens_per_frame=4, frames=32).eval()
x = torch.randn(2, 32, 4, 64)
full = m(x)

def stream(m, x, cf):
    L = cf * 4
    fi = torch.arange(L) // 4                       # chunk 内帧号
    bm = torch.zeros(L, L).masked_fill(fi[None, :] > fi[:, None], float("-inf"))
    caches, outs = [None] * len(m.encoder.layers), []
    for t in range(0, x.shape[1], cf):
        h = x[:, t:t + cf].reshape(x.shape[0], -1, x.shape[-1])
        for i, layer in enumerate(m.encoder.layers):
            y = layer.norm1(h)
            q, k, v = F.linear(y, layer.self_attn.in_proj_weight,
                               layer.self_attn.in_proj_bias).chunk(3, dim=-1)
            B, T, d = y.shape
            H, Dh = layer.self_attn.num_heads, d // layer.self_attn.num_heads
            sp = lambda z: z.view(B, T, H, Dh).transpose(1, 2)
            q, k, v = sp(q), sp(k), sp(v)
            if caches[i] is not None:
                n_hist = caches[i][0].shape[2]          # 已缓存的 token 数
                k = torch.cat([caches[i][0], k], dim=2)
                v = torch.cat([caches[i][1], v], dim=2)
                # 历史列全 0 + chunk 内块因果（§2.4）
                mask = torch.cat([torch.zeros(T, n_hist), bm], dim=1)
            else:
                mask = bm
            caches[i] = (k, v)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            h = h + layer.self_attn.out_proj(o.transpose(1, 2).reshape(B, T, d))
            h = h + layer._ff_block(layer.norm2(h))
        outs.append(m.encoder.norm(h).view(x.shape[0], -1, 4, d))
    return torch.cat(outs, dim=1)

for cf in (1, 2, 4):
    print(cf, (stream(m, x, cf) - full).abs().max().item())   # ≈ 7e-07
```

注意：`_ff_block` 是私有方法，原型阶段可以借用；正式实现应改为
`layer.linear2(layer.dropout(layer.activation(layer.linear1(...))))`，
避免依赖私有 API。

## 附录 B — 数字来源

| 数字 | 来源 |
|---|---|
| 231.4M 参数 | 6 层 × 38.5M（`d=1792`, `ratio=4`，`4d²+8d² = 12d²`） |
| 116 GMACs / 3.86 GMACs | 本机按 §4.1 公式核算，与 `DESIGN_v10` §6 的 231.4M 一致 |
| 7.2e-07 等价性 | 本机实测（fp32，`frames=32`，2~6 层，三种 chunk 粒度） |
| 0.39（漏 mask） | 本机实测，同一配置 |
| 1.0079 输出 std | 本机实测（随机权重，8~480 帧） |
| 4.70 / 6.05 / 7.43 注意力熵 | 本机实测（随机权重，F=32/120/480） |
| 11.1 TFLOP/clip（readout） | `DESIGN_v10` 的分析结论 |
