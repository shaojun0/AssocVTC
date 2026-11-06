# DESIGN — vlm8b 读出侧改为循环架构（预算 = 循环步数）

> 日期：2026-11-06 ｜ 状态：**方案 v1，未实现，待实验验证**
> 对象：代码分支 `feat/vlm8b-v1`（tip `0340f40`）；循环读出参考实现与权重在 main：`model_v2_linattn.py`（`RecurrentOutputQueryDecoder` / `GrowthBlock`）+ 阶段 2 ckpt `exp448/B/ckpt_best.pt`（远端）
> 性质：技术讨论文档，本文**不含代码改动**。证据强度标注：`[代码]` 代码事实（带 file:line）、`[文档]` 仓库文档/实测报告、`[推断]` 推理。
> 支撑材料：vlm8b 分支结构分析与 pp_zh 机制考据为本地分析文档（未入库）；其结论已内联，本文可独立阅读。

---

## 0. 一句话结论

**能做，而且这是目前结构性最顺、代价最小的一条改造路径**——因为 pp_zh 的 looped transformer 本来就是一个**读出侧**的循环（`q_t = q_base + Y_{t-1}`、权重共享栈被施加 T 次、每步读一块 register），而 `feat/vlm8b-v1` 恰好是 pp_zh 架构里那条**"1 层 MLP → 语言模型"的通路 B**单独成项目；循环的参考实现与训练好的权重（479.7M 循环解码器）就躺在同一个仓库的同一份阶段 2 checkpoint 里，vlm8b 分支只是没把它接上。

但要把话说全：**改架构本身不难，难的是让"预算"真的被激活**。当前 run1 的 k 曲线几乎是平的（ΔCE(k8−k1) = −0.032，k≥2 即饱和），根因是 8 个视觉 token 只是 32 个 register 的线性 4:1 重分组、token 之间零通信、没有顺序语义。循环读出能提供的恰恰是"token t 建立在 token 1..t−1 之上、每步多读 4 个 register"的**因果增量语义**——这是 pp_zh 论文的立论，也是 vlm8b 当前最缺的东西。

**推荐方案（V1，预算 = 循环步数）**：第 t 步产出第 t 个视觉 token，步 t 只读第 t 块 4 个 register（与现有 token 分组一一对应），carry 走 BPTT；预算 k = 前 k 步 = 前 4k 个 register = k 个 LM token。LM 侧、数据侧、评测侧**零改动**。用阶段 2 的 decoder 热启动，新增参数约 5.3M，算力 <0.1%，显存 +0.4~4.8 GB/卡（取决于热启动规模）。

**否决项**：把 Qwen 主干改成循环。实测 8.93 s/step，S=8 约 71 s/step ≈ 75 h/epoch；激活 ×S 会直接顶穿 80G；且 Qwen3.5 是 24 层线性注意力 + 8 层全注意力的 hybrid，无预训练权重可继承。这不是这个预算档位能做的实验。

---

## 1. 先把参照物说清楚：pp_zh 的 looped transformer 是什么、在代码哪

### 1.1 结构（`[代码]` `model_v2.py` / `model_v2_linattn.py`；`[文档]` pp_zh `03_方法.md` §3.2–3.6）

```
图像 → 编码器 [cls; s_1..s_K; patch_1..patch_N]（全双向、无 mask，13/24 层）
      → 末层 LN 取前 1+K 个 → z_cls (1,D)、z_s (K,D)；patch token 不进读出
读出：A = [z_cls; z_s] + Π                (K+1, D_dec)
      Q_0                              (N, D_dec)   ← 固定查询基，行 ↔ patch
      for t = 1..T:                                 ← 权重共享的同一组 L 层 decoder
          Y_t = Stack( Q_0 + Y_{t-1},  A[lo_t : hi_t+1] )   ← 只读"本步那块" register
          loss += L1( PixelHead(Y_t), image_patches ) / T    ← 每步直接预测整图、等权
      F_hat = Y_pix[:, -1]                             ← 推理取末步
```

三个必须同时成立的设计要素（这是 pp_zh 论文的实证结论，不是实现细节）：

| 要素 | 结论 | 证据 |
|---|---|---|
| carry 不 detach（BPTT） | detach→BPTT：像素 L1 23.72→**16.42（−30.8%）**，逐步曲线由平转降 | `[文档]` repo README / `doc/2026-09-16/REPORT_v2_bptt_vs_detach.md`；梯度扇出 1→9.49 |
| 每步都有直接监督 | 只有末步监督 + detach 会退化成"T 个共享权重的并行头" | 同上；pp_zh §3.6 |
| 恰好一个 step identity | 每步独立 `[cls]` 槽最佳（−10.3%）；0 个或 2 个 identity 都会双稳态塌缩 | `[文档]` `doc/2026-09-12/EXPERIMENT_step_identity_arm4_stepidcls_stepid.md`；arm-3/arm-4a |

> 注：pp_zh 论文 §4 报的数来自 **224²/240M 的 linattn 变体**（K=32、T=16、fixed block=2），不是 `model_v2.py` 的 K=576/24 步主线；两代数字不可横比（两代实现的操作点不同，见 `doc/2026-11-04/RUN_448_960M_3stage.md`）。

### 1.2 这个仓库里的三代实现，与 vlm8b 的位置

| 代 | 模型 | 操作点 | 状态 |
|---|---|---|---|
| L1 | `model_v2.py` | 448×252 / N=576 / K=32 / T=24 / square 窗口 | 论文早期主线，数字已被 L2 取代 |
| L2 | `model_v2_linattn.py` | 224² / N=196 / K=32 / T=16 / fixed block=2 / 240M+240M | **pp_zh 论文 §4 的模型** |
| L2-448 | 同上，448²/p32/960M | N=196 / K=32 / T=16 / block=2 / enc 480.5M + dec 479.7M | **vlm8b 阶段 2 ckpt 的来源**（`/root/autodl-tmp/exp448/B/ckpt_best.pt`） |
| vlm8b | `feat/vlm8b-v1` 的 `model.py` | 只取 L2-448 的 **encoder**；z_s(32,1744) → 8 个 LM token | 当前分支 |

关键对照：**pp_zh 论文 §3.1 明确写了两条并列通路**——通路 A 是循环读出到像素重建，通路 B 是"经一层 MLP 投影交给语言模型"；论文还自己标注"语言侧尚未开展实验"。`feat/vlm8b-v1` 就是通路 B 的独立工程实现。所以"像 pp_zh 那样改成 looped transformer"= **把通路 B 的 MLP 换成通路 A 的循环读出**（或让 A/B 共享同一个循环读出），而不是去动语言模型主干。

---

## 2. 现状：vlm8b 分支是一台"被拆掉循环"的机器

### 2.1 代码事实：视觉侧现在是纯线性重分组，无任何循环

`[代码]` `model.py:152-166`：

```python
tokens = self.token_proj(self.token_norm(z_s))        # (B,32,1744) → (B,32,640)，逐 token
return self.budget_norm(tokens.reshape(B, 8, 4*640))  # (B,8,2560)
```

- 8 个 visual token **互不通信**，第 j 个 token = `z_s[4j:4j+4]` 的线性压缩；预算 k = 前 4k 个 `z_s` token（`[代码]` `model.py:113-118`）。
- 整个视觉侧循环参数只有 **1.125M**；`encoder.py` 只 vendored 了 encoder 三个类，解码器/像素头没带过来（`[代码]` `encoder.py:10-12`）。
- 分支内 grep 不到任何功能性 loop/recurrent 代码（对分支全文件 grep `loop|recurr|iter|weight.?tied|readout|growth` 只命中 4 条非功能残迹：`encoder.py:155` 的 GrowthBlock 注释、`README.md:55` 的 Qwen recurrent mask、`model.py:244-247` 的 `_tied_weights_keys`、`ds_zero3.json:14` 的 `stage3_max_reuse_distance`）。

### 2.2 被丢掉的 479.7M 循环解码器

L2-448 阶段 2 的 ckpt 是完整的 `SRPhase1LinAttn`：`encoder`(480.5M, 13×1744) + `decoder`(**479.7M, 7×2048**，`RecurrentOutputQueryDecoder`，steps=1..16 / fixed block=2)（`[代码]` `model_v2_linattn.py:805-915`；`[文档]` `doc/2026-11-04/RUN_448_960M_3stage.md` §1、`doc/2026-11-04/REPORT_448_960M_3stage.md` §2）。vlm8b 的加载器只抽 `encoder.*` 前缀（`[代码]` `model.py:194-202`），479.7M 循环读出权重原样躺在 ckpt 里没被用。

### 2.3 症状：预算曲线几乎平

`[文档]` `doc/2026-11-06/REPORT_vlm8b_1ep.md` §2（n=250 随机子集，seed 42，修正了旧版按类排序的偏置）：

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| CE | 0.4989 | 0.4684 | 0.4683 | 0.4678 | 0.4683 | 0.4669 | 0.4668 | 0.4666 |
| acc | 12.40% | 14.00% | 13.60% | 11.60% | 10.40% | 12.80% | 11.60% | 12.00% |

ΔCE(k8−k1) = −0.032，k≥2 即饱和；训练内 per-budget CE 各档几乎无差（0.507–0.519）。视觉消融（normal 12% / null 1.2% / shuffle 0.8%）证明**视觉通路是真的，但读得浅**。

> 本地 `vlm8b_eval/eval.md` 里 26–31% 的一版是**按类排序子集**的旧结果，已被上面这份随机抽样报告取代（`doc/2026-11-06/REPORT_vlm8b_1ep.md` §1）。

### 2.4 为什么平：当前结构没有给"预算"任何语义

`[推断]` 三条结构性原因，环环相扣：

1. **token 之间没有信息流**。token 1 和 token 8 是同一张图不同 4 个 register 的平行投影；多给 7 个 token，模型只是多拿到 7 个**独立**信息块，而不是"更完整的一张图"。
2. **没有顺序语义**。哪个 register 落到哪个位置完全由 encoder 训练时决定，投影头没有理由让"前 4k 个"比"随便 4k 个"更粗或更细；"k 越小越难"的单调关系不被任何结构保证。
3. **8 个 token 对同一个文本目标、各自独立**。这与 pp_zh 记录过的"零增量自锁"是同构的病：后一个 token 的最优解是"复制/冗余/无关紧要"，于是 CE 在 k≥2 就饱和。

对照：pp_zh 用"权重共享循环 + 每步读新一块 register + carry"来强制"第 t 步必须在第 t−1 步基础上增量"，这是目前唯一被该仓库验证过能制造"逐步细化"的结构。

---

## 3. 改造落点：三种"looped"位置与取舍

| # | 落点 | 改法 | 成本 | 判断 |
|---|---|---|---|---|
| **L1** | **视觉读出侧**（`model.py` projector） | 用权重共享循环栈把 `[z_cls; z_s]` 逐块读成 8 个 LM token | 算力 <0.1%；显存 +0.4~4.8 GB/卡；不动 LM/数据/评测 | ✅ **推荐** |
| L2 | encoder 内部（13 层里重复某几层） | 让某 block 被复用 S 次 | 算力 +2%；E1 冻结期不可训；加参撞 `strict=True` 加载（`model.py:202`） | ❌ 不推荐（pp_zh 也明确"不在主干加循环"） |
| L3 | Qwen 主干 / 序列级循环 | 复用 LM 层栈多次前向 | 墙钟 ×S ≈ 71 s/step、75 h/epoch；激活 ×S；hybrid 层不可简单复用；无权重继承 | ❌ 否决 |
| L0 | 过渡臂：`z_s` 上做共享块细化（vlm8b-analyst 的 AP-2） | 32 个 token 上跑 S 次共享 block，再线性投影 | +0.25% 算力；token 数不变 | ⚠️ 只能当"加深度"对照，拿不到"预算=步数"语义 |

> L0 的价值：它是**最便宜的赌注**——如果连"在 z_s 上多做几次共享块细化"都没有增益，那循环读出大概率也不会有，可以先花半天排除一种可能。但它本质是"把 encoder 末尾再加几层并循环"，与 pp_zh 的"读出侧循环"不是同一个东西。

---

## 4. 推荐方案 V1：预算 = 循环步数（Budget = Loop Steps）

### 4.1 设计目标

一句话：**让"第 k 个预算"第一次有明确含义**——预算 k 就是读出执行了 k 步：
- 只读过前 4k 个 register（信息截断是真的）；
- 只产出 k 个 LM token（token 截断是真的）；
- 只算了 k 次读出（算力截断是真的）。

三者对齐，正是 pp_zh 论文"截断能力"在 VLM 侧的对应物。

```
现状（线性重分组）                          V1（预算 = 循环步数）
z_s (32×1744)                              z_cls(1) ┐
  │ LN+Linear(1744→640) 逐 token            z_s(32) ─┴→ mem_proj→norm→+pos ⇒ A (33×d)
  │ reshape(8, 4×640=2560)                         │
  │ LN(2560)                                Q[1] ───┴→ Stack(Q[1],       A[0..4])  → V1
  ▼ 8 个 token 互不通信                      Q[2]+Y1 ──→ Stack(Q[2]+Y1,   A[5..8])  → V2
[V1 V2 … V8]（每个 = 4 个 register）         …
  ▼                                        Q[8]+Y7 ──→ Stack(Q[8]+Y7,A[29..32]) → V8
预算 k = 前 4k 个 register                    预算 k = 前 k 步 = 前 4k register = k token
（信息截断真、token 截断真、算力截断假）       （三者对齐；V_t 出现在第 t..8 行 ⇒ 每步都有 CE 监督）
```

### 4.2 结构与公式

```
memory  A = [z_cls; z_s] (B, 33, 1744) → mem_proj(1744→d) → mem_norm → + mem_pos   (B,33,d)
查询    Q     (8, d)            ← 每行 = 一个 token 的"提问模板"（唯一的 step identity）
步     t = 1..8, 窗口 fixed block=4：
         W_t = A[:, lo_t : hi_t+1]        # 步1: [z_cls + reg1..4]；步t: reg[4t-3..4t]
         Y_t = Stack( Q[t] + Y_{t-1}, W_t )          # Stack = L 层，权重跨步共享
         V_t = OutProj( norm_out(Y_t) )  → (B, 1, 2560)
         Y_0 = 0
vis   = cat(V_1..V_8) → (B, 8, 2560) → budget_norm(2560) → 接现有 8 行预算流程
```

- `Stack`：2 层（pp_zh 原式）或 7 层（沿用阶段 2 ckpt 的 dec 配置）；每步只有 **1 个 query token**，所以自注意力退化为恒等，实际是"交叉注意力 + FFN"被重复施加 8 次，等效深度 2×8=16 或 7×8=56。
- carry **不 detach**（BPTT）。8 步 × 1 query 的 BPTT 代价可以忽略（见 §7）。
- **不要再加 step embedding**：8 个 query 行本身就是 8 个 step identity，再加一套就是 pp_zh arm-4a 的"两套 identity 互相干扰"（0.39763 vs 最佳 0.37442）。

### 4.3 伪代码（落到 `model.py` 的最小形态）

```python
class LoopVisualReadout(nn.Module):
    """z_s/z_cls → 8 个 LM token；第 t 步产出第 t 个 token。"""

    def __init__(self, enc_dim=1744, dim=2048, depth=7, heads=16, mlp_ratio=4.0,
                 n_tokens=8, n_specials=32, lm_hidden=2560, block=4):
        super().__init__()
        self.mem_proj = nn.Linear(enc_dim, dim) if enc_dim != dim else nn.Identity()
        self.mem_norm = nn.LayerNorm(dim)
        self.mem_pos  = nn.Parameter(torch.zeros(1, 1 + n_specials, dim))   # 与 ckpt 同名同形状
        self.q_bank   = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)     # 8 个独立 query 行
        self.stack    = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim*mlp_ratio,
                                       batch_first=True, norm_first=True),
            num_layers=depth)                                               # 权重跨步共享
        self.norm_out = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, lm_hidden)
        self.steps   = list(range(1, n_tokens + 1))
        self.windows = step_windows(n_specials, self.steps, plan="fixed", block=block)
        # fixed/block=4 ⇒ [(0,4),(5,8),(9,12),...,(29,32)]，8 步覆盖全部 32 个 register

    def forward(self, z_cls, z_s):                       # (B,1,1744), (B,32,1744)
        mem = self.mem_norm(self.mem_proj(torch.cat([z_cls, z_s], 1))) + self.mem_pos
        outs, Y = [], None
        for t, (lo, hi) in enumerate(self.windows):
            q = self.q_bank[t].view(1, 1, -1).expand(mem.size(0), -1, -1)
            q_in = q if Y is None else q + Y              # carry：BPTT，不 detach
            Y = self.stack(q_in, mem[:, lo:hi + 1])       # (B,1,d)
            outs.append(self.out_proj(self.norm_out(Y)))  # (B,1,2560)
        return torch.cat(outs, dim=1)                     # (B,8,2560)
```

> `step_windows(...)` 是 main 分支 `model_v2_linattn.py:628-675` 的现成函数（20 行，fixed/square/cumulative 三种计划），vendored 进本分支即可；`fixed + block=4` 的窗口正好是 `[(0,4),(5,8),...,(29,32)]`，与现有"每 4 个 register 一个 token"的分组一一对应。

对接只需三处（见 4.4），`expand_budget_rows` / LM 前向 / CE / 评测**一行都不用改**。

### 4.4 三处工程对接点（契约清单）

| 位置 | 现状 | 要改什么 |
|---|---|---|
| `AssocVTCEncoder.forward`（`model.py:204-206`） | 只返回 `self.net(x)[1]`（z_s）；`encoder.py` 本身已返回 `(z_cls, z_s)` 二元组 | 改成返回 `(z_cls, z_s)`；`model.py:295` 相应取两个值 |
| `model.py` 的 `self.projector` | `BudgetProjector`，输出 vis | 换成 `LoopVisualReadout`（或加 `--readout {linear,loop}` 开关保留原路径做对照） |
| `train.py:build_param_groups` | 三个组：lm / projector / encoder（`train.py:148-157`） | **不用改**——循环模块挂在 `model.projector` 下即自动进 proj 组；但 `lr_proj` 要从 1e-4 降到 ~1e-5~3e-5（热启动的栈不能用投影头的大 lr） |
| `train.py:108-126` / `eval.py:95` | `--init-from`/eval 只允许缺 `lm.lm_head.weight` | 若循环模块新加键，需把允许缺失集合放宽（否则续训/评测直接断言失败） |
| `smoke.py:147-173` | 验证"行 k 只由 vis[:k] 决定" | 增加两项：① S=1/`block=33` 时退化为"全读"；② 前缀因果性（改 `z_s[4]` 只影响 vis[1] 及之后，不影响 vis[0]） |

### 4.5 权重来源：三档热启动

阶段 2 ckpt 的 `decoder.*` 里可直接复用的键：`stack.*`（7×2048，约 470M）、`mem_proj.*`、`mem_norm.*`、`mem_pos`、`norm_out.*`；**要丢弃/重建**的：`query_base (196,2048)`、`query_pos (196,2048)`、`pixel_head.*`（3072 输出）。新增：`q_bank (8,2048)=16K`、`out_proj (2048→2560)=5.24M`。

| 档 | 栈 | 复用参数 | 新增参数 | 显存增量/卡（ZeRO-2） | 适用 |
|---|---|---|---|---|---|
| **A 全量热启动** | 7 层 × 2048，沿用 ckpt | ~474M | ~5.3M | **+4.8 GB**（72.7→~77.5，很紧） | 论文级"最忠实"臂 |
| **B 浅热启动** | 2 层 × 2048，取 ckpt 的 layer 0–1 | ~134M | ~5.3M | +1.3 GB | ✅ **首跑推荐**（逼近 pp_zh 的 2 层栈深度） |
| **C 小栈从零** | 2 层 × 1024 | 0 | ~36M | +0.4 GB | 最便宜的假设检验 |
| D 冻结 ckpt 解码器做特征 | 7 层 × 2048 冻结，每步池化 | 全部冻结 | pooling 头 | 与 A 同 | 只当探针，不推荐主力 |

> `[推断]` A 档显存偏紧：当前实测 72.7 GB/卡、激活约 38 GB（`doc/2026-11-06/REPORT_vlm8b_1ep.md` §5、`[文档]`），+4.8 GB 参数态接近 80 G 上限，需要降 micro-batch 或改 ZeRO-3（当前 torch 2.12 + deepspeed 0.19.9 有 backward 死锁前科，`train.py:41-43`）。B 档是更稳的第一步。

### 4.6 监督从哪来：8 行预算 CE **免费**提供逐步监督

这是本方案最漂亮的一点：pp_zh 在 VLM 侧最难的地方是"每步 CE 要过一遍 LLM，成本 ×T"（这是 pp_zh 方案迁到 VLM 侧的最大成本项）。而 vlm8b 的并行 8 预算训练**本来就是每步过 8 次 LLM**（`model.py:305-346`），于是：

- 行 k 的视觉前缀 = `vis[:k]` = 循环前 k 步的输出；
- 第 t 步的 token 出现在第 t..8 行里 ⇒ 自动获得 8−t+1 个 CE 监督项；
- 不需要额外损失、不需要改 `forward_rows`、不需要增加任何 LLM 前向。

换句话说：**8 行预算并行原本是为了训练效率，但把它和循环读出拼起来，它恰好变成了"逐步深监督"**——pp_zh 论文里"每步等权 L1"在 VLM 侧的同位替换，而且是免费的。这也避开了 pp_zh 的 detach 臂教训（每步无梯度 → 平线）。

### 4.7 可选：像素辅助头（把两条通路重新统一）

如果想更贴近 pp_zh 的"双通路共享循环"，可以在 `Y_t` 上再接一个小 `PixelHead`（复用 ckpt 里的 2048→3072 头）对原图 patch 做 L1，权重 λ 小（如 0.1），单步成本约 `8 × 196×3072` 个值，相对 LLM 可忽略。价值：

- 给循环一个**密集**信号（CE 只有 4 个 answer token，梯度很稀）；
- 阶段 2 的像素语义可以在 E1（encoder 冻结）期间"锚住"循环，防止它被 CE 带偏（见风险 R4）；
- 论文叙事上，VLM 变成"同一个循环读出 + 两个头（像素 / LM）"，即 pp_zh 图 1 的两条通路由**同一个循环**实现，这是一个比现状更强的主张。

代价：需要 `data.py` 额外吐 patch 目标（`image_to_patches` 现成），以及 λ 的调参。建议作为 V1 的第二阶段选项，不放进首跑。

### 4.8 与 pp_zh 原式的差异（写论文时必须说清）

| 维度 | pp_zh | V1（VLM 侧） |
|---|---|---|
| 循环输出 | 每步预测**整图**（N 个 patch） | 每步产出**一个新 token**（共 8 个） |
| 查询 | `query_base` 196 行（行↔patch），跨步重复使用 | 8 行，每步用一行（行=token 身份） |
| 监督 | 每步像素 L1（等权） | 每步由预算行 k≥t 的文本 CE 监督（权重由 loss 归一化决定） |
| 读出 | `F_hat = 末步` | 每个预算 k 用前 k 步（**每步都可用**，不取末步） |
| 截断 | 训练后提前停步，无需重训 | 训练时就在所有 k 上训（预算集合 = 1..8） |
| 循环语义 | 同一步细化整幅图 | 同一循环**生长** token 序列（ALIT/GrowthBlock 式生长） |

---

### 4.9 本机原型验证（已跑通，缩小版 20 行）

在 torch 2.8 CPU 上写了等价原型（`doc/2026-11-06/artifacts_looped/proto_loop_readout_check.py`，dim=256 / depth=2 / lm_hidden=64 的缩小版），验证了三件对方案成立最关键的事：

| 检查 | 实测结果 | 含义 |
|---|---|---|
| 形状 / 窗口 | vis `(2,8,64)`；windows `[(0,4),(5,8),…,(29,32)]` | 8 步恰好覆盖 32 个 register + z_cls（首步多含 z_cls，与 main 的 fixed 计划一致） |
| 前缀因果性 | 扰动 `z_s[4:8]`：vis[0] 变化 **0.0**，vis[1..7] 有变化；扰动 `z_s[20]`：vis[0..4] 全 **0.0**，只有 vis[5..7] 有变化 | **预算 k 的前缀语义成立**：token t 只依赖前 4t 个 register，截断不会"偷看"后面的 register |
| BPTT vs detach | 用末 token 的 loss 反传：`grad q_bank[0]` = **4.08**、共享栈 layer0 的 grad norm = **1.93**；detach 版同位置 = **0.0** | pp_zh 的"detach 静默掐断早步信用分配"在 VLM 侧同样成立 ⇒ 必须 BPTT |

这三项正好是把现有 `smoke.py`（7 项）扩到 9–10 项时要加的检查：形状契约、前缀因果性、BPTT 存在性。

---

## 5. 备选变体与适用场景

| 变体 | 一句话 | 优点 | 代价 | 建议 |
|---|---|---|---|---|
| **V1**（本方案） | 8 步 × 1 token，预算=步数 | 预算语义最干净；每步有免费监督；pp_zh 截断叙事直接迁移 | 每步 1 个 self-attn 退化（无 token 间自注意力） | ✅ 首跑 |
| **V1′** | 第 t 步对 query 1..t 做 causal self-attn，只读出第 t 个 | 有 token 间通信，更接近"真正的 looped transformer" | 计算 8→36 个 query-step（仍 <0.5%）；实现稍复杂 | 若 V1 有信号但增益有限，再做 |
| **V2** | 8 个 query 行被同一组 8 步反复精修，取末步输出 8 token（pp_zh 原式的直接移植） | 代码最少（直接搬 `RecurrentOutputQueryDecoder`，只换 head）；query 行可细化为"整图—token"映射 | 预算 = token 前缀，与"步数"解耦；"预算=算力"叙事弱 | 作为 V1 的消融臂 |
| **V3** | 完整复用 ckpt 的 196-query 循环 + 每步池化出一个 token | 最大程度继承 479.7M 预训练读出；可同时保留像素头 | 算力 +19%（8 步 × 7 层 × 196 token）；池化头是新的 | 预算充足时的"最忠实"臂 |
| **V0** | z_s 上做共享块细化（L0），再接线性投影 | 最便宜（+0.25%），E1 可训 | 不是读出循环，拿不到预算=步数 | 半天排除法 |

---

## 6. 为什么值得做：机制论证 + 反例对齐

1. **补上"顺序/增量"这个结构性缺口**：现在 8 个 token 是同一张图的 8 个平行切片；循环让 token t 看到 token 1..t−1 的表示（carry）并只新增 4 个 register 的信息 ⇒ 前缀 k 天然是"累积的充分统计量"。这是 pp_zh "由粗到细"的唯一结构条件。
2. **让预算的三种截断对齐**：信息（前 4k register）、token（前 k 个）、算力（前 k 步）第一次是同一个 k。当前只有第一种是"真"的，token 数变化对 LLM 的注意力成本有影响，但读出算力完全不变。
3. **避开 pp_zh 踩过的三个坑**（这是该仓库最有价值的资产）：
   - detach → 平线：V1 用 BPTT（代价可忽略，因为只有 1 个 query）。
   - 只有末步监督 → 中间档不可用：V1 由 8 行预算免费提供每步监督。
   - 两套 identity → 塌缩：V1 只用 8 个 query 行，不再加 step embedding。
4. **反例警示（必须写进实验设计）**：pp_zh 的 K=120/576 时代逐 step L1 **先降后升**（K=120 最低点在第 7 步 12.4877，末步 12.5095；24 步配置最优在第 17 步，之后 slightly worse）——"每步直接预测同一目标"会让后步过冲。VLM 侧的对应现象就是"k=8 反而不如 k=4/5"。所以**评测必须报每档曲线**，不能只看 k=8；如果出现 U 形，那是机制信号，不是噪声，要当作结果分析（这也是 pp_zh 论文自己写进 §4 的诚实做法）。
5. **"改架构"不会顺带解决欠训/遗忘**：run1 的 acc 12%、通用能力 75%→1.7% 是数据量/训练量/窄域全参 SFT 的问题（`doc/2026-11-06/REPORT_vlm8b_1ep.md` §4/§8），循环读出对此无能为力。别把两件事混在一个臂里。

---

## 7. 成本与排期

| 项 | 数值 | 依据 |
|---|---|---|
| 现模型 | LLM ≈3.86B（文档估）+ enc 480.5M + proj 1.125M ≈ **4.34B** | `[文档]` DESIGN §3；`[代码]` 结构手算 |
| V1-B（浅热启动 2×2048） | +~140M 参数，+1.3 GB/卡 | `[推断]` 按 ZeRO-2 参数/梯度/Adam 口径估算 |
| V1-C（小栈 2×1024） | +~36M，+0.4 GB/卡 | 同上 |
| V1-A（全量热启动 7×2048） | +~479M，+4.8 GB/卡（72.7→~77.5，紧） | 同上 |
| 读出算力（V1，7 层栈） | ≈3.8 GMACs/图 ≈ **0.1%** 的 LLM 前向（8 行 × 120 tok × 4B ≈ 3.84 TMACs/图）；2 层栈约 1.1 GMACs/图 | `[推断]` 手算 |
| V3（196-query 版） | ≈736 GMACs/图 ≈ **19%** | `[推断]` 手算 |
| 代码量 | `model.py` 新增 ~200 行（模块+加载+开关）；`train.py` ~20 行（`--readout`、`--loop-init`、lr 组）；`smoke.py` +2 项；`eval.py` 0 | `[代码]` 现有结构 |
| 本机 CPU smoke | 半天（torch 2.8 可用，`smoke.py` 现有 7 项全 PASS 的口径不变） | `[文档]` README §4 |
| 远端判定成本 | 5k 图子集 1 epoch ≈ 157 步 × 8.93 s ≈ **23 min/臂** | `[推断]` 按 32 图/step、8.93 s/step |
| 全量 1 epoch（对照用） | 9.44 h × 2 卡 | `[文档]` `doc/2026-11-06/REPORT_vlm8b_1ep.md` §1 |

**排期建议**：不要打断 run2（ETA 11-07 06:00）。新建 `feat/vlm8b-loop` 分支：

1. Day 1：`model.py` 加 `LoopVisualReadout` + `--readout/--loop-init/--loop-depth` 开关；本机 CPU smoke（含新增 2 项）；remote `--smoke` 2 步验证 DeepSpeed 下参数组与保存路径。
2. Day 2：5k 子集 A/B：臂 0 线性投影 / 臂 C 小栈从零 / 臂 B 浅热启动；各 1 epoch（~25 min），比较 per-budget 曲线。
3. 判定通过后再排全量 1 epoch（~9.5 h/臂）或并入预算课程。

---

## 8. 风险清单

| # | 风险 | 来源 | 对策 |
|---|---|---|---|
| R1 | 循环退化成"8 个并行头"（平线） | pp_zh detach 臂实测 | BPTT + 8 行预算的逐步监督（本方案天然满足）；若仍平，查 per-budget CE 曲线是否随 k 单调 |
| R2 | identity 双稳态塌缩 | pp_zh arm-3/arm-4a | 只用 8 个 query 行，不加 step embedding；smoke 加"改 z_s[4] 不影响 vis[0]"检查 |
| R3 | 热启动的分布偏移（像素目标 → CE） | `[推断]` | 先浅热启动/小栈；E1 冻结 encoder 时观察 per-budget CE；必要时加像素辅助（§4.7） |
| R4 | bf16 下 8 步 carry 数值漂移 | pp_zh fp16+8bit 抹掉循环收益的前科 | carry/norm 用 fp32；固定 seed 与线性臂对拍；报"同配置重跑分叉"的量级 |
| R5 | 显存顶穿 | 实测 72.7/80 GB | 浅热启动（+1.3 GB）或小栈（+0.4 GB）；必要时降 micro-batch；不要贸然开 ZeRO-3（死锁前科） |
| R6 | 新参数漏出 param groups 被静默不训 | `train.py:148-157` 只认 lm/projector/encoder | 循环模块必须挂在 `model.projector` 下；smoke 加"所有可训练参数都在 optimizer 里"的断言 |
| R7 | `--init-from` / `eval.py` 断言失败 | `train.py:123-124`、`eval.py:95` | 放宽 missing 集合（只允许缺 `lm.lm_head.weight`） |
| R8 | 评测判不出"循环有没有用" | `eval.py` 只有 CE+EM，缺 L1/L2/L3、bootstrap、m 曲线 | 至少报 per-budget CE/acc 曲线 + ΔCE/Δacc + 截断-满预算 gap；按评测协议补 m ∈ {8,16,32,64} 的 m–质量曲线与 bootstrap CI |
| R9 | 把它当"解决欠训/遗忘"的方案 | run1 实测 75%→1.7% | 分开归因；循环只解决"读出结构"，不解决数据/训练量/灾难遗忘 |

---

## 9. 最小可证伪实验（判定标准）

**臂**（同数据、同 step 数、同 lr 组、seed 固定）：

- **臂 0**：现行线性 projector（基线 = run2/课程同口径）。
- **臂 1**：V1 + 小栈（2×1024 从零）。
- **臂 2**：V1 + 浅热启动（2×2048，取 ckpt layer 0–1）。
- **臂 3**（可选）：V2（pp_zh 原式移植，末步输出 8 token）。
- **臂 4**（可选）：V0（z_s 共享块细化）——用来分离"加深度"与"读出语义"。

**规模**：5k 图子集，1 epoch/臂（~25 min/臂），n=500 随机子集评测，seed 42。

**指标**（缺一不可）：

1. per-budget CE 与 acc 曲线（k=1..8）；
2. ΔCE = CE(k=1) − CE(k=8)、Δacc = acc(k=8) − acc(k=1)（当前基线 ΔCE=−0.032、Δacc=−0.4%）；
3. 截断-满预算 gap（评测协议的 m–质量曲线口径：m ∈ {8,16,32,64}）；
4. 视觉消融（null/shuffle）在 k=1 与 k=8 上的差异；
5. 墙钟/显存（确认 <2% 单步开销）。

**判定规则**：

- 通过：臂 1/2 的 per-budget 曲线**开始单调**（或 ΔCE/Δacc 显著大于基线），且 k=8 不低于基线（对齐税 ≤ 噪声）。
- 不通过但曲线变 U 形：说明循环有效但"后步过冲"（pp_zh 同款现象），加 budget 权重（`inv-k` 或对 k=8 降权）再判。
- 完全无差别：问题不在读出结构，而在这批数据（4-token 闭集答案太短，预算信号天然饱和，`doc/2026-11-05/DESIGN_8budget_vlm_finetune.md` §2.1 已预警），应转 think 数据/更长答案，而不是继续调架构。
- 训练内 per-budget CE（`per_budget_ce.jsonl`）必须与评测曲线同向，否则先查 shift/掩码契约（`smoke.py` 的 P0 项）。

---

## 10. 结论与建议动作

1. **可以做，且推荐**：把视觉读出侧从"1 层线性重分组"换成 pp_zh 式的权重共享循环读出（V1），预算 k 绑定循环步 k。改动集中在 `model.py` 一个模块 + 加载/开关，LM/数据/评测零改动。
2. **用现成资产**：阶段 2 ckpt 里的 479.7M 循环解码器就是训练好的读出栈，浅层热启动即可；这是"pp_zh 架构 → vlm8b"最省的一段路。
3. **三个必须同时满足的条件**（否则重演平线）：BPTT 不 detach；每步都有直接监督（8 行预算 CE 免费给）；恰好一个 step identity（8 个 query 行，别加 step embedding）。
4. **先 5k 子集 A/B 再全量**：单臂 25 min，一次判定"循环是否让预算曲线站起来"。
5. **不要动 Qwen 主干**：预算档位不允许；pp_zh 的循环本来也不在语言侧。
6. **诚实边界**：循环读出是结构改动，不解决欠训、灾难遗忘、标签噪声；若 think 数据/更长答案能让预算曲线自然站起来，那循环的边际价值需要重新评估——这也是消融臂 0 必须同口径的原因。

---

## 附录 A：事实来源索引

| 结论 | 证据 |
|---|---|
| pp_zh 循环读出的结构/公式/每步监督 | 循环读出压缩论文（`/userdata/paper/pp_zh`）§3.2–3.6；`[代码]` `model_v2.py:583-612,768-774`；`model_v2_linattn.py:805-915` |
| detach→BPTT −30.8%、扇出 1→9.49、曲线由平转降 | `[文档]` `doc/2026-09-16/REPORT_v2_bptt_vs_detach.md`、`doc/2026-09-16/RESULTS_v2_bptt.md` |
| step identity 必须恰好一个 | `[文档]` `doc/2026-09-12/EXPERIMENT_step_identity_arm4_stepidcls_stepid.md`、`doc/2026-09-12/REPORT_step_identity_AB.md` |
| 逐 step L1 先降后升（K=120 第 7 步最低、末步回升） | `[文档]` `doc/2026-09-16/REPORT_ksweep_k120.md`、`doc/2026-09-16/ANALYSIS_ksweep_stepwise_refinement.md` |
| vlm8b 视觉侧纯线性重分组、无循环 | `[代码]` `feat/vlm8b-v1` `model.py:152-166`（逐 token LN+Linear 后 reshape，无跨 token 混合）、`model.py:113-118`（预算=前 4k register）；全文件 grep 无循环实现 |
| 阶段 2 ckpt 含 enc 480.5M + dec 479.7M（7×2048，16 步） | `[文档]` `doc/2026-11-04/RUN_448_960M_3stage.md` §1；`doc/2026-11-04/REPORT_448_960M_3stage.md` §2 |
| 448/960M 阶段 2 逐步 PSNR 21.01→26.16（14–16 步平台） | `[文档]` `doc/2026-11-04/REPORT_448_960M_3stage.md` §3.2 |
| run1 预算曲线几乎平（ΔCE −0.032、acc 12%） | `[文档]` `doc/2026-11-06/REPORT_vlm8b_1ep.md` §2 |
| 8.93 s/step、72.7 GB/卡、1 epoch 9.44 h | 同上 §1/§5 |
| 接入点/契约/显存估算 | 本文 §4.4/§7；`[代码]` `feat/vlm8b-v1` `train.py:148-157`（param groups）、`model.py:202`（strict 加载） |
| Qwen 主干循环不可行（×S 墙钟/显存） | `[推断]` 本文 §3/§7：8.93 s/step × S=8 ≈ 71 s/step、激活 ×S |

---

## 附录 B：后续验证入口

- **已落地（2026-11-06）**：分支 `feat/vlm8b-loop`（commit `a16863b`）——从 `feat/vlm8b-v1` 分叉、只含 12 个文件的同构最小工程，实现本方案 V1：`--readout loop`（默认 `linear` 与基线逐位一致，可直接 A/B）。本机 `python3 smoke.py` **23 项 PASS / 0 FAIL**（CPU，torch 2.8），覆盖前缀因果性与 BPTT/detach；**GPU / DeepSpeed 尚未跑**。
- 代码基线：`feat/vlm8b-v1`（tip `0340f40`）；新分支不改基线，也不影响远端在跑的 run2。
- 首跑判定：5k 图子集 1 epoch/臂（~25 min/臂），见 §9 的臂与指标；训练内信号 `per_budget_ce.jsonl`，评测信号 `eval.py` 的 per-budget CE/acc。
- 立项前提：先确认 think 数据/更长答案下"现行线性投影"的 per-budget 曲线是否自然单调（见 §9 判定规则），再决定循环读出的定位（"让预算站起来"还是"同 token 数更强/更省"）。
