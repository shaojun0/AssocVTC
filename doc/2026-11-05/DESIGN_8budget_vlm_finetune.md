# DESIGN — Qwen3.5-4B 全参微调：8 预算并行训练（AssocVTC 448²/p32/960M 阶段2 编码器）

- 日期: 2026-11-05 ｜ 状态: **方案 v1，待用户审核**（未开跑；下文每个数值都给来源）
- 用户已定（2026-11-05）：**全参微调**、**DeepSpeed ZeRO-3**、**并行训练 8 个预算**、**用 mask 防泄露**、**8 个预算预测同一段文本**、**先冻结编码器训 LLM 1 epoch，再解冻训 2 epoch**
- 一句话：把阶段2 模型的 `z_s`（32 token × 1744 维）压成 **8 个 2560 维 visual token**（Qwen3.5-4B 的 LM hidden），同一张图一次训练步里**并行算 k=1…8 全部预算**（每个预算喂前 k 个 token、预测同一段文本），3 epoch 全参微调（E1 冻结编码器 / E2–E3 解冻）。

---

## 0. 用户指令 → 实现映射

| 用户要求 | 本方案落地 | 备注 |
|---|---|---|
| 全量微调 | LLM（32 层、文本 ~4B）+ 投影头全参；E2–E3 加编码器 | 不用 LoRA；ZeRO-3 分片 |
| DeepSpeed ZeRO-3 | `zero_optimization.stage=3` + bf16 + `stage3_gather_16bit_weights_on_model_save` | Step-0 与 FSDP2 对比选快的（§7） |
| 并行预算训练，直接 8 个预算 | 每个训练样本 = 同一张图的 **8 行**（k=1…8），一次 fwd/bwd | 行间完全独立（§4） |
| 用 mask 防泄露 | 行内**截断**（预算 k 只有 k 个 token，不存在"看不见的 token"）+ 批内 **right-padding 2D mask**；**不打包成单序列**（§4.2 有代码事实说明为什么） | 另有 3 个数值自检（§7.2） |
| 预测文本相同 | 同一张图 8 行的 assistant 目标逐 token 相同；loss = 8 行 CE 的（加权）平均 | 跨图文本不同 |
| 冻结编码器 1 ep → 解冻 2 ep | E1：encoder+specials `requires_grad=False`；E2–E3：全解冻，组 lr 不同 | 总 3 epoch |

---

## 1. 目标与判定标准

**科学问题**：MQT-LLaVA 用「随机抽一个预算」训练可变视觉 token 模型；本实验改成「每步 8 个预算并行」，问三件事：

1. **满预算不掉点**：并行训练下 k=8 的质量 vs 只训 k=8 的单预算模型（对齐税 ≤ 噪声）；
2. **低预算真被训出来**：k=1/2/4 的准确率显著高于「k=8 模型直接截断」（minival 里截断差 2.18×）；
3. **解冻编码器加不加分**：E1 vs E3 的 per-budget 曲线（编码器从"像素重建目标"转向"文本目标"是否更好，是否有崩塌）。

**判定顺序**（minival 的教训）：先确认收敛步数（单预算臂），再比并行 vs 单预算；否则会把欠训练当机制差。

---

## 2. 数据

### 2.1 长度实测（remote `/root/venv_qwen` tokenizer，n=150/文件）

| 文件 | messages 字段 | prompt 中位/p90 | answer 中位/p90 | 单行长度（8 visual + prompt + answer） |
|---|---|---|---|---|
| `train_nothink_all.jsonl` | `messages_short` | 108 / 108 | **4 / 4** | **120** |
| `train_nothink_all.jsonl` | `messages`（含 100 类闭集） | 776 / 776 | 4 / 4 | 788 |
| `train_think_all.jsonl` | `messages_short` | 520 / 634 | **58 / 71** | **586** |
| `train_think_all.jsonl` | `messages` | 1216 / 1330 | 58 / 71 | 1281 |

> 来源：`exp_qwen/in100_selfdistill/sft/`（nothink 全量 121,689；think 目前 15,232，API 收口后 ~38k）。
> `messages_short` 的 system/user 不含 100 类清单 ⇒ 省 ~670 tok/条。**训练用 `messages_short`**（评测也用 short，保证口径一致；`messages` 版留作对照）。

### 2.2 三个候选数据方案（成本差很大，需签字）

| 方案 | 训练集 | 每图 token（8 行） | 每 epoch token | 3 epoch 估算 | 评价 |
|---|---|---|---|---|---|
| **D1 nothink-only（推荐首跑）** | `train_nothink_all` 121,689 | 960 | 116.8M | ~12–18 h | 全覆盖、快；但答案是 4-token 类别名，**预算信号可能饱和在 k=2**（这本身也是结论） |
| D2 think-only | `train_think_all` 15,232（会到 ~38k） | 4,688 | 71.4M | ~7–11 h（按 15k） | 58-token 推理答案，**最能体现预算价值**；但覆盖面小、且 think 抽的是 30% 子集 |
| D3 mix（nothink 全量 + think 全量） | 121,689 + 15,232 | 960 / 4,688 | 188M（+61%） | ~20–28 h | 信息最全，最贵；think 权重可调 |

> **建议**：先跑 D1 打通全链路 + 出 k 曲线；D2 作为第二个「长文本预算」实验（数据齐了就 1 epoch 起步）。若时间只够一次，选 D3 但把 nothink 子采样到 60k。
> **注意**：nothink 有 ~11% 模型答错的闭集内样本（自蒸馏分布），`_strict` 版只保留 `gt_hit` ⇒ 首跑建议 **D1 用 `_all`（分布真实）/ 对照 `_strict`（标签干净）两臂小规模比 1 epoch**，再定主跑用哪个。

### 2.3 图像与文本管线

- 图像：原 JPEG → **AssocVTC encoder 的 448 变换**（`data_linattn.py` 的 train transform：`RandomResizedCrop(448, scale=(0.35,1.0)) + HFlip + ImageNet Normalize`；val：`Resize(512, BICUBIC) + CenterCrop(448) + Normalize`）→ encoder → `z_s (B,32,1744)`。
  - ⚠️ 训练期用增强（RRC+flip）与 teacher 生成时的「长边 resize 448」视图有差异；text 目标是类别名，增强不破坏标签。若担心分布漂移，可改确定性变换（Step-0 二选一，默认**用增强**，与 encoder 原训练口径一致）。
- 文本：`system` + `user`（`messages_short` 前两条）→ 拼成 ids；assistant 答案为 labels。
- **序列布局**（每行）：`[system ids] [k 个 visual token] [user ids] [assistant answer ids]`，`labels=-100` 除 answer 外。visual token 用我们自己的投影 embedding 直接替换（`inputs_embeds` 拼接），**不走 Qwen 原生 vision tower**（也不喂 `pixel_values`）。
- 位置：`position_ids=None` ⇒ Qwen3.5 文本模型走 `arange(0..L-1)`（代码事实：`modeling_qwen3_5.py` 文本 forward 的默认分支，**与 attention_mask 无关**），所以每行位置自然、不需要手工 3D mrope。

---

## 3. 架构：从 z_s 到 8 个 visual token

```
image ──AssocVTC encoder（阶段2 B/ckpt_best）──▶ z_s (B, 32, 1744)
                                                     │  proj: Linear(1744→640)  （逐 token）
                                                     ▼
                                               (B, 32, 640) = 20,480 值
                                                     │  reshape (B, 8, 4×640=2560)
                                                     ▼
                                          vis = (B, 8, 2560)   ← 与 Qwen LM hidden 同维
```
- **预算 k** = 用 `vis[:, :k]`（即 `z_s[:4k]`）⇒ 8 个预算天然嵌套（单调前缀），与 MQT 的 Matryoshka 性质一致。
- 预算集合 = **{1,2,3,4,5,6,7,8}**（用户"直接 8 个预算"）。
- 投影头：`LayerNorm(1744) → Linear(1744,640)`，拼成 2560 后 `LayerNorm(2560)`（初始化 N(0,0.02)；Step-0 会做一次 sanity：冻结 LLM 时投影头能否让 loss 下降）。
- 常量对照：`1744`（enc_dim，见 `B/args.json`）、`8×4×640 = 20,480 = 32×640`、`4×640=2560 = text_config.hidden_size`（config 实测）。

---

## 4. 并行 8 预算与「防泄露」的实现（本方案最关键的技术选择）

### 4.1 采用：8 行批（每图 8 行，行内截断 + 批内 right-padding mask）

- 一个训练样本 = 同一张图 × 8 行，行 k 只含 `vis[:k]` + 文本。
- 8 行的**文本 token 完全相同**，只有 visual 前缀长度不同 → 长度为 `k + T`（T=文本长度），batch 内 right-pad 到 `8+T`（只有 7 个位置的 pad 浪费）。
- `attention_mask`：真实 token=1，pad=0（标准 right padding）。**行与行之间天然独立**（不同 batch 行），不存在"变体看到变体"。
- **防泄露 = ① 截断（k 个 token 就是 k 个，没有隐藏 token）+ ② padding mask（pad 不参与 attention/线性状态）**。
- `labels`：answer 之外全 `-100`；loss 只在 answer 上算（`logits_to_keep` 只保留 answer 位置，见 §4.3）。
- **按 k 分组是免费的优化**：同一批图像的文本（`messages_short`）逐 token 相同、答案 4 tok ⇒ 8 行只有 8 种长度（`k+112`），按 k 分成 8 组后组内零 padding，组间各算一次 forward（或按最长长度 padding 一次算完，只浪费 7 个位置）。
- 训练 = 一次 fwd/bwd 处理 `B_img × 8` 行；推理 = 每张图跑 8 次单行（或 batch 内 8 行同图）出 8 条结果。

### 4.2 不采用：8 预算打包成一条序列 + 4D 块对角 mask（minival / AssocVTC 的做法）——**在 Qwen3.5 上不安全**

代码事实（transformers 5.22.1 `models/qwen3_5/modeling_qwen3_5.py`）：
- Qwen3.5-4B 文本模型是 **hybrid**：32 层中 **24 层 `linear_attention`（gated delta rule, fla/triton 核）+ 8 层 `full_attention`**（`config.text_config.layer_types`，每 4 层 1 个 full）。
- full-attention 层吃 `create_causal_mask(...)`（4D 块对角可行）；
- **linear 层吃 `create_recurrent_attention_mask(...)`：当传入 4D 自定义 mask 时它 `return None`（docstring 明写"already a custom 4D attention mask → no 2D padding signal"）⇒ linear 层会完全忽略 4D mask，跨段泄露**。
- 要在打包下正确隔离，需要 `cu_seqlens`（fla varlen）+ 每段位置重置，**属于改模型代码的高风险项**，且与 ZeRO-3/编译叠加后更难排查。
- 另：linear 层内 `apply_mask_to_padding_states()` 在 conv1d **之前**把 pad 位置 hidden 置零 ⇒ 2D mask 在 linear 层是"零化"语义（正确但 conv 窗口边界有 3 token 的近似），所以 **2D padding mask 路径是安全的**，4D 路径不是。

⇒ **结论：用 8 行批（2D mask）而不是打包（4D mask）**。它同样满足"并行 8 预算 + mask 防泄露"，且与推理形态（k 个 token 直接喂）完全一致。

### 4.3 loss 与显存细节

- 8 行 CE 平均：`L = (1/8) Σ_k CE_k`（主）；第二臂 `w_k ∝ 1/k` 归一化（低预算更难；minival §6.2 建议）。
- `logits_to_keep`：只保留 answer 位置（nothink 4 个）→ logits 显存从 `B×8×120×248,320` 降到 `B×8×4×248,320`（B=4 时 ~3.2GB fp16 → 100MB 级）；CE 在 fp32 分块算。
- 视觉 token 的 label=-100（不监督），梯度经文本 CE 反传到投影头（E2 起也到 encoder）。

---

## 5. 训练协议

| 项 | 值 | 说明 |
|---|---|---|
| 优化器 | AdamW (β=0.9/0.95, eps=1e-8), wd 0.1（LLM）/0.05（proj/enc） | 与 Qwen 预训练口径接近 |
| lr | LLM **1e-5**；proj **1e-4**；encoder（E2–E3）**5e-6** | cosine → 0，warmup 3% |
| 批次 | micro `B_img=2`/GPU（16 行）× accum 8 × 2 GPU ⇒ **global 32 img/step（256 行）** | 视 Step-0 显存再调 |
| 步数 | nothink：121,689/32 = **3,803 步/epoch** ⇒ 3 ep = 11,409 步 | D2/D3 见 §2.2 |
| 精度 | bf16 参数/激活，fp32 master（ZeRO-3 默认） | `torch.amp` 不必再用 autocast |
| 梯度检查点 | 开（`use_reentrant=False`），必开（Hybrid 层 + 全参 4B） | 墙钟 +20~30% |
| 冻结策略 | **E1**：encoder+specials 冻结（`requires_grad=False`、`eval()`、不进优化器组），只训 LLM+proj；**E2–E3**：全解冻 | 用户指定 |
| 评测 | 每 epoch 末：val 随机 1,000 张 × 8 预算（快）；最终：val_nothink_all 5,000 × 8 | §9 |
| ckpt | 只存 `ckpt_best` + `ckpt_last`（bf16 权重，~9–10GB/份）；开 `stage3_gather_16bit_weights_on_model_save` | `/root/autodl-tmp` 44G 可用，2 份 20G，够；中途不存 optimizer |
| 日志 | 每 20 步：loss/per-budget CE（8 个）/lr/gnorm/peak mem/tokens/s；JSONL | 便于画 budget 曲线 |

---

## 6. 资源与时间估算（2×A800 80G PCIe）

- 参数（按 `text_config` 实测维数估算）：embed（tied）0.636B + 8 个 full-attn 层 × ~97M + 24 个 linear(gated-delta) 层 × ~102M ≈ **3.86B 文本参数**；+ encoder 0.482B（E2–E3 可训，E1 冻结）⇒ **N≈4.34B**；删除 Qwen vision tower 后总显存按此计。
- 每个 token fwd+bwd ≈ 6·N·T；nothink：30,720 tok/step ⇒ `6×4.34e9×30,720 = 8.0e14` FLOPs/step；2×A800 实算 ~250 TFLOPS ⇒ **3.2 s/step**；加 ZeRO-3 PCIe all-gather（每步 ~2×8.7GB 参数流量，暴露 1–1.5 s）⇒ **4.5–5.5 s/step** ⇒ **4.8–5.8 h/epoch**（3 ep ≈ **14–17 h**）。
- 3 epoch（nothink）≈ **13–16 h**；D2（think 15k）≈ 952 步/epoch × ~10–12 s ⇒ 2.6–3.2 h/epoch ⇒ **8–10 h/3ep**；D3 ≈ 20–28 h。
- 显存粗估（ZeRO-3，每卡）：参数 bf16 4.34G + fp32 master 8.7G + Adam 17.4G + grad bf16 4.34G ≈ **34.8G**；激活（grad-ckpt, 16 行×120 tok）≈ 4–8G；logits 分块 ⇒ **~45–50G/卡**，80G 有余量。
  - 若显存/墙钟不理想：ZeRO-3 → ZeRO-2（不切参数、无 per-layer all-gather，PCIe 友好）或 FSDP2，Step-0 实测三选一（§7.3）。
- 磁盘：模型 9.3G（`/root/qwen_models`，不在 autodl-tmp）；数据 1.4G；ckpt 2 份 ~20G + 日志 <1G ⇒ `autodl-tmp` 44G 够。

---

## 7. Step 0：开跑前必须过的门禁（预计 1.5–2.5 h，含装包）

### 7.1 环境
- [ ] `deepspeed` 安装（当前 **未装**；`pip download deepspeed` 已验证可拉到 `0.19.9`）→ 在 `/root/venv_qwen` 内装（**不新建 venv**，避免与数据生成脚本的 transformers 版本漂移；装完重启训练进程即可，不影响正在跑的生成进程）。
- [ ] `ninja`/`nvcc` 可用性确认（deepspeed CPU 优化器不用，但 JIT 编译可能需要）；不可用则退回 FSDP2。
- [ ] 确认 `Qwen3_5ForConditionalGeneration.from_pretrained(..., dtype=bf16, attn_implementation="sdpa")` 可加载；`del model.model.visual` 释放 vision tower（~几百 M）。

### 7.2 掩码/泄漏数值自检（纯 fwd，几分钟）
1. **行独立性**：同一行单独 forward 的 loss vs 混在 8 行 batch 里 forward 的 loss，`max|Δ| < 1e-4`（bf16 容差）。
2. **截断隔离**：把 k=1 行的 visual token 换成完全不同的 z_s（模拟"别的图"），k=8 行 loss 不变（`max|Δ|=0`）；反之亦然。
3. **pad 隔离**：把 padding 位置的 embedding 换成随机值，所有行 loss 不变（`max|Δ|=0`）。
4. **预算单调性**（未训练模型即可看）：CE(k) 不应出现 nan/爆炸；训练 100 步后应看到 k 越大 CE 越低（否则 mask/拼接实现有 bug）。

### 7.3 性能标定（真正决定 Step-0 通过与否）
- [ ] 50 步 fwd+bwd（含 optimizer）实测：**s/step、peak mem、tokens/s**；外推 3 epoch 墙钟。
- [ ] **ZeRO-3 vs ZeRO-2 vs FSDP2** 三选一（同 micro-batch）：取"能在 80G 内 + 最快"。
- [ ] 断点恢复：save → 杀进程 → resume 后 loss/step 连续。

### 7.4 单预算上界臂（必须先跑，约 0.5–1.5 h）
- `A: 只训 k=8`（每图 1 行，8× 便宜）500 步 → 确认 loss 下降 & 收敛速度，作为并行臂的同计算/同步数对照基准；也用来标定"收敛需多少步"。

---

## 8. 对照与消融（在 GPU 预算内的最小集合）

| # | 臂 | 目的 | 成本 |
|---|---|---|---|
| A | 单预算 k=8（3 ep 或到平台） | 对齐税 / 上界 | 1/8 |
| B | **并行 8 预算（本方案主臂）** | 主结果 | 1× |
| C | 并行 + `w_k∝1/k` | 权重敏感性 | 同 B（跑 1 ep 即可） |
| D | k=8 模型**截断**到 k=1/2/4（零训练） | "不联合训练就没有低预算能力"的直接证据 | 免费（只评测） |
| E（可选） | 随机预算（MQT 原式，每步 1 个 k） | 并行 vs 随机在**同每预算更新**下的对照 | 1× |

---

## 9. 评测协议

- **主指标**：`acc(k)`（greedy decode 后与 GT 类名完全匹配）+ `CE(k)`（teacher forcing），k=1…8，val 5,000（nothink）/ 1,534（think）。
- **派生指标**：`Δacc(k) = acc(8) − acc(1)`（预算收益）；`截断 gap = acc_parallel(k) − acc_single_k8_trunc(k)`（联合训练价值）；`E1 vs E3` 的 per-k 曲线（解冻收益）。
- **生成**：`generate(inputs_embeds=..., attention_mask=..., max_new_tokens=16, do_sample=False)`；batch 内 8 行同图 ⇒ 8 条结果；注意 right-padding 下 generate 要逐行或改 left-pad（Step-0 验证）。
- **不混用**：训练/评测用同一 prompt 版本（`messages_short`）与同一图像变换（val 确定性）；报告里写明。

---

## 10. 风险登记表（按严重度）

| # | 风险 | 证据/影响 | 缓解 |
|---|---|---|---|
| R1 | ZeRO-3 + 2 卡 PCIe all-gather 慢 | 每步 2×9GB 参数流量；可能 +30–100% 墙钟 | Step-0 比 ZeRO-2/FSDP2，择优 |
| R2 | hybrid linear 层 + 4D mask 会漏 | `create_recurrent_attention_mask` 4D→None | **用 8 行批（§4.2）**；不做打包 |
| R3 | 冻结编码器在 ZeRO-3 里仍要 gather | 每 fwd 多 ~1GB 通信 | 可接受；E1 也可先 cache z_s（121k×32×1744×2B=13.6G 磁盘，不推荐） |
| R4 | 全参 4B + grad ckpt + 8 行 显存 | 估 45–50G/卡 | micro 2 img/GPU 起，OOM 降 1；开 logits_to_keep |
| R5 | 解冻编码器后表示崩塌 | 文本 CE 无重建约束 | E2 用 encoder lr 5e-6；每 100 步记 `‖z_s‖`/有效秩；若崩塌回滚 E1 ckpt（保留） |
| R6 | nothink 答案仅 4 token ⇒ 预算饱和 | acc(k) 可能 k=2 就到顶 | 本身就是结论；另跑 D2（think 长文本）验证预算价值 |
| R7 | 数据标签 ~11% 错（自蒸馏） | acc 上限 ~89% | 先跑 `_strict` 小臂对照；报告说明 |
| R8 | deepspeed 装不上/不兼容 torch 2.12 | 当前未装 | 备选 FSDP2（torch 原生）；安装失败不阻塞方案 |
| R9 | ckpt 体积（9–10G/份）×2 + 日志 | 44G 空间 | 只存 best/last，不存 optimizer；空间不足时删 last |
| R10 | 数据未齐（think 生成在跑，~21:45 收口） | D2/D3 需等 merge | 今晚先 D1；think 数据齐后补 D2 |
| R11 | 与 S3z448/Qwen 生成抢 GPU | GPU0 训练到 ~21:40；GPU1 生成到 ~21:45 | VLM 训练 **21:45 后**启动；Step-0 可先做（纯 CPU/小显存） |
| R12 | 时间超支（D1 13–16h，D3 20–28h） | 机器按小时计费 | 先确认 3 ep 还是 1–2 ep（见签字点） |

---

## 11. 时间线建议

| 时间 | 动作 |
|---|---|
| 今晚（审核通过后，约 16:00–21:40） | 写实现（§12 文件清单）+ 装 deepspeed + **Step-0 门禁（除性能标定外，均可用 GPU1 空闲时段/CPU 做）** |
| 21:45 GPU 全空 | Step-0 性能标定（50 步）+ 单预算上界臂 A（500 步） |
| 22:00–22:30 | 若门禁全过：开 B 主臂（D1 nothink 3 ep，~14h） |
| 次日上午 | E1 结束（~22:00+5h≈03:00）→ 自动切 E2；E2/E3 到 ~13:00–15:00 |
| 完成后 | 评测 acc(k) 曲线 + 与 A/D 对照 + 报告；think 数据齐后补 D2 |

> 若时间不允许 3 ep（D1 要 14h），建议：**E1 1 ep + E2-E3 共 1 ep（即总 2 ep）**，或把 nothink 子采样到 60k/epoch（步数减半）。这两处请在签字点 §12.7 给指示。

---

## 12. 需要你确认的签字点（每项有默认推荐）

1. **数据臂**：D1 nothink-only 首跑（默认✅）；或 D3 mix（+61% 成本）；或等 think 齐后 D2。
2. **nothink 版本**：`_all`（真实分布，默认✅）还是 `_strict`（标签干净）。
3. **prompt**：`messages_short`（默认✅，省 670 tok/条）vs `messages`（含 100 类清单）。
4. **图像变换**：训练期 RRC+flip（默认✅，与 encoder 原训练一致）vs 确定性 resize+crop。
5. **loss 权重**：8 预算等权平均（默认✅）；第二臂 `w∝1/k`（1 ep 消融）。
6. **超参**：LLM lr 1e-5 / proj 1e-4 / enc(E2–E3) 5e-6；global batch 32 img（256 行）；warmup 3% — 可改。
7. **总 epoch / 时间盒**：3 ep（默认✅，~14h）vs 2 ep（~9h）vs nothink 子采样 60k×3ep（~8h）。
8. **ZeRO-3 vs FSDP2**：默认先 ZeRO-3，Step-0 若慢 >1.3× 则用 FSDP2（不改变语义）。
9. **评测频率**：每 epoch 末 1,000 张快评 + 最终 5,000 全评（默认✅）。
10. **是否要随机预算对照臂 E**（MQT 原式，额外 1× 成本）：默认只做 B，E 仅在同每预算更新对照有争议时补。

---

## 13. 实现文件计划（审核通过后落地，全部新文件，不动现有训练脚本）

| 文件 | 内容 |
|---|---|
| `tools/vlm/model_wrap.py` | 载 encoder（`model_v2_linattn.StandardAttentionEncoder`，从 `B/ckpt_best.pt` 读 cfg+权重）+ 投影头 + `vis → inputs_embeds`；冻结/解冻开关 |
| `tools/vlm/data_8budget.py` | 读 `sft/*.jsonl` + 图像 → `(z_s, text ids, labels, masks)`；每图展开 8 行；right-pad collate |
| `tools/vlm/train_8budget.py` | 训练循环（2 阶段冻结/解冻、per-budget 日志、ZeRO-3 与 FSDP2 两种启动路径、resume） |
| `tools/vlm/ds_zero3.json` | ZeRO-3 配置（bf16、grad ckpt、`stage3_gather_16bit_weights_on_model_save: true`、`zero_allow_untested_optimizer`） |
| `tools/vlm/smoke_mask.py` | §7.2 四个自检（行独立/截断隔离/pad 隔离/预算单调） |
| `tools/vlm/eval_8budget.py` | per-budget greedy 评测 + CE + 截断 gap |
| `tools/vlm/report_8budget.py` | 生成 acc(k)/CE(k) 表与图 + 与 A/D 臂对照 |

> 依赖：remote `/root/venv_qwen`（transformers 5.22.1 / torch 2.12.1 / fla 0.6.0）+ 新装 deepspeed；encoder 代码来自远端 `/root/autodl-tmp/exp448/code/AssocVTC`（已同步）。
