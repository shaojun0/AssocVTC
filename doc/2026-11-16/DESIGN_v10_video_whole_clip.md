# DESIGN — v10 全片段循环读出视频模型（2026-11-16）

> 代码：`model_v10.py`（模型 + 自检）、`train_v10.py`（训练）、`data_v10.py`（120 帧片段）
> 状态：**代码完成，本地自检通过（CPU）；尚未在 A800 上训练**。
> 上一轮（v9 并行读出，24h 视频 run）的结论是「时间通路仍为零」：`val_long` 上
> `rec − cold = −0.028 dB`，`z_cls` 相邻帧余弦 0.9992（塌缩）。v10 是针对这个负结果
> 重新设计的读出架构。

---

## 0. 一句话

冻结一个重训后的图像分词器（E + R），把 120 帧压成 **30 组 × 64 个 rate token
（7168 值/帧）**，再用**整片段循环读出**解码：每一步读一组 z、把 **120 帧整段**
重出一次，30 步逐档细化；率比目标窄一半，跨帧混合才有地方兑现。

---

## 1. 决策记录（本设计来自 2026-11-16 的三轮澄清）

| # | 议题 | 定论 |
|---|---|---|
| 1 | 循环还是并行 | **循环架构**（少算力下串行代价可接受；并行 v9 的时间通路已实测为零） |
| 2 | 编码器 | **重新训练**（原 v9 ckpt 是 32×1024 全宽，没有降维；`exp448/S3z448` 的 448 维权重已被清盘删除） |
| 3 | 中间态 mask 粒度 | 按帧分块因果，**不是**一帧一个 token；一帧 = 4 个 1792 向量互看，跨帧只看历史帧 |
| 3b | 宽度 | **唯一宽度 1792 = 4 token × 448**；`mid_dim/dec_dim` 只接受 0 或 1792，禁止"投影到别的宽度"（那是纯填充） |
| 4 | 读出粒度 | **每步重出全片**（用户定：牺牲算力换「后文可回改前文」，B 帧式双向细化） |
| 5 | 查询 | `Q` 可学习、随机初始化；$Y_0=0$，$Y_t$ 从 $Q+\gamma_tY_{t-1}$ 递推 |
| 6 | 率 | 每帧前 16 个 special token（前缀截断，论文的嵌套前缀读法） |
| 7 | 目标 | **全部 33 个位置**（z_cls + 32 token）——见 §3，率必须小于目标 |

---

## 2. 形状与值数（默认配置）

宽度只有一种：`chunk_tokens=4` 个 token 拼成一个 **1792** 向量（4×448）。
宽度和位置数由同一个常量决定，所以任何位置都不需要投影、也不需要填充：

```
x            (B, 120, 3, 448, 448)                        15 s @ 8 fps
E（冻结）    每帧独立前向 → z_cls (B,120,1,448) + z_s (B,120,32,448)
rate         前缀截断前 16 个 special → reshape (B,120,4,1792)   纯 reshape，零参数
middle       480 位置（120 帧 × 4），帧级因果，6 层 → (B,120,4,1792)
z            reshape → (B,30,16,1792) = 860,160 值/片段 = 7,168 值/帧
读出         Q(1,1089,1792) 随机初始化；30 步；每步 pred (B,120,33,448)
loss         mean_t L1(pred_t, 冻结编码器给的全量目标)   逐档深监督
R（冻结）    末档 pred → 像素 → PSNR（只用于评测/监控）
```

| 用户口径 | 本实现 | 说明 |
|---|---|---|
| 每帧 `16 × 448` = 7,168 值 | `rate_tokens=16, chunk_tokens=4` | 4 × 1792 |
| 每帧 `4 × 1792` | `rows_rate = 16/4 = 4` | 同一个量，不是"换视图" |
| 每步 cross-attn KV `16 × 1792` | `chunks_per_step = 16` | 4 帧 × 4 行 |
| decoder query `121 × 4 × 1792` | `121 × 9 × 1792 = 1089` | **9 行**：8 行装 32 个 special，第 9 行装 z_cls |
| 目标 `120 × 4 × 1792` | `120 × 9 × 1792` | 同上 |

**为什么 query 是 9 行而不是 4 行**：4 行只装得下 16 个 token，而冻结 R 的
`render()` 里有 `z_s.shape[1] == num_specials`（=32）的硬断言，少一个就出不了
像素。所以码率侧 4 行、预测侧 8+1 行 —— 这个不对称正是「率 < 目标」本身。

**零填充的量化口径**：码率侧每帧 `4 × 1792 = 7,168` 个数承载 `16 × 448 = 7,168`
个值，填充 **1.00×**；旧版把 token 直接当位置、宽 1024 时是
`33 × 1024 = 33,792` 承载 `33 × 448 = 14,784`，填充 **2.29×**。

---

## 3. 为什么要「率 < 目标」（本设计最关键的一条）

如果 decoder 的目标就是"被截断的那 16 个 token"，那么最好情况是逐位复现，
经冻结 R 渲染 = 图像分词器第 8 档的质量。以我们 448² 链的实测为例（本地
`exp448_bench/artifacts_s3z448/stage3z448_stepwise.json`）：

| 读入 token | 8 | **16** | 24 | 32 |
|---|---|---|---|---|
| PSNR (dB) | 24.42 | **25.86** | 26.18 | 26.30 |

⇒ 目标=16 token 时上限 25.86 dB，逐帧冷启动基线也是 25.86，**实验必然平局**
（v9 的 −0.028 dB 就是这样）。改成率 16 / 目标 33 之后，收益来自"用邻帧预测
被丢掉的细节"（P/B 帧），上限变成分词器满档 26.30 dB（同口径），且真实视频
的时间冗余远大于 ImageNet，落差应该更大。

**本版中间态只读码率侧的 16 个 token**（用户 prompt 的口径：`120×4×1792`）。
代价是 32→16 的降率由「前缀截断」固定完成、不是学出来的；好处是 middle 的输出
本身就是码率，不需要任何压缩级。若要做「编码端先看全量、再做跨帧比特重分配」，
得让 middle 读 9 行输入、产出 4 行输出 —— 那是一个真正的 9→4 投影（是压缩，
不是冗余），可以作为独立一档后续加回。

---

## 4. 读出侧细节

- **carry 与 v2 逐字对齐**：$h_t=Q+Y_{t-1}+\text{step\_emb}_t$，即
  `OutputQueryDecoder` 里的 `query_base + Y` —— 等权相加、**不在 carry 处手写
  LN**（归一化交给 `TransformerDecoderLayer(norm_first=True)` 的入口 LN，额外
  一层 LN 是自己造轮子）。可选 `--step_gate 1` 退回零初始化 $\gamma$
  （$\gamma_0=0$ ⇒ 训练起点等价于"单步读出"再逐步放开递推），**默认关（$w\equiv1$）**。
- **step_emb + kv_pos**：$z_t$ 每步同形，不给步位置模型分不清读到第几组（v9 踩过）。
- **self-attn 全双向、无 tgt_mask**（2026-11-17 按 v2 口径删除旧的块因果 mask）：
  1089 个位置 = 121 帧 × 9 行，任意两行互通，与 `OutputQueryDecoder` 一致。
  位置数比旧版（3960）少 3.6 倍，但注意力真的**跨帧**了 —— 这正是
  `chunk_tokens=4` 换来的：序列短了，才敢让 self-attn 跨帧。
  **streaming 由 memory 保证、与 mask 无关**：第 t 档的 cross-attn 只看到
  `z[:, t]` 的 16 行，未来档的 z 根本不在场。
  删除依据（`frames=12 / 3 步 / 2 层`，同权重对比，两项扰动）：

  | 扰动 | 有块因果 mask | 无 mask |
  |---|---|---|
  | 扰 `z[step1]` → step 0 输出 max\|Δ\| | **0** | **0** |
  | 扰"未来帧"的 Q → 前序帧输出 max\|Δ\| | **0** | 0.40 ~ 0.49 |

  第一行说明 rate 前缀从来不是 mask 守的（那是 memory 的职责，旧版 docstring 把
  这条因果链写错了）；第二行说明 mask 实际只挡住"同一步内跨帧状态互看"，而
  cross-attn 的 K/V 对所有位置相同、**没有位置间通道**，所以它不会破坏 mask，
  但也不需要 mask 来保护前缀。既然它与"整片段细化"的目的相冲突、又只省下
  4.7 MB 常量（dense SDPA 的 FLOPs 一点不省），就删掉了。
- **三处"同一信息再算一遍"全部省掉**：等宽 ⇒ 不建 `kv_proj`；8 行 1792 直接
  reshape 成 32×448 ⇒ special 侧不建 head；整段一条 batch ⇒ 不 `repeat_interleave`
  出 120 份 memory。trainable 只剩 Q、step_emb、kv_pos、cls_head 和 decoder 本体
  （carry 处没有 LN、没有门控；`--step_gate 1` 时才多一个 $\gamma$）。
- **深监督**：30 档平权（`--step_weights mean|lin|last` 可切）。停在第 t 档 =
  只读前 t+1 组 z = 时间前缀预算。

---

## 5. 训练编排（照论文三阶段）

论文 §3.2 的教训：前几档只读到很少 token 时最优解接近均值，从零训练会卡在平凡解。
v10 对应做法：

1. **单步预热（全读）**：对齐 v2 的 A 相（`train_v2.py --warm_steps`：把
   `decoder.steps` 设成 `[K]`，**一步读完全部 `[z_cls; z_s]`、一步出整图、
   loss 覆盖全部位置**）。v10 里 = `n_steps=1` + readout 的 memory 换成
   **整段 30 组 z（960 行，行位置 `kv_pos` + 步位置 `step_emb`）**、
   loss 覆盖全部 120 帧（`--stage1_mode full`，默认）。旧理解"只读第一组 z"
   不等价，保留为 `--stage1_mode first` 仅作对照；
2. **展开 30 档**：从阶段 1 权重热启动，深监督全部 30 步；
3. （可选）**降率退火**：先按 `rate_tokens=32`（满档）训，再线性降到 16 —— 对应
   论文的降维退火，让模型把信息重分配到更窄的率上。

`train_v10.py` 使用 **HF Trainer + TrainingArguments**：训练循环、梯度累积、
cosine 调度、checkpoint、分布式全部交给框架，本文件只保留 collator /
`compute_metrics` / 一个处理无 labels eval 的 `Trainer` 子类。

```bash
# 单卡
python train_v10.py --tok_ckpt ... --out ... --max_steps 20000

# 2 卡 DeepSpeed ZeRO-2（也支持 accelerate launch）
torchrun --nproc_per_node=2 train_v10.py --tok_ckpt ... --out ... \
  --deepspeed ds_zero2_v10.json --max_steps 20000
```

`--resume` 传 Trainer checkpoint 目录（如 `out/checkpoint-2000`），不要传旧的单个
`.pt`。`rate_tokens` 决定 middle 的输入行数和每步的 chunk 数（`chunks_per_step`），
所以**换率等于换结构、必须重训**；`rate_tokens` 还要能被 `chunk_tokens` 整除。
更稳的做法：直接用 `rate_tokens=16` 一段训完，单步预热用 `n_steps=1` 前 1~2k 步。

### 5.1 checkpoint 是自描述的（AutoModel + trust_remote_code）

`V10Config` 是 `PretrainedConfig`、`VideoV10` 是 `PreTrainedModel`，两者都在模块
末尾 `register_for_auto_class`，`config.auto_map` 指到 `model_v10`。于是
`out/final/` 里除权重外还有 `config.json`（含 `auto_map`）和**模型代码本身**：

```
out/final/
  model.safetensors        tokenizer.* + middle.* + readout.*（含冻结分词器）
  config.json              V10Config.to_dict() + auto_map + architectures
  model_v10.py             本文件（custom_object_save 拷进来的）
  model_v2_linattn.py      相对导入的依赖，一起拷
```

```python
from transformers import AutoModel
m = AutoModel.from_pretrained("out/final", trust_remote_code=True)
```

三个为了这件事做的调整：

1. **`tok_cfg` 存了分词器的完整结构**（`VideoV10.__init__` 里 `asdict(tokenizer.cfg)`
   写回 config）。换台机器 `--tok_ckpt` 路径失效时，仍能按同样形状重建，
   权重从 `model.safetensors` 里来 —— 不依赖原始 `.pt`。
2. **`model_v10.py` 对 `model_v2_linattn` 做「相对导入 + sys.path 兜底」**：
   HF 的动态模块会把 `.py` 拷到 `~/.cache/huggingface/modules/...` 后独立执行，
   那里既不是包目录也不在 `sys.path` 上。相对导入同时让 `custom_object_save`
   顺着正则把 `model_v2_linattn.py` 一起拷走。
3. **`_init_weights` 是空操作**：子模块各自定义初始化（γ 零初始化、Q 随机）；
   走默认 `normal_(0, init_std)` 会把它们全部覆盖。
4. **`from_pretrained` 之后重新冻结分词器**：transformers 的加载路径用
   `_load_parameter_into_model` 把每个浮点张量包成
   `nn.Parameter(tensor, requires_grad=True)`（走 `setattr`，`load_state_dict`
   的 post-hook 拦不到），所以 `AutoModel.from_pretrained` 出来的模型分词器是
   **可训**的。`VideoV10.from_pretrained` 覆写后在末尾调一次
   `freeze_tokenizer()`，实测 trainable 从 126K 回到 44K（= 去掉 82K 分词器）。

注意磁盘：冻结分词器的权重也在 `model.safetensors` 里（默认 `1 份分词器 + 388M`
可训练），所以每个 checkpoint 都是这个体量。这是自包含换来的代价；真要省，
得让 `tok_ckpt` 成为硬依赖并跳过 `tokenizer.*` 的保存。

---

## 6. 参数量 / 算力（默认配置实测公式）

| 部分 | 参数量 | 说明 |
|---|---|---|
| middle | 231.4M | 6 层 × 38.5M（d=1792, ratio 4）；等宽，**无 in_proj**；出口 LN 走 `nn.TransformerEncoder(norm=…)` 自带槽，入口不建 LN（上游 z_s 已是分词器 post-LN token） |
| readout | 157.1M | Q 1089×1792=2.0M；4 层 × 38.5M；`cls_head` 1792→448 |
| **合计可训练** | **388.5M** | 冻结分词器不计；special 侧无 head、无 kv_proj |

FLOPs/样本（fwd）：middle ≈ 0.22 TFLOP（480 位置 × 6 层）＋ readout ≈ 10.1 TFLOP
（30 步 × 4 层 × 1089 位置 @ d=1792）⇒ **≈ 10.3 TFLOP**，含反向约 31 TFLOP。

与旧几何（d=1024 / 3960 位置）对比：

| 量 | 旧（1024×3960） | 新（1792×1089） | 比值 |
|---|---|---|---|
| 可训练参数 | 144.6M | 388.5M | 2.7× |
| readout FLOPs/clip | 12.0 T | 10.1 T | 0.84× |
| 码率侧填充 | 2.29× | 1.00× | — |

参数涨 2.7× 是因为 1792² = 3.06 × 1024²，而位置只少 3.6 倍。**这是唯一要权衡的
地方**：显存/步时吃紧时先砍 `mid_depth` 6→4（−77M），readout 深度留着不动。

（真瓶颈仍是冻结编码器：120 帧/样本前向 ≈ 94 GFLOP/帧，**建议加 latent 缓存**，见 §8。）

---

## 7. 编码器重训方案（用户决定重训；本节的数字是待执行的任务）

`train_linattn.py` 就是论文三阶段实现，直接复用：

```bash
# 阶段 A：单步读满（T=1，窗口 [0,K]），从零
python train_linattn.py --img_size 448 --patch 32 --num_specials 32 \
  --enc_dim 1744 --enc_depth 13 --enc_heads 16 --dec_dim 2048 --dec_depth 7 \
  --dec_recurrent 0 --epochs 45 --batch_size 128 --lr 2e-4 --zs_dim 0 ...
# 阶段 B：16 档循环读出（每步 2 token），从 A 热启动
python train_linattn.py ... --dec_recurrent 1 --dec_steps "" --dec_step_plan fixed \
  --dec_block 2 --epochs 6 --batch_size 64 --init_from <A>/ckpt_best.pt ...
# 阶段 S3：输出投影降到 448 维（论文式降维），从 B 热启动
python train_linattn.py ... --zs_dim 448 --epochs 3 --batch_size 64 \
  --init_from <B>/ckpt_best.pt ...
```

实测参照（2026-11-05，`exp448/{A,B,S3z448}`，双 A800）：A 13.5 h + B 8.1 h +
S3 7.1 h ≈ **25.6 h**；S3z448 末档 **26.1369 dB**（2560 val），逐档曲线见 §3 表。
便宜替代：**编码器原生 448 宽**（enc_dim=448，不需要 zs 投影，像 in100 那条
4.8 亿参数级模型），成本低一个量级，但 448² 分辨率下没有实测数字，需要自己量。

> ⚠️ 权重已被清盘：`run_v9_img/final_model.pt` 是唯一活着的 448²/p32 图像 ckpt，
> 且**没有** 448 维瓶颈；`exp448/{A,B,S3,S3z448}` 只剩 args/log/eval/json。

---

## 8. 未做 / 待定

1. **latent 缓存**：冻结编码器后应离线缓存（120 帧 × 33 × 448 bf16 = **3.5 MB/clip**；
   全量 32k 条 ≈ 113 GB 装不下，2,000 条 ≈ 7 GB 可行）。当前实现是每步在环计算。
2. **R 只用于评测**：训练损失在 latent 空间，`--psnr_eval 1` 时每个 val 批次才渲染一次。
3. **中间态注意力掩码**用 (L,L) bool mask，未走 flash-attn 的 causal 快路径；
   3960² 的 mask 每层一次乘法，先求正确性，后续可换分块实现。
4. **`rate_dim=0`（原生宽度）** 分支已留好，便于先用 1024 宽度跑通再换 448。
5. **打包视图**（4×1792）没有单独实现：`64×448` 与 `16×1792` 是同一张量的不同视图，
   中间态 token 粒度用 33/帧（未打包），对注意力语义等价（只差一个线性层）。
6. **流式/因果**：本设计是离线 tokenizer（组内 4 帧双向、readout 侧 self-attn
   全双向且每步重出全片，后文可回改前文）；读端唯一的因果是"z 前缀"，由
   `memory=z[:, t]` 保证，不依赖任何 self-attn 掩码。若要逐帧流式输出，需要另做
   "每步只出新 4 帧"的变体（计算量约 1/20）。

---

## 9. 诚实边界

- 真压缩只有"16/32 × 448 维 → 7168 值/帧"这一条（相对满档 latent 2.3×）；
  中间态与 reshape 是**等值重排**，不是压缩。若只看率，v10 不优于图像分词器
  第 8 档的 25.86 dB；收益只能来自"率窄于目标"时跨帧预测被丢的细节。
- 所有 dB 都是像素 PSNR，不等于下游任务保真；论文 §4 已写明不做跨文献数值对比。
- 架构尚未训练；在 A800 上跑通前，本文件的算力/耗时都是估算。
- 30 步 × 整片段重出是**刻意选择**的昂贵方案（用户定）；如果实测收益撑不起算力，
  第一个回退是"每步只出新 4 帧"的流式变体。

## 实现状态（2026-11-18 审计补记）

读本文档时**不要默认以下条目已实现**（子智能体审计发现「第二类不一致」）：

| 文档条目 | 本仓库主线状态 |
|---|---|
| 单步全读预热（v2 A 相口径） | **已实现（2026-11-19）**：`train_v10.py --stage1_epochs n --stage1_mode full --stage2_epochs m`，阶段 1 = 1 步 + memory 为整段 z（960 行）+ loss 覆盖全帧；单次 run 内按 `global_step` 自动切到 30 步渐进读出。阶段 1 评测带 `_s1` 后缀且不更新 `best_trainable.pt`（见 `run_v10_train_2stage.sh`） |
| 降率退火 | **未实现**（仍只有 `--stage1_epochs` 的单步预热） |
| checkpoint 自描述（`AutoModel` + `trust_remote_code`） | **未实现**（该实现只在远端 `75ac271` 那条历史线；本线 `VideoV10` 仍是 `nn.Module`） |
| 读出 carry 与 v2 对齐（无 LN / 无 γ / 全双向 self-attn） | **已实现（2026-11-18 17:30）**：默认 `carry_ln=0, step_gate=0, readout_mask=0`；旧口径用开关显式复现 |
| middle 两种模式 | **已实现**：`packed`（帧因果，默认）/ `crossq`（可学 query 双向交叉注意力，实验 A；rate 不再是渐进码流） |
| `loss_read` 的帧权重 | 默认口径下 frame0 权重 ≈ frame119 的 **120×**；`--loss_frame_norm 1` 可消掉 |
