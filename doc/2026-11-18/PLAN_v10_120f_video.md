# PLAN — v10 全片段循环读出模型在视频上的训练（2026-11-18）

> 承接 `doc/2026-11-16/DESIGN_v10_video_whole_clip.md`。本文件记录**实际执行的配置**
> 与在 MUSA（4×MTT S4000）上跑通的编排差异，结果见 `REPORT_v10_120f_video.md`。

## 0. 一句话

用 `doc/2026-11-18` 训好的 448²→16×448 图像分词器（冻结 E+R），在 32,165 条
120 帧（15 s @ 8 fps）448² 真实视频片段上训 v10：冻结编码器把每帧压到
16×448 = **7,168 值/帧**（860,160 值/片段），middle 做帧间混合，读出侧 30 步
整片段循环重出，深监督逐步 latent L1。

## 1. 实际配置（与设计文档的差异）

| 项 | 设计文档 | 实际 | 理由 |
|---|---|---|---|
| 每帧率 | 16×448 = 7168 值 | 同 | 用户口径 A |
| chunk_tokens（打包宽度） | 4（宽 1792，可训练 ~387M） | **2（宽 896，可训练 78.6M）** | MUSA 实测：chunk4 单样本 1.31 s、chunk2 1.10 s（含编码），参数减半后更快；chunk1 因 2057² 注意力反而更慢（1.42 s） |
| target_tokens | 32（率<目标） | **16（= num_specials）** | 分词器 K=16；`render` 要全量 K 才能出像素，`target_tokens` 必须 = K。因此末档上界 = 冻结分词器满读（视频 val 上 **30.53 dB**，见 §4） |
| 中间态/读出 dim | 1024 | 896 = width | 代码约束 `mid_dim == dec_dim == width` |
| middle | 6 层 d1024 ratio4 | 6 层 d896 ratio2 | 同上（38.6M） |
| 读出 | Q 3960×1024、4 层、30 步 | Q 1089×896（121×9 行）、4 层、30 步 | rows_pred = 16/2+1 = 9（8 行 special + 1 行 z_cls）；P = 64/2 = 32 行/步 |
| 读出 memory | 每步只看当步 `z_t`，整段信息靠 `γ_t·Y_{t-1}` 递推搬运 | **`prefix`：每步 cross-attn 看已读的 `z_0..z_t`** | 原版 30 步递推搬运整段信息在 32k 片段上训不起来（§1.1 有实测） |
| 读出输出 | 纯 TransformerDecoder 输出 | **+ 直抄残差**：已读帧的 rate 行原样加到输出行（`copy_skip`） | 否则会塌到"与输入无关的模板"（§1.1） |
| 损失 | 30 步 × 全部帧 | **`loss_read`：第 t 步只在已读帧上算 L1** | 未读帧不可知，会把梯度稀释成均值解 |
| 可训练参数 | ~140M | **78.63M**（middle 38.60 + readout 40.03） | 见上 |
| 冻结分词器 | ~240M+120M（旧） | **486.7M**（s2 ckpt，enc 248.4M + dec 238.3M） | 复用 2026-11-18 的 448²/16 token 分词器 |

张量形状（B=2 微批）：

```
x            (B,120,3,448,448)              15 s @ 8fps，原生 448×448（无 resize）
E（冻结）     (B*120,3,448,448) → z_cls (B,120,1,448), z_s (B,120,16,448)
打包          rate = z_s[:16] → (B,120,8,896)    rows_rate = 16/2 = 8
middle 6 层   (B,120,8,896) → 同形（帧内 block 因果）
z            (B,30,32,896) = 860,160 值/片段
读出 30 步    L = 121×9 = 1089 位置，每步 cross-attn 到 32 个 key
pred          (B,120,17,448) = z_cls + 16 special
loss          mean_t L1(pred_t, 全量目标)，30 档平权
```

## 1.1 为什么改读出侧（2026-11-18 中午的实测诊断）

设计原版（每步 memory = `z_t`，整段 120 帧的信息靠 30 步 `Y` 递推搬运）跑 1400 步后：

| 观测 | 数值 |
|---|---|
| 逐步 latent L1（4 条片段） | 0.356, 0.352, …, 0.342 —— **一条直线**，与"读了几组 z"无关 |
| eval PSNR | **12.75 dB**（末档；上界 30.53） |
| middle 输出跨片段 std | 初始 0.29 → **0.0155**（塌掉）；pred 跨片段差 **0.0002** |

即模型完全不吃 memory，学成了"与输入无关的逐位置模板"——30 步深监督里未读帧占绝大多数
位置，最优解就是均值，梯度把注意力压成 0。中间做过的排除项：

1. **单片段过拟合**：1.13 → **0.036**（300 步，lr 1e-3）⇒ 架构本身可学，不是写错；
2. 换成 `prefix` memory（只改 attention 范围）：单片段仍可学，但全量数据 loss 卡 0.49，
   且 **lr 1e-3 与 5e-4 的 loss 曲线逐位相同**（输出已是常量）⇒ 缺的不是学习率；
3. 参数探针：`|pred(clip A) − pred(clip B)| = 0.0003`，middle 输出跨片段 std 塌到 0.015。

最终三件套（都可单开关回退，默认值 = 设计原版）：

- `--mem_mode prefix`：第 t 步 cross-attn 的 key = 已读帧的 rate 行，key/query 都带
  「帧位置 + 行号」嵌入；
- `--copy_skip 1`：输出行 0..7 直接 `+= ` 该帧 rate 行（16 个 special 精确写出，
  模型只需学 z_cls 与未读帧），从结构上消灭均值解；
- `--loss_read 1`：第 t 步只惩罚已读帧，梯度不再被"不可知的未来帧"稀释。

改完 400 步就正常：loss 1.13 → 0.12，PSNR 9 → 19.7 dB；正式 run 4000 步到 **27.6–28.0 dB**
（`v10_120f/best_trainable.pt`，best `eval_psnr=27.98 @step3000`）。

> 并行读出这份代码在 main 上独立成 **`model_v9.py`**（配套 `train_v9.py` /
> `eval_v9_stepwise.py` / `run_v9_train.sh`），代码导读见
> `doc/2026-11-18/README_v9_parallel_readout.md`；`test_v9_parity.py` 保证
> `VideoV9` 与 `VideoV10(mem_mode="prefix")` 同权重输出逐位一致。

## 1.2 循环 vs 并行读出：单片段过拟合对照（2026-11-18 下午）

用户安排：**先单片段过拟合跑一些步数，确认循环架构能拟合，再切循环架构**，同条件
对照两种架构。单片段是唯一能把「架构/优化器根本学不动」与「泛化差」分开的实验。

矩阵（同一片段 clip 0，800 步，lr 8e-4 常数，bf16，`loss_read=1`，weight decay 0.05）：

| 架构 | 说明 | 末档 latent L1 | 末档 PSNR |
|---|---|---|---|
| `prefix`（v9 并行） | 基线：cross-attn 看已读前缀 | 0.0275 | **29.28 dB** |
| `step` 纯递推 | 设计原版：memory=当步 z_t | 0.0266 | 28.86 dB |
| `step` + 直抄 | 已读帧直接抄，递推只管未读帧 | 0.0253 | **35.08 dB** |
| `step` + 直抄 + 并行预热 | 从正式训练 `best_trainable.pt` 初始化后切循环 | 0.0181 | 30.96 dB |
| 上界 | 冻结分词器满读同片段 | — | 36.05 dB |

读法/结论：

1. **循环架构能拟合单片段**，而且不输并行（纯递推 28.86 vs 并行 29.28）——说明
   「架构写错」可以排除，之前全量塌成均值解是**优化/监督口径**问题。
2. 加上直抄残差后循环+直抄 **+5.8 dB**（35.08，逼近 36.05 上界）；作为对照，并行
   +直抄只到 29.28。逐档曲线（`artifacts/overfit_v10/overfit_v10_arch.png`）里
   循环+直抄的已读帧 L1 也最低。
3. **单片段过拟合不能外推泛化**（历史上循环架构单片段能拟合、全量却塌），必须
   全量训练裁决。已启动同条件全量对照（与 `v10_120f` 只差 `mem_mode`）：
   `MEM=step COPY=1 LOSSREAD=1 OUT=.../v10_120f_step`，4000 步，日志
   `logs/v10_train_step.log`。
4. 分解探针（`probe_overfit_breakdown.py`）：单片段上 loss/PSNR 主要由 **z_cls 行**
   与 special 行的误差结构决定；同样的 latent L1 在不同架构下像素 PSNR 可差 6 dB，
   说明误差的空间模式（哪些 token 错）比 L1 均值更决定 PSNR。

产物：`doc/2026-11-18/artifacts/overfit_v10/{prefix,step_nocopy,step_copy,step_warm}.jsonl`
+ `overfit_v10_arch.png`；脚本 `overfit_v10.py` / `run_v10_overfit.sh` / `plot_overfit.py`。
代码侧新增：`copy_skip` 不再只属于 prefix（step 也能直抄，CLI `--copy_skip`），
`train_v10.py --init_trainable` 支持「并行拟合 → 切循环」的课程学习。

## 1.3 架构复核：v10 读出 vs v2（2026-11-18 晚，用户提问后补查）

用户复核口径：**v10 的循环解码架构应与 model_v2 的 `OutputQueryDecoder` 一致**。
逐行对照后确认：**审核口径本身没漏**（远端 main 上 `946226d` 已经按这个口径改完），
漏的是**本地训练线没有同步这个改动**——今天训练 27.98 dB 用的是 2026-11-16 的旧版
读出（带 LN + γ 门控 + 块因果 mask），不是 v2 对齐版。

| 机制 | v2（`model_v2.py:596-608`） | 本地训练版（`model_v10.py` b345942） | v2 对齐版（`946226d`/`75ac271`） |
|---|---|---|---|
| 循环 | 同一 stack 在 \|T\| 步上反复施加，carry 走 BPTT | 同 | 同 |
| carry | `Y = query_base + Y` 裸加，**无 LN** | `h = LN(Q + γ_t·Y) + step_emb[t]`（有 `n_in`） | `h = Q + Y + step_emb[t]`（无 LN） |
| 门控 γ | 无 | 零初始化 γ，**默认开** | `step_gate` 默认关（w≡1） |
| self-attn mask | 单步内 N 行**全双向** | 1089 行**帧级因果 mask**（`_mask` + `tgt_mask`） | 全双向、**无 tgt_mask** |
| 每步输出 | N 行 = 整图 patch | 121×9 = 整段 120 帧 | 同左 |
| 每步 memory | 当步 `[z_cls;z_s]` 块切片 | 当步 `z_t` / 已读前缀（v9） | 当步 `z_t` |
| 目标 | 像素（PixelHead） | 冻结分词器 latent（17 行） | 同左 |
| middle | 无（DINO 直连 decoder） | 新增 `TemporalMiddle` | 同左（6 层） |
| 训练侧 shortcut | 无 | `copy_skip` 直抄 + `loss_read` 只监督已读帧 | 无 |
| 宽度/深度 | d=dim，depth=2 | d=896（2×448），depth=4 | d=1792（4×448），depth=4 |

时间线（解释为什么训练版没对齐）：

- `ef6fdd2`（11-16）：v10 初版 = LN + γ + 块因果 mask（**我们本地线的分叉点**）；
- `50e8ff6`/`946226d`（11-17 10:23）：按用户复核口径**去 mask / 门控 / LN**，
  并把 carry 逐字对齐 v2（`Q + Y + step_emb`），同时换原生 `nn.Transformer*`、
  宽度 1792；`50e8ff6` 就是配套的 `exp_carry_ln.py`（量化 carry 处 LN/门控的作用）；
- 本地线（`44ab442`…`63846dc`）从 `ef6fdd2` 分叉，直接在这份**旧读出**上做了
  MUSA 移植、120 帧通路、prefix/copy_skip/loss_read，**从未合入 946226d**。

结论（对之前归因的修正）：

1. 「全量塌成均值解」发生在**未对齐 v2 的旧读出**上（LN + γ + 因果 mask）；
   它能不能用「监督口径」解释要分开看——未读帧全监督 + 均值解是叠加因素，
   但**对齐版（去 LN/γ/mask）在 120 帧全量上还没训练过**，不能断言它的行为。
2. 今天 27.98 dB（以及正在跑的 step+copy 全量对照）都是**旧读出 + shortcut** 的
   结果，不是「与 v2 一致的架构」的结果；论文/报告里必须写清这一点。
3. 下一步应做：**v2 对齐版 vs 旧版**的同条件消融（单片段先筛，再全量），
   把「carry/门控/mask 三处偏差」和「copy_skip/loss_read 两处 shortcut」解耦。

## 2. 数据

- 根：`/root/autodl-tmp/video_recon/data3`（42 GB，原生 120 帧 448² 8 fps）
  - `train/index.jsonl` **32,165** 条，`val/index.jsonl` **1,142** 条，`val_long` 60 条长片
  - 来源：actnetqa 23,031 / kinetics400 4,451 / ucf101 3,675 / mixkit 935 / actnet 73
- 训练输入 = 离线缓存的 `(z_cls, z_s)`（见 §3）；评测仍用原始 mp4（要算像素 PSNR）。

## 3. 三段流程

1. **缓存**（`build_latent_cache.py`，~85 min，59 GB）
   `torchrun` 不用：每卡一个**独立单进程**（`MUSA_VISIBLE_DEVICES=i`，交错 20 s 启动），
   各写 `rank{i}.npy`（float16，(n_i,120,17,448)）。全局样本 i → shard `i%4` 第 `i//4` 行。
   实测 1.6–1.8 clip/s/卡，缓存值与在线编码 **逐位相同**（maxdiff = 0）。
2. **训练**（`run_v10_train.sh`，4 卡 DDP/MCCL，~2.7 h）
   `--mem_mode prefix --loss_read 1 --copy_skip 1`；全局批 16 = micro 2 × accum 2 × 4 卡；
   lr **8e-4** cosine、warmup 2%、wd 0.05、clip 1.0、bf16、4000 步（≈64k 样本 ≈ 2 epoch）。
   评测回调每 200 步在 2 条 val 片段上报 latent L1 / PSNR / 分词器满读 PSNR，
   并按 PSNR 存 `best_trainable.pt`。
3. **逐档评测**（`eval_v10_stepwise.py`，60 条 val 片段，~10 min）
   30 档全部渲染，出「读 t 组 z → PSNR」曲线 + 逐片段 JSON + 图。

## 4. MUSA 实测数字（本机，不是估算）

| 项 | 数字 |
|---|---|
| 冻结编码器（120 帧/片段，bf16） | **0.369 s/片段**，峰值 4.5 GiB |
| middle fwd+bwd（B=1） | 0.092 s |
| 缓存训练步（无编码器） | B=1 **0.737 s/样本**；B=2 **0.605 s/样本**（峰值 25.8 GiB） |
| 训练步（在线编码，B=1） | 1.10–1.31 s/样本（取决于 chunk） |
| 4 卡 DDP 单步（全局 16） | 2.83 s → 4000 步 ≈ 2.7 h |
| 冻结分词器满读基线（视频 val 2 条） | **30.53 dB**（对比 IN-100 图像 27.32 dB —— 视频帧更简单） |

## 5. MUSA 上踩的坑（都已固化在代码里）

1. **SDPA 必须 math**：`train_v10.py:set_sdpa_backend()` 在 MUSA 上默认关 flash。
   分词器 enc head_dim 1248/16=78、dec 1440/16=90，命中 muDNN flash 反向
   `head_dim%8≠0` 的数值错误（doc/2026-11-17/REPORT_musa_sdpa_bwd.md，上游
   torch_musa#162）。
2. **DDP c10d logger timer 崩溃**：`_patch_ddp_musa()` 类级替换
   `DDP._pre_forward`，只删掉 `logger.set_runtime_stats_and_log()`。
3. **4 进程同时初始化 MCCL + 首上下文会把驱动卡死**：缓存阶段实测 hang 在
   `flush_workqueue`、GPU 0%、sshd 一度拒绝连接；改成 4 个独立单进程后正常。
4. **transformers 5.13 的 eval metric**：forward 返回 dict 时 `compute_metrics`
   拿不到 `l1/psnr`（只剩 runtime 指标），`metric_for_best_model` 直接 KeyError。
   改为 `V10EvalCallback` 自己评测（直接调 `VideoV10.evaluate()`）。
5. **RNG 存档**：`torch_musa` 没有 `torch.cuda.random`，`V10Trainer` 覆写
   `_save_rng_state/_load_rng_state` 走 `torch.musa.*`。

## 6. 复现命令

```bash
# 0) 同步代码
rsync -az model_v10.py train_v10.py build_latent_cache.py eval_v10_stepwise.py \
  run_v10_cache.sh run_v10_train.sh run_v10_smoke.sh sha1:/root/AssocVTC_v10/

# 1) 缓存（85 min，59 GB）
ssh sha1 'cd /root/AssocVTC_v10 && nohup bash run_v10_cache.sh \
  > /root/autodl-tmp/tok448_v2/logs/v10_cache_driver.log 2>&1 &'

# 2) 训练（4 h）
ssh sha1 'cd /root/AssocVTC_v10 && nohup bash run_v10_train.sh \
  > /root/autodl-tmp/tok448_v2/logs/v10_train.log 2>&1 &'

# 3) 逐档评测
ssh sha1 'cd /root/AssocVTC_v10 && SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0 \
  python eval_v10_stepwise.py --trainable /root/autodl-tmp/tok448_v2/v10_120f/best_trainable.pt \
  --n_clips 60 --out /root/autodl-tmp/tok448_v2/v10_120f/eval_stepwise.json'
```

## 7. 诚实边界

- 率 = 目标 = 16（分词器 K=16 的硬约束）：末档最好情况就是**复现冻结分词器的
  满读结果**（视频 val 30.53 dB），v10 的收益只能来自「少读几组 z 时用帧间预测
  补细节」（逐档曲线），以及时间一致性；不存在「率窄于目标 → 用邻帧回收被丢
  细节」的额外收益（设计文档 §3 的论证在 K=16 下不成立）。
- loss 是 latent L1，不是像素指标；像素 PSNR 只经冻结 R 渲染后计算。
- 单 seed；val 1142 条里取 2 条做训练中监控、60 条做逐档评测。

### 1.3.1 处置（2026-11-18 17:30）：main 的 model_v10 已对齐 v2，旧 run 停掉重训

- **停掉**旧读出的全量 run（`v10_120f_step`，旧 run 到 832/4000，日志
  `logs/v10_train_step.log` 保留）；旧 prefix 产物（27.98 dB）仍保留在 `v10_120f`。
- `model_v10.py` 读出按 946226d 口径改：**carry 无 LN**（`h = Q + Y + step_emb`）、
  **默认无 γ 门控**（w≡1）、**读出 self-attn 全双向无 tgt_mask**；三者各留一个
  开关（`--carry_ln / --step_gate / --readout_mask`，默认 0），只为复现 2026-11-16
  旧产物/做消融；`run_v10_train.sh`（27.98 dB 配方）已显式打开旧开关。
- `eval_v10_stepwise.py`：按权重自动检测旧 ckpt 的 `carry_ln`/`readout_mask`，
  历史 ckpt 仍可严格加载评测。
- **新 run**：`MEM=step COPY=1 LOSSREAD=1 STEPGATE=0 CARRYLN=0 MASK=0`，
  与旧 run 只差架构（监督配方不变，控制变量），输出
  `/root/autodl-tmp/tok448_v2/v10_120f_v2align`，日志 `logs/v10_train_v2align.log`，
  2.42 s/step，ETA ≈ 2 h 42 min。
- 待办：v2-pure 对照（`COPY=0 LOSSREAD=0`，全程平权监督、无 shortcut）作为第二组。

## 1.4 实验 A：middle 改成「双向交叉查询重采样」（2026-11-18 18:05 起跑）

用户指出：v2 的 decoder 第 1 步就能读到编码器给的全局摘要（`z_cls` 在 memory 里），
而 v10 的 `TemporalMiddle` 是**帧级因果** encoder——第 t 步的 z 只含前 4(t+1) 帧。
要求改成：**E 输出 (120×rows×d) 作为交叉注意力的 K/V，同形状随机初始化 query
去查询**（即 middle 做 v2 式摘要）。

实现：`--mid_mode crossq` → `CrossQueryMiddle`（`model_v10.py`）：

```
Q (1, F·T_f, d) 随机初始化（行↔(帧,行)）;  KV = E 打包行 + 可学 kv_pos
nn.TransformerDecoder（query 间 self-attn + 到 E 的 cross-attn，6 层，**无 mask**）
→ 输出 (B,F,T_f,d)，与 packed 版同形状（readout 不用改）
```

- 参数：d=896 时 ~57.9M（packed 版 38.6M）；算力可忽略（middle 每样本只算一次）。
- **语义代价（必须写进报告）**：rate 不再是「第 t 组只含前 4(t+1) 帧」的渐进码流，
  每行都是全片摘要；逐档曲线不再是"读前 t 组帧 vs 质量"，改成 v2 口径
  「读多少 token 的质量」；`copy_skip/loss_read` 里"已读帧"的字面含义变弱。
- 运行：`v10_120f_v2midA`，与 `v2align` 同配方（step+copy+loss_read，读出已对齐
  v2），**只换 middle**；日志 `logs/v10_train_v2midA.log`，2.49 s/step。

## 1.5 子智能体审计：还有哪些「设计意图 vs 实现」的类似问题

2026-11-18 18:00 由只读审计子智能体给出（无 P0，按严重度）：

1. **P1 评测口径**：`eval_v10_stepwise.py` 用整片 PSNR/L1，而 `loss_read=1` 时中间
   档未读帧无监督 ⇒ "read t groups" 曲线把"猜未来"当成绩。应加 read-frame-only 曲线。
2. **P1 文档/实现第二类不一致**：DESIGN 里 `n_steps=1` 单步预热/降率退火、checkpoint
   AutoModel 自描述都**未实现**（`train_v10` 只有 LR warmup；`VideoV10` 仍是 `nn.Module`）。
3. **P1 z_cls 不进率**：readout memory 永远没有 z_cls；v2 第 1 步就能读。27.98→30.53
   的缺口主要是 z_cls 回归误差（probe：zcls_l1=0.0725）⇒ DESIGN §3「率<目标」在 K=16 失效。
4. **P1 loss_read 帧权重极不均**：per-step 对已读帧取均值 ⇒ 解析算 frame0 权重约是
   frame119 的 **120×**（早期帧过训、末尾帧几乎无监督）。可加 per-frame 归一化消融。
5. **P1 v9 未对齐**：`model_v9.py` 仍是 carry LN + γ 默认开 + 因果 mask，且无开关
   ⇒ v9-vs-v10 的 27.32/27.98 差异混着架构与 3 处偏差，无法归因。
6. **P1 crossq 语义**：切双向 middle 后 rate 非前缀、逐档/渐进口径与 copy_skip 前提都变；
   eval/PLAN 需同步（本条已在 §1.4 记录）。
7. **P2**：copy_skip 使 specials 直抄、递推主要只影响 z_cls ⇒ "读出重建"措辞夸大；
   `evaluate().latent_l1` 在 loss_read=1 时是 masked L1，与 §1.1 的 0.49（全帧）不可比；
   默认配置（target_tokens=32、注释 1792、loss_read=0+copy_skip=1）不可用；
   位置嵌入的 4+ 处差异（middle 无位置、readout 再补 kv_pos、norm_out 多一层 LN）；
   best 用固定 2 条 val 选（选择噪声 ~0.1–0.3 dB）；cache docstring 仍教 torchrun 4 卡。

## 1.6 实验 A 结果：crossq（双向 middle）塌了（2026-11-18 晚）

`v10_120f_v2midA`（crossq middle + step + copy_skip + loss_read，读出对齐 v2）
跑到 **1971/4000** 时 eval 一直停在 **10.6–13.5 dB**（latent L1 0.27–0.42），
远低于同配方 packed middle 在 step200 的 20.5 dB ⇒ **双向交叉查询 middle 在
全量数据上塌了**（与旧架构原版塌的样子一致，只是更早/更低）。已停止该 run
（日志 `logs/v10_train_v2midA.log` 保留，partial 产物已清理）。

初步判断（待单片段过拟合确认）：
- crossq 把**全部 960 行**都变成"全片摘要"，readout 的 step memory（当步 32 行）
  虽然含全片信息，但**每行不再对应"某帧某行"的确定内容**——copy_skip 想直抄
  的"该帧 rate 行"与"全片摘要行"语义错位，模型最容易学的是"所有行输出同一
  个全片均值"，于是退化成模板；
- 这提示：**"摘要"不能靠把所有行都变成摘要**，而应像 v2 那样——**一行全局
  摘要（z_cls）+ 逐帧局部 token 并存**（即 PLAN §1.4 里的方案 C）。

今晚保留的对照：packed middle + 已对齐 v2 读出的全量 run（`v10_120f_v2align`），
作为"对齐架构 vs 旧架构"的同配方基线（此前只跑到 759 步就被实验 A 替换）。

## 1.7 实验 B：两阶段训练（单步预热 → 30 步循环，2026-11-19 上午）

**用户口径**：先不换架构，试「第一阶段单步解码 n 个 epoch，然后 30 步循环解码
m 个 epoch」。动机是 DESIGN §5 的论文式配方：多步循环早期步只读到极少信息，
条件最优解接近均值，从零直接训 30 步会卡在平凡解；先用最容易的单步把通路
训起来，再展开 30 档深监督。

**"单步"必须对齐原 v2 的 A 相（2026-11-19 09:30 用户纠正）**：v2 的
`train_v2.py --warm_steps N` 把 `decoder.steps` 设成 `[K]`（`single_steps =
[model.num_specials]`），`step_windows` 首步 `lo=0, hi=K` ⇒ **一步读完全部
`[z_cls; z_s]`、一步出整图**；loss = 各步直接预测 L1 平权，A 相只有 1 步 ⇒
**全部位置都在 loss 里**。因此 v10 阶段 1 = `n_steps=1` + memory 换成**整段
30 组 z（960 行）** + loss 覆盖全部 120 帧（`--stage1_mode full`，默认）。
最初的实现（`n_steps=1` 只读 z_0、只监督前 4 帧）**不等价**，保留为
`--stage1_mode first` 仅作对照，不是本轮口径。

**实现**（`train_v10.py` / `run_v10_train_2stage.sh`）：
- `--stage1_epochs n --stage2_epochs m`：总步数 = (n+m)×steps_per_epoch
  （steps_per_epoch 由 `len(train)/(batch×accum×world)` 反推，本机 = 2011），
  LR 仍是覆盖全程的单条 cosine（warmup_frac 0.02）；
- 阶段 1（`--stage1_mode full`，默认）：`n_steps=1` + readout 的 cross-attn
  memory = **整段 z 的全部 30×32=960 行**（memory 位置 = 行位置 `kv_pos`（32）
  叠加 z 步位置 `step_emb`（30），复用既有参数、不新增/不破坏旧 ckpt），
  一步出全部 1089 位置 × 9 行；`copy_skip` 下 120 帧都有直抄残差；
  loss = 全部帧的 latent L1（full_read 下自动关掉 `loss_read` 的已读帧 mask，
  等价 v2 A 相的"全部位置监督"）；
- 阶段 2：切回全 30 步，口径与 `v10_120f_v2align` 完全一致（packed middle、
  step 读出、carry 无 LN/无 γ、全双向 self-attn、copy_skip + loss_read）；
- 切换在 `V10Trainer.compute_loss` 里按 `state.global_step` 完成，不用重启、
  不用手工热启动；阶段 1 的 eval（`eval_*_s1`）**不写** `best_trainable.pt`，
  因为全 30 步 PSNR 在阶段 1 无意义。

**本轮配置**：`v10_120f_2stage`，n=m=2（4022 + 4022 步 = 8044），lr 8e-4，
batch 2×2×4 卡（16 clips/step）。阶段 1（旧 first 口径）实测 0.24 s/it（vs 30 步
2.4 s/it）⇒ 全读口径同量级（多出的只是 cross-attn key 从 32 行变 960 行）。
阶段 1 评测（2 条 val，带 `_s1`）同时报 latent L1 与像素 PSNR —— 单步全读能
出全片，PSNR 在此口径下有意义；旧 first 口径的 PSNR 无意义。

**要回答的问题**：同配方下 2 epoch 单步预热 + 2 epoch 循环，末档 PSNR 能否
超过 `v10_120f_v2align`（同 4000 步主训练、无预热，跑到 2210 步时 26.05 dB /
上界 30.53）与旧架构 `v10_120f`（27.98 dB）。逐档曲线（`psnr_read_mean`、
`zcls_l1_mean`、`spec_l1_mean`）由收尾评测给出。

**结果（2026-11-19 13:17 收尾；8044/8044 步跑满，3h04m 墙钟，2.56 s/it）**：
- 训练内（2 条固定 val，上界 30.53 dB）：阶段 1（单步全读）19.58@200 →
  21.25@1000 → 25.29@1800 → 26.70@2600 → **28.06@3400**（阶段 1 共 ~20 min）；
  阶段 2 起点 27.94、最好 **28.45 @ step 5400**、末次 28.41 ⇒ **30 步渐进循环只比
  "单步全读"多 +0.39 dB**。
- 逐档评测（60 val，上界 30.055±3.50）：末档 **24.76 ± 3.28 dB**
  （min 18.87 / max 35.89），距上界平均 **5.30** dB；整片 PSNR 随读档单调
  9.35 → 24.76；`z_cls` L1 0.423 → **0.374**（几乎不动），`special` L1
  0.565 → **0.0139**。
- ⇒ **special 已基本无损（~1.4% 量级）**，5.3 dB 缺口几乎全在 `z_cls`；
  `copy_skip` 下 special 走的是 middle 的直抄、读出循环对 special 几乎无贡献
  ⇒ 这轮实际上没检验"循环读出 vs 并行读出"的能力差异，只检验了"z_cls/摘要"通路。
- 对照（**60 val 同一批片段、同一评测口径**，2026-11-19 14:37 出齐，
  `doc/2026-11-19/artifacts/baselines60/comparison.md`）：

  | run | 末档 PSNR | 距上界 | z_cls L1 | special L1 |
  |---|---|---|---|---|
  | **v10_120f_2stage（本轮：2 预热 ep + 2 主训 ep）** | **24.76 ± 3.28** | **5.30** | 0.3743 | 0.0139 |
  | v10_120f 旧配方（4000 步，旧读出 LN+γ+mask、prefix） | 24.12 ± 3.31 | 5.94 | 0.3834 | 0.0175 |
  | v10_120f_v2align（aligned step、无预热、仅 2210 步） | 23.94 ± 2.94 | 6.12 | 0.3864 | 0.0222 |

  ⇒ 两阶段在 60-clip 上比旧配方 **+0.64 dB**、比同架构无预热（且只训到 2210 步）
  **+0.82 dB**；逐档曲线（t=1..30）在所有档位都更高。2-clip 口径同样 +0.47。
  **但三条 run 的 z_cls 误差几乎一样（0.374/0.383/0.386）** ⇒ z_cls 缺口是
  **结构性**的，与训练量、与读出 memory 模式（step vs prefix）都无关——旧配方的
  prefix 读出也没能把 z_cls 压下去。

- **逐档解码成真实视频（2026-11-19 14:10，`visualize_stepwise_video.py`，val 第
  0/1/2 条，档位 1/5/10/15/20/25/30）**：把第 t 档的整片 120 帧渲染成 mp4 后可见
  ——**前 4t 帧就是真实视频画面**（`copy_skip` 直抄 + 正常 `render`），**之后的
  未读帧是噪声/均值**，分界正好落在 4t/120 处并随档位右移；末档 120 帧全部真实。
  帧间运动量（相邻帧平均绝对差）与 GT 的比值：末档 **0.94–1.01**（运动量与真视频
  一致），早期档 0.15–0.9（未读区域是静态噪声把均值拖低）。三条 clip 的
  PSNR 7.1→21.7 / 6.6→35.2 / 10.6→28.6 dB，末档分别距各自上界
  **3.5 / 0.7 / 8.6** dB（clip1 几乎顶到天花板）。
  ⇒ 「渐进解码出真实视频」**成立、但只对已读前缀成立**；`loss_read=1` 下模型
  从没被要求预测未来帧，所以这是"视频逐档长出来"，不是"每档都在预测未来"。
  产物：`doc/2026-11-19/artifacts/v10_120f_2stage/viz{,_224}/`（contact sheet +
  `viz.json`；`viz_224/` 是压缩版视频进 git，全分辨率 45 MB 只在本地 `viz_full/`，
  已 gitignore；脚本 `visualize_stepwise_video.py` 可随时重跑）。

**结论**：两阶段成立且阶段 1 很便宜（4022 步 ≈ 20 min 就到 28.06），但收益被
`z_cls` 这条结构性缺口卡住：`z_cls` 不在 rate 里、step 模式末步 memory 只有
`z_29` ⇒ 早期帧的 `z_cls` 无法从自身内容预测（读更多档也不降）。

**下一步候选（按优先级）**：
1. **方案 D**（§1.4）：rate = `z_cls` + 15 specials，让读出 memory 里出现 z_cls；
2. 读出 carry 显式搬运早期帧 `z_cls`（step 模式末步只有 `z_t`，需靠递推 Y 记住）；
3. 先做 v2-pure（`COPY=0 LOSSREAD=0`）对照，确认循环读出本身有没有学到时间通路；
4. 方案 C（causal middle + 1 个全局摘要 token）；`loss_frame_norm` 消融。
