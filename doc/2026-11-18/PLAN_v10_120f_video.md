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
