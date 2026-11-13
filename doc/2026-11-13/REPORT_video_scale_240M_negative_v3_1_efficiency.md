# 视频中间态规模化实验报告（2026-11-13）：240M 负结果 + v3_1 效率实测

> 机器：① 摩尔线程 2×MTT S4000 48G（`musa`，bf16，手写 fp32 flat all-reduce DP）；② Tesla V100-PCIE-32G（fp16，torch 2.4.1+cu121）。
> 数据：真实视频 448×448，8 fps，片段 120 帧（15 s）。`data3/train` 32,165 条（MUSA 全量 34 GB；V100 上搬了 14,000 条子集），`val` 1,142 条，`val_long` 60 条。
> 模型：v7 = commit `d123ee7`（model_v3）在 448/patch32 上的移植（编码器 register 式 ViT，中间态帧间循环交叉注意力，解码器 N=196 行查询 + 采样步 + 卷积像素头 + 深监督）。
> 附件：`code/`（v3_1 移植与测速脚本，可直接复现）、`evidence/`（评测/测速/自检原始输出）。

---

## 0. 一句话结论

1. **240M×3（703M 总参）在 ~2.9 卡·天下没有学到东西**：训练内 PSNR 14.2–14.5 dB，`val` 冷启动 11.0 dB，`val_long` 上 cold/rec/warm **完全相同（≈14.04 dB）**——历史帧对重建的贡献 ≤0.01 dB，模型只学到"比灰色均值好 3.5 dB 的糊图"，远低于 copy-prev 基线 23.6 dB。
2. **算力账**：6 条 240M run 合计只喂了约 50 万监督帧 ≈ 17 小时视频 ≈ 数据集（386 万帧）的 12%。**一遍 epoch 都没看到**，所以这是"欠拟合"而不是"架构不行"。
3. **v3_1（`model_v3_1.py`，commit `7c4e42c`）确实快，但收益取决于哪一段是瓶颈**：
   - MUSA（中间态是瓶颈）：**3.9×**（同参数量、同 F/window）；
   - MUSA 真实数据 + 双卡 DP：**2.3×**（23.8 → 9.9–11.2 s/步）；
   - V100（decode 是瓶颈）：**v3_1 的收益主要来自它顺带做的"帧维折 batch decode"**，单这一项就有 2.35×，GDN 中间态在长窗口 240M 上持平甚至 −13%。
4. 顺手抓到一个**会静默废掉整个训练**的坑：**V100 的 fp16 autocast 没配 loss scaling 时，输出端梯度 ~1e-7 会被 fp16 下溢成 0，全模型梯度为 0，训练只剩 weight decay**（证据见 §4）。

---

## 1. 240M 训练结果（MUSA 2×S4000，DP，bf16）

| run | 关键设置 | steps | s/步 | 时长 | 卡·时 | 训练内 PSNR | `val` 冷启动 起→终 |
|---|---|---:|---:|---:|---:|---:|---|
| `run_v7_a` | 冻结编码器 | 880 | 8.1 | 2.0 h | 4.0 | 13.26 | 10.61 → 11.02 |
| `run_v7_b` | 冻结编码器（长跑） | 1500 | 30.4 | 12.7 h | 25.4 | 14.38 | 10.99 → 10.79 |
| `run_v7_inject` | 冻结 + CanvasInject | 740 | 31.8 | 6.5 h | 13.1 | 13.64 | 10.99 → 11.02 |
| `run_v7_prefix` | 冻结 + prefix 读窗 | 350 | 31.9 | 3.1 h | 6.2 | 13.32 | 10.99 → 10.93 |
| `run_v7_c` | **解冻全参**（faithful） | 770 | 23.8 | 5.1 h | 10.2 | 14.23 | 10.79 → **11.01** |
| `run_v7_c_inject` | 解冻 + inject | 740 | 24.8 | 5.1 h | 10.2 | 14.54 | 11.02 → **11.12** |

合计 ≈ **69 卡·时 ≈ 2.9 卡·天**，约 **50 万监督帧**（≈17 小时视频 ≈ 数据集 12%）。

- 训练内 PSNR 12.6 → 14.5 有缓慢上升，但 `val` 冷启动始终 10.8–11.1 dB 横盘；解冻编码器、CanvasInject、prefix 读窗三个改动互相差 ≤0.1 dB（噪声级）。
- 早期数据规模实验里出现过的"中间态生成解码器基底"在这里**没有复现**：in-batch 涨、val 不涨，是典型欠拟合 + 泛化不足。

## 2. 评测（`val_long` 4 条 × 120 帧，bf16，`eval_v7.py`）

`run_v7_c`（解冻全参）：

| 协议 | PSNR (dB) | Δ vs rec |
|---|---:|---:|
| cold（逐帧 F=1） | 14.048 | +0.000 |
| rec（max_hist=0） | 14.047 | +0.000 |
| hist_1 / 4 / 8 / 16 | 14.039 | **−0.008 / −0.009** |
| warm（全历史） | 14.039 | **−0.009** |

`run_v7_c_inject`：cold 13.995 / rec 13.995 / hist 13.996 / warm 13.996（**Δ ≤ 0.001 dB**）。

基线（同 4 条片段）：**copy_prev 23.638 dB / copy_best 23.965 dB**（copy_best 帧龄中位 1 帧）；
`val` 全集参考：gray 常数 10.47 dB、blur 25.9 dB。

三个观察：

1. **时间通路贡献 ≈ 0**：cold = rec = hist(k) = warm，甚至 warm 比 rec 低 0.009 dB。"给更长历史"完全没有用。
2. **连单帧都还差 9.6 dB**：14.0 vs copy_prev 23.6。模型重建的是比均值好一点的糊图——这解释了为什么时间建模无从发挥作用：**基线（复制上一帧）都比模型强 9.6 dB，任何"融合历史"的机制都没有收益空间**。
3. **潜变量几乎不随帧变化**：相邻帧 `z_cls` 余弦 **0.9977**，`z_cls` 检索 top-1 帧龄中位 = 1 帧（余弦 0.9987）。即中间态/编码器把时间变化压扁了——历史信息在进入解码器前就被抹平。这是下一步要查的头号嫌疑（而非"历史机制不够强"）。

## 3. v3_1 效率实测（移植为 `code/model_v8.py` + `code/gdn.py`）

### 3.1 移植方式

`model_v3_1.py`（commit `7c4e42c`）依赖 `transformers>=5.22.2` 的 `Qwen3_5GatedDeltaNet`，而那版 transformers **硬性要求 `torch>=2.6`**；V100(sm_70) 上只有 torch 2.4.1 能跑。做法：
把 GDN 的**纯 PyTorch 回退路径**（`fla`/`causal_conv1d` kernel 缺席时库自己走的那条）摘成 `gdn.py`，并改成"只算 recurrent state、不算 token 输出"（中间态用固定 query 读出，本来就不需要输出，省 2 个 matmul）。数值自检（`python gdn.py`）：chunked state ≡ 逐 token delta rule（atol 1e-5）、流式 carry ≡ 整段一次、conv 流式 ≡ 整段。

`model_v8.py` 保留 v7 的编码器/解码器/损失/TBPTT 口径，把中间态换成 v3_1 的「固定 query + GDN 状态递推」，decode 改为帧维折 batch（可 `dec.frame_chunk` 分块控显存）。cfg 开关可拆开归因：
- `mid.kind = "gdn" | "recurrent"`（v3_1 中间态 / v7 中间态）
- `dec.frame_batch = true | false`（帧维折 batch / 逐帧循环）

自检全过：F=1 与视频首帧逐位一致、`Q_f` 不读当前帧、梯度全通（含 GDN 的 conv/in_proj/A_log）、帧维折 batch ≡ 逐帧 loop（recurrent 路径 atol 1e-6）。

### 3.2 MUSA（240M，bs4，F=32 / n_warm=24，bf16，单卡合成张量）

| 变体 | 参数量 | s/步 | ms/监督帧 |
|---|---:|---:|---:|
| v7（recurrent 中间态 + 逐帧 decode） | 703M | 33.0 | 1032 |
| v7 + 帧维折 batch（只改 decode） | 703M | 31.6 | 986 |
| **v3_1（= v8）** | 661M | **8.4** | **263** |

→ **3.9×**；而"只折 decode"几乎没用（1.05×）。**在 MUSA 上瓶颈是中间态**：v7 每帧都要对"增长的历史 KV"重做投影，总代价 O(F²·K·d²)（F=32、K=32 时最后一帧的 mem 长 1025 token）；v3_1 的状态是常数大小（O(H·head_dim²)）。

### 3.3 MUSA 真实数据 + 双卡 DP（bs4/rank，hist 8/16/24，p_cold 0.3，即原 v7 run 同款设置）

| | s/步 | 监督帧/步 | ms/监督帧 | 一天能看 |
|---|---:|---:|---:|---:|
| v7（`run_v7_c` 实测） | 23.8 | 64 | 372 | ~23 万帧 |
| **v3_1** | **9.9–11.2** | 64 | **~164** | **~53 万帧** |

显存 28.4 GB/rank，14 步无 MCCL 报错。→ 同样 2 天量的 240M 训练，v3_1 约 **0.9 天**跑完。

### 3.4 V100（fp16，单卡，与另一训练共卡，比例可比）

| 配置 | v7 | v7+折 batch | v3_1 | 提速 |
|---|---:|---:|---:|---:|
| 32M（88–93M），bs4，F=32/w24 | 5.40 s | — | 3.51 s | **1.54×** |
| 32M，bs2，F=8 冷启动 | 5.96 s | 3.43 s | 1.87 s | **3.19×**（v3_1 vs v7） |
| 240M，bs1，F=16/w8 | 2.97 s | 1.23 s | 1.39 s（−13% vs 折 batch） | 1.4–2.1× |

→ 在 V100 上 v3_1 的收益**主要来自 decode 帧维折 batch**（240M 单这一项 2.35×），GDN 中间态在 240M 长窗口下不再有额外收益（略慢 13%，因为多了 chunked UT 求解）。**结论：v3_1 里"帧维折 batch"这一改动应该无条件保留；GDN 中间态在慢卡 / 长历史上才是主角。**

## 4. 工程坑：fp16 无 loss scaling → 梯度全 0（会静默废掉整轮训练）

V100 上 32M 图像阶段跑了 2.3 小时，`val` 冷启动 10.6767 → 10.6735（横盘）。逐张量比对 `step500.pt` 与 `step2000.pt`：

- **所有参数都满足 `y = 0.989321·x`（残差 4e-6）** —— 1500 步里唯一的更新是 AdamW 的 weight decay；
- 同一前向、同样数据：fp16 autocast 下 encoder/middle/decoder 的 **grad_norm 全为 0（297 个张量无一非零）**；fp32 下为 1.9e-2 / 1.8e-3 / 2.5e-2。

原因：本工程的 loss 是对 (batch×帧×步×像素) 取均值，单像素梯度 ~1e-7，低于 fp16 最小次正规数 6e-8 → 反向全下溢。
修复：`train_v7.py` 加 `GradScaler`（`unscale_` → clip → `scaler.step/update`，ckpt 内保存 scaler 状态），并在日志打印 clip 前 grad_norm。修复后实测 grad_norm 0.36/0.83/0.65、80 步内 loss 0.2345→0.2063、in-batch PSNR 11.69→12.28。
**bfloat16 不受影响**（指数范围同 fp32），所以 MUSA 上的 240M 实验有效。

## 5. 算力账：为什么 240M 现在一定是负结果

| 口径 | 数值 |
|---|---|
| 数据集一轮 | 32,165 条 × 120 帧 = **386 万** 监督帧 |
| 240M v7 实际 | 6 条 run 合计 **~50 万帧**（12% 一轮）≈ 69 卡·时 |
| 240M v3_1 | **~53 万帧/天**（双 S4000）→ 一轮 ≈ 7.3 天 |
| 32M v3_1 | ~65 万帧/13 h（V100 单卡，按 1.54× 折算）→ 一轮 ≈ 2.6 天 |
| 参考基线 | copy-prev 23.0–23.6 dB / blur 25.9 dB（无需训练即可达到） |

即：**要让 240M 这个规模有意义，至少要多 1–2 个数量级的算力；把 v3_1 的 2.3–3.9× 用满也要一周一轮**。

## 6. 结论与下一步

1. **240M/703M 这个规模的结论是"算力不足导致的欠拟合"**，不是架构被证伪；`val`/`val_long` 上 cold=rec=warm 说明时间通路没起作用，而单帧重建离 copy 基线还差 9.6 dB。
2. **v3_1 的收益要分机器看**：MUSA（慢卡、中间态瓶颈）3.9×；V100（decode 瓶颈）主要靠帧维折 batch（2.35×@240M）。两者都应保留。
3. **优先级建议**：先把"单帧能重构 / 潜变量不塌缩"解决（相邻帧 `z_cls` 余弦 0.9977 是直接线索），再谈时间建模与规模；解码器/信息通路改造优先于加卡。
4. 待办：
   - 32M v3_1 的收敛对照（K=16、8 步 × 每步 2 码字）；
   - 解码器改造后，用 `code/bench_v8.py` 在同一 `F/n_warm/bs` 口径下复测单步时间；
   - `fp16` 一律带 GradScaler；`bf16` 不需要。

## 7. 复现

```bash
# 0) GDN 数值自检 + v3_1 移植自检（CPU 即可）
python gdn.py && python model_v8.py

# 1) 同参数量单步时间（v7 / v7+折batch / v3_1）
python bench_v8.py --cfg cfg_v8_240m.json --variant v7,v7b,v8 \
       --frame-chunk 4 --bs 4 --F 32 --n-warm 24 --amp-dtype bf16   # MUSA
python bench_v8.py --cfg cfg_v8_32b.json  --variant v7,v7b,v8 \
       --frame-chunk 4 --bs 4 --F 32 --n-warm 24 --amp-dtype fp16   # V100

# 2) 训练（cfg 里 "arch": "v8" 走 v3_1，否则 v7；两阶段看护脚本 run_v8.sh）
bash run_v8.sh     # image 2h → video 14h，fp16+scaler，BS 8

# 3) 评测（cold / rec / warm / hist-sweep / copy 基线）
python eval_v7.py --ckpt <ckpt> --cfg-json cfg_v7b.json \
       --root data3/val_long --clips 4 --window 120 --hist-sweep 1,4,8,16 \
       --amp bf16 --out <out_dir>
```
