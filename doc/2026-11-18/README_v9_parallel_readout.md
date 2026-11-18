# v9 代码导读 — 并行（前缀）读出视频模型

> 这个分支（`v9`）就是 **2026-11-18 下午训练出 `eval_psnr = 27.98 dB`（视频 val，
> 冻结分词器满读上界 30.53 dB）的那份代码**，原样保存，方便对照学习。
> 注意：文件名仍叫 `model_v10.py` / `train_v10.py`，因为它是同一套「视频通路」
> 代码；**架构范式**是 v9 的并行读出（见下）。循环（递推）读出的改动在 `main`
> 分支，不在这里。

## 1. 范式一句话

图像/video 的 rate 是 30 步 × 每步 1 组 z（共 120 帧）。v9 的读出是**并行**的：
第 t 步直接把「已读过的前 t+1 组 z」全部当 cross-attn 的 key，**不依赖第 t-1 步
的输出**，30 个档位之间可以独立并行计算。反过来，v10 设计原版是**串行递归**：
第 t 步只看当步 z_t，历史信息靠 `Y_{t-1}` 递推搬运（30 步必须顺序算）。

## 2. 文件地图

| 文件 | 干什么 |
|---|---|
| `model_v10.py` | 模型：`FrozenTokenizer` / `TemporalMiddle` / `WholeClipDecoder` / `VideoV10` |
| `data_v10.py` | 120 帧 / 448² 的片段数据集（PyAV 解码） |
| `build_latent_cache.py` | 离线跑冻结 E，把 `(z_cls, z_s)` 存成 shard（训练不再解 mp4/跑 487M E） |
| `train_v10.py` | HF Trainer 训练：DDP + bf16 + `V10EvalCallback`（自定义 eval 落 `best_trainable.pt`） |
| `eval_v10_stepwise.py` | 逐档评测：30 步逐步 PSNR / latent L1 曲线 |
| `run_v10_cache.sh` | 4 个**独立单进程**建缓存（MUSA 上不要 4 进程同时初始化 MCCL，会卡驱动） |
| `run_v10_train.sh` | 4 卡正式训练（本分支=并行读出口径） |
| `doc/2026-11-18/PLAN_v10_120f_video.md` | 设计/配置/踩坑/复现 |
| `doc/2026-11-18/REPORT_tok448_2stage.md` | 冻结分词器（448²→16×448）的两阶段训练报告 |

## 3. 数据流（一个训练 step 的形状变化）

```
mp4 120帧×448² ──冻结 E──► z_cls (B,120,1,448) + z_s (B,120,16,448)   ← 离线缓存的全部内容
                                   │
                    rate = z_s[:, :16]（率=16 token，前缀截断，无投影）
                                   ▼
              reshape (B,120,8,896)  [chunk_tokens=2 → 一个向量 2×448]
                                   │  TemporalMiddle：6 层 d=896，帧内互看、跨帧因果
                                   ▼
                     z (B,120,8,896) → reshape (B,30,32,896)：每步 4 帧 × 8 行
                                   │
               ┌───────────────────┴────────────────────┐
               │ 读出：30 步，每一步都重出整段 121×9 个位置 │
               └───────────────────┬────────────────────┘
                                   ▼
        pred (B,F,17,448)：第 0 行 z_cls，1..16 是 16 个 special
                                   │
                            冻结 R（末档）──► 像素 (B,F,3,448,448)
```

### 3.1 读出每一步（`WholeClipDecoder.forward`，prefix 分支）

```python
h  = LN(Q + γ_t · Y_{t-1}) + step_emb[t]      # Q: (1,1089,896) 可学习查询；γ 零初始化
nf = min(F, (t+1)*group)                      # 已读帧数
keys = z_rows[:nf] + frame_emb[:nf] + row_emb # 已读前缀 + 帧位置 + 行号
Y  = Decoder(tgt=h + qe, memory=keys)         # 帧级因果 self-attn + cross-attn
out[t] = split(LN(Y)) + resid                 # resid = 已读帧的 rate 行直抄
```

- **`copy_skip`（直抄残差）**：`resid[:, :nf, :rr] = z_rows[:, :nf]`。已读帧
  rate 行原样加到输出行上（第 r 行 = 该帧第 r 个 rate 向量）。这样「把读到的
  帧抄到输出」是零学习成本的；没有它时实测会塌成「与输入无关的模板」
  （middle 输出跨片段 std 0.29→0.0155，pred 跨片段差 0.0002）。
- **`loss_read`（已读帧 masked loss）**：第 t 步只在「已读帧」上算 L1。
  否则 30 步里绝大多数位置是未读帧，均值解是局部极小，梯度把 attention 压成 0。
- 信息论上 v9 只在「已读帧」上被监督；未读帧的输出没有训练信号，所以
  **整片段 PSNR 要按「读全量的第 30 档」读**（eval 里就是末档）。

### 3.2 三处「零冗余」设计

1. **不建 `kv_proj`**：z 本来就是 896 宽，投影只是把同一向量再线性混合一次；
2. **不建 specials head**：8 行 896 直接 reshape 成 16×448（`chunk_tokens=2`），
   只有 z_cls 一行要 `cls_head`（896→448）；
3. **不复制 memory**：整段是一条 batch，cross-attn 直接对前缀 key，不需要
   `repeat_interleave` 出 120 份。

## 4. 训练配方（本分支跑出 27.98 dB 的那次）

- 数据：32,165 train / 1,142 val 片段，120 帧 8fps、原生 448²
- 可训练：middle 38.6M + 读出 40.0M ≈ 78.7M；冻结 E+R 486.7M
- 率 = 目标 = 16 token/帧（7168 值/帧，860,160 值/片段），**没有压缩级**
- 4×MTT S4000，bf16 autocast，AdamW lr 8e-4、warmup 2%、cosine 4000 步、
  wd 0.05、grad clip 1.0；micro-batch 2 × accum 2 × 4 卡 = 全局 16
- 单步 2.83 s，总 3 h 12 min；训练中每 200 步在 2 条 val 上测 PSNR

结果：`eval_latent_l1 0.148 → 0.0286`，`eval_psnr 20.25 → 27.98 dB`（@step3000），
冻结分词器满读上界 **30.53 dB**。缺口约 2.6 dB，来自 **z_cls 不在率里**（要从
16 个 special 反推，相对误差 ~40%）。

## 5. 复现

```bash
# 0) 同步（本地 → 远端）
rsync -az model_v10.py data_v10.py build_latent_cache.py train_v10.py \
  eval_v10_stepwise.py run_v10_cache.sh run_v10_train.sh sha1:/root/AssocVTC_v10/

# 1) 缓存（4 卡独立进程，85 min，59 GB）
ssh sha1 'cd /root/AssocVTC_v10 && nohup bash run_v10_cache.sh > logs/cache.log 2>&1 &'

# 2) 训练（4 卡，3 h 12 min）
ssh sha1 'cd /root/AssocVTC_v10 && nohup bash run_v10_train.sh > logs/train.log 2>&1 &'

# 3) 逐档评测
ssh sha1 'cd /root/AssocVTC_v10 && SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0 \
  python eval_v10_stepwise.py --trainable <out>/best_trainable.pt \
  --n_clips 60 --out <out>/eval_stepwise.json'
```

## 6. 读代码建议顺序

1. `V10Config`（宽度/率/步数/目标的硬约束，`model_v10.py` 开头 dataclass）
2. `FrozenTokenizer.encode/render`（率与目标从哪来）
3. `TemporalMiddle`（帧内 block 因果 + 跨帧因果）
4. `WholeClipDecoder.forward`（本分支的 prefix 分支 = v9 并行读出）
5. `VideoV10.forward`（loss_read 的 mask 怎么造）
6. `train_v10.py` 的 `V10EvalCallback`（为什么要自定义 eval）

## 7. 边界（诚实标注）

- 率=目标=16（分词器 K=16 的硬约束）：末档最好情况只是**复现冻结分词器满读**
  （30.53 dB）；v9 不存在「率窄于目标 → 用邻帧回收被丢细节」的收益。
- 逐档曲线只在「已读帧」上有意义（`loss_read=1`）；未读帧无监督。
- loss 是 latent L1；像素 PSNR 只经冻结 R 渲染后计算；单 seed。
