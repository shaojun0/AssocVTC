# v9 代码导读 — 并行（前缀）读出视频模型

> `main` 分支上的 v9 = **`model_v9.py`**（并行/前缀读出）。就是 2026-11-18 训练出
> `eval_psnr = 27.98 dB` 的那份代码（视频 val，120 帧 × 448²；上界 = 冻结分词器
> 满读 **30.53 dB**）。v10（`model_v10.py`）是串行递归读出，见对照说明 §1。

## 1. 范式一句话

30 步读出每一步都重出整段 120 帧，区别只在 memory：

| | v9（本文件） | v10 |
|---|---|---|
| 第 t 步 cross-attn 的 key | **已读前缀**：前 t+1 组 z（每组 4 帧 × 8 行），key/query 带「帧位置 + 行号」嵌入 | 当步 z_t（32 行），历史靠 `Y_{t-1}` 递推搬运 |
| 步间依赖 | 无 ⇒ 30 个档位可**独立并行**计算 | 严格串行（后一步依赖前一步状态） |
| 逐步细化 | 没有「后文回改前文」 | 有 |
| 收益来源 | 已读帧立刻精确可出（直抄残差）+ 未读帧由前缀条件预测 | 递推状态 + 逐步细化 |

> 名字沿用 v10 的「全片段重出 / 1792→896 宽度 / 三处零冗余」口径，只是把 memory
> 从当步 z 换成已读前缀，并加直抄残差 + 已读帧 loss —— 这两件是「能训起来」的关键。

## 2. 文件地图

| 文件 | 干什么 |
|---|---|
| `model_v9.py` | **v9 模型**：`V9Config` / `FrozenTokenizer` / `TemporalMiddle` / `ParallelReadout` / `VideoV9`；`python model_v9.py` 可 CPU 自检 |
| `train_v9.py` | v9 训练（HF Trainer + DDP + bf16 + `V9EvalCallback`，落 `best_trainable.pt`） |
| `eval_v9_stepwise.py` | 逐档评测：30 步逐步 PSNR / latent L1 曲线 |
| `run_v9_train.sh` | 4 卡正式训练（本分支配方：`loss_read=1, copy_skip=1`） |
| `data_v10.py` | 120 帧 / 448² 片段数据集（v9/v10 共用，不改） |
| `build_latent_cache.py` + `run_v10_cache.sh` | 离线缓存冻结 E 输出 `(z_cls, z_s)`（v9/v10 共用，不改） |
| `test_v9_parity.py` | 自检：`VideoV9` 与 `VideoV10(mem_mode="prefix")` 同权重输出逐位相同 |
| `model_v10.py` | v10 串行递归读出（含 `mem_mode="step"` 与 `"prefix"` 两种 memory） |
| `doc/2026-11-18/PLAN_v10_120f_video.md` | 设计/配置/踩坑/复现/单片段过拟合对照（§1.2） |
| `doc/2026-11-18/REPORT_tok448_2stage.md` | 冻结分词器（448² → 16×448）两阶段训练报告 |

## 3. 数据流（一个 step 的形状变化）

```
mp4 120帧×448² ──冻结 E──► z_cls (B,120,1,448) + z_s (B,120,16,448)   ← 离线缓存的全部内容
                                   │
              rate = z_s[:, :16]（率=16 token，前缀截断，无投影）
                                   ▼
              reshape (B,120,8,896)  [chunk_tokens=2 → 一个向量 2×448=896]
                                   │  TemporalMiddle：6 层 d=896，帧内互看、跨帧因果
                                   ▼
                     z (B,120,8,896) → reshape (B,30,32,896)：每步 4 帧 × 8 行
                                   │
               ┌───────────────────┴────────────────────┐
               │ 读出：30 步，每一步都重出整段 121×9 个位置 │ ← v9 里 30 步可并行
               └───────────────────┬────────────────────┘
                                   ▼
        pred (B,F,17,448)：第 0 行 z_cls，1..16 是 16 个 special
                                   │
                            冻结 R（末档）──► 像素 (B,F,3,448,448)
```

### 3.1 读出每一步（`ParallelReadout.forward`）

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
- v9 只在「已读帧」上被监督；未读帧的输出没有训练信号，所以**整片段 PSNR 要按
  「读全量的第 30 档」读**（eval 里就是末档）。

### 3.2 三处「零冗余」设计

1. **不建 `kv_proj`**：z 本来就是 896 宽，投影只是把同一向量再线性混合一次；
2. **不建 specials head**：8 行 896 直接 reshape 成 16×448（`chunk_tokens=2`），
   只有 z_cls 一行要 `cls_head`（896→448）；
3. **不复制 memory**：整段是一条 batch，cross-attn 直接对前缀 key，不需要
   `repeat_interleave` 出 120 份。

## 4. 训练配方（跑出 27.98 dB 的那次）

- 数据：32,165 train / 1,142 val 片段，120 帧 8fps、原生 448²
- 可训练：middle 38.6M + 读出 40.0M ≈ 78.7M；冻结 E+R 486.7M
- 率 = 目标 = 16 token/帧（7,168 值/帧，860,160 值/片段），**没有压缩级**
- 4×MTT S4000，bf16 autocast，AdamW lr 8e-4、warmup 2%、cosine 4000 步、
  wd 0.05、grad clip 1.0；micro-batch 2 × accum 2 × 4 卡 = 全局 16
- 单步 2.83 s，总 3 h 12 min；训练中每 200 步在 2 条 val 上测 PSNR

结果：`eval_latent_l1 0.148 → 0.0286`，`eval_psnr 20.25 → 27.98 dB`（@step3000），
冻结分词器满读上界 **30.53 dB**。缺口约 2.6 dB，来自 **z_cls 不在率里**（要从
16 个 special 反推，相对误差 ~40%）。

## 5. 复现

```bash
# 0) 同步（本地 → 远端）
rsync -az model_v9.py train_v9.py eval_v9_stepwise.py run_v9_train.sh \
  data_v10.py build_latent_cache.py sha1:/root/AssocVTC_v10/

# 1) 缓存（4 卡独立进程，85 min，59 GB；v9/v10 共用）
ssh sha1 'cd /root/AssocVTC_v10 && nohup bash run_v10_cache.sh > logs/cache.log 2>&1 &'

# 2) 训练（4 卡，3 h 12 min）
ssh sha1 'cd /root/AssocVTC_v10 && nohup bash run_v9_train.sh > logs/v9_train.log 2>&1 &'

# 3) 逐档评测
ssh sha1 'cd /root/AssocVTC_v10 && SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0 \
  python eval_v9_stepwise.py --trainable <out>/best_trainable.pt \
  --n_clips 60 --out <out>/eval_stepwise.json'
```

自检（不需要 GPU/ckpt）：

```bash
python model_v9.py        # 小号模型形状/因果/梯度/20 步过拟合
python test_v9_parity.py  # VideoV9 == VideoV10(mem_mode="prefix") 逐位一致
```

## 6. 读代码建议顺序

1. `V9Config`（宽度/率/步数/目标的硬约束）
2. `FrozenTokenizer.encode/render`（率与目标从哪来）
3. `TemporalMiddle`（帧内 block 因果 + 跨帧因果）
4. `ParallelReadout.forward`（v9 的核心：已读前缀 + 直抄残差）
5. `VideoV9.forward`（`loss_read` 的 mask 怎么造）
6. `train_v9.py` 的 `V9EvalCallback`（为什么要自定义 eval）

## 7. 边界（诚实标注）

- 率=目标=16（分词器 K=16 的硬约束）：末档最好情况只是**复现冻结分词器满读**
  （30.53 dB）；v9 不存在「率窄于目标 → 用邻帧回收被丢细节」的收益。
- 逐档曲线只在「已读帧」上有意义（`loss_read=1`）；未读帧无监督。
- loss 是 latent L1；像素 PSNR 只经冻结 R 渲染后计算；单 seed。
