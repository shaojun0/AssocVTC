# 我们的 KD 代码 vs ALIT 官方代码：逐项对比与改进点

- 时间：2026-11-23
- 对象 A（我们）：`kd_loss.py` + `train_linattn.py` 的 KD 分支 + `model_v2_linattn.py` 的损失口径（commit `741a1af`）
- 对象 B（ALIT）：官方仓库 `adaptive-length-tokenizer`（本地 `/home/linaro/dsh/alit_240m`，commit `2042e57`）
- 结论（一句话）：**两边都叫"蒸馏"，但做的不是同一件事。ALIT 是把 teacher 的 latent 当回归目标、把
  teacher 的 decoder 原样留作渲染器、且 stage-1 里没有任何像素损失；我们是像素级输出拟合 + 自建
  decoder 从零学。这三点差异里，前两点就是它 rFID 8.25 与我们的 86.50 的病灶。**

---

## 0. 先更正 `PLAN_distill.md` §0 的一句错话

PLAN 里写"ALIT **没有** teacher-student 知识蒸馏，它被称作 distillation 的是 2D→1D token 转置机制"。
读完代码这句是**错的**（或至少只描述了 stage-2 的一个侧面）：

| 证据（ALIT 代码） | 事实 |
|---|---|
| `adaptive_tokenizers/adaptive_vae.py:282-284` | `forward_loss(gt_code, decoded_code) = MSELoss(gt_code, decoded_code)`，`gt_code` 来自 **frozen base tokenizer** 的 encoder |
| `adaptive_tokenizers/adaptive_vae.py:120-122` | `reconstruct_images()` = `self.base_tokenizer.vae.decode(decoded_code)` ⇒ **像素由 teacher 的 decoder 渲染** |
| `adaptive_tokenizers/adaptive_vqgan.py:301-316` | VQGAN 版：`nll_loss` = label-smoothing CE 对 **teacher 的 codebook index** + `code_loss` = MSE(teacher code embedding, soft-expected student embedding) |
| `adaptive_tokenizers/adaptive_vae.py:247-250` | stage-1 的 loss **只有** `Σ_t lambda[t]·code_loss`（+ 量化器项），**没有任何像素损失** |
| `modules/base_tokenizers.py:52-60` | stage-1 时 base tokenizer `requires_grad=False` 且 `forward` 在 `no_grad` 内，latent 乘 `0.2325` 做尺度归一 |
| `main_full_finetuning.py:127-131` | stage-2 才把 base tokenizer 设成 `is_requires_grad=True`，用 `VQLPIPSWithDiscriminator`（L1+LPIPS+GAN，`disc_weight=0.2`）在 **渲染出来的像素 vs GT** 上训 |
| `run_scripts/latent_distillation_pretrain.sh` | stage-1：200 epoch、`blr 1.5e-4`、`bs 32×8GPU`、warmup 20 |
| `run_scripts/full_finetuning.sh` | stage-2：200 epoch、`blr 1e-4`、`bs 12×?GPU` |

也就是说 ALIT 的配方是**两段式、目标不冲突**：

1. **stage-1（latent distillation）**：学"从图像 → teacher 的 latent"，损失只有 latent MSE/CE；
2. **stage-2（GAN finetune）**：teacher decoder 渲染，损失只有像素 L1+LPIPS+GAN vs GT。

我们现在的做法是把 1 和 2 塞进**同一个** loss：`L1(8 步, x) + λ·L1(末步, t)`，两边目标互相拉扯，
且没有任何一段是"纯 latent"的。这是结构性的差异，不是超参差异。

---

## 1. 逐项对比

| 维度 | ALIT（VAE 版 stage-1/2） | ALIT（VQGAN 版 stage-1/2） | 我们（`kd1`） |
|---|---|---|---|
| teacher | frozen LDM KL-VAE（`vae.ckpt`） | frozen VQGAN（72.1M） | frozen `sd-vae-ft-mse`（83.7M） |
| **蒸馏对象** | **latent `z`（16×16×16=4096 值）** | **codebook index（CE）+ soft code（MSE）** | **像素 `D(E(x))`（3×256×256）** |
| 损失度量 | `MSE(z_t, z_hat)`，`z` 先乘 0.2325 | CE(16384 类, label smoothing 0.1) + MSE(embedding) | `L1(pred01, t01)`，`t01∈[0,1]` |
| 像素损失 | stage-1 **没有**；stage-2 有（vs GT） | 同 | 与 KD **同时**存在（vs GT） |
| 渲染器（测试时） | **teacher 的 VAE decoder（stage-1 冻结）** | **teacher 的 VQGAN decoder** | 我们自己的 decoder（从零用像素 L1 训出来，无预训练先验） |
| 监督的步数 | **每一步都算**，权重 `[2.5,2.0,1.5,1.25,1.0,1.0,1.0,1.0]` | 同 | **只末步**，λ 常数（ramp 后 =1.0） |
| 目标冲突 | 无（单一目标） | 无 | **有**（`L1(x)` 与 `L1(t)` 稳定点 = 二者平均） |
| 学生输入 | teacher tokens 与 patch-embed 拼接后 `F.normalize` | 同 | 原图 patches |
| 正则 | dynamic halting（已近猜对的 image token 换 mask） | 同（按 GT 概率阈值 0.55） | 无 |
| 目标噪声 | `.sample()`（随机采样 latent） | 硬 index | `.mode()`（确定性） |
| EMA | 有（eval/可视化用 ema） | 有 | 无 |
| 预算 | stage-1 200ep + stage-2 200ep，IN-100，从零 | 同 | 2 epoch 续训（14ep 收敛点上） |

**两个"人无我有"的讽刺点**：

- 我们已经有 ALIT 那套"逐步加权 + 随机抽步"的设施（`model_v2_linattn.py:1485-1512` 的
  `_aggregate_loss`、`--loss_weights`、`--sample_grad_iters`，注释里就写着 "ALIT/FID protocol"），
  但 KD 这一臂**一个都没用**；
- ALIT 的 stage-1 反而**不**用像素损失，而我们花了一半的梯度预算在像素 L1 上。

---

## 2. 为什么我们这一臂必然白跑（用我们自己的数字解释）

### 2.1 L1 在我们的工作点上几乎与 FID 解耦

同 5000 张 IN-100 val、同一 FID 口径（`doc/2026-11-23/artifacts/sdvae_fast.json`）：

| 臂 | L1(·, GT) /255 | PSNR | rFID |
|---|---|---|---|
| GT 模糊 σ=0.5 | 2.36 | 37.57 | 0.91 |
| GT 模糊 σ=1.0 | 6.46 | 28.65 | 9.15 |
| **SD-VAE 重建**（teacher） | **9.31** | 25.93 | **3.44** |
| GT 模糊 σ=2.0 | 10.31 | 24.56 | 27.20 |
| **我们 k256x12_t8（3072 值）** | ≈12.98（由 `l1_norm 0.22499` 推算） | 22.67 | **86.50** |

看第 3、4 行：**teacher 的 L1（9.31）比 σ=2 模糊（10.31）还好一点点，但 rFID 差 8 倍**
（3.44 vs 27.20）。同一张表里 L1 与 rFID 甚至不是单调对应的 ⇒ 在 9–13/255 这个区间里，
**L1 基本上看不见 FID 关心的东西（纹理/高频结构）**。

### 2.2 KD 项的实测值正好落在这个"盲区"

`kd1` 训练日志（`tok448_work/artifacts/kd1/`）实测 `kd = L1(pred01, t01)`：

- 1120 步区间内 `kd ∈ [0.039, 0.053]`，**无趋势**（首 60 步 0.0397 → 末 1120 步 0.0424）；
- 对应 `11.5/255`，与 `L1(pred, x) ≈ 13.0`、`L1(t, x) = 9.3` 同量级。

也就是说：把 pred 完全搬到位到 t，L1 只从 13.0 降到 9.3（**−21%**，PSNR +3.3 dB），
但 rFID 会从 86.5 变 3.44（**−96%**）。**这 21% 的 L1 里，真正对应 FID 的那部分又只占一小块**
（因为 teacher 自己的 L1 里有 9.3 全是"噪声"级误差、FID 却不罚）。L1 梯度不会告诉我们该往哪 21% 走。

### 2.3 再加一个目标冲突，就锁死了

`loss = L1(pred, x) + 1.0·L1(pred, t)`，`t` 与 `x` 是两个不同目标点。两个 L1 都偏好"低对比、
平滑、向均值回归"的解 —— 于是它们的加权平均稳定点仍然是糊的，只是糊的"风格"折中。
`kd1` 的 ep1 val PSNR 22.558（起点 22.607）正好是这个现象的签名：**没变化，因为没被往任何一个
方向推动**。这与"teacher 没被拟合"的另一种解释（尺度过小/梯度太小）不同：kd 项与基础 L1 在
像素空间里是**同量级**的，问题不是它小，是它**指的方向被基础项抵消、且与 FID 无关**。

⇒ 结论：**换度量（latent/LPIPS）+ 换渲染器（teacher decoder），而不是调 λ。** 若继续在像素 L1
上试 λ=3/5，`NOTES_rd_positioning` 里那条"纯 R-D"结论同样会重演。

---

## 3. 改进点（按"性价比 × 上限"排序）

### P0-d（20 min，不训练，先做）：量"VAE 流形投影"的天花板

问题：如果把我们现在的重建图**塞回 SD-VAE 的流形**（`ours_r = D(E(ours))`），rFID 是多少？
这是"latent 路线"能拿到的**上界代理**——它等于假设 z_hat 精确等于"我们自己重建图的 z"。

- 若 `rFID(real, D(E(ours)))` ≪ 86.5（比如 15–30）⇒ 差距几乎全部来自"输出不在 VAE 流形上"，
  §3-I2 值得做，且这条路的上限很高；
- 若仍 ≈70+ ⇒ 我们的重建丢的信息在 latent 里也补不回来，latent 路线要打折，转 refiner。

落点：给 `tools/fast_rfid.py` 加 `--vae_roundtrip 1`（默认关，附加第二个臂，不改原臂），
一步得到 `ours` 与 `D(E(ours))` 两组 PSNR/rFID。命令：
`python3 fast_rfid.py --ckpt <ckpt_best> --vae_roundtrip 1 --out p0d.json`。

### I1（必做）：把 KD 的度量从像素 L1 换成 teacher latent MSE

- 目标：`z_t = sdvae.encode(x).latent_dist.mode()`，`z_hat` 由我们预测；
- **尺度**：两边都乘 `0.18215`（SD 约定的 latent 归一化；ALIT 对应 `mul_(0.2325)`）再算 MSE，
  否则 MSE 被大方差通道主导；
- 理由：LDM 的 latent 空间是**去相关、按方差缩放**过的，MSE 在那里对纹理敏感（§2.1 的病根），
  且 ALIT 的 stage-1 全部收益就建立在这上面；
- 落点：`kd_loss.py` 增加 `TeacherLatent`（只跑 encoder，不跑 decoder ⇒ 便宜一半，见 I6），
  `train_linattn.py` 增加 `--kd_space {pixel,latent}`。

### I2（上限最高，ALIT 同构）：payload → `z_hat` → **frozen SD-VAE decoder** 渲染

这是 ALIT 真正的结构：**它从不训练像素 decoder**，只在 stage-2 用 GAN 微调 teacher 的 decoder。
我们让 3072 值 payload 直接回归 teacher latent，再由冻结的 SD-VAE decoder 出图：

```
z_hat_t = z_head(Y_pix_t)                # (B,T,N,768) → (B,T,4,32,32)
z_hat_t = z_hat_t / 0.18215              # 反归一化到 SD 的 z 尺度
recon_t = sdvae.decode(z_hat_t).sample   # 冻结, fp32, no_grad 之外可微
loss   = MSE(z_hat_t*0.18215, z_t*0.18215)   (+ 可选 LPIPS(recon_t, x))
```

- **rate 不变**：payload 仍是 16×192 = 3072 值（step1 = 384）；
- **上限** = teacher 的 3.44 + latent 预测误差。参考 ALIT：它在**完全同 rate**（256 token × 12 维
  = 3072 值）下 rFID 8.25 —— 它的 payload 比我们小、teacher（VQGAN）也不比 SD-VAE 强，所以
  **"同 rate 下 8–15" 是一个有依据的预期**，而不是 3.44；
- 对我们的 R-D 阶梯天然友好：8 步各自有 `z_hat_t`，逐速率点直接 `decode(z_hat_t)` 即可；
- 成本：先做"最小改动版"（保留现有解码器主干，加一个 `z_head` 从 `Y_pix` 上做 stride-8 卷积下采样），
  跑通后再考虑把像素头整个换掉（更 ALIT，但也更侵入）；
- 风险（须诚实写进论文）：**解码器不再是我们自己的**。但 ALIT 同样借 VQGAN decoder（stage-2 还
  一起微调），这条"借 decoder"在同类工作里是标配；我们要在对比表里写清 teacher/decoder 的归属。
- 评测落点：`fast_rfid.py` 加一条 `--render sdvae` 路径（把 `recon_t` 换成 `decode(z_hat_t)`），
  与 I2 一起改。

### I3（必做，和 I1/I2 一起）：消掉目标冲突

三选一（推荐 3）：

1. `loss = L1(pred, t)`，完全放弃 `L1(x)`：最干净，但"输出会变成 SD-VAE 风格"，PSNR 上限被锁在 25.93；
2. `loss = L1(pred, t) + 0.1·L1(pred, x)`：留一个防漂移锚，λ_x 从小（0.05–0.1）起；
3. **两段式（ALIT 原样）**：stage-A 只做 `MSE(z_hat, z_t)`（无像素损失）；stage-B 只在渲染图上做
   `L1+LPIPS+GAN vs GT`。我们已经有一整套 perc/rfid/gan 挂载点（`train_linattn.py:_pred01/_tgt01`），
   stage-B 几乎零新代码。

### I4：逐步监督 + 逐步权重（设施已存在，只是没接）

- ALIT 对**每个 rollout 迭代**都算损失，权重 `[2.5,2.0,1.5,1.25,1.0,1.0,1.0,1.0]`；
- 我们的 `out["Y_pix"]` 就是 `(B,T,N,pp)`，逐步 KD 是**纯加法**（`patches_to_image` 摊平 T 维即可），
  当前只挂末步是浪费；
- 但**权重方向要和 ALIT 相反**：ALIT 早期迭代 rate 也低（1→4 token）却给 2.5×，因为它的目标是
  "课程"（早期迭代要够准才能做 halting 决策）；我们早期步只有 384 值，把 teacher 目标挂上去
  等于让 384 值去装 4096 值的细节 ⇒ 建议**随步递增**（如 `0.2,0.3,0.5,0.7,0.9,1.0,1.0,1.2`），
  或至少 `step>=6` 才开 KD。这条可以用 `--loss_weights` 直接跑，零代码；
- `--sample_grad_iters`（ALIT 的省算力开关）同理，我们已实现。

### I5：λ 调度与学习率（别再手调常数 λ）

- 现有 ramp：`kd_start_frac=0.02`、`kd_ramp_frac=0.03` ⇒ 110 步就满 1.0，在收敛点上等于突然换目标；
- 可复用 `gan_adaptive` 的**梯度范数比**机制自动定 λ：让 `‖∇_pred(L_kd)‖ ≈ c·‖∇_pred(L_recon)‖`，
  比手调更不容易被量纲骗（尤其 I1 换成 latent MSE 后量纲完全变了）；
- lr：换目标等于离开 L1 的极小点，2e-5 太小（`kd1` 跑 1120 步 lr 才从 2e-5 降到 1e-5，等于没动）。

### I6：teacher 只跑 encoder + 离线缓存 z（直接省一半步时）

- 我们的 `SDVAETeacher.forward` 每 micro-step 都跑 `E+D`（83.7M）；换成 I1/I2 后**只用 E**（≈34M）；
- 更进一步：**离线缓存** `z_t`（mode，fp16）—— 126k 张 × 4096 值 × 2B = **≈1.0 GB**，随便存。
  KD 训练从此零 teacher 开销（现在 vram 27.9G / 20 img/s，比基线 22–24 img/s 慢，就是白花的）；
- ALIT 没做这个缓存（它每步都跑 frozen encoder），是我们的便宜可占。

### I7：小项

| 项 | ALIT | 我们 | 建议 |
|---|---|---|---|
| EMA | 有，eval 用 ema | 无 | 加（短 FT 里 EMA 是免费稳定器） |
| 目标随机性 | `.sample()` | `.mode()` | **保留 `.mode()`**：蒸馏要确定性目标，`.sample()` 会给 MSE 加噪声底 |
| 输入归一化 | 拼 teacher token 后 `F.normalize` | 原图 patches | 不抄（会引入 teacher encoder 做前端，改变"我们自己的编码器"叙事；要做就明写） |
| dynamic halting | 有（防 decoder 抄输入） | 无，但**我们的 decoder 只看 payload**（`decode` 的 memory = `[z_cls; z_s; 生长 token]`，无图像旁路） | 不用抄：我们没有"抄输入"的旁路可堵 |
| 量化器 | VQ 12 维×32 级 | 无（连续） | 不抄 |

### I8：训练 regime（最扎心的一条）

ALIT stage-1 = **200 epoch × 8GPU × bs32 ≈ 25M 样本**，且蒸馏是**从零开始的唯一目标**；
我们 = **2 epoch ≈ 1.6M 样本**，在 14 epoch 的 L1 收敛点上当**辅助项**。
差 14 倍的样本量 + 不同的收敛位置。如果 I1/I2 有效，正确做法是**从零开始就把 latent 目标当主目标**
（等于重跑一条训练曲线，成本≈一条 14ep 曲线），而不是在终态上贴补丁。
`kd1` 这种 arm 的定位应该改为"快速证否/证实"，不要指望它把 86.5 压到目标线。

---

## 4. 如果只做两件事

1. **P0-d**（20 min，不训练）—— 先量 `D(E(ours))` 的 rFID，决定 latent 路线的天花板；
2. **I1+I3(方案 3)+I4**：latent MSE 主目标 + 两段式 + 逐步递增权重，起一条 6–8 epoch 的训练
   （而不是 2 epoch 辅助项），用 `--loss_weights` 和现成的 `--kd_space latent` 跑。

I2 是真正的"ALIT 复刻"，但它改评测路径（decoder 归属变了），建议在 I1 看到 latent MSE 能降之后再做。

---

## 5. 对论文/定位的影响（写清楚，别埋）

- 我们的 rFID 86.5 与 ALIT 8.25 的差距，**主因是"我们把 decoder 之前的所有先验都从零学"**，
  而 ALIT 直接 **reuse 一个预训练 decoder**；同 rate 下这不是"tokenizer 设计"的差距，
  所以论文里不能把两者放进同一张"方法对比"表就完事，必须标注 decoder 来源；
- 一旦按 I2 做，我们的卖点从"完全自研的端到端 tokenizer"变成"**可变速率、连续 12 维 token 的
  latent 重压缩层** + 冻结强 decoder"——这仍然是有故事的（ALIT 只有离散且 token 数只能取 2 的幂，
  我们是连续 12 维 + 任意 8 档速率），但必须在文中如实交代。

---

## 6. 预登记预测（看到数据前写死，防止事后编故事）

| 实验 | 预测 |
|---|---|
| P0-d `rFID(real, D(E(ours)))` | **20–45**（若 >60，则我上面的归因错，latent 路线要重估） |
| I1（2 epoch 辅助项，latent MSE） | PSNR −0.2±0.3 dB，rFID **60–80**（比 kd1 好，但远不到 ≤30） |
| I2（6–8 epoch，latent 主目标，decoder 冻结） | rFID **8–25**（ALIT 同 rate 是 8.25；若 >40 说明 z 预测误差主导） |
| kd1（像素 L1，2 epoch） | rFID **75–95**，与 86.5 同噪声内（已在跑，17:24 截断） |
