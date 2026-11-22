# 下一阶段计划：解码端 Refiner（post-filter）路线 + 冲 Q1 的完整实验设计

- 写定时间：2026-11-22（本会话结束前）
- 前置：`doc/2026-11-22/REPORT_k256x12_t8.md`（256 token×12 = 3072 values/图、8 步循环解码、24h 全流程结果）
- 机器状态：远端已 `shutdown -h now`，**下次会话先开机**再按 §6 校验资产

---

## 0. 一句话目标

在**不改循环解码架构**、rate 固定 3072 values/图（256²）的前提下，用一个**解码端 refiner（post-filter）**把满速率 rFID 从当前的 **86.50** 压到 **≤ 30**（ALIT-S 8.25 的 3.6 倍以内），并给出完整的 rate-distortion-perception 曲线与诚实的参数/预算口径 ⇒ **TMM / TCSVT 级别（Q1）投稿包**。

---

## 1. 核心判断（这是整个计划的立论，写作时要用）

1. **解码端容量不计入 rate**。率失真框架里只有码流（3072 values）受约束，解码器/后处理器的参数量是"免费"的 —— ALIT 用 173M 解码器 + GAN 拿到 8.25，我们 480.6M + refiner 并不违反同一约束。**但必须在论文里给出参数量表与同参数 baseline**，否则审稿人判 unfair。
2. **信息不免费**。refiner 只能（a）重新分配已有信息（去块/去模糊）、（b）幻觉合理纹理。所以它必然 **FID↓ / PSNR↓ / LPIPS↓**，是 tradeoff 不是免费提升。必须同时报 PSNR+SSIM+LPIPS+DISTS。
3. **refiner 的真正价值是"隔离对抗信号"**，不是白拿 FID。本会话已证明：对抗信号直接加在主解码器末步会把主干拉坏
   （GAN 首轮：rFID 86.50→153.02，且 step7→8 PSNR 21.90→21.60 出现**单调性反转**）。
   把对抗关进一个冻结主干的小 refiner 里，才能既保住 L1 的结构/保真度，又学到纹理分布。
4. **一个 refiner 服务 8 个速率点**（32→256 token 的 8 个逐步输出都过同一个 refiner）—— 这是与"单模型循环多速率"卖点同源的故事，也是与 ALIT 逐 rate 评测的结构性差异。

---

## 2. 分阶段计划（含时间/GPU 预算与"决策门"）

### P0-a（10 min，无训练，无下载）：把 86.5 的差距**分解**开

**目的**：先判断"我们的重建离 ALIT 差在内容还是差在纹理统计"，避免盲目上 refiner。

做法：对 GT 做可控退化，测 `FID(real, degrade(real))`：
- 高斯模糊 σ ∈ {0.5, 1.0, 2.0}（模拟过平滑/纹理缺失）
- JPEG q ∈ {10, 20, 30}（模拟压缩伪影）
- 8×8 块均值（模拟 DC-only / patch-DC 基线）

判读：
- 若 `FID(real, blur(real))` 在 σ=1 时就已 50–60 ⇒ 我们的 86.5 主要是"糊"，refiner 路线天花板高，直接进 P1。
- 若 blur 只带来 <15 FID ⇒ 差距在内容/结构错位，refiner 无用，应转向"加率 / 更强先验 / 换 token 化方式"（计划 B，§5）。

产出：`tools/fid_degrade_decomp.py` + 一张分解表（写进论文的分析章节，本身就是卖点）。

### P0-b（1–2h，无训练，需下载权重）：**post-filter 天花板实验**

**目的**：用**预训练**生成式后处理直接测 FID，得到"如果能训到多好"的上界。这是决定是否值得投入 P1 的**决策门**。

做法：`tools/postfilter_eval.py`
1. 载入 `/root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt`（L1 14ep），对 IN-100 val 5000 张跑 8 步累计解码；
2. 对重建施加预训练后处理（不训练、不下载除权重外的东西）：
   - 首推 **SD-Turbo / SD1.5 img2img，strength 0.15–0.25，256²**（1.5–4GB 权重；最强、最省事）
   - 备选 **Real-ESRGAN x1/x4→resize**（依赖 basicsr，安装风险较高）
3. 全程内存内算 rFID（复用 `fast_rfid.py` 的 FID 累加器，避免落盘 PNG）；
4. 输出**每一步**的 {PSNR, rFID, LPIPS} 三联表。

判读（决策门）：
| 结果 | 决策 |
|---|---|
| rFID ≤ 40 | **走 P1**：refiner 路线天花板足够，训一个 6–10M 的 refiner 去逼近 |
| 40 < rFID ≤ 65 | 谨慎走 P1，但同时准备"加率/强先验"的对照实验 |
| rFID > 65 | **不走 P1**：说明瓶颈在信息量，转计划 B |

风险：MUSA 上跑 diffusers 可能不通（attention/bf16）。缓解：先在 1 张图上做 3 分钟 smoke；不通则退到 Real-ESRGAN 或自训 refiner（跳过天花板，承担不确定性）。

### P1（4–6h，4 卡）：训练 refiner 头

**架构**（新文件 `refiner_head.py`）：
```
base(3×256×256) ──► [ResBlock × k + 2× downsample ×2 + upsample ×2 + skip] ──► res(3×256×256)
refined = clamp(base + s·res, 0, 1)      # s 可学习标量, 初始 0.1 ⇒ 起点等价于 base
```
- 规模：6–10M 参数（相对主干 480.6M 约 1.5–2%，论文要给这张表）；
- **纯图像进/图像出**：同一个 refiner 直接作用于 8 个逐步输出，天然支持 8 个速率点；
- 不引入任何 GT 侧信息（防"作弊"质疑）。

**训练配置**（新脚本 `run_refiner_ft.sh`，复用本会话已验证的组件）：
| 项 | 取值 | 说明 |
|---|---|---|
| 主干 | **冻结**（或 lr 1e-6，仅微调） | 保住 L1 结构与 PSNR，避免重演 GAN 首轮的 &minus;20 面板损失 |
| refiner lr | 2e-4 → 1e-5 cosine | — |
| 损失 | `L = 1.0·L1(ref,gt) + 0.1·VGG_perc(ref,gt) + λ_adv·L_hinge_adv` | L1 项是**颜色/结构锚**，必不可少 |
| 对抗 | 复用 `patchgan.py`：70×70、ndf=64、hinge、**λ 自适应** `λ = 0.8·‖∂L_rec/∂w‖/‖∂L_adv/∂w‖`（w = refiner 末端卷积权重） | 本会话已验证可用 |
| D lr | **2e-5**（不是首轮的 1e-4） | D 太强是首轮失败主因之一 |
| 步数采样 | 每个 batch **随机抽 1 个速率点**（step ∈ 1..8） | 1× 成本覆盖全速率；也避免 8× 开销 |
| 对抗作用位置 | refiner 的**输出**（不是主干末步） | 本会话 §4.7 的核心教训 |
| 预算 | 4–6h | — |

**验收**：满速率 rFID **≤ 60** 且 PSNR ≥ 21.5 dB（允许 −1.2 dB 交换）；中间速率点不得出现单调性反转（step7→8 必须继续变好）。

### P2（可选，1–4 天）：主干长训练

现况：14 ep（13.7h）→ rFID 86.50，且快照曲线 ep8 118.88 → ep12 90.93 → ep14 86.50 **仍未收敛**。
- 100 ep ≈ 98h（4 天）⇒ 预期 rFID 落到 55–70 区间（外推，需实测确认）。
- 建议与 P1 联合：先训主干 60–100 ep，再训 refiner；或主干与 refiner 交替（冻结/解冻循环）。
- 若时间不够，退而求其次：**30 ep（2 天）** + refiner。

### P3（半 1 天）：指标体系补齐

现在 `fast_rfid.py` 只出 PSNR / SSIM / L1 / rFID。Q1 必需 **LPIPS + DISTS**：
- LPIPS：`torchvision` 已有 AlexNet/VGG 权重（注意走 full build 的 PYTHONPATH），自实现 10 行；
- DISTS：`pip install dists-pytorch`（需外网；离线则用 LPIPS + FID 代替并在文中说明）；
- 输出成**rate-distortion-perception 三联图**（bpp vs PSNR / LPIPS / rFID），这是该领域现在的标准图表。

### P4（1–2 天）：消融与写作

必须有的表：
1. **速率-性能主表**：8 个速率点 × {PSNR, SSIM, LPIPS, rFID} vs ALIT-S / ALIT-B（同 256²，3072 values 对齐）；
2. **参数量与预算表**：主干 480.6M + refiner X M，训练 epoch 数、GPU 小时数、ALIT 对照口径；
3. **组件消融**：refiner 有无 / 对抗有无 / 感知权重 / D lr / 对抗作用步（末步单点 vs 全 8 步 —— 用本会话的负结果做正向证据）；
4. **逐步解码分析**：8 步的 PSNR+rFID 曲线、饱和点、单调性；
5. **P0-a 的 FID 分解表**（糊 vs 伪影 vs 内容）。

---

## 3. 与 ALIT 的差距当前在哪（本会话已确证，写作时直接引用）

| 项 | 我们（L1 14ep） | ALIT-S | 倍数 |
|---|---|---|---|
| 满速率 rFID（3072 values, 256²） | **86.50** | 8.25 | 10.5× |
| 32 token rFID | 178.07 | 22.57 | 7.9× |
| PSNR（满速率×5000 张） | 22.67 dB（patchDC +4.04） | — | — |

已排除的杠杆（**别再重复花时间**）：
- 瓶颈宽度：192→448 只换来全曲线 +1 dB，且把与 T=1 的差距从 3.38 拉到 4.17 dB；
- 分辨率口径：已对齐 256²，rFID 从 448² 的 ~102 降到 86.5；
- 容量：480.6M（ALIT 的 2.8×）但 rFID 差 10×；
- **事后加感知损失**：VGG16 特征 L1（w=0.15, 3h）让 8 点**全线变差**（86.50→95.53）；
- **事后整模型加对抗**：PatchGAN 1.5h 让满速率 **86.50→153.02**，并破坏逐步单调性。

⇒ 剩下的杠杆只有：**解码端 refiner（P1）**、**更长训练（P2）**、**更强先验/codebook（计划 B）**。

---

## 4. 现有资产（下次会话直接用，避免重跑）

### 远端 `/root/AssocVTC_tok448/`（非 git 工作副本，已含全部改动）
```
train_linattn.py        # 含 --perc_weight/--perc_layer/--perc_resize、
                        #    --gan_weight/--gan_adaptive/--gan_ndf/--gan_layers/
                        #    --gan_d_lr/--gan_start_frac/--gan_ramp_frac
                        #    + _init_bottleneck_warm_start 修复（源 ckpt 已含瓶颈时跳过 SVD）
model_v2_linattn.py     # 循环解码器（T=8 cumulative），勿改架构
perc_loss.py            # VGGPerceptualLoss(layers=3,8,15, resize)
patchgan.py             # NLayerDiscriminator(70×70) + PatchGANTrainer
                        #   （hinge / D 不包 DDP / 手动 allreduce / grad_ratio 用叶子参数 / lam_at ramp）
rfid_loss.py            # BatchRFIDLoss（可微 Fréchet Gram）
fast_rfid.py            # 内存内 8 点 rFID 评测（--ckpt --out --tag --img_size --batch_size --amp）
data_linattn.py metrics_linattn.py
```

### 远端 `/root/autodl-tmp/tok256x12_t8/`
```
s2/ckpt_best.pt                  # ★ L1 14ep 最优（22.61 dB val / rFID 86.50）—— P1 的起点
s2/train_log.jsonl s2/eval_val.json s2/samples/epoch0{01..14}_val.png
snaps/ep4.pt ep8.pt ep12.pt      # 训练进程快照（rFID: ep8 118.88 / ep12 90.93）
perc1/s2/ckpt_best.pt            # 感知损失臂（负结果，备查）
gan1/s2/ckpt_last.pt             # PatchGAN 臂（负结果，备查）
gan1/snaps/gan_1140.pt
t8_l1_ep14_fast.json t8_l1_ep8_fast.json t8_l1_ep12_fast.json perc1_fast.json gan1/gan1_fast.json
```

### 启动脚本 `/root/autodl-tmp/`
```
run_k256_d12.sh     # 阶段1(从零 L1 14ep) + 阶段2(rFID loss，因 bug 报废，含 eval)
run_perc_ft.sh      # 感知微调（SRC/TAG/PW/BUDGET/LR/RFID 环境变量）
run_gan_ft.sh       # PatchGAN 微调（SRC/TAG/GW/BUDGET/LR/DLR；GW 是**梯度比**语义）
gan_smoke.sh        # GAN 冒烟（18 步，验证 D/G 与 allreduce）
ensure_run.sh snap_ckpt.sh
```

### 本地 `/home/linaro/dsh/tok448_work/`
报告、全部消融 JSON、脚本副本、`artifacts/k256x12_t8/`、`analyze_t8.py`（生成 markdown 表 + ALIT 插值对照）。

### 环境（MUSA，4× MTT S4000 48GB）
```bash
cd /root/AssocVTC_tok448
export PYTHONPATH=/home/torchvision/build/lib.linux-x86_64-cpython-310   # 预训练权重必需
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 DDP_BACKEND=mccl DDP_IMPL=patched \
  torchrun --nproc_per_node=4 --master_port=29703 train_linattn.py ...
```

---

## 5. 风险与备选（计划 B）

| 风险 | 概率 | 缓解 / 备选 |
|---|---|---|
| P0-b 天花板不够（post-filter 后 FID > 65） | 中 | 转计划 B：加率（448² / 更多 token）或引入 VQ codebook + 更强先验；把"信息瓶颈"作为论文的**正面结论**（给出每个速率的信息上界分析） |
| MUSA 跑不动 diffusers | 中 | 退到 Real-ESRGAN；再不行就跳 P0-b 直接训 refiner（承担不确定性） |
| refiner 训练出现模式坍塌 / 颜色漂移 | 中 | L1 锚权重加大、D lr 再降、每 200 步存样本图人工看 |
| 4 卡机器被别人占用 / 磁盘满 | 低 | 每 epoch 只留 `ckpt_last`+`ckpt_best`；`snaps/` 用完即删（每份 1.9G） |
| 时间不够做 P2（100 ep） | 高 | 用 30 ep + refiner 出主表，在 limitations 里说明预算 vs ALIT 的差异 |

---

## 6. 下次会话的开机检查清单（5 分钟）

```bash
ssh -p 17677 -o StrictHostKeyChecking=no -o ConnectTimeout=25 root@connect.sha1.seetacloud.com
date; df -h /root/autodl-tmp | tail -1
ls -la /root/autodl-tmp/tok256x12_t8/s2/ckpt_best.pt
ls /root/autodl-tmp/tok256x12_t8/snaps/ /root/autodl-tmp/tok256x12_t8/perc1 /root/autodl-tmp/tok256x12_t8/gan1
ls /root/autodl-tmp/tok256x12_t8/*.json
pgrep -f "train_linattn.py" | wc -l        # 应为 0
python3 -c "import torch, torch_musa; print(torch.musa.is_available())"
# 快速自检：一次 20 步 smoke（不落盘）
bash /root/autodl-tmp/gan_smoke.sh
```
注意：**实例关机后 `/root/autodl-tmp` 内容通常保留，但务必先确认 ckpt 还在**；若丢失，用 `run_k256_d12.sh` 重跑阶段 1（13.7h）。

---

## 7. MUSA / 本实验线的踩坑清单（务必先读，能省数小时）

1. **`Tensor.cuda()` 不接受位置参数**（如 `.cuda(0)`）⇒ 一律用 `.to("cuda")` / `.to(_DEVICE)`。
2. **`torch.autograd.grad(loss, 非叶子张量)` 在 MUSA 上不可用**（报 "differentiated Tensor not used in graph"）⇒ 对**叶子参数**求导（本会话用解码器末端 `pixel_head` 权重，等价 VQGAN 的 `get_last_layer()`）。
3. **判别器不要包 DDP**（会触发双 backward 语义问题）⇒ D 保持裸模块 + 梯度手动 `dist.all_reduce` + 初始化 `dist.broadcast`。
4. **`pgrep -f "<名字>"` 会匹配自己的 ssh 命令行** ⇒ 判活用标志文件，或用 `foo[.]py` 的方括号转义；本会话两次因为这个"假阳性/自杀"。
5. **不要 scp 覆盖正在执行的脚本**（bash 读偏移会错乱，导致训练后的自动评估不触发）⇒ 上传成 `.new` 再 `mv`。
6. **SSH 中转会限流**（`kex_exchange_identification: Connection closed`）⇒ 间隔 ≥60s、最多重试 3 次、巡检间隔 ≥10min。
7. **bf16 下 `torch.linalg.svd` 未实现** ⇒ Fréchet 计算包 `torch.autocast("cuda", enabled=False)`。
8. 8 步 micro-32 会 OOM ⇒ 本配置用 micro 8 × accum 4（等效 128）。
9. **远端时钟才是权威**（本机与其有数十分钟偏移），ETA/截止时间以远端 `date` 为准。
10. `fast_rfid.py` 5000 张 8 点约 5–14 min，跑之前确认没有训练在抢卡。

---

## 8. 验收标准（什么叫这一步成功）

| 阶段 | 通过条件 |
|---|---|
| P0-a | 得到清晰的 FID 分解表，能回答"糊 vs 伪影 vs 内容"各占多少 |
| P0-b | 拿到 post-filter 天花板数字，并据此做出走/不走 P1 的明确决策 |
| P1 | 满速率 rFID ≤ 60（理想 ≤ 30），PSNR ≥ 21.5 dB，逐步曲线 **无单调性反转** |
| 投稿包 | 8 点 × 4 指标主表 + 参数量/预算表 + 5 组消融 + RDP 三联图 + 诚实的 limitations |

---

## 9. 论文定位与审稿人预答

- **定位**：TMM / TCSVT（Q1）。卖点组合 = 单一循环解码器的**多速率** token 化（8 个操作点一次前向）+ **解码端 refiner** 把感知质量补齐 + 系统性的边界分析。
- **强卖点（本会话已挣到）**：单变量消融把"杠杆在哪"画清楚了 —— 容量不是杠杆、瓶颈宽度不是杠杆、**损失设计与对抗信号的作用位置才是**；并给出了两个可复现的负结果（感知损失 −9 FID、末步单点对抗 +66 FID 且破坏单调性）。
- 预设三个审稿问题与答法：
  1. **"rFID 差 ALIT 10×，凭什么发？"** → 给 §3 的口径分解（预算 14 vs 400 ep、无 codebook、无对抗）+ P1/P2 修正后的数字；把"同等预算下的 ablate"作为主表。
  2. **"refiner 算不算作弊？"** → 率失真框架下解码端免费（引 ALIT 自身解码器规模），给参数量表与"同参数量 refiner 用 L1 训"的对照（证明收益来自对抗/感知目标而非纯粹容量）。
  3. **"多速率是单模型吗？"** → 是；给同一 ckpt 的 8 点曲线（这是与逐 rate 训练的结构性差异）。
