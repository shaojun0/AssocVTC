# REPORT — 官方 ALIT 单卡复现：实测吞吐、兼容性 patch、30 epoch 起步后停机

> 日期：**2026-10-17**
> 上游：**ALIT**（Adaptive Length Image Tokenization via Recurrent Allocation，[arXiv:2411.02393](https://arxiv.org/abs/2411.02393)）
> 代码：[`ShivamDuggal4/adaptive-length-tokenizer`](https://github.com/ShivamDuggal4/adaptive-length-tokenizer)，**commit `8c0b4ec`**（2025-02-11）
> 机器：AutoDL 单卡 **RTX 4080 SUPER 32 GB**（本实例 `ssh -p 34340`，2026-10-17 21:1x 已关机），128 核 / 503 GB / 178 GB 数据盘
> 数据：**ImageNet-100 train 126,689 张**（本实例**没有** IN-100 val 5,000）
> 产物：[`artifacts/alit_official_repro/`](artifacts/alit_official_repro/)（日志 / ladder / 脚本 / patches / 环境快照）
> 关联：[`ANALYSIS_alit_vs_assocvtc.md`](../2026-10-16/ANALYSIS_alit_vs_assocvtc.md)（机制对照）、[`REPORT_alit_adaptive.md`](../2026-10-16/REPORT_alit_adaptive.md)（12 项移植实验）

---

## 1. 一句话结论

**官方 ALIT 在本机跑通了**：`alit_tiny`（= 论文 Ours-S，enc/dec 各 8 层 × d512，**52.8M 可训**）在 256² / IN-100 / stage-1（8 个 rollout 全部监督）下实测
**0.478 s/iter、7918 iter/epoch ⇒ 1.05 h/epoch**（bs16×accum16=有效 256，GPU 91%，275 W，20.3 GB，≈44 TFLOPS 有效算力）。

按此换算：**论文 stage-1 原配方 200 epoch = 单卡 8.6 天**；本次按 **30 epoch（31.6 h）** 起步，实际跑到 **epoch 0 完成 + epoch 1 进行中**后按用户要求关机（epoch 1 的进度未落盘，见 §6）。

> 这条线的定位：**给移植线补一个官方参考点**，不改变主线判决。移植线的结论（自适应 K 被否证、后步加权是有效杠杆）见 2026-10-16 的报告。

---

## 2. 这次要回答什么

主线（`model_v2.py` / `model_v2_linattn.py`）已经把 ALIT 的机制抄了一遍并跑了 12 项自适应实验，但一直缺两件东西：

1. **官方参考点**：ALIT 声称的 "32→256 token 单调阶梯" 在官方代码里到底长什么样、斜率多大、需要多少 epoch 才出现——移植线的 PSNR 阶梯没有可比对象；
2. **预算**：后续要不要投 200 epoch、要不要上多卡，需要一个实测的 per-epoch 成本，而不是 FLOPs 估算。

本次把两件都做了（第 2 件已完成，第 1 件因关机只拿到 epoch 0 的平线，见 §5）。

---

## 3. 配置与实测吞吐

### 3.1 配置

| 项 | 值 |
|---|---|
| 模型 | `alit_tiny` = 论文 Ours-S（`__init__.py` 里 `alit_small` 其实是 8 层 × **1024** ≈ 215M，**做 60M 要用 `alit_tiny`**） |
| 参数量 | 可训 **52.8M**；含冻结 VQGAN(72.1M) 共 125.0M |
| 输入 | 256×256，VQGAN f16 → 16×16 = 256 个 2D token |
| rollout | `max_rollout_iters=8`，`num_init_latent_tokens=32` ⇒ 32/64/…/256 |
| 量化 | `--quantize_latent --factorize_latent`（论文 Ours-S 的设置） |
| 数据 | IN-100 train 126,689；`RandomResizedCrop(256, 0.8~1.0)` + HFlip + **Normalize(0.5)**（见 §4） |
| 优化 | bs 16 × accum 16 = 有效 256；AdamW，blr 1.5e-4，wd 0.05，warmup 3 epoch，30 epoch 余弦 |
| 显存 | 峰值 20.3 GB（训练）→ 32 GB 卡可用 bs 16~20 |

### 3.2 吞吐（`artifacts/alit_official_repro/bench_alit_tiny.txt`，8 迭代全监督）

| bs | ms/step | img/s | 峰值显存 | 折算 epoch |
|---|---|---|---|---|
| 4 | 353 | 11.3 | 5.1 GiB | 3.1 h |
| 8 | 377 | 21.2 | 8.9 GiB | 1.66 h |
| 12 | 410 | 29.3 | 12.7 GiB | 1.20 h |
| **16** | 477 | **33.6** | 16.5 GiB | **1.05 h** |
| 20 | 583 | 34.3 | 20.3 GiB | 1.03 h |
| 24 | 704 | 34.1 | 24.1 GiB | 1.03 h |
| 16 + 只采 2 个迭代算梯度 | 311 | **51.4** | 10.9 GiB | 0.68 h |

实测训练进程与 bench 一致（0.478 s/iter，7918 iter/epoch）。

**换算（IN-100 126,689 张）**

| 目标 | 32 GB（bs16） | 16 GB（bs8） |
|---|---|---|
| 10 epoch | 10.5 h | 16.6 h |
| 30 epoch（本次） | **31.6 h** | 50 h |
| 200 epoch（论文 stage-1） | **8.6 天** | 13.8 天 |
| 200 epoch + `sample_grad_iters=2` | 5.7 天 | 9.2 天 |

> 结论：**单卡复现论文 stage-1 是"周"级**，不是"天"级；想跑完 200 epoch 建议多卡或直接砍到 30~50 epoch。`sample_grad_iters` 省法 1.5~1.6×，但它改变训练分布（论文默认 8 个迭代全监督），对比实验时不要混用。

---

## 4. 与论文的可比性边界（必须先看）

1. **base tokenizer 是替代品**（最关键）：论文用 Mage 的 VQGAN（Google Drive，本机被墙），本次改用
   HF `Nehc/vqgan_imagenet_f16_1024` 的 `ckpts/last.ckpt`（958 MB，官方 taming 结构、ImageNet f16、`n_embed=1024`，ddconfig 与论文 yaml 完全一致）。
   实测重建 **PSNR 20.2 dB / 256²，肉眼偏糊**（`vqgan_recon_check.png`：上=原图、中=[0,1] 输入、下=[-1,1] 输入）。
   ⇒ **绝对 FID 不可与论文 Table 1 对比**；只能比"阶梯形状/斜率/单调性"。
2. **数据变换改为 [-1,1]**：官方 taming ckpt 按 [-1,1] 训练，ALIT 代码默认 [0,1]（那是给 Mage 版的）。我们加了 `Normalize(0.5)`，
   这是本次唯一实质偏离论文代码的地方。
3. **没有 IN-100 val**：`eval_ladder.py` 用 train 的固定尾部 128 张做监控，数字不能与论文 val 对齐。
4. **只跑 stage-1**：论文的 FID 数字来自 stage-2（像素 + GAN + 解冻 VQGAN），本次未触及。
5. **`alit_tiny` 与论文 "Ours-S" 的对应**：论文附录表写 small = depth 8 / dim 512，与代码 `alit_tiny` 一致（`alit_small` 是 dim 1024，别选错）。

---

## 5. 管线验证（关机前已确认能跑通的部分）

| 环节 | 状态 | 证据 |
|---|---|---|
| 官方代码 + 依赖 | ✅ | torch 2.12.1+cu130 / timm 1.0.30 / PL 2.6.6（`env.txt`） |
| 5 处兼容 patch | ✅ | `patches.diff` + `taming_model/`（§5.1） |
| VQGAN ckpt 加载/重建 | ✅ | `vqgan_recon_check.png` | 
| stage-1 训练 | ✅ 稳定 | nll_loss 6.96(init)→**6.21**(epoch0 avg)，code_loss 0.61→0.59；6.96≈ln(1024) 是预期初值 |
| checkpoint | ✅ 1.33 GB/个 | 每 epoch 落 `checkpoint-last.pth`，每 5 epoch 额外落 `checkpoint-N.pth` |
| 断点续跑 | ✅ 实测 | 20:28 手动 kill 后自动从 `checkpoint-last`(epoch0) 恢复，lr 正确接上 |
| 逐档阶梯评测 | ✅ 打通 | `eval_ladder.py`：8 档 32→256，出 json + 拼图 |
| epoch-0 阶梯 | ⚠️ 平线（**预期内**） | 62.6±0.3 /255：warmup 阶段 lr 只到 5e-5、EMA 还停在 init，本来就该是平的；**这条数据不能用来说"阶梯不存在"** |
| 自动 ladder watcher | ✅ 已部署 | `ladder_watch.sh` 每 5 min 轮询，对 epoch 0/5/10/15/20/25 自动评测（随机关机未产出 5 及以后） |

### 5.1 必须打的 5 处兼容 patch（否则跑不起来）

| # | 现象 | 修法 | 文件 |
|---|---|---|---|
| 1 | `AttributeError: module 'timm.optim.optim_factory' has no attribute 'add_weight_decay'` | 换 `param_groups_weight_decay`（等价） | `main_pretrain.py` / `main_full_finetuning.py` |
| 2 | 官方仓库 vendored 的 taming 是 **Mage 变体**（shortcut out→out、conv_out 1×1、无 bias），与拿到的**官方 taming** ckpt 形状不符 | 换回官方 taming `model.py`（整文件替换，见 `taming_model/`） | `base_tokenizers/.../diffusionmodules/model.py` |
| 3 | 官方 taming VQModel 有 `quant_conv`/`post_quant_conv`，ALIT 的 Mage 变体没有；ckpt 还带 `loss.*` | 补两个 1×1 conv（初始化和 encode/decode 都走它们）；加载时过滤 `loss.*` | `base_tokenizers/taming/models/vqgan.py` |
| 4 | torch ≥ 2.6 `torch.load` 默认 `weights_only=True`，**ckpt 恢复直接炸**（含 `argparse.Namespace`） | 所有 `torch.load` 补 `weights_only=False` | `utils/misc.py` / `evaluate_rfid.py` / 两个 taming 文件 |
| 5 | 原版只在 `epoch % 40 == 0` 存 ckpt，长跑丢工作 | 加 `--save_freq`；`checkpoint-last` 本来就每 epoch 落（保留） | `main_pretrain.py` |

另：数据 transform 加 `Normalize(0.5,0.5)`（§4.2）。

**复现步骤**（新机器上，约 40 min 到能开跑）：
```bash
git clone https://github.com/ShivamDuggal4/adaptive-length-tokenizer.git alit && cd alit
git checkout 8c0b4ec                                  # 本文 patch 的基线
git apply /path/to/artifacts/alit_official_repro/patches.diff
cp /path/to/artifacts/alit_official_repro/taming_model/model.py \
   base_tokenizers/taming/modules/diffusionmodules/model.py
mkdir -p base_tokenizers/pretrained_models && \
  # 958MB，HF 直连或本地下载后 scp（本机 hf-mirror 仅 24KB/s）
  wget -O base_tokenizers/pretrained_models/vqgan.ckpt \
  https://huggingface.co/Nehc/vqgan_imagenet_f16_1024/resolve/main/ckpts/last.ckpt
pip install omegaconf einops timm pytorch-lightning wandb
# 数据：ImageFolder 布局，data_path/train/<class>/*.jpg，126,689 张
bash scripts/run_s1_30ep.sh
```

---

## 6. 停机状态与续跑

- 启动：**2026-10-17 19:18:43**；关机：**20:40 左右**（用户指令）
- 完成：**epoch 0**（`checkpoint-0.pth` / `checkpoint-last.pth`，各 1.33 GB，在实例数据盘 `/root/autodl-tmp/alit_out/s1_tiny_in100/`）
- 丢失：epoch 1 的约 10 min 进度（mid-epoch 不落盘，符合上游设计）
- 数据盘 `autodl-tmp` 跨关机保留 ⇒ 下次开机可原地续跑：

```bash
# 续跑（自动从 checkpoint-last 恢复；EPOCHS 保持 30 才不破坏余弦调度）
cd /root/autodl-tmp/alit && EPOCHS=30 ./run_s1_30ep.sh
# 看训练 / 手动阶梯
tail -f /root/autodl-tmp/alit_out/s1_tiny_in100/train.log
PYTHONPATH=. python eval_ladder.py --ckpt /root/autodl-tmp/alit_out/s1_tiny_in100/checkpoint-last.pth \
  --which ema --n 128 --bs 8 --out /root/autodl-tmp/alit_out/ladder/manual
# 停
pkill -f run_s1_30ep.sh; pkill -f main_pretrain.py; pkill -f ladder_watch.sh
```

---

## 7. 下一步建议（按性价比排序）

1. **先续跑到 epoch 10（约 10.5 h）再判读**：warmup 3 epoch + EMA，5 epoch 才开始出斜率，10 epoch 的 32→256 阶梯才够看；
2. **不要为了 FID 跑 200 epoch**：① base tokenizer 是替代品，绝对数不可比；② 要 FID 必须做 stage-2（GAN + 解冻 VQGAN），那是另一个量级；③ 我们要的"阶梯形状/斜率"在 30 epoch 内就能定；
3. **要更快的参考点**：把 `max_rollout_iters` 降到 4~5（32→128 token）+ 128²，成本降到 ~1/4，先确认"逐轮加记忆是否产生更陡的阶梯"，再决定投不投；
4. **移植线的对照实验**（真正影响主线的那一步）：用这台机器把 `b3_grow`（已有逐轮加记忆）与官方阶梯放在**同一 base tokenizer、同一数据、同一像素评测口径**下比，而不是比 PSNR 的绝对值。

---

## 8. 附：产物清单（`artifacts/alit_official_repro/`）

| 文件 | 内容 |
|---|---|
| `REPRO_NOTES.md` | 服务器上原样的复现笔记（含命令、patch 说明、论文附录基线） |
| `patches.diff` | 5 处 patch 的 unified diff（基于上游 `8c0b4ec`） |
| `taming_model/model.py` | 官方 taming 版 `model.py`（**整文件替换**用），`.alit_vendored_bak` 是上游 Mage 版备份 |
| `bench_alit_tiny.txt` | 8 档 batch/梯度的吞吐与显存实测 |
| `s1_progress.json` | 训练配置 + 实测吞吐 + 已完成 epoch + ckpt 清单（机器可读） |
| `train_s1_full.txt` | stage-1 完整训练日志（2,300 行级，8 个迭代的 nll/code loss 逐步） |
| `driver_s1.txt` | 自动续跑 driver 日志（含两次启动与断点恢复） |
| `env.txt` | GPU/驱动/torch/timm/PL 版本、CPU、内存 |
| `ladder_epoch0.{json,png}` | epoch-0 的 32→256 阶梯（平线，**启动态基线**，勿当结论） |
| `vqgan_recon_check.png` | 替代 base tokenizer 的重建检查（上=原图/中=[0,1]/下=[-1,1]） |
| `scripts/run_s1_30ep.sh` | 训练 driver（自动续跑、每 epoch 存 ckpt） |
| `scripts/bench_alit.py` | 吞吐/显存 bench |
| `scripts/eval_ladder.py` | 32→256 逐档 L1/PSNR + 拼图（分批，避免 OOM） |
| `scripts/ladder_watch.sh` | 后台轮询自动评测 |

**未入库**：checkpoint（1.33 GB/个）与 958 MB VQGAN 权重——都在实例数据盘上，按需从 HF/续跑重建。
