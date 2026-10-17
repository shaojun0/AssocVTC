# ALIT 单卡复现笔记（2026-10-17, RTX 4080 32G / AutoDL）

## 0. 结论（实测）
- 配置 `alit_tiny` = 论文 Ours-S（enc/dec 各 8 层 × d512）= **52.8M 可训 + 72.1M 冻结 VQGAN = 125.0M 总参数**
- 256² / IN-100（126,689 训练图）/ stage-1（8 rollouts，8 个迭代都算 loss）
- **1.03–1.05 h / epoch**（bs16×accum16=有效 batch256, 0.478 s/iter, 7918 iter/epoch, GPU 88%, 275W, 20 GB 显存）
- 200 epoch（论文 stage-1 原配方）≈ **8.6 天**；30 epoch ≈ **31.6 h**
- 省法：`sample_grad_iters=2`（只随机 2 个迭代算梯度）→ 55 img/s（1.64×）→ 200 epoch ≈ 5.2 天
- 批量吞吐（bench_alit.py 实测，8 迭代全解码）：bs4 10.7 / bs8 20.9 / bs12 28.5 / **bs16 33.6** / bs20 34.3 / bs24 34.1 img/s
  - 峰值显存：bs8 8.9 / bs16 16.5 / bs24 24.1 GiB ⇒ 16G 卡用 bs8 可行，32G 卡 bs16–20 最优
- 有效算力 ≈ 33.6 img/s × 1.3 TFLOPs/img ≈ **44 TFLOPS**（~23% MFU）

## 1. 与官方代码的差异（必须知道的 4 处 patch）
1. `main_pretrain.py` / `main_full_finetuning.py`: `optim_factory.add_weight_decay` → `param_groups_weight_decay`
   （timm 1.0.30 删了旧 API；等价功能）
2. `base_tokenizers/taming/modules/diffusionmodules/model.py`：换回**官方 taming** 版本
   （官方仓库 vendored 的是 Mage 版：ResnetBlock shortcut 是 out→out、conv_out 1×1、无 bias；
    我们拿到的 VQGAN ckpt 是官方 taming 结构：in→out、conv_out 3×3、有 bias）
3. `base_tokenizers/taming/models/vqgan.py`：补 `quant_conv` / `post_quant_conv`（官方 taming 结构），
   `init_from_ckpt` 过滤 `loss.*` 键（官方 ckpt 带感知损失/判别器）
4. 数据 transform 加 `Normalize(0.5,0.5)`（→[-1,1]）：因为我们用的是官方 taming ImageNet VQGAN（按 [-1,1] 训练）；
   论文用的 Mage VQGAN 是 [0,1]。**这是替代 base tokenizer 带来的唯一实质性偏差**。
   另：`--save_freq` 新增（原版 40 epoch 存一次，长跑丢工作）

## 2. base tokenizer 替代
- 论文 = Mage 的 VQGAN（Google Drive，本机被墙）→ 改用 **HF `Nehc/vqgan_imagenet_f16_1024` 的 last.ckpt**
  （958 MB，官方 taming 结构、ImageNet f16、n_embed=1024，与论文 ddconfig 完全一致）
- 实测重建：PSNR 20.2 dB / 256² （弱于 Mage 版，粗看偏糊）⇒ **绝对 FID 不可与论文比，只比趋势与相对差**
- HF 镜像（hf-mirror.com）在本机只有 ~24 KB/s，ckpt 是本地下载后 scp 过来的（~4 MB/s）

## 3. 数据
- `/root/autodl-tmp/datasets/imagenet100`（100 类 × train，126,689 张，ImageFolder 类目录）
- `alit_in100/{train,val}` 是指向它的 symlink（官方 loader 读 `data_path/train`）
- ⚠️ 没有 IN-100 val（5000 张）；`eval_ladder.py` 用训练集**固定尾部子集**做监控，数字不能与论文 val 直接对齐

## 4. 命令
```bash
# 训练（已带自动续跑）
cd /root/autodl-tmp/alit && EPOCHS=30 ./run_s1_30ep.sh      # 输出 /root/autodl-tmp/alit_out/s1_tiny_in100
# 监控
tail -f /root/autodl-tmp/alit_out/s1_tiny_in100/train.log
# 逐 token 预算重建阶梯（图 + json）
python eval_ladder.py --ckpt <OUT>/checkpoint-last.pth --which ema --n 128 \
       --out /root/autodl-tmp/alit_out/ladder/manual
# 吞吐/显存 bench
python bench_alit.py alit_tiny
```
- 自动 ladder：`ladder_watch.sh`（后台轮询 checkpoint-*.pth 逐个评测，结果在 `/root/autodl-tmp/alit_out/ladder/`）

## 5. 与论文可比的对象
论文附录 IN-100 / stage1 / small(8层,512) 的 L1(×255) 阶梯：
| tokens | 32 | 64 | 96 | 128 | 160 | 192 | 224 | 256 |
|---|---|---|---|---|---|---|---|---|
| 论文 200ep | 71.53 | 49.18 | 35.11 | 26.97 | 20.31 | 15.38 | 10.84 | 8.77 |
（我们 base tokenizer 不同 + val 集不同，只能看"单调下降"和斜率形状）

## 6. 运行状态（2026-10-17 20:30 更新）
- **正在跑**：stage-1，30 epoch，2026-10-17 19:18 启动（20:28 做过一次断点续跑验证），
  ETA ≈ 2026-10-19 04:00（30 × 1.05 h）。
- 进度：epoch 0 已出 `checkpoint-0.pth`（1.42 GB）；nll_loss 6.96（init）→ 5.93（epoch 1），code_loss 0.61 → 0.53
- 自动评测：`ladder_watch.sh` 每 5 min 轮询，对 checkpoint-{0,5,10,15,20,25}.pth 各跑一次 32→256 token 阶梯
  → `/root/autodl-tmp/alit_out/ladder/checkpoint-N/{ladder.json, ladder_grid.png}`
- 预期可交付（按 1.05 h/epoch）：
  | epoch | 墙钟 | 用途 |
  |---|---|---|
  | 5  | ~5.3 h | 阶梯开始出现斜率（第一个有意义的点） |
  | 10 | ~10.5 h | 变长 RD 曲线可用 |
  | 20 | ~21 h | 接近论文趋势（论文是 200 ep） |
  | 30 | ~31.6 h | 本次调度终点（余弦退火完成） |
- 停/续：
  ```bash
  pkill -f run_s1_30ep.sh; pkill -f main_pretrain.py; pkill -f ladder_watch.sh   # 停
  cd /root/autodl-tmp/alit && EPOCHS=30 ./run_s1_30ep.sh                          # 续（自动从 checkpoint-last 恢复）
  ```
- 显存现状：训练 20.3 GB + 评测 ~9 GB（bs8），32 GB 卡够用；16 GB 卡需把训练 bs 降到 8。
