# REPORT — MultiResSR 多分辨率六源实验「解码器 ×10」重跑 + multi 代码通用化（2026-10-14）

> 实例：AutoDL `ssh -p 15990 root@connect.nma1.seetacloud.com`
> （`autodl-container-027f4cad3d-42a24ba7`，os `bclinux` 4.19.90，112 vCPU / 1007 GB RAM）
> 硬件：**2× NVIDIA A800 80GB PCIe**（单卡可见 81840 MiB）
> 软件：Python 3.12.3（miniconda3 base）/ **torch 2.12.1+cu130** / transformers 4.57.6 /
> accelerate 1.15.0 / pyarrow 26.0.0 / numpy 2.4.6 / Pillow 12.2.0
> 代码工作树：本地 `SR-Diffusion-v3`（含本轮新增/重写）→ 远端 `/root/autodl-tmp/code/SR-Diffusion-v3/`
>
> 上一轮（被回收的机器）：`2× RTX 4080 SUPER 32G`，1 epoch / 40,242 步 / 12h33m /
> 峰值 23.52 GiB，见 `doc/2026-09-26/RUN_multires_natural.md`。

---

## 0. 一句话结论

1. **解码器参数量已按 10 倍落地**：`depth 2→2`、`width 1024→3072`、`heads 8→24`、
   `mlp_ratio 4.0→5.12` ⇒ 解码器 **140.9 M → 1409.3 M（×10.001）**，全模型 445.3 M → 1713.6 M。
   参数量是**算出来的**（`tools/count_params.py` 用真实模块统计），不是估的。
2. **multi 代码已重写为设备无关 + 框架驱动**：平台差异收敛进 `srdiff_runtime.py`
   （CUDA / 海光 DCU(ROCm) / 昇腾 NPU / XPU / CPU），训练循环/DDP/AMP/梯度累积/裁剪/
   调度器/续训全部交给 **HuggingFace Accelerate**（Trainer 内部同款），
   手写 `DistributedDataParallel` / `init_process_group("nccl")` / `autocast("cuda")` /
   `torch.cuda.*` / lr 曲线**全部删除**。
3. **新机器上已实跑验证**：2 卡 DDP + bf16 + 双梯度检查点，×10 解码器
   单卡峰值 **58.9 GiB / 80 GiB**，混合桶 **≈2.7 s/step**。
   正式 run 按 **2 epoch / 77,402 步/卡**（索引 259,641 ⇒ 38,701 步/epoch）启动，
   跑到 **step 4240（loss 0.869）** 后按用户要求停止并关机（内网 DCU 接棒，见 §5.5）。

---

## 1. 用户 2026-10-14 的两条要求 → 本轮的改动

| # | 要求 | 旧代码的问题 | 本轮改动 |
|---|---|---|---|
| 1 | 「要支持 DCU、NPU 等其它机器」 | `train_multi.py` 里 7 处 CUDA 硬编码：`torch.cuda.set_device` / `torch.device("cuda")` / `dist.init_process_group("nccl")` / `torch.cuda.get_device_name` / `torch.autocast("cuda")` / `torch.cuda.max_memory_allocated`×2；数据/模型/输出路径全硬编码 `/root/autodl-tmp` | 新增 `srdiff_runtime.py`：设备串/分布式后端/自动混合精度/显存统计/注意力实现全部按平台解析；`data_multi.py` 的六源改为**相对 `<data_root>`**；默认路径可用 `SRDIFF_DATA_ROOT` / `SRDIFF_DINO_DIR` / `SRDIFF_OUTPUT_DIR` 覆盖；`tools/prep_data.py` 一条命令在任意机器重建数据布局 |
| 2 | 「自己造轮子手写循环」 | 手写 DDP 包装、rank/world 判定、`dist.barrier`、autocast、`clip_grad_norm_`、手写 warmup+cosine `lr_scale`、手写 ckpt 存取 | 训练循环改由 **Accelerate** 驱动：`Accelerator(accumulate/backward/clip_grad_norm_/prepare)` + `get_cosine_schedule_with_warmup`（分组基础 lr 由 AdamW `param_groups` 承载）；`accelerate launch` 与 `torchrun` 都能起 |
| 3 | 「解码器参数量整体提升 10 倍」 | 解码器规模只有 `depth/heads/mlp_ratio` 三个旋钮，且无法验证实际倍数 | `MultiResSR` 新增 `decoder_width`（= `OutputQueryDecoder.stack_dim`，`model_v2` 早已支持、multi 未接）；新增 `tools/count_params.py` 精确计数；本轮取 ×10.001 配置 |

### 1.1 删掉的手写代码（逐项对照）

| 旧代码（2026-09-26） | 现在 |
|---|---|
| `torch.cuda.set_device(local_rank)` + `dev = torch.device("cuda", local_rank)` | `Runtime.device()` / `accelerator.device` |
| `dist.init_process_group("nccl")` | `Accelerator(...)` 自动选 `nccl`/`hccl`/`ccl`/`gloo` |
| `DDP(model, device_ids=..., find_unused_parameters=False, gradient_as_bucket_view=True)` | `accelerator.prepare(model, opt, sched)`（`DistributedDataParallelKwargs` 传参） |
| `torch.autocast("cuda", dtype=...)` | `Accelerator(mixed_precision="bf16")` 包装 `model.forward` |
| `dist.barrier()` | `accelerator.wait_for_everyone()` |
| `torch.nn.utils.clip_grad_norm_(...)` | `accelerator.clip_grad_norm_()`（自动 unscale + 跨卡） |
| 手写 `lr_scale(step)`（warmup+cosine） | `transformers.get_cosine_schedule_with_warmup` |
| `torch.cuda.max_memory_allocated(dev)` | `Runtime.peak_memory_gib()`（NPU 走 `torch.npu`） |
| `--reset_step_on_resume` 手工归零 | 语义保留，调度器按 `start_step` 推进或归零 |

仍然留在本文件里的「业务特化」只有两件（**框架无法替代**）：
① 按编码器输入形状分桶的 batch sampler（HF Dinov2 无 token mask，padding 会被
register 看到 ⇒ 只能同尺寸成批）；② 4 个解码器按「原图最适配分辨率」加权 2/5+3×1/5 的 L1。

---

## 2. 可移植性矩阵（实测 / 未实测分开写）

| 平台 | 设备串 | 分布式后端 | 状态 |
|---|---|---|---|
| NVIDIA CUDA | `cuda` | `nccl` | ✅ **本轮实测**：2× A800，`accelerate launch --multi_gpu`，bf16 + 双梯度检查点跑通 |
| 海光 DCU（ROCm/HIP） | `cuda`（别名 `dcu`） | `nccl`（即 RCCL） | ❌ **未适配/未实测**：只做到「代码路径就位」，从未在 DCU 真机上跑过一次。逐项风险与验收步骤见 §2.1 |
| 昇腾 NPU | `npu` | `hccl` | ⚠️ 路径已就位（自动 `import torch_npu`；`attn_implementation` 在 `auto` 下降级 `eager`），**未实测** |
| Intel XPU | `xpu` | `ccl` | ⚠️ 路径已就位，**未实测** |
| CPU | `cpu` | `gloo` | ✅ 本地冒烟实测跑通（`--smoke`，fp32），**训练不可行**（DINOv2-large） |

「就位」的确切含义：代码里不再有任何平台分支散落在训练逻辑中，全部经
`srdiff_runtime.Runtime` 解析；未实测的平台还需要在真机上过一遍 L1/L2/L3 阶梯
（见 §6 复现步骤），本报告不把「未实测」写成「支持」。

### 2.1 海光 DCU 适配现状（用户 2026-10-14 追问后逐条核实）

**结论先说：没有适配完，一次都没在 DCU 上跑过。** 目前只完成「代码路径就位」。
核实过程中还发现文档原先夸了一句 —— `Runtime.is_rocm()` 当时是**死代码**
（定义了但没有任何调用点）；本轮已修：`is_rocm` 变成实例属性并**真正接进
`attn_implementation()`**，DCU/ROCm 上 `auto` 会取保守的 `eager`。

**已就位（代码层，无需改代码即可在 DCU 上启动）**

| 项 | 实现 |
|---|---|
| 设备探测 | `torch.version.hip` 非空即认 ROCm；新增 `--device_type dcu` 别名（解析成 `cuda`，并在非 ROCm 环境给出 warn） |
| 设备可见性 | `CUDA_VISIBLE_DEVICES` **和** `HIP_VISIBLE_DEVICES` 都设（须在 `import torch` 前） |
| 分布式后端 | 自动 `nccl`（ROCm 下即 RCCL）；`torchrun` 与 `accelerate launch` 都能起 |
| 精度 | `bf16/fp16/fp32` 三档；`is_bf16_supported()=False` 时自动降 fp32；AMP 走设备无关 autocast |
| 注意力 | ROCm 上 `auto` → `eager`（见下）；可显式 `--attn_implementation sdpa` |
| 显存统计 | `Runtime.peak_memory_gib()` 走 `torch.cuda`（ROCm 同一套 API） |
| 依赖 | **不依赖 bitsandbytes**（默认 AdamW），DCU 上没有 bnb 也不影响 |

**未验证 + 预期会卡的点（按风险排序）**

| # | 风险 | 事实/依据 | 处理 |
|---|---|---|---|
| 1 | **torch 装错版本** | DCU 必须装 dtkit + ROCm 版 torch，`torch.version.hip` 才非空。装了 CUDA 版 torch 的 DCU 机器上 `torch.cuda.is_available()` 直接 False ⇒ 会静默落到 CPU | 先跑自检；`--device_type dcu` 在非 ROCm 上会 warn |
| 2 | **bf16 可能不可用** | 海光 Z100 级（gfx906）没有 bf16 矩阵核；K100-AI 有 | 显式 `--precision fp32`（或 fp16）；不要依赖自动降级 |
| 3 | **SDPA 退化/报错** | gfx906/gfx928 通常没有 CK flash-attn 后端，SDPA 退 math（能跑但慢）；个别 dtkit 版本上 SDPA 直接抛错 | `auto` 在 ROCm 上已默认 `eager`；确认无误后可切 `sdpa` 提速 |
| 4 | **RCCL 多卡通信** | 多卡 DDP 依赖 RCCL，DCU 上版本/拓扑问题比 CUDA 多 | 用自检的分布式项单机 `all_reduce` 先验；再上 2 卡冒烟 |
| 5 | **显存可能不够** | ×10 解码器在 A800 上峰值 **58.9 GiB/卡**（bs 上限 6 / patch_budget 6144）；DCU 单卡常见 16–64 GB | 按显存调小 `--batch_size` / `--patch_budget`，或 `--grad_accum` 补全局 batch |
| 6 | **性能未知** | DCU 的 bf16/fp16 吞吐与 DINOv2 24 层 + 解码器 12 步 BPTT 的实际 it/s 无任何实测 | 先 `--smoke --max_steps 3` 量 s/step，再排期 |

**DCU 上要跑的训练验收阶梯（一条命令一步）**

```bash
# L0 平台自检（本轮新写的工具，任意机器都能跑；FAIL 项必须清零）
python tools/check_platform.py --device_type dcu \
    --dino_dir /root/autodl-tmp/models/dinov2-large
#   → 逐项 PASS/FAIL：设备/显存 API/bf16/autocast/SDPA/梯度检查点/RCCL/DINOv2 前向

# L1 参数量自检（不需 GPU）
python tools/count_params.py --depth 2 --width 3072 --heads 24 --mlp_ratio 5.12

# L2 建模型（确认 ×10 结构与 encoder 都建得起来）
python train_multi.py --count_only --device_type dcu --precision fp32 \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --data_root /root/autodl-tmp/data --decoder_width 3072 --heads 24 \
    --decoder_mlp_ratio 5.12 --output_dir output/dcu_check

# L3 单机单卡冒烟 → 多卡正式
NUM_GPUS=1 DEVICE_TYPE=dcu PRECISION=fp32 LAUNCHER=python BS=2 BUDGET=2048 \
    STEPS=6 bash run_multi_10x.sh --smoke --limit 32
NUM_GPUS=8 DEVICE_TYPE=dcu LAUNCHER=torchrun bash run_multi_10x.sh
```

> `tools/check_platform.py` 已在本轮的两台机器上自测过：CPU（Debian/aarch64/py3.9/torch2.8）
> 与 NVIDIA A800（torch 2.12.1+cu130）全绿。它在 DCU 机器上就是那份**判定依据** ——
> 哪些项 PASS、哪些 FAIL，直接对应上表哪条风险落地了。

### 2.2 fp16 还是 bf16：判定规则（用户 2026-10-14 追问）

**本质差异**（两者都是 AMP：权重与优化器状态**始终 fp32**，只有前向/反向计算被 cast）：

| | 指数位 | 尾数位 | 数值范围 | 精度 | 需要 GradScaler |
|---|---|---|---|---|---|
| fp16 | 5 | 10 | ±65504（**会溢出/下溢**） | 较高 | **需要**（动态 loss scale） |
| bf16 | 8（= fp32） | 7 | ≈1e38（**与 fp32 同范围**） | 较低 | 不需要 |

⇒ **bf16 = fp32 的动态范围 + 更差的精度；fp16 = 更高精度 + 窄范围。**

**决策顺序（按优先级）**

1. **先看硬件速率，同速选 bf16。** 用 `python tools/check_platform.py --device_type dcu --bench`
   实测三者 TFLOPS。bf16 与 fp16 同速时**无条件选 bf16**：不需要 GradScaler、不会溢出跳步。
2. **bf16 慢或不可用 ⇒ 用 fp16。** 部分 DCU 代际（gfx906 一族）**没有 bf16 硬件支持**；
   此时 fp16 是唯一快路径（可能快 8 倍），代价是 GradScaler + 偶发跳步。
   > 代码自动降级已按此修正（2026-10-14）：`bf16 不可用 ⇒ 降 fp16`，**不再降到 fp32**——
   > 原来的「降 fp32」在 354 vs 44 TFLOPS 的卡上等于白送 8 倍时间。
3. **必须 fp32 的情况**：`--precision fp32` 显式指定。用于排查数值问题、或复现历史 fp32 口径
   （如 09-24 的 448 主线两相配方）。不要因为「怕掉精度」而用 fp32 —— 训练全程
   fp32 只在前向/反向，权重本来就是 fp32。
4. **本仓库的实测证据偏向 bf16**：A800 那条 2 epoch run 全程 bf16，loss 1.05 → 0.87 平滑、
   0 条 NaN、峰值 58.9 GiB、只有 2 条 DDP 警告。而这条工作负载的 grad_norm **出现过 68**
   （历史上 448 主线崩过一次 132）⇒ 尖峰是常态，fp16 下会有跳步（GradScaler 会自愈，
   但 `step` 计数照走 ⇒ 长跑有轻微漂移）。**除非硬件逼你，否则 bf16 更省心。**
5. **评测精度跟训练保持一致**（`eval_stepwise.py --precision` 同值）；要和 09-26 的
   26.10 px 逐位对齐就用当时的 bf16。

**一句话**：`bench 显示同速 → bf16；bf16 慢/没有 → fp16；显式排查 → fp32`。

---

## 3. 解码器参数量：基线 vs ×10

`python tools/count_params.py --scale 10` 的实跑输出（meta device 建模型，只数参数量）：

| 配置 | 4×decoder (M) | pixel_head (M) | special_bank (M) | **解码器合计 (M)** | vs 基线 |
|---|---:|---:|---:|---:|---:|
| **基线** depth 2 / width 1024 / heads 8 / mlp 4.0 | 137.5 | 3.30 | 0.15 | **140.9** | 1.000× |
| 候选 depth 2 / width 3072 / heads 24 / mlp 5.0 | 1387.7 | 3.30 | 0.15 | 1391.2 | 9.873× |
| **本轮** depth 2 / width 3072 / heads 24 / **mlp 5.12** | 1405.8 | 3.30 | 0.15 | **1409.3** | **10.001×** |

- 「解码器」口径 = 4 个 `OutputQueryDecoder` + `SpecialTokenBank` + `PixelHead`
  （`model_multi.param_breakdown()`，训练启动时也会打印同一套数字）。
- **只动宽度与 FFN，深度保持 2 不变**，`heads` 从 8→24 保持 `head_dim=128`（与基线一致），
  K=144 / 12 步 / 4 档分辨率 / 损失权重与 2026-09-26 完全相同 ⇒ 唯一实验变量是解码器容量。
- 全模型 445.3 M → **1713.6 M**（远端启动日志实测），解码器占比 31.6% → 82.2%。

---

## 4. 远端新机器的环境与数据（旧实例已回收，全部重建）

### 4.1 环境

```bash
export PATH=/root/miniconda3/bin:$PATH
pip install "transformers>=4.46,<5.0" accelerate datasets pyarrow safetensors tqdm
# → transformers 4.57.6 / accelerate 1.15.0 / pyarrow 26.0.0（pip 走 mirrors.aliyun.com，
#   tuna 镜像 403、pypi.org 可通；hf-mirror 直连实测 62 MB/s，huggingface.co 直连被墙）
```

### 4.2 数据（`tools/prep_data.py`，新写的可移植下载器）

| 源 | 布局 | 张数 | 来源 | 用时 |
|---|---|---:|---|---:|
| ImageNet ILSVRC2012 val | `imagenet_val/` | 50,000 | `mrm8488/ImageNet1K-val` parquet | 1.8 min |
| COCO2017 train | `coco/train2017/` | 118,287 | `pcuenq/coco-2017-mirror` zip | 见下 |
| Vimeo-90K im1 | `vimeo/` | **73,191** | `danjacobellis/vimeo90k_triplet` parquet | 6.8 min |
| DIV2K train HR | `div2k/DIV2K_train_HR/` | 800 | `yangtao9009/DIV2K` | ~1 min |
| Flickr2K | `flickr2k/` | 2,650 | `yangtao9009/Flickr2K` | ~2 min |
| OpenImages V7 子集 | `openimages/` | ~15,000 | `bitmind/open-images-v7-subset` parquet | 见下 |

> ⚠️ **数据偏差（必须记住）**：`danjacobellis/vimeo90k_triplet` 的 train split 实际只有
> **73,191** 个 clip（3 帧/clip，已核对 71/71 shard 全部下载成功、且每 shard
> `im1` 行数 ≈ 总行数/3）。2026-09-26 那轮的 91,701 来自 AutoDL **公共数据盘
> `/root/autodl-pub`**，本实例没有该挂载（`ls /root/autodl-pub` 不存在），
> HF 镜像上也找不到 91,701 版本的 triplet 镜像（`nasim69/vimeo90k-septuplet`
> 只剩一个 `.part1`、`lanikoworld/Vimeo90K` 空仓、`spzhu/Ranked-Vimeo-90K`
> 是 0.88 GiB 的排序子集；官方 `data.csail.mit.edu` 返回 403）。
> ⇒ 本轮索引 ≈ **259,641** 条（vimeo 少 18,510 条，占比 33.0% → 28.2%），
> 六源配比与本轮之后的所有结论都要按这个口径读。

---

## 5. 远端实跑记录

### 5.1 2 卡冒烟（`--only_patches 576 --smoke`）

```
[env] device=cuda name=NVIDIA A800 80GB PCIe count=2 torch=2.12.1+cu130 rocm=None
      backend=nccl | world=2 rank=0 | precision=bf16 grad_accum=1 | attn=sdpa
[model] 编码器 304.4M | 4×解码器 1405.8M + bank 0.15M + pixel_head 3.30M
        = 解码器合计 1409.3M (82.2% of total)
[model] 全模型 1713.6M | 可训 1713.6M | K=144 | steps=[1,4,9,…,144]
[data] 样本 446 | 桶 6 种 | 每卡 36 个 batch/epoch（2 卡已对齐）
[1/8] loss=1.08877 ... peak=43.38GiB 5.137s/step
[6/8] loss=1.74264 ... peak=56.15GiB 3.449s/step
```

### 5.4 正式 run（2 epoch）

用户 2026-10-14 追加：「这次训练 2 epoch」。索引 259,641 条 ⇒ `epoch_steps.py` 实测
**38,701 步/卡/epoch**，**2 epoch = 77,402 步/卡**（先用 40,243 步的 09-26 同预算起过一次，
收到 2 epoch 指示后立即停掉重起，产物归档在 `output/multires_10x_aborted_40243/`）。

```bash
cd /root/autodl-tmp/code/SR-Diffusion-v3
EPOCHS=2 NUM_GPUS=2 LAUNCHER=accelerate OUTPUT_DIR=/root/autodl-tmp/output/multires_10x \
  setsid nohup bash run_multi_10x.sh > /root/autodl-tmp/logs/run_multi_10x.log 2>&1 &
# 启动器自动算：每 epoch 38701 步/卡 × 2 epoch = 77402 步/卡
```

启动日志（实测）：

```
[env] device=cuda name=NVIDIA A800 80GB PCIe count=2 torch=2.12.1+cu130 rocm=None
      backend=nccl | world=2 rank=0 | precision=bf16 grad_accum=1 | attn=sdpa
[model] 编码器 304.4M | 4×解码器 1405.8M + bank 0.15M + pixel_head 3.30M
        = 解码器合计 1409.3M (82.2% of total)
[model] 全模型 1713.6M | 可训 1713.6M | K=144 | steps=[1,4,9,…,144]
[optim] AdamW 分组 lr: new=0.0003 dino=0.00015 | cosine warmup=2322 / max_steps=77402
[data] 样本 259641 | 桶 315 种 | top5 [(73531,(252,448)),(39594,(364,546)),(28415,(574,756)),…]
[data] 每卡 38701 个 batch/epoch（2 卡已对齐, batch 大小随桶变化）
[ 20/77402] loss=1.05186 dec=[1.309,1.308,1.300,1.315] gnorm=0.89  peak=58.87GiB 2.985s/step
[100/77402] loss=0.92893 dec=[0.769,0.762,0.753,0.787] gnorm=4.50  peak=58.93GiB 2.674s/step
```

| 量 | 实测 |
|---|---|
| 每卡峰值显存 | **58.9 GiB / 80 GiB**（bf16 + 编码器&解码器双梯度检查点） |
| 步速 | **2.63–2.99 s/step**（前 100 步，混合桶；09-26 同机对照见 §7） |
| 每卡优化步预算 | 38,701 × 2 = **77,402** |
| **ETA（按 2.67 s/step）** | **≈ 57 小时 ≈ 2.4 天**（单卡峰值 58.9 GiB，双卡 100% util） |
| ckpt 体积 | **20.56 GB / 个**（fp32 权重 6.9 GB + Adam 状态 13.7 GB；09-26 是 1.26 GB） |
| ckpt 策略 | `--save_every 2000 --keep_ckpts 4` ⇒ 最多占 **82 GB**（不轮转要写 ~780 GB，盘装不下） |
| 已知告警 | 2 条 DDP `Grad strides do not match bucket view strides`（与 09-26 同款，「This is not an error」） |

监控（任意时刻）：

```bash
tail -f /root/autodl-tmp/output/multires_10x/launch.log          # 主日志
tail -5 /root/autodl-tmp/output/multires_10x/train_log.jsonl     # 结构化逐步记录
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ls -la /root/autodl-tmp/output/multires_10x/                     # ckpt 轮转情况
```

跑完后的验收（留出集逐 step 曲线，与 09-26 §6 直接对比）：

```bash
python eval_stepwise.py \
  --ckpt /root/autodl-tmp/output/multires_10x/final_model.pt \
  --data_root /root/autodl-tmp/data/coco/val2017 \
  --out /root/autodl-tmp/output/multires_10x/eval_stepwise_cocoval.json
```

### 5.5 停止记录（2026-10-14 14:44，用户要求）

用户在内网用 DCU 同步训练 ⇒ 外网这台不再消耗余额，**主动停止并关机**。
（重启路径、数据资产清单写在数据盘的 `output/multires_10x/SHUTDOWN.md`。）

| 项 | 停机时实测 |
|---|---|
| 最后优化步 | **4240 / 77402**（1 epoch = 38701 步 ⇒ 约 0.11 epoch） |
| loss | 0.86921（20 步滑动均值），per_decoder [0.841,0.844,0.842,0.852]（四档同步、无塌缩） |
| 步速 / 峰值 | 2.565 s/step；58.93 GiB / 80 GiB |
| 产出 | `ckpt_2000.pt`、`ckpt_4000.pt`（各 20.56 GB，共 39 GB）—— ×10 解码器真实训过的权重 |
| 日志归档 | `doc/2026-10-14/run_logs_multires_10x/{launch.log,train_log.jsonl,SHUTDOWN.md}`（212 条逐步记录） |
| 未受影响 | ×10 参数表、显存/吞吐、可移植性证据、代码与文档全部在停机前完成并已推 GitHub |

⚠️ **因此本轮没有「跑完的 ×10 臂」**：4240 步的 loss 曲线只能说明「训得动、四档不塌缩」，
**不能**用来回答 09-26 §6 的逐步精炼问题。那件事要么在外网重开实例跑完 2 epoch，
要么直接在内网 DCU 上跑（`run_multi_10x.sh` 已设备无关，`tools/prep_data.py --verify`
可直接验收数据布局）。

---

## 6. 复现步骤（换任意机器）

```bash
# 0) 环境 + 平台自检（先确认这台机器真的能用，别一上来就长时间跑）
export PATH=/root/miniconda3/bin:$PATH
pip install "transformers>=4.46,<5.0" accelerate pyarrow safetensors tqdm
python tools/check_platform.py --dino_dir /root/autodl-tmp/models/dinov2-large
#   → 全 PASS 才继续；FAIL 项见 §2.1 的风险表

# 1) 数据（六源 ~64 GiB；先跑这个再做别的）
python tools/prep_data.py --data_root /root/autodl-tmp/data \
    --endpoint https://hf-mirror.com \
    --sources div2k flickr2k imagenet vimeo openimages coco --dl_threads 4 --ex_threads 8
# 幂等：每个源成功后在 <data_root>/.prep_done/<src> 落标记

# 2) DINOv2-large 权重（离线，1.16 GB）
mkdir -p /root/autodl-tmp/models/dinov2-large && cd $_
for f in config.json preprocessor_config.json model.safetensors; do
  curl -sL --fail -C - -o $f https://hf-mirror.com/facebook/dinov2-large/resolve/main/$f
done

# 3) 参数量自检（不需要 GPU，秒级）
python tools/count_params.py --depth 2 --width 3072 --heads 24 --mlp_ratio 5.12

# 4) 单卡冒烟 → 多卡正式 run
NUM_GPUS=2 bash run_multi_10x.sh                # DEVICE_TYPE=auto
# 昇腾/DCU 集群：NUM_GPUS=8 LAUNCHER=torchrun DEVICE_TYPE=npu bash run_multi_10x.sh
```

验收阶梯（照内网复现方案的 L1–L4 口径）：
**L1** `python tools/count_params.py` 出表 → **L2** `python train_multi.py --count_only` 建模型 +
打印 `解码器合计 1409.3M` → **L3** `--smoke` loss 下降 + 写出 `final_model.pt` →
**L4** 正式 run 的 `train_log.jsonl` 单调趋势 +（可选）`eval_stepwise.py` 留出集逐 step 曲线。

---

## 7. 未做 / 下一步

1. **×10 与 ×1 的同机对照**：本轮只跑了 ×10 臂。要严格回答「解码器容量是不是瓶颈」，
   应在**同一台机器、同一份数据**上再跑一次 ×1（旧配方）臂，否则只能与 09-26 的
   跨机器/跨数据数字比。
2. **DCU / NPU 真机验证**：本报告只到「代码路径就位」，需要在真机上过
   `tools/check_platform.py` → L1/L2/L3（§2.1 给了 DCU 的逐条风险与命令）。
   DCU 上还差一件事：`--precision fp32` 底下的 s/step 实测（决定 2 epoch 要跑几天）。
3. **Vimeo-90K 补齐**：拿到 91,701 版本的 triplet 后可重建索引，把六源配比对齐 09-26。
4. 后段 80 个 token 只买到 0.20 px 的问题（09-26 §6 的「最该做的一件事」）本轮未动
   ——×10 解码器恰好是检验「是解码器容量不够，还是 K=144 的信息已被前 48 个 token 吃干」
   的对照臂。
