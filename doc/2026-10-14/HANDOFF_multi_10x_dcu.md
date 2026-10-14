# HANDOFF — MultiResSR 解码器 ×10（外网 A800 已停 + 内网 DCU 在跑）

> 交接时间：**2026-10-14**（用户明确「后续在新会话聊」）
> 配套报告：[`REPORT_multi_portable.md`](REPORT_multi_portable.md)（本目录，含全部实测数字与推导）
> 归档日志：[`run_logs_multires_10x/`](run_logs_multires_10x/)

---

## 1. 现在是什么状态（一句话各自）

| 臂 | 位置 | 状态 |
|---|---|---|
| **外网 A800** | `ssh -p 15990 root@connect.nma1.seetacloud.com`（2×A800 80GB） | **已停止并关机**（用户要求，省余额）。停机在 step **4240/77402**，loss 0.869，产出 `ckpt_2000.pt` / `ckpt_4000.pt`（各 20.56 GB）留在实例数据盘；数据/索引/权重都还在数据盘，重启即可续 |
| **内网 DCU** | `/data/srdiff-20260926/SR-Diffusion-v3-main/`，8 卡 × 144 GB | **在跑**。已跑通 → 调 batch OOM → 已定策（不死磕 batch，改 `math+efficient` + 把显存给关重算）。**没有跑完的 ×10 臂** |
| 代码 / 文档 | GitHub `shaojun0/SR-Diffusion-v3` @ **`ccdc1c1`** | 已同步，工作树干净 |

**结论优先级**：解码器 ×10 的**参数量与可移植性**都已验证并入库；**「×10 是否改变逐步精炼结论」尚未回答** —— 需要内网把 2 epoch 跑完 + 出留出集曲线 + 建议同机补一条 ×1 臂。

---

## 2. 已经拿到、不用重做的结论

- **解码器参数量 140.9 M → 1409.3 M（×10.001）**，全模型 445.3 M → 1713.6 M
  （`depth 2` 不变 / `width 1024→3072` / `heads 8→24`（head_dim 仍 128）/ `mlp_ratio 4.0→5.12`）。
  验算：`python tools/count_params.py --depth 2 --width 3072 --heads 24 --mlp_ratio 5.12`
- **A800 实测**：2 卡 bf16 + 双梯度检查点，峰值 **58.93 GiB/卡**、**2.565–2.687 s/step**、
  单 ckpt **20.56 GB**、loss 1.05 → 0.869（4240 步，四档分项同步无塌缩）
- **数据**：六源 **259,641**（ImageNet 50k / COCO 118k / Vimeo **73,191** / DIV2K 800 /
  Flickr2K 2,650 / OpenImages 18,984），索引 315 桶，**1 epoch = 38,701 步/卡（2 卡）**
- **墙钟测算**（paper 规格 44/354 TFLOPS + A800 标定 μ=17.6%）：8 卡 fp16 ≈ **12.7 h**、
  8 卡 fp32 ≈ **102 h**、8 卡 fp16+关重算 ≈ **9.5 h**（详见报告 §2.3；属外推，需 smoke 校准）

---

## 3. 本轮新增的工具（都在 main 里）

| 工具 | 用途 |
|---|---|
| `tools/check_platform.py` | 一条命令判定「这台机器能不能跑」：设备/显存 API/bf16/autocast/SDPA/梯度检查点/分布式 all_reduce/DINOv2 前向，`--bench` 还实测 fp32/fp16/bf16 的 TFLOPS |
| `tools/diag_sdpa.py` | 按解码器真实形状逐 SDPA 后端跑，定位 DCU 上哪个融合后端崩 |
| `tools/plan_batch.py` | (BS, patch_budget, 卡数) 的后果表：每卡步/epoch、全局平均 batch、最大 N 桶 bs（显存风险）、drop_loss 比例 |
| `tools/prep_data.py` | 任意机器重建六源布局；`--verify [--build_index]` 校验张数 + 算每 epoch 步数与 max_steps |
| `tools/count_params.py` | 参数量精确统计（×N 的验收口径） |
| `run_multi_10x.sh` | 可移植入口；环境变量见下 |

```bash
# 启动器支持的环境变量（全部可覆盖）
NUM_GPUS LAUNCHER(accelerate|torchrun|python) DEVICE_TYPE(auto|cuda|dcu|npu|xpu|cpu)
DATA_ROOT DINO_DIR OUTPUT_DIR CACHE MASTER_PORT PYTHON
BS BUDGET GRAD_ACCUM PRECISION(bf16|fp16|fp32) SDPA_BACKEND ATTN_IMPL
GRAD_CKPT_DECODER(1|0) GRAD_CKPT_ENCODER(1|0)
DEC_W DEC_D DEC_H DEC_MLP LR_NEW LR_DINO WARMUP_RATIO
EPOCHS STEPS SAVE_EVERY KEEP_CKPTS LOG_EVERY NUM_WORKERS EXTRA_ARGS
```

---

## 4. 坑清单（每条都真实踩过，不是推测）

1. **`--batch_size` 只是上限**，实际 `bs = max(1, min(batch_size, patch_budget // N))`，N 随桶变（最大 4096）
   ⇒ 只调 `--batch_size` 对大 N 桶**完全无效**。09-26 run 设 6，实测每卡平均只有 **3.35**。
   要全桶吃满 `BS` ⇒ `patch_budget ≥ BS × N_max`。**调 batch 不省墙钟**（总 FLOPs 固定，只动 MFU ±20%）。
2. **DCU 上 bf16 的 SDPA 融合后端会崩**：`IndexError: select(): index 0 out of range for tensor of size [0] at dimension 0`
   —— 本地已排除形状原因（零 batch/零 query/零 kv/空 memory 喂 math 后端全过）。处置 `--sdpa_backend math`
   （**全局**生效：梯度检查点的重算在 backward，局部 `sdpa_kernel` 覆盖不到）。想提速先跑 `tools/diag_sdpa.py`，
   若 `efficient` 不崩就用 `math+efficient`（不物化 N×N，省几十 GB）。
3. **`--attn_implementation` 管不到解码器**：它只作用于 HF 的 DINOv2 编码器；解码器是 `nn.MultiheadAttention`，
   永远走 SDPA ⇒ 控制它必须用 `--sdpa_backend`。
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 必须设**（老脚本有，我的可移植版一开始漏了）。
   315 种桶形状 + 变长 batch ⇒ 极易碎片化 OOM。已在启动器里默认导出。
5. **bf16 不可用时降级到 fp16，不是 fp32**（354 vs 44 TFLOPS = 8× 陷阱）。已在代码里按此修。
   判据：同速优先 bf16（范围= fp32，不需要 GradScaler）；bf16 慢/没有 ⇒ fp16；显式排障 ⇒ fp32。
6. **`index_cache.json` 存的是绝对路径** ⇒ 换挂载点必须删掉重建（rank0 建一次，259,641 条约 3–4 分钟）。
7. **单 ckpt 20.56 GB**（×10 后：fp32 权重 6.9 + Adam 13.7）⇒ 必带 `--keep_ckpts`，否则 2 epoch 要写 ~780 GB。
8. **Vimeo 只有 73,191**（HF 镜像 `danjacobellis/vimeo90k_triplet` 就是这么多；官方 91,701 在已回收实例的
   `/root/autodl-pub`，本机无此挂载）⇒ 索引 259,641 而非 278,151。两边同源 ⇒ 数据可比。
9. **`epoch_steps.py` 的第 4 列才是 steps_per_rank**（第 5 列是 global_batches）；启动器会自动算，
   换 BS/BUDGET/卡数后**务必重算**（8 卡与 2 卡的步数差 4 倍，别抄 77,402）。
10. **全局 batch 变了就要调 lr**（线性规则 `lr × 新/旧`，`WARMUP_RATIO` 保持），否则是没记录的配方变更。

---

## 5. 新会话从这里接（建议顺序）

```bash
# 0) 先确认代码是最新的（内网那份 train_multi.py 实测比 main 少几行，必须重拉）
cd <code>/SR-Diffusion-v3 && git pull && git log --oneline -1   # 期望 ccdc1c1

# 1) 平台自检（FAIL 清零再走）+ 参数量自检
python tools/check_platform.py --device_type dcu --dino_dir <DINO> --bench
python tools/count_params.py --depth 2 --width 3072 --heads 24 --mlp_ratio 5.12

# 2) 选 SDPA 后端（先 diag，再拿结论）
python tools/diag_sdpa.py --device_type dcu --batch_size 6 --decoder_width 3072 --heads 24 --encoder

# 3) 定 batch（不死磕；先看表，再用最大 N 桶冒烟量峰值）
python tools/plan_batch.py --cache <data_root>/index_cache.json --world 8
BS=6 BUDGET=6144 SDPA_BACKEND=math+efficient ... bash run_multi_10x.sh --smoke --only_patches <Nmax> --max_steps 3
#   看日志 peak_gib；日志每步已带 N= / bs=，OOM 时直接看最后一行

# 4) 正式跑（2 epoch；max_steps 由启动器算）
NUM_GPUS=8 DEVICE_TYPE=dcu LAUNCHER=torchrun PRECISION=bf16 \
  SDPA_BACKEND=math+efficient GRAD_CKPT_DECODER=0 \
  DATA_ROOT=/data/srdiff-20260926 DINO_DIR=<DINO> OUTPUT_DIR=<OUT> EPOCHS=2 \
  bash run_multi_10x.sh

# 5) 跑完后出留出集逐 step 曲线（与 09-26 的 31.25 → 26.10 px 对比）
python eval_stepwise.py --ckpt <OUT>/final_model.pt \
  --data_root /data/srdiff-20260926/coco/val2017 --out <OUT>/eval_stepwise_cocoval.json
```

**待办（按价值排序）**

1. 内网 DCU 跑完 2 epoch ×10 臂 → 出 `eval_stepwise` 逐 step 曲线。
2. **同机补一条 ×1 臂**（`DEC_W=0 DEC_H=8 DEC_MLP=4`）—— 跨机器/跨数据对照不可比，这是唯一能严格回答
   「解码器容量是不是瓶颈」的做法。
3. 回答 09-26 §6 遗留问题：后段 80 个 token 只买到 0.20 px，是解码器容量不够，还是 K=144 的信息已被前 48 个 token 吃干。
4. 可选：补齐 Vimeo-90K 到 91,701 后重建索引，把六源配比对齐 09-26。
5. 可选：外网实例重启续训（数据/索引/权重/ckpt 都在盘上；`--resume .../ckpt_4000.pt` 不带 `--reset_step_on_resume`）。

---

## 6. 外网实例重启续训的准确命令

```bash
# 实例开机后（数据盘保留）：数据/索引/dinov2/ckpt 都在 /root/autodl-tmp
cd /root/autodl-tmp/code/SR-Diffusion-v3
STEPS=77402 EPOCHS=2 NUM_GPUS=2 LAUNCHER=accelerate OUTPUT_DIR=/root/autodl-tmp/output/multires_10x \
  setsid nohup bash run_multi_10x.sh > /root/autodl-tmp/logs/run_multi_10x.log 2>&1 &
```
（该实例的停机说明与资产清单也在盘上：`/root/autodl-tmp/output/multires_10x/SHUTDOWN.md`）
