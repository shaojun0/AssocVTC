"""train_multi — MultiResSR 多分辨率六源自然图像微调（设备无关版, 2026-10-14）

本文件 = 2026-09-26 版本的重构。用户 2026-10-14 的两条要求是这次重构的全部动机：

**要求 1：通用性（要能跑 DCU / NPU / 其它机器）**
    旧版把设备写死在 CUDA 上（`torch.cuda.set_device` / `device="cuda"` /
    `dist.init_process_group("nccl")` / `torch.autocast("cuda")` /
    `torch.cuda.max_memory_allocated` / 硬编码 `/root/autodl-tmp`）。
    现在：所有平台差异走 `srdiff_runtime.Runtime`（NVIDIA CUDA / 海光 DCU(ROCm) /
    昇腾 NPU(torch_npu) / Intel XPU / CPU），路径默认值全部可用环境变量覆盖，
    启动方式 `accelerate launch` 与 `torchrun` 都可以（DCU/NPU 集群常用后者）。
    平台适配现状见 `doc/2026-10-14/REPORT_multi_portable.md` 的矩阵（实测/未实测分列）。

**要求 2：不手写训练循环（不造轮子）**
    DDP 包装 / 混合精度 autocast / 梯度累积 / 梯度裁剪 / 断点续训 / 设备放置
    → 全部交给 HuggingFace **Accelerate**（`transformers.Trainer` 内部用的就是它，
    且它是官方指定的「自定义循环」框架）。本文件只保留真正业务特有的两件事：
      (a) 按编码器输入形状分桶的 batch sampler（HF Dinov2 无 token mask，
          padding 会被 register 看到 ⇒ 只能同尺寸成批）；
      (b) 4 个解码器按「原图最适配分辨率」加权的 L1。
    主循环里不再出现 `DistributedDataParallel` / `init_process_group` /
    `torch.cuda.*` / 手写 `autocast` / 手写 lr 曲线（改用
    `get_cosine_schedule_with_warmup`，分组基础 lr 由 AdamW param_groups 承载）。
    lr 调度用 `--reset_step_on_resume` 语义与旧版逐位一致（新调度器 + 归零）。

**要求 3：解码器参数量整体 ×10**
    `--decoder_width`（= 解码器 stack 的 d_model）/ `--decoder_depth` /
    `--decoder_mlp_ratio` / `--heads` 四个旋钮。精确参数量由
    `tools/count_params.py` 用真实模块统计（不是估算）。
    本轮采用配置：**depth=2 / width=3072 / heads=24 / mlp_ratio=5.12**
    ⇒ 解码器 140.9 M → **1409.3 M（×10.001）**，全模型 445.3 M → 1695.5 M。

用法::

    # 单机 2 卡（AutoDL / 任何 CUDA 机器）
    accelerate launch --multi_gpu --num_processes 2 train_multi.py \
        --data_root /root/autodl-tmp/data --dino_dir /root/autodl-tmp/models/dinov2-large \
        --output_dir /root/autodl-tmp/output/multires_10x \
        --batch_size 6 --patch_budget 6144 --grad_ckpt_decoder --precision bf16 \
        --decoder_width 3072 --decoder_depth 2 --heads 24 --decoder_mlp_ratio 5.12 \
        --max_steps 40243

    # 昇腾 NPU / 海光 DCU 集群（torchrun 亦可，accelerate 自动识别 RANK/WORLD_SIZE）
    torchrun --nproc_per_node=8 train_multi.py --device_type npu ...   # backend 自动 hccl

    # 只建模型/数参数，不训练
    python train_multi.py --count_only --decoder_width 3072 --heads 24 --decoder_mlp_ratio 5.12
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_multi import (BucketBatchSampler, MultiResCollator, MultiSourceDataset,
                        build_index, build_sources)
from model_multi import RESOS, MultiResSR, param_breakdown

# transformers 顶层有时不导出 Dinov2Model（4.57.6 实测）⇒ 走子模块
from transformers.models.dinov2 import Dinov2Model
from transformers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from srdiff_runtime import Runtime

ENV = os.environ.get


def get_args():
    p = argparse.ArgumentParser()
    # ── 路径（默认值可用环境变量覆盖, 不再硬编码 /root/autodl-tmp）──
    p.add_argument("--data_root", default=ENV("SRDIFF_DATA_ROOT",
                                              "/root/autodl-tmp/data"))
    p.add_argument("--dino_dir", default=ENV("SRDIFF_DINO_DIR",
                                             "/root/autodl-tmp/models/dinov2-large"))
    p.add_argument("--output_dir", default=ENV("SRDIFF_OUTPUT_DIR",
                                               "/root/autodl-tmp/output/multires"))
    p.add_argument("--data_cache", default="",
                   help="索引 json；默认 <data_root>/index_cache.json")
    p.add_argument("--long_side_cap", type=int, default=1280)
    # ── 设备 / 精度（设备无关）──
    p.add_argument("--device_type", default=ENV("SRDIFF_DEVICE", "auto"),
                   choices=["auto", "cuda", "dcu", "npu", "xpu", "mps", "cpu"],
                   help="dcu 是别名（海光 DCU 用 ROCm 版 torch，设备串仍是 cuda），"
                        "显式指定时会走更保守的注意力实现默认值；见 srdiff_runtime")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn_implementation", default="auto",
                   choices=["auto", "sdpa", "eager", "flash_attention_2"])
    p.add_argument("--no_pin_memory", action="store_true")
    # ── 模型 ──
    p.add_argument("--num_specials", type=int, default=144)
    p.add_argument("--decoder_depth", type=int, default=2)
    p.add_argument("--decoder_width", type=int, default=0,
                   help="解码器 stack 的 d_model；0 = 与编码器 dim 相同（历史口径）。"
                        "「解码器参数量 ×10」= 调这个 + depth + mlp_ratio，"
                        "精确值用 tools/count_params.py 验")
    p.add_argument("--decoder_mlp_ratio", "--mlp_ratio", dest="mlp_ratio",
                   type=float, default=4.0)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--step_plan", default="square", choices=["square", "fixed"])
    p.add_argument("--block", type=int, default=12)
    p.add_argument("--grad_ckpt_decoder", action="store_true",
                   help="4 个解码器各整段 12 步做一次检查点重算（省激活, 慢 ~30%%）")
    p.add_argument("--no_grad_ckpt", action="store_true", help="关掉编码器梯度检查点")
    p.add_argument("--freeze_dino", action="store_true")
    # ── 优化 ──
    p.add_argument("--lr_new", type=float, default=3e-4)
    p.add_argument("--lr_dino", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--find_unused_parameters", action="store_true")
    p.add_argument("--no_bucket_view", action="store_true")
    # ── 数据 ──
    p.add_argument("--batch_size", type=int, default=8, help="每卡 batch 上限（同桶内）")
    p.add_argument("--patch_budget", type=int, default=8192,
                   help="每批编码器 patch 数上限：bs=min(batch_size, budget/N)，0=关闭")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条样本（调试）")
    p.add_argument("--only_patches", type=int, default=0,
                   help="只保留编码器 patch 数 = N 的桶（显存冒烟用）")
    # ── 训练节奏 ──
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--keep_ckpts", type=int, default=0,
                   help="只保留最近 N 个 ckpt（0=全留）。×10 解码器后单个 ckpt ≈ 20 GB"
                        "（fp32 权重 6.9 GB + Adam 状态 13.7 GB），全留会把数据盘写满")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--reset_step_on_resume", action="store_true",
                   help="载入权重/优化器但把优化步归零（新一轮 warmup+cosine 必带）")
    p.add_argument("--smoke", action="store_true", help="只跑几步 + 打印显存")
    p.add_argument("--count_only", action="store_true",
                   help="只建模型 + 打印参数量（不训练）")
    return p.parse_args()


def load_dinov2(dino_dir: str, attn_implementation: str) -> nn.Module:
    """离线加载 DINOv2（不联网）。权重目录须含 config.json + model.safetensors。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return Dinov2Model.from_pretrained(dino_dir, local_files_only=True,
                                       attn_implementation=attn_implementation)


def build_dataloader(accelerator, args, log):
    """索引（rank0 建 + 落盘）→ Dataset → 按形状分桶的 batch sampler。"""
    cache = args.data_cache or os.path.join(args.data_root, "index_cache.json")
    sources = build_sources(args.data_root)
    if not os.path.exists(cache):
        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            build_index(sources, cache, args.long_side_cap)
        accelerator.wait_for_everyone()
    index = build_index(sources=[], cache=cache) if os.path.exists(cache) \
        else build_index(sources, "", args.long_side_cap)
    if args.limit:
        index = index[:args.limit]
    if args.only_patches:
        index = [r for r in index
                 if (r[1] // 14) * (r[2] // 14) == args.only_patches]
    if not index:
        raise SystemExit("[data] 过滤后没有样本（检查 --data_root / --only_patches）")
    shapes = [(bh, bw) for (_, bw, bh, _) in index]
    uniq = set(shapes)
    top = sorted(((shapes.count(s), s) for s in uniq), reverse=True)[:5]
    log(f"[data] 样本 {len(index)} | 桶 {len(uniq)} 种 | top5 {top}")

    ds = MultiSourceDataset(index, RESOS)
    sampler = BucketBatchSampler(shapes, args.batch_size, shuffle=True,
                                 seed=args.seed, drop_last=True,
                                 rank=accelerator.process_index,
                                 world_size=accelerator.num_processes,
                                 patch_budget=args.patch_budget)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                    collate_fn=MultiResCollator(),
                    pin_memory=(not args.no_pin_memory)
                    and accelerator.device.type in ("cuda", "xpu"),
                    persistent_workers=args.num_workers > 0,
                    prefetch_factor=4 if args.num_workers > 0 else None)
    log(f"[data] 每卡 {len(sampler)} 个 batch/epoch"
        f"（{accelerator.num_processes} 卡已对齐, batch 大小随桶变化）")
    return dl, sampler


def build_model(args, rt, log):
    dino = load_dinov2(args.dino_dir, rt.attn_implementation(args.attn_implementation))
    dim = dino.config.hidden_size
    if not args.no_grad_ckpt:
        dino.gradient_checkpointing_enable()
    model = MultiResSR(dino, num_specials=args.num_specials, dim=dim,
                       heads=args.heads, depth=args.decoder_depth,
                       mlp_ratio=args.mlp_ratio, step_plan=args.step_plan,
                       block=args.block, decoder_ckpt=args.grad_ckpt_decoder,
                       decoder_width=args.decoder_width)
    if args.freeze_dino:
        for p in model.dinov2.parameters():
            p.requires_grad_(False)
    # HF Dinov2 的 mask_token 在我们的 register 路径里用不到 ⇒ 必须关梯度,
    # 否则 DDP(find_unused_parameters=False) 会因为未使用参数报错
    for n, p in model.dinov2.named_parameters():
        if "mask_token" in n:
            p.requires_grad_(False)
    rep = param_breakdown(model)
    g = rep["groups"]
    log(f"[model] 编码器 {g['dinov2'] / 1e6:.1f}M | 4×解码器 {g['decoders'] / 1e6:.1f}M "
        f"+ bank {g['special_bank'] / 1e6:.2f}M + pixel_head {g['pixel_head'] / 1e6:.2f}M "
        f"= 解码器合计 {rep['decoder_total'] / 1e6:.1f}M "
        f"({rep['decoder_ratio_of_total'] * 100:.1f}% of total)")
    log(f"[model] 全模型 {rep['total'] / 1e6:.1f}M | 可训 "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M | "
        f"K={args.num_specials} | steps={model.steps}")
    log(f"[model] 解码器: depth={args.decoder_depth} "
        f"width={model.decoder_width or dim} heads={args.heads} "
        f"mlp_ratio={args.mlp_ratio} | 损失口径与 2026-09-26 一致（4 档加权, 直接预测）")
    return model, rep


def build_optimizer(model, args, max_steps):
    dino_ids = {id(p) for p in model.dinov2.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in dino_ids],
         "lr": args.lr_new, "name": "new"},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) in dino_ids],
         "lr": args.lr_dino, "name": "dino"},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    warmup = max(1, int(max_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup,
                                            num_training_steps=max_steps)
    return opt, sched, warmup


def main():
    args = get_args()
    runtime = Runtime(args.device_type)
    dev = runtime.device()

    precision = args.precision
    if runtime.device_type == "cpu":
        precision = "fp32"          # CPU 上 bf16 autocast 无意义
    if precision == "bf16" and runtime.device_type == "cuda" and not runtime.supports_bf16():
        print("[warn] 本卡不支持 bf16 ⇒ 降级 fp32")
        precision = "fp32"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=runtime.mixed_precision_flag(precision),
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters,
            gradient_as_bucket_view=not args.no_bucket_view)],
    )
    is_main = accelerator.is_main_process

    def log(*a):
        if is_main:
            print(*a, flush=True)

    d = runtime.describe()
    log(f"[env] device={d['device_type']} name={d['device_name']} "
        f"count={d['device_count']} torch={d['torch']} rocm={d['rocm_hip']} "
        f"backend={d['dist_backend']} | world={accelerator.num_processes} "
        f"rank={accelerator.process_index} | precision={precision} "
        f"grad_accum={args.grad_accum} | attn={runtime.attn_implementation(args.attn_implementation)}")
    if runtime.device_type == "cpu":
        log("[env] ⚠️ CPU：只能冒烟，DINOv2-large 在 CPU 上不可训练")
    os.makedirs(args.output_dir, exist_ok=True)

    model, rep = build_model(args, runtime, log)
    accelerator.wait_for_everyone()

    if args.count_only:
        if is_main:
            with open(os.path.join(args.output_dir, "params.json"), "w") as f:
                json.dump({"breakdown": rep, "args": vars(args)}, f,
                          indent=2, ensure_ascii=False)
            torch.save({"model": model.state_dict()},
                       os.path.join(args.output_dir, "init_state.pt"))
            log(f"[count_only] 参数量 → {args.output_dir}/params.json")
        return

    max_steps = int(args.max_steps)
    opt, sched, warmup = build_optimizer(model, args, max_steps)
    log(f"[optim] AdamW 分组 lr: new={args.lr_new} dino={args.lr_dino} | "
        f"cosine warmup={warmup} / max_steps={max_steps} | wd={args.weight_decay}")

    # ── 续训：在 prepare 之前载入（模型/优化器都是裸对象, 避免包装层差异）──
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ck_step = ck.get("step", 0)
        if args.reset_step_on_resume:
            start_step = 0
        else:
            start_step = int(ck_step)
            for _ in range(start_step):     # 新调度器推到对应位置（纯 python, 毫秒级）
                sched.step()
        log(f"[resume] {args.resume} @ ckpt_step={ck_step} → start_step={start_step} "
            f"(reset={bool(args.reset_step_on_resume)})")

    dl, sampler = build_dataloader(accelerator, args, log)
    # Accelerate 接管：设备放置（CUDA/NPU/XPU/CPU）、DDP 包装、AMP autocast。
    # 桶采样器已按 process_index/num_processes 分片（batch 大小随桶变化，
    # 交给 accelerate 的 BatchSamplerShard 反而要受 even_batches 约束），
    # 所以 dataloader 不再二次分片，仅由本循环把 batch 放到 accelerator.device。
    model, opt, sched = accelerator.prepare(model, opt, sched)
    trainable = [p for p in accelerator.unwrap_model(model).parameters()
                 if p.requires_grad]

    step = start_step
    t0 = time.time()
    running, running_n = 0.0, 0
    stop = False
    jsonl = open(os.path.join(args.output_dir, "train_log.jsonl"), "a") if is_main else None

    log(f"[train] 开始：world={accelerator.num_processes} batch上限={args.batch_size} "
        f"patch_budget={args.patch_budget} 精度={precision} "
        f"grad_ckpt_decoder={args.grad_ckpt_decoder}")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in dl:
            if step >= max_steps:
                stop = True
                break
            with accelerator.accumulate(model):
                px = batch["pixel_values"].to(dev, non_blocking=True)
                tgt = [t.to(dev, non_blocking=True) for t in batch["targets"]]
                best = batch["best_idx"].to(dev, non_blocking=True)
                out = model(px, tgt, best)
                loss = out["loss"]
                accelerator.backward(loss)
                gnorm = None
                if accelerator.sync_gradients:
                    gnorm = accelerator.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            step += 1
            running += float(out["loss"].detach())
            running_n += 1
            if step % args.log_every == 0:
                peak = runtime.peak_memory_gib()
                lrs = {g.get("name", str(i)): round(g["lr"], 8)
                       for i, g in enumerate(opt.param_groups)}
                rec = {"step": step, "epoch": epoch,
                       "loss": round(running / max(1, running_n), 5),
                       "recon": round(float(out["recon"].detach()), 5),
                       "per_decoder": [round(float(x), 5) for x in out["per_decoder"]],
                       "grad_norm": round(float(gnorm), 4) if gnorm is not None else None,
                       "lr": lrs, "peak_gib": round(peak, 2),
                       "sec_per_step": round((time.time() - t0)
                                             / max(1, step - start_step), 3)}
                if is_main:
                    jsonl.write(json.dumps(rec) + "\n")
                    jsonl.flush()
                    log(f"[{step}/{max_steps}] loss={rec['loss']} recon={rec['recon']} "
                        f"dec={rec['per_decoder']} gnorm={rec['grad_norm']} "
                        f"peak={rec['peak_gib']}GiB {rec['sec_per_step']}s/step")
                running, running_n = 0.0, 0
            if step % args.save_every == 0:
                accelerator.wait_for_everyone()
                if is_main:
                    raw = accelerator.unwrap_model(model)
                    ckpt = os.path.join(args.output_dir, f"ckpt_{step}.pt")
                    torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                                "step": step, "args": vars(args), "params": rep}, ckpt)
                    log(f"[save] {os.path.basename(ckpt)} "
                        f"({os.path.getsize(ckpt) / 2 ** 30:.1f} GiB)")
                    if args.keep_ckpts > 0:      # 轮转：×10 解码器后单 ckpt ≈20 GB
                        olds = sorted(
                            (int(f.split("_")[1].split(".")[0]), f)
                            for f in os.listdir(args.output_dir)
                            if f.startswith("ckpt_") and f.endswith(".pt"))
                        for _, fn in olds[:-args.keep_ckpts]:
                            os.remove(os.path.join(args.output_dir, fn))
                            log(f"[save] 删除旧 ckpt {fn}（keep_ckpts={args.keep_ckpts}）")
                accelerator.wait_for_everyone()
            if args.smoke and step >= start_step + 6:
                stop = True
                break
        if stop:
            break

    accelerator.wait_for_everyone()
    if is_main:
        raw = accelerator.unwrap_model(model)
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                    "step": step, "args": vars(args), "params": rep},
                   os.path.join(args.output_dir, "final_model.pt"))
        log(f"[done] step={step} peak={runtime.peak_memory_gib():.2f}GiB "
            f"→ {args.output_dir}/final_model.pt")
        if jsonl:
            jsonl.close()
    if accelerator.num_processes > 1:
        accelerator.end_training()


if __name__ == "__main__":
    main()
