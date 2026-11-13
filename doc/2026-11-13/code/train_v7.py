"""v7 训练脚本：448×448 真实视频重建（单卡 MUSA）。

架构（model_v7 = 主线 commit d123ee7 / model_v3 的设计）
  encoder : ViT 式 patch-32 编码器，229 token/帧（~241M）
  middle  : 帧间循环交叉注意力，KV = [mem_init] + 历史帧 z_s（~235M）
  decoder : 196 行查询 + 5 个平方采样步，每步直接预测整图（~240M）

阶段（--stage）
  image : 一个样本 = img_per 张独立图像 → (B*img_per, 1, 3, H, W) 各自 F=1 冷启动前向
  video : 一个样本 = [start-n_warm, start+window) 连续帧；前 n_warm 帧只预热 KV
          （编码 + 中间态 no_grad），后 window 帧 BPTT。历史只来自 z_s，预热帧不解码。

与 train_scale.py(v6) 的区别
  * 删掉 state_gen / readout / recon_vectors / hist_mem / val_budgets / div_weight /
    多卡 torch.distributed：帧循环、KV、BPTT 全部在 model 内部，前向整体交给 model。
  * 验证三档平均 PSNR：cold（每帧独立 F=1）/ rec（max_hist=0，只有上一帧 query 循环）
    / warm（max_hist=None，全部历史帧 z_s 进 KV）。

跑法见 run_v7.sh（两阶段配方）与 pipeline_v7.sh（看守重启）。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from typing import Dict, List

import torch

from data_real import (PlanBatchSampler, RealClipDataset, collate, decode_clip,
                       read_index)
from model_v7 import DEFAULT_CFG, init_encoder_from_vit
from model_v8 import build_any as build_model


# ---------------------------------------------------------------- 基础设施
def log(msg: str, fh=None):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def get_device(local_rank: int = 0) -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device(f"musa:{local_rank}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


AMP_DTYPE = "bf16"


def autocast_ctx(dev: torch.device, enabled: bool):
    if not enabled or dev.type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    dt = torch.float16 if AMP_DTYPE == "fp16" else torch.bfloat16
    return torch.autocast(device_type=dev.type, dtype=dt)


def mem_gb(dev: torch.device) -> float:
    """已分配的显存峰值（GB）；非 GPU 返回 0。"""
    try:
        if dev.type == "musa":
            return torch.musa.max_memory_allocated(dev) / 2 ** 30
        if dev.type == "cuda":
            return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    except Exception:
        pass
    return 0.0


def lr_scale(step: int, total: int, warmup: int, min_ratio: float = 0.05) -> float:
    """cosine + 线性 warmup（与 v6 一致）。"""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


# ---------------------------------------------------------------- 前向
def loss_mode_at(step: int, total: int, args) -> str:
    """深监督步权重课程：last(0-10%) -> lin(10-35%) -> mean(35%+)。"""
    if not args.loss_curriculum:
        return args.loss_weight
    f = step / max(1, total)
    if f < 0.10:
        return "last"
    if f < 0.35:
        return "lin"
    return "mean"


def forward_video(model, x: torch.Tensor, n_warm: int, args) -> Dict[str, torch.Tensor]:
    """x: (B,T,3,H,W)。前 n_warm 帧只预热 KV（无梯度），其余帧 BPTT。"""
    return model(x, n_warm=int(n_warm), step_weights=getattr(args, "_lw", args.loss_weight),
                 enc_chunk=args.enc_chunk)


def forward_image(model, x: torch.Tensor, args) -> Dict[str, torch.Tensor]:
    """x: (B,img_per,3,H,W)，img_per 帧是**独立图像** → (B*img_per,1,3,H,W) 冷启动前向。"""
    xf = x.flatten(0, 1).unsqueeze(1)
    return model(xf, n_warm=0, step_weights=getattr(args, "_lw", args.loss_weight),
                 enc_chunk=args.enc_chunk)


# ---------------------------------------------------------------- 验证
@torch.no_grad()
def evaluate(model, val_root: str, val_items: List[dict], args, dev) -> Dict[str, float]:
    """三档平均 PSNR：

    cold : 目标帧逐个独立前向（F=1, max_hist=0），纯冷启动 KV=[mem_init]
    rec  : 整段一次前向 n_warm=val_hist, max_hist=0（只有上一帧 query 的循环）
    warm : 整段一次前向 n_warm=val_hist, max_hist=None（全部历史帧 z_s 进 KV）
    """
    W, Hw = args.val_window, args.val_hist
    out: Dict[str, List[float]] = {k: [] for k in ("cold", "rec", "warm")}
    n = min(args.val_clips, len(val_items))
    for it in val_items[:n]:
        path = os.path.join(val_root, it["path"])
        start = max(Hw + 4, args.clip_frames // 2)          # 片段中段
        try:
            arr = decode_clip(path, start - Hw, start + W, args.img)
        except Exception:
            continue
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0).to(dev)  # (Hw+W,3,H,W)
        # ---- cold：每帧独立 F=1，PSNR 取 psnr_per_frame.mean()
        ps = []
        for j in range(Hw, Hw + W):
            o = model(x[j:j + 1].unsqueeze(0), n_warm=0, max_hist=0,
                      step_weights="mean", enc_chunk=args.enc_chunk)
            ps.append(float(o["psnr_per_frame"].mean()))
        out["cold"].append(sum(ps) / len(ps))
        # ---- rec / warm：整段一次前向，只统计目标帧（帧 >= val_hist）
        xv = x.unsqueeze(0)                                  # (1,Hw+W,3,H,W)
        for name, mh in (("rec", 0), ("warm", None)):
            o = model(xv, n_warm=Hw, max_hist=mh,
                      step_weights="mean", enc_chunk=args.enc_chunk)
            out[name].append(float(o["psnr"]))
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in out.items()}


def write_val(path: str, step: int, m: Dict[str, float]):
    with open(path, "a") as f:
        f.write(json.dumps(dict(step=step, **m)) + "\n")


# ---------------------------------------------------------------- 参数/优化器
def setup_trainable(model, args):
    """按 --train-enc/--train-mid/--train-dec 冻结模块；参数组 = 模块 × (weight_decay/no)"""
    flags = dict(encoder=args.train_enc, middle=args.train_mid, decoder=args.train_dec)
    for name, flag in flags.items():
        for p in getattr(model, name).parameters():
            p.requires_grad = bool(flag)
    groups = []
    for name, mult in (("encoder", args.lr_enc), ("middle", args.lr_mid),
                       ("decoder", args.lr_dec)):
        if not flags[name]:
            continue
        decay, no_decay = [], []
        for _, p in getattr(model, name).named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        if decay:
            groups.append(dict(params=decay, weight_decay=args.wd, lr=args.lr * mult,
                               base=args.lr * mult, tag=f"{name}.decay"))
        if no_decay:
            groups.append(dict(params=no_decay, weight_decay=0.0, lr=args.lr * mult,
                               base=args.lr * mult, tag=f"{name}.nodecay"))
    return groups


def sync_grads(params, world: int, bf16: bool = False):
    """手写数据并行同步：flat buffer + all_reduce(SUM)/world（同 train_scale.py，不用 DDP）。

    MUSA 的 DDP 内部计时器有 assert；bf16 sync 不稳，默认 fp32 flat buffer。
    world=1 时直接返回（单卡路径不受影响）。
    """
    if world <= 1:
        return
    import torch.distributed as dist
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    if bf16:
        flat = flat.to(torch.bfloat16)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat = flat.to(torch.float32).div_(world)
    for g, u in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(u)



# ---------------------------------------------------------------- 主流程
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="video", choices=["image", "video"])
    ap.add_argument("--data-root", default="/root/autodl-tmp/video_recon/data3/train")
    ap.add_argument("--index-name", default="index.jsonl")
    ap.add_argument("--val-root", default="/root/autodl-tmp/video_recon/data3/val")
    ap.add_argument("--val-index-name", default="index.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg-json", default="")
    ap.add_argument("--init", default="", help="只加载 ckpt['model']（图像→视频热启动）")
    ap.add_argument("--resume", default="", help="恢复 step/optimizer（崩溃续跑）")
    ap.add_argument("--vit-weights", default="", help="ViT-L/16 state_dict（初始化编码器）")
    ap.add_argument("--img", type=int, default=448)
    ap.add_argument("--clip-frames", type=int, default=120)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--hist-choices", default="8,16,32,48")
    ap.add_argument("--p-cold", type=float, default=0.3)
    ap.add_argument("--img-per", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积（真实 batch = batch*accum）")
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--lr-enc", type=float, default=0.1)
    ap.add_argument("--lr-mid", type=float, default=1.0)
    ap.add_argument("--lr-dec", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--train-enc", type=int, default=-1, help="-1=按 stage 自动（image 1 / video 0）")
    ap.add_argument("--train-mid", type=int, default=-1, help="-1=自动（1）")
    ap.add_argument("--train-dec", type=int, default=-1, help="-1=自动（1）")
    ap.add_argument("--steps", type=int, default=0, help="总步数（0=按 --hours 自动估算）")
    ap.add_argument("--hours", type=float, default=20.0)
    ap.add_argument("--auto-steps-after", type=int, default=40)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-clips", type=int, default=1)
    ap.add_argument("--val-window", type=int, default=6)
    ap.add_argument("--val-hist", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--enc-chunk", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"],
                    help="混合精度 dtype（V100 等不支持 bf16 的卡用 fp16）")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--loss-weight", default="mean", choices=["mean", "lin", "last"])
    ap.add_argument("--loss-curriculum", type=int, default=0,
                    help="1=深监督权重按进度切换 last(0-10%) -> lin(10-35%) -> mean(35%+)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--print-cfg", type=int, default=1)
    ap.add_argument("--reload-every", type=int, default=0,
                    help="每 N 步重读 index.jsonl（数据在训练中持续追加）")
    # ---- 数据并行（默认 world=1 即单卡，路径与行为不变）
    ap.add_argument("--world", type=int, default=1, help="进程数（1=单卡，默认）")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--local-rank", type=int, default=-1, help="-1=取 --rank")
    ap.add_argument("--master-addr", default="127.0.0.1")
    ap.add_argument("--master-port", default="29660")
    ap.add_argument("--sync-bf16", type=int, default=0,
                    help="all_reduce 用 bf16（旧经验不稳；默认 fp32 flat buffer）")
    return ap.parse_args()


def main():
    args = parse_args()
    global AMP_DTYPE
    AMP_DTYPE = str(args.amp_dtype)
    rank, world = int(args.rank), int(args.world)
    local_rank = int(args.local_rank) if args.local_rank >= 0 else rank
    dist = None
    if world > 1:                                        # 单卡（world=1）不 init、不 import
        import torch.distributed as dist_mod
        os.environ.setdefault("MASTER_ADDR", str(args.master_addr))
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        dist_mod.init_process_group(backend="mccl")
        dist = dist_mod
    dev = get_device(local_rank)
    if dev.type == "musa":
        torch.musa.set_device(local_rank)
    # DP 要求各 rank 初始参数逐位一致；各 rank 取不同数据由 PlanBatchSampler(rank, world) 保证
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    fh = open(os.path.join(args.out, "train.log" if world == 1 else f"train_r{rank}.log"), "a")
    log(f"stage={args.stage} out={args.out} device={dev} world={world} rank={rank}", fh)

    # 可训练开关默认值按阶段决定（image 1/1/1，video 0/1/1）
    if args.train_enc < 0:
        args.train_enc = 1 if args.stage == "image" else 0
    if args.train_mid < 0:
        args.train_mid = 1
    if args.train_dec < 0:
        args.train_dec = 1

    # ---- 模型
    cfg = copy.deepcopy(DEFAULT_CFG)
    if args.cfg_json:
        cfg = json.load(open(args.cfg_json))
    if args.print_cfg:
        log(f"cfg={json.dumps(cfg)}", fh)
    model = build_model(cfg)
    log(f"steps={model.steps} windows={model.decoder.windows}", fh)

    if args.vit_weights:
        log(f"init encoder from ViT: {args.vit_weights}", fh)
        ck = torch.load(args.vit_weights, map_location="cpu")
        sd = ck.get("state_dict", ck)
        info = init_encoder_from_vit(model.encoder, sd)
        log(f"vit init {info}", fh)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        log(f"init from {args.init}: missing={len(missing)} unexpected={len(unexpected)}", fh)
        if missing:
            log(f"  missing[:8]={list(missing)[:8]}", fh)
        if unexpected:
            log(f"  unexpected[:8]={list(unexpected)[:8]}", fh)
    model = model.to(dev)
    pg = model.param_groups()
    log("params " + " ".join(f"{k}={v/1e6:.1f}M" for k, v in pg.items()), fh)

    # ---- 数据
    hist_choices = [int(v) for v in str(args.hist_choices).split(",") if v.strip()]
    ds = RealClipDataset(args.data_root, args.index_name, clip_frames=args.clip_frames,
                         window=args.window, img=args.img,
                         hist_choices=hist_choices, p_cold=args.p_cold,
                         img_per=(args.img_per if args.stage == "image" else 0),
                         stage=args.stage)
    val_items = read_index(args.val_root, args.val_index_name) if args.val_root else []
    log(f"train clips={ds.n_clips} val clips={len(val_items)}", fh)

    # ---- 优化器
    groups = setup_trainable(model, args)
    if not groups:
        log("ERROR: 没有可训练参数（--train-enc/mid/dec 全 0）", fh)
        return
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    # fp16 必须配 loss scaling：本工程的 loss 是对 (batch×帧×步×像素) 取均值，输出端
    # 梯度只有 ~1e-7 量级，不缩放时 fp16 直接下溢成 0（实测：全模型 0 个非零梯度，
    # 训练只剩 AdamW 的 weight decay，权重两步之间只差一个 0.989 的标量）。bf16 的
    # 指数范围和 fp32 一样，不需要 scaler。
    USE_SCALER = bool(args.amp) and str(args.amp_dtype) == "fp16"
    scaler = None
    if USE_SCALER:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
        log(f"AMP=fp16 + GradScaler（初始 scale {scaler.get_scale():.0f}）", fh)

    total_steps = args.steps if args.steps > 0 else 10 ** 7
    start_step = 0
    best = -1e9
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:                       # 参数组变化时尽力而为
            log(f"resume: optimizer 状态未加载 ({e})", fh)
        if scaler is not None and ck.get("scaler"):
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception as e:
                log(f"resume: scaler 状态未加载 ({e})", fh)
        start_step = int(ck.get("step", 0))
        total_steps = int(ck.get("total_steps", total_steps))
        best = float(ck.get("best", -1e9))
        if args.steps > 0:                           # 显式 --steps 覆盖 ckpt 里的总步数
            total_steps = args.steps
        log(f"resume from step {start_step}/{total_steps}", fh)

    params = [p for p in model.parameters() if p.requires_grad]
    micro = max(1, int(args.accum))

    def make_loader(dataset, at_step: int):
        sampler = PlanBatchSampler(dataset.n_clips, args.batch, steps=10 ** 7,
                                   rank=rank, world=world, seed=args.seed,
                                   start=at_step * micro)
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, num_workers=args.workers,
            collate_fn=collate, pin_memory=False,
            persistent_workers=args.workers > 0,
            prefetch_factor=(4 if args.workers > 0 else None))

    loader = make_loader(ds, start_step)
    it = iter(loader)

    def run_val(step: int):
        # 验证只让 rank0 做（其它 rank 在下一步 all_reduce 处等待即可），结果只写一份
        if rank != 0 or not (args.val_every > 0 and val_items):
            return
        was_training = model.training
        model.eval()
        m = evaluate(model, args.val_root, val_items, args, dev)
        model.train(was_training)
        log("VAL " + " ".join(f"{k}={v:.3f}" for k, v in sorted(m.items())), fh)
        write_val(os.path.join(args.out, "val.jsonl"), step, m)

    t0 = time.time()
    step = start_step
    if start_step == 0:
        run_val(0)                                   # 新训练在 step 0 先测一次基线
    acc = 0
    run_loss = run_psnr = 0.0
    run_n = 0
    tok_s = time.time()
    hist_steps: List[float] = []
    while step < total_steps:
        # 到点停止：--hours 是每卡各自计时的，world>1 时用一次小 all_reduce 对齐退出点，
        # 否则一卡先退、另一卡会永久卡在下一步的 sync all_reduce 上
        stop = time.time() - t0 > args.hours * 3600 and step > start_step + 20
        if dist is not None:
            st = torch.tensor([int(stop)], dtype=torch.int32, device=dev)
            dist.all_reduce(st, op=dist.ReduceOp.MAX)
            stop = bool(int(st.item()))
        if stop:
            log(f"时间到 ({args.hours}h)，停止于 step {step}", fh)
            break
        batch = next(it)
        x = batch["x"].to(dev, non_blocking=True)
        args._lw = loss_mode_at(step, total_steps, args)
        with autocast_ctx(dev, bool(args.amp)):
            if args.stage == "image":
                out = forward_image(model, x, args)
            else:
                out = forward_video(model, x, batch["n_warm"], args)
            loss = out["loss"] / micro
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        acc += 1
        run_loss += out["loss"].item()
        run_psnr += float(out["psnr"])
        run_n += 1
        if acc < micro:
            continue
        acc = 0
        # ---- 一步优化
        if scaler is not None:
            scaler.unscale_(opt)                     # 先还原梯度再裁剪/同步
        if args.grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            if step % args.log_every == 0 and scaler is not None:
                log(f"grad_norm(before clip) {float(gn):.3f}", fh)
        if world > 1:
            sync_grads(params, world, bool(args.sync_bf16))
        scale = lr_scale(step, total_steps, args.warmup)
        for g in opt.param_groups:
            g["lr"] = g["base"] * scale
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        hist_steps.append(time.time() - tok_s)
        tok_s = time.time()
        if step % args.log_every == 0:
            sps = sum(hist_steps) / max(1, len(hist_steps))
            log(f"step {step} loss {run_loss/run_n:.4f} psnr {run_psnr/run_n:.2f} "
                f"lr {opt.param_groups[0]['lr']:.2e} {sps:.2f}s/step "
                f"elapsed {(time.time()-t0)/3600:.2f}h n_warm {batch['n_warm']} "
                f"mem {mem_gb(dev):.1f}G", fh)
            if rank == 0:
                with open(os.path.join(args.out, "log.jsonl"), "a") as f:
                    f.write(json.dumps(dict(step=step, loss=run_loss / run_n,
                                            psnr=run_psnr / run_n,
                                            lr=opt.param_groups[0]["lr"], sps=sps,
                                            n_warm=batch["n_warm"],
                                            mem_gb=round(mem_gb(dev), 2),
                                            elapsed=(time.time() - t0) / 3600)) + "\n")
            run_loss = run_psnr = 0.0
            run_n = 0
        # ---- 自动总步数（按实测速度把 cosine 排到 --hours）；world>1 时 rank0 算完广播，
        # 保证两卡 cosine 的 total 一致（否则 lr 不同会发散）
        if args.steps <= 0 and step == start_step + args.auto_steps_after:
            if rank == 0 and len(hist_steps) >= 5:
                sps = sum(hist_steps[-20:]) / max(1, len(hist_steps[-20:]))
                remain = max(60.0, args.hours * 3600 - (time.time() - t0))
                total_steps = step + int(remain / max(1e-3, sps))
                log(f"auto total_steps={total_steps} (sps={sps:.2f}s)", fh)
            if dist is not None:
                ts = torch.tensor([total_steps], dtype=torch.int64, device=dev)
                dist.broadcast(ts, src=0)
                total_steps = int(ts.item())
        # ---- 重读数据索引（训练中数据持续追加）
        if args.reload_every > 0 and step % args.reload_every == 0:
            try:
                ds = RealClipDataset(args.data_root, args.index_name,
                                     clip_frames=args.clip_frames, window=args.window,
                                     img=args.img, hist_choices=hist_choices,
                                     p_cold=args.p_cold,
                                     img_per=(args.img_per if args.stage == "image" else 0),
                                     stage=args.stage)
                if args.val_root:
                    val_items = read_index(args.val_root, args.val_index_name)
                loader = make_loader(ds, step)
                it = iter(loader)
                log(f"reload data: train clips={ds.n_clips} val clips={len(val_items)}", fh)
            except Exception as e:
                log(f"reload 失败（继续用旧索引）: {e}", fh)
        # ---- 验证 / 存盘
        if args.val_every > 0 and step % args.val_every == 0:
            run_val(step)
        if rank == 0 and args.save_every > 0 and step % args.save_every == 0:
            ck = dict(step=step, cfg=cfg, model=model.state_dict(),
                      opt=opt.state_dict(), total_steps=total_steps, best=best)
            torch.save(ck, os.path.join(args.out, "last.pt"))
            torch.save(ck, os.path.join(args.out, f"step{step}.pt"))
            log(f"save ckpt step {step} -> last.pt step{step}.pt", fh)
    # ---- 收尾
    if rank == 0:
        ck = dict(step=step, cfg=cfg, model=model.state_dict(),
                  opt=opt.state_dict(), total_steps=total_steps, best=best,
                  scaler=(scaler.state_dict() if scaler is not None else None))
        torch.save(ck, os.path.join(args.out, "last.pt"))
        torch.save(dict(cfg=cfg, model=model.state_dict(), step=step),
                   os.path.join(args.out, "final_model.pt"))
        log(f"DONE step {step} -> {args.out}/last.pt", fh)
    if dist is not None:
        dist.barrier()
        dist.destroy_process_group()
    fh.close()


if __name__ == "__main__":
    main()
