"""
训练脚本 — AssocVTC 实验版 (linattn): 无预训练编码器
=========================================================

从零训练 `model_v2_linattn.SRPhase1LinAttn`（线性注意力编码器 ~31.9M +
普通注意力解码器 ~31.2M）在 ImageNet-100 上做 224×224 → 224×224 像素重建。

    torchrun --nproc_per_node=1 train_linattn.py --output_dir out/linattn_k64

单卡即可（本仓库实验版无 DDP 需求; 多卡时每卡独立跑不同配置更方便对照）。
训练量: 121,689 train / 5,000 val（100 类 × 50 张等距抽取）;
batch 128 时 drop_last ⇒ **950 步/epoch**, 60 epoch = 57,000 步 ≈ 5.7h @ ~357 img/s。

关键超参（默认值）:
    lr 3e-4 (AdamW, cosine + warmup 5%), wd 0.05, grad_clip 1.0, bf16 autocast,
    L1 像素损失（归一化空间, 与主线同口径）。

产出（output_dir 下）:
    ckpt_last.pt / ckpt_best.pt   —— 含 cfg / args / 指标
    train_log.jsonl               —— 逐步日志
    status.json                   —— 监控用快照（step/epoch/loss/psnr/eta）
    samples/*.png                 —— 输入 | 重建 对照网格
    eval_val.json                 —— 最后一次验证的完整指标（含基线）
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from data_linattn import build_loaders
from metrics_linattn import psnr, ssim, per_patch_dc_baseline
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, LinAttnConfig,
                              SRPhase1LinAttn, patches_to_image,
                              parse_loss_weights)


# ─────────────────────────── 工具 ───────────────────────────

def param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def log_jsonl(path: str, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_status(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def denorm01(pix: torch.Tensor, hw: int, patch: int) -> torch.Tensor:
    """(B,N,3pp) 归一化像素 patch → (B,3,hw,hw) **[0,1]**。

    ⚠️ 单位陷阱（踩过）: `IMAGENET_MEAN/STD` 是 **[0,1] 标度**（0.485/0.229），
    所以 `patches_to_image` 反归一化后直接就是 [0,1]，**不能再除以 255**。
    反之 `model_v2.patches_to_image` 的 mean/std 口径是 0-255 标度（所以那里
    才需要 /255）。本函数用 clamp_255=False 关掉 0-255 截断，改成显式
    clamp(0,1)（预测可能越界, 截断到有效像素范围后再算 PSNR/SSIM 才公平）。
    """
    img = patches_to_image(pix, hw, hw, patch, IMAGENET_MEAN, IMAGENET_STD,
                           clamp_255=False)
    return img.clamp(0.0, 1.0)


def save_pair_grid(inp01: torch.Tensor, rec01: torch.Tensor, path: str, n: int = 8):
    """上排原图 / 下排重建, 各 n 张。"""
    n = min(n, inp01.shape[0])
    a = make_grid(inp01[:n], nrow=n, padding=2, value_range=(0, 1))
    b = make_grid(rec01[:n], nrow=n, padding=2, value_range=(0, 1))
    canvas = torch.cat([a, torch.ones(3, 4, a.shape[2], device=a.device), b], dim=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(canvas, path)


# ─────────────────────────── 验证 ───────────────────────────

@torch.no_grad()
def evaluate(model, loader, patch: int, img_size: int, amp: bool,
             max_batches: int = 0, save_grid: str = "", max_images: int = 0):
    model.eval()
    # 按图数加权（与 eval_linattn 同口径; 早期按批等权会系统性偏低 ~0.06dB）
    tot = {"l1": 0.0, "p": 0.0, "s": 0.0, "dc": 0.0, "c": 0.0}
    n_img = 0
    grid_in, grid_rec = [], []
    const01 = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    for bi, (x, _) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = x.cuda(non_blocking=True)
        if max_images and n_img >= max_images:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        Bb = x.shape[0]
        tot["l1"] += float(F.l1_loss(out["F_hat"].float(),
                                     out["target_pix"].float())) * Bb
        rec01 = denorm01(out["F_hat"].float(), img_size, patch)
        tgt01 = denorm01(out["target_pix"].float(), img_size, patch)
        # ★运行时护栏: 反归一化单位口径一旦再写错（多除 255）, 目标图均值会从
        #   ~0.45 掉到 ~0.0018, PSNR/SSIM 会整体虚高 ~48dB —— 直接报错而不是
        #   产出一份"看着很漂亮"的错误指标（2026-10-15 独立审查 finding 3）。
        assert 0.02 < float(tgt01.mean()) < 0.98, (
            f"反归一化单位口径疑似错误: tgt01.mean()={float(tgt01.mean()):.6f} "
            f"(真实图像应≈0.45)")
        tot["p"] += float(psnr(rec01, tgt01).sum())
        tot["s"] += float(ssim(rec01, tgt01).sum())
        tot["dc"] += float(per_patch_dc_baseline(tgt01, patch).sum())
        # 常数均值图基线: 拿"ImageNet 均值图"去和**真实 val 目标**比（不是和黑图比）
        tot["c"] += float(psnr(const01.expand_as(tgt01).to(tgt01.device),
                               tgt01).sum())
        n_img += x.shape[0]
        if save_grid and len(grid_in) * x.shape[0] < 8:
            grid_in.append(tgt01[:4].cpu())
            grid_rec.append(rec01[:4].cpu())
    model.train()
    res = {
        "n_images": n_img,
        "l1_norm": tot["l1"] / max(1, n_img),
        "psnr": tot["p"] / max(1, n_img),
        "ssim": tot["s"] / max(1, n_img),
        "psnr_mean_image_baseline": tot["c"] / max(1, n_img),
        "psnr_patch_dc_baseline": tot["dc"] / max(1, n_img),
        "pred_clamped_to_01": True,   # 指标用 clamp(0,1) 后的预测（基线天然在域内）
    }
    res["psnr_gain_over_patch_dc"] = res["psnr"] - res["psnr_patch_dc_baseline"]
    if save_grid and grid_in:
        save_pair_grid(torch.cat(grid_in), torch.cat(grid_rec), save_grid, 8)
    return res


# ─────────────────────────── 主流程 ───────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--output_dir", default="output/linattn_k64")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--num_specials", type=int, default=64, help="K: z_s 压缩 token 数")
    ap.add_argument("--special_init_std", type=float, default=0.02,
                    help="special token 初始化标准差（默认 0.02 = 原行为; 2026-10-22 "
                         "K≥128 A 相坍缩诊断的对称性探针）")
    # 编码器
    ap.add_argument("--enc_dim", type=int, default=448)
    ap.add_argument("--enc_depth", type=int, default=13)
    ap.add_argument("--enc_heads", type=int, default=8)
    ap.add_argument("--enc_mlp_ratio", type=float, default=4.0)
    ap.add_argument("--enc_dwconv", type=int, default=1)
    ap.add_argument("--enc_qk_norm", type=int, default=1,
                    help="线性注意力 QK-Norm（稳定性修复 1, 见模型 docstring）")
    ap.add_argument("--enc_attn", default="linear",
                    choices=["linear", "linattn", "std", "standard", "softmax"],
                    help="编码器注意力形式: linear=昨日臂(线性注意力, 默认); "
                         "std=**普通 transformer** softmax 缩放点积注意力（消融臂）")
    ap.add_argument("--enc_std_qk_norm", type=int, default=0,
                    help="enc_attn=std 时的 QK-Norm（默认 0 = 普通 transformer 原貌）")
    ap.add_argument("--dec_recurrent", type=int, default=0,
                    help="1 = 循环读出（主线口径, 每步只读自己那块 z_s, BPTT carry）")
    ap.add_argument("--dec_steps", default="",
                    help="逗号分隔采样步; 空 = square_block_starts(K)（K=64 → 8 步）")
    ap.add_argument("--step_plan", default="square",
                    choices=["square", "fixed", "cumulative"],
                    help="读窗口计划: square=默认主线口径; fixed=每步固定 block 个; "
                         "cumulative=dec_steps 解释为每步累计 z_s 数（固定 T 步 × 总预算 K "
                         "的消融, 生成器 model_v2_linattn.scaled_square_cumsum）")
    ap.add_argument("--block", type=int, default=0,
                    help="plan=fixed 时每步读入的 z_s 数（默认 0 = 不用）")
    ap.add_argument("--allow_k_gt_n", type=int, default=0,
                    help="1 = 显式允许 K>N（预算超过 patch 数, 已不是压缩; "
                         "仅 2026-10-21 的 K=256 探针臂）")
    ap.add_argument("--dec_attn", default="std", choices=["std", "linear"],
                    help="解码器注意力形式: std=nn.TransformerDecoder softmax（默认）; "
                         "linear=自注意力与交叉注意力都用线性注意力（**全线性**臂）")
    ap.add_argument("--dec_linear_qk_norm", type=int, default=1,
                    help="dec_attn=linear 时的 QK-Norm（默认 1 = 线性注意力稳定性修复）")
    ap.add_argument("--dec_grad_ckpt", type=int, default=0,
                    help="dec_attn=linear 时对解码器层做梯度检查点（数学等价, 反向重算; "
                         "全线性+BPTT 在 bs64 会 OOM, 置 1 可放进显存）")
    ap.add_argument("--freeze_encoder", type=int, default=0,
                    help="1 = 冻结编码器（只训解码器; 逐步精炼实验用）")
    # ── E-B1..B4（全部「新增 + 默认关」; 见 PLAN_alit_adaptive_experiments.md §4/§6）──
    ap.add_argument("--loss_weights", nargs="*", default=[],
                    help="E-B1 逐迭代损失权重: 逗号或空格分隔（如 "
                         "'2.5,2,1.5,1.25,1,1,1,1'）; 空 = 平权（现状）; "
                         "也支持 '--loss_weights 2.5 2 1.5 ...'; 长度须 = T 或 1（广播）")
    ap.add_argument("--dh_mode", type=int, default=0, choices=[0, 1, 2, 3],
                    help="E-B2 损失侧动态停机: 0=关(现状); 1=硬区域掩码(第t步只监督"
                         "region_t 行区); 2=软停机(按上一步逐patch误差的每图分位); "
                         "3=2 + 已完成 patch 的 carry stop-gradient")
    ap.add_argument("--dh_quantile", type=float, default=0.5,
                    help="dh_mode 2/3 的分位阈值: 0.5 = 保留误差最大的 50%% patch")
    ap.add_argument("--sample_grad_iters", type=int, default=0,
                    help="E-B4 随机预算采样: >0 时每步随机抽 2..N 个迭代算 loss "
                         "（其余步只前向）; 0 = 全步（现状）")
    ap.add_argument("--refiner_layers", type=int, default=0,
                    help="E-B3 读侧生长 GrowthBlock 层数: 0=关(现状), 1–2 "
                         "（d=256 时 1 层≈1.05M / 2 层≈1.84M）")
    ap.add_argument("--refiner_dim", type=int, default=256, help="GrowthBlock 内部维度")
    ap.add_argument("--refiner_heads", type=int, default=8)
    ap.add_argument("--refiner_mlp_ratio", type=float, default=4.0)
    ap.add_argument("--refiner_grow", type=int, default=16,
                    help="每步新增的生长 token 数（解码器读集合 = [z_s; L_1..L_it]）")
    ap.add_argument("--refiner_steps", type=int, default=0,
                    help="生长步数; 0 = 跟随解码器步数 T")
    ap.add_argument("--init_from", default="",
                    help="warm start ckpt（循环版可从单步版 ckpt 热启动, 只缺零初始化 mem_pos）")
    # 解码器
    ap.add_argument("--dec_dim", type=int, default=512)
    ap.add_argument("--dec_depth", type=int, default=7)
    ap.add_argument("--dec_heads", type=int, default=8)
    ap.add_argument("--dec_mlp_ratio", type=float, default=4.0)
    ap.add_argument("--pixel_hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    # 优化
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.08)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="梯度累积步数（默认 1 = 原行为逐位不变）。>1 时 loader 批量 = "
                         "batch_size/accum_steps, **等效批量与优化步数都不变**"
                         "（每步梯度 = accum 个 micro-batch 均值, 数学等价）, 只为把 "
                         "BPTT 步数很大（T=16/24/32）时的激活显存放下来 "
                         "（2026-10-23「固定 n=2 × T=16/24/32」臂: bs64 在 T=32 会 OOM）")
    ap.add_argument("--skip_grad", type=float, default=50.0,
                    help="clip 前 grad_norm 超过该值就跳过本步更新（0=关）; "
                         "线性注意力低精度间歇爆炸的护栏, 见模型 docstring")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--sample_interval", type=int, default=2000)
    ap.add_argument("--save_every_epochs", type=int, default=1)
    # 控制
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时提前停（冒烟）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时限制训练集大小（冒烟）")
    ap.add_argument("--time_budget_h", type=float, default=0.0, help=">0 时到点保存退出")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--smoke", action="store_true", help="冒烟: 快速配置 + 小数据")
    return ap.parse_args()


def build_cfg(args) -> LinAttnConfig:
    """CLI args → `LinAttnConfig`（单独提出便于 CPU 单测: 不进 CUDA / 数据管线）。

    E-B1..B4 的新参数全部「空/0 = 默认关」⇒ 与改造前的 cfg 逐字段相同。
    """
    return LinAttnConfig(
        img_size=args.img_size, patch=args.patch, num_specials=args.num_specials,
        special_init_std=float(args.special_init_std),
        enc_dim=args.enc_dim, enc_depth=args.enc_depth, enc_heads=args.enc_heads,
        enc_mlp_ratio=args.enc_mlp_ratio, enc_dwconv=bool(args.enc_dwconv),
        enc_qk_norm=bool(args.enc_qk_norm), enc_attn=args.enc_attn,
        enc_std_qk_norm=bool(args.enc_std_qk_norm),
        dec_recurrent=bool(args.dec_recurrent), dec_steps=args.dec_steps,
        dec_step_plan=args.step_plan, dec_block=int(args.block),
        allow_k_gt_n=bool(args.allow_k_gt_n),
        dec_attn=args.dec_attn, dec_linear_qk_norm=bool(args.dec_linear_qk_norm),
        dec_grad_ckpt=bool(args.dec_grad_ckpt),
        dec_dim=args.dec_dim, dec_depth=args.dec_depth, dec_heads=args.dec_heads,
        dec_mlp_ratio=args.dec_mlp_ratio, pixel_hidden=args.pixel_hidden,
        dropout=args.dropout,
        # ── E-B1..B4（空/0 = 默认关 ⇒ 旧路径逐位不变）──
        loss_weights=tuple(parse_loss_weights(args.loss_weights)),
        dh_mode=int(args.dh_mode), dh_quantile=float(args.dh_quantile),
        sample_grad_iters=int(args.sample_grad_iters),
        refiner_layers=int(args.refiner_layers), refiner_dim=int(args.refiner_dim),
        refiner_heads=int(args.refiner_heads),
        refiner_mlp_ratio=float(args.refiner_mlp_ratio),
        refiner_grow=int(args.refiner_grow), refiner_steps=int(args.refiner_steps))


def load_init_from(model: nn.Module, path: str) -> dict:
    """warm start: 允许缺 `decoder.mem_pos`（单步 → 循环）与 `refiner.*`（E-B3 新增块）。

    其余 missing 一律报错（形状不符/配置不匹配必须立刻失败）; 有多余 key 也报错。
    返回 `{"missing": [...], "unexpected": [...]}` 便于调用方记录。
    """
    ck0 = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck0["model"], strict=False)
    print(f"[init_from] {path} → missing={list(missing)} "
          f"unexpected={list(unexpected)}", flush=True)
    assert not unexpected, f"ckpt 有多余 key: {unexpected[:5]}"
    allowed = {"decoder.mem_pos"} | {k for k in missing if k.startswith("refiner.")}
    assert set(missing) <= allowed, \
        f"除 mem_pos/refiner.* 外还有未加载参数（形状不符?）: " \
        f"{[k for k in missing if k not in allowed][:8]}"
    return {"missing": list(missing), "unexpected": list(unexpected)}


def main():
    args = parse_args()
    if args.smoke:
        args.limit = args.limit or 512
        args.max_steps = args.max_steps or 30
        args.val_batches = 2
        args.log_interval = 5
        args.num_workers = 8
        args.sample_interval = 20
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    status_path = os.path.join(args.output_dir, "status.json")
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=1)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = build_cfg(args)
    model = SRPhase1LinAttn(cfg).cuda()
    if args.init_from:
        load_init_from(model, args.init_from)
    if args.freeze_encoder:
        for prm in model.encoder.parameters():
            prm.requires_grad_(False)
        n_tr = sum(prm.numel() for prm in model.parameters() if prm.requires_grad)
        print(f"[freeze] 冻结编码器 ⇒ 可训参数 {n_tr/1e6:.3f} M "
              f"(原 {sum(prm.numel() for prm in model.parameters())/1e6:.3f} M)", flush=True)
    pc = model.count_params()
    n_steps = (len(model.decoder.steps) if getattr(model, "recurrent", False) else 1)
    print(f"[decode] recurrent={getattr(model,'recurrent',False)} "
          f"dec_attn={cfg.dec_attn} → {type(model.decoder).__name__} T={n_steps}"
          + (f" steps={model.decoder.steps}" if getattr(model,'recurrent',False) else ""),
          flush=True)
    print(f"[cfg] N={cfg.num_patches} K={cfg.num_specials} "
          f"enc={cfg.enc_dim}x{cfg.enc_depth} dec={cfg.dec_dim}x{cfg.dec_depth} "
          f"enc_attn={cfg.enc_attn} → {type(model.encoder).__name__}", flush=True)
    print(f"[alit] loss_weights={model.loss_weights or '(平权)'} "
          f"dh_mode={model.dh_mode} dh_quantile={model.dh_quantile} "
          f"sample_grad_iters={model.sample_grad_iters} "
          f"refiner={type(model.refiner).__name__ if model.refiner else 'off'} "
          f"(layers={cfg.refiner_layers} dim={cfg.refiner_dim} grow={cfg.refiner_grow} "
          f"steps={cfg.refiner_steps or 'T'})", flush=True)
    print(f"[params] {pc}", flush=True)

    accum = max(1, int(args.accum_steps))
    assert args.batch_size % accum == 0, \
        f"--batch_size({args.batch_size}) 须能被 --accum_steps({accum}) 整除"
    micro_bs = args.batch_size // accum
    if accum > 1:
        print(f"[accum] 等效批量 {args.batch_size} = micro {micro_bs} × {accum} "
              f"累积, 优化步数不变（val 每批 {micro_bs} × 批数×{accum} 保持同样张数）",
              flush=True)
    tr_loader, va_loader, tr_ds, va_ds = build_loaders(
        args.data_dir, args.img_size, micro_bs, args.val_per_class,
        args.num_workers, args.limit, args.val_batches * accum)
    steps_per_epoch = len(tr_loader) // accum
    total_steps = (args.max_steps if args.max_steps > 0
                   else steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_frac))
    print(f"[data] train={len(tr_ds)} val={len(va_ds)} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} "
          f"warmup={warmup}", flush=True)

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay),
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    start_step, start_epoch, best_psnr = 0, 0, -1e9
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        start_epoch = ck.get("epoch", 0)
        best_psnr = ck.get("best_psnr", -1e9)
        print(f"[resume] {args.resume} step={start_step} epoch={start_epoch} "
              f"best_psnr={best_psnr:.3f}", flush=True)

    def save_ckpt(path, step, epoch, best, tag=""):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "cfg": cfg.to_dict(), "args": vars(args), "step": step,
                    "epoch": epoch, "best_psnr": best, "tag": tag,
                    "params": pc}, path)

    t0 = time.time()
    run_imgs, run_t0 = 0, time.time()
    n_skipped = 0
    loss_sum, rec_sum = 0.0, 0.0     # 一个优化步内 accum 个 micro-batch 的累计
    model.train()
    step = start_step
    stop = False
    epoch = start_epoch
    # 终止条件用**步数**（而非 epoch 数）: 这样 --time_budget_h 中途打断后从
    # ckpt 续跑也能正好补满 total_steps, cosine 能走到 min_lr
    # （2026-10-15 独立审查 finding 6: 旧版按 epoch 数终止会少跑 <1 epoch）。
    while step < total_steps and not stop:
        epoch += 1
        for bi, (x, _) in enumerate(tr_loader):
            if step >= total_steps:
                stop = True
                break
            lr = lr_at(step, total_steps, warmup, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.cuda(non_blocking=True)
            if bi % accum == 0:
                opt.zero_grad(set_to_none=True)
                loss_sum, rec_sum = 0.0, 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                out = model(x)
                loss = out["loss"]
            (loss / accum).backward()      # accum=1 时与原行为逐位相同
            loss_sum += float(loss.item())
            rec_sum += float(out["recon"].item())
            if (bi + 1) % accum != 0:
                continue                    # 累积未满 ⇒ 不做优化步
            if args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    [prm for prm in model.parameters() if prm.requires_grad],
                    args.grad_clip)
            else:
                gn = torch.tensor(0.0)
            # ★梯度爆炸跳步护栏（线性注意力稳定性修复 3）: 前向产出巨大 activation
            #   时参数本身还完好, 只要**不做这一步更新**就不会被毁（实测不设护栏时
            #   grad_norm 冲到 3e8, 一步就把 PSNR 从 19.6dB 打到 11.8dB 且不可恢复）。
            if args.skip_grad > 0 and float(gn) > args.skip_grad:
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
            else:
                opt.step()
            step += 1
            run_imgs += x.shape[0] * accum
            loss_avg = loss_sum / accum
            rec_avg = rec_sum / accum

            if step % args.log_interval == 0 or step - start_step <= 3:
                el = time.time() - t0
                ips = run_imgs / max(1e-6, time.time() - run_t0)
                eta_h = (total_steps - step) / max(1e-6, ips) * args.batch_size / 3600
                rec = {
                    "step": step, "epoch": epoch, "loss": round(loss_avg, 5),
                    "recon": round(rec_avg, 5),
                    "lr": lr, "grad_norm": round(float(gn), 4), "skipped": n_skipped, "T": n_steps,
                    "img_per_s": round(ips, 1), "elapsed_min": round(el / 60, 2),
                    "eta_h": round(eta_h, 2),
                    "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
                print(f"[{el/60:7.2f}m] step {step}/{total_steps} ep{epoch} "
                      f"loss={loss_avg:.4f} recon={rec_avg:.4f} "
                      f"lr={lr:.2e} gn={float(gn):.2f} skip={n_skipped} {ips:.0f}img/s "
                      f"vram={rec['vram_gb']:.1f}G eta={eta_h:.2f}h", flush=True)
                log_jsonl(log_path, rec)
                write_status(status_path, {**rec, "best_psnr": best_psnr,
                                           "running": True})

            if args.sample_interval and step % args.sample_interval == 0:
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                        o = model(x[:8])
                    g_in = denorm01(o["target_pix"].float(), args.img_size, args.patch)
                    g_rec = denorm01(o["F_hat"].float(), args.img_size, args.patch)
                    save_pair_grid(g_in, g_rec,
                                   os.path.join(args.output_dir, "samples",
                                                f"step{step:07d}.png"))

            if args.time_budget_h > 0 and (time.time() - t0) / 3600 >= args.time_budget_h:
                print(f"[stop] 达到时间预算 {args.time_budget_h}h", flush=True)
                stop = True
                break

        # ── epoch 末验证 + 存档 ──
        if args.val_batches > 0 and (step > start_step or epoch > 1):
            vr = evaluate(model, va_loader, args.patch, args.img_size,
                          bool(args.amp), args.val_batches * accum,
                          save_grid=os.path.join(args.output_dir, "samples",
                                                 f"epoch{epoch:03d}_val.png"))
            vr.update({"step": step, "epoch": epoch, "split": "val"})
            print(f"[val] ep{epoch} step{step} l1={vr['l1_norm']:.4f} "
                  f"psnr={vr['psnr']:.2f}dB ssim={vr['ssim']:.4f} "
                  f"(mean基线 {vr['psnr_mean_image_baseline']:.2f}dB, "
                  f"patchDC基线 {vr['psnr_patch_dc_baseline']:.2f}dB)", flush=True)
            log_jsonl(log_path, vr)
            with open(os.path.join(args.output_dir, "eval_val.json"), "w") as f:
                json.dump(vr, f, ensure_ascii=False, indent=1)
            if vr["psnr"] > best_psnr:
                best_psnr = vr["psnr"]
                save_ckpt(os.path.join(args.output_dir, "ckpt_best.pt"),
                          step, epoch, best_psnr, "best")
            write_status(status_path, {"step": step, "epoch": epoch,
                                       "val_psnr": vr["psnr"], "val_l1": vr["l1_norm"],
                                       "best_psnr": best_psnr,
                                       "elapsed_min": round((time.time() - t0) / 60, 2),
                                       "running": not stop})
        if args.save_every_epochs and epoch % args.save_every_epochs == 0:
            save_ckpt(os.path.join(args.output_dir, "ckpt_last.pt"),
                      step, epoch, best_psnr, "last")

    save_ckpt(os.path.join(args.output_dir, "ckpt_last.pt"), step, epoch,
              best_psnr, "last")
    if best_psnr < -1e8 and args.val_batches == 0:
        save_ckpt(os.path.join(args.output_dir, "ckpt_best.pt"), step, epoch,
                  best_psnr, "notvalidated")
    dt = (time.time() - t0) / 60
    print(f"[done] steps={step} epochs={epoch} best_val_psnr={best_psnr:.2f} "
          f"wall={dt:.1f}min", flush=True)
    write_status(status_path, {"step": step, "epoch": epoch,
                               "best_psnr": best_psnr,
                               "elapsed_min": round(dt, 2), "running": False,
                               "done": True})


if __name__ == "__main__":
    main()
