"""
训练脚本 — AssocVTC 实验版 (linattn): 无预训练编码器
=========================================================

从零训练 `model_v2_linattn.SRPhase1LinAttn`（线性注意力编码器 ~31.9M +
普通注意力解码器 ~31.2M）在 ImageNet-100 上做 224×224 → 224×224 像素重建。

    torchrun --nproc_per_node=1 train_linattn.py --output_dir out/linattn_k64

**单卡默认路径逐位不变**; 设了 `WORLD_SIZE>1`（torchrun 注入）时自动启用 DDP:
    torchrun --nproc_per_node=2 train_linattn.py ...   # ⇒ 2 卡 DDP
`--batch_size` 始终是**全局等效批量**, 每卡 micro = batch_size/(accum_steps×world);
steps/epoch、优化器、调度、等效批量语义全部与单卡完全相同（2026-11-04 双卡提速改造）。

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
import contextlib
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import make_grid, save_image

from data_linattn import build_loaders
from metrics_linattn import psnr, ssim, per_patch_dc_baseline
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, LinAttnConfig,
                              SRPhase1LinAttn, patches_to_image,
                              parse_loss_weights)


# ─────────────────────── 分布式（DDP）辅助 ───────────────────────

# 单卡（或 rank0）时 _IS_MAIN=True ⇒ `_p` 与内置 `print` 行为完全一致。
_IS_MAIN = True


def _p(*a, **kw):
    """rank0-only 打印（单卡路径与 `print` 逐位等价, 保证旧行为不变）。"""
    if _IS_MAIN:
        print(*a, **kw)


def ddp_env():
    """读 torchrun 注入的分布式环境变量; 未设置 ⇒ (1, 0, 0) 单卡路径。"""
    def _i(key, default):
        v = os.environ.get(key, "")
        try:
            return int(v)
        except (TypeError, ValueError):
            return default
    return _i("WORLD_SIZE", 1), _i("RANK", 0), _i("LOCAL_RANK", 0)


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
    ap.add_argument("--zs_dim", type=int, default=0,
                    help="编码器末端 FC 瓶颈维数（2026-10-26; 0=关）。zs_dim=448 ⇒ "
                         "每个 z_s token 从 enc_dim 压到 448 维、每步 2×448=896 个数、"
                         "每图 32×448=14,336 个值（= 旧 63M 宽度）")
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
    ap.add_argument("--save_opt", type=int, default=1,
                    help="1 = ckpt 里存 AdamW 状态（默认, 与原行为一致）；0 = 只存 model/cfg/args"
                         "（2026-10-24 容量阶梯: 360M 档 ckpt 含 opt 约 11.6G/个, 数据盘放不下; "
                         "本线不 --resume, 评测/热启动只用 model 权重 ⇒ 不存 opt 无副作用）")
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
        refiner_grow=int(args.refiner_grow), refiner_steps=int(args.refiner_steps),
        zs_dim=int(getattr(args, "zs_dim", 0)))


def _init_bottleneck_warm_start(model: nn.Module, sd: dict) -> None:
    """`--zs_dim>0` 的 warm start 初始化（2026-10-26）。

    目标: 让"编码器末端 FC（enc_dim→zs_dim）+ 解码器 mem_proj（zs_dim→dec_dim）"
    这一对在初始化时**尽量逼近旧的 mem_proj（enc_dim→dec_dim）**，即最小化
    「多出来的瓶颈」带来的初始退化，然后全量微调。

    做法（对旧 `decoder.mem_proj.weight` W (dec_dim, enc_dim) 做 SVD）:
        W ≈ U_d · diag(S_d) · Vh_d          # 最优秩 zs_dim 近似（Eckart–Young）
        zs_proj.weight = zcls_proj.weight = Vh_d        # (zs_dim, enc_dim) 下投影
        decoder.mem_proj.weight = U_d · diag(S_d)       # (dec_dim, zs_dim) 上投影
    ⇒ 初始 `mem_proj(FC(z))` = W 的最优秩 zs_dim 近似（与旧 memory 同量级/同方向）。

    退化路径: 旧 ckpt 没有 `decoder.mem_proj.weight`（enc_dim==dec_dim 时 mem_proj 是
    Identity）⇒ FC 用截断恒等（取前 zs_dim 维）。
    """
    if getattr(model, "zs_proj", None) is None:
        return
    d = int(model.zs_dim)
    key = "decoder.mem_proj.weight"
    with torch.no_grad():
        if key in sd:
            W = sd[key].float()                       # (dec_dim, enc_dim)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            Vh_d = Vh[:d].contiguous()                # (d, enc_dim)
            U_d = (U[:, :d] * S[:d]).contiguous()     # (dec_dim, d)
            model.zs_proj.weight.copy_(Vh_d)
            model.zcls_proj.weight.copy_(Vh_d)
            model.zs_proj.bias.zero_()
            model.zcls_proj.bias.zero_()
            model.decoder.mem_proj.weight.copy_(U_d)
            # 初始近似误差（相对 Frobenius）: 反映瓶颈带来的初始退化
            approx = U_d @ Vh_d
            rel = float((W - approx).norm() / W.norm())
            _p(f"[init_from] zs_dim={d} 瓶颈 warm start: FC=SVD 前 {d} 个右奇异向量, "
                  f"mem_proj {tuple(W.shape)} → {tuple(U_d.shape)} "
                  f"（秩-{d} 近似相对误差 {rel:.3f}）", flush=True)
            return
        for m in (model.zs_proj, model.zcls_proj):
            m.weight.zero_()
            m.bias.zero_()
            for i in range(d):
                m.weight[i, i] = 1.0
    _p(f"[init_from] zs_dim={d} 瓶颈 warm start: FC=截断恒等（无 mem_proj 权重）",
          flush=True)


def load_init_from(model: nn.Module, path: str) -> dict:
    """warm start: 允许缺 `decoder.mem_pos`（单步 → 循环）/ `refiner.*`（E-B3 新增块）
    / `zs_proj.*`+`zcls_proj.*`（2026-10-26 编码器末端 FC 瓶颈）; 也允许
    `decoder.mem_proj.*` 的形状变化（瓶颈臂: (dec_dim,enc_dim) → (dec_dim,zs_dim)）。

    其余 missing 一律报错（形状不符/配置不匹配必须立刻失败）; 有多余 key 也报错。
    返回 `{"missing": [...], "unexpected": [...], "shape_mismatch": [...]}`。
    """
    ck0 = torch.load(path, map_location="cpu", weights_only=False)
    raw = dict(ck0["model"])
    want = model.state_dict()
    mism = {k: v for k, v in raw.items()
            if k in want and tuple(v.shape) != tuple(want[k].shape)}
    for k in mism:                       # 形状不符的先摘出来, 稍后按瓶颈规则重初始化
        raw.pop(k)
    if mism:
        _p(f"[init_from] 形状变化（按瓶颈规则重初始化）: "
              f"{ {k: (tuple(v.shape), tuple(want[k].shape)) for k, v in mism.items()} }",
              flush=True)
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if getattr(model, "zs_proj", None) is not None:
        _init_bottleneck_warm_start(model, ck0["model"])
    _p(f"[init_from] {path} → missing={list(missing)} "
          f"unexpected={list(unexpected)}", flush=True)
    assert not unexpected, f"ckpt 有多余 key: {unexpected[:5]}"
    allowed = ({"decoder.mem_pos"}
               | set(mism)                      # 形状变化: 已按瓶颈规则重初始化
               | {k for k in missing if k.startswith("refiner.")}
               | {k for k in missing if k.startswith(("zs_proj.", "zcls_proj."))})
    assert set(missing) <= allowed, \
        f"除 mem_pos/refiner.*/FC 瓶颈外还有未加载参数（形状不符?）: " \
        f"{[k for k in missing if k not in allowed][:8]}"
    return {"missing": list(missing), "unexpected": list(unexpected),
            "shape_mismatch": list(mism)}


def main():
    global _IS_MAIN
    args = parse_args()
    if args.smoke:
        args.limit = args.limit or 512
        args.max_steps = args.max_steps or 30
        args.val_batches = 2
        args.log_interval = 5
        args.num_workers = 8
        args.sample_interval = 20
    # ── 分布式（DDP）: 只有 torchrun 注入 WORLD_SIZE>1 才启用 ──────────────
    # 未设置时 world=1/rank=0 ⇒ 下面所有 `if ddp:` 分支都不执行, 单卡路径逐位不变。
    world, rank, local_rank = ddp_env()
    ddp = world > 1
    _IS_MAIN = (rank == 0)
    if ddp:
        backend = os.environ.get("DDP_BACKEND", "nccl")
        ndev = torch.cuda.device_count()
        # 正常 2 卡: local_rank 即物理卡号。同卡多 rank 的 plumbing 测试
        # (CUDA_VISIBLE_DEVICES 只暴露 1 张卡) 时取模, 两个 rank 都落 device 0。
        dev = local_rank if local_rank < ndev else local_rank % max(1, ndev)
        torch.cuda.set_device(dev)
        local_rank = dev
        dist.init_process_group(backend=backend)
        assert dist.get_world_size() == world, \
            f"torchrun WORLD_SIZE={world} ≠ 进程组 world_size={dist.get_world_size()}"
        # 每个 rank 各打一行（rank≠0 也打: 冒烟时用来确认两个进程都起来了/各自 device）
        print(f"[ddp] rank={rank}/{world} local_rank={local_rank} "
              f"backend={backend} device=cuda:{dev}"
              + (" （⚠️ 多 rank 共卡调试: 仅 gloo 可跑, NCCL 会拒绝 Duplicate GPU）"
                 if ndev < world else ""), flush=True)

    if _IS_MAIN:
        os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    status_path = os.path.join(args.output_dir, "status.json")
    if _IS_MAIN:      # 日志/args 只由 rank0 落盘
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
        _p(f"[freeze] 冻结编码器 ⇒ 可训参数 {n_tr/1e6:.3f} M "
              f"(原 {sum(prm.numel() for prm in model.parameters())/1e6:.3f} M)", flush=True)
    pc = model.count_params()
    n_steps = (len(model.decoder.steps) if getattr(model, "recurrent", False) else 1)
    _p(f"[decode] recurrent={getattr(model,'recurrent',False)} "
          f"dec_attn={cfg.dec_attn} → {type(model.decoder).__name__} T={n_steps}"
          + (f" steps={model.decoder.steps}" if getattr(model,'recurrent',False) else ""),
          flush=True)
    _p(f"[cfg] N={cfg.num_patches} K={cfg.num_specials} "
          f"enc={cfg.enc_dim}x{cfg.enc_depth} dec={cfg.dec_dim}x{cfg.dec_depth} "
          f"enc_attn={cfg.enc_attn} → {type(model.encoder).__name__} "
          f"zs_dim={getattr(cfg, 'zs_dim', 0)} "
          f"({pc['values_per_image']} 个值/图)", flush=True)
    _p(f"[alit] loss_weights={model.loss_weights or '(平权)'} "
          f"dh_mode={model.dh_mode} dh_quantile={model.dh_quantile} "
          f"sample_grad_iters={model.sample_grad_iters} "
          f"refiner={type(model.refiner).__name__ if model.refiner else 'off'} "
          f"(layers={cfg.refiner_layers} dim={cfg.refiner_dim} grow={cfg.refiner_grow} "
          f"steps={cfg.refiner_steps or 'T'})", flush=True)
    _p(f"[params] {pc}", flush=True)

    raw_model = model            # DDP 包装前的裸模型: state_dict / evaluate / 采样图用
    if ddp:
        model = DDP(raw_model, device_ids=[local_rank],
                    find_unused_parameters=True)

    accum = max(1, int(args.accum_steps))
    # ★ `--batch_size` 恒为**全局等效批量**: 每卡 micro = batch_size/(accum×world)。
    #   单卡时 world=1 ⇒ micro = batch_size/accum, 与改造前逐位相同。
    assert args.batch_size % (accum * world) == 0, \
        f"--batch_size({args.batch_size}) 须能被 accum_steps×world_size" \
        f"({accum}×{world}={accum * world}) 整除"
    micro_bs = args.batch_size // (accum * world)
    if accum > 1:
        _p(f"[accum] 等效批量 {args.batch_size} = micro {micro_bs} × {accum} "
              f"累积, 优化步数不变（val 每批 {micro_bs} × 批数×{accum} 保持同样张数）",
              flush=True)
    if ddp:
        _p(f"[ddp] 等效批量 {args.batch_size} = 每卡 micro {micro_bs} × 每卡 accum "
              f"{accum} × {world} 卡（与单卡 micro {args.batch_size // accum} × "
              f"accum {accum} 等价, 优化步数不变）", flush=True)
    # val 张数口径: 单卡 val 图数 = micro×accum×val_batches = batch_size×val_batches,
    # DDP 下换成每个 rank 的 micro/accum, 故批数要乘 world 才能保持同样张数。
    val_batches_n = args.val_batches * accum * world
    tr_loader, va_loader, tr_ds, va_ds = build_loaders(
        args.data_dir, args.img_size, micro_bs, args.val_per_class,
        args.num_workers, args.limit, val_batches_n)
    tr_sampler = None
    if ddp:
        # 每卡只看自己那份（DistributedSampler 自动垫齐到 ceil(N/world)）
        tr_sampler = DistributedSampler(tr_ds, num_replicas=world, rank=rank,
                                        shuffle=True, seed=args.seed,
                                        drop_last=False)
        tr_loader = DataLoader(tr_ds, batch_size=micro_bs, sampler=tr_sampler,
                               drop_last=True, num_workers=args.num_workers,
                               pin_memory=True,
                               persistent_workers=args.num_workers > 0,
                               prefetch_factor=4 if args.num_workers > 0 else None)
    steps_per_epoch = len(tr_loader) // accum
    total_steps = (args.max_steps if args.max_steps > 0
                   else steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_frac))
    _p(f"[data] train={len(tr_ds)} val={len(va_ds)} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} "
          f"warmup={warmup}", flush=True)
    if ddp:
        # ★等价批量语义守卫: steps/epoch 必须与「同样超参的单卡跑」完全一致,
        #   否则等效批量/优化步数被悄悄改了（每个 rank 各打一行, 供冒烟核对）。
        micro_single = args.batch_size // accum
        ref_single = (len(tr_ds) // micro_single) // accum
        assert steps_per_epoch == ref_single, (
            f"DDP steps/epoch={steps_per_epoch} ≠ 单卡参考 {ref_single}"
            f"（train={len(tr_ds)} micro={micro_bs} accum={accum} world={world}）")
        print(f"[ddp] rank={rank} sampler num_samples={len(tr_sampler)} "
              f"loader_batches={len(tr_loader)} micro={micro_bs} accum={accum} "
              f"⇒ steps/epoch={steps_per_epoch} == 单卡参考 {ref_single} ✓",
              flush=True)

    opt = torch.optim.AdamW(param_groups(raw_model, args.weight_decay),
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    start_step, start_epoch, best_psnr = 0, 0, -1e9
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        start_epoch = ck.get("epoch", 0)
        best_psnr = ck.get("best_psnr", -1e9)
        _p(f"[resume] {args.resume} step={start_step} epoch={start_epoch} "
              f"best_psnr={best_psnr:.3f}", flush=True)

    def save_ckpt(path, step, epoch, best, tag=""):
        blob = {"model": raw_model.state_dict(),   # 裸模型: 不带 DDP 的 "module." 前缀
                "cfg": cfg.to_dict(), "args": vars(args), "step": step,
                "epoch": epoch, "best_psnr": best, "tag": tag,
                "params": pc}
        if ddp:                                    # 单卡 ckpt 逐字节不变: 只在 DDP 下加此键
            blob["ddp"] = {"world_size": world, "rank": rank, "backend": backend,
                           "micro_bs": micro_bs, "accum": accum}
        if args.save_opt:
            blob["opt"] = opt.state_dict()      # 默认路径与原行为逐位相同
        torch.save(blob, path)

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
        if ddp:
            tr_sampler.set_epoch(epoch)   # 每 epoch 换一份不重复的全局打乱划分
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
            last_micro = ((bi + 1) % accum == 0)
            # DDP 梯度累积: 只有本优化步的最后一次 micro 才 allreduce, 其余 no_sync
            # （否则每个 micro 都同步一次, 通信量 ×accum 且语义不等价）。
            sync_ctx = (model.no_sync() if ddp and not last_micro
                        else contextlib.nullcontext())
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                    out = model(x)
                    loss = out["loss"]
                (loss / accum).backward()      # accum=1 时与原行为逐位相同
            loss_sum += float(loss.item())
            rec_sum += float(out["recon"].item())
            if not last_micro:
                continue                    # 累积未满 ⇒ 不做优化步
            if args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    [prm for prm in raw_model.parameters() if prm.requires_grad],
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
            run_imgs += x.shape[0] * accum * world   # DDP: ×world = 全局吞吐（单卡恒等）
            loss_avg = loss_sum / accum
            rec_avg = rec_sum / accum

            # 日志/status 只由 rank0 写（否则两个 rank 抢同一个 status.json.tmp，
            # os.replace 会 ENOENT；train_log.jsonl 也会被写两遍）。
            if _IS_MAIN and (step % args.log_interval == 0 or step - start_step <= 3):
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
                _p(f"[{el/60:7.2f}m] step {step}/{total_steps} ep{epoch} "
                      f"loss={loss_avg:.4f} recon={rec_avg:.4f} "
                      f"lr={lr:.2e} gn={float(gn):.2f} skip={n_skipped} {ips:.0f}img/s "
                      f"vram={rec['vram_gb']:.1f}G eta={eta_h:.2f}h", flush=True)
                log_jsonl(log_path, rec)
                write_status(status_path, {**rec, "best_psnr": best_psnr,
                                           "running": True})

            if _IS_MAIN and args.sample_interval and step % args.sample_interval == 0:
                # ⚠️ 用 raw_model: DDP 包装后的 forward 含 buffer 广播集合通信,
                #    只有 rank0 调用会 hang 住另一个 rank。
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                        o = raw_model(x[:8])
                    g_in = denorm01(o["target_pix"].float(), args.img_size, args.patch)
                    g_rec = denorm01(o["F_hat"].float(), args.img_size, args.patch)
                    save_pair_grid(g_in, g_rec,
                                   os.path.join(args.output_dir, "samples",
                                                f"step{step:07d}.png"))

            if args.time_budget_h > 0 and (time.time() - t0) / 3600 >= args.time_budget_h:
                _p(f"[stop] 达到时间预算 {args.time_budget_h}h", flush=True)
                stop = True
                break

        # ── epoch 末验证 + 存档 ──
        # val 只在 rank0 做（其余 rank 在 barrier 上等）⇒ 用 raw_model 规避 DDP
        # forward 的集合通信; 前后各一次 barrier 保证不会一个 rank 在 allreduce 上
        # 等另一个 rank 先去做 val。
        do_val = (args.val_batches > 0 and (step > start_step or epoch > 1))
        if ddp:
            dist.barrier()
        if do_val and _IS_MAIN:
            vr = evaluate(raw_model, va_loader, args.patch, args.img_size,
                          bool(args.amp), args.val_batches * accum * world,
                          save_grid=os.path.join(args.output_dir, "samples",
                                                 f"epoch{epoch:03d}_val.png"))
            vr.update({"step": step, "epoch": epoch, "split": "val"})
            _p(f"[val] ep{epoch} step{step} l1={vr['l1_norm']:.4f} "
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
        if ddp:
            dist.barrier()      # rank≠0 等 rank0 做完 val + ckpt/status 落盘
        if _IS_MAIN and args.save_every_epochs and epoch % args.save_every_epochs == 0:
            save_ckpt(os.path.join(args.output_dir, "ckpt_last.pt"),
                      step, epoch, best_psnr, "last")

    # 收尾: 只 rank0 落盘/打日志; 其它 rank 先 barrier 等落盘完再退出。
    if ddp:
        dist.barrier()
    if _IS_MAIN:
        save_ckpt(os.path.join(args.output_dir, "ckpt_last.pt"), step, epoch,
                  best_psnr, "last")
        if best_psnr < -1e8 and args.val_batches == 0:
            save_ckpt(os.path.join(args.output_dir, "ckpt_best.pt"), step, epoch,
                      best_psnr, "notvalidated")
        dt = (time.time() - t0) / 60
        _p(f"[done] steps={step} epochs={epoch} best_val_psnr={best_psnr:.2f} "
              f"wall={dt:.1f}min", flush=True)
        write_status(status_path, {"step": step, "epoch": epoch,
                                   "best_psnr": best_psnr,
                                   "elapsed_min": round(dt, 2), "running": False,
                                   "done": True})
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
