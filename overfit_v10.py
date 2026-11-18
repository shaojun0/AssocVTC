#!/usr/bin/env python
"""v10 单片段过拟合对照：循环(step) vs 并行(prefix) 读出架构。

为什么（2026-11-18 用户安排）
---------------------------
先用「单片段过拟合」确认循环架构到底能不能拟合，再在「并行已经拟合」的权重
上切到循环架构（课程学习），然后同条件对照两种架构 —— 单片段上过拟合是唯一
能把「是不是这个架构/优化器根本学不动」与「泛化差」分开的实验。

四个独立单卡作业（一卡一个进程；MUSA 上不要多进程同卡，见 build_latent_cache 注释）
--------------------------------------------------------------------------------
  prefix      并行基线：cross-attn over 已读前缀（当前正式训练用的架构）
  step_nocopy 纯循环：memory=当步 z_t，递推 Y 是搬运历史的**唯一**通路
  step_copy   循环 + 直抄残差：已读帧直接抄（递推只负责未读帧）
  step_warm   从并行正式训练 best_trainable.pt 初始化后切循环（课程学习）

指标（每 --eval_every 步，整段 30 步，与训练同 bf16 前向；最后一步 = 读全量）
  l1_all     末档全片段 latent L1（17 行全算）
  psnr       末档经冻结 R 出像素、对原始帧的 PSNR
  tok_full   冻结分词器自身末档 PSNR（该片段的「读全量」上界）
  l1_whole   逐档全帧 latent L1 曲线（均值）
  l1_read    逐档「已读帧」latent L1 曲线（均值；loss_read=1 时这才是训练指标）

用法（单卡，独立进程启动）::

    SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0 python overfit_v10.py \
        --arch prefix --steps 800 --out /root/autodl-tmp/tok448_v2/overfit/prefix
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v10 import V10Config, VideoV10, psnr_from_l1     # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v10 单片段过拟合（架构对照）")
    p.add_argument("--tok_ckpt", type=str,
                   default="/root/autodl-tmp/tok448_v2/s2/ckpt_best.pt")
    p.add_argument("--latent_cache", type=str,
                   default="/root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train")
    p.add_argument("--data_root", type=str,
                   default="/root/autodl-tmp/video_recon/data3",
                   help="只用于取该片段的原始帧算 PSNR")
    p.add_argument("--clip", type=int, default=0,
                   help="缓存全局下标（= ClipDataset train 下标，缓存是按同序切的）")
    p.add_argument("--arch", choices=["prefix", "step"], default="prefix")
    p.add_argument("--copy_skip", type=int, default=1)
    p.add_argument("--loss_read", type=int, default=1)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--init_ckpt", type=str, default="",
                   help="best_trainable.pt：只加载 middle/readout（课程学习用）")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_psnr", action="store_true",
                   help="不解码原始片段（只报 latent 指标，快一点）")
    return p.parse_args()


def set_sdpa_math() -> None:
    """MUSA 上 flash 反向对 head_dim%8!=0 数值错，统一 math（见 11-17 报告）。"""
    for name in ("flash", "mem_efficient", "cudnn"):
        try:
            getattr(torch.backends.cuda, f"enable_{name}_sdp")(False)
        except Exception:
            pass
    try:
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def build_cfg(a: argparse.Namespace) -> V10Config:
    return V10Config(
        tok_ckpt=a.tok_ckpt, frames=120, group=4,
        rate_tokens=16, rate_dim=448, target_tokens=16, chunk_tokens=2,
        mid_dim=0, mid_depth=6, mid_heads=16, mid_ratio=2.0,
        dec_dim=0, dec_depth=4, dec_heads=16, dec_ratio=2.0, dec_drop=0.0,
        step_gate=True, step_weights="mean",
        mem_mode=a.arch, loss_read=bool(a.loss_read),
        copy_skip=bool(a.copy_skip))


def load_cached_clip(cache_dir: str, idx: int) -> torch.Tensor:
    """缓存全局下标 idx → (1,F,1+K,dim) float32。布局见 build_latent_cache.py。"""
    with open(os.path.join(cache_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    world = int(meta["world"])
    r, row = int(idx) % world, int(idx) // world
    mm = np.load(os.path.join(cache_dir, f"rank{r}.npy"), mmap_mode="r")
    arr = np.array(mm[row])                              # (F,1+K,dim) fp16
    return torch.from_numpy(arr).float()[None]


def evaluate(m: VideoV10, z_cls: torch.Tensor, z_s: torch.Tensor,
             x: torch.Tensor | None) -> dict:
    """整段 30 步读出 → 末档 latent L1 / PSNR + 逐档曲线（默认全部读全量）。"""
    cfg = m.cfg
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc = m.encode_rate_latents(z_cls, z_s)
        target = m.target_latents(enc["z_cls"], enc["z_s"])      # (1,F,T,c)
        preds = m.readout(enc["z"], n_steps=None,
                          return_step_preds=True)["preds"]       # (1,S,F,T,c)
        err = (preds.float() - target.float().unsqueeze(1)).abs()
        per_frame = err.mean(dim=(3, 4))[0]                      # (S,F)
        S, F = per_frame.shape
        grp = int(cfg.group)
        nf = torch.clamp((torch.arange(S, device=per_frame.device) + 1) * grp,
                         max=F)                                   # (S,)
        frame = torch.arange(F, device=per_frame.device)
        mask = (frame[None, :] < nf[:, None]).float()             # (S,F)
        l1_whole = per_frame.mean(dim=1)                          # (S,)
        l1_read = (per_frame * mask).sum(1) / mask.sum(1).clamp_min(1)
        out = dict(
            l1_all=float(per_frame[-1].mean()),
            l1_whole=[round(float(v), 4) for v in l1_whole.tolist()],
            l1_read=[round(float(v), 4) for v in l1_read.tolist()],
        )
        if x is not None:
            # render 要 (B,F,·,c)：单片段也必须有 batch 维（少一个维度会被
            # reshape 成 (F*K, ...) 然后 matmul 直接报形状错）
            p = preds[0, -1][None]                               # (1,F,1+K,c)
            pred_pix = m.tokenizer.render(p[:, :, :1].contiguous(),
                                          p[:, :, 1:].contiguous())
            gt = target[0][None]
            gt_pix = m.tokenizer.render(gt[:, :, :1].contiguous(),
                                        gt[:, :, 1:].contiguous())
            out["psnr"] = float(psnr_from_l1(pred_pix, x))
            out["tok_full"] = float(psnr_from_l1(gt_pix, x))
    m.train()
    return out


def main() -> None:
    a = parse_args()
    set_sdpa_math()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda", 0)

    cfg = build_cfg(a)
    m = VideoV10(cfg).to(dev)
    for p in m.tokenizer.parameters():
        p.requires_grad_(False)
    m.train()

    if a.init_ckpt:
        blob = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        r1 = m.middle.load_state_dict(blob["middle"], strict=False)
        r2 = m.readout.load_state_dict(blob["readout"], strict=False)
        print(f"[init] {a.init_ckpt} step={blob.get('step')} "
              f"mid_missing={len(r1.missing_keys)} read_missing={len(r2.missing_keys)} "
              f"read_unexpected={len(r2.unexpected_keys)}", flush=True)

    t = load_cached_clip(a.latent_cache, a.clip).to(dev)
    z_cls, z_s = t[:, :, :1], t[:, :, 1:]                   # (1,F,1,c)/(1,F,K,c)
    x = None
    if not a.no_psnr:
        from data_v10 import ClipDataset
        x = ClipDataset(a.data_root, "train", frames=cfg.frames,
                        img=448)[a.clip].to(dev)[None]      # (1,F,3,H,W)
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay)
    print(f"[cfg] arch={cfg.mem_mode} copy_skip={cfg.copy_skip} "
          f"loss_read={cfg.loss_read} clip={a.clip} steps={a.steps} lr={a.lr} "
          f"trainable={sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)

    rows = []
    t0 = time.time()
    for it in range(a.steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = m(z_cls=z_cls, z_s=z_s)
            loss = res["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, a.grad_clip)
        opt.step()

        if it % a.eval_every == 0 or it == a.steps:
            ev = evaluate(m, z_cls, z_s, x)
            row = dict(it=it, loss=round(float(loss.detach()), 5), **ev)
            rows.append(row)
            print(f"[{a.arch}{'+copy' if cfg.copy_skip else ''}] "
                  f"it {it:4d} loss {row['loss']:.5f} "
                  f"l1_all {row['l1_all']:.4f} "
                  f"psnr {row.get('psnr', float('nan')):.2f} "
                  f"tok_full {row.get('tok_full', float('nan')):.2f} "
                  f"el {(time.time()-t0)/60:.1f}min", flush=True)
            with open(os.path.join(a.out, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save(dict(
        middle=m.middle.state_dict(), readout=m.readout.state_dict(),
        step=a.steps, metrics=rows[-1], cfg=asdict(m.cfg),
        tok_ckpt=a.tok_ckpt),
        os.path.join(a.out, "trainable.pt"))
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, ensure_ascii=False, indent=2)
    print(f"[done] {a.out}  {rows[-1]}", flush=True)


if __name__ == "__main__":
    main()
