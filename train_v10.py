"""v10 训练脚本：冻结图像分词器（E + R），只训「中间态 + 整片段循环读出」。

用法（A800 上一条命令即可）::

    /root/venv_v31/bin/python train_v10.py \
        --data_root /root/autodl-tmp/video_recon/data3 \
        --tok_ckpt  /root/autodl-tmp/tok448/ckpt_best.pt \
        --out       /root/autodl-tmp/v10_wholeclip \
        --batch 2 --accum_steps 2 --lr 2e-4 --max_steps 20000

关键口径（详见 doc/2026-11-16/DESIGN_v10_video_whole_clip.md）
------------------------------------------------------------
* 冻结分词器：E 与 R 都是同一个 `SRPhase1LinAttn` ckpt（`--tok_ckpt`）；
  训练只更新 middle / compressor / readout，日志里有 `trainable` 参数量。
* 率 / 目标：率 = 每帧前 `--rate_tokens` 个 special token（默认 16），
  目标 = 全部 `--target_tokens` + z_cls（默认 32+1）。率 < 目标是**故意的**：
  否则上限就是逐帧冷启动基线，实验必然平局。
* 损失：逐步平权 latent L1（`--step_weights mean|lin|last`），深监督全部 30 步。
* 评测：`--psnr_eval 1` 时经冻结 R 渲染出像素 PSNR，同时报两个参照
  `psnr`（本模型）与 `psnr_tok_full`（冻结分词器自身末档 = 每帧冷启动上界）。

自检：``python train_v10.py --smoke``（小数据集/小步数，本地 CPU 也能跑）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict

import torch
from torch.utils.data import DataLoader

from data_v10 import ClipDataset
from model_v10 import V10Config, VideoV10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v10 全片段循环读出训练")
    # 数据
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp/video_recon/data3")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--img", type=int, default=448)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条（调试）")
    p.add_argument("--num_workers", type=int, default=8)
    # 分词器
    p.add_argument("--tok_ckpt", type=str, required=True)
    # 结构
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--rate_tokens", type=int, default=16)
    p.add_argument("--rate_dim", type=int, default=448)
    p.add_argument("--target_tokens", type=int, default=32)
    p.add_argument("--mid_dim", type=int, default=1024)
    p.add_argument("--mid_depth", type=int, default=6)
    p.add_argument("--mid_heads", type=int, default=16)
    p.add_argument("--mid_ratio", type=float, default=4.0)
    p.add_argument("--dec_dim", type=int, default=1024)
    p.add_argument("--dec_depth", type=int, default=4)
    p.add_argument("--dec_heads", type=int, default=16)
    p.add_argument("--dec_ratio", type=float, default=2.0)
    p.add_argument("--step_gate", type=int, default=1)
    p.add_argument("--step_weights", type=str, default="mean")
    # 优化
    p.add_argument("--batch", type=int, default=2, help="micro-batch（每步的片段数）")
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--amp", type=int, default=1)
    # 日志/评测/保存
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=2)
    p.add_argument("--psnr_eval", type=int, default=1)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true",
                   help="用随机张量代替真实片段（验通路用，不落盘数据）")
    return p.parse_args()


class SyntheticClips(torch.utils.data.Dataset):
    """随机片段：只用来验证训练通路/显存（真实训练必须用 --data_root）。"""

    def __init__(self, n: int, frames: int, img: int):
        self.n, self.frames, self.img = int(n), int(frames), int(img)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(1000 + int(i))
        return torch.rand(self.frames, 3, self.img, self.img, generator=g)


def build_cfg(a: argparse.Namespace) -> V10Config:
    return V10Config(
        tok_ckpt=a.tok_ckpt, frames=a.frames, group=a.group,
        rate_tokens=a.rate_tokens, rate_dim=a.rate_dim,
        target_tokens=a.target_tokens,
        mid_dim=a.mid_dim, mid_depth=a.mid_depth, mid_heads=a.mid_heads,
        mid_ratio=a.mid_ratio, dec_dim=a.dec_dim, dec_depth=a.dec_depth,
        dec_heads=a.dec_heads, dec_ratio=a.dec_ratio,
        step_gate=bool(a.step_gate), step_weights=a.step_weights)


def lr_at(step: int, total: int, base: float, min_lr: float, warm: int) -> float:
    if step < warm:
        return base * (step + 1) / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * min(t, 1.0)))


class Meter:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.step = 0
        self.loss = 0.0
        self.n = 0

    def add(self, loss: float) -> None:
        self.loss += float(loss)
        self.n += 1

    def reset(self) -> None:
        self.loss, self.n = 0.0, 0

    @property
    def mean(self) -> float:
        return self.loss / max(1, self.n)


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    cfg = build_cfg(a)
    log_path = os.path.join(a.out, "train_log.jsonl")
    status_path = os.path.join(a.out, "status.json")

    # ── 数据
    if a.synthetic:
        tr = SyntheticClips(32, a.frames, a.img)
        va = SyntheticClips(4, a.frames, a.img)
    else:
        tr = ClipDataset(a.data_root, "train", frames=a.frames, img=a.img,
                         limit=a.limit)
        va = ClipDataset(a.data_root, "val", frames=a.frames, img=a.img,
                         limit=max(8, 4 * a.val_batches))
    if a.smoke:
        a.max_steps = min(a.max_steps, 4)
        a.batch = min(a.batch, 1)
        a.num_workers = 0
        a.val_every = 2
        a.val_batches = 1
        a.psnr_eval = 0
    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True,
                       num_workers=a.num_workers, pin_memory=True,
                       persistent_workers=bool(a.num_workers))
    dl_va = DataLoader(va, batch_size=1, shuffle=False,
                       num_workers=min(2, a.num_workers), pin_memory=True)

    # ── 模型（分词器在 FrozenTokenizer 里已冻结）
    model = VideoV10(cfg).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay,
                            betas=(0.9, 0.95))
    warm = int(a.warmup_frac * a.max_steps)
    amp = bool(a.amp) and dev.type == "cuda"
    # bf16 不需要 GradScaler（与 fp16 不同，v9 那次静默失效的坑不在这里）
    print(f"[cfg] {json.dumps(asdict(cfg), ensure_ascii=False)}", flush=True)
    print(f"[model] trainable={n_train/1e6:.2f}M "
          f"{ {k: round(v/1e6, 2) for k, v in model.trainable_param_groups().items()} } "
          f"values/frame={cfg.values_per_frame()} values/clip={cfg.values_per_clip()} "
          f"positions={cfg.positions_per_clip}", flush=True)
    print(f"[data] train={len(tr)} val={len(va)} device={dev} amp={amp} "
          f"steps={a.max_steps} batch={a.batch}×{a.accum_steps}", flush=True)

    start_step = 0
    if a.resume and os.path.isfile(a.resume):
        blob = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=False)
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])
        start_step = int(blob.get("step", 0))
        print(f"[resume] {a.resume} @ step {start_step}", flush=True)

    meter = Meter()
    best = float("inf")
    it = iter(dl_tr)

    def save(tag: str, step: int, metrics: Dict) -> None:
        blob = dict(model=model.state_dict(), cfg=asdict(cfg), args=vars(a),
                    step=step, metrics=metrics)
        torch.save(blob, os.path.join(a.out, f"ckpt_{tag}.pt"))

    for step in range(start_step, a.max_steps):
        lr = lr_at(step, a.max_steps, a.lr, a.min_lr, warm)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(a.accum_steps):
            try:
                x = next(it)
            except StopIteration:
                it = iter(dl_tr)
                x = next(it)
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(x)
                loss = out["loss"] / a.accum_steps
            loss.backward()
            loss_acc += float(loss) * a.accum_steps
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.grad_clip)
        opt.step()
        meter.add(loss_acc)
        meter.step = step + 1

        if (step + 1) % a.log_every == 0:
            el = time.time() - meter.t0
            sps = (step + 1) / max(1e-9, el)
            eta_h = (a.max_steps - step - 1) / max(1e-9, sps) / 3600.0
            vram = (torch.cuda.max_memory_allocated() / 2**30
                    if dev.type == "cuda" else 0.0)
            rec = dict(step=step + 1, loss=round(meter.mean, 5),
                       lr=round(lr, 8), gn=round(float(gn), 3),
                       sps=round(sps, 3), eta_h=round(eta_h, 3),
                       vram_gb=round(vram, 2))
            print(f"[{el/60:6.2f}m] step {step+1}/{a.max_steps} "
                  f"loss={rec['loss']:.5f} lr={lr:.2e} gn={rec['gn']} "
                  f"{sps:.3f} it/s eta={eta_h:.2f}h vram={vram:.1f}G", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(status_path, "w") as f:
                json.dump(dict(step=step + 1, loss=rec["loss"], running=True,
                               eta_h=rec["eta_h"], it_per_s=rec["sps"]), f)
            meter.reset()

        # ── 验证
        do_val = (step + 1) % a.val_every == 0 or step + 1 == a.max_steps
        if do_val:
            model.eval()
            lat, ps, psf = [], [], []
            with torch.no_grad():
                for i, xb in enumerate(dl_va):
                    if i >= a.val_batches:
                        break
                    xb = xb.to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        ev = model.evaluate(xb, pixel_psnr=bool(a.psnr_eval))
                    lat.append(ev["latent_l1"])
                    if "psnr" in ev:
                        ps.append(ev["psnr"])
                        psf.append(ev["psnr_tok_full"])
            met = dict(latent_l1=round(sum(lat) / len(lat), 5))
            if ps:
                met["psnr"] = round(sum(ps) / len(ps), 4)
                met["psnr_tok_full"] = round(sum(psf) / len(psf), 4)
            print(f"[{step+1}] VAL {met}", flush=True)
            with open(os.path.join(a.out, "val.jsonl"), "a") as f:
                f.write(json.dumps(dict(step=step + 1, **met)) + "\n")
            if met["latent_l1"] < best:
                best = met["latent_l1"]
                save("best", step + 1, met)
            model.train()

        if (step + 1) % a.save_every == 0:
            save("last", step + 1, dict(latent_l1=best))
            with open(status_path, "w") as f:
                json.dump(dict(step=step + 1, loss=round(meter.mean, 5),
                               best_latent_l1=best, running=True), f)

    save("final", a.max_steps, dict(latent_l1=best))
    with open(status_path, "w") as f:
        json.dump(dict(step=a.max_steps, best_latent_l1=best, running=False,
                       done=True), f)
    print(f"[done] best latent_l1={best:.5f} out={a.out}", flush=True)


if __name__ == "__main__":
    main()
