"""v10 训练后评测：逐档（读前 t 组 z）的像素 PSNR / latent L1 曲线。

训练时的 `V10EvalCallback` 只报**末档**（读全 30 组）指标；本脚本把 30 档
全部渲染出来，画「率 vs 质量」曲线（论文的 stepwise 口径），并落逐片段 JSON
便于复核/画图。

用法（单卡，1~2 分钟/片段，60 片段约 10 min）::

    SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0 python eval_v10_stepwise.py \
        --trainable /root/autodl-tmp/tok448_v2/v10_120f/best_trainable.pt \
        --data_root /root/autodl-tmp/video_recon/data3 --n_clips 60 \
        --out /root/autodl-tmp/tok448_v2/v10_120f/eval_stepwise.json

`best_trainable.pt` 由 train_v10.py 的评测回调写出（只含 middle/readout），
分词器仍从 `cfg.tok_ckpt` 加载；也接受 Trainer 的 `final/` 目录（整模型）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v10 import V10Config, VideoV10, psnr_from_l1   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v10 逐档 PSNR 评测")
    p.add_argument("--trainable", type=str, required=True,
                   help="best_trainable.pt 或 Trainer final 目录")
    p.add_argument("--data_root", type=str,
                   default="/root/autodl-tmp/video_recon/data3")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--n_clips", type=int, default=60)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--dump_per_clip", type=int, default=1)
    p.add_argument("--no_fig", type=int, default=0)
    p.add_argument("--n_steps_max", type=int, default=0,
                   help="只评前 N 档（0 = 全 30 档）")
    return p.parse_args()


def _set_sdpa_math() -> None:
    for name in ("flash", "mem_efficient"):
        try:
            getattr(torch.backends.cuda, f"enable_{name}_sdp")(False)
        except Exception:
            pass
    try:
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def load_model(path: str) -> VideoV10:
    if os.path.isdir(path):
        from model_v10 import FrozenTokenizer
        cfg_path = os.path.join(path, "config.json")
        sd_path = os.path.join(path, "model.safetensors")
        if not os.path.isfile(cfg_path):                   # Trainer 的 final/ 布局
            cand = [f for f in os.listdir(path) if f.endswith(".json")]
            raise RuntimeError(f"{path} 里没有 config.json；有 {cand}")
        raise RuntimeError("请传 best_trainable.pt（final/ 目录需先转成 pt）")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = dict(blob.get("cfg") or {})
    # 2026-11-18 17:xx 之前训练的 ckpt 是「旧口径」读出：carry 有 LN、self-attn
    # 有帧级因果 mask。为让历史 ckpt 仍能严格加载/评测，这里按权重自动还原：
    if "carry_ln" not in cfg_dict:
        cfg_dict["carry_ln"] = any(k.startswith("n_in.")
                                   for k in (blob.get("readout") or {}))
    if "readout_mask" not in cfg_dict:
        cfg_dict["readout_mask"] = True
    names = {f.name for f in fields(V10Config)}
    cfg = V10Config(**{k: v for k, v in cfg_dict.items() if k in names})
    model = VideoV10(cfg)
    model.middle.load_state_dict(blob["middle"], strict=True)
    model.readout.load_state_dict(blob["readout"], strict=True)
    return model


def main() -> None:
    a = parse_args()
    _set_sdpa_math()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    model = load_model(a.trainable).to(dev).eval()
    cfg = model.cfg
    S = cfg.n_steps
    n_max = int(a.n_steps_max) if a.n_steps_max else S
    n_max = max(1, min(S, n_max))
    print(f"[cfg] chunk_tokens={cfg.chunk_tokens} width={cfg.width} "
          f"L={cfg.positions_per_clip} steps={S} 评 {n_max} 档  "
          f"rate={cfg.values_per_frame()} 值/帧", flush=True)

    from data_v10 import ClipDataset
    ds = ClipDataset(a.data_root, a.split, frames=cfg.frames, img=int(model.tokenizer.img),
                     limit=max(0, a.limit) if a.limit else 0)
    n_clips = min(len(ds), int(a.n_clips) if a.n_clips else len(ds))
    print(f"[data] {a.split} 共 {len(ds)} 片段，评 {n_clips} 条", flush=True)

    per_clip = []
    t0 = time.time()
    for i in range(n_clips):
        x = ds[i].unsqueeze(0).to(dev)                     # (1,F,3,H,W)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z_cls, z_s = model.tokenizer.encode(x)
            enc = model.encode_rate_latents(z_cls, z_s)
            target = model.target_latents(z_cls, z_s)      # (1,F,1+K,c)
            out = model.readout(enc["z"], n_steps=n_max,
                                return_step_preds=True)
            preds = out["preds"][0]                        # (n,F,1+K,c)
            psnrs, l1s = [], []
            for t in range(preds.shape[0]):
                pt = preds[t:t + 1]                        # (1,F,1+K,c)
                pix = model.tokenizer.render(pt[:, :, :1].contiguous(),
                                             pt[:, :, 1:].contiguous())
                psnrs.append(float(psnr_from_l1(pix, x).mean()))
                l1s.append(float((pt - target).abs().mean()))
            up = model.tokenizer.render(z_cls, z_s)
            psnr_full = float(psnr_from_l1(up, x).mean())
        rec = dict(idx=i, psnr_steps=psnrs, l1_steps=l1s,
                   psnr_tok_full=psnr_full)
        per_clip.append(rec)
        if i % 5 == 0 or i == n_clips - 1:
            el = time.time() - t0
            print(f"[{i + 1}/{n_clips}] step{n_max} psnr={psnrs[-1]:.3f} "
                  f"tok_full={psnr_full:.3f}  {(i + 1) / el:.2f} clip/s "
                  f"eta {(n_clips - i - 1) * el / (i + 1) / 60:.1f} min", flush=True)

    P = np.asarray([r["psnr_steps"] for r in per_clip])     # (N,n_max)
    L = np.asarray([r["l1_steps"] for r in per_clip])
    TF = np.asarray([r["psnr_tok_full"] for r in per_clip])
    curves = dict(
        n_clips=int(P.shape[0]), n_steps=int(P.shape[1]),
        psnr_mean=P.mean(0).tolist(), psnr_per_clip_mean=P.mean(1).tolist(),
        l1_mean=L.mean(0).tolist(), psnr_tok_full_mean=float(TF.mean()),
        psnr_tok_full_std=float(TF.std()),
    )
    summary = dict(curves=curves, cfg=cfg.__dict__,
                   args=vars(a), per_clip=per_clip if a.dump_per_clip else None)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[out] {a.out}", flush=True)
    print("[curve] PSNR per step:",
          " ".join(f"{v:.2f}" for v in curves["psnr_mean"]), flush=True)
    print(f"[curve] tokenizer full-read upper bound: {TF.mean():.3f} dB "
          f"(+/- {TF.std():.3f})", flush=True)

    if not a.no_fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            xs = np.arange(1, P.shape[1] + 1)
            ax[0].plot(xs, P.mean(0), "-o", ms=3, label="v10 (read t groups)")
            ax[0].axhline(TF.mean(), ls="--", c="gray",
                          label="tokenizer full read")
            ax[0].set_xlabel("z steps read (rate budget)")
            ax[0].set_ylabel("PSNR (dB)")
            ax[0].set_title(f"v10 stepwise PSNR ({n_clips} val clips)")
            ax[0].grid(alpha=.3)
            ax[0].legend()
            ax[1].plot(xs, L.mean(0), "-o", ms=3, c="C1")
            ax[1].set_xlabel("z steps read")
            ax[1].set_ylabel("latent L1")
            ax[1].set_title("v10 stepwise latent L1")
            ax[1].grid(alpha=.3)
            fig.tight_layout()
            png = os.path.splitext(a.out or "fig_stepwise")[0] + "_fig.png"
            fig.savefig(png, dpi=130)
            print(f"[fig] {png}", flush=True)
        except Exception as exc:
            print(f"[fig] 画图跳过: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
