"""把 v10 的「逐档读出」解码成**真实视频**：第 t 档（读前 t 组 z）渲染整片 120 帧。

用途（2026-11-19 用户要求）：30 步渐进训练的每一步到底能不能解码出**真实视频**，
而不只是 PSNR 数字。对每个被检片段、每个检查档位输出一个 mp4（8 fps / 15 s），
外加 GT 与「冻结分词器满读」上界视频、一张多档 × 多帧的 contact sheet。

用法::

    SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=1 python visualize_stepwise_video.py \
        --trainable /root/autodl-tmp/tok448_v2/v10_120f_2stage/best_trainable.pt \
        --clips 0,1,2 --steps 1,5,10,15,20,25,30 \
        --out /root/autodl-tmp/tok448_v2/v10_120f_2stage/viz

产物（<out>/）：
  clip{i}_step{tt}.mp4     第 tt 档的整片预测
  clip{i}_gt.mp4           真值
  clip{i}_tokfull.mp4      冻结分词器满读上界（同率下的天花板画面）
  contact_clip{i}.png      行 = 档位（含 GT/上界），列 = 8 个等间隔帧
  viz.json                 每档 PSNR + 时间一致性（相邻帧差比 GT 的 flicker 比）

注意：8 fps / 448² / 120 帧（15 s）与训练数据一致。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v10_stepwise as ev  # noqa: E402  （复用 load_model / SDPA 设置）
from model_v10 import psnr_from_l1  # noqa: E402

from data_v10 import ClipDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trainable", type=str, required=True)
    p.add_argument("--data_root", type=str,
                   default="/root/autodl-tmp/video_recon/data3")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--clips", type=str, default="0,1,2",
                   help="片段下标，逗号分隔（val split 内）")
    p.add_argument("--steps", type=str, default="1,5,10,15,20,25,30",
                   help="要出视频的档位（1-based，读前 t 组 z）")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--sheet_frames", type=int, default=8)
    p.add_argument("--sheet_px", type=int, default=112)
    return p.parse_args()


def to_u8(pix: torch.Tensor) -> np.ndarray:
    """(F,3,H,W) ∈[0,1] → (F,H,W,3) uint8。"""
    x = pix.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def write_mp4(path: str, frames_u8: np.ndarray, fps: int) -> None:
    import imageio
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                           macro_block_size=1, pixelformat="yuv420p")
    try:
        for f in frames_u8:
            w.append_data(f)
    finally:
        w.close()


def temporal_stats(frames_u8: np.ndarray) -> dict:
    """时间一致性：相邻帧平均绝对差（0-255 尺度）；flicker 比 = pred/gt。"""
    f = frames_u8.astype(np.float32)
    d = np.abs(f[1:] - f[:-1]).mean()
    return dict(frame_diff=float(d))


def main() -> None:
    a = parse_args()
    ev._set_sdpa_math()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    model = ev.load_model(a.trainable).to(dev).eval()
    cfg = model.cfg
    os.makedirs(a.out, exist_ok=True)
    clips = [int(s) for s in a.clips.split(",") if s.strip()]
    steps = sorted({int(s) for s in a.steps.split(",") if s.strip()})
    steps = [min(max(1, s), cfg.n_steps) for s in steps]

    ds = ClipDataset(a.data_root, a.split, frames=cfg.frames,
                     img=int(model.tokenizer.img))
    print(f"[data] {a.split} 共 {len(ds)} 片段；检查 {clips}；档位 {steps}",
          flush=True)
    report = dict(clips=clips, steps=steps, fps=a.fps, per_clip=[])

    for ci in clips:
        x = ds[ci].unsqueeze(0).to(dev)                    # (1,F,3,H,W)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z_cls, z_s = model.tokenizer.encode(x)
            enc = model.encode_rate_latents(z_cls, z_s)
            out = model.readout(enc["z"], n_steps=max(steps),
                                return_step_preds=True)
            preds = out["preds"][0].float()                # (S,F,1+K,c)
            up = model.tokenizer.render(z_cls, z_s)        # (1,F,3,H,W) 上界
        gt_u8 = to_u8(x[0])
        write_mp4(os.path.join(a.out, f"clip{ci}_gt.mp4"), gt_u8, a.fps)
        up_u8 = to_u8(up[0])
        write_mp4(os.path.join(a.out, f"clip{ci}_tokfull.mp4"), up_u8, a.fps)
        gt_t = temporal_stats(gt_u8)
        info = dict(clip=ci, gt_psnr_tokfull=float(
            psnr_from_l1(up, x).mean()), gt_frame_diff=gt_t["frame_diff"],
            steps=[])
        # contact sheet 的帧下标（等间隔）
        nf = cfg.frames
        k = min(a.sheet_frames, nf)
        fidx = np.linspace(0, nf - 1, k).round().astype(int).tolist()
        rows = [("gt", gt_u8), ("tokfull", up_u8)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in steps:
                pt = preds[t - 1:t]                        # (1,F,1+K,c)
                pix = model.tokenizer.render(pt[:, :, :1].contiguous(),
                                             pt[:, :, 1:].contiguous())
                u8 = to_u8(pix[0])
                write_mp4(os.path.join(a.out, f"clip{ci}_step{t:02d}.mp4"),
                          u8, a.fps)
                ts = temporal_stats(u8)
                psnr = float(psnr_from_l1(pix, x).mean())
                info["steps"].append(dict(step=t, psnr_all=psnr,
                                          frame_diff=ts["frame_diff"],
                                          flicker=ts["frame_diff"] /
                                          max(gt_t["frame_diff"], 1e-6)))
                rows.append((f"step{t:02d}", u8))
                print(f"[clip{ci}] step {t:2d} psnr_all={psnr:.2f} "
                      f"frame_diff={ts['frame_diff']:.2f} "
                      f"flicker={info['steps'][-1]['flicker']:.2f}", flush=True)
        # contact sheet
        from PIL import Image
        S = a.sheet_px
        sheet = Image.new("RGB", (k * S, len(rows) * S), (0, 0, 0))
        for r, (_nm, u8) in enumerate(rows):
            for c, fi in enumerate(fidx):
                im = Image.fromarray(u8[fi]).resize((S, S), Image.BILINEAR)
                sheet.paste(im, (c * S, r * S))
        sp = os.path.join(a.out, f"contact_clip{ci}.png")
        sheet.save(sp)
        print(f"[sheet] {sp}  行(上→下)={[n for n, _ in rows]} 列帧={fidx}",
              flush=True)
        report["per_clip"].append(info)

    with open(os.path.join(a.out, "viz.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
