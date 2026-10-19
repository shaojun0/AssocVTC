#!/usr/bin/env python3
"""SAVOIAS 低/中/高 human complexity 重建网格（ALIT Fig.4-left 复刻）

在远端 GPU 上跑：读 SAVOIAS People-Art 中复杂度 ≈10/50/90 的三张图，
用 `stdattn_k64_rec` 的 BPTT 前向，输出 original + step1/2/4/7 的重建网格。

用法（需在仓库根目录、GPU 机器上）：
    python3 tools/render_savoias_grid.py \
        --images /root/autodl-tmp/savoias/Art \
        --scores /root/autodl-tmp/savoias_gt/gt_art.npy \
        --ckpt /root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt \
        --out doc/2026-10-19/artifacts/savoias_complexity/savoias_recon_grid.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval_stepwise_linattn import load_model, MEAN_255, STD_255  # noqa: E402
from model_v2_linattn import patches_to_image  # noqa: E402
from data_linattn import build_transform  # noqa: E402


def pick(scores, target):
    return min(scores.items(), key=lambda kv: abs(kv[1] - target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/root/autodl-tmp/savoias/Art")
    ap.add_argument("--scores", default="/root/autodl-tmp/savoias_gt/gt_art.npy")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/srdiff_stdattn/"
                                     "stdattn_k64_rec/ckpt_best.pt")
    ap.add_argument("--out", default=str(REPO / "doc/2026-10-19/artifacts/"
                                        "savoias_complexity/savoias_recon_grid.png"))
    ap.add_argument("--targets", default="10,50,90")
    a = ap.parse_args()

    gt = np.load(a.scores, allow_pickle=True).item()
    scores = {int(k): int(gt[k][1]) for k in gt}
    targets = [int(x) for x in a.targets.split(",")]
    picked, used = [], set()
    for t in targets:
        for k in sorted(scores, key=lambda k: abs(scores[k] - t)):
            if k in used:
                continue
            used.add(k); picked.append((int(scores[k]), k)); break
    print("picked:", [(s, f"{k}.jpg") for s, k in picked], flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model(a.ckpt, dev)
    patch, img = cfg.patch, cfg.img_size
    tf = build_transform(img, False)
    xs, orig = [], []
    for s, k in picked:
        im = Image.open(Path(a.images) / f"{k}.jpg").convert("RGB")
        orig.append(np.array(im.resize((img, img), Image.BICUBIC)))
        xs.append(tf(im).unsqueeze(0))
    x = torch.cat(xs, 0).to(dev)
    with torch.no_grad():
        out = model(x)
    pix, tgt = out["Y_pix"].float(), out["target_pix"].float()
    B, T, N, _ = pix.shape
    p255 = patches_to_image(pix.reshape(B * T, N, -1), img, img, patch,
                            MEAN_255, STD_255, clamp_255=False).reshape(B, T, 3, img, img)
    t255 = patches_to_image(tgt, img, img, patch, MEAN_255, STD_255,
                            clamp_255=False).reshape(B, 3, img, img)
    p255 = p255.clamp(0, 255).cpu().numpy()
    t255 = t255.clamp(0, 255).cpu().numpy()

    steps_show = [None, 0, 1, 3, 6]          # original / step1(3) / 2(8) / 4(24) / 7(63)
    titles = ["original", "step1 (3 tok)", "step2 (8 tok)",
              "step4 (24 tok)", "step7 (63 tok)"]
    fig, axes = plt.subplots(len(picked), len(steps_show),
                             figsize=(10.5, 2.2 * len(picked)))
    for r, (s, k) in enumerate(picked):
        for c, st in enumerate(steps_show):
            ax = axes[r, c]
            ax.imshow((t255[r] if st is None else p255[r, st]).transpose(1, 2, 0)
                      .astype(np.uint8))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        axes[r, 0].set_ylabel(f"complexity {s}\n({k}.jpg)", fontsize=8)
    fig.suptitle("Reconstruction vs. human-annotated complexity (SAVOIAS People-Art, "
                 "OOD; standard Transformer BPTT)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("[plot]", out)


if __name__ == "__main__":
    main()
