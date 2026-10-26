#!/usr/bin/env python3
"""SAVOIAS 低/中/高 human complexity 重建网格（任意 ckpt，含 240M/T=16）

读 SAVOIAS People-Art 中复杂度最接近 `--targets` 的若干张图，用 ckpt 的 BPTT
前向，输出 original + `--steps` 指定读出步的重建。

两种用法：

1) **远端出图块，本地合成**（推荐；图注中文标签在本地画，避免远端缺字体）:

       # 远端 GPU（只写图块）
       PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python \\
           tools/render_savoias_grid.py \\
           --images /root/autodl-tmp/savoias_pick/Art \\
           --scores /root/autodl-tmp/savoias_pick/gt/gt_art.npy \\
           --ckpt /root/autodl-tmp/srdiff_scale/stdattn_m240_t16/ckpt_best.pt \\
           --targets 10,48,89 --steps 1,2,4,8,16 \\
           --save_tiles /root/autodl-tmp/savoias_pick/tiles_m240

       # 本地合成（tools/compose_recon_grid.py）
       python3 tools/compose_recon_grid.py --tiles <本地目录> --out 图/fig_recon_grid_m240.png

2) **直接出整图**（英文标签；不传 `--save_tiles` 时）:

       python3 tools/render_savoias_grid.py --ckpt .../ckpt_best.pt \\
           --steps 1,2,4,8,16 --out /tmp/grid.png

历史：旧 63M/K=64 的 Fig.4 由本脚本早期版本（`--steps 1,2,4,7`，递增窗口
3/8/24/63 token）生成；步号现在从 1 起算，token 数由 `model.decoder.windows`
的累计 hi 推出，对 square/fixed/cumulative 三种 plan 都成立。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval_stepwise_linattn import load_model, MEAN_255, STD_255  # noqa: E402
from model_v2_linattn import patches_to_image  # noqa: E402
from data_linattn import build_transform  # noqa: E402


def pick(scores, targets):
    """按目标分数选图，同一张图不重复用。"""
    picked, used = [], set()
    for t in targets:
        for k in sorted(scores, key=lambda k: abs(scores[k] - t)):
            if k in used:
                continue
            used.add(k)
            picked.append((int(scores[k]), k))
            break
    return picked


def resolve_image(img_dir: Path, k: int, ext: str):
    p = img_dir / f"{k}{ext}"
    if p.exists():
        return p
    hits = sorted(img_dir.glob(f"{k}.*"))
    if not hits:
        raise FileNotFoundError(f"{img_dir} 下找不到 {k}.*")
    return hits[0]


def save_tile(arr_chw, path: Path, up: int):
    """(3,H,W) [0,255] → PNG；up>1 时 bicubic 上采样（只为印刷清晰）。"""
    im = Image.fromarray(arr_chw.transpose(1, 2, 0).clip(0, 255).astype(np.uint8))
    if up > 1:
        im = im.resize((im.width * up, im.height * up), Image.BICUBIC)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/root/autodl-tmp/savoias/Art",
                    help="SAVOIAS 图像目录（Art/Scenes/Objects）")
    ap.add_argument("--scores", default="/root/autodl-tmp/savoias/gt/gt_art.npy",
                    help="gt_{art,scenes,objects}.npy：{index: [filename, score]}")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/ckpts/m240/ckpt_best.pt")
    ap.add_argument("--out", default=str(REPO / "doc/2026-10-26/artifacts/"
                                        "ood_savoias/fig_savoias_recon_grid.png"))
    ap.add_argument("--targets", default="10,48,89", help="目标复杂度（逗号分隔）")
    ap.add_argument("--steps", default="1,2,4,7", help="读出步（1 起算，逗号分隔）")
    ap.add_argument("--save_tiles", default="", help="只存图块到该目录（不画整图）")
    ap.add_argument("--tile_up", type=int, default=2, help="图块上采样倍数")
    ap.add_argument("--img_ext", default=".jpg")
    a = ap.parse_args()

    gt = np.load(a.scores, allow_pickle=True).item()
    scores = {int(k): int(gt[k][1]) for k in gt}
    targets = [int(x) for x in a.targets.split(",") if x.strip()]
    picked = pick(scores, targets)
    print("picked:", [(s, k) for s, k in picked], flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model(a.ckpt, dev)
    patch, img = cfg.patch, cfg.img_size
    recurrent = getattr(model, "recurrent", False)
    windows = list(model.decoder.windows) if recurrent else [None]
    T = len(windows)
    # 累计读入 z_s 数：每步读窗口的上界（A 坐标 1..K），三种 plan 通用
    cum = [int(hi) for (lo, hi) in windows] if recurrent else [int(cfg.num_specials)]
    steps = [int(s) for s in a.steps.split(",") if s.strip()]
    for s in steps:
        assert 1 <= s <= T, f"step {s} 超出范围 1..{T}"
    idx = [s - 1 for s in steps]
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {a.ckpt}\n  参数 {n_par / 1e6:.2f} M, K={cfg.num_specials}, "
          f"T={T}, cum_z_s={cum}\n  show steps={steps} → tokens={[cum[i] for i in idx]}",
          flush=True)

    img_dir = Path(a.images)
    tf = build_transform(img, False)
    xs, orig = [], []
    for s, k in picked:
        p = resolve_image(img_dir, k, a.img_ext)
        im = Image.open(p).convert("RGB")
        orig.append(np.array(im.resize((img, img), Image.BICUBIC)))
        xs.append(tf(im).unsqueeze(0))
    x = torch.cat(xs, 0).to(dev)
    with torch.no_grad():
        out = model(x)
    pix, tgt = out["Y_pix"].float(), out["target_pix"].float()
    B, T_out, N, _ = pix.shape
    assert T_out == T
    p255 = patches_to_image(pix.reshape(B * T, N, -1), img, img, patch,
                            MEAN_255, STD_255, clamp_255=False
                            ).reshape(B, T, 3, img, img).clamp(0, 255).cpu().numpy()
    t255 = patches_to_image(tgt, img, img, patch, MEAN_255, STD_255,
                            clamp_255=False).reshape(B, 3, img, img
                                                     ).clamp(0, 255).cpu().numpy()
    print(f"[recon] t255.mean()={float(t255.mean()):.2f} (应 ≈115, 0-255 口径)",
          flush=True)

    if a.save_tiles:
        d = Path(a.save_tiles)
        d.mkdir(parents=True, exist_ok=True)
        for r, (s, k) in enumerate(picked):
            save_tile(t255[r], d / f"row{r}_orig.png", a.tile_up)
            for c, i in enumerate(idx):
                save_tile(p255[r, i], d / f"row{r}_step{steps[c]:02d}.png", a.tile_up)
        l1_px = np.abs(p255[:, idx] - t255[:, None]).mean(axis=(2, 3, 4))  # (R,S)
        meta = {
            "ckpt": str(a.ckpt), "params_M": round(n_par / 1e6, 2),
            "K": int(cfg.num_specials), "T": T, "img_size": img,
            "steps": steps, "tokens": [cum[i] for i in idx],
            "rows": [{"score": s, "file": resolve_image(img_dir, k, a.img_ext).name}
                     for s, k in picked],
            "l1_px": [[round(float(v), 3) for v in row] for row in l1_px],
            "tile_up": a.tile_up,
        }
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(f"[tiles] {d}  ({len(picked)} 行 × {len(steps) + 1} 列)", flush=True)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [None] + idx
    titles = ["original"] + [f"step{s} ({cum[i]} tok)" for s, i in zip(steps, idx)]
    fig, axes = plt.subplots(len(picked), len(cols),
                             figsize=(2.1 * len(cols), 2.1 * len(picked)))
    axes = np.atleast_2d(axes)
    for r, (s, k) in enumerate(picked):
        for c, i in enumerate(cols):
            ax = axes[r, c]
            ax.imshow((t255[r] if i is None else p255[r, i]).transpose(1, 2, 0)
                      .astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        axes[r, 0].set_ylabel(f"complexity {s}\n({k})", fontsize=8)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("[plot]", out)


if __name__ == "__main__":
    main()
