#!/usr/bin/env python3
"""把 `render_savoias_grid.py --save_tiles` 产出的图块合成论文图（中文标签）。

图块目录里应有：`meta.json` + `row{r}_orig.png` + `row{r}_step{SS}.png`
（由远端 GPU 脚本写出），本脚本只做排版，不需要 GPU。

用法：
    python3 tools/compose_recon_grid.py \
        --tiles doc/2026-10-26/artifacts/ood_savoias/recon_tiles_m240 \
        --out 图/fig_recon_grid_m240.png

列标签取自 meta.json 的 steps/tokens（第 t 步累计 token 数），行标签取
targets 的复杂度分数与文件名，与正文描述一一对应。
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

CJK = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "Source Han Sans SC",
       "SimHei", "Microsoft YaHei", "DejaVu Sans"]


def setup_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK:
        if name in have:
            plt.rcParams["font.sans-serif"] = [name] + CJK
            break
    plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True, help="输出 PNG（同目录同名 .pdf 一并写出）")
    ap.add_argument("--title-fs", type=float, default=10.5)
    ap.add_argument("--label-fs", type=float, default=9.5)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    setup_font()
    d = Path(a.tiles)
    meta = json.loads((d / "meta.json").read_text())
    steps, toks = meta["steps"], meta["tokens"]
    rows = meta["rows"]

    n_r, n_c = len(rows), len(steps) + 1
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.28 * n_c, 2.42 * n_r),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for r in range(n_r):
        for c in range(n_c):
            ax = axes[r][c]
            f = d / (f"row{r}_orig.png" if c == 0 else f"row{r}_step{steps[c - 1]:02d}.png")
            ax.imshow(Image.open(f).convert("RGB"), interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6)
                s.set_color("0.35")
            if r == 0:
                t = "原始图像" if c == 0 else f"第 {steps[c - 1]} 步（{toks[c - 1]} token）"
                ax.set_title(t, fontsize=a.title_fs, pad=5)
        sc = rows[r]["score"]
        fn = rows[r]["file"]
        axes[r][0].set_ylabel(f"复杂度 {sc}\n（{fn}）", fontsize=a.label_fs, labelpad=7)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.915, bottom=0.01)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    print("[compose]", out, out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
