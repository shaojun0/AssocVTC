#!/usr/bin/env python3
"""count_params.py — 精确统计 MultiResSR 参数量，「解码器整体 ×10」的验收口径。

为什么需要它：用户 2026-10-14 要求「把解码器参数量整体提升 10 倍」。参数量必须
**算出来**而不是估出来——`OutputQueryDecoder` 的规模由
`depth`（层数）× `stack_dim`（宽度）× `mlp_ratio`（FFN 倍率）× `num_patches`（查询行数）
共同决定，随手把某一项 ×10 并不会让总参数量正好 ×10。本脚本用 meta device 建模型
（不占内存）逐个配置统计，并给出与基线的比值。

基线（2026-09-26 六源 run 实际用的配置）::

    depth=2  width=1024(=dim, 无投影)  heads=8  mlp_ratio=4.0  K=144

用法::

    python tools/count_params.py                      # 基线 + 「最接近 ×10」候选网格
    python tools/count_params.py --scale 5            # 换成找 ×5
    python tools/count_params.py --depth 2 --width 3072 --heads 12 --mlp_ratio 5
    python tools/count_params.py --json out.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
# 注意：部分 transformers 安装的顶层 `from transformers import Dinov2Model` 会
# ImportError（4.57.6 实测），但子模块一定在 ⇒ 统一走子模块。
from transformers.models.dinov2 import Dinov2Config, Dinov2Model  # noqa: E402

from model_multi import MultiResSR, RESOS, param_breakdown  # noqa: E402

BASELINE = dict(depth=2, width=0, heads=8, mlp_ratio=4.0)


def build(depth, width, heads, mlp_ratio, dim=1024, num_specials=144,
          resos=RESOS, device="meta"):
    """meta device 上建模型（只数参数量，不分配显存/内存）。"""
    cfg = Dinov2Config(hidden_size=dim, num_hidden_layers=24,
                       num_attention_heads=16, intermediate_size=4096,
                       image_size=518, patch_size=14)
    with torch.device(device):
        pass
        dino = Dinov2Model(cfg)
        model = MultiResSR(dino, num_specials=num_specials, dim=dim,
                           heads=heads, depth=depth, mlp_ratio=mlp_ratio,
                           resos=resos, decoder_width=width)
    return model


def measure(**kw) -> dict:
    m = build(**kw)
    b = param_breakdown(m)
    g = b["groups"]
    return {
        "config": {k: kw[k] for k in ("depth", "width", "heads", "mlp_ratio")},
        "width_effective": int(m.decoder_width or m.dim),
        "dinov2_M": g["dinov2"] / 1e6,
        "special_bank_M": g["special_bank"] / 1e6,
        "decoders_M": g["decoders"] / 1e6,
        "pixel_head_M": g["pixel_head"] / 1e6,
        "decoder_total_M": b["decoder_total"] / 1e6,
        "total_M": b["total"] / 1e6,
        "decoder_share": b["decoder_ratio_of_total"],
    }


def fmt_row(r, ratio=None):
    return (f"| {r['config']['depth']:>2} | {r['width_effective']:>5} | "
            f"{r['config']['heads']:>2} | {r['config']['mlp_ratio']:>4} | "
            f"{r['decoders_M']:>8.1f} | {r['pixel_head_M']:>5.2f} | "
            f"**{r['decoder_total_M']:>8.1f}** | "
            f"{(f'{ratio:.3f}×' if ratio else '—'):>7} |")


HEADER = ("| depth | width | heads | mlp | 4×decoder(M) | pixel_head(M) | "
          "解码器合计(M) | vs 基线 |\n|---:|---:|---:|---:|---:|---:|---:|---:|")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--width", type=int, default=None, help="0 = 与 dim 相同（无投影）")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=10.0,
                    help="目标倍数（用于候选网格排序）")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--json", default="", help="把结果写 JSON")
    args = ap.parse_args()

    base = measure(**BASELINE, dim=args.dim)
    print(f"# MultiResSR 参数量（K=144, dim={args.dim}, 目标 = 解码器 ×{args.scale:g}）\n")
    print("基线 = 2026-09-26 六源 run 的实际配置\n")
    print(HEADER)
    print(fmt_row(base))
    print(f"\n基线解码器合计 **{base['decoder_total_M']:.1f} M** "
          f"(4×decoder {base['decoders_M']:.1f} + bank {base['special_bank_M']:.2f} "
          f"+ pixel_head {base['pixel_head_M']:.2f})，全模型 {base['total_M']:.1f} M，"
          f"解码器占比 {base['decoder_share'] * 100:.1f}%\n")

    out = {"baseline": base, "target_scale": args.scale, "candidates": []}
    target = base["decoder_total_M"] * args.scale

    if args.depth is not None:
        r = measure(depth=args.depth, width=args.width or 0, heads=args.heads,
                    mlp_ratio=args.mlp_ratio, dim=args.dim)
        print("指定配置：\n")
        print(HEADER)
        print(fmt_row(r, r["decoder_total_M"] / base["decoder_total_M"]))
        print(f"\n解码器 {r['decoder_total_M']:.1f} M = 基线的 "
              f"**{r['decoder_total_M'] / base['decoder_total_M']:.3f}×**（目标 {args.scale:g}×），"
              f"全模型 {r['total_M']:.1f} M")
        out["candidates"].append(r)
    else:
        cands = []
        for depth in (2, 3, 4, 5):
            for width in (0, 1536, 2048, 2560, 3072, 3584, 4096):
                if width and width % args.heads:
                    continue
                for mlp in (4.0, 5.0, 6.0, 8.0):
                    r = measure(depth=depth, width=width, heads=args.heads,
                                mlp_ratio=mlp, dim=args.dim)
                    cands.append(r)
        # 只保留结构合法的（width 能被 heads 整除已保证），按接近目标排序
        cands.sort(key=lambda r: abs(r["decoder_total_M"] - target))
        top = [r for r in cands if abs(r["decoder_total_M"] - target) / target < 0.25][:15]
        print(f"最接近 ×{args.scale:g} 的候选（解码器 {target:.0f} M ± 25%）：\n")
        print(HEADER)
        for r in top:
            print(fmt_row(r, r["decoder_total_M"] / base["decoder_total_M"]))
        out["candidates"] = top
        if top:
            b = top[0]
            print(f"\n最接近的一个：depth={b['config']['depth']} "
                  f"width={b['width_effective']} heads={b['config']['heads']} "
                  f"mlp_ratio={b['config']['mlp_ratio']} ⇒ "
                  f"{b['decoder_total_M'] / base['decoder_total_M']:.3f}×"
                  f"（解码器 {b['decoder_total_M']:.1f} M，全模型 {b['total_M']:.1f} M）")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
