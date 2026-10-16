#!/usr/bin/env python3
"""E-A5 第 1 项（z_s 冗余度）— AssocVTC 零训练分析小工具
================================================================
对循环臂 ckpt 的编码器 `z_s`（K×D, 一次前向的固定产物）跑
`metrics_linattn.zs_stats`，产出 `effective_rank*` / 两两 cos / 逐 token 方差。
判据（PLAN §3 E-A5）：`effective_rank ≪ K` 且 `pair_cos_mean > 0.9` ⇒ z_s 冗余仍在。

用法::

    python doc/2026-10-16/artifacts/alit/zs_redundancy.py \
      --ckpt /root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt \
      --num_images 2000 \
      --out doc/2026-10-16/artifacts/alit/zs_redundancy_stdattn_rec.json

口径: 与 eval_linattn.py 相同的 val 变换 + bf16 autocast（`--amp 1`）；只取编码器
`model.encode(x)[1]`，不跑解码器。注意力熵部分本脚本不做（PLAN 允许后置）。
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

# 向上找到仓库根（含 model_v2_linattn.py），支持脚本被放在 artifacts/ 下
_HERE = os.path.dirname(os.path.abspath(__file__))
for _ in range(8):
    if os.path.exists(os.path.join(_HERE, "model_v2_linattn.py")):
        sys.path.insert(0, _HERE)
        break
    _HERE = os.path.dirname(_HERE)
else:
    raise SystemExit("找不到仓库根（model_v2_linattn.py）；请把本脚本放在仓库内")

from data_linattn import build_datasets            # noqa: E402
from metrics_linattn import zs_stats               # noqa: E402
from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--num_images", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, va = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    n = min(len(va), args.num_images) if args.num_images > 0 else len(va)
    dl = DataLoader(Subset(va, range(n)), batch_size=args.batch_size,
                    shuffle=False, drop_last=False, num_workers=args.num_workers,
                    pin_memory=True)
    print(f"[zs] val n={n} device={dev} amp={args.amp}", flush=True)

    out = {"n_images": n, "data_dir": args.data_dir, "img_size": args.img_size,
           "amp": args.amp, "results": {}}
    for ckpt in args.ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                               if k in LinAttnConfig.__dataclass_fields__})
        model = SRPhase1LinAttn(cfg).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()

        chunks = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=bool(dev == "cuda" and args.amp)):
                    z_s = model.encode(x)[1]                 # (B,K,D)
                chunks.append(z_s.float().cpu())
        z = torch.cat(chunks, 0)
        assert z.dim() == 3 and z.shape[0] == n, f"z_s 形状异常: {tuple(z.shape)}"
        st = zs_stats(z)                                     # (B,K,D) 精确统计
        st.update({"ckpt": ckpt, "step": ck.get("step"), "epoch": ck.get("epoch"),
                   "tag": ck.get("tag", ""), "cfg": ck["cfg"]})
        key = os.path.basename(os.path.dirname(ckpt))
        out["results"][key] = st
        print(f"[zs] {key}: K={st['K']} D={st['D']} eff_rank={st['effective_rank']:.3f}"
              f" ({st['effective_rank_norm']:.3f}×K) eff_rank_centered="
              f"{st['effective_rank_centered']:.3f} cos_mean={st['pair_cos_mean']:.4f}"
              f" frac_cos>0.9={st['pair_cos_frac_gt_0_9']:.4f}", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
