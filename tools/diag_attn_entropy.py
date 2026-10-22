#!/usr/bin/env python3
"""diag_attn_entropy.py — std（softmax）编码器各层注意力「饱和程度」诊断（2026-10-22）

为什么: 2026-10-21 的 K=128/256 A 相落进「z_s 冗余坍缩」差盆地, 怀疑是 std 编码器
**没有 QK-Norm** 时注意力 logits 随序列变长而放大 → softmax 饱和（每行权重集中到 1–2 个
key）→ K 个 special 被同化 → z_s 冗余。本脚本用 forward hook 抓每层 qkv, 重算注意力,
输出逐层统计:

  · qn/kn     : q、k 的每 token L2 范数均值（logits 尺度 ∝ qn·kn/√d）
  · logit_max : 每行 logits 的最大值（越大越接近"硬选择"）
  · ent       : 每行注意力熵（nats）均值（越小越饱和; ln(L)=完全均匀为上界）
  · ent_frac  : ent / ln(L)（1=均匀, →0=饱和）
  · maxw      : 每行最大注意力权重均值（→1 饱和）

用法:
  PYTHONPATH=$PWD python tools/diag_attn_entropy.py --out diag_attn.json --n 8 \
      --ckpts a.pt b.pt ...        # 默认 CPU, 避免占用训练 GPU
"""
import argparse
import json
import math
import os

import torch
from torch.utils.data import DataLoader

from data_linattn import build_datasets
from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    m = SRPhase1LinAttn(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg, ck


@torch.no_grad()
def layer_stats(model, x, device):
    """返回 [(qn, kn, logit_max, ent, ent_frac, maxw), ...]（每层一个 tuple）。"""
    blocks = [b for b in model.encoder.blocks]
    recs = []

    def make_hook(attn):
        def hook(mod, inp, out):
            # inp[0]: (B,L,D) → 重算注意力统计
            qkv = out                                   # (B,L,3D)
            B, L, _ = qkv.shape
            q, k, v = qkv.chunk(3, dim=-1)
            H = attn.heads
            d = attn.head_dim
            q = q.view(B, L, H, d).transpose(1, 2)      # (B,H,L,d)
            k = k.view(B, L, H, d).transpose(1, 2)
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            lmax = logits.max(dim=-1).values.mean().item()
            p = torch.softmax(logits.float(), dim=-1)   # (B,H,L,L)
            ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
            maxw = p.max(dim=-1).values.mean().item()
            recs.append((q.norm(dim=-1).mean().item(),
                         k.norm(dim=-1).mean().item(),
                         lmax, ent, ent / math.log(L), maxw))
        return hook

    handles = [b.attn.qkv.register_forward_hook(make_hook(b.attn))
               for b in blocks]
    try:
        model.encode(x)
    finally:
        for h in handles:
            h.remove()
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = args.device
    _, va = build_datasets(args.data_dir, 224, 50)
    stride = max(1, len(va) // args.n)
    idx = list(range(0, len(va), stride))[:args.n]
    dl = DataLoader(torch.utils.data.Subset(va, idx), batch_size=args.n,
                    shuffle=False, num_workers=2)
    xs = torch.cat([x for x, _ in dl], 0)[:args.n].to(dev)

    res = {}
    for p in args.ckpts:
        if not os.path.isfile(p):
            print(f"[skip] {p} 不存在")
            continue
        m, cfg, ck = load_model(p, dev)
        st = layer_stats(m, xs, dev)
        K = cfg.num_specials
        # 逐层均值 + 末层（最接近 z_s 的那层）
        mean = [sum(col) / len(st) for col in zip(*st)]
        last = st[-1]
        res[os.path.basename(os.path.dirname(p)) + "/ckpt_best.pt"] = {
            "ckpt": p, "K": K, "L": 1 + K + cfg.num_patches,
            "logit_max_last": last[2], "ent_last": last[3],
            "ent_frac_last": last[4], "maxw_last": last[5],
            "ent_frac_mean": mean[4], "maxw_mean": mean[5],
            "qn_mean": mean[0], "kn_mean": mean[1], "layers": st,
        }
        print(f"{os.path.basename(os.path.dirname(p)):<16} K={K:>3} L={1+K+cfg.num_patches:>3} "
              f"ent_frac_last={last[4]:.3f} maxw_last={last[5]:.3f} "
              f"logit_max={last[2]:.2f} ent_frac_mean={mean[4]:.3f}")
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
