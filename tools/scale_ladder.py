#!/usr/bin/env python3
"""scale_ladder.py — 「容量阶梯」消融的架构选型：把 enc/dec 参数量抬到 60/120/240/360M（2026-10-24）

用户口径（2026-10-24）：
    「这次只跑 n/step=2，step=16，但是编码和解码器参数量逐步的增加，先各 60M，
     再各 120M，再各 240M，最后加到 360M。」

scaling 方案（刻意简单、可解释）：
  · **深度钉死**在基线（encoder 13 层 / decoder 7 层），只**加宽 dim** ⇒ 容量轴单一 = 宽度，
    且 BPTT 的层数不变（显存/时间随宽度线性、而不是 depth×T 双乘，见 §局限）；
  · 基线 = `enc_dim 448 ×13`（31.85M）+ `dec_dim 512 ×7`（31.17M）≈ 63M（K=32、N=196）；
  · 每个目标档在候选 (dim, heads) 里挑**实测参数量最接近**的（用 `SRPhase1LinAttn.count_params()`
    精确计数，含 patch_embed / pos / z_s / pixel_head / query / mem_pos）；heads 选使 head_dim≈64–96。

用法（服务器仓库根目录, 纯 CPU）:
    python tools/scale_ladder.py                 # 打印候选表 + 每个目标的推荐
    python tools/scale_ladder.py --json out.json # 同时落盘推荐配置（给 runner 用）
"""
import argparse
import json
import os
import sys

TARGETS = [60e6, 120e6, 240e6, 360e6]

# 基线（2026-10-19/21/23 的 63M 臂）
BASE = dict(enc_dim=448, enc_depth=13, enc_heads=8, dec_dim=512, dec_depth=7, dec_heads=8)

# 候选：深度钉死, 只扫宽度；heads 取能整除且 head_dim 在 56..112 的
ENC_CANDS = [(576, 8), (592, 8), (608, 8), (624, 12), (640, 8), (656, 8),
             (672, 8), (704, 8), (736, 8), (768, 8), (832, 8), (864, 16),
             (896, 14), (928, 16), (960, 15), (992, 16), (1088, 16),
             (1152, 16), (1216, 16), (1248, 16), (1280, 16),
             (1408, 16), (1472, 16), (1504, 16), (1536, 16), (1600, 16),
             (1664, 16)]
DEC_CANDS = [(640, 8), (672, 8), (704, 8), (736, 8), (768, 8), (800, 16),
             (832, 13), (896, 14), (960, 15), (1024, 16),
             (1088, 17), (1152, 18), (1184, 16), (1216, 16), (1280, 16),
             (1344, 16), (1408, 16), (1440, 16), (1472, 16), (1504, 16),
             (1536, 16), (1600, 16), (1664, 16), (1680, 16), (1728, 16),
             (1792, 16), (1920, 15), (1984, 16), (2048, 32), (2112, 16)]


def build(enc_dim, enc_depth, enc_heads, dec_dim, dec_depth, dec_heads, T=16, K=32):
    from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn
    steps = ",".join(str(i) for i in range(1, T + 1))
    cfg = LinAttnConfig(
        num_specials=K, enc_attn="std", enc_dim=enc_dim, enc_depth=enc_depth,
        enc_heads=enc_heads, dec_dim=dec_dim, dec_depth=dec_depth,
        dec_heads=dec_heads, dec_recurrent=True, dec_steps=steps,
        dec_step_plan="fixed", dec_block=2)
    m = SRPhase1LinAttn(cfg)
    pc = m.count_params()
    del m
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    tgt = {}
    print(f"[base] enc {BASE['enc_dim']}x{BASE['enc_depth']} / "
          f"dec {BASE['dec_dim']}x{BASE['dec_depth']}")
    pc = build(BASE["enc_dim"], BASE["enc_depth"], BASE["enc_heads"],
               BASE["dec_dim"], BASE["dec_depth"], BASE["dec_heads"])
    print(f"       {pc}")

    print("\n[enc 候选] (dim×depth, heads)  encoder 参数")
    enc = {}
    for dim, heads in ENC_CANDS:
        pc = build(dim, 13, heads, 512, 7, 8)
        enc[(dim, heads)] = pc["encoder"]
        print(f"  {dim:>4}x13 h{heads:<2} -> {pc['encoder']/1e6:8.2f}M")
    print("\n[dec 候选] (dim×depth, heads)  decoder 参数")
    dec = {}
    for dim, heads in DEC_CANDS:
        pc = build(448, 13, 8, dim, 7, heads)
        dec[(dim, heads)] = pc["decoder"]
        print(f"  {dim:>4}x7  h{heads:<2} -> {pc['decoder']/1e6:8.2f}M")

    print("\n[推荐] 每个目标取 |实测-目标| 最小的候选")
    out = {}
    for t in TARGETS:
        ed = min(enc, key=lambda k: abs(enc[k] - t))
        dd = min(dec, key=lambda k: abs(dec[k] - t))
        pc = build(ed[0], 13, ed[1], dd[0], 7, dd[1])
        out[str(int(t / 1e6))] = dict(
            target_M=int(t / 1e6),
            enc=dict(dim=ed[0], depth=13, heads=ed[1],
                     params_M=round(enc[ed] / 1e6, 2)),
            dec=dict(dim=dd[0], depth=7, heads=dd[1],
                     params_M=round(dec[dd] / 1e6, 2)),
            total_M=round(pc["total"] / 1e6, 2))
        print(f"  {int(t/1e6):>3}M 档: enc {ed[0]}x13 h{ed[1]}={enc[ed]/1e6:.2f}M | "
              f"dec {dd[0]}x7 h{dd[1]}={dec[dd]/1e6:.2f}M | total={pc['total']/1e6:.2f}M")
        print(f"         {pc}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"[out] {args.json}")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
