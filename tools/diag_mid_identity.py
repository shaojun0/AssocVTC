"""诊断：middle(CrossQueryMiddle) 输出与输入 rate 的逐位置关系（是否接近恒等）。
若接近恒等，读出只需学逆映射；若被抹平/错位，读出很难从摘要重建细节。
"""
import os, sys, json, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_v10_stepwise import load_model, _set_sdpa_math
from data_v10 import ClipDataset

ap = argparse.ArgumentParser()
ap.add_argument("--trainable", required=True)
ap.add_argument("--clips", default="0,1,2,3")
a = ap.parse_args()
_set_sdpa_math()
dev = torch.device("cuda", 0); torch.cuda.set_device(0)
m = load_model(a.trainable).to(dev).eval()
cfg = m.cfg
ds = ClipDataset("/root/autodl-tmp/video_recon/data3", "val", frames=cfg.frames,
                 img=int(m.tokenizer.img))
rows = []
with torch.no_grad():
    for ci in [int(s) for s in a.clips.split(",") if s.strip()]:
        x = ds[ci].unsqueeze(0).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zc, zs = m.tokenizer.encode(x)
            enc = m.encode_rate_latents(zc, zs)
        p, mid = enc["packed"].float(), enc["mid"].float()   # (1,F,8,896)
        cos_row = torch.nn.functional.cosine_similarity(p, mid, dim=-1)  # (1,F,8)
        # 行内混叠程度：同一位置 vs 其它位置
        rows.append(dict(clip=ci,
                         cos_same_pos=float(cos_row.mean()),
                         cos_same_pos_std=float(cos_row.std()),
                         packed_norm=float(p.norm(dim=-1).mean()),
                         mid_norm=float(mid.norm(dim=-1).mean())))
print(json.dumps(rows, ensure_ascii=False, indent=2))
