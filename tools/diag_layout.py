"""诊断：merger 之后 token 的空间排布是否 row-major —— 用缓存特征快速对比几种 layout。
只跑 1000 张、250 步、解码器宽 128：比较训练 L1（layout 错了会明显更差）。
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_enc_dec import ReconDataset, list_split, Decoder
from probe_enc_dec_vendor import VisionEnc, VENDORS

dev="cuda"; torch.manual_seed(0)
vendor=sys.argv[1] if len(sys.argv)>1 else "glm5_flash"
stage=sys.argv[2] if len(sys.argv)>2 else "post"
res=int(sys.argv[3]) if len(sys.argv)>3 else 0
n_train, n_val, steps, bs, width = 1000, 200, 250, 32, 128
spec=dict(VENDORS[vendor])
enc=VisionEnc(vendor, dev, stage=stage, res=(res or None))
tr,va=list_split("/root/autodl-tmp/datasets/imagenet100", 10, 2)   # 每类10张 train, 2张 val
tr,va=tr[:n_train], va[:n_val]
def extract(paths):
    ds=ReconDataset(paths, enc.res, spec["mean"], spec["std"])
    fo=[]; to=[]
    with torch.no_grad():
        for i in range(0,len(ds),16):
            xs=torch.stack([ds[j][0] for j in range(i,min(i+16,len(ds)))]).to(dev)
            ts=torch.stack([ds[j][1] for j in range(i,min(i+16,len(ds)))])
            f=enc.forward(xs)
            fo.append(f.float().cpu().to(torch.float16)); to.append(ts)
    return torch.cat(fo), torch.cat(to)
t=time.time(); Ftr,Ttr=extract(tr); Fva,Tva=extract(va)
print(f"[diag] feats {tuple(Ftr.shape)} extracted {time.time()-t:.0f}s", flush=True)
g=Ftr.shape[-1]
def variant(f, v):
    B,C,h,w=f.shape
    if v=="row": return f
    if v=="trans": return f.transpose(-1,-2)
    if v=="unshuffle":   # token 顺序 = (2,2,h/2,w/2) 的 pixel-unshuffle ⇒ 还原成 2h x 2w
        hh,ww=h//2,w//2
        return f.view(B,C,2,2,hh,ww).permute(0,1,4,2,5,3).reshape(B,C,2*hh,2*ww)
    if v=="shuffle":
        idx=torch.randperm(h*w, generator=torch.Generator().manual_seed(0))
        return f.view(B,C,-1)[:, :, idx].view(B,C,h,w)
    raise ValueError(v)
for v in ("row","trans","unshuffle","shuffle"):
    Ftr2, Fva2 = variant(Ftr,v), variant(Fva,v)
    gg=Ftr2.shape[-1]
    if gg not in (8,16,32,64):
        gg=int(2**round(np.log2(gg)))
    dec=Decoder(Ftr2.shape[1], width=width, grid=gg, out=256).to(dev)
    opt=torch.optim.AdamW(dec.parameters(), lr=2e-4, weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps, pct_start=0.1)
    n=Ftr2.shape[0]; it=0
    for st in range(1,steps+1):
        ids=(torch.randperm(n)[:bs]) if it%n==0 else ids
        x=Ftr2[ids].to(dev).float(); y=Ttr[ids].to(dev)
        r=dec(x); loss=F.l1_loss(r,y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(),1.0); opt.step(); sch.step()
    dec.eval()
    with torch.no_grad():
        ps=0.0; m=Fva2.shape[0]
        for i in range(0,m,32):
            x=Fva2[i:i+32].to(dev).float(); y=Tva[i:i+32].to(dev)
            r=dec(x); ps+=float((-10*torch.log10(((r-y)**2).mean(dim=(1,2,3)).clamp_min(1e-12))).sum())
        ps/=m
    print(f"[diag] {vendor} stage={stage} res={enc.res} layout={v:10s} grid={gg:2d} trainL1={float(loss):.4f} valPSNR={ps:.2f}dB", flush=True)
