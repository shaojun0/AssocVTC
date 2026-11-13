import json, torch, sys
sys.path.insert(0, "/root/autodl-tmp/video_recon/code3")
from model_v8 import build_any
cfg = json.load(open("/root/autodl-tmp/video_recon/code3/cfg_v32b.json"))
dev = "cuda"
for mode in ("fp16", "fp32"):
    torch.manual_seed(0)
    m = build_any(cfg).to(dev)
    m.train()
    x = torch.rand(2, 8, 3, 448, 448, device=dev)
    if mode == "fp16":
        with torch.autocast("cuda", dtype=torch.float16):
            out = m(x, n_warm=2)
    else:
        out = m(x, n_warm=2)
    loss = out["loss"]
    print(f"[{mode}] loss dtype={loss.dtype} value={loss.item():.6f}")
    loss.backward()
    for name in ("encoder", "middle", "decoder"):
        mod = getattr(m, name)
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        nz = [g for g in gs if g.abs().sum() > 0]
        gn = sum(g.float().pow(2).sum() for g in gs).sqrt().item()
        print(f"   {name}: tensors={len(gs)} nonzero={len(nz)} grad_norm={gn:.3e} "
              f"max={max(g.abs().max().item() for g in gs):.3e}")
    del m, x, out, loss
    torch.cuda.empty_cache()
