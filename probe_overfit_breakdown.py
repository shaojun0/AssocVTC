"""单片段过拟合结果的分解：z_cls 误差 vs specials 误差 vs 像素 PSNR。"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v10 import V10Config, VideoV10, psnr_from_l1
from data_v10 import ClipDataset

for nm in ("flash", "mem_efficient", "cudnn"):
    try:
        getattr(torch.backends.cuda, f"enable_{nm}_sdp")(False)
    except Exception:
        pass
torch.backends.cuda.enable_math_sdp(True)

ROOT = "/root/autodl-tmp/tok448_v2/overfit"
C = "/root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train"
dev = torch.device("cuda", 0)
with open(os.path.join(C, "meta.json")) as f:
    meta = json.load(f)
mm = np.load(os.path.join(C, f"rank{0}.npy"), mmap_mode="r")
t = torch.from_numpy(np.array(mm[0])).float()[None].to(dev)
z_cls, z_s = t[:, :, :1], t[:, :, 1:]
x = ClipDataset("/root/autodl-tmp/video_recon/data3", "train",
                frames=120, img=448)[0].to(dev)[None]

rows = {}
for name in ("prefix", "step_nocopy", "step_copy", "step_warm"):
    p_ = os.path.join(ROOT, name, "trainable.pt")
    if not os.path.isfile(p_):
        print(name, "MISSING")
        continue
    blob = torch.load(p_, map_location="cpu", weights_only=False)
    fields = set(V10Config.__dataclass_fields__)
    cfg = V10Config(**{k: v for k, v in blob["cfg"].items() if k in fields})
    m = VideoV10(cfg).to(dev)
    m.middle.load_state_dict(blob["middle"])
    m.readout.load_state_dict(blob["readout"], strict=False)
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc = m.encode_rate_latents(z_cls, z_s)
        tgt = m.target_latents(enc["z_cls"], enc["z_s"])[0]        # (F,17,c)
        pr = m.readout(enc["z"], return_step_preds=True)["preds"][0, -1]
        pz, ps = pr[None], tgt[None]
        r_pred = m.tokenizer.render(pz[:, :, :1], pz[:, :, 1:])
        r_gt = m.tokenizer.render(ps[:, :, :1], ps[:, :, 1:])
        mix_z = torch.cat([pz[:, :, :1], ps[:, :, 1:]], 2)
        r_z = m.tokenizer.render(mix_z[:, :, :1], mix_z[:, :, 1:])
        mix_s = torch.cat([ps[:, :, :1], pz[:, :, 1:]], 2)
        r_s = m.tokenizer.render(mix_s[:, :, :1], mix_s[:, :, 1:])
        out = dict(
            zcls_l1=float((pr[:, 0] - tgt[:, 0]).abs().mean()),
            spec_l1=float((pr[:, 1:] - tgt[:, 1:]).abs().mean()),
            all_l1=float((pr - tgt).abs().mean()),
            psnr=float(psnr_from_l1(r_pred, x)),
            psnr_only_zcls=float(psnr_from_l1(r_z, x)),
            psnr_only_spec=float(psnr_from_l1(r_s, x)),
            psnr_gt=float(psnr_from_l1(r_gt, x)),
        )
    rows[name] = out
    print(name, json.dumps({k: round(v, 4) for k, v in out.items()}), flush=True)
