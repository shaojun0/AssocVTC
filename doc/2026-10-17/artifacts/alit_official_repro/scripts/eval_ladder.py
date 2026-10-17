"""ALIT stage-1 检查：每个 rollout 迭代（32..256 token）的重建质量阶梯。

用法: python eval_ladder.py --ckpt <ckpt.pth> [--n 128] [--bs 16] [--out outdir]
"""
import argparse, json, os, glob
import torch, torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import adaptive_tokenizers

p = argparse.ArgumentParser()
p.add_argument('--ckpt', default='')
p.add_argument('--which', default='ema', choices=['ema', 'model'])
p.add_argument('--model', default='alit_tiny')
p.add_argument('--n', type=int, default=128)
p.add_argument('--bs', type=int, default=16)
p.add_argument('--out', default='/root/autodl-tmp/alit_out/ladder')
p.add_argument('--data', default='/root/autodl-tmp/datasets/alit_in100/train')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = 'cuda'

model = adaptive_tokenizers.__dict__[args.model](
    base_tokenizer_args={"id": "vqgan", "is_requires_grad": False},
    quantize_latent=True, factorize_latent=True,
    train_stage="latent_distillation_pretrain").to(dev).eval()
epoch = None
if args.ckpt:
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    key = args.which if args.which in sd else 'model'
    epoch = sd.get('epoch')
    missing, unexpected = model.load_state_dict(sd[key], strict=False)
    print(f"loaded {args.ckpt} [{key}] epoch={epoch} missing={len(missing)} unexpected={len(unexpected)}")
    del sd
torch.cuda.empty_cache()

files = sorted(glob.glob(os.path.join(args.data, '*', '*.JPEG')) +
               glob.glob(os.path.join(args.data, '*', '*.jpg')))[-args.n:]
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                         transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
print(f"eval images={len(files)} bs={args.bs}")

sum_l1, sum_mse, cnt, grid = {}, {}, 0, []
with torch.no_grad():
    for s in range(0, len(files), args.bs):
        chunk = files[s:s + args.bs]
        imgs = torch.stack([tf(Image.open(f).convert('RGB')) for f in chunk]).to(dev)
        logs = model(imgs, reconstruction_iters="all")
        for i, d in enumerate(logs):
            k = f'reconstructed_imgs_{i}'
            if k not in d:
                continue
            rec = d[k].clamp(-1, 1)
            a, b = (rec + 1) / 2, (imgs + 1) / 2
            sum_l1[i] = sum_l1.get(i, 0.0) + F.l1_loss(a, b, reduction='mean').item() * len(chunk)
            sum_mse[i] = sum_mse.get(i, 0.0) + F.mse_loss(a, b, reduction='mean').item() * len(chunk)
            if len(grid) < 3:
                grid.append(rec[:3])
        cnt += len(chunk)

res = {'n': cnt, 'ckpt': args.ckpt, 'epoch': epoch, 'l1_255': {}, 'psnr': {}}
for i in sorted(sum_l1):
    n_tok = 32 * (i + 1)
    res['l1_255'][n_tok] = round(sum_l1[i] / cnt * 255, 3)
    res['psnr'][n_tok] = round(10 * torch.log10(torch.tensor(1.0 / max(sum_mse[i] / cnt, 1e-12))).item(), 3)
    print(f"tokens={n_tok:>3}  L1={res['l1_255'][n_tok]:6.2f}/255  PSNR={res['psnr'][n_tok]:5.2f} dB")
if grid:
    g = torch.cat(grid, 0)                      # 每档 3 张，前 8 档 => 24 张
    save_image((g + 1) / 2, os.path.join(args.out, 'ladder_grid.png'), nrow=3, padding=2)
json.dump(res, open(os.path.join(args.out, 'ladder.json'), 'w'), indent=1)
print("saved ->", args.out)
