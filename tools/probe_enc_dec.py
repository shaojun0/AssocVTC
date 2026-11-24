"""冻结视觉编码器的**可解码性探针**：编码器特征 → 统一解码器 → 还原 256×256 图像。

问题：各家大模型供应商的视觉编码器（CLIP / SigLIP / ViT / DINOv2 / SAM / MAE …）
      到底编码了多少**视觉信息**？做法：冻结编码器，只训一个小解码器把特征还原成图，
      还原质量（PSNR/SSIM/L1）就是这个编码器表示的"可解码信息量"的直接下界证据。

协议（所有编码器完全一致，除编码器本身）：
  · 数据：ImageNet-100 目录（`<data_dir>/<cls>/*.jpg`），按类确定划分
    train = 每类 [10, 10+N) 张、val = 每类 [0,10) 张（顺序固定, seed 只影响数据顺序抖动）;
  · 目标：`fit256`（短边缩到 256 + center crop 256）; 编码器输入 = 同一张图缩放到它
    自己的 native 分辨率（**不做增强**, 三组对照共用同一批图）;
  · 特征：ViT 类取第 `--layer` 层（默认最后一层）的**空间 patch token**（丢 CLS），
    reshape 成 (C,h,w); SAM 取 image encoder 输出; SD-VAE 取 latent（未缩放）;
  · 统一瓶颈：特征 → `adaptive_avg_pool2d(g×g)`（h,w>g 时）或 bilinear 放大到 g×g
    （h,w<g 时）⇒ 所有编码器都变成 **g×g×C 的同一形状语义**（g=16 ⇒ 每编码器
    空间位置数相同, 只剩通道数差异 = 特征维度）;
  · 解码器：`Conv1x1(C→W)` + 4×[Upsample×2, Conv3×3, GroupNorm, GELU] + `Conv3×3→3`，
    **所有编码器完全同一个解码器**（只有第一层输入通道不同），W=256 ⇒ ~2.4M 参数;
  · 损失/优化：L1 + AdamW(2e-4, cosine, warmup 100) + 固定 `--epochs`/`--steps` 预算;
  · 指标：val(1000 张) 的 PSNR / SSIM / L1。

对照组（不需要下载、用来给"信息量"定位）：
  · `randvit_l`: 同结构 ViT-L 随机初始化（**没训练过** ⇒ 可解码信息下限）;
  · `pixel16`: 不做编码, 直接把图 adaptive-avg-pool 到 16×16×3（768 值的平凡码）;
  · `sdvae`: SD-VAE 的 latent（4×32×32=4096 值, 生成式编码器参考上限）;
  · `raw_*`: 见 REGISTRY。

用法：
  python3 probe_enc_dec.py --enc clip_l14 --out /tmp/clip_l14.json
  python3 probe_enc_dec.py --list
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────── 编码器注册表 ───────────────────────────
# mean/std 用各家预处理口径（不下载 processor, 避免额外文件/版本坑）。
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SD_MEAN, SD_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

REGISTRY = {
    # ── OpenAI ──
    "clip_l14":     dict(hf="openai/clip-vit-large-patch14", family="clip", res=224,
                         mean=CLIP_MEAN, std=CLIP_STD, vendor="OpenAI", note="CLIP ViT-L/14"),
    "clip_b32":     dict(hf="openai/clip-vit-base-patch32", family="clip", res=224,
                         mean=CLIP_MEAN, std=CLIP_STD, vendor="OpenAI", note="CLIP ViT-B/32"),
    # ── Google ──
    "siglip_so400m": dict(hf="google/siglip-so400m-patch14-384", family="siglip", res=384,
                          mean=SD_MEAN, std=SD_STD, vendor="Google", note="SigLIP SO400M/14"),
    "vit_l16":      dict(hf="google/vit-large-patch16-224-in21k", family="vit", res=224,
                         mean=IN_MEAN, std=IN_STD, vendor="Google", note="ViT-L/16 (IN-21k 监督)"),
    "vit_h14":      dict(hf="google/vit-huge-patch14-224-in21k", family="vit", res=224,
                         mean=IN_MEAN, std=IN_STD, vendor="Google", note="ViT-H/14 (IN-21k 监督)"),
    # ── Meta ──
    "dinov2_l":     dict(hf="facebook/dinov2-large", family="dino", res=224,
                         mean=IN_MEAN, std=IN_STD, vendor="Meta", note="DINOv2-L/14 (自监督)"),
    "dinov2_g":     dict(hf="facebook/dinov2-giant", family="dino", res=224,
                         mean=IN_MEAN, std=IN_STD, vendor="Meta", note="DINOv2-g/14 (自监督)"),
    "mae_h":        dict(hf="facebook/vit-mae-huge", family="mae", res=224,
                         mean=IN_MEAN, std=IN_STD, vendor="Meta", note="MAE ViT-H/14 (自监督)"),
    "sam_h":        dict(hf="facebook/sam-vit-huge", family="sam", res=1024,
                         mean=IN_MEAN, std=IN_STD, vendor="Meta", note="SAM ViT-H image encoder"),
    # ── Alibaba ──
    "qwen2vl_2b":   dict(hf="Qwen/Qwen2-VL-2B-Instruct", family="qwenvl", res=448,
                         mean=IN_MEAN, std=IN_STD, vendor="Alibaba", note="Qwen2-VL 2B ViT"),
    # ── 生成式参考 / 对照 ──
    "sdvae":        dict(hf="stabilityai/sd-vae-ft-mse", family="sdvae", res=256,
                         mean=SD_MEAN, std=SD_STD, vendor="Stability", note="SD-VAE (生成式, 4096 值)"),
    "pixel16":      dict(family="pixel", res=256, mean=SD_MEAN, std=SD_STD,
                         vendor="—", note="16×16 像素下采样（768 值, 无编码器）"),
    "randvit_l":    dict(hf="google/vit-large-patch16-224-in21k", family="randvit", res=224,
                         mean=IN_MEAN, std=IN_STD, vendor="—", note="ViT-L 随机初始化（对照下限）"),
}


# ─────────────────────────── 数据 ───────────────────────────
def list_split(data_dir, train_per_class, val_per_class):
    """按类确定划分：val = 每类前 val_per_class 张, train = 之后 train_per_class 张。"""
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    tr, va = [], []
    for c in classes:
        fs = sorted(f for f in os.listdir(os.path.join(data_dir, c))
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
        va += [os.path.join(data_dir, c, f) for f in fs[:val_per_class]]
        tr += [os.path.join(data_dir, c, f) for f in fs[val_per_class:val_per_class + train_per_class]]
    return tr, va


def _fit256(im):
    """短边缩到 256 + center crop 256（与评测口径一致, 不做增强）。"""
    w, h = im.size
    s = 256.0 / min(w, h)
    im = im.resize((max(256, int(round(w * s))), max(256, int(round(h * s)))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - 256) // 2, (h - 256) // 2
    return im.crop((left, top, left + 256, top + 256))


class ReconDataset(torch.utils.data.Dataset):
    def __init__(self, paths, enc_res, mean, std):
        self.paths, self.res = paths, int(enc_res)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = _fit256(Image.open(self.paths[i]).convert("RGB"))
        tgt = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0     # (3,256,256)
        enc = im.resize((self.res, self.res), Image.BICUBIC)
        x = np.asarray(enc, dtype=np.float32).transpose(2, 0, 1) / 255.0     # (3,res,res)
        x = (torch.from_numpy(x) - self.mean) / self.std
        return x.contiguous(), torch.from_numpy(tgt).contiguous()


# ─────────────────────────── 编码器适配 ───────────────────────────
class EncoderAdapter:
    """统一接口: `forward(x)` → (B,C,h,w) 空间特征图（已脱离 CLS/全局 token）。"""

    def __init__(self, spec, layer=1.0, device="cuda"):
        self.spec, self.device = spec, device
        self.layer = layer
        self.family = spec["family"]
        self.res = int(spec["res"])
        self.model = self._build(layer)

    def _build(self, layer):
        fam = self.family
        hf = self.spec.get("hf")
        from transformers import AutoModel, AutoConfig
        if self.family == "pixel":
            return None
        if self.family == "random":
            from transformers import ViTModel
            cfg = AutoConfig.from_pretrained(hf)
            return ViTModel(cfg).to(self.device).eval()
        if fam == "randvit":
            from transformers import ViTModel, AutoConfig
            cfg = AutoConfig.from_pretrained(hf)
            return ViTModel(cfg).to(self.device).eval()      # 随机权重, 不加载 ckpt
        if fam == "clip":
            from transformers import CLIPVisionModel
            return CLIPVisionModel.from_pretrained(hf, torch_dtype=torch.bfloat16)
        if fam == "siglip":
            from transformers import SiglipVisionModel
            return SiglipVisionModel.from_pretrained(hf, torch_dtype=torch.bfloat16)
        if fam == "vit":
            from transformers import ViTModel
            return ViTModel.from_pretrained(hf, torch_dtype=torch.bfloat16)
        if fam == "dino":
            from transformers import Dinov2Model
            return Dinov2Model.from_pretrained(hf, torch_dtype=torch.bfloat16)
        if fam == "mae":
            from transformers import ViTMAEModel
            return ViTMAEModel.from_pretrained(hf, torch_dtype=torch.bfloat16)
        if fam == "sam":
            from transformers import SamModel
            return SamModel.from_pretrained(hf, torch_dtype=torch.bfloat16)
        if fam == "sdvae":
            from diffusers import AutoencoderKL
            return AutoencoderKL.from_pretrained(hf, local_files_only=False,
                                                 torch_dtype=torch.float32)
        if fam == "qwenvl":
            self._qwen_tokenizer = None
            from transformers import Qwen2VLForConditionalGeneration
            return Qwen2VLForConditionalGeneration.from_pretrained(
                hf, torch_dtype=torch.bfloat16)
        raise ValueError(f"未知 family {fam}")

    def to(self):
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        return self

    def _take_layer(self, hs, layer):
        n = len(hs)
        idx = int(round(float(layer) * (n - 1))) if isinstance(layer, float) and layer <= 1.0 \
            else int(layer)
        idx = max(0, min(n - 1, idx))
        return hs[idx], idx

    @torch.no_grad()
    def forward(self, x):
        fam = self.family
        if fam == "pixel":
            # 16×16×3 = 768 值的"平凡码"（无编码器）; 与 grid=16 对齐
            f = F.adaptive_avg_pool2d(x.float(), (16, 16))
            return f
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if fam in ("clip", "siglip", "vit", "dino", "mae", "randvit"):
                o = self.model(pixel_values=x, output_hidden_states=True)
                hs = o.hidden_states
                f, self.layer_idx = self._take_layer(hs, self.layer)
                f = f.float()
                if fam in ("clip", "vit", "dino", "mae", "randvit"):
                    f = f[:, 1:]                       # 丢 CLS（有）
                n = f.shape[1]
                g = int(round(math.sqrt(n)))
                assert g * g == n, f"token 数 {n} 不是平方数（grid {g}）"
                return f.transpose(1, 2).reshape(f.shape[0], f.shape[-1], g, g)
            if fam == "sam":
                f = self.model.get_image_features(pixel_values=x)   # (B,256,64,64)
                return f.float()
            if fam == "sdvae":
                z = self.model.encode(x.float() * 2.0 - 1.0).latent_dist.mode()
                return z.float()                                     # (B,4,32,32)
            if fam == "qwenvl":
                vis = self.model.model.visual if hasattr(self.model, "model") else self.model.visual
                # 动态分辨率: 每张图 grid = (res/14, res/14)
                pv = x.float()
                patch = 14
                gh = gw = self.res // patch
                grid = torch.tensor([[1, gh, gw]] * pv.shape[0], dtype=torch.long,
                                    device=pv.device)
                out = vis(pv, grid_thw=grid)
                f = out[0] if isinstance(out, (tuple, list)) else out
                f = f.float()                                        # (B, tokens, C)
                return f.transpose(1, 2).reshape(pv.shape[0], -1, gh, gw)
        raise ValueError(fam)


# ─────────────────────────── 解码器 / 指标 ───────────────────────────
class Decoder(nn.Module):
    """特征 (B,C,g,g) → 图 (B,3,256,256)。所有编码器共用（只换第一层输入通道）。"""

    def __init__(self, cin, width=256, grid=16, out=256):
        super().__init__()
        self.grid = int(grid)
        self.proj = nn.Conv2d(cin, width, 1)
        n_up = int(round(math.log2(out // self.grid)))
        assert 2 ** n_up * self.grid == out
        blocks = []
        for _ in range(n_up):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(width, width, 3, 1, 1),
                       nn.GroupNorm(8, width), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(width, 3, 3, 1, 1)

    def forward(self, f):
        if f.shape[-1] != self.grid:
            f = (F.adaptive_avg_pool2d(f, (self.grid, self.grid))
                 if f.shape[-1] > self.grid
                 else F.interpolate(f, size=(self.grid, self.grid), mode="bilinear",
                                    align_corners=False))
        return torch.sigmoid(self.head(self.body(self.proj(f))))


def _gauss_window(dev, k=11, s=1.5):
    x = torch.arange(k, device=dev, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * s ** 2))
    g = g / g.sum()
    return g.view(1, 1, 1, k)


def ssim(a, b, dev):
    """a,b: (B,3,H,W) ∈[0,1] ⇒ 平均 SSIM（11×11 gaussian, 标准口径）。"""
    g = _gauss_window(dev)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    out = 0.0
    for c in range(a.shape[1]):
        x, y = a[:, c:c + 1], b[:, c:c + 1]
        k = g.expand(1, 1, -1, -1)
        k2 = k.expand(1, 1, -1, -1)
        mu_x = F.conv2d(x, k2, padding=5)
        mu_y = F.conv2d(y, k2, padding=5)
        sx = F.conv2d(x * x, k2, padding=5) - mu_x ** 2
        sy = F.conv2d(y * y, k2, padding=5) - mu_y ** 2
        sxy = F.conv2d(x * y, k2, padding=5) - mu_x * mu_y
        s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
            ((mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2))
        out = out + s.mean()
    return float(out / a.shape[1])


def evaluate(dec, enc, loader, dev):
    dec.eval()
    ps, ss, l1, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, tgt in loader:
            x, tgt = x.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            f = enc.forward(x)
            r = dec(f.float())
            mse = ((r - tgt) ** 2).mean(dim=(1, 2, 3))
            ps += float((-10.0 * torch.log10(mse.clamp_min(1e-12))).sum())
            l1 += float((r - tgt).abs().mean(dim=(1, 2, 3)).sum())
            ss += ssim(r, tgt, dev) * r.shape[0]
            n += r.shape[0]
    return dict(psnr=ps / n, ssim=ss / n, l1=l1 / n, n=n)


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="clip_l14")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--train_per_class", type=int, default=120, help="每类训练图数（×100 类）")
    ap.add_argument("--val_per_class", type=int, default=10, help="每类验证图数（×100 类）")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grid", type=int, default=16, help="统一特征网格（所有编码器相同）")
    ap.add_argument("--width", type=int, default=256, help="解码器宽度")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--layer", type=float, default=1.0,
                    help="ViT 取第几层（0..1 比例, 或整数层号; 默认最后一层）")
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时截断（debug 用）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.list:
        for k, v in REGISTRY.items():
            print(f"{k:16s} {v['vendor']:9s} res={v['res']:4d} {v['note']}")
        return

    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = REGISTRY[args.enc]

    tr_p, va_p = list_split(args.data_dir, args.train_per_class, args.val_per_class)
    ds_tr = ReconDataset(tr_p, spec["res"], spec["mean"], spec["std"])
    ds_va = ReconDataset(va_p, spec["res"], spec["mean"], spec["std"])
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                                        num_workers=args.workers, pin_memory=False,
                                        drop_last=True, persistent_workers=args.workers > 0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                                        num_workers=args.workers, pin_memory=False)

    t0 = time.time()
    enc = EncoderAdapter(spec, layer=args.layer, device=dev).to()
    t_load = time.time() - t0
    cin = int(enc.forward(torch.zeros(2, 3, spec["res"], spec["res"], device=dev)).shape[1])
    featshape = tuple(enc.forward(torch.zeros(1, 3, spec["res"], spec["res"], device=dev)).shape[1:])

    dec = Decoder(cin, width=args.width, grid=args.grid, out=256).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_ep = max(1, len(ds_tr) // args.batch)
    total = max(1, int(steps_per_ep * args.epochs))
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos")

    print(f"[probe] enc={args.enc} vendor={spec['vendor']} res={spec['res']} "
          f"layer={args.layer} feat={featshape} cin={cin} grid={args.grid} "
          f"train={len(ds_tr)} val={len(ds_va)} steps={total} "
          f"params(dec)={sum(p.numel() for p in dec.parameters())/1e6:.2f}M "
          f"load={t_load:.1f}s", flush=True)

    it = iter(dl_tr)
    t1 = time.time()
    run_l1, run_n = 0.0, 0
    for st in range(1, total + 1):
        try:
            x, tgt = next(it)
        except StopIteration:
            it = iter(dl_tr)
            x, tgt = next(it)
        x, tgt = x.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
        f = enc.forward(x)
        r = dec(f.float())
        loss = F.l1_loss(r, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        sched.step()
        run_l1 += float(loss) * r.shape[0]
        run_n += r.shape[0]
        if st % 50 == 0 or st == 1:
            el = time.time() - t1
            print(f"[probe] step {st}/{total} loss={run_l1/max(1,run_n):.4f} "
                  f"{run_n/max(1e-6,el):.0f}img/s elapsed={el/60:.1f}m", flush=True)
            run_l1, run_n = 0.0, 0
    t_train = time.time() - t1

    m = evaluate(dec, enc, dl_va, dev)
    m.update(enc=args.enc, vendor=spec["vendor"], note=spec["note"],
             enc_res=spec["res"], layer=args.layer, feat_shape=list(featshape),
             feat_values=int(np.prod(featshape)), grid=args.grid,
             dec_width=args.width, dec_params=int(sum(p.numel() for p in dec.parameters())),
             train_images=len(ds_tr), val_images=len(ds_va), steps=total,
             load_sec=round(t_load, 1), train_sec=round(t_train, 1),
             img_per_s=round(len(ds_tr) * args.epochs / max(1e-6, t_train), 1),
             tag=args.tag, gpu=os.environ.get("MUSA_VISIBLE_DEVICES", ""))
    print(f"[probe] RESULT {args.enc}: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
          f"L1={m['l1']:.4f} (feat {featshape})", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(m, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
