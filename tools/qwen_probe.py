#!/usr/bin/env python3
"""qwen_probe.py — 「同参数量 AssocVTC 交叉注意力还原探针」（单步）在 Qwen 视觉 token 上的
PCA 截断还原实验。

设计（2026-10-30）:
  · memory = Qwen3.8-27B 仅图像编码器输出的 **49 个合并 token（5120 维）**，按 token 顺序；
    PCA 在**整个 IN-100 训练集池化**的 token 上拟合（train_pca 产物）。
  · 「1σ/2σ/3σ/全部」= 保留累计方差 68.27% / 95.45% / 99.73% / 100% 的主成分，
    输入 = 池化均值 + 前 r 个主成分重建（`mean + U_r U_r^T (x-mean)`），
    其中 r 由 pca_xyz.json 给出（`dims_1sigma/2sigma/3sigma`）。
  · 探针 = AssocVTC 240M 档的**解码器**（`OutputQueryDecoder`, dim=1440, depth=7,
    heads=16, N=196, K=49）+ mem_proj(5120→1440) + 学习 z_cls，**单步**（steps=[49]，
    一次读完全部 49 个 memory token），PixelHead → 224² 像素。参数量 244.69M（decoder
    主体 237.31M ≈ AssocVTC 240M 档 237.11M，+7.37M 输入适配）。
  · 反量化兼容两版 token 存储契约（初版 per-token absmax / B2 位置中心化+逐通道归一,
    后者见 tokens/quant_ref.npz；脚本自动检测）。
  · **patch=16**（224/16=14 ⇒ N=196；与仓库 224² 从零实验线 model_v2_linattn 的
    16×16×3=768 像素 patch 口径一致）。⚠️ 2026-10-30 修正：原稿写成 `(RES//14)**2`
    会得 256 而注释写 196，且 model_v2.patches_to_image 硬编码 patch=14（224² 需 256
    patch），与 N=196 冲突直接 assert 崩溃；现改用 model_v2_linattn 的 16px 版
    image_to_patches/patches_to_image（布局口径完全相同，仅 patch 边长 14→16）。
  · 4 个 arm 共用**同一架构/同一初始化种子/同一数据顺序/同一预算**，唯一自变量 = r。

用法:
  python tools/qwen_probe.py --mode bench
  python tools/qwen_probe.py --mode train --r 5120 --epochs 8 --out /root/autodl-tmp/qwen_vtc/probe/r5120
  python tools/qwen_probe.py --mode eval  --ckpt .../best.pt --r <r> --out .../eval_r<r>.json
  python tools/qwen_probe.py --mode meanbase
"""
import argparse
import io
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/root/autodl-tmp/qwen_vtc/code")

from model_v2 import OutputQueryDecoder, PixelHead  # noqa: E402
from metrics_linattn import psnr, ssim  # noqa: E402

RES = 224
PATCH = 16                      # 224/16 = 14 ⇒ N=196（与 240M 档从零实验线一致）
N_PATCH = (RES // PATCH) ** 2   # 196
PATCH_PX = 3 * PATCH * PATCH    # 768
N_TOK = 49
MEM_DIM = 5120
DIM = 1440                      # AssocVTC 240M 档解码器宽度
DEPTH = 7
HEADS = 16
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

R = "/root/autodl-tmp/qwen_vtc"
# 训练集 zip：默认 AutoDL 公共盘；QWEN_ZIP 可指向本地副本（提 token 时已复制 14.4GB 到 /root/img100.zip）
ZIP_PATH = os.environ.get("QWEN_ZIP", "/root/autodl-pub/ImageNet100/imagenet100.zip")


# ───────────────────────── 模型 ─────────────────────────
class CrossAttnProbe(nn.Module):
    def __init__(self, mem_dim=MEM_DIM, dim=DIM, depth=DEPTH, heads=HEADS,
                 num_patches=N_PATCH, num_specials=N_TOK, mlp_ratio=4.0,
                 steps=None):
        super().__init__()
        self.mem_dim = mem_dim
        self.dim = dim
        self.num_patches = num_patches
        self.num_specials = num_specials
        self.mem_proj = nn.Linear(mem_dim, dim)
        self.z_cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.decoder = OutputQueryDecoder(
            dim=dim, num_patches=num_patches, num_specials=num_specials,
            steps=steps if steps is not None else [num_specials],
            depth=depth, heads=heads, mlp_ratio=mlp_ratio)
        self.pixel_head = PixelHead(dim=dim, patch_px=PATCH_PX)
        self.target_pix = None

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens):
        """tokens: (B,49,5120) float → pixels (B,196,768)（归一化空间）"""
        B = tokens.shape[0]
        h = self.mem_proj(tokens)                      # (B,49,1440)
        zc = self.z_cls.expand(B, -1, -1)
        Y = self.decoder(zc, h)                        # (B,1,196,1440)
        return self.pixel_head(Y)[:, 0]                # (B,196,768)


def image_to_patches(x):
    """(B,3,224,224) [-1,1] → (B,196,768)（16px patch, row-major, 通道优先）

    布局 = model_v2_linattn.image_to_patches(x, patch=16)（本文件内联同一实现,
    免去远端多带一个 57KB 依赖；与主线 model_v2 的 target_pix 口径相同, 仅 patch
    边长 14→16, 见 model_v2_linattn.py 的布局权威定义）。
    """
    B, C, H, W = x.shape
    gh, gw = H // PATCH, W // PATCH
    return x.reshape(B, C, gh, PATCH, gw, PATCH) \
            .permute(0, 2, 4, 1, 3, 5) \
            .reshape(B, gh * gw, C * PATCH * PATCH)


def patches_to_pixels(pred_pix):
    """(B,196,768) 归一化空间 → (B,224,224,3) in [0,1]

    严格逆 = model_v2_linattn.patches_to_image(..., patch=16, clamp_255=False)
    （mean/std 用 [0,1] 标度 0.5/0.5, 反归一化后直接落 [0,1], 不再 /255）。
    """
    B, N, px = pred_pix.shape
    assert px == PATCH_PX and N == N_PATCH, (px, N)
    img = pred_pix.reshape(B, RES // PATCH, RES // PATCH, 3, PATCH, PATCH) \
                  .permute(0, 3, 1, 4, 2, 5) \
                  .reshape(B, 3, RES, RES)
    m = torch.as_tensor(MEAN, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    sd = torch.as_tensor(STD, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    img = img * sd + m
    return img.permute(0, 2, 3, 1)


# ───────────────────────── 数据 ─────────────────────────
class TokensWithImages(torch.utils.data.Dataset):
    """训练用: int8 token memmap + zip JPEG（target）。"""

    def __init__(self, tok_path, scale_path, names):
        self.tok = np.load(tok_path, mmap_mode="r")
        self.scale = np.load(scale_path, mmap_mode="r")
        self.names = names
        self._z = None
        assert len(self.tok) == len(names), (len(self.tok), len(names))

    def __len__(self):
        return len(self.tok)

    def __getitem__(self, i):
        if self._z is None:
            self._z = zipfile.ZipFile(ZIP_PATH)
        with self._z.open(self.names[i]) as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        a = np.asarray(img.resize((RES, RES), Image.BICUBIC), np.uint8)
        return i, torch.from_numpy(np.asarray(self.tok[i])), \
            torch.from_numpy(np.asarray(self.scale[i])), torch.from_numpy(a)


class _IndexedSubset(torch.utils.data.Dataset):
    """按固定索引取子集（4 臂同子集、同预算）。__getitem__ 仍返回全局 index。"""

    def __init__(self, base, idx):
        self.base = base
        self.idx = np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[int(self.idx[i])]


class TokensWithValImages(torch.utils.data.Dataset):
    """评测用: fp16 token memmap + 已落盘 val 图像 (5000,224,224,3) uint8。"""

    def __init__(self, tok_path, img_path):
        self.tok = np.load(tok_path, mmap_mode="r")
        self.imgs = np.load(img_path, mmap_mode="r")

    def __len__(self):
        return len(self.tok)

    def __getitem__(self, i):
        return i, torch.from_numpy(np.asarray(self.tok[i])), torch.from_numpy(np.asarray(self.imgs[i]))


def _img_to_tensor(a):
    """uint8 (B,224,224,3) → (B,3,224,224) float in [-1,1]"""
    x = a.permute(0, 3, 1, 2).float() / 255.0
    return x * 2.0 - 1.0


class Projector:
    """PCA 截断投影: mean + U_r U_r^T (x - mean)，r>=dim 时恒等。"""

    def __init__(self, pca_dir, r, device="cuda", dim=MEM_DIM):
        self.r = int(r)
        self.identity = self.r >= dim
        if not self.identity:
            mean = np.load(os.path.join(pca_dir, "mean.npy"))
            comps = np.load(os.path.join(pca_dir, "components.npy"), mmap_mode="r")
            assert comps.shape[0] == dim and self.r <= comps.shape[1], comps.shape
            U = np.asarray(comps[:, : self.r], dtype=np.float32)
            self.mean = torch.from_numpy(mean.astype(np.float32)).to(device)
            self.U = torch.from_numpy(U).to(device)             # (D,r)
            self.P = None

    def __call__(self, x):
        """x: (B,49,5120) float32 on device → 同 shape"""
        if self.identity:
            return x
        B, S, D = x.shape
        xf = x.reshape(B * S, D)
        z = torch.matmul(xf - self.mean, self.U)                # (B*S, r)
        out = torch.matmul(z, self.U.t()) + self.mean
        return out.reshape(B, S, D)


# ───────────────────────── 反量化（兼容两版存储契约） ─────────────────────────
def build_dequant(device="cuda"):
    """返回反量化上下文 ref。

    两版 token 存储契约（2026-10-30）:
      v1（初版, per-token absmax）:      x = q * sc
      v2（B2, 位置中心化+逐通道归一+逐 token int8, 有 tokens/quant_ref.npz）:
          x = (q * sc) * sd + mu_p,  mu_p (49,5120), sd (5120,)
        —— 量化参考 mu_p/sd 由 val 5000 张 fp16 token 估计（只影响存储刻度）。
    自动按 quant_ref.npz 是否存在选择；两种情况都对 (B,49,5120) int8→float32。
    """
    p = os.path.join(R, "tokens", "quant_ref.npz")
    if os.path.exists(p):
        z = np.load(p)
        mu = torch.from_numpy(z["mu_p"].astype(np.float32)).to(device).view(1, N_TOK, MEM_DIM)
        sd = torch.from_numpy(z["sd"].astype(np.float32)).to(device).view(1, 1, MEM_DIM)
        return ("b2", mu, sd)
    return ("v1", None, None)


def dequant(q, sc, ref):
    """q: (B,49,5120) float（int8 转来）, sc: (B,49) float → 同 shape float32"""
    mode, mu, sd = ref
    y = q * sc.unsqueeze(-1)
    if mode == "b2":
        return y * sd + mu
    return y


# ───────────────────────── ckpt 保存（fp16 省盘） ─────────────────────────
def save_ckpt(path, model, args, ep, subset_psnr):
    """state_dict 存 fp16（fp32 每份 0.98GB → fp16 0.49GB；AutoDL 盘只剩 ~13GB）。

    fp16 只是**落盘压缩**：加载回 fp32 模型时 load_state_dict 自动 copy_ 升回 fp32,
    数值 = fp32 训练值的 fp16 舍入（相对 1e-3, 对 PSNR/SSIM 影响 <0.001 dB）。
    """
    sd = {k: (v.detach().half() if v.is_floating_point() else v.detach().clone())
          for k, v in model.state_dict().items()}
    torch.save({"model": sd, "args": vars(args), "ep": ep,
                "subset_psnr": subset_psnr, "ckpt_dtype": "fp16"}, path)


# ───────────────────────── engine ─────────────────────────
def build_model(device="cuda"):
    m = CrossAttnProbe().to(device)
    return m


def load_meta():
    meta = json.load(open(os.path.join(R, "tokens", "meta.json")))
    return meta


def make_opt(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model, loader, proj, device="cuda", max_batches=None):
    model.eval()
    P, S, L, cnt = [], [], [], 0
    for b, (idx, tok, img) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        tok = tok.to(device, non_blocking=True).float()
        tok = proj(tok)
        img = _img_to_tensor(img).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pix = model(tok)
        pred = patches_to_pixels(pix.float())          # (B,224,224,3) in [0,1]
        tgt = (img.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1)
        pr = pred.permute(0, 3, 1, 2).clamp(0, 1)
        tg = tgt.permute(0, 3, 1, 2)
        P.append(psnr(pr, tg, max_val=1.0).cpu())
        S.append(ssim(pr, tg).cpu())
        L.append(F.l1_loss(pr, tg, reduction="none").mean(dim=(1, 2, 3)).cpu() * 255.0)
        cnt += len(idx)
    model.train()
    P, S, L = torch.cat(P), torch.cat(S), torch.cat(L)
    return {"n": cnt, "psnr": P.mean().item(), "ssim": S.mean().item(),
            "l1_255": L.mean().item(), "psnr_all": P, "ssim_all": S, "l1_all": L}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out
    os.makedirs(out, exist_ok=True)
    meta = load_meta()
    names = json.load(open(os.path.join(R, "tokens", "train_index.json")))["entries"]
    ds = TokensWithImages(os.path.join(R, "tokens", "train_tokens_int8.npy"),
                          os.path.join(R, "tokens", "train_scale.npy"), names)
    if args.limit:
        ds.tok = ds.tok[: args.limit]
        ds.names = ds.names[: args.limit]
    if args.subset_idx:
        idx = np.load(args.subset_idx)
        ds = _IndexedSubset(ds, idx)
        print(f"[probe] subset_idx {args.subset_idx}: {len(idx)} 张（seed 固定；4 臂同子集）", flush=True)
    proj = Projector(args.pca_dir, args.r, device)
    val_ds = TokensWithValImages(os.path.join(R, "tokens", "val_tokens_fp16.npy"),
                                 os.path.join(R, "tokens", "val_images_224.npy"))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    eval_idx = np.arange(len(val_ds)) if not args.eval_n else \
        np.random.default_rng(123).choice(len(val_ds), size=min(args.eval_n, len(val_ds)), replace=False)
    sub_ds = torch.utils.data.Subset(val_ds, np.sort(eval_idx))
    sub_loader = torch.utils.data.DataLoader(sub_ds, batch_size=args.eval_bs, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True,
                                     num_workers=args.workers, pin_memory=True,
                                     drop_last=True, persistent_workers=args.workers > 0,
                                     prefetch_factor=4 if args.workers else None)
    model = build_model(device)
    dq = build_dequant(device)
    nparam = model.n_params()
    print(f"[probe] arm r={args.r} params={nparam/1e6:.3f}M quant={dq[0]}", flush=True)
    opt = make_opt(model, args.lr, args.weight_decay)
    steps_per_epoch = len(dl)
    total_steps = steps_per_epoch * args.epochs
    warm = min(args.warmup, total_steps // 10)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / max(warm, 1)
        t = (step - warm) / max(total_steps - warm, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    scaler = torch.autocast("cuda", dtype=torch.bfloat16)
    log = open(os.path.join(out, "train_log.jsonl"), "w")
    json.dump({k: v for k, v in vars(args).items()}, open(os.path.join(out, "args.json"), "w"), indent=1)
    best = -1e9
    gstep = 0
    t0 = time.time()
    for ep in range(args.epochs):
        for b, (idx, tok, scale, img) in enumerate(dl):
            lr = lr_at(gstep)
            for g in opt.param_groups:
                g["lr"] = lr
            tok = tok.to(device, non_blocking=True).float()
            sc = scale.to(device, non_blocking=True).float()
            with torch.no_grad():
                tok = proj(dequant(tok, sc, dq))              # dequantize（v1/v2 兼容）
            img = _img_to_tensor(img).to(device)
            with scaler:
                pix = model(tok)
                loss = F.l1_loss(pix, image_to_patches(img))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
            if gstep % 20 == 0:
                rec = {"step": gstep, "ep": ep, "loss": loss.item(), "lr": lr,
                       "imgs": gstep * args.bs,
                       "img_s": gstep * args.bs / (time.time() - t0)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if gstep % 200 == 0:
                    print(json.dumps(rec), flush=True)
            if args.max_steps and gstep >= args.max_steps:
                break
        m = evaluate(model, sub_loader, proj, device, max_batches=None)
        print(f"[probe] ep{ep} subset PSNR {m['psnr']:.4f} SSIM {m['ssim']:.4f} "
              f"L1 {m['l1_255']:.3f} | {gstep} steps {(time.time()-t0)/60:.1f} min", flush=True)
        lastp = os.path.join(out, "last.pt")
        save_ckpt(lastp, model, args, ep, m["psnr"])
        if m["psnr"] > best:
            best = m["psnr"]
            save_ckpt(os.path.join(out, "best.pt"), model, args, ep, m["psnr"])
        json.dump({"best_subset_psnr": best, "steps": gstep},
                  open(os.path.join(out, "train_state.json"), "w"))
        if args.max_steps and gstep >= args.max_steps:
            break
    # 只留 best.pt（last.pt 仅训练中用于崩溃恢复；盘只剩 ~13GB, 4 臂不能都留）
    try:
        os.remove(os.path.join(out, "last.pt"))
    except OSError:
        pass
    print(f"[probe] DONE best subset PSNR {best:.4f} in {(time.time()-t0)/60:.1f} min", flush=True)
    log.close()


@torch.no_grad()
def eval_mode(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = TokensWithValImages(os.path.join(R, "tokens", "val_tokens_fp16.npy"),
                                 os.path.join(R, "tokens", "val_images_224.npy"))
    if args.limit:
        val_ds.tok = val_ds.tok[: args.limit]
    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    proj = Projector(args.pca_dir, args.r, device)
    model = build_model(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    m = evaluate(model, loader, proj, device)
    res = {"arm_r": args.r, "ckpt": args.ckpt, "ckpt_ep": ck.get("ep"),
           "n": m["n"], "psnr": m["psnr"], "ssim": m["ssim"], "l1_255": m["l1_255"],
           "params_M": model.n_params() / 1e6,
           "subset_psnr_at_ckpt": ck.get("subset_psnr")}
    json.dump(res, open(args.out, "w"), indent=1)
    pd = {"idx": list(range(len(m["psnr_all"]))),
          "psnr": m["psnr_all"].tolist(), "ssim": m["ssim_all"].tolist(),
          "l1_255": m["l1_all"].tolist()}
    np.save(args.out.replace(".json", "_per_image.npy"), pd)
    print("[eval]", json.dumps({k: v for k, v in res.items()}))


def meanbase(args):
    """平凡基线: 用训练集平均图（= 池化均值图）预测 val 的 PSNR。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = json.load(open(os.path.join(R, "tokens", "train_index.json")))["entries"]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(names), size=args.n, replace=False)
    acc = np.zeros((RES, RES, 3), np.float64)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for i in sel:
            a = np.asarray(Image.open(io.BytesIO(z.read(names[i]))).convert("RGB")
                           .resize((RES, RES), Image.BICUBIC), np.float64)
            acc += a
    mean_img = acc / len(sel)
    imgs = np.load(os.path.join(R, "tokens", "val_images_224.npy"))
    mean_t = torch.from_numpy(mean_img.astype(np.float32) / 255.0).permute(2, 0, 1)
    # 分块算指标: ssim 的 depthwise conv 在整批 5000 上 im2col 会爆内存（>100GB）
    bs = max(1, int(args.eval_bs))
    P, S, L = [], [], []
    for i0 in range(0, len(imgs), bs):
        tgt = torch.from_numpy(imgs[i0:i0 + bs].astype(np.float32) / 255.0) \
            .permute(0, 3, 1, 2)
        pred = mean_t.unsqueeze(0).expand_as(tgt).contiguous()
        P.append(psnr(pred, tgt, max_val=1.0))
        S.append(ssim(pred, tgt))
        L.append(F.l1_loss(pred, tgt, reduction="none").mean(dim=(1, 2, 3)) * 255.0)
    P, S, L = torch.cat(P), torch.cat(S), torch.cat(L)
    res = {"psnr": P.mean().item(), "ssim": S.mean().item(), "l1_255": L.mean().item(),
           "n_train_images_for_mean": len(sel), "n_val": int(len(P))}
    json.dump(res, open(args.out, "w"), indent=1)
    np.save(os.path.join(os.path.dirname(args.out), "mean_image_224.npy"), mean_img.astype(np.float32))
    print("[meanbase]", json.dumps(res))


def bench(args):
    device = "cuda"
    meta = load_meta()
    names = json.load(open(os.path.join(R, "tokens", "train_index.json")))["entries"]
    ds = TokensWithImages(os.path.join(R, "tokens", "train_tokens_int8.npy"),
                          os.path.join(R, "tokens", "train_scale.npy"), names)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs,
                                     shuffle=bool(args.bench_shuffle),
                                     num_workers=args.workers,
                                     pin_memory=True, drop_last=True,
                                     persistent_workers=args.workers > 0,
                                     prefetch_factor=args.prefetch if args.workers else None)
    model = build_model(device)
    proj = Projector(args.pca_dir, args.r, device)
    dq = build_dequant(device)
    opt = make_opt(model, args.lr, args.weight_decay)
    print(f"[bench] r={args.r} bs={args.bs} workers={args.workers} "
          f"shuffle={bool(args.bench_shuffle)} prefetch={args.prefetch} "
          f"zip={os.path.basename(ZIP_PATH)} params={model.n_params()/1e6:.3f}M quant={dq[0]}")
    n = 0
    t0 = time.time()
    for b, (idx, tok, scale, img) in enumerate(dl):
        tok = tok.to(device, non_blocking=True).float()
        sc = scale.to(device, non_blocking=True).float()
        with torch.no_grad():
            tok = proj(dequant(tok, sc, dq))
        img = _img_to_tensor(img).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pix = model(tok)
            loss = F.l1_loss(pix, image_to_patches(img))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n += args.bs
        if b == 5:
            t0 = time.time()
            n = 0
        if b >= 25:
            break
    el = time.time() - t0
    print(f"[bench] {n/el:.1f} img/s  ({el/n*1000:.2f} ms/img)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "bench", "meanbase"], default="bench")
    ap.add_argument("--pca_dir", default=os.path.join(R, "pca"))
    ap.add_argument("--r", type=int, default=5120)
    ap.add_argument("--out", default=os.path.join(R, "probe", "debug"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--bench_shuffle", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subset_idx", default="")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_mode(args)
    elif args.mode == "meanbase":
        meanbase(args)
    else:
        bench(args)


if __name__ == "__main__":
    main()
