#!/usr/bin/env python3
"""qwen_probe_v2.py — 探针 v2：**用 AssocVTC 240M 档的真实解码器类**（RecurrentOutputQueryDecoder）
+ 从 240M ckpt（B 相 t16）热启动，在 Qwen 视觉 token 的 PCA 截断上做单步还原。

与 v1 的差别（v1 用 model_v2.py 的 OutputQueryDecoder，缺 mem_proj/mem_norm/mem_pos/query_pos，
并非 AssocVTC 240M 档的真实结构）：
  · 解码器 = `model_v2_linattn.RecurrentOutputQueryDecoder(enc_dim=5120, dim=1440, depth=7,
    heads=16, mlp_ratio=4.0, patch=16, num_patches=196, num_specials=49, pixel_hidden=1024,
    steps=[49])` → **242.73M**（其中 stack+query+pos+pixel_head 与 240M 档 237.11M 逐 key 同构，
    只有 mem_proj 由 (1248→1440) 变宽为 (5120→1440) ⇒ +5.58M）
  · pixel 目标空间 = **DINOv2 归一化**（mean/std 与 ckpt 训练一致），PSNR 仍按 0-255 口径算
  · 其余（数据、PCA 截断、PSNR/SSIM 口径、评测集）与 v1 相同
"""
import argparse, io, json, math, os, sys, time, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = "/root/autodl-tmp/qwen_vtc/code"
sys.path.insert(0, ROOT)
from model_v2_linattn import RecurrentOutputQueryDecoder  # noqa: E402
from metrics_linattn import psnr, ssim  # noqa: E402
import qwen_probe as QP  # 复用数据类/Projector/meanbase 等  # noqa: E402

RES, PATCH = 224, 16
N_PATCH = (RES // PATCH) ** 2
PATCH_PX = 3 * PATCH * PATCH
N_TOK, MEM_DIM = 49, 5120
DIM, DEPTH, HEADS, PIXEL_HIDDEN = 1440, 7, 16, 1024
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0
R = "/root/autodl-tmp/qwen_vtc"


class ProbeV2(nn.Module):
    def __init__(self, mem_dim=MEM_DIM, dim=DIM, depth=DEPTH, heads=HEADS,
                 num_patches=N_PATCH, num_specials=N_TOK, mlp_ratio=4.0,
                 pixel_hidden=PIXEL_HIDDEN):
        super().__init__()
        self.decoder = RecurrentOutputQueryDecoder(
            enc_dim=mem_dim, dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio,
            patch=PATCH, num_patches=num_patches, num_specials=num_specials,
            pixel_hidden=pixel_hidden, steps=[num_specials], step_plan="square")

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens):
        B = tokens.shape[0]
        z_cls = tokens.mean(dim=1, keepdim=True)          # 无 z_cls 来源 ⇒ 用 49 token 均值
        pix = self.decoder(z_cls, tokens)                 # (B,1,N,PATCH_PX)
        return pix[:, 0]


def img_to_tensor(u8):
    """uint8 (B,224,224,3) → DINOv2 归一化 (B,3,224,224)"""
    x = u8.permute(0, 3, 1, 2).float()
    return (x - DINO_MEAN.to(x.device)) / DINO_STD.to(x.device)


def img_to_patches(x):
    B, C, H, W = x.shape
    g = H // PATCH
    return x.reshape(B, C, g, PATCH, g, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C * PATCH * PATCH)


def patches_to_img01(pix):
    """(B,196,768) DINOv2 归一化 → (B,224,224,3) in [0,1]"""
    B = pix.shape[0]
    im = pix.reshape(B, 14, 14, 3, PATCH, PATCH).permute(0, 1, 4, 2, 5, 3).reshape(B, RES, RES, 3)
    im = im * DINO_STD.view(1, 1, 1, 3).to(im.device) + DINO_MEAN.view(1, 1, 1, 3).to(im.device)
    return (im.clamp(0, 255) / 255.0)


def build_model(device="cuda", init=None):
    m = ProbeV2().to(device)
    if init:
        ck = torch.load(init, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"[init] loaded {init}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        print(f"[init] missing[:8]={missing[:8]} unexpected[:8]={unexpected[:8]}", flush=True)
        m._init_info = {"missing": missing, "unexpected": unexpected}
    return m


@torch.no_grad()
def evaluate(model, loader, proj, device="cuda", max_batches=None):
    model.eval()
    P, S, L, cnt = [], [], [], 0
    for b, (idx, tok, img) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        tok = tok.to(device, non_blocking=True).float()
        tok = proj(tok)
        u8 = img.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pix = model(tok)
        pred = patches_to_img01(pix.float()).permute(0, 3, 1, 2).clamp(0, 1)
        tgt = (u8.permute(0, 3, 1, 2).float() / 255.0)
        P.append(psnr(pred, tgt, max_val=1.0).cpu())
        S.append(ssim(pred, tgt).cpu())
        L.append(F.l1_loss(pred, tgt, reduction="none").mean(dim=(1, 2, 3)).cpu() * 255.0)
        cnt += len(idx)
    model.train()
    P, S, L = torch.cat(P), torch.cat(S), torch.cat(L)
    return {"n": cnt, "psnr": P.mean().item(), "ssim": S.mean().item(),
            "l1_255": L.mean().item(), "psnr_all": P, "ssim_all": S, "l1_all": L}


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out; os.makedirs(out, exist_ok=True)
    names = json.load(open(os.path.join(R, "tokens", "train_index.json")))["entries"]
    ds = QP.TokensWithImages(os.path.join(R, "tokens", "train_tokens_int8.npy"),
                             os.path.join(R, "tokens", "train_scale.npy"), names)
    ref = QP.build_dequant(device)
    proj = QP.Projector(args.pca_dir, args.r, device)
    val_ds = QP.TokensWithValImages(os.path.join(R, "tokens", "val_tokens_fp16.npy"),
                                    os.path.join(R, "tokens", "val_images_224.npy"))
    rng = np.random.default_rng(123)
    sub_idx = np.sort(rng.choice(len(val_ds), size=min(512, len(val_ds)), replace=False))
    sub_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_ds, sub_idx),
                                             batch_size=args.eval_bs, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    if args.subset:
        idx = np.sort(np.random.default_rng(0).choice(len(ds), size=min(args.subset, len(ds)), replace=False))
        ds = torch.utils.data.Subset(ds, idx)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                                     prefetch_factor=8 if args.workers else None)
    model = build_model(device, args.init)
    print(f"[probe_v2] r={args.r} params={model.n_params()/1e6:.2f}M", flush=True)
    opt = QP.make_opt(model, args.lr, args.weight_decay)
    steps_per_epoch = len(dl); total = steps_per_epoch * args.epochs
    warm = min(args.warmup, max(total // 10, 1))
    def lr_at(step):
        if step < warm: return args.lr * (step + 1) / max(warm, 1)
        t = (step - warm) / max(total - warm, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    log = open(os.path.join(out, "train_log.jsonl"), "w")
    json.dump({k: v for k, v in vars(args).items()}, open(os.path.join(out, "args.json"), "w"), indent=1)
    best = -1e9; gstep = 0; t0 = time.time()
    for ep in range(args.epochs):
        for idx, tok, sc, img in dl:
            for g in opt.param_groups: g["lr"] = lr_at(gstep)
            tok = tok.to(device, non_blocking=True).float()
            sc = sc.to(device, non_blocking=True).float()
            tok = QP.dequant(tok, sc, ref)
            with torch.no_grad():
                tok = proj(tok)
            u8 = img.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pix = model(tok)
                loss = F.l1_loss(pix, img_to_patches(img_to_tensor(u8)))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); gstep += 1
            if gstep % 20 == 0:
                rec = {"step": gstep, "ep": ep, "loss": loss.item(), "lr": lr_at(gstep),
                       "imgs": gstep * args.bs, "img_s": gstep * args.bs / (time.time() - t0)}
                log.write(json.dumps(rec) + "\n"); log.flush()
                if gstep % 200 == 0: print(json.dumps(rec), flush=True)
        m = evaluate(model, sub_loader, proj, device)
        print(f"[probe_v2] ep{ep} subset PSNR {m['psnr']:.4f} SSIM {m['ssim']:.4f} L1 {m['l1_255']:.3f} | "
              f"{gstep} steps {(time.time()-t0)/60:.1f} min", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "ep": ep,
                    "subset_psnr": m["psnr"]}, os.path.join(out, "last.pt"))
        torch.save({"model": {k: v.half() for k, v in model.state_dict().items()}, "args": vars(args),
                    "ep": ep, "subset_psnr": m["psnr"]}, os.path.join(out, "best.pt"))
        json.dump({"best_subset_psnr": max(best, m["psnr"]), "steps": gstep},
                  open(os.path.join(out, "train_state.json"), "w"))
        best = max(best, m["psnr"])
    print(f"[probe_v2] DONE best subset {best:.4f} in {(time.time()-t0)/60:.1f} min", flush=True)
    log.close()


@torch.no_grad()
def eval_mode(args):
    device = "cuda"
    val_ds = QP.TokensWithValImages(os.path.join(R, "tokens", "val_tokens_fp16.npy"),
                                    os.path.join(R, "tokens", "val_images_224.npy"))
    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    proj = QP.Projector(args.pca_dir, args.r, device)
    model = build_model(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = {k: v.float() for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    m = evaluate(model, loader, proj, device)
    res = {"arm_r": args.r, "ckpt": args.ckpt, "ckpt_ep": ck.get("ep"), "n": m["n"],
           "psnr": m["psnr"], "ssim": m["ssim"], "l1_255": m["l1_255"],
           "params_M": model.n_params() / 1e6, "subset_psnr_at_ckpt": ck.get("subset_psnr")}
    json.dump(res, open(args.out, "w"), indent=1)
    np.save(args.out.replace(".json", "_per_image.npy"),
            {"psnr": m["psnr_all"].tolist(), "ssim": m["ssim_all"].tolist(), "l1_255": m["l1_all"].tolist()})
    print("[eval]", json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--pca_dir", default=os.path.join(R, "pca"))
    ap.add_argument("--r", type=int, default=5120)
    ap.add_argument("--out", default=os.path.join(R, "probe", "v2debug"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--init", default=os.path.join(R, "probe", "init_m240_decoder.pt"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "train": train(args)
    else: eval_mode(args)


if __name__ == "__main__":
    main()
