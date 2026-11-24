#!/usr/bin/env python3
"""四家厂商视觉编码器的**可解码性探针**（冻结 vision tower → 同一解码器还原 256）。

模型：DeepSeek-V4.1-Flash-1017 / GLM-5.3-Flash / Qwen3.8-Flash-Next / MiMo-V2.6-Flash-RL。
这些仓库的 LLM 主体 180–510GB，我们**只取 vision tower 权重**（`tools/fetch_vision_weights.py`
按 safetensors byte-range 抽出，见 /root/autodl-tmp/vis_enc/*.safetensors）。

协议与 `tools/probe_enc_dec.py` 完全一致（同数据集划分/同解码器/同预算/同目标 256×256），
差别只在：每个编码器用**自己原生分辨率**输入 + 自己的 patch/merge 规则 + 原生特征网格
（不压缩空间）,特征取 vision tower 的标准输出（GLM/Qwen/MiMo 取 merger 之后,即送进 LLM
的 token；DeepSeek 取其 ViT 输出,它没有内置 merger）。

用法：
  python3 probe_enc_dec_vendor.py --vendor glm5_flash --selftest
  python3 probe_enc_dec_vendor.py --vendor glm5_flash --out /root/autodl-tmp/tok256x12_t8/probe_vendor/glm5_flash.json \
      --save_grid .../grids/glm5_flash.png
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_enc_dec import Decoder, ReconDataset, list_split, evaluate    # noqa: E402

ROOT = os.environ.get("VIS_ENC_DIR", "/root/autodl-tmp/vis_enc")

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HALF_MEAN, HALF_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

VENDORS = {
    "glm5_flash": dict(
        repo="zai-org/GLM-5.3-Flash", vendor="Zai/GLM", cfg="glm",
        ckpt="glm_visual.safetensors", strip="model.visual.",
        res=448, patch=14, temporal=2, merge=2,
        mean=CLIP_MEAN, std=CLIP_STD, note="GLM-5.3-Flash vision (24L, 1024d, patch14)"),
    "qwen38_flash": dict(
        repo="Qwen/Qwen3.8-Flash-Next", vendor="Alibaba/Qwen", cfg="qwen",
        ckpt="qwen_visual.safetensors", strip="model.visual.",
        res=512, patch=16, temporal=2, merge=2,
        mean=HALF_MEAN, std=HALF_STD, note="Qwen3.8-Flash-Next vision (27L, 1152d, patch16)"),
    "mimo26_flash": dict(
        repo="XiaomiMiMo/MiMo-V2.6-Flash-RL", vendor="Xiaomi/MiMo", cfg="mimo",
        ckpt="mimo_visual.safetensors", strip="visual.",
        res=448, patch=16, temporal=2, merge=2,
        mean=IN_MEAN, std=IN_STD, note="MiMo-V2.6-Flash vision (28L, 1280d, patch16, window attn)"),
    "deepseek_v41_flash": dict(
        repo="deepseek-ai/DeepSeek-V4.1-Flash-1017", vendor="DeepSeek", cfg="ds",
        ckpt="ds_visual_1.safetensors", strip="vision.",
        res=448, patch=14, temporal=1, merge=1,
        mean=HALF_MEAN, std=HALF_STD, note="DeepSeek-V4.1-Flash-1017 ViT (32L, 1024d, patch14)"),
}


# ─────────────────────── patchify（与各家 processor 同序）───────────────────────
def patchify_conv3d(x, p, t):
    """(B,3,H,W) → (B, N, C*t*p*p), N=(H/p)*(W/p)；顺序 = C,T,p,p（对应 Conv3d 权重）。"""
    B, C, H, W = x.shape
    gh, gw = H // p, W // p
    y = x[:, :, None].expand(B, C, t, H, W)
    y = y.reshape(B, C, t, gh, p, gw, p).permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    return y.view(B, gh * gw, C * t * p * p), gh, gw


def patchify_2d(x, p):
    """(B,3,H,W) → (B, N, C*p*p)；顺序 = C,p,p（DeepSeek 的 ViT PatchEmbed 用 Linear 吃这个）。"""
    B, C, H, W = x.shape
    gh, gw = H // p, W // p
    y = x.reshape(B, C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).contiguous()
    return y.view(B, gh * gw, C * p * p), gh, gw


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VisionEnc(nn.Module):
    def __init__(self, key, dev):
        super().__init__()
        self.key, self.dev = key, dev
        s = VENDORS[key]
        self.s, self.res = s, int(s["res"])
        vc = json.load(open(f"{ROOT}/cfg_{s['cfg']}.json"))["vision_config"]
        if key == "glm5_flash":
            from transformers.models.glm5_next import Glm5NextVisionConfig, Glm5NextVisionModel
            self.m = Glm5NextVisionModel(Glm5NextVisionConfig(**vc))
            self.kind = "hf"
        elif key == "qwen38_flash":
            from transformers.models.qwen4_exp import Qwen4ExpVisionConfig, Qwen4ExpVisionModel
            self.m = Qwen4ExpVisionModel(Qwen4ExpVisionConfig(**vc))
            self.kind = "hf"
        elif key == "mimo26_flash":
            sys.path.insert(0, f"{ROOT}/code")
            import importlib
            mm = importlib.import_module("mimo_pkg.modeling_mimo_v2")
            self.m = mm.MiMoVisionTransformer(SimpleNamespace(**vc))
            self.kind = "mimo"
        elif key == "deepseek_v41_flash":
            mod = _load_module(f"{ROOT}/code/ds_vision.py", "ds_vision")
            args = SimpleNamespace(
                vision_dim=vc["hidden_size"], vision_n_layers=vc["num_hidden_layers"],
                vision_n_heads=vc["num_attention_heads"], vision_inter_dim=vc["intermediate_size"],
                vision_patch_size=vc["patch_size"], vision_rope_theta=vc["rope_theta"],
                vision_downsample_ratio=vc["downsample_ratio"],
                vision_max_n_token=vc["max_image_tokens"],
                vision_min_pixels=vc["min_pixels"],
                vision_max_wh_ratio=vc.get("max_wh_ratio"))
            _orig = mod.get_vision_cos_sin
            def _cos_sin(nh, nw, dim, theta, _o=_orig, _d=dev):
                c, s = _o(nh, nw, dim, theta)
                return c.to(_d), s.to(_d)
            mod.get_vision_cos_sin = _cos_sin
            self.m = mod.ViT(args)
            self.kind = "ds"
        else:
            raise SystemExit(f"unknown vendor {key}")

        from safetensors.torch import load_file
        sd = load_file(os.path.join(ROOT, s["ckpt"]))
        sd = {k[len(s["strip"]):]: v for k, v in sd.items() if k.startswith(s["strip"])}
        missing, unexpected = self.m.load_state_dict(sd, strict=False)
        n_missing = len([k for k in missing])
        print(f"[vendor] {key}: ckpt keys={len(sd)} missing={n_missing} "
              f"unexpected={len(unexpected)}", flush=True)
        if n_missing or unexpected:
            print(f"[vendor]   missing[:6]={list(missing)[:6]}")
            print(f"[vendor]   unexpected[:6]={list(unexpected)[:6]}")
        # ckpt 里没有的键（本仓库 modeling 与训练时 bias 开关略不同）⇒ 置 0，等价于"无 bias"
        if missing:
            named = dict(self.m.named_parameters())
            with torch.no_grad():
                for k in missing:
                    if k in named and named[k].dim() >= 1:
                        named[k].zero_()
            print(f"[vendor]   zeroed {len(missing)} missing params", flush=True)
        self.m = self.m.to(dev).eval()
        if any(p.dtype == torch.float32 for p in self.m.parameters()):
            # 权重保存为 fp32；推理用 bf16（与各家训练 dtype 一致）
            pass
        self.m = self.m.to(torch.bfloat16)
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        s, B = self.s, x.shape[0]
        p, t = s["patch"], s["temporal"]
        if self.kind == "ds":
            patches, gh, gw = patchify_2d(x, p)
            outs = []
            for b in range(B):
                o = self.m(patches[b].to(torch.bfloat16), gh, gw)
                outs.append(o.float().reshape(gh, gw, -1).permute(2, 0, 1))
            return torch.stack(outs, 0)
        hidden, gh, gw = patchify_conv3d(x, p, t)
        grid = torch.tensor([[1, gh, gw]] * B, device=x.device, dtype=torch.long)
        o = self.m(hidden.to(torch.bfloat16), grid)
        if not isinstance(o, torch.Tensor):      # HF 返回 BaseModelOutputWithPooling
            o = getattr(o, "last_hidden_state", None)
            if o is None:
                o = o[0] if isinstance(o, tuple) else o
        m = s["merge"]
        gh2, gw2 = gh // m, gw // m
        o = o.float()
        if o.dim() == 2:                        # (total_tokens, C)
            o = o.reshape(B, gh2 * gw2, -1)
        elif o.dim() == 3:                      # (B, tokens, C)
            pass
        else:
            raise SystemExit(f"unexpected vision out {tuple(o.shape)}")
        return o.transpose(1, 2).reshape(B, -1, gh2, gw2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", default="glm5_flash",
                    help=" | ".join(VENDORS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--train_per_class", type=int, default=200)
    ap.add_argument("--val_per_class", type=int, default=10)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--save_grid", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.list:
        for k, v in VENDORS.items():
            print(f"{k:22s} {v['vendor']:14s} res={v['res']} {v['note']}")
        return

    dev = "cuda"
    torch.manual_seed(0)
    spec = VENDORS[args.vendor]
    t0 = time.time()
    enc = VisionEnc(args.vendor, dev)
    print(f"[vendor] built {args.vendor} in {time.time()-t0:.1f}s", flush=True)
    if args.selftest:
        x = torch.rand(2, 3, spec["res"], spec["res"], device=dev) * 2 - 1
        f = enc.forward(x)
        print(f"[vendor] SELFTEST ok: out {tuple(f.shape)} "
              f"mean={f.mean().item():.4f} std={f.std().item():.4f} "
              f"absmax={f.abs().max().item():.3f}", flush=True)
        return

    tr_p, va_p = list_split(args.data_dir, args.train_per_class, args.val_per_class)
    ds_tr = ReconDataset(tr_p, spec["res"], spec["mean"], spec["std"])
    ds_va = ReconDataset(va_p, spec["res"], spec["mean"], spec["std"])
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                                        num_workers=args.workers, pin_memory=False,
                                        drop_last=True, persistent_workers=args.workers > 0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                                        num_workers=args.workers, pin_memory=False)

    fs = tuple(enc.forward(torch.zeros(1, 3, spec["res"], spec["res"], device=dev)).shape[1:])
    g_native = int(fs[-1])
    grid = (g_native if g_native in (8, 16, 32, 64)
            else int(2 ** round(math.log2(g_native))))
    grid = max(8, min(64, grid))
    cin = int(fs[0])
    dec = Decoder(cin, width=args.width, grid=grid, out=256).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01)

    steps_per_ep = max(1, len(ds_tr) // args.batch)
    total = max(1, int(steps_per_ep * args.epochs))
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos")
    print(f"[vendor] {args.vendor} feat={fs} cin={cin} grid={g_native}->{grid} "
          f"train={len(ds_tr)} val={len(ds_va)} steps={total} "
          f"dec={sum(p.numel() for p in dec.parameters())/1e6:.2f}M", flush=True)

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
        run_l1 += float(loss.detach()) * r.shape[0]
        run_n += r.shape[0]
        if st % 50 == 0 or st == 1:
            el = time.time() - t1
            print(f"[vendor] step {st}/{total} loss={run_l1/max(1,run_n):.4f} "
                  f"{run_n/max(1e-6,el):.0f}img/s elapsed={el/60:.1f}m", flush=True)
            run_l1, run_n = 0.0, 0
    t_train = time.time() - t1

    m = evaluate(dec, enc, dl_va, dev)
    if args.save_grid:
        from PIL import ImageDraw
        dec.eval()
        with torch.no_grad():
            x, tgt = next(iter(dl_va))
            x4, t4 = x[:4].to(dev), tgt[:4].to(dev)
            r4 = dec(enc.forward(x4).float()).clamp(0, 1)
        n, cell = x4.shape[0], 256
        canvas = Image.new("RGB", (cell * 2, cell * n), (255, 255, 255))
        for i in range(n):
            gt = Image.fromarray((t4[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            rc = Image.fromarray((r4[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            canvas.paste(gt, (0, i * cell)); canvas.paste(rc, (cell, i * cell))
        ImageDraw.Draw(canvas).text(
            (6, 4), f"GT | recon  {args.vendor}  PSNR={m['psnr']:.2f}", fill=(255, 0, 0))
        os.makedirs(os.path.dirname(args.save_grid), exist_ok=True)
        canvas.save(args.save_grid)

    m.update(vendor=args.vendor, lab=spec["vendor"], note=spec["note"], repo=spec["repo"],
             enc_res=spec["res"], feat_shape=list(fs), feat_values=int(np.prod(fs)),
             grid=grid, grid_native=g_native, grid_mode="native",
             dec_width=args.width, dec_params=int(sum(p.numel() for p in dec.parameters())),
             train_images=len(ds_tr), val_images=len(ds_va), steps=total,
             train_sec=round(t_train, 1),
             img_per_s=round(len(ds_tr) * args.epochs / max(1e-6, t_train), 1),
             tag=args.tag, gpu=os.environ.get("MUSA_VISIBLE_DEVICES", ""))
    print(f"[vendor] RESULT {args.vendor}: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
          f"L1={m['l1']:.4f} (feat {fs})", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(m, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
