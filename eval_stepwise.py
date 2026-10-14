"""逐 step 评测：MultiResSR 有没有「逐步精炼」。

做法（COCO2017 val 留出集，训练只用了 train2017）：
1. 每个样本按训练同一套分桶喂编码器，encode 一次；
2. 4 个解码器各自跑完整 12 步（square 计划：1,4,…,144），拿到**每一步的直接预测**；
3. 反归一化到 0-255，逐步算 L1 / PSNR；
4. 输出：逐分辨率逐步曲线、best-fit 聚合曲线、逐图最优停步分布、ΔL1 衰减、
   以及每步的累计 token 数。

2026-10-14 改动（配合 ×10 解码器与通用化）:
  · 模型结构从 **ckpt 里的 `args`** 重建（`decoder_width/depth/heads/mlp_ratio/
    num_specials/step_plan`）——否则 ×10 的 ckpt 会因形状不符 load 失败；
  · 设备/精度走 `srdiff_runtime`，不再硬编码 `cuda:0` / `autocast("cuda")` /
    `/root/autodl-tmp`；
  · 留出集默认 = `<data_root>/coco/val2017`（`tools/prep_data.py` 的下游布局）。

产物：json + 打印表格。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from data_multi import (DEFAULT_DATA_ROOT, DINO_MEAN, DINO_STD, BucketBatchSampler,
                        MultiResCollator, MultiSourceDataset, build_index)
from model_multi import RESOS, MultiResSR
from srdiff_runtime import Runtime

ENV = os.environ.get
STD = torch.tensor(DINO_STD, dtype=torch.float32).view(3, 1)
MEAN = torch.tensor(DINO_MEAN, dtype=torch.float32).view(3, 1)


def denorm255(p: torch.Tensor) -> torch.Tensor:
    """(...,588) 归一化 → 0-255；588 = (3,196) 通道优先。"""
    sh = p.shape[:-1]
    return (p.reshape(*sh, 3, 196) * STD.to(p.device) + MEAN.to(p.device)
            ).reshape(*sh, 3 * 14 * 14)


def build_from_ckpt(ck, dino_dir, attn_impl, rt, log):
    """按 ckpt 里保存的 args 重建模型（×10 配置也能正确重建）。"""
    a = ck.get("args") or {}
    dim = None
    from transformers.models.dinov2 import Dinov2Model
    dino = Dinov2Model.from_pretrained(dino_dir, local_files_only=True,
                                       attn_implementation=attn_impl)
    dim = dino.config.hidden_size
    kwargs = dict(
        num_specials=int(a.get("num_specials", 144)), dim=dim,
        heads=int(a.get("heads", 8)), depth=int(a.get("decoder_depth", 2)),
        mlp_ratio=float(a.get("mlp_ratio", 4.0)),
        step_plan=a.get("step_plan", "square"), block=int(a.get("block", 12)),
        decoder_width=int(a.get("decoder_width", 0)),
    )
    log(f"[eval] 从 ckpt args 重建模型: {kwargs}")
    model = MultiResSR(dino, **kwargs)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"[eval] state_dict 不匹配: missing={len(missing)} "
                         f"unexpected={len(unexpected)}（ckpt 与代码代际不一致？）")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        ENV("SRDIFF_OUTPUT_DIR", "/root/autodl-tmp/output/multires_10x"),
        "final_model.pt"))
    ap.add_argument("--dino_dir", default=ENV("SRDIFF_DINO_DIR",
                                              "/root/autodl-tmp/models/dinov2-large"))
    ap.add_argument("--data_root", default=os.path.join(DEFAULT_DATA_ROOT, "coco", "val2017"),
                    help="留出集图片目录（默认 COCO2017 val2017）")
    ap.add_argument("--cache", default="", help="默认 <data_root>/../index_coco_val.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--patch_budget", type=int, default=6144)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device_type", default=ENV("SRDIFF_DEVICE", "auto"))
    ap.add_argument("--attn_implementation", default="auto")
    ap.add_argument("--out", default="", help="默认 <ckpt 目录>/eval_stepwise.json")
    args = ap.parse_args()

    rt = Runtime(args.device_type)
    DEV = rt.device()
    cache = args.cache or os.path.join(os.path.dirname(os.path.abspath(args.data_root)),
                                       "index_coco_val.json")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   "eval_stepwise.json")

    def log(*a):
        print(*a, flush=True)

    # 注意 build_index 的 cap 是 files[:cap] ⇒ 0 会取 0 张，这里用大数表示"不截断"
    src = [(args.data_root, (".jpg", ".jpeg", ".png"), 10 ** 9)]
    idx = build_index(src, cache, 1280)
    if args.limit:
        idx = idx[:args.limit]
    log(f"[eval] 留出集 {len(idx)} 张 from {args.data_root} | device={rt}")

    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_from_ckpt(ck, args.dino_dir, rt.attn_implementation(
        args.attn_implementation), rt, log)
    model = model.to(DEV).eval()
    T = len(model.steps)
    log(f"[eval] ckpt step={ck.get('step')} steps={model.steps} "
        f"decoder_width={model.decoder_width or model.dim}")

    ds = MultiSourceDataset(idx, RESOS)
    shapes = [(bh, bw) for (_, bw, bh, _) in idx]
    sampler = BucketBatchSampler(shapes, args.batch_size, shuffle=False,
                                 drop_last=False, rank=0, world_size=1,
                                 patch_budget=args.patch_budget)
    coll = MultiResCollator()

    n_res = len(RESOS)
    sum_l1 = np.zeros((n_res, T))          # 逐分辨率逐步 L1
    sum_mse = np.zeros((n_res, T))
    cnt = np.zeros(n_res)
    sum_l1_bf = np.zeros(T)                # best-fit 聚合
    sum_mse_bf = np.zeros(T)
    sum_l1_const = np.zeros(n_res)         # 每图均色常数基线
    per_img_best_curve = []
    t0 = time.time()
    n_img = 0
    for bi, batch_ids in enumerate(sampler):
        items = [ds[i] for i in batch_ids]
        b = coll(items)
        px = b["pixel_values"].to(DEV)
        tgt = [t.to(DEV) for t in b["targets"]]
        best = b["best_idx"].to(DEV)
        B = px.shape[0]
        with torch.no_grad(), rt.amp_context(args.precision):
            z_cls, z_s = model.encode(px)
            all_l1 = torch.zeros(B, n_res, T, device=DEV)
            all_mse = torch.zeros(B, n_res, T, device=DEV)
            for di, dec in enumerate(model.decoders):
                Y = dec(z_cls, z_s)                     # (B,T,N_i,D)
                pix = model.pixel_head(Y)               # (B,T,N_i,588)
                t255 = denorm255(tgt[di].float())       # (B,N_i,588)
                mu = t255.mean(dim=(1, 2), keepdim=True)   # 逐图均色基线
                sum_l1_const[di] += float((t255 - mu).abs().mean(dim=(1, 2)).sum())
                for t in range(T):
                    p255 = denorm255(pix[:, t].float())
                    d = (p255 - t255)
                    all_l1[:, di, t] = d.abs().mean(dim=(1, 2))
                    all_mse[:, di, t] = (d ** 2).mean(dim=(1, 2))
            bf = all_l1[torch.arange(B, device=DEV), best, :]        # (B,T)
            all_l1_np = all_l1.cpu().numpy()
            all_mse_np = all_mse.cpu().numpy()
            bf_np = bf.cpu().numpy()
            for di in range(n_res):
                sum_l1[di] += all_l1_np[:, di, :].sum(0)
                sum_mse[di] += all_mse_np[:, di, :].sum(0)
                cnt[di] += B
            sum_l1_bf += bf_np.sum(0)
            sum_mse_bf += all_mse_np[np.arange(B), best.cpu().numpy(), :].sum(0)
            per_img_best_curve.append(bf_np)
            n_img += B
        if (bi + 1) % 50 == 0:
            log(f"  {n_img} 张, {time.time() - t0:.0f}s")

    L1 = sum_l1 / cnt[:, None]                 # (n_res,T)
    PSNR = 10 * np.log10(255.0 ** 2 / (sum_mse / cnt[:, None]))
    L1_bf = sum_l1_bf / n_img
    PSNR_bf = 10 * np.log10(255.0 ** 2 / (sum_mse_bf / n_img))
    const = sum_l1_const / cnt
    curves = np.concatenate(per_img_best_curve, 0)          # (n,T) best-fit 逐图
    best_step = curves.argmin(1)

    log(f"\n=== 留出集 n={n_img}（COCO2017 val，训练未见）===")
    log("step  t   z_s累计  " + "".join(f"{w}x{h:<6}" for w, h in RESOS) + "best-fit  PSNR_bf")
    cum = [3, 8, 15, 24, 35, 48, 63, 80, 99, 120, 143, 144]
    for t in range(T):
        row = f"{t + 1:4d} {model.steps[t]:4d} {cum[t]:8d}  "
        row += "".join(f"{L1[i, t]:9.2f} " for i in range(n_res))
        row += f"{L1_bf[t]:9.2f} {PSNR_bf[t]:9.2f}"
        log(row)
    log("\n均色常数基线(0-255 L1): " +
        "  ".join(f"{w}x{h}={const[i]:.2f}" for i, (w, h) in enumerate(RESOS)))
    log(f"best-fit 首步 {L1_bf[0]:.2f} → 末步 {L1_bf[-1]:.2f} → 最优 {L1_bf.min():.2f}"
        f" @step{int(L1_bf.argmin()) + 1}")
    d = np.diff(L1_bf)
    drop = L1_bf[0] - L1_bf.min()
    log("逐步 ΔL1(best-fit, 负=变好): " + " ".join(f"{x:+.2f}" for x in d))
    if drop > 0:
        for k in (1, 2, 3, 4, 6, 12):
            if k <= T:
                log(f"  前 {k:2d} 步已拿到总降幅的 {(L1_bf[0] - L1_bf[k - 1]) / drop * 100:5.1f}%")
    hist = np.bincount(best_step, minlength=T)
    log("逐图最优停步分布: " + " ".join(f"{i + 1}:{h}" for i, h in enumerate(hist)))
    oracle = curves.min(1).mean()
    log(f"oracle(逐图选最优步) {oracle:.3f} vs 固定末步 {curves[:, -1].mean():.3f} "
        f"→ 早停最大收益 {curves[:, -1].mean() - oracle:.3f} px")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"n": n_img, "steps": model.steps, "cum_tokens": cum,
               "decoder_width": int(model.decoder_width or model.dim),
               "L1_per_reso": L1.tolist(), "PSNR_per_reso": PSNR.tolist(),
               "L1_bestfit": L1_bf.tolist(), "PSNR_bestfit": PSNR_bf.tolist(),
               "const_baseline": const.tolist(),
               "best_step_hist": hist.tolist(),
               "oracle": float(oracle)}, open(out, "w"), indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
