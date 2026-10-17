#!/usr/bin/env python3
"""E-B2 判据探针（linattn 版）：README §4.1 三条 + 逐区域隔离 L1
================================================================================
背景：`doc/2026-09-11/probe_step_collapse.py` 是同一判据的历史探针，但它绑在
`model_v2.py` + `data_v2.py`（DINOv2 主线 + parquet），不能直接跑 AssocVTC 的
linattn 臂。本脚本用**同一份判据定义**在 linattn 上重实现，只加一次前向。

产出（每个 ckpt）::

  · step_px_scale[t]      = mean |Y_pix[:, t]|          （坍缩主判据; 归一化空间 + 0-255 空间）
  · scale_ratio[t]        = step_px_scale[t] / step_px_scale[0]
  · prog_curve[t]         = 第 t 步直接预测的 0-255 L1（= eval_stepwise 的 L1_255[t] 口径）
  · E_px[t][r]            = 第 t 步预测在**第 r 个区域行带**（region_slices(N,T)）上的 0-255 L1
  · E_px_diag[t]          = E_px[t][t]（"各步隔离 L1 在自己区域上相近"的判据量）

⚠️ 口径边界（必须随数字一起报）:
  1. 判据 (1) 的阈值 0.015 出自旧探针（`model_v2` 的 pixel_head 输出空间、576 patch/5 区）；
     本脚本是 196 patch/8 区、PixelHead(dim→768)，**不能假定绝对尺度可比** ⇒ 同时给出
     归一化与 0-255 两列，阈值可比性交 alit-report 裁定。
  2. `E_px` 的区域取 `region_slices(N, T)` 的 patch 下标区间（与 `--dh_mode 1` 的硬区域
     掩码**同一函数**），L1 在 patch 内先对 (3,patch,patch) 求均值再对区域求均值。

用法::

    python doc/2026-10-16/artifacts/alit/probe_region_linattn.py \
      --ckpt /root/autodl-tmp/srdiff_alit/b2_dh1/ckpt_best.pt \
      --out doc/2026-10-16/artifacts/alit/probe_region_b2_dh1.json
    # 自洽校验（可选）: --check_stepwise <stepwise_<arm>.json> 对比 prog_curve
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _ in range(8):
    if os.path.exists(os.path.join(_HERE, "model_v2_linattn.py")):
        sys.path.insert(0, _HERE)
        break
    _HERE = os.path.dirname(_HERE)
else:
    raise SystemExit("找不到仓库根（model_v2_linattn.py）")

from data_linattn import build_datasets                      # noqa: E402
from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD,   # noqa: E402
                              LinAttnConfig, SRPhase1LinAttn, region_slices)
from torch.utils.data import DataLoader, Subset              # noqa: E402

MEAN_255 = [m * 255.0 for m in IMAGENET_MEAN]
STD_255 = [s * 255.0 for s in IMAGENET_STD]


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    model = SRPhase1LinAttn(cfg).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"ckpt 有多余 key: {unexpected[:5]}"
    assert set(missing) <= {"decoder.mem_pos"}, f"未加载参数: {list(missing)[:8]}"
    model.eval()
    return model, cfg, ck, list(missing)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--num_images", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--check_stepwise", default="",
                    help="可选的 stepwise json, 用于自洽校验 prog_curve")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, va = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    n = min(len(va), args.num_images) if args.num_images > 0 else len(va)
    dl = DataLoader(Subset(va, range(n)), batch_size=args.batch_size, shuffle=False,
                    drop_last=False, num_workers=args.num_workers, pin_memory=True)
    print(f"[probe] val n={n} device={dev} amp={args.amp}", flush=True)

    out = {"n_images": n, "results": {},
           "space_note": ("判据(1)阈值 0.015 出自 model_v2 旧探针空间; 本表给归一化与 0-255 "
                          "两列, 可比性交 alit-report 裁定"),
           "E_px_definition": ("区域=region_slices(N,T) patch 下标区间(与 --dh_mode 1 同一函数); "
                               "L1 先在 patch 内对 (3,patch,patch) 求均值, 再对区域/图求均值")}

    for ckpt in args.ckpt:
        model, cfg, ck, missing = load_model(ckpt, dev)
        T = len(model.decoder.windows)
        regions = region_slices(cfg.num_patches, T)
        p = cfg.patch
        # ⚠️ 两个不同维数的 scale 张量: pr 是 6 维 (B,T,N,3,p,p), tg 是 5 维 (B,N,3,p,p);
        #    用 6 维 scale 乘 5 维 tg 会在**前面插入一维** ⇒ 广播时 N 与 T 错位。
        s6 = torch.tensor(STD_255, device=dev).view(1, 1, 1, 3, 1, 1)
        s5 = torch.tensor(STD_255, device=dev).view(1, 1, 3, 1, 1)

        sum_scale_norm = np.zeros(T)
        sum_scale_255 = np.zeros(T)
        sum_prog = np.zeros(T)
        sum_E = np.zeros((T, len(regions)))
        seen = 0
        for x, _ in dl:
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=bool(dev == "cuda" and args.amp)):
                o = model(x)
            pred = o["Y_pix"].float()                     # (B,T,N,3pp) 归一化空间
            tgt = o["target_pix"].float()                 # (B,N,3pp)
            B = pred.shape[0]
            N = cfg.num_patches
            pr = pred.reshape(B, T, N, 3, p, p)
            tg = tgt.reshape(B, N, 3, p, p)
            pr255 = pr * s6
            tg255 = tg * s5
            # 逐步尺度（坍缩判据）
            sum_scale_norm += pr.abs().mean(dim=(2, 3, 4, 5)).sum(0).cpu().numpy()
            sum_scale_255 += pr255.abs().mean(dim=(2, 3, 4, 5)).sum(0).cpu().numpy()
            # 逐步 0-255 L1（= eval_stepwise 的 L1_255 口径: 每图先对所有像素求均值）
            err = (pr255 - tg255.unsqueeze(1)).abs().mean(dim=(3, 4, 5))   # (B,T,N)
            sum_prog += err.mean(dim=2).sum(0).cpu().numpy()
            for r, (lo, hi) in enumerate(regions):
                sum_E[:, r] += err[:, :, lo:hi].mean(dim=2).sum(0).cpu().numpy()
            seen += B

        res = {
            "ckpt": ckpt, "step": ck.get("step"), "epoch": ck.get("epoch"),
            "T": T, "n_patches": cfg.num_patches,
            "cum_z_s": [int(min((int(np.sqrt(t)) + 1) ** 2 - 1, cfg.num_specials))
                        for t in (model.decoder.steps or list(range(1, T + 1)))],
            "regions": [list(r) for r in regions],
            "load_missing": missing,
            "step_px_scale_norm": (sum_scale_norm / seen).tolist(),
            "step_px_scale_255": (sum_scale_255 / seen).tolist(),
            "scale_ratio_norm": (sum_scale_norm / max(sum_scale_norm[0], 1e-12)).tolist(),
            "prog_curve_L1_255": (sum_prog / seen).tolist(),
            "E_px": (sum_E / seen).tolist(),
            "E_px_diag": [float(sum_E[t, t] / seen) for t in range(T)],
        }
        # ── 尺度无关的相对判据（Lead 要求 (2)）: /step1 与 /max_t ──
        mn, m255 = sum_scale_norm / seen, sum_scale_255 / seen
        res["scale_ratio_over_max_norm"] = (mn / max(mn.max(), 1e-12)).tolist()
        res["scale_ratio_over_max_255"] = (m255 / max(m255.max(), 1e-12)).tolist()
        # ── 判据 (3) 的汇总量: 对角 vs 非对角、对角离散度 ──
        E = sum_E / seen
        off, ratio = [], []
        for t in range(T):
            others = [E[t, r] for r in range(T) if r != t]
            o = float(np.mean(others)) if others else float("nan")
            off.append(o)
            ratio.append(float(E[t, t] / o) if o > 0 else float("nan"))
        diag = np.asarray(res["E_px_diag"], dtype=np.float64)
        res["E_px_offdiag_mean"] = off
        res["E_px_diag_over_offdiag"] = ratio
        res["E_px_diag_mean"] = float(diag.mean())
        res["E_px_diag_cv"] = float(diag.std() / max(diag.mean(), 1e-12))
        res["E_px_diag_over_offdiag_mean"] = float(np.nanmean(ratio))
        out["results"][os.path.basename(os.path.dirname(ckpt))] = res
        print(f"[probe] {os.path.basename(os.path.dirname(ckpt))}: "
              f"scale_norm={[round(v,4) for v in res['step_px_scale_norm']]}")
        print(f"        scale_ratio(/step1)=[{', '.join(f'{v:.3f}' for v in res['scale_ratio_norm'])}]"
              f"  scale_ratio(/max)=[{', '.join(f'{v:.3f}' for v in res['scale_ratio_over_max_norm'])}]")
        print(f"        prog_L1_255={[round(v,4) for v in res['prog_curve_L1_255']]}")
        print(f"        E_px_diag={[round(v,3) for v in res['E_px_diag']]}"
              f"  diag/offdiag={[round(v,3) for v in res['E_px_diag_over_offdiag']]}"
              f"  diag_cv={res['E_px_diag_cv']:.4f}")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    if args.check_stepwise:
        # 自洽校验: prog_curve 必须与 eval_stepwise 的 L1_255 逐位一致（同口径）。
        # 传目录 ⇒ 按 `stepwise_<arm>.json` 自动逐臂匹配; 传文件 ⇒ 仅比对同名臂。
        import glob
        refs = {}
        if os.path.isdir(args.check_stepwise):
            for p in glob.glob(os.path.join(args.check_stepwise, "stepwise_*.json")):
                arm = os.path.basename(p)[len("stepwise_"):-len(".json")]
                refs[arm] = p
        else:
            arm = os.path.basename(args.check_stepwise)
            arm = arm[len("stepwise_"):-len(".json")] if arm.startswith("stepwise_") else arm
            refs[arm] = args.check_stepwise
        for key, res in out["results"].items():
            if key not in refs:
                print(f"[check] {key}: 无对应 stepwise json, 跳过")
                continue
            ref = json.load(open(refs[key]))
            a = np.asarray(res["prog_curve_L1_255"])
            b = np.asarray(ref["L1_255"])
            d = float(np.abs(a - b).max()) if a.shape == b.shape else float("nan")
            print(f"[check] {key} prog_curve vs stepwise L1_255: max|Δ|={d:.3e} (应≈0: 同口径)")
            out[f"check_{key}"] = {"max_abs_diff_vs_stepwise": d,
                                   "ref": refs[key]}
        worst = max([abs(v["max_abs_diff_vs_stepwise"]) for k, v in out.items()
                     if k.startswith("check_")], default=0.0)
        print(f"[check] 全部臂 worst max|Δ| = {worst:.3e}")
        out["check_worst_max_abs_diff"] = worst

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
