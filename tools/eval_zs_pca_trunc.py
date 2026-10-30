#!/usr/bin/env python3
"""T3: 零训练「PCA 截断 → 解码器」重建对照（AssocVTC 240M 基线, IN-100 val）。

问题
----
降维实验的 zs448/zs224 臂是**训练**出来的编码器末端 FC 瓶颈（`Linear(1248→448/224)`,
warm start = 对 decoder.mem_proj.weight 做 SVD 秩近似，不是数据 PCA）。本脚本做
零训练对照: 在**数据侧**对每个 token 位置单独拟合 PCA，把 z_s 投影到前 r 个主成分
再反投影回 1248 维，直接送**未改动的基线解码器**重建：
    r ∈ {1248（恒等对照）, 448, 224}
z_cls 用其自身（单位置）PCA 前 r 个主成分。

口径（与 eval_linattn.py 同口径）
-------------------------------
* 模型: `SRPhase1LinAttn(LinAttnConfig(**ck['cfg']))`，`strict=False` 加载、eval、
  CPU only、fp32、`torch.no_grad()`。
* 解码: `model.decode(z_cls, z_s, x)`（16 步 fixed block2 循环读出，取末步 F_hat）。
  直接用 decode 而不走 forward，避免重新 encode（省 CPU）。
* 指标: `metrics_linattn.psnr`（[0,1] 空间、预测 clamp 到 [0,1]）/ `ssim`（11×11
  高斯窗、valid 裁边）/ `l1_norm`（**归一化空间** L1，= eval_linattn.py 的 l1_norm）。
* 子集: 与 `eval_stepwise_linattn.py` 的 stride 采样约定一致
  `idx = arange(0, 5000, stride)[:N_sub]`（默认 N_sub=500 ⇒ stride=10，覆盖全部
  100 类、每类 5 张；IN-100 val 按类别分组存放，连续前缀只含前几类，故必须等间距）。
  所有 r 用**同一子集**。全量 5000 的 base 数字只作上下文，不做逐位比较。
* PCA 基在**全部 5000 张**（含子集图像）上拟合 ⇒ transductive/oracle 上界。

产物（out_dir/）
---------------
* `trunc_eval.json`: 配置、子集定义/索引、benchmark 与各 r 的 PSNR/L1_norm/SSIM、
  恒等对照 vs raw 的差值、PCA 方差保留统计、耗时、sha256
* `per_image.csv`: 每图每 variant 的 psnr/l1_norm/ssim
* `pca_bases_top448.npz`: 每位置均值 (32,1248) + 前 448 主成分 (32,448,1248) fp32
* `trunc_eval.log`: 由 nohup 重定向（脚本自身只 print）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

CST = timezone(timedelta(hours=8))
SCRIPT_NAME = "eval_zs_pca_trunc.py"


def log(msg: str) -> None:
    print(f"[{datetime.now(CST).isoformat(timespec='seconds')}] {msg}", flush=True)


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fit_pca_per_position(z: np.ndarray, max_rank: int):
    """z: (N,K,D) float32/64 → per-position PCA（float64 SVD）。

    返回 meta list[dict]（每位置 spectrum 统计）、means (K,D) fp64、
    components (K,max_rank,D) fp32（前 max_rank 个主成分, 行 = 主成分方向）。
    """
    N, K, D = z.shape
    means = np.empty((K, D), dtype=np.float64)
    comps = np.empty((K, max_rank, D), dtype=np.float32)
    meta = []
    for k in range(K):
        X = np.asarray(z[:, k, :], dtype=np.float64)
        mu = X.mean(axis=0)
        means[k] = mu
        Xc = X - mu
        # 经济型 SVD: Xc = U S Vh, Vh 行 = 主成分方向（单位向量）
        _, S, Vh = np.linalg.svd(Xc, full_matrices=False)
        assert Vh.shape == (min(N, D), D)
        comps[k] = Vh[:max_rank].astype(np.float32)
        lam = (S.astype(np.float64) ** 2) / max(N - 1, 1)     # 样本协方差特征值
        tot = float(lam.sum())
        cum = np.cumsum(lam) / max(tot, 1e-30)
        max_rank_eff = int(min(N - 1, D, S.size))
        # 秩阈值（T3 只需 448/224 的保留比例, 顺带记录 64/128/624）
        def frac(r):
            return float(cum[min(r, cum.size) - 1]) if r > 0 else 0.0
        meta.append({
            "pos": k,
            "n": int(N),
            "dim": int(D),
            "rank_eff": max_rank_eff,
            "total_variance": tot,
            "eigenvalue_top10": [float(v) for v in lam[:10]],
            "cumvar_at": {str(r): frac(r) for r in (32, 64, 128, 224, 448, 624, 1248)},
            "top1_frac": float(lam[0] / max(tot, 1e-30)),
            "participation_ratio": float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-30)),
        })
        log(f"  PCA pos {k:2d}: rank_eff={max_rank_eff} "
            f"cum@224={frac(224):.4f} cum@448={frac(448):.4f} "
            f"PR={meta[-1]['participation_ratio']:.1f}")
    return means, comps, meta


def project_r(z: np.ndarray, mean: np.ndarray, comps: np.ndarray, r: int,
              rows: np.ndarray) -> np.ndarray:
    """把 z[rows, k, :] 投影到前 r 主成分再反投影（float64 计算，返回 fp32）。

    z: (N,K,D); mean: (K,D); comps: (K,max_rank,D) fp32（行=主成分方向）。
    """
    N, K, D = z.shape
    out = np.empty((len(rows), K, D), dtype=np.float32)
    rr = min(r, comps.shape[1])
    for k in range(K):
        X = np.asarray(z[rows, k, :], dtype=np.float64)
        V = comps[k, :rr].astype(np.float64)                  # (rr, D)
        Xc = X - mean[k]
        out[:, k, :] = ((Xc @ V.T) @ V + mean[k]).astype(np.float32)
    return out


def run_paired_arms(args, build_datasets, psnr, ssim, IMAGENET_MEAN,
                    IMAGENET_STD, LinAttnConfig, SRPhase1LinAttn,
                    patches_to_image) -> int:
    """配对收尾: 在**同一 200 张等间距子集**上评测三臂（base / zs448 / zs224）。

    口径与 `eval_linattn.py` 一致（末步 F_hat：PSNR 在 [0,1] 空间对预测 clamp、
    SSIM 11×11 高斯窗 valid、l1_norm 在归一化空间），但子集改用与 T3 /
    `eval_stepwise_linattn.py` 相同的 stride 等间距采样（`eval_linattn.py` 内部是
    `range(n)` 前缀采样，会只覆盖前几类，不能用于与 T3 配对）。
    """
    from torch.utils.data import DataLoader, Subset
    import csv as _csv

    arms = {}
    for spec in args.paired_arms.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, path = spec.split("=", 1)
        arms[name.strip()] = path.strip()
    assert "base" in arms, "配对模式必须包含 base 臂（与 T3 raw 对齐做管线交叉验证）"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    t0_all = time.time()
    t0_wall = datetime.now(CST)
    log(f"=== paired arms 模式: {list(arms)} ===")

    _, va_ds = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    N = len(va_ds)
    n = min(int(args.paired_num_images), N)
    stride = max(1, N // n)
    idx = list(range(0, N, stride))[:n]
    assert len(idx) == n, f"stride 采样不足: {len(idx)} != {n}"
    log(f"subset: N_sub={len(idx)} stride={stride} "
        f"first={idx[:5]} last={idx[-3:]} arange(0,{N},{stride})")
    all_paths = [va_ds.items[i][0] for i in idx]
    all_labels = [int(va_ds.items[i][1]) for i in idx]

    # 子集与 T3 trunc_eval.json 交叉核对
    cross = {"trunc_eval_json": None}
    tj_path = os.path.join(out_dir, "trunc_eval.json")
    if os.path.exists(tj_path):
        with open(tj_path) as f:
            tj = json.load(f)
        same = [int(v) for v in tj["subset"]["indices"]] == idx
        cross["trunc_eval_json"] = {
            "path": tj_path,
            "subset_indices_match": bool(same),
            "trunc_raw_psnr_mean": tj["results"]["raw"]["psnr_mean"],
            "trunc_raw_l1_norm_mean": tj["results"]["raw"]["l1_norm_mean"],
            "trunc_raw_ssim_mean": tj["results"]["raw"]["ssim_mean"],
        }
        assert same, "配对子集与 trunc_eval.json 子集不一致!"
        log(f"subset 与 trunc_eval.json 一致 ✓")

    loader = DataLoader(Subset(va_ds, idx), batch_size=args.batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=args.num_workers, pin_memory=False)
    results, per_image, timings, ckpt_meta = {}, {}, {}, {}
    patch, img = args.img_size // 14, args.img_size      # patch 从 cfg 覆盖
    for name, ckpt_path in arms.items():
        t_arm = time.time()
        log(f"--- arm {name}: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfgd = {k: v for k, v in ck["cfg"].items()
                if k in LinAttnConfig.__dataclass_fields__}
        cfg = LinAttnConfig(**cfgd)
        model = SRPhase1LinAttn(cfg)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        assert not unexpected, f"{name} 多余 key: {unexpected[:5]}"
        assert set(missing) <= {"decoder.mem_pos"}, f"{name} 未加载: {missing[:5]}"
        model.eval()
        patch = int(cfg.patch)
        T = len(model.decoder.steps) if model.recurrent else 1
        ckpt_meta[name] = {"path": ckpt_path, "sha256": sha256_file(ckpt_path),
                           "step": ck.get("step"), "epoch": ck.get("epoch"),
                           "zs_dim": int(getattr(model, "zs_dim", 0) or 0),
                           "T": int(T), "missing_keys": list(missing)}
        ps, l1s, sss = [], [], []
        per_image[name] = []
        off = 0
        for x, _ in loader:
            with torch.no_grad():
                out = model(x)
            f_hat = out["F_hat"].float()
            tgt = out["target_pix"].float()
            l1 = (f_hat - tgt).abs().flatten(1).mean(1)       # 归一化空间
            rec01 = patches_to_image(f_hat, img, img, patch, IMAGENET_MEAN,
                                     IMAGENET_STD, clamp_255=False).clamp(0.0, 1.0)
            tgt01 = patches_to_image(tgt, img, img, patch, IMAGENET_MEAN,
                                     IMAGENET_STD, clamp_255=False).clamp(0.0, 1.0)
            assert 0.02 < float(tgt01.mean()) < 0.98, "反归一化口径异常"
            p = psnr(rec01, tgt01)
            s = ssim(rec01, tgt01)
            B = x.shape[0]
            for j in range(B):
                gi = idx[off + j]
                rec = {"variant": name, "index": int(gi), "label": int(all_labels[off + j]),
                       "path": all_paths[off + j], "psnr": float(p[j]),
                       "l1_norm": float(l1[j]), "ssim": float(s[j])}
                per_image[name].append(rec)
                ps.append(rec["psnr"]); l1s.append(rec["l1_norm"]); sss.append(rec["ssim"])
            off += B
            if off % (args.batch_size * 5) == 0 or off == n:
                log(f"    [{name}] {off}/{n} imgs "
                    f"({(time.time()-t_arm)/max(off,1):.3f}s/img)")
        assert off == n
        timings[name] = round(time.time() - t_arm, 1)
        results[name] = {
            "n_images": n,
            "psnr_mean": float(np.mean(ps)),
            "psnr_std": float(np.std(ps)),
            "l1_norm_mean": float(np.mean(l1s)),
            "ssim_mean": float(np.mean(sss)),
            "seconds": timings[name],
            "s_per_img": round(timings[name] / n, 3),
        }
        log(f"arm {name}: PSNR={results[name]['psnr_mean']:.4f} dB "
            f"L1_norm={results[name]['l1_norm_mean']:.5f} "
            f"SSIM={results[name]['ssim_mean']:.4f} ({timings[name]}s)")
        del model

    # ── 交叉核对 base vs T3 raw（per_image.csv）──
    cross["base_vs_trunc_raw"] = {}
    pi_path = os.path.join(out_dir, "per_image.csv")
    if os.path.exists(pi_path):
        raw = {}
        with open(pi_path) as f:
            for row in _csv.DictReader(f):
                if row["variant"] == "raw":
                    raw[int(row["index"])] = float(row["psnr"])
        d = [r["psnr"] - raw[r["index"]] for r in per_image["base"]
             if r["index"] in raw]
        if d:
            cross["base_vs_trunc_raw"] = {
                "n": len(d), "delta_psnr_mean": float(np.mean(d)),
                "delta_psnr_max_abs": float(np.max(np.abs(d))),
                "pass_lt_0.05db": bool(abs(float(np.mean(d))) < 0.05),
                "note": ("base 臂在本脚本用 model(x)（重新 encode）; T3 raw 用 T1 "
                         "token 的 decode。同一模型同一子集, 差异应来自 encode 的 "
                         "batch 形状级浮点差异（可忽略）。"),
            }
            log(f"crosscheck base vs T3 raw: ΔPSNR mean={np.mean(d):+.6f} dB "
                f"max|Δ|={np.max(np.abs(d)):.6f} (pass={abs(float(np.mean(d))) < 0.05})")

    res = {
        "script": SCRIPT_NAME,
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "mode": "paired_arms",
        "created_at": datetime.now(CST).isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "threads": int(torch.get_num_threads()),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "args": sys.argv,
        "arms": list(arms),
        "arm_ckpts": ckpt_meta,
        "subset": {
            "definition": f"N_sub={n}, stride={stride}, arange(0,{N},{stride})",
            "n_images": n, "stride": stride, "indices": idx,
            "note": ("与 T3/eval_stepwise_linattn.py 的 stride 采样一致; 覆盖全部 "
                     "100 类（每类 2 张）。三臂 + T3 的 r448/r224 全部同一子集。"),
        },
        "results": results,
        "crosscheck": cross,
        "timing": {"t_start": t0_wall.isoformat(timespec="seconds"),
                   "t_end": datetime.now(CST).isoformat(timespec="seconds"),
                   "total_s": round(time.time() - t0_all, 1),
                   "per_arm_s": timings},
        "notes": ("口径与 eval_linattn.py 相同: 末步 F_hat, PSNR 在 [0,1] 空间对预测 "
                  "clamp, SSIM 11×11 高斯窗 valid 裁边, l1_norm 在归一化空间。"
                  "zs448/zs224 臂为其训练后 ckpt（3 epoch 全量微调, warm start 对 "
                  "decoder.mem_proj.weight 做 SVD 秩近似, 不是数据 PCA）; T3 的 "
                  "r448/r224 是零训练数据侧 PCA 截断 + 基线解码器。"),
    }
    out = args.paired_out or os.path.join(out_dir, f"paired_arms_{n}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"saved {out}")

    # per-image csv（三臂合并, 便于复核）
    csv_path = os.path.join(out_dir, f"per_image_paired_{n}.csv")
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["arm", "index", "label", "psnr", "l1_norm", "ssim", "path"])
        for name in arms:
            for rec in per_image[name]:
                w.writerow([rec["variant"], rec["index"], rec["label"],
                            f"{rec['psnr']:.6f}", f"{rec['l1_norm']:.8f}",
                            f"{rec['ssim']:.6f}", rec["path"]])
    log(f"saved {csv_path}")
    log("=== 配对表（同一 %d 张）===" % n)
    for name in arms:
        r = results[name]
        log(f"  {name:8s} PSNR={r['psnr_mean']:.4f} dB  "
            f"L1_norm={r['l1_norm_mean']:.5f}  SSIM={r['ssim_mean']:.4f}  "
            f"({r['seconds']}s)")
    if cross.get("base_vs_trunc_raw"):
        log(f"  交叉验证 base(model(x)) vs T3 raw(token decode): "
            f"ΔPSNR={cross['base_vs_trunc_raw']['delta_psnr_mean']:+.6f} dB")
    log(f"ALL DONE in {time.time()-t0_all:.1f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tokens", default="", help="T1 的 fp32 token npz")
    ap.add_argument("--paired_arms", default="",
                    help="配对收尾模式: 'base=<ckpt>,zs448=<ckpt>,zs224=<ckpt>'; "
                         "给出时只跑「三臂同子集」评测, 忽略 --ckpt/--tokens")
    ap.add_argument("--paired_num_images", type=int, default=200,
                    help="配对评测子集大小（默认 200 ⇒ stride=25, 与 T3 一致）")
    ap.add_argument("--paired_out", default="",
                    help="配对结果 json（默认 out_dir/paired_arms_200.json）")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_images", type=int, default=500,
                    help="评测子集大小（默认 500 ⇒ stride=10, 覆盖全部 100 类）")
    ap.add_argument("--auto_reduce_to", type=int, default=200,
                    help="benchmark 超过 --max_s_per_img 时降级到的子集大小")
    ap.add_argument("--max_s_per_img", type=float, default=0.5)
    ap.add_argument("--ranks", default="1248,448,224")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--benchmark_n", type=int, default=8)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--repo", default="/root/AssocVTC")
    ap.add_argument("--save_bases", action="store_true", default=True)
    args = ap.parse_args()

    t_start = time.time()
    t_start_wall = datetime.now(CST)
    os.makedirs(args.out_dir, exist_ok=True)
    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]
    assert 1248 in ranks, "必须包含 r=1248 恒等对照"

    try:
        torch.set_num_interop_threads(1)
    except Exception as e:  # noqa: BLE001
        log(f"warning: set_num_interop_threads: {e}")
    torch.set_num_threads(int(args.threads))
    log(f"torch={torch.__version__} numpy={np.__version__} "
        f"threads={torch.get_num_threads()}")

    sys.path.insert(0, args.repo)
    from data_linattn import build_datasets                       # noqa: E402
    from metrics_linattn import psnr, ssim                        # noqa: E402
    from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD,    # noqa: E402
                                  LinAttnConfig, SRPhase1LinAttn,
                                  patches_to_image)

    if args.paired_arms:
        return run_paired_arms(args, build_datasets, psnr, ssim,
                               IMAGENET_MEAN, IMAGENET_STD, LinAttnConfig,
                               SRPhase1LinAttn, patches_to_image)

    assert args.ckpt and args.tokens, "常规模式需要 --ckpt 与 --tokens"

    # ── token ──
    log(f"loading tokens {args.tokens}")
    tok = np.load(args.tokens)
    z_s = np.asarray(tok["z_s"], dtype=np.float32)
    z_cls = np.asarray(tok["z_cls"], dtype=np.float32)
    labels = np.asarray(tok["labels"])
    paths = [str(p) for p in tok["paths"]]
    N, K, D = z_s.shape
    Dc = z_cls.shape[-1]
    assert K == 32 and D == 1248 and Dc == 1248, \
        f"token shape 异常: z_s={z_s.shape}, z_cls={z_cls.shape}"
    tok_sha = sha256_file(args.tokens)
    log(f"tokens: z_s={z_s.shape} z_cls={z_cls.shape} N={N} sha256={tok_sha[:16]}...")

    # ── 子集（等间距; 与 eval_stepwise_linattn.py 的 stride 采样一致）──
    n_sub = min(int(args.num_images), N)
    stride = max(1, N // n_sub)
    idx = list(range(0, N, stride))[:n_sub]
    log(f"subset: N_sub={len(idx)} stride={stride} first={idx[:5]} last={idx[-3:]}")

    # ── 模型 ──
    log(f"loading ckpt {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfgd = {k: v for k, v in ck["cfg"].items() if k in LinAttnConfig.__dataclass_fields__}
    cfg = LinAttnConfig(**cfgd)
    model = SRPhase1LinAttn(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, f"state_dict 不匹配 {missing[:5]} {unexpected[:5]}"
    model.eval()
    log(f"model: T={(len(model.decoder.steps) if model.recurrent else 1)}, "
        f"K={cfg.num_specials}, N={cfg.num_patches}, zs_dim={model.zs_dim}")

    # ── 数据（只用于取 x 与 target）──
    _, va_ds = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    assert len(va_ds) == N, f"val 数量 {len(va_ds)} != token N={N}"
    assert [va_ds.items[i][0] for i in idx] == [paths[i] for i in idx], \
        "token paths 与 val.items 顺序不一致"
    loader = DataLoader(Subset(va_ds, idx), batch_size=args.batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=args.num_workers, pin_memory=False)
    xb_all = []                      # 子集图像，按子集顺序（一次性载入, 500×3×224×224 ≈ 300MB）
    for x, y in loader:
        xb_all.append(x)
        assert np.array_equal(y.numpy(), labels[idx][sum(t.shape[0] for t in xb_all) - x.shape[0]:
                                                    sum(t.shape[0] for t in xb_all)]), \
            "标签顺序异常"
    xb_all = torch.cat(xb_all, 0)
    log(f"subset images loaded: {tuple(xb_all.shape)}")

    # ── per-position PCA（全部 5000 张拟合）──
    log("fitting per-position PCA on ALL tokens (float64 SVD) ...")
    t_pca = time.time()
    max_rank = max(ranks)
    means_s, comps_s, meta_s = fit_pca_per_position(z_s, max_rank)
    means_c, comps_c, meta_c = fit_pca_per_position(z_cls, max_rank)
    pca_fit_s = time.time() - t_pca
    log(f"PCA fit done in {pca_fit_s:.1f}s")

    # ── 子集 token variants（raw + 每个 r）──
    rows = np.asarray(idx, dtype=np.int64)
    variants = {"raw": (z_cls[rows].copy(), z_s[rows].copy())}
    for r in ranks:
        variants[f"r{r}"] = (project_r(z_cls, means_c, comps_c, r, rows),
                             project_r(z_s, means_s, comps_s, r, rows))
        log(f"variant r={r}: z_cls{tuple(variants[f'r{r}'][0].shape)} "
            f"z_s{tuple(variants[f'r{r}'][1].shape)}")
    # ── 解码度量 ──
    const01 = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    patch = cfg.patch
    img = cfg.img_size
    per_image = {name: [] for name in variants}     # name -> list[dict]
    var_times = {name: 0.0 for name in variants}

    def eval_variant(name: str, n_limit: int | None = None) -> dict:
        t_v = time.time()
        zc_all, zs_all = variants[name]
        n_eval = len(idx) if n_limit is None else min(n_limit, len(idx))
        ml = n_eval * K * D
        ps_list, l1_list, ss_list = [], [], []
        for i0 in range(0, n_eval, args.batch_size):
            i1 = min(i0 + args.batch_size, n_eval)
            x = xb_all[i0:i1]
            zc = torch.from_numpy(zc_all[i0:i1])
            zs = torch.from_numpy(zs_all[i0:i1])
            with torch.no_grad():
                out = model.decode(zc, zs, x)
            f_hat = out["F_hat"].float()
            tgt = out["target_pix"].float()
            l1 = (f_hat - tgt).abs().flatten(1).mean(1)           # 归一化空间
            rec01 = patches_to_image(f_hat, img, img, patch, IMAGENET_MEAN,
                                     IMAGENET_STD, clamp_255=False).clamp(0.0, 1.0)
            tgt01 = patches_to_image(tgt, img, img, patch, IMAGENET_MEAN,
                                     IMAGENET_STD, clamp_255=False).clamp(0.0, 1.0)
            assert 0.02 < float(tgt01.mean()) < 0.98, "反归一化口径异常"
            p = psnr(rec01, tgt01)
            s = ssim(rec01, tgt01)
            for j in range(i1 - i0):
                gi = idx[i0 + j]
                rec = {"variant": name, "index": int(gi), "label": int(labels[gi]),
                       "path": paths[gi], "psnr": float(p[j]),
                       "l1_norm": float(l1[j]), "ssim": float(s[j])}
                per_image[name].append(rec)
                ps_list.append(rec["psnr"]); l1_list.append(rec["l1_norm"])
                ss_list.append(rec["ssim"])
            if (i0 // args.batch_size) % 10 == 0:
                log(f"    [{name}] {i1}/{n_eval} imgs "
                    f"({(time.time()-t_v)/max(i1,1):.3f}s/img)")
        var_times[name] = time.time() - t_v
        return {"n_images": n_eval,
                "psnr_mean": float(np.mean(ps_list)),
                "l1_norm_mean": float(np.mean(l1_list)),
                "ssim_mean": float(np.mean(ss_list)),
                "psnr_std": float(np.std(ps_list)),
                "seconds": round(var_times[name], 1),
                "s_per_img": round(var_times[name] / max(n_eval, 1), 3)}

    # ── benchmark 8 张（raw）──
    n_bench = min(int(args.benchmark_n), len(idx))
    log(f"benchmark: decode {n_bench} imgs (raw variant) ...")
    bench = eval_variant("raw", n_limit=n_bench)
    s_per_img = bench["s_per_img"]
    per_image["raw"] = []       # benchmark 的 8 张不重复计入全量 per_image.csv
    var_times["raw"] = 0.0
    log(f"benchmark: {s_per_img:.3f} s/img (batch={args.batch_size}, "
        f"threads={args.threads}) → 全子集 {len(idx)} 张约 "
        f"{s_per_img*len(idx)*len(variants)/60:.1f} min")
    subset_note = f"N_sub={len(idx)}, stride={stride}, arange(0,{N},{stride})"
    reduced = False
    if s_per_img > args.max_s_per_img and len(idx) > args.auto_reduce_to:
        n_sub = int(args.auto_reduce_to)
        stride = max(1, N // n_sub)
        idx = list(range(0, N, stride))[:n_sub]
        rows = np.asarray(idx, dtype=np.int64)
        log(f"⚠️ {s_per_img:.3f}s/img > {args.max_s_per_img}: 子集降级到 "
            f"N_sub={len(idx)}, stride={stride}")
        subset_note = (f"降级后 N_sub={len(idx)}, stride={stride}, "
                       f"arange(0,{N},{stride})（benchmark {s_per_img:.3f}s/img 超过 "
                       f"{args.max_s_per_img}s/img）")
        reduced = True
        # 用新子集重建 variants / x
        xb_all = torch.stack([torch.as_tensor(va_ds[i][0]) for i in idx], 0)
        variants = {"raw": (z_cls[rows].copy(), z_s[rows].copy())}
        for r in ranks:
            variants[f"r{r}"] = (project_r(z_cls, means_c, comps_c, r, rows),
                                 project_r(z_s, means_s, comps_s, r, rows))
        per_image = {name: [] for name in variants}
        var_times = {name: 0.0 for name in variants}

    # ── 恒等对照（r=1248）应≈raw: 量化 token 空间差异（在最终子集上算）──
    ident = variants["r1248"]
    tok_check = {
        "z_s_max_abs_diff": float(np.max(np.abs(ident[1] - variants["raw"][1]))),
        "z_cls_max_abs_diff": float(np.max(np.abs(ident[0] - variants["raw"][0]))),
        "z_s_rel_rmse": float(np.sqrt(np.mean((ident[1] - variants["raw"][1]) ** 2)) /
                              (np.sqrt(np.mean(variants["raw"][1] ** 2)) + 1e-30)),
    }
    log(f"identity-vs-raw token diff: {tok_check}")

    # ── 全量评测所有 variant（同一子集）──
    log(f"full eval: {len(idx)} imgs × {len(variants)} variants ...")
    t_dec = time.time()
    results = {}
    for name in variants:
        log(f"=== variant {name} ===")
        results[name] = eval_variant(name)
        log(f"variant {name}: PSNR={results[name]['psnr_mean']:.4f} dB "
            f"L1_norm={results[name]['l1_norm_mean']:.5f} "
            f"SSIM={results[name]['ssim_mean']:.4f} "
            f"({results[name]['seconds']}s)")
    dec_elapsed = time.time() - t_dec

    # ── 对照检查 ──
    base = results["raw"]
    checks = {"identity_vs_raw": {}, "r448_vs_base_delta_db": None,
              "r224_vs_base_delta_db": None}
    for name in variants:
        if name == "raw":
            continue
        d = [pi["psnr"] - pr["psnr"] for pi, pr in
             zip(per_image[name], per_image["raw"])]
        checks["identity_vs_raw" if name == "r1248" else f"{name}_vs_raw"] = {
            "delta_psnr_mean": float(np.mean(d)),
            "delta_psnr_max_abs": float(np.max(np.abs(d))),
            "delta_l1_norm_mean": float(np.mean(
                [a["l1_norm"] - b["l1_norm"] for a, b in
                 zip(per_image[name], per_image["raw"])])),
        }
    ident_ok = abs(checks["identity_vs_raw"]["delta_psnr_mean"]) < 0.05

    # PCA 方差保留统计
    def var_stats(meta):
        out = {}
        for r in ranks:
            key = str(r)
            vals = [m["cumvar_at"].get(key) for m in meta]
            vals = [v for v in vals if v is not None]
            out[f"cumvar_at_{r}"] = {
                "median": float(np.median(vals)) if vals else None,
                "min": float(np.min(vals)) if vals else None,
                "max": float(np.max(vals)) if vals else None,
                "per_pos": [m["cumvar_at"].get(key) for m in meta],
            }
        out["participation_ratio_median"] = float(np.median(
            [m["participation_ratio"] for m in meta]))
        return out

    # ── 落盘 ──
    if args.save_bases:
        bases_path = os.path.join(args.out_dir, "pca_bases_top448.npz")
        if reduced:
            # 降级时仍存全部 5000 拟合结果（不受子集影响）
            pass
        np.savez(bases_path,
                 zs_mean=means_s.astype(np.float32),
                 zs_components=comps_s[:, :min(448, comps_s.shape[1])].astype(np.float32),
                 zcls_mean=means_c.astype(np.float32),
                 zcls_components=comps_c[:, :min(448, comps_c.shape[1])].astype(np.float32),
                 fit_indices=np.arange(N))
        log(f"saved {bases_path} ({os.path.getsize(bases_path)/1e6:.1f} MB)")

    # per_image.csv
    csv_path = os.path.join(args.out_dir, "per_image.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "index", "label", "psnr", "l1_norm", "ssim", "path"])
        for name in variants:
            for rec in per_image[name]:
                w.writerow([rec["variant"], rec["index"], rec["label"],
                            f"{rec['psnr']:.6f}", f"{rec['l1_norm']:.8f}",
                            f"{rec['ssim']:.6f}", rec["path"]])
    log(f"saved {csv_path}")

    elapsed = time.time() - t_start
    res = {
        "script": SCRIPT_NAME,
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "created_at": datetime.now(CST).isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "device": "cpu",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": int(torch.get_num_threads()),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "ckpt": {"path": args.ckpt, "sha256": sha256_file(args.ckpt),
                 "step": ck.get("step"), "epoch": ck.get("epoch"),
                 "zs_dim": int(model.zs_dim),
                 "T": int(len(model.decoder.steps) if model.recurrent else 1),
                 "dec_steps": list(model.decoder.steps) if model.recurrent else None,
                 "dec_step_plan": cfg.dec_step_plan, "dec_block": cfg.dec_block,
                 "K": int(cfg.num_specials), "N_patches": int(cfg.num_patches),
                 "patch": int(cfg.patch)},
        "tokens": {"path": args.tokens, "sha256": tok_sha,
                   "z_s_shape": list(z_s.shape), "z_cls_shape": list(z_cls.shape),
                   "num_images_fit_pca": int(N)},
        "subset": {
            "definition": subset_note,
            "n_images": len(idx),
            "stride": int(stride),
            "indices": [int(i) for i in idx],
            "reduced_from_benchmark": bool(reduced),
            "benchmark": bench,
            "max_s_per_img_threshold": float(args.max_s_per_img),
            "note": ("等间距抽样 idx=arange(0,5000,stride)[:N]（与 "
                     "eval_stepwise_linattn.py 的 stride 采样一致）；IN-100 val "
                     "按类别分组, 连续前缀只含前几类, 等间距才覆盖全部 100 类。"
                     "所有 variant 用同一子集。"),
        },
        "ranks": ranks,
        "pca": {
            "fit_on": "all 5000 val tokens（含评测子集图像 ⇒ transductive/oracle 上界）",
            "method": "per-position float64 SVD（经济型）, 中心化, 主成分=Vh 行",
            "zs": var_stats(meta_s),
            "z_cls": var_stats(meta_c),
            "per_pos": {"zs": meta_s, "z_cls": meta_c},
        },
        "results": results,
        "checks": {
            **checks,
            "identity_vs_raw_pass_lt_0.05db": bool(ident_ok),
            "identity_token_roundtrip_vs_raw": tok_check,
        },
        "context_reference_full_5000": {   # 既有基线数字, 不重跑
            "base_16step_laststep_psnr_db": 27.936,
            "zs448_arm_psnr_db": 27.688,
            "zs224_arm_psnr_db": 26.08,
            "note": ("既有数字为全量 5000 张、GPU 评测; 本实验为 CPU fp32 + 子集 + "
                     "零训练, 只能相对比较。zs448/zs224 臂的瓶颈是**训练出来的** "
                     "编码器末端 FC, warm start = 对 decoder.mem_proj.weight 做 SVD "
                     "秩近似（不是数据 PCA）; 本实验是数据侧 PCA 上界。"),
        },
        "timing": {
            "t_start": t_start_wall.isoformat(timespec="seconds"),
            "t_end": datetime.now(CST).isoformat(timespec="seconds"),
            "total_s": round(elapsed, 1),
            "pca_fit_s": round(pca_fit_s, 1),
            "decode_all_variants_s": round(dec_elapsed, 1),
            "per_variant_s": {k: round(v, 1) for k, v in var_times.items()},
        },
        "artifacts": {},
    }
    json_path = os.path.join(args.out_dir, "trunc_eval.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"saved {json_path}")
    log(f"ALL DONE in {elapsed:.1f}s")
    log(f"对照: raw(base子集) PSNR={base['psnr_mean']:.4f} | " +
        " | ".join(f"{n}={results[n]['psnr_mean']:.4f}" for n in variants if n != "raw"))
    log(f"恒等对照 ΔPSNR(mean)={checks['identity_vs_raw']['delta_psnr_mean']:+.6f} dB "
        f"(pass={ident_ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
