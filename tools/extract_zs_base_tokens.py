#!/usr/bin/env python3
"""提取 AssocVTC 240M **基线（降维训练之前, zs_dim=0）** 的 encoder 输出 token。

在 IN-100 val 上只用 `model.encode()`（不跑解码器）前向，落盘 z_s / z_cls 供后续
按位置 PCA 与「PCA 截断→解码器」零训练对照实验复用。

口径
----
* 数据: `data_linattn.build_datasets(data_dir, img_size=224, val_per_class=50)` 的
  **val** 子集（len=5000）；顺序 = `va.items` 顺序（scan 时类别目录名/文件名排序，
  见 data_linattn.scan_imagenet100 + split_index），`shuffle=False`。
* 图像变换: val transform = Resize(256, BICUBIC) → CenterCrop(224) → ToTensor →
  Normalize(ImageNet mean/std)，与既有评测逐位同口径。
* 模型: `SRPhase1LinAttn(LinAttnConfig(**ck['cfg']))`，`load_state_dict(strict=False)`
  且要求 missing/unexpected 均为空；`eval()` + `torch.no_grad()`，CPU only。
* 只调 `model.encode(x)`：`z_cls:(B,1,1248)`、`z_s:(B,32,1248)` fp32。

产物（out_dir/）
--------------
* `zs_base_m240_t16_fp32.npz`: z_s (N,32,1248) float32、z_cls (N,1,1248) float32、
  labels int16、paths 字符串数组
* `zs_base_m240_t16_fp16.npz`: 同内容 fp16（供本地 8GB 机器分析）
* `meta.json`: ckpt/脚本 sha256、ck['step']/['epoch']、cfg 关键项、data_dir、
  val_per_class、image order 说明、transform、torch 版本、线程数、逐 batch 耗时、
  起止时间、样本数、checks（NaN/Inf、fp16-vs-fp32、重跑一致性）
* `stats.json`: 按 token 位置的范数 mean/std 等（32 个位置，z_cls 单位置）

用法（远程 AutoDL）
------------------
    nohup /root/miniconda3/bin/python \
        /root/autodl-tmp/assocvtc_zpca/extract_zs_base_tokens.py \
        --ckpt /root/autodl-tmp/srdiff_scale/stdattn_m240_t16/ckpt_best.pt \
        --data_dir /root/autodl-tmp/datasets/imagenet100 \
        --out_dir /root/autodl-tmp/assocvtc_zpca/tokens \
        --threads 64 > /root/autodl-tmp/assocvtc_zpca/extract.log 2>&1 &

冒烟: 加 `--num_images 16 --out_dir .../smoke`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

CST = timezone(timedelta(hours=8))
SCRIPT_NAME = "extract_zs_base_tokens.py"


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


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_stats(x: np.ndarray) -> dict:
    """per-position 范数统计 (N,K,D) → 列表。"""
    norm = np.linalg.norm(x.reshape(x.shape[0], x.shape[1], -1), axis=2)  # (N,K)
    mean_vec = x.reshape(x.shape[0], x.shape[1], -1).mean(axis=0)          # (K,D_flat)
    std_vec = x.reshape(x.shape[0], x.shape[1], -1).std(axis=0)            # (K,D_flat)
    return {
        "pos_norm_mean": [float(v) for v in norm.mean(axis=0)],
        "pos_norm_std": [float(v) for v in norm.std(axis=0)],
        "pos_mean_vector_norm": [float(v) for v in np.linalg.norm(mean_vec, axis=1)],
        "pos_coord_std_mean": [float(v) for v in std_vec.mean(axis=1)],
        "global_norm_mean": float(norm.mean()),
        "global_norm_std": float(norm.std()),
        "global_coord_mean_abs": float(np.abs(x).mean()),
        "global_coord_std": float(x.std()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_images", type=int, default=0,
                    help="0=全部 val（5000）; 冒烟用 16")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--val_per_class", type=int, default=50)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--tag", default="zs_base_m240_t16",
                    help="产物文件名前缀（默认 zs_base_m240_t16）")
    ap.add_argument("--repo", default="/root/AssocVTC",
                    help="AssocVTC 代码目录（只读 import）")
    args = ap.parse_args()

    t_start_wall = datetime.now(CST)
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 线程（最先设置，避免 torch 初始化后再设 interop 报错）──
    try:
        torch.set_num_interop_threads(1)
    except Exception as e:  # noqa: BLE001 - 已初始化则忽略
        log(f"warning: set_num_interop_threads failed: {e}")
    torch.set_num_threads(int(args.threads))
    log(f"torch={torch.__version__} numpy={np.__version__} threads={torch.get_num_threads()} "
        f"interop={torch.get_num_interop_threads()}")

    sys.path.insert(0, args.repo)
    from data_linattn import build_datasets                # noqa: E402
    from model_v2_linattn import SRPhase1LinAttn, LinAttnConfig  # noqa: E402

    # ── ckpt hash + 加载 ──
    log(f"hashing ckpt {args.ckpt} ...")
    t_ck = time.time()
    ckpt_sha = sha256_file(args.ckpt)
    log(f"ckpt sha256={ckpt_sha[:16]}... ({time.time()-t_ck:.1f}s)")

    log("loading ckpt ...")
    t_ck = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfgd = {k: v for k, v in ck["cfg"].items() if k in LinAttnConfig.__dataclass_fields__}
    cfg = LinAttnConfig(**cfgd)
    model = SRPhase1LinAttn(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, \
        f"state_dict 不匹配: missing={missing[:8]} unexpected={unexpected[:8]}"
    model.eval()
    log(f"model loaded in {time.time()-t_ck:.1f}s | step={ck.get('step')} "
        f"epoch={ck.get('epoch')} | zs_dim={getattr(model,'zs_dim',None)} "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── 数据 ──
    _, va = build_datasets(args.data_dir, args.img_size, args.val_per_class)
    n_total = len(va)
    n = n_total if args.num_images in (0, None) else min(int(args.num_images), n_total)
    log(f"val len={n_total} | 取前 {n} 张 (shuffle=False, 顺序=va.items)")

    subset = Subset(va, list(range(n)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False,
                        drop_last=False)
    paths = [va.items[i][0] for i in range(n)]
    labels = np.asarray([int(va.items[i][1]) for i in range(n)], dtype=np.int16)

    zs_buf, zc_buf = None, None
    batch_times, data_times = [], []
    it = iter(loader)
    n_batches = (n + args.batch_size - 1) // args.batch_size
    t_loop = time.time()
    for bi in range(n_batches):
        t0 = time.time()
        x, y = next(it)
        t1 = time.time()
        with torch.no_grad():
            z_cls, z_s = model.encode(x)
        t2 = time.time()
        zc = z_cls.numpy().astype(np.float32, copy=False)
        zs = z_s.numpy().astype(np.float32, copy=False)
        if zs_buf is None:
            zs_buf = np.empty((n,) + zs.shape[1:], dtype=np.float32)
            zc_buf = np.empty((n,) + zc.shape[1:], dtype=np.float32)
            log(f"token shape per batch: z_s={tuple(zs.shape)} z_cls={tuple(zc.shape)}")
        i0 = bi * args.batch_size
        i1 = min(i0 + zs.shape[0], n)
        zs_buf[i0:i1] = zs[:i1 - i0]
        zc_buf[i0:i1] = zc[:i1 - i0]
        assert np.array_equal(y.numpy().astype(np.int16), labels[i0:i1]), \
            f"label 顺序异常 batch={bi}"
        if np.isnan(zs).any() or np.isinf(zs).any():
            raise FloatingPointError(f"z_s 出现 NaN/Inf, batch={bi}")
        data_times.append(t1 - t0)
        batch_times.append(t2 - t1)
        done = i1
        el = time.time() - t_loop
        speed = done / max(el, 1e-9)
        eta = (n - done) / max(speed, 1e-9)
        log(f"batch {bi+1}/{n_batches} imgs={i1}/{n} "
            f"data={t1-t0:.3f}s fwd={t2-t1:.3f}s ({batch_times[-1]/zs.shape[0]:.4f}s/img) "
            f"avg={np.mean(batch_times):.3f}s eta={eta/60:.1f}min")

    elapsed = time.time() - t_loop
    log(f"forward done: {n} imgs in {elapsed:.1f}s = {elapsed/n*1000:.1f} ms/img "
        f"(batch mean {np.mean(batch_times):.3f}s, total incl. load {time.time()-t_start:.1f}s)")

    # ── checks ──
    checks = {}
    checks["nan_inf_fp32"] = {"z_s": bool(not np.isfinite(zs_buf).all()),
                              "z_cls": bool(not np.isfinite(zc_buf).all())}
    assert not checks["nan_inf_fp32"]["z_s"], "z_s 含 NaN/Inf"
    assert not checks["nan_inf_fp32"]["z_cls"], "z_cls 含 NaN/Inf"

    zs16 = zs_buf.astype(np.float16)
    zc16 = zc_buf.astype(np.float16)
    checks["fp16_vs_fp32"] = {
        "z_s_max_abs_diff": float(np.max(np.abs(zs16.astype(np.float32) - zs_buf))),
        "z_cls_max_abs_diff": float(np.max(np.abs(zc16.astype(np.float32) - zc_buf))),
        "z_s_scale_rms": float(np.sqrt(np.mean(zs_buf.astype(np.float64) ** 2))),
    }

    # 重跑一致性: (a) 完全相同的 batch（含未满 batch）→ 应逐位相同;
    #                 (b) 3 张单独组 batch → 记录 max|diff|（batch 形状不同可能非逐位）
    repro = {}
    cand = [0, min(1, n - 1), n // 2, n - 1, min(args.batch_size - 1, n - 1)]
    cand = sorted(set(i for i in cand if 0 <= i < n))
    idx = cand[:3]
    x3 = torch.stack([va[i][0] for i in idx])
    with torch.no_grad():
        zc3, zs3 = model.encode(x3)
    for name, arr, ref in (("z_s", zs3.numpy(), zs_buf[idx]),
                           ("z_cls", zc3.numpy(), zc_buf[idx])):
        d = np.abs(arr.astype(np.float64) - ref.astype(np.float64))
        repro[f"single_batch_{name}"] = {
            "indices": idx,
            "bitwise_equal": bool(np.array_equal(arr, ref)),
            "max_abs_diff": float(d.max()),
        }
    # (a) 完全相同的 batch 内容重跑
    xb = torch.stack([va[i][0] for i in range(min(args.batch_size, n))])
    with torch.no_grad():
        zcb, zsb = model.encode(xb)
    repro["same_batch_replay"] = {
        "n": int(min(args.batch_size, n)),
        "z_s_bitwise_equal": bool(np.array_equal(zsb.numpy(), zs_buf[:xb.shape[0]])),
        "z_cls_bitwise_equal": bool(np.array_equal(zcb.numpy(), zc_buf[:xb.shape[0]])),
        "z_s_max_abs_diff": float(np.abs(zsb.numpy().astype(np.float64) -
                                         zs_buf[:xb.shape[0]].astype(np.float64)).max()),
    }
    checks["repro"] = repro
    for k, v in repro.items():
        log(f"check repro {k}: {v}")

    # ── 落盘 ──
    fp32_path = os.path.join(args.out_dir, f"{args.tag}_fp32.npz")
    fp16_path = os.path.join(args.out_dir, f"{args.tag}_fp16.npz")
    np.savez(fp32_path, z_s=zs_buf, z_cls=zc_buf, labels=labels,
             paths=np.asarray(paths))
    log(f"saved {fp32_path} ({os.path.getsize(fp32_path)/1e6:.1f} MB)")
    np.savez(fp16_path, z_s=zs16, z_cls=zc16, labels=labels,
             paths=np.asarray(paths))
    log(f"saved {fp16_path} ({os.path.getsize(fp16_path)/1e6:.1f} MB)")

    def _sha(p):
        return sha256_file(p)

    sha_fp32, sha_fp16 = _sha(fp32_path), _sha(fp16_path)
    log(f"sha256 fp32={sha_fp32}")
    log(f"sha256 fp16={sha_fp16}")

    # ── stats.json ──
    stats = {
        "num_images": int(n),
        "zs_shape": list(zs_buf.shape),
        "zcls_shape": list(zc_buf.shape),
        "z_s": tensor_stats(zs_buf),
        "z_cls": tensor_stats(zc_buf),
        "label_hist_first10": {int(c): int((labels == c).sum())
                               for c in np.unique(labels)[:10]},
        "note": ("pos_norm_*: 每位置 token 的 L2 范数在 5000 张上的 mean/std; "
                 "pos_mean_vector_norm: 位置均值向量范数; "
                 "pos_coord_std_mean: 位置内坐标维度 std 的均值"),
    }
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    log(f"saved {stats_path}")

    # ── meta.json ──
    transform_repr = repr(va.transform)
    meta = {
        "script": SCRIPT_NAME,
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "script_argv": sys.argv,
        "created_at": datetime.now(CST).isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
        "num_workers": int(args.num_workers),
        "batch_size": int(args.batch_size),
        "ckpt": {
            "path": args.ckpt,
            "sha256": ckpt_sha,
            "step": ck.get("step"),
            "epoch": ck.get("epoch"),
            "zs_dim": int(getattr(model, "zs_dim", -1)),
            "params_total_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
            "cfg_key_fields": {k: cfgd[k] for k in (
                "img_size", "patch", "num_specials", "enc_dim", "enc_depth",
                "enc_heads", "enc_attn", "dec_dim", "dec_depth", "dec_heads",
                "dec_recurrent", "dec_steps", "dec_step_plan", "dec_block",
                "dec_attn", "zs_dim") if k in cfgd},
        },
        "data": {
            "data_dir": args.data_dir,
            "val_per_class": int(args.val_per_class),
            "num_images": int(n),
            "val_total": int(n_total),
            "order": ("va.items 顺序（data_linattn.build_datasets: "
                      "scan_imagenet100 按类别目录名排序、目录内文件名排序, "
                      "split_index 取每类最后 val_per_class 张为 val）; "
                      "DataLoader shuffle=False; paths 一并落盘"),
            "transform": transform_repr,
            "subset_note": ("full" if n == n_total else f"前 {n} 张（子集）"),
            "first_path": paths[0] if paths else None,
            "last_path": paths[-1] if paths else None,
            "paths_sha256": sha256_bytes("\n".join(paths).encode()),
        },
        "timing": {
            "t_start": t_start_wall.isoformat(timespec="seconds"),
            "t_end": datetime.now(CST).isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - t_start, 1),
            "forward_loop_s": round(elapsed, 1),
            "ms_per_image_fwd": round(elapsed / n * 1000, 2),
            "batch_seconds": [round(t, 4) for t in batch_times],
            "batch_data_seconds": [round(t, 4) for t in data_times],
            "batch_seconds_mean": round(float(np.mean(batch_times)), 4),
            "batch_seconds_median": round(float(np.median(batch_times)), 4),
        },
        "artifacts": {
            "fp32": {"path": fp32_path, "sha256": sha_fp32,
                     "size_bytes": os.path.getsize(fp32_path)},
            "fp16": {"path": fp16_path, "sha256": sha_fp16,
                     "size_bytes": os.path.getsize(fp16_path)},
        },
        "shapes": {"z_s": list(zs_buf.shape), "z_cls": list(zc_buf.shape),
                   "labels": list(labels.shape), "paths": len(paths)},
        "checks": checks,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    log(f"saved {meta_path}")
    log(f"ALL DONE in {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
