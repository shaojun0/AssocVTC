#!/usr/bin/env python3
"""qwen_reextract_b2.py — 重新提取 IN-100 train token，改用 **B2 量化**（位置中心化 + 逐通道归一 +
逐 token int8），把 per-token-absmax int8 的 14% 相对误差降到 ~0.5%（token 存储精度 ~46 dB）。

背景：首轮 per-token absmax int8 实测 quant rel_rmse=0.1418（token PSNR 16.97 dB）——因为 Qwen 视觉
token 有巨大共享偏置/离群维（per-token absmax 中位 68，但逐元素 RMS 仅 1.28），per-token 尺度把
内容精度吃掉了。实测（val 4,000 图）：
  A per-token absmax            : rel_rmse 0.1419
  C 位置中心化+逐通道 int8       : rel_rmse 0.0101
  B2 位置中心化+逐通道归一+逐token int8: rel_rmse **0.0049**（采用）

量化参考（mu_p (49,5120)、sd (5120,)）由 **val 5,000 张的 fp16 token** 估计（只影响存储刻度，
不进入 PCA/训练数据；报告里注明）。
存储契约（在旧契约上增加一个 2MB 的 quant_ref.npz，其余文件名/形状不变）:
  tokens/train_tokens_int8.npy (N,49,5120) int8   # q = round((x-mu_p)/sd/sc)
  tokens/train_scale.npy       (N,49) fp16        # sc = 逐 token absmax of y=(x-mu_p)/sd /127
  tokens/quant_ref.npz         mu_p (49,5120) f32, sd (5120,) f32
  反量化: x = (q.float() * sc[...,None]) * sd[None,None,:] + mu_p[None,:,:]
用法:
  QWEN_ZIP=/root/autodl-pub/... python code/re_extract_b2.py --bs 256 --workers 20 --prefetch 8
"""
import argparse, io, json, os, sys, time, zipfile
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/qwen_vtc/code")
import qwen_extract_vision_tokens as Q  # noqa: E402

R = "/root/autodl-tmp/qwen_vtc"
N_TOK, DIM = 49, 5120


def build_refs(tokens_dir, chunk=1000):
    V = np.load(os.path.join(tokens_dir, "val_tokens_fp16.npy"), mmap_mode="r")
    n = len(V)
    s = np.zeros((N_TOK, DIM), np.float64)
    t0 = time.time()
    for lo in range(0, n, chunk):
        x = np.asarray(V[lo:lo + chunk], dtype=np.float32)
        s += x.sum(0, dtype=np.float64)
    mu = (s / n).astype(np.float32)
    amax = np.zeros(DIM, np.float32)
    for lo in range(0, n, chunk):
        x = np.asarray(V[lo:lo + chunk], dtype=np.float32) - mu[None]
        amax = np.maximum(amax, np.abs(x).reshape(-1, DIM).max(0))
    sd = (amax / 127.0).astype(np.float32)
    ref = {"mu_p": mu, "sd": sd}
    np.savez(os.path.join(tokens_dir, "quant_ref.npz"), **ref)
    print(f"[refs] mu_p {mu.shape} sd median {np.median(sd):.5f} max {sd.max():.4f} "
          f"{(time.time()-t0):.1f}s", flush=True)
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = os.path.join(R, "tokens")
    mu, sd = build_refs(out)
    mu_t = torch.from_numpy(mu).to(args.device)
    sd_t = torch.from_numpy(sd).to(args.device)

    model, _ = Q.build_vision(device=args.device)
    proc = None
    names, classes = Q.train_zip_entries()
    if args.limit:
        names = names[: args.limit]
    N = len(names)
    mm = np.lib.format.open_memmap(os.path.join(out, "train_tokens_int8.npy"), mode="w+",
                                   dtype=np.int8, shape=(N, N_TOK, DIM))
    sc = np.lib.format.open_memmap(os.path.join(out, "train_scale.npy"), mode="w+",
                                   dtype=np.float16, shape=(N, N_TOK))
    json.dump({"entries": names, "classes": classes},
              open(os.path.join(out, "train_index.json"), "w"))
    rng = np.random.default_rng(0)
    sub_idx = np.sort(rng.choice(N, size=min(args.subset, N), replace=False))
    sub_pos = {int(i): k for k, i in enumerate(sub_idx)}
    sub = np.zeros((len(sub_idx), N_TOK, DIM), np.float16)

    ds = Q.ZipImageDataset(names)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True,
        prefetch_factor=args.prefetch, persistent_workers=True,
        collate_fn=Q._collate_u8, worker_init_fn=Q._worker_init)
    t0 = time.time()
    done = 0
    for it, (idx, u8) in enumerate(loader):
        idx = idx.numpy()
        pv, thw = Q.patchify_batch(u8, device=args.device)
        emb = Q.encode_pv(model, pv, thw, n_img=len(idx), device=args.device)   # (B,49,5120) f32
        xc = emb - mu_t[None]
        y = xc / sd_t[None, None, :]
        s_t = y.abs().amax(-1).clamp_min(1e-8) / 127.0
        q = torch.round(y / s_t[..., None]).clamp(-127, 127).to(torch.int8)
        mm[idx] = q.cpu().numpy()
        sc[idx] = s_t.cpu().numpy().astype(np.float16)
        for b, gi in enumerate(idx):
            k = sub_pos.get(int(gi))
            if k is not None:
                sub[k] = emb[b].float().cpu().numpy().astype(np.float16)
        done += len(idx)
        if it % 20 == 0 or done == N:
            mm.flush(); sc.flush()
            print(f"[extract] {done}/{N} {done/(time.time()-t0):.1f} img/s "
                  f"eta {(N-done)/max(done/(time.time()-t0),1e-9)/60:.1f} min", flush=True)
    mm.flush(); sc.flush()
    np.save(os.path.join(out, "train_fp16_subset.npy"), sub)
    np.save(os.path.join(out, "train_fp16_subset_idx.npy"), sub_idx)
    # 保真度：用落盘 memmap 反量化 vs fp16 子集
    mm2 = np.load(os.path.join(out, "train_tokens_int8.npy"), mmap_mode="r")
    sc2 = np.load(os.path.join(out, "train_scale.npy"), mmap_mode="r")
    sse = ref = amax_d = 0.0
    for j in range(0, len(sub_idx), 500):
        ii = sub_idx[j:j + 500]
        qq = np.asarray(mm2[ii], dtype=np.float32)
        ss = np.asarray(sc2[ii], dtype=np.float32)
        rec = (qq * ss[..., None]) * sd[None, None, :] + mu[None]
        d = rec - sub[j:j + 500].astype(np.float32)
        amax_d = max(amax_d, float(np.abs(d).max()))
        sse += float((d.astype(np.float64) ** 2).sum())
        ref += float((sub[j:j + 500].astype(np.float64) ** 2).sum())
    rel = (sse / ref) ** 0.5
    qc = {"scheme": "posmean_perchannel_pertoken_int8",
          "n_tokens_checked": int(len(sub_idx) * N_TOK),
          "max_abs_diff": amax_d, "rel_rmse": float(rel),
          "token_psnr_db": float(-20 * np.log10(max(rel, 1e-12))),
          "ref_note": "mu_p/sd 由 val 5,000 张 fp16 token 估计（只影响存储刻度）"}
    json.dump(qc, open(os.path.join(out, "quant_check.json"), "w"), indent=1)
    print("[quant]", json.dumps(qc), flush=True)
    meta = {"n_train": int(N), "n_val": int(len(np.load(os.path.join(out, "val_labels.npy")))),
            "tokens_per_image": N_TOK, "dim": DIM, "resolution": 224,
            "quant_scheme": "posmean_perchannel_pertoken_int8",
            "quant_ref": "quant_ref.npz", "val_tokens_dtype": "float16",
            "train_tokens_dtype": "int8", "model": "Qwen/Qwen3.8-27B visual tower (460.73M)",
            "vision_only_sha256": "ef7ccc6d74936c63ee52d99a437fd9870708da7b383edab27ab0419fc669b9a8",
            "files": {f: os.path.getsize(os.path.join(out, f)) for f in
                      ("train_tokens_int8.npy", "train_scale.npy", "quant_ref.npz",
                       "train_fp16_subset.npy", "train_index.json", "val_tokens_fp16.npy",
                       "val_images_224.npy", "val_labels.npy")},
            "throughput_img_s": float(N / (time.time() - t0)),
            "elapsed_min": float((time.time() - t0) / 60),
            "zip_source": Q.ZIP_PATH, "fast_path": True,
            "patchify_equiv_max_abs_diff": 1.1920928955078125e-07,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("[meta] written. DONE", flush=True)


if __name__ == "__main__":
    main()
