#!/usr/bin/env python3
"""qwen_feature_sanity.py — Qwen 视觉 token 提取的**特征健全性检查**（线性探针）。

做法：每图 49 个 5120 维 token 做 mean-pool（5120 维）→ 按 train 统计标准化 →
sklearn LogisticRegression（多类，C=1.0）→ IN-100 val 5,000 张上测 top-1 / top-5。
目的不是刷精度，而是验证 token 提取（顺序、归一化、分辨率、未旋转/打乱）正确：
随机猜 1%；full-val top-1 < 30% 会显式报警。

两个口径（都会报告）：
- **primary（Lead 指定）**：按 `train_index.json` 的 entries **顺序取前 N 张**（默认 5 万）。
  ⚠️ entries 按 (class, name) 排序，前 5 万只覆盖前 ~40 类 ⇒ full-val top-1 天然上限 ~40%，
  因此同时报告 **seen-class top-1/top-5（只在已见类上重归一化）**；
- **control（100 类对照）**：随机取 N_control 张（默认 3 万，覆盖全部 100 类）→ full-val
  top-1/top-5，用来回答「>60% 预期」是否满足。

输入（--prep）:
  train_tokens_int8.npy (N,49,5120) int8 / train_scale.npy (N,49) fp16 / quant_ref.npz（B2）
  train_index.json {"entries":[...], "classes":[...]}
  val_tokens_fp16.npy (5000,49,5120) fp16 / val_labels.npy (5000,) int64

输出（--out）: feature_sanity.json（含 primary/control 的 top-1/top-5+seen、训练用时、曲线）

用法:
  python tools/qwen_feature_sanity.py \
      --prep /root/autodl-tmp/qwen_vtc/tokens --out /root/autodl-tmp/qwen_vtc/pca/feature_sanity.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

R = "/root/autodl-tmp/qwen_vtc"
DIM = 5120
N_TOK = 49


def log(msg):
    print(f"[sanity] {msg}", flush=True)


def load_labels(prep):
    idx = json.load(open(os.path.join(prep, "train_index.json")))
    entries, classes = idx["entries"], idx["classes"]
    cls2i = {c: i for i, c in enumerate(classes)}
    labels = np.array([cls2i[e.split("/")[1]] for e in entries], dtype=np.int64)
    return labels, len(classes)


def load_refs(prep, device):
    p = os.path.join(prep, "quant_ref.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return (torch.from_numpy(z["mu_p"].astype(np.float32)).to(device),
            torch.from_numpy(z["sd"].astype(np.float32)).to(device))


def pool_train(tok_path, scale_path, rows, refs, chunk=512):
    """int8+B2 反量化 → mean-pool（49 token 平均）→ (n,5120) f32。"""
    tok = np.load(tok_path, mmap_mode="r")
    sc = np.load(scale_path, mmap_mode="r")
    n = len(rows)
    out = np.empty((n, DIM), dtype=np.float32)
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        r0 = np.asarray(rows[lo:hi])
        order = np.argsort(r0)                      # 排序只影响读取顺序，最后按 order 还原
        r = r0[order]
        q = np.asarray(tok[r], dtype=np.int8).reshape(-1, N_TOK, DIM)
        s = np.asarray(sc[r], dtype=np.float32).reshape(-1, N_TOK, 1)
        x = torch.from_numpy(q.astype(np.float32))
        x.mul_(torch.from_numpy(s))
        if refs is not None:
            mu, sd = refs
            x.mul_(sd).view(len(r), N_TOK, DIM).add_(mu)
        feat = x.numpy().mean(1)
        out[lo:hi] = feat[np.argsort(order)]
        if (lo // chunk) % 20 == 0:
            log(f"pool train {hi}/{n} {(time.time()-t0):.1f}s")
    return out


def pool_val(val_path, chunk=512):
    val = np.load(val_path, mmap_mode="r")
    n = len(val)
    out = np.empty((n, DIM), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        out[lo:hi] = np.asarray(val[lo:hi], dtype=np.float32).mean(1)
    return out


def probe_linear(Xtr, ytr, Xva, yva, lam=10.0, n_classes=None, C=None):
    """标准化 + 闭式 ridge 线性分类器（等价线性探针，快且确定）→ top1/top5 + seen 重归一化指标。"""
    n_classes = int(n_classes or (max(ytr.max(), yva.max()) + 1))
    t0 = time.time()
    mu = Xtr.mean(0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(Xtr.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    Ztr = (Xtr - mu) / sd
    Zva = (Xva - mu) / sd
    Y = np.zeros((len(Ztr), n_classes), dtype=np.float32)
    Y[np.arange(len(Ztr)), ytr] = 1.0
    Y -= 1.0 / n_classes
    XtX = (Ztr.T @ Ztr).astype(np.float64)
    XtX[np.diag_indices_from(XtX)] += lam
    W = np.linalg.solve(XtX, (Ztr.T @ Y).astype(np.float64))      # (D, C)
    scores = Zva.astype(np.float64) @ W
    t_fit = time.time() - t0

    def acc(sc, y, classes=None):
        y_idx = y if classes is None else np.searchsorted(classes, y)
        keep = np.ones(len(y), dtype=bool) if classes is None else np.isin(y, classes)
        out = {"n": int(keep.sum())}
        if classes is not None:
            sc = sc[:, classes]
        kk = min(5, sc.shape[1])
        pred = np.argpartition(-sc, kk - 1, axis=1)[:, :kk]
        for k in (1, 5):
            k2 = min(k, sc.shape[1])
            hit = (pred[:, :k2] == y_idx[:, None]).any(1) & keep
            out[f"top{k}"] = float(hit.sum() / max(keep.sum(), 1))
        out["top1_all"] = float((np.argmax(sc, 1) == y_idx).mean())
        return out

    full = acc(scores, yva)
    seen = np.unique(ytr)
    full_seen = acc(scores, yva, classes=seen)
    return {
        "lam": lam, "n_train": int(len(Xtr)), "seen_classes": int(len(seen)),
        "pca_dim": None, "pca_explained_var": None, "pca_s": 0.0,
        "top1": full["top1"], "top5": full["top5"],
        "seen_top1": full_seen["top1"], "seen_top5": full_seen["top5"],
        "n_val_seen": full_seen["n"], "fit_s": t_fit,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=os.path.join(R, "tokens"))
    ap.add_argument("--out", default=os.path.join(R, "pca", "feature_sanity.json"))
    ap.add_argument("--n_train", type=int, default=50000, help="primary: train_index 顺序前 N 张")
    ap.add_argument("--subset_mode", choices=["first", "random"], default="first")
    ap.add_argument("--n_control", type=int, default=30000, help="control: 随机 N 张（覆盖 100 类）")
    ap.add_argument("--lam", type=float, default=10.0, help="ridge 正则（线性探针）")
    ap.add_argument("--cache_dir", default="", help="池化特征缓存（.npz），重跑快")
    ap.add_argument("--no_control", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(os.path.dirname(args.out), "_unittest", "feat_cache")
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device("cpu")                     # 特征池化用 CPU，不占 GPU
    refs = load_refs(args.prep, device)
    labels, n_classes = load_labels(args.prep)
    n_full = len(labels)
    tok_path = os.path.join(args.prep, "train_tokens_int8.npy")
    sc_path = os.path.join(args.prep, "train_scale.npy")

    # val 特征（只算一次）
    val_cache = os.path.join(cache_dir, "val_pool.npz")
    if os.path.exists(val_cache):
        d = np.load(val_cache)
        Xva, yva = d["X"], d["y"]
    else:
        Xva = pool_val(os.path.join(args.prep, "val_tokens_fp16.npy"))
        yva = np.load(os.path.join(args.prep, "val_labels.npy"))
        np.savez(val_cache, X=Xva, y=yva)
    log(f"val features: {Xva.shape}")

    def get_pool(subset_mode, n):
        n = min(n, n_full)
        if subset_mode == "random" and n < n_full:
            rows = np.random.default_rng(0).permutation(n_full)[:n]
        else:
            rows = np.arange(n, dtype=np.int64)
        tag = f"{subset_mode}_{n}"
        cache = os.path.join(cache_dir, f"train_pool_{tag}.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            return d["X"], d["y"], rows
        X = pool_train(tok_path, sc_path, rows, refs)
        y = labels[rows]
        np.savez(cache, X=X, y=y)
        return X, y, rows

    res = {"check": "mean-pooled Qwen vision tokens → linear classifier (IN-100)",
           "method": "standardize → closed-form ridge linear classifier (lam=10, one-hot centred targets)",
           "n_val": int(len(Xva)), "n_classes": int(n_classes), "chance_top1": 1.0 / n_classes,
           "quant_scheme": ("b2_posmean_perchannel_pertoken_int8" if refs is not None
                            else "v1_per_token_absmax_int8"),
           "note": ("primary = train_index 顺序前 N 张（entries 按 class,name 排序 ⇒ 只覆盖前 ~40 类，"
                    "full-val top1 上限 ~40%，看 seen_top1 才是 token 质量）；"
                    "control = 随机 N 张（覆盖全部 100 类），用于对照 >60% 预期。")}

    Xtr, ytr, rows = get_pool(args.subset_mode, args.n_train)
    log(f"primary pooled: {Xtr.shape} seen_classes={np.unique(ytr).size}")
    per, _ = probe_linear(Xtr, ytr, Xva, yva, n_classes=n_classes)
    res["primary"] = {**per, "subset_mode": args.subset_mode,
                      "label_range": [int(ytr.min()), int(ytr.max())],
                      "top1_best": per["top1"], "top5_best": per["top5"],
                      "top1_best_seen": per["seen_top1"], "top5_best_seen": per["seen_top5"],
                      "n_classes_seen_in_train": per["seen_classes"],
                      "n_train": int(len(Xtr)), "train_min": per["fit_s"] / 60.0}
    log("primary: " + json.dumps({k: res["primary"][k] for k in
                                  ("top1", "top5", "seen_top1", "seen_top5", "seen_classes",
                                   "pca_dim", "pca_explained_var", "pca_s", "fit_s")}))
    del Xtr
    if not args.no_control and args.n_control:
        Xc, yc, _ = get_pool("random", args.n_control)
        log(f"control pooled: {Xc.shape} seen_classes={np.unique(yc).size}")
        con, _ = probe_linear(Xc, yc, Xva, yva, n_classes=n_classes)
        res["control"] = con
        log("control: " + json.dumps({k: con[k] for k in
                                      ("top1", "top5", "seen_top1", "seen_top5", "seen_classes", "fit_s")}))
        del Xc

    prim = res["primary"]
    res["alarm"] = bool(prim["seen_top1"] < 0.30 and (res.get("control", {}).get("top1", 1.0) < 0.30))
    res["alarm_note"] = ("token 提取链路可能有问题（已见类 top1 与 100 类对照都 <30%），需排查"
                         if res["alarm"] else "OK")
    res["elapsed_min"] = (time.time() - t_start) / 60
    json.dump(res, open(args.out, "w"), indent=1)
    log("RESULT: " + json.dumps({k: res.get(k) for k in
                                ("primary", "control", "alarm", "elapsed_min")}, ensure_ascii=False)[:900])
    log(f"saved {args.out} ({res['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
