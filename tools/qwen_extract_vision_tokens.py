#!/usr/bin/env python3
"""qwen_extract_vision_tokens.py — 用 Qwen3.8-27B 的**仅图像编码器**给 ImageNet-100 抽 token。

实验口径（2026-10-30，用户）: 下载 Qwen3.8-27B 的仅图像编码器 → 输入 IN-100 **训练集**
得到图像 embedding（visual token 序列）→ 同图按 token 顺序（row-major 合并网格）拼接 →
全训练集池化做 PCA → 取 1σ/2σ/3σ/全部 主成分，喂「同参数量 AssocVTC 交叉注意力单步还原探针」。

产物（默认 --out_root /root/autodl-tmp/qwen_vtc）:
  tokens/train_tokens_int8.npy   (N,49,5120) int8   per-token absmax 量化（30.5 GB）
  tokens/train_scale.npy         (N,49) float16     per-token absmax 反量化 scale
  tokens/train_index.json        {"entries":[...], "classes":[...]} 与 int8 行一一对应
  tokens/train_fp16_subset.npy   (S,49,5120) float16 固定随机子集（保精度校验用）
  tokens/train_fp16_subset_idx.npy (S,) int64
  tokens/val_tokens_fp16.npy     (5000,49,5120) float16
  tokens/meta.json
  tokens/quant_check.json        int8 量化保真度（子集上 max|Δ| / 相对误差 / token PSNR）

另含 `--mode prep_val`: 从 ILSVRC2012 val tar + devkit GT 里筛出 IN-100 的 5,000 张，
落 val_images_224.npy (5000,224,224,3) uint8 + val_labels.npy + val_paths.json。

用法:
  python tools/qwen_extract_vision_tokens.py --mode prep_val
  python tools/qwen_extract_vision_tokens.py --mode smoke          # 4 张冒烟
  python tools/qwen_extract_vision_tokens.py --mode extract --split val
  python tools/qwen_extract_vision_tokens.py --mode extract --split train --workers 16
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch
from PIL import Image

W_DEFAULT = "/root/autodl-tmp/qwen_vtc/weights"
# 训练集 zip：默认 AutoDL 公共盘（NFS，随机读慢）；可用 QWEN_ZIP 指向本地副本提速
ZIP_PATH = os.environ.get("QWEN_ZIP", "/root/autodl-pub/ImageNet100/imagenet100.zip")
VAL_TAR = "/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar"
DEVKIT = "/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz"
RES = 224
GRID = RES // 16          # 14 patches/side
MERGE = 2
N_TOK = (GRID // MERGE) ** 2   # 49
OUT_DIM = 5120


# ───────────────────────────── 图像编码器 ─────────────────────────────
def build_vision(weights_dir=W_DEFAULT, device="cuda"):
    from safetensors.torch import load_file
    from transformers.models.qwen3_5 import Qwen3_5VisionConfig, Qwen3_5VisionModel

    cfg_all = json.load(open(os.path.join(weights_dir, "config.json")))
    vcfg = Qwen3_5VisionConfig(**cfg_all["vision_config"])
    model = Qwen3_5VisionModel(vcfg).eval()
    sd = load_file(os.path.join(weights_dir, "vision_only.safetensors"))
    sd = {k[len("model.visual."):] if k.startswith("model.visual.") else k: v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"权重不匹配 missing={missing[:5]} unexpected={unexpected[:5]}")
    model = model.to(device=device, dtype=torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"[qwen] vision tower loaded: {n/1e6:.2f}M params, dim={vcfg.out_hidden_size}")
    return model, vcfg


def build_proc(weights_dir=W_DEFAULT):
    from transformers import Qwen2VLImageProcessor
    pre = json.load(open(os.path.join(weights_dir, "preprocessor_config.json")))
    kw = dict(
        size={"shortest_edge": RES * RES, "longest_edge": RES * RES},
        patch_size=int(pre["patch_size"]),
        temporal_patch_size=int(pre["temporal_patch_size"]),
        merge_size=int(pre["merge_size"]),
        image_mean=list(pre["image_mean"]),
        image_std=list(pre["image_std"]),
        do_resize=True, do_rescale=True, do_normalize=True,
        resample=3,  # bicubic
    )
    try:
        return Qwen2VLImageProcessor(**kw)
    except TypeError:
        from transformers import Qwen2VLImageProcessorPil
        return Qwen2VLImageProcessorPil(**kw)


def to_pixel_values(proc, images):
    """list[PIL] → (pixel_values, grid_thw)，并断言每图 N_TOK 个 merged token。"""
    out = proc(images=images, return_tensors="pt")
    pv = out["pixel_values"]
    thw = out["image_grid_thw"]
    assert tuple(thw[0].tolist()) == (1, GRID, GRID), f"grid={thw.tolist()}"
    return pv, thw


# ── 快速路径：DataLoader worker 只做 decode + PIL bicubic resize→224 uint8，
# normalize + patchify 在主进程 GPU 上一次完成（processor 的 CPU patchify ≈19ms/img 是旧链路主瓶颈）。
_PCFG = None


def patch_cfg():
    global _PCFG
    if _PCFG is None:
        pre = json.load(open(os.path.join(W_DEFAULT, "preprocessor_config.json")))
        _PCFG = dict(mean=list(pre["image_mean"]), std=list(pre["image_std"]),
                     ps=int(pre["patch_size"]), tps=int(pre["temporal_patch_size"]),
                     ms=int(pre["merge_size"]))
    return _PCFG


def patchify_batch(u8, device=None):
    """u8 (B,224,224,3) uint8 → (pixel_values (B*196,1536) float32, grid_thw (B,3) int64)

    精确复刻 Qwen2VLImageProcessor._preprocess：rescale 1/255 → normalize(mean,std) → patchify。
    patchify 维度重排与 transformers 源码逐行一致：
      [B,C,gh/ms,ms,ps,gw/ms,ms,ps] → permute(0,2,5,3,6,1,4,7) → 时间维复制 → (B,gh*gw,C*T*ps*ps)
    """
    cfg = patch_cfg()
    if device is not None:
        u8 = u8.to(device, non_blocking=True)
    B = u8.shape[0]
    ps, ms = cfg["ps"], cfg["ms"]
    gh = gw = RES // ps                                     # 14
    x = u8.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    mean = torch.tensor(cfg["mean"], device=x.device, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(cfg["std"], device=x.device, dtype=torch.float32)[None, :, None, None]
    x = (x - mean) / std
    x = x.reshape(B, 3, gh // ms, ms, ps, gw // ms, ms, ps)
    x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)                   # (B,7,7,2,2,3,16,16)
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, cfg["tps"], -1, -1)
    x = x.reshape(B, gh * gw, 3 * cfg["tps"] * ps * ps)
    thw = torch.tensor([[1, gh, gw]] * B, dtype=torch.long, device=u8.device)
    return x.reshape(B * gh * gw, 3 * cfg["tps"] * ps * ps), thw


def check_patchify_equiv(out_root, n=8, rng_seed=0):
    """随机 n 张 val 图：processor 输出 vs GPU patchify 输出，返回 max|Δ|（float32）。"""
    arr = np.load(os.path.join(out_root, "tokens", "val_images_224.npy"), mmap_mode="r")
    rng = np.random.default_rng(rng_seed)
    pick = np.sort(rng.choice(len(arr), size=n, replace=False))
    u8 = torch.from_numpy(np.stack([np.asarray(arr[i], dtype=np.uint8) for i in pick]))
    pv_fast, _ = patchify_batch(u8, device="cuda")
    pv_fast = pv_fast.cpu()
    proc = build_proc()
    pv_ref, _ = to_pixel_values(proc, [Image.fromarray(x) for x in u8.numpy()])
    return float((pv_fast - pv_ref.float()).abs().max())


def _collate_u8(items):
    idx = torch.tensor([it[0] for it in items], dtype=torch.long)
    u8 = torch.from_numpy(np.stack([it[1] for it in items]))    # (B,224,224,3) uint8
    return idx, u8


def _worker_init(_worker_id):
    torch.set_num_threads(1)


@torch.no_grad()
def encode_pv(model, pv, thw, n_img=None, device="cuda"):
    """pixel_values (B*196,1536) + grid (B,3) → (B,49,5120) float32，留在 device 上。"""
    B = int(thw.shape[0]) if n_img is None else n_img
    pv = pv.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    thw = thw.to(device=device, non_blocking=True)
    out = model(pv, thw)                      # BaseModelOutputWithPooling
    # pooler_output = patch merger 之后的 merged visual token: (B*49, 5120)
    # last_hidden_state 是 merger 之前 (B*196, 1152)，不能用
    emb = getattr(out, "pooler_output", None)
    if emb is None:
        emb = getattr(out, "last_hidden_state", out)
    if not torch.is_tensor(emb):
        raise RuntimeError(f"unexpected vision output type {type(out)}")
    emb = emb.float().reshape(B, N_TOK, -1)
    assert emb.shape[1:] == (N_TOK, OUT_DIM), f"bad shape {tuple(emb.shape)}"
    return emb


@torch.no_grad()
def encode_batch(model, proc, images, device="cuda"):
    """list[PIL] → (B,49,5120) float32 (CPU)。smoke 用，走主进程 proc。"""
    pv, thw = to_pixel_values(proc, images)
    return encode_pv(model, pv, thw, n_img=len(images), device=device).cpu()


# ───────────────────────────── 数据 ─────────────────────────────
def train_zip_entries():
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
    imgs = [n for n in names if n.lower().endswith((".jpeg", ".jpg", ".png"))]
    imgs.sort(key=lambda n: (n.split("/")[1], n))
    classes = sorted({n.split("/")[1] for n in imgs})
    assert len(classes) == 100, f"class 数 {len(classes)} != 100"
    return imgs, classes


class ZipImageDataset(torch.utils.data.Dataset):
    """按索引从 zip 取图，并在 **worker 内** resize 224 + processor → pixel_values。"""

    def __init__(self, names, res=RES):
        self.names = names
        self.res = res
        self._z = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self._z is None:
            self._z = zipfile.ZipFile(ZIP_PATH)
        with self._z.open(self.names[i]) as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        img = img.resize((self.res, self.res), Image.BICUBIC)
        return i, np.asarray(img, dtype=np.uint8)


class ValImageDataset(torch.utils.data.Dataset):
    """val_images_224.npy → uint8 (224,224,3)（worker 只读、不预处理）。"""

    def __init__(self, arr_path, res=RES):
        self.arr = np.load(arr_path, mmap_mode="r")
        self.res = res

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return i, np.asarray(self.arr[i], dtype=np.uint8)


# ───────────────────────────── prep_val ─────────────────────────────
def prep_val(out_root):
    import tarfile
    import scipy.io as sio
    from torch.utils.data import DataLoader

    out = os.path.join(out_root, "tokens")
    os.makedirs(out, exist_ok=True)
    dev = os.path.join(out_root, "devkit")
    if not os.path.isdir(dev):
        os.makedirs(dev, exist_ok=True)
        with tarfile.open(DEVKIT) as t:
            t.extractall(dev)
    # devkit 目录名带版本号，递归找
    gt_file = synset_file = None
    for root, dirs, files in os.walk(dev):
        for f in files:
            if f == "ILSVRC2012_validation_ground_truth.txt":
                gt_file = os.path.join(root, f)
        if "meta.mat" in files:
            synset_file = os.path.join(root, "meta.mat")
    assert gt_file and synset_file, (gt_file, synset_file)
    gt = np.loadtxt(gt_file, dtype=np.int64)                    # (50000,) 取值 1..1000
    meta = sio.loadmat(synset_file)
    # devkit meta.mat 的 synsets 是 (N,1) struct 数组，字段含 ILSVRC2012_ID / WNID；
    # 不能直接 np.array(s) 取字符串。这里建 ILSVRC2012_ID(1..1000) -> WNID 的映射。
    syn_arr = meta["synsets"].ravel()
    id_key = "ILSVRC2012_ID"
    assert id_key in syn_arr.dtype.names, syn_arr.dtype.names
    syn_ids = np.array([int(np.ravel(e[id_key])[0]) for e in syn_arr])
    syn_wnids = np.array([str(np.ravel(e["WNID"])[0]) for e in syn_arr])
    id2wnid = {int(i): w for i, w in zip(syn_ids, syn_wnids)}
    assert set(range(1, 1001)) <= set(id2wnid), "devkit synsets 未覆盖 ILSVRC2012 ID 1..1000"
    assert gt.min() == 1 and gt.max() == 1000, (gt.min(), gt.max())
    # 交叉校验：位置 gt-1 的 WNID 与按 ID=gt 查到的一致（防止任何 off-by-one 歧义）
    assert all(syn_wnids[g - 1] == id2wnid[int(g)] for g in gt), "devkit 位置序与 ID 序不一致"
    _, classes = train_zip_entries()
    keep = {s: i for i, s in enumerate(classes)}
    # val tar 内文件按名字升序 = GT 行序（文件名 ..._%08d.JPEG），用序号定位 GT 行
    import re
    with tarfile.open(VAL_TAR) as t:
        members = [m for m in t.getmembers() if m.name.lower().endswith(".jpeg")]
    members.sort(key=lambda m: m.name)
    nums = [int(re.search(r"_(\d+)\.JPEG$", m.name).group(1)) for m in members]
    assert nums == list(range(1, len(members) + 1)), "val tar 文件名序号不连续"
    assert len(members) == 50000, len(members)
    sel = []          # (member, cls_idx_in_100)
    for k, m in enumerate(members):
        cls_all = int(gt[nums[k] - 1])                 # 1..1000
        syn = id2wnid[cls_all]                         # ← 原代码 synsets[gt[k]] 是 off-by-one
        if syn in keep:
            sel.append((m, keep[syn]))
    counts = np.bincount([c for _, c in sel], minlength=100)
    print(f"[prep_val] selected {len(sel)} / {len(members)}; per-class {counts.min()}..{counts.max()}")
    assert len(sel) == 5000 and counts.min() == 50 and counts.max() == 50, (len(sel), counts.min(), counts.max())
    assert len({c for _, c in sel}) == 100, "有类别没被 val 覆盖"
    # 人工核对：val 路径 → 100 类内 label → synset（Lead 要求），并落盘
    checks = []
    for j in (0, 1, len(sel) // 2):
        m, c = sel[j]
        checks.append({"path": m.name, "label_in100": int(c), "synset": classes[c]})
    for ck in checks:
        print(f"[prep_val] check {ck}")
    json.dump({"n_selected": len(sel), "per_class_min": int(counts.min()),
               "per_class_max": int(counts.max()),
               "n_classes": int(len({c for _, c in sel})),
               "classes": classes, "check": checks},
              open(os.path.join(out, "val_label_map_check.json"), "w"), indent=1)

    arr = np.zeros((len(sel), RES, RES, 3), np.uint8)
    labels = np.zeros(len(sel), np.int64)
    paths = []
    with tarfile.open(VAL_TAR) as t:
        for j, (m, c) in enumerate(sel):
            f = t.extractfile(m)
            img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((RES, RES), Image.BICUBIC)
            arr[j] = np.asarray(img, np.uint8)
            labels[j] = c
            paths.append(m.name)
    np.save(os.path.join(out, "val_images_224.npy"), arr)
    np.save(os.path.join(out, "val_labels.npy"), labels)
    json.dump({"paths": paths, "classes": classes}, open(os.path.join(out, "val_paths.json"), "w"))
    print(f"[prep_val] saved {out}/val_images_224.npy {arr.shape} {arr.nbytes/1e6:.1f}MB")


# ───────────────────────────── extract ─────────────────────────────
def extract_split(args, out_root, model, proc, device):
    out = os.path.join(out_root, "tokens")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if args.split == "train":
        names, classes = train_zip_entries()
        ds = ZipImageDataset(names)
        if args.limit:
            ds.names = ds.names[: args.limit]
            names = names[: args.limit]
        N = len(ds)
        mm = np.lib.format.open_memmap(os.path.join(out, "train_tokens_int8.npy"), mode="w+",
                                       dtype=np.int8, shape=(N, N_TOK, OUT_DIM))
        sc = np.lib.format.open_memmap(os.path.join(out, "train_scale.npy"), mode="w+",
                                       dtype=np.float16, shape=(N, N_TOK))
        json.dump({"entries": names, "classes": classes},
                  open(os.path.join(out, "train_index.json"), "w"))
        rng = np.random.default_rng(0)
        sub_idx = np.sort(rng.choice(N, size=min(args.subset, N), replace=False))
        sub_idx_set = {int(i): k for k, i in enumerate(sub_idx)}
        sub = np.zeros((len(sub_idx), N_TOK, OUT_DIM), np.float16)
        qmax_abs = 0.0
        q_sum_sq = 0.0
        q_sum_ref = 0.0
        n_vals = 0
    else:
        arr_path = os.path.join(out, "val_images_224.npy")
        assert os.path.exists(arr_path), "先跑 --mode prep_val"
        ds = ValImageDataset(arr_path)
        N = len(ds)
        mm = np.lib.format.open_memmap(os.path.join(out, "val_tokens_fp16.npy"), mode="w+",
                                       dtype=np.float16, shape=(N, N_TOK, OUT_DIM))
        sc = None
        if args.limit:
            ds.arr = ds.arr[: args.limit]
            N = len(ds)
    # 快速路径等价性验证（processor vs GPU patchify），失败则回退主进程 processor
    equiv_max = None
    use_fast = bool(args.fast)
    if use_fast:
        try:
            equiv_max = check_patchify_equiv(out_root)
        except Exception as e:  # noqa: BLE001
            print(f"[patchify] equiv check 失败: {e}", flush=True)
            equiv_max = None
        use_fast = equiv_max is not None and equiv_max < 1e-4
        json.dump({"max_abs_diff": equiv_max, "fast_path": bool(use_fast),
                   "threshold": 1e-4, "n_images": 8},
                  open(os.path.join(out, "patchify_equiv.json"), "w"), indent=1)
        print(f"[patchify] fast_path={use_fast} processor-vs-GPU max|Δ|={equiv_max}", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
        pin_memory=True, prefetch_factor=args.prefetch if args.workers else None,
        persistent_workers=args.workers > 0, collate_fn=_collate_u8,
        worker_init_fn=_worker_init)
    done = 0
    for it, (idx, u8) in enumerate(loader):
        idx = idx.numpy()
        if use_fast:
            pv, thw = patchify_batch(u8, device=device)     # GPU 上 normalize+patchify
        else:
            pv, thw = to_pixel_values(proc, [Image.fromarray(a) for a in u8.numpy()])
        emb = encode_pv(model, pv, thw)                    # (B,49,5120) fp32，在 GPU 上
        if args.split == "train":
            # 量化在 GPU 上完成，只把 int8 token + fp16 scale 传回 CPU（省 ≥2/3 传输）
            amax = emb.abs().amax(dim=-1).clamp_min(1e-8)      # (B,49)
            scale = amax / 127.0
            q = torch.round(emb / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
            deq = q.float() * scale.unsqueeze(-1)
            d = (deq - emb)
            qmax_abs = max(qmax_abs, d.abs().max().item())
            q_sum_sq += (d ** 2).sum().item()
            q_sum_ref += (emb ** 2).sum().item()
            n_vals += emb.numel()
            mm[idx] = q.cpu().numpy()
            sc[idx] = scale.cpu().numpy().astype(np.float16)
            sub_pos = [(b, sub_idx_set[int(gi)]) for b, gi in enumerate(idx)
                       if int(gi) in sub_idx_set]
            if sub_pos:
                sub[ [k for _, k in sub_pos] ] = emb[
                    torch.tensor([b for b, _ in sub_pos], device=emb.device)
                ].half().cpu().numpy()
            if it % 50 == 0:
                mm.flush()
                sc.flush()
        else:
            mm[idx] = emb.half().cpu().numpy()
        done += len(idx)
        if it % 20 == 0 or done == N:
            el = time.time() - t0
            print(f"[extract:{args.split}] {done}/{N} {done/el:.1f} img/s "
                  f"eta {(N-done)/(done/el)/60:.1f} min", flush=True)
    mm.flush()
    if sc is not None:
        sc.flush()
        np.save(os.path.join(out, "train_fp16_subset.npy"), sub)
        np.save(os.path.join(out, "train_fp16_subset_idx.npy"), sub_idx)
        rel = (q_sum_sq / max(q_sum_ref, 1e-12)) ** 0.5
        qc = {"n_tokens_checked": int(n_vals // OUT_DIM), "max_abs_diff": qmax_abs,
              "rel_rmse": float(rel),
              "token_psnr_db": float(-20 * np.log10(max(rel, 1e-12))),
              "subset_idx": [int(x) for x in sub_idx[:20]]}
        json.dump(qc, open(os.path.join(out, "quant_check.json"), "w"), indent=1)
        print("[quant]", json.dumps(qc))
    print(f"[extract:{args.split}] DONE {done} imgs in {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prep_val", "smoke", "extract"], default="extract")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--out_root", default="/root/autodl-tmp/qwen_vtc")
    ap.add_argument("--weights", default=W_DEFAULT)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--fast", type=int, default=1, help="1=GPU patchify 快速路径(默认), 0=强制 processor")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    if args.mode == "prep_val":
        prep_val(args.out_root)
        return
    model, vcfg = build_vision(args.weights, args.device)
    proc = build_proc(args.weights)
    if args.mode == "smoke":
        names, classes = train_zip_entries()
        with zipfile.ZipFile(ZIP_PATH) as z:
            imgs = [Image.open(io.BytesIO(z.read(names[i]))).convert("RGB").resize((RES, RES), Image.BICUBIC)
                    for i in [0, 1, 500, 70000]]
        emb = encode_batch(model, proc, imgs, args.device)
        assert tuple(emb.shape) == (4, N_TOK, OUT_DIM), tuple(emb.shape)
        assert torch.isfinite(emb).all(), "有 nan/inf"
        norms = emb.norm(dim=(1, 2))
        within = emb.std(dim=1).mean().item()          # 同图 49 个 token 之间的差异
        # per-token 余弦：emb[0]/emb[1] 均为 (49,5120)，dim=-1 得 49 个相似度
        cs = torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=-1)
        print("[smoke] tokens", tuple(emb.shape), "mean", emb.mean().item(), "std", emb.std().item())
        print("[smoke] per-image L2 norm", norms.tolist())
        print("[smoke] within-image token std", within)
        print("[smoke] cos(img0,img1) mean", cs.mean().item(), "std", cs.std().item())
        assert (norms > 1.0).all(), f"L2 norm 太小 {norms}"
        assert within > 1e-3, f"同图 49 token 几乎相同 (std={within})"
        assert cs.std().item() > 1e-4, "两图 token 余弦是常数"
        assert cs.mean().item() < 0.999, "两图 token 余弦全 1"
        print("[smoke] PASS")
        return
    extract_split(args, args.out_root, model, proc, args.device)


if __name__ == "__main__":
    main()
