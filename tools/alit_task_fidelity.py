#!/usr/bin/env python3
"""E-C1 任务保真分类探针 — 自适应 token 预算 under IN-100 top-1
===============================================================================

PLAN `doc/2026-10-16/PLAN_alit_adaptive_experiments.md` §5 **E-C1** 的落地工具：
把 E-A2（TSC = 重建 L1 阈值选 token 数, 见 `tools/alit_adaptive_k.py`）的
**per-image 停步**接到「冻结编码器 + 线性分类探针」的 IN-100 val top-1 上, 在
**等平均 token 数**口径下比较 "自适应 K" vs "固定 K"。

流水线
------
1. **冻结编码器提特征**（默认 `--ckpt` 的 encoder, `model.encode(x) → (z_cls, z_s)`）:
   每图 z_s (K, D_enc) 与 z_cls (1, D_enc); 预算 k 的特征 =
   ``mean-pool(z_s[:k])`` （`--use_cls 1` 再拼 z_cls）。
2. **每档一个线性分类器**: 同一确定性划分（`data_linattn.build_datasets`,
   `--val_per_class 50` ⇒ val 5,000; train 子集 `--train_per_class 100`/类 =
   10,000 张, 类内等距抽取）; 特征**只在 train 子集上标准化**（per-dim mean/std）
   后训单层 Linear（`--probe logreg` 全批 Adam / `--probe lstsq` ridge 闭式）。
   对每个 k ∈ `--ks` 报 val top-1, 得 by-k 曲线。
3. **自适应 vs 等预算固定档**:
   · 自适应 top-1 = 每图用它自己的 k_i 调第 k_i 档分类器;
   · 对照 = 固定档曲线在 ``mean(k_i)`` 处**线性插值**的 top-1（等平均 token 数,
     ALIT Fig.6 的正确读法, **不是**等最大 token 数）;
   · 每档曲线 / 自适应 / 差值 Δ 都给 **paired bootstrap CI**（同一批重采样索引,
     且每个重采样样本内重新计算 mean(k_i) 再插值 ⇒ 预算是配对的）。
4. 输出 json + png 到 `--out` / `--out` 同名 .png（默认
   `doc/2026-10-16/artifacts/alit/`）。

`--tsc_json` 接口（**E-A2 → E-C1 的契约**）
------------------------------------------
本工具接受三种形态, 匹配优先级 ``path > filename > index``:

(a) **规范形态**（推荐, 也是本工具 `--smoke` 写出的样例）::

      {"per_image": [{"index": 0,               # val 划分内下标（可省）
                      "filename": "n01440764_xx.JPEG",   # 可省
                      "path": "/abs/path.JPEG",          # 可省, 最可靠
                      "k": 15, "step": 3},               # k = z_s token 数（必须）
                      ...],
       "k_grid": [3, 8, 15, 24, 35, 48, 63, 64],
       "cum_z_s": [3, 8, 15, 24, 35, 48, 63, 64],
       "tau": 6.5, "source": "E-A2 TSC"}

    其中 ``k``（别名 ``tokens``/``num_tokens``）是**该图实际读入的 z_s 前缀长度**。
    `step` 给的是循环读出的步号（0-based）, ``k = cum_z_s[step]``。
    `index` 语义 = `data_linattn.build_datasets(data_dir, img_size, val_per_class)`
    的 **val** 顺序（= `eval_stepwise_linattn.py --dump_per_image` 里 `images[j].index`）。

(b) **位置对齐的 k 列表**（与 val 顺序一一对应, 长度须 = val 张数; 也可包成
    ``{"k": [...]}`` / ``{"k_list": [...]}``）::

      [15, 8, 35, ..., 63]

(c) **直接喂 E-A1 的 `--dump_per_image` json**（免中间格式）::

      python tools/alit_task_fidelity.py --ckpt <ckpt> \
        --tsc_json doc/.../artifacts/alit/per_image_stdattn_k64_rec.json \
        --tsc_taus auto --tsc_n_taus 9 ...

    此时本工具**自己按 E-A2 的定义**算每图 k: 取最小 t 使
    ``L1_255[t] <= tau``（无满足则取末步 = capped）, ``k_i = cum_z_s[t]``;
    `--tsc_tau` 给单点, `--tsc_taus auto` 用与 `alit_adaptive_k.default_taus`
    同款的几何网格（末步 p10 → 首步 p50）扫一族阈值 ⇒ 得到一条
    "自适应 (mean k, top-1)" 曲线, 而不是单点。

**诚实边界（必读, 也写进了 json 的 `honesty` 字段）**
------------------------------------------------------
1. **线性分类探针 ≠ Phase 2 生成任务**。IN-100 100 类 top-1 只回答"z_s 前缀里
   还剩多少**类别可线性读出**的信息"。它**不能**替代 GOAL §2 的 caption/幻觉口径
   （E-C2, 独立排期）; 本工具的所有结论只能表述为"任务保真的**分类代理**"。
   把它当唯一验收 = 过度解读。
2. **等平均 token 数口径**: 自适应图的 k_i 逐图不同, 与固定档 K 比较时**必须**
   比"平均 token 数相同"的固定档（本工具在生产线上插值），**不能**比固定 K=64,
   也不能比等最大 token 数。mean(k) 与固定档 top-1 的插值存在格点误差（由 `--ks`
   密度决定）, 故同时报最近格点值。
3. **探针自身是训练出来的**: 每档一个 probe（`--train_per_class`,`--probe_*`）。
   不同 k 的 probe 收敛程度若有差异会污染曲线 ⇒ 工具同时报各档 train top-1,
   曲线出现非单调时先查这里。
4. **单种子 + 无方差**: 默认 seed 0; 只有 bootstrap CI 跨过 0 的差异才可谈
   "有/无"; <0.5 pp 的差值不作结论（PLAN §9.1）。
5. **TSC-TOI 对齐 gap**: TSC 用的是**重建** L1（自监督、免费）, 这里量的是
   **分类** TOI; 两者不必然对齐（PLAN §5 备注）。本工具额外报 oracle 上界
   （"若知道标签, 每图最小需要多少 token"）以量化这个 gap, 但那是**上界**,
   不是可实现的策略。
6. **`--smoke` 只是管线证据**: 合成数据 + 随机初始化编码器（无 `--ckpt` 时）只验证
   提取/缓存/探针/bootstrap/画图能跑通, **没有任何科学含义**。

用法
----
本地 CPU 冒烟（无数据/无 ckpt 也能跑, 合成数据 + 小随机 std 编码器）::

    python3 tools/alit_task_fidelity.py --smoke \
        --out doc/2026-10-16/artifacts/alit/smoke_task_fidelity.json

服务器全量（alit-runner; GPU 独占; 见交付汇报里的确切命令）::

    python tools/alit_task_fidelity.py \
      --ckpt /root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt \
      --data_dir /root/autodl-tmp/datasets/imagenet100 \
      --tsc_json doc/2026-10-16/artifacts/alit/per_image_stdattn_k64_rec.json \
      --tsc_taus auto --tsc_n_taus 9 \
      --train_per_class 100 --val_per_class 50 --batch_size 64 --num_workers 16 \
      --cache /root/autodl-tmp/alit_cache/zs_stdattn_k64_rec \
      --out doc/2026-10-16/artifacts/alit/task_fidelity_stdattn_rec.json

依赖: 只用 `data_linattn`（不改）+ `model_v2_linattn`（不改）+ torch/numpy/matplotlib。
不依赖本地绝对路径; 路径全部参数化, 可直接 rsync 到服务器跑。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_linattn import ImageNet100, build_transform, scan_imagenet100, split_index  # noqa: E402
from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn  # noqa: E402

DEFAULT_KS = "1,4,8,12,16,24,32,40,48,56,64"
SMOKE_KS = "4,8,16"
DEFAULT_OUT = "doc/2026-10-16/artifacts/alit/task_fidelity.json"

HONESTY = [
    "分类线性探针 ≠ Phase 2 生成任务: 这里只测 'z_s 前缀中类别可线性读出的信息', "
    "不能替代 GOAL §2 的 caption/幻觉口径 (E-C2)。",
    "等平均 token 数 (mean k) 口径: 自适应逐图 k_i 必须与 '平均 token 数相同' 的固定档"
    "比较, 不能与固定 K=64 或等最大 token 数比较; 插值存在格点误差, 故同时报最近格点值。",
    "每档 probe 独立训练: 曲线非单调时先看各档 train top-1 (欠拟合/过拟合) 再谈科学结论。",
    "单种子 (默认 seed 0): 只有 paired bootstrap CI 不跨 0 的差异可谈方向; <0.5 pp 不作结论。",
    "TSC (重建 L1, 自监督) 与 TOI (分类) 不必然对齐; oracle 上界只量化 gap, 不是可实现策略。",
    "--smoke 是管线证据: 合成数据 + 随机编码器, 无科学含义。",
]

EPILOG = """\
--tsc_json 接口（三形态皆可）:
  1) 规范 per_image: {"per_image":[{"index":i,"filename":str,"path":str,"k":int,"step":int}, ...],
                      "k_grid":[..],"cum_z_s":[..],"tau":float}   ← 推荐
  2) 位置对齐 k 列表: [k0, k1, ...]（或 {"k":[...]}）与 val 顺序一一对应
  3) E-A1 dump: 直接给 eval_stepwise_linattn.py --dump_per_image 的 json + --tsc_tau[s]
匹配优先级 path > filename > index（index = data_linattn val 划分顺序）;
k = 该图读入的 z_s 前缀长度; step → k 用 cum_z_s[step]。

本地 CPU 冒烟（无数据/无 ckpt）:
  python3 tools/alit_task_fidelity.py --smoke --out doc/2026-10-16/artifacts/alit/smoke_task_fidelity.json

服务器全量（alit-runner, GPU 独占）:
  python tools/alit_task_fidelity.py --ckpt /root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt --data_dir /root/autodl-tmp/datasets/imagenet100 --tsc_json doc/2026-10-16/artifacts/alit/per_image_stdattn_k64_rec.json --tsc_taus auto --tsc_n_taus 9 --train_per_class 100 --val_per_class 50 --batch_size 64 --num_workers 16 --cache /root/autodl-tmp/alit_cache/zs_stdattn_k64_rec --out doc/2026-10-16/artifacts/alit/task_fidelity_stdattn_rec.json

诚实边界: 分类线性探针 ≠ Phase 2 生成任务; 等平均 token 数口径必须配对插值;
单种子, 只信 bootstrap CI 不跨 0 的方向。
"""


# ══════════════════════════ 通用小工具 ══════════════════════════

def _csv_ints(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def _ci(x: np.ndarray, level: float) -> Tuple[float, float]:
    a = (1.0 - float(level)) / 2.0 * 100.0
    lo, hi = np.percentile(np.asarray(x, dtype=np.float64), [a, 100.0 - a])
    return float(lo), float(hi)


def _even_subsample(n_total: int, n_want: int) -> List[int]:
    """确定性等距抽样（不是取前缀; 前缀会让验证类分布偏斜）。"""
    if n_want <= 0 or n_want >= n_total:
        return list(range(n_total))
    idx = np.linspace(0, n_total - 1, n_want)
    return [int(round(v)) for v in idx]


def _pick_per_class(items: Sequence[Tuple[str, int]], per_class: int
                    ) -> List[Tuple[str, int]]:
    """每类**类内等距**抽 per_class 张（确定性; 与 data_linattn.split_index 风格一致）。"""
    if per_class <= 0:
        return list(items)
    by_cls: Dict[int, List[Tuple[str, int]]] = {}
    for p, c in items:
        by_cls.setdefault(int(c), []).append((p, int(c)))
    out: List[Tuple[str, int]] = []
    for c in sorted(by_cls):
        lst = by_cls[c]
        n = len(lst)
        k = min(per_class, n)
        if k <= 0:
            continue
        picks = ([int(round(i * (n - 1) / (k - 1))) for i in range(k)]
                 if k > 1 else [0])
        seen = set()
        for i in picks:
            if i not in seen:
                seen.add(i)
                out.append(lst[i])
    return out


def _ds_paths(ds) -> List[str]:
    if hasattr(ds, "paths"):
        return [str(p) for p in ds.paths]
    if hasattr(ds, "items"):
        return [str(p) for p, _ in ds.items]
    return [f"synthetic://{i}" for i in range(len(ds))]


# ══════════════════════════ 数据集（真实 / 合成） ══════════════════════════

class SyntheticProbeDataset(torch.utils.data.Dataset):
    """`--smoke` 专用: 每类一个恒定低频色底 + 噪声（有可学信号, 便于验证探针管线）。

    **不是**数据集划分口径的一部分, 只用于无数据环境下的管线自检: 目标只是让
    探针 val top-1 明显高于 chance, 从而 bootstrap/等预算对照确有数值可跑。
    """

    def __init__(self, n: int, n_classes: int, img_size: int, seed: int, tag: str):
        g = torch.Generator().manual_seed(int(seed))
        col = torch.randn(n_classes, 3, 1, 1, generator=g)     # 每类一个恒定色
        y = torch.arange(n) % n_classes
        x = col[y].expand(-1, 3, img_size, img_size).clone() \
            + 0.35 * torch.randn(n, 3, img_size, img_size, generator=g)
        self.x, self.y = x, y
        self.paths = [f"synthetic://{tag}/{i}.png" for i in range(n)]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int):
        return self.x[i], int(self.y[i])


def build_probe_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset,
                                        str, bool]:
    """→ (train_ds, val_ds, source, synthetic?)。

    · 真实数据: 复用 `data_linattn` 的确定性划分（**不修改** data_linattn.py）;
      val = `--val_per_class` × 100（再按 `--val_n` 等距子采样）;
      train = `--train_per_class`/类（类内等距）。
    · `--smoke` 且无可用 `--data_dir` ⇒ 合成数据集（小样本）。
    """
    use_real = (bool(args.data_dir) and os.path.isdir(args.data_dir)
                and not args.smoke_force_synthetic)
    items = None
    if use_real:
        try:
            items = scan_imagenet100(args.data_dir)
        except Exception as e:                                   # noqa: BLE001
            if not args.smoke:
                raise
            print(f"[warn] --smoke: 读不到真实数据 ({e}); 退到合成数据")
            use_real = False
    if use_real:
        assert items is not None
        tr, va = split_index(items, args.val_per_class)
        tr_sub = _pick_per_class(tr, args.train_per_class)
        va_sub = [va[i] for i in _even_subsample(len(va), args.val_n)]
        tf = build_transform(args.img_size, train=bool(args.train_aug))
        return (ImageNet100(tr_sub, tf, args.img_size),
                ImageNet100(va_sub, tf, args.img_size), "imagenet100", False)
    n_cls = int(args.smoke_n_classes)
    n_tr = int(args.smoke_n_train)
    n_va = int(args.smoke_n_val)
    return (SyntheticProbeDataset(n_tr, n_cls, args.img_size, args.seed, "train"),
            SyntheticProbeDataset(n_va, n_cls, args.img_size, args.seed + 1, "val"),
            "synthetic(smoke)", True)


# ══════════════════════════ 编码器加载 / 特征提取 / 缓存 ══════════════════════════

def _ckpt_signature(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"


def load_frozen_encoder(ckpt_path: str, device: str) -> Tuple[nn.Module, LinAttnConfig, dict]:
    """载入 ckpt 的编码器（与 `eval_stepwise_linattn.load_model` 同口径, 容忍新增模块）。

    · `unexpected` key 一律报错（ckpt 与代码结构不符）;
    · `missing` 只允许 `decoder.mem_pos`（BPTT 臂新增, 零初始化）与 `refiner.*`
      （E-B3 GrowthBlock, 只用于训练, 推理解码不受影响）。
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in LinAttnConfig.__dataclass_fields__})
    model = SRPhase1LinAttn(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"ckpt 有多余 key: {list(unexpected)[:8]}"
    bad = [m for m in missing
           if m != "decoder.mem_pos" and not m.startswith("refiner.")]
    assert not bad, f"未加载参数: {bad[:8]}"
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, {"step": ck.get("step"), "epoch": ck.get("epoch"),
                        "missing_ok": list(missing), "cfg": cfg.to_dict()}


def make_random_smoke_model(args) -> Tuple[nn.Module, LinAttnConfig]:
    """`--smoke` 无 ckpt: 小随机 std 编码器（只验证接口/管线, 无科学含义）。"""
    cfg = LinAttnConfig(enc_attn="std", num_specials=int(args.smoke_K),
                        img_size=int(args.smoke_img_size), patch=int(args.smoke_patch),
                        enc_dim=int(args.smoke_dim), enc_depth=int(args.smoke_depth),
                        enc_heads=4, dec_dim=int(args.smoke_dim), dec_depth=2,
                        dec_heads=4, dec_attn="std", dec_recurrent=False)
    model = SRPhase1LinAttn(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def _meta_signature(ckpt_sig: str, args, tr_ds, va_ds, K: int, D: int) -> str:
    payload = {
        "ckpt": ckpt_sig, "K": K, "D": D, "img_size": args.img_size,
        "val_per_class": args.val_per_class, "val_n": args.val_n,
        "train_per_class": args.train_per_class, "train_aug": int(bool(args.train_aug)),
        "seed": args.seed,
        "n_train": len(tr_ds), "n_val": len(va_ds),
        "train_head": _ds_paths(tr_ds)[:5], "train_tail": _ds_paths(tr_ds)[-5:],
        "val_head": _ds_paths(va_ds)[:5], "val_tail": _ds_paths(va_ds)[-5:],
        "source": "synthetic" if isinstance(tr_ds, SyntheticProbeDataset) else "files",
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


@torch.no_grad()
def extract_split(ds, model, args, device: str, split: str, cache_file: str,
                  signature: str) -> dict:
    """冻结编码器一次前向 → z_s/z_cls 全量; 支持磁盘缓存（`--cache`）。

    返回 {"z_s": (N,K,D) float16/32, "z_cls": (N,1,D), "labels": (N,) int64,
          "paths": [str], "cache": "hit"/"miss"/"off"}。
    """
    if cache_file and os.path.exists(cache_file) and not args.refresh_cache:
        blob = torch.load(cache_file, map_location="cpu", weights_only=False)
        if blob.get("signature") == signature:
            print(f"[cache] {split}: HIT {cache_file} "
                  f"(N={blob['z_s'].shape[0]} K={blob['z_s'].shape[1]} "
                  f"D={blob['z_s'].shape[2]})")
            blob["cache"] = "hit"
            return blob
        print(f"[cache] {split}: 签名不匹配, 重算并覆盖 {cache_file}")
    dt = args.cache_dtype
    dtype = torch.float16 if dt == "fp16" else torch.float32
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(device == "cuda"),
        persistent_workers=bool(int(args.num_workers) > 0))
    zs_l, zc_l, y_l = [], [], []
    t0 = time.time()
    for bi, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(device == "cuda" and args.amp)):
            z_cls, z_s = model.encode(x)
        zs_l.append(z_s.detach().float().to(dtype))
        zc_l.append(z_cls.detach().float().to(dtype))
        y_l.append(y.detach().long())
        if (bi + 1) % 20 == 0 or bi == 0:
            print(f"    [{split}] {sum(z.shape[0] for z in zs_l)}/{len(ds)} 图 "
                  f"({time.time() - t0:.1f}s)", flush=True)
    z_s = torch.cat(zs_l, 0)
    z_cls = torch.cat(zc_l, 0)
    labels = torch.cat(y_l, 0)
    assert torch.isfinite(z_s.float()).all(), "z_s 出现非有限值"
    assert torch.isfinite(z_cls.float()).all(), "z_cls 出现非有限值"
    blob = {"z_s": z_s, "z_cls": z_cls, "labels": labels, "paths": _ds_paths(ds),
            "signature": signature, "cache": "off" if not cache_file else "miss"}
    print(f"[extract] {split}: {tuple(z_s.shape)} in {time.time() - t0:.1f}s "
          f"({dtype})")
    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        torch.save(blob, cache_file)
        print(f"[cache] {split}: SAVE {cache_file}")
    return blob


def prefix_mean(z_s: torch.Tensor, k: int, chunk: int = 1024) -> torch.Tensor:
    """mean-pool(z_s[:, :k]) → (N, D) float32（分块, 避免大中间张量）。"""
    K = int(z_s.shape[1])
    assert 1 <= k <= K, f"k={k} 越界 (K={K})"
    out = torch.empty(z_s.shape[0], z_s.shape[2], dtype=torch.float32)
    for s in range(0, z_s.shape[0], chunk):
        out[s:s + chunk] = z_s[s:s + chunk, :k].float().mean(dim=1)
    return out


# ══════════════════════════ 线性探针 ══════════════════════════

def train_probe(Xtr: torch.Tensor, ytr: torch.Tensor, n_classes: int, args,
                device: str) -> Tuple[dict, float, float]:
    """单层线性分类器（特征在 train 上标准化）。→ (probe, train_top1, final_loss)"""
    X = Xtr.to(device)
    y = ytr.to(device)
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sd
    if args.probe == "lstsq":
        Xb = torch.cat([Xn, torch.ones(Xn.shape[0], 1, device=device)], dim=1)
        Y = F.one_hot(y, n_classes).float()
        D1 = Xb.shape[1]
        A = Xb.t() @ Xb + float(args.ridge) * torch.eye(D1, device=device)
        W_full = torch.linalg.solve(A, Xb.t() @ Y)                # (D+1, C)
        W, b = W_full[:-1], W_full[-1]
        with torch.no_grad():
            logits = Xn @ W + b
            tr_acc = float((logits.argmax(1) == y).float().mean())
            loss = float(F.cross_entropy(logits, y))
    else:
        torch.manual_seed(int(args.seed))
        lin = nn.Linear(X.shape[1], n_classes).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=float(args.probe_lr),
                                weight_decay=float(args.probe_wd))
        steps = max(1, int(args.probe_steps))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps, eta_min=float(args.probe_lr) * 0.01)
        loss = float("nan")
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            logits = lin(Xn)
            loss_t = F.cross_entropy(logits, y)
            loss_t.backward()
            opt.step()
            sched.step()
            loss = float(loss_t.detach())
        with torch.no_grad():
            tr_acc = float((lin(Xn).argmax(1) == y).float().mean())
        # 统一成 (D, C) 口径（nn.Linear.weight 是 (out, in)）
        W = lin.weight.detach().t().contiguous()
        b = lin.bias.detach()
    return {"mu": mu, "sd": sd, "W": W, "b": b}, tr_acc, loss


@torch.no_grad()
def probe_correct(probe: dict, X: torch.Tensor, y: torch.Tensor, device: str,
                  chunk: int = 4096) -> np.ndarray:
    """→ (N,) bool, 是否 top-1 命中。"""
    out = []
    for s in range(0, X.shape[0], chunk):
        xb = (X[s:s + chunk].to(device) - probe["mu"]) / probe["sd"]
        pred = (xb @ probe["W"] + probe["b"]).argmax(1)
        out.append(pred.eq(y[s:s + chunk].to(device)))
    return torch.cat(out).cpu().numpy().astype(bool)


# ══════════════════════════ TSC json 读取（E-A2 → E-C1 契约） ══════════════════════════

def _match_tables(val_paths: Sequence[str]) -> Tuple[dict, dict]:
    by_path = {p: i for i, p in enumerate(val_paths)}
    by_name: Dict[str, int] = {}
    dup = set()
    for i, p in enumerate(val_paths):
        nm = os.path.basename(p)
        if nm in by_name:
            dup.add(nm)
        by_name[nm] = i
    for nm in dup:
        by_name.pop(nm, None)
    return by_path, by_name


def _default_taus(L1: np.ndarray, n_taus: int) -> List[float]:
    """与 `tools/alit_adaptive_k.default_taus` 同款: 末步 p10 → 首步 p50 几何网格。"""
    lo = float(np.percentile(L1[:, -1], 10))
    hi = float(np.percentile(L1[:, 0], 50))
    if not (0 < lo < hi):
        lo, hi = float(max(L1.min(), 1e-3)), float(max(L1.max(), 1e-2))
    if not (0 < lo < hi):
        lo, hi = 1e-3, 1.0
    return [float(x) for x in np.geomspace(lo, hi, max(2, int(n_taus)))]


def load_tsc_k_sets(path: str, val_paths: Sequence[str], args) -> Tuple[List[dict], dict]:
    """读 `--tsc_json`, 返回 (k_sets, interface_info)。

    k_sets: [{"label","tau","k": (n_val,) int (-1 = 未匹配), "matched": int}, ...]
    """
    with open(path) as f:
        d = json.load(f)
    n_val = len(val_paths)
    by_path, by_name = _match_tables(val_paths)
    info: dict = {"path": path, "form": None, "tau_grid": None, "match_key": {}}

    def _finish(k: np.ndarray, label: str, tau: Optional[float], form: str) -> dict:
        matched = int((k >= 0).sum())
        info["form"] = form
        return {"label": label, "tau": tau, "k": k, "matched": matched}

    # ── (b) 位置对齐列表 ──
    if isinstance(d, list):
        if len(d) != n_val:
            raise SystemExit(f"--tsc_json 是列表, 长度 {len(d)} != val {n_val}; "
                             f"位置对齐形式必须等长（或用规范 per_image 形式按 path 匹配）")
        return [_finish(np.asarray(d, dtype=np.int64), "positional", None, "k_list")], info
    if isinstance(d, dict) and "per_image" not in d and "images" not in d:
        for key in ("k", "k_list", "k_per_image", "tokens", "budgets"):
            if key in d and isinstance(d[key], list):
                if len(d[key]) != n_val:
                    raise SystemExit(f"--tsc_json['{key}'] 长度 {len(d[key])} != val {n_val}")
                return [_finish(np.asarray(d[key], dtype=np.int64), f"positional:{key}",
                                d.get("tau"), f"k_list({key})")], info

    # ── (c) E-A1 --dump_per_image（按 tau 现场推 k_i）──
    if isinstance(d, dict) and "images" in d:
        imgs = d["images"]
        cum = [int(x) for x in (d.get("cum_z_s") or [])]
        if not cum:
            raise SystemExit("stepwise dump 缺 cum_z_s, 无法把停步换成 token 数")
        if len(cum) != len(imgs[0]["L1_255"]):
            raise SystemExit(f"cum_z_s 长度 {len(cum)} != 步数 {len(imgs[0]['L1_255'])}")
        L1_all = np.asarray([im["L1_255"] for im in imgs], dtype=np.float64)
        # 每图 → val 下标
        idxs = np.full(len(imgs), -1, dtype=np.int64)
        for j, im in enumerate(imgs):
            for key, table in (("path", by_path), ("file_name", by_name)):
                v = im.get(key)
                if v and v in table:
                    idxs[j] = table[v]
                    break
            if idxs[j] < 0 and "index" in im:
                gi = int(im["index"])
                if 0 <= gi < n_val:
                    idxs[j] = gi
        # 阈值优先级: --tsc_tau >= 0（单点） > --tsc_taus 列表 > "auto" 几何网格
        if float(args.tsc_tau) >= 0.0:
            taus = [float(args.tsc_tau)]
        else:
            spec = str(args.tsc_taus).strip().lower()
            if spec in ("", "auto"):
                taus = _default_taus(L1_all, int(args.tsc_n_taus))
                info["tau_grid"] = "auto(末步p10→首步p50)"
            else:
                taus = [float(x) for x in spec.split(",") if x.strip()]
        info["n_dump_images"] = len(imgs)
        info["stepwise_ckpt"] = d.get("ckpt")
        info["cum_z_s"] = cum
        info["match_key"] = {"path_or_filename_or_index": int((idxs >= 0).sum())}
        k_sets = []
        for tau in taus:
            ok = L1_all <= float(tau)
            any_ok = ok.any(1)
            t_idx = np.where(any_ok, ok.argmax(1), L1_all.shape[1] - 1)
            k_dump = np.asarray([cum[int(t)] for t in t_idx], dtype=np.int64)
            k = np.full(n_val, -1, dtype=np.int64)
            k[idxs[idxs >= 0]] = k_dump[idxs >= 0]
            k_sets.append(_finish(k, f"tsc_tau={tau:.4f}", float(tau), "stepwise_dump"))
        info["form"] = "stepwise_dump(E-A1 --dump_per_image)"
        info["tau_grid"] = taus
        return k_sets, info

    # ── (a) 规范 per_image ──
    per = d.get("per_image") if isinstance(d, dict) else None
    if per is None:
        raise SystemExit(f"无法识别的 --tsc_json 形态: {path}")
    k = np.full(n_val, -1, dtype=np.int64)
    n_key = {"path": 0, "filename": 0, "index": 0}
    for rec in per:
        ki = rec.get("k", rec.get("tokens", rec.get("num_tokens")))
        if ki is None:
            continue
        gi = -1
        if rec.get("path") and rec["path"] in by_path:
            gi, key = by_path[rec["path"]], "path"
        elif rec.get("filename") and rec["filename"] in by_name:
            gi, key = by_name[rec["filename"]], "filename"
        elif "index" in rec and 0 <= int(rec["index"]) < n_val:
            gi, key = int(rec["index"]), "index"
        else:
            continue
        k[gi] = int(ki)
        n_key[key] += 1
    info["match_key"] = n_key
    info["k_grid"] = d.get("k_grid")
    info["cum_z_s"] = d.get("cum_z_s")
    info["source"] = d.get("source")
    return [_finish(k, "per_image", d.get("tau"), "per_image")], info


# ══════════════════════════ bootstrap / 判据 ══════════════════════════

def _bootstrap_counts(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n, B) 每列 = 一次 paired 重采样的出现次数（供 C @ counts 快速求均值）。"""
    rng = np.random.default_rng(int(seed))
    counts = np.zeros((n, int(n_boot)), dtype=np.float32)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        counts[:, b] = np.bincount(idx, minlength=n)
    return counts


def evaluate(ks: Sequence[int], C_fixed: np.ndarray, k_sets: List[dict], n_boot: int,
             seed: int, ci_level: float, min_effect_pp: float) -> dict:
    """等平均 token 数对照 + paired bootstrap CI。

    C_fixed: (n_k, n_eval) 0/1; k_sets: 每图 k ∈ ks（已对齐到同一 eval 子集）。
    """
    ks_arr = np.asarray(ks, dtype=np.float64)
    n = C_fixed.shape[1]
    counts = _bootstrap_counts(n, n_boot, seed)
    fixed_curve = C_fixed.mean(axis=1)
    fixed_b = (C_fixed.astype(np.float64) @ counts) / n              # (n_k, B)
    lo = np.percentile(fixed_b, (1 - ci_level) / 2 * 100, axis=1)
    hi = np.percentile(fixed_b, (1 + ci_level) / 2 * 100, axis=1)

    rows = []
    for s in k_sets:
        k_i = np.asarray(s["k"], dtype=np.int64)
        gi = np.searchsorted(np.asarray(ks, dtype=np.int64), k_i)
        assert np.all(np.asarray(ks, dtype=np.int64)[gi] == k_i), "k_i 不在 --ks 网格内"
        ca = C_fixed[gi, np.arange(n)].astype(np.float64)
        acc = float(ca.mean())
        acc_b = (ca @ counts) / n
        acc_lo, acc_hi = _ci(acc_b, ci_level)
        kbar = float(k_i.mean())
        kbar_b = (k_i.astype(np.float64) @ counts) / n
        fixed_b_at = np.asarray([np.interp(kbar_b[b], ks_arr, fixed_b[:, b])
                                 for b in range(counts.shape[1])])
        delta_b = acc_b - fixed_b_at
        d_lo, d_hi = _ci(delta_b, ci_level)
        fixed_point = float(np.interp(kbar, ks_arr, fixed_curve))
        j_near = int(np.argmin(np.abs(ks_arr - kbar)))
        delta_pp = (acc - fixed_point) * 100.0
        verdict = ("adaptive_better"
                   if (delta_pp >= min_effect_pp and d_lo * 100 > 0) else
                   "fixed_better"
                   if (delta_pp <= -min_effect_pp and d_hi * 100 < 0) else
                   "indistinguishable")
        rows.append({
            "label": s["label"], "tau": s.get("tau"), "n": int(n),
            "mean_k": kbar, "median_k": float(np.median(k_i)),
            "std_k": float(k_i.std()),
            "adaptive_top1": acc, "adaptive_top1_ci": [acc_lo, acc_hi],
            "fixed_top1_at_equal_budget": fixed_point,
            "fixed_top1_at_equal_budget_ci": [
                float(np.interp(kbar, ks_arr, lo)),
                float(np.interp(kbar, ks_arr, hi))],
            "fixed_k_nearest_grid": float(ks_arr[j_near]),
            "fixed_top1_nearest_grid": float(fixed_curve[j_near]),
            "delta_pp": delta_pp,
            "delta_pp_ci": [d_lo * 100.0, d_hi * 100.0],
            "verdict": verdict,
        })
    return {
        "fixed_curve": [
            {"k": int(ks[i]), "top1": float(fixed_curve[i]),
             "top1_ci": [float(lo[i]), float(hi[i])]}
            for i in range(len(ks))],
        "adaptive_curve": rows,
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed),
                      "ci_level": float(ci_level), "paired": True,
                      "note": "每样本内重算 mean(k_i) 再对该固定档曲线插值 ⇒ 预算是配对的"},
        "min_effect_pp": float(min_effect_pp),
    }


# ══════════════════════════ 画图 ══════════════════════════

def make_plot(png_path: str, fixed_curve: List[dict], rows: List[dict],
              k_i_primary: np.ndarray, K: int, chance: float, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                        # noqa: BLE001
        print(f"[warn] matplotlib 不可用, 跳过 png: {e}")
        return False
    ks = [r["k"] for r in fixed_curve]
    top = np.asarray([r["top1"] for r in fixed_curve]) * 100.0
    lo = np.asarray([r["top1_ci"][0] for r in fixed_curve]) * 100.0
    hi = np.asarray([r["top1_ci"][1] for r in fixed_curve]) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    ax.fill_between(ks, lo, hi, color="tab:blue", alpha=0.15,
                    label="fixed-K probe 95% CI")
    ax.plot(ks, top, "-o", ms=3.5, color="tab:blue", label="fixed-K probe")
    for r in rows:
        ax.errorbar([r["mean_k"]], [r["adaptive_top1"] * 100.0],
                    yerr=[[(r["adaptive_top1"] - r["adaptive_top1_ci"][0]) * 100.0],
                          [(r["adaptive_top1_ci"][1] - r["adaptive_top1"]) * 100.0]],
                    fmt="*", ms=11, capsize=3, color="tab:red",
                    label=f"adaptive (tau={r['tau']:.2f})" if r["tau"] is not None
                    else "adaptive")
        ax.plot([r["mean_k"], r["mean_k"]],
                [r["fixed_top1_at_equal_budget"] * 100.0, r["adaptive_top1"] * 100.0],
                ":", color="gray", lw=0.8)
    if rows:
        p = rows[0]
        ax.annotate(f"d={p['delta_pp']:+.2f} pp\nCI[{p['delta_pp_ci'][0]:+.2f},"
                    f"{p['delta_pp_ci'][1]:+.2f}]",
                    xy=(p["mean_k"], (p["adaptive_top1"] + p["fixed_top1_at_equal_budget"])
                        / 2 * 100.0),
                    xytext=(8, -14), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.6"))
    ax.axhline(chance * 100.0, ls=":", color="0.4", lw=1, label=f"chance ({chance*100:.1f}%)")
    ax.set_xlabel("z_s prefix budget k (tokens)")
    ax.set_ylabel("IN-100 val top-1 (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax2 = axes[1]
    ax2.hist(np.asarray(k_i_primary), bins=np.arange(0.5, K + 1.5, 1.0),
             color="tab:red", alpha=0.75)
    ax2.axvline(float(np.mean(k_i_primary)), color="k", ls="--", lw=1,
                label=f"mean k={float(np.mean(k_i_primary)):.1f}")
    ax2.axvline(K, color="tab:blue", ls=":", lw=1, label=f"K={K}")
    ax2.set_xlabel("TSC-selected k per image"); ax2.set_ylabel("images")
    ax2.set_title("per-image TSC budget (primary policy)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7)
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[png] {png_path}")
    return True


# ══════════════════════════ main ══════════════════════════

def _apply_smoke_defaults(args):
    """`--smoke`: 只在参数仍是默认值时收紧（用户显式给的参数优先）。"""
    if not args.smoke:
        return
    if args.ks == DEFAULT_KS:
        args.ks = SMOKE_KS
    smoke_over = {"n_boot": 200, "probe_steps": 150, "batch_size": 32, "num_workers": 0}
    for name, default in (("n_boot", 1000), ("probe_steps", 600), ("batch_size", 64),
                          ("num_workers", 8)):
        if getattr(args, name) == default:
            setattr(args, name, smoke_over[name])
    if not args.ckpt:
        args.img_size = int(args.smoke_img_size)
        print(f"[smoke] 无 --ckpt: 随机 std 编码器 "
              f"(dim={args.smoke_dim}, depth={args.smoke_depth}, K={args.smoke_K}, "
              f"img={args.smoke_img_size}) — 仅管线证据, 无科学含义")


def main():
    ap = argparse.ArgumentParser(
        description="E-C1 任务保真分类探针: 自适应 token 预算 vs 等平均 token 数固定档 "
                    "(IN-100 top-1, 冻结编码器 + 线性探针)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)
    # ── 模型 / 数据 ──
    ap.add_argument("--ckpt", default="",
                    help="冻结编码器 ckpt (默认空: 仅 --smoke 可用, 随机小编码器)")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100",
                    help="ImageNet-100 根目录（100 个 synset 子目录）")
    ap.add_argument("--val_per_class", type=int, default=50,
                    help="与 data_linattn 一致的确定性 val 划分参数")
    ap.add_argument("--val_n", type=int, default=0,
                    help="val 等距子采样张数（0 = 全部; 正式跑必须 0 以对齐 TSC json）")
    ap.add_argument("--train_per_class", type=int, default=100,
                    help="探针训练集: 每类张数（默认 100/类 = 10,000）")
    ap.add_argument("--img_size", type=int, default=224,
                    help="输入分辨率（默认 224; 须被 cfg.patch 整除）")
    ap.add_argument("--train_aug", type=int, default=0,
                    help="探针训练特征是否用 train 增强变换（默认 0 = 与 val 同口径）")
    ap.add_argument("--ks", default=DEFAULT_KS,
                    help="固定档 token 预算网格（逗号分隔; 自适应 k_i 会自动并入）")
    ap.add_argument("--use_cls", type=int, default=0,
                    help="特征是否拼接 z_cls（默认 0 = 纯 mean-pool(z_s[:k]), 只测预算效应）")
    # ── 提取 ──
    ap.add_argument("--batch_size", type=int, default=64, help="编码器前向 batch（显存友好）")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1, help="cuda 上 bf16 autocast（cpu 自动关）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--cache", default="",
                    help="特征缓存前缀 → <cache>.train.pt / <cache>.val.pt（避免重复前向）")
    ap.add_argument("--cache_dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--refresh_cache", type=int, default=0, help="1 = 忽略已有缓存重算")
    ap.add_argument("--extract_only", type=int, default=0, help="1 = 只建缓存后退出")
    # ── 探针 ──
    ap.add_argument("--probe", default="logreg", choices=["logreg", "lstsq"],
                    help="logreg = 单层 Linear 全批 Adam; lstsq = ridge 闭式")
    ap.add_argument("--probe_steps", type=int, default=600)
    ap.add_argument("--probe_lr", type=float, default=0.01)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--ridge", type=float, default=1e-2)
    # ── TSC（E-A2 接口）──
    ap.add_argument("--tsc_json", default="",
                    help="E-A2 per-image 停步 json（规范 per_image / k 列表 / "
                         "E-A1 --dump_per_image 三种形态皆可）")
    ap.add_argument("--tsc_tau", type=float, default=-1.0,
                    help="--tsc_json 是 E-A1 dump 时的单点阈值（L1_255）; >=0 时优先于 --tsc_taus")
    ap.add_argument("--tsc_taus", default="auto",
                    help="'auto' = 同 alit_adaptive_k 的几何网格扫一族人; 或逗号分隔阈值")
    ap.add_argument("--tsc_n_taus", type=int, default=9)
    ap.add_argument("--allow_partial", type=int, default=0,
                    help="1 = 允许 TSC json 只覆盖部分 val（自适应/固定都限制在同一子集）")
    # ── 统计 / 输出 ──
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_effect_pp", type=float, default=0.5,
                    help="判据说 '有差' 的最小效应（百分点）")
    ap.add_argument("--primary_tau_frac", type=float, default=0.6,
                    help="主策略 = mean_k/K 最接近该比例的 tau 行（ALIT ≈60%% 预算）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--png", default="")
    # ── smoke ──
    ap.add_argument("--smoke", type=int, default=0,
                    help="本地 CPU 冒烟: 小样本/少档位; 无 --ckpt 时用随机小编码器 + 合成数据")
    ap.add_argument("--smoke_n_classes", type=int, default=8)
    ap.add_argument("--smoke_n_train", type=int, default=320)
    ap.add_argument("--smoke_n_val", type=int, default=200)
    ap.add_argument("--smoke_K", type=int, default=16)
    ap.add_argument("--smoke_dim", type=int, default=64)
    ap.add_argument("--smoke_depth", type=int, default=3)
    ap.add_argument("--smoke_img_size", type=int, default=64)
    ap.add_argument("--smoke_patch", type=int, default=8)
    ap.add_argument("--smoke_tsc", type=int, default=1,
                    help="1 = 合成一份规范形态 --tsc_json 供本次 smoke 使用")
    ap.add_argument("--smoke_force_synthetic", type=int, default=0)
    args = ap.parse_args()
    _apply_smoke_defaults(args)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_png = args.png or (os.path.splitext(args.out)[0] + ".png")

    # ── 1. 模型 ──
    if args.ckpt:
        model, cfg, ck_meta = load_frozen_encoder(args.ckpt, device)
        ckpt_sig = _ckpt_signature(args.ckpt)
    else:
        if not args.smoke:
            raise SystemExit("必须给 --ckpt（除非 --smoke）")
        model, cfg = make_random_smoke_model(args)
        ck_meta = {"step": None, "epoch": None, "missing_ok": [], "cfg": cfg.to_dict()}
        ckpt_sig = f"random(smoke,dim={args.smoke_dim},depth={args.smoke_depth})"
    model = model.to(device).eval()
    K = int(cfg.num_specials)
    print(f"[model] enc_attn={cfg.enc_attn} K={K} N={cfg.num_patches} "
          f"dim={cfg.enc_dim}  img_size={cfg.img_size}  device={device}")

    # 输入分辨率: 真实 ckpt 时以 cfg 为准（编码器 pos_embed 可插值, 但默认同口径）
    if args.ckpt and args.img_size != int(cfg.img_size):
        print(f"[warn] --img_size={args.img_size} != cfg.img_size={cfg.img_size}; "
              f"编码器会对 pos 做线性插值, 与训练口径不同（仅用于敏感性测试）")

    # ── 2. 数据 ──
    tr_ds, va_ds, source, synthetic = build_probe_datasets(args)
    val_paths = _ds_paths(va_ds)
    print(f"[data] source={source} train={len(tr_ds)} val={len(va_ds)} "
          f"(val_per_class={args.val_per_class}, train_per_class={args.train_per_class})")

    # ── 3. 特征（冻结前向 + 缓存）──
    sig = _meta_signature(ckpt_sig, args, tr_ds, va_ds, K, int(cfg.enc_dim))
    cache_tr = f"{args.cache}.train.pt" if args.cache else ""
    cache_va = f"{args.cache}.val.pt" if args.cache else ""
    t0 = time.time()
    blob_tr = extract_split(tr_ds, model, args, device, "train", cache_tr, sig)
    blob_va = extract_split(va_ds, model, args, device, "val", cache_va, sig)
    sec_extract = time.time() - t0
    y_tr = blob_tr["labels"].long()
    y_va = blob_va["labels"].long()
    n_classes = int(max(int(y_tr.max()) + 1, int(y_va.max()) + 1))
    assert args.use_cls in (0, 1)
    if args.use_cls:
        def feats(blob, k, chunk=1024):
            zs = blob["z_s"]
            out = torch.empty(zs.shape[0], zs.shape[2] * 2, dtype=torch.float32)
            for s in range(0, zs.shape[0], chunk):
                out[s:s + chunk] = torch.cat(
                    [prefix_mean(zs[s:s + chunk], k, chunk=chunk),
                     blob["z_cls"][s:s + chunk].float().squeeze(1)], dim=1)
            return out
    else:
        feats = lambda blob, k, chunk=1024: prefix_mean(blob["z_s"], k, chunk=chunk)  # noqa: E731

    if args.extract_only:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"meta": {"extract_only": True, "ckpt": args.ckpt,
                                "K": K, "n_train": len(tr_ds), "n_val": len(va_ds),
                                "signature": sig, "cache": args.cache,
                                "sec_extract": sec_extract}, "honesty": HONESTY},
                      f, ensure_ascii=False, indent=1)
        print(f"[out] {args.out} (extract_only)")
        return

    # ── 4. TSC k_i（E-A2 接口）──
    k_sets, tsc_info = [], {"path": None, "form": None}
    eval_idx = np.arange(len(va_ds))
    if args.smoke and args.smoke_tsc and not args.tsc_json:
        # smoke: 合成一份**规范形态** --tsc_json（接口证据; 无科学含义）
        k_pool = _csv_ints(args.ks)
        rng = np.random.default_rng(int(args.seed))
        w = np.linspace(1.6, 0.6, len(k_pool))
        w = w / w.sum()
        k_vals = rng.choice(np.asarray(k_pool), size=len(va_ds), p=w).astype(np.int64)
        per = [{"index": int(i), "filename": os.path.basename(val_paths[i]),
                "path": val_paths[i], "k": int(k_vals[i]),
                "step": None, "source": "SMOKE synthetic (no science)"}
               for i in range(len(va_ds))]
        smoke_tsc = os.path.join(os.path.dirname(args.out) or ".",
                                 "smoke_tsc_input.json")
        os.makedirs(os.path.dirname(smoke_tsc) or ".", exist_ok=True)
        with open(smoke_tsc, "w") as f:
            json.dump({"per_image": per, "k_grid": sorted(k_pool), "tau": None,
                       "source": "SMOKE synthetic per_image (E-C1 --tsc_json 规范形态样例)"},
                      f, ensure_ascii=False, indent=1)
        args.tsc_json = smoke_tsc
        print(f"[smoke] 合成 --tsc_json（规范形态）: {smoke_tsc}  "
              f"mean_k={k_vals.mean():.2f}")
    if args.tsc_json:
        raw_sets, tsc_info = load_tsc_k_sets(args.tsc_json, val_paths, args)
        cov = [int(s["matched"]) for s in raw_sets]
        n_cov = min(cov) if cov else 0
        if n_cov < len(va_ds):
            msg = (f"--tsc_json 只覆盖 {cov} / {len(va_ds)} 张 val")
            if not args.allow_partial:
                raise SystemExit(msg + "（加 --allow_partial 1 可把自适应与固定档同时限制"
                                        "在该子集上比较）")
            print(f"[warn] {msg} → 限制到该子集")
        keep = np.ones(len(va_ds), dtype=bool)
        for s in raw_sets:
            keep &= (s["k"] >= 0)
        eval_idx = np.where(keep)[0]
        if eval_idx.size == 0:
            raise SystemExit("--tsc_json 与 val 无任何可匹配项（检查 path/filename/index 口径）")
        k_sets = [{"label": s["label"], "tau": s["tau"], "k": s["k"][eval_idx]}
                  for s in raw_sets]
        print(f"[tsc] {tsc_info['form']}  → eval 子集 {eval_idx.size} 张; "
              f"匹配键 {tsc_info.get('match_key')}")

    # ── 5. 预算网格（并入 TSC 的 k_i）+ 逐档探针 ──
    ks = sorted({k for k in _csv_ints(args.ks) if 1 <= k <= K})
    ks = sorted(set(ks) | {int(v) for s in k_sets for v in np.unique(s["k"])})
    assert ks, "预算网格为空"
    if args.ks != DEFAULT_KS and any(k > K for k in _csv_ints(args.ks)):
        print(f"[warn] --ks 中 > K={K} 的档被丢弃")
    print(f"[probe] {len(ks)} 档: {ks}  features={'mean(z_s[:k])+z_cls' if args.use_cls else 'mean(z_s[:k])'}")
    X_va = {k: feats(blob_va, k)[eval_idx] for k in ks}
    y_eval = y_va[eval_idx]
    C_fixed = np.zeros((len(ks), eval_idx.size), dtype=np.float32)
    curve, t_probe0 = [], time.time()
    for j, k in enumerate(ks):
        Xtr = feats(blob_tr, k)
        probe, tr_acc, loss = train_probe(Xtr, y_tr, n_classes, args, device)
        del Xtr
        ok = probe_correct(probe, X_va[k], y_eval, device)
        C_fixed[j] = ok.astype(np.float32)
        curve.append({"k": int(k), "top1": float(ok.mean()),
                      "n_correct": int(ok.sum()), "n": int(ok.size),
                      "train_top1": float(tr_acc), "final_loss": float(loss),
                      "feat_dim": int(probe["W"].shape[0])})
        print(f"  k={k:3d}  val_top1={ok.mean()*100:6.2f}%  train_top1={tr_acc*100:6.2f}%  "
              f"loss={loss:.4f}", flush=True)
    sec_probe = time.time() - t_probe0

    # ── 6. 等预算对照 + bootstrap（无 TSC 时也给出逐档 CI）──
    ev = evaluate(ks, C_fixed, k_sets, int(args.n_boot), int(args.seed),
                  float(args.ci), float(args.min_effect_pp))
    if not k_sets:
        ev["note"] = "未给 --tsc_json: 只报固定档曲线（含 bootstrap CI）, 无自适应对照"
    # 逐档 CI 与 probe 训练信息合并
    ci_map = {r["k"]: r for r in ev["fixed_curve"]}
    for row in curve:
        row["top1_ci"] = ci_map[row["k"]]["top1_ci"]
    ev["fixed_curve"] = curve

    # ── 7. oracle 上界（TSC-TOI 对齐 gap 的量化, 仅上界）──
    any_ok = C_fixed.any(axis=0)
    ks_arr = np.asarray(ks)
    min_k_correct = ks_arr[C_fixed.argmax(axis=0)]
    oracle = {
        "any_k_correct_rate": float(any_ok.mean()),
        "mean_min_k_all_images": float(np.where(any_ok, min_k_correct, K).mean()),
        "mean_min_k_correct_only": (float(min_k_correct[any_ok].mean())
                                    if bool(any_ok.any()) else None),
        "frac_images_needing_K": float((~any_ok).mean()),
        "chance_top1": 1.0 / float(n_classes),
        "note": "上界: 知道标签时每图最小需要多少 token; 不可实现, 只量化 TSC-TOI gap",
    }

    # ── 8. 主策略行 + 总体判据 ──
    primary = None
    if k_sets:
        target = float(args.primary_tau_frac) * K
        primary = min(k_sets, key=lambda s: abs(float(s["k"].mean()) - target))
    e_c1 = {}
    if ev["adaptive_curve"]:
        rows = ev["adaptive_curve"]
        prim_row = next(r for r in rows if r["label"] == primary["label"]) if primary else rows[0]
        e_c1 = {
            "primary_label": primary["label"] if primary else None,
            "primary_mean_k": prim_row["mean_k"],
            "primary_mean_k_over_K": prim_row["mean_k"] / float(K),
            "primary_adaptive_top1": prim_row["adaptive_top1"],
            "primary_fixed_top1_at_equal_budget": prim_row["fixed_top1_at_equal_budget"],
            "primary_delta_pp": prim_row["delta_pp"],
            "primary_delta_pp_ci": prim_row["delta_pp_ci"],
            "primary_verdict": prim_row["verdict"],
            "max_delta_pp_over_sweep": float(max(r["delta_pp"] for r in rows)),
            "min_delta_pp_over_sweep": float(min(r["delta_pp"] for r in rows)),
            "any_row_adaptive_better": any(r["verdict"] == "adaptive_better" for r in rows),
            "any_row_fixed_better": any(r["verdict"] == "fixed_better" for r in rows),
            "decision_rule": (f"行判据: delta_pp >= +{args.min_effect_pp} 且 CI 下界 >0 ⇒ "
                              f"adaptive_better; <= -{args.min_effect_pp} 且 CI 上界 <0 ⇒ "
                              f"fixed_better; 否则 indistinguishable（PLAN §8 E-C1: "
                              f"等平均 token 数下自适应 ≥ 固定 K 才算成立）"),
            "caveat": "分类线性探针 ≠ Phase 2 生成任务; 单种子; 见 honesty",
        }

    # ── 9. 落盘 ──
    meta = {
        "tool": "tools/alit_task_fidelity.py", "task": "E-C1 (§5) / task-5",
        "ckpt": args.ckpt or None, "ckpt_signature": ckpt_sig,
        "ckpt_meta": {"step": ck_meta.get("step"), "epoch": ck_meta.get("epoch"),
                      "missing_ok": ck_meta.get("missing_ok")},
        "cfg": {k: v for k, v in ck_meta["cfg"].items()
                if k in ("enc_attn", "enc_dim", "enc_depth", "enc_heads", "num_specials",
                         "img_size", "patch", "dec_recurrent", "dec_dim", "dec_depth")},
        "K": K, "N": int(cfg.num_patches), "feature_dim": int(cfg.enc_dim) * (2 if args.use_cls else 1),
        "data_dir": args.data_dir, "data_source": source, "synthetic": bool(synthetic),
        "split": {"val_per_class": args.val_per_class, "n_train": len(tr_ds),
                  "n_val": len(va_ds), "n_eval": int(eval_idx.size),
                  "train_per_class": args.train_per_class, "val_n": args.val_n},
        "feature": {"pool": "mean-prefix", "use_cls": bool(args.use_cls),
                    "train_aug": bool(args.train_aug),
                    "note": "预算 k 的特征 = mean-pool(z_s[:, :k])（+z_cls 可选）"},
        "probe": {"kind": args.probe, "steps": args.probe_steps, "lr": args.probe_lr,
                  "wd": args.probe_wd, "ridge": args.ridge,
                  "standardize": "train per-dim mean/std"},
        "ks": ks, "n_classes": n_classes, "chance_top1": 1.0 / n_classes,
        "tsc": tsc_info, "tsc_source_k_sets": [s["label"] for s in k_sets],
        "device": device, "batch_size": args.batch_size, "amp": bool(args.amp),
        "cache": args.cache or None,
        "cache_status": {"train": blob_tr.get("cache"), "val": blob_va.get("cache")},
        "seed": args.seed, "n_boot": args.n_boot, "ci": args.ci,
        "min_effect_pp": args.min_effect_pp, "primary_tau_frac": args.primary_tau_frac,
        "sec_extract": sec_extract, "sec_probe": sec_probe,
        "wall_sec": time.time() - t0,
    }
    res = {"meta": meta, "top1_by_k": curve,
           "adaptive_curve": ev["adaptive_curve"],
           "bootstrap": ev.get("bootstrap"), "oracle_upper_bound": oracle,
           "E_C1_verdict": e_c1, "honesty": HONESTY,
           "tsc_interface": {
               "canonical": '{"per_image":[{"index":i,"filename":str,"path":str,"k":int,'
                            '"step":int}],"k_grid":[int],"cum_z_s":[int],"tau":float}',
               "k_list": "[k0, k1, ...] 与 val 顺序一一对应（或 {\"k\":[...]}）",
               "eadump": "E-A1 --dump_per_image json + --tsc_tau/--tsc_taus（本工具现场推 k_i）",
               "match_priority": "path > filename > index(val 划分下标)",
               "k_semantics": "k = 该图读入的 z_s 前缀长度; step → k 用 cum_z_s[step]"},
           }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ── 10. 画图 + 终端摘要 ──
    make_plot(out_png, curve, ev["adaptive_curve"],
              (np.asarray(primary["k"]) if primary is not None else np.array([K])),
              K, 1.0 / n_classes,
              "E-C1 task-fidelity probe" + (" [SMOKE]" if args.smoke else ""))

    print("\n=== E-C1 任务保真分类探针（等平均 token 数口径）===")
    print(f"ckpt  : {args.ckpt or '(random smoke)'}")
    print(f"data  : {source}  train={len(tr_ds)}  val={len(va_ds)}  eval={eval_idx.size}"
          f"  classes={n_classes}  chance={100.0/n_classes:.2f}%")
    print("k     : " + "  ".join(f"{r['k']}:{r['top1']*100:.1f}%"
                                 f"[{r['top1_ci'][0]*100:.1f},{r['top1_ci'][1]*100:.1f}]"
                                 for r in curve))
    for r in ev["adaptive_curve"]:
        print(f"tau={r['tau'] if r['tau'] is not None else float('nan'):7.3f}  "
              f"mean_k={r['mean_k']:5.2f} (K={K}, {r['mean_k']/K*100:4.1f}%)  "
              f"adaptive={r['adaptive_top1']*100:6.2f}%  "
              f"CIfix@budget={r['fixed_top1_at_equal_budget']*100:6.2f}%  "
              f"Δ={r['delta_pp']:+6.2f}pp [{r['delta_pp_ci'][0]:+.2f},{r['delta_pp_ci'][1]:+.2f}]"
              f"  → {r['verdict']}")
    if e_c1:
        print(f"primary: {e_c1['primary_label']}  mean_k/K={e_c1['primary_mean_k_over_K']:.3f}"
              f"  Δ={e_c1['primary_delta_pp']:+.2f}pp  → {e_c1['primary_verdict']}")
        print(f"oracle : any_k_correct={oracle['any_k_correct_rate']*100:.1f}%  "
              f"mean_min_k={oracle['mean_min_k_all_images']:.1f}(all)/"
              f"{oracle['mean_min_k_correct_only']} (只算可命中)  （上界, 不可实现）")
    print(f"[out] {args.out}")
    print("[honesty] 分类线性探针 ≠ Phase 2 生成任务; 等平均 token 数口径已配对; "
          "单种子 + bootstrap CI 才可谈方向。")


if __name__ == "__main__":
    main()
