"""
ImageNet-100 数据管线 — AssocVTC 实验版 (linattn)
=======================================================

数据集: `/root/autodl-tmp/datasets/imagenet100/<synset>/*.JPEG`（**直接**是 100 个
    synset 目录, 没有多余一层）。
    AutoDL 公共盘 `/root/autodl-pub/ImageNet100/imagenet100.zip` 内的顶层是
    `imagenet100/`, 解压到 `/root/autodl-tmp/datasets/` 后即为上面的布局。
    100 类 × ~1267 张 = 126,689 张**无官方 val 划分**（只有 train 全部图像）。

划分 (确定性, 可复现, 无随机性):
    每类按**文件名排序**后, 以等距步长抽取 `val_per_class` 张作验证集, 其余训练。
    ⇒ train = 126,689 − 100×val_per_class; 默认 val_per_class=50 ⇒ val=5000。
    与类别顺序、磁盘顺序无关。

变换:
    train: RandomResizedCrop(224, scale=(0.35,1.0)) + RandomHorizontalFlip + Normalize
    val:   Resize(256, BICUBIC) + CenterCrop(224) + Normalize
    —— 归一化统一用 `model_v2_linattn.IMAGENET_MEAN/STD`（像素目标同一口径）。
"""

import json
import os
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from model_v2_linattn import IMAGENET_MEAN, IMAGENET_STD


def scan_imagenet100(root: str) -> List[Tuple[str, int]]:
    """返回 [(图片绝对路径, 类别索引)], 类别按 synset 名排序（索引稳定）。"""
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    assert len(classes) == 100, f"{root} 下类别数 = {len(classes)}, 期望 100"
    items: List[Tuple[str, int]] = []
    for ci, c in enumerate(classes):
        cdir = os.path.join(root, c)
        files = sorted(f for f in os.listdir(cdir)
                       if f.lower().endswith((".jpeg", ".jpg", ".png")))
        assert files, f"{cdir} 为空"
        for f in files:
            items.append((os.path.join(cdir, f), ci))
    return items


def split_index(items: List[Tuple[str, int]], val_per_class: int = 50
                ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """按类内等距步长划分 train/val（确定性）。"""
    by_cls: dict = {}
    for p, c in items:
        by_cls.setdefault(c, []).append((p, c))
    train, val = [], []
    for c in sorted(by_cls):
        lst = by_cls[c]
        n = len(lst)
        k = min(val_per_class, max(0, n - 1))
        if k <= 0:
            train.extend(lst)
            continue
        step = max(1, n // k)
        val_idx = set(range(0, n, step))
        # 等距抽取可能超出 k 个 ⇒ 再均匀下采样到恰好 k 个
        ordered = sorted(val_idx)
        if len(ordered) > k:
            keep = [ordered[int(round(i * (len(ordered) - 1) / (k - 1)))]
                    for i in range(k)] if k > 1 else [ordered[0]]
            val_idx = set(keep)
        for i, x in enumerate(lst):
            (val if i in val_idx else train).append(x)
    return train, val


class ImageNet100(Dataset):
    def __init__(self, items: Sequence[Tuple[str, int]], transform, img_size: int = 224):
        self.items = list(items)
        self.transform = transform
        self.img_size = img_size
        if len(self.items) == 0:
            raise ValueError("空数据集")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), label


def build_transform(img_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3),
                interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_datasets(data_dir: str, img_size: int = 224, val_per_class: int = 50
                   ) -> Tuple[ImageNet100, ImageNet100]:
    items = scan_imagenet100(data_dir)
    tr, va = split_index(items, val_per_class)
    return (ImageNet100(tr, build_transform(img_size, True), img_size),
            ImageNet100(va, build_transform(img_size, False), img_size))


def build_loaders(data_dir: str, img_size: int = 224, batch_size: int = 64,
                  val_per_class: int = 50, num_workers: int = 16,
                  limit: int = 0, val_batches: int = 0
                  ) -> Tuple[DataLoader, DataLoader, ImageNet100, ImageNet100]:
    tr_ds, va_ds = build_datasets(data_dir, img_size, val_per_class)
    if limit > 0:
        tr_ds = torch.utils.data.Subset(tr_ds, range(min(limit, len(tr_ds))))
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=num_workers > 0,
                    prefetch_factor=4 if num_workers > 0 else None)
    n_val = len(va_ds)
    if val_batches > 0:
        n_val = min(n_val, val_batches * batch_size)
    # 等距抽样（不是取前缀）: 取前缀会退化成"只在前若干个类上验证",
    # 验证指标被类别分布带偏（2026-10-15 独立审查 finding 13）。
    stride = max(1, len(va_ds) // max(1, n_val))
    idx = list(range(0, len(va_ds), stride))[:n_val]
    va_sub = torch.utils.data.Subset(va_ds, idx)
    n_va_workers = 0 if num_workers <= 0 else max(2, num_workers // 4)
    va = DataLoader(va_sub, batch_size=batch_size, shuffle=False, drop_last=False,
                    num_workers=n_va_workers, pin_memory=True,
                    persistent_workers=n_va_workers > 0)
    return tr, va, tr_ds, va_ds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/autodl-tmp/datasets/imagenet100")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_per_class", type=int, default=50)
    a = ap.parse_args()
    ds_tr, ds_va = build_datasets(a.data_dir, a.img_size, a.val_per_class)
    print(json.dumps({"train": len(ds_tr), "val": len(ds_va),
                      "total": len(ds_tr) + len(ds_va),
                      "val_per_class": a.val_per_class}, ensure_ascii=False))
    x, y = ds_tr[0]
    xv, yv = ds_va[0]
    print("train sample:", tuple(x.shape), x.dtype, float(x.mean()), y)
    print("val   sample:", tuple(xv.shape), float(xv.min()), float(xv.max()), yv)
