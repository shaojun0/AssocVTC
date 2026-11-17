"""torchvision 的最小纯 PIL/torch 替身（无 CUDA kernel、无 C++ 扩展）。

为什么存在
----------
训练机 `connect.sha1`（4×MTT S4000 / MUSA）上装的 `torchvision` 是一个**空壳**
（`__version__='0.0.0'`，`torchvision.transforms` 里没有 `Compose`），任何
`from torchvision import transforms` 的代码在那里直接 AttributeError。
本模块只实现仓库数据管线/采样图真正用到的子集，**API 与 torchvision 对齐**：

    transforms.Compose / RandomResizedCrop / RandomHorizontalFlip /
    ToTensor / Normalize / Resize / CenterCrop、InterpolationMode
    make_grid / save_image

数值口径与 torchvision 一致（RandomResizedCrop 的 area/aspect 采样、
PIL BICUBIC 重采样、ToTensor=/255）。差异只在实现（纯 PIL + torch 张量运算），
因此不能指望逐位复现 torchvision 的随机数序列 —— 本仓库的训练本就从零开始，
数据增强只需保持同一分布。

用法（替换两行 import 即可）::

    import tvmini as transforms                  # 原 from torchvision import transforms
    from tvmini import InterpolationMode          # 原 from torchvision.transforms import ...
    from tvmini import make_grid, save_image      # 原 from torchvision.utils import ...

自检：``python tvmini.py``（CPU 即可，检查形状/取值范围与 grid 拼图）。
"""
from __future__ import annotations

import math
import random
from enum import Enum
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = [
    "InterpolationMode", "Compose", "RandomResizedCrop", "RandomHorizontalFlip",
    "ToTensor", "Normalize", "Resize", "CenterCrop",
    "make_grid", "save_image",
]

_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_NEAREST = getattr(Image, "Resampling", Image).NEAREST


class InterpolationMode(str, Enum):
    """与 torchvision.transforms.InterpolationMode 同名子集（值不重要）。"""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


def _pil_filter(interpolation) -> int:
    if interpolation in (InterpolationMode.BICUBIC, "bicubic", None, 3):
        return _BICUBIC
    if interpolation in (InterpolationMode.BILINEAR, "bilinear", 2):
        return _BILINEAR
    if interpolation in (InterpolationMode.NEAREST, "nearest", 0):
        return _NEAREST
    raise ValueError(f"不支持的 interpolation: {interpolation!r}")


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    raise TypeError(f"需要 PIL.Image 或 ndarray, got {type(img)!r}")


def _to_chw_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → (3,H,W) float32 ∈[0,1]（torchvision ToTensor 口径）。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    a = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0)
    return t.contiguous()


class Compose:
    def __init__(self, tfs: Sequence):
        self.transforms = list(tfs)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        body = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{body}])"


class RandomResizedCrop:
    """torchvision 同名语义：随机面积/长宽比裁剪后再 resize 到 size×size。

    默认参数与 `data_linattn.build_transform` 一致（scale=(0.35,1.0),
    ratio=(3/4,4/3), BICUBIC）。
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BICUBIC):
        self.size = (int(size), int(size)) if isinstance(size, int) else tuple(size)
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.interpolation = interpolation

    def __call__(self, img):
        img = _to_pil(img)
        w, h = img.size
        if self.size[0] == 0 or self.size[1] == 0:
            raise ValueError("size 不能为 0")
        area = h * w
        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            cw = int(round(math.sqrt(target_area * aspect_ratio)))
            ch = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < cw <= w and 0 < ch <= h:
                i = random.randint(0, h - ch)
                j = random.randint(0, w - cw)
                crop = img.crop((j, i, j + cw, i + ch))
                return crop.resize(self.size, _pil_filter(self.interpolation))
        # 兜底：torchvision 也会走中心裁剪
        in_ratio = w / h
        if in_ratio < self.ratio[0]:
            cw = w
            ch = int(round(cw / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            ch = h
            cw = int(round(ch * self.ratio[1]))
        else:
            cw, ch = w, h
        i = (h - ch) // 2
        j = (w - cw) // 2
        crop = img.crop((j, i, j + cw, i + ch))
        return crop.resize(self.size, _pil_filter(self.interpolation))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img):
        img = _to_pil(img)
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class ToTensor:
    def __call__(self, img):
        return _to_chw_tensor(_to_pil(img))


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(3, 1, 1)

    def __call__(self, t: torch.Tensor):
        return (t - self.mean) / self.std


class Resize:
    """torchvision 语义：int size = 短边缩放到 size、保持长宽比。"""

    def __init__(self, size, interpolation=InterpolationMode.BICUBIC):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        img = _to_pil(img)
        w, h = img.size
        if w <= h:                       # 短边是 w
            ow = self.size
            oh = max(1, int(round(h * self.size / w)))
        else:
            oh = self.size
            ow = max(1, int(round(w * self.size / h)))
        return img.resize((ow, oh), _pil_filter(self.interpolation))


class CenterCrop:
    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, int) else tuple(size)

    def __call__(self, img):
        img = _to_pil(img)
        w, h = img.size
        cw, ch = self.size
        if cw > w or ch > h:
            raise ValueError(f"CenterCrop 尺寸 {self.size} 大于图像 {(w, h)}")
        i = (h - ch) // 2
        j = (w - cw) // 2
        return img.crop((j, i, j + cw, i + ch))


# ─────────────────────────── 采样图/拼图 ───────────────────────────
def make_grid(tensors, nrow: int = 8, padding: int = 2,
              value_range: Tuple[float, float] | None = (0.0, 1.0)) -> torch.Tensor:
    """(B,3,H,W) 或 list[Tensor] → (3, H', W') 网格图（torchvision 口径）。"""
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    imgs = [t if t.dim() == 4 else t.unsqueeze(0) for t in tensors]
    n = sum(t.shape[0] for t in imgs)
    if n == 0:
        raise ValueError("make_grid 需要至少一张图")
    _, c, h, w = imgs[0].shape
    nrow = min(int(nrow), n)
    ncol = int(math.ceil(n / nrow))
    grid = torch.full((c, nrow * h + padding * (nrow - 1),
                       ncol * w + padding * (ncol - 1)), float("nan"))
    k = 0
    for t in imgs:
        t = t.detach().to(torch.float32).cpu()
        for b in range(t.shape[0]):
            r, cc = divmod(k, ncol)
            y = r * (h + padding)
            x = cc * (w + padding)
            grid[:, y:y + h, x:x + w] = t[b]
            k += 1
    if value_range is not None:
        grid = grid.clamp_(float(value_range[0]), float(value_range[1]))
    return grid


def save_image(tensor, path: str, nrow: int = 8, padding: int = 2,
               value_range: Tuple[float, float] | None = (0.0, 1.0)) -> None:
    """单图 (3,H,W) 或多图网格 → PNG 落盘。"""
    t = tensor
    if t.dim() == 3:
        g = t
    else:
        g = make_grid(t, nrow=nrow, padding=padding, value_range=value_range)
    if value_range is not None:
        g = g.clamp(float(value_range[0]), float(value_range[1]))
    arr = (g.detach().to(torch.float32).cpu() * 255.0).round().clamp(0, 255)
    arr = arr.to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


if __name__ == "__main__":            # 自检（CPU 即可）
    from PIL import Image as _I
    img = _I.new("RGB", (500, 375))
    px = np.random.RandomState(0).randint(0, 255, (375, 500, 3), dtype=np.uint8)
    img = _I.fromarray(px)

    tf = Compose([
        RandomResizedCrop(448, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3),
                          interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img)
    assert x.shape == (3, 448, 448), x.shape
    assert x.dtype == torch.float32
    vt = Compose([Resize(512, interpolation=InterpolationMode.BICUBIC),
                  CenterCrop(448), ToTensor(),
                  Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    xv = vt(img)
    assert xv.shape == (3, 448, 448), xv.shape

    g = make_grid(torch.rand(5, 3, 448, 448), nrow=5, padding=2, value_range=(0, 1))
    assert g.shape == (3, 5 * 448 + 4 * 2, 448), g.shape   # nrow=5, ncol=1
    save_image(g, "/tmp/tvmini_selftest.png", value_range=(0, 1))
    import os
    assert os.path.getsize("/tmp/tvmini_selftest.png") > 0
    print("[tvmini] OK", tuple(x.shape), tuple(xv.shape), tuple(g.shape))
