"""v10 的片段数据集：120 帧 / 15 s @ 8 fps / 448×448 的真实视频片段。

目录口径（与远端 `video_recon/data3` 一致）::

    <root>/<split>/index.jsonl    每行 {"path": "clips/xx.mp4", "src": ..., ...}
    <root>/<split>/clips/*.mp4    path 相对 <root>/<split>

一个样本 = **整段 120 帧**（v10 每一步都要重出整段，所以不分窗口）。
解码依赖 PyAV（`av`）；没有 av 时退回 cv2。

自检：``python data_v10.py <root>``（打印条数、抽查一条的形状/范围）
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["read_index", "decode_clip", "ClipDataset"]


def read_index(split_dir: str, name: str = "index.jsonl") -> List[dict]:
    path = os.path.join(split_dir, name)
    items: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def decode_clip(path: str, lo: int, hi: int, img: int = 448) -> np.ndarray:
    """解码 [lo, hi) 帧 → (hi-lo, img, img, 3) uint8；不足则重复末帧补齐。"""
    out: List[np.ndarray] = []
    try:
        import av
        with av.open(path) as c:
            st = c.streams.video[0]
            for i, fr in enumerate(c.decode(st)):
                if i < lo:
                    continue
                if i >= hi:
                    break
                a = fr.to_ndarray(format="rgb24")
                if a.shape[0] != img or a.shape[1] != img:
                    import cv2
                    a = cv2.resize(a, (img, img), interpolation=cv2.INTER_AREA)
                out.append(a)
    except ImportError:                                    # 无 av → cv2
        import cv2
        cap = cv2.VideoCapture(path)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i < lo:
                i += 1
                continue
            if i >= hi:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if fr.shape[0] != img or fr.shape[1] != img:
                fr = cv2.resize(fr, (img, img), interpolation=cv2.INTER_AREA)
            out.append(fr)
            i += 1
        cap.release()
    if not out:
        raise RuntimeError(f"decode 空: {path} [{lo},{hi})")
    while len(out) < hi - lo:
        out.append(out[-1])
    return np.stack(out[: hi - lo], 0)


class ClipDataset(Dataset):
    """整段片段：返回 (F, 3, H, W) float32 ∈[0,1]。"""

    def __init__(self, root: str, split: str = "train", frames: int = 120,
                 img: int = 448, limit: int = 0):
        self.dir = os.path.join(root, split)
        self.items = read_index(self.dir)
        if limit and int(limit) > 0:
            self.items = self.items[: int(limit)]
        self.frames = int(frames)
        self.img = int(img)
        assert self.items, f"{self.dir} 无样本"

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        it = self.items[int(idx)]
        arr = decode_clip(os.path.join(self.dir, it["path"]), 0, self.frames,
                          img=self.img)                    # (F,H,W,3) uint8
        x = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)  # (F,3,H,W)
        return x.contiguous()


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/video_recon/data3"
    ds = ClipDataset(root, "train", frames=120, img=448)
    print(f"[data] {root} train 条数 = {len(ds)}")
    x = ds[0]
    print("[shape]", tuple(x.shape), "range", round(float(x.min()), 3),
          round(float(x.max()), 3))
    assert x.shape == (120, 3, 448, 448)
    dsv = ClipDataset(root, "val", frames=120, img=448)
    print(f"[data] val 条数 = {len(dsv)}")
    print("OK")
