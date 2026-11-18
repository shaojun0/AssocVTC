"""离线缓存 v10 的冻结编码器输出（z_cls + z_s）——训练时不再解码/编码。

为什么要有这一步：v10 的冻结分词器（487M）对 120 帧 × 448² 做前向，
单样本 0.37 s（4 卡上占单步时间 ~30%），而它的输出对每个片段是常量
（分词器冻结、输入不变）。离线算一次、训练反复用，顺带把 mp4 解码
和 448² resize 从训练循环里拿掉。

用法（4 卡，每卡一个 shard）::

    SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 build_latent_cache.py \
        --data_root /root/autodl-tmp/video_recon/data3 --split train \
        --tok_ckpt /root/autodl-tmp/tok448_v2/s2/ckpt_best.pt \
        --out /root/autodl-tmp/tok448_v2/latent_cache/s2_120f/train \
        --frames 120 --img 448 --batch 4

产出：``<out>/rank{r}.npy``（float16，(n_r, frames, 1+K, dim)）+ ``meta.json``。
全局样本 ``i`` 落在 shard ``i % world`` 的第 ``i // world`` 行；行内第 0 行是
z_cls、1..K 是 special token —— 与 ``VideoV10.target_latents`` 的拼接顺序一致。
训练侧见 ``train_v10.py --latent_cache <out>``。

注意：缓存的是 **bf16 前向结果转 float16**（latent 量级 |z|<100，fp16 表示
bf16 值无损；比 uint16 位打包简单）。分词器权重若变（S3→其他），缓存必须重算。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_v10 import ClipDataset                      # noqa: E402
from model_v10 import FrozenTokenizer                # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="缓存 v10 冻结编码器输出")
    p.add_argument("--data_root", type=str,
                   default="/root/autodl-tmp/video_recon/data3")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--tok_ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--img", type=int, default=448)
    p.add_argument("--batch", type=int, default=4, help="每卡一次前向几个片段")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条（调试）")
    p.add_argument("--rank", type=int, default=-1,
                   help="本进程负责的 shard 号（-1 = 从 RANK 环境变量取；不启 NCCL/MCCL）")
    p.add_argument("--world", type=int, default=-1,
                   help="shard 总数（-1 = 从 WORLD_SIZE 环境变量取）")
    return p.parse_args()


def _set_sdpa_math() -> None:
    for name in ("flash", "mem_efficient"):
        try:
            getattr(torch.backends.cuda, f"enable_{name}_sdp")(False)
        except Exception:
            pass
    try:
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def main() -> None:
    a = parse_args()
    _set_sdpa_math()
    os.makedirs(a.out, exist_ok=True)

    # 不开进程组：每个 shard 完全独立（只写自己的文件），避免 MUSA 上
    # 4 进程同时建 MCCL + 首次上下文初始化时把驱动卡死（实测会 hang 在
    # flush_workqueue / GPU 0% 上，连 sshd 都拒绝连接）。启动器给每个进程
    # 一个 MUSA_VISIBLE_DEVICES=<i>，这里统一落到 cuda:0。
    rank = int(a.rank if a.rank >= 0 else os.environ.get("RANK", "0"))
    world = int(a.world if a.world >= 0 else os.environ.get("WORLD_SIZE", "1"))
    dev = torch.device("cuda", 0)

    tok = FrozenTokenizer(a.tok_ckpt).to(dev)
    tok.eval()
    F, rows, dim = int(a.frames), 1 + int(tok.num_specials), int(tok.code_dim)

    ds = ClipDataset(a.data_root, a.split, frames=F, img=a.img)
    n_all = len(ds)
    n = min(n_all, int(a.limit)) if a.limit else n_all
    idx = list(range(rank, n, world))
    print(f"[rank{rank}] {a.split}: 总 {n_all} 条，本次 {n} 条，本卡 {len(idx)} 条 "
          f"(frames={F} rows={rows} dim={dim})", flush=True)

    path = os.path.join(a.out, f"rank{rank}.npy")
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                   shape=(len(idx), F, rows, dim))

    def collate(batch):
        return torch.stack(batch, dim=0)                 # (B,F,3,H,W)

    loader = DataLoader(Subset(ds, idx), batch_size=int(a.batch), shuffle=False,
                        num_workers=int(a.num_workers), drop_last=False,
                        collate_fn=collate, pin_memory=False,
                        prefetch_factor=2)
    off = 0
    t0 = time.time()
    for bi, x in enumerate(loader):
        x = x.to(dev, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z_cls, z_s = tok.encode(x)                   # (B,F,1,c),(B,F,K,c)
            z = torch.cat([z_cls, z_s], dim=2)           # (B,F,1+K,c)
        arr = z.float().half().cpu().numpy()
        mm[off:off + arr.shape[0]] = arr
        off += arr.shape[0]
        if bi % 20 == 0:
            el = time.time() - t0
            rate = off / max(el, 1e-6)
            print(f"[rank{rank}] {off}/{len(idx)}  {rate:.2f} clip/s  "
                  f"eta {max(0.0, (len(idx) - off) / max(rate, 1e-9)) / 60:.1f} min "
                  f"el {el/60:.1f} min", flush=True)
            mm.flush()
    mm.flush()
    del mm
    print(f"[rank{rank}] done {off} 条 {time.time() - t0:.0f}s", flush=True)

    if rank == 0:
        meta = dict(n=int(n), world=int(world), frames=F, rows=rows, dim=dim,
                    dtype="float16", split=a.split, tok_ckpt=a.tok_ckpt,
                    data_root=a.data_root, batch=int(a.batch),
                    n_all=int(n_all))
        with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        total = sum(int(np.load(os.path.join(a.out, f"rank{r}.npy"),
                                mmap_mode="r").shape[0]) for r in range(world))
        print(f"[rank0] meta.json 写好；{total} 条 = sum(shard)", flush=True)



if __name__ == "__main__":
    main()
