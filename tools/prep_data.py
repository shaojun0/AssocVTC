#!/usr/bin/env python3
"""prep_data.py — 一条命令在**任意机器**上重建 train_multi 的六源自然图像布局。

为什么要这个脚本（2026-10-14）：
  · 旧的多分辨率六源实验数据只存在于已回收的 AutoDL 实例盘上，换机即失；
  · 三台机器的数据准备脚本（AutoDL 上的临时脚本 / 移动硬盘 fetch_all.sh /
    内网 unpack_data.sh）互不通用，且硬编码 `/root/autodl-tmp`。
  本脚本只依赖 `data_root` + HF 镜像地址，用同一份代码在 AutoDL / 内网 / 任意
  有外网的机器上重建同样的布局，并在最后打印每个源的文件数。

布局（与 data_multi.SOURCES 一一对应）::

    <data_root>/imagenet_val/*.jpg          50,000   mrm8488/ImageNet1K-val
    <data_root>/coco/train2017/*.jpg       118,287   pcuenq/coco-2017-mirror
    <data_root>/coco/val2017/*.jpg           5,000   （留出集，eval_stepwise 用）
    <data_root>/vimeo/*.png                 91,701   danjacobellis/vimeo90k_triplet (im1)
    <data_root>/div2k/DIV2K_train_HR/*.png     800   yangtao9009/DIV2K
    <data_root>/flickr2k/*.png               2,650   yangtao9009/Flickr2K
    <data_root>/openimages/*.jpg           ~19,000   bitmind/open-images-v7-subset

用法::

    python tools/prep_data.py --data_root /root/autodl-tmp/data \
        --sources imagenet coco vimeo div2k flickr2k openimages

  · 幂等：每个源成功后在 <data_root>/.prep_done/<src> 落标记；重跑自动跳过
    （`--force <src>` 可强制重做）。
  · 断点续传：所有下载都走 `curl -C -`，单文件失败不会拖垮整轮。
  · 线程并发：parquet/图片适合多线程；大 zip 走单线程（镜像对大文件可能限速，
    可用 `--zip_threads`/`--flickr_individual` 切换策略）。
"""
import argparse
import concurrent.futures as cf
import glob
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time

import pyarrow.parquet as pq
from huggingface_hub import HfApi

# ── 数据源注册表：name → 仓库 + 文件模式 + 解出目录 ──────────────────────────
REGISTRY = {
    "imagenet": dict(repo="mrm8488/ImageNet1K-val", rtype="dataset",
                     pattern="data/train-*.parquet", out="imagenet_val",
                     ext=".jpg", expect=50000),
    "vimeo": dict(repo="danjacobellis/vimeo90k_triplet", rtype="dataset",
                  pattern="data/train-*.parquet", out="vimeo",
                  ext=".png", expect=91701, only_basename="im1.png"),
    "openimages": dict(repo="bitmind/open-images-v7-subset", rtype="dataset",
                       pattern="data/train-*.parquet", out="openimages",
                       ext=".jpg", expect=15000, shards=12),
    "coco": dict(repo="pcuenq/coco-2017-mirror", rtype="dataset",
                 pattern="*.zip", out="coco", kind="zip", expect=118287),
    "div2k": dict(repo="yangtao9009/DIV2K", rtype="dataset",
                  pattern="DIV2K_train.zip", out="div2k/DIV2K_train_HR",
                  kind="zip", expect=800),
    "flickr2k": dict(repo="yangtao9009/Flickr2K", rtype="dataset",
                     pattern="Flickr2K.zip", out="flickr2k",
                     kind="zip", expect=2650),
}
ALL = list(REGISTRY)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def log(*a):
    print(f"[{time.strftime('%F %T')}]", *a, flush=True)


def sniff_ext(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:2] == b"BM":
        return ".bmp"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    return ""


def img_bytes(row):
    """parquet 行 → bytes。兼容 struct{bytes,path} / bytes / PIL.Image 三种形态。

    坑（2026-10-14 内网复现方案）：HF `datasets` 会把 Image feature 自动解码成
    PIL.Image，只用 `row["image"]["bytes"]` 会 TypeError；这里三分支都兜住。
    """
    v = row.get("image")
    if v is None:
        return None, ""
    if hasattr(v, "save"):                       # PIL.Image
        buf = io.BytesIO()
        v.convert("RGB").save(buf, format="PNG" if v.mode == "RGBA" else "JPEG")
        return buf.getvalue(), ""
    if isinstance(v, dict):
        return v.get("bytes"), v.get("path") or ""
    if isinstance(v, (bytes, bytearray)):
        return bytes(v), ""
    return None, ""


# ── 下载 ────────────────────────────────────────────────────────────────────
def curl(url: str, dest: str, tries: int = 4, min_bytes: int = 1024) -> bool:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    for i in range(tries):
        r = subprocess.run(
            ["curl", "-sSL", "--fail", "-C", "-", "--retry", "3",
             "--retry-delay", "5", "--connect-timeout", "30",
             "--max-time", "3600", "-o", dest, url],
            capture_output=True)
        if r.returncode == 0 and os.path.exists(dest) \
                and os.path.getsize(dest) >= min_bytes:
            return True
        log(f"   [retry {i + 1}/{tries}] {os.path.basename(dest)} rc={r.returncode} "
            f"{r.stderr.decode()[-120:]}")
        time.sleep(5 * (i + 1))
    return False


def hf_files(repo: str, rtype: str, endpoint: str, pattern: str):
    api = HfApi(endpoint=endpoint)
    info = (api.dataset_info(repo, files_metadata=True) if rtype == "dataset"
            else api.model_info(repo, files_metadata=True))
    import fnmatch
    out = [(f.rfilename, f.size or 0) for f in info.siblings
           if not f.rfilename.startswith(".")
           and fnmatch.fnmatch(f.rfilename, pattern)]
    out.sort()
    return out


def repo_base(endpoint: str, repo: str, rtype: str) -> str:
    kind = "datasets/" if rtype == "dataset" else ""
    return f"{endpoint.rstrip('/')}/{kind}{repo}/resolve/main"


# ── 解包：parquet image 列 ──────────────────────────────────────────────────
def extract_parquet(paths, out_dir, ext=".jpg", only_basename=None,
                    per_shard=0, workers=6, tag=""):
    os.makedirs(out_dir, exist_ok=True)

    def one(item):
        idx, path = item
        n = 0
        pf = pq.ParquetFile(path)
        for bi, batch in enumerate(pf.iter_batches(batch_size=64, columns=["image"])):
            for ri, row in enumerate(batch.to_pylist()):
                b, p = img_bytes(row)
                if b is None:
                    continue
                if only_basename and os.path.basename(p) != only_basename:
                    continue
                if per_shard and n >= per_shard:
                    break
                e = sniff_ext(b)
                name = f"{tag}{idx:05d}_{bi * 64 + ri:07d}{(e or ext)}"
                with open(os.path.join(out_dir, name), "wb") as fh:
                    fh.write(b)
                n += 1
            else:
                continue
            break
        return os.path.basename(path), n

    total = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for name, n in ex.map(one, list(enumerate(paths)), chunksize=1):
            total += n
            log(f"   [extract] {name}: +{n} (累计 {total})")
    return total


def extract_zip(zip_path, out_dir, flatten=True, keep_exts=IMG_EXTS):
    """zip → out_dir（可摊平）。返回解出的图片数。"""
    import zipfile
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(keep_exts)]
        for m in members:
            data = z.read(m)
            name = os.path.basename(m) if flatten else m
            if not name:
                continue
            dst = os.path.join(out_dir, name)
            if os.path.exists(dst) and os.path.getsize(dst) == len(data):
                n += 1
                continue
            with open(dst, "wb") as fh:
                fh.write(data)
            n += 1
            if n % 2000 == 0:
                log(f"   [unzip] {n} …")
    return n


# ── 各源 ────────────────────────────────────────────────────────────────────
def do_parquet(name, spec, args):
    raw = os.path.join(args.raw, name)
    os.makedirs(raw, exist_ok=True)
    files = hf_files(spec["repo"], spec["rtype"], args.endpoint, spec["pattern"])
    if spec.get("shards"):
        files = files[:spec["shards"]]
    if not files:
        log(f"!! {name}: 仓库里没有匹配 {spec['pattern']} 的文件")
        return 0
    log(f"== {name}: {len(files)} 个 parquet "
        f"({sum(s for _, s in files) / 2 ** 30:.2f} GiB) → {raw}")
    base = repo_base(args.endpoint, spec["repo"], spec["rtype"])
    paths = []

    def dl(it):
        f, size = it
        dst = os.path.join(raw, os.path.basename(f))
        if os.path.exists(dst) and size and abs(os.path.getsize(dst) - size) == 0:
            return dst
        ok = curl(f"{base}/{f}", dst, min_bytes=min(size, 1 << 20))
        if ok and size and os.path.getsize(dst) != size:
            log(f"   [size] {f}: {os.path.getsize(dst)} != {size}，重下")
            os.remove(dst)
            ok = curl(f"{base}/{f}", dst, min_bytes=size - 1024)
        return dst if ok else None

    with cf.ThreadPoolExecutor(max_workers=args.dl_threads) as ex:
        for p in ex.map(dl, files, chunksize=1):
            if p:
                paths.append(p)
            elif args.strict:
                return 0
    log(f"== {name}: 下载完成 {len(paths)}/{len(files)}")
    if len(paths) < len(files) and args.strict:
        return 0
    n = extract_parquet(paths, os.path.join(args.data_root, spec["out"]),
                        ext=spec["ext"], only_basename=spec.get("only_basename"),
                        workers=args.ex_threads, tag=f"{name}_")
    if args.keep_raw:
        return n
    shutil.rmtree(raw, ignore_errors=True)
    return n


def do_zip(name, spec, args):
    files = hf_files(spec["repo"], spec["rtype"], args.endpoint, spec["pattern"])
    if not files:
        log(f"!! {name}: 仓库里没有匹配 {spec['pattern']}")
        return 0
    base = repo_base(args.endpoint, spec["repo"], spec["rtype"])
    raw = os.path.join(args.raw, name)
    os.makedirs(raw, exist_ok=True)
    out = os.path.join(args.data_root, spec["out"])
    n = 0
    for f, size in files:
        dst = os.path.join(raw, os.path.basename(f))
        if not (os.path.exists(dst) and (not size or os.path.getsize(dst) == size)):
            log(f"== {name}: 下载 {f} ({size / 2 ** 30:.2f} GiB)")
            if not curl(f"{base}/{f}", dst, tries=args.tries):
                log(f"!! {name}: {f} 下载失败")
                return 0
        if not f.lower().endswith(".zip"):
            continue
        if name == "coco":
            # COCO：解到 <data_root>/coco/ 下（zip 内含 train2017/ val2017/）
            log(f"== coco: 解 {os.path.basename(f)} → {args.data_root}/coco")
            subprocess.run(["unzip", "-q", "-o", dst, "-d",
                            os.path.join(args.data_root, "coco")], check=True)
            n = max(n, len(glob.glob(os.path.join(
                args.data_root, "coco", "train2017", "*.jpg"))))
        else:
            log(f"== {name}: 解 {os.path.basename(f)} → {out}")
            n = extract_zip(dst, out)
    if not args.keep_raw:
        shutil.rmtree(raw, ignore_errors=True)
    return n


def count_dir(d):
    return sum(1 for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
               if f.lower().endswith(IMG_EXTS))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=os.environ.get("SRDIFF_DATA_ROOT",
                                                          "/root/autodl-tmp/data"))
    ap.add_argument("--sources", nargs="+", default=["all"],
                    help=f"要准备的源，all 或 {' '.join(ALL)}")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT",
                                                         "https://hf-mirror.com"))
    ap.add_argument("--raw", default="", help="原始 parquet/zip 暂存目录（默认 <data_root>/_raw）")
    ap.add_argument("--force", nargs="*", default=[], help="强制重做这些源")
    ap.add_argument("--dl_threads", type=int, default=4, help="并发下载的 parquet shard 数")
    ap.add_argument("--ex_threads", type=int, default=6, help="并发解包的 shard 数")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--keep_raw", action="store_true", help="保留原始文件（默认解完即删省盘）")
    ap.add_argument("--strict", action="store_true", help="任一 shard 失败即终止该源")
    args = ap.parse_args()

    args.data_root = os.path.abspath(args.data_root)
    args.raw = args.raw or os.path.join(args.data_root, "_raw")
    srcs = ALL if "all" in args.sources else args.sources
    for s in srcs:
        if s not in REGISTRY:
            raise SystemExit(f"未知数据源 {s}；可选 {ALL}")
    os.makedirs(args.data_root, exist_ok=True)
    done_dir = os.path.join(args.data_root, ".prep_done")
    os.makedirs(done_dir, exist_ok=True)
    log(f"data_root={args.data_root} endpoint={args.endpoint} sources={srcs}")

    report = {}
    for s in srcs:
        spec = REGISTRY[s]
        marker = os.path.join(done_dir, s)
        d = os.path.join(args.data_root, spec["out"])
        if os.path.exists(marker) and s not in args.force:
            n = count_dir(d)
            log(f"-- {s}: 已完成标记，现有 {n} 张，跳过（--force {s} 可重做）")
            report[s] = n
            continue
        t0 = time.time()
        n = do_parquet(s, spec, args) if spec.get("kind") != "zip" else do_zip(s, spec, args)
        log(f"-- {s}: {n} 张，用时 {(time.time() - t0) / 60:.1f} min")
        report[s] = n
        if n > 0:
            with open(marker, "w") as fh:
                fh.write(f"{time.strftime('%F %T')} n={n}\n")

    log("=" * 60)
    log("最终布局：")
    for s in srcs:
        d = os.path.join(args.data_root, REGISTRY[s]["out"])
        log(f"   {REGISTRY[s]['out']:<28} {count_dir(d):>8} 张")
    free = shutil.disk_usage(args.data_root).free / 2 ** 30
    log(f"   剩余磁盘 {free:.1f} GiB")
    log("PREP_DONE")
    return 0 if all(v > 0 for v in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
