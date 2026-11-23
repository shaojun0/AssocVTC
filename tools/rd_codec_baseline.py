"""纯 R-D 锚点：在同一套 IN-100 val 5000 张上量 JPEG / WebP 的 bpp-PSNR 曲线。

目的：判断"纯率失真压缩"这条定位能不能用——把我们的 22.67 dB @ 3072 values 换算成 bpp 后，
     和经典编解码器在同 rate 上比。若连 JPEG 都比不过，纯 R-D 定位就不成立。

协议与 data_linattn 的 val transform 一致：Resize(292, BICUBIC) + CenterCrop(256)，
参考图 = 该 256² 的 uint8 数组（float 版即 /255，与训练/评测逐位一致）。
按 JPEG 的 bytes*8/(256*256) 算 bpp（256² 的 65536 像素）。
"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/torchvision/build/lib.linux-x86_64-cpython-310")
sys.path.insert(0, "/root/AssocVTC_tok448")
from data_linattn import scan_imagenet100, split_index  # noqa: E402

DATA = "/root/autodl-tmp/datasets/imagenet100"
IMGSZ = 256
NETWORK = os.environ.get("NETWORK", "jpeg/5000")

def prep(img):
    """等价于 torchvision Resize(292,BICUBIC)+CenterCrop(256)（纯 PIL 实现，
    避开本机 /usr/local 那个空壳 torchvision）。"""
    img = img.convert("RGB")
    short = int(IMGSZ * 256 / 224)                      # 292
    w, h = img.size
    if w <= h:
        nw, nh = short, max(short, int(round(h * short / w)))
    else:
        nh, nw = short, max(short, int(round(w * short / h)))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - IMGSZ) // 2, (nh - IMGSZ) // 2
    return img.crop((left, top, left + IMGSZ, top + IMGSZ))

# 我们的 rate 换算成 bpp
RATES = {
    "ours_8bit": 3072 * 8 / (IMGSZ * IMGSZ),      # 3072 values × 8 bit
    "ours_12bit": 3072 * 12 / (IMGSZ * IMGSZ),    # 3072 values × 12 bit
    "alit_256tok_5bit": 256 * 12 * 5 / (IMGSZ * IMGSZ),  # ALIT: 256 tok × 12 dim × 32 级
}
QUAL = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 75, 90]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    items = scan_imagenet100(DATA)
    _, va = split_index(items, 50)
    LIMIT = int(os.environ.get("LIMIT", "0"))
    if LIMIT > 0:
        va = list(va)[:LIMIT]
    print(f"[rd] val images = {len(va)}", flush=True)
    res = {"jpeg": {q: {"bpp": [], "psnr": []} for q in QUAL},
           "webp": {q: {"bpp": [], "psnr": []} for q in QUAL}}
    for i, (path, _lbl) in enumerate(va):
        with open(path, "rb") as f:
            img = prep(Image.open(f))
        ref = np.asarray(img, dtype=np.uint8)
        for q in QUAL:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=False)
            nb = buf.tell()
            dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"),
                             dtype=np.uint8)
            res["jpeg"][q]["bpp"].append(nb * 8 / (IMGSZ * IMGSZ))
            res["jpeg"][q]["psnr"].append(psnr(ref, dec))
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=q)
            nb = buf.tell()
            dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"),
                             dtype=np.uint8)
            res["webp"][q]["bpp"].append(nb * 8 / (IMGSZ * IMGSZ))
            res["webp"][q]["psnr"].append(psnr(ref, dec))
        if (i + 1) % 1000 == 0:
            print(f"[rd] {i+1}/{len(va)} …", flush=True)

    out = {"n_images": len(va), "img_size": IMGSZ, "rates_bpp": RATES, "curves": {}}
    for codec in ("jpeg", "webp"):
        rows = []
        print(f"\n[rd] {codec.upper()}")
        print(f"{'q':>4} {'bpp':>8} {'PSNR':>8}")
        for q in QUAL:
            d = res[codec][q]
            bpp = float(np.mean(d["bpp"]))
            p = float(np.mean(d["psnr"]))
            rows.append({"quality": q, "bpp": bpp, "psnr": p})
            print(f"{q:>4} {bpp:>8.4f} {p:>8.3f}")
        out["curves"][codec] = rows

    # 在三个 rate 上插值
    print("\n[rd] 同 rate 对比（PSNR 线性插值到给定 bpp）")
    print(f"{'codec':>6} {'rate':>18} {'bpp':>8} {'PSNR':>8}")
    out["at_rate"] = {}
    for codec in ("jpeg", "webp"):
        rows = sorted(out["curves"][codec], key=lambda r: r["bpp"])
        xs = np.array([r["bpp"] for r in rows])
        ys = np.array([r["psnr"] for r in rows])
        for name, bpp in RATES.items():
            p = float(np.interp(bpp, xs, ys))
            out["at_rate"][f"{codec}@{name}"] = {"bpp": bpp, "psnr": p}
            print(f"{codec:>6} {name:>18} {bpp:>8.4f} {p:>8.3f}")
    out["ours_l1"] = {"psnr": 22.666, "values": 3072,
                      "note": "k256x12_t8 L1 ep14, step8, 5000 张, fast_rfid 口径"}
    os.makedirs(os.path.dirname(NETWORK) or ".", exist_ok=True)
    with open(NETWORK, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[rd] -> {NETWORK}", flush=True)


if __name__ == "__main__":
    main()
