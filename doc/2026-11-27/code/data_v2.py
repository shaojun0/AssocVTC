"""
AssocVTC Phase 1 v2 — 数据管线
=================================================
旋转 + 等比缩放 + 填充 到 16:9 画布（默认 1600×900）:
    · 轮廓不变形 —— 只做刚体旋转 + 均匀缩放（无拉伸/裁剪/透视）。
    · 信息量最大化 —— 网格搜索最优旋转角 θ*，使"画布内内容面积"
      （= 缩放因子 s = min(W/bw, H/bh)）最大；θ* 自动覆盖 0°/90°/中间角。
    · 画布 (1600×900) → 模型输入 (448×252) 同为 16:9, 再缩放无二次变形。

设计（解耦）:
    fit_to_canvas      —— 纯函数 (PIL → PIL), 无状态, 训练/推理同一路径
    ParquetImageDataset—— 只吐原始 image bytes（解码/变换在 collate 做）
    V2Collator         —— batch → (B,3,H,W) DINO 归一化张量

自检: python data_v2.py
"""
import io

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

CANVAS_W, CANVAS_H = 1600, 900          # 16:9 画布（"屏幕分辨率比率"）
DINO_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
DINO_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0


# ═══════════════════════════════════════════════════════════════
# 最优旋转角 — 纯浮点数学, 无图像操作
# ═══════════════════════════════════════════════════════════════

def fit_angle(w: int, h: int, cw: int = CANVAS_W, ch: int = CANVAS_H,
              step: float = 0.5) -> float:
    """返回使缩放因子 s(θ)=min(cw/bw, ch/bh) 最大的旋转角 θ*（弧度）。

    θ ∈ [0°, 90°] 即覆盖全部朝向（旋转 ±θ 包围盒相同; 90° 覆盖竖图翻转）。
    bw(θ)=w·cosθ+h·sinθ, bh(θ)=w·sinθ+h·cosθ 为旋转后包围盒尺寸。
    粗网格 + 局部细化（有效精度 ≈ 0.05°）。
    """
    thetas = np.deg2rad(np.arange(0.0, 90.0 + step, step))
    c, s = np.cos(thetas), np.sin(thetas)
    bw = w * c + h * s
    bh = w * s + h * c
    scale = np.minimum(cw / bw, ch / bh)
    t0 = thetas[int(np.argmax(scale))]

    ts = np.linspace(max(0.0, t0 - np.deg2rad(step)),
                     min(np.pi / 2, t0 + np.deg2rad(step)), 21)
    c, s = np.cos(ts), np.sin(ts)
    bw = w * c + h * s
    bh = w * s + h * c
    scale = np.minimum(cw / bw, ch / bh)
    return float(ts[int(np.argmax(scale))])


def fit_to_canvas(img: Image.Image, canvas=(CANVAS_W, CANVAS_H),
                  angle_step: float = 0.5,
                  resample: int = Image.BICUBIC,
                  fill=(0, 0, 0),
                  return_box: bool = False):
    """img → 旋转到最优角度 → 均匀缩放 → 居中填充到 canvas。返回 PIL RGB。

    输出恒为 (canvas_w, canvas_h)。内容零拉伸（各向同性缩放）。

    return_box=True 时返回 ``(out, box)``，``box = (left, top, nw, nh)`` 是内容在
    画布中的粘贴框。**仅供评测侧构造"内容区 mask"用**（PLAN §5 的 padding 问题）；
    默认 False ⇒ 老调用方行为逐位不变（训练路径不受影响）。
    """
    w, h = img.size
    theta = fit_angle(w, h, canvas[0], canvas[1], angle_step)
    deg = np.rad2deg(theta)
    if deg > 1e-3:
        img = img.rotate(deg, resample=resample, expand=True)   # 刚体旋转
    bw, bh = img.size
    s = min(canvas[0] / bw, canvas[1] / bh)                     # 等比 contain
    nw, nh = int(round(bw * s)), int(round(bh * s))
    if (nw, nh) != (bw, bh):
        img = img.resize((nw, nh), resample)                    # 均匀缩放
    out = Image.new("RGB", canvas, fill)
    box = ((canvas[0] - nw) // 2, (canvas[1] - nh) // 2, nw, nh)
    out.paste(img, box[:2])
    if return_box:
        return out, box
    return out


# ═══════════════════════════════════════════════════════════════
# Dataset — parquet（datasets 库）→ 原始 image bytes + 中文 caption
# ═══════════════════════════════════════════════════════════════

class ParquetImageDataset(Dataset):
    """__getitem__ → {"image_bytes": bytes, "image_caption": str|None,
    "violations": str|None}。解码/变换/归一化/编码交给 collate。

    data_dir 下形如 train-*.parquet / test-*.parquet 的分片, image 列为
    struct{bytes, path}（datasets parquet 默认结构）。
    """

    def __init__(self, parquet_files, limit: int = 0):
        from datasets import load_dataset
        self.ds = load_dataset("parquet", data_files=list(parquet_files),
                               split="train")
        if limit > 0 and limit < len(self.ds):
            self.ds = self.ds.select(range(limit))
        print(f"[data] {len(self.ds)} 条 from {len(parquet_files)} parquet")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        im = row["image"]
        return {
            "image_bytes": im["bytes"] if isinstance(im, dict) else im,
            "image_path": im.get("path") if isinstance(im, dict) else None,
            "image_caption": row.get("image_caption"),
            "violations": row.get("violations"),
        }


# ═══════════════════════════════════════════════════════════════
# Collator — bytes → fit_to_canvas(1600:900) → 模型输入 → 张量
# （text 模式可选: caption → Qwen tokenizer → text_ids）
# ═══════════════════════════════════════════════════════════════

DEFAULT_TEXT_TEMPLATE = "描述这张建筑工地图片：{caption}"


class V2Collator:
    def __init__(self, model_size=(448, 252), canvas=(CANVAS_W, CANVAS_H),
                 angle_step: float = 0.5, tokenizer=None,
                 max_text_len: int = 256, pad_token_id: int = 0,
                 text_template: str = DEFAULT_TEXT_TEMPLATE,
                 return_mask: bool = False):
        self.model_w, self.model_h = model_size
        assert abs(self.model_w / self.model_h - canvas[0] / canvas[1]) < 1e-6, \
            "模型输入必须与画布同为 16:9, 否则缩放会变形"
        self.canvas, self.angle_step = canvas, angle_step
        self.tokenizer = tokenizer          # None = 纯重建模式
        self.max_text_len = max_text_len
        self.pad_token_id = pad_token_id
        self.text_template = text_template
        # return_mask=True ⇒ 额外返回 "content_mask" (B, H, W) bool（True=真内容,
        # False=letterbox padding）。**只给评测用**，默认 False ⇒ 训练路径零变化。
        self.return_mask = return_mask

    def _format_text(self, caption, violations) -> str:
        cap = caption or ""
        text = self.text_template.format(caption=cap)
        if violations:
            text += f"\n隐患：{violations}"
        return text

    def __call__(self, batch: list) -> dict:
        xs = []
        masks = []
        for item in batch:
            img = Image.open(io.BytesIO(item["image_bytes"]))
            img = ImageOps.exif_transpose(img).convert("RGB")
            if self.return_mask:
                img, box = fit_to_canvas(img, self.canvas, self.angle_step,
                                         return_box=True)
            else:
                img = fit_to_canvas(img, self.canvas, self.angle_step)
            img = img.resize((self.model_w, self.model_h), Image.BICUBIC)
            if self.return_mask:
                l, t, bw_, bh_ = box
                m = Image.new("L", self.canvas, 0)
                m.paste(Image.new("L", (bw_, bh_), 255), (l, t))
                m = m.resize((self.model_w, self.model_h), Image.NEAREST)
                masks.append(torch.from_numpy(np.asarray(m) > 127))
            arr = np.asarray(img, np.float32)
            arr = (arr - DINO_MEAN) / DINO_STD
            xs.append(torch.from_numpy(arr).permute(2, 0, 1))
        out = {"pixel_values": torch.stack(xs)}
        if self.return_mask:
            out["content_mask"] = torch.stack(masks)          # (B,H,W) bool
            # ⚠️ image_path 必须与 content_mask 同开关: Trainer 会把 collator 的**所有**
            # key 当 kwargs 传给 model.forward, 多一个 key 就是 TypeError
            # （2026-09-22 冒烟实测: forward() got an unexpected keyword argument）。
            out["image_path"] = [it.get("image_path") for it in batch]
        if self.tokenizer is not None:
            # 文字: template(+隐患) → tokenize(截断) → batch 内动态 padding
            texts = [self._format_text(it["image_caption"], it["violations"])
                     for it in batch]
            ids = self.tokenizer(texts, truncation=True,
                                 max_length=self.max_text_len)["input_ids"]
            T = max(max(len(t) for t in ids), 2)          # 错位 CE 至少 2 token
            padded = torch.full((len(ids), T), self.pad_token_id, dtype=torch.long)
            for r, t in enumerate(ids):
                padded[r, :len(t)] = torch.tensor(t[:T], dtype=torch.long)
            out["text_ids"] = padded
        return out


# ═══════════════════════════════════════════════════════════════
# 自检: python data_v2.py
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1) 最优角: 16:9 原图 → 不旋转; 竖图 → 旋转 90°; 网格最优 ≥ 0°/90°
    a = fit_angle(1600, 900)
    assert abs(a) < 1e-3, f"16:9 应不旋转, got {a}"
    a = fit_angle(900, 1600)
    assert abs(a - np.pi / 2) < np.deg2rad(1.0), f"竖图应旋 90°, got {a}"
    for (w, h) in [(1000, 1000), (1200, 900), (1920, 1080), (1000, 2000),
                   (906, 765), (720, 576)]:
        t = fit_angle(w, h)
        s0 = min(1600 / w, 900 / h)
        s90 = min(1600 / h, 900 / w)
        c, s = np.cos(t), np.sin(t)
        sbest = min(1600 / (w * c + h * s), 900 / (w * s + h * c))
        assert sbest >= s0 - 1e-6 and sbest >= s90 - 1e-6, (w, h, t, sbest, s0, s90)
    print("[ok] fit_angle: 0°/90°/中间角全部 ≥ 轴对齐, 16:9 不旋转, 竖图旋 90°")

    # 2) fit_to_canvas: 恒 1600×900, 内容不变形（画布内完整）
    for size in [(1600, 900), (900, 1600), (1200, 900), (1000, 1000),
                 (2000, 800), (700, 500)]:
        img = Image.new("RGB", size, (255, 0, 0))
        out = fit_to_canvas(img)
        assert out.size == (1600, 900), out.size
        arr = np.asarray(out)
        assert arr.shape == (900, 1600, 3)
    print("[ok] fit_to_canvas: 输出恒 1600×900")

    # 3) 内容面积最大化: 旋转方案的红色像素数 ≥ 不旋转方案
    img = Image.new("RGB", (1000, 2000), (255, 0, 0))
    out_rot = np.asarray(fit_to_canvas(img))
    out_flat = np.asarray(fit_to_canvas(img, ))
    img0 = img.rotate(0, resample=Image.BICUBIC, expand=True)
    s0 = min(1600 / img0.size[0], 900 / img0.size[1])
    nw0, nh0 = int(round(img0.size[0] * s0)), int(round(img0.size[1] * s0))
    canvas0 = Image.new("RGB", (1600, 900), (0, 0, 0))
    canvas0.paste(img0.resize((nw0, nh0)), ((1600 - nw0) // 2, (900 - nh0) // 2))
    n_rot = (out_rot.sum(axis=2) > 0).sum()
    n_flat = (np.asarray(canvas0).sum(axis=2) > 0).sum()
    assert n_rot >= n_flat, (n_rot, n_flat)
    print(f"[ok] 内容最大化: 旋转 {n_rot} px ≥ 不旋转 {n_flat} px")

    # 4) Collator: 形状/归一化范围
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGB", (1200, 900), (128, 64, 200)).save(buf, format="JPEG")
    b = buf.getvalue()
    coll = V2Collator()
    out = coll([{"image_bytes": b, "image_caption": None, "violations": None},
                {"image_bytes": b, "image_caption": None, "violations": None}])
    assert out["pixel_values"].shape == (2, 3, 252, 448), out["pixel_values"].shape
    assert out["pixel_values"].dtype == torch.float32
    assert out["pixel_values"].abs().max() < 10
    print(f"[ok] V2Collator: {tuple(out['pixel_values'].shape)} 归一化范围 "
          f"[{out['pixel_values'].min():.2f}, {out['pixel_values'].max():.2f}]")

    # 5) 文字模式: 假 tokenizer → text_ids (动态 padding, ≥2, pad 补位)
    class FakeTok:
        pad_token_id = 7
        def __call__(self, texts, **kw):
            return {"input_ids": [list(range(2, 2 + (len(t) % 5) + 1)) for t in texts]}
    coll_t = V2Collator(tokenizer=FakeTok(), max_text_len=8, pad_token_id=7)
    out_t = coll_t([{"image_bytes": b, "image_caption": "塔吊", "violations": None},
                    {"image_bytes": b, "image_caption": None, "violations": "无安全帽"}])
    assert out_t["text_ids"].shape == (2, 5), out_t["text_ids"].shape
    assert out_t["text_ids"].min() >= 2 and (out_t["text_ids"] == 7).any()  # 有 pad
    assert out_t["text_ids"].size(1) >= 2
    print(f"[ok] V2Collator text: text_ids {tuple(out_t['text_ids'].shape)} "
          f"pad={out_t['text_ids'][-1].tolist()}")

    print("\nALL DATA CHECKS PASSED")
