#!/usr/bin/env python
"""ChartQA train split 扩容（4,500 行）：ahmed-masry/ChartQA。

lmms-lab/ChartQA 只有 test（旧构建已用满 2,500 = 1,500 train + 1,000 heldout）。
官方 train split（18,000 条，human/augmented 混合，含图+query+label）在
ahmed-masry/ChartQA 仓库可得 ⇒ 取 4,500 条补足，chatqa 任务达到 6,000。

产出 source="chartqa_train"（与 textvqa_train 同样的组织方式），id 前缀
chartqa_train_，图片 img/chartqa_train/。会检查与旧 heldout（ChartQA test 切分）
的问题文本重叠，若很高说明 train/test 的图-题复用严重，需要人工复核。
"""
import json
import os
import re
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = "/root/autodl-tmp/vlm8b/data_eval"
IMG_DIR = os.path.join(ROOT, "img")
OUT = os.path.join(ROOT, "expand_chartqa_train.jsonl")
N_TARGET = 4500
MAX_CHARS, MAX_ANS, MAX_SIDE = 1400, 60, 640
SYS_VQA = ('你是多模态问答助手。结合图像与题目，给出最简短的答案，'
           '只回复 "The answer is <答案>"，不要解释。')


def norm_ans(text):
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    s = str(text).replace("\u00a0", " ").strip()
    s = s.split("\n")[0].strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip("。.") if len(s) > 1 else s
    return s


def norm_question(text):
    s = str(text or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def save_image(image, n):
    if image is None:
        return None
    try:
        if isinstance(image, (bytes, bytearray)):
            import io
            im = Image.open(io.BytesIO(image))
        else:
            im = Image.open(image) if not hasattr(image, "convert") else image
        im = im.convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    os.makedirs(os.path.join(IMG_DIR, "chartqa_train"), exist_ok=True)
    path = os.path.join(IMG_DIR, "chartqa_train", f"{n:06d}.jpg")
    im.save(path, "JPEG", quality=90)
    return path


def main() -> int:
    ds = load_dataset("ahmed-masry/ChartQA", split="train", streaming=True)
    rows = []
    n = 0
    for row in ds:
        if n >= N_TARGET:
            break
        ans = norm_ans(row.get("label"))
        q = norm_question(row.get("query"))
        if not ans or not q or len(ans) > MAX_ANS or len(q) + len(ans) > MAX_CHARS:
            continue
        path = save_image(row.get("image"), n)
        if path is None:
            continue
        msg = [
            {"role": "system", "content": SYS_VQA},
            {"role": "user", "content": [{"type": "image", "image": path},
                                         {"type": "text", "text": q}]},
            {"role": "assistant", "content": f"The answer is {ans}"},
        ]
        rows.append({"id": f"chartqa_train_{n:06d}", "source": "chartqa_train",
                     "task": "short_vqa", "messages": msg,
                     "messages_short": json.loads(json.dumps(msg)), "gt": ans,
                     "image": path})
        n += 1
        if n % 500 == 0:
            print(f"  ...{n}/{N_TARGET}", flush=True)

    held_q = set()
    for line in open(os.path.join(ROOT, "mix_heldout.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") == "chartqa":
            held_q.add(r["messages"][1]["content"][1]["text"])
    clash = sum(1 for r in rows if r["messages"][1]["content"][1]["text"] in held_q)
    missing = sum(1 for r in rows if not os.path.exists(r["image"]))
    print(f"[chartqa-train] 与 heldout 同题: {clash}；图片缺失: {missing}")
    if clash or missing or len(rows) != N_TARGET:
        return 1

    with open(OUT, "w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[chartqa-train] 写出 {len(rows)} 行 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
