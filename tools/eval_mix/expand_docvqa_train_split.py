#!/usr/bin/env python
"""DocVQA train split 扩容（1,200 行）：nielsr/docvqa_1200_examples。

lmms-lab/DocVQA 的 "DocVQA" 配置只有 validation/test（validation 5,288 条已用尽，
test 无答案），官方 train split 不在该仓库 ⇒ 用 HF 上现成的 DocVQA train 示例集
（1,200 行，含图、英文 query、answers）补足。该集来自 DocVQA **train** split，
与我们的 heldout（validation 切分）不重叠。

产出 source="docvqa_train"（和 textvqa/textvqa_train 同样的组织方式），
id 前缀 docvqa_train_，图片 img/docvqa_train/。
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
OUT = os.path.join(ROOT, "expand_docvqa_train_split.jsonl")
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
        im = image.convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    os.makedirs(os.path.join(IMG_DIR, "docvqa_train"), exist_ok=True)
    path = os.path.join(IMG_DIR, "docvqa_train", f"{n:06d}.jpg")
    im.save(path, "JPEG", quality=90)
    return path


def main() -> int:
    ds = load_dataset("nielsr/docvqa_1200_examples", split="train", streaming=True)
    rows, seen_q = [], set()
    n = 0
    for row in ds:
        ans = norm_ans(row.get("answers") or row.get("answer"))
        query = row.get("query") or {}
        q = norm_question(query.get("en") if isinstance(query, dict) else query)
        if not ans or not q or len(ans) > MAX_ANS or len(q) + len(ans) > MAX_CHARS:
            continue
        if q in seen_q:                     # 同一问题多语言重复行：去重
            continue
        path = save_image(row.get("image"), n)
        if path is None:
            continue
        seen_q.add(q)
        msg = [
            {"role": "system", "content": SYS_VQA},
            {"role": "user", "content": [{"type": "image", "image": path},
                                         {"type": "text", "text": q}]},
            {"role": "assistant", "content": f"The answer is {ans}"},
        ]
        rows.append({"id": f"docvqa_train_{n:06d}", "source": "docvqa_train",
                     "task": "short_vqa", "messages": msg,
                     "messages_short": json.loads(json.dumps(msg)), "gt": ans,
                     "image": path})
        n += 1
        if n % 300 == 0:
            print(f"  ...{n}", flush=True)

    with open(OUT, "w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[docvqa-train] 写出 {len(rows)} 行 → {OUT}")

    # 与旧 heldout（docvqa 题库）问题重叠检查
    held_q = set()
    for line in open(os.path.join(ROOT, "mix_heldout.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") == "docvqa":
            held_q.add(r["messages"][1]["content"][1]["text"])
    clash = sum(1 for r in rows if r["messages"][1]["content"][1]["text"] in held_q)
    missing = sum(1 for r in rows if not os.path.exists(r["image"]))
    print(f"[docvqa-train] 与 heldout 同题: {clash}；图片缺失: {missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
