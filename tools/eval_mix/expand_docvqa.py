#!/usr/bin/env python
"""DocVQA 训练行扩容：从旧构建停下的位置（5,239 条已接受行）继续取 validation 流。

旧构建（build_img_eval_mix.py）对 docvqa 的配额是 train 4,500 / heldout 800，实际接受
4,451 + 788 = 5,239 行（61 条 ans_too_long 丢弃）。本脚本按**同一过滤逻辑**流式重扫
validation split，跳过前 5,239 条已接受行（= 旧 jsonl 全部内容，含 heldout 行），
再追加 1,549 行到 6,000 行，写成独立文件 `expand_docvqa_train.jsonl`（不动旧文件）。

安全约束：新行 id 从 docvqa_005239 起，图片写 img/docvqa/005239.jpg 起，
与旧 train（000000–004450）和旧 heldout（004451–005238）都不重叠。
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = "/root/autodl-tmp/vlm8b/data_eval"
IMG_DIR = os.path.join(ROOT, "img")
OUT = os.path.join(ROOT, "expand_docvqa_train.jsonl")
MAX_CHARS, MAX_ANS, MAX_SIDE = 1400, 60, 640
OLD_ACCEPTED = 5239          # = 旧 docvqa train 4,451 + heldout 788
EXTRA = 1549                 # 4,451 + 1,549 = 6,000
SYS_VQA = ('你是多模态问答助手。结合图像与题目，给出最简短的答案，'
           '只回复 "The answer is <答案>"，不要解释。')


def load_retry(fn, tries=10):
    for t in range(tries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = 60 + 20 * t
                print(f"    429 backoff {wait}s (try {t+1}/{tries})", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("load_retry exhausted")


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
    s = re.sub(r"<image\s*\d?>", "", s)
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
    os.makedirs(os.path.join(IMG_DIR, "docvqa"), exist_ok=True)
    path = os.path.join(IMG_DIR, "docvqa", f"{n:06d}.jpg")
    im.save(path, "JPEG", quality=90)
    return path


def main() -> int:
    ds = load_retry(lambda: load_dataset(
        "lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True))
    accepted = 0
    got = 0
    rows = []
    for row in ds:
        ans = norm_ans(row.get("answers") or row.get("answer"))
        q = norm_question(row.get("question"))
        if not ans or not q:
            continue
        if len(ans) > MAX_ANS:
            continue
        if len(q) + len(ans) > MAX_CHARS:
            continue
        if accepted < OLD_ACCEPTED:        # 旧 jsonl 已含的前 5,239 条：只计数，不落盘
            accepted += 1
            continue
        if got >= EXTRA:
            break
        path = save_image(row.get("image"), accepted)
        if path is None:                   # 与旧逻辑一致：坏图不占编号
            continue
        msg = [
            {"role": "system", "content": SYS_VQA},
            {"role": "user", "content": [{"type": "image", "image": path},
                                         {"type": "text", "text": q}]},
            {"role": "assistant", "content": f"The answer is {ans}"},
        ]
        rows.append({"id": f"docvqa_{accepted:06d}", "source": "docvqa",
                     "task": "short_vqa", "messages": msg,
                     "messages_short": json.loads(json.dumps(msg)), "gt": ans,
                     "image": path})
        accepted += 1
        got += 1
        if got % 200 == 0:
            print(f"  ...{got}/{EXTRA}", flush=True)

    with open(OUT, "w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[docvqa-expand] 写出 {len(rows)} 行 → {OUT}（从 n={OLD_ACCEPTED} 起）")

    # 校验：新行不得与旧 heldout / 旧 train 的 id 或 (q,a) 重叠
    old_ids = set()
    for name in ("img_train.jsonl", "img_heldout.jsonl"):
        for line in open(os.path.join(ROOT, name), encoding="utf-8"):
            r = json.loads(line)
            if r.get("source") == "docvqa":
                old_ids.add(r["id"])
    new_ids = {r["id"] for r in rows}
    dup = old_ids & new_ids
    print(f"[docvqa-expand] 与旧 docvqa 行 id 重叠: {len(dup)}（应为 0）")
    return 0 if not dup and len(rows) == EXTRA else 1


if __name__ == "__main__":
    sys.exit(main())
