#!/usr/bin/env python
"""把 Qwen3.5-4B 评测对齐的多模态数据集，转成 vlm8b 训练用 JSONL。

用法：
  python build_img_eval_mix.py --check     # 每源打印 2 条样例，人工核对字段
  python build_img_eval_mix.py --run      # 正式抽取（写 jsonl，可中断续跑）

输出（远端）：/root/autodl-tmp/vlm8b/data_eval/img_train.jsonl、img_heldout.jsonl、img_stats.json
图片：      /root/autodl-tmp/vlm8b/data_eval/img/<source>/<n>.jpg（最长边 640，JPEG q90）
"""
import argparse
import json
import os
import random
import re
import sys
import traceback

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = "/root/autodl-tmp/vlm8b/data_eval"
IMG_DIR = os.path.join(ROOT, "img")
MAX_CHARS = 1400          # prompt+answer 交给 tokenizer 前先按字符粗筛（<=1024 token）
MAX_ANS = 60              # 答案字符上限
MAX_SIDE = 640

SYS_MCQ = ('你是多模态答题助手。结合图像与题目，只回复 "The answer is <选项字母>"，不要解释。')
SYS_VQA = ('你是多模态问答助手。结合图像与题目，给出最简短的答案，只回复 "The answer is <答案>"，不要解释。')

# 每源：(train 配额, heldout 配额, task)
QUOTA = {
    "docvqa":      (4500, 800, "short_vqa"),
    "textvqa":     (4000, 1000, "short_vqa"),
    "chartqa":     (1500, 1000, "short_vqa"),
    "mmstar":      (1000, 500, "mcq_img"),
    "mathvista":   (700, 300, "short_vqa"),
    "mathvision":  (2000, 1000, "short_vqa"),
    "we_math":     (1200, 500, "short_vqa"),
    "mmmu":        (600, 300, "mcq_img"),
    "docvqa_train": (3000, 0, "short_vqa"),
    "textvqa_train": (2000, 0, "short_vqa"),
}

random.seed(42)

import time


def load_retry(fn, tries=10):
    """hf-mirror 的 api 限流（429）很常见：指数退避重试，别把整源判死。"""
    for t in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                wait = 60 + 20 * t
                print(f"    429 backoff {wait}s (try {t+1}/{tries})", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("load_retry exhausted")


def norm_ans(text, is_mcq=False):
    """答案清洗：去换行/多余空格；选择题取第一个大写字母。"""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    s = str(text).replace("\u00a0", " ").strip()
    s = s.split("\n")[0].strip()
    if is_mcq:
        m = re.search(r"\b([A-J])\b", s.upper())
        return m.group(1) if m else ""
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip("。.") if len(s) > 1 else s
    return s


def norm_question(text):
    s = str(text or "")
    s = re.sub(r"<image\s*\d?>", "", s)          # MMMU 的 <image 1>
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fmt_options(options):
    """options 可能是 list / str(list) / 带字母前缀的字符串。返回 A. xxx 逐行文本。"""
    if options is None:
        return ""
    if isinstance(options, str):
        s = options.strip()
        if s.startswith("["):
            try:
                options = json.loads(s.replace("'", '"'))
            except Exception:
                options = [x.strip() for x in s.strip("[]").split(",")]
        else:
            return s
    out = []
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        opt = str(opt).strip()
        if re.match(r"^[A-J][.、)]", opt):
            out.append(opt)
        else:
            out.append(f"{letter}. {opt}")
    return "\n".join(out)


class Sink:
    """负责落图 + 拼行 + 写 jsonl（增量追加，可续跑）。"""

    def __init__(self, run: bool):
        self.run = run
        self.count = {k: {"train": 0, "heldout": 0} for k in QUOTA}
        self.drop = {}
        os.makedirs(IMG_DIR, exist_ok=True)
        self.fh = {}
        if run:
            for split in ("train", "heldout"):
                path = os.path.join(ROOT, f"img_{split}.jsonl")
                self.fh[split] = open(path, "w", encoding="utf-8")

    def _save_image(self, image, source, n):
        if isinstance(image, dict):
            image = image.get("bytes") and Image.open(__import__("io").BytesIO(image["bytes"])) or None
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
        rel = os.path.join("img", source, f"{n:06d}.jpg")
        os.makedirs(os.path.join(IMG_DIR, source), exist_ok=True)
        im.save(os.path.join(ROOT, rel), "JPEG", quality=90)
        return os.path.join(ROOT, rel)

    def emit(self, source, n, image, question, answer, split="train", is_mcq=False):
        task = QUOTA[source][2]
        ans = norm_ans(answer, is_mcq=is_mcq or task == "mcq_img")
        q = norm_question(question)
        if not ans or not q:
            self._skip(source, "empty_ans_or_q")
            return False
        if len(ans) > MAX_ANS:
            self._skip(source, "ans_too_long")
            return False
        if len(q) + len(ans) > MAX_CHARS:
            self._skip(source, "text_too_long")
            return False
        row = {
            "id": f"{source}_{n:06d}",
            "source": source,
            "task": task,
            "messages": [
                {"role": "system", "content": SYS_MCQ if task == "mcq_img" else SYS_VQA},
                {"role": "user", "content": [
                    {"type": "image", "image": os.path.join(ROOT, "img", source, f"{n:06d}.jpg")},
                    {"type": "text", "text": q}]},
                {"role": "assistant", "content": f"The answer is {ans}"},
            ],
            "gt": ans,
        }
        row["image"] = row["messages"][1]["content"][0]["image"]
        row["messages_short"] = row["messages"]
        if not self.run:                      # check 模式只打印
            print(f"  [{source}#{n}] q={q[:90]!r} a={ans!r} img={getattr(image, 'size', None)}")
            return True
        path = self._save_image(image, source, n)
        if path is None:
            self._skip(source, "bad_image")
            return False
        row["image"] = path
        row["messages"][1]["content"][0]["image"] = path
        row["messages_short"] = row["messages"]
        self.fh[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count[source][split] += 1
        return True

    def _skip(self, source, why):
        d = self.drop.setdefault(source, {})
        d[why] = d.get(why, 0) + 1

    def close(self):
        for f in self.fh.values():
            f.close()
        stats = {"counts": self.count, "drops": self.drop,
                 "note": "train/heldout 配额见 QUOTA；新增源请同步 PLAN_eval_aligned_mixture.md"}
        path = os.path.join(ROOT, "img_stats.json")
        os.makedirs(ROOT, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(json.dumps(stats, ensure_ascii=False, indent=2))


def _take(ds, n):
    for i, row in enumerate(ds):
        if i >= n:
            return
        yield i, row


def src_docvqa(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True))
    n = 0
    for i, row in _take(ds, n_tr + n_ho):
        ans = row.get("answers") or row.get("answer")
        split = "train" if i < n_tr else "heldout"
        if sink.emit("docvqa", n, row["image"], row["question"], ans, split):
            n += 1


def src_textvqa(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("lmms-lab/TextVQA", split="validation", streaming=True))
    n = 0
    for i, row in _take(ds, n_tr + n_ho):
        ans = row.get("answers") or row.get("answer")
        split = "train" if i < n_tr else "heldout"
        if sink.emit("textvqa", n, row["image"], row["question"], ans, split):
            n += 1


def src_chartqa(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("lmms-lab/ChartQA", split="test", streaming=True))
    n = 0
    for i, row in _take(ds, n_tr + n_ho):
        split = "train" if i < n_tr else "heldout"
        if sink.emit("chartqa", n, row["image"], row["question"], row["answer"], split):
            n += 1


def src_mmstar(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("Lin-Chen/MMStar", split="val", streaming=True))
    n = 0
    for i, row in _take(ds, n_tr + n_ho):
        split = "train" if i < n_tr else "heldout"
        if sink.emit("mmstar", n, row["image"], row["question"], row["answer"], split):
            n += 1


def src_mathvista(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("AI4Math/MathVista", split="testmini", streaming=True))
    n = 0
    for i, row in _take(ds, n_tr + n_ho):
        q = row.get("query") or row.get("question") or ""
        choices = row.get("choices")
        if choices:
            q = f"{q}\n{fmt_options(choices)}"
        split = "train" if i < n_tr else "heldout"
        if sink.emit("mathvista", n, row["decoded_image"], q, row["answer"], split):
            n += 1


def src_mathvision(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("MathLLMs/MathVision", split="test", streaming=True))
    n = 0
    for i, row in _take(ds, (n_tr + n_ho) * 2):
        if n >= n_tr + n_ho:
            break
        q = norm_question(row["question"])
        opts = row.get("options")
        if opts:
            q = f"{q}\n{fmt_options(opts)}"
        split = "train" if n < n_tr else "heldout"
        if sink.emit("mathvision", n, row["decoded_image"], q, row["answer"], split):
            n += 1


def src_we_math(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("We-Math/We-Math", split="testmini", streaming=True))
    n = 0
    for i, row in _take(ds, (n_tr + n_ho) * 2):
        if n >= n_tr + n_ho:
            break
        opts = str(row.get("option") or "").strip()
        if opts:
            opts = "\n".join(x.strip() for x in opts.split(";") if x.strip())
        q = f"{norm_question(row['question'])}\n{opts}" if opts else norm_question(row["question"])
        split = "train" if n < n_tr else "heldout"
        if sink.emit("we_math", n, row["image_path"], q, row["answer"], split, is_mcq=True):
            n += 1


MMMU_SUBJECTS = ["Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
                 "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
                 "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
                 "Engineering", "Finance", "Geography", "History", "Literature", "Manage", "Marketing",
                 "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics",
                 "Psychology", "Public_Health", "Sociology"]


def src_docvqa_train(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("lmms-lab/DocVQA", "DocVQA", split="train", streaming=True))
    n = 0
    for i, row in _take(ds, n_tr * 2):
        if n >= n_tr:
            break
        if sink.emit("docvqa_train", n, row["image"], row["question"],
                     row.get("answers") or row.get("answer"), "train"):
            n += 1


def src_textvqa_train(sink, n_tr, n_ho):
    ds = load_retry(lambda: load_dataset("lmms-lab/TextVQA", split="train", streaming=True))
    n = 0
    for i, row in _take(ds, n_tr * 2):
        if n >= n_tr:
            break
        if sink.emit("textvqa_train", n, row["image"], row["question"],
                     row.get("answers") or row.get("answer"), "train"):
            n += 1


def src_mmmu(sink, n_tr, n_ho):
    n = 0
    for subj in MMMU_SUBJECTS:
        if n >= n_tr + n_ho:
            break
        try:
            ds = load_retry(lambda: load_dataset("MMMU/MMMU", subj, split="validation", streaming=True))
        except Exception as e:                                   # 单个学科失败不影响整体
            sink._skip("mmmu", f"load_fail_{subj}")
            print(f"  mmmu {subj} load fail: {e}", file=sys.stderr)
            continue
        for row in ds:
            if n >= n_tr + n_ho:
                break
            q = norm_question(row.get("question"))
            opts = row.get("options")
            if opts:
                q = f"{q}\n{fmt_options(opts)}"
            split = "train" if n < n_tr else "heldout"
            if sink.emit("mmmu", n, row.get("image_1"), q, row.get("answer"), split):
                n += 1


SRC_FN = {
    "docvqa": src_docvqa, "textvqa": src_textvqa, "chartqa": src_chartqa,
    "mmstar": src_mmstar, "mathvista": src_mathvista, "mathvision": src_mathvision,
    "we_math": src_we_math, "mmmu": src_mmmu,
    "docvqa_train": src_docvqa_train, "textvqa_train": src_textvqa_train,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default="", help="逗号分隔的源名")
    args = ap.parse_args()
    if not (args.check or args.run):
        ap.error("给 --check 或 --run")
    only = [s for s in args.only.split(",") if s] or list(QUOTA)
    sink = Sink(run=args.run)
    for name in only:
        n_tr, n_ho, _ = QUOTA[name]
        print(f"== {name} (train {n_tr} / heldout {n_ho})", flush=True)
        try:
            if args.check:
                SRC_FN[name](sink, 2, 0)          # 只打印 2 条
            else:
                SRC_FN[name](sink, n_tr, n_ho)
        except Exception as e:
            print(f"!! {name} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sink._skip(name, f"exception_{type(e).__name__}")
        if args.run:
            print(f"   done: {sink.count[name]}", flush=True)
    sink.close()


if __name__ == "__main__":
    main()
