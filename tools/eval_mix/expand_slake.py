#!/usr/bin/env python
"""SLAKE 训练行扩容：5,000 → 6,000 行（heldout 1,000 行完全不动）。

旧构建（_work/build_extra.py::src_slake）逻辑：train 池 = train.json + validation.json，
shuffle(seed=42) 后取前 5,000；heldout 池 = test.json 取前 1,000；编号跨 train+heldout
连续 ⇒ 旧 train id 0–4999，旧 heldout id 5000–5999。本脚本复用同一 shuffle，取
train 池第 5,001–6,000 条，编号从 6,000 起，写 `expand_slake_train.jsonl`。
"""
import json
import os
import sys

WORK = "/root/autodl-tmp/vlm8b/data_eval/_work"
sys.path.insert(0, WORK)
import build_extra as B  # noqa: E402  （模块有 __main__ 保护，导入安全）

ROOT = "/root/autodl-tmp/vlm8b/data_eval"
OUT = os.path.join(ROOT, "expand_slake_train.jsonl")
OLD_TRAIN = 5000
OLD_HELDOUT = 1000
NEW_TOTAL = 6000


def main() -> int:
    img_root = os.path.join(B.W, "slake", "imgs")
    pool_tr = []
    for name in ("train.json", "validation.json"):
        rows = json.load(open(os.path.join(B.W, "slake", name)))
        for r in rows:
            ans = B.short_answer(r.get("answer", ""))
            if ans is None:
                continue
            q = str(r.get("question") or "").strip()
            img = str(r.get("img_name") or "")
            path = os.path.join(img_root, img)
            if not q or not img or not os.path.exists(path):
                continue
            pool_tr.append(("short_vqa", q, ans, path))
    print(f"[slake-expand] train 池 {len(pool_tr)} 条")
    rng = B.new_rng()          # random.Random(SEED=42)，与旧构建同种子同顺序
    rng.shuffle(pool_tr)

    st = B.new_stats("short_vqa")
    rows = []
    n = OLD_TRAIN + OLD_HELDOUT          # 编号从 6,000 起（旧 heldout 占 5,000–5,999）
    for task, text, ans, img in pool_tr[OLD_TRAIN:NEW_TOTAL]:
        sample = B.make_sample("slake", n, task, text, ans, img, st)
        if sample is None:
            continue
        rows.append(sample)
        n += 1
    with open(OUT, "w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[slake-expand] 写出 {len(rows)} 行 → {OUT}（id 从 {OLD_TRAIN + OLD_HELDOUT} 起）")

    # 校验 1：新行 id 不得与旧 slake 行重叠
    old_ids = set()
    for name in ("img_extra_train.jsonl", "img_extra_heldout.jsonl"):
        for line in open(os.path.join(ROOT, name), encoding="utf-8"):
            r = json.loads(line)
            if r.get("source") == "slake":
                old_ids.add(r["id"])
    new_ids = {r["id"] for r in rows}
    print(f"[slake-expand] 与旧 slake id 重叠: {len(old_ids & new_ids)}（应为 0）")

    # 校验 2：新行问答不得与旧 heldout 重复（若 shuffle 顺序不符会在这里暴露）
    held = set()
    for line in open(os.path.join(ROOT, "img_extra_heldout.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") == "slake":
            held.add((r["gt"], r["messages"][1]["content"][1]["text"]))
    clash = sum(1 for r in rows if (r["gt"], r["messages"][1]["content"][1]["text"]) in held)
    print(f"[slake-expand] 与旧 heldout 问答重复: {clash}（应为 0）")

    # 校验 3：图片文件数 > 旧数量，且新图都存在
    missing = sum(1 for r in rows if not os.path.exists(r["image"]))
    print(f"[slake-expand] 图片缺失: {missing}（应为 0）")
    return 0 if (len(rows) == NEW_TOTAL - OLD_TRAIN and not old_ids & new_ids
                 and clash == 0 and missing == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
