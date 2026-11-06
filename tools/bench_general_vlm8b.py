"""通用能力小bench：弱智吧式脑筋急转弯 / 算术 / 常识，对比 base Qwen3.5-4B 与 vlm8b 微调后的 LM。

用法（remote）::

    # 两个都跑（各自加载一份 4B，建议分卡）
    CUDA_VISIBLE_DEVICES=1 python bench_general_vlm8b.py \
        --qwen-dir /root/qwen_models/Qwen3.5-4B \
        --checkpoint /root/autodl-tmp/vlm8b/output/run1/final --which both

    # 只跑 base / 只跑微调
    ... --which base / ours

设计：只走文本通道（不喂图）。base 用 Qwen3_5ForConditionalGeneration 原始权重；
ours 用 VLM8Budget 载入 ckpt 后取 `.lm`（同一套输入 id / 生成参数 / 打分规则）。
"""
import argparse
import json
import re
import sys
import os

import torch

# ── 题库（每项：category, q, answer；answer 用 _norm 包含匹配）────────────────
BANK = [
    # 弱智吧式 / 脑筋急转弯（20）
    ("弱智吧", "树上停着十只鸟，猎人开枪打死了一只，树上还剩几只鸟？", "0"),
    ("弱智吧", "什么门永远关不上？", "球门"),
    ("弱智吧", "什么东西越洗越脏？", "水"),
    ("弱智吧", "一年中有几个月份包含28天？", "12"),
    ("弱智吧", "什么东西倒立过来以后数值会增加一半？", "6"),
    ("弱智吧", "什么东西天气越热，它爬得越高？", "温度计"),
    ("弱智吧", "什么布剪不断？", "瀑布"),
    ("弱智吧", "什么蛋不能吃？", "脸蛋"),
    ("弱智吧", "什么东西有五个头，但人们一点都不觉得奇怪？", "手"),
    ("弱智吧", "什么东西只能增加不能减少？", "年龄"),
    ("弱智吧", "什么车子寸步难行？", "风车"),
    ("弱智吧", "什么东西打破了才能用？", "鸡蛋"),
    ("弱智吧", "谁天天都要去看病？", "医生"),
    ("弱智吧", "什么东西有面没有口，有脚没有手？", "桌子"),
    ("弱智吧", "什么东西有嘴却不能说话？", "茶壶"),
    ("弱智吧", "什么书在书店里买不到？", "秘书"),
    ("弱智吧", "什么东西看不见摸不着却很重要？", "空气"),
    ("弱智吧", "什么东西越擦越小？", "橡皮"),
    ("弱智吧", "什么水永远用不完？", "口水"),
    ("弱智吧", "什么东西有脚却不会走路？", "椅子"),
    # 算术（20）
    ("算术", "23 + 48 = ? 只给数字。", "71"),
    ("算术", "7 × 8 = ? 只给数字。", "56"),
    ("算术", "100 - 37 = ? 只给数字。", "63"),
    ("算术", "144 ÷ 12 = ? 只给数字。", "12"),
    ("算术", "15 × 6 = ? 只给数字。", "90"),
    ("算术", "256 + 179 = ? 只给数字。", "435"),
    ("算术", "1000 - 863 = ? 只给数字。", "137"),
    ("算术", "9 × 9 + 8 = ? 只给数字。", "89"),
    ("算术", "(36 + 12) ÷ 6 = ? 只给数字。", "8"),
    ("算术", "19 × 21 = ? 只给数字。", "399"),
    ("算术", "78 ÷ 3 = ? 只给数字。", "26"),
    ("算术", "13 + 29 + 47 = ? 只给数字。", "89"),
    ("算术", "500 ÷ 25 = ? 只给数字。", "20"),
    ("算术", "37 × 4 - 49 = ? 只给数字。", "99"),
    ("算术", "1234 + 5678 = ? 只给数字。", "6912"),
    ("算术", "100 ÷ 4 + 6 = ? 只给数字。", "31"),
    ("算术", "85 - 17 - 28 = ? 只给数字。", "40"),
    ("算术", "11^2 + 8 = ? 只给数字。", "129"),
    ("算术", "64 ÷ 8 × 9 = ? 只给数字。", "72"),
    ("算术", "999 + 1 - 500 = ? 只给数字。", "500"),
    # 常识 / 知识（20）
    ("常识", "中国的首都是哪里？只给地名。", "北京"),
    ("常识", "水的化学式是什么？", "H2O"),
    ("常识", "光年是什么物理量的单位？", "距离"),
    ("常识", "一年通常有几天？只给数字。", "365"),
    ("常识", "地球到太阳的平均距离约多少千米？只给量级（如 1.5亿）。", "1.5亿"),
    ("常识", "人体最大的器官是什么？", "皮肤"),
    ("常识", "圆周率约等于多少？给 3 位小数。", "3.14"),
    ("常识", "太阳系中体积最大的行星是哪颗？", "木星"),
    ("常识", "《红楼梦》的作者是谁？", "曹雪芹"),
    ("常识", "电灯的发明者通常被认为是？", "爱迪生"),
    ("常识", "一个标准足球队场上有多少名球员？只给数字。", "11"),
    ("常识", "珠穆朗玛峰位于哪两个国家的边界？", "中国"),
    ("常识", "DNA 的中文全称是什么？", "脱氧核糖核酸"),
    ("常识", "水的沸点在标准大气压下是多少摄氏度？只给数字。", "100"),
    ("常识", "地球上最大的海洋是？", "太平洋"),
    ("常识", "中国的陆地面积约多少万平方千米？只给数字。", "960"),
    ("常识", "联合国总部位于哪个城市？", "纽约"),
    ("常识", "一年中有几个季节？只给数字。", "4"),
    ("常识", "黄金的化学元素符号是什么？", "Au"),
    ("常识", "光在真空中的速度约多少万千米每秒？只给数字。", "30"),
]

SYSTEM = "你是一个严谨的中文助手。只输出最终答案，不要解释过程，回答尽量简短。"


def _norm(text: str) -> str:
    return re.sub(r"[\s，。！？、；：\"'（）()\[\]【】]", "", str(text)).lower()


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


@torch.no_grad()
def run_batch(tokenizer, lm, questions, max_new_tokens: int = 48):
    """逐条贪心生成（用 input_ids 走纯文本通道）。"""
    outs = []
    for q in questions:
        prompt = build_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(lm.device)
        generated = lm.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        outs.append(text)
    return outs


def score(outs, bank):
    cats = {}
    details = []
    for out, (cat, q, ans) in zip(outs, bank):
        ok = _norm(ans) in _norm(out)
        cats.setdefault(cat, [0, 0])
        cats[cat][0] += int(ok)
        cats[cat][1] += 1
        details.append({"category": cat, "q": q, "answer": ans, "pred": out, "correct": ok})
    return {c: {"acc": v[0] / v[1], "n": v[1]} for c, v in cats.items()}, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", default="/root/qwen_models/Qwen3.5-4B")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--which", default="both", choices=["base", "ours", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bank = BANK[:args.limit] if args.limit else BANK
    questions = [q for _, q, _ in bank]
    results = {}

    if args.which in ("base", "both"):
        from transformers import Qwen3_5ForConditionalGeneration
        lm = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.qwen_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
            trust_remote_code=True, local_files_only=True).cuda().eval()
        outs = run_batch(tokenizer, lm, questions)
        cats, details = score(outs, bank)
        results["base"] = {"by_category": cats, "details": details}
        del lm
        torch.cuda.empty_cache()

    if args.which in ("ours", "both"):
        from model import VLM8Budget, VLM8BudgetConfig
        config = VLM8BudgetConfig.from_pretrained(args.checkpoint)
        model = VLM8Budget(config, grad_ckpt=False)
        from eval import load_trained_model
        model = load_trained_model(args.checkpoint, config)
        lm = model.lm
        outs = run_batch(tokenizer, lm, questions)
        cats, details = score(outs, bank)
        results["ours"] = {"by_category": cats, "details": details}

    print("\n=== 通用能力小bench（文本通道，不含图）===")
    cats = [c for c in ("弱智吧", "算术", "常识")]
    print(f"{'category':8s}" + "".join(f" {m:>10s}" for m in results))
    for c in cats:
        row = f"{c:8s}"
        for m in results:
            row += f" {results[m]['by_category'].get(c, {}).get('acc', 0) * 100:9.1f}%"
        print(row)
    for m in results:
        total = sum(v["acc"] * v["n"] for v in results[m]["by_category"].values())
        n = sum(v["n"] for v in results[m]["by_category"].values())
        print(f"{m}: overall {total / n * 100:.1f}% (n={n})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"写 {args.out}")


if __name__ == "__main__":
    main()
