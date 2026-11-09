"""
预算筛查分析 —— 逐样本 CE/acc 落盘 → 分源/分象限/分特征表 + 相关分析
=====================================================================

输入：`eval_dump_per_sample.py` 产出的逐行 jsonl（可多份，打 tag）。
输出：`analysis_screen.md` + `analysis_screen.json`（供报告引用）。

记号（**增益都定义成「CE 降低」= 正数更好**）：

  · `ce_k1` / `ce_k8`          normal 模式 k=1 / k=8 的 answer-only CE
  · `budget_gain = ce_k1 − ce_k8`        多给视觉 token 的收益
  · `budget_gain_k2 = ce_k1 − ce_k2`     只加第 2 个 token 的收益（边际）
  · `vision_gain = ce_null − ce_normal`  有视觉 vs 视觉置零（图到底有没有用）
  · `spec_gain  = ce_shuffle − ce_normal` 对图 vs 错图（是否真的在看这张图）
  · `mean_cost  = ce_mean − ce_normal`   8 个 token 取均值 vs 原始（≈0 ⇒ token 冗余）
  · `repeat_cost= ce_repeat − ce_normal` 只用第 1 个 token 复制 8 份 vs 原始

用法::

    python analyze_screen.py --dump heldout=/tmp/screen_heldout.jsonl \
        --dump train=/tmp/screen_train.jsonl --dump in100=/tmp/screen_in100.jsonl \
        --out-dir doc/2026-11-09/screen
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

IGNORE = -100


# ═══════════════════════════════════════════════════════════════
# 读取 / 派生指标
# ═══════════════════════════════════════════════════════════════

def load_dump(path: str, tag: str) -> pd.DataFrame:
    """逐行 jsonl → DataFrame（增益统一成「CE 降低」为正）。"""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    records: List[Dict[str, object]] = []
    for row in rows:
        ce_n, ce_null = row["ce"]["normal"], row["ce"]["null"]
        ce_sh, ce_mean = row["ce"]["shuffle"], row["ce"]["mean"]
        ce_rep = row["ce"]["repeat"]
        record = {
            "dataset": tag, "id": row["id"], "source": row["source"],
            "task": row["task"], "blank": bool(row["blank"]),
            "answer_len": row["answer_len"], "prompt_len": row["prompt_len"],
            "question_chars": row["question_chars"], "has_digit": bool(row["has_digit"]),
            "vis_cos": row["vis_cos_offdiag"], "vis_rank": row["vis_eff_rank"],
            "z_rank": row["z_eff_rank"],
            "ce_k1": ce_n[0], "ce_k2": ce_n[1], "ce_k8": ce_n[-1],
            "ce_null_k8": ce_null[-1], "ce_shuffle_k8": ce_sh[-1],
            "ce_mean_k8": ce_mean[-1], "ce_repeat_k8": ce_rep[-1],
            "budget_gain": ce_n[0] - ce_n[-1],
            "budget_gain_k2": ce_n[0] - ce_n[1],
            "vision_gain_k1": ce_null[0] - ce_n[0],
            "vision_gain_k8": ce_null[-1] - ce_n[-1],
            "spec_gain_k8": ce_sh[-1] - ce_n[-1],
            "mean_cost": ce_mean[-1] - ce_n[-1],
            "repeat_cost": ce_rep[-1] - ce_n[-1],
            "acc_k1": int(row["correct"]["1"]), "acc_k8": int(row["correct"]["8"]),
            "pred_k1": row["pred"]["1"], "pred_k8": row["pred"]["8"],
            "gt": row["gt"],
        }
        for index, value in enumerate(ce_n):
            record[f"ce_k{index + 1}"] = value
        records.append(record)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# 表格
# ═══════════════════════════════════════════════════════════════

def group_table(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """分组均值表（n / CE / 各类增益 / acc）。"""
    grouped = df.groupby(by, dropna=False)
    table = grouped.agg(
        n=("id", "size"),
        ce_k1=("ce_k1", "mean"), ce_k8=("ce_k8", "mean"),
        budget=("budget_gain", "mean"),
        budget_k2=("budget_gain_k2", "mean"),
        vision=("vision_gain_k8", "mean"),
        spec=("spec_gain_k8", "mean"),
        mean_cost=("mean_cost", "mean"),
        repeat=("repeat_cost", "mean"),
        acc1=("acc_k1", "mean"), acc8=("acc_k8", "mean"),
        vis_cos=("vis_cos", "mean"),
    ).reset_index()
    table["d_acc"] = table["acc8"] - table["acc1"]
    return table


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def table_md(table: pd.DataFrame, first: str = "source") -> str:
    """DataFrame → markdown 表；`first` 是分组列（bin 时按区间字符串输出）。"""
    columns = [first] + [c for c in
                         ["task", "blank", "n", "ce_k1", "ce_k8", "budget", "budget_k2",
                          "vision", "spec", "mean_cost", "repeat", "acc1", "acc8",
                          "d_acc"] if c in table.columns and c != first]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column == "n":
                cells.append(str(int(value)))
            elif column in ("blank",):
                cells.append("txt" if value else "img")
            elif column in ("acc1", "acc8", "d_acc"):
                cells.append(f"{value * 100:.1f}%")
            elif column in ("ce_k1", "ce_k8"):
                cells.append(f"{value:.3f}")
            elif column == first or column in (
                    "source", "task", "bin", "answer_len", "question_chars", "ce_k1"):
                cells.append(str(value))
            else:
                cells.append(_fmt(float(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def curve_table(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """每个 k 的平均 CE（看曲线在哪一档饱和/拐点）。"""
    columns = [f"ce_k{k}" for k in range(1, 9)]
    table = df.groupby(by, dropna=False)[columns].mean().reset_index()
    table.columns = list(by) + [f"k{k}" for k in range(1, 9)]
    table["argmin_k"] = table[[f"k{k}" for k in range(1, 9)]].values.argmin(axis=1) + 1
    return table


def spearman(df: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    """Spearman 相关（去 NaN）。"""
    pair = df[[x, y]].dropna()
    if len(pair) < 10:
        return {"rho": float("nan"), "p": float("nan"), "n": len(pair)}
    rho, p_value = stats.spearmanr(pair[x], pair[y])
    return {"rho": float(rho), "p": float(p_value), "n": int(len(pair))}


def binned_table(df: pd.DataFrame, column: str, bins: List[float],
                 label: str) -> pd.DataFrame:
    """按 column 分箱看 budget/vision 增益。"""
    frame = df.copy()
    frame["bin"] = pd.cut(frame[column], bins=bins, include_lowest=True)
    return group_table(frame, ["bin"]).rename(columns={"bin": label})


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析参数。"""
    p = argparse.ArgumentParser(description="budget screening analysis")
    p.add_argument("--dump", action="append", required=True,
                   help="tag=path，可重复")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> None:
    """跑分析并写 md/json。"""
    args = parse_args()
    frames = []
    for spec in args.dump:
        tag, path = spec.split("=", 1)
        frame = load_dump(path, tag)
        print(f"[load] {tag}: {len(frame)} rows")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    lines: List[str] = ["# vlm8b 预算筛查（逐样本）", ""]
    summary: Dict[str, object] = {}

    # ── 0) 总览 ──────────────────────────────────────────────
    lines += ["## 0. 总览", ""]
    overview = group_table(data, ["dataset", "blank"])
    lines += [table_md(overview, first="dataset"), ""]
    summary["overview"] = overview.to_dict(orient="records")

    for tag, frame in data.groupby("dataset"):
        lines += [f"## 1. `{tag}` 分源表（按 budget_gain 排序）", ""]
        table = group_table(frame, ["source", "task", "blank"])
        table = table.sort_values("budget", ascending=False)
        lines += [table_md(table), ""]
        summary[f"{tag}_by_source"] = table.to_dict(orient="records")

        lines += [f"### `{tag}` 每档 k 的平均 CE（曲线饱和点）", ""]
        curve = curve_table(frame, ["source", "blank"])
        curve["blank"] = curve["blank"].map({True: "txt", False: "img"})
        header = ["source", "kind"] + [f"k{k}" for k in range(1, 9)] + ["argmin_k"]
        lines += ["| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] * len(header)) + "|"]
        for _, row in curve.iterrows():
            cells = [str(row["source"]), str(row["blank"])]
            cells += [f"{row[f'k{k}']:.3f}" for k in range(1, 9)]
            cells.append(str(int(row["argmin_k"])))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        summary[f"{tag}_curve"] = curve.to_dict(orient="records")

        # 源内 rho(budget_gain, vision_gain)：预算收益是否只出现在"图有用"的行
        lines += [f"### `{tag}` 源内 Spearman(budget_gain, vision_gain)", ""]
        lines += ["| source | rho | p | n |", "|---|---|---|---|"]
        within = {}
        for (source, blank), group in frame.groupby(["source", "blank"], dropna=False):
            stat = spearman(group, "budget_gain", "vision_gain_k8")
            within[str(source)] = stat
            lines.append(f"| {source}{'/txt' if blank else ''} | {stat['rho']:+.3f} | "
                         f"{stat['p']:.1e} | {stat['n']} |")
        lines.append("")
        summary[f"{tag}_within_source_corr"] = within

        # 图像行 vs 文本行
        image_rows = frame[~frame["blank"]]
        text_rows = frame[frame["blank"]]
        if len(text_rows) > 0 and len(image_rows) > 0:
            lines += [f"### `{tag}` 图/文行对比", ""]
            compare = pd.DataFrame([
                {"kind": "image", **{k: float(v) for k, v in {
                    "n": len(image_rows),
                    "ce_k1": image_rows["ce_k1"].mean(),
                    "ce_k8": image_rows["ce_k8"].mean(),
                    "budget": image_rows["budget_gain"].mean(),
                    "vision": image_rows["vision_gain_k8"].mean(),
                    "spec": image_rows["spec_gain_k8"].mean(),
                    "mean_cost": image_rows["mean_cost"].mean(),
                    "acc1": image_rows["acc_k1"].mean(),
                    "acc8": image_rows["acc_k8"].mean()}.items()}},
                {"kind": "text", **{k: float(v) for k, v in {
                    "n": len(text_rows),
                    "ce_k1": text_rows["ce_k1"].mean(),
                    "ce_k8": text_rows["ce_k8"].mean(),
                    "budget": text_rows["budget_gain"].mean(),
                    "vision": text_rows["vision_gain_k8"].mean(),
                    "spec": text_rows["spec_gain_k8"].mean(),
                    "mean_cost": text_rows["mean_cost"].mean(),
                    "acc1": text_rows["acc_k1"].mean(),
                    "acc8": text_rows["acc_k8"].mean()}.items()}},
            ])
            compare["d_acc"] = compare["acc8"] - compare["acc1"]
            lines += [table_md(compare, first="kind"), ""]
            summary[f"{tag}_img_vs_text"] = compare.to_dict(orient="records")

        # 象限：视觉有没有用 × 预算有没有用
        lines += [f"### `{tag}` 四象限（vision_gain × budget_gain，行占比）", ""]
        frame = frame.copy()
        frame["vision_used"] = frame["vision_gain_k8"] > 0.02
        frame["budget_used"] = frame["budget_gain"] > 0.02
        quad = frame.groupby(["vision_used", "budget_used"]).size().unstack(fill_value=0)
        lines += [str(quad), ""]
        summary[f"{tag}_quadrant"] = {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                                      for k, v in quad.to_dict(orient="index").items()}

        # 相关
        corr = {}
        for target in ("vision_gain_k8", "ce_k1", "answer_len", "question_chars",
                       "vis_cos", "vis_rank"):
            corr[target] = spearman(frame, "budget_gain", target)
        lines += [f"### `{tag}` budget_gain 的 Spearman 相关", ""]
        lines += ["| x | rho | p | n |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['rho']:+.3f} | {v['p']:.2e} | {v['n']} |"
                  for k, v in corr.items()]
        lines.append("")
        summary[f"{tag}_corr"] = corr

        # 分箱
        if len(image_rows) >= 50:
            lines += [f"### `{tag}` 图像行分箱（answer_len / question_chars / ce_k1）", ""]
            for column, bins, label in (
                    ("answer_len", [0, 2, 4, 8, 16, 1e9], "answer_len 区间"),
                    ("question_chars", [0, 100, 200, 400, 1e9], "题面长度区间"),
                    ("ce_k1", [0, 0.1, 0.3, 0.7, 1.5, 1e9], "k1 CE 区间")):
                binned = binned_table(image_rows, column, bins, label)
                binned[label] = binned[label].astype(str)
                lines += [table_md(binned, first=label), ""]
                summary[f"{tag}_bins_{column}"] = binned.to_dict(orient="records")

    with open(os.path.join(args.out_dir, "analysis_screen.md"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_dir, "analysis_screen.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[write] {args.out_dir}/analysis_screen.md")


if __name__ == "__main__":
    main()
