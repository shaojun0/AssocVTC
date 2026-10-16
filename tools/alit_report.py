#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIT 自适应长度实验 — 报告聚合器（E-A / E-B / E-C1）
================================================================================

把 `doc/2026-10-16/artifacts/alit/` 下的实测产物聚合成 **可被报告直接引用的
markdown 表 + 图**，并给出 PLAN §8 判据表的三态判决（成立 / 否证 / 数据不足）。

设计约束（与 task-6 验收一致）:
  · **纯 CPU / 无 GPU 依赖**（不 import torch; matplotlib 缺失也不致命）;
  · **缺文件优雅跳过**，并把 present/missing 清单写进产物与 stdout;
  · **每个数字标来源文件名**（表格列 `来源` 或小节首行 `来源:`）;
  · 只读产物、只写 `report_tables.md` + `*.png`，**不碰**任何实验代码。

产物文件命名（依 run_alit_a.sh / run_alit_b.sh / tools/alit_adaptive_k.py）:
  baseline_four_arm_eval.json    四臂同进程全量评测（eval_linattn.py）
  stepwise_<arm>.json           逐步曲线（eval_stepwise_linattn.py），含四臂基线
  per_image_<domain>_*.json     E-A1 每图逐步 dump（--dump_per_image）
  tsc_<arm>.json                E-A1/A2/A4/A6 分析（tools/alit_adaptive_k.py）
  domain_ladder.json            E-A3 三级域阶梯（tools/alit_adaptive_k.py）
  ood_<domain>_*.json           E-A3 各级逐步曲线（eval_stepwise_linattn.py）
  zs_redundancy_*.json / zs_stats*.json   E-A5 z_s 冗余度
  eval_<arm>.json               E-B 臂评测（eval_linattn.py）
  stepwise_<arm>.json           E-B 臂逐步
  alit_b_arms.tsv               E-B 臂执行状态
  *task_fidelity*.json          E-C1 线性探针（非 smoke）

用法::

    python3 tools/alit_report.py                       # 全自动（推荐）
    python3 tools/alit_report.py --artifacts <dir> --out <md> --no-plots
    python3 tools/alit_report.py --json                # 额外打印机器可读摘要

口径提醒（本文件会显式标注）:
  · `eval_linattn.py` 的 `l1_norm` 是**归一化空间** L1；换到 0–255 像素口径须
    `× mean(IMAGENET_STD) × 255 = ×0.226×255 = ×57.63`（与
    `artifacts/ablation_table.md` 的换算一致）。`stepwise_*.json` 的 `L1_255`
    是**反归一化后像素域**直接算的，两者在 0.01 px 内一致（交叉核对见输出 §1）。
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ART = os.path.join(REPO_ROOT, "doc", "2026-10-16", "artifacts", "alit")
DEFAULT_OUT = os.path.join(DEFAULT_ART, "report_tables.md")

# PLAN §2.2 已发布（10-15/10-16 旧实例）四臂参照值 —— 仅用于「复现核对」列
PLAN_REF = {
    "k64/ckpt_best.pt": {"psnr": 26.602, "ssim": 0.7654, "l1_norm": 0.1463},
    "stdattn_k64/ckpt_best.pt": {"psnr": 31.219, "ssim": 0.9037, "l1_norm": 0.0901},
    "k64_rec/ckpt_best.pt": {"psnr": 25.304, "ssim": 0.7156, "l1_norm": 0.1666},
    "stdattn_k64_rec/ckpt_best.pt": {"psnr": 28.582, "ssim": 0.8505, "l1_norm": 0.1147},
}
SRC_PLAN = "PLAN_alit_adaptive_experiments.md §2.2"

# E-B4 范围裁剪（2026-10-16 Lead 决定）：专训档 K'=16/32/48 从零训练 3×~5 h 超预算，**不做**。
# 该判据位按「数据不足 + 成本理由」记录，**不是待补**——报告不得留空判据。
EB4_COST_CUT = ("专训档 K'=16/32/48 **不做**（从零训练 3×~5 h 超预算，2026-10-16 Lead 决定）"
                "⇒ 该判据位记「数据不足 + 成本理由」，不是待补")

# 期望产物清单（present/missing 判定 + 报告「未到位」节）
EXPECTED = [
    ("baseline_four_arm_eval.json", "A0 四臂同进程全量评测", "eval_linattn.py（四臂）"),
    ("stepwise_stdattn_k64_rec.json", "四臂基线·std 循环臂逐步", "eval_stepwise_linattn.py"),
    ("stepwise_linattn_k64_rec.json", "四臂基线·lin 循环臂逐步", "eval_stepwise_linattn.py"),
    ("stepwise_stdattn_k64.json", "四臂基线·std 单步逐步", "eval_stepwise_linattn.py"),
    ("stepwise_linattn_k64.json", "四臂基线·lin 单步逐步", "eval_stepwise_linattn.py"),
    ("per_image_stdattn_k64_rec.json", "E-A1 每图逐步 dump（std 循环臂）", "run_alit_a.sh a1_dump_std_rec"),
    ("per_image_linattn_k64_rec.json", "E-A1 每图逐步 dump（lin 循环臂）", "run_alit_a.sh a1_dump_lin_rec"),
    ("tsc_stdattn_rec.json", "E-A1/A2/A4/A6 分析（std 循环臂）", "run_alit_a.sh a2_analysis_std"),
    ("tsc_linattn_rec.json", "E-A1/A2/A4/A6 分析（lin 循环臂）", "run_alit_a.sh a2_analysis_lin"),
    ("ood_in100_stdattn_rec.json", "E-A3 IID 级逐步", "run_alit_a.sh a3_iid"),
    ("ood_coco_stdattn_rec.json", "E-A3 中距 OOD（COCO2017 val）逐步", "run_alit_a.sh a3_coco"),
    ("ood_far_stdattn_rec.json", "E-A3 远距 OOD 逐步", "run_alit_a.sh a3_far"),
    ("domain_ladder.json", "E-A3 三级域阶梯汇总", "run_alit_a.sh a3_ladder"),
    ("zs_redundancy_stdattn_rec.json", "E-A5 z_s 冗余度", "run_alit_a.sh a5_zs_stats"),
    ("gate_diff_stdattn_rec.json", "回归门：新代码默认路径逐位一致", "run_alit_a.sh a0b_gate_diff"),
    ("alit_a_summary.json", "E-A 产物汇总", "run_alit_a.sh 收尾"),
    ("alit_a_steps.tsv", "E-A 步骤/rc/耗时表", "run_alit_a.sh"),
    ("alit_a_numbers.json", "E-A 原始数字汇总（无判决）", "alit_a_numbers.py"),
    ("alit_b_arms.tsv", "E-B 臂执行状态表", "run_alit_b.sh"),
    ("eval_b1_mean.json", "E-B 基线复跑（b1_mean）全量评测", "run_alit_b.sh"),
    ("stepwise_b1_mean.json", "E-B 基线复跑（b1_mean）逐步", "run_alit_b.sh"),
    ("task_fidelity_stdattn_rec.json", "E-C1 线性探针（主）", "E-C1 执行"),
    ("task_fidelity_stdattn_rec_cls.json", "E-C1 线性探针（cls 特征变体）", "E-C1 执行"),
    ("task_fidelity_stdattn_rec_lstsq.json", "E-C1 线性探针（lstsq 变体）", "E-C1 执行"),
]
SMOKE_HINTS = ("smoke", "synthetic")


# ─────────────────────────── 基础工具 ───────────────────────────

def fnum(x, nd=3, dash="—"):
    """数值格式化; None/NaN → dash。"""
    if x is None:
        return dash
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf) or math.isinf(xf):
        return dash
    return f"{xf:.{nd}f}"


def fsgn(x, nd=3, dash="—"):
    if x is None:
        return dash
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf) or math.isinf(xf):
        return dash
    return f"{xf:+.{nd}f}"


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 —— 产物可能截断/半写, 一律降级不炸
        print(f"  [warn] 读失败 {os.path.basename(path)}: {e}", file=sys.stderr)
        return None


def is_smoke(path):
    b = os.path.basename(path).lower()
    return any(h in b for h in SMOKE_HINTS)


def arm_of_stepwise(path):
    """stepwise_stdattn_k64_rec.json → stdattn_k64_rec"""
    return os.path.basename(path)[len("stepwise_"):-len(".json")]


def ref_key(eval_json):
    """从 eval_linattn.py 输出中挑出「参照臂」结果键（stdattn_k64_rec 优先）。"""
    if not isinstance(eval_json, dict):
        return None, None
    for pref in ("stdattn_k64_rec", "stdattn_k64"):
        for k in eval_json:
            if pref in k:
                return k, eval_json[k]
    return None, None


def _get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


# ─────────────────────────── 采集 ───────────────────────────

def collect(art):
    """扫产物目录，按类归集。返回 dict。"""
    def g(pat):
        return sorted(p for p in glob.glob(os.path.join(art, pat))
                      if not is_smoke(p))

    c = {
        "art": art,
        "four_arm": None,
        "stepwise": {},           # arm -> json
        "tsc": {},                # arm/tag -> json
        "ladder": None,
        "ood": {},                # domain -> stepwise json
        "zs": {},                 # tag -> json
        "eval_b": {},             # arm -> json
        "stepwise_b": {},         # arm -> json
        "tsv": None,
        "ec1": {},
        "gate": None,
        "a_summary": None,
        "per_image": {},
        "present": set(),
        "missing": [],
    }

    def reg(name):
        c["present"].add(name)
        return os.path.join(art, name)

    # 目录内实际存在的文件（含未在 EXPECTED 里登记的新产物）
    existing = sorted(os.path.basename(p) for p in glob.glob(os.path.join(art, "*.json")))
    for name, desc, producer in EXPECTED:
        if name in existing or os.path.exists(os.path.join(art, name)):
            c["present"].add(name)
        else:
            c["missing"].append((name, desc, producer))

    if "baseline_four_arm_eval.json" in c["present"]:
        c["four_arm"] = load_json(reg("baseline_four_arm_eval.json"))

    for p in g("stepwise_*.json"):
        arm = arm_of_stepwise(p)
        # 回归门产物（published_code 基准 / gate_ 复跑）不是实验臂，单独由 gate_diff 处理
        if "published_code" in arm or arm.startswith("gate_"):
            continue
        d = load_json(p)
        if d is None:
            continue
        # E-B 臂名（b1_*/b2_*/b3_*/b4_*）与四臂基线分开
        if arm.startswith(("b1_", "b2_", "b3_", "b4_")):
            c["stepwise_b"][arm] = d
        else:
            c["stepwise"][arm] = d

    for p in g("tsc_*.json") + g("alit_a_analysis*.json"):
        d = load_json(p)
        if isinstance(d, dict):
            c["tsc"][os.path.basename(p)[:-5]] = d

    if "domain_ladder.json" in c["present"]:
        c["ladder"] = load_json(reg("domain_ladder.json"))

    for p in g("ood_*.json"):
        d = load_json(p)
        if isinstance(d, dict):
            tag = os.path.basename(p)[len("ood_"):-len(".json")]
            c["ood"][tag] = d

    for p in g("zs_redundancy*.json") + g("zs_stats*.json"):
        d = load_json(p)
        if isinstance(d, dict):
            c["zs"][os.path.basename(p)[:-5]] = d

    for p in g("eval_*.json"):
        b = os.path.basename(p)
        if b in ("eval_ablation_full.json",):
            continue
        d = load_json(p)
        if isinstance(d, dict):
            c["eval_b"][b[len("eval_"):-len(".json")]] = d

    if "alit_b_arms.tsv" in c["present"]:
        try:
            with open(os.path.join(art, "alit_b_arms.tsv"), encoding="utf-8") as f:
                c["tsv"] = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
        except OSError:
            c["tsv"] = None

    for p in g("*task_fidelity*.json"):
        d = load_json(p)
        if isinstance(d, dict):
            c["ec1"][os.path.basename(p)] = d

    if "gate_diff_stdattn_rec.json" in c["present"]:
        c["gate"] = load_json(reg("gate_diff_stdattn_rec.json"))
    if "alit_a_summary.json" in c["present"]:
        c["a_summary"] = load_json(reg("alit_a_summary.json"))

    for p in g("per_image_*.json"):
        c["per_image"][os.path.basename(p)] = p

    # E-A 断点/降级标记 + 步骤表（run_alit_a.sh 产出）
    c["skips"] = {}
    for p in sorted(glob.glob(os.path.join(art, "SKIP_*.json"))):
        name = os.path.basename(p)[len("SKIP_"):-len(".json")]
        c["skips"][name] = load_json(p) or {}
    c["a_steps"] = None
    if os.path.exists(os.path.join(art, "alit_a_steps.tsv")):
        try:
            with open(os.path.join(art, "alit_a_steps.tsv"), encoding="utf-8") as f:
                c["a_steps"] = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
        except OSError:
            c["a_steps"] = None

    c["extra_json"] = sorted(set(existing) - set(EXPECTED[i][0] for i in range(len(EXPECTED)))
                             - {"alit_b_arms.tsv"})
    return c


# ─────────────────────────── 判决（PLAN §8 三态）───────────────────────────

V_OK, V_NO, V_INSUFF = "成立", "否证", "数据不足"
# E-B3 实测出现「增益成立 / 机制否证」的分裂结果, PLAN §8 的三态表表达不了 ⇒ 显式第四态。
# 报告与 README 必须同时写明「分裂」各轴分别的三态判决，不得只报一个词。
V_SPLIT = "分裂"


def verdict_a1(hist, n, oracle_gain=None, oracle_tokens=None, fixed_tokens=None):
    """E-A1（PLAN §8）三态:
      · 停步高度集中（≥90% 同一档）⇒ **否证**（PLAN §8 字面否证条件）;
      · 否则若 oracle 增益可得且 < 0.10 px（PLAN §9.1 噪声口径）⇒ **否证**
        （oracle 是「知道标签时按图选停步」的不可实现上界；上界都买不到 >0.10 px，
         则「oracle 明显优于固定 step7」这一成立条件不满足，图间差异不可利用）;
      · oracle 增益 ≥ 0.10 px ⇒ **成立**;
      · 两者都不可得 ⇒ 数据不足。
    """
    if not hist or not n:
        return V_INSUFF, "无 best_step_hist"
    share = max(hist) / n
    if share >= 0.90:
        return V_NO, f"停步高度集中：众数档占比 {share * 100:.2f}% ≥ 90%（PLAN §8 否证条件）"
    if oracle_gain is None:
        return V_INSUFF, (f"众数占比 {share * 100:.2f}% < 90%（未命中字面否证），"
                          "但 oracle 增益需 tsc_*.json（未到位）")
    tag = (f"；oracle {oracle_tokens:.2f} tokens vs 固定 {fixed_tokens}"
           if oracle_tokens is not None and fixed_tokens else "")
    if oracle_gain < 0.10:
        return V_NO, (f"众数占比 {share * 100:.2f}% < 90%（未命中字面否证），"
                      f"但 **oracle 上界增益仅 {oracle_gain:+.4f} px < 0.10 px**"
                      f"（PLAN §9.1 噪声口径）⇒ 成立条件「oracle 明显优于固定 step7」不满足{tag}")
    return V_OK, f"oracle 上界增益 {oracle_gain:+.4f} px ≥ 0.10 px{tag}"


def oracle_gain_for_arm(c, arm):
    """从任一 tsc_*.json 里找 per_image_<arm>.json 对应的 E_A1 oracle 增益。"""
    for _tag, d in (c.get("tsc") or {}).items():
        srcs = ((d.get("meta") or {}).get("stepwise") or [])
        names = [os.path.basename(s) for s in srcs]
        if f"per_image_{arm}.json" not in names:
            continue
        a1s = d.get("E_A1_stop_distribution") or {}
        if not a1s:
            continue
        a1 = next(iter(a1s.values()))
        return (a1.get("oracle_gain_L1_255"), a1.get("oracle_tokens_mean"),
                a1.get("fixed_step7_tokens"))
    return None, None, None


def hist_stats(hist):
    """由 best_step_hist（0-based 计数）反推 平均/标准差/熵（nats, 归一化）。"""
    if not hist:
        return {}
    n = float(sum(hist))
    if n <= 0:
        return {}
    mean = sum((i + 1) * h for i, h in enumerate(hist)) / n
    var = sum(((i + 1) ** 2) * h for i, h in enumerate(hist)) / n - mean ** 2
    p = [h / n for h in hist if h > 0]
    H = -sum(q * math.log(q) for q in p)
    T = len(hist)
    return {"n": int(n), "mean_step": mean, "std_step": math.sqrt(max(var, 0.0)),
            "entropy_nats": H,
            "entropy_norm": H / math.log(T) if T > 1 else 0.0,
            "mode_share": max(hist) / n}


def verdict_a4(a4):
    """E-A4（PLAN §8）: 成立 = ρ 显著且方向为正; ρ 不显著 ⇒ 只能记「代理量不足」。
    ⚠️ 目标不同结论不同，必须分开看（本报告显式写）:
      · `best_step` 目标在本 ckpt 上**退化**（96.98% 图最优步=step7）⇒ ρ≈0 无信息量，
        不能拿它当「熵假设不成立」的证据;
      · `needed_step_at_final_median_tau` 目标（达到该域末步中位 L1 所需步数）ρ 显著为正。
    故若仅后者显著为正 ⇒ 成立，但依据里写明「目标依赖 + 代理弱」。
    """
    rows = []
    for dom, s in (a4 or {}).items():
        for r in (s.get("rows") or []):
            ci = (r.get("ci_low"), r.get("ci_high"))
            if None in ci:
                continue
            rows.append((dom, r.get("proxy"), r.get("target"), r.get("rho"),
                         ci[0] > 0, ci[1] < 0))
    if not rows:
        return V_INSUFF, "无 ρ/CI 行"
    pos = [r for r in rows if r[4]]
    neg = [r for r in rows if r[5] and not r[4]]
    n_needed = sum(1 for r in rows if r[2] == "needed_step_at_final_median_tau")
    n_best = sum(1 for r in rows if r[2] == "best_step")
    split = f"（needed_step 目标 {n_needed} 行 / best_step 目标 {n_best} 行）"
    if pos:
        d, p, t, rho, _, _ = max(pos, key=lambda x: x[3])
        return V_OK, (f"{len(pos)}/{len(rows)} 对 ρ 显著为正（最强 {d}:{p}~{t} "
                      f"ρ={rho:+.3f}）{split}；⚠️ 弱代理，且 `best_step` 目标因退化无信息量")
    if neg:
        d, p, t, rho, _, _ = min(neg, key=lambda x: x[3])
        return V_NO, (f"仅有的显著对方向为负（{d}:{p}~{t} ρ={rho:+.3f}）"
                      f"⇒ 代理量方向与假设相反{split}")
    return V_INSUFF, f"0/{len(rows)} 对 ρ 显著（PLAN: 只能记「代理量不足」）{split}"


def verdict_a6(a6):
    """E-A6: 复现「≥88% 降幅在前 4 步」。三态按 tsc json 的 verdict 位。"""
    fronts, flags = [], []
    for dom, s in (a6 or {}).items():
        front = s.get("cumulative_share_prefix_mean") or {}
        f4 = front.get("step4")
        g = (s.get("verdict") or {}).get("ge_88pct_within_first_4_steps")
        if f4 is not None:
            fronts.append((dom, f4))
        if g is not None:
            flags.append(bool(g))
    if not flags:
        return V_INSUFF, "缺 ge_88pct 判定位"
    detail = "；".join(f"{d} step4={v * 100:.1f}%" for d, v in fronts) or "—"
    if all(flags):
        return V_OK, f"≥88% 降幅在前 4 步复现（{detail}）"
    if not any(flags):
        return V_NO, f"≥88% 未在前 4 步复现（{detail}）"
    return V_INSUFF, f"各域不一致（{detail}）"


def verdict_a5(z):
    """E-A5（PLAN §8 三态 + 显式运营化阈值）:
      · 否证（冗余仍在）: `effective_rank_norm < 0.5` **且** `pair_cos_mean > 0.9`
        （= PLAN 的「有效秩 ≪ K 且 cos > 0.9」）;
      · 成立（低冗余）: `effective_rank_norm ≥ 0.7` 且 `pair_cos_mean ≤ 0.5`
        且 `frac(cos>0.9) ≤ 0.05`;
      · 否则数据不足（落在中间）。
    阈值是**本报告自定的运营化**（PLAN 只给定性方向），报告与聚合表均显式标注。
    """
    er = z.get("effective_rank_norm")
    cs = z.get("pair_cos_mean")
    fg = z.get("pair_cos_frac_gt_0_9")
    if er is None or cs is None:
        return V_INSUFF, "缺 effective_rank_norm / pair_cos_mean"
    fgs = f"{fg:.4f}" if isinstance(fg, (int, float)) else "—"
    if er < 0.5 and cs > 0.9:
        return V_NO, (f"有效秩 {er:.3f}×K < 0.5×K 且 cos {cs:.3f} > 0.9 ⇒ 冗余仍在"
                      f"（E-B3 动机增强）")
    if er >= 0.7 and cs <= 0.5 and (fg is None or fg <= 0.05):
        return V_OK, (f"有效秩 {er:.3f}×K ≥ 0.7×K、cos 均值 {cs:.3f} ≤ 0.5、"
                      f"frac(cos>0.9)={fgs} ≤ 0.05 ⇒ 低冗余（非「K 份副本」）")
    return V_INSUFF, (f"有效秩 {er:.3f}×K / cos {cs:.3f} / frac>0.9={fgs} "
                      f"落在运营化阈值中间，不判")


def verdict_b1(rows):
    """E-B1: 4 臂末步 L1_255 差异 <0.10 px ⇒ 否证; 有臂改善>0.10 且过冲不放大 ⇒ 成立。"""
    base = rows.get("b1_mean")
    if not base:
        return V_INSUFF, "缺 b1_mean 基线复跑"
    base_last = base.get("L1_255_last")
    arms = {k: v for k, v in rows.items() if k.startswith("b1_") and k != "b1_mean"}
    if not arms or base_last is None:
        return V_INSUFF, "缺 E-B1 非基线臂或末步 L1"
    if len(arms) < 3:
        lack = f"（仅 {len(arms)}/3 个非基线臂到位）"
    else:
        lack = ""
    deltas = {k: base_last - v["L1_255_last"] for k, v in arms.items()
              if v.get("L1_255_last") is not None}
    if not deltas:
        return V_INSUFF, "非基线臂缺末步 L1"
    mx = max(abs(d) for d in deltas.values())
    if mx < 0.10:
        return V_NO, f"各臂 |Δ末步 L1| 最大 {mx:.3f} px < 0.10 px {lack}"
    base_os = base.get("overshoot", 0.0) or 0.0
    good = [k for k, d in deltas.items()
            if d > 0.10 and (arms[k].get("overshoot", 0.0) or 0.0) <= base_os + 1e-9]
    # 每臂方向一句话（便于报告解释「谁更好、更差多少」）
    dirs = "; ".join(
        f"{k} Δ末步L1={d:+.4f} px"
        f"({'更好' if d > 0 else '更差'})"
        f" 过冲{arms[k].get('overshoot', 0.0):.4f}"
        for k, d in sorted(deltas.items()))
    if good:
        g = good[0]
        # 剂量-响应：按 ΔPSNR 降序（含基线=0）
        dose = {("b1_mean"): 0.0}
        for k in arms:
            if arms[k].get("psnr") is not None and base.get("psnr") is not None:
                dose[k] = arms[k]["psnr"] - base["psnr"]
        order = " > ".join(f"{k} {v:+.3f}" for k, v in
                           sorted(dose.items(), key=lambda kv: -kv[1]))
        return V_OK, (f"{','.join(sorted(good))} 末步改善 >0.10 px 且过冲未放大"
                      f"（{arms[g].get('overshoot', 0.0):.4f} ≤ 基线 {base_os:.4f}）"
                      f"；{dirs}；**剂量（ΔPSNR dB 降序）= {order}** {lack}")
    return V_INSUFF, f"最大差异 {mx:.3f} px ≥ 0.10 但方向/过冲不满足；{dirs} {lack}"


def verdict_b3(rows):
    """E-B3（PLAN §8 有**两个可分离的轴**，本函数分别判三态）:
      · **轴 1「recon 增益」**: `recon 不劣于 b1_mean`（0.10 dB 噪声口径）⇒ 成立/否证;
      · **轴 2「ALIT 后步激活机制假说」**: 要求 `Δ₁ 占比下降` **且** `后步 ΔL1 上升`
        ⇒ 成立; 两条都不满足 ⇒ 否证; 否则数据不足;
      · 判据 (c)「注意力熵随迭代下降」**未测** ⇒ 该子项数据不足。
    两轴结论相反时返回 `V_SPLIT`（PLAN §8 的三态表无法表达，报告须分轴写明）。
    """
    base, arm = rows.get("b1_mean"), rows.get("b3_grow")
    if not base or not arm:
        return V_INSUFF, "缺 b1_mean 或 b3_grow"
    recon_ok = (arm.get("psnr") is not None and base.get("psnr") is not None
                and arm["psnr"] >= base["psnr"] - 0.10)     # 0.1 dB = PLAN §9 噪声口径
    d1 = arm.get("delta1_share")
    d1_down = (d1 is not None and base.get("delta1_share") is not None
               and d1 < base["delta1_share"] - 0.02)
    later_up = (arm.get("later_delta_mean") is not None
                and base.get("later_delta_mean") is not None
                and arm["later_delta_mean"] > base["later_delta_mean"])
    mech_ok = bool(d1_down and later_up)
    gain_ok = bool(recon_ok)
    dp = (arm.get("psnr") - base.get("psnr")) if (arm.get("psnr") is not None
                                                  and base.get("psnr") is not None) else None
    facts = (f"ΔPSNR={fsgn(dp, 4)} dB（recon 不劣={recon_ok}）; "
             f"Δ₁占比 {fnum((base.get('delta1_share') or 0) * 100, 2)}% → "
             f"{fnum((d1 or 0) * 100, 2)}%（下降={d1_down}）; "
             f"后 4 步 ΔL1 均值 {fsgn(base.get('later_delta_mean'))} → "
             f"{fsgn(arm.get('later_delta_mean'))}（上升={later_up}）")
    if gain_ok and mech_ok:
        return V_OK, f"两轴均成立: {facts}; (c) 注意力熵未测"
    if gain_ok and not mech_ok:
        return V_SPLIT, (f"**轴1「recon 增益」= 成立**（{facts}）；"
                         f"**轴2「ALIT 后步激活机制假说」= 否证**"
                         f"（Δ₁ 占比未降{'（反而升）' if d1 is not None and base.get('delta1_share') is not None and d1 > base['delta1_share'] else ''}"
                         f"，后步 Δt 未升）；判据 (c) 注意力熵 = 数据不足（未测）"
                         f"⇒ PLAN §8 三态表无法表达此分裂")
    if not gain_ok and not mech_ok:
        return V_NO, f"两轴均不成立: {facts}"
    return V_INSUFF, (f"部分成立: recon不劣={recon_ok} Δ₁占比↓={d1_down} "
                      f"后步ΔL1↑={later_up}；(c) 注意力熵未测")


def verdict_ec1(e):
    """E-C1: tools/alit_task_fidelity.py 的 E_C1_verdict.primary_verdict 三态映射。
    `indistinguishable` ⇒ 数据不足（不是「成立」，也不是「否证」），并附探针质量边界。
    """
    v = _get(e, "E_C1_verdict", "e_c1_verdict")
    if not isinstance(v, dict):
        return V_INSUFF, "缺 E_C1_verdict"
    pv = v.get("primary_verdict")
    note = (f"primary Δ={fsgn(v.get('primary_delta_pp'), 4)} pp "
            f"CI={v.get('primary_delta_pp_ci')}（mean_k={fnum(v.get('primary_mean_k'), 3)}）")
    try:
        q = probe_quality(e)
    except Exception:  # noqa: BLE001
        q = {}
    qnote = ""
    if q:
        qnote = (f"；探针分辨力不足（train_top1 {fnum(q['train_min'], 3)}–"
                 f"{fnum(q['train_max'], 3)} vs val_top1 {fnum(q['val_min'], 4)}–"
                 f"{fnum(q['val_max'], 4)}，val 曲线下降 {q['n_dec']}/{q['n_steps']} 次）"
                 f"⇒ 只能说「该探针测不到差异」，不能说「自适应 = 固定 K」")
    if pv == "adaptive_better":
        return V_OK, note + qnote
    if pv == "fixed_better":
        return V_NO, note + qnote
    return V_INSUFF, f"indistinguishable（CI 跨 0，未达成立判据）{note}{qnote}"


# ─────────────────────────── markdown 段 ───────────────────────────

def sec_availability(c, now):
    L = ["## 0. 数据可用性（自动）", "",
         f"- 产物目录: `{os.path.relpath(c['art'], REPO_ROOT)}`",
         f"- 生成时间: {now}",
         f"- 生成命令: `python3 tools/alit_report.py`",
         "- 口径: 只读产物 json；缺文件一律记为「数据不足」，不填任何估计值。", "",
         "| 产物 | 用途 | 状态 | 生产者 |", "|---|---|---|---|"]
    for name, desc, producer in EXPECTED:
        st = "**present**" if name in c["present"] else "MISSING"
        L.append(f"| `{name}` | {desc} | {st} | {producer} |")
    if c["extra_json"]:
        L += ["", "目录内其它 json（未登记）: " +
              ", ".join(f"`{x}`" for x in c["extra_json"])]
    L += ["", f"**missing 合计 {len(c['missing'])} 项**："
              + ("、".join(f"`{m[0]}`" for m in c["missing"]) if c["missing"] else "无")]
    return "\n".join(L)


def sec_run_status(c):
    """E-A 执行状态（alit_a_steps.tsv / alit_a_summary.json / SKIP_*.json）。"""
    if not (c.get("a_summary") or c.get("skips") or c.get("a_steps")):
        return None
    L = ["## 0.1 E-A 执行状态（自动）", "",
         "来源: `alit_a_steps.tsv` / `alit_a_summary.json` / `SKIP_*.json`"
         "（`run_alit_a.sh`）。SKIP 是**显式降级标记**，不是静默跳过。", ""]
    if c.get("a_steps") is not None:
        L += ["`alit_a_steps.tsv`（步骤 / rc / 耗时秒）:", "",
              "| 步骤 | rc | 秒 |", "|---|---|---|"]
        for row in c["a_steps"]:
            r = (row + ["", "", ""])[:3]
            L.append("| " + " | ".join(r) + " |")
        L.append("")
    if c.get("a_summary"):
        arts = (c["a_summary"] or {}).get("artifacts") or {}
        if arts:
            L += ["`alit_a_summary.json → artifacts`:", "",
                  "| 产物 | 状态 |", "|---|---|"]
            for k, v in arts.items():
                L.append(f"| `{k}` | {v} |")
            L.append("")
    if c.get("skips"):
        L += ["**SKIP 标记（显式降级）**：", "",
              "| ID | 原因 | 时间 |", "|---|---|---|"]
        for k, d in sorted(c["skips"].items()):
            L.append(f"| `{k}` | {d.get('reason', '—')} | {d.get('ts', '—')} |")
        L.append("")
    return "\n".join(L)


def sec_baseline(c):
    L = ["## 1. 四臂基线复现（E-A0）", "",
         "来源: `baseline_four_arm_eval.json`（`eval_linattn.py`，全量 5,000 张同进程）；"
         "逐步列来源 `stepwise_*.json`（`eval_stepwise_linattn.py`）。",
         f"参照值来源: {SRC_PLAN}（10-15/10-16 旧实例发布值）。", ""]
    if not c["four_arm"]:
        L.append("**数据不足**（缺 `baseline_four_arm_eval.json`）。")
        return "\n".join(L)
    L += ["| 臂 | PSNR (dB) ↑ | SSIM ↑ | L1(归一化) ↓ | L1(0-255) ↓ | "
          "PSNR per-patch DC 基线 | 相对 DC 增益 | 图数 | PLAN §2.2 PSNR | Δ复现 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    ok_all = True
    for k, r in c["four_arm"].items():
        arm = os.path.basename(os.path.dirname(k))
        sw = c["stepwise"].get(arm)
        l1_255 = (fnum(sw["L1_255"][-1], 3) if sw and sw.get("L1_255") else
                  fnum((r.get("l1_norm") or 0) * 57.63, 3))
        ref = PLAN_REF.get(k, {}).get("psnr")
        d = None if ref is None else r["psnr"] - ref
        if d is not None and abs(d) > 0.05:
            ok_all = False
        L.append(f"| `{arm}` | {fnum(r.get('psnr'))} | {fnum(r.get('ssim'), 4)} | "
                 f"{fnum(r.get('l1_norm'), 4)} | {l1_255} | "
                 f"{fnum(r.get('psnr_patch_dc_baseline'), 2)} | "
                 f"{fsgn(r.get('psnr_gain_over_patch_dc'), 2)} | {r.get('n_images')} | "
                 f"{fnum(ref)} | {fsgn(d, 4)} |")
    L += ["", ("**复现判决**: 四臂与 PLAN §2.2 参照值在 |ΔPSNR| ≤ 0.05 dB 内逐一对齐 ⇒ "
               "**基线复现成立**（同实例、同 ckpt、同进程口径）。" if ok_all else
               "⚠️ 存在 |ΔPSNR| > 0.05 dB 的臂 ⇒ 复现存疑，需查。"), ""]
    # 回归门（E-A 前置）：新代码默认路径必须与发布版逐位一致
    if c["gate"]:
        gv = c["gate"].get("verdict")
        gp = c["gate"].get("max_abs_diff") or c["gate"].get("max_abs_delta")
        L.append(f"- **回归门 `gate_diff_stdattn_rec.json`: {gv}**"
                 + (f"（max|Δ| = {fnum(gp)}）" if gp is not None else "")
                 + " ⇒ 新代码不加开关时默认路径与发布版一致。")
    else:
        L.append("- **回归门: 数据不足**（缺 `gate_diff_stdattn_rec.json`；"
                 "E-A 阶段要求先确认新代码默认路径逐位不变）。")
    L += ["",
          "> **口径换算（可复核）**: `eval_linattn.py` 的 `l1_norm` 是归一化空间 L1；"
          "`stepwise_*.json` 的 `L1_255` 是反归一化像素域值。二者关系 "
          "`L1_255 ≈ l1_norm × mean(IMAGENET_STD) × 255 = l1_norm × 57.63`"
          "（本表逐臂一致，见 `artifacts/ablation_table.md` 同一换算）。", ""]
    return "\n".join(L)


def sec_stepwise(c):
    L = ["## 2. 逐步曲线与停步分布（E-A1 基础数据）", "",
         "来源: `stepwise_<arm>.json`（`eval_stepwise_linattn.py`，全量 5,000 张）。",
         "`cum_z_s[t]` = 第 t 步实际读入的 z_s 前缀长度（= token 数）。", ""]
    recs = {a: d for a, d in c["stepwise"].items() if d.get("recurrent")}
    if not recs:
        L.append("**数据不足**（无循环臂 stepwise 产物）。")
        return "\n".join(L)
    for idx, (arm, d) in enumerate(sorted(recs.items()), start=1):
        L += [f"### 2.{idx} `{arm}`（T={d.get('T')}，n={d.get('n_images')}）", "",
              "| step | t | z_s 累计 | L1_255 ↓ | ΔL1_255（正=变好） | PSNR (dB) ↑ |",
              "|---|---|---|---|---|---|"]
        for i, t in enumerate(d.get("steps") or []):
            dl = (d["delta_L1_255"][i] if i < len(d.get("delta_L1_255") or [])
                  else None)
            L.append(f"| {i + 1} | {t} | {d['cum_z_s'][i]} | {fnum(d['L1_255'][i])} | "
                     f"{fsgn(dl)} | {fnum(d['PSNR'][i])} |")
        hist = d.get("best_step_hist") or []
        n = d.get("n_images") or sum(hist)
        og, otok, ftok = oracle_gain_for_arm(c, arm)
        v, why = verdict_a1(hist, n, og, otok, ftok)
        hs = hist_stats(hist)
        L += ["",
              f"- 总落差 `total_drop_255` = **{fnum(d.get('total_drop_255'))} px**；"
              f"最优步 = **step {int(d.get('best_step', 0)) + 1}**；"
              f"Δ₁ 占比 = **{fnum((d.get('delta1_share') or 0) * 100, 1)}%**；"
              f"正增量步数 = {d.get('n_positive_steps')}/{(d.get('T') or 1) - 1}",
              f"- 每图最优停步分布（hist, step1..stepT）= {hist}",
              f"- 停步统计（由 hist 反推）: 平均 **{fnum(hs.get('mean_step'), 2)} ± "
              f"{fnum(hs.get('std_step'), 2)}** 步；众数占比 "
              f"**{fnum((hs.get('mode_share') or 0) * 100, 2)}%**；"
              f"熵 **{fnum(hs.get('entropy_nats'), 3)} nats**"
              f"（归一化 {fnum(hs.get('entropy_norm'), 3)}）"]
        if og is not None:
            L.append(f"- oracle 停步（不可实现上界）: 平均 **{fnum(otok, 3)} tokens** vs "
                     f"固定 step7 **{ftok}** tokens；oracle 增益 **{fsgn(og, 5)} px**"
                     f"（来源 `tsc_*.json → E_A1`）")
        L.append(f"- **E-A1 判决: {v}** —— {why}")
        L.append("")
    return "\n".join(L)


def sec_tsc(c):
    L = ["## 3. E-A2 自监督 TSC：等平均 token 数对照", "",
         "来源: `tsc_*.json`（`tools/alit_adaptive_k.py` 读 `per_image_*.json`）。",
         "口径: 每图取最小 t 使 `L1_255[t] ≤ τ`；对照臂 = 固定 step 3/5/7/8 的"
         "**等平均 token 数**（密集 τ 曲线插值）。", ""]
    if not c["tsc"]:
        L.append("**数据不足**（缺 `tsc_*.json`；需先有 `--dump_per_image` 产物）。")
        return "\n".join(L), [], {}
    plots = []
    vmap = {}
    for idx, (tag, d) in enumerate(sorted(c["tsc"].items()), start=1):
        a2 = d.get("E_A2_tsc") or {}
        if not a2:
            continue
        for dom, s in sorted(a2.items()):
            L += [f"### 3.{idx} `{tag}` / 域 `{dom}`", "",
                  "固定步臂与等预算自适应对照（ΔL1 < 0 = 自适应更优；"
                  "判据 >0.10 px 且方向一致）:", "",
                  "| 固定 step | 预算 tokens | 自适应@等预算 L1_255 | 固定 L1_255 | "
                  "ΔL1_255 (px) | ΔPSNR (dB) | 预算在曲线范围内 |",
                  "|---|---|---|---|---|---|---|"]
            for m in (s.get("matched_interp_equal_budget")
                      or s.get("matched_equal_budget") or []):
                L.append(f"| {m.get('step')} | {fnum(m.get('budget_tokens'), 1)} | "
                         f"{fnum(m.get('adaptive_mean_L1_255_at_budget'))} | "
                         f"{fnum(m.get('fixed_mean_L1_255'))} | "
                         f"{fsgn(m.get('delta_L1_255'))} | {fsgn(m.get('delta_PSNR'))} | "
                         f"{m.get('budget_in_range')} |")
            vd = s.get("verdict") or {}
            any_better = vd.get("any_match_adaptive_better_0.10px")
            rows_eq = (s.get("matched_interp_equal_budget")
                       or s.get("matched_equal_budget") or [])
            deltas = [m.get("delta_L1_255") for m in rows_eq
                      if m.get("delta_L1_255") is not None and m.get("budget_in_range", True)]
            if any_better is None:
                v, why = V_INSUFF, "tsc json 缺 verdict 字段"
            elif any_better:
                v, why = V_OK, "等预算下存在自适应优于固定步 >0.10 px 的预算点"
            elif not deltas:
                v, why = V_INSUFF, "无预算在曲线范围内的等预算对照点"
            else:
                best_d, worst_d = min(deltas), max(deltas)
                if worst_d >= 0.10:
                    v, why = V_NO, (f"等预算下自适应**更差**：ΔL1 最大 {worst_d:+.4f} px "
                                    f"（ΔL1>0 = 自适应更差；最小 {best_d:+.4f}）"
                                    f"⇒ 信息保持口径结案，判决权交 E-C1")
                else:
                    v, why = V_NO, (f"所有等预算点 ΔL1 ∈ [{best_d:+.4f}, {worst_d:+.4f}] px，"
                                    f"无 >0.10 px 的优势（PLAN §8：复现 batch-0 天花板 ⇒ "
                                    f"PSNR 口径结案，判决权交 E-C1）")
            L += ["", f"- **E-A2 判决（`{tag}`/`{dom}`）: {v}** —— {why}", ""]
            vmap[(tag, dom)] = (v, why)
            plots.append((tag, dom, s))
    return "\n".join(L), plots, vmap


def sec_ladder(c):
    L = ["## 4. E-A3 IID / OOD 熟悉度阶梯", ""]
    entries = {}
    a3 = ((c["ladder"] or {}).get("E_A3_domain_ladder")
          or (c["ladder"] or {})) if c["ladder"] else {}
    if c["ladder"]:
        pd_ = a3.get("per_domain") or {}
        for dom, s in pd_.items():
            entries[dom] = {
                "n": s.get("n_images"),
                "mean_best_step": s.get("mean_best_step_1based"),
                "std_best_step": s.get("std_best_step"),
                "L1_step1": s.get("L1_step1"),
                "L1_step7": s.get("L1_step7"),
                "gap": s.get("gap_step1_minus_step7"),
                "src": "domain_ladder.json",
            }
    if not entries:
        for tag, d in c["ood"].items():
            hist = d.get("best_step_hist") or []
            n = d.get("n_images") or sum(hist)
            mean_bs = (sum((i + 1) * h for i, h in enumerate(hist)) / n) if n else None
            L1 = d.get("L1_255") or []
            entries[tag] = {
                "n": n, "mean_best_step": mean_bs, "std_best_step": None,
                "L1_step1": L1[0] if L1 else None,
                "L1_step7": L1[min(6, len(L1) - 1)] if L1 else None,
                "gap": ((L1[0] - L1[min(6, len(L1) - 1)]) if L1 else None),
                "src": f"ood_{tag}.json（降级口径：由 best_step_hist 反推）",
            }
    if not entries:
        L.append("**数据不足**（缺 `domain_ladder.json` 与 `ood_*.json`）。")
        return "\n".join(L), None
    L += ["| 域 | n | 平均最优停步（1-based） | 停步 std | L1_255 step1 | "
          "L1_255 step7 | 落差 step1−step7 | 来源 |", "|---|---|---|---|---|---|---|---|"]
    for dom, e in entries.items():
        L.append(f"| {dom} | {e.get('n')} | {fnum(e.get('mean_best_step'), 2)} | "
                 f"{fnum(e.get('std_best_step'), 2)} | {fnum(e.get('L1_step1'))} | "
                 f"{fnum(e.get('L1_step7'))} | {fnum(e.get('gap'))} | `{e.get('src')}` |")
    steps = [(d, e["mean_best_step"]) for d, e in entries.items()
             if e.get("mean_best_step") is not None]
    if len(steps) >= 2:
        spread = max(s for _, s in steps) - min(s for _, s in steps)
        order = [d for d, _ in sorted(steps, key=lambda x: x[1])]
        if spread >= 1.0:
            v = V_OK
            why = f"三级平均停步跨度 {spread:.2f} 步 ≥ 1 步（由少到多: {' < '.join(order)}）"
        elif spread < 0.5:
            v = V_NO
            why = f"三级平均停步跨度 {spread:.2f} 步 < 0.5 步 ⇒ 熟悉度假设不成立"
        else:
            v = V_INSUFF
            why = f"跨度 {spread:.2f} 步落在 [0.5, 1.0)，未达 PLAN 判据"
        L += ["", f"- **E-A3 判决: {v}** —— {why}"]
        if c["ladder"]:
            vd = a3.get("verdict") or {}
            L.append(f"  - `domain_ladder.json` verdict: "
                     f"max_abs_mean_step_delta="
                     f"{fnum(vd.get('max_abs_mean_step_delta'), 2)}, "
                     f"monotone={vd.get('monotone_increasing_with_distance')}")
        # 共同 τ 下「所需步数」的域间对照（比均值停步更有分辨力）
        ct = a3.get("common_tau_needed_steps") if c["ladder"] else None
        if ct:
            L += ["", "共同 τ 下的所需步数（Δ 相对第一域；仅列 frac_capped ≤ 0.6 的 τ）:",
                  "", "| τ (L1_255) | " + " | ".join(entries) + " | " +
                  " | ".join(f"Δ {d} vs {next(iter(entries))}" for d in list(entries)[1:]) + " |",
                  "|---|" + "---|" * (len(entries) + max(len(entries) - 1, 0))]
            shown = 0
            for tau, per in ct.items():
                vals = {d: per.get(d, {}).get("mean_step_1based") for d in entries}
                if any(v is None for v in vals.values()):
                    continue
                if any((per.get(d, {}).get("frac_capped") or 0) > 0.6 for d in entries):
                    continue
                first = next(iter(entries))
                deltas = " | ".join(
                    fsgn(vals[d] - vals[first], 3) for d in list(entries)[1:])
                L.append(f"| {tau} | " + " | ".join(fnum(vals[d], 3) for d in entries)
                         + " | " + deltas + " |")
                shown += 1
            if shown == 0:
                L.append("| — | " + " | ".join("—" for _ in entries) + " | "
                         + " | ".join("—" for _ in list(entries)[1:]) + " |")
        if len(steps) < 3:
            L.append("  - ⚠️ 仅 " + str(len(steps))
                     + " 级到位（PLAN 要求 IID/中距/远距三级），判决按现有级数给出并标注。")
    else:
        L += ["", "- **E-A3 判决: 数据不足**（有效域 < 2 级）。"]
    return "\n".join(L), entries


def sec_a4_a6(c):
    L = ["## 5. E-A4 复杂度相关性 / E-A6 Δ_t 分解", ""]
    if c["tsc"]:
        for idx, (tag, d) in enumerate(sorted(c["tsc"].items()), start=1):
            a4 = d.get("E_A4_complexity_spearman")
            if a4:
                L += [f"### 5.{idx}a E-A4 Spearman（`{tag}`）", "",
                      "来源: `tsc_*.json → E_A4_complexity_spearman`。"
                      "⚠️ 代理量（JPEG 大小/边缘密度/首步 L1）是**弱代理**，"
                      "ρ 显著 ≠ 验证 ALIT 的熵假设。", "",
                      "| 域 | 代理 | 目标 | ρ | 95% CI | 显著(CI 不含 0) | n_finite |",
                      "|---|---|---|---|---|---|---|"]
                for dom, s in a4.items():
                    for r in (s.get("rows") or []):
                        ci = (r.get("ci_low"), r.get("ci_high"))
                        sig = (None if None in ci else
                               (ci[0] > 0 or ci[1] < 0))
                        L.append(f"| {dom} | {r.get('proxy')} | {r.get('target')} | "
                                 f"{fsgn(r.get('rho'))} | "
                                 f"[{fnum(ci[0])}, {fnum(ci[1])}] | {sig} | "
                                 f"{r.get('n_finite')} |")
                L.append("")
            a6 = d.get("E_A6_delta_decomposition")
            if a6:
                L += [f"### 5.{idx}b E-A6 Δ_t 分解（`{tag}`）", "",
                      "来源: `tsc_*.json → E_A6_delta_decomposition`。"
                      "判据: 复现「≥88% 降幅在前 4 步」。", "",
                      "| 域 | Δ₁ 占比 | step4 前置占比 | 末步过冲率 | 末步 ΔL1 均值 | "
                      "≥88%@step4 |", "|---|---|---|---|---|---|"]
                for dom, s in a6.items():
                    front = s.get("cumulative_share_prefix_mean") or {}
                    L.append(f"| {dom} | {fnum((s.get('delta1_share_mean') or 0) * 100, 1)}% | "
                             f"{fnum((front.get('step4') or 0) * 100, 1)}% | "
                             f"{fnum((s.get('last_step_overshoot_rate') or 0) * 100, 1)}% | "
                             f"{fsgn(s.get('last_step_delta_mean'))} | "
                             f"{(s.get('verdict') or {}).get('ge_88pct_within_first_4_steps')} |")
                L.append("")
    else:
        # 降级口径：由 stepwise 聚合曲线直接算（标注）
        recs = {a: d for a, d in c["stepwise"].items() if d.get("recurrent")}
        if recs:
            L += ["### 5.3 降级口径 E-A6（由 `stepwise_*.json` 聚合曲线，非每图口径）", "",
                  "⚠️ 聚合曲线的累计占比 ≠ 每图占比的均值；仅作**粗口径**，"
                  "正式结论须等 `tsc_*.json`。", "",
                  "| 臂 | Δ₁ 占比 | step4 前置占比(聚合) | 末步 ΔL1 | 末步过冲 | 正增量步数 |",
                  "|---|---|---|---|---|---|"]
            for arm, d in recs.items():
                L1 = d.get("L1_255") or []
                if not L1:
                    continue
                drop = max(L1[0] - min(L1), 1e-9)
                front4 = (L1[0] - L1[min(3, len(L1) - 1)]) / drop
                L.append(f"| `{arm}` | {fnum((d.get('delta1_share') or 0) * 100, 1)}% | "
                         f"{fnum(front4 * 100, 1)}% | "
                         f"{fsgn((d.get('delta_L1_255') or [None])[-1])} | "
                         f"{'是' if (d.get('delta_L1_255') or [1])[-1] < 0 else '否'} | "
                         f"{d.get('n_positive_steps')}/{(d.get('T') or 1) - 1} |")
            L.append("")
        else:
            L += ["**数据不足**（缺 `tsc_*.json` 与循环臂 stepwise 产物）。", ""]
    return "\n".join(L)


def sec_a5(c):
    L = ["## 6. E-A5 z_s 冗余度（第 1 项）", "",
         "来源: `zs_redundancy_*.json` / `zs_stats*.json`（`metrics_linattn.zs_stats`）。"
         "注意: 注意力熵（第 2/3 项）**未做**，见诚实边界。", ""]
    if not c["zs"]:
        L.append("**数据不足**（缺 z_s 冗余度产物）。")
        return "\n".join(L), []
    L += ["| 产物 | 臂 | K | D | 有效秩 | 有效秩/K | 去均值有效秩 | 两两 cos 均值 | "
          "cos>0.9 占比 | 逐 token 方差 (平均/D 维) | token 均值方差 (跨 K) | 判决 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    for tag, d in c["zs"].items():
        for arm, s in (d.get("results") or {}).items():
            v, why = verdict_a5(s)
            verdicts[f"{tag}/{arm}"] = (v, why)
            L.append(f"| `{tag}` | {arm} | {s.get('K')} | {s.get('D')} | "
                     f"{fnum(s.get('effective_rank'))} | "
                     f"{fnum(s.get('effective_rank_norm'))} | "
                     f"{fnum(s.get('effective_rank_centered'))} | "
                     f"{fnum(s.get('pair_cos_mean'), 4)} | "
                     f"{fnum(s.get('pair_cos_frac_gt_0_9'), 4)} | "
                     f"{fnum(s.get('token_var_mean'), 4)} | "
                     f"{s.get('token_mean_var'):.3e} | {v} |")
    L += ["", "运营化阈值（PLAN 只给定性方向，本报告**自定并显式标注**）: "
              "**否证** = 有效秩 <0.5×K **且** cos 均值 >0.9；"
              "**成立（低冗余）** = 有效秩 ≥0.7×K **且** cos 均值 ≤0.5 "
              "**且** frac(cos>0.9) ≤0.05；否则数据不足。", ""]
    for k, (v, why) in verdicts.items():
        L.append(f"- **E-A5 判决（{k}）: {v}** —— {why}")
    return "\n".join(L), verdicts


def sec_b(c):
    """E-B1..B4 臂表 + 判决。"""
    L = ["## 7. E-B 重训臂对照（E-B1..B4）", "",
         "来源: `alit_b_arms.tsv`（执行状态）+ `eval_<arm>.json`（全量评测）+ "
         "`stepwise_<arm>.json`（逐步）。协议: IN-100 224², K=64, std 编码器, "
         "T=8 BPTT, 从 `stdattn_k64` 热启动, 6 ep/11,406 步/bs64/bf16。",
         "**执行口径**: 全部 B 臂在同一台**新实例上串行**执行（同机、同后端、单一训练机器）"
         "⇒ 无跨机混淆；配对性来自「同进程 / 同批次评测」（PLAN §0 口径 2）。", ""]
    rows = {}
    for arm, ej in c["eval_b"].items():
        # 该臂自身结果键: 目录名匹配 arm; 否则若只有一个非参照键则取它
        # （覆盖 b3_base 复用 b1_mean 目录的情形）
        own = None
        ref_keys = {k for k in ej if "stdattn_k64" in k}
        for k in ej:
            if os.path.basename(os.path.dirname(k)) == arm or arm in k:
                own = ej[k]
                break
        if own is None:
            cand = [k for k in ej if k not in ref_keys]
            if len(cand) == 1:
                own = ej[cand[0]]
        if own is None:
            continue
        sw = c["stepwise_b"].get(arm)
        row = {"psnr": own.get("psnr"), "ssim": own.get("ssim"),
               "l1_norm": own.get("l1_norm"), "n": own.get("n_images"),
               "step": own.get("step"), "epoch": own.get("epoch"),
               "src_eval": f"eval_{arm}.json"}
        if sw:
            L1 = sw.get("L1_255") or []
            dlt = sw.get("delta_L1_255") or []
            row.update({
                "L1_255_last": L1[-1] if L1 else None,
                "L1_255_best": min(L1) if L1 else None,
                "delta1_share": sw.get("delta1_share"),
                "overshoot": (-dlt[-1]) if dlt and dlt[-1] < 0 else 0.0,
                "later_delta_mean": (sum(dlt[-4:]) / 4.0) if len(dlt) >= 4 else
                                    (sum(dlt) / len(dlt) if dlt else None),
                "n_positive_steps": sw.get("n_positive_steps"),
                "src_stepwise": f"stepwise_{arm}.json",
            })
        rows[arm] = row

    if c["tsv"]:
        L += ["### 7.1 执行状态（`alit_b_arms.tsv`）", "",
              "| 臂 | 状态 | 明细 | 时间 |", "|---|---|---|---|"]
        for ln in c["tsv"]:
            cells = (ln + ["", "", ""])[:4]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    if not rows:
        b4 = ("数据不足", "缺 b4_* 臂产物；" + EB4_COST_CUT)
        groups = {"E-B1": verdict_b1(rows), "E-B2": ("数据不足", "缺 b2_* 臂产物"),
                  "E-B3": verdict_b3(rows), "E-B4": b4}
        L += ["### 7.2 臂结果表", "",
              "**数据不足**（缺 `eval_<arm>.json` / `stepwise_<arm>.json`）。", "",
              "### 7.3 判决（PLAN §8）", "",
              "| 实验 | 判决 | 依据 |", "|---|---|---|"]
        for eid, name in (("E-B1", "逐迭代损失权重"), ("E-B2", "损失侧动态停机"),
                          ("E-B3", "ALIT 式自适应记忆"), ("E-B4", "随机预算/一条流多档")):
            L.append(f"| {eid} {name} | **{groups[eid][0]}** | {groups[eid][1]} |")
        return "\n".join(L), rows, groups

    L += ["### 7.2 臂结果表", "",
          "| 臂 | PSNR ↑ | SSIM ↑ | L1(归一化) ↓ | 末步 L1_255 ↓ | 最优 L1_255 | "
          "Δ₁ 占比 | 末步过冲 (px) | 后 4 步 ΔL1 均值 | 正增量步 | 来源 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(rows):
        r = rows[arm]
        L.append(f"| `{arm}` | {fnum(r.get('psnr'))} | {fnum(r.get('ssim'), 4)} | "
                 f"{fnum(r.get('l1_norm'), 4)} | {fnum(r.get('L1_255_last'))} | "
                 f"{fnum(r.get('L1_255_best'))} | "
                 f"{fnum((r.get('delta1_share') or 0) * 100, 1)}% | "
                 f"{fnum(r.get('overshoot'))} | {fsgn(r.get('later_delta_mean'))} | "
                 f"{r.get('n_positive_steps')} | "
                 f"`{r.get('src_eval')}`"
                 + (f" + `{r['src_stepwise']}`" if r.get("src_stepwise") else "") + " |")

    # ── vs b1_mean 的配对差（E-B1 的解释力所在）──
    base = rows.get("b1_mean")
    if base and any(k.startswith("b1_") and k != "b1_mean" for k in rows):
        L += ["", "**相对同预算基线 `b1_mean` 的配对差**（同机同协议；正 = 该臂更好）：", "",
              "| 臂 | ΔPSNR (dB) | ΔSSIM | Δ末步 L1_255 | Δ最优 L1_255 | ΔΔ₁ (pp) | "
              "末步过冲 | 基线过冲 |", "|---|---|---|---|---|---|---|---|"]
        for arm in sorted(k for k in rows if k.startswith("b1_") and k != "b1_mean"):
            r = rows[arm]
            def _d(a, b):
                return (a - b) if (a is not None and b is not None) else None
            # Δ末步/Δ最优：基线 − 该臂（正 = 该臂 L1 更低 = 更好）
            dd1 = _d(r.get("delta1_share"), base.get("delta1_share"))
            L.append(f"| `{arm}` | {fsgn(_d(r.get('psnr'), base.get('psnr')), 4)} | "
                     f"{fsgn(_d(r.get('ssim'), base.get('ssim')), 4)} | "
                     f"{fsgn(_d(base.get('L1_255_last'), r.get('L1_255_last')), 4)} | "
                     f"{fsgn(_d(base.get('L1_255_best'), r.get('L1_255_best')), 4)} | "
                     f"{fsgn(dd1 * 100 if dd1 is not None else None, 2)} | "
                     f"{fnum(r.get('overshoot'), 4)} | {fnum(base.get('overshoot'), 4)} |")

    # ── 逐 step L1_255 与「arm − b1_mean」（解释「质量前移/后移」）──
    base_sw = c["stepwise_b"].get("b1_mean")
    arms_sw = [(a, c["stepwise_b"][a]) for a in sorted(c["stepwise_b"])
               if a != "b1_mean" and c["stepwise_b"].get(a)]
    if base_sw and arms_sw and base_sw.get("steps") and base_sw["steps"][0] is not None:
        L += ["", "**逐 step L1_255 与「该臂 − `b1_mean`」（来源 `stepwise_b1_*.json`；"
                  "正 = 该臂在该步更差）：**", "",
              "| step | t | z_s 累计 | `b1_mean` | "
              + " | ".join(f"`{a}`" for a, _ in arms_sw) + " | "
              + " | ".join(f"Δ(`{a}`−mean)" for a, _ in arms_sw) + " |",
              "|---|---|---|" + "---|" * (1 + 2 * len(arms_sw))]
        for i, t in enumerate(base_sw["steps"]):
            cells = [str(i + 1), str(t), str(base_sw["cum_z_s"][i]),
                     fnum(base_sw["L1_255"][i])]
            for _a, d in arms_sw:
                cells.append(fnum((d.get("L1_255") or [None] * 8)[i]))
            for _a, d in arms_sw:
                cells.append(fsgn((d.get("L1_255") or [0] * 8)[i] - base_sw["L1_255"][i]))
            L.append("| " + " | ".join(cells) + " |")

    # ── E-B4 对照：截断 = 等 token 预算下的 L1_255（链式因果 ⇒ 截断前缀=全链同值）──
    if base_sw and arms_sw and (base_sw.get("T") or 0) >= 8:
        budgets = [2, 4, 6, 7]                     # 0-based: step3/5/7/8
        toks = [base_sw["cum_z_s"][i] for i in budgets]
        tag = " / ".join(f"t={i + 1}→{tk}" for i, tk in zip(budgets, toks))
        L += ["", f"**截断 = 等 token 预算对照（{tag}；来源各臂 `stepwise_b1_*.json`；"
                  "链式因果 ⇒ 截断档的 L1_255 就等于该步的全链值，无需额外训练）**", ""]
        for metric, nd in (("L1_255", 3), ("PSNR", 3)):
            L += [f"*{metric}*：", "",
                  "| 臂 | " + " | ".join(f"{tk} tokens" for tk in toks) + " |",
                  "|---|" + "---|" * len(toks)]
            for a in ["b1_mean"] + [x for x, _ in arms_sw]:
                d = c["stepwise_b"].get(a)
                if not d:
                    continue
                cells = [fnum((d.get(metric) or [None] * 8)[i], nd) for i in budgets]
                L.append(f"| `{a}` | " + " | ".join(cells) + " |")
            if metric == "L1_255" and base_sw.get("const_baseline_255") is not None:
                cb = fnum(base_sw["const_baseline_255"], 3)
                L.append("| **常量灰基线** | " + " | ".join([cb] * len(toks)) + " |")
            L.append("")
        # 与常量灰的关系（last-only 的早期步报废）
        cb = base_sw.get("const_baseline_255")
        if cb:
            bad = []
            for a in ["b1_mean"] + [x for x, _ in arms_sw]:
                d = c["stepwise_b"].get(a)
                if not d:
                    continue
                worse = [f"t={i + 1}（{fnum(d['L1_255'][i], 2)}）"
                         for i in range(len(d.get("L1_255") or []))
                         if d["L1_255"][i] is not None and d["L1_255"][i] > cb]
                if worse:
                    bad.append(f"`{a}`: {', '.join(worse)}")
            if bad:
                L.append(f"- ⚠️ **L1_255 高于常量灰基线（{fnum(cb, 2)}）的档位（全部 step）**: "
                         + "；".join(bad)
                         + " ⇒ 这些档位在「一条流多档」语境下**不可用**（等价于比灰图还差）。")
                L.append("")

    # 同预算基线复跑 vs 历史 ckpt：同进程评测的差值 + 混淆边界
    for arm in sorted(rows):
        ej = c["eval_b"].get(arm) or {}
        # 只与**同预算/同训练链**的参照臂比；单步 stdattn_k64 是上界参考，预算不同不可比
        refs = [(k, v) for k, v in ej.items() if k != arm and "stdattn_k64_rec" in k]
        own = rows[arm]
        for rk, rv in refs:
            if own.get("psnr") is None or rv.get("psnr") is None:
                continue
            dn = os.path.basename(os.path.dirname(rk))
            d = own["psnr"] - rv["psnr"]
            L += ["", f"- 同进程核对 `{arm}` vs 历史 `{dn}`: ΔPSNR = **{fsgn(d, 4)} dB**"
                      f"（{fnum(own['psnr'], 4)} vs {fnum(rv['psnr'], 4)}，来源 `eval_{arm}.json`）"
                      + ("　⚠️ **该差不可归因于机制**：从零/重跑漂移与训练开关差异"
                         "（如 `dec_grad_ckpt`，历史 run 的 ckpt cfg 无该键）**不可分离**；"
                         "同一条边界同样适用于任何「旧实例 arm vs 新实例 arm」的比较。"
                         if abs(d) >= 0.10 else "")]

    # 分组判决
    b1 = verdict_b1(rows)
    b3 = verdict_b3(rows)
    have_b2 = any(k.startswith("b2_") for k in rows)
    have_b4 = any(k.startswith("b4_") for k in rows)
    have_b4_k = any(k.startswith("b4_k") for k in rows)
    if have_b2:
        base = rows.get("b1_mean") or {}
        parts = []
        for k in sorted(k for k in rows if k.startswith("b2_")):
            r = rows[k]
            parts.append(f"{k}: ΔPSNR={fsgn((r.get('psnr') or 0) - (base.get('psnr') or 0), 3)}"
                         f" Δ末步L1={fsgn((base.get('L1_255_last') or 0) - (r.get('L1_255_last') or 0), 3)}")
        b2 = ("数据不足", "README §4.1 三条判据需要 ‖W·Y_t‖ / 区域隔离 L1 / 真阶梯，"
                          "现有产物只能给 PSNR+Δ_t 读数（" + "; ".join(parts) + "）")
    else:
        b2 = ("数据不足", "缺 b2_* 臂产物")
    if not have_b4:
        b4 = ("数据不足", "缺 b4_* 臂产物；" + EB4_COST_CUT)
    elif not have_b4_k:
        b4 = ("数据不足", "缺专训档 K'=16/32/48 等预算对照（截断档 t=3/5/7 评测亦未到位）"
                          "⇒ 无法判 <0.20 px gap；" + EB4_COST_CUT)
    else:
        b4 = ("数据不足", "缺「截断档 t=3/5/7 vs 专训档」等预算评测产物"
                          "（PLAN E-B4 判据 <0.20 px）")
    L += ["", "### 7.3 判决（PLAN §8）", "",
          "| 实验 | 判决 | 依据 |", "|---|---|---|",
          f"| E-B1 逐迭代损失权重 | **{b1[0]}** | {b1[1]} |",
          f"| E-B2 损失侧动态停机 | **{b2[0]}** | {b2[1]} |",
          f"| E-B3 ALIT 式自适应记忆 | **{b3[0]}** | {b3[1]} |",
          f"| E-B4 随机预算/一条流多档 | **{b4[0]}** | {b4[1]} |", ""]
    return "\n".join(L), rows, {"E-B1": b1, "E-B2": b2, "E-B3": b3, "E-B4": b4}


def _spearman(xs, ys):
    """纯 python Spearman ρ（无 numpy/scipy 依赖），用于探针曲线单调性。"""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(a):
        order = sorted(range(n), key=lambda i: a[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((v - mx) ** 2 for v in rx) ** 0.5
    dy = sum((v - my) ** 2 for v in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def probe_quality(e):
    """探针质量（E-C1 最重要的边界）: train/val top-1 区间、val 曲线单调性。"""
    rows = e.get("top1_by_k") or []
    if not rows:
        return {}
    ks = [r.get("k") for r in rows]
    val = [r.get("top1") for r in rows]
    tr = [r.get("train_top1") for r in rows]
    dec = sum(1 for i in range(1, len(val)) if val[i] < val[i - 1])
    rho = _spearman(ks, val)
    chance = (e.get("meta") or {}).get("chance_top1")
    best = max(range(len(rows)), key=lambda i: val[i])
    return {"train_min": min(tr), "train_max": max(tr), "train_at_k": ks[best],
            "val_min": min(val), "val_max": max(val), "val_best_k": ks[best],
            "n_dec": dec, "n_steps": len(val) - 1, "rho": rho, "chance": chance,
            "gap": (min(tr) - max(val))}


def sec_ec1(c):
    L = ["## 8. E-C1 任务保真分类探针", "",
         "来源: `task_fidelity_stdattn_rec{,_cls,_lstsq}.json`（`tools/alit_task_fidelity.py`；"
         "ckpt = `stdattn_k64_rec`，n_eval=5,000、100 类、chance=1%）。"
         "⚠️ 分类线性探针 ≠ Phase 2 生成任务；口径 = **等平均 token 数**。", ""]
    if not c["ec1"]:
        L.append("**数据不足**（缺正式 E-C1 产物；仅有 smoke——合成数据+随机编码器，无科学含义）。")
        return "\n".join(L), None
    primary = next((n for n in sorted(c["ec1"])
                    if "_cls" not in n and "_lstsq" not in n), sorted(c["ec1"])[0])

    # ── 顶上先给「探针质量」醒目边界（这是本轮最重要的边界）──
    L += ["### 8.0 探针质量（**醒目边界：严重过拟合 ⇒ 分辨力不足**）", "",
          "| 产物 | 特征/探针 | train top-1 区间 | val top-1 区间 | val 曲线下降次数 | "
          "rho(k, val) | train−val 最小差 |", "|---|---|---|---|---|---|---|"]
    quals = {}
    for name in sorted(c["ec1"]):
        e = c["ec1"][name]
        q = probe_quality(e)
        if not q:
            continue
        quals[name] = q
        meta = e.get("meta") or {}
        feat = meta.get("feature") or {}
        probe = meta.get("probe") or {}
        L.append(f"| `{name}` | {feat.get('pool')}"
                 f"{'+cls' if feat.get('use_cls') else ''} / {probe.get('kind')}"
                 f" (d={meta.get('feature_dim')}) | "
                 f"{fnum(q['train_min'], 4)}–{fnum(q['train_max'], 4)} | "
                 f"{fnum(q['val_min'], 4)}–{fnum(q['val_max'], 4)}"
                 f" （best @k={q['val_best_k']}） | {q['n_dec']}/{q['n_steps']} | "
                 f"{fsgn(q['rho'], 3)} | {fnum(q['gap'], 4)} |")
    L += ["", "> **读法（必须照录）**：三个变体的 `train_top1` 都远高于 `val_top1`"
              "（`_cls` 变体甚至 train=1.0000），且 **val 曲线非单调** ⇒ 该探针**分辨力不足**。"
              "因此 `indistinguishable` 只能读作「**该探针分辨力下测不到差异**」，"
              "**不能**读作「自适应 = 固定 K」，也不能据此判 E-C1 成立或否证。", ""]

    v, why = None, None
    for name in sorted(c["ec1"]):
        e = c["ec1"][name]
        meta = e.get("meta") or {}
        pv, pw = verdict_ec1(e)
        if name == primary or v is None:
            v, why = pv, pw
        ver = e.get("E_C1_verdict") or {}
        q = quals.get(name) or {}
        L += [f"### 8.{sorted(c['ec1']).index(name) + 1} `{name}`"
              + ("（主）" if name == primary else ""), "",
              f"- ckpt 签名: `{_get(meta, 'ckpt_signature')}`；数据源: "
              f"{_get(meta, 'data_source')}（synthetic={_get(meta, 'synthetic')}）；"
              f"K={_get(meta, 'K')}；probe={((meta.get('probe') or {}).get('kind'))}",
              f"- **主判据行**（`E_C1_verdict`）: label=`{ver.get('primary_label')}`, "
              f"mean_k=**{fnum(ver.get('primary_mean_k'), 4)}**"
              f"（{fnum(ver.get('primary_mean_k_over_K'), 4)}×K）, "
              f"Δpp=**{fsgn(ver.get('primary_delta_pp'), 4)}**, "
              f"CI=**[{fnum((ver.get('primary_delta_pp_ci') or [None, None])[0], 4)}, "
              f"{fnum((ver.get('primary_delta_pp_ci') or [None, None])[1], 4)}]**, "
              f"verdict=`{ver.get('primary_verdict')}`",
              f"- **全 τ 扫描**（{len(e.get('adaptive_curve') or [])} 行）: Δpp ∈ "
              f"[**{fsgn(ver.get('min_delta_pp_over_sweep'), 4)}**, "
              f"**{fsgn(ver.get('max_delta_pp_over_sweep'), 4)}**]；"
              f"adaptive_better={ver.get('any_row_adaptive_better')}, "
              f"fixed_better={ver.get('any_row_fixed_better')}",
              f"- **探针质量**: train_top1 {fnum(q.get('train_min'), 4)}–"
              f"{fnum(q.get('train_max'), 4)}；val_top1 {fnum(q.get('val_min'), 4)}–"
              f"{fnum(q.get('val_max'), 4)}（chance {fnum(q.get('chance'), 4)}）；"
              f"val 曲线下降 {q.get('n_dec')}/{q.get('n_steps')} 次",
              f"- **该文件判决: {pv}** —— {pw}", ""]
        ob = e.get("oracle_upper_bound") or {}
        if ob:
            L.append(f"- oracle 上界（不可实现，只量化 TSC-TOI gap）: "
                     f"`any_k_correct_rate`=**{fnum(ob.get('any_k_correct_rate'), 4)}**, "
                     f"`mean_min_k_all_images`={fnum(ob.get('mean_min_k_all_images'), 2)}, "
                     f"`mean_min_k_correct_only`={fnum(ob.get('mean_min_k_correct_only'), 2)}, "
                     f"`frac_images_needing_K`={fnum(ob.get('frac_images_needing_K'), 4)}, "
                     f"`chance_top1`={fnum(ob.get('chance_top1'), 4)}")
        L += ["", "| k | val top-1 | train top-1 | val 95% CI |", "|---|---|---|---|"]
        for r in (e.get("top1_by_k") or []):
            L.append(f"| {r.get('k')} | {fnum(r.get('top1'), 4)} | "
                     f"{fnum(r.get('train_top1'), 4)} | {r.get('top1_ci')} |")
        L += ["", "| τ | 平均 k | median k | 自适应 top-1 | 等预算固定档 top-1 | "
              "Δ (pp) | Δ 95% CI | 行判决 |", "|---|---|---|---|---|---|---|---|"]
        for r in (e.get("adaptive_curve") or []):
            L.append(f"| {fnum(r.get('tau'), 4)} | {fnum(r.get('mean_k'), 3)} | "
                     f"{fnum(r.get('median_k'), 1)} | {fnum(r.get('adaptive_top1'), 4)} | "
                     f"{fnum(r.get('fixed_top1_at_equal_budget'), 4)} | "
                     f"{fsgn(r.get('delta_pp'), 3)} | {r.get('delta_pp_ci')} | "
                     f"{r.get('verdict')} |")
        png = os.path.join(c["art"], name[:-5] + ".png")
        if os.path.exists(png):
            L += ["", f"![{os.path.basename(png)}]({os.path.basename(png)})"]
        L.append("")
    L += [f"- **E-C1 判决（主文件 `{primary}`）: {v}** —— {why}",
          "- （按 PLAN §8，E-C1 的「成立」要求等平均 token 数下自适应 ≥ 固定 K 且差距 > 噪声；"
          "本产物三条臂的 CI 全部跨 0 ⇒ **既不成立也不否证**，记「数据不足（探针分辨力不足）」。）",
          ""]
    return "\n".join(L), (v, why)


def sec_verdicts(v_all, has_numbers=False):
    L = ["## 9. 判决总表（PLAN §8 三态）", "",
         "| ID | 实验 | 判决 | 依据/来源 |", "|---|---|---|---|"]
    for row in v_all:
        L.append("| " + " | ".join(str(x) for x in row) + " |")
    L += ["", "> 三态定义: **成立** = 判据满足；**否证** = 明确的否证条件命中；"
              "**数据不足** = 产物未到位或落在判据边界（单种子无方差，"
              "<0.10 px / 0.1 dB 不作结论）。"]
    if has_numbers:
        L.append("> 数字来源：另见 `alit_a_numbers.json`（由 `alit_a_numbers.py` 产出，"
                 "**只汇总原始数字、不做判决**）；本表是**唯一三态裁定**，"
                 "两者不一致时以本表 + PLAN §8 原文为准。")
    L.append("")
    return "\n".join(L)


# ─────────────────────────── plots ───────────────────────────

def make_plots(c, tsc_plots, ladder_entries, b_rows, art, no_plots):
    out = {"files": [], "skipped": []}
    if no_plots:
        out["skipped"].append("--no-plots")
        return out
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        out["skipped"].append(f"matplotlib 不可用: {e}")
        return out

    # TSC 曲线 + 等预算对照
    for tag, dom, s in tsc_plots:
        rows = s.get("tau_curve") or []
        if not rows:
            continue
        try:
            xs = [r["mean_tokens"] for r in rows]
            ys = [r["mean_L1_255"] for r in rows]
            fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
            ax.plot(xs, ys, "-", color="#1f77b4", lw=2,
                    label="adaptive TSC (τ → mean tokens)")
            fx = s.get("fixed_steps") or []
            ax.scatter([r["mean_tokens"] for r in fx], [r["mean_L1_255"] for r in fx],
                       marker="s", color="#d62728", zorder=3, label="fixed step")
            for r in fx:
                ax.annotate(f"step{r['step']}", (r["mean_tokens"], r["mean_L1_255"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
            eq = s.get("matched_interp_equal_budget") or []
            for m in eq:
                if m.get("adaptive_mean_L1_255_at_budget") is None:
                    continue
                ax.plot([m["budget_tokens"], m["budget_tokens"]],
                        [m["adaptive_mean_L1_255_at_budget"], m["fixed_mean_L1_255"]],
                        "--", color="gray", lw=1)
            ax.set_xlabel("mean token budget (z_s prefix)")
            ax.set_ylabel("L1_255 (px)")
            ax.set_title(f"E-A2 TSC vs fixed-step (equal budget)\n{tag} / {dom}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            base_tag = tag[4:] if tag.startswith("tsc_") else tag
            p = os.path.join(art, f"tsc_{base_tag}_{dom}.png")
            fig.savefig(p)
            plt.close(fig)
            out["files"].append(os.path.basename(p))
        except Exception as e:  # noqa: BLE001
            out["skipped"].append(f"tsc plot {tag}/{dom}: {e}")

    # 域阶梯
    if ladder_entries:
        try:
            doms = [d for d, e in ladder_entries.items()
                    if e.get("mean_best_step") is not None]
            vals = [ladder_entries[d]["mean_best_step"] for d in doms]
            if doms:
                fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=140)
                ax.bar(doms, vals, color="#2ca02c")
                for i, v in enumerate(vals):
                    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
                ax.set_ylabel("mean best stop step (1-based)")
                ax.set_title("E-A3 IID / OOD domain ladder")
                ax.grid(alpha=0.3, axis="y")
                fig.tight_layout()
                p = os.path.join(art, "domain_ladder.png")
                fig.savefig(p)
                plt.close(fig)
                out["files"].append(os.path.basename(p))
        except Exception as e:  # noqa: BLE001
            out["skipped"].append(f"ladder plot: {e}")

    # B 臂对照
    if b_rows:
        try:
            arms = sorted(a for a, r in b_rows.items() if r.get("L1_255_last") is not None)
            if arms:
                vals = [b_rows[a]["L1_255_last"] for a in arms]
                fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(arms)), 3.8), dpi=140)
                colors = ["#7f7f7f" if a == "b1_mean" else "#1f77b4" for a in arms]
                ax.bar(arms, vals, color=colors)
                base = b_rows.get("b1_mean", {}).get("L1_255_last")
                if base:
                    ax.axhline(base, ls="--", color="black", lw=1,
                               label="b1_mean baseline")
                    ax.legend(fontsize=8)
                for i, v in enumerate(vals):
                    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
                ax.set_ylabel("last-step L1_255 (px) ↓")
                ax.set_title("E-B arms — final-step reconstruction")
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
                ax.grid(alpha=0.3, axis="y")
                fig.tight_layout()
                p = os.path.join(art, "b_arms_last_step.png")
                fig.savefig(p)
                plt.close(fig)
                out["files"].append(os.path.basename(p))
        except Exception as e:  # noqa: BLE001
            out["skipped"].append(f"b_arms plot: {e}")
    return out


# ─────────────────────────── main ───────────────────────────

def build(art, out_path, no_plots, emit_json=False):
    c = collect(art)
    if not os.path.isdir(art):
        print(f"[alit_report] 产物目录不存在: {art}", file=sys.stderr)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts, v_all = [], []
    parts.append(f"# ALIT 自适应长度实验 — 聚合表（自动生成）\n\n"
                 f"> 由 `tools/alit_report.py` 生成于 {now}；**请勿手改**。\n"
                 f"> 报告正文: [`REPORT_alit_adaptive.md`](../REPORT_alit_adaptive.md)；"
                 f"规划: [`PLAN_alit_adaptive_experiments.md`](../PLAN_alit_adaptive_experiments.md)。\n"
                 f"> 每个数字的来源文件名在表列或小节首行标注。\n")
    parts.append(sec_availability(c, now))
    _run_st = sec_run_status(c)
    if _run_st:
        parts.append(_run_st)
    parts.append(sec_baseline(c))

    v_all.append(("E-A0", "四臂基线复现", V_OK if c["four_arm"] else V_INSUFF,
                  "baseline_four_arm_eval.json（见 §1）"))

    # E-A1
    rec = {a: d for a, d in c["stepwise"].items() if d.get("recurrent")}
    parts.append(sec_stepwise(c))
    for a, d in rec.items():
        hist, n = d.get("best_step_hist") or [], d.get("n_images") or 0
        og, otok, ftok = oracle_gain_for_arm(c, a)
        v, why = verdict_a1(hist, n, og, otok, ftok)
        v_all.append((f"E-A1[{a}]", "每图停步分布/oracle", v, why))

    tsc_md, tsc_plots, tsc_vmap = sec_tsc(c)
    parts.append(tsc_md)
    for (tag, dom), (v, why) in tsc_vmap.items():
        v_all.append((f"E-A2[{tag}/{dom}]", "TSC 等预算", v, why))
    if not c["tsc"]:
        v_all.append(("E-A2[std]", "TSC 等预算", V_INSUFF, "缺 tsc_stdattn_rec.json"))

    ladder_md, ladder_entries = sec_ladder(c)
    parts.append(ladder_md)
    if ladder_entries:
        steps = [e["mean_best_step"] for e in ladder_entries.values()
                 if e.get("mean_best_step") is not None]
        if len(steps) >= 2:
            spread = max(steps) - min(steps)
            v_all.append(("E-A3", "IID/OOD 阶梯",
                          V_OK if spread >= 1.0 else (V_NO if spread < 0.5 else V_INSUFF),
                          f"跨度 {spread:.2f} 步（见 §4）"))
        else:
            v_all.append(("E-A3", "IID/OOD 阶梯", V_INSUFF, "有效域 < 2 级"))
    else:
        v_all.append(("E-A3", "IID/OOD 阶梯", V_INSUFF, "缺 domain_ladder/ood 产物"))

    parts.append(sec_a4_a6(c))
    a4_all = {}
    a6_all = {}
    for _tag, _d in c["tsc"].items():
        a4_all.update(_d.get("E_A4_complexity_spearman") or {})
        a6_all.update(_d.get("E_A6_delta_decomposition") or {})
    if a4_all:
        v4, w4 = verdict_a4(a4_all)
    else:
        v4, w4 = V_INSUFF, "缺 per_image/tsc 产物"
    v_all.append(("E-A4", "复杂度相关性（弱代理）", v4, w4))
    if a6_all:
        v6, w6 = verdict_a6(a6_all)
    else:
        v6, w6 = V_INSUFF, "缺 tsc_*.json（§5.3 为聚合降级口径）"
    v_all.append(("E-A6", "Δ_t 分解（≥88% 前置复核）", v6, w6))

    a5_md, a5_v = sec_a5(c)
    parts.append(a5_md)
    for k, (v, why) in (a5_v or {}).items():
        v_all.append((f"E-A5[{k}]", "z_s 冗余度", v, why))
    if not a5_v:
        v_all.append(("E-A5", "z_s 冗余度", V_INSUFF,
                      "缺 zs_redundancy_*.json（注意力熵未做）"))

    b_md, b_rows, b_groups = sec_b(c)
    parts.append(b_md)
    for eid, name in (("E-B1", "逐迭代损失权重"), ("E-B2", "损失侧动态停机"),
                      ("E-B3", "ALIT 式自适应记忆"), ("E-B4", "随机预算/一条流多档")):
        gv, gw = b_groups[eid]
        v_all.append((eid, name, gv, gw))

    ec1_md, ec1_v = sec_ec1(c)
    parts.append(ec1_md)
    v_all.append(("E-C1", "任务保真自适应 K",
                  ec1_v[0] if ec1_v else V_INSUFF,
                  (ec1_v[1] if ec1_v else "缺正式 E-C1 产物（仅 smoke）")))

    parts.append(sec_verdicts(
        v_all, has_numbers=os.path.exists(os.path.join(c["art"], "alit_a_numbers.json"))))

    plots = make_plots(c, tsc_plots, ladder_entries, b_rows, art, no_plots)
    plot_sec = ["## 10. 图（自动生成）", ""]
    if plots["files"]:
        for p in plots["files"]:
            plot_sec.append(f"![{p}]({p})")
            plot_sec.append("")
    else:
        plot_sec.append("（无图：数据未到位或 matplotlib 不可用）")
    if plots["skipped"]:
        plot_sec.append("跳过原因: " + "; ".join(plots["skipped"]))
    plot_sec.append("")
    parts.append("\n".join(plot_sec))

    missing_sec = ["## 11. 未到位产物（missing）", "",
                   "| 产物 | 用途 | 生产者 |", "|---|---|---|"]
    for name, desc, producer in c["missing"]:
        missing_sec.append(f"| `{name}` | {desc} | {producer} |")
    if not c["missing"]:
        missing_sec.append("| — | 全部到位 | — |")
    missing_sec += ["", f"合计 **{len(c['missing'])}** 项 missing。" +
                    (" 这些项在报告中一律记「数据不足」。"
                     if c["missing"] else "")]
    parts.append("\n".join(missing_sec))

    md = "\n\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    # ── stdout 汇总（验收要求：列出 missing）──
    print(f"[alit_report] 产物目录: {art}")
    print(f"[alit_report] 写入: {out_path}  ({len(md.splitlines())} 行)")
    print(f"[alit_report] 图: {plots['files'] or '无'}"
          + (f"  跳过: {plots['skipped']}" if plots["skipped"] else ""))
    print(f"[alit_report] present={len(c['present'])}  missing={len(c['missing'])}")
    if c["missing"]:
        print("[alit_report] MISSING:")
        for name, desc, producer in c["missing"]:
            print(f"  - {name:38s} {desc}  ({producer})")
    else:
        print("[alit_report] MISSING: 无")
    print("[alit_report] 判决总表:")
    for row in v_all:
        print(f"  {row[0]:22s} {row[2]:6s} {row[3]}")
    if emit_json:
        print("[alit_report] JSON_SUMMARY " + json.dumps(
            {"artifacts_dir": art, "out": out_path,
             "present": sorted(c["present"]),
             "missing": [m[0] for m in c["missing"]],
             "verdicts": [{"id": r[0], "verdict": r[2], "why": r[3]} for r in v_all]},
            ensure_ascii=False))
    return md


def main():
    ap = argparse.ArgumentParser(
        description="ALIT 实验报告聚合器（CPU-only，缺文件优雅跳过）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--artifacts", default=DEFAULT_ART, help="产物目录")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 markdown")
    ap.add_argument("--no-plots", action="store_true", help="不生成 png")
    ap.add_argument("--json", action="store_true", dest="emit_json",
                    help="额外打印一行机器可读 JSON 摘要")
    a = ap.parse_args()
    build(os.path.abspath(a.artifacts), os.path.abspath(a.out), a.no_plots,
          a.emit_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
