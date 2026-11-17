#!/usr/bin/env python3
"""SAVOIAS OOD 人评复杂度 × 重建误差分析（单步 / 任意 T 评测 dump 通用）
=======================================================================

用途
----
把 `eval_stepwise_linattn.py --dump_per_image` 的逐图产物，与 SAVOIAS 官方人评
复杂度 (`gt_{art,scenes,objects}.npy`) 按**文件名 stem（整数）** join，输出：

  <out>/summary.json   每子集 n / Spearman ρ 与 p / 复杂度三分位均值 L1 与 PSNR /
                       high−low 差（px 与相对 %）/ OOD−ID 差（若给 --id_per_image）
  <out>/tables.md      markdown 表（overview / 三分位 / OOD−ID / 分箱 / 汇总）
  <out>/fig_*.png/pdf  英文标签图（不依赖中文字体）:
                        · fig_complexity_l1_panels      三子集合图：散点 + 分箱均值曲线
                        · fig_complexity_l1_<sub>       每子集单图（art / scenes / objects）
                        · fig_rho_bar                   三子集 ρ(complexity,L1) / ρ(complexity,PSNR)
                        · fig_tercile_bar               三子集 × 三分位 L1（含 PSNR 副图）
                        · fig_ood_vs_id                 仅当给 --id_per_image
                        · fig_rho_vs_step               仅当 T>1 且给 --stepwise

设计口径（与 `doc/2026-10-26/REPORT_ood_savoias_scale.md` /
`tools/plot_savoias_scale.py` **完全一致，可直接对照数字**）
-----------------------------------------------------------------
  · L1_255 : 0–255 像素 L1、**不做 clamp**（canonical eval_stepwise_linattn.py 口径），
             取**末步**（T=1 单步模型即唯一那一步；可用 --metric_step 指定别的步）；
  · PSNR   : [0,1] clamp 空间、**逐图 PSNR 再平均**（主报告 eval_linattn.py 口径），末步；
  · 复杂度 : SAVOIAS 人评 0–100（gt npy 里是 int 或 np.int64）；
  · join   : `int(Path(record["file_name"]).stem)` ↔ gt 的文件名 stem（gt 里是
             `0.png`、图像可以是 `0.jpg`，所以必须按 stem 而不是按扩展名/全名 join）；
  · 三分位 : q1, q2 = np.percentile(S, [100/3, 200/3])；low: S<=q1，
             mid: q1<S<=q2，high: S>q2；gap = high−low；相对差 = gap/low*100；
  · Spearman: scipy.stats.spearmanr（complexity vs 指标），同时落 ρ 与 p。

与旧脚本 `tools/plot_savoias_scale.py` 的差异（数字口径不变）
-------------------------------------------------------------
  旧脚本面向"5 档容量阶梯"：默认 tags/ckpt/ID 参考全部硬编码、必须存在；只算 ρ
  不算 p；三分位只做 L1；图是容量/逐步曲线。本脚本面向**单次运行、任意 T（含
  T=1）**：--per_image 可以重复传、可传一个合并文件或三个子集文件；子集由
  per-image 里的 `category` / 图像路径 / 文件名提示自动判定；--stepwise 与
  --id_per_image 都是可选；额外给 p 值、PSNR 三分位、per-subset OOD−ID 与
  markdown 表。join/三分位/ρ 的算法逐位与旧脚本相同，因此 m240 旧 artifact 上
  重跑应与 REPORT 的 §4 数字完全一致（自测里做了该交叉验证）。

用法
----
  # 单步新模型（T=1）820 张一次 dump + ID 对照：
  python3 tools/analyze_savoias_single.py \\
      --per_image /root/autodl-tmp/tok448_eval/per_image_tok448.json \\
      --stepwise  /root/autodl-tmp/tok448_eval/stepwise_tok448.json \\
      --gt_dir    /root/autodl-tmp/savoias/gt \\
      --id_per_image /root/autodl-tmp/tok448_eval/per_image_id_in100.json \\
      --out       /root/autodl-tmp/tok448_eval/analysis \\
      --label     "448^2->16x448 single-step (T=1)"

  # 三个子集分别 dump（也可以只传一部分子集）：
  python3 tools/analyze_savoias_single.py \\
      --per_image art=per_image_art.json --per_image scenes=per_image_scenes.json \\
      --per_image objects=per_image_objects.json \\
      --gt_dir /root/autodl-tmp/savoias/gt --out <out>

  # 指定非末步（T>1 的多步模型）：
  python3 tools/analyze_savoias_single.py --per_image ... --gt_dir ... \\
      --out ... --metric_step 0

依赖：numpy + scipy（spearmanr）+ matplotlib（Agg）；pandas 不需要。
数据准备：`bash tools/fetch_savoias.sh <DATA_ROOT>`（gt 在 `<DATA_ROOT>/gt/`）。

自测（合成 500 张已知相关性 + 与旧 artifact 真实交叉验证）
-----------------------------------------------------------
  # 全量自测（合成数据；若本地有真实 gt 则追加真实 artifact 交叉验证）
  /usr/bin/python3 tools/selftest_analyze_savoias_single.py
  # 只跑合成数据部分（不联网、不碰真实 artifact）
  /usr/bin/python3 tools/selftest_analyze_savoias_single.py --skip-real
  # 手动等价命令（自测脚本内部即执行这些）：
  #   <out> 里应看到 summary.json / tables.md / 6+ 张 png+pdf；
  #   art 子集 ρ(L1) 应落在 0.80~0.95（合成 L1=0.05*complexity+8+noise, n=250），
  #   三分位分组与 gap 与手算逐位一致。

注意：本脚本只读输入、只写 --out 目录，不碰远端训练目录。
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    sys.exit(f"[error] matplotlib 不可用: {e}")

try:
    from scipy.stats import spearmanr
except Exception as e:  # pragma: no cover
    sys.exit(f"[error] 需要 scipy.stats.spearmanr（pip install scipy）: {e}")


# ─────────────────────────── 子集定义 ───────────────────────────
# key / 显示名 / 期望的图像目录名 / 图色 / 文件名/路径/类目名里可识别的别名
SUBSETS = [
    dict(key="art", pretty="People-Art", dir="Art", color="#d62728",
         aliases=("art", "people-art", "peopleart", "people_art")),
    dict(key="scenes", pretty="Places", dir="Scenes", color="#1f77b4",
         aliases=("scenes", "scene", "places", "place")),
    dict(key="objects", pretty="MSCOCO", dir="Objects", color="#2ca02c",
         aliases=("objects", "object", "mscoco", "ms-coco", "coco")),
]
SUB_BY_KEY = {s["key"]: s for s in SUBSETS}
SUB_KEYS = [s["key"] for s in SUBSETS]


def die(msg, code=2):
    """统一错误出口：明确报错 + 非 0 退出。"""
    sys.stderr.write(f"[error] {msg}\n")
    sys.exit(code)


# ─────────────────────────── 名称 / 子集判定 ───────────────────────────
def norm_name(s):
    """小写 + 非字母数字折叠成 '-'，便于匹配 Art/art/people-art/Art_420 等。"""
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def _match_subset_in_text(text):
    n = norm_name(text)
    if not n:
        return None
    for spec in SUBSETS:
        for alias in spec["aliases"]:
            if (n == alias or n.startswith(alias + "-")
                    or n.endswith("-" + alias) or ("-" + alias + "-") in n):
                return spec["key"]
    return None


def detect_subset(rec, hint_key, src_path):
    """按优先级判定一条 per-image 记录属于哪个子集。

    1) per-image 记录的 `category`（canonical dump = 图像所在目录名 Art/Scenes/Objects）；
    2) `path` 的各个路径分量（递归扫描时目录名带子集名）；
    3) `file_name`；
    4) 显式提示：`--per_image art=<file>` 的前缀，或该 per_image 文件名里的子集词
       （如 per_image_art.json / ood_scenes.json）。
    返回 key 或 None。
    """
    if hint_key:
        return hint_key
    cat = rec.get("category")
    if cat is not None:
        k = _match_subset_in_text(cat)
        if k:
            return k
    p = rec.get("path") or ""
    for part in reversed(Path(str(p)).parts):
        k = _match_subset_in_text(part)
        if k:
            return k
    k = _match_subset_in_text(rec.get("file_name") or "")
    if k:
        return k
    k = _match_subset_in_text(Path(str(src_path)).stem)
    return k


# ─────────────────────────── 基础 IO ───────────────────────────
def load_json(path, what="json"):
    p = Path(path)
    if not p.is_file():
        die(f"{what} 文件不存在: {p}")
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        die(f"{what} 解析失败 {p}: {e}")


def as_float_list(x):
    """标量 / list / ndarray → list[float]；None/空 → None。"""
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return [float(x)]
    try:
        out = [float(v) for v in x]
    except Exception:
        return None
    return out or None


def metric_at(rec, key, t, where):
    """取一条记录在 0-based 步 t 的指标值；缺失 → None（PSNR）或 die（L1）。"""
    v = as_float_list(rec.get(key))
    if v is None:
        if key == "L1_255":
            die(f"{where}: 记录缺少 {key}（该字段是必需项）: "
                f"file_name={rec.get('file_name')!r}")
        return None
    if not (-len(v) <= t < len(v)):
        die(f"{where}: 记录 {rec.get('file_name')!r} 的 {key} 只有 {len(v)} 步，"
            f"取不到第 {t} 步（0-based；--metric_step={t}）")
    return float(v[t])


def parse_per_image_args(vals):
    """--per_image 可重复传；支持 `art=<json>` / `scenes=<json>` / `objects=<json>` 前缀。"""
    if not vals:
        die("至少需要一次 --per_image <per_image.json>（可重复传，或用 art=... 前缀）")
    out, seen = [], set()
    for v in vals:
        hint, path = None, v
        if "=" in v:
            left, right = v.split("=", 1)
            k = _match_subset_in_text(left) if "://" not in v else None
            if k and right:
                hint, path = k, right
        p = Path(path).expanduser()
        if not p.is_file():
            die(f"--per_image 文件不存在: {p}")
        rp = str(p.resolve())
        if rp in seen:
            sys.stderr.write(f"[warn] --per_image 重复，已忽略: {p}\n")
            continue
        seen.add(rp)
        out.append((hint, p))
    return out


# ─────────────────────────── GT 读取 ───────────────────────────
def locate_gt_files(gt_dir):
    """在 gt_dir（或其 gt/ 子目录）里定位 gt_{art,scenes,objects}.npy。

    兼容 `fetch_savoias.sh` 的 `gt/gt_<key>.npy` 与官方原始名
    `global_ranking_<key>.npy`；找不到时给出**明确**期望路径。
    """
    if not gt_dir:
        die("缺少 --gt_dir（SAVOIAS gt npy 所在目录，如 /root/autodl-tmp/savoias/gt）")
    root = Path(gt_dir).expanduser()
    if not root.is_dir():
        die(f"--gt_dir 不存在或不是目录: {root}")
    search_dirs = [root] + ([root / "gt"] if (root / "gt").is_dir() else [])
    found, missing = {}, []
    for spec in SUBSETS:
        key = spec["key"]
        cand = []
        for d in search_dirs:
            cand += [d / f"gt_{key}.npy", d / f"global_ranking_{key}.npy"]
        hit = next((p for p in cand if p.is_file()), None)
        if hit is None:  # 宽松兜底：该目录下唯一包含 <key> 的 .npy
            g = [p for d in search_dirs for p in d.glob(f"*{key}*.npy")]
            g = sorted(set(g))
            if len(g) == 1:
                hit = g[0]
            elif len(g) > 1:
                die(f"gt 文件歧义（目录 {root}）：找到多个 *{key}*.npy: "
                    f"{[p.name for p in g]}；请整理成 gt_{key}.npy")
        if hit is None:
            missing.append(f"{search_dirs[0] / ('gt_' + key + '.npy')}")
        else:
            found[key] = hit
    if missing:
        die("缺少 SAVOIAS gt npy: " + " | ".join(missing)
            + f"\n  期望布局: {search_dirs[0]}/gt_{{art,scenes,objects}}.npy"
            + "（或 global_ranking_{art,scenes,objects}.npy）"
            + "\n  获取方式: bash tools/fetch_savoias.sh <DATA_ROOT>")
    return found


def load_gt_scores(path):
    """读 gt npy → ({stem:int -> score}, {name_lower -> score}, meta)。

    官方格式：`np.load(..., allow_pickle=True).item()` =
    `{index: [filename, score]}`（filename 如 '0.png'，score 0–100）。
    额外兼容：dict 值为纯 score（key 即文件名/下标）、list of [filename, score]。
    以**文件名 stem（整数）**为主键（gt 的 .png 与图像的 .jpg 对齐）。
    """
    p = Path(path)
    try:
        obj = np.load(p, allow_pickle=True)
    except Exception as e:
        die(f"gt npy 读取失败 {p}: {e}")
    try:
        obj = obj.item()
    except Exception:
        pass
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple, np.ndarray)):
        items = list(enumerate(np.asarray(obj, dtype=object)))
    else:
        die(f"gt npy 格式无法识别（期望 dict {{index: [filename, score]}}）: {p}")

    by_stem, by_name, n_dup = {}, {}, 0
    for k, v in items:
        if isinstance(v, dict):
            fname = v.get("filename") or v.get("file_name") or v.get("name")
            score = v.get("score", v.get("complexity"))
        elif isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            fname, score = v[0], v[1]
        else:
            fname, score = None, v
        try:
            score = float(score)
        except Exception:
            n_dup += 1
            continue
        if not np.isfinite(score):
            n_dup += 1
            continue
        fname = str(fname) if fname is not None else str(k)
        stem = Path(fname).stem
        score = int(score) if float(score).is_integer() else score
        if re.fullmatch(r"[+-]?\d+", stem):
            si = int(stem)
            if si in by_stem and by_stem[si] != score:
                n_dup += 1
            by_stem.setdefault(si, score)
        by_name.setdefault(fname.lower(), score)
        by_name.setdefault(norm_name(Path(fname).stem), score)
    if not by_stem and not by_name:
        die(f"gt npy 里没有可用条目: {p}")
    meta = {"path": str(p.resolve()), "n_entries": len(items),
            "n_by_stem": len(by_stem), "n_by_name": len(by_name),
            "n_bad": n_dup}
    return by_stem, by_name, meta


def join_score(rec, by_stem, by_name):
    """按 stem（优先）→ 全名 → stem 归一化名 join 复杂度；失败返回 (None, why)。"""
    fname = rec.get("file_name") or Path(str(rec.get("path") or "")).name
    if not fname:
        return None, "no filename"
    stem = Path(str(fname)).stem
    if re.fullmatch(r"[+-]?\d+", stem) and int(stem) in by_stem:
        return by_stem[int(stem)], "stem"
    low = str(fname).lower()
    if low in by_name:
        return by_name[low], "name"
    if norm_name(stem) in by_name:
        return by_name[norm_name(stem)], "norm-stem"
    return None, "unmatched"


# ─────────────────────────── 统计 ───────────────────────────
def spearman_xy(x, y):
    """Spearman ρ 与双侧 p（scipy）；退化输入返回 (nan, nan)。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), int(len(x))
    r = spearmanr(x, y)
    rho = float(r.correlation) if hasattr(r, "correlation") else float(r[0])
    p = float(r.pvalue) if hasattr(r, "pvalue") else float(r[1])
    return rho, p, int(len(x))


def quantile_bins(x, k):
    """分位分箱（与旧脚本 qbins 同口径）；返回 (bin_idx, edges)。"""
    x = np.asarray(x, float)
    u = np.unique(x)
    k = int(max(1, min(k, len(u))))
    qs = np.quantile(x, np.linspace(0, 1, k + 1))
    edges = np.quantile(x, np.linspace(0, 1, k + 1))
    adj = qs.copy()
    adj[0] -= 1e-9
    idx = np.clip(np.digitize(x, adj) - 1, 0, k - 1)
    return idx, edges, k


def binned_stats(x, y, k):
    """分箱均值 + SEM（与旧脚本 binned() 同口径）。"""
    idx, edges, kk = quantile_bins(x, k)
    xs, ys, es, ns = [], [], [], []
    for b in range(kk):
        m = idx == b
        if not m.any():
            continue
        xs.append(float(np.mean(np.asarray(x)[m])))
        ys.append(float(np.mean(np.asarray(y)[m])))
        es.append(float(np.std(np.asarray(y)[m], ddof=1) / np.sqrt(int(m.sum())))
                  if m.sum() > 1 else float("nan"))
        ns.append(int(m.sum()))
    return {"edges": [float(e) for e in edges], "complexity_mean": xs,
            "y_mean": ys, "y_sem": es, "n": ns, "k_effective": int(kk)}


def tercile_stats(S, y):
    """复杂度三分位（与 plot_savoias_scale.py terciles() 逐位同口径）。"""
    S = np.asarray(S, float)
    y = np.asarray(y, float)
    q1, q2 = np.percentile(S, [100 / 3, 200 / 3])
    masks = {"low": S <= q1, "mid": (S > q1) & (S <= q2), "high": S > q2}
    out = {"q1": float(q1), "q2": float(q2),
           "n": {k: int(m.sum()) for k, m in masks.items()},
           "complexity_mean": {k: (float(S[m].mean()) if m.any() else float("nan"))
                               for k, m in masks.items()}}
    for k, m in masks.items():
        out[k] = float(y[m].mean()) if m.any() else float("nan")
    out["gap_high_low"] = out["high"] - out["low"]
    out["gap_rel_pct"] = (out["gap_high_low"] / out["low"] * 100.0
                          if np.isfinite(out["low"]) and out["low"] != 0
                          else float("nan"))
    return out


def describe(x):
    x = np.asarray([v for v in np.asarray(x, float) if np.isfinite(v)], float)
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {"n": int(len(x)), "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)),
            "min": float(x.min()), "max": float(x.max())}


# ─────────────────────────── 数据装载 ───────────────────────────
def load_per_image_files(files):
    """读入所有 per_image json → (records, files_meta, T)。

    records: list of dict(subset, file_name, path, source, L1 list(0..T-1),
                          PSNR list|None, category)
    """
    records, files_meta = [], []
    T_expected = None
    for hint, p in files:
        d = load_json(p, "per-image json")
        if isinstance(d, list):
            images, top = d, {}
        elif isinstance(d, dict) and isinstance(d.get("images"), list):
            images, top = d["images"], d
        else:
            die(f"per-image json 里没有 'images' 列表: {p}")
        if not images:
            die(f"per-image json 的 images 为空: {p}")
        T_top = top.get("T")
        if T_top is None:
            first = next((r for r in images if as_float_list(r.get("L1_255"))), None)
            T_top = len(as_float_list(first.get("L1_255"))) if first else None
        if T_top is None:
            die(f"无法确定 T（T=1 单步模型应有 L1_255=[x]）: {p}")
        T_top = int(T_top)
        if T_expected is None:
            T_expected = T_top
        elif T_top != T_expected:
            die(f"多个 per-image 文件的 T 不一致: {p} T={T_top} vs 已有 T={T_expected}")
        for i, r in enumerate(images):
            where = f"{p.name}#images[{i}]"
            L1 = as_float_list(r.get("L1_255"))
            if L1 is None:
                die(f"{where}: 缺少必需字段 L1_255（请用 eval_stepwise_linattn.py "
                    f"--dump_per_image 重新 dump）")
            if len(L1) != T_top:
                die(f"{where}: L1_255 长度 {len(L1)} != T={T_top}"
                    f"（file_name={r.get('file_name')!r}）")
            PSNR = as_float_list(r.get("PSNR"))
            if PSNR is not None and len(PSNR) != T_top:
                sys.stderr.write(f"[warn] {where}: PSNR 长度 {len(PSNR)} != T={T_top}，"
                                 f"该图 PSNR 置空\n")
                PSNR = None
            fn = r.get("file_name") or Path(str(r.get("path") or "")).name
            if not fn:
                die(f"{where}: 记录既无 file_name 也无 path，无法 join gt")
            sub = detect_subset(r, hint, p)
            if sub is None:
                die(f"{where}: 无法判定子集（category={r.get('category')!r}, "
                    f"path={str(r.get('path'))[:120]!r}, 来源文件 {p.name}）。"
                    f"\n  处理：让 per-image 记录保留 canonical 的 category 字段，"
                    f"或用 `--per_image art=<file>` 显式标注子集")
            records.append({"subset": sub, "file_name": str(fn),
                            "path": str(r.get("path") or ""),
                            "source": str(p), "L1_255": L1, "PSNR": PSNR,
                            "category": r.get("category")})
        files_meta.append({
            "path": str(p.resolve()), "subset_hint": hint, "n_images": len(images),
            "T": T_top, "ckpt": top.get("ckpt"), "domain": top.get("domain"),
            "K": top.get("K"), "recurrent": top.get("recurrent"),
            "steps": top.get("steps"), "num_patches": top.get("num_patches"),
            "header_n_images": top.get("n_images")})
    if not records:
        die("所有 per-image 文件都没有记录")
    assert T_expected is not None
    return records, files_meta, int(T_expected)


def dedupe_records(records):
    """同 (subset, 文件名 stem) 重复时保留第一条（防重复传文件/跨文件重叠）。"""
    seen, out, n_dup = set(), [], 0
    for r in records:
        k = (r["subset"], Path(r["file_name"]).stem)
        if k in seen:
            n_dup += 1
            continue
        seen.add(k)
        out.append(r)
    return out, n_dup


def load_id_reference(path, T_expected, metric_step):
    """ID 对照：接受 per_image dump（'images'）或 stepwise 聚合（'L1_255' 数组）。"""
    d = load_json(path, "ID per-image json")
    if isinstance(d, dict) and isinstance(d.get("images"), list):
        L1s, Ps = [], []
        for r in d["images"]:
            l1 = as_float_list(r.get("L1_255"))
            ps = as_float_list(r.get("PSNR"))
            if l1 is None:
                continue
            t = metric_step if metric_step is not None and metric_step >= 0 else len(l1) - 1
            if not (-len(l1) <= t < len(l1)):
                die(f"ID dump 记录 {r.get('file_name')!r} 的 L1_255 只有 {len(l1)} 步，"
                    f"取不到 --metric_step={metric_step}")
            L1s.append(float(l1[t]))
            if ps is not None and -len(ps) <= t < len(ps):
                Ps.append(float(ps[t]))
        if not L1s:
            die(f"ID per-image json 里没有可用的 L1_255: {path}")
        return {"source": str(Path(path).resolve()), "n": len(L1s),
                "L1_255": float(np.mean(L1s)),
                "PSNR": float(np.mean(Ps)) if Ps else float("nan"),
                "kind": "per_image"}
    if isinstance(d, dict) and "L1_255" in d:
        l1 = as_float_list(d["L1_255"])
        if l1 is None:
            die(f"ID 聚合 json 的 L1_255 不可用: {path}")
        t = metric_step if metric_step is not None and metric_step >= 0 else len(l1) - 1
        if not (-len(l1) <= t < len(l1)):
            die(f"ID 聚合 json 只有 {len(l1)} 步，取不到 --metric_step={metric_step}")
        ps = as_float_list(d.get("PSNR"))
        return {"source": str(Path(path).resolve()),
                "n": int(d.get("n_images") or len(l1)),
                "L1_255": float(l1[t]),
                "PSNR": float(ps[t]) if ps is not None and -len(ps) <= t < len(ps)
                else float("nan"), "kind": "aggregate"}
    die(f"--id_per_image 既不是 per-image dump（无 'images'）也不是聚合 json"
        f"（无 'L1_255'）: {path}")


# ─────────────────────────── 画图 ───────────────────────────
def _fmt_p(p):
    if not np.isfinite(p):
        return "p=n/a"
    return "p<1e-4" if p < 1e-4 else f"p={p:.2g}"


def _l1_ylabel(T, t):
    return ("pixel L1 (0–255), single read (T=1)" if T == 1
            else f"pixel L1 (0–255) at step {t + 1}/{T}")


def _psnr_ylabel(T, t):
    return ("PSNR (dB), single read (T=1)" if T == 1
            else f"PSNR (dB) at step {t + 1}/{T}")


def _save(fig, out, stem, figs, dpi):
    fig.savefig(out / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    figs.extend([f"{stem}.png", f"{stem}.pdf"])


def fig_complexity_panels(sub_stats, out, T, t, label, figs, dpi, bins):
    """三子集合图 + 每子集单图：complexity–L1 散点 + 分位分箱均值曲线。"""
    active = [k for k in SUB_KEYS if k in sub_stats and sub_stats[k]["n"] > 0]
    ncol = max(1, len(active))
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.9), squeeze=False)
    for ax, key in zip(axes[0], active):
        st = sub_stats[key]
        spec = SUB_BY_KEY[key]
        S, L = np.asarray(st["_S"]), np.asarray(st["_L1"])
        ax.scatter(S, L, s=9, color="#9a9a9a", alpha=.40, edgecolors="none",
                   label=f"per-image (n={st['n']})", zorder=1)
        b = st["bins"]
        ax.errorbar(b["complexity_mean"], b["y_mean"], yerr=b["y_sem"], fmt="-o",
                    color=spec["color"], ms=4.5, lw=1.9, capsize=2, zorder=3,
                    label=f"quantile-bin mean ± SEM (k={len(b['y_mean'])})")
        ax.set_xlabel("SAVOIAS human-annotated complexity (0–100)", fontsize=9.5)
        ax.set_ylabel(_l1_ylabel(T, t), fontsize=9.5)
        ax.set_title(f"{spec['pretty']} (n={st['n']})\n"
                     f"ρ(L1)={st['spearman']['L1_255']['rho']:+.3f} "
                     f"({_fmt_p(st['spearman']['L1_255']['p'])})", fontsize=9.5)
        ax.grid(alpha=.28)
        ax.legend(fontsize=7.2, loc="upper left")
    fig.suptitle(f"SAVOIAS OOD: human complexity vs reconstruction error"
                 + (f" — {label}" if label else ""), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "fig_complexity_l1_panels", figs, dpi)

    # 每子集单图（png+pdf）
    for key in active:
        st = sub_stats[key]
        spec = SUB_BY_KEY[key]
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        S, L = np.asarray(st["_S"]), np.asarray(st["_L1"])
        ax.scatter(S, L, s=11, color="#9a9a9a", alpha=.42, edgecolors="none",
                   label=f"per-image (n={st['n']})", zorder=1)
        b = st["bins"]
        ax.errorbar(b["complexity_mean"], b["y_mean"], yerr=b["y_sem"], fmt="-o",
                    color=spec["color"], ms=4.5, lw=1.9, capsize=2, zorder=3,
                    label=f"quantile-bin mean ± SEM (k={len(b['y_mean'])})")
        ter = st["terciles"]
        ax.set_xlabel("SAVOIAS human-annotated complexity (0–100)", fontsize=10)
        ax.set_ylabel(_l1_ylabel(T, t), fontsize=10)
        ax.set_title(f"{spec['pretty']} (n={st['n']}): complexity ↑ ⇒ L1 ↑\n"
                     f"ρ(L1)={st['spearman']['L1_255']['rho']:+.3f} "
                     f"({_fmt_p(st['spearman']['L1_255']['p'])}), "
                     f"high−low tercile = {ter['gap_high_low']:+.2f} px "
                     f"({ter['gap_rel_pct']:+.1f}%)", fontsize=9)
        ax.grid(alpha=.28)
        ax.legend(fontsize=7.6, loc="upper left")
        fig.tight_layout()
        _save(fig, out, f"fig_complexity_l1_{key}", figs, dpi)


def fig_rho_bar(sub_stats, out, label, figs, dpi):
    """三子集 ρ 对比条形图（L1 与 PSNR 两个面板）。"""
    active = [k for k in SUB_KEYS if k in sub_stats and sub_stats[k]["n"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for ax, metric, ylab in [(axes[0], "L1_255", "Spearman ρ (complexity, L1_255)"),
                             (axes[1], "PSNR", "Spearman ρ (complexity, PSNR)")]:
        vals = [sub_stats[k]["spearman"][metric]["rho"] for k in active]
        ps = [sub_stats[k]["spearman"][metric]["p"] for k in active]
        col = [SUB_BY_KEY[k]["color"] for k in active]
        x = np.arange(len(active))
        bars = ax.bar(x, vals, width=.62, color=col, edgecolor="white", lw=.6)
        for xi, v, p in zip(x, vals, ps):
            if not np.isfinite(v):
                continue
            ax.text(xi, v + (0.04 if v >= 0 else -0.04), f"{v:+.3f}\n{_fmt_p(p)}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(0, color="k", lw=.7)
        ax.set_xticks(x)
        ax.set_xticklabels([SUB_BY_KEY[k]["pretty"] for k in active], fontsize=9)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_ylim(-1.25, 1.25)
        ax.grid(alpha=.25, axis="y")
        ax.set_title(f"complexity vs {metric} (per subset)", fontsize=10)
    fig.suptitle("SAVOIAS OOD complexity correlation: three human-rated subsets"
                 + (f" — {label}" if label else ""), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "fig_rho_bar", figs, dpi)


def fig_tercile_bar(sub_stats, out, T, t, label, figs, dpi):
    """三分位 L1 条形图（L1 与 PSNR 两个面板）。"""
    active = [k for k in SUB_KEYS if k in sub_stats and sub_stats[k]["n"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    groups = ["low", "mid", "high"]
    gcolors = ["#c6dbef", "#6baed6", "#2171b5"]
    width = .8 / len(active)
    for ax, metric, ylab in [(axes[0], "L1", _l1_ylabel(T, t)),
                             (axes[1], "PSNR", _psnr_ylabel(T, t))]:
        for gi, (g, gc) in enumerate(zip(groups, gcolors)):
            vals = [sub_stats[k][f"terciles{'_psnr' if metric == 'PSNR' else ''}"][g]
                    for k in active]
            x = np.arange(len(active)) + (gi - 1) * width
            ax.bar(x, vals, width=width * .92, color=gc, edgecolor="white", lw=.5,
                   label=f"{g} complexity tercile")
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(len(active)))
        ax.set_xticklabels([SUB_BY_KEY[k]["pretty"] for k in active], fontsize=9)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(alpha=.25, axis="y")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.18 if hi > 0 else hi * 0.82)
        ax.legend(fontsize=7.6, loc="lower right")
        ax.set_title(f"mean {metric} by complexity tercile", fontsize=10)
        for i, k in enumerate(active):
            ter = sub_stats[k]["terciles"]
            ax.text(i, ax.get_ylim()[1] * 0.98,
                    f"Δ={ter['gap_high_low']:+.2f}\n({ter['gap_rel_pct']:+.1f}%)",
                    ha="center", va="top", fontsize=7.2, color="#333333")
    fig.suptitle("Complexity penalty: high−low complexity tercile gap"
                 + (f" — {label}" if label else ""), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "fig_tercile_bar", figs, dpi)


def fig_ood_vs_id(sub_stats, id_ref, out, T, t, label, figs, dpi):
    """OOD 三子集 vs ID 参考（L1 / PSNR）。"""
    active = [k for k in SUB_KEYS if k in sub_stats and sub_stats[k]["n"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for ax, metric, ylab in [(axes[0], "L1_255", _l1_ylabel(T, t)),
                             (axes[1], "PSNR", _psnr_ylabel(T, t))]:
        vals = [sub_stats[k]["ood_minus_id"][metric] for k in active]
        col = [SUB_BY_KEY[k]["color"] for k in active]
        x = np.arange(len(active))
        ax.bar(x, vals, width=.6, color=col, edgecolor="white", lw=.6)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:+.3f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
        ax.axhline(0, color="k", lw=.7)
        ax.set_xticks(x)
        ax.set_xticklabels([SUB_BY_KEY[k]["pretty"] for k in active], fontsize=9)
        ax.set_ylabel(f"OOD − ID ({'L1_255 px' if metric == 'L1_255' else 'PSNR dB'})",
                      fontsize=10)
        ax.grid(alpha=.25, axis="y")
        ax.set_title(f"generalization gap vs ImageNet-100 ID (n={id_ref['n']})",
                     fontsize=9.5)
    fig.suptitle("OOD − ID: is the SAVOIAS penalty complexity-dependent?"
                 + (f" — {label}" if label else ""), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "fig_ood_vs_id", figs, dpi)


def fig_rho_vs_step(step_stats, out, label, figs, dpi):
    """ρ(complexity, L1) 随步数变化（仅 T>1）。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    for ax, metric in [(axes[0], "L1_255"), (axes[1], "PSNR")]:
        for key in SUB_KEYS:
            if key not in step_stats:
                continue
            v = step_stats[key][f"rho_{metric}"]
            x = np.arange(1, len(v) + 1)
            ax.plot(x, v, "-o", ms=3, lw=1.7, color=SUB_BY_KEY[key]["color"],
                    label=SUB_BY_KEY[key]["pretty"])
        ax.set_xlabel("decoding step t", fontsize=9.5)
        ax.set_ylabel(f"Spearman ρ (complexity, {metric})", fontsize=9.5)
        ax.axhline(0, color="k", lw=.6)
        ax.grid(alpha=.28)
        ax.legend(fontsize=7.6)
        ax.set_title(f"ρ vs step — {metric}", fontsize=9.8)
    fig.suptitle("Complexity correlation across decoding steps"
                 + (f" — {label}" if label else ""), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "fig_rho_vs_step", figs, dpi)


# ─────────────────────────── markdown 表 ───────────────────────────
def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def _sgn(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def write_tables_md(path, summary, T, t, label):
    S = summary["subsets"]
    active = [k for k in SUB_KEYS if k in S and S[k]["n"] > 0]
    L = []
    L.append(f"# SAVOIAS OOD complexity analysis{' — ' + label if label else ''}\n")
    L.append(f"Generated: {summary['generated_at']}  ")
    L.append(f"L1_255 / PSNR 取 **step {t + 1}/{T}**"
             + ("（单步模型 T=1：唯一一步）" if T == 1 else "（默认末步）")
             + "；L1_255 = 0–255 像素 L1（不 clamp），PSNR = [0,1] clamp 空间逐图 PSNR 再平均。\n")
    L.append("Inputs:")
    for f in summary["inputs"]["per_image"]:
        L.append(f"- per-image: `{f}`")
    L.append(f"- gt_dir: `{summary['inputs']['gt_dir']}`")
    if summary["inputs"].get("stepwise"):
        L.append(f"- stepwise: `{summary['inputs']['stepwise']}`")
    if summary["inputs"].get("id_per_image"):
        L.append(f"- id_per_image: `{summary['inputs']['id_per_image']}`")
    L.append("")

    L.append("## 1. Subset overview\n")
    L.append("| subset | n | complexity mean ± std | L1_255 ↓ | PSNR ↑ | "
             "Spearman ρ(complexity, L1) | p | Spearman ρ(complexity, PSNR) | p |")
    L.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for k in active:
        st = S[k]
        c = st["complexity"]
        L.append(f"| {st['pretty']} | {st['n']} | {_f(c['mean'], 1)} ± {_f(c['std'], 1)} | "
                 f"{_f(st['L1_255']['mean'])} | {_f(st['PSNR']['mean'])} | "
                 f"{_sgn(st['spearman']['L1_255']['rho'])} | "
                 f"{_fmt_p(st['spearman']['L1_255']['p'])} | "
                 f"{_sgn(st['spearman']['PSNR']['rho'])} | "
                 f"{_fmt_p(st['spearman']['PSNR']['p'])} |")
    if "aggregate" in summary:
        a = summary["aggregate"]
        L.append(f"| **all joined** | {a['n']} | — | {_f(a['L1_255'])} | "
                 f"{_f(a['PSNR'])} | — | — | — | — |")
    L.append("")

    L.append("## 2. Complexity terciles (q1/q2 = 33.3/66.7 percentile; "
             "low: S≤q1, mid: q1<S≤q2, high: S>q2)\n")
    L.append("| subset | n (low/mid/high) | low L1 | mid L1 | high L1 | "
             "high−low (px) | relative (%) | low PSNR | mid PSNR | high PSNR | "
             "high−low (dB) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in active:
        st = S[k]
        ter, terp = st["terciles"], st["terciles_psnr"]
        n = ter["n"]
        L.append(f"| {st['pretty']} | {n['low']}/{n['mid']}/{n['high']} | "
                 f"{_f(ter['low'])} | {_f(ter['mid'])} | {_f(ter['high'])} | "
                 f"{_f(ter['gap_high_low'])} | {_f(ter['gap_rel_pct'], 1)} | "
                 f"{_f(terp['low'])} | {_f(terp['mid'])} | {_f(terp['high'])} | "
                 f"{_sgn(terp['gap_high_low'])} |")
    L.append("")

    if "id_reference" in summary and summary["id_reference"]:
        idr = summary["id_reference"]
        L.append("## 3. OOD − ID (ID = ImageNet-100 val per-image reference)\n")
        L.append(f"ID reference: n={idr['n']}, L1_255={_f(idr['L1_255'])}, "
                 f"PSNR={_f(idr['PSNR'])} dB (`{idr['source']}`, {idr['kind']}).\n")
        L.append("| subset | L1_255 (OOD) | L1_255 (ID) | OOD−ID (px) | OOD−ID (%) | "
                 "PSNR (OOD) | PSNR (ID) | OOD−ID (dB) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k in active:
            st = S[k]
            d = st["ood_minus_id"]
            L.append(f"| {st['pretty']} | {_f(st['L1_255']['mean'])} | "
                     f"{_f(idr['L1_255'])} | {_sgn(d['L1_255'])} | "
                     f"{_sgn(d['L1_255_rel_pct'], 1)} | "
                     f"{_f(st['PSNR']['mean'])} | {_f(idr['PSNR'])} | "
                     f"{_sgn(d['PSNR'])} |")
        if "aggregate" in summary and summary["aggregate"].get("ood_minus_id"):
            a = summary["aggregate"]
            d = a["ood_minus_id"]
            L.append(f"| **all joined** | {_f(a['L1_255'])} | {_f(idr['L1_255'])} | "
                     f"{_sgn(d['L1_255'])} | {_sgn(d['L1_255_rel_pct'], 1)} | "
                     f"{_f(a['PSNR'])} | {_f(idr['PSNR'])} | {_sgn(d['PSNR'])} |")
        L.append("")

    L.append(f"## {4 if summary.get('id_reference') else 3}. Quantile bins "
             f"(k≤{summary['inputs']['bins']}) — complexity vs L1_255\n")
    L.append("| subset | bin | n | complexity mean | L1_255 mean | L1 SEM | "
             "PSNR mean | complexity range |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for k in active:
        st = S[k]
        b = st["bins"]
        ranges = [(b["edges"][i], b["edges"][i + 1]) for i in range(len(b["y_mean"]))]
        for i in range(len(b["y_mean"])):
            lo, hi = ranges[i]
            L.append(f"| {st['pretty']} | {i + 1} | {b['n'][i]} | "
                     f"{_f(b['complexity_mean'][i], 1)} | {_f(b['y_mean'][i])} | "
                     f"{_f(b['y_sem'][i])} | "
                     f"{_f(st['psnr_bins']['y_mean'][i]) if i < len(st['psnr_bins']['y_mean']) else '—'} | "
                     f"[{_f(lo, 1)}, {_f(hi, 1)}] |")
    L.append("")

    if "stepwise" in summary and summary["stepwise"]:
        sw = summary["stepwise"]
        L.append("## Stepwise aggregate (from --stepwise)\n")
        L.append("| t | step | L1_255 | PSNR |")
        L.append("|---:|---:|---:|---:|")
        steps = sw.get("steps") or list(range(1, len(sw["L1_255"]) + 1))
        for i, (l1, ps) in enumerate(zip(sw["L1_255"], sw["PSNR"])):
            L.append(f"| {i + 1} | {steps[i] if i < len(steps) else i + 1} | "
                     f"{_f(l1)} | {_f(ps)} |")
        if sw.get("best_step_1based") is not None:
            L.append(f"\nargmin L1_255 step = {sw['best_step_1based']} "
                     f"(T={sw['T']}, n={sw['n_images']}).")
        L.append("")

    L.append("## Notes\n")
    L.append("- Join: SAVOIAS gt `{index: [filename, score 0–100]}` 按 **文件名 stem（整数）** "
             "与 per-image `file_name` 对齐（gt 可能是 `.png`、图像是 `.jpg`，因此不能按全名 join）。")
    L.append("- 判据：ρ > 0 且 high−low 三分位差 > 0 ⇒ \"越复杂越难重建\"；"
             "relative % = (high−low)/low×100。")
    L.append("- 图：`fig_complexity_l1_panels.png/pdf`、`fig_complexity_l1_<sub>.png/pdf`、"
             "`fig_rho_bar.png/pdf`、`fig_tercile_bar.png/pdf`"
             + ("、`fig_ood_vs_id.png/pdf`" if summary.get("id_reference") else "")
             + ("、`fig_rho_vs_step.png/pdf`" if summary.get("stepwise") and T > 1 else "")
             + "。")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ─────────────────────────── 汇总计算 ───────────────────────────
def compute_subset_stats(key, recs, gt, T, t, n_bins):
    """一个子集的 join + ρ + 三分位 + 分箱。"""
    pretty = SUB_BY_KEY[key]["pretty"]
    by_stem, by_name, gt_meta = gt
    S, L1, PSNR, unmatched = [], [], [], []
    reasons = {}
    for r in recs:
        score, why = join_score(r, by_stem, by_name)
        if score is None:
            unmatched.append(Path(r["file_name"]).stem)
            reasons[why] = reasons.get(why, 0) + 1
            continue
        L1.append(metric_at(r, "L1_255", t, f"{key} join"))
        S.append(float(score))
        ps = metric_at(r, "PSNR", t, f"{key} join")
        PSNR.append(ps if ps is not None else float("nan"))
    n = len(S)
    st = {"pretty": pretty, "n": n, "n_gt": gt_meta["n_by_stem"],
          "n_unmatched": len(unmatched),
          "unmatched_examples": unmatched[:8],
          "unmatched_reasons": reasons,
          "gt_meta": gt_meta}
    if n == 0:
        return st, unmatched
    S = np.asarray(S, float)
    L1 = np.asarray(L1, float)
    PSNR = np.asarray(PSNR, float)
    has_psnr = np.isfinite(PSNR).any()

    st["complexity"] = describe(S)
    st["L1_255"] = describe(L1)
    st["PSNR"] = describe(PSNR)
    st["spearman"] = {
        "L1_255": dict(zip(("rho", "p", "n_used"), spearman_xy(S, L1))),
        "PSNR": dict(zip(("rho", "p", "n_used"), spearman_xy(S, PSNR)))
        if has_psnr else {"rho": float("nan"), "p": float("nan"), "n_used": 0}}
    st["terciles"] = tercile_stats(S, L1)
    st["terciles_psnr"] = tercile_stats(S, PSNR) if has_psnr else {
        "low": float("nan"), "mid": float("nan"), "high": float("nan"),
        "gap_high_low": float("nan"), "gap_rel_pct": float("nan")}
    if not has_psnr or "q1" not in st["terciles_psnr"]:
        st["terciles_psnr"]["q1"] = st["terciles"]["q1"]
        st["terciles_psnr"]["q2"] = st["terciles"]["q2"]
        st["terciles_psnr"]["n"] = dict(st["terciles"]["n"])
    st["bins"] = binned_stats(S, L1, n_bins)
    st["psnr_bins"] = binned_stats(S, PSNR, n_bins) if has_psnr else {
        "y_mean": [], "y_sem": [], "n": []}
    # 内部数组只用于画图/逐步，不落 JSON
    st["_S"], st["_L1"], st["_PSNR"] = S, L1, PSNR
    return st, unmatched


def main():
    ap = argparse.ArgumentParser(
        description="SAVOIAS OOD 人评复杂度 × 重建误差分析（单步 T=1 / 任意 T）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--per_image", "--per-image", action="append", default=[],
                    metavar="JSON", help="eval_stepwise_linattn.py --dump_per_image 产物；"
                    "可重复传；支持 `art=<f>`/`scenes=<f>`/`objects=<f>` 显式标注；"
                    "也可传一个含 820 张的合并文件")
    ap.add_argument("--gt_dir", "--gt-dir", default="", metavar="DIR",
                    help="SAVOIAS gt npy 目录（含 gt_{art,scenes,objects}.npy，"
                    "或 global_ranking_*.npy；也可给其父目录）")
    ap.add_argument("--stepwise", default="", metavar="JSON",
                    help="可选：eval_stepwise_linattn.py --out 聚合 json（单步模型 T=1 只有一行；"
                    "T>1 时追加 ρ-per-step 与 fig_rho_vs_step）")
    ap.add_argument("--id_per_image", "--id-per-image", default="", metavar="JSON",
                    help="可选：ImageNet-100 val 的 per-image dump（或聚合 json），作 ID 对照")
    ap.add_argument("--out", required=True, metavar="DIR", help="输出目录（summary.json/tables.md/图）")
    ap.add_argument("--metric_step", "--metric-step", "--step", dest="metric_step",
                    type=int, default=-1,
                    help="-1=末步（单步 T=1 即唯一一步）；>=0 为 0-based 步号")
    ap.add_argument("--bins", type=int, default=6, help="分位分箱数（图/表用）")
    ap.add_argument("--label", default="", help="图标题后缀 / 表头标签（如 'tok448 T=1'）")
    ap.add_argument("--dpi", type=int, default=300, help="png 图 DPI")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # 1) gt
    gt_files = locate_gt_files(args.gt_dir)
    gt = {k: load_gt_scores(p) for k, p in gt_files.items()}

    # 2) per-image（可能多个文件；支持显式子集前缀）
    files = parse_per_image_args(args.per_image)
    records, files_meta, T = load_per_image_files(files)
    records, n_dup = dedupe_records(records)
    if n_dup:
        sys.stderr.write(f"[warn] 忽略 {n_dup} 条重复记录（同 subset+stem）\n")
    t = T - 1 if args.metric_step < 0 else args.metric_step
    if not (0 <= t < T):
        die(f"--metric_step={args.metric_step} 越界：T={T}（0-based 合法范围 0..{T - 1}）")
    n_by_sub = {}
    for r in records:
        n_by_sub[r["subset"]] = n_by_sub.get(r["subset"], 0) + 1
    seen_subs = [k for k in SUB_KEYS if n_by_sub.get(k, 0) > 0]
    if not seen_subs:
        die("per-image 里没有可识别子集的记录；请检查 category/path/file_name 或显式 "
            "`--per_image art=<file>`")
    for k in SUB_KEYS:
        if n_by_sub.get(k, 0) == 0:
            sys.stderr.write(f"[warn] 子集 {k} ({SUB_BY_KEY[k]['pretty']}) 没有任何 "
                             f"per-image 记录（跳过）\n")

    # 3) 逐子集 join + 统计
    sub_stats = {}
    for key in seen_subs:
        recs = [r for r in records if r["subset"] == key]
        st, _ = compute_subset_stats(key, recs, gt[key], T, t, args.bins)
        if st["n"] == 0:
            die(f"子集 {key}: 记录 {len(recs)} 条但 join gt 全失败（0 命中）。"
                f"\n  gt 文件: {gt[key][2]['path']}（{gt[key][2]['n_by_stem']} 个 stem）"
                f"\n  记录文件名示例: {[r['file_name'] for r in recs[:5]]}"
                f"\n  gt 文件名示例: 见 gt npy；join 规则=文件名 stem（整数）")
        if st["n_unmatched"] > 0:
            sys.stderr.write(
                f"[warn] {key}: {st['n_unmatched']}/{len(recs)} 条记录未 join 上 gt"
                f"（原因 {st['unmatched_reasons']}；示例 stem {st['unmatched_examples']}）"
                f"\n  常见原因: per-image 文件名不是 gt 里的整数 stem（gt 为 "
                f"{gt[key][2]['n_by_stem']} 条 {SUBSETS[SUB_KEYS.index(key)]['dir']}/<int>.*）\n")
        if st["n"] < 3:
            sys.stderr.write(f"[warn] {key}: n={st['n']} 太小，ρ 不可靠\n")
        sub_stats[key] = st

    # 4) ID 对照
    id_ref = None
    if args.id_per_image:
        id_ref = load_id_reference(args.id_per_image, T, args.metric_step)
        for key, st in sub_stats.items():
            st["ood_minus_id"] = {
                "L1_255": st["L1_255"]["mean"] - id_ref["L1_255"],
                "L1_255_rel_pct": ((st["L1_255"]["mean"] - id_ref["L1_255"])
                                   / id_ref["L1_255"] * 100.0
                                   if id_ref["L1_255"] else float("nan")),
                "PSNR": (st["PSNR"]["mean"] - id_ref["PSNR"]
                         if np.isfinite(st["PSNR"]["mean"]) and
                         np.isfinite(id_ref["PSNR"]) else float("nan"))}

    # 5) 全量（join 上的）聚合
    all_L1 = np.concatenate([sub_stats[k]["_L1"] for k in sub_stats])
    all_PSNR = np.concatenate([sub_stats[k]["_PSNR"] for k in sub_stats])
    aggregate = {"n": int(len(all_L1)),
                 "L1_255": float(np.nanmean(all_L1)),
                 "PSNR": float(np.nanmean(all_PSNR)) if np.isfinite(all_PSNR).any()
                 else float("nan")}
    if id_ref is not None:
        aggregate["ood_minus_id"] = {
            "L1_255": aggregate["L1_255"] - id_ref["L1_255"],
            "L1_255_rel_pct": ((aggregate["L1_255"] - id_ref["L1_255"])
                               / id_ref["L1_255"] * 100.0
                               if id_ref["L1_255"] else float("nan")),
            "PSNR": (aggregate["PSNR"] - id_ref["PSNR"]
                     if np.isfinite(aggregate["PSNR"]) and
                     np.isfinite(id_ref["PSNR"]) else float("nan"))}

    # 6) stepwise（可选）
    stepwave = None
    if args.stepwise:
        sw = load_json(args.stepwise, "stepwise json")
        if not isinstance(sw, dict) or not isinstance(sw.get("L1_255"), list):
            die(f"--stepwise 里没有 L1_255 数组: {args.stepwise}")
        l1 = [float(v) for v in sw["L1_255"]]
        ps = [float(v) for v in sw.get("PSNR", [])] if isinstance(sw.get("PSNR"), list) else []
        T_sw = len(l1)
        if T_sw != T:
            sys.stderr.write(f"[warn] stepwise T={T_sw} 与 per-image T={T} 不一致；"
                             f"仅记录聚合值，不计算 per-step ρ\n")
        best = sw.get("best_step")
        stepwave = {"source": str(Path(args.stepwise).resolve()), "T": T_sw,
                    "steps": sw.get("steps"), "K": sw.get("K"),
                    "recurrent": sw.get("recurrent"),
                    "n_images": sw.get("n_images"),
                    "L1_255": l1, "PSNR": ps,
                    "best_step_0based": int(best) if best is not None else None,
                    "best_step_1based": int(best) + 1 if best is not None else None,
                    "cum_z_s": sw.get("cum_z_s"),
                    "const_baseline_255": sw.get("const_baseline_255")}

    # 7) per-step ρ（T>1 且 per-image 步数够）
    step_rho = {}
    if stepwave is not None and T > 1 and stepwave["T"] == T:
        for key, st in sub_stats.items():
            matched = [x for x in records if x["subset"] == key
                       and join_score(x, gt[key][0], gt[key][1])[0] is not None]
            Sx = np.asarray(st["_S"], float)
            rho_l1, rho_ps = [], []
            for tt in range(T):
                l1 = np.asarray([metric_at(r, "L1_255", tt, key) for r in matched],
                                float)
                pv = [metric_at(r, "PSNR", tt, key) for r in matched]
                pv = np.asarray([np.nan if v is None else v for v in pv], float)
                rho_l1.append(spearman_xy(Sx, l1)[0]
                              if len(l1) == len(Sx) else float("nan"))
                rho_ps.append(spearman_xy(Sx, pv)[0]
                              if len(pv) == len(Sx) else float("nan"))
            step_rho[key] = {"rho_L1_255": rho_l1, "rho_PSNR": rho_ps}

    # 8) summary.json
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = {
        "note": "SAVOIAS OOD human-complexity (0–100) vs reconstruction error. "
                "L1_255 = 0–255 pixel L1 without clamp; PSNR = per-image PSNR in "
                "[0,1]-clamped space, then averaged. Single-step model: T=1 ⇒ "
                "metric step is the only read. Terciles: percentile(100/3, 200/3).",
        "generated_at": generated,
        "label": args.label,
        "protocol": {
            "L1_255": "0-255 pixel L1, no clamp (canonical eval_stepwise_linattn.py)",
            "PSNR": "dB in [0,1] clamp space, per-image PSNR then averaged",
            "metric_step_0based": int(t), "metric_step_1based": int(t + 1),
            "T": int(T),
            "terciles": "q1,q2=np.percentile(S,[100/3,200/3]); "
                        "low: S<=q1, mid: q1<S<=q2, high: S>q2; "
                        "gap=high-low; rel%=gap/low*100",
            "spearman": "scipy.stats.spearmanr(complexity, metric), two-sided p",
            "join": "int(Path(file_name).stem) -> gt npy filename stem",
        },
        "inputs": {
            "per_image": [str(p.resolve()) for _, p in files],
            "per_image_subset_hints": {str(p.resolve()): h for h, p in files},
            "gt_dir": str(Path(args.gt_dir).expanduser().resolve()),
            "gt_files": {k: str(p.resolve()) for k, p in gt_files.items()},
            "stepwise": str(Path(args.stepwise).expanduser().resolve())
            if args.stepwise else None,
            "id_per_image": str(Path(args.id_per_image).expanduser().resolve())
            if args.id_per_image else None,
            "metric_step": int(args.metric_step), "bins": int(args.bins),
        },
        "per_image_files": files_meta,
        "subsets": {},
        "aggregate": aggregate,
        "id_reference": id_ref,
        "stepwise": stepwave,
        "stepwise_rho": step_rho or None,
        "figures": [],
        "tables_md": "tables.md",
    }
    for key, st in sub_stats.items():
        entry = {k: v for k, v in st.items()
                 if k not in ("_S", "_L1", "_PSNR")}
        entry["n_records"] = int(n_by_sub.get(key, 0))
        if "ood_minus_id" not in entry and id_ref is not None:
            entry["ood_minus_id"] = {
                "L1_255": entry["L1_255"]["mean"] - id_ref["L1_255"],
                "L1_255_rel_pct": float("nan"), "PSNR": float("nan")}
        summary["subsets"][key] = entry

    figs = []
    # 9) 图
    if sub_stats:
        fig_complexity_panels(sub_stats, out, T, t, args.label, figs, args.dpi,
                              args.bins)
        fig_rho_bar(sub_stats, out, args.label, figs, args.dpi)
        fig_tercile_bar(sub_stats, out, T, t, args.label, figs, args.dpi)
        if id_ref is not None:
            fig_ood_vs_id(sub_stats, id_ref, out, T, t, args.label, figs, args.dpi)
        if step_rho:
            fig_rho_vs_step(step_rho, out, args.label, figs, args.dpi)
    summary["figures"] = figs

    # 10) 落盘 tables.md / summary.json
    write_tables_md(out / "tables.md", summary, T, t, args.label)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # 11) 控制台摘要
    print(f"[analyze] out={out}")
    print(f"  per-image: {len(records)} records / {len(files)} file(s); T={T}; "
          f"metric step {t + 1}/{T}"
          + (" (single-step model)" if T == 1 else ""))
    if id_ref is not None:
        print(f"  ID reference: n={id_ref['n']} L1_255={id_ref['L1_255']:.3f} "
              f"PSNR={id_ref['PSNR']:.3f} dB")
    if stepwave is not None:
        print(f"  stepwise: T={stepwave['T']} n={stepwave['n_images']} "
              f"L1_255={stepwave['L1_255'][-1]:.3f} PSNR={stepwave['PSNR'][-1]:.3f} dB"
              if stepwave["PSNR"] else "")
    for key in SUB_KEYS:
        if key not in sub_stats:
            continue
        st = sub_stats[key]
        rho = st["spearman"]["L1_255"]["rho"]
        p = st["spearman"]["L1_255"]["p"]
        ter = st["terciles"]
        line = (f"  {st['pretty']:10s} n={st['n']:4d} "
                f"L1={st['L1_255']['mean']:7.3f} PSNR={st['PSNR']['mean']:7.3f} dB  "
                f"ρ(L1)={rho:+.3f} ({_fmt_p(p)})  "
                f"high−low={ter['gap_high_low']:+.3f}px ({ter['gap_rel_pct']:+.1f}%)")
        if "ood_minus_id" in st:
            line += f"  ΔL1(OOD−ID)={st['ood_minus_id']['L1_255']:+.3f}px"
        print(line)
    print(f"  wrote: summary.json, tables.md, {len(figs)} figure files")


if __name__ == "__main__":
    main()
