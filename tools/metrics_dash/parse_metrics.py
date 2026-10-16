#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AssocVTC 训练指标统一解析器。

把下列三类数据源解析成一份统一的 ``metrics.json`` + 若干 CSV：

1. 训练日志（HF Trainer / accelerate，**训练中会持续增长**）
   ``/root/train_logs/stack2x_slice05.log``      新实验
   ``/root/train_logs/blockdiag_slice05.log``    基线对照
   其中周期性打印的 Python dict 形如::

       {'loss': '1.117', 'grad_norm': '0.6628', 'learning_rate': '4.523e-05', 'epoch': '0.3653'}

   注意 **dict 里没有 step**，只有 ``epoch``；本实验
   ``steps_per_epoch = 7009 // (16*2) = 219``，故 ``step ≈ round(epoch * 219)``，
   并用日志里 tqdm 的 ``N/8760`` 交叉校验（见 ``tqdm_step`` / ``tqdm_total``）。
   eval 行形如::

       {'eval_loss': '0.3316', 'eval_recon': '0.3318', 'eval_runtime': '97.37', 'epoch': '40'}

2. 探针 JSON（step1~5 关键指标，训练结束后由既有链路生成）
   ``.../output/probe/probe_stack2x_slice05.json``
   字段：``step_px_scale``(5)、``prog_curve_255``(5)、``E_px``(5x5)、``z_s_within_std``、
   ``z_s_block_cos``(5)、``steps``、``tag``、``n_img``。支持同一 run 的**多个
   checkpoint** 探针文件，自动按 step 排序成"随训练进程的变化"。

3. 全量推理 JSON（训练后生成，当前可能不存在）
   ``.../output/phase1_v2_stack2x_slice05/infer_test.json``
   字段：``full_norm_l1``、``full_pixel_l1_255``、``step_pixel_l1_255``。

设计约束
--------
* **只读**：绝不写入 / 删除被监控的训练日志与 JSON，绝不触碰训练进程。
* **纯标准库**：不需要 torch / pandas，任意 Python 3.8+ 可跑。
* **幂等**：每次运行完整重写 ``metrics.json`` 与 CSV；文件缺失只是记为 missing。
* 日志正在被写入时也能安全解析（容忍半行，靠花括号配对失败自然跳过）。

用法::

    python parse_metrics.py                        # 输出到 ./data/
    python parse_metrics.py --out /root/sr-metrics-dash/data
    python parse_metrics.py --quiet                # 不打印摘要
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------------------
# 配置（可用环境变量 SRDASH_ROOT / SRDASH_LOGDIR 覆盖，便于在别处复现）
# --------------------------------------------------------------------------------------

PROJECT_ROOT = os.environ.get("SRDASH_ROOT", "/root/autodl-tmp/sr-diffusion-v3-stack2x")
TRAIN_LOG_DIR = os.environ.get("SRDASH_LOGDIR", "/root/train_logs")

#: 本实验 steps_per_epoch = 7009 // (16*2) = 219
STEPS_PER_EPOCH = int(os.environ.get("SRDASH_STEPS_PER_EPOCH", "219"))
#: 日志里 tqdm 的总步数（8760 = 219 * 40）
TOTAL_STEPS = int(os.environ.get("SRDASH_TOTAL_STEPS", "8760"))
#: 基线最终 eval_recon，用作图中参考水平线
BASELINE_EVAL_RECON = float(os.environ.get("SRDASH_BASELINE_RECON", "0.3318"))

PROBE_DIR = os.path.join(PROJECT_ROOT, "output", "probe")

#: 训练日志超过这么久没有新写入且未跑完 → 判定为 stalled（不要显示成 "running" 骗人）
STALE_SECONDS = int(os.environ.get("SRDASH_STALE_SECONDS", "420"))

RUNS = {
    # ---- 修复版 run（峰值 lr 1.5e-4 → 1.0e-4），当前正在训练 ----
    "stack2x_lr1e4_slice05": {
        "label": "stack2x lr1e-4 slice05（修复版）",
        "role": "fix",
        "color": "#2ca02c",
        "logs": [os.path.join(TRAIN_LOG_DIR, "stack2x_lr1e4_slice05.log")],
        "probe_files": [
            os.path.join(PROBE_DIR, "probe_stack2x_lr1e4_slice05.json"),
            os.path.join(TRAIN_LOG_DIR, "probe_stack2x_lr1e4_slice05.json"),
        ],
        # 注意 glob 必须精确到 lr1e4，否则会与旧 run 的 probe_stack2x_slice05* 互相串数据
        "probe_globs": [
            os.path.join(PROBE_DIR, "probe_stack2x_lr1e4_slice05*.json"),
            os.path.join(TRAIN_LOG_DIR, "probe_stack2x_lr1e4_slice05*.json"),
        ],
        "infer_files": [
            os.path.join(PROJECT_ROOT, "output", "phase1_v2_stack2x_lr1e4_slice05",
                         "infer_test.json"),
        ],
    },
    # ---- 旧 run（lr 1.5e-4，后步坍缩），日志停在 step 2500 ----
    "stack2x_slice05": {
        "label": "stack2x lr1.5e-4 slice05（旧）",
        "role": "old",
        "color": "#1f77b4",
        "logs": [os.path.join(TRAIN_LOG_DIR, "stack2x_slice05.log")],
        # 优先用任务书给出的正式路径，其次回落到历史目录
        "probe_files": [
            os.path.join(PROBE_DIR, "probe_stack2x_slice05.json"),
            os.path.join(TRAIN_LOG_DIR, "probe_stack2x_slice05.json"),
        ],
        # 精确前缀，避免把 probe_stack2x_lr1e4_slice05* 也扫进来
        "probe_globs": [
            os.path.join(PROBE_DIR, "probe_stack2x_slice05*.json"),
            os.path.join(TRAIN_LOG_DIR, "probe_stack2x_slice05*.json"),
        ],
        "infer_files": [
            os.path.join(PROJECT_ROOT, "output", "phase1_v2_stack2x_slice05", "infer_test.json"),
        ],
    },
    # ---- 基线对照 ----
    "baseline_blockdiag_slice05": {
        "label": "blockdiag slice05（基线）",
        "role": "baseline",
        "color": "#d62728",
        "logs": [os.path.join(TRAIN_LOG_DIR, "blockdiag_slice05.log")],
        "probe_files": [
            os.path.join(PROBE_DIR, "probe_baseline_blockdiag_slice05.json"),
            os.path.join(TRAIN_LOG_DIR, "probe_blockdiag.json"),
        ],
        "probe_globs": [
            os.path.join(PROBE_DIR, "probe_*baseline*blockdiag*slice05*.json"),
            os.path.join(TRAIN_LOG_DIR, "probe_*blockdiag*.json"),
        ],
        "infer_files": [
            os.path.join(PROJECT_ROOT, "output", "baseline_blockdiag_slice05_infer_test.json"),
            os.path.join(PROJECT_ROOT, "output", "phase1_v2_block_slice05_blockdiag_infer_test.json"),
        ],
    },
}

# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------

DICT_RE = re.compile(r"\{[^{}]*\}")
TQDM_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
STEP_IN_NAME_RE = re.compile(r"(?:step|ckpt|checkpoint|iter)[-_]?(\d+)", re.I)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _num(v):
    """把日志里的字符串数字转成 float；失败返回 None。"""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def read_text(path: str) -> str:
    """容忍并发写入 / 非法字节地读取文本。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def existing_globs(patterns):
    found = []
    for pat in patterns:
        try:
            found.extend(p for p in glob.glob(pat) if os.path.isfile(p))
        except OSError:
            pass
    return found


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def epoch_to_step(epoch):
    """本实验 dict 里没有 step，只有 epoch；step ≈ round(epoch * steps_per_epoch)。"""
    if epoch is None:
        return None
    return int(round(epoch * STEPS_PER_EPOCH))


# --------------------------------------------------------------------------------------
# 1) 训练日志解析
# --------------------------------------------------------------------------------------

def parse_train_log(path: str):
    """解析一份训练日志 -> (train_points, eval_points, meta)。"""
    text = read_text(path)
    train, evals = [], []

    for m in DICT_RE.finditer(text):
        chunk = m.group(0)
        if "loss" not in chunk:
            continue
        try:
            d = ast.literal_eval(chunk)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(d, dict):
            continue

        epoch = _num(d.get("epoch"))
        if "eval_recon" in d or "eval_loss" in d:
            point = {"step": epoch_to_step(epoch), "epoch": epoch}
            for k, v in d.items():
                n = _num(v)
                point[k] = n if (n is not None or k == "epoch") else v
            evals.append(point)
            continue

        if "loss" in d:
            point = {
                "step": epoch_to_step(epoch),
                "epoch": epoch,
                "loss": _num(d.get("loss")),
                "grad_norm": _num(d.get("grad_norm")),
                "learning_rate": _num(d.get("learning_rate")),
            }
            train.append(point)

    # 去重 + 按 step 排序（同一 step 只保留最后一条）
    def _dedup(points):
        by_step = {}
        for p in points:
            key = p["step"] if p["step"] is not None else id(p)
            by_step[key] = p
        return [by_step[k] for k in sorted(by_step, key=lambda x: (x is None, x))]

    train = _dedup(train)
    evals = _dedup(evals)

    tqdm_max, tqdm_total = None, None
    hits = TQDM_RE.findall(text)
    if hits:
        tqdm_total = max(int(t) for _, t in hits)
        tqdm_max = max(int(c) for c, _ in hits)

    meta = {
        "log_path": path,
        "log_mtime": os.path.getmtime(path) if os.path.isfile(path) else None,
        "log_bytes": os.path.getsize(path) if os.path.isfile(path) else 0,
        "n_train_points": len(train),
        "n_eval_points": len(evals),
        "tqdm_step": tqdm_max,
        "tqdm_total": tqdm_total,
        "total_steps": tqdm_total or TOTAL_STEPS,
        "last_step": train[-1]["step"] if train else (tqdm_max or None),
    }
    meta["log_age_seconds"] = (
        round(time.time() - meta["log_mtime"], 1) if meta["log_mtime"] else None
    )
    meta["progress"] = (
        round(meta["last_step"] / meta["total_steps"], 4)
        if meta["last_step"] and meta["total_steps"]
        else None
    )
    meta["status"] = classify_status(meta, evals, text)
    return train, evals, meta


def classify_status(meta, evals, text) -> str:
    """finished / running / stalled / unknown。

    ``stalled``：日志在 STALE_SECONDS 内没有任何新写入，但进度又没跑满
    —— 例如旧 run 在 step 2500 被中断，不能显示成 "running"。
    """
    if "TRAIN_EXIT=0" in text or "[final]" in text:
        return "finished"
    if meta["tqdm_step"] and meta["tqdm_total"] and meta["tqdm_step"] >= meta["tqdm_total"]:
        return "finished"
    if meta["n_train_points"] or meta["tqdm_step"]:
        age = meta.get("log_age_seconds")
        if age is not None and age > STALE_SECONDS:
            return "stalled"
        return "running"
    return "unknown"


# --------------------------------------------------------------------------------------
# 2) 探针 JSON 解析
# --------------------------------------------------------------------------------------

PROBE_LIST_FIELDS = ("step_px_scale", "prog_curve_255", "z_s_block_cos")
PROBE_MATRIX_FIELDS = ("E_px", "E_nrm")


def _as_float_list(v):
    if not isinstance(v, (list, tuple)):
        return None
    out = []
    for x in v:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _as_float_matrix(v):
    if not isinstance(v, (list, tuple)):
        return None
    rows = [_as_float_list(r) for r in v]
    return rows if all(r is not None for r in rows) else None


def probe_step_of(path: str, data: dict):
    """尽力推断该探针属于哪个训练 step（用于画'随训练进程变化'）。"""
    for key in ("step", "train_step", "global_step", "ckpt_step", "steps_trained"):
        n = data.get(key)
        if isinstance(n, (int, float)):
            return int(n)
    for key in ("ckpt", "checkpoint", "model_path", "path", "log_path"):
        v = data.get(key)
        if isinstance(v, str):
            m = STEP_IN_NAME_RE.search(v)
            if m:
                return int(m.group(1))
    m = STEP_IN_NAME_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def parse_probe(path: str):
    data = load_json(path)
    if not isinstance(data, dict):
        return None

    entry = {
        "path": path,
        "mtime": os.path.getmtime(path),
        "tag": data.get("tag"),
        "ckpt": data.get("ckpt"),
        "n_img": data.get("n_img"),
        "steps": data.get("steps") or [1, 4, 9, 16, 25],
        "query_mask_mode": data.get("query_mask_mode"),
        "num_specials": data.get("num_specials"),
        "regions": data.get("regions"),
        "blocks": data.get("blocks"),
        "z_s_within_std": _num(data.get("z_s_within_std")),
        "step": probe_step_of(path, data),
        "raw": {},
        "derived": {},
    }
    for f in PROBE_LIST_FIELDS:
        entry[f] = _as_float_list(data.get(f))
    for f in PROBE_MATRIX_FIELDS:
        entry[f] = _as_float_matrix(data.get(f))
    # 保留其余标量字段
    for k, v in data.items():
        if k not in entry and isinstance(v, (int, float, str)) and v is not None:
            entry["raw"][k] = v

    sps = entry.get("step_px_scale") or []
    sps_ok = [x for x in sps if x is not None]
    prog = entry.get("prog_curve_255") or []
    prog_ok = [x for x in prog if x is not None]
    cos = entry.get("z_s_block_cos") or []
    cos_ok = [x for x in cos if x is not None]
    d = entry["derived"]
    if sps_ok:
        d["step_px_scale_mean"] = sum(sps_ok) / len(sps_ok)
        d["step_px_scale_step1"] = sps_ok[0]
        if len(sps_ok) > 1:
            tail = sps_ok[1:]
            d["step_px_scale_tail_mean"] = sum(tail) / len(tail)
            d["step_px_scale_late_early_ratio"] = (
                d["step_px_scale_tail_mean"] / sps_ok[0] if sps_ok[0] else None
            )
    if prog_ok:
        d["prog_curve_first"] = prog_ok[0]
        d["prog_curve_final"] = prog_ok[-1]
        d["prog_curve_delta"] = prog_ok[-1] - prog_ok[0]
    if cos_ok:
        d["z_s_block_cos_mean"] = sum(cos_ok) / len(cos_ok)
    return entry


def collect_probes(spec):
    """返回 (primary_entry, history_entries)。

    同一 run 可能有多个 checkpoint 的探针文件 -> 按 step 排序构成历史序列。
    """
    paths = []
    primary = first_existing(spec.get("probe_files", []))
    if primary:
        paths.append(primary)
    for p in existing_globs(spec.get("probe_globs", [])):
        paths.append(p)

    uniq, seen = [], set()
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)

    entries = [e for e in (parse_probe(p) for p in uniq) if e]
    if not entries:
        return None, []

    # 有 step 的按 step 排前面并按 step 升序；无 step 的按 mtime 排在后面
    entries.sort(key=lambda e: (e["step"] is None, e["step"] or 0, e["mtime"]))

    # primary 定义为"显式路径命中的那个"，否则取 step 最大的
    prim = None
    if primary:
        for e in entries:
            if os.path.realpath(e["path"]) == os.path.realpath(primary):
                prim = e
                break
    if prim is None:
        prim = entries[-1]
    return prim, entries


# --------------------------------------------------------------------------------------
# 3) 推理 JSON
# --------------------------------------------------------------------------------------

INFER_SCALARS = ("full_norm_l1", "full_pixel_l1_255")
INFER_LISTS = ("step_pixel_l1_255",)


def parse_infer(path: str):
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    out = {"path": path, "mtime": os.path.getmtime(path), "data": {}}
    for k, v in data.items():
        if k in INFER_LISTS:
            out["data"][k] = _as_float_list(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out["data"][k] = v
    if "step_pixel_l1_255" in out["data"] and out["data"]["step_pixel_l1_255"]:
        vals = [x for x in out["data"]["step_pixel_l1_255"] if x is not None]
        if vals:
            out["data"].setdefault("full_pixel_l1_255", vals[-1])
    return out


# --------------------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------------------

def build_bundle() -> dict:
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "generated_at_ts": time.time(),
        "config": {
            "project_root": PROJECT_ROOT,
            "train_log_dir": TRAIN_LOG_DIR,
            "probe_dir": PROBE_DIR,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "total_steps": TOTAL_STEPS,
            "baseline_eval_recon": BASELINE_EVAL_RECON,
            "step_formula": "step = round(epoch * steps_per_epoch)",
        },
        "runs": list(RUNS.keys()),
        "run_meta": {
            k: {"label": v["label"], "role": v["role"], "color": v["color"]}
            for k, v in RUNS.items()
        },
        "series": {},
        "probes": {},
        "probe_history": {},
        "infer": {},
        "sources": {"ok": [], "missing": [], "notes": []},
    }

    for key, spec in RUNS.items():
        # --- 训练日志 ---
        log_path = first_existing(spec["logs"])
        if log_path:
            train, evals, meta = parse_train_log(log_path)
            bundle["series"][key] = {"train": train, "eval": evals, **meta}
            bundle["sources"]["ok"].append(log_path)
        else:
            bundle["series"][key] = {
                "train": [], "eval": [], "log_path": None, "n_train_points": 0,
                "n_eval_points": 0, "status": "missing", "last_step": None,
                "progress": None, "tqdm_step": None, "tqdm_total": None,
                "total_steps": TOTAL_STEPS,
            }
            bundle["sources"]["missing"].append(spec["logs"][0])

        # --- 探针 ---
        prim, hist = collect_probes(spec)
        if prim:
            bundle["probes"][key] = prim
            bundle["probe_history"][key] = hist
            for e in hist:
                bundle["sources"]["ok"].append(e["path"])
            if len(hist) > 1:
                bundle["sources"]["notes"].append(
                    f"{key}: 发现 {len(hist)} 个 checkpoint 探针，已按 step 排序"
                )
        else:
            bundle["probes"][key] = None
            bundle["probe_history"][key] = []
            bundle["sources"]["missing"].append(spec["probe_files"][0])

        # --- 推理 ---
        ip = first_existing(spec["infer_files"])
        if ip:
            bundle["infer"][key] = parse_infer(ip)
            bundle["sources"]["ok"].append(ip)
        else:
            bundle["infer"][key] = None
            bundle["sources"]["missing"].append(spec["infer_files"][0])

    return bundle


def write_csvs(bundle: dict, out_dir: str):
    with open(os.path.join(out_dir, "metrics_train.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "step", "epoch", "loss", "grad_norm", "learning_rate"])
        for run, s in bundle["series"].items():
            for p in s["train"]:
                w.writerow([run, p.get("step"), p.get("epoch"), p.get("loss"),
                            p.get("grad_norm"), p.get("learning_rate")])

    with open(os.path.join(out_dir, "metrics_eval.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "step", "epoch", "eval_loss", "eval_recon", "eval_runtime"])
        for run, s in bundle["series"].items():
            for p in s["eval"]:
                w.writerow([run, p.get("step"), p.get("epoch"), p.get("eval_loss"),
                            p.get("eval_recon"), p.get("eval_runtime")])

    with open(os.path.join(out_dir, "metrics_probe.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        head = ["run", "tag", "step", "n_img", "prog_curve_final", "z_s_within_std",
                "z_s_block_cos_mean", "step_px_scale_mean"]
        head += [f"step_px_scale_s{i+1}" for i in range(5)]
        head += [f"prog_curve_s{i+1}" for i in range(5)]
        w.writerow(head)
        for run, hist in bundle["probe_history"].items():
            for e in hist:
                sps = (e.get("step_px_scale") or [None] * 5)[:5]
                sps = (sps + [None] * 5)[:5]
                prog = (e.get("prog_curve_255") or [None] * 5)[:5]
                prog = (prog + [None] * 5)[:5]
                d = e.get("derived", {})
                w.writerow([run, e.get("tag"), e.get("step"), e.get("n_img"),
                            d.get("prog_curve_final"), e.get("z_s_within_std"),
                            d.get("z_s_block_cos_mean"), d.get("step_px_scale_mean")]
                           + list(sps) + list(prog))


def _fmt(v, nd=5):
    if v is None:
        return "None"
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return str(v)


def summarize(bundle: dict) -> str:
    lines = [f"[parse_metrics] {bundle['generated_at']}"]
    for key, s in bundle["series"].items():
        lines.append(
            "  %-28s status=%-8s train_pts=%-5d eval_pts=%-3d step=%-6s progress=%s"
            % (key, s.get("status"), s.get("n_train_points", 0), s.get("n_eval_points", 0),
               s.get("last_step"), s.get("progress"))
        )
    for key, p in bundle["probes"].items():
        if p:
            d = p.get("derived", {})
            sps = [round(x, 5) for x in (p.get("step_px_scale") or [])]
            lines.append(
                "  %-28s probe tag=%s step=%s step_px_scale=%s"
                % (key, p.get("tag"), p.get("step"), sps)
            )
            lines.append(
                "  %-28s       mean=%s z_s_within_std=%s prog_final=%s n_history=%d"
                % ("",
                   _fmt(d.get("step_px_scale_mean")),
                   _fmt(p.get("z_s_within_std")),
                   _fmt(d.get("prog_curve_final")),
                   len(bundle.get("probe_history", {}).get(key, [])))
            )
        else:
            lines.append("  %-28s probe MISSING" % key)
    for key, i in bundle["infer"].items():
        lines.append("  %-28s infer %s" % (key, "OK" if i else "MISSING"))
    if bundle["sources"]["missing"]:
        lines.append("  missing: " + ", ".join(bundle["sources"]["missing"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="解析 AssocVTC 训练指标")
    default_out = os.environ.get("SRDASH_DATA") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data")
    ap.add_argument("--out", default=default_out, help="输出目录（写 metrics.json 与 CSV）")
    ap.add_argument("--quiet", action="store_true", help="不打印摘要")
    ap.add_argument("--pretty", action="store_true", help="metrics.json 缩进输出")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    bundle = build_bundle()

    out_json = os.path.join(args.out, "metrics.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2 if args.pretty else None,
                  default=str)
        fh.write("\n")
    os.replace(tmp, out_json)  # 原子替换，避免看板读到半个文件

    write_csvs(bundle, args.out)

    if not args.quiet:
        print(summarize(bundle))
        print(f"[parse_metrics] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
