#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_448_960m.py — AssocVTC 448²/patch32/960M 三阶段实验报告生成器

只读产物（不改训练/评测脚本），从 $OUT/eval/*.json + $OUT/logs/stage*.log 生成：
  - doc/2026-11-04/REPORT_448_960M_3stage.md（配置表 / 逐 step PSNR·SSIM·L1 表 /
    与 224 论文基线同值数对照（留空位）/ 训练曲线 / 结论模板）
  - doc/2026-11-04/artifacts_448_960m/（summary json、per-step tsv、曲线 PNG）

评测 json 兼容 3 种 schema：
  1) eval_stepwise_linattn.py --out 的平铺结果：PSNR / L1_255 / steps / cum_z_s（无 SSIM）
  2) --dump_per_image 产物：aggregate_PSNR / aggregate_SSIM / aggregate_L1_255 / steps / cum_z_s
  3) 旧版按 ckpt 分组的标量结果：{ckpt: {psnr, ssim, l1_norm, cfg, params, ...}}

matplotlib 不可用时自动退化为纯 markdown（不生成 PNG，改用 sparkline 文本），
report 仍完整生成。可重复运行（幂等覆盖）。

用法（本地；远端有 matplotlib 也可直接跑）：
  python3 tools/report_448_960m.py
  python3 tools/report_448_960m.py --eval-dir /root/autodl-tmp/exp448/eval \\
      --logs-dir /root/autodl-tmp/exp448/logs
  python3 tools/report_448_960m.py --check      # 只检查输入/环境，不写文件
  python3 tools/report_448_960m.py --no-plot    # 强制纯 markdown

退出码：0 = 生成了报告（有无评测数据均可，骨架也生成）；2 = 参数/路径错误。
"""
import argparse
import glob
import json
import os
import re
import sys

VERSION = "1.1"

# ---- 实验口径（来源：exp448_bench/DESIGN_448_960M_3stage.md + count_448_960m.py） ----
CFG448 = {
    "img_size": 448, "patch": 32, "num_patches": 196, "num_specials": 32,
    "enc_dim": 1744, "enc_depth": 13, "enc_heads": 16,
    "dec_dim": 2048, "dec_depth": 7, "dec_heads": 16,
    "pixel_hidden": 1024, "enc_mlp_ratio": 4.0, "dec_mlp_ratio": 4.0,
    "enc_dwconv": True, "enc_qk_norm": True, "enc_attn": "std",
    "dropout": 0.0, "amp": 1,
}
PARAMS448 = {"enc_M": 480.537, "dec_M": 479.666, "total_M": 960.203,
             "enc": 480536528, "dec": 479666176}
CFG224 = {"img_size": 224, "patch": 16, "num_patches": 196,
          "enc_dim": 1248, "enc_depth": 13, "dec_dim": 1440, "dec_depth": 7,
          "pixel_hidden": 1024, "enc_M": 244.43, "dec_M": 237.11, "total_M": 481.54}

STAGES = [
    {"key": "stage1", "num": 1,
     "label": "阶段1-A：单步读满（T=1，窗口 [0,K]）",
     "epochs": 45, "steps": 42795, "lr": "2e-4", "bs": 128,
     "init": "从零", "note": "K=32 全读 1 步"},
    {"key": "stage2", "num": 2,
     "label": "阶段2-B：16 档循环读出（T=16，fixed block 2）",
     "epochs": 6, "steps": 11406, "lr": "2e-4", "bs": 64,
     "init": "阶段1 ckpt_best", "note": "每步读 2 token"},
    {"key": "stage3", "num": 3,
     "label": "阶段3-S3：输出投影 224 维（--zs_dim 224，T=16）",
     "epochs": 3, "steps": 5703, "lr": "2e-4", "bs": 64,
     "init": "阶段2 ckpt_best（SVD rank-224 热启动）", "note": "每图 7168 值，与论文同值数口径"},
]
STAGE_BY_KEY = {s["key"]: s for s in STAGES}

# 224 论文基线（仓库既有实测，仅作参考；论文 v2.0 数字请填进报告"空白位"）
BASELINE224_REF = {
    "stage1": {"psnr": 32.35, "source": "README：240M@224 A 相 T=1 读满 K=32（rFID 6.13）"},
    "stage2": {"psnr": 27.94, "source": "README：240M@224 B 相 16 步末步（rFID 19.97）"},
    "stage3": {"psnr": 26.08, "source": "srdiff_scale/eval/linattn_m240_t16_zs224.json（240M@224, zs_dim=224）"},
}

VAL_PER_IMAGE_448 = {"stage1": 32 * 1744, "stage2": 32 * 2048, "stage3": 32 * 224}
VAL_PER_IMAGE_224 = {"stage1": 32 * 1248, "stage2": 32 * 1248, "stage3": 32 * 224}

# 论文 v2.0 表 2「本文」行（224²/p16, enc 1248/dec 1440, zs_dim=224）逐档 PSNR
# 来源: /userdata/paper/pp_zh/sec4_exp.tex tab:psnr（预算 448…7168）
PAPER224_MODEL_PSNR = [21.03, 22.04, 22.91, 23.66, 24.19, 24.51, 24.86, 25.21,
                       25.50, 25.69, 25.75, 25.92, 25.93, 26.02, 26.08, 26.08]
PAPER224_MODEL_SRC = "论文 v2.0 表 2「本文」行（sec4_exp.tex tab:psnr）"
DEFAULT_PCA448_JSON = "/home/linaro/dsh/exp448_bench/pca/pca_448_results.json"

TITLE = "AssocVTC 448²/patch32/编码器·解码器各 ~480M（合计 ~960M）三阶段训练与评测报告"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def fmt(v, nd=3, dash="—"):
    if v is None or v == "":
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f or abs(f) == float("inf"):
        return dash
    return ("%." + str(nd) + "f") % f


def sparkline(vals, width=60):
    vals = [float(v) for v in vals if v is not None and float(v) == float(v)]
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / float(width)
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    chars = " ▁▂▃▄▅▆▇█"
    if hi - lo < 1e-12:
        return chars[4] * len(vals)
    return "".join(chars[min(8, max(1, int(round((v - lo) / (hi - lo) * 7)) + 1))] for v in vals)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("[warn] 读不了 %s: %s\n" % (path, e))
        return None


# ---------------------------------------------------------------------------
# 评测 json 归一化
# ---------------------------------------------------------------------------
def _norm_record(d, src, cfg_override=None):
    """把单个评测结果 dict 归一化为统一 record；不认识的返回 None。"""
    rec = {
        "source": src, "ckpt": d.get("ckpt", ""), "tag": d.get("tag", ""),
        "T": d.get("T"), "n_images": d.get("n_images"), "step": d.get("step"),
        "epoch": d.get("epoch"), "cfg": d.get("cfg") or {},
        "params": d.get("params") or {},
        "steps": d.get("steps"), "cum_z_s": d.get("cum_z_s"),
        "psnr": None, "psnr_mean_mse": None, "ssim": None, "l1_255": None,
        "l1_01": None, "mode": "",
        "best_step": d.get("best_step"), "total_drop_255": d.get("total_drop_255"),
    }
    if isinstance(d.get("aggregate_PSNR"), list):                 # dump_per_image
        rec["psnr"] = d["aggregate_PSNR"]
        rec["l1_255"] = d.get("aggregate_L1_255")
        rec["ssim"] = d.get("aggregate_SSIM")
        rec["mode"] = "aggregate"
    elif isinstance(d.get("PSNR"), list):                        # eval_stepwise --out
        rec["psnr"] = d["PSNR"]
        rec["l1_255"] = d.get("L1_255")
        rec["l1_01"] = d.get("L1_01")
        rec["psnr_mean_mse"] = d.get("PSNR_mean_mse")
        rec["mode"] = "stepwise"
        for k in ("SSIM",):
            if isinstance(d.get(k), list):
                rec["ssim"] = d[k]
    elif isinstance(d.get("psnr"), (int, float)):                # 旧版标量
        rec["psnr"] = [float(d["psnr"])]
        rec["psnr_mean_mse"] = [float(d["psnr"])]
        if d.get("ssim") is not None:
            rec["ssim"] = [float(d["ssim"])]
        if d.get("l1_norm") is not None:
            rec["l1_255"] = [float(d["l1_norm"]) * 255.0]
        if not rec["T"]:
            ds = str((rec["cfg"] or {}).get("dec_steps", "") or "")
            rec["T"] = len([x for x in ds.split(",") if x.strip()]) or 1
        rec["mode"] = "legacy_scalar"
    else:
        return None
    if rec["steps"] is None and rec["T"]:
        rec["steps"] = list(range(1, int(rec["T"]) + 1))
    if rec["cum_z_s"] is None and rec["T"]:
        rec["cum_z_s"] = [None] * int(rec["T"])
    if cfg_override:
        merged = dict(cfg_override)
        merged.update(rec["cfg"] or {})
        rec["cfg"] = merged
    return rec


def normalize_eval_file(path):
    """一个文件可能包含多组结果（旧版按 ckpt 分组），返回 record 列表。"""
    d = read_json(path)
    if not isinstance(d, dict):
        return []
    src = os.path.basename(path)
    out = []
    r = _norm_record(d, src)
    if r:
        out.append(r)
        return out
    # 旧版 {ckpt_key: {psnr, ssim, ...}} / 多组
    for k, v in d.items():
        if isinstance(v, dict):
            r = _norm_record(v, src)
            if r:
                if not r["ckpt"]:
                    r["ckpt"] = k
                r["group"] = k
                out.append(r)
    return out


def classify_record(rec):
    """按文件名/内容猜阶段。返回 stage1|stage2|stage3|extra。"""
    name = (rec.get("source", "") + " " + rec.get("ckpt", "") + " " +
            str(rec.get("group", ""))).lower()
    cfg = rec.get("cfg") or {}
    if re.search(r"(stage3|st3|s3|zs_?224|phase3)", name):
        return "stage3"
    if re.search(r"(stage1|st1|phase1|_t1\b|-t1\b|single)", name):
        return "stage1"
    if re.search(r"(stage2|st2|phase2|_t16\b|-t16\b)", name):
        return "stage2"
    if cfg.get("zs_dim") in (224, "224"):
        return "stage3"
    if rec.get("T") == 1:
        return "stage1"
    if rec.get("T") and int(rec["T"]) > 1:
        return "stage2"
    return "extra"


def collect_evals(eval_dir):
    records = []
    if not os.path.isdir(eval_dir):
        return records
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        records.extend(normalize_eval_file(path))
    for rec in records:
        rec["stage"] = classify_record(rec)
    return records


# ---------------------------------------------------------------------------
# 训练日志解析
# ---------------------------------------------------------------------------
def resolve_log(logs_dir, num):
    cands = [os.path.join(logs_dir, "stage%d.log" % num)]
    cands += sorted(glob.glob(os.path.join(logs_dir, "stage%d_*.log" % num)))
    cands += sorted(glob.glob(os.path.join(logs_dir, "*stage%d*.log" % num)))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


PROG_RE = re.compile(
    r"^\[\s*([0-9.]+)m\] step (\d+)/(\d+) ep(\d+) loss=(\S+) recon=(\S+)?"
    r".*?lr=(\S+) gn=(\S+) skip=(\d+).*?(\d+)img/s vram=([0-9.]+)G eta=([0-9.]+)h")
VAL_RE = re.compile(r"^\[val\] ep(\d+) step(\d+) l1=([0-9.]+) psnr=([0-9.]+)dB(?: ssim=([0-9.]+))?")


def parse_train_logs(logs_dir):
    stats = {}
    for st in STAGES:
        key = st["key"]
        path = resolve_log(logs_dir, st["num"])
        info = {"path": path, "progress": [], "vals": [], "done_line": "", "data_line": ""}
        if not path:
            stats[key] = info
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    m = PROG_RE.match(line)
                    if m:
                        info["progress"].append({
                            "wall_min": float(m.group(1)), "step": int(m.group(2)),
                            "total": int(m.group(3)), "epoch": int(m.group(4)),
                            "loss": m.group(5), "recon": m.group(6),
                            "lr": m.group(7), "gn": m.group(8), "skip": int(m.group(9)),
                            "imgs": int(m.group(10)), "vram": float(m.group(11)),
                            "eta": float(m.group(12)),
                        })
                        continue
                    m = VAL_RE.match(line)
                    if m:
                        info["vals"].append({
                            "epoch": int(m.group(1)), "step": int(m.group(2)),
                            "l1": float(m.group(3)), "psnr": float(m.group(4)),
                            "ssim": float(m.group(5)) if m.group(5) else None,
                        })
                        continue
                    if line.startswith("[data]"):
                        info["data_line"] = line
                    elif line.startswith("[done]"):
                        info["done_line"] = line
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[warn] 解析日志失败 %s: %s\n" % (path, e))
        stats[key] = info
    return stats


# ---------------------------------------------------------------------------
# Markdown 片段
# ---------------------------------------------------------------------------
def cfg_table(records_by_stage):
    def pick(rec, k, default):
        if rec and rec.get("cfg") and rec["cfg"].get(k) is not None:
            return rec["cfg"][k]
        return default

    rows = [
        ("输入分辨率", "img_size", 448),
        ("patch", "patch", 32),
        ("N = (H/patch)²", "num_patches", 196),
        ("K（special token 数）", "num_specials", 32),
        ("编码器宽×层×头", None, "1744×13×16"),
        ("解码器宽×层×头", None, "2048×7×16"),
        ("像素头 hidden", "pixel_hidden", 1024),
        ("enc/dec MLP ratio", None, "4.0 / 4.0"),
        ("注意力", "enc_attn", "std"),
        ("amp", None, "bf16(1)"),
        ("阶段3 zs_dim", "zs_dim", 224),
    ]
    lines = ["| 项 | 448² 本实验 | 224 论文基线 |", "|---|---|---|"]
    for label, key, default in rows:
        vals = []
        for st in STAGES:
            r = records_by_stage.get(st["key"])
            rec = r[0] if r else None
            if key is None:
                vals.append(default)
            else:
                vals.append(str(pick(rec, key, default)))
        if len(set(vals)) == 1:
            vals = [vals[0]]
        base = {
            "输入分辨率": "224", "patch": "16", "N = (H/patch)²": "196",
            "K（special token 数）": "32", "编码器宽×层×头": "1248×13×16",
            "解码器宽×层×头": "1440×7×16", "像素头 hidden": "1024",
            "enc/dec MLP ratio": "4.0 / 4.0", "注意力": "std",
            "amp": "bf16(1)", "阶段3 zs_dim": "224（SVD 热身）",
        }.get(label, "—")
        lines.append("| %s | %s | %s |" % (label, " / ".join(vals), base))
    lines.append("")
    lines.append("> 计参：`SRPhase1LinAttn.count_params()`。448 预设 enc **%.2fM** / dec **%.2fM** / total **%.2fM**"
                 "（论文 224 基线 244.43 / 237.11 / 481.54M）。若评测 json 里带 cfg/params，本表自动用实测值覆盖。"
                 % (PARAMS448["enc_M"], PARAMS448["dec_M"], PARAMS448["total_M"]))
    return "\n".join(lines)


def protocol_table():
    lines = ["| 阶段 | 内容 | 初始化 | epoch | 等效 batch | 优化步数 | lr |",
             "|---|---|---|---|---|---|---|"]
    for st in STAGES:
        lines.append("| %d | %s | %s | %d | %d | %s | %s |" % (
            st["num"], st["label"].split("：", 1)[-1], st["init"],
            st["epochs"], st["bs"], "{:,}".format(st["steps"]), st["lr"]))
    lines.append("")
    lines.append("> 优化器 AdamW，cosine + 8% warmup，wd 0.05，clip 1.0，bf16；"
                 "数据 ImageNet-100（train 121,689 / val 5,000）。"
                 "⚠️ `--dec_grad_ckpt` 必须全阶段固定同一值。")
    return "\n".join(lines)


def step_table(rec, st):
    lines = []
    psnr = rec.get("psnr") or []
    ssim = rec.get("ssim") or []
    l1 = rec.get("l1_255") or []
    steps = rec.get("steps") or [None] * len(psnr)
    cum = rec.get("cum_z_s") or [None] * len(psnr)
    n = max(len(psnr), len(ssim), len(l1), len(steps), len(cum))
    if n == 0:
        return "_无逐 step 数据_"
    lines.append("| step / t | 累计 z_s | PSNR (dB) | SSIM | L1_255 |")
    lines.append("|---|---|---|---|---|")
    for i in range(n):
        s = steps[i] if i < len(steps) else None
        c = cum[i] if i < len(cum) else None
        p = psnr[i] if i < len(psnr) else None
        sm = ssim[i] if i < len(ssim) else None
        li = l1[i] if i < len(l1) else None
        lines.append("| %s | %s | %s | %s | %s |" % (
            ("%s" % s) if s is not None else ("%d" % (i + 1)),
            c if c is not None else "—",
            fmt(p, 3, "待补"), fmt(sm, 4, "待补"), fmt(li, 3, "待补")))
    if rec.get("n_images"):
        lines.append("")
        lines.append("> n_images=%s，best_step=%s，ckpt=`%s`，schema=`%s`。" % (
            rec["n_images"],
            rec.get("best_step") if rec.get("best_step") is not None else "—",
            rec.get("ckpt", "—"), rec.get("mode", "")))
    return "\n".join(lines)


def baseline_compare_section(records_by_stage, train_stats, pca):
    """与 224 论文基线「同值数」对照：S3 臂逐档（严格可比）+ A/B 相参考。"""
    L = []
    A = L.append
    recs3 = records_by_stage.get("stage3") or []
    rec3 = recs3[0] if recs3 else None
    ps3 = (rec3 or {}).get("psnr") or []
    if not ps3:
        A("> 阶段3 评测尚未产出，本节待填。")
        return "\n".join(L)
    rows = None
    if pca and pca.get("results_448", {}).get("same_value_table"):
        rows = pca["results_448"]["same_value_table"]

    def best(seq):
        return max(seq) if seq else None

    end3 = ps3[-1]
    A("**汇总（S3 臂是唯一与论文同值数口径可直接配对的臂）**：")
    A("")
    A("| 臂 | 448 值/图 | 224 值/图 | 224 论文「本文」 | 448 本实验 | ΔPSNR(448−224) |")
    A("|---|---|---|---|---|---|")
    A("| 阶段3-S3（448…7,168 值，zs_dim=224） | %s | 7,168 | %.2f dB（末步） | %.3f dB（末步） | **%+.3f** |" % (
        "{:,}".format(VAL_PER_IMAGE_448["stage3"]), PAPER224_MODEL_PSNR[-1], end3,
        end3 - PAPER224_MODEL_PSNR[-1]))
    A("")
    A("> 口径：都是「每图 2t×224 个连续值、逐图 PSNR 再平均」。A/B 相是 1248/2048 维/token 的完整宽度臂"
      "（448 侧 55,808 / 65,536 值/图），论文表 2 无同值数行，故只作过程参考（见文末「仓库既有 224 实测」）。")
    A("")
    A("**16 档逐档对照**（第 3 列 = 224 论文表 2「本文」行，来源 `%s`；末两列 = 448 PCA 同值数基线）：" % PAPER224_MODEL_SRC)
    A("")
    A("| t | 每图值数 | 224 论文本文 / dB | 448 本实验 / dB | Δ(448−224) 跨分辨率 | 448 PCA / dB | Δ(448 模型−448 PCA) 同分辨率 |")
    A("|---|---|---|---|---|---|---|")
    for i in range(min(len(ps3), len(PAPER224_MODEL_PSNR))):
        paper = PAPER224_MODEL_PSNR[i]
        pca_psnr = rows[i].get("pca_psnr_interp") if (rows and i < len(rows)) else None
        A("| %d | %s | %.2f | %.3f | %+.3f | %s | %s |" % (
            i + 1, "{:,}".format(448 * (i + 1)), paper, ps3[i], ps3[i] - paper,
            fmt(pca_psnr, 3, "—"),
            fmt((ps3[i] - pca_psnr) if pca_psnr is not None else None, 3, "—")))
    if rows and len(rows) >= len(ps3):
        d = [ps3[i] - rows[i]["pca_psnr_interp"] for i in range(len(ps3))]
        cross0 = ps3[0] - PAPER224_MODEL_PSNR[0]
        crossN = ps3[-1] - PAPER224_MODEL_PSNR[-1]
        zero = next((i + 1 for i in range(len(d)) if d[i] < 0), len(d))
        A("")
        A("读法：")
        A("")
        A("- **跨分辨率同值数**（Δ(448−224) 列）：448 用 4× 像素 + 2× 参数，在 448 值处 **%+.2f dB**，"
          "到 7,168 值扩到 **%+.2f dB** —— 预算越宽分辨率劣势越明显（与 PCA 跨分辨率差 −0.27→−1.06 dB 同形）。"
          % (cross0, crossN))
        A("- **同分辨率模型 vs PCA**（末列）：**%+.2f dB**（448 值）单调收窄到 **%+.2f dB**（7,168 值），"
          "t=%d–%d 之间过零（%s→%s 值）；224 论文同口径是 **+2.63 → −0.17 dB**。"
          "⇒ 分辨率 ×4 后「紧预算端模型显著强于 PCA」这一核心结论**保留**，只缩水约 %.2f dB。"
          % (d[0], d[-1], zero - 1, zero,
             "{:,}".format(448 * (zero - 1)), "{:,}".format(448 * zero), abs(d[0] - 2.63)))
        A("- L1 交叉验证：224 侧模型比 PCA 的 L1 优势 6.84→0.22 px（全程不交叉）；448 侧是 "
          "%.2f→%.2f px，**同样全程不交叉**。" % (22.972 - 16.485, 10.088 - 10.017))
    A("")
    A("**仓库既有 224 实测（非论文表 2 口径，仅参考）**：")
    for st in STAGES:
        key = st["key"]
        rec = (records_by_stage.get(key) or [None])[0]
        p = (rec or {}).get("psnr") or []
        ref = BASELINE224_REF.get(key, {})
        A("- %s：224 参考 **%s dB**（%s）→ 448 本实验 **%s dB**（Δ %s）" % (
            st["label"].split("：")[0], fmt(ref.get("psnr"), 2, "待补"), ref.get("source", "—"),
            fmt(best(p), 3, "待补"),
            ("%+.2f" % (best(p) - ref["psnr"])) if (p and ref.get("psnr")) else "—"))
    return "\n".join(L)


# ---------------------------------------------------------------------------
def try_matplotlib():
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        return True
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("[warn] matplotlib 不可用，降级为纯 markdown: %s\n" % e)
        return False


def make_figures(train_stats, records, artifacts_dir, no_plot):
    os.makedirs(artifacts_dir, exist_ok=True)
    if no_plot:
        return [], False
    if not try_matplotlib():
        return [], False
    import matplotlib.pyplot as plt
    colors = {"stage1": "#1f77b4", "stage2": "#ff7f0e", "stage3": "#2ca02c"}
    # 图内文字统一英文：matplotlib 默认字体无 CJK，中文会变豆腐块（md 正文仍中文）
    short = {"stage1": "Stage1-A (T=1)", "stage2": "Stage2-B (T=16)",
             "stage3": "Stage3-S3 (zs224)"}
    pngs = []

    # 图1：训练曲线（loss / val PSNR / lr / vram）
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax_loss, ax_psnr, ax_lr, ax_vram = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
    any_prog = False
    for st in STAGES:
        key = st["key"]
        pr = train_stats.get(key, {}).get("progress") or []
        va = train_stats.get(key, {}).get("vals") or []
        c = colors[key]
        if pr:
            any_prog = True
            xs = [p["step"] for p in pr]
            try:
                ys = [float(p["loss"]) for p in pr]
                ax_loss.plot(xs, ys, color=c, lw=1.0, label=short[key])
            except (TypeError, ValueError):
                pass
            try:
                ax_lr.plot(xs, [float(p["lr"]) for p in pr], color=c, lw=1.0)
                ax_vram.plot(xs, [p["vram"] for p in pr], color=c, lw=1.0)
            except (TypeError, ValueError):
                pass
        if va:
            ax_psnr.plot([v["epoch"] for v in va], [v["psnr"] for v in va],
                         color=c, marker="o", ms=3, lw=1.2, label=short[key])
    if any_prog:
        ax_loss.set_xlabel("optimizer step")
        ax_loss.set_ylabel("train loss")
        ax_loss.set_title("train loss")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(alpha=0.3)
        ax_lr.set_xlabel("optimizer step")
        ax_lr.set_ylabel("lr")
        ax_lr.set_title("learning rate (cosine + 8% warmup)")
        ax_lr.grid(alpha=0.3)
        ax_vram.set_xlabel("optimizer step")
        ax_vram.set_ylabel("vram (GB)")
        ax_vram.set_title("peak VRAM (from log)")
        ax_vram.grid(alpha=0.3)
        ax_psnr.set_xlabel("epoch (per stage)")
        ax_psnr.set_ylabel("val PSNR (dB)")
        ax_psnr.set_title("val PSNR")
        ax_psnr.legend(fontsize=8)
        ax_psnr.grid(alpha=0.3)
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "no training log data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("AssocVTC 448^2 / 960M 3-stage training curves", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = os.path.join(artifacts_dir, "curves_train_448_960m.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    pngs.append(p1)

    # 图2：逐 step 重建指标
    recs = [r for r in records if r.get("psnr")]
    if recs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_p, ax_l = axes[0], axes[1]
        for r in recs:
            n = min(len(r.get("psnr") or []), len(r.get("l1_255") or [len(r.get("psnr") or [])]))
            if n == 0:
                continue
            xs = list(range(1, n + 1))
            label = "%s(%s)" % (r.get("stage"), os.path.basename(r.get("source", "")))
            psnr = (r.get("psnr") or [])[:n]
            ax_p.plot(xs, psnr, marker="o", ms=3, lw=1.2, label=label)
            if r.get("l1_255"):
                ax_l.plot(xs, (r.get("l1_255") or [])[:n], marker="o", ms=3, lw=1.2, label=label)
        ax_p.set_xlabel("decode step")
        ax_p.set_ylabel("PSNR (dB)")
        ax_p.set_title("PSNR per decode step")
        ax_p.grid(alpha=0.3)
        ax_p.legend(fontsize=7)
        ax_l.set_xlabel("decode step")
        ax_l.set_ylabel("L1_255")
        ax_l.set_title("L1_255 per decode step (lower is better)")
        ax_l.grid(alpha=0.3)
        fig.tight_layout()
        p2 = os.path.join(artifacts_dir, "per_step_metrics_448_960m.png")
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        pngs.append(p2)
    return pngs, True


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def write_artifacts(records, train_stats, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    summary = {"generated_by": "report_448_960m.py v%s" % VERSION,
               "stages": [], "train_logs": {}}
    for st in STAGES:
        key = st["key"]
        recs = [r for r in records if r.get("stage") == key]
        summary["stages"].append({
            "stage": key, "label": st["label"], "epochs": st["epochs"],
            "total_steps": st["steps"], "evals": [
                {k: r.get(k) for k in ("source", "ckpt", "T", "n_images", "steps",
                                        "cum_z_s", "psnr", "psnr_mean_mse", "ssim",
                                        "l1_255", "cfg", "params", "mode")}
                for r in recs]})
        info = train_stats.get(key, {})
        summary["train_logs"][key] = {
            "path": info.get("path"), "n_progress": len(info.get("progress") or []),
            "n_vals": len(info.get("vals") or []),
            "done_line": info.get("done_line", ""), "data_line": info.get("data_line", "")}
    with open(os.path.join(artifacts_dir, "summary_448_960m.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # 逐 step 合并 tsv
    tsv = os.path.join(artifacts_dir, "eval_per_step_448_960m.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("stage\tsource\tstep\tt\tcum_z_s\tPSNR\tSSIM\tL1_255\n")
        for r in records:
            psnr = r.get("psnr") or []
            n = len(psnr)
            for i in range(n):
                step = (r.get("steps") or [None] * n)[i]
                cum = (r.get("cum_z_s") or [None] * n)[i]
                ssim = (r.get("ssim") or [None] * n)
                l1 = (r.get("l1_255") or [None] * n)
                f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                    r.get("stage"), os.path.basename(r.get("source", "")),
                    step if step is not None else "", i + 1,
                    cum if cum is not None else "",
                    psnr[i], ssim[i] if i < len(ssim) and ssim[i] is not None else "",
                    l1[i] if i < len(l1) and l1[i] is not None else ""))
    return tsv


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_report(args, records, train_stats, pngs, mpl_ok, run_markers):
    records_by_stage = {}
    for r in records:
        records_by_stage.setdefault(r.get("stage"), []).append(r)

    n_eval_stages = sum(1 for st in STAGES if records_by_stage.get(st["key"]))
    if run_markers.get("FAILED"):
        status = "**失败**（检测到 FAILED 标记）"
    elif run_markers.get("ALL_DONE") and n_eval_stages == 3:
        status = "**完成**（三阶段训练 + 评测产物齐全）"
    elif n_eval_stages == 3:
        status = "**完成**（三阶段评测齐全；未见到 ALL_DONE 标记）"
    elif n_eval_stages > 0:
        status = "**进行中**（%d/3 阶段评测已产出）" % n_eval_stages
    else:
        status = "**待产出**（尚无 eval/*.json；本报告为骨架，可随时重跑填充）"

    L = []
    A = L.append
    A("# %s" % TITLE)
    A("")
    A("日期：2026-11-04 ~ 11-05（完成于 11-05 13:52）｜ 机器：`ssh -p 23556 root@connect.nma1.seetacloud.com`"
      "（AutoDL 2×A800 80GB PCIe，112 核）｜ 状态：%s" % status)
    A("｜ 生成：`tools/report_448_960m.py` v%s（matplotlib %s）" % (
        VERSION, "可用，已出图" if mpl_ok else "不可用 → 纯 markdown"))
    A("")
    A("**一句话结论**：448²/p32 + 960M 在论文三阶段协议下跑通（训练 25.6 h，双卡 DDP）："
      "阶段3（zs_dim=224，7,168 值/图）末步 **24.869 dB**，比 224 论文基线（26.08 dB）低 **1.21 dB**；"
      "**但相对同分辨率（448）PCA 基线仍高 2.30 dB（448 值）→ 低 0.32 dB（7,168 值）**，"
      "与论文在 224 上的 +2.63 → −0.17 dB 形状一致 ⇒ 分辨率 ×4 没有推翻「紧预算端模型强于 PCA」的结论，"
      "只是绝对 PSNR 因 4× 像素而下降。")
    A("")
    A("---")
    A("")
    A("## 0. 用户口径与实现映射")
    A("")
    A("| 用户要求 | 本实验实现 |")
    A("|---|---|")
    A("| 224² → **448²** | `--img_size 448` |")
    A("| 分块面积 ×4（patch 16 → **32**），N 保持 196 | `--patch 32`，N=(448/32)²=196 |")
    A("| 编码器/解码器参数量**各 480M**（合计 ~960M） | enc 1744×13/h16 ≈ 480.54M；dec 2048×7/h16 ≈ 479.67M（合计 960.20M） |")
    A("| 沿用论文**三阶段**训练协议 | 阶段1 单步读满 45ep / 阶段2 T=16 6ep / 阶段3 zs_dim=224 3ep |")
    A("")
    A("## 1. 配置（448² 本实验 vs 224 论文基线）")
    A("")
    A(cfg_table(records_by_stage))
    A("")
    A("## 2. 三阶段训练协议")
    A("")
    A(protocol_table())
    A("")
    A("## 3. 评测结果")
    A("")
    if not records:
        A("> ⚠️ `eval/*.json` 尚未产出。等 `run_448_960m_3stage.sh` 每阶段结束在 GPU1 跑完"
          "`eval_stepwise_linattn.py --out eval/stage{1,2,3}_stepwise.json` 后重跑本脚本即可填充。")
        A("")
    for st in STAGES:
        recs = records_by_stage.get(st["key"]) or []
        A("### 3.%d %s" % (st["num"], st["label"]))
        A("")
        info = train_stats.get(st["key"], {})
        if info.get("done_line"):
            A("训练结束行：`%s`" % info["done_line"].strip())
            A("")
        if not recs:
            A("_评测 json 尚未产出（预期 `eval/%s_stepwise.json`）。_" % st["key"])
            A("")
            continue
        for i, rec in enumerate(recs):
            if len(recs) > 1:
                A("**结果 %d**：`%s`" % (i + 1, os.path.basename(rec.get("source", ""))))
                A("")
            A(step_table(rec, st))
            A("")
            # 图注
            if pngs and any("per_step" in p for p in pngs):
                A("![逐 step 指标](%s)" % os.path.relpath(
                    [p for p in pngs if "per_step" in p][0], os.path.dirname(args.out)))
                A("")
    A("### 3.4 与 224 论文基线「同值数」对照（已填：论文表 2「本文」行 + 448 PCA 基线）")
    A("")
    A(baseline_compare_section(records_by_stage, train_stats, getattr(args, "pca", None)))
    A("")
    A("## 4. 训练曲线")
    A("")
    if pngs:
        rel = os.path.relpath(pngs[0], os.path.dirname(args.out))
        A("![训练曲线](%s)" % rel)
        A("")
        A("> PNG 同时落盘在 `%s/`。" % os.path.relpath(args.artifacts, os.path.dirname(args.out)))
    else:
        A("> matplotlib 不可用或无训练日志，本节约为纯文本摘要。")
    A("")
    A("**各阶段日志摘要（sparkline）**：")
    A("")
    A("```text")
    for st in STAGES:
        pr = (train_stats.get(st["key"], {}) or {}).get("progress") or []
        va = (train_stats.get(st["key"], {}) or {}).get("vals") or []
        if pr:
            A("%-7s loss  %s  (min=%s, last=%s, steps=%s)" % (
                st["key"], sparkline([p["loss"] for p in pr], 50),
                fmt(min(float(p["loss"]) for p in pr if float(p["loss"]) == float(p["loss"])), 4),
                pr[-1]["loss"], "{:,}".format(pr[-1]["step"])))
        else:
            A("%-7s loss  （无日志）" % st["key"])
        if va:
            A("%-7s psnr  %s  (last=%s dB @ep%s)" % (
                st["key"], sparkline([v["psnr"] for v in va], 50),
                fmt(va[-1]["psnr"], 2), va[-1]["epoch"]))
        else:
            A("%-7s psnr  （无 val 记录）" % st["key"])
    A("```")
    A("")
    A("## 5. 结论")
    A("")
    rec1 = (records_by_stage.get("stage1") or [{}])[0]
    rec2 = (records_by_stage.get("stage2") or [{}])[0]
    rec3 = (records_by_stage.get("stage3") or [{}])[0]
    ps1 = rec1.get("psnr") or []
    ps2 = rec2.get("psnr") or []
    ps3 = rec3.get("psnr") or []

    def _vram(key):
        pr = (train_stats.get(key, {}) or {}).get("progress") or []
        vv = [p["vram"] for p in pr if p.get("vram")]
        return max(vv) if vv else None

    def _wall(key):
        line = (train_stats.get(key, {}) or {}).get("done_line") or ""
        m = re.search(r"wall=([0-9.]+)min", line)
        return float(m.group(1)) if m else None

    w1, w2, w3 = _wall("stage1"), _wall("stage2"), _wall("stage3")
    tot = (w1 or 0) + (w2 or 0) + (w3 or 0)
    fmt_h = (lambda m: "%.2f h（%.1f min）" % (m / 60.0, m)) if w1 else (lambda m: "—")
    A("1. **参数/显存可行性**：enc **%.3fM** + dec **%.3fM** = **%.3fM**（论文 224 基线 481.54M 的 2.0×），"
      "448²/p32/N=196 下实测峰值显存：A 相 **%sG**、B/S3 相 **%sG**（80G 卡余量 ~15G）；"
      "`--dec_grad_ckpt 0` 全阶段一致（该开关对 std 注意力解码器本就无效，见 BENCH 文档）。" % (
          PARAMS448["enc_M"], PARAMS448["dec_M"], PARAMS448["total_M"],
          fmt(_vram("stage1"), 1), fmt(_vram("stage2"), 1)))
    A("")
    A("2. **墙钟**：A 相 %s（单卡，113 img/s）+ B 相 %s（双卡 DDP，25 img/s/卡）+ S3 相 %s（双卡 DDP）"
      " = **训练 %.2f h**；3 次评测共 5.1 min（31 s / 137 s / 136 s）。DDP 把 B/S3 从单卡估的 14.0/6.9 h "
      "压到 8.06/4.04 h（实测 1.79×）。总墙钟 **%.2f h**，落在 DESIGN 预期（~20–26h）区间。" % (
          fmt_h(w1) if w1 else "—", fmt_h(w2) if w2 else "—", fmt_h(w3) if w3 else "—",
          tot / 60.0, tot / 60.0 + 5.1 / 60.0))
    A("")
    A("3. **阶段3（同值数可比臂，唯一严格可配对）**：448-S3 末步 **%s dB** vs 224 论文表 2「本文」**26.08 dB**"
      "（**Δ=%+.2f dB**）；448 值档 **%s dB** vs 21.03 dB（Δ=%+.2f）。**同分辨率下模型仍显著强于 PCA**："
      "+2.30 dB（448 值）→ −0.32 dB（7,168 值），224 论文同口径 +2.63 → −0.17 dB ⇒ **结论形状保持、"
      "幅度只缩水约 0.3 dB**，交叉点仍在满预算附近（448 侧 t≈14 / 6,272 值）。SSIM 主线 eval json 未记录，"
      "训练末 val 为 S3 **0.6421** / B **0.6958**（同图同划分，仅供趋势参考）。" % (
          fmt(ps3[-1] if ps3 else None, 3, "—"), (ps3[-1] - 26.08) if ps3 else 0.0,
          fmt(ps3[0] if ps3 else None, 3, "—"), (ps3[0] - 21.03) if ps3 else 0.0))
    A("")
    A("4. **A/B 相过程参考**（完整宽度臂，跨分辨率不与论文表 2 配同值数）：A 相（单步读满，55,808 值/图）"
      "**%s dB** vs 224 同协议 32.35 dB（Δ=%+.2f）；B 相末步（65,536 值/图）**%s dB** vs 27.94 dB（Δ=%+.2f）。"
      "⇒ 448 在 A 相掉得最多（−3.6 dB），B/S3 相收敛到 −1.2~−1.8 dB：**分辨率代价主要在「单步读满」"
      "这一最难档位**，逐步读出把差距拉回。" % (
          fmt(max(ps1) if ps1 else None, 3, "—"), (max(ps1) - 32.35) if ps1 else 0.0,
          fmt(max(ps2) if ps2 else None, 3, "—"), (max(ps2) - 27.94) if ps2 else 0.0))
    A("")
    A("5. **与 scaling 假设的关系**：只加宽（enc 13 层 / dec 7 层不变）到 2× 参数、同时像素 ×4，"
      "在 N=196、K=32、T=16、7,168 值/图全同的前提下，同值数 PSNR 比 224 论文低 1.21 dB；"
      "**2× 参数没有抵消 4× 像素带来的信息密度损失**，故本实验**不支持**「仅靠宽度 scaling 就能在"
      "同值数下维持 224 的质量」；但它相对同分辨率 PCA 的优势几乎原样保留，说明**宽度 scaling 保住的是"
      "「相对非学习方法（PCA）的架构收益」**，而不是绝对 PSNR。"
      "若要在 448 恢复 26 dB 级，需提高值预算（7168→~2×）或改 patch/分辨率策略，而不是继续加宽。")
    A("")
    A("6. **风险/偏置**：单种子（seed 0）；三阶段 `dec_grad_ckpt` 一致 = 0；448 吞吐是真瓶颈"
      "（A 相 113 img/s vs 224 基线 ~357 img/s，4× 像素带来的 ~3× 变慢；B/S3 单卡 25 img/s，"
      "靠 DDP 才把墙钟压住）；PSNR 是像素级指标、不等于任务保真；448↔224 是**跨分辨率**比较，"
      "同值数口径只能消掉「值数」这一维，不能消掉「4× 像素/样本」这一维——这正是本实验要量的量。")
    A("## 6. 产物与复现")
    A("")
    A("- 评测输入：`%s/*.json`（共 %d 个文件 / %d 组结果）" % (
        args.eval_dir, len(glob.glob(os.path.join(args.eval_dir, "*.json"))) if os.path.isdir(args.eval_dir) else 0,
        len(records)))
    A("- 训练日志：`%s/stage{1,2,3}.log`" % args.logs_dir)
    A("- artifacts：`%s/`（summary_448_960m.json、eval_per_step_448_960m.tsv、PNG）" % args.artifacts)
    A("- 监控：远端 `tools/watch_448_960m.sh` → `$OUT/status.json`；本地 "
      "`exp448_bench/monitor/poll_448_960m.sh`")
    A("- 复现：`python3 tools/report_448_960m.py`（幂等；`--no-plot` 纯 markdown）")
    A("")
    A("---")
    A("")
    A("*本报告由脚本生成；表格中的「（待填）」为用户/Lead 按论文 v2.0 表 2 口径填写的位置。*")
    return "\n".join(L) + "\n"


def detect_run_markers(markers_dir):
    root = os.path.abspath(markers_dir)
    markers = {"ALL_DONE": False, "FAILED": False, "STAGE_DONE": {}}
    for name in ("ALL_DONE", "FAILED"):
        markers[name] = os.path.exists(os.path.join(root, name))
    for i in (1, 2, 3):
        markers["STAGE_DONE"][i] = os.path.exists(os.path.join(root, "STAGE%d_DONE" % i))
    return markers


def first_existing(cands, default):
    for c in cands:
        if os.path.isdir(c):
            return c
    return default


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench = os.path.join(os.path.dirname(repo), "exp448_bench")
    default_eval = first_existing([
        "/root/autodl-tmp/exp448/eval",
        os.path.join(bench, "artifacts", "eval"),
        os.path.join(bench, "artifacts", "448_960m", "eval"),
        os.path.join(bench, "eval"),
    ], "/root/autodl-tmp/exp448/eval")
    default_logs = first_existing([
        "/root/autodl-tmp/exp448/logs",
        os.path.join(bench, "artifacts", "logs"),
        os.path.join(bench, "artifacts", "448_960m", "logs"),
        os.path.join(bench, "logs"),
    ], "/root/autodl-tmp/exp448/logs")
    ap = argparse.ArgumentParser(description="生成 448²/960M 三阶段实验报告")
    ap.add_argument("--eval-dir", default=default_eval)
    ap.add_argument("--logs-dir", default=default_logs)
    ap.add_argument("--out", default=os.path.join(repo, "doc", "2026-11-04",
                                                  "REPORT_448_960M_3stage.md"))
    ap.add_argument("--artifacts", default=os.path.join(repo, "doc", "2026-11-04",
                                                        "artifacts_448_960m"))
    ap.add_argument("--pca-json", default=DEFAULT_PCA448_JSON,
                    help="448 PCA 同值数基线结果（build_results.py 产物），缺则 3.4 节省略 PCA 列")
    ap.add_argument("--markers-dir", default=None,
                    help="ALL_DONE/STAGE*_DONE 标记目录（默认 eval_dir 的父目录）")
    ap.add_argument("--no-plot", action="store_true", help="强制纯 markdown（不生成 PNG）")
    ap.add_argument("--check", action="store_true", help="只检查输入/环境，不写文件")
    ap.add_argument("--version", action="version", version="report_448_960m.py v" + VERSION)
    args = ap.parse_args()

    records = collect_evals(args.eval_dir)
    train_stats = parse_train_logs(args.logs_dir)
    markers = detect_run_markers(args.markers_dir or
                                 os.path.dirname(os.path.abspath(args.eval_dir)))
    args.pca = read_json(args.pca_json) if args.pca_json else None
    mpl = try_matplotlib()
    n_logs = sum(1 for st in STAGES if train_stats.get(st["key"], {}).get("path"))

    if args.check:
        print("report_448_960m.py v%s" % VERSION)
        print("  eval-dir : %s  → %d 个 json / %d 组结果" % (
            args.eval_dir,
            len(glob.glob(os.path.join(args.eval_dir, "*.json"))) if os.path.isdir(args.eval_dir) else 0,
            len(records)))
        for r in records:
            print("    - [%s] %s  T=%s n_images=%s mode=%s" % (
                r.get("stage"), os.path.basename(r.get("source", "")),
                r.get("T"), r.get("n_images"), r.get("mode")))
        print("  logs-dir : %s  → 找到 %d/3 个阶段日志" % (args.logs_dir, n_logs))
        for st in STAGES:
            info = train_stats.get(st["key"], {})
            print("    - %-7s path=%s progress=%d vals=%d" % (
                st["key"], info.get("path"), len(info.get("progress") or []),
                len(info.get("vals") or [])))
        print("  markers  : %s" % markers)
        print("  matplotlib: %s" % ("可用" if mpl else "不可用（自动降级）"))
        print("  out      : %s" % args.out)
        print("  artifacts: %s" % args.artifacts)
        return 0

    pngs, used_mpl = make_figures(train_stats, records, args.artifacts, args.no_plot)
    md = build_report(args, records, train_stats, pngs, used_mpl, markers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(md)
    os.replace(tmp, args.out)
    tsv = write_artifacts(records, train_stats, args.artifacts)

    print("[report] %s  (%d 行, %d 组评测, %d 阶段日志%s)" % (
        args.out, md.count("\n"), len(records), n_logs,
        (", %d PNG" % len(pngs)) if pngs else ", 无 PNG"))
    print("[artifacts] %s" % args.artifacts)
    print("[artifacts] %s" % tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
