#!/usr/bin/env python3
"""make_report.py — Qwen 视觉 token PCA 截断 × 同参数量 AssocVTC 单步还原探针
的汇总 / 画图 / 8 图网格。

用法（远端 /root/autodl-tmp/qwen_vtc/probe/）:
  python make_report.py --mode summary   # 只读 json/npy（CPU，几秒）
  python make_report.py --mode grid      # 需要 GPU：4 个 best.pt 对 8 张固定 val 推理
  python make_report.py --mode all

产物（都写在 probe/ 下）:
  summary.json
  fig_qwen_probe_psnr_vs_r.png / .pdf     # 4 训练臂 + 4 零训练截断点 + meanbase
  fig_qwen_probe_grid.png                 # GT + 4 臂重建，8 张 val 样例
  grid_recon_8.npy                        # 网格重建数组（8,5,224,224,3 float32）

口径: PSNR/SSIM/L1 一律 [0,1] 像素空间；SSIM 用 metrics_linattn.ssim (fp32)；
L1_255 = 0-255 刻度（[0,1] L1 × 255）。与 qwen_probe.py 的 evaluate 同源。
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

PD = "/root/autodl-tmp/qwen_vtc/probe"
ROOT = "/root/autodl-tmp/qwen_vtc"
PCA = os.path.join(ROOT, "pca")
TOK = os.path.join(ROOT, "tokens")
CODE = os.path.join(ROOT, "code")
PROBE_PY = os.path.join(CODE, "qwen_probe.py")

# 表/图顺序：由小 r 到大 r
ARM_ORDER = ["1sigma", "2sigma", "3sigma", "all"]
ARM_LABEL = {"1sigma": "1σ", "2sigma": "2σ", "3sigma": "3σ", "all": "全部 (5120)"}
FIG_LABEL = {"1σ": "1σ", "2σ": "2σ", "3σ": "3σ", "全部 (5120)": "all (5120)"}


def fig_label(d):
    """图内标签：避免 CJK 字体缺失（σ 是希腊字母, DejaVu 有）。"""
    lab = d.get("r_label") or ""
    return FIG_LABEL.get(lab, lab)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def read_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl(p):
    out = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def load_arms():
    arms = read_json(os.path.join(PCA, "arms.json"))
    if arms is None:
        raise SystemExit("[fatal] pca/arms.json 不存在")
    rmap = {k: int(arms[k]) for k in ARM_ORDER if k in arms}
    if "all" not in rmap:
        rmap["all"] = 5120
    return rmap


def cumvar_at(r):
    """累计解释方差 @ r（r>=5120 → 1.0）。"""
    p = os.path.join(PCA, "cumvar.npy")
    if r >= 5120 or not os.path.exists(p):
        return 1.0 if r >= 5120 else None
    cv = np.load(p)
    r = int(min(r, len(cv)))
    return float(cv[r - 1])


def parse_epoch_log(r):
    """从 nohup 日志提取每 epoch 子集指标: [(ep, psnr, ssim, l1, gstep, min)]"""
    p = os.path.join(PD, f"logs_r{r}.log")
    rows = []
    try:
        pat = re.compile(
            r"\[probe\] ep(\d+) subset PSNR ([\d.]+) SSIM ([\d.]+) "
            r"L1 ([\d.]+) \| (\d+) steps ([\d.]+) min")
        for line in open(p, errors="replace"):
            m = pat.search(line)
            if m:
                rows.append({"ep": int(m.group(1)), "subset_psnr": float(m.group(2)),
                             "subset_ssim": float(m.group(3)),
                             "subset_l1_255": float(m.group(4)),
                             "steps": int(m.group(5)),
                             "elapsed_min": float(m.group(6))})
    except Exception:
        pass
    return rows


def load_arm_result(r):
    """训练臂：eval r=该臂 r 的结果 + 训练元信息。"""
    ej = read_json(os.path.join(PD, f"eval_r{r}.json"))
    d = {"r": int(r), "r_label": None, "cumvar_at_r": cumvar_at(r),
         "eval_json": f"eval_r{r}.json" if ej else None,
         "ckpt": os.path.join(PD, f"r{r}", "best.pt") if os.path.exists(
             os.path.join(PD, f"r{r}", "best.pt")) else None,
         "epochs": None, "steps": None, "epoch_times_min": [], "epoch_log": []}
    args = read_json(os.path.join(PD, f"r{r}", "args.json")) or {}
    if "epochs" in args:
        d["epochs"] = int(args["epochs"])
    d["bs"] = args.get("bs")
    d["lr"] = args.get("lr")
    d["weight_decay"] = args.get("weight_decay")
    d["seed"] = args.get("seed")
    ts = read_json(os.path.join(PD, f"r{r}", "train_state.json")) or {}
    if "steps" in ts:
        d["steps"] = int(ts["steps"])
    if "best_subset_psnr" in ts:
        d["best_subset_psnr"] = float(ts["best_subset_psnr"])
    d["epoch_log"] = parse_epoch_log(r)
    d["epoch_times_min"] = [row["elapsed_min"] for row in d["epoch_log"]]
    if ej:
        d.update({"psnr": _r(ej.get("psnr"), 4), "ssim": _r(ej.get("ssim"), 4),
                  "l1_255": _r(ej.get("l1_255"), 3), "n": ej.get("n"),
                  "params_M": ej.get("params_M"),
                  "ckpt_ep": ej.get("ckpt_ep"),
                  "per_image_npy": f"eval_r{r}_per_image.npy"})
    return d


def load_zeroshot(r):
    """零训练截断：用 r=5120 臂 best.pt 在 r 截断输入上直接评。"""
    p = os.path.join(PD, f"eval_zeroshot_r{r}.json")
    ej = read_json(p)
    if ej is None:
        return None
    return {"r": int(r), "r_label": None,
            "cumvar_at_r": cumvar_at(r), "eval_json": f"eval_zeroshot_r{r}.json",
            "psnr": _r(ej.get("psnr"), 4), "ssim": _r(ej.get("ssim"), 4),
            "l1_255": _r(ej.get("l1_255"), 3), "n": ej.get("n")}


def load_data_section(rmap):
    tm = read_json(os.path.join(TOK, "meta.json")) or {}
    qc = read_json(os.path.join(TOK, "quant_check.json")) or {}
    return {
        "n_train": tm.get("n_train"), "n_val": tm.get("n_val"),
        "tokens_per_image": tm.get("tokens_per_image"),
        "dim": tm.get("dim"), "resolution": tm.get("resolution"),
        "quant_scheme": tm.get("quant_scheme"),
        "token_psnr_db": _r(qc.get("token_psnr_db"), 2),
        "quant_rel_rmse": qc.get("rel_rmse"),
        "quant_note": "train 存 int8 B2（位置中心化+逐通道归一+逐 token int8, "
                      "反量化 x=(q*sc)*sd+mu_p；mu_p/sd 由 val 5,000 张 fp16 token 估计，"
                      "只影响存储刻度）；val 评测输入用 fp16 原值 ⇒ 训练/评测输入精度不完全对称",
        "val_tokens_dtype": tm.get("val_tokens_dtype"),
        "model": tm.get("model"),
        "pca": {"fit_on": "IN-100 train 全量池化 token（49 位置合并，6,207,761 样本）",
                "n_samples": (tm.get("n_train") or 0) * (tm.get("tokens_per_image") or 0),
                "mean_subtracted": True, "components_desc": "列 = 降序特征向量",
                "arms_r": rmap},
    }


def build_summary():
    rmap = load_arms()
    label_of = {rmap[k]: ARM_LABEL[k] for k in rmap}
    meanbase = read_json(os.path.join(PD, "meanbase.json"))
    bench = read_json(os.path.join(PD, "bench.json"))

    arms = [load_arm_result(rmap[k]) for k in ARM_ORDER if k in rmap]
    for a in arms:
        a["r_label"] = label_of.get(a["r"])
    arms_by_r = {a["r"]: a for a in arms}

    # all 臂（r=5120）作为参照
    all_r = rmap.get("all", 5120)
    ref = arms_by_r.get(all_r) or {}
    ref_psnr = ref.get("psnr")
    mb_psnr = (meanbase or {}).get("psnr")

    for a in arms:
        if a.get("psnr") is not None and ref_psnr is not None:
            a["delta_vs_all"] = round(a["psnr"] - ref_psnr, 4)
        else:
            a["delta_vs_all"] = None
        if a.get("psnr") is not None and mb_psnr is not None:
            a["delta_vs_meanbase"] = round(a["psnr"] - mb_psnr, 4)
        else:
            a["delta_vs_meanbase"] = None

    # 零训练截断：r1/r2/r3 + r=5120（同 ckpt 全输入 = all 臂自身评测）
    zs = []
    for k in ["1sigma", "2sigma", "3sigma"]:
        if k in rmap:
            z = load_zeroshot(rmap[k])
            if z is not None:
                z["r_label"] = ARM_LABEL[k]
                zs.append(z)
    if ref.get("psnr") is not None:
        zs.append({"r": int(all_r), "r_label": "全部 (5120)", "cumvar_at_r": 1.0,
                   "eval_json": f"eval_r{all_r}.json",
                   "psnr": ref["psnr"], "ssim": ref.get("ssim"),
                   "l1_255": ref.get("l1_255"), "n": ref.get("n"),
                   "note": "= all 臂自身全输入评测（零训练截断的右端点，非独立点）"})
    zs.sort(key=lambda x: x["r"])
    for z in zs:
        z["delta_vs_all"] = (round(z["psnr"] - ref_psnr, 4)
                             if z.get("psnr") is not None and ref_psnr is not None else None)
        z["delta_vs_meanbase"] = (round(z["psnr"] - mb_psnr, 4)
                                  if z.get("psnr") is not None and mb_psnr is not None else None)

    epochs_set = {a["epochs"] for a in arms if a.get("epochs")}
    steps_set = {a["steps"] for a in arms if a.get("steps")}
    one_epoch = None
    for a in arms:
        if len(a["epoch_times_min"]) >= 1:
            one_epoch = a["epoch_times_min"][0]
            break
    pilot = None
    pdir = os.path.join(PD, "r5120_pilot")
    pargs = read_json(os.path.join(pdir, "args.json"))
    plog = parse_epoch_log("5120_pilot")
    if pargs is not None:
        pilot = {"dir": "r5120_pilot", "epochs": pargs.get("epochs"),
                 "epoch_log": plog}
        if plog:
            pilot["epoch_time_min"] = plog[-1]["elapsed_min"]
            pilot["subset_psnr"] = plog[-1]["subset_psnr"]
    protocol = {
        "arms_same": ("同架构/同 seed(0)/同数据顺序/同 epoch 预算/同超参；"
                      "唯一自变量 = 输入 PCA 截断维数 r"),
        "epochs": sorted(epochs_set)[0] if len(epochs_set) == 1 else sorted(epochs_set),
        "steps_per_arm": sorted(steps_set)[0] if len(steps_set) == 1 else sorted(steps_set),
        "bs": 64, "lr": 2e-4, "weight_decay": 0.05, "warmup": 500,
        "schedule": "cosine", "autocast": "bf16", "grad_clip": 1.0,
        "loss": "L1（归一化像素空间）", "opt": "AdamW",
        "train_set": "ImageNet-100 train（126,689 图，随机重排，drop_last=True）",
        "eval_subset_per_epoch": 512,
        "eval_full": 5000,
        "epoch_time_min_first": one_epoch,
        "pilot": pilot,
        "budget_rule": "E = floor(2.5h / (4 × epoch_time))，且 E ≥ 3",
    }
    if bench:
        protocol["bench"] = bench

    data_sec = load_data_section(rmap)
    chosen = (bench or {}).get("chosen") or {}
    E = None
    for a in arms:
        if a.get("epochs"):
            E = a["epochs"]
    n_train = data_sec.get("n_train") or 0
    training = {
        "subset_size": n_train,
        "subset_note": "全量 126,689（不做子集；预算规则 T≥150 分支）",
        "E": E,
        "bs": 128, "workers": 16, "lr": 2e-4, "weight_decay": 0.05,
        "warmup_steps": (min(500, ((E or 1) * (n_train // 128)) // 10) if n_train else None), "schedule": "cosine", "autocast": "bf16", "grad_clip": 1.0,
        "seed": 0, "opt": "AdamW", "loss": "L1（归一化像素空间）",
        "img_per_arm": (E * n_train) if E else None,
        "T_img_s_bench": chosen.get("img_s_observed_in_train"),
        "epoch_min_est": chosen.get("epoch_time_min_est"),
        "budget_note": "为 2h 预算：bs 提到 128（原设计 64）、E=2、全量数据、4 臂同协议；"
                       "lr 未按 batch 线性缩放（每图有效步数比 bs64 减半），臂间可比、"
                       "但与 bs64 协议的绝对值不可直接对照",
        "pilot": "跳过（Lead 决定：用 bench 吞吐直接定 E）",
        "arms_order": ["all(5120)", "r3σ", "r2σ", "r1σ"],
        "ckpt_dtype": "fp16 state_dict（落盘压缩；加载回 fp32）",
    }
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "Qwen3.8-27B 仅图像编码器 token（49×5120, 224²）的池化 PCA 截断 → "
                      "同参数量 AssocVTC 交叉注意力单步还原探针（4 臂）",
        "metric_protocol": "PSNR/SSIM/L1_255 均在 [0,1] 像素空间；SSIM = metrics_linattn.ssim "
                           "(11×11 高斯窗, fp32)；L1_255 = [0,1] L1 ×255；重建自 "
                           "patches_to_pixels（mean/std=0.5, patch=16, 224²）",
        "params_M": {
            "total": 244.6894,
            "decoder_OutputQueryDecoder": 232.7890,
            "pixel_head": 4.5248,
            "decoder_body_decoder_plus_pixel_head": 237.3138,
            "mem_proj_5120_to_1440": 7.3742,
            "z_cls": 1440,
            "assocvtc_240m_decoder_ref_M": 237.11,
            "note": "decoder body 237.31M vs AssocVTC 240M 档 237.11M = +0.09%；"
                    "输入适配 mem_proj + z_cls = +7.37M（+3.0%）",
        },
        "data": data_sec,
        "training": training,
        "protocol": protocol,
        "arms": arms,
        "meanbase": meanbase,
        "zeroshot": zs,
        "honest_boundaries": [
            "单种子（seed=0）；4 臂从零训练且预算相同，但不是超参搜索（lr/wd/warmup/bs 未调）",
            "为 2h 墙上预算：bs 提到 128（原设计 64）、E=2、全量 126,689 张、4 臂完全同协议；"
            "lr 仍是 2e-4（未按 batch 线性缩放）⇒ 每图有效优化步数比 bs64 少一半，"
            "臂间比较有效，绝对 PSNR 不可与 bs64/45ep 的 240M 参考线直接对照",
            "PCA 基在**训练集**（126,689 图 × 49 token 池化）上拟合，截断输入是训练集统计量 ⇒ "
            "可泛化口径而非 oracle（val 未参与拟合，但与 train 同域）",
            "train/val 同域（ImageNet-100），val 5,000 张训练未见",
            "像素指标（PSNR/SSIM/L1）≠ 任务保真（分类/检索/生成下游未测）",
            "探针含 +7.37M 输入适配层（mem_proj，+3.0%），decoder 主体与 240M 档对齐但整体略大",
            "每图 49 token × 5120 维、224²、patch=16（N=196）；Qwen token 网格 7×7，"
            "与像素 patch 网格 14×14 非同构映射（由探针自行学习）",
            "训练输入是 int8 B2 存储（token PSNR ~46.2 dB, rel_rmse 0.0049），val 评测输入是 "
            "fp16 原值（无损）；train/eval 输入精度不完全对称，但 B2 噪声远小于 PCA 截断误差",
            "ckpt 落盘为 fp16（省盘）：数值 = fp32 训练值的 fp16 舍入（相对 1e-3），"
            "对 PSNR/SSIM 影响 <0.001 dB 量级",
            "逐 epoch 子集 PSNR 是固定 512 张 val 子集（seed 123），全量 5,000 张只用于最终评测；"
            "best.pt 按子集 PSNR 选，存在轻微子集选择偏差",
            "all 臂 = 恒等输入（r=5120），与截断臂共享超参；4 臂训练数据顺序相同来自同一 "
            "torch RNG 种子，但不同进程/机器状态不保证逐 bit 一致",
        ],
        "artifacts": {
            "train_logs": sorted(glob.glob(os.path.join(PD, "logs_r*.log"))),
            "eval_jsons": sorted(glob.glob(os.path.join(PD, "eval_*.json"))),
            "fig_psnr": ["fig_qwen_probe_psnr_vs_r.png", "fig_qwen_probe_psnr_vs_r.pdf"],
            "fig_grid": "fig_qwen_probe_grid.png",
        },
    }
    # 总耗时：日志里的 elapsed 是"从训练开始到该 epoch 末"的累计分钟
    per_arm, tot = [], 0.0
    for a in arms:
        if a["epoch_times_min"]:
            last = max(a["epoch_times_min"])
            per_arm.append({"r": a["r"], "wall_min": round(last, 1)})
            tot += last
    if per_arm:
        summary["protocol"]["per_arm_wall_min"] = per_arm
        summary["protocol"]["estimated_train_wall_min"] = round(tot, 1)
    os.makedirs(PD, exist_ok=True)
    with open(os.path.join(PD, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("[summary] wrote", os.path.join(PD, "summary.json"))
    for a in summary["arms"]:
        print(" arm r=%-5d %-12s PSNR %s SSIM %s L1 %s (Δvs_all %s)"
              % (a["r"], a["r_label"], a.get("psnr"), a.get("ssim"),
                 a.get("l1_255"), a.get("delta_vs_all")))
    for z in summary["zeroshot"]:
        print(" zs  r=%-5d %-12s PSNR %s (Δvs_all %s)"
              % (z["r"], z["r_label"], z.get("psnr"), z.get("delta_vs_all")))
    if summary["meanbase"]:
        print(" meanbase PSNR %.4f SSIM %.4f L1 %.3f" % (
            summary["meanbase"].get("psnr", float("nan")),
            summary["meanbase"].get("ssim", float("nan")),
            summary["meanbase"].get("l1_255", float("nan"))))
    return summary


def fig_curve(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    arms = [a for a in summary["arms"] if a.get("psnr") is not None]
    zs = [z for z in summary["zeroshot"] if z.get("psnr") is not None]
    mb = summary.get("meanbase") or {}
    if not arms:
        print("[fig] 无训练臂结果，跳过")
        return
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, key, name in [(axes[0], "psnr", "PSNR (dB)"),
                          (axes[1], "ssim", "SSIM")]:
        arms_s = sorted(arms, key=lambda d: d["r"])
        zs_k = sorted([z for z in zs if z.get(key) is not None], key=lambda d: d["r"])
        ax.set_xscale("log", base=2)          # 必须先设 scale 再设 ticks（否则默认 locator 覆盖）
        # x 刻度 = 训练臂 r（1σ/2σ/3σ 附带累计解释方差）
        tick_r = sorted({a["r"] for a in arms_s})
        labs = {}
        for a in arms_s:
            if a["r"] >= 5120:
                labs[a["r"]] = f"all\nr={a['r']}"
            else:
                cv = a.get("cumvar_at_r")
                labs[a["r"]] = (f"{fig_label(a)}\nr={a['r']}\ncum={cv:.3f}"
                                if cv is not None else f"{fig_label(a)}\nr={a['r']}")
        ax.set_xticks(tick_r)
        ax.set_xticklabels([labs[t] for t in tick_r], fontsize=8.5)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(min(tick_r) * 0.62, max(tick_r) * 1.55)

        ax.plot([a["r"] for a in arms_s], [a[key] for a in arms_s], "o-",
                color="#1f77b4", lw=2.2, ms=9,
                label="trained arm (its own r, from scratch)", zorder=5)
        for a in arms_s:
            ax.annotate(f'{a[key]:.3f}' if key == "ssim" else f'{a[key]:.2f}',
                        (a["r"], a[key]), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9, color="#1f77b4")
        if zs_k:
            zmax = max(t for t in tick_r)
            zplot = [z for z in zs_k if z["r"] != zmax] or []
            zr = [z for z in zs_k]
            ax.plot([z["r"] for z in zr], [z[key] for z in zr], "s--",
                    color="#d62728", lw=2.0, ms=8,
                    label="zero-shot truncation (r=5120 ckpt, truncated input)",
                    zorder=4)
            for z in zplot:
                ax.annotate(f'{z[key]:.3f}' if key == "ssim" else f'{z[key]:.2f}',
                            (z["r"], z[key]), textcoords="offset points",
                            xytext=(0, -17), ha="center", fontsize=9,
                            color="#d62728")
        if key == "psnr" and mb.get("psnr") is not None:
            ax.axhline(mb["psnr"], color="gray", ls=":", lw=2,
                       label=f'mean-image baseline ({mb["psnr"]:.2f} dB)')
        else:
            if mb.get("ssim") is not None:
                ax.axhline(mb["ssim"], color="gray", ls=":", lw=2,
                           label=f'mean-image baseline ({mb["ssim"]:.3f})')
        ax.tick_params(axis="x", which="minor", length=2)
        ax.margins(y=0.14)
        ax.set_xlabel("number of retained PCs r  (log2 scale)")
        ax.set_ylabel(name)
        ax.grid(alpha=.3, which="major")
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Qwen visual tokens (49x5120, 224^2) under PCA truncation -> same-size "
                 "AssocVTC single-step reconstruction probe (244.69M)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        p = os.path.join(PD, f"fig_qwen_probe_psnr_vs_r.{ext}")
        fig.savefig(p, dpi=160 if ext == "png" else 300, bbox_inches="tight")
        print("[fig] wrote", p)
    plt.close(fig)


def fig_grid():
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, CODE)
    import qwen_probe as Q

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rmap = load_arms()
    r_to_label = {rmap[k]: ARM_LABEL[k] for k in rmap}
    val_tok = os.path.join(TOK, "val_tokens_fp16.npy")
    val_img = os.path.join(TOK, "val_images_224.npy")
    N = len(np.load(val_img, mmap_mode="r"))
    idx = np.random.default_rng(0).choice(N, size=8, replace=False)
    idx.sort()
    ti = np.load(val_tok, mmap_mode="r")
    ii = np.load(val_img, mmap_mode="r")
    toks = torch.from_numpy(np.asarray(ti[idx]).astype(np.float32)).to(device)
    imgs_u8 = np.asarray(ii[idx])
    gt = torch.from_numpy(imgs_u8).permute(0, 3, 1, 2).float() / 255.0   # (8,3,224,224)

    order = [k for k in ARM_ORDER if k in rmap]
    order = sorted(order, key=lambda k: -rmap[k])  # 大 r → 小 r
    rows = [("GT", gt)]
    recs = {}
    for k in order:
        r = rmap[k]
        ck = os.path.join(PD, f"r{r}", "best.pt")
        if not os.path.exists(ck):
            print("[grid] skip 缺失 ckpt", ck)
            continue
        proj = Q.Projector(PCA, r, device)
        model = Q.build_model(device)
        sd = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        model.eval()
        with torch.no_grad():
            t = proj(toks)
            with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" \
                    else torch.autocast("cpu", enabled=False):
                pix = model(t)
            rec = Q.patches_to_pixels(pix.float()).permute(0, 3, 1, 2).clamp(0, 1).cpu()
        recs[r] = rec
        rows.append((f"{FIG_LABEL.get(ARM_LABEL[k], ARM_LABEL[k])}  r={r}", rec))
    if len(rows) < 2:
        print("[grid] 没有可用 ckpt，跳过")
        return

    # 每行 PSNR（8 张平均）
    from metrics_linattn import psnr as _psnr
    psnrs = {}
    for name, im in rows:
        if name == "GT":
            continue
        psnrs[name] = float(_psnr(im, gt).mean())

    R, C = len(rows), 8
    fig, axes = plt.subplots(R, C, figsize=(C * 2.05, R * 2.35))
    for ri, (name, im) in enumerate(rows):
        for ci in range(C):
            ax = axes[ri, ci]
            ax.imshow(im[ci].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"val[{idx[ci]}]", fontsize=8)
            if ci == 0:
                lbl = name if name == "GT" else f"{name}\nPSNR {psnrs[name]:.2f} dB"
                ax.set_ylabel(lbl, fontsize=8.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle("Qwen visual tokens -> single-step reconstruction: 8 IN-100 val samples "
                 "(GT vs 4 trained arms)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(PD, "fig_qwen_probe_grid.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("[fig] wrote", p)
    plt.close(fig)
    arr = np.stack([im.numpy() for _, im in rows], axis=1)  # (8, R, 3, 224, 224)
    np.save(os.path.join(PD, "grid_recon_8.npy"), arr)
    json.dump({"indices": idx.tolist(), "rows": [n for n, _ in rows],
               "psnr8": psnrs},
              open(os.path.join(PD, "grid_meta.json"), "w"), indent=1)
    print("[grid] wrote grid_recon_8.npy / grid_meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["summary", "grid", "all"], default="all")
    args = ap.parse_args()
    if args.mode in ("summary", "all"):
        s = build_summary()
        fig_curve(s)
    if args.mode in ("grid", "all"):
        fig_grid()


if __name__ == "__main__":
    main()
