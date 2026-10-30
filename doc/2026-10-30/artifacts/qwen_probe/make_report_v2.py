#!/usr/bin/env python3
"""make_report_v2.py — 探针 v2（AssocVTC 240M 真解码器 + 热启动）的汇总 / 画图 / 8 图网格。

用法（远端 /root/autodl-tmp/qwen_vtc/probe/）:
  python make_report_v2.py --mode summary   # CPU，读 json/log 出 summary_v2.json + 曲线图
  python make_report_v2.py --mode grid      # GPU，4 臂 best.pt 对 8 张固定 val 推理出网格
  python make_report_v2.py --mode all

产物:
  probe/summary_v2.json
  probe/fig_qwen_probe_v2_psnr_vs_r.png / .pdf
  probe/fig_qwen_probe_v2_grid.png
  probe/grid_v2_recon_8.npy / grid_v2_meta.json
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
PROBE_V2 = os.path.join(CODE, "qwen_probe_v2.py")

ARM_ORDER = ["1sigma", "2sigma", "3sigma", "all"]
ARM_LABEL = {"1sigma": "1σ", "2sigma": "2σ", "3sigma": "3σ", "all": "全部 (5120)"}
FIG_LABEL = {"1σ": "1σ", "2σ": "2σ", "3σ": "3σ", "全部 (5120)": "all (5120)"}


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def read_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def load_arms():
    a = read_json(os.path.join(PCA, "arms.json"))
    if a is None:
        raise SystemExit("[fatal] pca/arms.json missing")
    return {k: int(a[k]) for k in ARM_ORDER if k in a}


def cumvar_at(r):
    p = os.path.join(PCA, "cumvar.npy")
    if r >= 5120:
        return 1.0
    if not os.path.exists(p):
        return None
    cv = np.load(p)
    return float(cv[min(int(r), len(cv)) - 1])


def parse_log(r, tag="v2_r"):
    """日志里: [init] loaded ...; [probe_v2] ep0 subset PSNR .. | N steps X min; [eval] {...}"""
    p = os.path.join(PD, f"logs_{tag}{r}.log")
    out = {"epochs": [], "init": None, "eval": None}
    try:
        pat_ep = re.compile(r"\[probe_v2\] ep(\d+) subset PSNR ([\d.]+) SSIM ([\d.]+) "
                            r"L1 ([\d.]+) \| (\d+) steps ([\d.]+) min")
        for line in open(p, errors="replace"):
            m = pat_ep.search(line)
            if m:
                out["epochs"].append({"ep": int(m.group(1)), "subset_psnr": float(m.group(2)),
                                      "subset_ssim": float(m.group(3)),
                                      "subset_l1_255": float(m.group(4)),
                                      "steps": int(m.group(5)),
                                      "elapsed_min": float(m.group(6))})
            if line.startswith("[init] loaded"):
                out["init"] = line.strip()
            if line.startswith("[eval]"):
                try:
                    out["eval"] = json.loads(line[len("[eval]"):].strip())
                except Exception:
                    pass
    except Exception:
        pass
    return out


def arm_result(r, label, init_load):
    ej = read_json(os.path.join(PD, f"eval_v2_r{r}.json"))
    d = {"r": int(r), "r_label": label, "cumvar_at_r": _r(cumvar_at(r), 6),
         "ckpt": f"probe/v2_r{r}/best.pt", "eval_json": f"eval_v2_r{r}.json"}
    args = read_json(os.path.join(PD, f"v2_r{r}", "args.json")) or {}
    d.update({"epochs": args.get("epochs"), "bs": args.get("bs"),
              "lr": args.get("lr"), "warmup": args.get("warmup"),
              "seed": args.get("seed"), "init": args.get("init")})
    lg = parse_log(r)
    d["epoch_log"] = lg["epochs"]
    if lg["epochs"]:
        d["steps"] = lg["epochs"][-1]["steps"]
        d["subset_psnr_final"] = lg["epochs"][-1]["subset_psnr"]
        d["wall_min"] = lg["epochs"][-1]["elapsed_min"]
    if ej:
        d.update({"psnr": _r(ej.get("psnr")), "ssim": _r(ej.get("ssim")),
                  "l1_255": _r(ej.get("l1_255"), 3), "n": ej.get("n"),
                  "params_M": _r(ej.get("params_M"), 3),
                  "ckpt_ep": ej.get("ckpt_ep")})
    if init_load:
        d["params_M"] = _r(init_load.get("params_M"), 3)
        d["init_missing_keys"] = init_load.get("missing")
        d["init_unexpected_keys"] = init_load.get("unexpected")
    return d


def zeroshot_entry(r, label, all_psnr):
    ej = read_json(os.path.join(PD, f"eval_v2_zeroshot_r{r}.json"))
    if ej is None:
        return None
    return {"r": int(r), "r_label": label, "cumvar_at_r": _r(cumvar_at(r), 6),
            "eval_json": f"eval_v2_zeroshot_r{r}.json",
            "psnr": _r(ej.get("psnr")), "ssim": _r(ej.get("ssim")),
            "l1_255": _r(ej.get("l1_255"), 3), "n": ej.get("n"),
            "delta_vs_all": _r((ej.get("psnr") or 0) - all_psnr) if all_psnr is not None else None}


def build_summary():
    rmap = load_arms()
    meanbase = read_json(os.path.join(PD, "meanbase.json"))
    bench = read_json(os.path.join(PD, "bench.json")) or {}
    init_load = read_json(os.path.join(PD, "init_load_report.json"))
    tm = read_json(os.path.join(TOK, "meta.json")) or {}
    qc = read_json(os.path.join(TOK, "quant_check.json")) or {}

    arms = []
    for k in ARM_ORDER:
        if k not in rmap:
            continue
        arms.append(arm_result(rmap[k], ARM_LABEL[k], init_load))
    arms_by_r = {a["r"]: a for a in arms}
    all_r = rmap.get("all", 5120)
    ref_psnr = (arms_by_r.get(all_r) or {}).get("psnr")
    mb_psnr = (meanbase or {}).get("psnr")
    for a in arms:
        a["delta_vs_all"] = _r(a["psnr"] - ref_psnr) if a.get("psnr") and ref_psnr else None
        a["delta_vs_meanbase"] = (_r(a["psnr"] - mb_psnr)
                                  if a.get("psnr") and mb_psnr else None)

    zs = []
    for k in ["1sigma", "2sigma", "3sigma"]:
        if k in rmap:
            z = zeroshot_entry(rmap[k], ARM_LABEL[k], ref_psnr)
            if z:
                z["delta_vs_meanbase"] = (_r(z["psnr"] - mb_psnr)
                                          if z.get("psnr") and mb_psnr else None)
                zs.append(z)
    if ref_psnr is not None:
        zs.append({"r": int(all_r), "r_label": "全部 (5120)", "cumvar_at_r": 1.0,
                   "eval_json": f"eval_v2_r{all_r}.json", "psnr": ref_psnr,
                   "ssim": (arms_by_r.get(all_r) or {}).get("ssim"),
                   "l1_255": (arms_by_r.get(all_r) or {}).get("l1_255"),
                   "n": (arms_by_r.get(all_r) or {}).get("n"), "delta_vs_all": 0.0,
                   "delta_vs_meanbase": _r(ref_psnr - mb_psnr) if mb_psnr else None,
                   "note": "= all 臂自身评测（零训练截断的右端点，非独立点）"})
    zs.sort(key=lambda x: x["r"])
    # 训练臂 vs 零训练截断（同 r 配对）：直接回答"截断后是否需要重训"
    zmap = {z["r"]: z for z in zs}
    trained_vs_zeroshot = []
    for a in arms:
        z = zmap.get(a["r"])
        if z is None or a.get("psnr") is None or z.get("psnr") is None:
            continue
        trained_vs_zeroshot.append({
            "r": a["r"], "r_label": a["r_label"], "cumvar_at_r": a["cumvar_at_r"],
            "trained_psnr": a["psnr"], "zeroshot_psnr": z["psnr"],
            "delta_trained_minus_zeroshot": _r(a["psnr"] - z["psnr"]),
            "trained_ssim": a.get("ssim"), "zeroshot_ssim": z.get("ssim"),
            "delta_ssim": _r((a["ssim"] or 0) - (z["ssim"] or 0))
            if a.get("ssim") is not None and z.get("ssim") is not None else None,
            "note": ("该 r 臂从零(热启动)重训 vs 用 r=5120 探针零训练直接吃截断输入；"
                     "差值 >0 = 重训有收益"),
        })

    # v1 from-scratch 参照
    scratch = {"all": read_json(os.path.join(PD, "eval_r5120_scratch.json"))}
    scratch["zeroshot"] = {}
    for R in [rmap.get("1sigma"), rmap.get("2sigma"), rmap.get("3sigma")]:
        if R:
            z = read_json(os.path.join(PD, f"eval_zeroshot_r{R}_scratch.json"))
            if z:
                scratch["zeroshot"][str(R)] = {"psnr": _r(z.get("psnr")),
                                               "ssim": _r(z.get("ssim")),
                                               "l1_255": _r(z.get("l1_255"), 3)}
    if scratch.get("all"):
        scratch["all"] = {"psnr": _r(scratch["all"].get("psnr")),
                          "ssim": _r(scratch["all"].get("ssim")),
                          "l1_255": _r(scratch["all"].get("l1_255"), 3),
                          "note": "v1 探针（model_v2.OutputQueryDecoder, 244.69M）从零训练 2 ep"}

    # 训练协议
    a0 = arms[0] if arms else {}
    n_train = tm.get("n_train") or 126689
    train_info = {
        "output_dirs": [f"probe/v2_r{a['r']}" for a in arms],
        "ckpt_dtype": "best.pt = fp16 state_dict；训练中 last.pt=fp32（每臂训完删除）",
        "init": "probe/init_m240_decoder.pt（AssocVTC 240M 档 B 相 t16 ckpt 的 decoder.* 权重；"
                "step 11406, best_psnr 27.81；mem_pos 33→50 行线性插值）",
        "init_missing_keys": (init_load or {}).get("missing"),
        "init_unexpected_keys": (init_load or {}).get("unexpected"),
        "params_M": _r((init_load or {}).get("params_M"), 3),
        "params_breakdown": "真实解码器 237.11M（decoder 递归读出，含 mem_proj/mem_norm/mem_pos/"
                            "query_pos/norm_out/pixel_head）+ mem_proj 输入适配 5120→1440 (+5.58M) = 242.71M",
        "lr": a0.get("lr"), "bs": a0.get("bs"), "warmup": a0.get("warmup"),
        "epochs": a0.get("epochs"), "seed": a0.get("seed"),
        "subset_size": n_train, "img_per_arm": (a0.get("epochs") or 0) * n_train,
        "pixel_space": "DINOv2 归一化（mean=IMAGENET_MEAN×255, std=IMAGENET_STD×255；"
                       "与 240M ckpt 的训练口径一致）；指标在 [0,1] 像素空间算",
        "bench": bench,
        "NOTE": "warm_start_from_AssocVTC_240M_decoder_T16",
    }
    # 每臂 wall
    walls = [a["wall_min"] for a in arms if a.get("wall_min")]
    if walls and a0.get("epochs"):
        train_info["per_arm_wall_min"] = [{"r": a["r"], "wall_min": a["wall_min"]}
                                          for a in arms if a.get("wall_min")]
        train_info["total_train_wall_min"] = _r(sum(walls), 1)
        train_info["estimated_total_with_evals_min"] = _r(sum(walls) + 5 * len(arms), 1)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "Qwen3.8-27B 视觉 token（49×5120, 224²）池化 PCA 截断 × "
                      "AssocVTC 240M 真解码器单步还原探针（热启动，4 臂）",
        "protocol_tag": "warm_start_from_AssocVTC_240M_decoder_T16",
        "metric_protocol": "PSNR/SSIM/L1_255 在 [0,1] 像素空间；SSIM=metrics_linattn.ssim "
                           "(11×11 高斯窗, fp32)；L1_255=[0,1] L1×255；重建用 v2 的 "
                           "patches_to_img01（DINOv2 反归一化, patch=16, 224²）",
        "data": {
            "n_train": n_train, "n_val": tm.get("n_val"),
            "tokens_per_image": tm.get("tokens_per_image"), "dim": tm.get("dim"),
            "resolution": tm.get("resolution"),
            "quant_scheme": tm.get("quant_scheme"),
            "token_psnr_db": _r(qc.get("token_psnr_db"), 2),
            "quant_rel_rmse": qc.get("rel_rmse"),
            "val_tokens_dtype": tm.get("val_tokens_dtype"),
            "model": tm.get("model"),
            "pca": {"fit_on": "IN-100 train 全量池化 token（6,207,761 样本）",
                    "arms_r": rmap, "cumvar_at": {a["r_label"]: a["cumvar_at_r"] for a in arms}},
        },
        "training": train_info,
        "arms": arms,
        "meanbase": meanbase,
        "zeroshot": zs,
        "trained_vs_zeroshot": trained_vs_zeroshot,
        "scratch_ref": scratch,
        "honest_boundaries": [
            "**热启动而非从零训练**：解码器主体来自 AssocVTC 240M 档在**其自身 DINOv2/自建编码器 "
            "token** 上的 B 相 ckpt（step 11406），本次只把 memory 换成 Qwen 视觉 token；"
            "唯一缺失键 = mem_proj（5120→1440，随机初始化，7.37M）",
            "4 臂同 init / 同 seed(0) / 同数据顺序 / 同 epoch 预算 / 同超参，唯一自变量 = PCA 截断维数 r",
            "单种子；非超参搜索（lr/wd/warmup/bs 未调）",
            "PCA 基在**训练集**（126,689×49 池化 token）上拟合 ⇒ 可泛化口径而非 oracle；"
            "val 未参与拟合（train/val 同域 IN-100）",
            "训练输入 int8 B2（token PSNR 46.17 dB, rel_rmse 0.0049；mu_p/sd 由 val 5,000 张 fp16 "
            "估计，仅存储刻度）；val 评测输入 fp16 原值 ⇒ 训练/评测输入精度不完全对称",
            "像素指标（PSNR/SSIM/L1）≠ 任务保真；未测分类/检索/生成下游",
            "PCA 是**方差最优**不是**重建最优**：池化 PC1 占 51.7% 方差且与 token 幅度/norm 相关 −0.999，"
            "r=20（1σ）主要保留幅度方向，内容维度大多在其外（top-49 才 78.3% 方差）⇒ r20 臂差于 "
            "r234/r754 属预期",
            "mem_proj 的 PCA-投影初始化（raw / 白化，把池化 PCA 前 1248 维投到 ckpt 的 1248 维空间）"
            "实测初始 PSNR 10.44/10.49 dB，不如随机初始化 10.77 dB（32 张 val）⇒ 未采用；"
            "ckpt 的 1248 维 token 空间与 Qwen PCA 空间方向不对应",
            "24 位分辨率 224²、patch=16（N=196）与 Qwen 的 7×7 视觉 token 网格非同构，"
            "由解码器自行学习映射",
            "ckpt 落盘 fp16（省盘）：相对 fp32 舍入 1e-3，对 PSNR/SSIM 影响 <0.001 dB 量级",
        ],
        "artifacts": {
            "eval_jsons": sorted(glob.glob(os.path.join(PD, "eval_*.json"))),
            "fig_psnr": ["fig_qwen_probe_v2_psnr_vs_r.png", "fig_qwen_probe_v2_psnr_vs_r.pdf"],
            "fig_grid": "fig_qwen_probe_v2_grid.png",
            "scratch_ref": "eval_r5120_scratch.json + eval_zeroshot_r*_scratch.json（v1 从零 2ep 对照）",
        },
    }
    with open(os.path.join(PD, "summary_v2.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("[summary_v2] wrote", os.path.join(PD, "summary_v2.json"))
    for a in summary["arms"]:
        print(" arm r=%-5d %-12s PSNR %s SSIM %s L1 %s (Δall %s) %s"
              % (a["r"], a["r_label"], a.get("psnr"), a.get("ssim"), a.get("l1_255"),
                 a.get("delta_vs_all"), f"[{a.get('wall_min')} min]" if a.get("wall_min") else ""))
    for z in summary["zeroshot"]:
        print(" zs  r=%-5d %-12s PSNR %s SSIM %s" % (z["r"], z["r_label"], z.get("psnr"),
                                                      z.get("ssim")))
    for t in summary["trained_vs_zeroshot"]:
        print(" paired r=%-5d %-12s trained %s vs zeroshot %s ⇒ Δ=%s dB"
              % (t["r"], t["r_label"], t["trained_psnr"], t["zeroshot_psnr"],
                 t["delta_trained_minus_zeroshot"]))
    if summary["meanbase"]:
        print(" meanbase PSNR %.3f SSIM %.3f L1 %.2f" % (
            summary["meanbase"].get("psnr"), summary["meanbase"].get("ssim"),
            summary["meanbase"].get("l1_255")))
    if scratch.get("all"):
        print(" scratch v1 all PSNR", scratch["all"].get("psnr"))
    return summary


def fig_curve(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    arms = [a for a in summary["arms"] if a.get("psnr") is not None]
    zs = [z for z in summary["zeroshot"] if z.get("psnr") is not None]
    mb = summary.get("meanbase") or {}
    scr = (summary.get("scratch_ref") or {}).get("all")
    if not arms:
        print("[fig] no arm results")
        return
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, key, name in [(axes[0], "psnr", "PSNR (dB)"), (axes[1], "ssim", "SSIM")]:
        arms_s = sorted(arms, key=lambda d: d["r"])
        zs_k = sorted([z for z in zs if z.get(key) is not None], key=lambda d: d["r"])
        ax.set_xscale("log", base=2)
        tick_r = sorted({a["r"] for a in arms_s})
        labs = {}
        for a in arms_s:
            if a["r"] >= 5120:
                labs[a["r"]] = f"all\nr={a['r']}"
            else:
                cv = a.get("cumvar_at_r")
                labs[a["r"]] = (f"{FIG_LABEL.get(a['r_label'], a['r_label'])}\nr={a['r']}\ncum={cv:.3f}"
                                if cv is not None else f"{a['r_label']}\nr={a['r']}")
        ax.set_xticks(tick_r)
        ax.set_xticklabels([labs[t] for t in tick_r], fontsize=8.5)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(min(tick_r) * 0.62, max(tick_r) * 1.55)
        ax.plot([a["r"] for a in arms_s], [a[key] for a in arms_s], "o-",
                color="#1f77b4", lw=2.2, ms=9, zorder=5,
                label="trained arm (warm start from 240M decoder, its own r)")
        for a in arms_s:
            ax.annotate(f'{a[key]:.3f}' if key == "ssim" else f'{a[key]:.2f}',
                        (a["r"], a[key]), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, color="#1f77b4")
        if zs_k:
            zr = zs_k
            ax.plot([z["r"] for z in zr], [z[key] for z in zr], "s--", color="#d62728",
                    lw=2.0, ms=8, zorder=4,
                    label="zero-shot truncation (r=5120 ckpt, truncated input)")
            for z in zr:
                if z["r"] < 5120:
                    ax.annotate(f'{z[key]:.3f}' if key == "ssim" else f'{z[key]:.2f}',
                                (z["r"], z[key]), textcoords="offset points", xytext=(0, -17),
                                ha="center", fontsize=9, color="#d62728")
        if key == "psnr":
            if mb.get("psnr") is not None:
                ax.axhline(mb["psnr"], color="gray", ls=":", lw=2,
                           label=f'mean-image baseline ({mb["psnr"]:.2f} dB)')
            if scr and scr.get("psnr") is not None:
                ax.axhline(scr["psnr"], color="#2ca02c", ls="-.", lw=1.6,
                           label=f'v1 from-scratch 2ep, all ({scr["psnr"]:.2f} dB)')
        else:
            if mb.get("ssim") is not None:
                ax.axhline(mb["ssim"], color="gray", ls=":", lw=2,
                           label=f'mean-image baseline ({mb["ssim"]:.3f})')
            if scr and scr.get("ssim") is not None:
                ax.axhline(scr["ssim"], color="#2ca02c", ls="-.", lw=1.6,
                           label=f'v1 from-scratch 2ep, all ({scr["ssim"]:.3f})')
        ax.set_xlabel("number of retained PCs r  (log2 scale)")
        ax.set_ylabel(name)
        ax.grid(alpha=.3, which="major")
        ax.margins(y=0.14)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Qwen visual tokens (49x5120, 224^2) -> AssocVTC 240M decoder single-step probe "
                 "(warm start, 242.71M): PCA truncation 1σ/2σ/3σ/all", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        p = os.path.join(PD, f"fig_qwen_probe_v2_psnr_vs_r.{ext}")
        fig.savefig(p, dpi=160 if ext == "png" else 300, bbox_inches="tight")
        print("[fig] wrote", p)
    plt.close(fig)


def fig_grid(rmap=None, arm_order=None):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, CODE)
    import qwen_probe_v2 as V2
    import qwen_probe as QP

    rmap = rmap or load_arms()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    order = sorted([k for k in ARM_ORDER if k in rmap], key=lambda k: -rmap[k])
    vt = np.load(os.path.join(TOK, "val_tokens_fp16.npy"), mmap_mode="r")
    vi = np.load(os.path.join(TOK, "val_images_224.npy"), mmap_mode="r")
    N = len(vi)
    idx = np.sort(np.random.default_rng(0).choice(N, size=8, replace=False))
    toks = torch.from_numpy(np.asarray(vt[idx]).astype(np.float32)).to(device)
    imgs = np.asarray(vi[idx])
    gt = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    rows = [("GT", gt)]
    for k in order:
        r = rmap[k]
        ck = os.path.join(PD, f"v2_r{r}", "best.pt")
        if not os.path.exists(ck):
            print("[grid] skip missing", ck)
            continue
        proj = QP.Projector(PCA, r, device)
        model = V2.build_model(device)
        sd = {kk: v.float() for kk, v in
              torch.load(ck, map_location=device, weights_only=False)["model"].items()}
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            t = proj(toks)
            with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else \
                    torch.autocast("cpu", enabled=False):
                pix = model(t)
            rec = V2.patches_to_img01(pix.float()).permute(0, 3, 1, 2).clamp(0, 1).cpu()
        cv = cumvar_at(r)
        lab = "all (5120)" if r >= 5120 else FIG_LABEL.get(ARM_LABEL[k], ARM_LABEL[k])
        rows.append((f"{lab}  r={r}\ncumvar={cv:.4f}" if cv is not None else f"{lab}  r={r}", rec))
        del model
        torch.cuda.empty_cache()
    if len(rows) < 2:
        print("[grid] no ckpt")
        return
    from metrics_linattn import psnr as _psnr
    ps = {}
    for name, im in rows[1:]:
        ps[name] = float(_psnr(im, gt).mean())
    R, C = len(rows), 8
    fig, axes = plt.subplots(R, C, figsize=(C * 2.05, R * 2.35))
    for ri, (name, im) in enumerate(rows):
        for ci in range(C):
            ax = axes[ri, ci]
            ax.imshow(im[ci].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"val[{idx[ci]}]", fontsize=8)
            if ci == 0:
                lbl = name if name == "GT" else f"{name}\nPSNR {ps[name]:.2f} dB"
                ax.set_ylabel(lbl, fontsize=8.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle("Qwen visual tokens -> AssocVTC 240M decoder probe (warm start, 242.71M): "
                 "8 IN-100 val samples; reconstruction via v2 patches_to_img01 "
                 "(DINOv2 de-normalization: x*IMAGENET_STD*255 + IMAGENET_MEAN*255, then [0,1])",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(PD, "fig_qwen_probe_v2_grid.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("[fig] wrote", p)
    plt.close(fig)
    arr = np.stack([im.numpy() for _, im in rows], axis=1)
    np.save(os.path.join(PD, "grid_v2_recon_8.npy"), arr)
    json.dump({"indices": idx.tolist(), "rows": [n for n, _ in rows], "psnr8": ps},
              open(os.path.join(PD, "grid_v2_meta.json"), "w"), indent=1)
    print("[grid] wrote grid_v2_recon_8.npy / grid_v2_meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["summary", "grid", "all"], default="all")
    a = ap.parse_args()
    if a.mode in ("summary", "all"):
        s = build_summary()
        fig_curve(s)
    if a.mode in ("grid", "all"):
        fig_grid()


if __name__ == "__main__":
    main()
