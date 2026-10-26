#!/usr/bin/env python3
"""容量阶梯 × SAVOIAS OOD 复杂度分析（2026-10-26）
=====================================================

输入（默认 `doc/2026-10-26/artifacts/ood_savoias/`）
--------------------------------------------------
  per_image_<tag>.json      `eval_stepwise_linattn.py --dump_per_image` 在 SAVOIAS
                            （820 张 = Art 420 + Scenes 200 + Objects 200, T=16）上的产物
  stepwise_<tag>.json       `eval_stepwise_linattn.py --out` 的逐步聚合
  per_image_ssim_<tag>.json 可选的补充 SSIM pass（ssim_pass.py；逐图逐步 SSIM）
  gt_{art,scenes,objects}.npy  SAVOIAS 官方人评复杂度 {index: [filename, score 0-100]}
                            —— 不随仓库分发，用 `tools/fetch_savoias.sh` 下载
                            （`--gt-dir`，默认 /root/autodl-tmp/savoias/gt）

输出（同名目录）
----------------
  ood_capacity_vs_quality.png/pdf  ① 容量 vs OOD/ID 末步 L1_255 / PSNR / SSIM
  ood_complexity_rho_tercile.png/pdf ② 每模型 ρ(复杂度, L1) 与三分位差
  ood_l1_vs_complexity_art.png/pdf ③ 复杂度–L1 散点/分箱（Art 420 张, 5 模型）
  ood_step_curves.png/pdf          ④ 逐步曲线（5 模型 × OOD 平均）
  summary.json                     全部数字（主表 / ID vs OOD / ρ / 三分位）
  per_image_joined.csv             逐图 join 后的原始数值

用法
----
  python3 tools/plot_savoias_scale.py
  python3 tools/plot_savoias_scale.py --data-dir <artifacts> --gt-dir <savoias>/gt
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "doc/2026-10-26/artifacts/ood_savoias"
DEFAULT_GT = Path("/root/autodl-tmp/savoias/gt")
ID_SCALE_DIR = REPO / "doc/2026-10-24/artifacts/scale/eval"
ID_BASE_JSON = REPO / "doc/2026-10-23/artifacts/n2bigstep/eval/stepwise_k32_t16.json"

# 容量档（tag → 总参数量 M, 显示名）。数值来自 doc/2026-10-24/REPORT_scale_ladder.md
TAGS = [("base63", 63.0, "63M (base)"),
        ("m60", 119.3, "119M"),
        ("m120", 238.2, "238M"),
        ("m240", 481.5, "482M"),
        ("m360", 720.6, "721M")]
TAG2M = {t: m for t, m, _ in TAGS}
TAG2LABEL = {t: lab for t, _, lab in TAGS}
COLORS = {"base63": "#7f7f7f", "m60": "#1f77b4", "m120": "#2ca02c",
          "m240": "#d62728", "m360": "#9467bd"}
SUBSETS = [("Art", "People-Art", "gt_art.npy"),
           ("Scenes", "Places", "gt_scenes.npy"),
           ("Objects", "MSCOCO", "gt_objects.npy")]

# ID 参考（同 T=16 / n=2 / K=32，ImageNet-100 val 5000 张，末步）
ID_STEPWISE = {"base63": ID_BASE_JSON,
               "m60": ID_SCALE_DIR / "stepwise_m60_t16.json",
               "m120": ID_SCALE_DIR / "stepwise_m120_t16.json",
               "m240": ID_SCALE_DIR / "stepwise_m240_t16.json",
               "m360": ID_SCALE_DIR / "stepwise_m360_t16.json"}
ID_EVAL = {"base63": REPO / "doc/2026-10-23/artifacts/n2bigstep/eval/linattn_k32_t16.json",
           "m60": ID_SCALE_DIR / "linattn_m60_t16.json",
           "m120": ID_SCALE_DIR / "linattn_m120_t16.json",
           "m240": ID_SCALE_DIR / "linattn_m240_t16.json",
           "m360": ID_SCALE_DIR / "linattn_m360_t16.json"}


# ─── 数据加载 ─────────────────────────────────────────────────────────────
def load_scores(gt_dir, fn):
    """SAVOIAS ground truth: {index: [filename, human_complexity_0_100]} → {int: score}"""
    path = Path(gt_dir) / fn
    o = np.load(path, allow_pickle=True).item()
    return {int(k): int(o[k][1]) for k in o}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_ssim(art_dir, tag):
    """可选补充 SSIM pass；返回 {file_name: 末步 SSIM} 或 None。"""
    p = Path(art_dir) / f"per_image_ssim_{tag}.json"
    if not p.exists():
        return None
    d = load_json(p)
    # 注意: 三个子集的文件名互相重名（Art/0.jpg 与 Scenes/0.jpg）, 必须用 path 做键
    return {r["path"]: float(r["SSIM"][-1]) for r in d["images"]}


def join(per_image, subset, scores, ssim_map=None):
    """per_image['images'] 按子集筛出并 join 复杂度（index = 文件名 stem）。"""
    S, L, P, B, SS, IDX = [], [], [], [], [], []
    for r in per_image["images"]:
        if r["category"] != subset:
            continue
        stem = int(Path(r["file_name"]).stem)
        if stem not in scores:
            continue
        S.append(scores[stem])
        L.append(r["L1_255"])
        P.append(r["PSNR"])
        B.append(r["best_step"])
        SS.append(ssim_map.get(r["path"], np.nan) if ssim_map else np.nan)
        IDX.append(stem)
    n_steps = len(per_image["aggregate_L1_255"])
    out = {"S": np.asarray(S, float),
           "L": np.asarray(L, float).reshape(len(L), n_steps) if L else np.zeros((0, n_steps)),
           "P": np.asarray(P, float).reshape(len(P), n_steps) if P else np.zeros((0, n_steps)),
           "B": np.asarray(B, int), "SS": np.asarray(SS, float),
           "IDX": np.asarray(IDX, int)}
    return out


def qbins(x, k=6):
    qs = np.quantile(x, np.linspace(0, 1, k + 1))
    qs[0] -= 1e-9
    return np.clip(np.digitize(x, qs) - 1, 0, k - 1)


def binned(x, y, k=6):
    idx = qbins(x, k)
    xs, ys, es = [], [], []
    for b in range(k):
        m = idx == b
        if not m.any():
            continue
        xs.append(x[m].mean())
        ys.append(y[m].mean())
        es.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
    return np.array(xs), np.array(ys), np.array(es)


def terciles(S, L_last):
    """复杂度三分位：low/mid/high 的末步 L1 均值与 high-low 差距。"""
    q1, q2 = np.percentile(S, [100 / 3, 200 / 3])
    out = {"q1": float(q1), "q2": float(q2),
           "n": {"low": int((S <= q1).sum()),
                 "mid": int(((S > q1) & (S <= q2)).sum()),
                 "high": int((S > q2).sum())}}
    for name, m in [("low", S <= q1), ("mid", (S > q1) & (S <= q2)),
                    ("high", S > q2)]:
        out[name] = float(L_last[m].mean()) if m.any() else float("nan")
    out["gap_high_low"] = out["high"] - out["low"]
    out["gap_rel_pct"] = (out["gap_high_low"] / out["low"] * 100
                          if out["low"] else float("nan"))
    out["q1_tercile_mean"] = float(S[S <= q1].mean())
    out["q3_tercile_mean"] = float(S[S > q2].mean())
    return out


# ─── 主流程 ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--gt-dir", default=str(DEFAULT_GT))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-csv", action="store_true")
    a = ap.parse_args()
    art_dir = Path(a.data_dir)
    out = Path(a.out_dir) if a.out_dir else art_dir
    out.mkdir(parents=True, exist_ok=True)

    scores = {sub: load_scores(a.gt_dir, fn) for sub, _, fn in SUBSETS}
    ssim_maps = {tag: load_ssim(art_dir, tag) for tag, _, _ in TAGS}

    # ── ID 参考 ──
    id_ref = {}
    for tag, _, _ in TAGS:
        sw = load_json(ID_STEPWISE[tag])
        ev = load_json(ID_EVAL[tag])
        ev = list(ev.values())[0] if isinstance(ev, dict) and "psnr" not in ev else ev
        id_ref[tag] = {"L1_255": float(sw["L1_255"][-1]),
                       "PSNR": float(sw["PSNR"][-1]),
                       "SSIM": float(ev.get("ssim", np.nan)),
                       "n_images": int(sw["n_images"]), "T": int(sw["T"])}

    summary = {
        "note": "SAVOIAS (human complexity, OOD for ImageNet-100 models) x capacity ladder "
                "(encoder/decoder each 60/120/240/360M, n=2 tokens/step, T=16, K=32). "
                "L1_255 = 0-255 pixel L1 at last step (T=16), PSNR per-image mean.",
        "tags": [t for t, _, _ in TAGS],
        "params_M": TAG2M,
        "subsets": {sub: {"pretty": pretty, "n_scored": len(scores[sub])}
                    for sub, pretty, _ in SUBSETS},
        "id_reference": id_ref,
        "models": {}, "complexity": {}, "figures": []}
    csv_rows = []

    for tag, _, label in TAGS:
        sw = load_json(art_dir / f"stepwise_{tag}.json")
        pi = load_json(art_dir / f"per_image_{tag}.json")
        T = len(sw["L1_255"])
        entry = {"label": label, "params_M": TAG2M[tag], "ckpt": sw.get("ckpt"),
                 "T": T, "steps": sw.get("steps"), "n_images": sw["n_images"],
                 "K": sw.get("K"),
                 "OOD": {"L1_255": [float(v) for v in sw["L1_255"]],
                         "PSNR": [float(v) for v in sw["PSNR"]],
                         "L1_255_last": float(sw["L1_255"][-1]),
                         "PSNR_last": float(sw["PSNR"][-1]),
                         "best_step_1based": int(sw["best_step"]) + 1,
                         "subsets": {}},
                 "subsets": {}}
        if ssim_maps[tag] is not None:
            sm = load_ssim(art_dir, tag)
            ss_last = np.array([sm[r["path"]] for r in pi["images"]
                                if r["path"] in sm])
            entry["OOD"]["SSIM_last"] = float(ss_last.mean())
        else:
            entry["OOD"]["SSIM_last"] = float("nan")

        # 子集指标（每图末步再平均）
        for sub, pretty, _ in SUBSETS:
            j = join(pi, sub, scores[sub], ssim_maps[tag])
            sub_entry = {"pretty": pretty, "n": int(len(j["S"])),
                         "L1_255_last": float(j["L"][:, -1].mean()),
                         "PSNR_last": float(j["P"][:, -1].mean()),
                         "SSIM_last": (float(j["SS"][~np.isnan(j["SS"])].mean())
                                       if np.isfinite(j["SS"]).any() else float("nan")),
                         "L1_255_curve": [float(v) for v in j["L"].mean(0)],
                         "PSNR_curve": [float(v) for v in j["P"].mean(0)],
                         "complexity_mean": float(j["S"].mean()),
                         "complexity_std": float(j["S"].std())}
            entry["subsets"][sub] = sub_entry

            # 复杂度相关性 / 三分位（逐子集）
            rho_l1 = float(spearmanr(j["S"], j["L"][:, -1]).correlation)
            rho_psnr = float(spearmanr(j["S"], j["P"][:, -1]).correlation)
            ter = terciles(j["S"], j["L"][:, -1])
            entry.setdefault("complexity", {})[sub] = {
                "pretty": pretty, "n": int(len(j["S"])),
                "rho_l1": rho_l1, "rho_psnr": rho_psnr, "terciles": ter,
                "rho_l1_per_step": [float(spearmanr(j["S"], j["L"][:, t]).correlation)
                                    for t in range(T)]}
            if not a.no_csv:
                for i in range(len(j["S"])):
                    row = {"tag": tag, "params_M": TAG2M[tag], "subset": sub,
                           "index": int(j["IDX"][i]), "complexity": int(j["S"][i]),
                           "best_step_1based": int(j["B"][i]) + 1,
                           "SSIM_last": (round(float(j["SS"][i]), 4)
                                         if np.isfinite(j["SS"][i]) else "")}
                    for t in range(T):
                        row[f"L1_255_t{t + 1}"] = round(float(j["L"][i, t]), 4)
                        row[f"PSNR_t{t + 1}"] = round(float(j["P"][i, t]), 4)
                    csv_rows.append(row)

        # ID vs OOD
        entry["ID"] = id_ref[tag]
        entry["OOD_minus_ID"] = {
            "L1_255": entry["OOD"]["L1_255_last"] - id_ref[tag]["L1_255"],
            "PSNR": entry["OOD"]["PSNR_last"] - id_ref[tag]["PSNR"],
            "SSIM": (entry["OOD"]["SSIM_last"] - id_ref[tag]["SSIM"]
                     if np.isfinite(entry["OOD"]["SSIM_last"])
                     and np.isfinite(id_ref[tag]["SSIM"]) else float("nan"))}
        summary["models"][tag] = entry
        summary["complexity"][tag] = entry["complexity"]
        print(f"[{tag}] OOD L1={entry['OOD']['L1_255_last']:.3f} "
              f"PSNR={entry['OOD']['PSNR_last']:.3f} "
              f"SSIM={entry['OOD']['SSIM_last']:.4f} | "
              f"ID L1={id_ref[tag]['L1_255']:.3f} "
              f"ΔL1={entry['OOD_minus_ID']['L1_255']:+.3f}px "
              f"ΔPSNR={entry['OOD_minus_ID']['PSNR']:+.3f}dB")
        for sub, pretty, _ in SUBSETS:
            c = entry["complexity"][sub]
            print(f"    {pretty:11s} n={c['n']:4d} L1={entry['subsets'][sub]['L1_255_last']:7.3f} "
                  f"ρ={c['rho_l1']:+.3f} gap(high-low)={c['terciles']['gap_high_low']:+.3f}px")

    # ── 图 ①：容量 vs OOD/ID 质量 ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    xs = [TAG2M[t] for t, _, _ in TAGS]
    panels = [("L1_255_last", "final pixel L1 (0–255) ↓", "L1_255"),
              ("PSNR_last", "final PSNR (dB) ↑", "PSNR"),
              ("SSIM_last", "final SSIM ↑", "SSIM")]
    sub_styles = {"Art": "--s", "Scenes": "--^", "Objects": "--D"}
    for ax, (key, ylab, iso) in zip(axes, panels):
        for sub, pretty, _ in SUBSETS:
            ys = [summary["models"][t]["subsets"][sub][key] for t, _, _ in TAGS]
            ax.plot(xs, ys, sub_styles[sub], color="#a0a0a0", ms=4, lw=1.2,
                    label=f"OOD {sub}")
        ood = [summary["models"][t]["OOD"][key] for t, _, _ in TAGS]
        ax.plot(xs, ood, "-o", color="#d62728", ms=6, lw=2.2, label="OOD full (820)")
        idv = [summary["models"][t]["ID"][iso] for t, _, _ in TAGS]
        ax.plot(xs, idv, "-o", color="#1f4e9c", ms=5, lw=1.8, label="ID (IN-100 val)")
        ax.set_xlabel("total parameters (M)", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.0f}" for x in xs], fontsize=7.5)
        ax.grid(alpha=.28)
        ax.legend(fontsize=7.4, loc="best")
    fig.suptitle("Capacity ladder vs reconstruction quality: SAVOIAS OOD (820) vs "
                 "ImageNet-100 ID (5,000 val); n=2 tok/step, T=16, K=32", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "ood_capacity_vs_quality.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "ood_capacity_vs_quality.pdf", bbox_inches="tight")
    plt.close(fig)
    summary["figures"].append("ood_capacity_vs_quality.png")

    # ── 图 ②：复杂度的 ρ 与三分位差（柱状, 5 模型 × 4 组）───────────────
    groups = [("Full(820)", "full")] + [(p, s) for s, p, _ in SUBSETS]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    gw = 0.8 / len(groups)
    for ax, metric, ylab in [
            (axes[0], "rho", "Spearman ρ (complexity, final L1)"),
            (axes[1], "gap", "high−low complexity tercile L1 gap (px)"),
            (axes[2], "rel", "high−low gap / low-tercile L1 (%)")]:
        for gi, (glabel, gkey) in enumerate(groups):
            vals = []
            for tag, _, _ in TAGS:
                if gkey == "full":
                    rhos, gaps, rels = [], [], []
                    for sub, _, _ in SUBSETS:
                        c = summary["complexity"][tag][sub]
                        rhos.append(c["rho_l1"])
                        gaps.append(c["terciles"]["gap_high_low"])
                        rels.append(c["terciles"]["gap_rel_pct"])
                    vals.append(np.mean(rhos) if metric == "rho"
                                else np.mean(gaps) if metric == "gap"
                                else np.mean(rels))
                else:
                    c = summary["complexity"][tag][gkey]
                    vals.append(c["rho_l1"] if metric == "rho" else
                                c["terciles"]["gap_high_low"] if metric == "gap" else
                                c["terciles"]["gap_rel_pct"])
            xpos = np.arange(len(TAGS)) + (gi - (len(groups) - 1) / 2) * gw
            ax.bar(xpos, vals, width=gw * 0.92, label=glabel,
                   color=plt.cm.viridis(gi / max(1, len(groups) - 1)),
                   edgecolor="white", lw=.4)
        ax.set_xticks(np.arange(len(TAGS)))
        ax.set_xticklabels([TAG2LABEL[t] for t, _, _ in TAGS], fontsize=7.5)
        ax.set_xlabel("capacity (total params)", fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(alpha=.25, axis="y")
        ax.axhline(0, color="k", lw=.6)
    axes[0].legend(fontsize=7.2, ncol=2, loc="best")
    fig.suptitle("Does capacity shrink the complexity penalty? "
                 "SAVOIAS human complexity vs final pixel L1 (per model)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "ood_complexity_rho_tercile.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "ood_complexity_rho_tercile.pdf", bbox_inches="tight")
    plt.close(fig)
    summary["figures"].append("ood_complexity_rho_tercile.png")

    # ── 图 ③：复杂度–L1 散点/分箱（Art 420 张, 5 模型）────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    art_all = None
    for tag, _, label in TAGS:
        pi = load_json(art_dir / f"per_image_{tag}.json")
        j = join(pi, "Art", scores["Art"], ssim_maps[tag])
        if art_all is None:
            art_all = j["S"]
            ax.scatter(j["S"], j["L"][:, -1], s=6, color="#c8c8c8", alpha=.45,
                       zorder=1, label="per-image (n=420)")
    for tag, _, label in TAGS:
        pi = load_json(art_dir / f"per_image_{tag}.json")
        j = join(pi, "Art", scores["Art"], ssim_maps[tag])
        bx, by, be = binned(j["S"], j["L"][:, -1], k=6)
        ax.errorbar(bx, by, yerr=be, fmt="-o", color=COLORS[tag], ms=4.5, lw=1.8,
                    capsize=2, label=f"{label}")
    ax.set_xlabel("SAVOIAS human-annotated complexity (0–100)", fontsize=10.5)
    ax.set_ylabel("final pixel L1 (0–255)", fontsize=10.5)
    ax.set_title("People-Art (420): more complex ⇒ worse reconstruction; "
                 "capacity shifts the curve down but does not flatten it", fontsize=9.5)
    ax.grid(alpha=.28)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ood_l1_vs_complexity_art.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "ood_l1_vs_complexity_art.pdf", bbox_inches="tight")
    plt.close(fig)
    summary["figures"].append("ood_l1_vs_complexity_art.png")

    # ── 图 ④：逐步曲线（5 模型 × OOD 全量平均）────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for tag, _, label in TAGS:
        sw = summary["models"][tag]["OOD"]
        steps = np.arange(1, len(sw["L1_255"]) + 1)
        axes[0].plot(steps, sw["L1_255"], "-o", color=COLORS[tag], ms=3.5, lw=1.7,
                     label=f"{label} (final {sw['L1_255_last']:.2f} px)")
        axes[1].plot(steps, sw["PSNR"], "-o", color=COLORS[tag], ms=3.5, lw=1.7,
                     label=f"{label} (final {sw['PSNR_last']:.2f} dB)")
    T = len(summary["models"][TAGS[0][0]]["OOD"]["L1_255"])
    for ax, ylab in [(axes[0], "mean pixel L1 (0–255)"), (axes[1], "mean PSNR (dB)")]:
        ax.set_xlabel("decoding step $t$ (2 z_s per step, K=32)", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.set_xticks(range(1, T + 1))
        ax.grid(alpha=.28)
        ax.legend(fontsize=7.6)
    axes[0].set_title("SAVOIAS OOD (820 images) — stepwise L1", fontsize=10)
    axes[1].set_title("SAVOIAS OOD (820 images) — stepwise PSNR", fontsize=10)
    fig.suptitle("Stepwise curves, capacity ladder (n=2 tok/step, T=16, K=32)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "ood_step_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "ood_step_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    summary["figures"].append("ood_step_curves.png")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    if csv_rows and not a.no_csv:
        fields = list(csv_rows[0].keys())
        with open(out / "per_image_joined.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
    print(f"[done] {out}  ({len(csv_rows)} csv rows)")


if __name__ == "__main__":
    main()
