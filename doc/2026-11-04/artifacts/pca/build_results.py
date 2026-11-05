#!/usr/bin/env python
"""build_results.py — 把 pca_448_baseline.py 的多次运行合并成交付 json + 打印 md 表格

输入（同目录，均为 pca_448_baseline.py 产物）：
  pca_448_ranks.json                       448²/p32 正式基线（全量 train 拟合基, r=1..40）
  pca_448_sens_5000.json / _20000.json     448 基拟合集大小敏感性
  pca_224_selfcheck_refit400.json          224²/p16 自洽（重拟合 400 张基）
  pca_224_selfcheck_stale_artifact_basis.json  224 自洽（用仓库 artifacts 的 pca_basis.pt）
  pca_224_sens_5000.json / _20000.json     224 基拟合集大小敏感性
输出：
  pca_448_results.json（交付）+ stdout 上的 markdown 表格
"""
from __future__ import annotations

import json
import os

D = os.path.dirname(os.path.abspath(__file__))


def L(name):
    with open(os.path.join(D, name), "r", encoding="utf-8") as f:
        return json.load(f)


MODEL_JSON_REMOTE = "/root/autodl-tmp/exp448/eval/stage3_stepwise.json"
# 本地默认（fetch_448_960m.sh 拉回后的路径）；可用环境变量 MODEL_JSON 覆盖
MODEL_JSON_LOCAL = "/home/linaro/dsh/exp448_bench/artifacts/eval/stage3_stepwise.json"
PAPER_224 = L("pca_224_selfcheck_refit400.json")["paper_224_check"]["paper_row_psnr"]
# 论文 v2.0 表 2「本文」行（224²/p16/240M×2, zs_dim=224 的 16 档 PSNR）
# 来源: /userdata/paper/pp_zh/sec4_exp.tex tab:psnr（预算 448…7168）
PAPER_224_MODEL_PSNR = [21.03, 22.04, 22.91, 23.66, 24.19, 24.51, 24.86, 25.21,
                        25.50, 25.69, 25.75, 25.92, 25.93, 26.02, 26.08, 26.08]


def model_json_path():
    for c in (os.environ.get("MODEL_JSON"), MODEL_JSON_LOCAL, MODEL_JSON_REMOTE):
        if c and os.path.isfile(c):
            return c
    return MODEL_JSON_LOCAL


def fill_model(rows, path):
    """本地把主线 S3 逐步 eval 填进 16 行模型列（不重跑 PCA，只做同值数对齐差值）。"""
    if not (path and os.path.isfile(path)):
        return None, f"模型列 json 不存在: {path}"
    m = json.load(open(path, encoding="utf-8"))
    ps = [float(v) for v in (m.get("PSNR") or m.get("psnr") or [])]
    l1 = m.get("L1_255") or m.get("l1_255")
    l1 = [float(v) for v in l1] if l1 else None
    if len(ps) < len(rows):
        return None, f"{path} 只有 {len(ps)} 档 PSNR（期望 {len(rows)}）"
    for row in rows:
        t = row["t"]
        row["model_psnr"] = round(ps[t - 1], 4)
        row["model_l1_255"] = round(l1[t - 1], 4) if l1 else None
        row["delta_psnr_model_minus_pca_interp"] = round(
            row["model_psnr"] - row["pca_psnr_interp"], 4)
    return m, "filled"


def main():
    d448 = L("pca_448_ranks.json")
    s448_5k = L("pca_448_sens_5000.json")
    s448_20k = L("pca_448_sens_20000.json")
    c224 = L("pca_224_selfcheck_refit400.json")
    c224_stale = L("pca_224_selfcheck_stale_artifact_basis.json")
    s224_5k = L("pca_224_sens_5000.json")
    s224_20k = L("pca_224_sens_20000.json")

    pr448 = d448["metrics"]["per_rank"]
    rows448 = d448["same_value_table"]
    mjson = model_json_path()
    model, model_status = fill_model(rows448, mjson)
    pr224 = c224["metrics"]["per_rank"]
    rows224 = c224["same_value_table"]

    # 448 vs 224 在同一个「每图连续值数」上的逐块 PCA 对照（两者都是训练集共享基）
    cross = []
    for r448, r224 in zip(rows448, rows224):
        assert r448["model_values"] == r224["model_values"]
        cross.append({
            "values_per_image": r448["model_values"],
            "pca448_psnr_interp": r448["pca_psnr_interp"],
            "pca224_psnr_interp": r224["pca_psnr_interp"],
            "delta_448_minus_224_dB": round(r448["pca_psnr_interp"]
                                            - r224["pca_psnr_interp"], 4),
            "pca448_l1_255_interp": r448["pca_l1_255_interp"],
            "pca224_l1_255_interp": r224["pca_l1_255_interp"],
        })

    # 448 基拟合集大小敏感性：与全量基（正式结果）逐 rank 比 PSNR
    def sens(base):
        pr = base["metrics"]["per_rank"]
        diffs = {r: round(pr448[r]["psnr"] - pr[r]["psnr"], 5) for r in pr448}
        mx = max(diffs, key=lambda r: abs(diffs[r]))
        return {"n_basis_images": base["basis"]["n_basis_images"],
                "n_patches": base["basis"]["n_patches"],
                "fit_seconds": base["basis"]["fit_seconds"],
                "max_abs_psnr_diff_vs_fulltrain": round(abs(diffs[mx]), 5),
                "argmax_rank": int(mx),
                "psnr_diff_per_rank": diffs}

    out = {
        "meta": {
            "purpose": "论文表 2 式「同值数 PSNR：本文 vs 逐块 PCA」的 448 扩展基线；"
                       "另含 224 自洽校验（与论文表 2 PCA 行 18.40…26.25 dB 对照）",
            "script": "AssocVTC/tools/pca_448_baseline.py（副本在 exp448_bench/pca/）",
            "created": d448["created"],
            "repo": d448["repo"],
            "protocol": {
                "data": "/root/autodl-tmp/datasets/imagenet100, val=5000（每类等距 50，"
                        "build_datasets(data_dir, img_size)，与主线同划分）",
                "pca_basis": "per-patch PCA、**训练集共享基、非逐图 oracle**、0–255 patch 空间、"
                             "协方差精确 eigh（float64 累加）",
                "values": "PCA 每图 = N×r（N=196，r=1..40）；模型 S3 臂每图 = 2t×224 个值"
                          "（t=1..16 ⇒ 448…7168），按 196r 对齐",
                "psnr": "[0,1] clamp、metrics_linattn.psnr 逐图再平均",
                "ssim": "metrics_linattn.ssim（11×11 高斯窗, 裁边界, 通道均值）, 输入 [0,1]",
                "l1_255": "0–255 空间、不 clamp、逐图再平均（eval_stepwise_linattn 口径）",
                "gpu": "GPU1（CUDA_VISIBLE_DEVICES=1）; 拟合峰值显存 ~575 MiB（batch 32, "
                       "实测 nvidia-smi）；GPU0 主线训练全程未触碰",
            },
        },
        "results_448": {
            "config": d448["config"],
            "basis": d448["basis"],
            "per_rank": pr448,
            "same_value_table": rows448,
            "timing": d448["timing"],
            "model_column": ({
                "status": "filled",
                "source": mjson,
                "ckpt": model.get("ckpt"),
                "step": model.get("step"),
                "n_images": model.get("n_images"),
                "T": model.get("T"), "K": model.get("K"),
                "psnr": model.get("PSNR") or model.get("psnr"),
                "l1_255": model.get("L1_255") or model.get("l1_255"),
                "note": "模型列 = 主线 S3 臂 448²/p32/960M/zs_dim224 的逐档 eval；"
                        "模型值数 = 2t×224，与 PCA 侧 196r 按「同值数」对齐（Δ = 模型 − PCA 插值）",
            } if model else {
                "status": "TODO",
                "expected_path": mjson,
                "note": f"读模型列失败: {model_status}",
            }),
        },
        "baseline_224_selfcheck": {
            "paper_row_psnr": PAPER_224,
            "refit_basis_images_400": {
                "vs_paper_max_abs_diff_dB": c224["paper_224_check"]["max_abs_diff"],
                "vs_paper_mean_abs_diff_dB": c224["paper_224_check"]["mean_abs_diff"],
                "pass_0p5dB": c224["paper_224_check"]["pass_0p5dB"],
                "diff_mine_minus_paper": c224["paper_224_check"]["diff_mine_minus_paper"],
                "mine_psnr_interp": c224["paper_224_check"]["mine_psnr_interp"],
                "vs_repo_pca_allranks_max_abs_psnr_diff": c224["compare_per_rank"]["max_abs_psnr_diff"],
                "vs_repo_pca_allranks_max_abs_l1_diff": c224["compare_per_rank"]["max_abs_l1_diff"],
                "n_ranks_compared": c224["compare_per_rank"]["n_ranks_compared"],
                "basis": c224["basis"],
                "per_rank": pr224,
                "timing": c224["timing"],
            },
            "stale_artifact_basis_note": {
                "basis_pt": "AssocVTC/doc/2026-10-26/artifacts/rfid_m240_pca/pca_basis.pt",
                "vs_paper_max_abs_diff_dB": c224_stale["paper_224_check"]["max_abs_diff"],
                "note": "仓库里的 pca_basis.pt 与重拟合 400 张基等效（同样复现论文行/ "
                        "pca_allranks，max|Δ|=0.005 dB）⇒ 口径链路二次确认。早先用 "
                        "--max_images 200 冒烟时看到 2.6 dB 偏差，根因是那 200 张（前 4 类）"
                        "内容偏难（r=40 时 PSNR 23.98 vs 全量 26.57），不是基线 bug；"
                        "正式数字一律用全量 5000 张 val。",
            },
            "basis_size_sensitivity": {
                "400": {"vs_paper_max_abs_diff_dB": c224["paper_224_check"]["max_abs_diff"]},
                "5000": {"vs_paper_max_abs_diff_dB": s224_5k["paper_224_check"]["max_abs_diff"]},
                "20000": {"vs_paper_max_abs_diff_dB": s224_20k["paper_224_check"]["max_abs_diff"]},
                "note": "论文 PCA 行本身用 400 张（eval_rfid_pca 默认）；基拟合集越大，相对论文行"
                        "只差 ~0.05 dB ⇒ 基拟合集大小对曲线影响可忽略（≤0.05 dB @ 224）",
            },
        },
        "basis_size_sensitivity_448": {
            "official_full_train": {
                "n_basis_images": d448["basis"]["n_basis_images"],
                "n_patches": d448["basis"]["n_patches"],
            },
            "5000": sens(s448_5k),
            "20000": sens(s448_20k),
            "note": "448 正式结果用**全量 train（121,689 张 / 23.85M patch）**拟合；"
                    "5k / 20k 子采样基与全量基相比，r=1..40 全档 PSNR 最大差 ≤0.006 dB ⇒ "
                    "本任务用全量而非子采样，子采样会损失的精度可忽略",
        },
        "compare_448_vs_224_same_values": cross,
        "limitations": [
            "PCA 基在 train 上拟合、val 上评测（非逐图 oracle，也非 transductive）；"
            "与论文表 2 的 PCA 行同族口径",
            "PCA 保留 N=196 个空间位置的码（每位置 r 个系数），模型把整图路由进 32 个 token；"
            "值数相同 ≠ 序列长度/结构相同——这是论文「同值数」对照的既定口径，不是 bug",
            "448 的 PCA 每图值数 N×r 与 224 相同（N 都是 196），但每 patch 覆盖 4× 像素"
            "（3072 维 vs 768 维）⇒ 同值数下 448 的 PCA 重建明显更弱（见 "
            "compare_448_vs_224_same_values，满预算 7168 值差 −1.06 dB）",
            "模型列已填（S3 臂 448²/p32/960M 逐档 eval）；模型 SSIM 在 eval json 里未记录，"
            "故本表只给 PSNR/L1_255 两列",
            "单种子；像素指标 ≠ 任务保真；SSIM 在主报告里只用于末步/汇总，本表为逐 rank 口径",
        ],
        "repro_commands": [
            "# 224 自洽（重拟合 400 张基, 对照论文表 2 PCA 行）",
            "CUDA_VISIBLE_DEVICES=1 ASSOCVTC_REPO=<repo> python pca_448_baseline.py "
            "--img_size 224 --patch 16 --basis_images 400 --ranks 1-40 "
            "--save_basis pca_basis_224_refit400.pt --out pca_224_selfcheck_refit400.json "
            "--compare_per_rank_json pca_allranks_224.json",
            "# 448 正式（全量 train 拟合 3072 维基, r=1..40, 5000 val）",
            "CUDA_VISIBLE_DEVICES=1 ASSOCVTC_REPO=<repo> python pca_448_baseline.py "
            "--img_size 448 --patch 32 --basis_images 0 --ranks 1-40 --batch_size 32 "
            "--num_workers 24 --save_basis pca_basis_448_fulltrain.pt --out pca_448_ranks.json",
            "# 模型列（S3 eval 产出后）",
            "CUDA_VISIBLE_DEVICES=1 ASSOCVTC_REPO=<repo> python pca_448_baseline.py "
            "--img_size 448 --patch 32 --basis_pt pca_basis_448_fulltrain.pt --ranks 1-40 "
            "--model_json /root/autodl-tmp/exp448/eval/stage3_stepwise.json "
            "--out pca_448_ranks.json",
        ],
    }
    with open(os.path.join(D, "pca_448_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"WROTE {os.path.join(D, 'pca_448_results.json')}")

    # ── markdown 表 1：448 每 r ──
    print("\n<!-- 448 per-rank -->")
    print("| r | 每图值数 (196r) | PSNR / dB | SSIM | L1_255 |")
    print("|---|---|---|---|---|")
    for r in range(1, 41):
        d = pr448[str(r)]
        print(f"| {r} | {d['values_per_image']} | {d['psnr']:.3f} | "
              f"{d['ssim']:.4f} | {d['l1_255']:.3f} |")

    # ── markdown 表 2：448 16 行同值数对照 ──
    print("\n<!-- 448 same-value 16 rows -->")
    print("| t | 模型值数 (2t×224) | PCA 最近 rank | PCA 实际值数 | 值数差 | "
          "PCA PSNR 最近rank | PCA PSNR 插值 | PCA SSIM 插值 | PCA L1_255 插值 | 模型 PSNR | Δ(模型−PCA) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows448:
        model = "TODO" if row["model_psnr"] is None else f"{row['model_psnr']:.3f}"
        delta = "TODO" if row["delta_psnr_model_minus_pca_interp"] is None \
            else f"{row['delta_psnr_model_minus_pca_interp']:+.3f}"
        print(f"| {row['t']} | {row['model_values']} | {row['pca_rank_nearest']} | "
              f"{row['pca_values_nearest']} | {row['pca_values_rel_diff_pct']:+.2f}% | "
              f"{row['pca_psnr_nearest']:.3f} | {row['pca_psnr_interp']:.3f} | "
              f"{row['pca_ssim_interp']:.4f} | {row['pca_l1_255_interp']:.3f} | "
              f"{model} | {delta} |")

    # ── markdown 表 2b：模型 vs 论文 224 表 2「本文」行（跨分辨率同值数）──
    print("\n<!-- 448 model vs paper-224 model same values -->")
    print("| t | 每图值数 | 论文 224 本文 / dB | 本实验 448 本文 / dB | Δ(448−224) |")
    print("|---|---|---|---|---|")
    dlist = []
    for row, paper in zip(rows448, PAPER_224_MODEL_PSNR):
        d = row["model_psnr"] - paper if row["model_psnr"] is not None else None
        dlist.append(d)
        mine = "TODO" if row["model_psnr"] is None else ("%.3f" % row["model_psnr"])
        dd = "TODO" if d is None else ("%+.3f" % d)
        print("| %d | %d | %.2f | %s | %s |" % (row["t"], row["model_values"], paper, mine, dd))
    if dlist and dlist[0] is not None:
        print("\nΔ 范围 %+.2f ~ %+.2f dB；448 值 %+.2f dB、7168 值 %+.2f dB"
              % (min(dlist), max(dlist), dlist[0], dlist[-1]))

    # ── markdown 表 3：224 自洽 ──
    print("\n<!-- 224 selfcheck -->")
    print("| 每图值数 | 论文表 2 PCA 行 / dB | 本次 224 重拟合 / dB | Δ / dB |")
    print("|---|---|---|---|")
    for row, paper, diff in zip(rows224, PAPER_224,
                                c224["paper_224_check"]["diff_mine_minus_paper"]):
        print(f"| {row['model_values']} | {paper:.2f} | {row['pca_psnr_interp']:.3f} | "
              f"{diff:+.3f} |")
    print(f"\nmax|Δ| = {c224['paper_224_check']['max_abs_diff']:.4f} dB, "
          f"mean|Δ| = {c224['paper_224_check']['mean_abs_diff']:.4f} dB")

    # ── markdown 表 4：448 vs 224 同值数 ──
    print("\n<!-- 448 vs 224 PCA same values -->")
    print("| 每图值数 | 224/p16 PCA PSNR | 448/p32 PCA PSNR | Δ(448−224) |")
    print("|---|---|---|---|")
    for c in cross:
        print(f"| {c['values_per_image']} | {c['pca224_psnr_interp']:.3f} | "
              f"{c['pca448_psnr_interp']:.3f} | {c['delta_448_minus_224_dB']:+.3f} |")


if __name__ == "__main__":
    main()
