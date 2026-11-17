#!/usr/bin/env python3
"""自测 `tools/analyze_savoias_single.py`
==========================================

三层自测：
  A. 合成数据（总计 500 张：Art 250 / Scenes 150 / Objects 100），已知相关性
     L1 = 0.05*complexity + 8 + N(0,0.8)，PSNR = 30 − 1.2*L1 + N(0,0.5)；
     合成 gt npy 用官方格式 `{index: [filename, score]}`（**gt 文件名 .png，
     图像 .jpg** ⇒ 专门验证"按整数 stem join"而不是按全名）。
     验证：ρ(L1) 与独立重算一致（<1e-9）且落在 0.80–0.95；ρ(PSNR)<0；
           三分位 low/mid/high/n/gap/rel% 与独立重算一致（<1e-9）；
           OOD−ID = 子集均值 − ID 均值（<1e-9）；三/单文件两种输入结果一致；
           T>1 的 --stepwise 分支产出 per-step ρ 与 fig_rho_vs_step。
  B. 错误路径：per_image 缺失 / gt 文件缺失 / --metric_step 越界 / join 全失败
     都必须非 0 退出且报错信息明确。
  C. 真实交叉验证（可选，需真实 gt）：用仓库里真实的 820 张 canonical dump
     `AssocVTC/doc/2026-10-26/artifacts/ood_savoias/per_image_m240.json`（T=16）
     + 真实 gt 重跑，与旧 `plot_savoias_scale.py` 产出的同目录 `summary.json`
     （REPORT_ood_savoias_scale.md §4 数字）逐位对比 ρ 与三分位 → 证明口径一致。
     gt 来源优先级：--gt-dir > /tmp/savoias_gt > 从 GitHub 下载到 /tmp/savoias_gt；
     都不可用则 SKIP（--skip-real 可显式跳过）。

用法
----
  /usr/bin/python3 tools/selftest_analyze_savoias_single.py
  /usr/bin/python3 tools/selftest_analyze_savoias_single.py --skip-real
  /usr/bin/python3 tools/selftest_analyze_savoias_single.py --gt-dir /root/autodl-tmp/savoias/gt
"""

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "analyze_savoias_single.py"
REAL_ART = REPO / "doc" / "2026-10-26" / "artifacts" / "ood_savoias"
REAL_PER_IMAGE = REAL_ART / "per_image_m240.json"
REAL_REF_SUMMARY = REAL_ART / "summary.json"
GT_URL = ("https://raw.githubusercontent.com/esaraee/Savoias-Dataset/master/"
          "Ground%20truth/npy/global_ranking_{key}.npy")

SUBSET_SPECS = [("art", "Art", "People-Art", 250),
                ("scenes", "Scenes", "Places", 150),
                ("objects", "Objects", "MSCOCO", 100)]

RESULTS = []  # (name, ok, detail)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def run(args, timeout=300):
    p = subprocess.run([sys.executable] + args, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def spearman(x, y):
    r = spearmanr(np.asarray(x, float), np.asarray(y, float))
    return float(r.correlation), float(r.pvalue)


def terciles(S, y):
    """与 analyze_savoias_single.py / plot_savoias_scale.py 相同的独立实现。"""
    S = np.asarray(S, float)
    y = np.asarray(y, float)
    q1, q2 = np.percentile(S, [100 / 3, 200 / 3])
    m = {"low": S <= q1, "mid": (S > q1) & (S <= q2), "high": S > q2}
    out = {"q1": float(q1), "q2": float(q2)}
    for k, mm in m.items():
        out[k] = float(y[mm].mean())
        out["n_" + k] = int(mm.sum())
    out["gap"] = out["high"] - out["low"]
    out["rel"] = out["gap"] / out["low"] * 100.0
    return out


# ─────────────────────────── 合成数据 ───────────────────────────
def make_synthetic(work):
    rng = np.random.default_rng(20261117)
    syn = {}
    for key, dirname, pretty, n in SUBSET_SPECS:
        c = rng.integers(0, 101, size=n)                       # 人评复杂度 0–100
        l1 = 0.05 * c + 8.0 + rng.normal(0, 0.8, size=n)       # 已知正相关
        psnr = 30.0 - 1.2 * l1 + rng.normal(0, 0.5, size=n)    # 已知负相关
        syn[key] = {"n": n, "dir": dirname, "pretty": pretty,
                    "c": c.astype(float), "l1": l1, "psnr": psnr}

    gt_dir = work / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for key, _, _, _ in SUBSET_SPECS:
        # 官方格式：{index: [filename, score]}；注意文件名是 .png
        gt = {i: [f"{i}.png", np.int64(int(c))]
              for i, c in enumerate(syn[key]["c"])}
        np.save(gt_dir / f"gt_{key}.npy", np.array(gt, dtype=object),
                allow_pickle=True)

    top = {"ckpt": "/root/autodl-tmp/ckpts/tok448/ckpt_best.pt", "domain": "savoias",
           "recurrent": False, "T": 1, "steps": [1], "cum_z_s": [16], "K": 16,
           "num_patches": 784, "n_images": 500,
           "proxy_note": "synthetic self-test"}

    def rec(key, i, with_category):
        s = syn[key]
        r = {"index": int(i),
             "path": f"/root/autodl-tmp/savoias/{s['dir']}/{i}.jpg",
             "file_name": f"{i}.jpg",
             "L1_255": [float(s["l1"][i])], "PSNR": [float(s["psnr"][i])],
             "best_step": 0, "read_step": 1, "best_L1_255": float(s["l1"][i]),
             "file_bytes": 1234, "label": -1}
        if with_category:
            r["category"] = s["dir"]
        return r

    # 模式 A：三个子集文件；三种不同的"子集判定路径"
    #   art   → 记录带 category（canonical）
    #   scenes→ 无 category，靠 path 里的 Scenes/ 目录
    #   objects→ 无 category、path 无子集词，靠 --per_image objects=<file> 显式提示
    files_a = {}
    for key, _, _, _ in SUBSET_SPECS:
        imgs = [rec(key, i, True) for i in range(syn[key]["n"])]
        d = dict(top)
        d["n_images"] = syn[key]["n"]
        if key == "scenes":
            for r in imgs:
                r.pop("category", None)
        if key == "objects":
            for r in imgs:
                r.pop("category", None)
                r["path"] = f"/tmp/saudi_objects/{r['index']}.jpg"
        p = work / f"per_image_{key}.json"
        with open(p, "w") as f:
            json.dump({"images": imgs}, f)
        files_a[key] = p

    # 模式 B：单个合并文件（canonical 820 张式）
    merged = {"images": [rec(k, i, True) for k, _, _, _ in SUBSET_SPECS
                         for i in range(syn[k]["n"])]}
    merged.update(top)
    merged["n_images"] = sum(s["n"] for s in syn.values())
    p_merged = work / "per_image_merged.json"
    with open(p_merged, "w") as f:
        json.dump(merged, f)

    # 聚合 stepwise（单步模型：只有一行）
    all_l1 = np.concatenate([syn[k]["l1"] for k, _, _, _ in SUBSET_SPECS])
    all_psnr = np.concatenate([syn[k]["psnr"] for k, _, _, _ in SUBSET_SPECS])
    p_sw = work / "stepwise_tok448.json"
    with open(p_sw, "w") as f:
        json.dump({"ckpt": top["ckpt"], "step": 12000, "epoch": 12, "K": 16,
                   "dec_step_plan": "", "recurrent": False, "T": 1, "steps": [1],
                   "windows": [None], "cum_z_s": [16], "n_images": 500,
                   "L1_255": [float(all_l1.mean())],
                   "PSNR": [float(all_psnr.mean())],
                   "const_baseline_255": 20.0, "best_step": 0}, f)

    # ID 对照（ImageNet-100 val 的 per-image dump；类目是 wnid）
    rng2 = np.random.default_rng(7)
    id_l1 = 9.6 + rng2.normal(0, 1.0, size=100)
    id_psnr = 21.0 - 1.1 * id_l1 + rng2.normal(0, 0.4, size=100)
    id_imgs = [{"index": i, "path": f"/root/autodl-tmp/datasets/imagenet100/val/n01440764/{i}.JPEG",
                "file_name": f"{i}.JPEG", "category": "n01440764", "label": 0,
                "L1_255": [float(id_l1[i])], "PSNR": [float(id_psnr[i])],
                "best_step": 0} for i in range(100)]
    p_id = work / "per_image_id_in100.json"
    with open(p_id, "w") as f:
        json.dump({"ckpt": top["ckpt"], "domain": "iid", "T": 1, "n_images": 100,
                   "images": id_imgs}, f)

    # 模式 C：T=3 多步（验证 --stepwise 的 per-step ρ 分支）
    steps = 3
    multi = {"images": [], "T": steps, "steps": [1, 2, 3], "cum_z_s": [2, 4, 6],
             "K": 6, "n_images": 500, "recurrent": True}
    for key, _, _, _ in SUBSET_SPECS:
        s = syn[key]
        for i in range(s["n"]):
            base = s["l1"][i]
            multi["images"].append({
                "index": int(i), "category": s["dir"],
                "path": f"/root/autodl-tmp/savoias/{s['dir']}/{i}.jpg",
                "file_name": f"{i}.jpg",
                "L1_255": [float(base * 1.35), float(base * 1.12), float(base)],
                "PSNR": [float(s["psnr"][i] - 1.6), float(s["psnr"][i] - 0.8),
                         float(s["psnr"][i])], "best_step": steps - 1})
    p_multi = work / "per_image_multi_t3.json"
    with open(p_multi, "w") as f:
        json.dump(multi, f)
    p_sw3 = work / "stepwise_t3.json"
    with open(p_sw3, "w") as f:
        json.dump({"ckpt": top["ckpt"], "recurrent": True, "T": steps,
                   "steps": [1, 2, 3], "cum_z_s": [2, 4, 6], "K": 6,
                   "n_images": 500,
                   "L1_255": [float(all_l1.mean() * 1.35),
                              float(all_l1.mean() * 1.12),
                              float(all_l1.mean())],
                   "PSNR": [float(all_psnr.mean() - 1.6),
                            float(all_psnr.mean() - 0.8),
                            float(all_psnr.mean())],
                   "best_step": 2}, f)
    return syn, files_a, p_merged, p_sw, p_id, p_multi, p_sw3, gt_dir, id_l1


# ─────────────────────────── 检查 ───────────────────────────
FIG_STEMS = ["fig_complexity_l1_panels", "fig_complexity_l1_art",
             "fig_complexity_l1_scenes", "fig_complexity_l1_objects",
             "fig_rho_bar", "fig_tercile_bar"]


def check_outputs(out, syn, id_l1, prefix, expect_id=True, expected_n_files=3):
    ok_all = True
    sp = out / "summary.json"
    tm = out / "tables.md"
    ok_all &= check(f"{prefix}summary.json+tables.md exist",
                    sp.is_file() and tm.is_file())
    if not sp.is_file():
        return None
    summary = json.load(open(sp))
    ok_all &= check(f"{prefix}summary parses; T=1, metric step 1",
                    summary["protocol"]["T"] == 1
                    and summary["protocol"]["metric_step_1based"] == 1)
    ok_all &= check(f"{prefix}per_image_files: {expected_n_files} file(s), all T=1",
                    len(summary["per_image_files"]) == expected_n_files
                    and all(f["T"] == 1 for f in summary["per_image_files"]))
    # 图存在且非空
    stems = list(FIG_STEMS) + (["fig_ood_vs_id"] if expect_id else [])
    for stem in stems:
        for ext in ("png", "pdf"):
            p = out / f"{stem}.{ext}"
            ok_all &= check(f"{prefix}fig {stem}.{ext}",
                            p.is_file() and p.stat().st_size > 2000,
                            f"{p.stat().st_size} B" if p.is_file() else "missing")
    md = tm.read_text()
    ok_all &= check(f"{prefix}tables.md has expected sections",
                    all(s in md for s in ("Spearman", "People-Art", "Places",
                                          "MSCOCO", "high−low"))
                    and ("OOD − ID" in md) == expect_id)
    # 数值：与独立实现逐位对比
    for key, dirname, pretty, n in SUBSET_SPECS:
        st = summary["subsets"][key]
        s = syn[key]
        ok_all &= check(f"{prefix}{key}: n={n}", st["n"] == n, f"got {st['n']}")
        rho_exp, p_exp = spearman(s["c"], s["l1"])
        rho_got = st["spearman"]["L1_255"]["rho"]
        p_got = st["spearman"]["L1_255"]["p"]
        ok_all &= check(f"{prefix}{key}: rho(L1) matches scipy recompute",
                        abs(rho_got - rho_exp) < 1e-9,
                        f"got {rho_got:+.6f} vs {rho_exp:+.6f}")
        ok_all &= check(f"{prefix}{key}: rho(L1) in expected 0.80–0.95",
                        0.80 <= rho_got <= 0.95, f"{rho_got:+.4f}")
        ok_all &= check(f"{prefix}{key}: p matches recompute",
                        abs(p_got - p_exp) < 1e-9, f"{p_got:.3e} vs {p_exp:.3e}")
        rho_ps_exp = spearman(s["c"], s["psnr"])[0]
        ok_all &= check(f"{prefix}{key}: rho(PSNR) negative & matches recompute",
                        abs(st["spearman"]["PSNR"]["rho"] - rho_ps_exp) < 1e-9
                        and rho_ps_exp < 0,
                        f"{st['spearman']['PSNR']['rho']:+.6f}")
        # 三分位
        exp = terciles(s["c"], s["l1"])
        ter = st["terciles"]
        ok_all &= check(
            f"{prefix}{key}: terciles (low/mid/high/n) match recompute",
            max(abs(ter["low"] - exp["low"]), abs(ter["mid"] - exp["mid"]),
                abs(ter["high"] - exp["high"]), abs(ter["gap_high_low"] - exp["gap"]),
                abs(ter["gap_rel_pct"] - exp["rel"])) < 1e-9
            and [ter["n"][g] for g in ("low", "mid", "high")]
            == [exp["n_low"], exp["n_mid"], exp["n_high"]],
            f"low/mid/high={ter['low']:.3f}/{ter['mid']:.3f}/{ter['high']:.3f}, "
            f"gap={ter['gap_high_low']:+.3f}px ({ter['gap_rel_pct']:+.1f}%)")
        ok_all &= check(f"{prefix}{key}: gap>0 (complexity ⇒ worse)",
                        ter["gap_high_low"] > 0)
        if expect_id:
            exp_d = st["L1_255"]["mean"] - float(id_l1.mean())
            ok_all &= check(f"{prefix}{key}: OOD−ID = subset mean − ID mean",
                            abs(st["ood_minus_id"]["L1_255"] - exp_d) < 1e-9,
                            f"{st['ood_minus_id']['L1_255']:+.4f} px "
                            f"(rel {st['ood_minus_id']['L1_255_rel_pct']:+.2f}%)")
    ok_all &= check(f"{prefix}aggregate n=500", summary["aggregate"]["n"] == 500)
    if expect_id:
        ok_all &= check(f"{prefix}id_reference present",
                        summary.get("id_reference") is not None
                        and abs(summary["id_reference"]["L1_255"]
                                - float(np.mean(id_l1))) < 1e-9)
    else:
        ok_all &= check(f"{prefix}no id_reference", summary.get("id_reference") is None)
    return summary


# ─────────────────────────── 真实交叉验证 ───────────────────────────
def ensure_real_gt(gt_dir_arg, work):
    candidates = []
    if gt_dir_arg:
        candidates.append(Path(gt_dir_arg))
    candidates.append(Path("/tmp/savoias_gt"))
    for d in candidates:
        if all((d / f"gt_{k}.npy").is_file()
               for k, _, _, _ in SUBSET_SPECS):
            return d
    # 下载
    d = Path("/tmp/savoias_gt")
    d.mkdir(parents=True, exist_ok=True)
    try:
        for key, _, _, _ in SUBSET_SPECS:
            p = d / f"gt_{key}.npy"
            if p.is_file():
                continue
            with urllib.request.urlopen(GT_URL.format(key=key), timeout=60) as r:
                p.write_bytes(r.read())
        if all((d / f"gt_{k}.npy").is_file() for k, _, _, _ in SUBSET_SPECS):
            return d
    except Exception as e:
        print(f"  [SKIP] 真实 gt 下载失败: {e}")
    return None


def check_real(gt_dir_arg, work):
    if not REAL_PER_IMAGE.is_file():
        check("real cross-check: per_image_m240.json exists", False,
              str(REAL_PER_IMAGE))
        return
    gt = ensure_real_gt(gt_dir_arg, work)
    if gt is None:
        print("  [SKIP] 真实交叉验证：无可用 gt（可 --gt-dir 指定或 --skip-real）")
        return
    out = work / "out_real"
    rc, so, se = run([str(TOOL), "--per_image", str(REAL_PER_IMAGE),
                      "--gt_dir", str(gt), "--out", str(out)])
    if rc != 0:
        check("real cross-check run exits 0", False, se[-800:])
        return
    summary = json.load(open(out / "summary.json"))
    ref = json.load(open(REAL_REF_SUMMARY))
    ref_tag = "m240"  # per_image_m240.json ↔ summary['models']['m240']
    print(f"  real dump: {REAL_PER_IMAGE.name} T={summary['protocol']['T']} "
          f"n_joined={summary['aggregate']['n']}")
    for key, dirname, pretty, _ in SUBSET_SPECS:
        st = summary["subsets"][key]
        exp = ref["complexity"][ref_tag][dirname]  # 旧 summary 用目录名 Art/Scenes/Objects
        rho_ok = abs(st["spearman"]["L1_255"]["rho"] - exp["rho_l1"]) < 1e-9
        rho_ps_ok = abs(st["spearman"]["PSNR"]["rho"] - exp["rho_psnr"]) < 1e-9
        ter_ok = (
            st["n"] == exp["n"]
            and abs(st["terciles"]["low"] - exp["terciles"]["low"]) < 1e-9
            and abs(st["terciles"]["mid"] - exp["terciles"]["mid"]) < 1e-9
            and abs(st["terciles"]["high"] - exp["terciles"]["high"]) < 1e-9
            and abs(st["terciles"]["gap_high_low"]
                    - exp["terciles"]["gap_high_low"]) < 1e-9
            and abs(st["terciles"]["gap_rel_pct"]
                    - exp["terciles"]["gap_rel_pct"]) < 1e-9)
        check(f"real {pretty}: n/rho/rho_psnr/terciles == old REPORT §4",
              rho_ok and rho_ps_ok and ter_ok,
              f"ρ={st['spearman']['L1_255']['rho']:+.6f} "
              f"(ref {exp['rho_l1']:+.6f}), gap={st['terciles']['gap_high_low']:.6f}px "
              f"(ref {exp['terciles']['gap_high_low']:.6f}), n={st['n']}")


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="selftest for analyze_savoias_single.py")
    ap.add_argument("--workdir", default="/tmp/savoias_selftest",
                    help="自测产物目录（会清空重建）")
    ap.add_argument("--skip-real", action="store_true",
                    help="跳过真实 artifact × 真实 gt 的交叉验证")
    ap.add_argument("--gt-dir", default="", help="真实 SAVOIAS gt 目录（可选）")
    ap.add_argument("--keep", action="store_true", help="保留上次产物目录")
    a = ap.parse_args()
    if not TOOL.is_file():
        sys.exit(f"[error] 找不到 {TOOL}")
    work = Path(a.workdir)
    if work.exists() and not a.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    print(f"[selftest] tool = {TOOL}\n[selftest] work = {work}")

    print("\n== A. 合成数据 (n=500; L1=0.05*complexity+8+noise) ==")
    (syn, files_a, p_merged, p_sw, p_id, p_multi, p_sw3,
     gt_dir, id_l1) = make_synthetic(work)

    out_a = work / "out_a_three_files"
    cmd_a = [str(TOOL),
             "--per_image", f"art={files_a['art']}",
             "--per_image", f"scenes={files_a['scenes']}",
             "--per_image", f"objects={files_a['objects']}",
             "--stepwise", str(p_sw), "--id_per_image", str(p_id),
             "--gt_dir", str(gt_dir), "--out", str(out_a),
             "--label", "synthetic T=1"]
    rc, so, se = run(cmd_a)
    check("A: exit 0 (3 files + stepwise + id)", rc == 0, (se or so)[-1200:])
    if rc == 0:
        check_outputs(out_a, syn, id_l1, "A ")
    if not a.keep:
        with open(work / "cmd_A.txt", "w") as f:
            f.write(" ".join(cmd_a) + "\n")

    print("\n== B. 单合并文件（无 stepwise / 无 ID） ==")
    out_b = work / "out_b_merged"
    rc, so, se = run([str(TOOL), "--per_image", str(p_merged),
                      "--gt_dir", str(gt_dir), "--out", str(out_b)])
    check("B: exit 0 (single merged file)", rc == 0, (se or so)[-1200:])
    if rc == 0:
        sb = check_outputs(out_b, syn, id_l1, "B ", expect_id=False,
                           expected_n_files=1)
        out_a_summary_ok = (out_a / "summary.json").is_file()
        sa = (json.load(open(out_a / "summary.json")) if out_a_summary_ok else {})
        if out_a_summary_ok:
            check("B: same rhos as mode A (input-mode invariant)",
                  all(abs(sb["subsets"][k]["spearman"]["L1_255"]["rho"]
                          - sa["subsets"][k]["spearman"]["L1_255"]["rho"]) < 1e-12
                      for k, _, _, _ in SUBSET_SPECS))
        else:
            check("B: same rhos as mode A (input-mode invariant)", False,
                  "mode A outputs missing")
        check("B: no fig_ood_vs_id / fig_rho_vs_step",
              not (out_b / "fig_ood_vs_id.png").exists()
              and not (out_b / "fig_rho_vs_step.png").exists())

    print("\n== C. T=3 多步 + --stepwise（per-step ρ 分支） ==")
    out_c = work / "out_c_multi_t3"
    rc, so, se = run([str(TOOL), "--per_image", str(p_multi),
                      "--stepwise", str(p_sw3), "--gt_dir", str(gt_dir),
                      "--out", str(out_c)])
    check("C: exit 0 (T=3 + stepwise)", rc == 0, (se or so)[-1200:])
    if rc == 0:
        sc = json.load(open(out_c / "summary.json"))
        check("C: metric step = last (3/3)",
              sc["protocol"]["metric_step_1based"] == 3
              and sc["protocol"]["T"] == 3)
        check("C: stepwise_rho present for all subsets, len=3",
              sc.get("stepwise_rho") is not None
              and all(len(sc["stepwise_rho"][k]["rho_L1_255"]) == 3
                      for k, _, _, _ in SUBSET_SPECS)
              and all(len(sc["stepwise_rho"][k]["rho_PSNR"]) == 3
                      for k, _, _, _ in SUBSET_SPECS))
        check("C: fig_rho_vs_step png+pdf exist",
              (out_c / "fig_rho_vs_step.png").is_file()
              and (out_c / "fig_rho_vs_step.pdf").is_file())
        check("C: stepwise aggregate recorded",
              sc["stepwise"]["T"] == 3
              and abs(sc["stepwise"]["L1_255"][-1] - np.mean(
                  [x["L1_255"][-1] for x in
                   json.load(open(p_multi))["images"]])) < 1e-6)

    print("\n== D. 错误路径（必须明确报错 + 非 0 退出） ==")
    rc, so, se = run([str(TOOL), "--per_image", str(work / "nope.json"),
                      "--gt_dir", str(gt_dir), "--out", str(work / "out_d1")])
    check("D1: missing per_image → clear error", rc != 0 and "不存在" in se,
          (se.strip().splitlines() or [""])[-1])
    bad_gt = work / "gt_missing"
    bad_gt.mkdir(exist_ok=True)
    for key, _, _, _ in SUBSET_SPECS[:2]:
        shutil.copy(gt_dir / f"gt_{key}.npy", bad_gt / f"gt_{key}.npy")
    rc, so, se = run([str(TOOL), "--per_image", str(files_a["art"]),
                      "--gt_dir", str(bad_gt), "--out", str(work / "out_d2")])
    check("D2: missing gt_objects.npy → clear error",
          rc != 0 and "缺少 SAVOIAS gt npy" in se,
          (se.strip().splitlines() or [""])[-1])
    rc, so, se = run([str(TOOL), "--per_image", str(files_a["art"]),
                      "--gt_dir", str(gt_dir), "--out", str(work / "out_d3"),
                      "--metric_step", "5"])
    check("D3: --metric_step 5 on T=1 → out-of-range error",
          rc != 0 and "越界" in se, (se.strip().splitlines() or [""])[-1])
    bad_gt2 = work / "gt_offset"
    bad_gt2.mkdir(exist_ok=True)
    for key, _, _, _ in SUBSET_SPECS:
        src = np.load(gt_dir / f"gt_{key}.npy", allow_pickle=True).item()
        off = {i + 5000: [f"{i + 5000}.png", v[1]] for i, v in src.items()}
        np.save(bad_gt2 / f"gt_{key}.npy", np.array(off, dtype=object),
                allow_pickle=True)
    rc, so, se = run([str(TOOL), "--per_image", str(files_a["art"]),
                      "--gt_dir", str(bad_gt2), "--out", str(work / "out_d4")])
    check("D4: all-stems-unmatched → clear join error",
          rc != 0 and "join gt 全失败" in se,
          (se.strip().splitlines() or [""])[-1])

    if not a.skip_real:
        print("\n== E. 真实交叉验证（canonical 820 张 dump × 真实 gt，T=16） ==")
        check_real(a.gt_dir, work)
    else:
        print("\n== E. 真实交叉验证：SKIP (--skip-real) ==")

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print(f"\n[selftest] {n_pass}/{len(RESULTS)} checks PASS, {n_fail} FAIL")
    if n_fail:
        print("FAILED CHECKS:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    print("[selftest] ALL PASS")


if __name__ == "__main__":
    main()
