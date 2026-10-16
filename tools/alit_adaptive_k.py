#!/usr/bin/env python3
"""ALIT 式自适应 token 数 — 零训练分析工装（E-A1 / E-A2 / E-A3 / E-A4 / E-A6）
================================================================================

读 `eval_stepwise_linattn.py --dump_per_image` 的 json, 产出 PLAN
(`doc/2026-10-16/PLAN_alit_adaptive_experiments.md` §3) 的零训练分析:

  · **E-A1 停步分布**: 每图最优停步的直方图/熵/方差/众数占比 + oracle 总 token
    数 vs 固定 step7 的 token 数之比与 L1 差;
  · **E-A2 TSC 曲线**: 每图取最小满足 `L1_255[t] <= τ` 的 t（读窗口 hi+1 即该图
    token 数）, 扫 τ 得 "平均 token 数 ↔ 平均 L1/PSNR"; 再与固定步
    {3,5,7,8} **等平均预算配对**对照, 报 ΔL1/ΔPSNR 与判据位;
  · **E-A3 域阶梯**（需多个 `--stepwise`）: 各域平均停步/所需步数差、共同 τ 下的
    所需步数、step1↔step7 落差;
  · **E-A4 复杂度相关性**: 弱代理（JPEG 文件大小 / 边缘密度 / step1 L1）与
    最优停步的 Spearman ρ + bootstrap CI;
  · **E-A6 Δ_t 分解**: 逐步边际 Δ_t、前置累计占比（≥88% 复核）、末步过冲率及
    其**类别交叉**与**边缘密度四分位分层**。

用法（路径按机器替换）::

    python tools/alit_adaptive_k.py --stepwise <dump.json> --out <out.json>
    python tools/alit_adaptive_k.py --stepwise iid.json coco.json --domains iid coco \
        --out alit_a.json

纯分析: 复用 `metrics_linattn.spearman_ci`（torch 由其间接引入）; 不依赖本地绝对
路径, 可直接 rsync 到服务器运行。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics_linattn import spearman_ci  # noqa: E402  (需先补仓库根到 sys.path)


# ────────────────────────── 载入 / 小工具 ──────────────────────────

def load_dump(path: str, domain: str = None) -> dict:
    """读一个 `--dump_per_image` json → 分析用数组字典。"""
    with open(path) as f:
        d = json.load(f)
    imgs = d.get("images") or []
    assert imgs, (f"{path}: 没有 images 字段 —— 需要 eval_stepwise_linattn.py "
                  f"--dump_per_image 产出的 json")
    L1 = np.asarray([im["L1_255"] for im in imgs], dtype=np.float64)
    PSNR = np.asarray([im["PSNR"] for im in imgs], dtype=np.float64)
    assert L1.ndim == 2 and L1.shape == PSNR.shape, "L1_255/PSNR 形状不一致"
    T = L1.shape[1]
    cum = d.get("cum_z_s") or list(range(1, T + 1))
    cum = [int(c) for c in cum][:T]
    assert len(cum) == T, f"{path}: cum_z_s 长度 {len(cum)} != T={T}"

    def _num(key, default=float("nan")):
        vals = []
        for im in imgs:
            v = im.get(key, None)
            vals.append(default if v is None else float(v))
        return np.asarray(vals, dtype=np.float64)

    info = {
        "path": path,
        "domain": domain or d.get("domain")
        or os.path.splitext(os.path.basename(path))[0],
        "L1": L1, "PSNR": PSNR, "T": T, "cum_z_s": cum,
        "steps": d.get("steps"),
        "n": int(len(imgs)),
        "best_step": L1.argmin(1),                      # 0-based
        "file_bytes": _num("file_bytes"),
        "edge_density": _num("edge_density"),
        "first_step_L1": L1[:, 0].copy(),
        "categories": [str(im.get("category") or "?") for im in imgs],
        "labels": [int(im.get("label", -1)) for im in imgs],
        "files": [str(im.get("path") or im.get("file_name") or "") for im in imgs],
    }
    return info


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _needed_step(info: dict, tau: float) -> dict:
    """每图最小 t 使 L1[t] <= τ; 无满足则取末步并计入 frac_capped。"""
    L1, T, cum = info["L1"], info["T"], info["cum_z_s"]
    ok = L1 <= tau
    any_ok = ok.any(1)
    t_idx = np.where(any_ok, ok.argmax(1), T - 1)
    tokens = np.asarray([cum[t] for t in t_idx], dtype=np.float64)
    return {"mean_step_1based": float(t_idx.mean()) + 1.0,
            "mean_tokens": float(tokens.mean()),
            "frac_capped": float((~any_ok).mean())}


def default_taus(L1: np.ndarray, n_taus: int = 40) -> list:
    """自动 τ 网格: 从末步的 p10 到首步的 p50（几何间距）。"""
    lo = float(np.percentile(L1[:, -1], 10))
    hi = float(np.percentile(L1[:, 0], 50))
    if not (0 < lo < hi):
        lo, hi = float(max(L1.min(), 1e-3)), float(max(L1.max(), 1e-2))
    if not (0 < lo < hi):
        lo, hi = 1e-3, 1.0
    return [float(x) for x in np.geomspace(lo, hi, max(2, int(n_taus)))]


# ────────────────────────── E-A1 ──────────────────────────

def section_a1(info: dict) -> dict:
    L1, T, cum, best = info["L1"], info["T"], info["cum_z_s"], info["best_step"]
    n = info["n"]
    hist = np.bincount(best, minlength=T).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    H = _entropy(prob)
    mode = int(hist.argmax())
    oracle_tokens = float(np.mean([cum[b] for b in best]))
    ref = min(6, T - 1)                                    # 固定 step7（T<7 时退到末步）
    oracle_l1 = float(np.mean(L1[np.arange(n), best]))
    fixed_l1 = float(L1[:, ref].mean())
    return {
        "n_images": n, "T": T, "cum_z_s": cum,
        "best_step_hist_1based": hist.astype(int).tolist(),
        "best_step_prob": [float(p) for p in prob],
        "mean_best_step_1based": float(best.mean()) + 1.0,
        "std_best_step": float(best.std()),
        "entropy_nats": H,
        "entropy_normalized": (H / math.log(T)) if T > 1 else 0.0,
        "mode_step_1based": mode + 1,
        "mode_share": float(prob[mode]),
        "oracle_tokens_mean": oracle_tokens,
        "fixed_step7_tokens": int(cum[ref]),
        "oracle_vs_fixed7_token_ratio": (oracle_tokens / cum[ref]) if cum[ref] else float("nan"),
        "oracle_L1_255": oracle_l1,
        "fixed_step7_L1_255": fixed_l1,
        "oracle_gain_L1_255": fixed_l1 - oracle_l1,
        "verdict": {
            "concentrated_ge_90pct": bool(prob[mode] >= 0.9),
            "note": "熵高/众数占比低 ⇒ 图间差异可利用 (PLAN E-A1); "
                    "≥90% 同一档 ⇒ 图间差异不可利用",
        },
    }


# ────────────────────────── E-A2 ──────────────────────────

def _adaptive_curve(info: dict, taus: list) -> list:
    """τ → (平均 token 数, 平均 L1/PSNR): 每图取最小 t 满足 L1[t] <= τ。"""
    L1, PSNR, T, cum = info["L1"], info["PSNR"], info["T"], info["cum_z_s"]
    n = info["n"]
    rows = []
    for tau in taus:
        ok = L1 <= tau
        any_ok = ok.any(1)
        t_idx = np.where(any_ok, ok.argmax(1), T - 1)
        tokens = np.asarray([cum[t] for t in t_idx], dtype=np.float64)
        rows.append({
            "tau": float(tau),
            "mean_tokens": float(tokens.mean()),
            "mean_stop_step_1based": float(t_idx.mean()) + 1.0,
            "mean_L1_255": float(L1[np.arange(n), t_idx].mean()),
            "mean_PSNR": float(PSNR[np.arange(n), t_idx].mean()),
            "frac_capped": float((~any_ok).mean()),
        })
    return rows


def section_a2(info: dict, taus: list, fixed_steps: list, n_dense: int = 201) -> dict:
    L1, PSNR, T, cum = info["L1"], info["PSNR"], info["T"], info["cum_z_s"]
    rows = _adaptive_curve(info, taus)                     # 用户/自动 τ 网格（报告用）
    fixed = []
    for k in fixed_steps:
        if 1 <= k <= T:
            fixed.append({"step": int(k), "mean_tokens": float(cum[k - 1]),
                          "mean_L1_255": float(L1[:, k - 1].mean()),
                          "mean_PSNR": float(PSNR[:, k - 1].mean())})
    # ── 等平均预算对照 ──
    # 两条口径: (a) 最近 τ 网格点（报 budget 相对偏差, 可能对不齐）;
    #          (b) 密集 τ 曲线上按**平均 token 数插值**（等预算的干净口径, 判据用它）。
    matched = []
    for fx in fixed:
        row = min(rows, key=lambda r: abs(r["mean_tokens"] - fx["mean_tokens"]))
        gap = abs(row["mean_tokens"] - fx["mean_tokens"]) / max(fx["mean_tokens"], 1e-9)
        matched.append({
            "step": fx["step"], "fixed_mean_tokens": fx["mean_tokens"],
            "adaptive_tau": row["tau"], "adaptive_mean_tokens": row["mean_tokens"],
            "budget_rel_gap": float(gap), "budget_matched_5pct": bool(gap <= 0.05),
            "fixed_mean_L1_255": fx["mean_L1_255"],
            "adaptive_mean_L1_255": row["mean_L1_255"],
            "delta_L1_255": row["mean_L1_255"] - fx["mean_L1_255"],
            "delta_PSNR": row["mean_PSNR"] - fx["mean_PSNR"],
            "adaptive_better": bool(row["mean_L1_255"] < fx["mean_L1_255"]),
        })
    # 密集曲线: τ 从 "全部 capped"(低) 到 "全部 step1"(高), 覆盖整个 token 区间
    lo = max(float(L1.min()) * 0.5, 1e-6)
    hi = max(float(L1[:, 0].max()) * 1.5, lo * 1.01)
    dense = _adaptive_curve(info, [float(x) for x in np.geomspace(lo, hi, n_dense)])
    tk = np.asarray([r["mean_tokens"] for r in dense])
    order = np.argsort(tk)
    tk_s = tk[order]
    l1_s = np.asarray([r["mean_L1_255"] for r in dense])[order]
    ps_s = np.asarray([r["mean_PSNR"] for r in dense])[order]
    interp = []
    for fx in fixed:
        B = fx["mean_tokens"]
        in_range = bool(tk_s.min() - 1e-9 <= B <= tk_s.max() + 1e-9)
        l1v = float(np.interp(B, tk_s, l1_s))
        psv = float(np.interp(B, tk_s, ps_s))
        interp.append({
            "step": fx["step"], "budget_tokens": B,
            "adaptive_mean_L1_255_at_budget": l1v,
            "adaptive_mean_PSNR_at_budget": psv,
            "fixed_mean_L1_255": fx["mean_L1_255"], "fixed_mean_PSNR": fx["mean_PSNR"],
            "delta_L1_255": l1v - fx["mean_L1_255"], "delta_PSNR": psv - fx["mean_PSNR"],
            "adaptive_better": bool(l1v < fx["mean_L1_255"]),
            "budget_in_range": in_range,
        })
    return {
        "tau_curve": rows, "fixed_steps": fixed,
        "matched_nearest_tau": matched,
        "matched_interp_equal_budget": interp,
        # 兼容旧键名（task 描述里的 "等预算对照"）
        "matched_equal_budget": interp,
        "verdict": {
            "any_match_adaptive_better_0.10px": bool(
                any(m["delta_L1_255"] <= -0.10 and m["budget_in_range"]
                    for m in interp)),
            "note": "delta_L1_255 < 0 = 自适应更优; 判据 >0.10 px 且方向一致 "
                    "(PLAN E-A2 §8); 等预算口径见 matched_interp_equal_budget "
                    "(密集 τ 曲线上插值), matched_nearest_tau 仅作参考",
        },
    }


# ────────────────────────── E-A3 ──────────────────────────

def section_a3(infos: list, n_tau: int = 9) -> dict:
    ref = infos[0]
    per_domain = {}
    for inf in infos:
        T = inf["T"]
        per_domain[inf["domain"]] = {
            "n_images": inf["n"],
            "mean_best_step_1based": float(inf["best_step"].mean()) + 1.0,
            "std_best_step": float(inf["best_step"].std()),
            "oracle_tokens_mean": float(np.mean([inf["cum_z_s"][b]
                                                 for b in inf["best_step"]])),
            "L1_step1": float(inf["L1"][:, 0].mean()),
            "L1_step7": float(inf["L1"][:, min(6, T - 1)].mean()),
            "gap_step1_minus_step7": float(inf["L1"][:, 0].mean()
                                           - inf["L1"][:, min(6, T - 1)].mean()),
        }
    taus = default_taus(ref["L1"], n_tau)
    common = {}
    for tau in taus:
        common[f"{tau:.4f}"] = {inf["domain"]: _needed_step(inf, tau)
                                for inf in infos}
    deltas = {inf["domain"]: per_domain[inf["domain"]]["mean_best_step_1based"]
              - per_domain[ref["domain"]]["mean_best_step_1based"]
              for inf in infos}
    ordinals = [per_domain[inf["domain"]]["mean_best_step_1based"] for inf in infos]
    return {
        "per_domain": per_domain,
        "delta_steps_vs_first": deltas,
        "common_tau_needed_steps": common,
        "verdict": {
            "max_abs_mean_step_delta": float(max(abs(v) for v in deltas.values())),
            "monotone_increasing_with_distance": bool(
                all(ordinals[i] <= ordinals[i + 1] for i in range(len(ordinals) - 1))),
            "note": "最大平均停步差 <0.5 步 ⇒ 熟悉度假设不成立 (PLAN E-A3 §8), "
                    "预期 OOD 所需步数更多",
        },
    }


# ────────────────────────── E-A4 ──────────────────────────

def section_a4(infos: list, n_boot: int, seed: int) -> dict:
    out = {}
    for inf in infos:
        L1 = inf["L1"].copy()
        T = inf["T"]
        tau_ref = float(np.median(L1[:, -1]))              # 该域末步中位水平
        ok = L1 <= tau_ref
        any_ok = ok.any(1)
        t_idx = np.where(any_ok, ok.argmax(1), T - 1).astype(np.float64)
        targets = {"best_step": inf["best_step"].astype(np.float64),
                   "needed_step_at_final_median_tau": t_idx}
        proxies = {"jpeg_bytes": inf["file_bytes"],
                   "edge_density": inf["edge_density"],
                   "first_step_L1": inf["first_step_L1"]}
        rows = []
        for pname, pv in proxies.items():
            for tname, tv in targets.items():
                r = spearman_ci(pv, tv, n_boot=n_boot, seed=seed)
                r.update(proxy=pname, target=tname, tau_ref=tau_ref,
                         n_finite=int(np.isfinite(pv).sum()))
                rows.append(r)
        out[inf["domain"]] = {
            "rows": rows,
            "honesty": "JPEG 大小/边缘密度是图像复杂度的**弱代理**, ρ 显著不等于"
                       "验证 ALIT 的熵假设; 代理不足时只能说 '代理量不足'",
        }
    return out


# ────────────────────────── E-A6 ──────────────────────────

def section_a6(infos: list, topk: int = 5, min_count: int = 20) -> dict:
    out = {}
    for inf in infos:
        L1, T, n = inf["L1"], inf["T"], inf["n"]
        delta = L1[:, :-1] - L1[:, 1:] if T > 1 else np.zeros((n, 0))  # 正 = 变好
        drop = np.maximum(L1[:, 0] - L1.min(1), 1e-9)
        cum_share = np.asarray([(L1[:, 0] - L1[:, t]) / drop for t in range(T)])
        front = {f"step{k}": float(cum_share[k - 1].mean()) for k in range(1, T + 1)}
        last_delta = delta[:, -1] if T > 1 else np.zeros(n)
        overshoot = last_delta < 0
        cats = np.asarray(inf["categories"])
        cat_rows = []
        for c in sorted(set(cats.tolist())):
            m = cats == c
            if int(m.sum()) < min_count:
                continue
            cat_rows.append({
                "category": c, "n": int(m.sum()),
                "overshoot_rate": float(overshoot[m].mean()),
                "mean_delta1_share": (float((delta[:, 0] / drop)[m].mean())
                                      if T > 1 else float("nan")),
                "mean_best_step_1based": float(inf["best_step"][m].mean()) + 1.0,
            })
        cat_rows.sort(key=lambda r: -r["overshoot_rate"])
        ed = inf["edge_density"]
        ed_f = ed[np.isfinite(ed)]
        q_rows = []
        if ed_f.size >= 8:
            qs = np.quantile(ed_f, [0.25, 0.5, 0.75])
            masks = [("Q1_low", ed <= qs[0]),
                     ("Q2", (ed > qs[0]) & (ed <= qs[1])),
                     ("Q3", (ed > qs[1]) & (ed <= qs[2])),
                     ("Q4_high", ed > qs[2])]
            for name, m in masks:
                if int(m.sum()) > 0:
                    q_rows.append({"quartile": name, "n": int(m.sum()),
                                   "overshoot_rate": float(overshoot[m].mean()),
                                   "mean_best_step_1based": float(
                                       inf["best_step"][m].mean()) + 1.0})
        out[inf["domain"]] = {
            "delta_mean_per_step": [float(v) for v in (delta.mean(0)
                                                       if delta.size else [])],
            "delta1_share_mean": (float((delta[:, 0] / drop).mean())
                                  if T > 1 else float("nan")),
            "cumulative_share_prefix_mean": front,
            "last_step_overshoot_rate": float(overshoot.mean()),
            "last_step_delta_mean": float(last_delta.mean()),
            "by_category_top_overshoot": cat_rows[:topk],
            "by_category_bottom_overshoot": cat_rows[-topk:],
            "by_edge_density_quartile": q_rows,
            "verdict": {
                "ge_88pct_within_first_4_steps": (
                    bool(front.get("step4", 0.0) >= 0.88) if T >= 4 else None),
                "note": "末步过冲 = Δ_T<0 的图占比; 类别表只含 n>=min_count; "
                        "复现 ANALYSIS_ksweep_stepwise_refinement / "
                        "REPORT_stdattn_ablation §4 口径",
            },
        }
    return out


# ────────────────────────── main ──────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="ALIT 式自适应 token 数零训练分析 (E-A1/A2/A3/A4/A6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stepwise", nargs="+", required=True,
                    help="一个或多个 --dump_per_image json; 第一个视为 IID/参照域")
    ap.add_argument("--domains", nargs="*", default=None,
                    help="域标签（与 --stepwise 一一对应）; 缺省用 json 内 domain 字段")
    ap.add_argument("--out", required=True, help="输出汇总 json")
    ap.add_argument("--taus", default="",
                    help="逗号分隔 τ 网格 (L1_255 px); 空 = 自动几何网格")
    ap.add_argument("--n_taus", type=int, default=40, help="自动 τ 网格点数")
    ap.add_argument("--fixed_steps", default="3,5,7,8",
                    help="E-A2 固定步对照臂（1-based）")
    ap.add_argument("--n_boot", type=int, default=1000, help="E-A4 bootstrap 次数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5, help="E-A6 类别表条数")
    a = ap.parse_args()

    assert not a.domains or len(a.domains) == len(a.stepwise), \
        f"--domains {len(a.domains)} 个 != --stepwise {len(a.stepwise)} 个"
    infos = [load_dump(p, a.domains[i] if a.domains else None)
             for i, p in enumerate(a.stepwise)]
    taus = ([float(x) for x in a.taus.split(",") if x.strip()]
            or default_taus(infos[0]["L1"], a.n_taus))
    fixed = [int(x) for x in a.fixed_steps.split(",") if x.strip()]

    res = {
        "meta": {
            "stepwise": list(a.stepwise),
            "domains": [i["domain"] for i in infos],
            "n_images": {i["domain"]: i["n"] for i in infos},
            "T": {i["domain"]: i["T"] for i in infos},
            "cum_z_s": {i["domain"]: i["cum_z_s"] for i in infos},
            "tau_grid": taus, "n_taus": len(taus),
            "fixed_steps": fixed, "n_boot": a.n_boot, "seed": a.seed,
            "source": "eval_stepwise_linattn.py --dump_per_image",
        },
        "E_A1_stop_distribution": {i["domain"]: section_a1(i) for i in infos},
        "E_A2_tsc": {i["domain"]: section_a2(i, taus, fixed) for i in infos},
        "E_A3_domain_ladder": section_a3(infos) if len(infos) > 1 else None,
        "E_A4_complexity_spearman": section_a4(infos, a.n_boot, a.seed),
        "E_A6_delta_decomposition": section_a6(infos, a.topk),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ── 终端摘要（每一段都打印, 便于服务器上快速核对产出）──
    print(f"[alit_adaptive_k] {len(infos)} 域  T={infos[0]['T']}  "
          f"n={[i['n'] for i in infos]}  τ∈[{taus[0]:.3f},{taus[-1]:.3f}]×{len(taus)}")
    for dom, s in res["E_A1_stop_distribution"].items():
        print(f"  E-A1 [{dom}] 众数 step{s['mode_step_1based']} 占 {s['mode_share']*100:.1f}%"
              f" | 熵 {s['entropy_nats']:.3f} (norm {s['entropy_normalized']:.3f})"
              f" | μstep {s['mean_best_step_1based']:.2f}±{s['std_best_step']:.2f}"
              f" | oracle tokens {s['oracle_tokens_mean']:.1f} vs step7 "
              f"{s['fixed_step7_tokens']} (×{s['oracle_vs_fixed7_token_ratio']:.2f})"
              f" | oracle 增益 {s['oracle_gain_L1_255']:+.3f} px")
    for dom, s in res["E_A2_tsc"].items():
        print(f"  E-A2 [{dom}] τ 点数 {len(s['tau_curve'])}  "
              f"固定步 {[f['step'] for f in s['fixed_steps']]}")
        near = {m["step"]: m for m in s["matched_nearest_tau"]}
        for m in s["matched_interp_equal_budget"]:
            n = near[m["step"]]
            print(f"       等预算(插值): {m['budget_tokens']:.1f} tokens → "
                  f"ΔL1={m['delta_L1_255']:+.3f}px ΔPSNR={m['delta_PSNR']:+.3f}dB"
                  f"  | 最近τ网格: τ={n['adaptive_tau']:.3f} "
                  f"tokens={n['adaptive_mean_tokens']:.1f} "
                  f"(gap {n['budget_rel_gap']*100:.1f}%)")
    if res["E_A3_domain_ladder"] is not None:
        s = res["E_A3_domain_ladder"]
        print(f"  E-A3 平均停步: " + "  ".join(
            f"{d}:{v['mean_best_step_1based']:.2f}" for d, v in s["per_domain"].items())
            + f"  | Δ vs {infos[0]['domain']}: "
            + ", ".join(f"{d}:{v:+.2f}" for d, v in s["delta_steps_vs_first"].items()))
    for dom, s in res["E_A4_complexity_spearman"].items():
        sig = [r for r in s["rows"] if np.isfinite(r["rho"])]
        print(f"  E-A4 [{dom}] " + "  ".join(
            f"{r['proxy']}~{r['target']}: ρ={r['rho']:+.3f}"
            f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
            for r in sig[:3]) + f"  ({len(sig)} 对)")
    for dom, s in res["E_A6_delta_decomposition"].items():
        top = s["by_category_top_overshoot"][0] if s["by_category_top_overshoot"] else None
        print(f"  E-A6 [{dom}] Δ1占比 {s['delta1_share_mean']*100:.1f}%"
              f" | step4 前置占比 "
              f"{s['cumulative_share_prefix_mean'].get('step4', float('nan'))*100:.1f}%"
              f" | 末步过冲率 {s['last_step_overshoot_rate']*100:.1f}%"
              f" | 最高过冲类别 {top['category'] if top else '—'}")
    print(f"[out] {a.out}")
    return res


if __name__ == "__main__":
    main()
