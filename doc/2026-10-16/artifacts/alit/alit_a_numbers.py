#!/usr/bin/env python3
"""E-A 原始数字汇总（**不做判决**）
================================================================
读取 `tools/alit_adaptive_k.py` / `zs_redundancy.py` / `gate_diff.py` 的产出，
把关键**原始数字**集中到一个 json，便于报告引用。

⚠️ 本文件**不产出任何判决文本**：三态判决（成立/否证/数据不足）唯一裁定方是
alit-report 的 `tools/alit_report.py` + PLAN §8，见 `report_tables.md`。

用法::

    python doc/2026-10-16/artifacts/alit/alit_a_numbers.py --dir doc/2026-10-16/artifacts/alit
"""
import argparse
import json
import os


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


NOTE = ("本文件只汇总原始数字, 不含判决; 三态判决(成立/否证/数据不足)由 "
        "alit-report 的 tools/alit_report.py + PLAN §8 唯一裁定, 见 report_tables.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="doc/2026-10-16/artifacts/alit")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    A = args.dir
    out = {"dir": A, "note": NOTE, "key_numbers": {}}

    tsc = _load(os.path.join(A, "tsc_stdattn_rec.json"))
    ladder = _load(os.path.join(A, "domain_ladder.json"))
    zs = _load(os.path.join(A, "zs_redundancy_stdattn_rec.json"))
    gate = _load(os.path.join(A, "gate_diff_stdattn_rec.json"))
    base = _load(os.path.join(A, "baseline_four_arm_eval.json"))

    if gate:
        out["key_numbers"]["S2_gate"] = {"verdict_field": gate.get("verdict"),
                                         "max_abs_diff": gate.get("max_abs_diff"),
                                         "n_numeric_diffs": gate.get("n_numeric_diffs"),
                                         "n_struct_diffs": gate.get("n_struct_diffs")}

    if base:
        out["key_numbers"]["baseline_psnr"] = {k: v["psnr"] for k, v in base.items()}

    if tsc:
        keys = tsc.get("E_A1_stop_distribution", {})
        for key in keys:
            a1 = keys[key]
            out["key_numbers"].setdefault("E_A1", {})[key] = {
                "mode_step_1based": a1["mode_step_1based"], "mode_share": a1["mode_share"],
                "entropy_normalized": a1["entropy_normalized"],
                "std_best_step": a1["std_best_step"],
                "mean_best_step_1based": a1["mean_best_step_1based"],
                "oracle_tokens_mean": a1["oracle_tokens_mean"],
                "fixed_step7_tokens": a1["fixed_step7_tokens"],
                "oracle_vs_fixed7_token_ratio": a1["oracle_vs_fixed7_token_ratio"],
                "oracle_gain_L1_255": a1.get("oracle_gain_L1_255"),
                "best_step_hist_1based": a1["best_step_hist_1based"]}
        for key, a2 in tsc.get("E_A2_tsc", {}).items():
            out["key_numbers"].setdefault("E_A2", {})[key] = {
                "matched_equal_budget": a2.get("matched_equal_budget"),
                "matched_nearest_tau": a2.get("matched_nearest_tau"),
                "fixed_steps": a2.get("fixed_steps")}
        for key, a6 in tsc.get("E_A6_delta_decomposition", {}).items():
            out["key_numbers"].setdefault("E_A6", {})[key] = {
                "delta1_share_mean": a6["delta1_share_mean"],
                "cumulative_share_prefix_mean": a6["cumulative_share_prefix_mean"],
                "last_step_overshoot_rate": a6["last_step_overshoot_rate"],
                "last_step_delta_mean": a6["last_step_delta_mean"],
                "delta_mean_per_step": a6["delta_mean_per_step"]}
        for key, a4 in tsc.get("E_A4_complexity_spearman", {}).items():
            out["key_numbers"].setdefault("E_A4", {})[key] = {
                "rows": a4.get("rows"), "honesty": a4.get("honesty")}

    if ladder and ladder.get("E_A3_domain_ladder"):
        a3 = ladder["E_A3_domain_ladder"]
        out["key_numbers"]["E_A3"] = {
            "delta_steps_vs_first": a3.get("delta_steps_vs_first"),
            "per_domain": a3.get("per_domain"),
            "common_tau_needed_steps": a3.get("common_tau_needed_steps")}

    if zs:
        for key, s in zs.get("results", {}).items():
            out["key_numbers"].setdefault("E_A5", {})[key] = {
                "K": s["K"], "D": s["D"], "n_images": zs.get("n_images"),
                "effective_rank": s["effective_rank"],
                "effective_rank_norm": s["effective_rank_norm"],
                "effective_rank_centered": s["effective_rank_centered"],
                "pair_cos_mean": s["pair_cos_mean"],
                "pair_cos_q50": s["pair_cos_q50"], "pair_cos_q90": s["pair_cos_q90"],
                "pair_cos_frac_gt_0_9": s["pair_cos_frac_gt_0_9"],
                "token_var_mean": s["token_var_mean"],
                "token_mean_var": s["token_mean_var"]}

    op = args.out or os.path.join(A, "alit_a_numbers.json")
    with open(op, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[numbers] {op}")
    print(f"[numbers] blocks: {sorted(out['key_numbers'])}")
    print(f"[numbers] note: {NOTE}")


if __name__ == "__main__":
    main()
