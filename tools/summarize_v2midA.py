#!/usr/bin/env python
"""实验 A（crossq 双向 middle）收尾：读评测 JSON + 训练日志 → summary.md。"""
from __future__ import annotations

import json
import os
import re

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--art", type=str, default="doc/2026-11-18/artifacts/v2midA")
_ap.add_argument("--name", type=str, default="v10_120f_v2midA")
_A = _ap.parse_args()
ART = _A.art
OUTR = f"/root/autodl-tmp/tok448_v2/{_A.name}"


def main() -> None:
    os.makedirs(ART, exist_ok=True)
    ev_path = os.path.join(ART, "eval_stepwise.json")
    log_path = os.path.join(ART, f"v10_train_{_A.name}.log")
    lines = []
    ev = None
    if os.path.isfile(ev_path):
        ev = json.load(open(ev_path, encoding="utf-8"))
    if ev is not None:
        c = ev["curves"]
        lines.append(f"# {_A.name} 评测\n")
        lines.append(f"- 片段数: {c['n_clips']}（val），档数: {c['n_steps']}")
        lines.append(f"- 冻结分词器满读上界: {c['psnr_tok_full_mean']:.3f} dB "
                     f"(±{c['psnr_tok_full_std']:.3f})")
        steps = [0, 4, 9, 14, 19, 24, 29]
        names = ["末档/读全量", "t=5", "t=10", "t=15", "t=20", "t=25", "t=30"]
        lines.append("\n| 档位 | PSNR 整片 | PSNR 已读帧 | latent L1 整片 | "
                     "latent L1 已读帧 | z_cls 误差 | special 误差 |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, nm in zip(steps, names):
            lines.append(
                f"| {nm} | {c['psnr_all_mean'][k]:.2f} | {c['psnr_read_mean'][k]:.2f} | "
                f"{c['l1_all_mean'][k]:.4f} | {c['l1_read_mean'][k]:.4f} | "
                f"{c['zcls_l1_mean'][k]:.4f} | {c['spec_l1_mean'][k]:.4f} |")
        lines.append("\n末档：整片与已读帧重合（读全量）。")
    if os.path.isfile(log_path):
        best = None
        hist = []
        for ln in open(log_path, encoding="utf-8", errors="ignore"):
            for m in re.finditer(r"\{'eval_latent_l1': '([\d.eE-]+)'.*?'eval_psnr': '([\d.eE-]+)'", ln):
                hist.append((float(m.group(1)), float(m.group(2))))
        if hist:
            best = max(hist, key=lambda t: t[1])
            lines.append("\n\n## 训练中 eval（2 条 val 片段）")
            lines.append(f"- 记录 {len(hist)} 次，最好 eval_psnr = **{best[1]:.2f} dB** "
                         f"(latent_l1={best[0]:.4f})；末次 = {hist[-1][1]:.2f} dB")
    if lines:
        out = os.path.join(ART, "summary.md")
        open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"[summary] {out}")
    else:
        print("[summary] 没有可用数据")


if __name__ == "__main__":
    main()
