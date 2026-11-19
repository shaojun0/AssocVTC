#!/usr/bin/env python
"""v10 收尾：读逐档评测 JSON + 训练日志 → summary.md（与具体实验无关，通用）。

用法::

    python tools/summarize_v2midA.py --art doc/2026-11-19/artifacts/<NAME> --name <NAME>

产物 = <art>/summary.md。统计口径：
  * 逐档曲线（整片 / 已读帧 / z_cls 分解）取 `curves.*`；
  * 末档 = t=30（读全量，整片与已读帧重合）；
  * 训练中 eval 分两段：阶段 1（带 `_s1` 后缀）与阶段 2；best_trainable
    的 step 从日志 `[done] best_psnr=... @step=...` 行读。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics as st

_ap = argparse.ArgumentParser()
_ap.add_argument("--art", type=str, default="doc/2026-11-18/artifacts/v2midA")
_ap.add_argument("--name", type=str, default="v10_120f_v2midA")
_A = _ap.parse_args()
ART = _A.art


def _parse_train_log(path: str):
    """返回 (s1, s2, best)：两组 [(epoch, psnr, latent_l1)] + [done] 行信息。"""
    s1, s2, done = [], [], None
    if not os.path.isfile(path):
        return s1, s2, done
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            for seg in re.findall(r"\{'eval_[^}]*\}", ln):
                try:
                    d = ast.literal_eval(seg)
                except (ValueError, SyntaxError):
                    continue
                ep = float(d.get("epoch", 0) or 0)
                l1 = float(d.get("eval_latent_l1") or d.get("eval_latent_l1_s1") or 0)
                tf = float(d.get("eval_psnr_tok_full_s1")
                           or d.get("eval_psnr_tok_full") or 0)
                if "eval_psnr_s1" in d:
                    s1.append((ep, float(d["eval_psnr_s1"]), l1, tf))
                elif "eval_psnr" in d:
                    s2.append((ep, float(d["eval_psnr"]), l1, tf))
            m = re.search(r"\[done\] best_psnr=([\d.]+) @step=(\d+)", ln)
            if m:
                done = (float(m.group(1)), int(m.group(2)))
    return s1, s2, done


def _seg_line(tag: str, hist, spe: float | None, ep2step):
    if not hist:
        return None
    best = max(hist, key=lambda t: t[1])
    s = (f"- **{tag}**：{len(hist)} 次 · 首个 {hist[0][1]:.2f} → 末个 "
         f"{hist[-1][1]:.2f} · 最好 **{best[1]:.2f} dB**")
    if ep2step is not None:
        s += f" (step≈{int(round(best[0] * ep2step))})"
    s += f" · 对应 latent_l1={best[2]:.4f}"
    if hist[0][3]:
        s += f"（2 条 val 上界 {hist[0][3]:.2f}）"
    return s


def main() -> None:
    os.makedirs(ART, exist_ok=True)
    ev_path = os.path.join(ART, "eval_stepwise.json")
    log_path = os.path.join(ART, f"v10_train_{_A.name}.log")
    lines: list[str] = []
    ev = None
    if os.path.isfile(ev_path):
        ev = json.load(open(ev_path, encoding="utf-8"))
    n_steps = None
    tok_full = None
    if ev is not None:
        c = ev["curves"]
        n_steps = int(c["n_steps"])
        tok_full = float(c["psnr_tok_full_mean"])
        lines.append(f"# {_A.name} 评测\n")
        lines.append(f"- 片段数: {c['n_clips']}（val），档数: {n_steps}")
        lines.append(f"- 冻结分词器满读上界: {c['psnr_tok_full_mean']:.3f} dB "
                     f"(±{c['psnr_tok_full_std']:.3f})")
        idxs = [0, 4, 9, 14, 19, 24, n_steps - 1]
        names = [f"t={i + 1}" for i in idxs[:-1]] + [f"t={idxs[-1] + 1}（末档/全读）"]
        lines.append("\n| 档位 | PSNR 整片 | PSNR 已读帧 | latent L1 整片 | "
                     "latent L1 已读帧 | z_cls 误差 | special 误差 |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, nm in zip(idxs, names):
            lines.append(
                f"| {nm} | {c['psnr_all_mean'][k]:.2f} | {c['psnr_read_mean'][k]:.2f} | "
                f"{c['l1_all_mean'][k]:.4f} | {c['l1_read_mean'][k]:.4f} | "
                f"{c['zcls_l1_mean'][k]:.4f} | {c['spec_l1_mean'][k]:.4f} |")
        lines.append("\n末档：整片与已读帧重合（读全量）。")
        # 逐片段：末档 PSNR 与冻结分词器上界
        pc = ev.get("per_clip") or []
        if pc and "psnr_all_steps" in pc[0]:
            fin = [x["psnr_all_steps"][-1] for x in pc]
            ub = [x["psnr_tok_full"] for x in pc]
            gap = [a - b for a, b in zip(ub, fin)]
            lines.append(f"- 末档逐片段: **{st.mean(fin):.2f} ± {st.pstdev(fin):.2f}** dB "
                         f"(min {min(fin):.2f} / max {max(fin):.2f})")
            lines.append(f"- 距上界（逐片段）: 平均 **{st.mean(gap):.2f}** dB "
                         f"(min {min(gap):.2f} / max {max(gap):.2f})")
            lines.append(f"- z_cls L1 末档均值 {st.mean([x['zcls_l1_steps'][-1] for x in pc]):.4f}"
                         f" · special L1 末档均值 {st.mean([x['spec_l1_steps'][-1] for x in pc]):.4f}")

    s1, s2, done = _parse_train_log(log_path)
    if s1 or s2 or done:
        # epoch 的参考步数（从日志跑批进度不可靠，用 --name 里的 run 反推）:
        # 这里只给估计值，标注 step≈
        ep2step = None
        lines.append("\n## 训练中 eval（2 条固定 val 片段）")
        for tag, hist in (("阶段 1（单步全读，`_s1`）", s1), ("阶段 2（30 步循环）", s2)):
            ln = _seg_line(tag, hist, tok_full, ep2step)
            if ln:
                lines.append(ln)
        if done:
            lines.append(f"- `best_trainable.pt`: **{done[0]:.2f} dB @ step {done[1]}**")
    if lines:
        out = os.path.join(ART, "summary.md")
        open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"[summary] {out}")
    else:
        print("[summary] 没有可用数据")


if __name__ == "__main__":
    main()
