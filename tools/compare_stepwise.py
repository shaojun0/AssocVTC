#!/usr/bin/env python
"""对比多个 eval_stepwise.json（不同 run / 架构 / 配方），输出 markdown 表。

用法::

    python3 tools/compare_stepwise.py --out cmp.md \
        "v10_120f(旧配方)"=path/a.json "v10_120f_2stage"=path/b.json

口径：末档（读全量）用 `per_clip` 求 mean±std（与训练内 2-clip 指标不可直接比），
上界 = 同一批片段的冻结分词器满读 PSNR；另列若干读档的整片 PSNR 作为率-质量曲线。
"""
from __future__ import annotations

import argparse
import json
import statistics as st

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default="")
_ap.add_argument("runs", nargs="+", help="label=path/to/eval_stepwise.json")
_A = _ap.parse_args()


def load(spec: str):
    if "=" not in spec:
        raise SystemExit(f"参数需为 label=path: {spec}")
    label, path = spec.split("=", 1)
    d = json.load(open(path, encoding="utf-8"))
    c = d["curves"]
    pc = d.get("per_clip") or []
    n = int(c["n_steps"])
    fin = [x["psnr_all_steps"][-1] for x in pc] if pc else [c["psnr_all_mean"][-1]]
    ub = [x["psnr_tok_full"] for x in pc] if pc else [c["psnr_tok_full_mean"]]
    gap = [a - b for a, b in zip(ub, fin)]
    zc = [x["zcls_l1_steps"][-1] for x in pc] if pc else [c["zcls_l1_mean"][-1]]
    sp = [x["spec_l1_steps"][-1] for x in pc] if pc else [c["spec_l1_mean"][-1]]
    return dict(label=label, path=path, n=n, n_clips=int(c["n_clips"]),
                fin=st.mean(fin), fin_std=st.pstdev(fin),
                ub=st.mean(ub), gap=st.mean(gap),
                zc=st.mean(zc), sp=st.mean(sp),
                curve=[c["psnr_all_mean"][k] for k in
                       (0, 4, 9, 14, 19, 24, n - 1)])


runs = [load(s) for s in _A.runs]
lines = ["# stepwise 对比（60 val 片段）", ""]
lines.append("| run | 片段 | 末档 PSNR | 距上界 | 上界 | z_cls L1 | special L1 | "
             "t=1 | t=5 | t=10 | t=15 | t=20 | t=25 | t=30 |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in runs:
    cs = " | ".join(f"{v:.2f}" for v in r["curve"])
    lines.append(f"| {r['label']} | {r['n_clips']} | **{r['fin']:.2f} ± {r['fin_std']:.2f}** | "
                 f"{r['gap']:.2f} | {r['ub']:.2f} | {r['zc']:.4f} | {r['sp']:.4f} | {cs} |")
txt = "\n".join(lines) + "\n"
print(txt)
if _A.out:
    open(_A.out, "w", encoding="utf-8").write(txt)
    print(f"[out] {_A.out}")
