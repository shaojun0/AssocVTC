#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兜底看板：把 metrics.json 渲染成一个自包含的静态 HTML（Plotly CDN）。

仅在 Streamlit 装不上时由 ``run_dash.sh`` 自动启用（``python -m http.server`` 发布）。
平时也可以手动生成一份用于离线分享 / 归档。

用法::

    python make_static_html.py --metrics data/metrics.json --out data/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"


def _traces(bundle, key):
    meta = bundle.get("run_meta", {})
    series = bundle.get("series", {})
    out = []
    for run in bundle.get("runs", []):
        pts = series.get(run, {}).get("train", [])
        xs = [p.get("step") for p in pts if p.get(key) is not None]
        ys = [p.get(key) for p in pts if p.get(key) is not None]
        if not xs:
            continue
        out.append({
            "x": xs, "y": ys, "mode": "lines", "type": "scatter",
            "name": meta.get(run, {}).get("label", run),
            "line": {"color": meta.get(run, {}).get("color", "#333")},
        })
    return out


def build_html(bundle: dict) -> str:
    cfg = bundle.get("config", {})
    meta = bundle.get("run_meta", {})
    probes = bundle.get("probes", {})
    infer = bundle.get("infer", {})
    baseline_recon = cfg.get("baseline_eval_recon", 0.3318)

    # --- eval_recon ---
    eval_traces, has_eval = [], False
    for run in bundle.get("runs", []):
        pts = bundle.get("series", {}).get(run, {}).get("eval", [])
        xs = [p.get("step") for p in pts if p.get("eval_recon") is not None]
        ys = [p.get("eval_recon") for p in pts if p.get("eval_recon") is not None]
        if xs:
            has_eval = True
            eval_traces.append({
                "x": xs, "y": ys, "mode": "lines+markers", "type": "scatter",
                "name": meta.get(run, {}).get("label", run),
                "line": {"color": meta.get(run, {}).get("color", "#333"), "width": 3},
            })
    eval_layout = {
        "title": f"eval_recon vs step（虚线=基线 {baseline_recon}）",
        "shapes": [{"type": "line", "x0": 0, "x1": 1, "xref": "paper",
                    "y0": baseline_recon, "y1": baseline_recon,
                    "line": {"dash": "dash", "color": "gray"}}],
    }

    # --- 探针柱状 ---
    probe_bars = []
    for run in bundle.get("runs", []):
        e = probes.get(run)
        sps = (e or {}).get("step_px_scale")
        if not sps:
            continue
        probe_bars.append({
            "x": [f"step{i}" for i in range(1, len(sps) + 1)], "y": sps,
            "type": "bar", "name": meta.get(run, {}).get("label", run),
        })

    heatmaps = []
    for run in bundle.get("runs", []):
        e = probes.get(run)
        if not e or not e.get("E_px"):
            continue
        mat = e["E_px"]
        regions = e.get("regions") or [[i, i + 1] for i in range(len(mat[0]))]
        heatmaps.append({
            "z": mat,
            "x": [f"r{i+1}[{a},{b})" for i, (a, b) in enumerate(regions)],
            "y": [f"step{i}" for i in range(1, len(mat) + 1)],
            "type": "heatmap", "colorscale": "Viridis",
            "title": f"E_px · {meta.get(run, {}).get('label', run)}",
        })

    # --- 数据源状态 ---
    src = bundle.get("sources", {})
    ok_rows = "".join(f"<li><code>{html.escape(p)}</code></li>" for p in src.get("ok", []))
    miss_rows = "".join(f"<li><code>{html.escape(p)}</code></li>" for p in src.get("missing", []))

    def panel(div_id, title, traces, layout=None, note=""):
        return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      {f'<p class="note">{html.escape(note)}</p>' if note else ''}
      <div id="{div_id}" class="plot"></div>
    </section>
    <script>
      Plotly.newPlot("{div_id}", {json.dumps(traces)}, Object.assign(
        {{template: "plotly_white", height: 380,
          margin: {{l: 60, r: 20, t: 50, b: 50}},
          legend: {{orientation: "h", y: 1.05}}}},
        {json.dumps(layout or {})}));
    </script>"""

    parts = [
        panel("p_loss", "① train loss vs step", _traces(bundle, "loss")),
        panel("p_grad", "② grad_norm vs step",
              _traces(bundle, "grad_norm"), {"yaxis": {"type": "log"}}),
        panel("p_lr", "③ learning_rate vs step", _traces(bundle, "learning_rate")),
    ]
    if has_eval:
        parts.append(panel("p_eval", "④ eval_recon vs step", eval_traces, eval_layout))
    else:
        parts.append('<section class="card"><h2>④ eval_recon vs step</h2>'
                     '<p class="note">新实验暂无 eval 行（约每 2000 step 一次）；'
                     f'基线参考值 {baseline_recon}</p></section>')
    parts.append(panel("p_probe", "⑤ step1~5 的平均值 step_px_scale", probe_bars,
                       note="每步未累加增量的平均绝对值"))
    for i, hm in enumerate(heatmaps):
        title = hm.pop("title")
        parts.append(panel(f"p_heat{i}", f"⑥ {title}", [hm]))

    infer_rows = "".join(
        f"<tr><td>{html.escape(r)}</td><td>{html.escape(str((infer.get(r) or {}).get('path')))}</td>"
        f"<td>{html.escape(str(((infer.get(r) or {}).get('data') or {}).get('full_pixel_l1_255')))}</td></tr>"
        for r in bundle.get("runs", []))

    status_line = " · ".join(
        f"{html.escape(meta.get(r, {}).get('label', r))}: "
        f"{bundle.get('series', {}).get(r, {}).get('status', '?')} "
        f"step={bundle.get('series', {}).get(r, {}).get('last_step')}"
        for r in bundle.get("runs", []))

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AssocVTC 训练指标看板（静态）</title>
<script src="{PLOTLY_CDN}"></script>
<style>
 body {{ font-family: -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
        margin: 0; background: #f7f8fa; color: #1c1e21; }}
 header {{ padding: 20px 28px; background: #fff; border-bottom: 1px solid #e4e6eb; }}
 h1 {{ margin: 0 0 6px; font-size: 22px; }}
 .sub {{ color: #606770; font-size: 13px; }}
 .wrap {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
          gap: 16px; padding: 20px 28px 60px; }}
 .card {{ background: #fff; border: 1px solid #e4e6eb; border-radius: 10px;
          padding: 8px 14px 4px; }}
 .card h2 {{ font-size: 15px; margin: 8px 0 2px; }}
 .note {{ color: #606770; font-size: 12px; margin: 2px 0 6px; }}
 .plot {{ width: 100%; }}
 table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
 td, th {{ border: 1px solid #e4e6eb; padding: 6px 8px; text-align: left; }}
 code {{ background: #f0f2f5; padding: 1px 4px; border-radius: 4px; font-size: 12px; }}
 .full {{ grid-column: 1 / -1; }}
</style></head>
<body>
<header>
  <h1>📈 AssocVTC 训练指标看板（静态兜底版）</h1>
  <div class="sub">生成时间 {html.escape(str(bundle.get('generated_at')))}
    ｜ {status_line}<br/>
    自动刷新：由服务器后台循环重写本页；浏览器每 60 秒 reload 一次。
  </div>
</header>
<div class="wrap">
{''.join(parts)}
  <section class="card full">
    <h2>⑦ 全量推理结果</h2>
    <table><tr><th>run</th><th>path</th><th>full_pixel_l1_255</th></tr>
    {infer_rows or '<tr><td colspan="3">尚未生成</td></tr>'}</table>
  </section>
  <section class="card">
    <h2>✅ 已就绪数据源 ({len(src.get('ok', []))})</h2><ul>{ok_rows}</ul>
  </section>
  <section class="card">
    <h2>⏳ 缺失数据源 ({len(src.get('missing', []))})</h2><ul>{miss_rows}</ul>
  </section>
</div>
<script>setTimeout(function(){{ location.reload(); }}, 60000);</script>
</body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成静态看板 HTML")
    default_m = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metrics.json")
    ap.add_argument("--metrics", default=os.environ.get("SRDASH_METRICS", default_m))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.metrics):
        print(f"[make_static_html] metrics 不存在: {args.metrics}", file=sys.stderr)
        return 1
    out = args.out or os.path.join(os.path.dirname(args.metrics), "index.html")
    with open(args.metrics, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(build_html(bundle))
    os.replace(tmp, out)
    print(f"[make_static_html] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
