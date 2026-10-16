#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AssocVTC 训练指标交互看板（Streamlit，6008 端口）。

数据来源： ``parse_metrics.py`` 生成的 ``metrics.json``（由 ``run_dash.sh`` 的刷新
循环每 30-60s 重写一次）。本应用**只读**该 JSON，不接触训练日志/训练进程。

启动::

    /root/dashvenv/bin/streamlit run app_streamlit.py \\
        --server.address 0.0.0.0 --server.port 6008 --server.headless true
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.environ.get("SRDASH_METRICS", os.path.join(HERE, "data", "metrics.json"))
DEFAULT_REFRESH = int(os.environ.get("SRDASH_REFRESH", "30"))

PLOTLY_TEMPLATE = "plotly_white"
BASELINE_RECON_FALLBACK = 0.3318

st.set_page_config(page_title="AssocVTC 训练指标看板", page_icon="📈",
                   layout="wide")


# --------------------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------------------

def load_bundle():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        return {"_error": f"无法读取 {METRICS_PATH}: {exc}"}


def chart(fig, **kwargs):
    """兼容新旧 Streamlit 的 plotly 宽度参数。"""
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def empty_note(msg):
    st.info(msg, icon="⏳")


def _xy(points, ykey, xkey="step"):
    xs, ys = [], []
    for p in points:
        x, y = p.get(xkey), p.get(ykey)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def base_layout(fig, title, ytitle, xtitle="train step"):
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE, height=380,
        margin=dict(l=60, r=20, t=50, b=50),
        xaxis_title=xtitle, yaxis_title=ytitle,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# --------------------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------------------

def render():
    bundle = load_bundle()
    if "_error" in bundle:
        st.error(bundle["_error"])
        st.stop()

    cfg = bundle.get("config", {})
    meta = bundle.get("run_meta", {})
    series = bundle.get("series", {})
    probes = bundle.get("probes", {})
    probe_hist = bundle.get("probe_history", {})
    infer = bundle.get("infer", {})
    baseline_recon = cfg.get("baseline_eval_recon", BASELINE_RECON_FALLBACK)
    runs = bundle.get("runs", [])

    # ---------------- 侧栏 ----------------
    with st.sidebar:
        st.header("⚙️ 控制")
        refresh = st.selectbox("自动刷新", [15, 30, 60, 120, 300],
                               index=[15, 30, 60, 120, 300].index(DEFAULT_REFRESH)
                               if DEFAULT_REFRESH in (15, 30, 60, 120, 300) else 1,
                               format_func=lambda s: f"{s} 秒")
        show_runs = st.multiselect(
            "显示的 run", runs, default=runs,
            format_func=lambda k: meta.get(k, {}).get("label", k))
        logy_grad = st.checkbox("grad_norm 用对数轴", value=True)
        logy_loss = st.checkbox("loss 用对数轴", value=False)
        st.divider()
        st.caption(f"metrics.json\n`{METRICS_PATH}`")
        st.caption(f"生成时间：{bundle.get('generated_at', '?')}")
        if st.button("🔄 立即刷新", width="stretch"):
            st.rerun()

    # ---------------- 顶部状态 ----------------
    st.title("📈 AssocVTC 训练指标看板")
    st.caption(
        "数据源：训练日志 `{}` + 探针 JSON + 推理 JSON ｜ 刷新由后台循环驱动".format(
            cfg.get("train_log_dir", "?"))
    )

    cols = st.columns(max(len(runs), 1) + 1)
    for i, run in enumerate(runs):
        s = series.get(run, {})
        with cols[i]:
            label = meta.get(run, {}).get("label", run)
            status = s.get("status")
            age = s.get("log_age_seconds")
            age_txt = ("刚刚" if (age is not None and age < 120)
                       else (f"{age/60:.0f} 分钟前" if age is not None else "无日志"))
            st.metric(
                f"{label} · step", s.get("last_step") or "—", delta=f"{status}",
                delta_color="normal" if status == "running"
                else ("inverse" if status in ("stalled", "missing") else "off"),
                help=f"日志：{s.get('log_path')}\n最后写入：{age_txt}")
            st.progress(min(max(s.get("progress") or 0.0, 0.0), 1.0),
                        text=f"进度 {((s.get('progress') or 0) * 100):.1f}%  "
                             f"({s.get('n_train_points', 0)} 条日志点 · 日志{age_txt})")
    with cols[-1]:
        n_ok = len(bundle.get("sources", {}).get("ok", []))
        n_miss = len(bundle.get("sources", {}).get("missing", []))
        st.metric("数据源", f"{n_ok} OK", delta=f"{n_miss} 缺失",
                  delta_color="normal" if n_miss == 0 else "inverse")

    tabs = st.tabs(["损失/梯度/学习率", "step1~5 探针", "E_px 热力图 & 曲线",
                    "推理结果 & 数据源"])

    # ================= Tab 1 =================
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for run in show_runs:
                xs, ys = _xy(series.get(run, {}).get("train", []), "loss")
                if xs:
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", name=meta.get(run, {}).get("label", run),
                        line=dict(color=meta.get(run, {}).get("color"))))
            base_layout(fig, "① train loss vs step", "loss")
            if logy_loss:
                fig.update_yaxes(type="log")
            chart(fig) if fig.data else empty_note("暂无 train loss 数据（等待日志积累）")

        with c2:
            fig = go.Figure()
            for run in show_runs:
                xs, ys = _xy(series.get(run, {}).get("train", []), "grad_norm")
                if xs:
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", name=meta.get(run, {}).get("label", run),
                        line=dict(color=meta.get(run, {}).get("color"))))
            base_layout(fig, "② 梯度平均变化 grad_norm vs step", "grad_norm")
            if logy_grad:
                fig.update_yaxes(type="log")
            chart(fig) if fig.data else empty_note("暂无 grad_norm 数据")

        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            for run in show_runs:
                xs, ys = _xy(series.get(run, {}).get("train", []), "learning_rate")
                if xs:
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", name=meta.get(run, {}).get("label", run),
                        line=dict(color=meta.get(run, {}).get("color"))))
            base_layout(fig, "③ learning_rate vs step", "learning_rate")
            chart(fig) if fig.data else empty_note("暂无学习率数据")

        with c4:
            fig = go.Figure()
            any_eval = False
            for run in show_runs:
                xs, ys = _xy(series.get(run, {}).get("eval", []), "eval_recon")
                if xs:
                    any_eval = True
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        name=meta.get(run, {}).get("label", run),
                        line=dict(color=meta.get(run, {}).get("color"), width=3),
                        marker=dict(size=9)))
            fig.add_hline(y=baseline_recon, line_dash="dash", line_color="gray",
                          annotation_text=f"基线 final eval_recon = {baseline_recon}",
                          annotation_position="bottom right")
            base_layout(fig, "④ eval_recon vs step（虚线=基线 0.3318）", "eval_recon")
            if any_eval:
                chart(fig)
            else:
                empty_note("新实验尚无 eval 行（eval 约每 2000 step 一次）。"
                           "基线 5 个 eval 点已就绪，切换侧栏 run 可见。")

        st.caption(
            "⚠️ 探针的 `step_px_scale` 是 **每步未累加增量的平均绝对值**；"
            "基线 step1≈1.03 → step5≈0.033 说明后步坍缩。"
        )

        # ---- 三路关键指标对照表（baseline / 旧 / 修复版）----
        st.subheader("三路关键指标对照")
        cmp_rows = []
        for run in runs:
            s = series.get(run, {})
            tr, ev = s.get("train") or [], s.get("eval") or []
            p = probes.get(run) or {}
            sps = p.get("step_px_scale") or [None] * 5
            last = tr[-1] if tr else {}
            last_ev = ev[-1] if ev else {}
            cmp_rows.append({
                "run": meta.get(run, {}).get("label", run),
                "状态": s.get("status"),
                "最后 step": s.get("last_step"),
                "train loss": last.get("loss"),
                "grad_norm": last.get("grad_norm"),
                "lr": last.get("learning_rate"),
                "eval_recon": last_ev.get("eval_recon"),
                "探针 step": p.get("step"),
                "探针 tag": p.get("tag"),
                "px_scale step1": sps[0] if len(sps) > 0 else None,
                "px_scale step5": sps[4] if len(sps) > 4 else None,
                "z_s_within_std": p.get("z_s_within_std"),
            })
        st.dataframe(cmp_rows, width="stretch", hide_index=True)
        st.caption(
            f"基线参考 eval_recon = **{baseline_recon}**（图④虚线）。"
            "`px_scale step1 → step5` 落差越小说明后步越没有贡献（坍缩）。"
        )

    # ================= Tab 2 =================
    with tabs[1]:
        st.subheader("step1~5 的平均值变化（探针 step_px_scale）")
        steps_axis = list(range(1, 6))

        def probe_bars(entry, label, color, key):
            if not entry:
                return False
            sps = entry.get("step_px_scale") or []
            if len(sps) < 5:
                return False
            st.plotly_chart(
                base_layout(go.Figure([go.Bar(
                    x=[f"step{i}" for i in steps_axis], y=sps,
                    marker_color=color, name=label)]),
                    f"{label} · step_px_scale（tag={entry.get('tag')}, "
                    f"step={entry.get('step')}, n_img={entry.get('n_img')}）",
                    "step_px_scale"), key=key)
            return True

        # 每个 run 一格柱状图（自动适配 N 路对照：基线 / 旧 / 修复版）
        bar_runs = show_runs or runs
        ncol = min(len(bar_runs), 3) or 1
        cols = st.columns(ncol)
        for idx, run in enumerate(bar_runs):
            with cols[idx % ncol]:
                p = probes.get(run)
                label = meta.get(run, {}).get("label", run)
                if not probe_bars(p, label, meta.get(run, {}).get("color", "#333"),
                                  f"bar_{run}"):
                    expected = os.path.join(cfg.get("probe_dir") or "", f"probe_{run}.json")
                    empty_note(f"探针缺失，等待生成：\n\n`{expected}`")

        st.divider()
        st.subheader("随训练进程的变化（多 checkpoint 探针）")
        fig = go.Figure()
        has_hist = False
        for run in bar_runs:
            hist = [e for e in probe_hist.get(run, []) if e.get("step") is not None]
            if not hist:
                continue
            has_hist = True
            for i in steps_axis:
                xs = [e["step"] for e in hist]
                ys = [(e.get("step_px_scale") or [None] * 5)[i - 1] for e in hist]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers",
                    name=f"{meta.get(run, {}).get('label', run)} · step{i}",
                    legendgroup=run))
        base_layout(fig, "step_px_scale 各采样步 vs 训练 step", "step_px_scale")
        fig.update_yaxes(type="log")
        if has_hist:
            chart(fig)
        else:
            empty_note("目前每个 run 只有 1 个探针文件，尚不能画随训练进程的曲线。\n\n"
                       "**如何加上新 checkpoint 的探针数据**：把新的探针 JSON 放到 "
                       f"`{cfg.get('probe_dir')}/` 下，文件名带上 step 号"
                       "（如 `probe_stack2x_lr1e4_slice05_step4000.json`），"
                       "或让 JSON 内含 `step` 字段 —— 刷新循环会自动纳入。详见 README。")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for run in bar_runs:
                e = probes.get(run)
                if not e:
                    continue
                fig.add_trace(go.Bar(x=steps_axis, y=e.get("z_s_block_cos"),
                                     marker_color=meta.get(run, {}).get("color"),
                                     name=meta.get(run, {}).get("label", run)))
            base_layout(fig, "块内 cos（z_s_block_cos）vs 采样步", "cos", "采样步")
            chart(fig) if fig.data else empty_note("暂无 z_s_block_cos")
        with c2:
            rows = []
            for run in bar_runs:
                e = probes.get(run)
                if not e:
                    continue
                d = e.get("derived", {})
                rows.append({
                    "run": run, "tag": e.get("tag"), "probe step": e.get("step"),
                    "z_s_within_std": e.get("z_s_within_std"),
                    "step_px_scale 均值": d.get("step_px_scale_mean"),
                    "step1": d.get("step_px_scale_step1"),
                    "step2~5 均值": d.get("step_px_scale_tail_mean"),
                    "后段/首步 比": d.get("step_px_scale_late_early_ratio"),
                })
            st.markdown("**探针汇总**")
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                empty_note("暂无探针数据")

    # ================= Tab 3 =================
    with tabs[2]:
        st.subheader("E_px：step × region 的 0-255 L1 矩阵")
        any_heat = False
        for run in bar_runs:
            e = probes.get(run)
            if not e or not e.get("E_px"):
                continue
            any_heat = True
            mat = e["E_px"]
            regions = e.get("regions") or [[i, i + 1] for i in range(len(mat[0]))]
            rlabels = [f"r{i+1}\n[{a},{b})" for i, (a, b) in enumerate(regions)]
            fig = go.Figure(go.Heatmap(
                z=mat, x=rlabels, y=[f"step{i}" for i in range(1, len(mat) + 1)],
                colorscale="Viridis", colorbar=dict(title="L1 (0-255)"),
                text=[[f"{v:.2f}" for v in row] for row in mat],
                texttemplate="%{text}"))
            fig.update_layout(
                title=f"E_px 热力图 · {meta.get(run, {}).get('label', run)} "
                      f"(tag={e.get('tag')}, step={e.get('step')})",
                template=PLOTLY_TEMPLATE, height=400,
                xaxis_title="region（token 区间）", yaxis_title="采样步")
            chart(fig, key=f"heat_{run}")
        if not any_heat:
            empty_note("暂无 E_px：等探针 JSON 生成。")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for run in bar_runs:
                e = probes.get(run)
                if not e:
                    continue
                fig.add_trace(go.Scatter(
                    x=steps_axis, y=e.get("prog_curve_255"), mode="lines+markers",
                    name=meta.get(run, {}).get("label", run),
                    line=dict(color=meta.get(run, {}).get("color"), width=3),
                    marker=dict(size=9)))
            base_layout(fig, "prog_curve_255：5 步累积重建 L1", "L1 (0-255)", "采样步")
            chart(fig) if fig.data else empty_note("暂无 prog_curve_255")
        with c2:
            fig = go.Figure()
            for run in bar_runs:
                e = probes.get(run)
                if not e:
                    continue
                fig.add_trace(go.Bar(
                    x=[meta.get(run, {}).get("label", run)],
                    y=[e.get("z_s_within_std")],
                    marker_color=meta.get(run, {}).get("color"),
                    name=meta.get(run, {}).get("label", run)))
            base_layout(fig, "z_s_within_std（步内 std，坍缩指标）", "std", "run")
            chart(fig) if fig.data else empty_note("暂无 z_s_within_std")

        st.divider()
        st.subheader("E_nrm（可选，归一化矩阵）")
        any_nrm = False
        for run in bar_runs:
            e = probes.get(run)
            if not e or not e.get("E_nrm"):
                continue
            any_nrm = True
            with st.expander(f"{meta.get(run, {}).get('label', run)} · E_nrm", expanded=False):
                mat = e["E_nrm"]
                regions = e.get("regions") or [[i, i + 1] for i in range(len(mat[0]))]
                fig = go.Figure(go.Heatmap(
                    z=mat, x=[f"r{i+1}[{a},{b})" for i, (a, b) in enumerate(regions)],
                    y=[f"step{i}" for i in range(1, len(mat) + 1)],
                    colorscale="Cividis", colorbar=dict(title="L1 norm")))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                                  xaxis_title="region", yaxis_title="采样步")
                chart(fig, key=f"enrm_{run}")
        if not any_nrm:
            empty_note("暂无 E_nrm")

    # ================= Tab 4 =================
    with tabs[3]:
        st.subheader("全量推理结果（训练后生成）")
        rows = []
        for run in runs:
            inf = infer.get(run)
            if not inf:
                continue
            d = inf.get("data", {})
            rows.append({
                "run": run, "path": inf.get("path"),
                "full_norm_l1": d.get("full_norm_l1"),
                "full_pixel_l1_255": d.get("full_pixel_l1_255"),
                "step_pixel_l1_255": d.get("step_pixel_l1_255"),
            })
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            fig = go.Figure()
            for run in runs:
                inf = infer.get(run)
                if not inf:
                    continue
                sp = inf.get("data", {}).get("step_pixel_l1_255")
                if sp:
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(sp) + 1)), y=sp, mode="lines+markers",
                        name=meta.get(run, {}).get("label", run)))
            base_layout(fig, "推理 step_pixel_l1_255", "L1 (0-255)", "采样步")
            if fig.data:
                chart(fig)
        else:
            empty_note(
                "推理 JSON 尚未生成。预期路径：\n\n"
                + "\n".join(f"- `{p}`" for p in [
                    os.path.join(cfg.get("project_root", ""), "output",
                                 "phase1_v2_stack2x_slice05", "infer_test.json"),
                    os.path.join(cfg.get("project_root", ""), "output",
                                 "baseline_blockdiag_slice05_infer_test.json")])
            )

        st.divider()
        st.subheader("数据源状态")
        src = bundle.get("sources", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**✅ 已就绪 ({len(src.get('ok', []))})**")
            for p in src.get("ok", []):
                st.markdown(f"- `{p}`")
        with c2:
            st.markdown(f"**⏳ 缺失 ({len(src.get('missing', []))})**")
            for p in src.get("missing", []):
                st.markdown(f"- `{p}`")
        for n in src.get("notes", []):
            st.caption("ℹ️ " + n)

        with st.expander("原始 metrics.json"):
            st.json(bundle)

    st.caption(f"页面渲染于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ "
               f"每 {refresh} 秒自动刷新（可在侧栏调整）")

    return refresh


def main():
    try:
        refresh = render()
    except Exception as exc:  # 渲染出错时不要把整个页面搞崩
        st.error(f"渲染异常：{exc}")
        st.exception(exc)
        refresh = DEFAULT_REFRESH

    # 自动刷新：优先用 st.fragment(run_every=...)，否则退回 HTML meta 定时 reload
    if hasattr(st, "fragment"):
        try:
            st.fragment(run_every=f"{refresh}s")(lambda: None)()
            return
        except Exception:
            pass
    try:
        import streamlit.components.v1 as components
        components.html(
            f"<script>setTimeout(function(){{window.parent.location.reload();}},"
            f"{refresh * 1000});</script>", height=0)
    except Exception:
        pass


main()
