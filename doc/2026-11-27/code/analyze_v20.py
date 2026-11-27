#!/usr/bin/env python
"""analyze_v20.py — v2.0(224²) 三阶段 + 阶段3-CE 的 rFID/PSNR 逐步曲线与汇总表。

输入（fast_rfid.py 产物）:
    <st>/eval/s1_rfid.json      阶段1 T=1（像素头）
    <st>/eval/s2_rfid.json      阶段2 T=16（像素头）
    <st>/eval/s3_rfid.json      阶段3-CE（argmax 离散码 → VQ 解码）
    <st>/eval/s3_last_rfid.json 阶段3-CE ckpt_last（可选）
    <st>/p0_gate_llamagen_5000.json   P0 天花板（n=5000；缺失时回退 p0_gate_llamagen.json 并警告）

输出:
    <st>/fig_step_rfid.png   x=累计 z_s(=2t)，曲线: s1 单点 / s2 像素 / s3-CE / P0 天花板水平线
    <st>/fig_step_psnr.png   同 x，s2 vs s3-CE 的 PSNR（+ P0 PSNR 参考线）
    <st>/TABLE.md            末步汇总 + 逐步表 + 缺文件清单

缺文件/坏 JSON 只跳过并写进表（不崩）；matplotlib 不可用时只写 TABLE.md。

用法:
    python analyze_v20.py --st /root/autodl-tmp/v20_224
"""

import argparse
import datetime
import json
import os

# (key, 文件名, 图例, 颜色, 线型, 是否画进图)
SERIES = [
    ("s1", "s1_rfid.json", "阶段1 T=1（像素头）", "#1f77b4", "none"),
    ("s2", "s2_rfid.json", "阶段2 T=16（像素头, 像素 L1）", "#2ca02c", "-"),
    ("s3", "s3_rfid.json", "阶段3-CE（token→VQ 解码）", "#d62728", "-"),
    ("s3_last", "s3_last_rfid.json", "阶段3-CE ckpt_last", "#d62728", "--"),
]
PLOT_KEYS = ("s1", "s2", "s3")          # 图里只画这 3 条 + 天花板线（s3_last 只进表）
# 本机无 CJK 字体（19 个字体全是 DejaVu 系）⇒ 图内一律英文, 中文说明放 TABLE.md, 避免豆腐块
EN_LABEL = {"s1": "s1 T=1 (pixel head)", "s2": "s2 T=16 (pixel head, L1)",
            "s3": "s3-CE (tokens -> VQ decode)", "s3_last": "s3-CE ckpt_last"}
P0_CANDIDATES = ["p0_gate_llamagen_5000.json", "p0_gate_llamagen.json"]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_series(path):
    """→ {"meta":..., "rows":[{"t","x","rfid","psnr","l1"}...]} 或 None。"""
    d = load_json(path)
    if not d:
        return None
    ps = d.get("per_step") or {}
    rows = []
    for k in sorted(ps, key=lambda s: int(s)):
        v = ps[k] or {}
        t = int(v.get("step", k))
        rows.append(dict(t=t,
                         x=float(v.get("cum_z_s", 2 * t)),   # 链上 = fixed/block2 ⇒ cum = 2t
                         rfid=v.get("rfid"), psnr=v.get("psnr"), l1=v.get("l1_255")))
    if not rows:
        return None
    rows.sort(key=lambda r: r["t"])
    return dict(meta=d, rows=rows)


def load_p0(st):
    for name in P0_CANDIDATES:
        d = load_json(os.path.join(st, name))
        if d:
            d["_file"] = name
            if "5000" not in name:
                print(f"[analyze] ⚠️ P0 用回退文件 {name}（n={d.get('n')}）; "
                      f"主结论应使用 p0_gate_llamagen_5000.json（n=5000）", flush=True)
            return d
    return None


def fmt(v, nd=3):
    return "—" if v is None else f"{float(v):.{nd}f}"


def write_table(st, loaded, p0, out_path):
    L = []
    add = L.append
    add("# v2.0 (224²) 三阶段 + 阶段3-CE：rFID / PSNR / L1 汇总")
    add("")
    add(f"生成: `analyze_v20.py --st {st}` · {datetime.datetime.now().isoformat(timespec='seconds')}")
    add("口径: rFID = FID-InceptionV3 pool3 2048d（resize 299, uint8 量化, 内存累积）; "
        "PSNR/L1 在 [0,1] / 0-255 上；x = 累计读取 z_s 数（链上 fixed/block2 ⇒ 2t）。")
    add("")
    add("## 0. 源文件")
    add("")
    add("| 文件 | 状态 | n_images | T | 备注 |")
    add("|---|---|---|---|---|")
    for key, fname, label, _c, _ls in SERIES:
        s = loaded.get(key)
        if s:
            m = s["meta"]
            note = str(m.get("tag", "")) or "—"
            add(f"| eval/{fname} | ok | {m.get('n_images', '?')} | {m.get('T', len(s['rows']))} | {note} |")
        elif key == "s3_last":
            add(f"| eval/{fname} | missing（可选） | — | — | ckpt_last 评测不存在 |")
        else:
            add(f"| eval/{fname} | **missing** | — | — | 未评测/未跑到 |")
    add("")
    add("## 1. 末步汇总（最后一步 = 累计 32 个 z_s）")
    add("")
    add("| 臂 | 累计 z_s | rFID ↓ | PSNR ↑ | L1_255 ↓ |")
    add("|---|---|---|---|---|")
    for key, fname, label, _c, _ls in SERIES:
        s = loaded.get(key)
        if not s:
            add(f"| {label} | — | — | — | — |")
            continue
        last = s["rows"][-1]
        add(f"| {label} | {last['x']:.0f} | {fmt(last['rfid'])} | {fmt(last['psnr'])} | {fmt(last['l1'])} |")
    if p0:
        add(f"| **P0 天花板** LlamaGen VQ-16（n={p0.get('n', '?')}） | — | "
            f"**{fmt(p0.get('rfid'))}** | {fmt(p0.get('psnr'))} | {fmt(p0.get('l1_255'))} |")
    else:
        add("| P0 天花板 | — | missing | — | — |")
    add("")
    if p0:
        add(f"> P0 = tokenizer 自身重建上限（`{p0['_file']}`，n={p0.get('n')}, 100 类全覆盖）"
            f"⇒ s3-CE 的 rFID 不可能低于 **{fmt(p0.get('rfid'))}**。FID 绝对值不可与论文 FID 直接比"
            f"（n=5000 与 50k 不同；n<2048 时协方差奇异会虚高，见 P0_REPORT.md §2）。")
        add("")
    add("## 2. 逐步表")
    add("")
    have = [k for k in PLOT_KEYS if loaded.get(k)]
    xs = sorted({r["t"] for k in have for r in loaded[k]["rows"]})
    add("| t | 累计 z_s | s1 rFID | s2 rFID | s3-CE rFID | s2 PSNR | s3-CE PSNR |")
    add("|---|---|---|---|---|---|---|")

    def cell(key, t, field):
        s = loaded.get(key)
        if not s:
            return "—"
        for r in s["rows"]:
            if r["t"] == t:
                return fmt(r[field])
        return "—"

    for t in xs:
        x = next((r["x"] for k in have for r in loaded[k]["rows"] if r["t"] == t), 2 * t)
        add(f"| {t} | {x:.0f} | {cell('s1', t, 'rfid')} | {cell('s2', t, 'rfid')} | "
            f"{cell('s3', t, 'rfid')} | {cell('s2', t, 'psnr')} | {cell('s3', t, 'psnr')} |")
    add("")
    add("## 3. 缺文件 / 复跑")
    add("")
    missing = [f"eval/{f}" for k, f, *_ in SERIES if not loaded.get(k) and k != "s3_last"]
    add(f"- 缺: {', '.join(missing) if missing else '无'}")
    add("- 复跑: `python analyze_v20.py --st /root/autodl-tmp/v20_224`")
    add("")
    with open(out_path, "w") as f:
        f.write("\n".join(L))
    return L


def write_figures(st, loaded, p0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:      # matplotlib 缺失 ⇒ 只写表（按要求退化）
        print(f"[analyze] matplotlib 不可用（{type(e).__name__}: {e}）⇒ 只写 TABLE.md", flush=True)
        return []

    made = []
    colors = {k: c for k, _f, _l, c, _ls in SERIES}
    styles = {k: ls for k, _f, _l, _c, ls in SERIES}

    # ── fig 1: rFID ──
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key in PLOT_KEYS:
        s = loaded.get(key)
        if not s:
            continue
        xs = [r["x"] for r in s["rows"]]
        ys = [r["rfid"] for r in s["rows"]]
        if any(y is None for y in ys):
            continue
        if key == "s1":
            ax.plot(xs, ys, marker="o", ls="none", ms=9, color=colors[key],
                    label=f"{EN_LABEL[key]}: {ys[-1]:.2f}")
        else:
            ax.plot(xs, ys, marker="o", ms=4, ls=styles[key], color=colors[key],
                    label=f"{EN_LABEL[key]}: {ys[-1]:.2f}")
    if p0 and p0.get("rfid") is not None:
        ax.axhline(float(p0["rfid"]), color="0.35", ls=":", lw=2,
                   label=f"P0 ceiling LlamaGen VQ-16 = {float(p0['rfid']):.3f} (n={p0.get('n')})")
    ax.set_xlabel("cumulative z_s read (= 2t)")
    ax.set_ylabel("rFID down (FID-InceptionV3 pool3, n=5000)")
    ax.set_title("v2.0 (224^2) step-wise rFID: pixel head vs Stage3-CE vs tokenizer ceiling")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:      # 数据全缺时避免 legend 空警告
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p1 = os.path.join(st, "fig_step_rfid.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(p1)

    # ── fig 2: PSNR ──
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key in ("s2", "s3"):
        s = loaded.get(key)
        if not s:
            continue
        xs = [r["x"] for r in s["rows"]]
        ys = [r["psnr"] for r in s["rows"]]
        if any(y is None for y in ys):
            continue
        ax.plot(xs, ys, marker="o", ms=4, ls="-", color=colors[key],
                label=f"{EN_LABEL[key]}: {ys[-1]:.2f} dB")
    if p0 and p0.get("psnr") is not None:
        ax.axhline(float(p0["psnr"]), color="0.35", ls=":", lw=2,
                   label=f"P0 VQ-16 recon PSNR = {float(p0['psnr']):.2f} dB (n={p0.get('n')})")
    ax.set_xlabel("cumulative z_s read (= 2t)")
    ax.set_ylabel("PSNR up (dB)")
    ax.set_title("v2.0 (224^2) step-wise PSNR: s2 pixel head vs s3-CE (VQ render)")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p2 = os.path.join(st, "fig_step_psnr.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    made.append(p2)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", default="/root/autodl-tmp/v20_224", help="实验根目录（含 eval/）")
    ap.add_argument("--p0", default="", help="P0 JSON 路径（默认 <st>/p0_gate_llamagen_5000.json）")
    args = ap.parse_args()
    st = os.path.abspath(args.st)
    ev = os.path.join(st, "eval")

    loaded = {}
    for key, fname, *_ in SERIES:
        s = load_series(os.path.join(ev, fname))
        loaded[key] = s
        print(f"[analyze] {fname}: {'ok T=' + str(len(s['rows'])) if s else 'missing'}", flush=True)

    if args.p0:
        p0 = load_json(args.p0)
        if p0:
            p0["_file"] = os.path.basename(args.p0)
    else:
        p0 = load_p0(st)
    print(f"[analyze] P0: {p0['_file'] if p0 else 'missing'}"
          + (f" rFID={float(p0['rfid']):.3f} n={p0.get('n')}" if p0 and p0.get("rfid") is not None else ""),
          flush=True)

    os.makedirs(st, exist_ok=True)
    table = write_table(st, loaded, p0, os.path.join(st, "TABLE.md"))
    print(f"[analyze] -> {os.path.join(st, 'TABLE.md')}", flush=True)
    figs = write_figures(st, loaded, p0)
    for p in figs:
        print(f"[analyze] -> {p}", flush=True)
    if not figs:
        print("[analyze] 无图（matplotlib 缺失或没数据）", flush=True)
    # 控制台简报
    for key, fname, label, _c, _ls in SERIES:
        s = loaded.get(key)
        if s:
            last = s["rows"][-1]
            print(f"[analyze] {label}: cum_z={last['x']:.0f} rFID={fmt(last['rfid'])} "
                  f"PSNR={fmt(last['psnr'])} L1_255={fmt(last['l1'])}", flush=True)


if __name__ == "__main__":
    main()
