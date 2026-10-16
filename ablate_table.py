#!/usr/bin/env python3
"""六臂架构矩阵对比表 — AssocVTC 编码器/解码器注意力形式 × 读出方式
================================================================================

从 `eval_linattn.py` 的多臂输出 json 生成"架构矩阵"表。臂的归类**不靠文件名**,
而是读 checkpoint 里的 `cfg` 字段（`enc_attn` / `dec_attn` / `dec_recurrent`）——
这样即使日后改了目录命名, 归类也不会错（缺失字段按历史默认值补齐: 线性编码器 +
普通 softmax 解码器）。

    python ablate_table.py \
        --eval /root/autodl-tmp/srdiff_alllinear/eval_ablation6.json \
        --stepwise-new alllinear_k64_rec=/root/autodl-tmp/srdiff_alllinear/stepwise_alllinear_rec.json \
        --stepwise-old k64_rec=doc/2026-10-15/artifacts/recurrent/stepwise.json \
        --out doc/2026-10-16/artifacts/ablation_matrix.md

输出:
  1. 架构矩阵（行 = 编码器注意力 × 解码器注意力; 列 = 单步 T=1 / 循环 BPTT T=8）, 格 = PSNR
  2. 完整数值表（PSNR / SSIM / L1 / 基线 / 参数量 / 步数）
  3. 配对差汇总（对每个"只动一个因素"的配对给出 Δ）
"""

import argparse
import json
import os
import sys

_IMG_STD_MEAN = (0.229 + 0.224 + 0.225) / 3.0
L1_NORM_TO_255 = _IMG_STD_MEAN * 255.0     # 归一化空间 L1 → 0-255 像素口径（见摘注）

ENC_LABEL = {"linear": "线性注意力", "linattn": "线性注意力", "std": "普通 softmax",
             "standard": "普通 softmax", "softmax": "普通 softmax"}
DEC_LABEL = {"std": "普通 softmax", "linear": "线性注意力"}


def load(p):
    with open(p) as f:
        return json.load(f)


def arm_of(rec, name):
    """从 ckpt 的 cfg 读出架构归属（缺失字段按历史默认补齐）。"""
    cfg = rec.get("cfg") or {}
    enc = str(cfg.get("enc_attn", "linear")).lower()
    dec = str(cfg.get("dec_attn", "std")).lower()
    rec_flag = bool(cfg.get("dec_recurrent", False))
    return {
        "dir": name,
        "enc": "linear" if enc in ("linear", "linattn") else "std",
        "dec": "linear" if dec == "linear" else "std",
        "recurrent": rec_flag,
        "psnr": rec.get("psnr"),
        "ssim": rec.get("ssim"),
        "l1": rec.get("l1_norm"),
        "l1_255": None if rec.get("l1_norm") is None else rec["l1_norm"] * L1_NORM_TO_255,
        "dc": rec.get("psnr_patch_dc_baseline"),
        "mean": rec.get("psnr_mean_image_baseline"),
        "gain": rec.get("psnr_gain_over_patch_dc"),
        "params": rec.get("params") or {},
        "step": rec.get("step"),
        "epoch": rec.get("epoch"),
        "n": rec.get("n_images"),
        "T": len((cfg.get("dec_steps") or "").split(",")) if cfg.get("dec_steps") else
             (8 if rec_flag else 1),
    }


def fmt(v, nd=3, dash="—"):
    return dash if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--stepwise-new", action="append", default=[],
                    help="标签=路径（可多次）")
    ap.add_argument("--stepwise-old", action="append", default=[],
                    help="标签=路径（可多次）")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ev = load(a.eval)
    arms = [arm_of(v, k.split("/")[0]) for k, v in ev.items()]
    arms.sort(key=lambda x: (x["recurrent"], x["enc"], x["dec"]))
    by = {(x["enc"], x["dec"], x["recurrent"]): x for x in arms}

    out = ["# 六臂架构矩阵 — 编码器/解码器注意力形式 × 读出方式", ""]
    out.append(f"- 归类依据 = checkpoint 里的 `cfg.enc_attn / cfg.dec_attn / cfg.dec_recurrent`（不靠文件名）")
    out.append(f"- 评测：**全量留出集，同一进程、同一批次**（n={arms[0]['n'] if arms else '?'}）")
    out.append(f"- PSNR 口径：逐图 PSNR 再平均（与 2026-10-15 报告同口径）")
    out.append(f"- `L1(0-255)` 由归一化空间换算：`l1_norm × 0.226 × 255 = ×{L1_NORM_TO_255:.2f}`")
    out.append("")

    # ── 1. 架构矩阵（PSNR）──
    out.append("## 1. 架构矩阵 — PSNR (dB)")
    out.append("")
    out.append("| 编码器注意力 \\ 解码器 / 读出 | 普通 softmax · 单步 T=1 | 普通 softmax · **BPTT** T=8 "
               "| 线性注意力 · 单步 T=1 | 线性注意力 · **BPTT** T=8 |")
    out.append("|---|---|---|---|---|")
    for enc in ("std", "linear"):
        cells = []
        for dec in ("std", "linear"):
            for rec in (False, True):
                x = by.get((enc, dec, rec))
                cells.append(fmt(x["psnr"], 3) if x else "（未跑）")
        out.append(f"| **{ENC_LABEL[enc]}** | " + " | ".join(cells) + " |")
    out.append("")

    # ── 2. 完整数值表 ──
    out.append("## 2. 完整数值表")
    out.append("")
    out.append("| 编码器 | 解码器 | 读出 | 臂目录 | PSNR↑ | SSIM↑ | L1(归一)↓ | L1(0-255)↓ | "
               "DC基线 | 相对DC增益 | enc M | dec M | 步/ep | n |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in arms:
        p = x["params"]
        out.append(
            f"| {ENC_LABEL[x['enc']]} | {DEC_LABEL[x['dec']]} | "
            f"{'**BPTT T=8**' if x['recurrent'] else '单步 T=1'} | `{x['dir']}` | "
            f"**{fmt(x['psnr'])}** | {fmt(x['ssim'], 4)} | {fmt(x['l1'], 4)} | {fmt(x['l1_255'])} | "
            f"{fmt(x['dc'], 2)} | {fmt(x['gain'], 2)} | {p.get('enc_M', '—')} | {p.get('dec_M', '—')} | "
            f"{x['step']}/{x['epoch']} | {x['n']} |")
    out.append("")

    # ── 3. 配对差（只动一个因素）──
    out.append("## 3. 配对差汇总（每个配对只动一个因素）")
    out.append("")
    out.append("| 配对（A → B） | 变动的因素 | ΔPSNR (dB) | ΔSSIM | ΔL1(归一) |")
    out.append("|---|---|---|---|---|")
    pairs = [
        (("linear", "std", False), ("std", "std", False), "编码器: 线性 → 普通 softmax（单步）"),
        (("linear", "std", True), ("std", "std", True), "编码器: 线性 → 普通 softmax（BPTT）"),
        (("linear", "std", False), ("linear", "linear", False), "解码器: 普通 softmax → 线性（单步）"),
        (("linear", "std", True), ("linear", "linear", True), "解码器: 普通 softmax → 线性（**BPTT**）"),
        (("std", "std", False), ("std", "std", True), "读出: 单步 → BPTT（全 softmax）"),
        (("linear", "std", False), ("linear", "std", True), "读出: 单步 → BPTT（线性 enc）"),
        (("linear", "linear", False), ("linear", "linear", True), "读出: 单步 → BPTT（**全线性**）"),
    ]
    for ka, kb, desc in pairs:
        xa, xb = by.get(ka), by.get(kb)
        if not xa or not xb:
            out.append(f"| {desc} | — | （缺臂） | | |")
            continue
        out.append(f"| `{xa['dir']}` → `{xb['dir']}` | {desc} | "
                   f"**{xb['psnr'] - xa['psnr']:+.3f}** | {xb['ssim'] - xa['ssim']:+.4f} | "
                   f"{xb['l1'] - xa['l1']:+.4f} |")
    out.append("")

    # ── 4. 逐步曲线（可选）──
    for tag, path in [s.split("=", 1) for s in
                      (a.stepwise_new + a.stepwise_old) if "=" in s]:
        if not os.path.exists(path):
            continue
        d = load(path)
        out.append(f"### 逐步（BPTT）: {tag}")
        out.append("")
        out.append("| step | t | 累计 z_s | L1_255 | PSNR | ΔL1 |")
        out.append("|---|---|---|---|---|---|")
        cum = d.get("cum_z_s") or [None] * len(d.get("steps") or [])
        dl = d.get("delta_L1_255") or [None] * len(d.get("steps") or [])
        for i, t in enumerate(d.get("steps") or []):
            d_i = dl[i] if i < len(dl) else None
            out.append(f"| {i + 1} | {t} | {cum[i]} | {fmt(d['L1_255'][i])} | {fmt(d['PSNR'][i])} | "
                       f"{'—' if d_i is None else f'{d_i:+.3f}'} |")
        out.append("")
        out.append(f"总落差 {fmt(d.get('total_drop_255'))} px; 最优 step "
                   f"{(d.get('best_step') or 0) + 1}; Δ₁ 占比 "
                   f"{fmt((d.get('delta1_share') or 0) * 100, 1)}%; "
                   f"正增量步数 {d.get('n_positive_steps')}; "
                   f"最优停步分布 {d.get('best_step_hist')}")
        out.append("")

    text = "\n".join(out) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)
        print(f"[out] {a.out}")
    print(text)


if __name__ == "__main__":
    main()
