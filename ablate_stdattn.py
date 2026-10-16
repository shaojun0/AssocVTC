#!/usr/bin/env python3
"""消融实验对比表生成器 — 编码器: 线性注意力 vs 普通 transformer
================================================================================

读取 `eval_linattn.py` 的四臂全量留出集结果 + 两臂逐步重建结果, 生成
markdown 对比表（配对比较: 同解码方式下"只换编码器"）。

    python ablate_stdattn.py \
        --eval /root/autodl-tmp/srdiff_stdattn/eval_ablation_full.json \
        --stepwise-new /root/autodl-tmp/srdiff_stdattn/stepwise_stdattn_rec.json \
        --stepwise-old doc/2026-10-15/artifacts/recurrent/stepwise.json \
        --out doc/2026-10-16/artifacts/ablation_table.md

设计原则: 只做**配对**展示（linear vs std 在同一读出方式下）, 并把"不可比"的地方
（预算/种子/数据）显式标注出来 —— 避免把不同口径的数字放进同一列。
"""

import argparse
import json
import os
import sys

# eval json 的 key = basename(dirname(ckpt)) + "/" + basename(ckpt)
ARM_SINGLE_OLD = "k64"
ARM_REC_OLD = "k64_rec"
ARM_SINGLE_NEW = "stdattn_k64"
ARM_REC_NEW = "stdattn_k64_rec"

# ⚠️ 单位换算（踩过的坑, 别想当然）: eval json 里的 `l1_norm` 是 **ImageNet 归一化空间**
# 的 L1（(x−mean)/std 之后）。要换成「0-255 像素」口径, 不能乘 255 —— 那样会得到
# 0.1463×255 = 37.3, 而昨日报告的 8.433 是**像素空间**的量。正确的换算是先乘回
# 通道 std（三个通道取均值 0.226）, 再乘 255:
#     0.1463 (归一化) × 0.226 (std) × 255 = 8.43  ✓ 与 doc/2026-10-15 REPORT §3.1 一致
#     0.16662 (归一化) × 0.226 × 255 = 9.60     ✓ 与 k64_rec 逐步表末步 9.603 一致
# （`stepwise_*.json` 里的 L1_255 是**直接在像素空间实测**的, 与此换算无关。）
_IMG_STD_MEAN = (0.229 + 0.224 + 0.225) / 3.0
L1_NORM_TO_255 = _IMG_STD_MEAN * 255.0


def load_json(path):
    with open(path) as f:
        return json.load(f)


def index_by_dir(eval_json):
    """{"<dir>/<file>": rec} → {"<dir>": rec}"""
    out = {}
    for k, v in eval_json.items():
        out[k.split("/")[0]] = v
    return out


def fmt(v, nd=3, suffix=""):
    return "—" if v is None else f"{v:.{nd}f}{suffix}"


def arm_row(name, rec):
    """一条臂在"重建质量"上的全部口径。"""
    return {
        "arm": name,
        "step": rec.get("step"),
        "epoch": rec.get("epoch"),
        "n": rec.get("n_images"),
        "psnr": rec.get("psnr"),
        "ssim": rec.get("ssim"),
        "l1": rec.get("l1_norm"),
        "l1_255": None if rec.get("l1_norm") is None else rec["l1_norm"] * L1_NORM_TO_255,
        "dc": rec.get("psnr_patch_dc_baseline"),
        "mean": rec.get("psnr_mean_image_baseline"),
        "gain": rec.get("psnr_gain_over_patch_dc"),
        "enc_M": (rec.get("params") or {}).get("enc_M"),
        "params": rec.get("params"),
    }


def pair_block(title, rec_old, rec_new, old_label, new_label, note=""):
    """一个"配对比较"小节: 线性编码器 vs 普通 transformer 编码器。"""
    a, b = arm_row(old_label, rec_old), arm_row(new_label, rec_new)
    lines = [f"### {title}", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += [
        "| 指标 | 线性注意力编码器 | 普通 transformer 编码器 | Δ（普通 − 线性） |",
        "|---|---|---|---|",
        f"| 编码器参数量 | {a['enc_M']} M | {b['enc_M']} M | "
        f"{b['enc_M'] - a['enc_M']:+.3f} M |",
        f"| **PSNR (dB) ↑** | **{fmt(a['psnr'])}** | **{fmt(b['psnr'])}** | "
        f"**{b['psnr'] - a['psnr']:+.3f}** |",
        f"| **SSIM ↑** | **{fmt(a['ssim'], 4)}** | **{fmt(b['ssim'], 4)}** | "
        f"**{b['ssim'] - a['ssim']:+.4f}** |",
        f"| L1 (归一化空间) ↓ | {fmt(a['l1'], 4)} | {fmt(b['l1'], 4)} | "
        f"{b['l1'] - a['l1']:+.4f} |",
        f"| L1 (0-255 像素) ↓ | {fmt(a['l1_255'], 3)} | {fmt(b['l1_255'], 3)} | "
        f"{b['l1_255'] - a['l1_255']:+.3f} |",
        f"| per-patch DC 基线 PSNR | {fmt(a['dc'], 2)} | {fmt(b['dc'], 2)} | "
        f"{b['dc'] - a['dc']:+.2f} |",
        f"| 相对 DC 基线增益 (dB) | {fmt(a['gain'], 2)} | {fmt(b['gain'], 2)} | "
        f"{b['gain'] - a['gain']:+.2f} |",
        f"| 常量灰基线 PSNR | {fmt(a['mean'], 2)} | {fmt(b['mean'], 2)} | "
        f"{b['mean'] - a['mean']:+.2f} |",
        f"| 评测图数 | {a['n']} | {b['n']} | — |",
        f"| 训练步 / epoch | {a['step']} / {a['epoch']} | {b['step']} / {b['epoch']} | — |",
        "",
    ]
    return lines


def stepwise_block(sw_new, sw_old, new_label, old_label):
    """逐步（循环/BPTT）重建曲线对比。"""
    if sw_new is None:
        return []
    lines = ["### 逐步（循环 BPTT）重建曲线对比", ""]
    st = sw_new.get("steps")
    l_new = sw_new.get("L1_255") or []
    l_old = (sw_old or {}).get("L1_255") or []
    ps_new = sw_new.get("PSNR") or []
    ps_old = (sw_old or {}).get("PSNR") or []
    lines += [
        "| step | t (累计 z_s) | L1_255 "
        f"{old_label} | L1_255 {new_label} | ΔL1 | PSNR {old_label} | PSNR {new_label} |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(st or []):
        cum = (sw_new.get("cum_z_s") or [None] * len(st))[i]
        a = l_old[i] if i < len(l_old) else None
        b = l_new[i] if i < len(l_new) else None
        pa = ps_old[i] if i < len(ps_old) else None
        pb = ps_new[i] if i < len(ps_new) else None
        lines.append(
            f"| {i + 1} | {t} | {fmt(a, 3)} | {fmt(b, 3)} | "
            f"{(b - a):+.3f} | {fmt(pa, 3)} | {fmt(pb, 3)} |" if a is not None and b is not None
            else f"| {i + 1} | {t} | {fmt(a, 3)} | {fmt(b, 3)} | — | {fmt(pa, 3)} | {fmt(pb, 3)} |")
    lines.append("")
    # 形态量（与昨日报告同口径）
    for label, sw in ((old_label, sw_old), (new_label, sw_new)):
        if not sw:
            continue
        d = sw.get("delta_L1_255") or []
        lines += [
            f"**{label}**: 总落差 {fmt(sw.get('total_drop_255'), 3)} px; "
            f"最优 step {None if sw.get('best_step') is None else sw['best_step'] + 1}; "
            f"Δ₁ 占比 {fmt((sw.get('delta1_share') or 0) * 100, 1, '%')}; "
            f"正增量步数 {sw.get('n_positive_steps')}/{len(d)}; "
            f"逐步增量 [{' '.join(f'{x:+.3f}' for x in d)}]",
            "",
        ]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval_linattn.py 的多臂输出 json")
    ap.add_argument("--stepwise-new", default="", help="今日 std BPTT 臂的逐步 json")
    ap.add_argument("--stepwise-old", default="", help="昨日 linattn BPTT 臂的逐步 json")
    ap.add_argument("--out", default="", help="输出 markdown 路径（空 = 只打印）")
    ap.add_argument("--old-label", default="线性注意力编码器（昨日）")
    ap.add_argument("--new-label", default="普通 transformer 编码器（今日）")
    args = ap.parse_args()

    by_dir = index_by_dir(load_json(args.eval))
    sw_new = load_json(args.stepwise_new) if args.stepwise_new and os.path.exists(args.stepwise_new) else None
    sw_old = load_json(args.stepwise_old) if args.stepwise_old and os.path.exists(args.stepwise_old) else None

    need = [ARM_SINGLE_OLD, ARM_SINGLE_NEW, ARM_REC_OLD, ARM_REC_NEW]
    missing = [d for d in need if d not in by_dir]
    if missing:
        print(f"[warn] eval json 缺少臂: {missing}（现有: {sorted(by_dir)}）", file=sys.stderr)

    out = ["# 消融对比表 — 编码器: 线性注意力 vs 普通 transformer", ""]
    out.append(f"- 权重: `--enc_attn linear`（昨日）/ `--enc_attn std`（今日, 普通 transformer）")
    out.append(f"- 解码器 / K / 数据 / 分辨率 / 损失 / 优化器: 两臂**逐项相同**（唯一自变量 = 编码器注意力形式）")
    out.append(f"- 评测: 全量留出集 {by_dir.get(ARM_SINGLE_NEW, {}).get('n_images', '?')} 张, 四臂**同进程同批次**同口径")
    out.append(f"- PSNR 口径: **逐图 PSNR 再平均**（与 2026-10-15 报告同口径）")
    out.append(f"- `L1 (0-255 像素)` 由归一化空间 L1 换算: `l1_norm × {_IMG_STD_MEAN} × 255 = l1_norm × {L1_NORM_TO_255:.2f}`"
               f"（与昨日报告 §3.1 的 8.433 ↔ 0.1463 换算一致）")
    out.append("")

    if ARM_SINGLE_OLD in by_dir and ARM_SINGLE_NEW in by_dir:
        out += pair_block(
            "配对 1 — 单步并行解码（T=1）, 从零训练 45 epoch / 42750 步 / bs128",
            by_dir[ARM_SINGLE_OLD], by_dir[ARM_SINGLE_NEW],
            args.old_label, args.new_label,
            note="两臂均为**从零**训练, 同 schedule —— 唯一自变量是编码器注意力形式。")
    if ARM_REC_OLD in by_dir and ARM_REC_NEW in by_dir:
        out += pair_block(
            "配对 2 — 循环读出 / **BPTT**（T=8）, 从单步版热启动微调 6 epoch / 11406 步 / bs64",
            by_dir[ARM_REC_OLD], by_dir[ARM_REC_NEW],
            args.old_label, args.new_label,
            note="两臂均为**从各自单步版热启动、6 epoch 同协议**（昨日 §3.4 已标注该预算偏短，"
                 "故本表**不**用于比较单步 vs BPTT，只用于同读出方式下换编码器的配对比较）。")
    out += stepwise_block(sw_new, sw_old, args.new_label, args.old_label)

    text = "\n".join(out) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"[out] {args.out}")
    print(text)


if __name__ == "__main__":
    main()
