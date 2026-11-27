#!/usr/bin/env python
"""v20_224_summarize.py — 汇总 v2.0(224²) 三阶段 + 阶段3-CE 的 rFID/指标 → SUMMARY.md"""
import json
import os
import sys

ST = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/v20_224"
EV = os.path.join(ST, "eval")
OUT = os.path.join(ST, "SUMMARY.md")


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


L = []


def add(s=""):
    L.append(s)
    print(s, flush=True)


def row(d, tag):
    if not d:
        return f"| {tag} | — | — | — | — |"
    ps = d.get("per_step", {})
    ks = sorted(ps, key=int)
    last = ps[ks[-1]] if ks else {}
    return (f"| {tag} | {d.get('n_images','?')} | {last.get('rfid', float('nan')):.3f} | "
            f"{last.get('psnr', float('nan')):.3f} | {last.get('l1_255', float('nan')):.3f} |")


def main():
    s1 = load(os.path.join(EV, "s1_rfid.json"))
    s2 = load(os.path.join(EV, "s2_rfid.json"))
    s3 = load(os.path.join(EV, "s3_rfid.json"))
    s3l = load(os.path.join(EV, "s3_last_rfid.json"))
    p0 = load(os.path.join(ST, "p0_gate_llamagen_5000.json"))
    ev1 = load(os.path.join(ST, "s1", "eval_val.json"))
    ev2 = load(os.path.join(ST, "s2", "eval_val.json"))
    ev3 = load(os.path.join(ST, "s3", "eval_val.json"))

    add("# v2.0 (224², K=32, T=16) — 阶段3 换成「对齐 LlamaGen VQ-16 离散码的交叉熵」")
    add()
    add("口径: 论文 pp_zh v2.0 = IN-100 / 224² / patch16 / K=32 / enc 13L·1248·h16 / "
        "dec 7L·1440·h16；s1 T=1 45ep(bs128)，s2 T=16 6ep(bs64)，s3 3ep(bs64, zs_dim=224)。")
    add("本实验唯一改动: **s3 的损失从像素 L1 换成 对 LlamaGen VQ-16(f16, 16384 codes) "
        "给原图输出的 14×14=196 个离散码 的逐步交叉熵**；读出循环/carry/heat-start 与 v2.0 相同。")
    add()
    add("## 1. 末步重建指标（5000 张 IN-100 val；rFID = FID-InceptionV3 pool3 2048d, resize299）")
    add()
    add("| 臂 | n | 末步 rFID ↓ | 末步 PSNR ↑ | 末步 L1(0-255) ↓ |")
    add("|---|---|---|---|---|")
    add(row(s1, "阶段1 T=1（像素头，读满 K=32）"))
    add(row(s2, "阶段2 T=16（像素头，full width 1248）"))
    add(row(s3, "阶段3-CE T=16（token 头→VQ decoder, zs 224）"))
    if s3l:
        add(row(s3l, "阶段3-CE ckpt_last"))
    add()
    if p0:
        add(f"**tokenizer 天花板（P0 决策门）**: LlamaGen VQ-16 自己对 224²/IN-100 val 的重建 "
            f"rFID = **{p0.get('rfid', float('nan')):.3f}**, PSNR = {p0.get('psnr', float('nan')):.3f} "
            f"(n={p0.get('n','?')}) ⇒ 阶段3-CE 的 rFID 不可能低于这个数。")
        add()
    add("## 2. 逐步 rFID 曲线（每步 2 个 z_s）")
    add()
    add("| 步 t | 累计 z_s | s1(T=1) | s2 像素 rFID | s3-CE rFID | s3-CE PSNR |")
    add("|---|---|---|---|---|---|")
    for t in range(1, 17):
        def g(d, k):
            if not d:
                return "—"
            v = d.get("per_step", {}).get(str(t))
            return "—" if v is None else f"{v.get(k):.3f}"
        add(f"| {t} | {2*t} | " + g(s1, "rfid") + " | " + g(s2, "rfid") + " | "
            + g(s3, "rfid") + " | " + g(s3, "psnr") + " |")
    add()
    add("## 3. 训练期 val（阶段3-CE: token 交叉熵 / top-1；像素口径不可比）")
    add()
    if ev3:
        add(f"- s3 iter: tok_ce={ev3.get('tok_ce'):.4f}, tok_acc={ev3.get('tok_acc'):.4f}, "
            f"T={ev3.get('T')}")
        cps = ev3.get("tok_ce_per_step") or []
        aps = ev3.get("tok_acc_per_step") or []
        if cps:
            add(f"- 逐步 CE: {[round(float(v),3) for v in cps]}")
            add(f"- 逐步 acc: {[round(float(v),4) for v in aps]}")
    if ev1:
        add(f"- s1 best val: PSNR={ev1.get('psnr'):.3f} L1={ev1.get('l1_norm'):.4f}")
    if ev2:
        add(f"- s2 best val: PSNR={ev2.get('psnr'):.3f} L1={ev2.get('l1_norm'):.4f}")
    add()
    add("## 4. 诚实边界")
    add()
    add("- 单种子；s1/s2 与论文 v2.0 同协议但重跑在本机（4×S4000, bf16, DDP4, --dec_grad_ckpt 1），"
        "绝对 PSNR 与论文 A800 单卡有跨机差异。")
    add("- s3-CE 的 rFID 受 tokenizer 重建天花板约束；与像素头 s2 的 rFID 不是同一渲染器，"
        "只能作为「阶段3 换目标后整条投放路径的端到端质量」对照。")
    add("- rFID 是**配对**的集合级分布度量（不含逐图精度），与 PSNR 不可互相替代。")
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[summarize] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
