"""AssocVTC v7 评测：**历史长度 / 历史有无**消融 + copy 基线 + 检索指标。

v7（model_v7.py，主线 d123ee7 的 model_v3 设计）没有"每帧 token 预算"旋钮，
评测口径从 v6 的"每帧预算 b 步"改成"history 消融"：

    cold  : 逐帧独立 F=1、max_hist=0（KV 只有 mem_init，纯冷启动）
    rec   : 整段一次前向、max_hist=0（只有"上一帧 Q → 下一帧 query"的递归，无历史 z_s）
    warm  : 整段一次前向、max_hist=None（全部历史帧 z_s 进 KV，中间态完整版）
    hist  : max_hist ∈ {1,2,4,8,16}（0=rec、None=warm）→ 历史长度-PSNR 曲线

基线完全照 eval_scale.py 的口径：
    copy_prev : 复制上一帧
    copy_best : 历史里"最相似"的帧（像素 MSE 最优）+ 帧龄统计
    retr_*    : z_cls 余弦检索（max / top_age），诊断历史里谁最像当前帧

输出：results.json（全局均值 + 逐片段 + 逐帧数组）、控制台表格、vis.png、results.md。

用法（远端 GPU1，见 eval_v7.sh）：
    MUSA_VISIBLE_DEVICES=1 python3 -u eval_v7.py --ckpt <ckpt.pt> --clips 4 \
        --window 120 --out /root/autodl-tmp/video_recon/eval_v7_long
    # 无 ckpt 时验证通路（PSNR ~10dB 无意义）：
    ... --ckpt /dev/null --random-init 1 --clips 1 --window 16 --out .../smoke_eval_v7

注意：
* 协议前向一律 `m(x, n_warm=0, max_hist=..., step_weights="last")`，**不要 return_pix**
  （120 帧 × 5 步的像素张量会撑爆显存）；只有 vis 对少量帧单独取像素。
* 输入 x 按 chunk 传 `enc_chunk`，编码器显存有界；B 恒为 1（逐帧循环在模型内部）。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import time

import numpy as np
import torch

from data_real import decode_clip, read_index
from model_mini import psnr_from_l1
from model_v8 import build_any as build_model


# ================================================================= 设备 / 显存
def get_device(lr: int = 0):
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device(f"musa:{lr}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{lr}")
    return torch.device("cpu")


def _peak_mem_mb() -> float:
    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            return float(torch.musa.max_memory_allocated() / 2 ** 20)
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:
        pass
    return float("nan")


def _reset_peak():
    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            torch.musa.reset_peak_memory_stats()
        elif torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _amp_ctx(args, dev):
    """--amp bf16 时给前向套 autocast（默认 none = fp32，PSNR 口径与 v6 一致）。"""
    if args.amp == "bf16" and dev.type in ("musa", "cuda"):
        return torch.autocast(dev.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ================================================================= 模型 / 数据
def load_model(args, dev):
    cfg_path = args.cfg_json or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cfg_v7.json")
    if args.random_init:
        cfg = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        model = build_model(cfg)
        src = f"随机初始化（无 ckpt，PSNR 无意义；cfg={cfg_path if cfg else 'DEFAULT_CFG'}）"
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        model = build_model(cfg)
        model.load_state_dict(ck["model"])
        src = args.ckpt
    model = model.to(dev).eval().requires_grad_(False)
    return model, cfg, src


def list_paths(args):
    idx = os.path.join(args.root, args.index_name)
    if os.path.exists(idx):
        items = read_index(args.root, args.index_name)
        paths = [os.path.join(args.root, it["path"]) for it in items]
    else:                                   # 退化：目录下直接放 *.mp4
        paths = sorted(os.path.join(args.root, f) for f in os.listdir(args.root)
                       if f.endswith(".mp4"))
    return paths


# ================================================================= 统计小工具
def _mean(vals):
    v = [float(v) for v in vals if v is not None]
    return float(np.mean(v)) if v else None


def _median(vals):
    v = [float(v) for v in vals if v is not None]
    return float(statistics.median(v)) if v else None


def _f(v):
    """JSON 安全：tensor / numpy → python float。"""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return float(v.item())
    return float(v)


def _s(v, nd=3):
    """格式化：None → '-'。"""
    return f"{v:.{nd}f}" if v is not None else "-"


def _finite(v):
    """JSON 安全：非有限值 → None。"""
    return float(v) if (v is not None and math.isfinite(float(v))) else None


# ================================================================= 协议
@torch.no_grad()
def run_cold(model, x, args, dev):
    """cold：逐帧独立 F=1、max_hist=0，按 chunk 拼 batch 加速。

    注意：F=1 时 model.forward 的 psnr_per_frame 只有 1 项（batch 均值），
    b>1 必须自己从像素算逐样本 PSNR。这里 return_pix 只用于 F=1 的 chunk
    （(b,1,5,3,448,448) ≈ 96MB@b=8，安全）；整段协议**不用** return_pix。
    """
    ps = []
    chunk = max(1, args.chunk)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk].unsqueeze(1)                    # (b,1,3,H,W)
        b = int(xb.shape[0])
        with _amp_ctx(args, dev):
            if b == 1:
                out = model(xb, n_warm=0, max_hist=0, step_weights="last")
                ps.append(_f(out["psnr_per_frame"][0]))
            else:
                out = model(xb, n_warm=0, max_hist=0, step_weights="last",
                            return_pix=True)
                pix = out["pix"][:, 0, -1].float()          # (b,3,H,W) 最后一步
                ps.extend(_f(v) for v in psnr_from_l1(pix, x[i:i + b]))
    return ps


@torch.no_grad()
def run_seq(model, x, max_hist, args, dev):
    """整段一次前向（B=1）：rec=0 / warm=None / hist=k。返回逐帧 PSNR 列表。"""
    with _amp_ctx(args, dev):
        out = model(x.unsqueeze(0), n_warm=0, max_hist=max_hist,
                    step_weights="last", enc_chunk=max(0, args.enc_chunk))
    return [_f(v) for v in out["psnr_per_frame"].float().cpu().tolist()]


@torch.no_grad()
def copy_baselines(x):
    """copy_prev（上一帧）/ copy_best（历史里 MSE 最优帧 + 帧龄）。第 0 帧无历史 → None。"""
    F = int(x.shape[0])
    prev = [None]
    for j in range(1, F):
        prev.append(_f(psnr_from_l1(x[j - 1:j], x[j:j + 1])))
    best, ages = [None], [None]
    for j in range(1, F):
        tgt = x[j:j + 1]
        # 与 eval_scale.py 同口径：MSE → PSNR（max=1），取历史最优
        d = ((x[:j] - tgt) ** 2).flatten(1).mean(1).clamp_min(1e-10)
        ps = -10.0 * torch.log10(d)
        best.append(_f(ps.max()))
        ages.append(int(j - int(ps.argmax().item())))
    return dict(prev=_mean(prev), best=_mean(best),
                best_age_median=_median(ages), best_age_mean=_mean(ages),
                prev_per_frame=prev, best_per_frame=best, best_ages=ages)


@torch.no_grad()
def retr_metrics(model, x, args):
    """z_cls 余弦检索：当前帧 vs 历史各帧（逐帧 max cosine + top 帧龄），照 eval_scale.py。"""
    zc, _ = model.encode(x.unsqueeze(0), chunk=max(0, args.enc_chunk))
    z = torch.nn.functional.normalize(zc[0, :, 0].float(), dim=-1).cpu()   # (F,D)
    sim = z @ z.T
    F = int(z.shape[0])
    cos_max, top_age = [None], [None]
    for j in range(1, F):
        s = sim[j, :j]
        cos_max.append(_f(s.max()))
        top_age.append(int(j - int(s.argmax().item())))
    prev_cos = [None] + [_f(sim[j, j - 1]) for j in range(1, F)]
    return dict(cos_max=_mean(cos_max), cos_max_median=_median(cos_max),
                top_age_median=_median(top_age), top_age_mean=_mean(top_age),
                cos_max_per_frame=cos_max, top_age_per_frame=top_age,
                prev_cos=_mean(prev_cos), prev_cos_per_frame=prev_cos)


@torch.no_grad()
def run_clip(model, x, args, dev):
    """一条片段的全部协议。x: (F,3,H,W) on dev。"""
    out = {"n_frames": int(x.shape[0]), "secs": {}}
    t_clip = time.time()

    # ---- 基线（纯像素，最便宜，先算）
    t = time.time()
    out["copy"] = copy_baselines(x)
    out["secs"]["copy"] = time.time() - t

    # ---- 检索指标（一次编码；z_s 只在这里用，算完即释放）
    if args.run_retr:
        t = time.time()
        out["retr"] = retr_metrics(model, x, args)
        out["secs"]["retr"] = time.time() - t

    # ---- cold
    if args.run_cold:
        t = time.time()
        ps = run_cold(model, x, args, dev)
        out["cold"] = dict(psnr=_mean(ps), per_frame=ps)
        out["secs"]["cold"] = time.time() - t
        print(f"  cold  {out['cold']['psnr']:.3f} dB  ({out['secs']['cold']:.1f}s)",
              flush=True)

    # ---- rec（max_hist=0）
    if args.run_rec:
        t = time.time()
        ps = run_seq(model, x, 0, args, dev)
        out["rec"] = dict(psnr=_mean(ps), per_frame=ps)
        out["secs"]["rec"] = time.time() - t
        print(f"  rec   {out['rec']['psnr']:.3f} dB  ({out['secs']['rec']:.1f}s)",
              flush=True)

    # ---- hist sweep
    if args.run_hist and args.hist_sweep:
        out["hist"] = {}
        for k in args.hist_sweep:
            t = time.time()
            ps = run_seq(model, x, k, args, dev)
            out["hist"][str(k)] = dict(psnr=_mean(ps), per_frame=ps)
            out["secs"][f"hist_{k}"] = time.time() - t
            print(f"  hist={k:<3d}{out['hist'][str(k)]['psnr']:.3f} dB  "
                  f"({out['secs'][f'hist_{k}']:.1f}s)", flush=True)

    # ---- warm（max_hist=None，KV 全历史；显存/时间最大，放最后）
    if args.run_warm:
        t = time.time()
        ps = run_seq(model, x, None, args, dev)
        out["warm"] = dict(psnr=_mean(ps), per_frame=ps)
        out["secs"]["warm"] = time.time() - t
        print(f"  warm  {out['warm']['psnr']:.3f} dB  ({out['secs']['warm']:.1f}s)",
              flush=True)

    out["secs"]["total"] = time.time() - t_clip
    return out


# ================================================================= 汇总 / 打印
def agg_protocol(per_clip, key, sub=None):
    """逐片段均值 + 逐帧池化均值。sub=None → c[key]；否则 c[key][sub]（hist sweep 用）。"""
    def _get(c):
        d = c.get(key, {})
        return d.get(sub) if sub is not None else d
    clips = [_get(c)["psnr"] for c in per_clip
             if _get(c) and _get(c).get("psnr") is not None]
    frames = []
    for c in per_clip:
        d = _get(c)
        if d:
            frames.extend(v for v in d["per_frame"] if v is not None)
    return dict(psnr=_mean(clips) if clips else None,
                psnr_pooled=_mean(frames) if frames else None,
                per_clip=[_f(v) for v in clips])


def summarize(res, args, model, n_clips):
    per_clip = res["per_clip"]
    summ = {"n_clips": n_clips, "protocols": {}, "hist_curve": {}, "baselines": {},
            "retrieval": {}, "secs": {}, "peak_mem_mb": res.get("peak_mem_mb")}
    order = []
    if args.run_cold:
        summ["protocols"]["cold"] = agg_protocol(per_clip, "cold")
        order.append("cold")
    if args.run_rec:
        summ["protocols"]["rec"] = agg_protocol(per_clip, "rec")
        order.append("rec")
    if args.run_hist and args.hist_sweep:
        for k in args.hist_sweep:
            key = f"hist_{k}"
            summ["protocols"][key] = agg_protocol(per_clip, "hist", str(k))
            order.append(key)
    if args.run_warm:
        summ["protocols"]["warm"] = agg_protocol(per_clip, "warm")
        order.append("warm")
    summ["order"] = order
    # hist 曲线：0=rec、k=hist k、None=warm（顺序：0,1,2,4,8,16,None）
    if args.run_hist and args.hist_sweep:
        rec = summ["protocols"].get("rec", {}).get("psnr")
        if rec is not None:
            summ["hist_curve"]["0"] = rec
        for k in args.hist_sweep:
            summ["hist_curve"][str(k)] = summ["protocols"][f"hist_{k}"]["psnr"]
        warm = summ["protocols"].get("warm", {}).get("psnr")
        if warm is not None:
            summ["hist_curve"]["None"] = warm

    # ---- 基线
    prevs = [c["copy"]["prev"] for c in per_clip]
    bests = [c["copy"]["best"] for c in per_clip]
    ages = [c["copy"]["best_age_median"] for c in per_clip]
    prev_pool = [v for c in per_clip for v in c["copy"]["prev_per_frame"] if v is not None]
    best_pool = [v for c in per_clip for v in c["copy"]["best_per_frame"] if v is not None]
    summ["baselines"] = dict(
        copy_prev=_mean(prevs), copy_prev_pooled=_mean(prev_pool),
        copy_best=_mean(bests), copy_best_pooled=_mean(best_pool),
        copy_best_age_median=_median(ages),
        copy_best_age_mean=_mean([c["copy"]["best_age_mean"] for c in per_clip]))
    # ---- 检索
    if args.run_retr and per_clip and "retr" in per_clip[0]:
        cms = [c["retr"]["cos_max"] for c in per_clip]
        cms_pool = [v for c in per_clip for v in c["retr"]["cos_max_per_frame"]
                    if v is not None]
        summ["retrieval"] = dict(
            cos_max=_mean(cms), cos_max_pooled=_mean(cms_pool),
            top_age_median=_median([c["retr"]["top_age_median"] for c in per_clip]),
            top_age_mean=_mean([c["retr"]["top_age_mean"] for c in per_clip]),
            prev_cos=_mean([c["retr"]["prev_cos"] for c in per_clip]))
    # ---- 耗时
    summ["secs"]["total"] = _mean([c["secs"]["total"] for c in per_clip])
    for k in ("cold", "rec", "warm", "copy", "retr"):
        summ["secs"][k] = _mean([c["secs"].get(k) for c in per_clip])
    for k in (args.hist_sweep or []):
        summ["secs"][f"hist_{k}"] = _mean([c["secs"].get(f"hist_{k}") for c in per_clip])
    return summ


def print_summary(res, args):
    s = res["summary"]
    print("\n=== SUMMARY ===")
    print(f"clips={s['n_clips']}  window={res['window']}  "
          f"root={res['root']}  amp={args.amp}")
    rec = s["protocols"].get("rec", {}).get("psnr")
    cold = s["protocols"].get("cold", {}).get("psnr")

    def _d(v, ref):
        return f"{(v - ref):+.3f}" if (v is not None and ref is not None) else "  -  "

    print(f"{'protocol':<14}{'PSNR(dB)':>11}{'Δvs rec':>10}{'Δvs cold':>10}{'s/clip':>9}")
    for key in s["order"]:
        p = s["protocols"][key]
        secs = s["secs"].get(key)
        secs_s = f"{secs:9.1f}" if secs is not None else f"{'-':>9}"
        print(f"{key:<14}{p['psnr']:11.3f}{_d(p['psnr'], rec):>10}"
              f"{_d(p['psnr'], cold):>10}{secs_s}")
    if s["hist_curve"]:
        print("\nhist 曲线（历史长度 → PSNR）: " +
              " | ".join(f"{k}={v:.3f}" for k, v in s["hist_curve"].items()
                         if v is not None))
        if rec is not None:
            print("  相对 rec: " +
                  " | ".join(f"{k}={(v - rec):+.3f}" for k, v in s["hist_curve"].items()
                             if v is not None))
    b = s["baselines"]
    print("\n--- 基线 / 检索（全局均值）---")
    print(f"copy_prev     {_s(b['copy_prev'])} dB     copy_best  {_s(b['copy_best'])} dB"
          f"（帧龄中位 {_s(b['copy_best_age_median'], 1)}）")
    if s["retrieval"]:
        r = s["retrieval"]
        print(f"retr_cos_max  {_s(r['cos_max'], 4)}            "
              f"top_age 中位 {_s(r['top_age_median'], 1)}     "
              f"prev_cos {_s(r['prev_cos'], 4)}")
    print(f"\n总耗时 {res['total_secs']:.0f}s（{s['secs']['total']:.1f}s/clip），"
          f"峰值显存 {_s(s['peak_mem_mb'], 0)} MB")


# ================================================================= results.md
def write_report(path, res, args):
    s = res["summary"]
    rec = s["protocols"].get("rec", {}).get("psnr")
    cold = s["protocols"].get("cold", {}).get("psnr")
    L = []
    L.append("# v7 评测：历史长度 / 历史有无消融")
    L.append("")
    L.append(f"- **ckpt**: `{res['ckpt']}`" +
             ("（**随机初始化，PSNR 无意义，仅验证通路**）" if args.random_init else ""))
    L.append(f"- **root**: `{res['root']}`，前 {s['n_clips']} 条片段，"
             f"window={res['window']} 帧/片段")
    L.append(f"- **设备**: `{res['device']}`（MUSA_VISIBLE_DEVICES="
             f"{os.environ.get('MUSA_VISIBLE_DEVICES', '?')}），amp={args.amp}")
    L.append(f"- **耗时/显存**: {s['secs']['total']:.1f}s/clip，"
             f"峰值 {_s(s['peak_mem_mb'], 0)} MB；总 {res['total_secs']:.0f}s"
             "（GPU1 与其他任务共享，计时仅供参考）")
    L.append("")
    L.append("## 协议（全局均值）")
    L.append("")
    L.append("| 协议 | PSNR (dB) | Δ vs rec | Δ vs cold | s/clip | 说明 |")
    L.append("|---|---:|---:|---:|---:|---|")
    desc = {"cold": "逐帧 F=1，KV=[mem_init]",
            "rec": "整段前向，max_hist=0（仅 Q 递归）",
            "warm": "整段前向，max_hist=None（全历史）"}
    for key in s["order"]:
        p = s["protocols"][key]
        if key in desc:
            d = desc[key]
        else:
            d = f"整段前向，max_hist={key.split('_', 1)[1]}"
        dr = f"{p['psnr'] - rec:+.3f}" if rec is not None else "-"
        dc = f"{p['psnr'] - cold:+.3f}" if cold is not None else "-"
        secs = s["secs"].get(key)
        secs_s = f"{secs:.1f}" if secs is not None else "-"
        L.append(f"| {key} | {p['psnr']:.3f} | {dr} | {dc} | "
                 f"{secs_s} | {d} |")
    L.append("")
    if s["baselines"]:
        b = s["baselines"]
        L.append("## 基线 / 检索")
        L.append("")
        L.append("| 指标 | 数值 |")
        L.append("|---|---:|")
        L.append(f"| copy_prev (dB) | {_s(b['copy_prev'])} |")
        L.append(f"| copy_best (dB) | {_s(b['copy_best'])} |")
        L.append(f"| copy_best 帧龄中位 (帧) | {_s(b['copy_best_age_median'], 1)} |")
        if s["retrieval"]:
            r = s["retrieval"]
            L.append(f"| retr_cos_max | {_s(r['cos_max'], 4)} |")
            L.append(f"| retr top_age 中位 (帧) | {_s(r['top_age_median'], 1)} |")
            L.append(f"| 相邻帧 z_cls 余弦 | {_s(r['prev_cos'], 4)} |")
        L.append("")
    if s["hist_curve"]:
        L.append("## 历史长度曲线")
        L.append("")
        L.append("| 历史长度 | " + " | ".join(s["hist_curve"].keys()) + " |")
        L.append("|---|" + "---:|" * len(s["hist_curve"]))
        L.append("| PSNR (dB) | " + " | ".join(
            f"{v:.3f}" if v is not None else "-" for v in s["hist_curve"].values()) + " |")
        L.append("| Δ vs rec | " + " | ".join(
            f"{v - rec:+.3f}" if (v is not None and rec is not None) else "-"
            for v in s["hist_curve"].values()) + " |")
        L.append("")
    L.append("## 逐片段 PSNR (dB)")
    L.append("")
    cols = ["cold", "rec"] + [f"hist_{k}" for k in (args.hist_sweep or [])] + ["warm"]
    L.append("| clip | 帧数 | " + " | ".join(cols) + " |")
    L.append("|---|---:|" + "---:|" * len(cols))
    for c in res["per_clip"]:
        vals = []
        for col in cols:
            if col in ("rec", "cold", "warm"):
                v = c.get(col, {}).get("psnr")
            else:
                v = c.get("hist", {}).get(col.split("_", 1)[1], {}).get("psnr")
            vals.append(f"{v:.3f}" if v is not None else "-")
        L.append(f"| {c['path']} | {c['n_frames']} | " + " | ".join(vals) + " |")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print("saved", path)


# ================================================================= 可视化
@torch.no_grad()
def save_vis(model, args, dev, paths, cfg, out_png):
    """vis.png：每条片段 1 行，列 = GT / cold / rec / warm 各若干帧（取窗口末尾帧）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.fontManager.addfont(
            "/userdata/texlive/2026/texmf-dist/fonts/truetype/public/lxgw-fonts/LXGWNeoXiHei.ttf")
        plt.rcParams["font.family"] = "LXGW Neo XiHei"
    except Exception:
        pass

    protos = [p for p in ("cold", "rec", "warm") if getattr(args, f"run_{p}")]
    if not protos:
        print("vis: 没有启用的协议，跳过")
        return
    img = int(cfg.get("img", 448) if cfg else 448)
    nf = max(1, args.vis_frames)
    rows = [p for p in paths if os.path.exists(p)]
    if not rows:
        return
    ncol = nf * (1 + len(protos))
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(1.65 * ncol, 1.9 * len(rows) + 0.5),
                             squeeze=False)
    for r, path in enumerate(rows):
        w_vis = args.window if args.vis_window <= 0 else min(args.vis_window, args.window)
        w_vis = max(nf, w_vis)
        arr = decode_clip(path, 0, w_vis, img)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0).to(dev)
        sel = list(range(x.shape[0] - nf, x.shape[0]))
        pe = {}                                     # (proto, j) -> (3,H,W) float cpu
        if any(p in protos for p in ("rec", "warm")):
            zc, zs = model.encode(x.unsqueeze(0), chunk=max(0, args.enc_chunk))
        for p in protos:
            if p == "cold":
                for j in sel:
                    with _amp_ctx(args, dev):
                        o = model(x[j:j + 1].unsqueeze(0), n_warm=0, max_hist=0,
                                  step_weights="last", return_pix=True)
                    pe[("cold", j)] = o["pix"][0, 0, -1].float().cpu()
            else:
                mh = 0 if p == "rec" else None
                Q = model.middle(zs, max_hist=mh)
                for j in sel:
                    pix = model.decoder(zc[:, j], zs[:, j], Q[:, j],
                                        return_pix=True)["pix"][0, -1]
                    pe[(p, j)] = pix.float().cpu()
        # GT（占 nf 列）+ 各协议（各占 nf 列）
        for fi, j in enumerate(sel):
            axes[r, fi].imshow(x[j].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            axes[r, fi].set_title(f"GT f{j}", fontsize=6)
        axes[r, 0].set_ylabel(os.path.basename(path)[:22], fontsize=6)
        for pi, p in enumerate(protos):
            for fi, j in enumerate(sel):
                c = nf + pi * nf + fi
                pix = pe[(p, j)]
                axes[r, c].imshow(pix.permute(1, 2, 0).numpy().clip(0, 1))
                db = float(psnr_from_l1(pix.unsqueeze(0).to(dev),
                                        x[j:j + 1]))
                axes[r, c].set_title(f"{p} f{j} {db:.1f}dB", fontsize=6)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("saved", out_png)


# ================================================================= main
def parse_args():
    ap = argparse.ArgumentParser(description="v7 评测：历史长度消融 + copy 基线")
    ap.add_argument("--ckpt", required=True,
                    help="ckpt.pt（含 cfg/model）；--random-init 1 时忽略")
    ap.add_argument("--random-init", type=int, default=0,
                    help="1 = 不加载 ckpt，用 cfg_v7.json 随机初始化（仅验证通路）")
    ap.add_argument("--cfg-json", default="", help="random-init 时用的结构配置，默认 cfg_v7.json")
    ap.add_argument("--root", default="/root/autodl-tmp/video_recon/data3/val_long",
                    help="片段目录（含 index.jsonl；无则按 *.mp4 排序）")
    ap.add_argument("--index-name", default="index.jsonl")
    ap.add_argument("--clips", type=int, default=4, help="取前多少条片段")
    ap.add_argument("--window", type=int, default=120, help="每条片段取多少帧做目标")
    ap.add_argument("--chunk", type=int, default=8,
                    help="cold 协议的批大小（F=1×chunk；1=逐帧，最省显存）")
    ap.add_argument("--enc-chunk", type=int, default=8,
                    help="编码器批大小（0=一次全编码，显存占用大）")
    ap.add_argument("--hist-sweep", default="1,2,4,8,16",
                    help="max_hist 扫描列表；空串 = 不做 hist sweep")
    ap.add_argument("--protocols", default="all",
                    help="cold,rec,warm,hist 的子集或 all")
    ap.add_argument("--retr", type=int, default=1, help="1 = 算 z_cls 检索指标")
    ap.add_argument("--amp", default="none", choices=["none", "bf16"],
                    help="bf16 加速（PSNR 略变；正式数字建议 none）")
    ap.add_argument("--vis", type=int, default=1, help="1 = 存 vis.png")
    ap.add_argument("--vis-window", type=int, default=0,
                    help="vis 用多少帧（0=同 --window）；越大越慢")
    ap.add_argument("--vis-frames", type=int, default=2, help="每个协议在 vis 里显示几帧")
    ap.add_argument("--out", required=True, help="输出目录")
    args = ap.parse_args()
    # ---- 协议集合
    ps = args.protocols.strip().lower()
    if ps in ("", "all"):
        want = {"cold", "rec", "warm", "hist"}
    else:
        want = {x.strip() for x in ps.split(",") if x.strip()}
        bad = want - {"cold", "rec", "warm", "hist"}
        if bad:
            ap.error(f"--protocols 里有未知项: {sorted(bad)}")
    args.run_cold = "cold" in want
    args.run_rec = "rec" in want
    args.run_warm = "warm" in want
    args.run_hist = "hist" in want
    args.run_retr = bool(args.retr)
    args.run_copy = True                    # 纯像素基线，几乎免费，恒算
    # ---- hist 扫描
    sweep = []
    for s in (args.hist_sweep or "").split(","):
        s = s.strip().lower()
        if not s:
            continue
        sweep.append(None if s in ("none", "null") else int(s))
    args.hist_sweep = sweep
    if args.run_hist and not sweep:
        args.run_hist = False
    return args


def main():
    args = parse_args()
    torch.manual_seed(0)
    dev = get_device(0)
    os.makedirs(args.out, exist_ok=True)

    paths = list_paths(args)
    if not paths:
        raise SystemExit(f"没有片段：{args.root}")
    paths = paths[:max(1, args.clips)]
    print(f"device={dev}  torch={torch.__version__}  "
          f"MUSA_VISIBLE_DEVICES={os.environ.get('MUSA_VISIBLE_DEVICES', '?')}")
    print(f"root={args.root}  clips={len(paths)}  window={args.window}  "
          f"chunk={args.chunk}  enc_chunk={args.enc_chunk}  amp={args.amp}")
    print(f"protocols: cold={args.run_cold} rec={args.run_rec} "
          f"warm={args.run_warm} hist={args.hist_sweep if args.run_hist else '-'}")

    model, cfg, src = load_model(args, dev)
    pg = model.param_groups()
    print(f"loaded: {src}")
    print("params:", {k: f"{v / 1e6:.1f}M" for k, v in pg.items()},
          " steps:", model.steps)
    _reset_peak()
    res = {"ckpt": src, "random_init": bool(args.random_init), "root": args.root,
           "device": str(dev), "window": args.window, "chunk": args.chunk,
           "enc_chunk": args.enc_chunk, "amp": args.amp, "clips_requested": args.clips,
           "n_clips": 0, "per_clip": [], "total_secs": 0.0}
    t_all = time.time()
    n_ok = 0
    ok_paths = []
    for pi, path in enumerate(paths):
        try:
            arr = decode_clip(path, 0, args.window, int(cfg.get("img", 448) if cfg else 448))
        except Exception as e:
            print("skip", path, repr(e)[:160])
            continue
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0).to(dev)
        print(f"\n[{pi}] {os.path.basename(path)}  frames={x.shape[0]}", flush=True)
        _reset_peak()
        out = run_clip(model, x, args, dev)
        out["path"] = os.path.basename(path)
        out["peak_mem_mb"] = _finite(_peak_mem_mb())
        res["per_clip"].append(out)
        ok_paths.append(path)
        n_ok += 1
        del x
        print(f"  copy_prev={_s(out['copy']['prev'], 2)}  copy_best={_s(out['copy']['best'], 2)}  "
              f"({_s(out['copy']['best_age_median'], 0)} 帧龄中位)  "
              f"peak={_s(out['peak_mem_mb'], 0)}MB  {out['secs']['total']:.1f}s", flush=True)
    res["n_clips"] = n_ok
    if n_ok == 0:
        raise SystemExit("所有片段都失败，没有结果")
    peaks = [c["peak_mem_mb"] for c in res["per_clip"] if c["peak_mem_mb"] is not None]
    res["peak_mem_mb"] = max(peaks) if peaks else None
    res["total_secs"] = time.time() - t_all
    res["summary"] = summarize(res, args, model, n_ok)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, allow_nan=False)
    print("\nwrote", os.path.join(args.out, "results.json"))
    print_summary(res, args)
    write_report(os.path.join(args.out, "results.md"), res, args)
    if args.vis:
        try:
            save_vis(model, args, dev, ok_paths, cfg,
                     os.path.join(args.out, "vis.png"))
        except Exception as e:                     # vis 失败不影响数字
            print("vis 失败:", repr(e)[:200])


if __name__ == "__main__":
    main()
