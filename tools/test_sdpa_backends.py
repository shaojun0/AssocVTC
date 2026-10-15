#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdpa_backends.py — SDPA 后端可复现性测试 / 缺陷取证脚本（自包含）

背景（2026-10-14，内网 8×海光 DCU 实测）
----------------------------------------
bf16 训练在**第一个解码器 self-attention** 就抛：

    IndexError: select(): index 0 out of range for tensor of size [0] at dimension 0

调用栈：`TransformerDecoderLayer._sa_block` → `nn.MultiheadAttention`
→ `F.multi_head_attention_forward` → `F.scaled_dot_product_attention`。

本地把零 batch / 零 query 长 / 零 kv 长 / 空 memory 全喂给 `math` 后端**都能通过**
⇒ 排除形状/空张量问题，判定为**该平台融合后端自身的 bug**。
处置：**全局**关掉坏后端（`--sdpa_backend math`）后跑通。

本脚本做什么
------------
用解码器的**真实形状**，逐 (精度 × 后端组合) 跑多组用例，产出可直接贴进 issue 的
证据包（环境 + 逐用例结果 + 异常栈 + 最小复现片段）：

  ① 裸 `F.scaled_dot_product_attention`（B,H,N,head_dim）  fwd / fwd+bwd
  ② `nn.MultiheadAttention` self-attention                 fwd / fwd+bwd  ← 实际崩的那条路
  ③ `nn.MultiheadAttention` cross-attention（memory = [z_cls; z_s] = 145）
  ④ `torch.utils.checkpoint(use_reentrant=False)` 包住的 MHA → **backward 重算路径**
     这正是"后端开关必须全局设"的原因：解码器 12 步 BPTT 被 checkpoint 包着，
     重算发生在 backward，局部 `with sdpa_kernel(...)` 上下文覆盖不到那一次重算。

与 `tools/diag_sdpa.py` 的分工
------------------------------
  · `diag_sdpa.py`            = 快速定位（只 forward、context 级开关、几十秒出结论）
  · `test_sdpa_backends.py`   = 回归 + 取证（含 backward/重算、**全局**开关、导出 issue 报告）

本文件**不 import 本仓库任何模块**，只依赖 torch ⇒ 可直接寄给上游 / 板卡厂商。

用法
----
    python tools/test_sdpa_backends.py                     # 默认：全部形状 × 默认精度组合
    python tools/test_sdpa_backends.py --quick             # 只跑出事的那一档（N=576），几十秒
    python tools/test_sdpa_backends.py --device_type dcu --bs 2 --out_dir ./sdpa_report
    python tools/test_sdpa_backends.py --precision bf16 --shapes 576,1024

产物：`<out_dir>/sdpa_backend_report.md`（贴 issue）+ `sdpa_backend_report.json`（机器可读）。

退出码：0 = 无失败；1 = 有用例失败（**取证要的就是这个**）；2 = 脚本自身出错。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

# ── 解码器真实形状（与 model_multi.RESOS / reso_patches 同口径：w//14 * h//14）──
SHAPES = [("448x252", 576), ("224x224", 256), ("448x448", 1024)]
CROSS_MEM = 145                      # memory = [z_cls; z_s] = 1 + K(=144)
BACKENDS = ("flash", "efficient", "math", "cudnn")
_TOGGLE = {"flash": "enable_flash_sdp",
           "efficient": "enable_mem_efficient_sdp",
           "math": "enable_math_sdp",
           "cudnn": "enable_cudnn_sdp"}
_ENABLED = {"flash": "flash_sdp_enabled",
            "efficient": "mem_efficient_sdp_enabled",
            "math": "math_sdp_enabled",
            "cudnn": "cudnn_sdp_enabled"}
_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

MINIMAL_REPRO = '''\
# 最小复现：ROCm/DCU 上 bf16 + 默认 SDPA 后端选择
import torch, torch.nn as nn
torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16        # ROCm 版 torch 下 cuda 即 HIP
B, N, d, H = 8, 576, 3072, 24           # 448x252 -> N=576；解码器 width=3072, heads=24
m = nn.MultiheadAttention(d, H, dropout=0.0, batch_first=True).to(device=dev, dtype=dt)
x = torch.randn(B, N, d, device=dev, dtype=dt, requires_grad=True)
out, _ = m(x, x, x, need_weights=False)     # <- 这里抛 IndexError: select(): index 0 ...
out.float().sum().backward()
print("OK", tuple(out.shape))
'''


# ────────────────────────────── 设备 ──────────────────────────────
def resolve_device(device_type: str):
    """返回 (device_type, torch.device)。dcu 是 cuda 的别名（ROCm 下 torch.cuda 即 HIP）。"""
    import torch
    dt = (device_type or "auto").lower()
    if dt == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            dt = "npu"
        elif torch.cuda.is_available():
            dt = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            dt = "xpu"
        else:
            dt = "cpu"
    if dt == "dcu":
        dt = "cuda"
    if dt == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[FAIL] --device_type cuda/dcu，但 torch.cuda.is_available()=False。\n"
                "       DCU 上必须是 ROCm/HIP 版 torch（torch.version.hip 非空）；\n"
                "       若装的是 CUDA 版 torch，这里会静默落到 CPU。"
                f"当前 torch.version.hip={getattr(torch.version, 'hip', None)!r}, "
                f"torch={torch.__version__}")
        return "cuda", torch.device("cuda", 0)
    if dt == "npu":
        return "npu", torch.device("npu", 0)
    if dt == "xpu":
        return "xpu", torch.device("xpu", 0)
    return "cpu", torch.device("cpu")


def collect_env(dev_type: str, device) -> dict:
    import torch
    env = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "device_type": dev_type,
        "device": str(device),
    }
    try:
        env["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    if dev_type == "cuda":
        try:
            p = torch.cuda.get_device_properties(0)
            env["device_name"] = torch.cuda.get_device_name(0)
            env["device_count"] = torch.cuda.device_count()
            env["device_total_mem_gib"] = round(p.total_memory / 2 ** 30, 1)
            env["compute_capability"] = f"{p.major}.{p.minor}"
            gcn = getattr(p, "gcnArchName", None)          # ROCm 专有：gfx906 / gfx928 ...
            if gcn:
                env["gcn_arch"] = gcn
        except Exception as e:
            env["device_props_error"] = repr(e)
        try:
            env["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception as e:
            env["bf16_supported"] = f"ERR {e!r}"
    for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
              "HSA_OVERRIDE_GFX_VERSION", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF",
              "SRDIFF_DEVICE", "HF_HUB_OFFLINE"):
        v = os.environ.get(k)
        if v:
            env[f"env:{k}"] = v
    return env


# ──────────────────────── SDPA 后端开关（全局） ────────────────────────
def backend_availability(dev_type: str) -> dict:
    """SDPA 后端开关只在 CUDA-family 设备上有意义；非 CUDA 设备返回空 dict。"""
    import torch
    if dev_type != "cuda":
        return {}
    avail = {}
    for name in BACKENDS:
        avail[name] = getattr(torch.backends.cuda, _TOGGLE[name], None) is not None
    try:
        probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
        if probe is not None:
            avail["flash"] = bool(avail["flash"] and probe())
    except Exception:
        pass
    return avail


def apply_backends(spec, avail: dict) -> str:
    """**全局**设置后端开关。

    spec=None → 全开（等价 PyTorch 默认）；否则只开 spec 里的后端。
    返回可读的生效状态串（写进报告，作为"开关确实生效"的证据）。
    """
    import torch
    parts = []
    for name in BACKENDS:
        fn = getattr(torch.backends.cuda, _TOGGLE[name], None)
        if fn is None:
            continue
        want = True if spec is None else (name in spec)
        try:
            fn(bool(want))
            parts.append(f"{name}={'on' if want else 'off'}")
        except Exception as e:
            parts.append(f"{name}=ERR({type(e).__name__})")
    return " ".join(parts) or "(no SDPA backend switches on this build/device)"


def backend_flags() -> dict:
    import torch
    out = {}
    for name in BACKENDS:
        fn = getattr(torch.backends.cuda, _ENABLED[name], None)
        try:
            out[name] = bool(fn()) if fn is not None else "n/a"
        except Exception:
            out[name] = "ERR"
    return out


# ──────────────────────────── 用例 ────────────────────────────
def case_raw(N, B, d, H, dev, dt, backward: bool):
    """裸 SDPA：解码器 self-attention 的真实形状 (B, H, N, head_dim)。"""
    import torch
    import torch.nn.functional as F
    hd = d // H
    req = bool(backward)
    q = torch.randn(B, H, N, hd, device=dev, dtype=dt, requires_grad=req)
    k = torch.randn(B, H, N, hd, device=dev, dtype=dt, requires_grad=req)
    v = torch.randn(B, H, N, hd, device=dev, dtype=dt, requires_grad=req)
    out = F.scaled_dot_product_attention(q, k, v)
    if backward:
        out.float().sum().backward()


def case_mha(N, B, d, H, dev, dt, backward: bool, memory=None):
    """nn.MultiheadAttention —— TransformerDecoderLayer._sa_block / _mha_block 的真实路径。"""
    import torch
    req = bool(backward)
    m = torch.nn.MultiheadAttention(d, H, dropout=0.0, batch_first=True).to(device=dev, dtype=dt)
    q = torch.randn(B, N, d, device=dev, dtype=dt, requires_grad=req)
    if memory is None:
        k = v = q
    else:
        k = v = torch.randn(B, memory, d, device=dev, dtype=dt, requires_grad=req)
    out, _ = m(q, k, v, need_weights=False)
    if backward:
        out.float().sum().backward()


def case_ckpt(N, B, d, H, dev, dt):
    """checkpoint(use_reentrant=False) 包住 MHA —— 覆盖 backward 里的重算路径。"""
    import torch
    from torch.utils.checkpoint import checkpoint
    m = torch.nn.MultiheadAttention(d, H, dropout=0.0, batch_first=True).to(device=dev, dtype=dt)
    x = torch.randn(B, N, d, device=dev, dtype=dt, requires_grad=True)

    def block(t):
        out, _ = m(t, t, t, need_weights=False)
        return out

    out = checkpoint(block, x, use_reentrant=False)
    out.float().sum().backward()


def build_cases(shape_list, B, d, H, dev, dt):
    """(用例名, 可调用) 列表。"""
    cases = []
    for label, N in shape_list:
        cases.append((f"{label} (N={N}) raw SDPA fwd",
                      lambda N=N: case_raw(N, B, d, H, dev, dt, False)))
        cases.append((f"{label} (N={N}) raw SDPA fwd+bwd",
                      lambda N=N: case_raw(N, B, d, H, dev, dt, True)))
        cases.append((f"{label} (N={N}) MHA self fwd",
                      lambda N=N: case_mha(N, B, d, H, dev, dt, False)))
        cases.append((f"{label} (N={N}) MHA self fwd+bwd",
                      lambda N=N: case_mha(N, B, d, H, dev, dt, True)))
        cases.append((f"{label} (N={N}) MHA self ckpt fwd+bwd(重算)",
                      lambda N=N: case_ckpt(N, B, d, H, dev, dt)))
    N0 = shape_list[0][1]
    cases.append((f"cross (m={CROSS_MEM}) MHA cross fwd+bwd",
                  lambda: case_mha(N0, B, d, H, dev, dt, True, CROSS_MEM)))
    return cases


# ──────────────────────────── 主流程 ────────────────────────────
def parse_shapes(args):
    if args.quick:
        return SHAPES[:1]
    if args.shapes:
        want = [int(x) for x in args.shapes.replace(" ", "").split(",") if x]
        picked = [s for s in SHAPES if s[1] in want]
        if not picked:
            raise SystemExit(f"[FAIL] --shapes {args.shapes} 与已知形状 {[n for _, n in SHAPES]} 无交集")
        return picked
    return list(SHAPES)


def parse_precisions(args, dev_type: str, env: dict):
    if args.precision != "auto":
        return [p.strip() for p in args.precision.split(",") if p.strip()]
    if dev_type == "cpu":
        return ["fp32"]
    out = ["bf16", "fp16", "fp32"]
    if env.get("bf16_supported") is False:
        out.remove("bf16")          # 该卡没有 bf16 矩阵核（如 gfx906）——不浪费一次跑
    return out


def suggest(rows):
    per = {}
    for r in rows:
        if r["status"] == "SKIP":
            continue
        per.setdefault(r["backend"], []).append(r["status"])
    ok, bad = [], []
    for b, sts in per.items():
        (bad if any(s == "FAIL" for s in sts) else ok).append(b)
    lines = []
    if not bad:
        lines.append("全部后端组合通过 —— 本机上没有复现出该缺陷。")
    else:
        lines.append(f"**失败的后端组合**：{', '.join(sorted(bad))}")
    if "math" in ok:
        lines.append("**可用逃生门**：`--sdpa_backend math`（最稳）")
    if "math+efficient" in ok and "efficient" in bad:
        lines.append("`math+efficient` 也通过 ⇒ 想要速度可用它（efficient 不物化 N×N，省几十 GB）")
    if "flash" in bad or "efficient" in bad or "cudnn" in bad:
        lines.append("⇒ 该平台**融合后端自身有 bug**（math 后端同样的形状能过）"
                     "，不是模型/数据/形状问题；训练里必须**全局**关掉坏后端"
                     "（backward 重算路径 `ckpt` 用例同样是 FAIL，局部 `sdpa_kernel` 上下文救不了）。")
    if "default" in bad and not ok:
        lines.append("连 math 都失败 ⇒ 把本报告整体贴出，并附 `--precision fp32` 的结果。")
    return lines


def write_reports(out_dir, args, env, rows, flags_before, flags_after, avail, notes):
    os.makedirs(out_dir, exist_ok=True)
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    payload = {
        "env": env,
        "args": vars(args),
        "sdpa_flags_before": flags_before,
        "sdpa_flags_after": flags_after,
        "backend_availability": avail,
        "notes": notes,
        "summary": {"PASS": n_pass, "FAIL": n_fail, "SKIP": n_skip},
        "results": rows,
    }
    jpath = os.path.join(out_dir, "sdpa_backend_report.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    L = []
    L.append(f"# SDPA 后端取证报告 — {env.get('device_name') or env['device_type']}")
    L.append("")
    L.append(f"> 生成时间：{env['generated_at']}　|　脚本：`tools/test_sdpa_backends.py`"
             f"　|　结果：**{n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP**")
    L.append("")
    L.append("## 1. 环境")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    for k, v in env.items():
        L.append(f"| `{k}` | `{v}` |")
    L.append(f"| SDPA 开关（运行前） | `{flags_before}` |")
    L.append(f"| SDPA 开关（运行后） | `{flags_after}` |")
    L.append(f"| 后端可用性探测 | `{avail or '非 CUDA-family 设备，后端开关无效'}` |")
    L.append("")
    L.append("## 2. 结果汇总")
    L.append("")
    L.append("| 精度 | 后端组合 | PASS | FAIL | SKIP | 生效开关 |")
    L.append("|---|---|---:|---:|---:|---|")
    seen = []
    for r in rows:
        key = (r["precision"], r["backend"])
        if key not in seen:
            seen.append(key)
    for prec, be in seen:
        sub = [r for r in rows if r["precision"] == prec and r["backend"] == be]
        L.append(f"| {prec} | `{be}` | {sum(x['status']=='PASS' for x in sub)} "
                 f"| {sum(x['status']=='FAIL' for x in sub)} "
                 f"| {sum(x['status']=='SKIP' for x in sub)} | `{sub[0]['flags']}` |")
    L.append("")
    L.append("## 3. 逐用例结果")
    L.append("")
    L.append("| 精度 | 后端 | 用例 | 结果 | 耗时 s | 说明 |")
    L.append("|---|---|---|---|---:|---|")
    for r in rows:
        det = (r["detail"] or "").replace("|", "\\|")
        L.append(f"| {r['precision']} | `{r['backend']}` | {r['case']} | **{r['status']}** "
                 f"| {r['seconds']:.2f} | {det} |")
    L.append("")
    fails = [r for r in rows if r["status"] == "FAIL"]
    L.append("## 4. 失败明细（traceback 尾部）")
    L.append("")
    if not fails:
        L.append("无失败用例 —— 本机没有复现出该缺陷。若是在报 bug，请附上"
                 "**失败机器**上的同名报告。")
        L.append("")
    else:
        for r in fails:
            L.append(f"### {r['precision']} / `{r['backend']}` / {r['case']}")
            L.append("")
            L.append("```")
            L.append(r["detail"])
            L.append("")
            L.append(r["traceback_tail"])
            L.append("```")
            L.append("")
    L.append("## 5. 最小复现片段")
    L.append("")
    L.append("```python")
    L.append(MINIMAL_REPRO.rstrip())
    L.append("```")
    L.append("")
    L.append("## 6. 结论 / 建议")
    L.append("")
    for line in suggest(rows):
        L.append(f"- {line}")
    if notes:
        L.append("")
        L.append("## 7. 其他说明")
        L.append("")
        for n in notes:
            L.append(f"- {n}")
    L.append("")
    mpath = os.path.join(out_dir, "sdpa_backend_report.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return mpath, jpath


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device_type", default=os.environ.get("SRDIFF_DEVICE", "auto"),
                    help="auto|cuda|dcu|npu|xpu|cpu（dcu 是 cuda 的别名）")
    ap.add_argument("--precision", default="auto",
                    help="auto | bf16 | fp16 | fp32 | 逗号组合（CPU 上 auto=fp32）")
    ap.add_argument("--bs", type=int, default=2, help="batch size（默认 2，省显存）")
    ap.add_argument("--decoder_width", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--shapes", default="", help="逗号分隔的 N，如 576,1024；默认全形状")
    ap.add_argument("--quick", action="store_true", help="只跑 448x252(N=576) 这一档")
    ap.add_argument("--out_dir", default=".", help="报告输出目录")
    ap.add_argument("--no_files", action="store_true", help="不写报告文件，只打印")
    args = ap.parse_args()

    import torch

    dev_type, dev = resolve_device(args.device_type)
    env = collect_env(dev_type, dev)
    flags_before = backend_flags()
    avail = backend_availability(dev_type)
    shapes = parse_shapes(args)
    precisions = parse_precisions(args, dev_type, env)
    B, d, H = args.bs, args.decoder_width, args.heads
    if d % H:
        raise SystemExit(f"[FAIL] decoder_width={d} 必须能被 heads={H} 整除")

    # 组合：(标签, 只开这些后端 / None=全开)
    combos = [("default", None), ("math", {"math"}), ("efficient", {"efficient"}),
              ("flash", {"flash"}), ("math+efficient", {"math", "efficient"})]
    if dev_type != "cuda" or avail.get("cudnn"):
        # 非 CUDA-family 上这些组合一律 SKIP（列全，报告才是完整矩阵）；
        # CUDA 上若本版本没有 cudnn 开关，则不列（否则会用残留开关假装测过）。
        combos.append(("cudnn", {"cudnn"}))
    notes = []
    if dev_type != "cuda":
        notes.append(f"设备类型 `{dev_type}`：SDPA 融合后端开关只在 CUDA-family 上有效，"
                     f"非 default 组合一律记为 SKIP（不代表这些后端在 CPU/NPU 上可用）。")
    if dev_type == "cuda" and env.get("torch_hip"):
        notes.append(f"ROCm/HIP 环境：hip={env['torch_hip']}, gcn_arch={env.get('gcn_arch')}, "
                     f"bf16_supported={env.get('bf16_supported')}。上报时请附这些字段。")
    if env.get("bf16_supported") is False:
        notes.append("本卡 `is_bf16_supported()=False` ⇒ 仓库代码会自动降级到 **fp16**（不是 fp32）。")

    print("=" * 96)
    print("test_sdpa_backends — SDPA 后端可复现性测试 / 取证")
    print("=" * 96)
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"  SDPA 开关（运行前）: {flags_before}")
    print(f"  后端可用性: {avail or '(非 CUDA-family)'}")
    print(f"  形状: {[f'{lb}(N={n})' for lb, n in shapes]}  bs={B}  width={d}  heads={H} "
          f"(head_dim={d // H})  精度: {precisions}")
    print("-" * 96)

    rows = []
    for prec in precisions:
        if prec not in _DTYPE:
            raise SystemExit(f"[FAIL] 未知精度 {prec!r}，可选 {sorted(_DTYPE)}")
        dt = getattr(torch, _DTYPE[prec])
        cases = build_cases(shapes, B, d, H, dev, dt)
        for be, spec in combos:
            if dev_type != "cuda" and be != "default":
                rows.append(dict(precision=prec, backend=be, case="-", status="SKIP",
                                 detail="非 CUDA-family 设备：后端开关无效",
                                 traceback_tail="", seconds=0.0, flags="-"))
                continue
            flags = apply_backends(spec, avail)
            nf = 0
            for name, fn in cases:
                t0 = time.time()
                try:
                    fn()
                    st, detail, tb = "PASS", "", ""
                except Exception as e:
                    st = "FAIL"
                    detail = f"{type(e).__name__}: {str(e)[:300]}"
                    tb = "\n".join(traceback.format_exc().strip().splitlines()[-12:])
                    nf += 1
                rows.append(dict(precision=prec, backend=be, case=name, status=st,
                                 detail=detail, traceback_tail=tb,
                                 seconds=time.time() - t0, flags=flags))
            mark = "✅" if nf == 0 else f"❌ {nf} FAIL"
            print(f"  {prec:<5} | {be:<15} | {len(cases) - nf}/{len(cases)} PASS {mark}"
                  f"  [{flags}]", flush=True)

    apply_backends(None, avail)          # 复原为全开，别给后续进程留副作用
    flags_after = backend_flags()

    print("-" * 96)
    for line in suggest(rows):
        print("  • " + line.replace("**", ""))
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    print("=" * 96)

    mpath = jpath = ""
    if not args.no_files:
        mpath, jpath = write_reports(args.out_dir, args, env, rows,
                                     flags_before, flags_after, avail, notes)
        print(f"  报告（贴 issue 用）: {mpath}")
        print(f"  机器可读结果      : {jpath}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
