#!/usr/bin/env python3
"""diag_sdpa.py — 定位 DCU/ROCm 上 SDPA 融合后端崩溃

背景（2026-10-14 用户实测）：DCU 上 bf16 跑训练，第一个解码器 self-attention 就抛

    IndexError: select(): index 0 out of range for tensor of size [0] at dimension 0

调用栈落在 `nn.TransformerDecoderLayer._sa_block → nn.MultiheadAttention →
F.multi_head_attention_forward → F.scaled_dot_product_attention`。

**关键认知**：`--attn_implementation` 只管 HF 的 DINOv2 编码器；解码器用的是
`nn.MultiheadAttention`，它**永远**走 SDPA，没有开关。所以那个参数救不了这个错。

本地已排除「形状/空张量」这一大类：把零 batch、零 query 长、零 kv 长、空 memory
全喂给 **math** 后端都能正常返回 ⇒ 不是形状问题，是**该平台融合后端自身的 bug**。

本脚本用**解码器的真实形状**逐后端跑一遍，几十秒出结论：哪个后端能用、哪个崩。

用法::

    python tools/diag_sdpa.py                                   # 默认形状 (bs=3)
    python tools/diag_sdpa.py --batch_size 6 --decoder_width 3072 --heads 24
    python tools/diag_sdpa.py --encoder                          # 顺带测 DINOv2 编码器的 SDPA 形状

结论怎么用：把失败的后端关掉即可（**全局**，因为梯度检查点的重算在 backward 里）：

    --sdpa_backend math                 # 最稳，通常能立刻解封
    --sdpa_backend math+efficient       # math 保住正确性 + efficient 给速度（若它不崩）
    --sdpa_backend default              # 关掉本策略（默认全开）
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESOS = [(448, 252), (252, 448), (224, 224), (448, 448)]


def patches(w, h):                      # 与 model_multi.reso_patches 同口径
    return (w // 14) * (h // 14)


def record(results, name, fn):
    """跑一遍 fn（内部自己设后端上下文），把 PASS/FAIL 记进 results。"""
    try:
        fn()
        results.append((name, "PASS", ""))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:150]}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device_type", default=os.environ.get("SRDIFF_DEVICE", "auto"))
    ap.add_argument("--batch_size", type=int, default=3)
    ap.add_argument("--decoder_width", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--encoder", action="store_true", help="加测 DINOv2 编码器的 SDPA 形状")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from srdiff_runtime import Runtime, is_rocm

    rt = Runtime(args.device_type)
    dev = rt.device()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    print("=" * 78)
    print("diag_sdpa — SDPA 融合后端定位")
    print("=" * 78)
    print(f"torch={torch.__version__}  hip={getattr(torch.version, 'hip', None)}  "
          f"cuda={getattr(torch.version, 'cuda', None)}  device={dev}")
    print(f"is_rocm={is_rocm()}  dtype={args.precision}  "
          f"bs={args.batch_size} width={args.decoder_width} heads={args.heads}")
    flags = {}
    for k in ("flash", "mem_efficient", "math"):
        fn = getattr(torch.backends.cuda, f"{k}_sdp_enabled", None)
        flags[k] = bool(fn()) if fn else "n/a"
    print(f"当前后端开关: {flags}")
    try:
        print(f"flash_attention_available={torch.backends.cuda.is_flash_attention_available()}")
    except Exception:
        pass

    B, d, H = args.batch_size, args.decoder_width, args.heads
    assert d % H == 0, f"decoder_width {d} 必须能被 heads {H} 整除"
    hd = d // H
    print(f"\nhead_dim={hd}（解码器 stack 的真实值：{d}/{H}）")

    # 后端组合：默认 / 各单后端 / math+efficient
    combos = [("default", None),
              ("math", dict(enable_math=True, enable_flash=False, enable_mem_efficient=False)),
              ("efficient", dict(enable_math=False, enable_flash=False, enable_mem_efficient=True)),
              ("flash", dict(enable_math=False, enable_flash=True, enable_mem_efficient=False)),
              ("math+efficient", dict(enable_math=True, enable_flash=False, enable_mem_efficient=True))]

    import contextlib

    def ctx(flags):
        if flags is None:
            return contextlib.nullcontext()
        return torch.backends.cuda.sdp_kernel(**flags)

    results = []
    for label, fl in combos:
        # ① 裸 SDPA：解码器 self-attention 的形状 (B,H,N,head_dim)
        for w, h in RESOS:
            N = patches(w, h)
            def _raw(N=N):
                q = torch.randn(B, H, N, hd, device=dev, dtype=dt)
                k = torch.randn(B, H, N, hd, device=dev, dtype=dt)
                v = torch.randn(B, H, N, hd, device=dev, dtype=dt)
                with ctx(fl):
                    F.scaled_dot_product_attention(q, k, v)
            record(results, f"{label:<15} 裸SDPA self  N={N:<5}", _raw)

        # ② nn.MultiheadAttention —— 实际崩掉的那条路（TransformerDecoderLayer._sa_block）
        for w, h in RESOS:
            N = patches(w, h)
            def _mha(N=N):
                m = torch.nn.MultiheadAttention(d, H, dropout=0.0, batch_first=True).to(dev, dt)
                x = torch.randn(B, N, d, device=dev, dtype=dt)
                with ctx(fl):
                    m(x, x, x, need_weights=False)
            record(results, f"{label:<15} MHA self      N={N:<5}", _mha)

        # ③ 解码器的 cross-attention：memory = [z_cls; z_s] = 145
        def _mha_cross(N=patches(*RESOS[0])):
            m = torch.nn.MultiheadAttention(d, H, dropout=0.0, batch_first=True).to(dev, dt)
            q = torch.randn(B, N, d, device=dev, dtype=dt)
            mem = torch.randn(B, 145, d, device=dev, dtype=dt)
            with ctx(fl):
                m(q, mem, mem, need_weights=False)
        record(results, f"{label:<15} MHA cross     m=145 ", _mha_cross)

    # ④ 编码器侧（HF DINOv2）：head_dim=64
    if args.encoder:
        for L in (721, 1162, 1025):       # 1+144+N（N=576 / 1018 / 880）
            def _enc(L=L):
                q = torch.randn(B, 16, L, 64, device=dev, dtype=dt)
                with ctx(None):
                    F.scaled_dot_product_attention(q, q, q)
            record(results, f"encoder(16头,hd=64) L={L:<5}", _enc)

    print("\n" + "=" * 78)
    print(f"{'场景':<34}{'结果':<7}说明")
    print("-" * 78)
    seen = {}
    for name, st, det in results:
        print(f"{name:<34}{st:<7}{det}")
        seen.setdefault(name.split()[0], []).append(st)
    print("-" * 78)
    print("按后端汇总：")
    for label, fl in combos:
        sts = [st for name, st, _ in results if name.startswith(label)]
        bad = sum(s == "FAIL" for s in sts)
        print(f"  {label:<15} {'全部通过 ✅' if bad == 0 else f'{bad}/{len(sts)} 失败 ❌'}")

    ok = [lb for lb, fl in combos
          if all(st == "PASS" for name, st, _ in results if name.startswith(lb))]
    print("\n结论：")
    if ok:
        print(f"  可用后端组合: {', '.join(ok)}")
        if "math" in ok:
            print("  建议先跑通:  --sdpa_backend math")
        if "math+efficient" in ok and "math+efficient" != ok[0]:
            print("  想要速度（若 efficient 不崩）: --sdpa_backend math+efficient")
    else:
        print("  所有组合都有失败 —— 请把上面完整输出发回；"
              "可能需要改用 fp32（--precision fp32）或换 torch/DTK 版本")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
