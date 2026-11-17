#!/usr/bin/env python3
"""min_repro.py -- minimal, standalone reproducer for the MUSA SDPA-backward bug.

Pure `torch` (no project code, no checkpoint, no data).  Builds one random
Q/K/V/G (G = upstream gradient), runs `F.scaled_dot_product_attention`
forward+backward under several dtype x backend configurations, and compares
dQ/dK/dV against a hand-written **fp64 reference computed on CPU** (muDNN has no
fp64 SDPA path, so the reference cannot be another mudnn call).

Per-row output (all rows use seed-identical inputs):
    dQ dK dV  relative L2 error vs fp64 reference     healthy bf16: ~5e-3
    aQ aK aV  ||grad|| / ||grad_ref||                 healthy: ~1.0
    fwd_loss  |loss - loss_ref| / |loss_ref|

Root cause demonstrated here: torch_musa dispatches bf16/fp16 SDPA with
`head_dim <= 128 and head_dim >= 64` to the muDNN **flash** kernel, but that
kernel is numerically wrong in BACKWARD whenever `head_dim % 8 != 0`
(78 and 90 are the head dims of the model that trained on this box:
enc 1248/16 = 78, dec 1440/16 = 90).  Forward is unaffected, dV is unaffected,
dQ/dK are off by ~1e-2..1e-1 and get worse as logits grow.  Forcing the math
kernel (`torch.backends.cuda.enable_flash_sdp(False)`) reproduces fp64 to bf16
precision.  `head_dim % 8 == 0` (e.g. 64) is fine -> this is a kernel bug, not
"bf16 is imprecise": fp16 and bf16 give the *same* wrong dQ.

Usage:
    MUSA_VISIBLE_DEVICES=0 python min_repro.py                 # key comparison
    MUSA_VISIBLE_DEVICES=0 python min_repro.py --sweep dim      # head_dim sweep
    MUSA_VISIBLE_DEVICES=0 python min_repro.py --sweep scale    # logit magnitude
    MUSA_VISIBLE_DEVICES=0 python min_repro.py --sweep N        # seq length
    MUSA_VISIBLE_DEVICES=0 python min_repro.py --sweep causal   # is_causal
    MUSA_VISIBLE_DEVICES=0 python min_repro.py --sweep loss     # loss shape
Exit code is 1 if any bf16 flash row exceeds 2% dQ error (CI-friendly).
"""
import argparse
import math

import torch
import torch.nn.functional as F

DEV = "cuda"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp32": torch.float32, "fp64": torch.float64}


# ---------------------------------------------------------------- backend setup
def set_backend(name):
    """MUSA: only `flash` and `math` exist; `mem_efficient` is never selected.

    torch_musa overrides `_fused_sdp_choice` (csrc/aten/ops/attention/mudnn/
    SDPBackendSelect.cpp + SDPUtils.h): flash if the shape/dtype checks pass,
    else math, else RuntimeError.  `can_use_flash_attention()` returns False on
    MUSA even when flash is used -- do not trust it for introspection.
    """
    ef = torch.backends.cuda.enable_flash_sdp
    em = torch.backends.cuda.enable_mem_efficient_sdp
    ea = torch.backends.cuda.enable_math_sdp
    if name == "default":          # what training does out of the box
        ef(True), em(True), ea(True)
    elif name == "math":           # workaround: one flag is enough
        ef(False), em(False), ea(True)
    elif name == "flash":          # force flash; error if unusable
        ef(True), em(False), ea(False)
    elif name == "mem_efficient":
        ef(False), em(True), ea(False)
    else:
        raise ValueError(name)


def backend_state():
    c = torch.backends.cuda
    return "flash={} mem_eff={} math={}".format(
        int(c.flash_sdp_enabled()), int(c.mem_efficient_sdp_enabled()),
        int(c.math_sdp_enabled()))


# --------------------------------------------------------------------- workload
def base_cfg(B=2, H=8, N=213, D=78):
    """Defaults = the real model shape: seq 213 (1 cls + 16 specials + 196 patch),
    head_dim 78 (enc_dim 1248 / enc_heads 16)."""
    return {"B": B, "H": H, "N": N, "D": D, "scale": None, "qscale": 1.0}


def make_inputs(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    B, H, N, D = cfg["B"], cfg["H"], cfg["N"], cfg["D"]
    q = torch.randn(B, H, N, D, generator=g) * cfg["qscale"]
    k = torch.randn(B, H, N, D, generator=g) * cfg["qscale"]
    v = torch.randn(B, H, N, D, generator=g)
    go = torch.randn(B, H, N, D, generator=g)
    return q.to(DEV), k.to(DEV), v.to(DEV), go.to(DEV)


def ref_fp64_cpu(cfg, inputs):
    """Hand-written softmax attention in float64 on CPU -- independent of muDNN."""
    q, k, v, go = (x.double().cpu().requires_grad_(True) for x in inputs)
    s = cfg["scale"] if cfg["scale"] is not None else 1.0 / math.sqrt(q.size(-1))
    out = (((q @ k.transpose(-1, -2)) * s).softmax(-1)) @ v
    loss = (out * go).sum()
    loss.backward()
    return {"loss": float(loss), "g": [x.grad for x in (q, k, v)]}


def run_device(cfg, dtype, backend, inputs, causal=False):
    q, k, v, go = inputs
    set_backend(backend)
    td = DTYPES[dtype]
    qq, kk, vv = (x.to(td).detach().requires_grad_(True) for x in (q, k, v))
    out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=None,
                                        dropout_p=0.0, is_causal=causal,
                                        scale=cfg["scale"])
    loss = (out * go.to(td)).sum()
    loss.backward()
    return {"loss": float(loss.detach().float()), "g": [x.grad for x in (qq, kk, vv)]}


def one_row(cfg, dtype, backend, inputs, ref, causal=False):
    row = {"dtype": dtype, "backend": backend, "err": None}
    try:
        c = run_device(cfg, dtype, backend, inputs, causal)
    except Exception as e:                                   # noqa: BLE001
        row["err"] = "{}: {}".format(type(e).__name__,
                                     str(e).split("\n")[0][:80])
        return row
    row["floss"] = abs(c["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-30)
    for i, n in enumerate("QKV"):
        g = c["g"][i].detach().double().cpu()
        nb = float(ref["g"][i].norm())
        row["d" + n] = float((g - ref["g"][i]).norm() / nb) if nb else float("nan")
        row["a" + n] = float(g.norm() / nb) if nb else float("nan")
    return row


def table(rows):
    out = ["{:<26s} {:>9s} {:>9s} {:>9s} {:>7s} {:>7s} {:>7s} {:>9s}".format(
        "case", "dQ", "dK", "dV", "aQ", "aK", "aV", "fwd_loss")]
    for r in rows:
        tag = "{} / {}".format(r["backend"], r["dtype"])
        if r["err"]:
            out.append("{:<26s} {}".format(tag, r["err"]))
        else:
            out.append(("{:<26s} {:9.3e} {:9.3e} {:9.3e} {:7.3f} {:7.3f} "
                        "{:7.3f} {:9.2e}").format(
                tag, r["dQ"], r["dK"], r["dV"], r["aQ"], r["aK"], r["aV"],
                r["floss"]))
    return "\n".join(out)


def header(cfg, extra=""):
    return ("=== B={B} H={H} N={N} D={D} qk_std={qscale} scale={scale} {ex}"
            "  (ref = fp64 CPU) ===").format(ex=extra, **cfg)


# ---------------------------------------------------------------------- sweeps
def cases(dim_control=True):
    c = [("default", "bf16"), ("math", "bf16"), ("default", "fp16"),
         ("default", "fp32")]
    return c


def sweep(kind, seed):
    if kind == "dim":
        cfgs = [(base_cfg(D=d), "D={}".format(d))
                for d in (56, 60, 64, 66, 68, 72, 76, 78, 80, 84, 88, 90,
                          96, 100, 112, 128)]
        causal = False
    elif kind == "scale":
        cfgs = [(dict(base_cfg(), qscale=s), "qk_std={}".format(s))
                for s in (1, 2, 4, 8, 16)]
        causal = False
    elif kind == "N":
        cfgs = [(base_cfg(N=n), "N={}".format(n))
                for n in (17, 64, 128, 196, 213, 384, 512)]
        causal = False
    elif kind == "causal":
        cfgs = [(base_cfg(), "d78"), (base_cfg(D=64), "d64")]
        causal = None
    elif kind == "loss":
        cfgs = [(base_cfg(), "d78")]
        causal = False
    else:
        raise ValueError(kind)

    bad = []
    for cfg, extra in cfgs:
        print("\n" + header(cfg, extra))
        rows = []
        if kind == "causal":
            for c in (False, True):
                ref = ref_fp64_cpu(cfg, make_inputs(cfg, seed))
                for be, dt in (("default", "bf16"), ("math", "bf16")):
                    r = one_row(cfg, dt, be, make_inputs(cfg, seed), ref, causal=c)
                    r["backend"] = "{}[{}]".format(be, "causal" if c else "full")
                    rows.append(r)
        else:
            ref = ref_fp64_cpu(cfg, make_inputs(cfg, seed))
            for be, dt in cases():
                rows.append(one_row(cfg, dt, be, make_inputs(cfg, seed), ref))
        print(table(rows))
        for r in rows:
            if r["err"] is None and r["backend"].startswith("default") \
                    and r["dtype"] == "bf16" and r["dQ"] > 2e-2:
                bad.append((extra, r["dQ"]))
    if bad:
        print("\n[FAIL] bf16 flash dQ error > 2% in: {}".format(
            ", ".join("{}={:.1e}".format(e, d) for e, d in bad)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=None,
                    choices=["dim", "scale", "N", "causal", "loss"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("torch={} torch.version.musa={} torch_musa={} dev={} SDPA[{}]".format(
        torch.__version__, getattr(torch.version, "musa", None),
        _musa_version(), torch.musa.get_device_name(0), backend_state()))

    if a.sweep:
        raise SystemExit(sweep(a.sweep, a.seed))

    rc = 0
    for cfg, extra in ((base_cfg(), "MODEL SHAPE: head_dim=78 (1248/16)"),
                       (base_cfg(D=64), "CONTROL: head_dim=64 (8-aligned)")):
        print("\n" + header(cfg, extra))
        ref = ref_fp64_cpu(cfg, make_inputs(cfg, a.seed))
        rows = [one_row(cfg, dt, be, make_inputs(cfg, a.seed), ref)
                for be, dt in cases()]
        print(table(rows))
        if "78" in extra and rows[0]["dQ"] > 2e-2 and rows[1]["dQ"] < 1e-2:
            print("--> flash/bf16 dQ error {:.1e} vs math/bf16 {:.1e}: "
                  "flash BACKWARD is wrong for head_dim=78, math is fine.".format(
                      rows[0]["dQ"], rows[1]["dQ"]))
            rc = 1
    print("\nKey reading: `default` == muDNN flash. Compare dQ of "
          "default/bf16 against math/bf16 (same dtype -> isolates the kernel) "
          "and against default/fp32.")
    return rc


def _musa_version():
    try:
        import torch_musa
        return getattr(torch_musa, "__version__", "?")
    except Exception:                                        # noqa: BLE001
        return "?"


if __name__ == "__main__":
    raise SystemExit(main())
