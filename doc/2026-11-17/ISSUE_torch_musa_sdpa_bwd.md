# Issue draft — torch_musa (MooreThreads/torch_musa)

<!--
Channel: https://github.com/MooreThreads/torch_musa/issues/new
Related: #162 (item 6, "MuDNNFlashSDPABwd fails in autocast training"), maintainer lijing-mt replied 2026-11-13.
Suggested labels: bug
Post the whole file below this line. Attach: min_repro.py, env.txt (or paste the env block).
-->

## Title

`[torch_musa][muDNN][SDPA] Flash-attention BACKWARD returns wrong dQ/dK when head_dim % 8 != 0 (MTT S4000, bf16 and fp16)`


## Environment

```
- GPU: 4x Moore Threads MTT S4000 48 GiB (compute capability 2.2, arch 220)
- Driver: MUSA Driver 5.2.0-server   (mthreads-gmi 2.4.2)
- MUSA SDK: 5.2.0   (muDNN 3.4.0, runtime banner "muDNN(v3400)")
- PyTorch: 2.11.0.post1        (torch.version.git_version = 2b86b38358bd692be2b1fc5c37c0fdd6e12f8db5)
- torch_musa: 2.11.0.post1+b431668
              (torch_musa.version.git_version = b43166890d500f2c8a28ba1e8fdaf18e8ae3d24d)
- Python 3.10.12, Linux x86_64
- torch.backends.cuda: flash_sdp_enabled=1, mem_efficient_sdp_enabled=1, math_sdp_enabled=1
```


## Summary

`F.scaled_dot_product_attention` on **bf16/fp16 with a backward graph** is dispatched to the
muDNN flash kernel (`MuDNNFlashSDPAFwd` / `MuDNNFlashSDPABwd`). Its **backward is numerically
wrong whenever `head_dim % 8 != 0`**: `dQ`/`dK` are off by 2.6e-2 … 1.1e-1 in relative L2 norm,
while `dV` and the forward output are fine.

The error is **independent of dtype** — bf16 and fp16 produce the *same* wrong `dQ`
(they differ from each other by only 4.9e-3, i.e. ordinary dtype noise, but both differ from an
fp64 reference by ~3e-2). This is therefore an algorithmic/indexing defect in the flash backward
kernel, not a rounding issue, and it cannot be avoided by switching to fp16.

`head_dim % 8 == 0` is completely healthy (error at dtype-noise level), which gives a clean
work-around and a clean gate for the dispatcher.

This was found while training a 486 M-parameter ViT-style model on 4x S4000: with
`head_dim = 78` (encoder) and `head_dim = 90` (decoder) the encoder gradients drifted and then
exploded (grad-norm ratio bf16-vs-fp32 up to 44x … 231x, `lr=2e-4`), while the same recipe is
stable on an A800 (CUDA). Forcing the math kernel makes the run reproduce CUDA exactly and
removes all skipped steps. Assuming flash is safe for `head_dim <= 128` is the trap here.


## Minimal reproduction

Self-contained, ~40 lines, no project code, no checkpoint, runs in a few seconds.
`head_dim = 78` and `90` are the head dims of the model above; `64` is the aligned control.

```python
import math, torch, torch.nn.functional as F

def sdpa_grads(D, backend, dtype, B=2, H=8, N=213, seed=0):
    """dQ/dK/dV of  out = SDPA(q,k,v);  loss = (out*g).sum()"""
    g0 = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, N, D, generator=g0).cuda()
    k = torch.randn(B, H, N, D, generator=g0).cuda()
    v = torch.randn(B, H, N, D, generator=g0).cuda()
    g = torch.randn(B, H, N, D, generator=g0).cuda()
    c = torch.backends.cuda
    c.enable_flash_sdp(backend == "flash")
    c.enable_math_sdp(backend == "math")
    qq, kk, vv = (x.to(dtype).requires_grad_() for x in (q, k, v))
    out = F.scaled_dot_product_attention(qq, kk, vv)
    (out * g.to(dtype)).sum().backward()
    return [x.grad.float().cpu() for x in (qq, kk, vv)]

def ref_fp64(D, B=2, H=8, N=213, seed=0):
    """same computation in float64 on CPU (muDNN has no fp64 SDPA)"""
    g0 = torch.Generator().manual_seed(seed)
    q, k, v, g = (torch.randn(B, H, N, D, generator=g0).double().requires_grad_()
                  for _ in range(4))
    out = (((q @ k.transpose(-1, -2)) / math.sqrt(D)).softmax(-1)) @ v
    (out * g).sum().backward()
    return [x.grad for x in (q, k, v)]

for D in (64, 78, 90):
    R = ref_fp64(D)
    for backend in ("flash", "math"):
        for dt in (torch.bfloat16, torch.float16):
            G = sdpa_grads(D, backend, dt)
            rel = [float((a.double() - b).norm() / b.norm()) for a, b in zip(G, R)]
            print("head_dim=%3d %-5s %-8s  dQ=%.2e  dK=%.2e  dV=%.2e"
                  % (D, backend, str(dt).split(".")[1], *rel))
```

### Actual output (MTT S4000, 2.11.0.post1+b431668)

```
head_dim= 64 flash bfloat16  dQ=4.83e-03  dK=4.47e-03  dV=3.80e-03
head_dim= 64 flash float16   dQ=5.96e-04  dK=5.53e-04  dV=4.74e-04     <- aligned: healthy
head_dim= 64 math  bfloat16  dQ=5.97e-03  dK=5.98e-03  dV=5.12e-03
head_dim= 64 math  float16   dQ=7.41e-04  dK=7.45e-04  dV=6.47e-04
head_dim= 78 flash bfloat16  dQ=2.61e-02  dK=1.82e-02  dV=3.76e-03     <- WRONG (5.4x bf16 noise)
head_dim= 78 flash float16   dQ=2.57e-02  dK=1.77e-02  dV=4.67e-04     <- WRONG (43x fp16 noise)
head_dim= 78 math  bfloat16  dQ=6.02e-03  dK=5.98e-03  dV=5.18e-03
head_dim= 78 math  float16   dQ=7.35e-04  dK=7.36e-04  dV=6.33e-04
head_dim= 90 flash bfloat16  dQ=4.25e-02  dK=3.11e-02  dV=3.70e-03     <- WRONG
head_dim= 90 flash float16   dQ=4.22e-02  dK=3.08e-02  dV=4.61e-04     <- WRONG
head_dim= 90 math  bfloat16  dQ=5.95e-03  dK=5.95e-03  dV=5.01e-03
head_dim= 90 math  float16   dQ=7.41e-04  dK=7.41e-04  dV=6.27e-04
```

Note `head_dim=78 math float16` (7.35e-04) vs `head_dim=78 flash float16` (2.57e-02): the math
kernel is **35x** more accurate, and the flash error is ~26x the fp16 machine epsilon (2^-10).
Since the fp16 and bf16 flash results are equally wrong, precision is not the explanation.

### head_dim sweep (bf16 / fp16, `dQ` relative error vs fp64, B=2 H=8 N=213)

| head_dim | 56 | 60 | 64 | 65 | 66 | 68 | 72 | 76 | **78** | 80 | 84 | 88 | **90** | 96 | 100 | 112 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 flash | 6.1e-3¹ | 6.0e-3¹ | 4.8e-3 | error² | 4.8e-2 | 4.0e-2 | 4.8e-3 | 3.7e-2 | **2.6e-2** | 4.8e-3 | 3.7e-2 | 4.8e-3 | **4.2e-2** | 4.8e-3 | 3.1e-2 | 4.8e-3 | 4.8e-3 |
| fp16 flash | 7.5e-4¹ | 7.5e-4¹ | 6.0e-4 | error² | 4.8e-2 | 4.0e-2 | 6.0e-4 | 3.6e-2 | **2.6e-2** | 6.0e-4 | 3.7e-2 | 6.0e-4 | **4.2e-2** | 6.0e-4 | 3.0e-2 | 6.0e-4 | 6.0e-4 |

¹ head_dim < 64 is rejected for the backward pass by the dispatcher, so math runs — correct.
² head_dim = 65 raises `Run SDPA Flash FWD ... Flash Attention 2 Not Support HeadDim Odd`
  (kernel rejects it, but the dispatcher let it through — see "Additional issues").

**Rule: broken iff `head_dim % 8 != 0` (and `head_dim >= 64`, i.e. the flash-backward path).**


## Expected vs actual

| | expected | actual |
|---|---|---|
| `head_dim=78/90`, flash, bf16 | dQ error ≈ bf16 rounding (≈5e-3) | **2.6e-2 … 4.2e-2** |
| `head_dim=78/90`, flash, fp16 | dQ error ≈ fp16 rounding (≈6e-4) | **2.6e-2 … 4.2e-2** (same as bf16) |
| `head_dim=64/72/80/…`, flash | dQ error ≈ dtype rounding | matches, 4.8e-3 / 6.0e-4 |
| forward output, `head_dim=78` | within bf16 rounding | OK (loss rel. err ~1e-2 bf16, ~2e-4 fp16) |
| `dV` | within dtype rounding | OK (3.8e-3 bf16 / 4.7e-4 fp16) |
| math kernel, any `head_dim` | reference | OK |

The error is **not** a scale error: `||dQ|| / ||dQ_ref|| = 1.000`, i.e. the gradient magnitude is
right but its **direction is wrong**, which is much more damaging for training than a small
uniform bias.


## Aggravating factors (all measured)

* **Larger logits amplify it.** Scaling `q,k` by `s` (so the logit std grows like `s²`),
  `dQ` error at `head_dim=78`, B=2 H=8 N=213, bf16:

  | `s` | 1 | 2 | 4 | 8 | 16 |
  |---|---|---|---|---|---|
  | flash | 2.6e-2 | 0.21 | 0.67 | 1.41 | 2.90 |
  | math  | 6.0e-3 | 1.4e-2 | 5.0e-2 | 0.18 | 0.78 |
  | **flash / math** | **4.3x** | **15.7x** | **13.5x** | **7.9x** | **3.7x** |

  For the aligned control `head_dim=64` the same ratio is **0.8x / 0.6x / 0.55x / 0.47x / 0.58x**
  (flash is *better* than math at every scale). Reporting the flash-vs-math ratio at the same
  `head_dim` removes the effect of the fp64 reference itself becoming ill-conditioned when the
  softmax saturates — and it shows the defect is specific to unaligned head dims at all
  magnitudes.
  This is why the bug only shows up after the model has been training for a while: as logits grow,
  the flash backward silently degrades from "2 %" to "several hundred %" wrong.
* **`is_causal=True` makes it worse**: `head_dim=78` → 1.8e-1 (vs 4.4e-3 for `head_dim=64`).
* **Layout-dependent**: contiguous inputs → 7.1e-2, non-contiguous
  (`x.reshape(B,N,H,D).transpose(1,2)`, the usual `nn.MultiheadAttention`-style layout) → 2.7e-2.
  Different layouts give different wrong answers, which again points at indexing rather than
  rounding. (Both layouts are wrong.)
* **Independent of `N`** (17 … 512 all wrong at `head_dim=78`), of batch/head count, of explicit
  `scale`, and of `dropout_p` (`dropout_p > 0` is rejected by
  `check_musa_attention_input`, so it correctly falls back to math).


## Why the dispatcher lets this through

`torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h` (installed 2.11.0.post1+b431668):

```cpp
const bool fwd_only = (!input_requires_grad(params));
const bool is_head_dim_qkv_128 = (qk_head_dim <= 128 && v_head_dim <= 128);
bool enable = is_head_dim_qkv_128;
if (arch == 220) {
  if (fwd_only) { enable = enable || is_head_dim_qkv_160; }
  else          { enable = enable && (qk_head_dim >= 64 && v_head_dim >= 64); }  // only >= 64
}
...
return check_tensor_dtype(params, musa_allowed_dtypes /* {kHalf, kBFloat16} */, true);
```

There is **no `% 8` alignment check**, so `head_dim = 78/90` reaches `MuDNNFlashSDPABwd`.

`main` (blob `01fc7175…`, newer than the installed `97bf4a4c…`) already documents the constraint,
but it is only applied when `dropout > 0` — the training path with `dropout_p == 0` is still
ungated:

```cpp
  // mutlass tme require head dim % 8 == 0
  if ((enable && has_dropout) &&
      !(qk_head_dim % 8 == 0 && v_head_dim % 8 == 0)) {
    enable = false;
  }
```


## Suggested fix

In `check_musa_attention_input()` (`SDPUtils.h`), apply the alignment requirement on **every**
path, so unaligned head dims silently fall back to math instead of returning wrong gradients:

```diff
-  // mutlass tme require head dim % 8 == 0
-  if ((enable && has_dropout) &&
-      !(qk_head_dim % 8 == 0 && v_head_dim % 8 == 0)) {
-    enable = false;
-  }
+  // mutlass tme requires head dim % 8 == 0. The backward kernel silently
+  // returns wrong dQ/dK for unaligned head dims (forward and dV are fine),
+  // so gate it on every path, not only when dropout is active.
+  if (enable && !(qk_head_dim % 8 == 0 && v_head_dim % 8 == 0)) {
+    enable = false;
+  }
```

A correctness reference for the muDNN side would be nice as well: `MuDNNFlashSDPABwd` should be
validated against `MuDNNMathSDPABwd` for a few unaligned head dims (66/78/90/100) in CI.


## Additional (separate, smaller) issues found while reducing this one

1. **Odd head dim is not rejected by the dispatcher.** `head_dim=65`, bf16, backward graph:
   the dispatcher returns flash, then the kernel raises
   `MuDNNFlashSDPAFwd MUDNN failed in: Run SDPA Flash FWD` with
   `Flash Attention 2 Not Support HeadDim Odd`. It should be filtered out in
   `check_musa_attention_input` and fall back to math, like `head_dim < 64` already is.
2. **`qk_head_dim != v_head_dim` is silently catastrophically wrong.** `qk_dim=78, v_dim=64`
   gives `dQ = nan`; `qk_dim=64, v_dim=78` gives `dQ rel err 4.0`, `dV rel err 2.2`, while the
   math path raises `MuDNNMathSDPAFwd MUDNN failed in: Run SDPA Math FWD`.
   `main` already adds `const bool sym = (qk_head_dim == v_head_dim);` — please also make sure a
   non-symmetric case falls back to a working path rather than NaN.
3. **Introspection APIs are misleading on MUSA**: `torch.backends.cuda.can_use_flash_attention()`
   and `is_flash_attention_available()` return `False` even when the flash path *is* selected
   (they call the generic ATen implementation, which does not know about the MUSA override of
   `_fused_sdp_choice`). `mem_efficient_sdp_enabled()` is `True` but the backend is never
   selectable (`enable_mem_efficient_sdp(True)` + flash/math off → `RuntimeError: Invalid backend
   configuration for musa!`). Consider making these reflect the MUSA dispatcher; right now it is
   impossible to tell from Python which kernel ran.


## Impact / work-around

* Impact: **every bf16/fp16 training run on S4000 whose head dim is not a multiple of 8**, i.e.
  most odd-dim ViT variants (`dim/heads = 78`, `90`, `100`, …). Silent — no error, no warning,
  only wrong gradients.
* Work-around (validated, one flag is enough — math is enabled by default):

  ```python
  torch.backends.cuda.enable_flash_sdp(False)     # -> dispatcher picks math
  ```

  Cost measured at `B=4, H=16, N=213, head_dim=78`, bf16 fwd+bwd:
  math **0.71 ms** vs flash **2.46 ms** per iteration — the math kernel is *faster* at these
  shapes, so the work-around is free. (For larger `N` the trade-off may flip; re-measure.)
* Alternative work-around: choose `head_dim % 8 == 0` (e.g. 1248/16 = 78 → 1248/13 = 96 or
  1280/16 = 80).


## Attachments

* `min_repro.py` — the extended reproducer (`--sweep dim|scale|N|causal`), returns exit code 1
  when a bf16 flash row exceeds 2 % dQ error.
* `env.txt` — full `pip freeze` / `mthreads-gmi` / `torch` version dump.

---

## 中文摘要（给同事看，不用贴到 issue）

`torch_musa 2.11.0.post1` 在 MTT S4000 上把 bf16/fp16 的 `F.scaled_dot_product_attention`
（带反向图时）派发到 muDNN 的 flash kernel，而**该 kernel 的反向在 `head_dim % 8 != 0` 时
把 dQ/dK 算错**（前向和 dV 正常）：

* `head_dim=78`（=1248/16，本仓编码器）dQ 相对误差 bf16 **2.6e-2**、fp16 **2.6e-2**；
  `head_dim=90`（=1440/16，解码器）**4.2e-2**。**fp16 和 bf16 一样错** ⇒ 不是精度问题。
* `head_dim=64/72/80/88/96/128`（8 的倍数）误差 = dtype 理论噪声（bf16 4.8e-3 / fp16 6.0e-4）。
* 同一 dtype 换 math 后端 → 回到理论噪声（fp16 差 **35 倍**）。
* 误差随 logits 变大单调放大（qk×4 → 0.63；×16 → 1.05），所以"训练到中后期才炸"。
* 规避：`torch.backends.cuda.enable_flash_sdp(False)`（math 默认开，一个开关就够）；
  N=213/head_dim=78 上 math 反而比 flash 快（0.71 vs 2.46 ms），成本≈0。
* 根治：上游 `SDPUtils.h` 把 `head_dim % 8 == 0` 的检查从 `has_dropout` 分支里放出来
  （见上面 patch），不满足就 fallback 到 math。
