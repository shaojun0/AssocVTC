# 已提交到 MooreThreads/torch_musa#162 的评论

> ✅ **已提交**：2026-11-17 13:53:56Z（UTC），作者 `shaojun0`
> 链接：https://github.com/MooreThreads/torch_musa/issues/162#issuecomment-6950049064
> 提交内容 = 本文件 `---` 之后的正文（5192 字符）。

> 目标 issue: https://github.com/MooreThreads/torch_musa/issues/162
> （meta-issue「8 ready-to-file issues」，其中第 6 条 `MuDNNFlashSDPABwd fails in autocast
>  training — mobilenetv4_hybrid`；维护者 `lijing-mt` 2026-11-13 已回复 "we are currently
> analyzing it."）
> 提交身份：`gh` 已登录 `shaojun0`（本仓 owner）。

---

## Root cause of item #6 (`MuDNNFlashSDPABwd` in autocast training): `head_dim % 8 != 0`

We reproduced the same failure on the same stack while training a 486M ViT tokenizer
(encoder `head_dim = 1248/16 = 78`, decoder `1440/16 = 90`, bf16 autocast, lr 2e-4) on
4×MTT S4000, and reduced it to a single condition.

**Summary.** bf16/fp16 `F.scaled_dot_product_attention` **with a backward graph** is
dispatched to muDNN flash. Its **backward returns wrong dQ/dK whenever `head_dim % 8 != 0`**;
forward and dV are correct. bf16 and fp16 produce the *same* wrong dQ (mutual diff 4.9e-3 =
dtype noise; both ~3e-2 off an fp64 reference) ⇒ algorithmic/indexing defect, not rounding —
switching to fp16 does not help. `math` is correct. The error grows monotonically with logit
magnitude, which is why it surfaces late in training as gradient spikes rather than at step 0.

### Evidence (B=2, H=8, N=213; ref = fp64 CPU; dQ relative error)

| head_dim | flash bf16 | flash fp16 | math bf16 | math fp16 |
|---|---|---|---|---|
| 64 (8-aligned, control) | 4.8e-3 | 6.0e-4 | 6.0e-3 | 7.4e-4 |
| 72 / 80 / 88 / 96 (8-aligned) | ~4.8e-3 | ~6.0e-4 | — | — |
| **78** | **2.6e-2** | **2.6e-2** | 6.0e-3 | 7.4e-4 |
| **90** | **4.2e-2** | **4.2e-2** | 6.0e-3 | 7.4e-4 |
| 66 / 68 / 76 / 84 / 100 (all `%8!=0`) | 3.1e-2 … 4.8e-2 | — | — | — |

- `head_dim` sweep 56…128: **broken iff `head_dim % 8 != 0`** (56/60 are rejected and fall
  back to math → correct; 65 is dispatched and then raises inside the kernel).
- Scaling with logits (D=78, bf16): qk std ×1/×2/×4/×8/×16 → dQ
  **2.6e-2 / 0.21 / 0.67 / 1.41 / 2.90**; math at the same conditions
  6.0e-3 / 1.4e-2 / 5.0e-2 / 0.18 / 0.78 ⇒ flash/math ratio 4.3…15.7 (at D=64 the ratio is <1).
- `dV` is fine (3.8e-3); forward loss is fine (bf16 1.6e-2, fp16 2.1e-4, D=64 identical).
- Independent of N (17…512), B/H, contiguous vs transposed layout, explicit `scale`.
- `is_causal=True` is worse (1.8e-1 at D=78). `dropout_p>0` is rejected and correctly falls
  back to math (that fallback path is fine).

### Minimal reproduction (~18 s, self-contained; also runs `--sweep dim|scale|N|causal`)

```python
import torch, torch.nn.functional as F

def grads(mode, dtype, D=78, B=2, H=8, N=213):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda"); k = torch.randn_like(q); v = torch.randn_like(q)
    g = torch.randn(B, H, N, D, device="cuda")
    q, k, v = (t.to(dtype).requires_grad_(True) for t in (q, k, v))
    if mode == "math":
        torch.backends.cuda.enable_flash_sdp(False)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dtype is torch.bfloat16)):
        out = F.scaled_dot_product_attention(q, k, v)
    (out * g).sum().backward()
    return [t.grad.float().detach().cpu() for t in (q, k, v)]

# ref: same computation in fp64 on CPU; compare dQ of ("default"=flash, bf16) vs ("math", bf16)
```

Expected: dQ error at dtype-rounding level for every supported `head_dim`.
Actual: D=78 flash bf16 **2.61e-2** (math 6.02e-3); D=90 **4.25e-2**; D=64 control 4.83e-3.

### Source-level hint

`torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h::check_musa_attention_input()` currently
validates only `64 <= head_dim <= 128` and dtype. `main` already carries
`// mutlass tme require head dim % 8 == 0`, but the guard sits **behind `has_dropout`**, so the
dropout-free training path is unguarded — upgrading to `main`/post2 does not fix it.

Suggested one-line fix: move the `% 8` check out of the `has_dropout` branch so that a
non-conforming `head_dim` disables flash and falls back to math:

```cpp
// current: if (enable && has_dropout && !(qk_dim % 8 == 0 && v_dim % 8 == 0)) enable = false;
if (enable && !(qk_dim % 8 == 0 && v_dim % 8 == 0)) enable = false;
```

Suggested CI: compare `MuDNNFlashSDPABwd` against `MuDNNMathSDPABwd` for head_dim 66/78/90/100.

### Minor issues seen in the same code path

- `head_dim = 65` is dispatched and then raises `Flash Attention 2 Not Support HeadDim Odd`
  inside the kernel instead of falling back.
- `qk_dim != v_dim`: flash returns NaN / ~4.0-scale garbage grads, while math raises.
- `can_use_flash_attention()` / `is_flash_attention_available()` return **False** on MUSA even
  when flash is actually used ⇒ cannot be used to detect the active backend.
- `mem_efficient` is never selectable on MUSA (direct RuntimeError).

### Impact and workaround

Our 486M ViT tokenizer (enc head_dim 78, dec head_dim 90, bf16 autocast) diverges from epoch 2
on MUSA (encoder grad norms 1e2…1e5, `--skip_grad` firing on 40 % of steps, val PSNR collapsing)
while the identical recipe is stable on A800. **Workaround validated**: call
`torch.backends.cuda.enable_flash_sdp(False)` (math). End-to-end cost on our step: **+8 %**
(1.019 → 1.098 s/step); at B4/H16/N213/D78 bf16 fwd+bwd the math kernel was actually faster
(0.71 ms vs 2.46 ms). If flash is needed, choose `head_dim % 8 == 0`.

Environment: same as the meta-issue (4×MTT S4000, MUSA driver 5.2.0-server, MUSA SDK
5.2.0 / muDNN 3.4.0, PyTorch 2.11.0.post1 `2b86b38`, torch_musa 2.11.0.post1+b431668
`b431668`, Python 3.10.12).

Attachments: `min_repro.py` (extended sweeps, exit code 1 on >2 % dQ error), `env.txt`.
