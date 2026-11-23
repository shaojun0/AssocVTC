"""纯 CPU 测试: latent KD 通路（I1/I2）—— 2026-11-23.

不依赖 diffusers / timm / 数据 / GPU:

    python3 tools/test_kd_latent_cpu.py        # 全过 exit 0, 否则非 0

覆盖（对应 `doc/2026-11-23/NOTES_kd_vs_alit_code.md` §3-I1/I2 的验收项）:

  1. `z_head=1` ⇒ `z_hat` 形状 `(B,T,4,H/8,W/8)`;
  2. `MSE(z_hat*s, z_t*s)` 反传后 `z_head` / `zs_proj` / **编码器**梯度全部非零
     ⇒ 这是"KD 梯度能否到达被传输 payload"的**实验证据**（打印最大梯度值）;
  3. `z_head=0` ⇒ `forward` 无 `z_hat` 键、`state_dict()` 键与不带该字段时**完全相同**
     （旧 ckpt 仍可 `strict=True` 载入）;
  4. `kd_loss.latent_mse` == 手算 `((a*s − b*s)**2).mean()`;
  5. diffusers 不可用（或本地无 SD-VAE 权重）⇒ teacher 相关检查 `SKIP`, **不算失败**。

⚠️ 这个测试只碰 CPU 与随机张量: 不读 val 数据、不跑 teacher 的 teacher（除了第 5 项）。
"""
import os
import sys
import traceback

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                    # 让 `model_v2_linattn` / `kd_loss` 可 import

import kd_loss                                            # noqa: E402
from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn  # noqa: E402

SCALE = 0.18215
FAILED = []


class SkipTest(Exception):
    """不算失败的跳过（缺 diffusers / 缺权重）。"""


def check(name, fn):
    try:
        info = fn()
        print(f"[PASS] {name}" + (f" | {info}" if info else ""))
    except SkipTest as e:
        print(f"[SKIP] {name} | {e}")
    except Exception as e:                    # noqa: BLE001
        print(f"[FAIL] {name} | {type(e).__name__}: {e}")
        traceback.print_exc()
        FAILED.append(name)


def small_cfg(**kw):
    """小配置（CPU 秒级）: 64²/patch16 ⇒ N=16, 3 步循环读出, z_hat 8×8。"""
    base = dict(img_size=64, patch=16, num_specials=8, enc_dim=64, enc_depth=2,
                enc_heads=4, dec_dim=64, dec_depth=2, dec_heads=4,
                pixel_hidden=128, dec_recurrent=True, dec_steps="1,4,8", zs_dim=12)
    base.update(kw)
    return LinAttnConfig(**base)


def gmax(mod):
    """模块内最大 |grad|（无梯度 ⇒ 0.0）。"""
    gs = [float(p.grad.abs().max()) for p in mod.parameters() if p.grad is not None]
    return max(gs) if gs else 0.0


# ── 1. z_hat 形状 + 潜变量头参数量 ───────────────────────────────────────────
def t_shape():
    torch.manual_seed(0)
    m = SRPhase1LinAttn(small_cfg(z_head=1))
    out = m(torch.randn(2, 3, 64, 64))
    z = out["z_hat"]
    assert z.shape == (2, 3, 4, 8, 8), z.shape          # (B,T,4,H/8,W/8)
    n = sum(p.numel() for p in m.z_head.parameters())
    assert n < 1_000_000, f"z_head 参数 {n:,} 应远小于 1M"
    return f"z_hat{tuple(z.shape)} z_head 参数 {n:,} (<1M)"


# ── 2. KD 梯度能到达 payload（z_head / zs_proj / 编码器）────────────────────────
def t_grad_to_payload():
    torch.manual_seed(0)
    m = SRPhase1LinAttn(small_cfg(z_head=1)).train()
    out = m(torch.randn(2, 3, 64, 64))
    torch.manual_seed(7)
    z_t = torch.randn_like(out["z_hat"])                # 假教师 latent（未缩放口径）
    loss = kd_loss.latent_mse(out["z_hat"], z_t, SCALE)
    loss.backward()                                     # 只有 latent 损失 ⇒ 梯度必经 payload
    g_zh, g_zs, g_enc = gmax(m.z_head), gmax(m.zs_proj), gmax(m.encoder)
    assert g_zh > 0.0, f"z_head 无梯度 ({g_zh})"
    assert g_zs > 0.0, f"zs_proj（被传输的 payload）无梯度 ({g_zs})"
    assert g_enc > 0.0, f"编码器无梯度 ({g_enc})"
    return (f"loss={float(loss.detach()):.6f}; max|g| z_head={g_zh:.3e} "
            f"zs_proj={g_zs:.3e} encoder={g_enc:.3e} (全部 >0 ⇒ 梯度可达 payload)")


# ── 3. z_head=0 = 旧行为（无 z_hat 键 / state_dict 键不变 / strict 互载）────────
def t_off_is_old():
    torch.manual_seed(0)
    m_off = SRPhase1LinAttn(small_cfg(z_head=0))        # 显式关
    m_old = SRPhase1LinAttn(small_cfg())               # 不写该字段（= 改造前的 cfg 口径）
    k_off, k_old = set(m_off.state_dict()), set(m_old.state_dict())
    assert k_off == k_old, f"state_dict 键变了: {k_off ^ k_old}"
    assert not any(k.startswith("z_head") for k in k_off), "z_head=0 仍注册了参数"
    m_old.load_state_dict(m_off.state_dict(), strict=True)   # 旧 ckpt strict=True 可载入
    m_off.load_state_dict(m_old.state_dict(), strict=True)
    out = m_off(torch.randn(2, 3, 64, 64))
    assert "z_hat" not in out, f"z_head=0 却出现了 z_hat 键: {sorted(out)}"
    assert {"loss", "recon", "F_hat", "Y_pix", "target_pix", "step_l1"} <= set(out)
    # 开着的模型一定多出 z_head.* 键（证明两条路径确实不同, 不是两边都关着）
    k_on = set(SRPhase1LinAttn(small_cfg(z_head=1)).state_dict())
    assert (k_on - k_off) == {k for k in k_on if k.startswith("z_head")}, \
        f"开 z_head 后多出的键不止 z_head.*: {sorted((k_on - k_off))}"
    return (f"keys={len(k_off)} 个（与旧 cfg 逐 key 同, strict=True 互载 OK）; "
            f"forward 无 z_hat; 开时多 {len(k_on - k_off)} 个 z_head.* key")


# ── 4. latent_mse == 手算（按 scale 缩放后）────────────────────────────────────
def t_latent_mse():
    torch.manual_seed(0)
    a, b = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    for s in (SCALE, 1.0, 0.5):
        got = float(kd_loss.latent_mse(a, b, s))
        want = float(((a * s - b * s) ** 2).mean())
        assert abs(got - want) <= 1e-7 * max(1.0, abs(want)), (s, got, want)
    assert abs(float(kd_loss.LATENT_SCALE) - SCALE) < 1e-12, kd_loss.LATENT_SCALE
    assert abs(float(kd_loss.latent_mse(a, b)) - float(
        kd_loss.latent_mse(a, b, SCALE))) < 1e-12, "默认 scale 不是 LATENT_SCALE"
    return f"scale∈{{0.18215,1.0,0.5}} 均与 ((a·s−b·s)²).mean() 一致; LATENT_SCALE={SCALE}"


# ── 5. teacher（需要 diffusers + 本地 sd-vae 权重; 缺 ⇒ SKIP）──────────────────
def t_teacher():
    try:
        import diffusers  # noqa: F401
    except Exception as e:                    # noqa: BLE001
        raise SkipTest(f"diffusers 不可用 ({type(e).__name__})")
    try:
        from kd_loss import SDVAETeacher
        t = SDVAETeacher()                    # 默认 sd-vae-ft-mse + local_files_only
    except Exception as e:                    # noqa: BLE001
        raise SkipTest(f"SD-VAE 权重不可用 ({type(e).__name__}: {e})")
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    z = t.encode_codes(x)
    assert z.shape == (2, t.latent_channels, 64 // t.down, 64 // t.down), z.shape
    assert not z.requires_grad, "encode_codes 应在 no_grad 内"
    y = t.decode_latent(z)
    assert y.shape == x.shape and 0.0 <= float(y.min()) <= float(y.max()) <= 1.0
    assert torch.allclose(y, t(x), atol=1e-6), "decode_latent(encode_codes(x)) != forward(x)"
    assert float(kd_loss.latent_mse(z, z, SCALE)) == 0.0
    return (f"latent={tuple(z.shape)} (down={t.down}); "
            f"decode_latent∘encode_codes == forward OK; latent_mse(z,z)=0 OK")


def main():
    print("=" * 78)
    print("[kd-latent CPU] 纯 CPU 测试: z_head / latent_mse / 梯度可达 payload")
    check("1. z_head=1 ⇒ z_hat (B,T,4,H/8,W/8)", t_shape)
    check("2. latent KD 梯度到达 payload（z_head/zs_proj/encoder）", t_grad_to_payload)
    check("3. z_head=0 ⇒ 无 z_hat、state_dict 键不变", t_off_is_old)
    check("4. latent_mse == 手算 MSE（按 scale 缩放）", t_latent_mse)
    check("5. teacher latent 往返（缺 diffusers/权重则 SKIP）", t_teacher)
    print("=" * 78)
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("ALL CPU CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
