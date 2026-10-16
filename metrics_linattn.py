"""
重建指标 — AssocVTC 实验版 (linattn)
==========================================

口径（与仓库主线一致, 见 doc/2026-09-23/ANALYSIS_res_sweep_rootcause.md）:
    · PSNR 单独看**不可信**（"预测均值"也能拿不低的 PSNR）。本模块因此同时
      给出**两个基线**参照, 而不是只报一个 PSNR:
        1. `mean_image_baseline`: 全数据集均值图（"预测均值"型平凡解）;
        2. `per_patch_dc_baseline`: 每张图自己的 16×16 patch 均值（per-patch DC）,
           即"不做任何跨 patch 推理"的朴素重建。
      ⇒ 重建模型只有在**同时**显著优于这两个基线时, PSNR 才有意义。
    · SSIM 用 11×11 高斯窗 (σ=1.5), C1=0.01², C2=0.03², 逐通道后取通道均值;
      边界按**参考实现（Wang 原始 MATLAB / skimage）**的口径: 滤波后裁掉
      pad=(win−1)//2 的边界再平均（零填充 + 裁剪 == valid 卷积, 二者在保留
      区域内逐位相同）。不裁边界会与 skimage 有 ~0.003 的系统性偏差
      （2026-10-15 独立审查 finding 11）。
    · 两个参照基线（**必须打在真实目标上, 不是黑图**）:
        - `per_patch_dc_baseline`: 每图像素 patch 均值重建（per-patch DC）;
        - "常量灰基线"（train/eval 里现算的 psnr_mean_image_baseline）: 用
          ImageNet 均值灰图去比真实 val 目标。`mean_image_baseline()` 是
          "用真实数据集均值图"的可选版本, 当前未被调用（finding 9）。
    · 全部指标在 **[0,1] 像素空间** 上算（pred/target 由归一化空间反变换得到）。
"""

from typing import Optional

import torch
import torch.nn.functional as F


def mse_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) → (B,) 每图 MSE。"""
    return ((pred - target) ** 2).flatten(1).mean(dim=1)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
         eps: float = 1e-10) -> torch.Tensor:
    """(B,3,H,W) → (B,) PSNR (dB)。"""
    return 10.0 * torch.log10(max_val ** 2 / (mse_per_image(pred, target) + eps))


def _gaussian_window(window_size: int = 11, sigma: float = 1.5,
                     channels: int = 3, device=None, dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])
    return w.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5, data_range: float = 1.0) -> torch.Tensor:
    """(B,3,H,W) in [0,data_range] → (B,) SSIM（逐通道后取均值）。"""
    B, C, H, W = pred.shape
    if H < window_size or W < window_size:
        return torch.full((B,), float("nan"), device=pred.device)
    win = _gaussian_window(window_size, sigma, C, pred.device, pred.dtype)
    pad = window_size // 2
    mu_p = F.conv2d(pred, win, padding=pad, groups=C)
    mu_t = F.conv2d(target, win, padding=pad, groups=C)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p2 = F.conv2d(pred * pred, win, padding=pad, groups=C) - mu_p2
    sig_t2 = F.conv2d(target * target, win, padding=pad, groups=C) - mu_t2
    sig_pt = F.conv2d(pred * target, win, padding=pad, groups=C) - mu_pt
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    s = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / \
        ((mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2))
    # 裁掉 pad 边界（与 skimage / 原始 MATLAB 的 "valid" 口径一致;
    # 零填充+裁剪在保留区域内与 valid 卷积逐位相同, 故填充方式无关紧要）
    if pad > 0 and H > 2 * pad and W > 2 * pad:
        s = s[..., pad:-pad, pad:-pad]
    return s.mean(dim=(1, 2, 3))


def mean_image_baseline(train_mean_01: torch.Tensor, target_01: torch.Tensor
                        ) -> torch.Tensor:
    """(3,H,W) 数据集均值图 broadcast 到 (B,3,H,W) 后对 target 的 PSNR。"""
    B = target_01.shape[0]
    m = train_mean_01.to(target_01.device).unsqueeze(0).expand(B, -1, -1, -1)
    return psnr(m, target_01)


def per_patch_dc_baseline(target_01: torch.Tensor, patch: int = 16) -> torch.Tensor:
    """每张图自己的 patch 均值重建（per-patch DC, 不做跨 patch 推理）→ PSNR。

    这是"编码器什么都没学到"的强基线: 只要模型不会比它差才算真会重建。
    """
    B, C, H, W = target_01.shape
    gh, gw = H // patch, W // patch
    p = target_01.reshape(B, C, gh, patch, gw, patch)
    dc = p.mean(dim=(3, 5), keepdim=True).expand_as(p)
    return psnr(dc.reshape(B, C, H, W), target_01)


# ═══ 自检（python metrics_linattn.py）═══
# 2026-10-15 加: 训练冒烟时曾出现 "PSNR=61dB 而 L1=0.85" 的自相矛盾数值,
# 根因是**反归一化的单位口径**（IMAGENET_MEAN/STD 是 [0,1] 标度, 却被又除了
# 一次 255 ⇒ 所有图像缩到 1/255, MSE 变得极小 ⇒ PSNR 虚高到 60+dB、SSIM≈0.99）。
# 训练损失不受影响（两侧同口径）, 但**指标与可视化全错**。以下 §4 是该 bug 的
# 回归测试: 只要单位口径再错一次, 这里立刻失败。

if __name__ == "__main__":
    import torch as _t
    from model_v2_linattn import (IMAGENET_MEAN, IMAGENET_STD, image_to_patches,
                                  patches_to_image)

    _t.manual_seed(0)
    print("=" * 74)
    print("[1] PSNR 定义（已知 MSE）")
    tgt = _t.rand(4, 3, 64, 64)
    assert abs(psnr(tgt, tgt).mean().item() - 100.0) < 1e-3, "自比应为满分"
    d = _t.full_like(tgt, 0.1)
    got = psnr(tgt + d, tgt).mean().item()
    assert abs(got - 20.0) < 1e-3, f"MSE=0.01 应得 20dB, got {got}"
    print(f"    自比 100dB; MSE=0.01 → {got:.3f}dB (期望 20)")

    print("[2] SSIM 定义（自比 / 对称 / 单调）")
    s_self = ssim(tgt, tgt).mean().item()
    assert abs(s_self - 1.0) < 1e-6, f"SSIM 自比应为 1, got {s_self}"
    a, b = _t.rand(2, 3, 64, 64), _t.rand(2, 3, 64, 64)
    assert abs(ssim(a, b).mean().item() - ssim(b, a).mean().item()) < 1e-6
    s_near = ssim(tgt, (tgt + 0.02 * _t.randn_like(tgt)).clamp(0, 1)).mean().item()
    s_far = ssim(tgt, (tgt + 0.3 * _t.randn_like(tgt)).clamp(0, 1)).mean().item()
    assert s_near > s_far, "SSIM 应随扰动单调下降"
    # 边界口径: 与"valid 卷积（不跨边界）"的独立实现逐位对齐
    import torch.nn.functional as _F

    def _ref_ssim(p, t, ws=11, sigma=1.5):
        B, C, H, W = p.shape
        co = _t.arange(ws, dtype=p.dtype) - ws // 2
        g = _t.exp(-(co ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w = (g[:, None] @ g[None, :]).expand(C, 1, ws, ws).contiguous()
        cv = lambda z: _F.conv2d(z, w, groups=C)          # valid, 无填充
        mp, mt = cv(p), cv(t)
        sp2 = cv(p * p) - mp ** 2
        st2 = cv(t * t) - mt ** 2
        spt = cv(p * t) - mp * mt
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        return (((2 * mp * mt + c1) * (2 * spt + c2)) /
                ((mp ** 2 + mt ** 2 + c1) * (sp2 + st2 + c2))).mean(dim=(1, 2, 3))

    # 同一张图: 用同一对图比较口径
    p2 = (tgt + 0.1 * _t.randn_like(tgt)).clamp(0, 1)
    assert abs(ssim(tgt, p2).mean().item() - _ref_ssim(tgt, p2).mean().item()) < 1e-5, \
        "SSIM 边界口径与 valid 卷积参考实现不一致"
    print(f"    自比 {s_self:.6f}; 小扰动 {s_near:.4f} > 大扰动 {s_far:.4f}; "
          f"对称 OK; 边界口径 == valid 卷积参考 "
          f"({ssim(tgt, p2).mean().item():.6f})")

    print("[3] 基线口径（必须打在**真实**目标上, 不是黑图）")
    # 合成"类自然图"（平滑梯度 + 噪声）
    yy, xx = _t.meshgrid(_t.linspace(0, 1, 64), _t.linspace(0, 1, 64),
                         indexing="ij")
    nat = (0.5 + 0.3 * xx + 0.1 * yy).unsqueeze(0).expand(4, 3, 64, 64).clone()
    nat = (nat + 0.02 * _t.randn_like(nat)).clamp(0, 1)
    dc = per_patch_dc_baseline(nat, 16).mean().item()
    const = _t.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).expand_as(nat)
    cm = psnr(const, nat).mean().item()
    assert 15.0 < dc < 45.0, f"patch-DC 基线 {dc}dB 不在合理范围（15-45）"
    assert 4.0 < cm < 20.0, f"均值图基线 {cm}dB 不在合理范围（4-20）"
    print(f"    patch-DC 基线 {dc:.2f}dB; 均值图基线 {cm:.2f}dB (均为合理量级)")

    print("[4] ★回归: 反归一化单位口径（[0,1] 标度 mean/std, 不许再 /255）")
    x01 = _t.rand(2, 3, 64, 64)
    m = _t.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    s = _t.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    xn = (x01 - m) / s                              # 归一化（训练输入口径）
    pix = image_to_patches(xn, 16)
    back = patches_to_image(pix, 64, 64, 16, IMAGENET_MEAN, IMAGENET_STD,
                            clamp_255=False).clamp(0.0, 1.0)
    err = (back - x01).abs().max().item()
    assert err < 1e-5, f"反归一化回不到 [0,1]: max|diff|={err}"

    # 旧 bug 的机制: 两侧图像都被额外缩到 1/255 ⇒ MSE 缩 1/255² ⇒ PSNR 虚高 +48dB。
    # 用一个"很差的预测"（常数灰）演示: 正确口径 ~7dB, 旧口径 ~55dB（自相矛盾:
    # 明明 L1 很大却 PSNR 很高）。
    bad_pred = _t.full_like(x01, 0.45)
    ok_db = psnr(bad_pred, x01).mean().item()
    buggy_db = psnr(bad_pred / 255.0, x01 / 255.0).mean().item()
    assert ok_db < 12.0, f"常数预测的 PSNR 应很低, got {ok_db}"
    assert buggy_db - ok_db > 40.0, "旧 bug 的虚高未复现, 自检失效"
    print(f"    往返 max|diff|={err:.2e} (OK); 常数预测: 正确 {ok_db:.1f}dB vs "
          f"旧口径 {buggy_db:.1f}dB (+{buggy_db-ok_db:.1f}dB 虚高) ⇒ 能拦住")

    print("[5] ★回归: 直接跑**真实调用点**（train_linattn.denorm01）")
    # §4 只测了 model_v2_linattn 的往返; 真正的 bug 出在消费方 train/eval 的
    # denorm01 上。这里直接 import 并断言, 保证再写错 /255 时立刻失败
    # （2026-10-15 独立审查 finding 3）。
    from train_linattn import denorm01 as _denorm01
    x01b = _t.rand(2, 3, 224, 224)
    xnb = (x01b - m) / s
    pixb = image_to_patches(xnb, 16)
    backb = _denorm01(pixb, 224, 16)
    assert backb.shape == x01b.shape, backb.shape
    assert 0.0 <= float(backb.min()) and float(backb.max()) <= 1.0, \
        f"denorm01 应返回 [0,1], got [{backb.min()}, {backb.max()}]"
    e = (backb - x01b).abs().max().item()
    assert e < 1e-5, f"train_linattn.denorm01 往返失败: max|diff|={e}"
    assert 0.2 < float(backb.mean()) < 0.8, \
        f"denorm01 均值 {float(backb.mean()):.5f} 不合理（多除 255 会掉到 ~0.002）"
    print(f"    train_linattn.denorm01 往返 max|diff|={e:.2e}, "
          f"mean={float(backb.mean()):.4f} ∈ [0,1] OK")
    print("=" * 74)
    print("ALL CHECKS PASSED")
