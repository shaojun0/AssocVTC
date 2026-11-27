"""LlamaGen VQ-16 (f16, 16,384-entry codebook) tokenizer wrapper —— ids in / pixels out.

独立模块（刻意"干净"）: 只依赖 stdlib + torch; 官方 LlamaGen 源码树在**首次实例化时**
才惰性 import（只加它的路径, 不 import 本仓库任何其它模块, 不强制 GPU）。

API
---
    tok = LlamaGenVQ16(ckpt="/root/autodl-tmp/vq/llamagen_vq16_c2i.pt", device="cpu")
    tok.grid(224)                # (14, 14)  —— img_size 必须是 downsample=16 的倍数
    ids = tok.encode_ids(x01)    # (B,3,H,W) in [0,1]  ->  (B, h*w) int64, h=H/16, w=W/16
    rec = tok.decode_ids(ids)    # (B,h*w) 或 (B,h,w) ->  (B,3,16h,16w) in [0,1]
    rec = tok.recon(x01)         # decode_ids(encode_ids(x01)) —— 只走 token 的重建

id 顺序（row-major）
--------------------
LlamaGen 的 `VectorQuantizer.forward` 先把特征 `b c h w -> b h w c` 再 flatten，
因此扁平 id 序列的第 k 个对应网格的 `(k // w, k % w)`（行主序）；`decode_code`
用 `reshape(B,h,w,C).permute(0,3,1,2)` 还原。⇒ 扁平 ids / 网格 ids 与 encode
输出完全一致（`__main__` 自检逐像素验证）。

数值口径
--------
* 模型 fp32、全部接口 `torch.no_grad()`、参数 `requires_grad=False`、eval 模式。
* `decode_ids` 的 decoder 原始输出可越出 [-1,1]（实测约 [-1.3, 1.6]），这里统一做
  `((o + 1) / 2).clamp(0, 1)` 映到 [0,1]（与 LlamaGen 取图口径一致）。
* ⚠️ LlamaGen 的 quantizer 在 eval 下用 straight-through `z + (z_q - z).detach()`
  返回 quant：fp32 下它与 codebook 向量差 ~3e-8，decoder 会把这个舍入差放大到
  ~1e-4 像素量级。`decode_ids` 走的是纯 codebook 路径（`decode_code`），即**与 id
  严格对应**的像素；`decode(encode(x))` 的那条路才带 straight-through 舍入。
  两者在 float64 下差 ~4e-14（数学上同一个量），见 `__main__` 自检。
"""

import math
import os
import sys
import time

import torch

# ─── 默认位置（可用环境变量覆盖; 只在本文件首次实例化时才需要存在）───────────────
LLAMAGEN_ROOT = os.environ.get("LLAMAGEN_ROOT", "/root/autodl-tmp/vq/LlamaGen")
DEFAULT_CKPT = os.environ.get(
    "LLAMAGEN_VQ16_CKPT", "/root/autodl-tmp/vq/llamagen_vq16_c2i.pt")

_VQ_MODELS = None      # 惰性缓存: VQ_models 字典


def _vq_models():
    """惰性 import 官方 LlamaGen 的 VQ 模型工厂（避免 import 本模块即依赖它）。"""
    global _VQ_MODELS
    if _VQ_MODELS is None:
        root = os.path.abspath(LLAMAGEN_ROOT)
        if not os.path.isdir(root):
            raise RuntimeError(
                f"找不到 LlamaGen 源码树: {root}（可用 LLAMAGEN_ROOT 环境变量指定）")
        if root not in sys.path:
            sys.path.insert(0, root)
        from tokenizer.tokenizer_image.vq_model import VQ_models
        _VQ_MODELS = VQ_models
    return _VQ_MODELS


def _load_state_dict(ckpt):
    """兼容 {'model': sd} 与裸 sd 两种 ckpt 格式。"""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        return ck["model"]
    return ck


class LlamaGenVQ16:
    """LlamaGen VQ-16 编解码包装: 16× 下采样 / 16,384 codes / codebook dim 8。

    输入像素一律 **[0,1]**（内部自己转 `2x-1`），输出像素一律 **[0,1]**。
    """

    n_embed = 16384          # codebook 条目数
    downsample = 16          # 空间下采样倍数（f16）
    codebook_dim = 8         # 每个 code 的嵌入维数
    name = "LlamaGen-VQ-16"

    def __init__(self, ckpt=DEFAULT_CKPT, device="cpu"):
        self.ckpt = str(ckpt)
        self.device = torch.device(device)
        if not os.path.isfile(self.ckpt):
            raise FileNotFoundError(f"找不到 VQ-16 权重: {self.ckpt}")
        VQ_models = _vq_models()
        net = VQ_models["VQ-16"](codebook_size=self.n_embed,
                                 codebook_embed_dim=self.codebook_dim)
        net.load_state_dict(_load_state_dict(self.ckpt), strict=True)
        net.to(self.device).float().eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.codebook_dim = int(net.quantize.e_dim)   # 以模型实际配置为准（=8）

    # ─── 网格 ────────────────────────────────────────────────────────────────
    def grid(self, img_size):
        """`img_size` → `(h, w)` 的 code 网格。要求每个边长都是 `downsample` 的正整数倍。"""
        if isinstance(img_size, (tuple, list)):
            if len(img_size) != 2:
                raise ValueError(f"img_size 元组应为 (H, W), got {img_size!r}")
            hw = (int(img_size[0]), int(img_size[1]))
        else:
            hw = (int(img_size), int(img_size))
        for s in hw:
            if s <= 0 or s % self.downsample:
                raise ValueError(
                    f"img_size 必须是 {self.downsample} 的正整数倍, got {s}")
        return (hw[0] // self.downsample, hw[1] // self.downsample)

    # ─── 内部工具 ────────────────────────────────────────────────────────────
    def _as_x01(self, x01):
        x = torch.as_tensor(x01)
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"x01 形状应为 (B,3,H,W), got {tuple(x.shape)}")
        h, w = self.grid((x.shape[-2], x.shape[-1]))
        if not x.is_floating_point():
            x = x.float()
        return x.to(self.device, torch.float32), h, w

    # ─── 编码 / 解码 ────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_ids(self, x01):
        """(B,3,H,W) ∈ [0,1] → (B, h*w) int64，row-major（见模块 docstring）。"""
        x, h, w = self._as_x01(x01)
        latent, _, [_, _, ids] = self.net.encode(2.0 * x - 1.0)
        if tuple(latent.shape[-2:]) != (h, w):
            raise RuntimeError(
                f"latent 空间尺寸 {tuple(latent.shape[-2:])} != 期望 {(h, w)}")
        ids = ids.reshape(x.shape[0], h * w)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.n_embed):
            raise RuntimeError("encode 返回的 id 越界（codebook 与模型不匹配?）")
        return ids.contiguous()

    @torch.no_grad()
    def decode_ids(self, ids):
        """(B,h*w) 或 (B,h,w) ids → (B,3,16h,16w) ∈ [0,1]。

        扁平 ids 只能推断正方形网格（h*w 需为完全平方数）；非正方形请传 (B,h,w)。
        """
        ids = torch.as_tensor(ids)
        if ids.dim() == 3:
            B, h, w = ids.shape
        elif ids.dim() == 2:
            B, n = ids.shape
            h = w = math.isqrt(n)
            if h * w != n:
                raise ValueError(
                    f"扁平 ids 长度 {n} 不是完全平方数 ⇒ 无法推断网格；"
                    f"请传 (B,h,w) 形状的 ids")
        else:
            raise ValueError(f"ids 形状应为 (B,h*w) 或 (B,h,w), got {tuple(ids.shape)}")
        if ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            raise TypeError(f"ids 必须是整型张量, got {ids.dtype}")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.n_embed):
            raise ValueError(f"id 越界: 应在 [0,{self.n_embed})")
        ids = ids.reshape(B, h * w).to(self.device, torch.long)
        out = self.net.decode_code(ids, (B, self.codebook_dim, h, w))
        return ((out.float() + 1.0) * 0.5).clamp_(0.0, 1.0)

    @torch.no_grad()
    def recon(self, x01):
        """只走 token 的重建: `decode_ids(encode_ids(x01))`，输出 (B,3,H,W) ∈ [0,1]。"""
        return self.decode_ids(self.encode_ids(x01))

    def __repr__(self):
        return (f"{self.__class__.__name__}(n_embed={self.n_embed}, "
                f"downsample={self.downsample}, codebook_dim={self.codebook_dim}, "
                f"device={self.device}, ckpt={self.ckpt!r})")


# ═══ 自检: python vq_tokenizer.py ════════════════════════════════════════════

def _self_check():
    torch.manual_seed(0)
    print("=" * 78)
    tok = LlamaGenVQ16(device="cpu")
    print(f"[0] {tok}")
    print(f"    grid(224)={tok.grid(224)}  grid(256)={tok.grid(256)}  "
          f"grid(448)={tok.grid(448)}")

    # ── [1] 形状 / 值域 / 任意 B / 耗时 ────────────────────────────────────
    print("[1] encode → decode → recon（B=1/2/5, 224²）")
    for B in (1, 2, 5):
        x = torch.rand(B, 3, 224, 224)
        t0 = time.time()
        ids = tok.encode_ids(x)
        t1 = time.time()
        rec = tok.decode_ids(ids)
        t2 = time.time()
        rec2 = tok.recon(x)
        t3 = time.time()
        assert tuple(ids.shape) == (B, 196), ids.shape
        assert ids.dtype == torch.int64, ids.dtype
        assert tuple(rec.shape) == (B, 3, 224, 224), rec.shape
        assert 0.0 <= float(rec.min()) and float(rec.max()) <= 1.0, \
            f"decode 输出应在 [0,1], got [{rec.min()}, {rec.max()}]"
        assert torch.equal(rec2, rec), "recon 应等价于 decode_ids(encode_ids(x))"
        assert tuple(rec2.shape) == tuple(x.shape)
        print(f"    B={B}: ids{tuple(ids.shape)} {ids.dtype}  "
              f"recon{tuple(rec2.shape)} ∈[{rec2.min():.3f},{rec2.max():.3f}]  "
              f"enc {t1-t0:.2f}s dec {t2-t1:.2f}s "
              f"({(t2-t0)/B*1000:.0f} ms/img)")

    # ── [2] 224² + 非 224 的整倍数 ─────────────────────────────────────────
    print("[2] 其它尺寸 / ids 的扁平 vs 网格形式")
    x256 = torch.rand(2, 3, 256, 256)
    ids256 = tok.encode_ids(x256)
    assert tuple(ids256.shape) == (2, 256), ids256.shape      # 16×16
    rec256 = tok.decode_ids(ids256)
    assert tuple(rec256.shape) == (2, 3, 256, 256), rec256.shape
    assert torch.equal(rec256, tok.decode_ids(ids256.reshape(2, 16, 16))), \
        "扁平 (B,h*w) 与网格 (B,h,w) 两种 ids 输入必须等价"
    assert torch.equal(rec256, tok.recon(x256))
    try:
        tok.grid(220)
        raise AssertionError("img_size=220 应被拒绝（非 16 的倍数）")
    except ValueError:
        pass
    print(f"    256² → grid {tok.grid(256)} / ids{tuple(ids256.shape)} / "
          f"recon{tuple(rec256.shape)}; grid(220) 正确报错; 扁平==网格 OK")

    # ── [3] ids 一致性: decode_ids(encode_ids(x)) vs 各参考路径 ──────────────
    print("[3] ★ ids 一致性自检（224², B=2）")
    x = torch.rand(2, 3, 224, 224)
    B, h, w = 2, 14, 14
    ids = tok.encode_ids(x)                                  # 扁平, row-major
    rec = tok.decode_ids(ids)
    with torch.no_grad():
        quant, _, [_, _, ids2] = tok.net.encode(2.0 * x - 1.0)   # straight-through 路径
        rec_st = ((tok.net.decode(quant).float() + 1.0) * 0.5).clamp_(0.0, 1.0)
        entry = tok.net.quantize.get_codebook_entry(
            ids.reshape(B, h * w), (B, tok.codebook_dim, h, w))
        rec_cb = ((tok.net.decode(entry).float() + 1.0) * 0.5).clamp_(0.0, 1.0)
        rec_dc = tok.decode_ids(ids2.reshape(B, h * w))
    assert torch.equal(ids, ids2.reshape(B, h * w)), "encode_ids 的 id 顺序必须与官方 encode 一致"
    d_cb = float((rec - rec_cb).abs().max())
    d_st = float((rec - rec_st).abs().max())
    d_dc = float((rec - rec_dc).abs().max())
    print(f"    ids 与官方 encode 逐位相同: {torch.equal(ids, ids2.reshape(B, h*w))}")
    print(f"    vs decode(codebook[ids])      max|Δ| = {d_cb:.3e}   (期望 0, 同一条路径)")
    print(f"    vs decode_code(官方 ids)       max|Δ| = {d_dc:.3e}   (期望 0)")
    print(f"    vs decode(encode(x)) [fp32]    max|Δ| = {d_st:.3e}   "
          f"(straight-through 舍入 ~3e-8 被 decoder 放大)")
    assert d_cb == 0.0 and d_dc == 0.0, (d_cb, d_dc)
    assert d_st < 1e-3, d_st          # fp32 下实测 ~8.5e-5, 见下方 f64 证明

    # ── [4] float64 证明: 上述两条路径数学上同一个量 ────────────────────────
    print("[4] ★ float64 复算: decode(encode(x)) 与 decode_ids(encode_ids(x)) 应重合")
    VQ_models = _vq_models()
    net64 = VQ_models["VQ-16"](codebook_size=tok.n_embed,
                               codebook_embed_dim=tok.codebook_dim).double()
    net64.load_state_dict({k: v.double() for k, v in _load_state_dict(tok.ckpt).items()},
                          strict=True)
    net64.eval()
    with torch.no_grad():
        x64 = (2.0 * x.double() - 1.0)
        quant64, _, [_, _, ids64] = net64.encode(x64)
        rec64_ids = net64.decode_code(ids64.reshape(B, h * w), (B, tok.codebook_dim, h, w))
        rec64_ids = ((rec64_ids + 1.0) * 0.5).clamp_(0.0, 1.0)
        rec64_st = ((net64.decode(quant64) + 1.0) * 0.5).clamp_(0.0, 1.0)
    d64 = float((rec64_ids - rec64_st).abs().max())
    assert torch.equal(ids64.reshape(B, h * w), ids), "f64 下 id 也应完全相同"
    print(f"    f64 max|Δ| = {d64:.3e}  (fp32 的 {d_st:.3e} 纯属舍入; spec 的 <1e-5 在此成立)")
    assert d64 < 1e-5, d64

    print("=" * 78)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
