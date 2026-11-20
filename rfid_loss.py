"""批次 rFID（reconstruction FID）辅助损失 —— 2026-11-20.

背景（用户口径，参考会话 session-0389af9d 最后一条记录的结论）
------------------------------------------------------------------
rFID = 把 FID 用在「重建图 vs 原图」上。它**可微、可以当损失**
（先例: Mathiasen & Hvilshøj, "Backpropagating through Fréchet Inception
Distance", arXiv:2009.14075, FastFID；GAN 上 FID 22→11），但它是**分布级**
统计量、不惩罚逐图张冠李戴 ⇒ 只能当**辅助项**，主损失仍必须是逐样本像素 L1。

本实现刻意做的选择
------------------
1. 特征 = 官方 FID InceptionV3 的 pool3（2048-d，`pytorch_fid.inception.InceptionV3`，
   resize 299×299 + [0,1]→[-1,1]，与 `tools/eval_rfid.py` **同一权重/同口径**）。
   冻结、eval 模式（BN 用 running stats）；梯度只经输入图反传到解码器。
2. 「批次」= 当前 GPU 的 micro-batch（默认 32 图/卡；全局 128 由 DDP 各卡本地项
   均值给出）。不做跨卡 all_gather —— 避免 DDP「同一全局标量 × averaging」把
   梯度整体缩小 world 倍的坑。
3. Fréchet 用**精确 Gram 恒等式**，不需要 2048×2048 的 `sqrtm`（FastFID 的诉求）：
   设中心化特征 X, Y ∈ R^{n×D}，Σ = XᵀX/(n-1)，
       tr((Σx Σy)^{1/2}) = sum_sv(X Yᵀ) / (n-1)
   ⇒ FID = ||μx−μy||² + (||X||_F² + ||Y||_F² − 2·sum_sv(X Yᵀ)) / (n-1)
   只需 n×n 的 SVD，且 `torch.linalg.svdvals` 可反传。batch 小于特征维也**数值精确**。
4. 量纲：默认除以特征维 D（`normalize_by_dim`），使损失尺度 ~O(0.01–1)，便于配权重。
   原始 rFID 同时返回，用于日志监控（防 reward hacking）。
"""
import os
import sys

# MUSA 机器上 /usr/local 的 torchvision 是空壳（没有 models），完整 build 在
# /home/torchvision/build/...；pytorch_fid.inception 依赖 torchvision.models。
# 先补 path、清掉可能的空壳 torchvision，再 import。
_TV_BUILD = "/home/torchvision/build/lib.linux-x86_64-cpython-310"
if os.path.isdir(_TV_BUILD) and _TV_BUILD not in sys.path:
    sys.path.insert(0, _TV_BUILD)

import torch
import torch.nn as nn

try:
    import torchvision  # noqa: F401
    _TV_OK = hasattr(torchvision, "models")
except Exception:                                    # pragma: no cover
    _TV_OK = False

if not _TV_OK and _TV_BUILD in sys.path:
    # 之前可能已 import 到空壳 torchvision（train_linattn 顶部 try 过）
    for _k in [k for k in list(sys.modules)
               if k == "torchvision" or k.startswith("torchvision.")]:
        del sys.modules[_k]
    import torchvision                           # noqa: F401
    _TV_OK = hasattr(torchvision, "models")

if not _TV_OK:                                       # pragma: no cover
    raise ImportError(
        "BatchRFIDLoss 需要完整的 torchvision（pytorch_fid.inception 依赖 "
        f"torchvision.models）。请 export PYTHONPATH={_TV_BUILD}:$PYTHONPATH")

from pytorch_fid.inception import InceptionV3      # noqa: E402


def frechet_distance(fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """两组 (n, D) 特征之间的 Fréchet 距离（可微，精确，仅 n×n SVD）。

    fx/fy: (n, D)，同一 batch 的重建 / 真值 Inception 特征。

    ⚠️ 必须在 fp32 下算：训练循环外层是 bf16 autocast，`dx @ dyᵀ` 会被 autocast
    成 bf16，而 torch_musa 的 `linalg_svdvals` 只有 fp32 kernel（bf16 直接报
    `linalg_svd_cpu not implemented for BFloat16`）。这里显式关 autocast。
    """
    assert fx.dim() == 2 and fx.shape == fy.shape, (fx.shape, fy.shape)
    n = int(fx.shape[0])
    assert n >= 2, "rFID 需要 batch >= 2"
    with torch.autocast("cuda", enabled=False):
        fx, fy = fx.float(), fy.float()
        mx, my = fx.mean(dim=0), fy.mean(dim=0)
        dx, dy = fx - mx, fy - my
        gram = dx @ dy.transpose(0, 1)              # (n, n), fp32
        sv = torch.linalg.svdvals(gram)             # 可反传
        tr_terms = (dx.pow(2).sum() + dy.pow(2).sum()
                    - 2.0 * sv.sum()) / float(n - 1)
        return (mx - my).pow(2).sum() + tr_terms


class BatchRFIDLoss(nn.Module):
    """冻结 FID-InceptionV3 pool3 + 可微 Fréchet 距离（批次级）。

    forward(pred01, tgt01) → {"loss": rFID/D（或 rFID）, "raw": rFID}
    两张输入都是 (B, 3, H, W)、值域 [0, 1]（与 `model_v2_linattn.reconstruct` 同口径）。
    """

    def __init__(self, block_idx: int = 3, dim: int = 2048,
                 normalize_by_dim: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.normalize_by_dim = bool(normalize_by_dim)
        self.inception = InceptionV3([int(block_idx)],
                                     resize_input=True,
                                     normalize_input=True,
                                     requires_grad=False)
        self.inception.eval()
        for p in self.inception.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):     # 永远保持 eval（冻结 BN）
        super().train(False)
        self.inception.eval()
        return self

    def _features(self, img01: torch.Tensor, grad: bool) -> torch.Tensor:
        ctx = torch.enable_grad if grad else torch.no_grad
        with ctx():
            with torch.autocast("cuda", enabled=False):
                out = self.inception(img01.float())
        f = out[0].flatten(1).float()
        if f.shape[1] > self.dim:
            f = f[:, :self.dim]
        return f

    def forward(self, pred01: torch.Tensor, tgt01: torch.Tensor) -> dict:
        fp = self._features(pred01, grad=True)
        ft = self._features(tgt01, grad=False)
        raw = frechet_distance(fp, ft)
        loss = raw / float(fp.shape[1]) if self.normalize_by_dim else raw
        return {"loss": loss, "raw": raw}
