"""VGG16 感知损失（L1 on ImageNet-normalized VGG16 features）。

用途：纯逐像素 L1 训练出来的重建偏模糊，FID/rFID 这类分布级指标被模糊严重惩罚。
感知损失是压低 rFID 最便宜的手段（不需要 GAN），常用于把"模糊但 PSNR 高"换成
"略低 PSNR 但纹理真实"。参考 Johnson et al. 2016 / LPIPS 的 VGG 特征 L1 变体。

接口与 rfid_loss.BatchRFIDLoss 保持一致：`loss = fn(pred01, tgt01)`，
pred 为 [B,3,H,W]、[0,1]（梯度只回传 pred）。fp32 计算（在 autocast 关闭的上下文里调用）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """VGG16(ImageNet) features L1。

    layers: VGG16.features 里的模块下标（默认 3=relu1_2, 8=relu2_2, 15=relu3_3）
    resize: 送入 VGG 前的短边尺寸（默认 224；越大越贵，纹理项更细）
    """

    def __init__(self, layers=(3, 8, 15), resize=224, layer_weights=None):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        try:
            feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except Exception as e:                              # 离线兜底
            print(f"[perc] VGG16 预训练权重加载失败({e})，退回随机权重（不推荐）", flush=True)
            feats = vgg16(weights=None).features
        self.vgg = feats.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.layers = sorted(int(v) for v in layers)
        self.resize = int(resize or 0)
        w = layer_weights or [1.0] * len(self.layers)
        self.weights = [float(x) for x in w]
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def train(self, mode=True):          # 永远保持 eval（BN 用 running stats）
        super().train(False)
        return self

    def _pre(self, x01):
        x = x01.float().clamp(0.0, 1.0)
        if self.resize and x.shape[-1] != self.resize:
            x = F.interpolate(x, size=(self.resize, self.resize),
                              mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, pred01, tgt01):
        xp = self._pre(pred01)
        with torch.no_grad():                 # 目标分支不建图
            xt = self._pre(tgt01)
        loss = pred01.new_zeros(())
        used, i = 0, 0
        for li, layer in enumerate(self.vgg):
            xp, xt = layer(xp), layer(xt)
            if li in self.layers:
                w = self.weights[self.layers.index(li)]
                loss = loss + w * F.l1_loss(xp, xt)
                used += w
        return loss / max(used, 1e-8)
