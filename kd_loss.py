"""SD-VAE 蒸馏 teacher **v2**: 一个 teacher、两个出口（像素 / latent）—— 2026-11-23.

背景与病因（`doc/2026-11-23/NOTES_kd_vs_alit_code.md` §2，三个数字）
--------------------------------------------------------------------
同 5000 张 IN-100 val、同 FID 口径：

    臂                                L1/255   PSNR    rFID
    SD-VAE 重建（= 本 teacher）         9.31    25.93    3.44
    GT 模糊 σ=2.0                      10.31    24.56   27.20

⇒ **教师自己的 L1（9.31）比 σ=2 模糊（10.31）还好，rFID 却差 8 倍（3.44 vs 27.20）**。
在 9–13/255 这个工作区间里 L1 与 FID 基本上**解耦**：把 pred 完全搬到位到教师输出，
L1 只从 ~13/255 降到 9.31/255（−21%），rFID 却从 86.50 变 3.44（−96%）。所以"像素 L1
到教师解码图"这条监督（kd1）指的梯度方向与 FID 关心的东西几乎无关，且与基础 L1(x)
互相拉扯（§2.3）。

**为什么改成监督 `z` 而不是像素**（I1）：
  · LDM 的 latent 是**去相关、按方差缩放**过的，纹理/高频结构在那里才有分量，
    像素 L1 里它们只是被平均掉的噪声；
  · ALIT 官方 stage-1 的全部收益就建立在这上面：`adaptive_vae.py:282-284` 的
    `forward_loss = MSE(gt_code, decoded_code)`，`gt_code` 来自 frozen teacher encoder，
    且 stage-1 **没有任何像素损失**（`adaptive_vae.py:247-250`）；像素由 teacher 自己的
    decoder 渲染（`adaptive_vae.py:120-122`），对应我们的 `render=sdvae`（I2）；
  · 尺度约定（**2026-11-24 更正**）：默认 `scale=1.0`，即直接回归**未缩放** latent。
    第一版 kd2 用了 SD 的 `0.18215`，理由是"归一化后 MSE 不被大方差通道主导"——这是
    **概念错误**：两边同乘一个正常数只把损失/梯度整体乘 `s²`（0.18215² = 0.0332，
    即 λ 被暗削弱 30 倍），**完全不会改变各通道的相对权重**。ALIT 的 `mul_(0.2325)`
    在其 stage-1 里无害，只因为那一阶段**只有 MSE 这一条损失**、λ 由它自己定。
    ⇒ 我们的 KD 强度统一由 `--kd_weight` 表达；`--kd_latent_scale 0.18215` 仅保留为
      "复现 kd2-#1 那版 ckpt"的开关（ckpt 的 args.json 会记录实际值）。

本文件的两个出口
----------------
1. `encode_codes(x01)` ⇒ 教师 latent `(B,4,H/8,W/8)`（**未缩放**，`latent_dist.mode()`）;
   训练侧的 `z_t` 就是它（只跑 encoder ⇒ 比 v1 省一半 teacher 开销，见 NOTES §3-I6）；
2. `forward(x01)` ⇒ 教师**重建图** `(B,3,H,W) ∈[0,1]`（v1 原样保留，`--kd_space pixel`
   的旧路径仍用它，逐位不变）。

新增 `decode_latent(z)`（= `decode(z)` 的 [0,1] 包装）：`--render sdvae` 的评测路径
（`train_linattn.py` 的 val / `tools/fast_rfid.py`）用它把 `z_hat`（**未缩放**）渲染成图。
⚠️ 绝不能写成 `z_hat / scale`：`MSE(z_hat·s, z_t·s)` 的最优解是 `z_hat = z_t`，
   再除 `s` 等于把 latent 放大 `1/s≈5.5` 倍 ⇒ 出图趋近均值图（kd2-#1 的 Bug 1）。

⚠️ 目标分支只喂 GT（`_tgt01`），绝不喂学生输出 —— 否则 teacher 会被学生的伪影带偏，
   且"教师看 GT"这条边界在论文里要写清楚（这里看的是训练图 x，不是额外的标签信息）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 默认 **1.0**（= 直接回归未缩放 latent, λ 语义干净）。0.18215 是 SD 的
# `config.scaling_factor`（`z_norm = z * 0.18215`），它属于**扩散调度**的约定，
# 不是"给 MSE 调通道权重"的工具；见文件头的 2026-11-24 更正。
# 保留 knob 只为复现 kd2-#1（`--kd_latent_scale 0.18215 --kd_weight 1.0`）。
LATENT_SCALE = 1.0


def latent_mse(pred_z: torch.Tensor, gt_z: torch.Tensor,
               scale: float = LATENT_SCALE) -> torch.Tensor:
    """latent MSE：`scale=1.0` 时就是 `mean((pred − gt)²)`。

    ⚠️ 恒定 `scale` 是**全局**缩放：`MSE(pred·s, gt·s) = s² · MSE(pred, gt)`，
       损失值和梯度都乘 `s²`，**不改各通道比例**。所以它唯一的效果是"暗改 λ"。
       强度请用 `--kd_weight` 表达；`scale≠1` 只用于复现历史 ckpt。
    · 纯函数、无状态、可微（pred_z 一侧建图，gt_z 一侧调用方自己 no_grad）;
    · `scale` 显式传入而不是读模块常量 ⇒ 调用方（train/fast_rfid）与 ckpt 里的
      `--kd_latent_scale` 一致，不会出现"训练一套尺度、评测另一套"。
    """
    return F.mse_loss(pred_z * scale, gt_z * scale)


class SDVAETeacher(nn.Module):
    """SD-VAE teacher（冻结）：`encode_codes` 出 latent、`forward` 出重建图。

    - 永远 `eval()`（BN/attention 不随训练模式变）；
    - 全参数冻结（`requires_grad=False`）；
    - 两个出口都在 `torch.no_grad()` 内：teacher 只是"标签生成器"，不参与反传。

    `latent_scale` 只是**记录**在实例上（日志/描述用），真正的缩放由调用方传给
    `latent_mse`（见上面的理由：尺度必须与 ckpt 记录一致）。
    """

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse",
                 local_files_only: bool = True,
                 latent_scale: float = LATENT_SCALE):
        super().__init__()
        from diffusers import AutoencoderKL     # 惰性 import: 本模块 import 期不碰 diffusers
        self.vae = AutoencoderKL.from_pretrained(
            model_id, local_files_only=bool(local_files_only))
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        cfg = self.vae.config
        self.model_id = model_id
        self.down = 2 ** (len(cfg.block_out_channels) - 1)
        self.latent_channels = int(cfg.latent_channels)
        self.rate = self.latent_channels  # per latent pixel; ×(H/down)² 由调用方算
        self.latent_scale = float(latent_scale)

    def train(self, mode: bool = True):        # 永远 eval
        super().train(False)
        self.vae.eval()
        return self

    # ── 出口 1: 只要 latent（I1/I6: 只跑 encoder ⇒ 省一半 teacher 开销）──
    @torch.no_grad()
    def encode_codes(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: (B,3,H,W) ∈ [0,1]（fp32）⇒ 未缩放 latent (B,4,H/8,W/8)。

        · 内部先 `×2−1` 到 SD-VAE 的输入域，与 `forward` 同一口径；
        · 用 `latent_dist.mode()`（确定性目标，**不用** `.sample()`：蒸馏要确定目标，
          采样会给 MSE 加噪声底，见 NOTES §3-I7）;
        · 返回的是**未缩放**的 z（SD 的 `z*0.18215` 由 `latent_mse`/调用方做）。
        """
        with torch.autocast("cuda", enabled=False):
            x = x01.float().clamp(0.0, 1.0) * 2.0 - 1.0
            return self.vae.encode(x).latent_dist.mode()

    # ── 出口 1': z → 图（I2 评测/验证的投放路径: 冻结 teacher decoder）──
    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B,4,H/8,W/8) **未缩放** latent ⇒ (B,3,H,W) ∈ [0,1]。

        ⚠️ 传的是 `z_hat` **原值**，不要 `/ scale`（见文件头 Bug 1 说明）。
        与 `adaptive_vae.py:120-122`
        `reconstruct_images = base_tokenizer.vae.decode(decoded_code)` 同构。
        """
        with torch.autocast("cuda", enabled=False):
            y = self.vae.decode(z.float()).sample
        return ((y.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    # ── 出口 2: 重建图（v1 原样 ⇒ `--kd_space pixel` 逐位不变）──
    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: (B,3,H,W) ∈ [0,1] ⇒ teacher 重建 (B,3,H,W) ∈ [0,1]。

        ⚠️ 函数体是 v1 的**原样**（逐位不变）。不要为了去重改写成
        `decode_latent(encode_codes(x01))` —— 那样无法保证与 kd1 的旧日志逐位一致。
        """
        with torch.autocast("cuda", enabled=False):
            x = x01.float().clamp(0.0, 1.0) * 2.0 - 1.0
            z = self.vae.encode(x).latent_dist.mode()
            y = self.vae.decode(z).sample
        return ((y.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    def rate_values(self, H: int) -> int:
        """latent 值数/图 @H×H（= `latent_channels × (H/down)²`，如 256² ⇒ 4096）。"""
        lat = int(H) // self.down
        return self.latent_channels * lat * lat

    def describe(self, H: int) -> str:
        n = sum(p.numel() for p in self.vae.parameters())
        lat = H // self.down
        return (f"teacher={self.model_id} params={n/1e6:.1f}M "
                f"latent={self.latent_channels}x{lat}x{lat} "
                f"rate={self.latent_channels*lat*lat} values/img (H={H})")
