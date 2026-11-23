"""SD-VAE 输出蒸馏 teacher —— 2026-11-23.

背景（`PLAN_distill_20261123.md`）
--------------------------------
主解码器在 3072 values/图 上的 rFID 是 86.50，而同值数下 ALIT 是 8.25 ⇒ 瓶颈不是信息量，
是**解码器先验**。LDM 的 VAE（`sd-vae-ft-mse`）正是用 L1 + LPIPS 感知 + PatchGAN 对抗
训出来的解码器（我们买不起的配方），rate = 4×32×32 = **4096 values** @256²（≈我们的 1.33×），
纯卷积、83.7M 参数、可离线跑。

P0-c 实测（IN-100 val 5000 张，与 fast_rfid.py 完全同口径；见 tools/p0c_teacher_fid.py 与
`doc/2026-11-23/artifacts/P0c_SUMMARY.md`）：
    SD-VAE 重建 (4096 values/图):  PSNR 25.93  rFID 3.44
    我们的 L1    (3072 values/图):  PSNR 22.67  rFID 86.50
    GT 模糊 σ=1.0: PSNR 28.65 rFID 9.15 ／ σ=2.0: PSNR 24.56 rFID 27.20
⇒ 我们的重建在 PSNR 与 rFID 两个轴上都比 σ=2 的模糊更差；而同 rate 量级的 SD-VAE 能到 3.44
  ⇒ 瓶颈不是"信息/速率"而是**解码器先验**，把 teacher 输出当监督目标在原理上可达。

用法（在 train_linattn.py 里）:
    loss += kd_lam * L1(F_hat_01, teacher(gt_01))
    teacher 冻结、永远 eval、fp32、no_grad（目标分支不建图）。

⚠️ 目标分支**只喂 GT**（`_tgt01`），绝不喂学生输出 —— 否则 teacher 会被学生的伪影带偏，
   且"教师看 GT"这条边界在论文里要写清楚（这里看的是训练图 x，不是额外的标签信息）。
"""
import torch
import torch.nn as nn


class SDVAETeacher(nn.Module):
    """SD-VAE 重建作为蒸馏目标：t = D(E(x))，返回 [0,1] 图。

    - 永远 `eval()`（BN/attention 不随训练模式变）；
    - 全参数冻结；
    - `forward` 在 `torch.no_grad()` 内：teacher 只是"标签生成器"，不参与反传。
    """

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse",
                 local_files_only: bool = True):
        super().__init__()
        from diffusers import AutoencoderKL
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

    def train(self, mode: bool = True):        # 永远 eval
        super().train(False)
        self.vae.eval()
        return self

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: (B,3,H,W) ∈ [0,1] ⇒ teacher 重建 (B,3,H,W) ∈ [0,1]。"""
        with torch.autocast("cuda", enabled=False):
            x = x01.float().clamp(0.0, 1.0) * 2.0 - 1.0
            z = self.vae.encode(x).latent_dist.mode()
            y = self.vae.decode(z).sample
        return ((y.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    def describe(self, H: int) -> str:
        n = sum(p.numel() for p in self.vae.parameters())
        lat = H // self.down
        return (f"teacher={self.model_id} params={n/1e6:.1f}M "
                f"latent={self.latent_channels}x{lat}x{lat} "
                f"rate={self.latent_channels*lat*lat} values/img (H={H})")
