"""PatchGAN 判别器 + 训练器（重建任务的对抗微调用, VQGAN 风格 hinge 损失）。

设计要点（针对本项目约束）:
- **Pix2pix 70×70 感受野的 PatchGAN**：只看局部 patch 真伪, 不要求全局一致性 ⇒ 对
  重建任务（结构由 L1 保证）只负责补高频纹理, 与判别器容量无关。
- **hinge 损失**（非饱和, 训练稳定）:  D: relu(1-D(real)) + relu(1+D(fake));
  G: -D(fake)。
- **D 不包 DDP**: 本仓库 MUSA 环境里 DDP 双 backward（G 一次 / D 一次）语义不干净,
  故 D 保持裸模块, 由外部手动 allreduce 梯度 + 初始 broadcast, 与 ManualDDP 同思路。
- **float32 跑 D**（autocast 关闭）: InstanceNorm/谱归一在 bf16 下不稳, D 只有 ~3M 参数,
  开 float32 的代价可忽略。
- 监控量: `d_real`/`d_fake`（判别器输出均值）、`acc_real`/`acc_fake`（>0 的比例）⇒
  用来判断 D 是否过强（此时应下调 --gan_weight 或放宽 D 的 lr）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLayerDiscriminator(nn.Module):
    """Pix2pix 的 NLayerDiscriminator（默认 n_layers=3 ⇒ 70×70 感受野）。"""

    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_layers = int(n_layers)
        kw, padw = 4, 1

        def norm(c):
            # 判别器用小 batch 时 InstanceNorm 比 BatchNorm 稳（不跨 rank 统计）
            return nn.InstanceNorm2d(c, affine=True, track_running_stats=False)

        seq = [nn.Conv2d(in_ch, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, self.n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 2, padw, bias=False),
                    norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** self.n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 1, padw, bias=False),
                norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * mult, 1, kw, 1, padw)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class PatchGANTrainer:
    """包装判别器 + 优化器；提供 G 侧的对抗项和 D 侧的更新。"""

    def __init__(self, ndf: int = 64, n_layers: int = 3, lr: float = 1e-4,
                 device=None, ddp: bool = False, world: int = 1, rank: int = 0,
                 in_ch: int = 3):
        self.device = device
        self.ddp = bool(ddp)
        self.world = int(world)
        self.rank = int(rank)
        self.D = NLayerDiscriminator(in_ch=in_ch, ndf=ndf,
                                     n_layers=n_layers).to(device)
        # 各 rank 起始参数必须一致（D 不走 DDP ⇒ 手动 broadcast）
        if self.ddp:
            import torch.distributed as dist
            for p in self.D.parameters():
                dist.broadcast(p.data, src=0)
        self.opt = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.9))
        self.last = {}          # 最近一次 D 更新的监控量

    # ── G 侧：对抗项（要求 fake 带梯度）────────────────────────────────
    def g_adv_loss(self, fake):
        # D 用 float32；.float() 可微 ⇒ 梯度能回到 bf16 的重建张量
        with torch.autocast("cuda", enabled=False):
            out = self.D(fake.float())
            return -out.mean()

    # ── D 侧：hinge 更新 ─────────────────────────────────────────────
    def update_d(self, real, fake):
        import torch.distributed as dist
        self.D.train()
        with torch.autocast("cuda", enabled=False):
            pr = self.D(real.detach().float())
            pf = self.D(fake.detach().float())
            loss = 0.5 * (F.relu(1.0 - pr).mean() + F.relu(1.0 + pf).mean())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.ddp:
            for p in self.D.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.mul_(1.0 / self.world)
        self.opt.step()
        with torch.no_grad():
            self.last = {
                "d_loss": float(loss),
                "d_real": float(pr.mean()),
                "d_fake": float(pf.mean()),
                "acc_real": float((pr > 0).float().mean()),
                "acc_fake": float((pf > 0).float().mean()),
            }
        return self.last

    # ── 自适应 λ 用：末端输出层的权重（叶子参数，VQGAN get_last_layer 同思路）──
    @staticmethod
    def last_layer_weight(raw_model):
        """返回解码器末端 pixel_head 的最后一个 Conv/Linear 权重。"""
        dec = getattr(raw_model, "decoder", None)
        head = getattr(dec, "pixel_head", None) if dec is not None else None
        mod = head if head is not None else dec
        if mod is None:
            return None
        for m in reversed(list(mod.modules())):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                return m.weight
        return None

    def grad_ratio(self, loss_rec, loss_adv, weight):
        """‖∂L_rec/∂w‖ / ‖∂L_adv/∂w‖（w = 末端输出层权重，VQGAN 的 λ 自适应）。

        对**叶子参数**求导而不是对非叶子输出张量 —— MUSA 的 autograd 对非叶子
        input 会报 "differentiated Tensor not used in graph"，对叶子权重正常。
        """
        if weight is None:
            return None
        g_rec = torch.autograd.grad(loss_rec, weight, retain_graph=True,
                                    allow_unused=True)[0]
        g_adv = torch.autograd.grad(loss_adv, weight, retain_graph=True,
                                    allow_unused=True)[0]
        if g_rec is None or g_adv is None:
            return None
        return float(g_rec.norm()) / (float(g_adv.norm()) + 1e-8)

    @staticmethod
    def lam_at(step: int, start_step: int, ramp_steps: int, target: float) -> float:
        """对抗权重 ramp：start 之后线性升到 target，避免开局把已收敛的模型打乱。"""
        if target <= 0.0 or step < start_step:
            return 0.0
        return target * min(1.0, (step - start_step + 1) / max(1, ramp_steps))
