"""对照实验: carry `query_base + Y` 要不要加 LayerNorm。

背景（model_v2.py:608）
    Y = self.stack(self.stack_in(Y), A_in[:, lo:hi+1])
    Y = self.stack_out(Y)
    Y = self.query_base + Y            ← 这一步没有 LN
但 self.stack 的 DecoderLayer 是 norm_first=True，层内第一个操作是
`norm1(tgt)`。所以「加不加 LN」真正改变的**不是注意力看到什么**，而是
**残差基的尺度**。本脚本把这件事拆成四个可测量的量。

E1 等价性   LN(LN(x)) ≈ LN(x)（γ=1, β=0）⇒ 注意力输入几乎不变
E2 尺度     逐步残差基范数 + 分支占比 branch/base（新信息能改动多少）
E3 梯度     BPTT 下各步 dL/d(查询输入) 的放大倍数
E4 影响力   扰动本步 z，看还能把输出改动多少（相对尺度）

全部是前向或单次 backward，不需要训练、不需要数据。
注意: 试图用合成任务做「训练 A/B」不可靠（本脚本试过三版都停在同一点，
差异只反映输出尺度而非信息利用），要定论请在真实数据上跑短 A/B。

用法::

    python exp_carry_ln.py            # 全部四段
    python exp_carry_ln.py --quick    # 只跑 E1-E3
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v2 import OutputQueryDecoder


# ═══════════════════════════════════════════════════════ 三个实验臂
class CarryArm(OutputQueryDecoder):
    """在真实 OutputQueryDecoder 上只改 carry 一行。

    arm="none" → Y = query_base + Y        （model_v2.py 现状）
    arm="ln"   → Y = LN(query_base + Y)
    arm="off"  → Y = query_base            （对照组：彻底断开 carry）

    forward 是父类逐行复制 + 一行替换；arm="none" 时断言与父类逐位相同，
    保证这份副本没有偏离真实实现。
    """

    def __init__(self, *args, arm: str = "none", **kwargs):
        super().__init__(*args, **kwargs)
        assert arm in ("none", "ln", "off"), arm
        self.arm = arm
        self.carry_norm = nn.LayerNorm(self.query_base.shape[-1])

    def _next_query(self, y: torch.Tensor) -> torch.Tensor:
        if self.arm == "off":
            return self.query_base.unsqueeze(0).expand_as(y)
        s = self.query_base + y
        return s if self.arm == "none" else self.carry_norm(s)

    def forward(self, z_cls: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        B = z_cls.shape[0]
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed
        A_in = self.stack_in(A)
        Y = self.query_base.unsqueeze(0).expand(B, -1, -1)
        Y_total, tgt_log = [], []
        windows = self.windows_for_probe()
        for i, t in enumerate(self.steps):
            lo, hi = windows[i]
            tgt_log.append(Y)                    # 该步的查询输入（诊断用）
            Y = self.stack(self.stack_in(Y), A_in[:, lo:hi + 1])
            Y = self.stack_out(Y)
            Y_total.append(Y)
            Y = self._next_query(Y)
        out = torch.stack(Y_total, dim=1)
        self.last_windows = windows
        self.last_Y = out
        self.last_tgt = tgt_log
        return out

    def windows_for_probe(self):
        from model_v2 import step_windows
        return step_windows(self.num_specials, self.steps,
                            self.step_plan, self.block)


def build(arm: str, dim: int, n: int, k: int, depth: int, heads: int,
          ratio: float = 2.0, seed: int = 0) -> CarryArm:
    torch.manual_seed(seed)
    return CarryArm(dim=dim, num_patches=n, num_specials=k, heads=heads,
                    depth=depth, mlp_ratio=ratio, arm=arm)


# ═══════════════════════════════════════════════════════ E1 等价性
@torch.no_grad()
def e1_equivalence(dim: int, n: int, k: int, depth: int, heads: int,
                  ratio: float = 2.0) -> None:
    """LN 加在 carry 上，到底改变了什么？

    pre-norm 层内第一件事是 norm1(tgt)，所以 LN(carry) 之后又被归一一次
    ⇒ 注意力**输入**几乎不变（LN∘LN ≈ LN）。真正变的是**残差基**的尺度：
    层输出 = 残差基 + 注意力分支 + FFN 分支，基多大直接决定"直通"占多少。
    """
    m = build("none", dim, n, k, depth, heads, ratio)
    z_cls, z_s = torch.randn(2, 1, dim), torch.randn(2, k, dim)
    out = m(z_cls, z_s)                                    # (B,T,N,D)
    layer = m.stack.layers[0]
    qb = m.query_base.unsqueeze(0).expand(2, -1, -1)

    print("── E1 加 LN 改变了什么（残差基尺度，不是注意力输入）──")
    print("   step   base:无LN   base:有LN   注意力输入max|Δ|   cos")
    for t in (0, out.shape[1] - 1):
        base = qb if t == 0 else qb + out[:, t - 1]
        base_ln = m.carry_norm(base)
        d = (layer.norm1(base) - layer.norm1(base_ln)).abs().max().item()
        c = F.cosine_similarity(layer.norm1(base).flatten(1),
                                layer.norm1(base_ln).flatten(1)).mean().item()
        print(f"   {t:4d}   {base.norm(dim=-1).mean():8.3f}"
              f"   {base_ln.norm(dim=-1).mean():8.3f}   {d:16.3e}   {c:.8f}")
    print(f"   （sqrt(dim) = {math.sqrt(dim):.2f}；LN 把基**钉**在这个尺度上，"
          f"注意力看到的内容不变）\n")


# ═══════════════════════════════════════════════════════ E2 尺度漂移
@torch.no_grad()
def e2_scale(dim: int, n: int, k: int, depth: int, heads: int,
             ratio: float = 2.0) -> None:
    """逐步看残差基怎么漂，以及「新信息分支」相对基的占比。

    branch = 层输出 - 残差基（= 自注意力 + 交叉注意力 + FFN 三条残差之和）。
    ||branch|| / ||base|| 就是"这一步能改多少"。它趋于 0 时，无论读进什么
    新内容，输出都≈把上一步原样搬过去 —— 这正是 carry 无 LN 的病。
    """
    z_cls, z_s = torch.randn(2, 1, dim), torch.randn(2, k, dim)
    res = {}
    for arm in ("none", "ln"):
        m = build(arm, dim, n, k, depth, heads, ratio)
        m.eval()
        Y = m(z_cls, z_s)
        qb = m.query_base.unsqueeze(0).expand(Y.shape[0], -1, -1)
        bases = [qb]
        for t in range(1, Y.shape[1]):
            bases.append(m._next_query(Y[:, t - 1]))
        res[arm] = (Y, torch.stack(bases, 1))

    T = res["none"][0].shape[1]
    print(f"── E2 逐步尺度漂移（|T| = {T}, d = {dim}）──")
    print("         none: ||base||   ||out||   branch/base     "
          "ln: ||base||   branch/base   cos(out_t,out_t-1)")
    prev = None
    for t in range(T):
        row = []
        for arm in ("none", "ln"):
            Y, base = res[arm]
            bn = base[:, t].norm(dim=-1).mean().item()
            on = Y[:, t].norm(dim=-1).mean().item()
            br = (Y[:, t] - base[:, t]).norm(dim=-1).mean().item() / bn
            row += [bn, br]
        c = (float("nan") if prev is None else
             F.cosine_similarity(res["none"][0][:, t].flatten(1),
                                 prev.flatten(1)).mean().item())
        prev = res["none"][0][:, t]
        print(f"   t={t:2d}  {row[0]:11.3f} {res['none'][0][:,t].norm(dim=-1).mean():9.3f}"
              f"   {row[1]:10.3f}     {row[2]:11.3f}   {row[3]:10.3f}   {c:10.4f}")
    print()


# ═══════════════════════════════════════════════════════ E3 梯度
def e3_grad(dim: int, n: int, k: int, depth: int, heads: int,
            ratio: float = 2.0) -> None:
    """只监督最后一步，看 BPTT 把梯度沿 carry 链送回各步时放大/衰减多少。

    量的是 dL/d(tgt_t)：tgt_t = carry(上一步输出)，是链上的每个接口。
    相邻两步的比值 = 单步 Jacobian 的放大倍数。
      · 无 LN：tgt 里有恒等项 ⇒ 梯度可原样穿过（放大器 ≈ 1）
      · 有 LN：恒等项被 LN 破坏 ⇒ 梯度必须过 LN 的 Jacobian（会收缩）
    """
    print("── E3 BPTT 梯度（只监督最后一步；量 dL/d(每步查询输入)）──")
    z_cls, z_s = torch.randn(2, 1, dim), torch.randn(2, k, dim)
    for arm in ("none", "ln", "off"):
        m = build(arm, dim, n, k, depth, heads, ratio)
        m.train()
        out = m(z_cls, z_s)
        loss = out[:, -1].pow(2).mean()
        grads = torch.autograd.grad(loss, m.last_tgt + [m.query_base])
        per = [g.norm().item() for g in grads[:-1]]
        gq = grads[-1].norm().item()
        # 反向链上的单步放大倍数：g_t / g_{t+1}
        amps = [per[t] / max(per[t + 1], 1e-30) for t in range(len(per) - 1)]
        print(f"  arm={arm:4s} ||dL/dQ||={gq:8.3e}  ||Q||={m.query_base.norm():.3f}"
              f"  相对 {gq/m.query_base.norm():8.2e}")
        print(f"    各步 dL/d(tgt): 首 {per[0]:.2e}  中 {per[len(per)//2]:.2e}"
              f"  末 {per[-1]:.2e}   （首/末 = {per[0]/max(per[-1],1e-30):.2e}）")
        print(f"    单步放大 g_t/g_(t+1): 均值 {sum(amps)/len(amps):.3f}"
              f"  最大 {max(amps):.3f}  最小 {min(amps):.3f}")
    print()


# ═══════════════════════════════════════════════════════ E4 训练 A/B
# ═══════════════════════════════════════════════════════ E4 新信息影响力
@torch.no_grad()
def e4_sensitivity(dim: int, n: int, k: int, depth: int, heads: int,
                   ratio: float = 2.0) -> None:
    """第 t 步读到的新信息，还能把**第 t 步的输出**改动多少？

    扰动该步 window 内的 z（δ 相对幅度固定），量 ||Δout_t||：
      rel = ||Δout_t|| / ||out_t||   相对自身尺度还能动多少
      abs = ||Δout_t|| / ||Δz||      单位输入的绝对灵敏度
    no-LN 时 base 随 t 线性增长 ⇒ rel 应随 t 衰减（新信息被已积累的状态淹没）；
    LN 把 base 钉住 ⇒ rel 基本恒定。这是"后几步学不动"的直接证据。
    """
    print("── E4 新信息影响力（扰动本步 window 内的 z）──")
    z_cls, z_s = torch.randn(2, 1, dim), torch.randn(2, k, dim)
    res = {}
    for arm in ("none", "ln"):
        m = build(arm, dim, n, k, depth, heads, ratio)
        m.eval()
        base = m(z_cls, z_s)
        deltas = []
        win = m.windows_for_probe()
        for i in range(len(m.steps)):
            lo, hi = win[i]
            dz = torch.randn_like(z_s) * 0.1
            z2, c2 = z_s.clone(), z_cls.clone()
            if lo == 0:                       # 首窗口 = [z_cls; z_s[:hi]]，只动 z_cls
                c2 = c2 + torch.randn_like(c2) * 0.1
                z2[:, :hi] = z2[:, :hi] + dz[:, :hi]
            else:                             # A 坐标的 lo 对应 z_s 的 lo-1
                z2[:, lo - 1:hi] = z2[:, lo - 1:hi] + dz[:, lo - 1:hi]
            o2 = m(c2, z2)
            rel, ab = [], []
            for t in range(base.shape[1]):
                d = (o2[:, t] - base[:, t]).norm(dim=-1).mean().item()
                rel.append(d / base[:, t].norm(dim=-1).mean().item())
                ab.append(d / dz.norm(dim=-1).mean().item())
            deltas.append((rel, ab))
        res[arm] = deltas

    T = len(res["none"])
    print("   扰动步 → 看哪一步的输出变化；下表是 rel（相对灵敏度）")
    print("   观察步:", "  ".join(f"t={t}" for t in range(T)))
    for i in range(T):
        for arm in ("none", "ln"):
            rel = res[arm][i][0]
            print(f"  {arm:4s} 扰 z_{m.steps[i]:2d}   "
                  + "  ".join(f"{rel[t]*100:6.2f}%" if t >= i else "   --  "
                              for t in range(T)))
        print()
    print("  末步相对灵敏度（扰 z_0）: none="
          f"{res['none'][0][0][-1]*100:.3f}%  ln={res['ln'][0][0][-1]*100:.3f}%")
    print("  末步相对灵敏度（扰 z_末）: none="
          f"{res['none'][-1][0][-1]*100:.3f}%  ln={res['ln'][-1][0][-1]*100:.3f}%")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过 E4")
    a = ap.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    big = dict(dim=256, n=256, k=256, depth=2, heads=8, ratio=2.0)
    small = dict(dim=64, n=16, k=16, depth=2, heads=4, ratio=2.0)

    print("=" * 74)
    print("对照: carry = query_base + Y 是否加 LayerNorm")
    print("=" * 74)
    e1_equivalence(**big)
    e2_scale(**big)
    e3_grad(**big)
    if not a.quick:
        e4_sensitivity(**big)


if __name__ == "__main__":
    main()
