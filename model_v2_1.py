"""
model_v2_1 — 方案A: 解码器离线循环精炼状态 + 尾部单监督（基于 model_v2.py）
=================================================================

动机（2026-11-13, 对话中定名"方案A"）:
    model_v2 的解码器是 looped transformer（同一组 stack 权重在 |T| 个采样
    步上反复施加, 每步读**新**一块 z_s, carry 走 BPTT, 每步直接出图）——
    它的循环是"**信息调度器**": 每遍输入都在变, |T| 一动, 信息计划（步值/
    块边界/K 推导）跟着动。
    与之正交的另一路是 LLM Sleep 式的**离线循环** [13]: 循环是"**算力磨盘**"
    ——同一份输入磨 N 遍, 把多步计算累积进一个**持久状态**, 预测阶段只做
    单次前向（在线延迟不变, 额外算力全部花在无人等待的 consolidation 段）。
    方案A = 把两种循环的三个结构差异一次改齐（与 model_v2 逐点对照）:
      1. 循环携带: Y（查询/激活, 每步即弃） → **状态 S**（K+1 行槽位, 跨遍
         存活, 尾部被读出）;
      2. 每遍输入: 递增读块（信息驱动） → **固定全量 A**（算力驱动: N 遍读
         同一份上下文, 纯粹"多磨几遍", 与读窗口步值无关）;
      3. 监督挂点: 每步深监督 mean_t L1 → **尾部单监督**（循环内不产生损失,
         中间态可以自由当"半成品", 只对最终读出负责）。
    新超参 num_loops（N）因此是**纯算力旋钮**: v2 做不了"固定信息计划、单独
    扫 N"的实验（|T| 动 = 信息计划动）, v2_1 可以——这是复现"循环深度当
    算力旋钮, N↑ → 能力↑" [14] 的前置条件。前向次数 = N+1（N 遍精炼 + 1 次
    读出）; 对齐 v2 的 |T| 次前向取 N = |T|−1（成本口径见架构段末）。

架构（register 式编码段与 model_v2 **一字不差**; 解码段按方案A重写）:
    specials 直接拼进 DINO 输入 [cls; specials(K); patches(N)], 24 层全双向
    → z_cls/z_s; 解码器:
        A    = [z_cls; z_s] + pos_embed              (B, K+1, D)
        A_in = stack_in(A)                           ← 每遍循环的**固定** memory
        S   ← state_init                             ← 可学习初始状态 (1, K+1, D)
        for _ in range(num_loops):                   ← **离线循环** N 遍, 输入相同
            S = stack_out(stack(stack_in(S), A_in))  ← tgt=状态, memory=A_in
        Y = stack_out(stack(stack_in(query_base), S)) ← **在线读出**: 单次前向
        F_hat = PixelHead(Y)                          ← 唯一输出 = 唯一监督
    · S 是"权重状态"的松散类比（[13] 的 fast weight 是 SSM 矩阵; 这里是对
      K+1 个状态槽位的持续改写, 槽位 k 不必然对应 register k, 由注意力学）。
    · 跨遍 carry 不 detach ⇒ BPTT 穿过全部 N 遍（梯度教 stack "怎么把上下文
      磨成尾部读出好用的 S"）; 与 model_v2 的 BPTT 口径一致, 仍是唯一路径。
    · state_init 每样本现算现弃、不跨步持久 ⇒ 无 BatchNorm 式"θ 更新后
      状态陈旧"问题（S = f_θ(x) 的派生量, 每次 forward 用当前 θ 重算）。
    · 成本口径: 前向次数 N+1（v2: |T|）; 每次 cross-attn 的 memory 宽度恒为
      K+1（v2: 递增块宽, 总量 ≈ K）; self-attn 每次同为 N 行。等次数下 v2_1
      的 cross-attn 略贵、结构更简单——本版优先把口径改对, 效率后议。

与 model_v2 的接口差异（旧旋钮一律 TypeError, 不静默忽略）:
    · 删除: decoder_steps / skip_steps / max_steps / step_plan / block 及
      全部读窗口机械（square_block_starts / fixed_block_starts /
      step_windows / derive_num_specials / select_steps / build_block_mask /
      build_causal_query_mask / build_prefix_mask 本文件不再消费; 复现 v2
      行为请直接用 model_v2.py）。
      动机: 解码器每遍读**全量** A ⇒ K 个位置每遍都被读, 不存在"花瓶
      register"问题, derive_num_specials 失去存在理由; K<N 压缩实验从
      "切步集"简化为"直接给 num_specials"（缺省 K=N 全量, 向后兼容）。
    · 新增: num_loops（N, 默认 4）; state_init（(1,K+1,D), 随机小值初始化,
      可学习）。
    · 损失: L = L1(PixelHead(Y), target)（单点; recon ≡ loss）; 返回 dict
      = {"loss","recon","F_hat","state","target_pix"}——**没有 Y_pix**
      （逐步预测与 |T| 维已删, 拿 v2 的 Y_pix 消费代码过来会 KeyError,
      这是故意的: 宁可响亮报错也不静默降级）。
    · ⚠️ checkpoint 不兼容: state_init 是新键, 且循环语义已变——v2 的
      final_model.pt 在本代码上 strict load 必报缺键; 即便手动补键, 语义
      也不同（v2 的 |T| 步读块 vs v2_1 的 N 遍磨状态）。不要跨版混用权重。
    · 本版仍只做**图像** (B,3,H,W), 无 model_v3 的帧维/视频路径。

踩坑记录（重要）:
    **对称性破缺**: 状态槽位若用全零初始化, TransformerDecoderLayer 的
    self-attn/cross-attn（norm_first）对**相同 query 必输出相同行**——K+1
    个槽位从第一遍起恒等, 之后永远恒等, 状态容量塌成 1 行, N 遍循环等于
    磨同一个槽。故 state_init 用 **randn×0.02**（与 query_base 同款, 逐槽
    不同且可学习）; 自检 §2c 钉死"槽位可区分"。

用法
----
    model = SRPhase1V21(dinov2=dinov2, num_patches=256, dim=768,
                        num_specials=32, num_loops=4)
    out = model(pixel_values)        # {"loss","recon","F_hat","state","target_pix"}
    loss = out["loss"]; loss.backward()
    model.eval()                     # 推理同路径

    自检: python model_v2_1.py

参考资料（承接 model_v2 的 [1]–[12], 新增）:
    [13] Lee et al. (2026), Language Models Need Sleep: Offline Recurrence for
        Improved Online Inference——"离线循环"口径出处（睡眠=N 遍同一输入
        精炼持久状态, 预测阶段单次前向; 梯度流经被反复精炼的状态而非特征）
        https://arxiv.org/abs/2605.26099
    [14] Geiping et al. (2025), Looped Transformers are Better at Learning
        Learning Algorithms——循环深度当算力旋钮（N↑ → 能力↑）的依据;
        方案A 解耦 N 与读窗口, 即为在像素重建任务上复现该曲线的前置条件
        https://arxiv.org/abs/2504.16752
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence


# ═══ SpecialTokenBank — 特殊 token 池（输入相同, 仅位置编码不同）═══

class SpecialTokenBank(nn.Module):
    """每个 special/register 位置一个特殊 token：共享可学习向量 + 逐位置可学习位置编码。

    token (1,1,D) 对所有位置/所有图像相同; pos (1,K,D) 提供位置区分。
    特殊 token 位置的表示 z_s 是 decoder 输入; 图像 patch 编码不进 decoder
    （只作为编码器输入参与聚合）。K = register/specials 数（= num_specials,
    **直接给**: 缺省 K=N 全量, K<N 即压缩; v2_1 已删读窗口, 不再从步集推导
    （derive_num_specials 随之删除, 见文件头; 历史参数名 num_patches 已改名
    为 num_tokens——K 与 N 解耦后两者不再恒等）。
    """

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_tokens, -1) + self.pos      # (B,K,D)

# ═══ PixelHead — 特征 → 像素 patch 解码头 ═══

class PixelHead(nn.Module):
    """每 patch 特征 (B,N,D) → 像素 patch (B,N,14*14*3)。

    2 层 MLP（dim→hidden→588）: 回应"解码器参数量不够欠拟合"的担忧
    （v2 是单层 Linear, 0.6M; 这里是 ~2.1M@dim=1024,hidden=2048, 且带
    非线性）。输出不加激活: 像素按 DINO_MEAN/STD 归一化(范围≈[-2,2]),
    L1 直接监督归一化空间, 评估时再反归一化。
    """

    def __init__(self, dim: int, patch_px: int = 14 * 14 * 3, hidden: int = 2048):
        super().__init__()
        self.patch_px = patch_px
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, patch_px))

    def forward(self, feat: Tensor) -> Tensor:
        """feat: (..., N, D) → (..., N, patch_px)"""
        return self.net(feat)


# ═══ patches_to_image — 像素 patch → 图像（target_pix 布局的唯一反变换）═══

def patches_to_image(pixels: Tensor, H: int, W: int,
                     mean: Optional[Sequence[float]] = None,
                     std: Optional[Sequence[float]] = None,
                     clamp_255: bool = True) -> Tensor:
    """(B,N,588) 像素 patch → (B,H,W,3) 图像（给定 mean/std 时反归一化到 0-255）。

    **布局的权威定义 = SRPhase1V2.decode 里 target_pix 的构造**: patch 向量 =
    (C,14,14) **通道优先**（展开序 = c*14*14 + dy*14 + dx）, patch 顺序
    row-major（先 y 后 x）, N=(H/14)*(W/14)。本函数是它的**严格逆**,
    model_v2.py 自检 §1b 有往返断言（patches_to_image(target_pix,
    clamp_255=False) == 输入图 (B,H,W,C) 逐位相等）。

    踩坑记录（2026-09-15 修）: 消费方历史上把反变换写成
    `reshape(B, gh, gw, 14, 14, 3).permute(0,1,3,2,4,5)`——把 588 当成
    "(14,14,3) 通道在后"读, 与 decode 的通道优先布局不符 ⇒ 每张"重建图"的
    每个 14×14 patch 内部像素被打乱（实测往返 vs 原图 max|diff|=255 /
    mean≈85; 修正后为 0）。训练不受影响（loss 两侧同用同一个 patch 平铺
    向量, 无需求逆）, 但可视化与 0-255 口径的 L1 都不可信。消费方
    （infer_v2_test.py / visualize_recon_pixel.py）一律调用本函数。

    clamp_255: 只在反归一化后生效（默认 True）; 纯布局往返（输入已在
    归一化空间、不传 mean/std）传 False, 否则会被 clamp 破坏数值。
    """
    B, N, px = pixels.shape
    assert px == 3 * 14 * 14, f"patch 向量长度应为 588, got {px}"
    assert (H // 14) * (W // 14) == N, \
        f"N={N} 与 {H}x{W} 的 patch 数 {(H // 14) * (W // 14)} 不符"
    img = pixels.reshape(B, H // 14, W // 14, 3, 14, 14) \
                .permute(0, 1, 4, 2, 5, 3) \
                .reshape(B, H, W, 3)
    if mean is not None and std is not None:
        m = torch.as_tensor(mean, dtype=img.dtype,
                            device=img.device).view(1, 1, 1, 3)
        s = torch.as_tensor(std, dtype=img.dtype,
                            device=img.device).view(1, 1, 1, 3)
        img = img * s + m
    if clamp_255:
        img = img.clamp(0.0, 255.0)
    return img

# ═══ OutputQueryDecoder — 状态循环 + 尾部单读出解码器（方案A）═══

class OutputQueryDecoder(nn.Module):
    """方案A 解码器: [z_cls; z_s] 当上下文 A (B,S,D), S=K+1; N 遍循环把同一份
    A 磨进状态 (B,S,D), 尾部单次前向从 query_base 读出 (B,N,D)。

    机制（forward 三段, 与 model_v2 的 |T| 步循环逐点对照）:
        A = [z_cls; z_s] + pos_embed; A_in = stack_in(A)   ← 固定 memory（投影一次）
        # ── 离线循环: N 遍精炼状态, 每遍输入完全相同（同一份 A_in）──
        S = state_init.expand(B)                            ← 可学习初始状态 (1,S,D)
        for _ in range(num_loops):
            S = stack_out(stack(stack_in(S), A_in))         ← tgt=状态, memory=固定
        # ── 在线读出: 单次前向（无逐步输出 / 无深监督）──
        Y = stack_out(stack(stack_in(query_base), stack_in(S)))
    · **输入固定** ⇒ 循环是"算力调度"（同一材料磨 N 遍, 每遍基于上一遍的 S
      做更深变换）, 不是 v2 的"信息调度"（每步读新块）; 与读窗口/步值无关。
    · 循环体与 v2 单步同构: stack_in → stack → stack_out, 状态活在 **dim
      空间**（stack_dim 加宽时每遍前后各过一次投影, 与 v2 每步同款）。
    · stack 调用次数 = num_loops + 1, 参数量与 num_loops 无关（权重共享）
      ——自检 §2/§7 分别用 forward hook 计数与参数量断言钉死。
    · carry 走 **BPTT**（无 detach, 与 v2 同款唯一路径）: 跨遍梯度存在, 即
      存在"让 S 成为尾部读出好用状态"的梯度通路; state_init 收到尾部 loss
      的回流（自检 §3b）——梯度穿过全部 N 遍回到初始状态。
    · 输出 (B,N,D)（v2 是 (B,|T|,N,D)）; 行 k ↔ patch k 不变。
    · 最近一次 forward 的尾部状态存 self.last_state（decode 取出当 "state"）。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 num_specials: Optional[int] = None,
                 num_loops: int = 4, mlp_ratio: float = 4.0,
                 heads: int = 8, depth: int = 2,
                 stack_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.num_patches = num_patches                # 查询基行数 = N（输出 N 个 patch 预测）
        self.num_specials = num_patches if num_specials is None else int(num_specials)
        assert 1 <= self.num_specials <= num_patches, \
            f"num_specials 越界: K={self.num_specials} (须 1 ≤ K ≤ N={num_patches})"
        self.num_loops = int(num_loops)
        assert self.num_loops >= 1, f"num_loops 须 ≥1, got {num_loops}"
        S = self.num_specials + 1                                 # z_cls + K 个 z_s
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)  # 行 k↔patch k
        # 可学习初始状态。**随机小值、不可全零**（见文件头"对称性破缺"踩坑:
        # 全零 ⇒ 所有槽位恒等 ⇒ 状态容量塌成 1 行）; 形状 (1,S,D), 与 pos_embed
        # 同款初始化, 逐槽位可区分、可学习。
        self.state_init = nn.Parameter(torch.randn(1, S, dim) * 0.02)
        # 标准解码器堆叠（与 model_v2 完全同款: depth 层 TransformerDecoderLayer,
        # 每层 = 自注意力 + 交叉注意力(memory) + FFN + 残差, norm_first）。
        # stack_dim 语义也与 v2 一致: None/0 ⇒ stack_dim == dim, 不加投影
        # （零额外参数, 与原实现逐位一致）; 显式放大 ⇒ stack 前后各一层 Linear。
        self.stack_dim = int(dim if not stack_dim else stack_dim)
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.stack_dim, nhead=heads,
                dim_feedforward=int(self.stack_dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True),
            num_layers=depth,
        )
        self.stack_in = (nn.Identity() if self.stack_dim == dim
                         else nn.Linear(dim, self.stack_dim))
        self.stack_out = (nn.Identity() if self.stack_dim == dim
                          else nn.Linear(self.stack_dim, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, S, dim) * 0.02)
        self.last_state: Optional[Tensor] = None      # 尾部精炼后的 S（诊断/decode 用）

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        assert z_cls.dim() == 3 and z_s.dim() == 3, \
            f"z_cls/z_s 须为 (B,·,D), got {tuple(z_cls.shape)}/{tuple(z_s.shape)}"
        assert z_cls.shape[1] == 1, f"z_cls 应为 1 列, got {z_cls.shape[1]}"
        assert z_s.shape[1] == self.num_specials, \
            f"z_s 列数 {z_s.shape[1]} != num_specials(K)={self.num_specials}"
        assert z_cls.shape[0] == z_s.shape[0], \
            f"batch 不一致: z_cls {z_cls.shape[0]} vs z_s {z_s.shape[0]}"
        B = z_cls.shape[0]
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed       # (B,S,D)
        A_in = self.stack_in(A)                                   # memory 投影一次, N 遍共用
        # ── 离线循环: N 遍精炼状态（每遍输入 = 同一份 A_in, 不 detach ⇒ BPTT）──
        S = self.state_init.expand(B, -1, -1)                     # (B,S,D) 状态槽
        for _ in range(self.num_loops):
            S = self.stack_out(self.stack(self.stack_in(S), A_in))
        # ── 在线读出: 单次前向（query_base 直接查状态 S）──
        Y = self.query_base.unsqueeze(0).expand(B, -1, -1)        # (B,N,D)
        Y = self.stack_out(self.stack(self.stack_in(Y), self.stack_in(S)))
        self.last_state = S
        return Y


# ═══ SRPhase1V21 — 主模型（register 式编码继承 v2; 解码 = 方案A）═══

class SRPhase1V21(nn.Module):
    """register 式主模型（方案A）。编码段（SpecialTokenBank + DINO 24 层全双向
    → z_cls/z_s）与 SRPhase1V2 **一字不差**; 差异全在解码段与损失口径:

    · 解码 = OutputQueryDecoder（N 遍状态循环 + 尾部单读出, 见其 docstring）;
      num_loops（N）是纯算力旋钮, 与读窗口/步值彻底无关（v2 的
      decoder_steps/skip/max/step_plan/block 已删除, 传即 TypeError）。
    · K（num_specials）直接给: 缺省 K=N 全量（向后兼容 v2 默认行为）;
      K<N 即真压缩——解码器每遍读全量 A, K 个位置不会沦为"花瓶"。
    · 损失 = **尾部单监督**: L = L1(PixelHead(Y), target_pix); recon ≡ loss
      （v2 是 mean_t 深监督, F_hat=最后一步——本版循环内不产生损失,
      中间态自由当"半成品", 只对最终读出负责）。
    · 输入仍只认 (B,3,H,W)（图像, 无 v3 帧维）; 训练/推理同一 forward。

    dict: {"loss", "recon"(≡loss), "F_hat"(B,N,588 唯一输出),
           "state"(尾部精炼状态 B,K+1,D, 诊断), "target_pix"(B,N,588)}
    ——没有 "Y_pix"（逐步预测已删, 见文件头接口差异）。
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_px: int = 14 * 14 * 3,
        decoder_depth: int = 2,
        num_specials: Optional[int] = None,
        num_loops: int = 4,
        stack_dim: int = 0,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_px = patch_px

        K = num_patches if num_specials is None else int(num_specials)
        assert 1 <= K <= num_patches, \
            f"num_specials 越界: K={K} (须 1 ≤ K ≤ N={num_patches})"
        self.num_specials = K
        self.num_loops = int(num_loops)

        self.special_bank = SpecialTokenBank(num_tokens=K, dim=dim)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          num_specials=K,
                                          num_loops=self.num_loops,
                                          mlp_ratio=mlp_ratio, heads=heads,
                                          depth=decoder_depth,
                                          stack_dim=stack_dim,
                                          dropout=decoder_dropout)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px)

    # ── encode: 输入 → 解码器输入 z_cls, z_s（与 SRPhase1V2 逐字相同）──
    def encode(self, pixel_values: Tensor):
        """输入 → 解码器输入 (z_cls, z_s), 训练/推理同一路径。

        register 式: specials 作为额外 token 拼进 DINO 输入序列, 由 DINO
        24 层直接算出 z_s（register token 式, Darcet et al.）——深层网络做
        内容路由, 修 F1（special 无内容输入）与 F2（z_s 冗余全局摘要）。
        """
        return self._encode_register(pixel_values)

    def _encode_register(self, pixel_values: Tensor):
        """specials 直接进 DINO 输入 [cls; specials(K); patches(N)]
        (1+K+N token, K=self.num_specials)。

        HF Dinov2Model 无 token 级注意力 mask API, 故 DINO 内全双向注意力
        （重建任务无时序因果需求, register 惯例亦然）; specials 用
        SpecialTokenBank（共享 token + 逐位置可学习 pos, 不带 DINO PE）。
        24 层跑完 → layernorm → 取 [cls; 前 K 个 specials] 当 z_cls/z_s
        （全部来自**末层**）。
        """
        x = pixel_values                                # (B,3,H,W)
        emb = self.dinov2.embeddings(x)                 # (B,1+N,D) [cls; patches] + PE
        specials = self.special_bank(x.shape[0], x.device)   # (B,K,D) token+pos
        seq = torch.cat([emb[:, :1], specials, emb[:, 1:]], dim=1)   # (B,1+K+N,D)
        for layer in self.dinov2.encoder.layer:         # DINO 24 层（全双向）
            out = layer(seq)
            seq = out[0] if isinstance(out, (tuple, list)) else out
        seq = self.dinov2.layernorm(seq)                # (B,1+K+N,D)
        return seq[:, :1], seq[:, 1:1 + self.num_specials]

    # ── decode: 解码尾（方案A 解码器 → PixelHead → 单点像素 L1）──
    def decode(self, z_cls: Tensor, z_s: Tensor, pixel_values: Tensor) -> dict:
        """解码器 + 像素头 + **单点**像素 L1（方案A: 尾部监督, 无 mean_t）。

        Y = decoder(z_cls, z_s) 是**唯一**读出（(B,N,D), 无 |T| 维）:
            loss = L1(PixelHead(Y), target_pix)   ← 唯一损失
            recon ≡ loss                          ← 监控别名（v2 里 recon=最后
                                                  一步 L1; 本版两者同值）
            state = decoder.last_state            ← 尾部精炼后的 S（诊断）

        dict: {"loss", "recon", "F_hat", "state", "target_pix"}——没有 "Y_pix"。
        """
        x = pixel_values
        B, C, H, W = x.shape
        N = self.num_patches
        Y = self.decoder(z_cls, z_s)                        # (B,N,D) 单次读出
        # 像素目标: (B,3,H,W) 归一化像素 → (B,N,588) patch
        # 布局 = decode 里 target_pix 的构造: (C,14,14) 通道优先, patch row-major
        target_pix = x.reshape(B, C, H // 14, 14, W // 14, 14) \
                      .permute(0, 2, 4, 1, 3, 5) \
                      .reshape(B, N, C * 14 * 14)       # (B,N,588) 归一化像素
        F_pix = self.pixel_head(Y)                      # (B,N,588) 唯一输出
        loss = F.l1_loss(F_pix, target_pix)             # 尾部单监督
        return {"loss": loss, "recon": loss, "F_hat": F_pix,
                "state": self.decoder.last_state, "target_pix": target_pix}

    # ── forward（训练/推理同一路径, 无分支）──
    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,3,H,W) 归一化像素 → dict{loss, recon, F_hat, state,
        target_pix}

        监督**原始像素**而非 DINO patch 特征（特征目标退化: 工地图特征
        空间近常数, 学质心即低 L1 是假收敛）。H,W 须为 14 的倍数。
        """
        x = pixel_values                                # (B,3,H,W)
        B, C, H, W = x.shape
        N = self.num_patches
        assert W % 14 == 0 and H % 14 == 0, "输入须为 14 的倍数"
        assert (W // 14) * (H // 14) == N, \
            f"输入 {W}x{H} 产生 {(W//14)*(H//14)} patches, 但模型 num_patches={N}"
        z_cls, z_s = self.encode(x)
        return self.decode(z_cls, z_s, x)


# ═══ 自检（python model_v2_1.py）═══
#   1. 形状/单点损失/state（register 式全路径）+ patch↔图像往返
#   2. 解码器结构: stack 调用 = num_loops+1; N 遍共用同一 memory（输入固定）;
#      旧旋钮 TypeError; state_init 对称性破缺（随机初始化）
#   3. 梯度流向（整模型可训, 含 state_init）+ BPTT 穿过全部 N 遍
#   4. eval 同路径
#   5. stack 加宽（stack_dim ≠ dim）
#   6. K<N 压缩（K 直接给, 无花瓶）
#   7. num_loops 解耦: 参数量与 N 恒等 / 前向次数 = N+1 / 输出随 N 变
if __name__ == "__main__":
    torch.manual_seed(0)

    # ── 假的 DINOv2: 结构对齐 HF Dinov2Model（register 式直接走层）──
    class _FakeDinoLayer(nn.Module):
        """HF Dinov2EncoderLayer 属性布局 + 真实单头自注意力 forward。"""
        def __init__(self, dim: int, mlp_ratio: float = 4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            att = nn.Module()
            att.attention = nn.Module()
            att.attention.query = nn.Linear(dim, dim)
            att.attention.key = nn.Linear(dim, dim)
            att.attention.value = nn.Linear(dim, dim)
            att.output = nn.Module()
            att.output.dense = nn.Linear(dim, dim)
            self.attention = att
            mlp = nn.Module()
            mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
            mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
            self.mlp = mlp

        def forward(self, hidden_states: Tensor, head_mask=None,
                    output_attentions=False):
            n = self.norm1(hidden_states)
            q = self.attention.attention.query(n)
            k = self.attention.attention.key(n)
            v = self.attention.attention.value(n)
            scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h = hidden_states + self.attention.output.dense(attn @ v)
            h = h + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(h))))
            return (h, None)

    class _FakeEmbeddings(nn.Module):
        """对齐 HF Dinov2Embeddings 的最小结构: conv patch + cls + PE(+插值)。"""
        def __init__(self, dim: int, num_patches: int):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.patch_embeddings = nn.Conv2d(3, dim, kernel_size=14, stride=14)
            self.position_embeddings = nn.Parameter(
                torch.randn(1, num_patches + 1, dim) * 0.02)
            self.dropout = nn.Identity()

        def forward(self, pixel_values: Tensor, bool_masked_pos=None,
                    interpolate_pos_encoding=None) -> Tensor:
            B = pixel_values.shape[0]
            patch = self.patch_embeddings(pixel_values)      # (B,D,h,w)
            patch = patch.flatten(2).transpose(1, 2)         # (B,N,D)
            cls = self.cls_token.expand(B, -1, -1)
            emb = torch.cat([cls, patch], dim=1)             # (B,1+N,D)
            pe = self.position_embeddings
            if emb.shape[1] != pe.shape[1]:                  # 对齐 HF 自动插值
                pe = F.interpolate(pe.transpose(1, 2).unsqueeze(0),
                                   size=(emb.shape[1],), mode="linear",
                                   align_corners=False).squeeze(0).transpose(1, 2)
            return self.dropout(emb + pe)

    class FakeDino(nn.Module):
        """最小假 DINO: 只含 register 式用到的 embeddings / encoder.layer / layernorm。"""
        def __init__(self, dim: int = 64, n_layers: int = 4, num_patches: int = 16):
            super().__init__()
            self.embeddings = _FakeEmbeddings(dim, num_patches)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(
                [_FakeDinoLayer(dim) for _ in range(n_layers)])
            self.layernorm = nn.LayerNorm(dim)

    N, D = 16, 64
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V21(dino, num_patches=N, dim=D, num_loops=3)   # K 缺省 = N
    x = torch.randn(2, 3, 56, 56)
    B, C, H, W = x.shape

    # ── 1. 形状: F_hat = 唯一输出; 单点损失; state = 尾部精炼状态 ──
    PATCH_PX = 14 * 14 * 3
    out = model(x)
    assert set(out.keys()) == {"loss", "recon", "F_hat", "state", "target_pix"}, \
        f"输出键应为方案A五件套, got {sorted(out)}"
    assert out["F_hat"].shape == (2, N, PATCH_PX), out["F_hat"].shape
    assert out["loss"].shape == () and out["recon"].shape == ()
    # 像素目标提取与模型内部一致（布局口径 = decode 的 target_pix, 通道优先）
    target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
              .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)
    assert torch.isclose(out["recon"],
                         F.l1_loss(out["F_hat"], target)), "recon 应为像素 L1"
    assert torch.isclose(out["loss"], out["recon"]), \
        "v2_1 是尾部单监督: loss ≡ recon（v2 是 mean_t 深监督, 两者不同）"
    assert "Y_pix" not in out, \
        "逐步预测键已删（无 |T| 维）——别把 model_v2 的 Y_pix 消费代码搬过来"
    K = model.num_specials
    assert out["state"].shape == (2, K + 1, D), out["state"].shape
    # ── 1b. 像素 patch ↔ 图像 往返（布局回归测试, 与 v2 逐字相同）──
    rt = patches_to_image(out["target_pix"], H, W, clamp_255=False)
    assert rt.shape == (2, H, W, 3), rt.shape
    assert torch.allclose(rt, x.permute(0, 2, 3, 1)), \
        "patches_to_image 必须与 decode 里 target_pix 布局互为严格逆（通道优先）"
    img255 = patches_to_image(out["F_hat"], H, W,
                              mean=[0.485 * 255, 0.456 * 255, 0.425 * 255],
                              std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    lo, hi = float(img255.detach().min()), float(img255.detach().max())
    assert lo >= 0.0 and hi <= 255.0, f"反归一化范围越界 [{lo}, {hi}]"
    print(f"[ok] shapes: F_hat{tuple(out['F_hat'].shape)} (像素 {PATCH_PX}D), "
          f"单点 loss={out['loss'].item():.4f} ≡ recon, state{tuple(out['state'].shape)}; "
          f"patch↔图像往返逐位相等, 0-255 反归一化范围 [{lo:.1f}, {hi:.1f}]")

    # ── 2. 解码器结构: stack 调用 = num_loops+1; N 遍共用同一 memory ──
    assert model.decoder.num_loops == 3, model.decoder.num_loops
    assert not hasattr(model.decoder, "attn_mask") and \
        not hasattr(model.decoder, "tgt_mask"), "方案A无掩码（切片读窗口已删）"
    model.eval()
    mems = []
    _h = model.decoder.stack.register_forward_hook(lambda m, a, o: mems.append(a[1]))
    with torch.no_grad():
        model(x)
    _h.remove()
    assert len(mems) == model.decoder.num_loops + 1, \
        f"stack 调用应 = num_loops+1 = {model.decoder.num_loops + 1}, got {len(mems)}"
    assert all(m is mems[0] for m in mems[:model.decoder.num_loops]), \
        "离线循环的 N 遍必须读同一份 memory（A_in, 输入固定=算力调度）"
    assert mems[-1] is not mems[0], \
        "尾部读出的 memory 应是状态 S（不是 A_in）——循环与读出角色分离"
    model.train()
    print(f"[ok] 解码器: stack 调用 {len(mems)} = num_loops(3)+1; "
          f"循环 3 遍共用同一 memory, 读出用状态 S; 无 attn/tgt 掩码")

    # ── 2b. 旧旋钮一律 TypeError（读窗口/深监督开关随方案A删除, 不静默忽略）──
    for kw in ({"decoder_steps": [1, 4, 9]}, {"skip_steps": 1}, {"max_steps": 3},
               {"step_plan": "fixed"}, {"block": 4},
               {"loss_mode": "cumulative"}, {"loss_decouple": True}):
        try:
            SRPhase1V21(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D, **kw)
            raise AssertionError(f"{kw} 应被拒绝（方案A已删读窗口/损失口径开关）")
        except TypeError as e:
            assert "unexpected keyword" in str(e), f"{kw}: {e}"
    print("[ok] 旧旋钮 (decoder_steps/skip/max/step_plan/block/loss_mode/"
          "loss_decouple) 传即 TypeError（不静默忽略）")

    # ── 2c. state_init: 可学习 + 随机初始化（对称性破缺, 防全零塌缩）──
    si = model.decoder.state_init
    assert si.shape == (1, K + 1, D) and si.requires_grad, \
        f"state_init 形状/可学习性不对: {tuple(si.shape)}"
    assert si.detach().abs().sum().item() > 0 and \
        (si.detach()[0, 0] - si.detach()[0, 1]).abs().sum().item() > 0, \
        "状态槽初始不得全零/恒等: 全零 ⇒ 槽位恒等 ⇒ 状态容量塌成 1 行"
    print(f"[ok] state_init {tuple(si.shape)} 可学习且逐槽随机可区分"
          f"（对称性破缺, 防状态塌缩）")

    # ── 3. 梯度流向（整模型可训, 含 state_init / query_base / PixelHead）──
    out["loss"].backward()
    for name, p in [("dino.embeddings.patch_embeddings.weight",
                     dino.embeddings.patch_embeddings.weight),
                    ("dino.embeddings.position_embeddings",
                     dino.embeddings.position_embeddings),
                    ("dino.encoder.layer.0.mlp.fc1.weight",
                     dino.encoder.layer[0].mlp.fc1.weight),
                    ("decoder.stack.layers[0].self_attn.in_proj_weight",
                     model.decoder.stack.layers[0].self_attn.in_proj_weight),
                    ("decoder.stack.layers[-1].multihead_attn.in_proj_weight",
                     model.decoder.stack.layers[-1].multihead_attn.in_proj_weight),
                    ("decoder.stack.layers[-1].linear1.weight",
                     model.decoder.stack.layers[-1].linear1.weight),
                    ("decoder.query_base", model.decoder.query_base),
                    ("decoder.state_init", model.decoder.state_init),
                    ("special_bank.pos", model.special_bank.pos),
                    ("pixel_head.net[0].weight", model.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print("[ok] 梯度: DINO 嵌入/层 + OutputQueryDecoder(stack/query_base/"
          "state_init)/SpecialTokenBank/PixelHead 全部可训")

    # ── 3b. BPTT 穿过全部 N 遍: 尾部单点 loss 必须回流到 state_init ──
    _g = model.decoder.state_init.grad
    assert _g is not None and _g.abs().sum() > 0, \
        "state_init 收不到梯度 ⇒ 循环被 detach/截断了（方案A要求 BPTT 全通）"
    print("[ok] BPTT: 尾部单点 loss 穿过 3 遍状态循环回到 state_init（无 detach）")

    # ── 4. 推理: 同一 forward（eval + no_grad）──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["F_hat"].shape == (2, N, PATCH_PX)
    model.train()
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    # ── 5. stack 加宽（stack_dim ≠ dim, 前后 Linear 投影; 状态活在 dim 空间）──
    m_wide = SRPhase1V21(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                         num_loops=2, heads=4, decoder_depth=4, stack_dim=2 * D,
                         decoder_dropout=0.05)
    dec = m_wide.decoder
    assert dec.stack_dim == 2 * D, dec.stack_dim
    assert isinstance(dec.stack_in, nn.Linear) and dec.stack_in.in_features == D \
        and dec.stack_in.out_features == 2 * D, "输入侧应为 dim→stack_dim 投影"
    assert isinstance(dec.stack_out, nn.Linear) and dec.stack_out.in_features == 2 * D \
        and dec.stack_out.out_features == D, "输出侧应为 stack_dim→dim 投影"
    assert len(dec.stack.layers) == 4, "加宽路径 depth 应生效"
    assert dec.stack.layers[0].linear1.in_features == 2 * D
    assert dec.stack.layers[0].linear1.out_features == 4 * 2 * D
    assert dec.stack.layers[0].dropout1.p == 0.05
    # 循环体 = stack_in → stack → stack_out ⇒ 状态每遍回到 dim 空间（与 v2 单步同构）
    assert dec.state_init.shape == (1, N + 1, D), \
        "state_init 应在 dim 空间（循环体含 stack_out, 与 v2 单步同构）"
    out_w = m_wide(x)
    assert out_w["F_hat"].shape == (2, N, PATCH_PX), out_w["F_hat"].shape
    assert out_w["loss"].shape == ()
    # 加宽权重可 save/load 往返（训练收尾 final_model.pt 依赖此路径）
    m_wide.load_state_dict(m_wide.state_dict(), strict=True)
    print(f"[ok] self.stack 加宽: d_model {D}→{dec.stack_dim}, heads=4, "
          f"depth=4, dropout=0.05, 投影 {tuple(dec.stack_in.weight.shape)}→"
          f"{tuple(dec.stack_out.weight.shape)}, state_init 留在 dim 空间")

    # ── 6. K<N 压缩（v2_1: K 直接给, 不再从步集推导; 每遍读全量 ⇒ 无花瓶）──
    m_k = SRPhase1V21(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                      num_specials=8, num_loops=2)
    assert m_k.num_specials == 8, m_k.num_specials
    z_s_k = m_k.encode(x)[1]
    assert z_s_k.shape == (2, 8, D), z_s_k.shape         # 编码器只生成 K=8 个 register
    out_k = m_k(x)
    assert out_k["F_hat"].shape == (2, N, PATCH_PX)
    assert out_k["state"].shape == (2, 9, D), out_k["state"].shape
    out_k["loss"].backward()
    for name, p in [("m_k.special_bank.pos", m_k.special_bank.pos),
                    ("m_k.decoder.query_base", m_k.decoder.query_base),
                    ("m_k.decoder.state_init", m_k.decoder.state_init),
                    ("m_k.pixel_head.net[0].weight", m_k.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    # K 越界 → 清晰报错
    try:
        SRPhase1V21(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                    num_specials=N + 1)
        raise AssertionError("K>N 应报错")
    except AssertionError as err:
        assert "越界" in str(err), err
    print(f"[ok] K<N 压缩: N={N} 给 K=8 ⇒ z_s(2,8,D)/state(2,9,D) 形状对, "
          f"键侧/查询侧/状态/像素头梯度全通; K>N 报'越界'")

    # ── 7. num_loops 解耦: 参数量与 N 恒等 / 前向次数 = N+1 / 输出随 N 变 ──
    m1 = SRPhase1V21(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D, num_loops=1)
    m8 = SRPhase1V21(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D, num_loops=8)
    n1 = sum(p.numel() for p in m1.parameters())
    n8 = sum(p.numel() for p in m8.parameters())
    assert n1 == n8, f"参数量必须与 num_loops 无关（权重共享）: {n1} vs {n8}"
    _cnt = []
    _h7 = m8.decoder.stack.register_forward_hook(lambda m, a, o: _cnt.append(1))
    with torch.no_grad():
        f8 = m8(x)["F_hat"]
    _h7.remove()
    assert len(_cnt) == 9, f"num_loops=8 ⇒ stack 调用应 9 次, got {len(_cnt)}"
    with torch.no_grad():
        f1 = m1(x)["F_hat"]
    assert not torch.allclose(f1, f8), "不同 N 应产生不同计算（N 是纯算力旋钮）"
    print(f"[ok] num_loops 解耦: N=1 与 N=8 参数量恒等 ({n1}); "
          f"前向次数 = N+1 (9 次); 输出随 N 变化")

    print("\nALL CHECKS PASSED")
