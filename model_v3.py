"""
AssocVTC Phase 1 v2/v3 — 训练脚手架（register 式, 无 ReEncoder; 视频模态）
=================================================================

项目目标（权威版见 doc/2026-08-28/GOAL_compression_for_nlp.md）:
    用像素重建当代理任务, 训练编码器（DINOv2 + register specials）的
    "token 压缩 + 联想"能力; 训练完成后冻结编码器接入 Qwen 做 NLP 解码
    （验收标准 = Phase 2 文字生成质量）。像素重建 = 信息保持的直接探针
    （能还原像素 ⇒ z 携带整图信息 ⇒ 语义信息必然在）。纹理级高频按文献
    [3] 属"死信息"不追; 压缩率 K 是练联想的真正杠杆——K 与 N 解耦: N
    只管输入 patch 数与解码查询行数, K（register/specials 数）由解码器
    最终采样步集自动推导（K≤N; K=N 即历史全量行为, K<N 即真压缩）。

沿革: v2 曾含 ReEncoder 路径与"可学习前缀预算"机制, 均已按 YAGNI 整块
    移除（git 历史可找回）; register 式（2026-08-28 起）为唯一路径。
    解码器侧: **并行路径**（|T| 步一次算完 + 跨步 tgt_mask + query_mask_mode
    开关）自 2026-09-15 起被**循环架构**取代并整块删除; 需要复现旧结果时从
    git 取回（`git log -- model_v2.py`, 最后一个含并行路径的提交 = 043ef2a）。
    循环 = 同一组解码器权重在 |T| 个采样步上反复施加、每步把上一步输出作为
    下一步查询（looped transformer 口径, 见参考资料 [10]/[11]）; 不动点口径
    （把循环解到收敛 + 隐式微分, 见 [12]）本文**未采用**——固定 |T| 步 + BPTT。

    视频模态（2026-11-12, model_v3）: 每帧仍由同一编码器压成 K 个 z_s token
    （240M 配置 K=32）; MiddleEncoder 沿帧做循环交叉注意力（上一步输出当
    下一步 query, KV = [cls] + **历史帧** z_s, 不含当前帧）, 输出的 Q_f 当
    该帧解码器的查询基; 图像 = 单帧视频（F=1, KV=[cls] 冷启动）。编码器/
    解码器结构不变, 时序信息只经 MiddleEncoder 引入。详见架构段。

架构（register 式, 无选择、无预算）:
    specials 作为额外 token 直接拼进 DINOv2 输入序列
    [cls; specials(K); patches(N)] (1+K+N token, K 的推导见下), 由 DINO
    24 层直接算出 z_s（register token 式, Darcet et al.）——深层网络做
    "内容路由", 修 F1（special 无内容输入）与 F2（z_s 冗余全局摘要）。
    DINO 内全双向注意力（HF 无 token 级 mask API）; z_s[k] 依赖全部 patch
    （含 j>k）, "前缀稳定性"不成立——渐进语义由解码器的**读窗口**提供:
    第 i 个采样步 t 的 cross-attention 只吃自己那个平方块这一段 z_s
    （memory = A[:, lo:hi+1], 块起止 k=⌊√t⌋, hi=min((k+1)²−1, K),
    非首步 lo=max(k²,1); 首步 lo=0 ⇒ 额外含位置 0 = z_cls 及其前的全部元素,
    口径与 build_block_mask 逐位一致）——切片即窗口, 无需显式掩码。
    无 ReEncoder（省 51.6M 参数）。
    ⚠️ 2026-09-23: **逐层 tap / 特征金字塔（layer_tap）已整块从 main 移除**
    （用户口径: main 只保留主线目标, 实验性分支不进主线）。历史实现与实测见
    `doc/2026-09-22/DESIGN_layer_tap_pyramid.md` /
    `doc/2026-09-22/REPORT_layer_tap_pyramid_224.md`（只作存档, 代码已不在）。
    register 数 K（num_specials）= 解码器实际读取的 z_s 范围: 默认由
    "最终生效采样步集"自动推导 K = min( max_{t∈steps}((⌊√t⌋+1)²−1), N )
    （公式/动机/示例见 derive_num_specials 与 doc/2026-09-02/
    DESIGN_v2_num_specials_from_max_steps.md）——消除"花瓶 register":
    编码器生成、解码器从不读、却经 24 层全双向注意力的前向/反向耦合
    参与训练动力学并干扰读窗口的多余位置。全量默认（无 decoder_steps /
    skip/max 切片）时 steps = square_block_starts(N) ⇒ K=N（向后兼容）;
    例: N=576 切片 [4:9] ⇒ steps=[25,36,49,64,81] ⇒ K=99; steps=[64] ⇒ K=80。
    MiddleEncoder（**视频帧间循环, 2026-11-12**）: query_base 从解码器迁入
                    （(N,D) 参数, 行 k↔patch k）; 对第 f 帧（f=0..F−1）:
                        Q_{-1} = query_base
                        KV_f = [cls]                       ← f=0（冷启动）
                             = [cls; z_s[0]; …; z_s[f−1]]  ← f>0（1+K·f）
                        Q_f = MiddleEncoder(Q_{f−1}, KV_f)
                    KV 的"所有历史帧"**不含当前帧**; [cls] 是可学习初始化
                    向量（mem_init）, 不是编码器的 z_cls。每帧 z_s 有 K 个
                    token（240M 配置 K=32 ⇒ 历史 32×n 个 token）。结构 =
                    TransformerDecoderLayer 堆叠（query 自注意力 + 交叉
                    注意力(KV_f) + FFN, norm_first）, depth 默认 1; 帧间状态
                    不 detach ⇒ 跨帧 BPTT。
    OutputQueryDecoder（**顺序循环, 唯一路径; 不再持 query_base**）:
                    [z_cls; z_s] 为时序序列 A (S=K+1); 在采样时刻 T_sub =
                    steps（默认 square_block_starts, 自动适配任意上界;
                    SRPhase1V2 里切片先于 K 推导, 上界 = N, 见 select_steps）
                    上循环（query = MiddleEncoder 的 Q_f）:
                        Y = query                            ← step1 查询
                        for t in steps:                      ← 每步读自己那块, |T| 步
                            lo, hi = 步 t 的块起止（首步 lo=0, 含 z_cls）
                            Y = stack(stack_in(Y), stack_in(A[:, lo:hi+1]))
                            Y = stack_out(Y)
                            Y = query + Y                    ← 喂给下一步当查询（BPTT, 不 detach）
                    查询行 k ↔ patch k（行数 = N 不变）; 单步内 N 行查询
                    全双向自注意力。图像/视频首帧: F=1, KV=[cls], Q_0 当查询。
    L = mean_{f,t} L1(PixelHead(Y_{f,t}), patch_f)  ← **直接预测**口径: 每帧
        每个采样步的输出 Y_{f,t} 自己过 PixelHead 直接预测该帧整图, 各帧各步
        平权全覆盖损失（无累加/集成: 2026-09-15 前的 cumsum 累加口径已整块
        删除, 需要复现旧结果时从 git 取回。carry 走 **BPTT**（2026-09-23 起
        唯一路径, 无 detach 开关）⇒ 存在"让 Y_{f,t-1} 成为对下一步有用的
        草稿"的梯度通路, 见 OutputQueryDecoder.forward / SRPhase1V2.decode）
        F_hat = Y_pix[:, :, -1]（每帧最后一步的直接预测）;
        recon = mean_f L1(F_hat_f, patch_f) = 监控量（归一化空间）

    输入: 模型内部**只认** (B,F,3,H,W); 图像 (B,3,H,W) 只在 forward 入口
    `unsqueeze(1)` 成 F=1, 之后没有图像分支。输出**始终带 F 维**
    （图像 (B,1,...)）; 同 batch 固定 F, 未做变长 padding。
    ⚠️ checkpoint: model_v3 的 MiddleEncoder 是**新参数**（middle.query_base /
    mem_init / stack.*）, 旧 model_v2 的 decoder.query_base 键不存在 ⇒ 旧
    final_model.pt strict load 会报缺键/多键; 需按新结构重训或显式重映射,
    不要拿旧权重直接在本代码上推理。

踩坑记录（重要）:
    torch 2.x 的 bool 注意力掩码约定是 True=屏蔽（nn.TransformerEncoder
    Layer / MultiheadAttention / TransformerDecoderLayer）, 与直觉相反;
    F.scaled_dot_product_attention 的 bool 掩码实测 True=允许（torch 2.8
    本机验证）。掩码统一用加法浮点(-inf=屏蔽)规避歧义。

用法
----
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768)
    out = model(pixel_values)                 # (B,3,H,W) 或 (B,F,3,H,W)
    loss = out["loss"]; loss.backward()
    model.eval()                              # 推理同路径

    返回: loss/recon 标量; F_hat (B,F,N,588); Y_pix (B,F,|T|,N,588);
    target_pix (B,F,N,588)。图像输入也只是 F=1, 不再压掉 F 维。

    自检: python model_v3.py

参考资料:
    [1] Hawthorne et al. (ICML 2022), Perceiver AR——输出查询 + 因果掩码解码器
        https://mlanthology.org/icml/2022/hawthorne2022icml-generalpurpose/
    [2] Li et al. (ICML 2023), BLIP-2——可学习查询基 / Q-Former 查询桥接
        https://arxiv.org/abs/2301.12597
    [3] Fan et al. (2026)——视觉 token 稀疏性/冗余分析（"死信息"依据）
        https://arxiv.org/abs/2603.00510
    [4] Apedo et al. (2026), SVD-Prune——视觉 token 剪枝（与已移除的预算机制相关）
        https://arxiv.org/abs/2604.11530
    [5] Vahdat & Kautz (NeurIPS 2020), NVAE——隐变量容量取舍对照
        https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html
    [6] Gao & Shou (2025), D-AR——顺序扩散 tokenizer;"token 顺序定义渐进细化"参考
        https://arxiv.org/abs/2505.23660
    [7] Liang et al. (ICLR 2022), EViT——注意力分数排序剪枝 + 不重要 token 融合
        （与 [4] 同域; 预算机制回归时的简单替代）
        https://arxiv.org/abs/2202.07815
        https://github.com/youweiliang/evit
    [8] Darcet et al. (ICLR 2024), Vision Transformers Need Registers——register
        token 式出处（架构段所引的 Darcet et al. 即此文）
        https://arxiv.org/abs/2309.16588
    [9] Jaegle et al. (ICML 2022), Perceiver IO——输出查询定义输出结构的通用化;
        query_base 行 k↔patch k 的直接出处（2026-11-12 起 query_base 由
        MiddleEncoder 持有, 仍作帧间循环的初始查询）
        https://arxiv.org/abs/2107.14795
    [10] Giannou et al. (ICML 2023), Looped Transformers as Programmable Computers
        ——"looped transformer" 口径出处: 同一组权重被反复施加、输入序列承载
        指令与中间状态; 对应本文解码器"同一个 OutputQueryDecoder 在 |T| 个
        采样步上循环、carry 走 BPTT"的读出方式（架构段所引 looped transformer
        即此文）
        https://arxiv.org/abs/2301.13196
        https://proceedings.mlr.press/v202/giannou23a/giannou23a.pdf
    [11] Dehghani et al. (ICLR 2019), Universal Transformers——循环深度
        （recurrent depth）+ 跨深度权重共享的原始出处; 循环读出"反复施加同一
        解码器"的最早形式（该文另有 per-position halting / ACT, 本项目未用,
        不构成依据）
        https://arxiv.org/abs/1807.03819
    [12] Bai et al. (NeurIPS 2019), Deep Equilibrium Models（DEQ）——**不动点
        口径**: 权重共享的无限深网络, 前向 = 解不动点 z* = f_θ(z*, x), 后向
        走隐函数定理 ⇒ 显存与有效深度无关; 与 [10]/[11] 的循环读出同族, 但
        **不是同一个对象**——本文每步只读自己那块 z_s（映射随步变 ⇒ 非自治,
        严格无不动点, 只有慢点/幽灵不动点）, 且跑固定 |T| 步 + BPTT; DEQ 是
        迭代到收敛 + 隐式微分（本项目**未实现**, 仅作对照与"解到收敛"的出路）
        https://arxiv.org/abs/1909.01377
        https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html
        https://github.com/locuslab/deq
"""

import math

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
    由解码器最终采样步集推导, 见 derive_num_specials; 历史参数名
    num_patches 已改名为 num_tokens——K 与 N 解耦后两者不再恒等）。
    """

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_tokens, -1) + self.pos      # (B,K,D)


# ═══ build_prefix_mask — 块状前缀掩码（仅旧诊断脚本 gen_mask_gif.py 引用）═══

def build_prefix_mask(seq_len: int, z_start: int, z_end: int,
                      device: torch.device = None) -> Tensor:
    """块状前缀注意力掩码（torch bool, True=屏蔽）。

    布局: [z 区域 (z_start..z_end-1) | 尾部 (z_end..seq_len-1)]
        · z 行 i: 屏蔽 z 列 (i+1..z_end-1)（因果链）, 可看尾部（全局）
        · 尾部行: 全开放（全局）
    """
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(z_start, z_end):
        m[i, i + 1:z_end] = True          # 屏蔽 z 内部"后面的"
    return m


# ═══ square_block_starts — 分块采样计划（块起点 = 平方数, 自动适配任意 N）═══

def square_block_starts(num_patches: int) -> list:
    """decoder 采样时刻 T_sub = {k² | 1 ≤ k² ≤ N}（块起点 = 平方数）。

    块 k = [k², min((k+1)²-1, N)]; 步数 = ⌊√N⌋, 自动适配任意 N
    （N=256→16 步, N=576→24 步）。例: square_block_starts(12) → [1,4,9]。
    本函数已通用: 入参语义 = "分块计划上界", 不要求与 patch 数 N 恒等
    （见 derive_num_specials / select_steps 的调用方说明）。
    """
    K = math.isqrt(num_patches)
    return [k * k for k in range(1, K + 1)]


def fixed_block_starts(num_patches: int, block: int) -> list:
    """固定宽度分块计划的**步值** = 自然数 1..⌈N/block⌉。

    动机（2026-09-23 用户口径）: square_block_starts 的步值是平方数
    （1,4,9,16,…）, 步值本身在跳变; 本计划让步值按**自然数**增长, 每步读
    **固定 block 个** z_s（末块不足则截断）。块几何见
    step_windows(..., plan="fixed"): 第 k 步读 z_s 的第 k 块
    （A 坐标 [(k−1)·block+1, min(k·block, K)]; 首步 lo=0 额外含 z_cls）。

    例: N=16, block=4 → [1,2,3,4]（4 步）; N=576, block=4 → [1..144]。
    block=1 即"每步读 1 个 token"（|T|=N, 24× 成本, 见报告 §成本）。
    """
    assert block >= 1, f"block 须 ≥1, got {block}"
    return list(range(1, (num_patches + block - 1) // block + 1))


def step_windows(num_specials: int, steps: Sequence[int],
                 plan: str = "square", block: int = 0) -> list:
    """每个采样步在 A 坐标（0=z_cls, 1..K=z_s）下的读窗口 `[(lo, hi), …]`。

    · `plan="square"`（**默认, 历史唯一口径**）: k=⌊√t⌋,
      hi=min((k+1)²−1, K); 非首步 lo=max(k²,1), 首步 lo=0（额外含 z_cls）。
    · `plan="fixed"`: k=t（步值即块号）, hi=min(k·block, K);
      非首步 lo=(k−1)·block+1, 首步 lo=0。

    本函数是 `OutputQueryDecoder.forward` 直接切 `A[:, lo:hi+1]` 的**唯一来源**,
    语义与 `build_block_mask` 逐位一致; 工具侧也用它算"实际累计读入 bpp"。
    """
    assert plan in ("square", "fixed"), f"未知 plan={plan}"
    if plan == "fixed":
        assert block >= 1, f"plan=fixed 须 block ≥1, got {block}"
    out = []
    for i, t in enumerate(steps):
        if plan == "fixed":
            k = int(t)
            hi = min(k * block, num_specials)
            lo = 0 if i == 0 else (k - 1) * block + 1
        else:
            k = math.isqrt(int(t))
            hi = min((k + 1) ** 2 - 1, num_specials)
            lo = 0 if i == 0 else max(k * k, 1)
        assert lo <= hi, \
            f"步 {t} 的读窗口 [{lo},{hi}] 为空 (K={num_specials}, plan={plan})"
        out.append((lo, hi))
    return out


# ═══ derive_num_specials — K 由"最终生效采样步集"自动推导（无花瓶）═══

def derive_num_specials(num_patches: int, steps: Sequence[int]) -> int:
    """K = min( max_{t∈steps} ((⌊√t⌋+1)²−1), num_patches ); steps 为空 → num_patches。

    动机（花瓶 register 问题）: register 式里编码器生成 N 个 register, 但
    当训练只读 steps 覆盖的部分 register 时（如 slice [4:9] → steps=
    [25,36,49,64,81], 块 5..9 最大读到位置 99）, 位置 100..N 的 register
    从不被解码器读——它们不是摆设: 24 层全双向注意力的前向/反向耦合使
    花瓶收到大梯度（读窗口 pos 梯度 |Σ|=6.35 vs 花瓶区 9.81）且逐维扰动
    花瓶输入会让读窗口输出变化 ~68%, 干扰训练动力学。修复 = K 与 N 解耦:
    编码器 register 数 = 解码器实际读的范围, 不存在花瓶。K<N 时同时实现
    真压缩（N 只管输入 patch 数与解码查询行数, K 只管键数/register 数）。

    公式: 步 t 所在的平方块 k=⌊√t⌋ 的块终点 = (k+1)²−1;
        K = min( max_{t∈steps} ((⌊√t⌋+1)²−1), num_patches )
    全量默认（无 decoder_steps、无 skip/max 切片）时 steps =
    square_block_starts(num_patches), K 计算结果 = N（保持现状, 向后兼容）。
    例 1: N=576, slice [4:9] → steps=[25,36,49,64,81], max t=81 →
        (9+1)²−1=99 → K=99（读 1..99, 全部被覆盖, 无花瓶）。
    例 2: 显式 steps=[64] → (⌊√64⌋+1)²−1 = 80 → K=80。
    数学性质: K ≥ max(steps); 对 square_block_starts 的连续切片, register
    1..K 每个位置至少被某个步允许（首步前缀规则覆盖开头, 各块连续铺满）。
    显式非平方 steps 不保证全覆盖（文档注明即可）。

    注意: K 推导只依赖步集本身, 与 skip_steps 起点无关。
    """
    if not steps:
        return num_patches
    max_t = max(int(t) for t in steps)
    return min((math.isqrt(max_t) + 1) ** 2 - 1, num_patches)


# ═══ select_steps — 最终生效采样步集（SRPhase1V2 / OutputQueryDecoder 共用）═══

def select_steps(num_patches: int, decoder_steps: Optional[Sequence[int]] = None,
                 skip_steps: Optional[int] = None,
                 max_steps: Optional[int] = None) -> list:
    """最终生效采样步集: 显式 decoder_steps 原样（去重排序, 不切片）; None 时
    默认计划 square_block_starts(num_patches) 再按 skip_steps/max_steps 做
    Python 切片（如 [4:9] 取中段, 每个保留步仍按自身步值归属分块）。

    入参 num_patches 语义 = "默认分块计划上界": SRPhase1V2 在 K 未知时传
    N=num_patches（切片先于 K 推导, 与 train/infer CLI 的 slice 索引一致）;
    OutputQueryDecoder 独立使用时传自己的 num_specials（K, 已推导完毕）。
    两处对同一配置得到同一步集（SRPhase1V2 把选好的 steps 显式传给
    decoder, decoder 只去重排序不再切片）。

    校验: 步集非空; skip/max 切片索引合法; 步值 ∈ [0, num_patches]。
    """
    explicit = decoder_steps is not None
    if explicit:
        base = sorted(set(int(s) for s in decoder_steps))
    else:
        base = square_block_starts(num_patches)
    assert base and all(0 <= s <= num_patches for s in base), \
        f"steps 越界: {base} (上界={num_patches})"
    if explicit:
        return base
    lo = 0 if skip_steps is None else int(skip_steps)
    hi = len(base) if max_steps is None else int(max_steps)
    assert 0 <= lo < hi <= len(base), \
        f"skip_steps/max_steps 越界: skip={lo} max={hi} " \
        f"(计划共 {len(base)} 步 {base})"
    out = base[lo:hi]
    assert out, f"切片后无采样步: base[{lo}:{hi}] of {base} 为空"
    return out


# ═══ build_block_mask — 分块注意力掩码（每步只见自己的 z_s 块）═══

def build_block_mask(num_tokens: int, steps: Sequence[int],
                     num_queries: Optional[int] = None,
                     device: torch.device = None) -> Tensor:
    """分块注意力掩码（torch float, -inf=屏蔽, 0=允许）：每步只见自己的 z_s 块。

    块号 k = ⌊√t⌋, 块 k = [k², min((k+1)²-1, num_tokens)]（平方数边界铺满
    1..num_tokens; num_tokens = register/specials 数 K）。**第一个步特殊**:
    能看到自己块之前的所有元素（含位置 0 = z_cls）——避免
    skip_steps/max_steps 切片丢弃的前段信息完全不被使用; 其余步只允许
    自己的块, 位置 0 屏蔽。

    返回 (|T|·Q, num_tokens+1)：Q = 每采样步的查询行数（默认 = num_tokens,
    即历史 K==N 全量行为; K<N 时解码器须传 num_queries=N——每步 N 行 patch
    查询共享同一掩码行, 掩码高度按查询行数铺, 列数按 K+1 铺）。
    步值须 ∈ [0, num_tokens]（构造方已断言）; t=0 无块（仅显式传 0 时出现,
    默认计划不含 0）。
    """
    S = num_tokens + 1                      # z_cls + K 个 z_s
    Q = num_tokens if num_queries is None else int(num_queries)
    rows = []
    for i, t in enumerate(steps):
        k = math.isqrt(int(t))              # 块号 = 步值所在的平方块
        hi = min((k + 1) * (k + 1) - 1, num_tokens)
        row = torch.full((S,), float("-inf"), device=device)
        if i == 0:
            row[:hi + 1] = 0.0              # 第一个步: 前面所有元素（含 z_cls）到块终点
        else:
            lo = max(k * k, 1)              # 位置 0 (z_cls) 屏蔽
            if lo <= hi:
                row[lo:hi + 1] = 0.0        # 只允许自己块内的 z_s
        rows.append(row)
    return torch.stack(rows).repeat_interleave(Q, dim=0)


# ═══ build_causal_query_mask — 查询自注意力掩码（块因果 / 块对角; 历史遗留）═══

# 查询自注意力掩码的合法模式。**当前架构不消费它**: 解码器每步只跑自己那 N 行
# 查询（单步内 N 行全双向自注意力）, 根本不存在"跨步查询自注意力"; 读窗口是
# 直接切 A[:, lo:hi+1]（步 t 所在的平方块, 首步含 z_cls）当 memory, 也不需要显式掩码。此处保留
# build_causal_query_mask 纯函数 + 模式常量, 仅供**诊断/可视化脚本**复现历史
# 掩码形状（doc/2026-09-10/smoke_query_mask_mode.py、output/mask_viz/ 等）。
#   "blockdiag"（默认）: 块对角——每步只 attend 自己那 num_queries 行。
#   "causal"            : 块下三角（步 t 可见步 ≤ t）——并行路径的历史行为。
QUERY_MASK_MODES = ("causal", "blockdiag")
DEFAULT_QUERY_MASK_MODE = "blockdiag"

# 训练损失口径（SRPhase1V2.decode）—— **唯一口径, 无开关**: 每个采样步的输出
# Y_t 自己过 PixelHead **直接**预测整图（不做任何累加/集成）, 各步平权:
#     L = mean_t L1(PixelHead(Y_t), target)
# 历史遗留: 曾有 loss_mode="cumulative"/"final" + loss_decouple 两个开关切
# "cumsum 累加结果"口径; 2026-09-15 起整块删除（连同模块级常量 LOSS_MODES）,
# 需要复现旧产物时从 git 取回当时的 model_v2.py。
# F_hat = Y_pix[:, -1]（最后一步的直接预测）, recon = L1(F_hat, target)。


def build_causal_query_mask(num_steps: int, num_queries: int,
                            device: torch.device = None,
                            mode: str = DEFAULT_QUERY_MASK_MODE) -> Tensor:
    """查询自注意力掩码（torch float, -inf=屏蔽, 0=允许）。返回 (L,L) 方阵,
    L = num_steps·num_queries; 查询行序 = (t,k) 展平
    （index = t*num_queries + k, 每采样步 num_queries 行 patch 查询连排;
    列 = 同一套查询行）。

    **历史遗留（当前架构不消费）**: 本函数是并行解码路径（|T| 步一次算完,
    跨步查询自注意力）的 tgt_mask 构造器; 循环架构每步只跑自己那 N 行查询,
    故训练/推理路径不再调用它。保留原因: 既有诊断/可视化脚本按此名引用,
    且它精确记录了两代掩码语义，便于对照历史产物。mode="blockdiag" 恰等价于
    循环路径"单步内 N 行全双向"的隐含语义。

    函数名沿用历史名; 两种语义由 mode 选择（合法值见 QUERY_MASK_MODES）。
    **默认 mode 自 2026-09-10 起为 "blockdiag"**（原默认 "causal"）。

    · mode="blockdiag"（**默认**）—— **块对角**: 步 t 的行只能 attend 自己
      那 num_queries 行。每步退化为一个独立的 num_queries×num_queries 稠密
      双向注意力, 步与步在自注意力上完全隔离。此时:
        - memory_mask 的"每步只见自己的 z_s 块"在整条前向路径上字面成立;
        - loss 的跨步梯度回流由循环 carry 提供（decode 侧无累加）⇒ 2026-09-23
          起 carry 走 **BPTT**（不 detach）, 跨步梯度**存在**; 本掩码函数是
          历史遗留纯函数, 仅供参考（当前循环架构每步只跑自己那 N 行查询,
          不使用 tgt_mask）。
      **渐进语义不依赖本掩码**: 它来自 memory_mask 侧的信息量递增（每步只
      读自己块, 允许列数随步单调增 4→5→7→9→11）, 故 blockdiag 不破坏渐进性。
      **局限（实测）**: 对 register 塌缩**中性**——块内非种子成员的梯度差在
      两种模式下都恰为 0（该对称性只由"自己的 cross-attn + 块内 pos"决定,
      与跨步自注意力无关）, 故它不是塌缩的修复手段。
      **架构含义**: 步间前向的信息流动被移除——模型从"5 步串行、共享权重的
      循环结构"变成"5 个共享权重的并行预测头"（各步输出各自直接预测整图）。
      参数效率不降（权重仍共享）, 但每步的有效输入从累积前缀缩回自己那一块。

    · mode="causal"（历史行为, 3151bab 之前的全部产物）—— **块下三角**:
      步 t 的行可 attend 步 ≤ t 的所有行（同步内全双向, patch 间可交换信息;
      + 之前步）, 未来步屏蔽。目的: 后步查询内容（z_s 采样种子）不再经查询
      自注意力影响前步输出。
      **范围限定（2026-09-10 实测）**: 该模式只保证"后步不泄露进前步";
      反方向（前步 → 后步）的泄露被允许且实际发生——因 depth≥2 时第一层
      输出的各行已含"自己那块的 cross-attn 结果", 第二层的块下三角会把
      **之前所有块的整块信息**带进后步。故步 t 的实际依赖集是**累积前缀**
      （z_s 从位置 1 到自己块末）, 而非"只有自己那一块"。实测 N=576/K=35/
      steps=[1,4,9,16,25]: 步 1→z_s[0:2], 步 4→z_s[0:7], 步 9→z_s[0:14],
      步 16→z_s[0:23], 步 25→z_s[0:34]。同理 loss 的跨步梯度回流也经由本
      掩码存在（步 25 的损失会推动块 1 的 register）。
    """
    assert mode in QUERY_MASK_MODES, \
        f"未知查询掩码 mode={mode!r}, 合法值 {QUERY_MASK_MODES}"
    T, Q = int(num_steps), int(num_queries)
    assert T >= 1 and Q >= 1, f"num_steps/num_queries 须 ≥1, got {T}/{Q}"
    L = T * Q
    m = torch.full((L, L), float("-inf"), device=device)
    for t in range(T):
        if mode == "causal":
            m[t * Q:(t + 1) * Q, :(t + 1) * Q] = 0.0
        else:                                        # mode == "blockdiag"
            m[t * Q:(t + 1) * Q, t * Q:(t + 1) * Q] = 0.0
    return m


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

    **布局的权威定义 = SRPhase1V2.decode 里 target_pix 的构造**, 现在是
    `F.unfold(patch=14, stride=14)`（ATen 定义: 列序 c*14*14 + dy*14 + dx,
    块序 row-major）, N=(H/14)*(W/14)。本函数 = 它的严格逆 `F.fold`。
    model_v3.py 自检 §1b 有往返断言（patches_to_image(target_pix,
    clamp_255=False) == 输入图 (B,H,W,C) 逐位相等）。

    踩坑记录（2026-09-15 修, 2026-11-13 改为 fold）: 消费方历史上把反变换
    写成 `reshape(B, gh, gw, 14, 14, 3).permute(...)`——把 588 当成
    "(14,14,3) 通道在后"读, 与 decode 的通道优先布局不符 ⇒ 每张"重建图"的
    每个 14×14 patch 内部像素被打乱（实测往返 vs 原图 max|diff|=255 /
    mean≈85; 修正后为 0）。训练不受影响（loss 两侧同用同一个 patch 平铺
    向量, 无需求逆）, 但可视化与 0-255 口径的 L1 都不可信。消费方
    （infer_v2_test.py / visualize_recon_pixel.py）一律调用本函数。
    改用 `F.fold` 后布局由库保证, 不再依赖手写 permute 顺序。

    clamp_255: 只在反归一化后生效（默认 True）; 纯布局往返（输入已在
    归一化空间、不传 mean/std）传 False, 否则会被 clamp 破坏数值。
    """
    B, N, px = pixels.shape
    assert px == 3 * 14 * 14, f"patch 向量长度应为 588, got {px}"
    assert (H // 14) * (W // 14) == N, \
        f"N={N} 与 {H}x{W} 的 patch 数 {(H // 14) * (W // 14)} 不符"
    img = F.fold(pixels.transpose(1, 2), output_size=(H, W),
                 kernel_size=14, stride=14)          # (B,3,H,W)
    img = img.permute(0, 2, 3, 1)                    # (B,H,W,3)
    if mean is not None and std is not None:
        m = torch.as_tensor(mean, dtype=img.dtype,
                            device=img.device).view(1, 1, 1, 3)
        s = torch.as_tensor(std, dtype=img.dtype,
                            device=img.device).view(1, 1, 1, 3)
        img = img * s + m
    if clamp_255:
        img = img.clamp(0.0, 255.0)
    return img


# ═══ OutputQueryDecoder — 输出查询注意力解码器（采样时刻上输出全部 patch）═══

class OutputQueryDecoder(nn.Module):
    """把 [z_cls; z_s] 当时序序列 A=(S,D), S=num_specials+1=K+1, 在采样时刻
    T_sub 上每步输出 (N,D) = 该步对全部 patch 的**直接**预测（每步各自过
    PixelHead 成像素, 无累加/集成; F_hat = 最后一步, 在 decode 做）。

    **查询来源（2026-11-12 起）**: 本模块**不再持有 query_base**——查询由
    MiddleEncoder 逐帧给出（`forward(z_cls, z_s, query)` 的 `query`）:
    视频第 f 帧的 Q_f 是 MiddleEncoder 用历史帧 z_s 递推出的 (B,N,D); 图像
    （F=1）用同一个 MiddleEncoder 的冷启动输出。`query_base` 参数已迁到
    MiddleEncoder（行 k↔patch k 的语义不变）。

    **机制（顺序循环——用户的原始写法）**:
    A = [z_cls; z_s] + pos_embed; 读窗口用**切片**给出: 第 i 个采样步 t 的
    memory = A[:, lo:hi+1], 块起止 k=⌊√t⌋, hi=min((k+1)²−1, K), 非首步
    lo=max(k²,1)（只读自己那块）, 首步 lo=0（含 z_cls 及其前的全部元素）——
    与 build_block_mask 的块口径逐位一致; 默认平方块计划（含其连续切片）下
    位置 1..K 全被某个步读到（无花瓶 register）。查询 = query（step1）或
    query + 上一步输出（喂回自己当下一步查询）; 循环按 steps 逐个走,
    **|T| = len(steps)**（每个采样步读自己那块, 不再丢尾步）。
    · num_specials（K）= z_s 长度; None = num_patches（历史 K==N 行为）。
      num_patches（N）只决定查询行数 = 输出 patch 预测数, 两者解耦后
      不再恒等——pos_embed 尺寸 (1,K+1,D), query 行数 = N。
    · 读窗口 = **平方块**（步 t 读 z_s[k²:(k+1)²−1], 首步额外含 z_cls）。
    · 每步跑一次 nn.TransformerDecoder 堆叠（自注意力 + 交叉注意力(memory=该步
      块切片) + FFN）; 单步内 N 行查询全双向自注意力。
    · 返回 (B,|T|,N,D) = 每步全部 patch 的**直接**预测（每步各自过 PixelHead;
      F_hat = 最后一步, 都在 decode 做）。
      **梯度走 BPTT**（唯一路径, 无开关）: 循环本体把 (query + 上一步输出)
      原样喂给下一步（**不 detach**）, decode 侧无累加 ⇒ 跨步梯度存在, 即存在
      "让 Y_{t-1} 成为对下一步有用的草稿"的梯度通路; detach 口径实测曲线是平线
      （doc/2026-09-22/REPORT_batch0_rd_earlystop.md §2b）, 开关已整块删除
    · 时刻采样（显存优化, 默认开启）: 采样计划 =
      select_steps(num_specials, steps, skip_steps, max_steps): 默认计划 =
      square_block_starts(K); skip_steps/max_steps 可选切片挑选; steps= 可
      自定义（原样不切片）。注意: SRPhase1V2 会先按 N 计划选好步再以显式
      steps 传入（两处结果一致, 见 select_steps 的 docstring）
    · 覆盖语义: 每个采样时刻都**直接**预测全部 N 个 patch, 每步的直接预测
      都被监督还原全部 patch（平权, 无累加）
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 mlp_ratio: float = 4.0,
                 steps: Optional[Sequence[int]] = None, heads: int = 8,
                 depth: int = 2, skip_steps: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 num_specials: Optional[int] = None,
                 stack_dim: Optional[int] = None, dropout: float = 0.0,
                 step_plan: str = "square", block: int = 0):
        super().__init__()
        self.num_patches = num_patches                 # 查询基行数 = N（输出 N 个 patch 预测, 不变）
        self.num_specials = num_patches if num_specials is None else int(num_specials)
        # 采样计划: 与 SRPhase1V2 共用 select_steps; 独立使用时默认计划
        # 上界 = K（num_specials）; 显式 steps 原样（调用方负责, 不切片）
        self.steps = select_steps(self.num_specials, steps,
                                  skip_steps, max_steps)
        # 读窗口计划（2026-09-23 新增, 默认保持历史口径）:
        #   "square"（默认）= 步值平方数, 步 t 读 z_s[k²:(k+1)²−1], k=⌊√t⌋;
        #   "fixed"        = 步值自然数, 第 k 步固定读 block 个 z_s。
        # 几何统一由 step_windows 给出; 首步都额外含 z_cls。z_s 全部来自末层。
        assert step_plan in ("square", "fixed"), f"未知 step_plan={step_plan}"
        self.step_plan = step_plan
        self.block = int(block)
        if step_plan == "fixed":
            assert self.block >= 1, f"step_plan=fixed 须 block ≥1, got {block}"
        # 最近一次 forward 实际使用的 A 坐标读窗口 [(lo, hi), …]（诊断/自检用）
        self.last_windows: Optional[list] = None
        S = self.num_specials + 1                                # z_cls + K z_s
        # 2026-11-12: query_base 已整块迁到 MiddleEncoder（视频模态的帧间循环
        # 初始查询）。本模块只消费调用方给的 query，不再自带查询参数。
        # carry = **BPTT**（唯一路径, 2026-09-23 起无开关）: (query + 上一步输出)
        # 原样喂给下一步（不 detach）⇒ 存在"让 Y_{t-1} 成为对下一步有用的草稿"的
        # 梯度通路; 实测 L1(t) 才有单调下降的阶梯（detach 口径曲线是平线, 见
        # doc/2026-09-22/REPORT_batch0_rd_earlystop.md §2b）。
        # 标准解码器堆叠: nn.TransformerDecoder 内部按 num_layers 深拷贝同一
        # decoder_layer 并顺序执行（含逐层传掩码）, 无需手写循环。每层 =
        # 自注意力 + 交叉注意力(memory=A) + FFN + 残差, 各层共享同一 memory
        #
        # stack_dim（= self.stack 的 d_model）与模型 dim 解耦:
        #   · 默认 None/0 ⇒ stack_dim == dim, 不加任何投影, 与原实现逐位一致;
        #   · 显式放大（如 2×dim=2048）⇒ 模型其余部分（pos_embed / query /
        #     PixelHead / DINO 输出）仍是 dim 维, 故在 self.stack 前后各加一层
        #     Linear: dim→stack_dim（输入侧, 查询 Y 与 memory A 共用同一投影）
        #     与 stack_dim→dim（输出侧, 交回 PixelHead / 直接预测损失）。
        # dim_feedforward 随 stack_dim 等比缩放（mlp_ratio 语义不变）。
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

    def forward(self, z_cls: Tensor, z_s: Tensor, query: Tensor) -> Tensor:
        """z_cls/z_s: 当前帧 (B,1,D)/(B,K,D); query: (B,N,D) 或 (N,D)。

        query 的来源 = MiddleEncoder 在当前帧的输出（视频第 f 帧 = Q_f;
        图像/首帧 = 用 [cls] 冷启动的 Q_0）。本模块不再持有 query_base。
        """
        assert z_cls.dim() == 3 and z_s.dim() == 3, \
            f"z_cls/z_s 须为 (B,·,D), got {tuple(z_cls.shape)}/{tuple(z_s.shape)}"
        assert z_cls.shape[1] == 1, f"z_cls 应为 1 列, got {z_cls.shape[1]}"
        assert z_s.shape[1] == self.num_specials, \
            f"z_s 列数 {z_s.shape[1]} != num_specials(K)={self.num_specials}"
        assert z_cls.shape[0] == z_s.shape[0], \
            f"batch 不一致: z_cls {z_cls.shape[0]} vs z_s {z_s.shape[0]}"
        B = z_cls.shape[0]
        # query 允许单独传 (N,D)（旧用法/单元测试）, 补 batch 维后与 A 同 batch;
        # (N,D) 必须显式 expand 成 3-D——TransformerDecoder 会把 2-D tgt 当
        # unbatched, 与 3-D memory 冲突。
        if query.dim() == 2:
            assert query.shape == (self.num_patches, z_cls.shape[-1]), \
                f"query 形状 {tuple(query.shape)} != ({self.num_patches}," \
                f"{z_cls.shape[-1]})"
            base = query.unsqueeze(0).expand(B, -1, -1)          # (B,N,D)
        else:
            assert query.dim() == 3 and query.shape[0] == B \
                and query.shape[1] == self.num_patches, \
                f"query 须为 (B,N,D), got {tuple(query.shape)}"
            base = query
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed      # (B,S,D)
        A_in = self.stack_in(A)                                  # 投影一次, 循环里只切片
        # step1 查询 = MiddleEncoder 给出的 query
        Y = base                                                 # (B,N,D)
        Y_total = []
        # 读窗口 = step_windows 给出的计划（plan="square" 与历史平方块口径
        # 逐位一致; plan="fixed" = 步值自然数、每步固定读 block 个）
        windows = step_windows(self.num_specials, self.steps,
                               self.step_plan, self.block)
        for i, t in enumerate(self.steps):
            lo, hi = windows[i]
            Y = self.stack(self.stack_in(Y), A_in[:, lo:hi + 1])
            Y = self.stack_out(Y)
            Y_total.append(Y)                                    # 该步全部 patch 预测
            # carry = BPTT（唯一路径, 不 detach）: 喂给下一步当查询（见 __init__ 注释）
            Y = base + Y
        Y = torch.stack(Y_total, dim=1)                          # (B,|T|,N,D) 沿步
        self.last_windows = windows                              # 本次实际读窗口（A 坐标）
        self.last_query = base                                   # 本次使用的查询基
        self.last_Y = Y                                          # 采样步全部 patch 预测
        return Y

# ═══ MiddleEncoder — 视频帧间循环交叉注意力（query_base 的新来源）═══

class MiddleEncoder(nn.Module):
    """帧间循环交叉注意力: 上一帧的输出作为下一帧的 query, KV = [cls] + 历史帧 z_s。

    **来源变更（2026-11-12）**: 原 `OutputQueryDecoder.query_base`（nn.Parameter,
    (N,D), 行 k↔patch k）整块迁到这里, 成为帧间循环的**初始查询**。解码器不再
    持有查询参数, 只消费本模块逐帧给出的 Q_f（`decode` 里传给 decoder）。

    **循环（帧 f = 0..F−1）**:
        Q_{-1} = query_base                              ← (N,D) 可学习, 本模块参数
        KV_0   = [cls]                                   ← 无历史（冷启动）
        KV_f   = [cls; z_s[0]; …; z_s[f−1]]              ← f>0, 1 + K·f 个 key
        Q_f    = MiddleEncoder(Q_{f−1}, KV_f)            ← 上一步输出当下一步 query
    返回 (B,F,N,D)。"所有历史帧"**不含当前帧**（用户口径 2026-11-12 确认）:
    首帧/单图 KV 只有 [cls]，这正是"交叉注意力输入刚开始是一个 [cls] 向量"。
    每帧 z_s 有 K 个 token（num_specials; 240M 配置 K=32 ⇒ 32×n 个历史 token）,
    K 由编码器实际输出决定, 本模块不硬编码 32。

    **结构**: 预归一化 `nn.TransformerDecoderLayer` 堆叠（query 自注意力 +
    交叉注意力(memory=KV_f) + FFN + 残差; `norm_first=True`, gelu）, 外加
    末层 LayerNorm（`nn.TransformerDecoder(norm=...)`）。self-attn 让 N 行查询
    在帧内交换信息（与 OutputQueryDecoder 同构, 用户 2026-11-12 选择）；
    参数 `depth` 默认 1, 控制跨帧状态容量。

    **图像 = 单帧视频**: F=1 时 KV_0=[cls], 输出 Q_0 直接当解码器查询——与
    视频首帧走**同一段代码、同一套权重**, 无图像/视频分支。

    **梯度**: 帧间状态与 KV 都不 detach ⇒ 跨帧 BPTT（与解码器 carry 口径一致）;
    后帧损失能反传到前帧 z_s 与本模块参数。

    `last_memory_lens` / `last_queries` 记录最近一次 forward 的 KV 长度序列与
    逐帧输出（诊断/自检用; 长度应为 [1, 1+K, 1+2K, …]）。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 heads: int = 8, depth: int = 1, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        # 原 OutputQueryDecoder.query_base: 行 k↔patch k 的查询基（初始化口径不变）
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        # KV 的常驻初始化向量 [cls]（无历史时 KV 只有它; 不是编码器的 z_cls）
        self.mem_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True),
            num_layers=depth,
            norm=nn.LayerNorm(dim),
        )
        # 最近一次 forward 的 KV 长度序列 / 逐帧查询（诊断、自检用）
        self.last_memory_lens: Optional[list] = None
        self.last_queries: Optional[Tensor] = None

    def forward(self, z_s: Tensor) -> Tensor:
        """z_s: (B,F,K,D) 逐帧编码器输出 → Q: (B,F,N,D) 逐帧解码器查询。

        F 维必须显式给出（图像传 F=1）; 同 batch 内帧数固定（未做变长 padding）。
        """
        assert z_s.dim() == 4, \
            f"z_s 须为 (B,F,K,D), got {tuple(z_s.shape)}"
        B, n_frames, K, D = z_s.shape
        assert D == self.dim, f"z_s 维度 {D} != dim={self.dim}"
        # Q_{-1} = query_base; 循环体内逐步换成上一帧输出
        state = self.query_base.unsqueeze(0).expand(B, -1, -1)   # (B,N,D)
        outs, lens = [], []
        for f in range(n_frames):
            # KV_f = [cls] + 历史帧 z_s[0..f-1]（不含当前帧 f）
            mem = self.mem_init.expand(B, -1, -1)                # (B,1,D)
            if f > 0:
                hist = z_s[:, :f].reshape(B, f * K, D)           # (B,f*K,D)
                mem = torch.cat([mem, hist], dim=1)              # (B,1+f*K,D)
            state = self.stack(state, mem)                       # 跨帧 BPTT（不 detach）
            outs.append(state)
            lens.append(mem.shape[1])
        Q = torch.stack(outs, dim=1)                             # (B,F,N,D)
        self.last_memory_lens = lens
        self.last_queries = Q
        return Q

# ═══ SRPhase1V2 — 主模型（register 式; 无预算; K=num_specials 由步集推导）═══

class SRPhase1V2(nn.Module):
    """register 式主模型。num_specials（K）与 num_patches（N）解耦:
    N = 输入 patch 数 = 解码查询行数 = 输出 patch 预测数（不变）; K =
    register/specials 数 = 解码器实际读取的 z_s 范围。默认（num_specials
    = None）K 由"最终生效采样步集"自动推导: 先按 N 计划选出步集
    steps_selected（显式 decoder_steps 原样; 否则 square_block_starts(N)
    再 skip_steps/max_steps 切片）, 再 K = derive_num_specials(N,
    steps_selected)——与解码器独立用法同一步集（构造时把 steps_selected
    以显式 steps 传给 OutputQueryDecoder, 由后者去重排序, 不再二次切片）。
    显式 num_specials=K 时断言 1≤K≤N 且 max(steps_selected)≤K（K 过小 =
    采样步超出 z_s 范围, 清晰报错; 复现旧 checkpoint 时须显式传回旧 K）。

    K=N（默认全量）序列 = [cls; specials(N); patches(N)] = 2N+1 token, 与
    历史一致; K<N 时序列 = 1+K+N token, register 1..K 全被解码器读（无
    "花瓶 register", 见 derive_num_specials / DESIGN doc）。

    **视频模态（2026-11-12 新增, 唯一处理路径）**: 模型内部**全部按**
    (B,F,3,H,W) 处理（F 为帧数; 同 batch 固定 F, 未做变长 padding）。图像
    (B,3,H,W) 只在 forward 入口 `unsqueeze(1)` 成 F=1, 之后没有任何图像
    分支; 输出也**始终带 F 维**（图像输出 (B,1,...)）。逐帧编码得
    z_cls (B,F,1,D) / z_s (B,F,K,D); MiddleEncoder 沿帧循环, 把上一帧输出
    当下一帧 query, 交叉注意力 KV = [cls] + 所有历史帧 z_s（**不含当前帧**）:
        Q_{-1} = query_base（MiddleEncoder 参数, 自 OutputQueryDecoder 迁入）
        KV_f   = [cls]                       f=0
               = [cls; z_s[0]; …; z_s[f−1]]  f>0
        Q_f    = MiddleEncoder(Q_{f−1}, KV_f)
    OutputQueryDecoder 对每帧用 Q_f 当 query 重建该帧全部 patch; 图像 F=1 时
    KV=[cls] 冷启动, 与视频首帧同一段代码/同一套权重。每帧 z_s 有 K 个 token
    （240M 配置 K=32）; 帧间 BPTT（不 detach）, 后帧损失可反传到前帧 z_s。

    解码器 = **顺序循环**（用户原始写法）: `Y = Q_f`（MiddleEncoder 输出）→
    对每个采样步 t 循环 `Y = stack(stack_in(Y), A[:, lo:hi+1])` →
    `Y = stack_out(Y)` → `Y = Q_f + Y`（喂回自己当下一步查询; carry 走 **BPTT**、
    不 detach, 见 OutputQueryDecoder.forward）, lo/hi = 步 t 那块 z_s 的起止
    （首步 lo=0 含 z_cls）; 返回 (B,|T|,N,D)。
    **并行路径**（|T| 步一次算完 + 跨步 tgt_mask + query_mask_mode）与循环的
    那些开关（recurrent_state/fuse/memory/step_embed/detach）都已整块删除。
    损失 = **直接预测**口径（无开关）: 每帧每步 Y_{f,t} 各自过 PixelHead 预测
    该帧整图, `L = mean_{f,t} L1(PixelHead(Y_{f,t}), target_f)`;
    F_hat = 每帧最后一步的直接预测（Y_pix[:, :, -1]）。旧的 loss_mode /
    loss_decouple（cumsum 累加口径）开关已一并删除, 传它们直接 TypeError
    （不静默忽略）。
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_steps: Optional[Sequence[int]] = None,
        patch_px: int = 14 * 14 * 3,
        decoder_depth: int = 2,
        middle_depth: int = 1,
        skip_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        num_specials: Optional[int] = None,
        stack_dim: int = 0,
        decoder_dropout: float = 0.0,
        step_plan: str = "square",
        block: int = 0,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_px = patch_px
        # 构造参数留档（子类 / 诊断脚本读这里, 不用去嗅探
        # nn.TransformerDecoderLayer 的内部属性 linear1 / dropout1）
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.decoder_dropout = decoder_dropout

        # 最终生效采样步集: 先按 N 计划选步（K 尚未推导; 与 train/infer
        # CLI 的 slice 索引口径一致, 见 select_steps）; K = 显式 num_specials
        # 或由步集自动推导（公式见 derive_num_specials）
        steps_selected = select_steps(num_patches, decoder_steps,
                                      skip_steps, max_steps)
        if num_specials is None:
            # plan="fixed" 的读窗口铺满 1..N（末块 hi=min(⌈N/block⌉·block,N)=N）
            # ⇒ K=N; plan="square" 沿用 derive_num_specials。
            K = (num_patches if step_plan == "fixed"
                 else derive_num_specials(num_patches, steps_selected))
        else:
            K = int(num_specials)
        assert 1 <= K <= num_patches, \
            f"num_specials 越界: K={K} (须 1 ≤ K ≤ N={num_patches})"
        self.num_specials = K
        max_t = max(steps_selected)
        assert max_t <= K, \
            f"num_specials(K)={K} 过小: 采样步最大 {max_t} > K " \
            f"(z_s 只有 {K} 个位置, 步 {max_t} 读不到); 请显式加大 K, 或 " \
            f"缩小 decoder_steps / skip_steps / max_steps"

        self.special_bank = SpecialTokenBank(num_tokens=K, dim=dim)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          mlp_ratio=mlp_ratio, heads=heads,
                                          steps=steps_selected,
                                          depth=decoder_depth,
                                          num_specials=K,
                                          stack_dim=stack_dim,
                                          dropout=decoder_dropout,
                                          step_plan=step_plan,
                                          block=block)
        # query_base 的新来源: 帧间循环交叉注意力（图像 = F=1 冷启动）
        self.middle = MiddleEncoder(dim=dim, num_patches=num_patches,
                                    heads=heads, depth=middle_depth,
                                    mlp_ratio=mlp_ratio,
                                    dropout=decoder_dropout)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px)

    # ── encode: 视频 (B,F,3,H,W) → 逐帧 z_cls, z_s ──
    def encode(self, pixel_values: Tensor):
        """视频输入 (B,F,3,H,W) → z_cls (B,F,1,D), z_s (B,F,K,D)。

        **唯一编码路径**: 图像在 forward 里 unsqueeze(1) 成 F=1 后也走这里。
        逐帧调用 `_encode_register`（不把 F 折进 batch: DINO 24 层在长视频
        上一次性展开会顶显存）。同 batch 内 F 固定（未做变长 padding）; 帧间
        不做任何编码器侧融合——时序信息全部由 MiddleEncoder 的交叉注意力
        引入（编码器结构不变）。register 式: specials 作为额外 token 拼进
        DINO 输入序列, 由 24 层直接算出 z_s（Darcet et al.）。
        """
        assert pixel_values.dim() == 5, \
            f"encode 须为 (B,F,3,H,W), got {tuple(pixel_values.shape)}"
        B, n_frames, C, H, W = pixel_values.shape
        z_cls_f, z_s_f = [], []
        for f in range(n_frames):
            zc, zs = self._encode_register(pixel_values[:, f])   # (B,1,D)/(B,K,D)
            z_cls_f.append(zc)
            z_s_f.append(zs)
        z_cls = torch.stack(z_cls_f, dim=1)                      # (B,F,1,D)
        z_s = torch.stack(z_s_f, dim=1)                          # (B,F,K,D)
        return z_cls, z_s

    def _encode_register(self, pixel_values: Tensor):
        """specials 直接进 DINO 输入序列 [cls; specials(K); patches(N)]
        (1+K+N token, K=self.num_specials)。

        HF Dinov2Model 无 token 级注意力 mask API, 故 DINO 内全双向注意力
        （重建任务无时序因果需求, register 惯例亦然）; 渐进语义由解码器
        分块掩码提供。embeddings() 复用 HF 的 cls/patch 嵌入 + 位置编码
        （含自动插值逻辑）; specials 用 SpecialTokenBank（共享 token +
        逐位置可学习 pos, 不带 DINO PE——与 patch 的位置关系完全学出）。

        24 层跑完 → layernorm → 取 [cls; 前 K 个 specials] 当 z_cls/z_s
        （全部来自**末层**）。逐层 tap / 特征金字塔已于 2026-09-23 整块移除。
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

    # ── 像素目标 / 解码 / forward（全部按 (B,F,...) 处理）──
    def _pixel_target(self, x: Tensor) -> Tensor:
        """(B,3,H,W) 归一化像素 → (B,N,patch_px) patch（通道优先布局）。

        布局权威定义 = `F.unfold`（ATen 里写死: 列序 = c*14*14 + dy*14 + dx,
        块序 row-major）, `patches_to_image` 的 `F.fold` 是它的严格逆。
        2026-11-13: 原来是手写 reshape/permute/reshape, 与 unfold 逐位等价
        但需自己维护布局口径（历史上"通道优先 vs 通道在后"踩过坑）,
        现直接调库。逐帧调用。
        """
        B, C, H, W = x.shape
        assert H % 14 == 0 and W % 14 == 0, "输入须为 14 的倍数"
        assert (H // 14) * (W // 14) == self.num_patches, \
            f"输入 {W}x{H} 产生 {(H // 14) * (W // 14)} patches, " \
            f"但模型 num_patches={self.num_patches}"
        assert C * 14 * 14 == self.patch_px, \
            f"通道数 {C} 对应 patch_px={C * 14 * 14}, 但模型 patch_px={self.patch_px}"
        return F.unfold(x, kernel_size=14, stride=14).transpose(1, 2)

    # ── decode: 逐帧解码 + 逐帧像素损失（唯一路径）──
    def decode(self, z_cls: Tensor, z_s: Tensor, pixel_values: Tensor) -> dict:
        """解码整段视频 (B,F,3,H,W): 每帧用 MiddleEncoder 的 Q_f 当查询。

        输入: z_cls (B,F,1,D), z_s (B,F,K,D), pixel_values (B,F,3,H,W)。
        内部先 `queries = self.middle(z_s)`（(B,F,N,D)），再逐帧跑
        `OutputQueryDecoder` 的 |T| 个采样步, 每帧每步都直接预测该帧整图
        （平权深监督）。图像在 forward 里补成 F=1 后也走这里, 不再有单帧分支。

        返回 dict（始终带 F 维）:
            loss         标量 = mean_{f,t} L1(PixelHead(Y_{f,t}), target_f)
            recon        标量 = mean_f L1(F_hat_f, target_f)（归一化空间）
            F_hat        (B,F,N,588) 每帧最后一步的直接预测
            Y_pix        (B,F,|T|,N,588) 每帧每步的直接预测
            target_pix   (B,F,N,588) 每帧像素目标
        逐帧查询在 `self.middle.last_queries`（诊断用, 不在返回 dict 里）。
        """
        assert pixel_values.dim() == 5, \
            f"decode 须为 (B,F,3,H,W), got {tuple(pixel_values.shape)}"
        assert z_cls.dim() == 4 and z_s.dim() == 4, \
            f"z_cls/z_s 须为 (B,F,·,D), got {tuple(z_cls.shape)}/{tuple(z_s.shape)}"
        B, n_frames = pixel_values.shape[:2]
        D = z_s.shape[-1]
        assert z_cls.shape == (B, n_frames, 1, D), \
            f"z_cls 须为 (B,F,1,D), got {tuple(z_cls.shape)}"
        assert z_s.shape == (B, n_frames, self.num_specials, D), \
            f"z_s 须为 (B,F,{self.num_specials},D), got {tuple(z_s.shape)}"
        queries = self.middle(z_s)                            # (B,F,N,D)
        Y_pix_f, per_step_f, target_f = [], [], []
        for f in range(n_frames):
            target = self._pixel_target(pixel_values[:, f])    # (B,N,588)
            Y = self.decoder(z_cls[:, f], z_s[:, f], queries[:, f])
            Y_pix = self.pixel_head(Y)                        # (B,|T|,N,588)
            per_step = F.l1_loss(
                Y_pix, target.unsqueeze(1).expand_as(Y_pix),
                reduction="none").mean(dim=(0, 2, 3))         # (|T|,)
            Y_pix_f.append(Y_pix)
            per_step_f.append(per_step)
            target_f.append(target)
        Y_pix = torch.stack(Y_pix_f, dim=1)                    # (B,F,|T|,N,588)
        target_pix = torch.stack(target_f, dim=1)              # (B,F,N,588)
        F_pix = Y_pix[:, :, -1]                                # (B,F,N,588)
        recon = F.l1_loss(F_pix, target_pix)
        # loss = 先对每帧求各步 L1, 再对帧平均（帧 × 步平权）
        loss = torch.stack(per_step_f, dim=0).mean()
        return {"loss": loss, "recon": recon, "F_hat": F_pix,
                "Y_pix": Y_pix, "target_pix": target_pix}

    # ── forward（唯一路径; 4D 图像仅转成 F=1 视频的 shape）──
    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,F,3,H,W) 视频; (B,3,H,W) 图像自动补 F=1。

        **只有这一处图像处理**: `unsqueeze(1)` 成 (B,1,3,H,W), 之后编码、
        MiddleEncoder、解码、损失全部按视频路径走, 输出**始终带 F 维**。

        监督**原始像素**而非 DINO patch 特征（特征目标退化: 工地图特征
        空间近常数, 学质心即低 L1 是假收敛）。H,W 须为 14 的倍数。
        损失: 每帧每个采样步的输出 Y_{f,t} 各自**直接**过 PixelHead 预测该帧
        整图, 平权全覆盖 L = mean_{f,t} L1(PixelHead(Y_{f,t}), target_pix_f)
        （无累加/集成）。F_hat = 每帧最后一步的直接预测; recon = 其 L1。

        返回 dict:
            loss        标量 = mean_{f,t} L1(PixelHead(Y_{f,t}), target_f)
            recon       标量 = mean_f L1(F_hat_f, target_f)（归一化空间）
            F_hat       (B,F,N,588)
            Y_pix       (B,F,|T|,N,588)
            target_pix  (B,F,N,588)
        图像输入 (4D) 返回 (B,1,...), 不再压掉 F 维。
        """
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)   # 图像 → (B,1,3,H,W)
        assert pixel_values.dim() == 5 and pixel_values.shape[2] == 3, \
            f"pixel_values 须为 (B,F,3,H,W) 或 (B,3,H,W), " \
            f"got {tuple(pixel_values.shape)}"
        z_cls, z_s = self.encode(pixel_values)
        return self.decode(z_cls, z_s, pixel_values)


# ═══ 自检（python model_v3.py）═══
#   1. 形状正确性（register 式全路径; 全部按 (B,F,...) 处理, 图像只在入口补 F=1）
#   2. OutputQueryDecoder 分块采样计划 + MiddleEncoder 结构（query_base 迁移）
#   3. 梯度流向（整模型可训）+ 直接预测损失结构（每步只收自己那一步的
#      Y 梯度）+ carry = BPTT 唯一路径（detach 开关已删）
#   4. eval 同路径
#   5. self.stack 加宽 + 6. 损失口径 + 7. 视频帧间循环（历史 KV 不含当前帧,
#      图像=单帧视频, 帧间 BPTT 梯度）

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
    model = SRPhase1V2(dino, num_patches=N, dim=D,
                       decoder_steps=square_block_starts(N))  # 显式传完整分块计划(不切片)
    # 像素目标绑定输入尺寸: N=16 patches ⇒ 输入须为 4×4 patch = 56×56 (14 的倍数)
    x = torch.randn(2, 3, 56, 56)
    B, C, H, W = x.shape

    # ── 1. 形状: 目标=像素; 统一口径 F_hat = (B,F,N,588)（图像 F=1）──
    PATCH_PX = 14 * 14 * 3
    out = model(x)
    assert out["F_hat"].shape == (2, 1, N, PATCH_PX), out["F_hat"].shape
    assert out["loss"].shape == () and out["recon"].shape == ()
    # 像素目标提取与模型内部一致
    target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
               .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)
    assert torch.isclose(out["recon"],
                         F.l1_loss(out["F_hat"], target.unsqueeze(1))), \
        "recon 应为像素 L1"
    # ── 1b. 像素 patch ↔ 图像 往返（布局回归测试, 2026-09-15）──
    # patches_to_image 必须是 decode 里 target_pix 布局（通道优先 (C,14,14)）
    # 的严格逆。历史上消费方把反变换写成 (14,14,3) 通道在后 ⇒ 每个 patch
    # 内部像素被打乱（可视化全错）, 这条断言就是那次 bug 的回归测试。
    rt = patches_to_image(out["target_pix"][:, 0], H, W,
                          clamp_255=False)                  # (B,H,W,C)
    assert rt.shape == (B, H, W, C), rt.shape
    assert torch.allclose(rt, x.permute(0, 2, 3, 1)), \
        "patches_to_image 必须与 decode 的 target_pix 布局互为严格逆（通道优先）"
    # 反归一化 + clamp 路径也要能跑通（推理/可视化走这条）
    img255 = patches_to_image(out["F_hat"][:, 0], H, W,
                              mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                              std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    lo, hi = float(img255.detach().min()), float(img255.detach().max())
    assert img255.shape == (B, H, W, C) and lo >= 0.0 and hi <= 255.0, \
        (img255.shape, lo, hi)
    print(f"[ok] shapes: F_hat{tuple(out['F_hat'].shape)} (像素 {PATCH_PX}D) "
          f"loss={out['loss'].item():.4f}; patch↔图像 往返逐位相等, "
          f"0-255 反归一化范围 [{lo:.1f}, {hi:.1f}]")

    # ── 2. OutputQueryDecoder: 采样计划 + 顺序循环(块切片当 memory) ──
    # 注: 解码器现在**没有** attn_mask / tgt_mask（读窗口是 A[:, lo:hi+1] 块切片,
    # 每步一个采样步、|T| = len(steps); 单步内 N 行全双向自注意力）, 故本段只钉
    # 采样计划与循环形状。
    T_steps = model.decoder.steps
    assert T_steps == square_block_starts(N), T_steps       # N=16 → [1,4,9,16]
    assert len(model.decoder.stack.layers) == 2, "默认 decoder_depth=2"
    assert not hasattr(model.decoder, "attn_mask"), \
        "切片式读窗口 ⇒ 不应再有 attn_mask"
    assert not hasattr(model.decoder, "tgt_mask"), \
        "单步内全双向自注意力 ⇒ 不应再有 tgt_mask"

    # ── 2b. build_causal_query_mask: **历史遗留纯函数**回归 ──
    # 当前架构不消费它（见函数 docstring）, 但它精确记录了两代掩码语义, 是复现/
    # 对照旧产物与诊断可视化的工具, 故仍须保证语义不被改坏。
    T3, Q3 = 3, 4
    tm_bd = build_causal_query_mask(T3, Q3, mode="blockdiag")
    tm_ca = build_causal_query_mask(T3, Q3, mode="causal")
    assert DEFAULT_QUERY_MASK_MODE == "blockdiag", DEFAULT_QUERY_MASK_MODE
    assert torch.equal(build_causal_query_mask(T3, Q3), tm_bd), \
        "默认 mode 应为 'blockdiag'"
    assert tm_bd.shape == (T3 * Q3, T3 * Q3), tm_bd.shape
    for ti in range(T3):
        row = tm_bd[ti * Q3]
        assert (row[ti * Q3:(ti + 1) * Q3] == 0).all(), \
            f"blockdiag 对角块 {ti} 应全允许（同步内全双向）"
        rest = torch.cat([row[:ti * Q3], row[(ti + 1) * Q3:]])
        assert (rest == float("-inf")).all(), \
            f"blockdiag 步 {ti} 不应见其它步的查询行"
        row = tm_ca[ti * Q3]
        assert (row[:(ti + 1) * Q3] == 0).all(), f"causal 步 {ti} 应见步 ≤{ti}"
        assert (row[(ti + 1) * Q3:] == float("-inf")).all(), \
            f"causal 步 {ti} 不应见未来步"
    # blockdiag 是 causal 的**真子集**（只砍跨步列, 不动同步内全双向）
    assert int(((tm_bd == 0) & (tm_ca != 0)).sum()) == 0, \
        "blockdiag 的允许集应为 causal 的子集（更严格）"
    assert int((tm_ca == 0).sum()) > int((tm_bd == 0).sum()), \
        "blockdiag 应真的砍掉跨步列（否则等同 causal）"
    # 非法 mode 立刻报错（不静默退化成默认）
    for bad in ("", "causal ", "diag", "block_diag", None):
        try:
            build_causal_query_mask(T3, Q3, mode=bad)
            raise AssertionError(f"mode={bad!r} 应被拒绝")
        except AssertionError as e:
            assert "未知查询掩码 mode" in str(e), f"mode={bad!r}: {e}"
    # 模型层: query_mask_mode 旋钮已随并行路径删除 ⇒ 传它必须**报错**而不是被
    # 静默忽略（否则消费方以为自己在切掩码语义, 实际什么都没发生）。
    for kw in ({"query_mask_mode": "causal"}, {"recurrent": True}):
        try:
            SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D, **kw)
            raise AssertionError(f"{kw} 应被拒绝（该接口已删除）")
        except TypeError as e:
            assert "unexpected keyword" in str(e), e
    print("[ok] query_mask_mode / recurrent 开关已删除: 传它们直接 TypeError"
          "（不静默忽略）; build_causal_query_mask 作为历史遗留纯函数语义不变")

    Y = model.decoder.last_Y                                 # 最后一帧 (B,|T|,N,D)
    assert Y.shape == (2, len(T_steps), N, D)
    # 直接预测: 每步 Y_t 各自过 PixelHead（无累加/集成）
    Y_pix = model.pixel_head(Y)                              # (B,|T|,N,588) 每步直接预测
    assert Y_pix.shape == (2, len(T_steps), N, PATCH_PX)
    assert torch.isclose(Y_pix, out["Y_pix"][:, 0]).all(), \
        "Y_pix 应为每步 Y 的直接像素预测（与 decode 同路径、逐位一致）"
    assert torch.isclose(out["F_hat"][:, 0], Y_pix[:, -1]).all(), \
        "F_hat 应为最后一步的直接预测（不再有累加）"
    per = F.l1_loss(Y_pix, target.unsqueeze(1).expand_as(Y_pix),
                    reduction="none").mean(dim=(0, 2, 3))    # (|T|,) 每步直接预测
    assert torch.isclose(out["loss"], per.mean()), "loss 应为各步直接预测的平权 L1"
    assert torch.isclose(out["recon"], per[-1]), "recon 应等于最后一步的直接预测 L1"
    # 计划自动适配任意 N（可扩展性）: 12→3 块, 256→16 块, 512→22 块
    assert square_block_starts(12) == [1, 4, 9], square_block_starts(12)
    assert len(square_block_starts(256)) == 16
    assert len(square_block_starts(512)) == 22
    # ── plan="fixed"（2026-09-23 新增）: 步值按自然数增长, 每步固定读 block 个 ──
    assert fixed_block_starts(16, 4) == [1, 2, 3, 4], fixed_block_starts(16, 4)
    assert fixed_block_starts(12, 4) == [1, 2, 3], fixed_block_starts(12, 4)
    assert fixed_block_starts(576, 4) == list(range(1, 145))
    assert fixed_block_starts(12, 1) == list(range(1, 13))
    # 窗口几何: N=12/K=12/block=4 → 块 [1..4][5..8][9..12], 首步 lo=0 含 z_cls
    w_fixed = step_windows(12, [1, 2, 3], plan="fixed", block=4)
    assert w_fixed == [(0, 4), (5, 8), (9, 12)], w_fixed
    # N=576/K=576/block=4 ⇒ 144 步铺满, 每步宽度 5(首)/4, 无空洞
    w576 = step_windows(576, fixed_block_starts(576, 4), plan="fixed", block=4)
    assert len(w576) == 144 and w576[0] == (0, 4) and w576[-1] == (573, 576)
    assert all(b - a + 1 == (5 if i == 0 else 4)
               for i, (a, b) in enumerate(w576)), "fixed 计划每步宽度须恒定"
    assert w576[-1][1] == 576, "fixed 计划须铺满 1..K"
    # square 计划经同一函数后与历史口径逐位一致（默认路径不回归）
    assert step_windows(12, [1, 4, 9]) == [(0, 3), (4, 8), (9, 12)]
    d_fix = OutputQueryDecoder(num_patches=16, dim=D, steps=fixed_block_starts(16, 4),
                              step_plan="fixed", block=4)
    assert d_fix.step_plan == "fixed" and d_fix.block == 4
    # query 现在由 MiddleEncoder 提供; 独立使用解码器时显式传 (B,N,D)
    d_fix(torch.randn(1, 1, D), torch.randn(1, 16, D),
          torch.randn(1, 16, D))                            # 循环跑得通
    assert d_fix.last_windows == [(0, 4), (5, 8), (9, 12), (13, 16)], d_fix.last_windows
    # 用户给定示例核对（N=12）: 第一个步 t=1 可见 [0..3]; 步 2 (t=4) 只看 [4..8]
    m12 = build_block_mask(12, [1, 4, 9])
    assert m12.shape == (3 * 12, 13), m12.shape
    r1, r2 = m12[0], m12[12]                                 # 步 1 / 步 2 首查询行
    assert (r1[0:4] == 0).all() and (r1[4:] == float("-inf")).all()
    assert (r2[4:9] == 0).all() and (r2[:4] == float("-inf")).all() \
        and (r2[9:] == float("-inf")).all()
    # 可选挑选（skip_steps/max_steps, 默认 None = 全部分块）: [4:9] 切片取中段
    d_slice = OutputQueryDecoder(num_patches=256, dim=D, skip_steps=4, max_steps=9)
    assert d_slice.steps == square_block_starts(256)[4:9], d_slice.steps
    assert d_slice.steps == [25, 36, 49, 64, 81], d_slice.steps
    d_slice(torch.randn(1, 1, D), torch.randn(1, 256, D),
            torch.randn(1, 256, D))                         # 循环跑得通
    # 默认 None = 全部分块（等价不切片）
    d_full = OutputQueryDecoder(num_patches=256, dim=D)
    assert d_full.steps == square_block_starts(256), "默认 None 应为全部分块"
    assert d_full.steps[0] == 1, "默认计划第一个步应为 t=1"
    d_full(torch.randn(1, 1, D), torch.randn(1, 256, D),
           torch.randn(1, 256, D))
    print(f"[ok] OutputQueryDecoder: {len(model.decoder.stack.layers)} 层 "
          f"TransformerDecoder, 分块采样 {len(T_steps)} 步 {T_steps}, "
          f"可选挑选 [4:9]={d_slice.steps}, 默认全量 {len(d_full.steps)} 步, "
          f"循环体 = 用户原始写法（切片当 memory; 无 attn_mask/tgt_mask）; "
          f"query 由 MiddleEncoder 提供（本模块不再有 query_base）")

    # ── 2a. MiddleEncoder: query_base 迁入 + 帧间循环结构 ──
    assert not hasattr(model.decoder, "query_base"), \
        "query_base 应已迁出 OutputQueryDecoder"
    assert isinstance(model.middle, MiddleEncoder)
    assert model.middle.query_base.shape == (N, D), model.middle.query_base.shape
    assert model.middle.mem_init.shape == (1, 1, D), model.middle.mem_init.shape
    assert len(model.middle.stack.layers) == 1, "middle_depth 默认应为 1"
    assert isinstance(model.middle.stack.layers[0], nn.TransformerDecoderLayer)
    m_mid2 = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                        middle_depth=2)
    assert len(m_mid2.middle.stack.layers) == 2, "middle_depth 应生效"
    # 解码器 forward 必须收到 query（旧的两参数调用应报 TypeError）
    try:
        model.decoder(torch.randn(1, 1, D), torch.randn(1, N, D))
        raise AssertionError("OutputQueryDecoder 不应再能省略 query")
    except TypeError as err:
        assert "query" in str(err), err
    # MiddleEncoder 的输入必须是显式 (B,F,K,D)（图像传 F=1）
    try:
        model.middle(torch.randn(1, N, D))
        raise AssertionError("MiddleEncoder 不应接受无 F 维输入")
    except AssertionError as err:
        assert "B,F,K,D" in str(err), err
    print(f"[ok] MiddleEncoder: query_base{tuple(model.middle.query_base.shape)} + "
          f"[cls]{tuple(model.middle.mem_init.shape)} + "
          f"{len(model.middle.stack.layers)} 层 TransformerDecoderLayer; "
          f"解码器已无 query_base（省略 query 报 TypeError）")

    # ── 2b. num_specials(K) 与 N 解耦: K 由最终采样步集自动推导（无花瓶）──
    # derive_num_specials 公式核对（权威示例）
    assert derive_num_specials(576, [25, 36, 49, 64, 81]) == 99, \
        "N=576 slice [4:9] → K=99"
    assert derive_num_specials(576, [64]) == 80, "显式 steps=[64] → K=80"
    assert derive_num_specials(16, [1, 4, 9, 16]) == 16, "全量默认 → K=N"
    assert derive_num_specials(16, []) == 16, "空步集 → K=N"
    # 配置: N=16, skip_steps=1 / max_steps=3 → 切片 square_block_starts(16)
    # [1,4,9,16][1:3] = [4,9] → K = min((⌊√9⌋+1)²−1, 16) = min(15,16) = 15
    dino_k = FakeDino(dim=D, num_patches=N)
    m_k = SRPhase1V2(dino_k, num_patches=N, dim=D, skip_steps=1, max_steps=3)
    assert m_k.num_specials == 15, m_k.num_specials
    assert m_k.decoder.steps == [4, 9], m_k.decoder.steps
    assert m_k.special_bank.pos.shape == (1, 15, D), m_k.special_bank.pos.shape
    assert m_k.decoder.pos_embed.shape == (1, 16, D), m_k.decoder.pos_embed.shape
    out_k = m_k(x)                                        # 同 x: (2,3,56,56)
    z_s_k = m_k.encode(x.unsqueeze(1))[1]
    assert z_s_k.shape == (2, 1, 15, D), z_s_k.shape      # 编码器只生成 K=15 个 register
    # 梯度仍全通: 显式 K<N 模型的键侧 / 查询侧 / 像素头都要收梯度
    out_k["loss"].backward()
    for name, p in [("m_k.special_bank.pos", m_k.special_bank.pos),
                    ("m_k.decoder.pos_embed", m_k.decoder.pos_embed),
                    ("m_k.middle.query_base", m_k.middle.query_base),
                    ("m_k.middle.mem_init", m_k.middle.mem_init),
                    ("m_k.pixel_head.net", m_k.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] num_specials 自动推导: N={N} slice[1:3] → steps=[4,9], "
          f"K={m_k.num_specials}, z_s{z_s_k.shape} 形状对, 梯度全通")

    # ── 2c. 显式 num_specials（旧 checkpoint 复现/手动指定路径）──
    dino_e = FakeDino(dim=D, num_patches=N)
    m_e = SRPhase1V2(dino_e, num_patches=N, dim=D, num_specials=8,
                     decoder_steps=[4])                    # 步 4 ≤ K=8
    assert m_e.num_specials == 8 and m_e.decoder.steps == [4], \
        (m_e.num_specials, m_e.decoder.steps)
    z_s_e = m_e.encode(x.unsqueeze(1))[1]
    assert z_s_e.shape == (2, 1, 8, D), z_s_e.shape
    out_e = m_e(x)
    assert out_e["F_hat"].shape == (2, 1, N, PATCH_PX)
    # 显式 K 过小 → 清晰报错（max(steps) > K）
    try:
        SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                   num_specials=4, decoder_steps=[9])
        raise AssertionError("应报 num_specials 过小错误")
    except AssertionError as err:
        assert "过小" in str(err), err
    print(f"[ok] 显式 num_specials: K=8 + steps=[4] → K=8, z_s{z_s_e.shape}, "
          f"K 过小(4 vs steps=[9])报错信息含'过小'")

    # ── 3. 梯度流向（整模型可训, 含 PixelHead; 首层自注意力 + 末层交叉/FFN）──
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
                    ("middle.stack.layers[0].multihead_attn.in_proj_weight",
                     model.middle.stack.layers[0].multihead_attn.in_proj_weight),
                    ("middle.stack.layers[0].linear1.weight",
                     model.middle.stack.layers[0].linear1.weight),
                    ("middle.query_base", model.middle.query_base),
                    ("middle.mem_init", model.middle.mem_init),
                    ("special_bank.pos", model.special_bank.pos),
                    ("pixel_head.net", model.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] 梯度: DINO 嵌入/层 + OutputQueryDecoder({len(model.decoder.stack.layers)}×"
          f"TransformerDecoderLayer) + MiddleEncoder({len(model.middle.stack.layers)}×"
          f"DecoderLayer, query_base/mem_init) / SpecialTokenBank/PixelHead 全部可训")

    # ── 3b. 直接预测口径的损失结构（无累加）+ BPTT 唯一路径 ──
    # 无累加 ⇒ dL/dY_t（对**输出张量**求导）只来自第 t 步自己的损失项。校验:
    #   (a) 全量梯度在位置 n == 仅第 n 步损失的梯度(÷|T|);
    #   (b) 第 n 步损失对其他位置 Y_{m≠n} 无梯度（decode 侧无累加）。
    # ⚠️ 这不等于"参数梯度按步解耦": carry 走 BPTT ⇒ 后面的步会经 carry 把梯度
    # 送回前面各步用过的参数（2026-09-23 起 detach 开关已整块删除）。
    out_b = model(x)
    Y_b = model.decoder.last_Y                              # 最后一帧 (B,|T|,N,D)
    Tb = Y_b.shape[1]
    assert torch.isclose(out_b["F_hat"][:, 0], out_b["Y_pix"][:, 0, -1]).all()
    per_b = F.l1_loss(out_b["Y_pix"],
                      out_b["target_pix"].unsqueeze(2).expand_as(out_b["Y_pix"]),
                      reduction="none").mean(dim=(0, 1, 3, 4))   # (|T|,) 每步损失
    g_total = torch.autograd.grad(out_b["loss"], Y_b, retain_graph=True)[0]
    for n in range(Tb):
        g_n = torch.autograd.grad(per_b[n] / Tb, Y_b,
                                  retain_graph=True)[0]          # 仅第 n 步损失
        others = [m for m in range(Tb) if m != n]
        assert torch.allclose(g_n[:, others],
                              torch.zeros_like(g_n[:, others]), atol=1e-5), \
            f"step {n} 的损失不应给其他步的预测梯度（无累加）"
        assert torch.allclose(g_total[:, n], g_n[:, n], atol=1e-5), \
            f"step {n} 的 Y 梯度应只来自自己那一步的损失（平权 1/{Tb}）"
    print(f"[ok] 直接预测损失结构: 每步 Y_t 的损失项只给自己那一步 Y 梯度"
          f"（÷|T|=1/{Tb}, 无累加）")
    # BPTT 唯一路径: 开关必须已删除; 并用最小图验证 carry 不 detach
    # （detach 时后一步损失回不到前一步 ⇒ 这里会是 1.0 而不是 2.0）
    assert not hasattr(model.decoder, "carry_detach"), \
        "carry_detach 开关应已删除（BPTT 是唯一路径）"
    _q = torch.zeros(1, requires_grad=True)
    _y0 = _q + 1.0
    _y1 = _q + _y0
    assert torch.autograd.grad(_y1.sum(), _q)[0].item() == 2.0, "carry 必须走 BPTT"
    print("[ok] carry = BPTT 唯一路径: carry_detach 开关已删除, 跨步梯度通路存在")

    # ── 4. 推理: 同一 forward（eval + no_grad）──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["F_hat"].shape == (2, 1, N, PATCH_PX)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    # ── 5. self.stack 加宽（stack_dim ≠ dim, 前后 Linear 投影）+ dropout ──
    # 默认（stack_dim=0 ⇒ stack_dim==dim）不加任何投影: 不新增 stack_in/out 参数
    # （与 stack_dim=0 的循环实现同构）。
    # ⚠️ checkpoint 兼容性（2026-11-12 更新）: model_v3 的 MiddleEncoder 是**新
    # 参数**（middle.query_base / mem_init / stack.*）, 原 decoder.query_base 已
    # 移除 ⇒ 与 model_v2 的 state_dict **不再逐 key 相同**, 旧 final_model.pt
    # strict load 会报缺键/多键; 需按新结构重训或显式重映射。要复现旧产物必须用
    # git 取回当时的 model_v2.py, 不要拿旧权重在本代码上推理。
    assert model.decoder.stack_dim == D
    assert isinstance(model.decoder.stack_in, nn.Identity), \
        "默认 stack_dim=dim 时输入侧应为 Identity（零额外参数）"
    assert isinstance(model.decoder.stack_out, nn.Identity)
    assert not [k for k in model.state_dict()
                if "stack_in" in k or "stack_out" in k], \
        "默认路径不应出现投影层参数"
    # 显式加宽: dim → stack_dim → dim; FFN 随 stack_dim 等比（mlp_ratio 不变）
    m_wide = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                        decoder_steps=square_block_starts(N), heads=4,
                        decoder_depth=4, stack_dim=2 * D, decoder_dropout=0.05)
    dec = m_wide.decoder
    assert dec.stack_dim == 2 * D, dec.stack_dim
    assert isinstance(dec.stack_in, nn.Linear) and dec.stack_in.in_features == D \
        and dec.stack_in.out_features == 2 * D, "输入侧应为 dim→stack_dim 投影"
    assert isinstance(dec.stack_out, nn.Linear) and dec.stack_out.in_features == 2 * D \
        and dec.stack_out.out_features == D, "输出侧应为 stack_dim→dim 投影"
    assert len(dec.stack.layers) == 4, "加宽路径 depth 应生效"
    assert dec.stack.layers[0].linear1.in_features == 2 * D
    assert dec.stack.layers[0].linear1.out_features == 4 * (2 * D), \
        "dim_feedforward 应随 stack_dim 等比缩放（mlp_ratio=4）"
    assert dec.stack.layers[0].dropout1.p == 0.05, "dropout 应透传进 stack"
    out_w = m_wide(x)
    assert out_w["F_hat"].shape == (2, 1, N, PATCH_PX), out_w["F_hat"].shape
    assert out_w["loss"].shape == ()
    # 加宽权重可 save/load 往返（训练收尾 final_model.pt 依赖此路径）
    m_wide.load_state_dict(m_wide.state_dict(), strict=True)
    print(f"[ok] self.stack 加宽: d_model {D}→{dec.stack_dim}, heads=4, "
          f"depth={len(dec.stack.layers)}, dropout=0.05, "
          f"投影 {tuple(dec.stack_in.weight.shape)}→{tuple(dec.stack_out.weight.shape)}")

    # ── 6. 损失口径: **直接预测**（每步 Y 各自过 PixelHead, 无累加）──
    # 注: 原 §6 的循环开关自检（rec_proj zero-init / 跨步信息流 / BPTT 截断 /
    # 读窗口三档 / open 步退化 + step_embed）随那些开关整块删除;
    # 循环体现在就是 forward 里那 4 行。
    # 6.9.0 唯一口径: loss = 各步直接预测的平权 L1（无 loss_mode/loss_decouple）
    _per = F.l1_loss(out["Y_pix"],
                     out["target_pix"].unsqueeze(2).expand_as(out["Y_pix"]),
                     reduction="none").mean(dim=(0, 1, 3, 4))
    assert torch.isclose(out["loss"], _per.mean()), "loss 应为各步直接预测 L1 的均值"
    assert not hasattr(model, "loss_mode") and not hasattr(model, "loss_decouple"), \
        "旧口径属性应已删除（消费方别再从模型上读 loss_mode/loss_decouple）"
    assert "LOSS_MODES" not in globals(), "模块级常量 LOSS_MODES 应已删除"
    # 6.9.1 F_hat = 每帧最后一步的直接预测; recon = 其 L1
    assert torch.isclose(out["F_hat"], out["Y_pix"][:, :, -1]).all(), \
        "F_hat 应为每帧最后一步的直接预测"
    assert torch.isclose(out["recon"], _per[-1]), "recon 应等于最后一步直接预测的 L1"
    # 6.9.2 旧开关传即 TypeError（不静默忽略: 否则消费方以为在切口径, 实际什么都没发生）
    for kw in ({"loss_mode": "cumulative"}, {"loss_mode": "final"},
               {"loss_decouple": True}, {"loss_decouple": False}):
        try:
            SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D, **kw)
            raise AssertionError(f"{kw} 应被拒绝（累加口径已删除）")
        except TypeError as e:
            assert "unexpected keyword" in str(e), e

    print("[ok] 损失口径: 直接预测 mean_{f,t} L1(PixelHead(Y_{f,t}), target)（无累加/cumsum）; "
          "F_hat = 每帧最后一步的直接预测, recon = 其 L1; "
          "旧 loss_mode/loss_decouple 传即 TypeError, LOSS_MODES 常量已删除")

    # ── 7. 视频模态: MiddleEncoder 帧间循环（历史 KV 不含当前帧）──
    # 结构: 第 f 帧 KV = [cls] + z_s[0..f-1] ⇒ 长度 [1, 1+K, 1+2K, …]; 图像
    # (F=1) 就是 KV=[cls] 的首帧, 与视频首帧逐位一致。
    FV = 3
    x_v = torch.randn(2, FV, 3, 56, 56)                     # (B,F,3,H,W)
    model.eval()
    with torch.no_grad():
        out_v = model(x_v)
        # 每帧像素目标: 与 decode 的 _pixel_target 同一布局（逐帧通道优先）
        target_v = x_v.reshape(2, FV, 3, 4, 14, 4, 14) \
                      .permute(0, 1, 3, 5, 2, 4, 6).reshape(2, FV, N, PATCH_PX)
    assert out_v["Y_pix"].shape == (2, FV, len(T_steps), N, PATCH_PX)
    assert out_v["F_hat"].shape == (2, FV, N, PATCH_PX)
    assert out_v["target_pix"].shape == (2, FV, N, PATCH_PX)
    assert torch.allclose(out_v["target_pix"], target_v)
    # KV 长度: [cls], [cls]+K, [cls]+2K ⇒ 历史帧不含当前帧（必须在后续图像
    # 前向前读取; 图像 F=1 会把 last_memory_lens 覆盖成 [1]）
    K = model.num_specials
    assert model.middle.last_memory_lens == [1, 1 + K, 1 + 2 * K], \
        model.middle.last_memory_lens
    assert model.middle.last_queries.shape == (2, FV, N, D)
    # 每帧每步平权 ⇒ loss = mean_{帧,步}; recon = 每帧最后一步 L1 的均值
    per_step_v = F.l1_loss(
        out_v["Y_pix"], target_v.unsqueeze(2).expand_as(out_v["Y_pix"]),
        reduction="none").mean(dim=(0, 1, 3, 4))            # (|T|,) 帧平均
    assert torch.isclose(out_v["loss"], per_step_v.mean()), \
        "视频 loss 应为帧 × 步平权 L1"
    assert torch.isclose(out_v["recon"], F.l1_loss(out_v["F_hat"], target_v)), \
        "视频 recon 应为每帧最后一步预测的 L1 均值"
    assert torch.allclose(out_v["F_hat"], out_v["Y_pix"][:, :, -1]), \
        "F_hat 应为每帧最后一步的直接预测"
    # 图像 = 单帧视频: 4D 与 5D(F=1) 输出逐位一致（都带 F 维）; 3 帧视频首帧也一致
    with torch.no_grad():
        out_i = model(x_v[:, 0])                            # 4D 图像 → (B,1,...)
        out_v1 = model(x_v[:, :1])                          # F=1 视频
    assert out_i["F_hat"].shape == (2, 1, N, PATCH_PX), out_i["F_hat"].shape
    assert torch.allclose(out_i["F_hat"], out_v1["F_hat"], atol=1e-6)
    assert torch.allclose(out_i["Y_pix"], out_v1["Y_pix"], atol=1e-6)
    assert torch.allclose(out_i["F_hat"], out_v["F_hat"][:, :1], atol=1e-6), \
        "视频首帧（无历史）应与单帧图像逐位一致"
    # 逐帧查询只依赖历史: Q_0 与 F=1 冷启动相同; 改第 0 帧只影响 Q_1/Q_2,
    # 改第 1 帧不影响 Q_1（当前帧不进 KV）但影响 Q_2
    with torch.no_grad():
        z_cls_v, z_s_v = model.encode(x_v)                  # (B,F,1,D)/(B,F,K,D)
        q0 = model.middle(z_s_v[:, :1])[:, 0]              # 只有 [cls]
        q = model.middle(z_s_v)
        z_s_mod0 = z_s_v.clone()
        z_s_mod0[:, 0] += 0.1
        q_mod0 = model.middle(z_s_mod0)
        z_s_mod1 = z_s_v.clone()
        z_s_mod1[:, 1] += 0.1
        q_mod1 = model.middle(z_s_mod1)
    assert torch.allclose(q[:, 0], q0, atol=1e-6), \
        "首帧查询应只来自 [cls] 冷启动"
    assert torch.allclose(q_mod0[:, 0], q[:, 0], atol=1e-6), \
        "第 0 帧的 z_s 不应影响 Q_0（历史不含当前帧）"
    assert not torch.allclose(q_mod0[:, 1], q[:, 1]), \
        "第 0 帧 z_s 应进入 Q_1 的 KV"
    assert not torch.allclose(q_mod0[:, 2], q[:, 2]), \
        "第 0 帧 z_s 应进入 Q_2 的 KV"
    assert torch.allclose(q_mod1[:, 1], q[:, 1], atol=1e-6), \
        "第 1 帧查询不应依赖第 1 帧自己的 z_s（历史不含当前帧）"
    assert not torch.allclose(q_mod1[:, 2], q[:, 2]), \
        "第 1 帧 z_s 应进入 Q_2 的 KV"
    # 帧间 BPTT: 第 1 帧查询对第 0 帧 z_s 有梯度, 对当前/后帧无梯度
    z_s_req = z_s_v.detach().clone().requires_grad_(True)
    q_req = model.middle(z_s_req)
    q_req[:, 1].sum().backward()
    assert z_s_req.grad[:, 0].abs().sum() > 0, "帧间 carry 必须走 BPTT（不 detach）"
    assert z_s_req.grad[:, 1].abs().sum() == 0, \
        "第 1 帧查询不应读到当前帧 z_s（历史不含当前帧）"
    assert z_s_req.grad[:, 2].abs().sum() == 0, \
        "第 1 帧查询不应读到未来帧 z_s"
    # 整段视频反传: MiddleEncoder / 编码器 / 解码器 / 像素头都收梯度
    model.zero_grad(set_to_none=True)
    out_v2 = model(x_v)
    out_v2["loss"].backward()
    for name, p in [("middle.query_base", model.middle.query_base),
                    ("middle.mem_init", model.middle.mem_init),
                    ("middle.stack.layers[0].multihead_attn.in_proj_weight",
                     model.middle.stack.layers[0].multihead_attn.in_proj_weight),
                    ("dino.embeddings.patch_embeddings.weight",
                     model.dinov2.embeddings.patch_embeddings.weight),
                    ("decoder.stack.layers[0].self_attn.in_proj_weight",
                     model.decoder.stack.layers[0].self_attn.in_proj_weight),
                    ("pixel_head.net", model.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"视频路径 {name} 收不到梯度"
    print(f"[ok] 视频模态: 输入 (B={2},F={FV},3,56,56) → Y_pix{tuple(out_v['Y_pix'].shape)}, "
          f"F_hat{tuple(out_v['F_hat'].shape)}; KV 长度 {model.middle.last_memory_lens} "
          f"(K={K}, 首帧只有 [cls]); 图像=单帧视频逐位一致; 帧间 BPTT 梯度全通")

    print("\nALL CHECKS PASSED")
