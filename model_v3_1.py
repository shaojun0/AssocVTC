"""
model_v3_1 — MiddleEncoder 复用 Qwen3.5 GatedDeltaNet
=================================================================

动机（2026-11-12）:
    model_v3 的 MiddleEncoder 是
        Q_f = MiddleEncoder(Q_{f-1}, KV_f)          ← 非线性帧间递推
    v3_1 最初把它换成“固定 query + 纯线性累积”:
        S_f = S_{f-1} + Σ Φ(k) v^T
    纯累积等价于 γ=1、无遗忘、无 delta 修正的线性 attention: 历史等权、
    无序, 且要存 (B,F,H,d,d) 的全部前缀状态。

    本版不再手写 GDN: 直接复用 transformers 5.22 的完整 Qwen3.5 block
        Qwen3_5GatedDeltaNet
    它内部包含 causal conv1d、in_proj_qkv/z/b/a、A_log/dt_bias 门控、
    torch_chunk_gated_delta_rule、RMSNormGated、out_proj。MiddleEncoder
    只做三件事:
        1. 把 [cls] + 各帧 z_s 按帧喂给 Qwen3_5GatedDeltaNet;
        2. 在每帧边界取 recurrent state S, 用固定 query 读出;
        3. 加 FFN 残差, 可选地每 4 层插 1 层 softmax full-attention。
    GDN 的 q/k/v/g/beta/conv/state update 全是库里的, 不再维护自定义
    的 q/k/v/g/b 投影和 WY/UT。

MiddleEncoder（LinearMiddleEncoder）:
    第 1 层 query 固定: q = query_base (N,D)（所有帧共享）
    历史 token: mem = [cls] + z_s[0..F-1]（1+F·K 个 token）
    第 f 帧读出（S_f 只含 [cls] + 帧 < f, 不含当前帧）:
        q_f = in_proj_qkv(LN(query_f)) 的 q 分支, L2Norm
        o_f = norm(S_f^T q_f * scale, gate=in_proj_z(LN(query_f)))
        Q_f = out_proj(o_f) + query_f   （再 + FFN 残差）
    其中 S_f 是 Qwen3_5GatedDeltaNet 在帧 f 之前的 recurrent state,
    布局 (d_k,d_v); norm/out_proj/in_proj_z 都复用同一个 GDN block 的权重。
    · [cls] 冷启动: 先用库 block 处理 [cls], 得到 S_0; Q_0 与单帧图像一致。
    · depth>1 时, 第 l 层的 q_f = 第 l−1 层的 Q_f（逐帧不同）, 但每层
      仍从同一份 mem 重新累积状态; 没有 Q_{f-1} 反馈边。
    · 帧间串行 F 步, 帧内 K 个 token 由库 block 一次算完; 推理状态显存
      O(B·H·d²), 训练反向图 O(B·F·H·d²)（长视频可用 checkpointing 降）。
    · 可选 hybrid: full_attn_every=4 时每 4 层插入 1 层 softmax 历史
      注意力（3 GDN : 1 full, 对齐 Qwen3.5 的 3:1 混合）; 该层就是一层
      标准 `nn.TransformerDecoder`（自注意力 + 交叉注意力 + FFN), 只在
      memory_mask 上表达"帧 f 只读 [cls] + 帧 < f"; 其余层仍是 GDN block。

训练用 cache:
    transformers 的 DynamicCache 为了 cudagraph 用 in-place copy_ 更新
    conv/recurrent state, 跨帧反传会触发 autograd version error。库里给了
    一个官方开关 `activate_past_recording()`: 打开后 update_conv_state
    改成"赋值"而非 copy_（padding / 裁剪回 kernel_size 的语义仍由库负责,
    配一次 crop(0) 收敛状态长度）。但 update_recurrent_state 是**无条件**
    copy_、没有开关, 所以只重写这一个方法（赋值代替 copy）。_GDNCache 因此
    只剩"打开 record_past + 重写 update_recurrent_state", 不再手抄 conv
    状态的滚动窗口。
    注: record_past 顺带关掉了库里 seq_len==1 的 `causal_conv1d_update`
    就地更新分支（modeling_qwen3_5.py 里 `not cache_params.is_recording_past`）,
    否则 num_specials=1（每帧 1 个 token）时训练会报 inplace 错。

与 v3 的差异:
    · v3  : Q_f 依赖 Q_{f-1}; 非线性; 帧间串行。
    · v3_1: 第 1 层 query 固定 query_base; 历史经 Qwen3.5 GDN block 递推;
      帧内并行, 帧间 scan; 可选 3:1 hybrid。
    · 输出接口/损失/图像=单帧视频完全继承 SRPhase1V2; decode 把帧维折进
      batch（decoder 逐帧独立）。
    · 编码器 (DINO) 仍逐帧循环; 帧维并行只针对 decode。

依赖:
    transformers>=5.22.2（Qwen3.5 模块）; 无 flash-linear-attention /
    causal_conv1d 时库会自动回退到 PyTorch 参考实现, 只是慢一些。

用法:
    from model_v3_1 import SRPhase1V31
    model = SRPhase1V31(dinov2=dino, num_patches=256, dim=1248, heads=8,
                        num_specials=32, decoder_steps=square_block_starts(32),
                        middle_depth=4, middle_full_attn_every=4)
    out = model(pixel_values)       # 4D 或 5D; 输出始终带 F 维
    自检: python model_v3_1.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as _qwen35
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
# Qwen3.5 的完整 GDN block（需要 transformers>=5.22.2；无 kernel 时回退 PyTorch）
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet, torch_chunk_gated_delta_rule)

# l2norm 是 Qwen3.5 模块内的 helper（非公开 API, 小版本可能挪位）:
# 拿不到就退回 F.normalize(p=2), 语义等价（x / max(||x||, eps)）。
l2norm = getattr(_qwen35, "l2norm", None) or (
    lambda x, dim=-1, eps=1e-6: F.normalize(x, p=2.0, dim=dim, eps=eps))

from model_v3 import SRPhase1V2


def _ffn(dim: int, hidden: int, dropout: float) -> nn.Sequential:
    """前馈层（MiddleEncoder 各层共用）。

    想换成 Qwen3.5 自己的 gated SiLU MLP 时, 把这里换成
    `Qwen3_5MLP(config, hidden)` 即可（config.hidden_act 已是 silu）;
    默认保持 2 层 GELU, 以免改 checkpoint 的参数名与容量。
    """
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, dim))


class _GDNCache(DynamicCache):
    """DynamicCache + record_past, 让 Qwen3_5GatedDeltaNet 跨帧可反传。

    transformers 的 cache 为了 cudagraph 用 `copy_` 原地更新 state; 训练时
    上一帧的 state 还被 autograd 引用, 原地覆盖会报 "modified by an inplace
    operation"。

    · conv 状态: 库自带开关 `activate_past_recording()`, 打开后
      update_conv_state 改成赋值（padding / 裁回 kernel_size 全由库负责）。
      配一次 `crop(0)` 把状态长度收敛回 kernel_size, 否则它会随帧数增长。
    · recurrent 状态: 库里是**无条件** copy_、没有开关, 故只重写这一个
      方法（直接存新 tensor）。
    这是 cache adapter, 不是 GDN 重实现。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activate_past_recording()
        # 传 config 时 DynamicCache 会预建 layer 列表; 空列表说明走向了
        # 惰性创建, record_past 就不会生效, 这里直接报错而不是静默退化。
        assert len(self.layers) == 1, \
            f"_GDNCache 需要预建的 1 个 layer, got {len(self.layers)}"

    def update_recurrent_state(self, recurrent_states: Tensor, layer_idx: int,
                               state_idx: int = 0, **kwargs) -> Tensor:
        layer = self.layers[layer_idx]
        layer.recurrent_states[state_idx] = recurrent_states
        layer.is_recurrent_states_initialized[state_idx] = True
        layer.has_previous_state[state_idx] = True
        return recurrent_states


class LinearMiddleLayer(nn.Module):
    """MiddleEncoder 的 GDN adapter: Qwen3.5 block + 固定 query 读出 + FFN。

    GDN 的 conv1d / in_proj_qkv/z/b/a / A_log/dt_bias / chunk gated delta
    rule / RMSNormGated / out_proj 全部来自 transformers 的 Qwen3.5 block。
    这里只负责:
        · 按帧把 [cls] / z_s 喂给 block, 收集帧边界 state S;
        · 用 S 一次性读出所有帧的固定 query;
        · 加 FFN 残差。
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 decay_range: tuple = (5e-4, 5e-3)):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 不能被 heads={heads} 整除"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)                     # mem 与 query 共用
        self.config = Qwen3_5TextConfig(
            hidden_size=dim, num_hidden_layers=1,
            num_attention_heads=heads, num_key_value_heads=heads,
            head_dim=self.head_dim,
            linear_num_key_heads=heads, linear_num_value_heads=heads,
            linear_key_head_dim=self.head_dim,
            linear_value_head_dim=self.head_dim,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention"],
            rms_norm_eps=1e-6, hidden_act="silu")
        self.gdn = Qwen3_5GatedDeltaNet(self.config, layer_idx=0)
        # Qwen 默认 A∈[0.01,16] 衰减过猛（一个 token 就忘光）, 历史跨不了帧。
        # A 的含义是每 token 遗忘率: γ_tok = exp(-A·softplus(a+dt_bias)),
        # 初始 softplus(1)≈1.31 ⇒ A=1e-3 时半衰期 ≈ 0.69/(1.31e-3) ≈ 530 token,
        # K=32 时 ≈ 17 帧, 8 帧内 γ≈0.72 ⇒ 接近"历史等权累积"。
        # 想让 Δ 规则真正起作用, 取半衰期 ≈ 1~2 帧: A ≈ 0.7/(1.5·K)
        # （K=32 ⇒ ≈1.5e-2）, 即构造时传 decay_range=(7e-3, 3e-2)。
        assert len(decay_range) == 2 and 0 < decay_range[0] < decay_range[1]
        with torch.no_grad():
            self.gdn.A_log.copy_(torch.log(
                torch.empty(heads).uniform_(*decay_range)))
        self.norm_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = _ffn(dim, hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: Tensor, mem: Tensor, n_frames: int,
                k_per_frame: int) -> Tensor:
        """q: (B,F,N,D) → (B,F,N,D); mem: (B,1+F·K,D)。"""
        B, L, D = mem.shape
        K = k_per_frame
        assert L == 1 + n_frames * K, (L, n_frames, K)
        assert q.shape[:2] == (B, n_frames), q.shape
        N, H, hd = q.shape[2], self.heads, self.head_dim

        # [cls], 帧 0, ..., 帧 F-2: 第 f 个 state 恰好是帧 f 读到的历史
        cache = _GDNCache(config=self.config)
        mem_n = self.norm(mem)
        states = []
        for chunk in torch.split(mem_n, [1] + [K] * n_frames, dim=1)[:-1]:
            self.gdn(chunk, cache_params=cache)
            states.append(cache.get_recurrent_state(0, 0))
            # record_past 下 conv 状态会累积全部历史, 读完整回 kernel_size
            # （只是切片, 不改数值也不改梯度）
            cache.crop(0)
        s = torch.stack(states, dim=1)                     # (B,F,H,d,d)
        # 诊断: crop(0) 后 conv 状态长度应恒为 kernel_size（否则 record_past
        # 下它随帧数增长, 每帧 conv 的算量也线性涨）
        self.last_conv_state_len = cache.layers[0].conv_states[0].shape[-1]

        # 固定 query 独立读出, 一次算完所有帧
        qn = self.norm(q)
        qp = l2norm(self.gdn.in_proj_qkv(qn)[..., :D].view(
            B, n_frames, N, H, hd), dim=-1, eps=1e-6)
        # 读出走 fp32: state 是 (d×d) 的累积量、值域远大于激活, bf16 累加
        # 误差会直接进 query; 这一步是整个 MiddleEncoder 唯一的跨帧大 matmul。
        o = torch.einsum("bfnhd,bfhdv->bfnhv",
                         qp.float(), s.float()) * self.scale
        gate = self.gdn.in_proj_z(qn).view(B, n_frames, N, H, hd)
        o = self.gdn.norm(o.reshape(-1, hd), gate.reshape(-1, hd))
        o = self.gdn.out_proj(o.view(B, n_frames, N, D)).to(q.dtype)

        out = q + self.drop(o)
        return out + self.ff(self.norm_ff(out))


class SoftmaxMiddleLayer(nn.Module):
    """hybrid 的 full-attention 层 = 一层标准 `nn.TransformerDecoder`。

    结构完全交给库（与 model_v3.MiddleEncoder 同构）:
        预归一化 TransformerDecoderLayer（query 自注意力 + 交叉注意力
        (memory = 历史) + FFN + 残差, gelu）+ 末层 LayerNorm。
    这里只负责把"每帧看到的历史前缀不同"翻译成一个 memory_mask。

    **memory_mask（唯一需要自己算的东西）**: 帧 f 的 query 只可见
    [cls] + 帧 < f, 即列 0..f·K（与 GDN 递推的"历史不含当前帧"一致）。
    因为每帧掩码不同, 把帧维折进 batch:
        tgt     = q.reshape(B·F, N, D)
        memory  = mem 展开成 (B·F, L, D)
        mask    = (B·F·H, N, L) bool, True = 屏蔽
    `_mha_block` 里是 `need_weights=False`, 所以这个 3D bool 掩码直接走
    SDPA, 不会物化 (B·F,H,N,L) 的注意力权重。

    与手写版的差别（2026-11-13 改）:
        · 多了 query 帧内自注意力（TransformerDecoderLayer 自带）——
          N 行 query 在同一帧内交换信息, 与 model_v3.MiddleEncoder 一致;
          手写版是逐 query 独立读出, 与 LinearMiddleLayer 一致。
        · 代价: memory 要按帧展开成 (B·F,L,D), K/V 投影也算 F 次
          （MHA 内部做, 不像手写版能"投影一次 + 广播"）。L=1+F·K 很小,
          且这层只占 hybrid 的 1/4, 换来的是一整层都是库代码。
    参数名随之变成 `stack.layers.0.{self_attn,multihead_attn,linear1,...}`,
    旧 hybrid checkpoint 不能 strict load（该配置默认未启用, 影响面=0）。
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 不能被 heads={heads} 整除"
        self.dim = dim
        self.heads = heads
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True),
            num_layers=1,
            norm=nn.LayerNorm(dim),
        )

    def forward(self, q: Tensor, mem: Tensor, n_frames: int,
                k_per_frame: int) -> Tensor:
        """q: (B,F,N,D) → (B,F,N,D); mem: (B,1+F·K,D)。"""
        B, L, D = mem.shape
        assert L == 1 + n_frames * k_per_frame, (L, n_frames, k_per_frame)
        assert q.shape[:2] == (B, n_frames), q.shape
        N, H = q.shape[2], self.heads
        BF = B * n_frames
        # 帧维折进 batch（decoder 逐帧独立, 只有掩码不同）
        tgt = q.reshape(BF, N, D)
        mem_rep = mem.unsqueeze(1).expand(
            B, n_frames, L, D).reshape(BF, L, D)
        # 帧 f 可见列 = [0 .. f·K]; (F,L) → (B·F·H, N, L), True = 屏蔽
        idx = torch.arange(L, device=mem.device)
        limits = torch.arange(n_frames, device=mem.device) * k_per_frame
        visible = idx[None] <= limits[:, None]                     # (F,L)
        mask = visible.view(1, n_frames, 1, 1, L).expand(
            B, -1, H, N, -1).reshape(BF * H, N, L)
        out = self.stack(tgt, mem_rep, memory_mask=~mask)          # (BF,N,D)
        return out.reshape(B, n_frames, N, D)


class LinearMiddleEncoder(nn.Module):
    """v3_1 的 MiddleEncoder: 固定 query + Qwen3.5 GDN block 递推。

    full_attn_every=4 时每 4 层插入 1 层 softmax 历史注意力（3:1 hybrid）,
    该层是一层标准 `nn.TransformerDecoder`（见 SoftmaxMiddleLayer）。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 heads: int = 8, depth: int = 1, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, full_attn_every: int = 0,
                 linear_decay_range: tuple = (5e-4, 5e-3)):
        super().__init__()
        assert depth >= 1, f"depth 须 ≥1, got {depth}"
        assert full_attn_every >= 0, full_attn_every
        self.dim = dim
        self.num_patches = num_patches
        self.heads = heads
        self.depth = depth
        self.full_attn_every = full_attn_every
        # 第 1 层固定 query（行 k↔patch k）: 所有帧共享, 不再随帧递推
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)
        # KV 的常驻初始化向量 [cls]（无历史时状态只由它产生）
        self.mem_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layers = []
        for i in range(depth):
            use_full = full_attn_every > 0 and (i + 1) % full_attn_every == 0
            if use_full:
                layers.append(SoftmaxMiddleLayer(
                    dim=dim, heads=heads, mlp_ratio=mlp_ratio,
                    dropout=dropout))
            else:
                layers.append(LinearMiddleLayer(
                    dim=dim, heads=heads, mlp_ratio=mlp_ratio,
                    dropout=dropout, decay_range=linear_decay_range))
        self.layers = nn.ModuleList(layers)
        self.norm_out = nn.LayerNorm(dim)
        # 最近一次 forward 的诊断量
        self.last_memory_lens = None
        self.last_queries = None

    def forward(self, z_s: Tensor) -> Tensor:
        """z_s: (B,F,K,D) → Q: (B,F,N,D)。"""
        assert z_s.dim() == 4, \
            f"z_s 须为 (B,F,K,D), got {tuple(z_s.shape)}"
        B, n_frames, K, D = z_s.shape
        assert D == self.dim, f"z_s 维度 {D} != dim={self.dim}"
        assert n_frames >= 1 and K >= 1, (n_frames, K)
        # 历史 token: [cls] + 所有帧 z_s（当前帧不进入自己的状态, 见 forward）
        mem = torch.cat([self.mem_init.expand(B, -1, -1),
                         z_s.reshape(B, n_frames * K, D)], dim=1)
        # (B,F,N,D): 所有帧共享同一个 query_base, expand 不复制
        q = self.query_base.unsqueeze(0).expand(B, n_frames, -1, -1)
        for layer in self.layers:
            q = layer(q, mem, n_frames, K)
        Q = self.norm_out(q)                                 # (B,F,N,D)
        self.last_memory_lens = [1 + K * f for f in range(n_frames)]
        self.last_queries = Q
        return Q


class SRPhase1V31(SRPhase1V2):
    """SRPhase1V2 + v3_1 的 GDN MiddleEncoder；其余接口/损失完全继承。

    GDN block 直接复用 transformers 的 `Qwen3_5GatedDeltaNet`
    （conv1d / in_proj_* / chunk GDN / RMSNormGated / out_proj）；
    无 flash-linear-attention / causal_conv1d 时自动回退 PyTorch。

    与 v3 / 旧 v3_1 的 checkpoint 不同：`middle.layers.*` 现在是
    `gdn.*`（in_proj_* / conv1d / norm / out_proj / A_log）+ `ff`，
    不能 strict load 旧权重; `query_base`/`mem_init` 的 shape 不变。
    """

    def __init__(self, *args, middle_depth: int = 1,
                 middle_full_attn_every: int = 0,
                 middle_decay_range: tuple = (5e-4, 5e-3), **kwargs):
        super().__init__(*args, middle_depth=middle_depth, **kwargs)
        # 父类把 heads / mlp_ratio / decoder_dropout 留档在 self 上（2026-11-13）:
        # 这里直接读, 不再去嗅探 nn.TransformerDecoderLayer 的内部属性
        # （linear1.out_features / dropout1.p —— PyTorch 改个属性名就静默出错）。
        # 这样即使调用方按位置传参, 拿到的也是 decoder 实际生效的值。
        # 替换掉父类构造的 v3 循环 MiddleEncoder
        self.middle = LinearMiddleEncoder(
            dim=self.dim, num_patches=self.num_patches, heads=self.heads,
            depth=middle_depth, mlp_ratio=self.mlp_ratio,
            dropout=self.decoder_dropout,
            full_attn_every=middle_full_attn_every,
            linear_decay_range=middle_decay_range)

    def decode(self, z_cls: Tensor, z_s: Tensor, pixel_values: Tensor) -> dict:
        """v3_1 的 decode: decoder 把帧维折进 batch。

        MiddleEncoder 内部是 GDN 的 F 步 scan; decoder/PixelHead 逐帧独立,
        所以帧维仍可折成 (B·F) 一次算完。输出 key/shape 与
        SRPhase1V2.decode 完全一致。
        """
        assert pixel_values.dim() == 5, \
            f"decode 须为 (B,F,3,H,W), got {tuple(pixel_values.shape)}"
        assert z_cls.dim() == 4 and z_s.dim() == 4, \
            f"z_cls/z_s 须为 (B,F,·,D), got {tuple(z_cls.shape)}/{tuple(z_s.shape)}"
        B, n_frames = pixel_values.shape[:2]
        D = z_s.shape[-1]
        assert z_cls.shape == (B, n_frames, 1, D), z_cls.shape
        assert z_s.shape == (B, n_frames, self.num_specials, D), z_s.shape
        queries = self.middle(z_s)                           # (B,F,N,D)
        # 帧维折进 batch: decoder/PixelHead 都是逐帧独立的
        zc = z_cls.reshape(B * n_frames, 1, D)
        zs = z_s.reshape(B * n_frames, self.num_specials, D)
        q = queries.reshape(B * n_frames, self.num_patches, D)
        Y = self.decoder(zc, zs, q)                          # (B·F,|T|,N,D)
        Y_pix = self.pixel_head(Y)                           # (B·F,|T|,N,588)
        target = self._pixel_target(
            pixel_values.reshape(B * n_frames, *pixel_values.shape[2:]))
        per_step = F.l1_loss(
            Y_pix, target.unsqueeze(1).expand_as(Y_pix),
            reduction="none").mean(dim=(0, 2, 3))            # (|T|,)
        T_steps, N, patch_px = Y_pix.shape[1], Y_pix.shape[2], Y_pix.shape[3]
        Y_pix = Y_pix.reshape(B, n_frames, T_steps, N, patch_px)
        target = target.reshape(B, n_frames, N, patch_px)
        # 保持 v3 的诊断语义: decoder.last_Y / last_query = 最后一帧
        Y_frames = Y.reshape(B, n_frames, T_steps, N, D)
        self.decoder.last_Y = Y_frames[:, -1]
        self.decoder.last_query = q.reshape(B, n_frames, N, D)[:, -1]
        F_pix = Y_pix[:, :, -1]                              # (B,F,N,588)
        recon = F.l1_loss(F_pix, target)
        loss = per_step.mean()                               # 帧 × 步平权
        return {"loss": loss, "recon": recon, "F_hat": F_pix,
                "Y_pix": Y_pix, "target_pix": target}


# 兼容旧命名：v3_1 的 MiddleEncoder 就是 GDN 版
MiddleEncoder = LinearMiddleEncoder


# ═══ 自检（python model_v3_1.py）═══

if __name__ == "__main__":
    torch.manual_seed(0)

    from transformers import Dinov2Config, Dinov2Model

    N, D = 16, 64
    PATCH_PX = 14 * 14 * 3
    cfg = Dinov2Config(hidden_size=D, num_hidden_layers=2,
                       num_attention_heads=4, intermediate_size=4 * D,
                       image_size=56, patch_size=14)
    dino = Dinov2Model(cfg)
    model = SRPhase1V31(dino, num_patches=N, dim=D, heads=4,
                        num_specials=8, decoder_steps=[1, 4],
                        middle_depth=2)
    model.eval()
    assert isinstance(model.middle, LinearMiddleEncoder)
    assert len(model.middle.layers) == 2
    assert not hasattr(model.middle, "stack"), \
        "v3_1 应替换掉 v3 的循环 MiddleEncoder"
    assert model.middle.query_base.shape == (N, D)
    assert model.middle.mem_init.shape == (1, 1, D)
    assert hasattr(model.middle.layers[0], "gdn")
    assert hasattr(model.middle.layers[0].gdn, "in_proj_qkv")
    assert hasattr(model.middle.layers[0].gdn, "conv1d")

    x = torch.randn(2, 3, 56, 56)
    x_v = torch.randn(2, 3, 3, 56, 56)                     # (B,F,3,H,W)

    with torch.no_grad():
        out = model(x)                                     # 4D 图像 → F=1
        out_v = model(x_v)
        zc_v, zs_v = model.encode(x_v)
        q = model.middle(zs_v)
    assert out["F_hat"].shape == (2, 1, N, PATCH_PX), out["F_hat"].shape
    assert out_v["F_hat"].shape == (2, 3, N, PATCH_PX), out_v["F_hat"].shape
    assert out_v["Y_pix"].shape == (2, 3, 2, N, PATCH_PX), out_v["Y_pix"].shape
    assert q.shape == (2, 3, N, D), q.shape
    assert torch.isfinite(q).all(), "GDN 状态/读出出现非有限值"
    K = model.num_specials
    assert model.middle.last_memory_lens == [1, 1 + K, 1 + 2 * K], \
        model.middle.last_memory_lens

    # 4D 图像与 F=1 视频逐位一致；3 帧视频首帧也应与单帧一致
    with torch.no_grad():
        out_i = model(x_v[:, 0])
        out_v1 = model(x_v[:, :1])
    assert torch.allclose(out_i["F_hat"], out_v1["F_hat"], atol=1e-6)
    assert torch.allclose(out_i["F_hat"], out_v["F_hat"][:, :1], atol=1e-6), \
        "首帧无历史, 应与单帧图像一致"

    # 历史依赖 / 当前帧不进自己的 KV（与 v3 语义一致）
    with torch.no_grad():
        q0 = model.middle(zs_v[:, :1])[:, 0]
        zs0 = zs_v.clone()
        zs0[:, 0] += 0.1
        qm0 = model.middle(zs0)
        zs1 = zs_v.clone()
        zs1[:, 1] += 0.1
        qm1 = model.middle(zs1)
    assert torch.allclose(q[:, 0], q0, atol=1e-6), "Q_0 应只由 [cls] 冷启动"
    assert torch.allclose(qm0[:, 0], q[:, 0], atol=1e-6)
    assert not torch.allclose(qm0[:, 1], q[:, 1]), "帧 0 应进入 Q_1 的历史"
    assert not torch.allclose(qm0[:, 2], q[:, 2]), "帧 0 应进入 Q_2 的历史"
    assert torch.allclose(qm1[:, 1], q[:, 1], atol=1e-6), \
        "Q_1 不应依赖帧 1 自己（历史不含当前帧）"
    assert not torch.allclose(qm1[:, 2], q[:, 2]), "帧 1 应进入 Q_2 的历史"

    # GDN 递推的帧间梯度: Q_1 对帧 0 有梯度, 对当前/未来帧为 0
    zr = zs_v.detach().clone().requires_grad_(True)
    qr = model.middle(zr)
    qr[:, 1].sum().backward()
    assert zr.grad[:, 0].abs().sum() > 0, "历史帧必须回传梯度"
    assert zr.grad[:, 1].abs().sum() == 0, \
        "Q_1 不应读到当前帧 z_s（历史不含当前帧）"
    assert zr.grad[:, 2].abs().sum() == 0, \
        "Q_1 不应读到未来帧 z_s"

    # 库函数 ↔ 逐 token gated delta rule（数值回归）
    lt = 5
    kt = torch.randn(2, lt, 4, 8)
    vt = torch.randn(2, lt, 4, 8)
    gt = -torch.rand(2, lt, 4)
    bt = torch.rand(2, lt, 4)
    _, s_lib = torch_chunk_gated_delta_rule(
        kt, kt, vt, gt, bt, chunk_size=lt, initial_state=None,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    kn = l2norm(kt, dim=-1, eps=1e-6)
    s_ref = torch.zeros(2, 4, 8, 8)
    for t in range(lt):
        s_ref = s_ref * gt[:, t].exp()[..., None, None]
        kv = (s_ref * kn[:, t, :, :, None]).sum(dim=-2)
        delta = (vt[:, t] - kv) * bt[:, t, :, None]
        s_ref = s_ref + kn[:, t, :, :, None] * delta[..., None, :]
    assert torch.allclose(s_lib, s_ref, atol=1e-5), \
        "库函数与逐 token gated delta rule 不一致"

    # 分帧 carry ≡ 整段一次调用（MiddleEncoder 帧循环的正确性）
    lc, kc = 8, 4
    qc = torch.randn(2, lc, 4, 8)
    kc_ = torch.randn(2, lc, 4, 8)
    vc = torch.randn(2, lc, 4, 8)
    gc = -torch.rand(2, lc, 4)
    bc = torch.rand(2, lc, 4)
    _, s_full = torch_chunk_gated_delta_rule(
        qc, kc_, vc, gc, bc, chunk_size=kc, initial_state=None,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    s_carry = torch.zeros(2, 4, 8, 8)
    for f in range(lc // kc):
        _, s_carry = torch_chunk_gated_delta_rule(
            qc[:, f * kc:(f + 1) * kc], kc_[:, f * kc:(f + 1) * kc],
            vc[:, f * kc:(f + 1) * kc], gc[:, f * kc:(f + 1) * kc],
            bc[:, f * kc:(f + 1) * kc], chunk_size=kc,
            initial_state=s_carry, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
    assert torch.allclose(s_carry, s_full, atol=1e-6), \
        "分帧 carry 与整段一次调用的状态不一致"

    # 并行 decode 与父类逐帧 decode 数值等价（帧维折 batch 的回归测试）
    with torch.no_grad():
        out_vec = model.decode(zc_v, zs_v, x_v)
        last_vec = model.decoder.last_Y.clone()
        out_loop = SRPhase1V2.decode(model, zc_v, zs_v, x_v)
    for key in ("F_hat", "Y_pix", "target_pix"):
        assert torch.allclose(out_vec[key], out_loop[key], atol=1e-6), key
    assert torch.isclose(out_vec["loss"], out_loop["loss"], atol=1e-6)
    assert torch.isclose(out_vec["recon"], out_loop["recon"], atol=1e-6)
    assert torch.allclose(last_vec, model.decoder.last_Y, atol=1e-6), \
        "并行 decode 的 decoder.last_Y 应与逐帧 decode 的最后一帧一致"
    assert model.decoder.last_Y.shape == (2, 2, N, D), model.decoder.last_Y.shape
    assert model.decoder.last_query.shape == (2, N, D), model.decoder.last_query.shape

    # 整段视频反传: GDN 门控/固定 query/编码器/解码器/像素头都收梯度
    model.zero_grad(set_to_none=True)
    out_b = model(x_v)
    out_b["loss"].backward()
    layer = model.middle.layers[0]
    for name, p in [
        ("middle.query_base", model.middle.query_base),
        ("middle.mem_init", model.middle.mem_init),
        ("middle.layers[0].gdn.in_proj_qkv.weight",
         layer.gdn.in_proj_qkv.weight),
        ("middle.layers[0].gdn.in_proj_z.weight",
         layer.gdn.in_proj_z.weight),
        ("middle.layers[0].gdn.in_proj_b.weight",
         layer.gdn.in_proj_b.weight),
        ("middle.layers[0].gdn.in_proj_a.weight",
         layer.gdn.in_proj_a.weight),
        ("middle.layers[0].gdn.conv1d.weight", layer.gdn.conv1d.weight),
        ("middle.layers[0].gdn.norm.weight", layer.gdn.norm.weight),
        ("middle.layers[0].gdn.out_proj.weight", layer.gdn.out_proj.weight),
        ("middle.layers[0].gdn.A_log", layer.gdn.A_log),
        ("middle.layers[1].gdn.in_proj_qkv.weight",
         model.middle.layers[1].gdn.in_proj_qkv.weight),
        ("dino.layernorm.weight", model.dinov2.layernorm.weight),
        ("decoder.stack.layers[0].self_attn.in_proj_weight",
         model.decoder.stack.layers[0].self_attn.in_proj_weight),
        ("pixel_head.net", model.pixel_head.net[0].weight),
    ]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"v3_1 路径 {name} 收不到梯度"

    # 加宽 stack_dim 路径: MiddleEncoder 仍在 dim, decoder 内部做 dim→2D→dim
    m_wide = SRPhase1V31(Dinov2Model(cfg), num_patches=N, dim=D, heads=4,
                         num_specials=8, decoder_steps=[1, 4],
                         stack_dim=2 * D, decoder_dropout=0.05)
    out_w = m_wide(x_v)
    assert out_w["F_hat"].shape == (2, 3, N, PATCH_PX), out_w["F_hat"].shape
    assert len(m_wide.middle.layers) == 1

    # 3:1 hybrid: 3 层 GDN + 1 层 softmax
    m_hybrid = SRPhase1V31(Dinov2Model(cfg), num_patches=N, dim=D, heads=4,
                           num_specials=8, decoder_steps=[1, 4],
                           middle_depth=4, middle_full_attn_every=4)
    m_hybrid.eval()
    assert isinstance(m_hybrid.middle.layers[0], LinearMiddleLayer)
    assert isinstance(m_hybrid.middle.layers[3], SoftmaxMiddleLayer)
    out_h = m_hybrid(x_v)
    assert out_h["F_hat"].shape == (2, 3, N, PATCH_PX), out_h["F_hat"].shape
    with torch.no_grad():
        zh = m_hybrid.middle(zs_v)
        zs0_h = zs_v.clone()
        zs0_h[:, 0] += 0.1
        zh0 = m_hybrid.middle(zs0_h)
        zs1_h = zs_v.clone()
        zs1_h[:, 1] += 0.1
        zh1 = m_hybrid.middle(zs1_h)
    assert torch.allclose(zh0[:, 0], zh[:, 0], atol=1e-6), \
        "hybrid 的 Q_0 应只由 [cls] 冷启动"
    assert not torch.allclose(zh0[:, 1], zh[:, 1]), \
        "hybrid 的 Q_1 应读到帧 0"
    assert torch.allclose(zh1[:, 1], zh[:, 1], atol=1e-6), \
        "hybrid 的 Q_1 不应读到帧 1 自己"
    assert not torch.allclose(zh1[:, 2], zh[:, 2]), \
        "hybrid 的 Q_2 应读到帧 1"
    # hybrid 也要能反传（softmax 层 + GDN 层）
    m_hybrid.zero_grad(set_to_none=True)
    out_h["loss"].backward()
    assert m_hybrid.middle.layers[3].stack.layers[0] \
        .multihead_attn.in_proj_weight.grad is not None
    assert m_hybrid.middle.layers[3].stack.layers[0] \
        .self_attn.in_proj_weight.grad is not None
    assert m_hybrid.middle.layers[0].gdn.in_proj_qkv.weight.grad is not None

    # K=1 回归（2026-11-13 修）: 每帧只有 1 个 token 时, 库的 GDN forward
    # 会走 `causal_conv1d_update` 那条**就地**写 conv state 的融合分支
    # （modeling_qwen3_5.py: `not cache_params.is_recording_past`）,
    # 绕过我们的 update_conv_state ⇒ 第 2 帧反传必报 inplace 错。
    # _GDNCache 开 record_past 后 is_recording_past() 为 True, 不再进该分支。
    m_k1 = SRPhase1V31(Dinov2Model(cfg), num_patches=N, dim=D, heads=4,
                       num_specials=1, decoder_steps=[1], middle_depth=1)
    m_k1.train()
    out_k1 = m_k1(x_v)
    out_k1["loss"].backward()
    assert out_k1["F_hat"].shape == (2, 3, N, PATCH_PX), out_k1["F_hat"].shape
    assert m_k1.middle.layers[0].gdn.in_proj_qkv.weight.grad is not None

    # conv 状态长度必须被 crop(0) 约束在 kernel_size（否则 record_past 下
    # 它随帧数线性增长, 每帧 conv 的算量也线性涨）
    k_conv = model.middle.layers[0].gdn.conv1d.kernel_size[0]
    model.middle(zs_v)
    assert model.middle.layers[0].last_conv_state_len == k_conv, \
        model.middle.layers[0].last_conv_state_len

    # decay_range 可配（A 的含义: 每 token 遗忘率, 半衰期 ≈ 0.69/(A·1.31)）
    m_dec = SRPhase1V31(Dinov2Model(cfg), num_patches=N, dim=D, heads=4,
                        num_specials=8, decoder_steps=[1, 4],
                        middle_decay_range=(7e-3, 3e-2))
    a = m_dec.middle.layers[0].gdn.A_log.exp()
    assert (a.min() >= 7e-3 - 1e-6) and (a.max() <= 3e-2 + 1e-6), (a.min(), a.max())

    print(f"[ok] v3_1 GDN MiddleEncoder: Q{tuple(q.shape)} "
          f"(固定 query + Qwen3.5 GDN block), KV 长度 "
          f"{model.middle.last_memory_lens} (K={K}); "
          f"图像=单帧视频逐位一致; 并行 decode ≡ 逐帧 decode; 梯度全通; "
          f"3:1 hybrid 可跑; K=1 可训; conv state 长度受 crop(0) 约束; "
          f"decay_range 可配")
    print("\nALL CHECKS PASSED")
