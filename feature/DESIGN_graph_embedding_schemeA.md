# DESIGN 方案A：GNN + 置换不变 Readout 的图级向量表示（Graph Embedding）

> 状态：设计提案（通用方法，未在仓库落地）。拟放入 `feature/` 作为"以最少无序向量保存图信息"的首选方案。
> 需求来源：图信息保存要求 —— **可有损**、**无序**（置换不变，同构图必须得到相同表示）、**向量化**（定长向量）、**最小化向量数量**（每图 1 个向量为理论最小）。
> 对应文档：方案B（谱指纹）/ 方案C（哈希/随机投影）见 §9 对比，另文详述。

---

## 1. 目标与需求约束

| 约束 | 含义 | 方案A 如何满足 |
|---|---|---|
| 无序 | 对任意节点编号重排 σ，有 $f(G) = f(G^\sigma)$ | 图神经网络节点级置换**等变** + Readout 置换**不变** ⇒ 组合不变（§4） |
| 向量化 | 输出为固定维度 $z \in \mathbb{R}^d$，可进向量库/ANN | 不变 Readout 聚合出恰好 1 个向量 |
| 最小向量数量 | 每个图只对应极少向量，最优 = 1 | 方案A 默认输出 1 个向量；"k 向量"仅在需要多原型时才引入（§6 注） |
| 有损 | 允许不同图映射到同一向量（碰撞） | 信息论必然，见 §2；表达力上界见 §5 |

输入：图 $G = (V, E, X)$，$|V|=n$，可选节点特征 $X \in \mathbb{R}^{n \times d_{in}}$（无特征则退化为结构学习，如用度/邻域 one-hot 编码初始化）。

## 2. 为什么"每图 1 个向量"就是最小（信息论视角）

- **无损**保存一个 n 节点图至少需要 $\Omega(n^2)$ 比特（邻接矩阵本身），任何固定小向量都做不到 —— "允许有损"是必须的。
- **有损**且只需在 k 类图之间做区分（检索/去重/分类）时，理论下限 $\log_2 k$ 比特：一个 d 维 float 向量编码约 $2^{32d}$ 种模式，d=16~512 对工程完全够用。
- 因此：**向量数量取 1 是最小**，维度 d 才是唯一可调的精度旋钮。不要用"多个向量"换精度 —— 那等价于增大维度但破坏了"最小数量"约束。

## 3. 方案A 架构（形式化）

$$
z = \underbrace{\text{MLP}_\phi}_{\text{Projector (可选)}} \circ \underbrace{\text{READOUT}}_{\text{置换不变聚合}} \big( \{ h_v : v \in V \} \big),
\qquad h = \underbrace{\text{GNN}_\theta(X, E)}_{\text{L 层消息传递}}
$$

三个模块：

1. **Encoder：GNN**（GCN / GraphSAGE / GAT / GIN，L 层）。逐层：
   $$h_v^{(l+1)} = \sigma\Big( \text{AGG}\big( \{ W h_u^{(l)} : u \in \mathcal{N}(v) \cup \{v\} \} \big) \Big)$$
   作用是**把结构信息揉进每个节点向量**（1 层看 1 跳邻域，L 层看 L 跳）。
2. **Readout：置换不变聚合** → 单个向量 $\bar h = \text{READOUT}(\{h_v\})$（sum / mean / max / attention，见 §6）。
3. **Projector：MLP**（可选）。把聚合向量映射到目标维度/度量空间，如对比学习中的余弦空间。

> 关键设计原则：**所有信息必须先进入"节点级表示"再聚合**。直接对节点属性做 sum 会丢掉结构；直接存邻接矩阵则依赖编号（违反无序）。

## 4. 置换不变性论证（为什么"无序"成立）

分两步：

1. **GNN 节点级置换等变**：消息传递只依赖邻居集合 $\mathcal{N}(v)$（集合运算，无编号概念），故输入打乱编号 $G^\sigma$ 时，输出表示也按同样方式打乱：
   $$h^\sigma_{\sigma(v)} = h_v \quad\Rightarrow\quad \{h^\sigma_v\} = \{h_v\} \text{（作为多重集）}$$
2. **READOUT 置换不变**：sum/mean/max 对输入顺序不敏感。

组合即得：$z(G^\sigma) = z(G)$。**这是结构性保证，不是学出来的** —— 只要不引入任何依赖节点顺序的算子（如按行拼接、位置编码），无序性恒成立，训练和推理都不会破坏它。

理论支撑（DeepSets, Zaheer et al. 2017）：定义在集合上的连续函数可被 $\phi(\sum_{x \in S} \psi(x))$ 形式任意逼近 —— sum 池化 + 足够强的 encoder（GNN）足以表达任意图级函数；且 **sum 保留基数（图大小）信息**，比 mean 信息量更完整。

## 5. 表达力与有损性分析

- **上界**：1 层 GNN + sum 池化的区分能力不超过 **1-WL 测试**（Xu et al. 2019）。需要更强区分时：用 GIN（可达 1-WL 上界）、加深层数（感受野变大）、或在 Readout 前拼接多尺度表示（各层 $h_v^{(l)}$ 一起聚合）。
- **有损不可避免**：存在非同构但表示相同的图（碰撞）。典型如正则图族 —— 只要 Readout 是"全局摘要"，局部排列细节必然丢失。若应用允许，碰撞率由维度 d 与图分布共同决定（实践中 d ≥ 128 对中规模图已足够低）。
- **无节点特征的结构图**：GNN 在随机/度初始化特征上仍能学结构（类似"匿名游走"的隐式实现），但收敛更慢，此时谱指纹（方案B）可能是更便宜的替代（§9）。

## 6. Readout 选型

| Readout | 公式 | 优点 | 缺点 |
|---|---|---|---|
| **sum**（推荐默认） | $\bar h = \sum_v h_v$ | 信息完整、保大小、严格不变 | 数值随 n 增长，聚合后建议 LayerNorm |
| mean | $\frac{1}{n}\sum_v h_v$ | 尺度归一 | 丢图大小信息 |
| max | $\max_v h_v$（逐维） | 强调显著节点 | 信息利用率低 |
| attention/gated | $\bar h = \sum_v \alpha_v h_v$，$\alpha = \text{softmax}(a(h_v))$ | 可学习权重 | 多一组参数；α 本身置换不变（对每个 v 独立打分） |

> **关于"最小向量数量"的注**：方案A 默认输出 **1 个向量**。若下游明确要求"k 个向量"（例如与仓库 K 个 special token 对齐），可用 attention Readout 输出 k 个原型（$\bar h_k = \sum_v \alpha_{v,k} h_v$，对每个原型一套打分权重，仍置换不变），或用层次池化（DiffPool）。但除非有强理由，先保持 1 向量 + 高维度。

## 7. 训练范式

- **有监督**（图分类/回归）：$z$ 接线性头 + cross-entropy/MSE。标注少时该范式最直接。
- **无监督/自监督**（无标签图库首选）：
  - **图对比学习**（GraphCL / InfoGraph）：同一图两次增强（删边/扰动特征/子图采样）为正对，批内其他图为负对，NT-Xent 损失。产出可用于检索/聚类/下游微调。
  - 辅助目标：边重建（Graph Autoencoder）等，帮助 Encoder 保留结构。
- **技巧**：聚合后 LayerNorm；对比头输出归一化到单位球（余弦度量）；节点特征先归一化。

## 8. 复杂度与可扩展性

- 每层时间 $O(|E| \cdot d + n \cdot d^2)$，空间 $O(n \cdot d)$。图规模中等（$n \lesssim 10^6$、稀疏）时 CPU/单卡 GPU 均可。
- 超大图：GraphSAGE 式邻居采样 / ClusterGCN 分块 —— Readout 不变性不受影响（仍然对"参与聚合的节点子集"做不变聚合）。

## 9. 三种方案对比与选型建议

| | 方案A：GNN + Readout | 方案B：谱指纹 | 方案C：哈希/随机投影 |
|---|---|---|---|
| 无序保证 | 结构性（严格） | 特征值多重集天然无序 | 结构性（聚合在多重集上） |
| 需要训练 | 是 | 否 | 否 |
| 结构+节点特征 | 都可处理 | 仅结构 | 仅结构（或需先编码特征） |
| 区分度 | **最高**（可学习） | 中（同谱图碰撞） | 低-中（碰撞可控） |
| 成本 | 训练成本高 | O(n³) 谱分解（可截断） | O(\|E\|)，最快 |
| 适用 | 质量优先、有训练资源 | 纯结构快速基线 | 海量图粗筛/去重 |

**建议**：质量优先且能训练 → **方案A（本方案）**；无训练资源的结构图粗筛 → 方案B 作免费基线；海量去重/召回 → 方案C 做一级粗筛再 A/B 精排。

## 10. 与 AssocVTC 的潜在关联（方向性探讨）

仓库主线 = 把图像信息**压缩进少量 special token**（联想能力 → Phase 2 NLP）。方案A 在方法论上与这条线同构，可作为 feature 探索：

1. **无序压缩的语义对齐**：register token 式的 $z_s$ 目前依赖序列顺序（specials 拼接 + 位置编码）。若场景/结构表示需要**对顺序鲁棒**（patch 打乱、对象无规范次序），方案A 的"置换不变 Readout"给出结构性的保证 —— 表示只取决于"内容多重集 + 结构"，不取决于摆放顺序。
2. **图→token 的桥**：若把图像内容组织为图（超像素/对象/横纵亲和图，见前序讨论），方案A 提供"整图 → 1 向量"或"多原型 → k 向量"的规范通道，k 向量版与仓库 K 个 special token 的数量对齐。
3. **后续实验建议**（如采纳）：
   - E1：DINOv2 patch 特征建 kNN 图 → 方案A 压出 $z$，作为额外全局 token 注入，对比现有 register 式 $z_s$ 的 Phase 2 文本质量；
   - E2：置换鲁棒性消融（随机打乱 patch 顺序，验证 $z$ 不变而序列式基线漂移）；
   - E3：k 原型 Readout（k=K）与 sum 单向量在重建/生成任务上的取舍。

## 11. 参考实现（PyTorch + torch_geometric 伪代码）

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

class SchemeA_GraphEncoder(torch.nn.Module):
    """方案A: GNN + 置换不变 Readout → 每图 1 个向量 z ∈ R^d (L2 归一化)"""

    def __init__(self, in_dim, hid_dim=256, out_dim=128, num_layers=3, readout="sum"):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_dim, hid_dim)]
            + [GCNConv(hid_dim, hid_dim) for _ in range(num_layers - 1)]
        )
        self.readout = readout
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x, edge_index, batch):
        # x: (总节点数, in_dim); edge_index: (2, 总边数); batch: 每个节点所属图
        hs = []
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            hs.append(h)                     # 多尺度: 也可把各层拼起来再聚合
        # Readout: 置换不变（sum 默认; 保留图大小信息）
        z = (global_add_pool(h, batch) if self.readout == "sum"
             else global_mean_pool(h, batch))
        z = self.proj(z)                     # (图数, out_dim)
        return F.normalize(z, dim=-1)        # 余弦度量空间

    # 无监督对比训练（NT-Xent）示意:
    #   z1 = model(aug1.x, aug1.edge_index, aug1.batch)   # 图增强 1
    #   z2 = model(aug2.x, aug2.edge_index, aug2.batch)   # 图增强 2
    #   loss = nt_xent(z1, z2, temperature=0.1)
```

要点：
- `batch` 张量记录每个节点所属图，`global_add_pool` 按图求和 → 每图 1 行 —— 向量数量最小化到 1；
- 任何 `edge_index` / 节点顺序重排都不改变输出（§4）；
- 无节点特征时令 `x = degree_onehot` 或可学习节点嵌入（新图则不可用，需换成结构特征）。

## 12. 结论

- 方案A 是"无序 + 向量化 + 最小向量数（1 个/图）+ 可有损"需求下**可行且质量最优**的方法：置换不变性是结构保证（GNN 等变 + 不变 Readout），表达力受 1-WL 约束可用 GIN/多层缓解，碰撞（有损）由维度 d 控制。
- 落地注意：默认 sum Readout + LayerNorm；先单向量后多原型；无标签用对比学习；大图用采样式 GNN。
- 备选：无训练资源 → 谱指纹（B）；海量粗筛 → 哈希（C）。

## 13. 参考文献

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017. https://arxiv.org/abs/1609.02907
- Hamilton et al., *Inductive Representation Learning on Large Graphs (GraphSAGE)*, NeurIPS 2017. https://arxiv.org/abs/1706.02216
- Veličković et al., *Graph Attention Networks*, ICLR 2018. https://arxiv.org/abs/1710.10903
- Xu et al., *How Powerful are Graph Neural Networks? (GIN / 1-WL)*, ICLR 2019. https://arxiv.org/abs/1810.00826
- Zaheer et al., *Deep Sets*, NeurIPS 2017. https://arxiv.org/abs/1703.06114
- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017. https://arxiv.org/abs/1704.01212
- Sun et al., *InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning*, 2019. https://arxiv.org/abs/1908.01000
- You et al., *Graph Contrastive Learning with Augmentations (GraphCL)*, NeurIPS 2020. https://arxiv.org/abs/2010.13902
- Ying et al., *Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool)*, NeurIPS 2018. https://arxiv.org/abs/1806.08804
