# REPORT：与本文思想相近的方法调研 + 实验做法对照 + 综述仓库评估

> 日期：2026-10-26（Asia/Shanghai）
> 起因：用户要求「找找有没有与我们的方法有类似思想的方法，看看有没有雷同，也看看别人是怎么实验的」，并给出综述仓库 <https://github.com/yaolinli/MLLM-Token-Compression>。
> 本文的"我们"指当前稿件：ViT 编码 → $K=32$ 个 1248 维**连续**特殊 token（无量化、无码本）→ 共享解码器按 $T=16$ 步**循环读出**（每步均匀读 2 个 token、BPTT 训练、逐步重建监督），因此同一模型沿读出步数给出 $2t$ 个 token 的**可截断嵌套码**；指标 rFID（主）+ PSNR/SSIM/L1；另有容量阶梯与 SAVOIAS 复杂度分析。
> 所有数字/结论均来自下列来源的原文页面（arXiv abs/html、PMLR、ICLR proceedings、综述 README），未核实的写明"未核实"。**不做数值跨协议比较**（各文的分辨率、留出集、底座都不同）。

---

## 0. 结论摘要（先看这 6 条）

1. **"单模型 + 可变 token 数 + 由粗到细"在 2025–2026 已成为一条密集赛道（variable-length 1D tokenizer），不是一个空白**。最接近的 6 个：FlexTok（ICML'25）、One-D-Piece（2025）、NesTok（2026-09）、CaTok（2026-03）、DetailFlow（2025-05）、ElasticTok（ICLR'25）；再外围：ChannelTok（2026-06）、SpectralAR（2025-06）、Semanticist（ICCV'25）、VideoFlexTok（2026-04）。
2. **雷同度判断：核心思想「高」，具体机制「中」**。它们绝大多数是"**码侧**可变"（nested dropout / tail token drop / 谱排序 / PCA 式排序 → 前缀即可用），我们是"**读出侧**可变"（码长固定、按解码步截断）。但要注意三处**机制级**重合：① **CaTok（2026-03）明确把"解码时间步"绑定到"token 区间"**（"selecting tokens over time intervals and binding them to the MeanFlow objective"）——与"第 $t$ 步读第 $t$ 段窗口"直接对应；② NesTok 的"cross-length training（全长序列指导短序列）"与我们的"逐步深监督"同向；③ **ALIT 与 CAT 已经在"连续、单模型、可变率"上做过**（ALIT 用编码侧 recurrent allocation + 外部 halt 准则；CAT 用嵌套 VAE + 外部复杂度打分器选 8/16/32×），我们"连续 token"这一条并不新，只是**把可变性放到解码读出侧且不依赖外部 halt 准则**。
3. **论文 §2 目前只引到 FlexTok / ElasticTok / nested dropout / MRL / M3 / ALIT**（见 `sec2_related.tex` 第 4、6、8、16 行，`ref.bib` 已有对应条目），且已明确写"本文不主张首次提出可变的预算"——这条自我定位是**正确且必要**的；但**缺 2025–2026 的 NesTok / CaTok / One-D-Piece / DetailFlow / ChannelTok / SpectralAR / Semanticist / VideoFlexTok**，审稿人很容易用"FlexTok/NesTok 已经做到单模型 1–256 token"来质疑新颖性。
4. **综述仓库与本文这条线几乎不重叠**：它是 MLLM **理解侧**视觉 token 压缩（pruning / merging / KV cache / projector）的论文索引（约 370 条），明确写"不综述纯视觉任务的 token 压缩"；**TiTok、FlexTok、VQGAN、RQ-VAE 等生成式 tokenizer 在 README 与论文 PDF 中 0 命中**，全文 rFID/FID/PSNR/SSIM/LPIPS 出现 **0 次**，也没有"重建率 vs 压缩率"数值表；仓库里带 `Adaptive_Tokenization` 标签的只有 **1 条**（VideoFlexTok），benchmark 套件标注 "Coming Soon"。可借鉴的只是它的**标签体系**（Position / Method / Ratio / Retrain-Plugin）与 §7.2 对"自适应预算"的开放问题表述，不能当作图像 tokenizer 的综述来引。
5. **一个必须正面回应的理论点**：nested dropout 在半线性/线性自编码器下与 PCA 等价（Rippel 等 2014，arXiv [1402.0915](https://arxiv.org/abs/1402.0915)；本轮只据其原文与后续引用，未逐式重推）。再叠加 NS 语义：我们的"按重要性排序、前缀可截断"在理论上很可能被审稿人读成"就是 PCA 的重新发现"。**我们的 PCA 基线（rFID 2.12@rank 204）恰好是这个问题的最强答案**——但要把话说透：逐块 PCA 是**每图独立最优线性码**，我们的读出前缀是**单模型共享的非线性码**，二者只在"有序性"上同源、在"是否每图重训、是否可跨图共享、是否支持任意步截断"上不同。建议在 §4.2/§5 显式加一段"与 PCA 的关系"（见 §5 草稿）。
6. **别人怎么实验（共性）**：ImageNet-1k 256×256（少数 DFN-2B、COCO-30k text）；**rFID 是重建质量的主指标**（我们选 rFID 与这条线一致）；预算点取 1/2/4/8/16/…/256 或 32/64/128/256；**几乎都配一个下游 AR 生成实验**（gFID / IS）来证明 token 序列"能用"；消融集中在"嵌套/尾部丢弃 vs 定长训练"、"cross-length 训练"、"解码器类型/采样步数"、"模型规模"。**我们的实验缺三块**：① token 的字节数口径（他们报 bytes）；② 同协议定长基线 / nested-dropout 基线；③ 下游 AR 生成（这是 tokenizer 论文的"标配"）。

---

## 1. 最相近方法一览

| 方法 | 出处 | 核心机制 | 单模型多预算 | 截断/可变方式 | 量化 | 与本文重合点 | 雷同度 |
|---|---|---|---|---|---|---|---|
| **FlexTok** | ICML 2025，arXiv [2502.13967](https://arxiv.org/abs/2502.13967) | 2D 图像 → **有序 1D 离散 token**（1–256）；nested dropout 学层次化顺序；**rectified-flow 解码器**保证任意前缀可解 | ✅ | 任意前缀（nested dropout 采样长度） | ✅ VQ（64k 码本） | "单模型多预算 + 前缀可截断 + 由粗到细"三件套全中 | **高** |
| **One-D-Piece** | 2025，arXiv [2501.10064](https://arxiv.org/abs/2501.10064) | 离散 1D tokenizer + **Tail Token Drop** 正则，把关键信息压到序列头部；质量可控压缩 | ✅ | 尾部丢弃 → 任意前缀 | ✅ | "尾部丢弃 = 我们按步截断"的码侧版本 | **高** |
| **NesTok** | 2026-09，arXiv [2609.36756](https://arxiv.org/abs/2609.36756) | 嵌套自对齐：**cross-length training**，用全长序列指导短序列，让短前缀逼近全长重建 | ✅ | 嵌套 dropout（任意前缀） | ✅ | "短前缀也要重建好"= 我们的逐步监督目标 | **高** |
| **CaTok** | 2026-03，arXiv [2603.06449](https://arxiv.org/abs/2603.06449) | 1D 因果 tokenizer + MeanFlow 解码器；**把 token 按时间区间选取并绑定到 flow 目标** | ✅（时间步↔token 区间） | 时间步/区间截断 | ✅（VQ 底座） | "解码步 ↔ token 区间"与本文"第 t 步读第 t 段窗口"直接对应 | **很高** |
| **DetailFlow** | 2025-05，arXiv [2505.21473](https://arxiv.org/abs/2505.21473) | 1D **coarse-to-fine**：next-detail prediction，用逐级退化图像监督分辨率感知的 token 序列 | ✅（长度固定但由粗到细） | 按"细节级"截断 | ✅ | 由粗到细的**监督信号构造**（我们靠读出步数） | **高** |
| **ElasticTok** | ICLR 2025，arXiv [2410.08368](https://arxiv.org/abs/2410.08368) | 训练时随机丢弃尾部 token（masking），推理时按重建阈值动态分配 token 数；图像 + 视频 | ✅ | 尾部随机丢弃 → 任意长度 | ✅ | "按重建阈值停"= 我们 §3.8 的自适应预算；**"动态分配"已被做过** | **高** |
| **ChannelTok** | 2026-06，arXiv [2606.04461](https://arxiv.org/abs/2606.04461) | **每个 latent 通道当一个 token**，CNN-Transformer 混合；尾部随机丢弃 → 保留前 $k$ 个通道 | ✅ | 前 $k$ 通道 | ✅ | "前缀 = 预算"的另一实现；rFID 2.92 / 159 M / 解码快 8.6× | 中高 |
| **SpectralAR** | 2025-06，arXiv [2506.10962](https://arxiv.org/abs/2506.10962) | **嵌套谱 token 化**：按频率从低到高排序，粗到细 AR | 部分（嵌套谱） | 谱层级 | ✅ | "按重要性排序、前缀承载低频/整体" | 中 |
| **Semanticist** | ICCV 2025，arXiv [2503.08685](https://arxiv.org/abs/2503.08685) | 把 **PCA 式结构**嵌进 token 空间：1D 因果序列，逐 token 解释方差递减 | 部分 | 前缀（PCA 式） | ✅（+ 扩散解码器） | **与本文 PCA 基线同源**：说明"PCA 式有序表示"是被认可的强基线 | 中（对 PCA 部分：高） |
| **VideoFlexTok** | 2026-04，arXiv [2604.12887](https://arxiv.org/abs/2604.12887) | FlexTok 的视频版：变长、粗到细时空 token | ✅ | 任意前缀 | ✅ | 同 FlexTok，扩展到视频 | 中 |
| Matryoshka MM / MRL / nested dropout | ICLR'25 / ICML'22 / ICLR'14 | 嵌套表示：低维/短前缀即可用 | ✅ | 维度或序列前缀 | — | 我们是该思想在"解码步"上的实例化 | 中 |

| **CAT**（Continuous Adaptive Tokenizer） | 2025，arXiv [2501.03120](https://arxiv.org/abs/2501.03120) | **嵌套 VAE**（单模型支持 8×/16×/32× 压缩率，中间层 skip），外部 caption+LLM 复杂度打分器逐图选压缩率 | ✅（8/16/32×） | 按复杂度选压缩率（非截断） | ❌ 连续 latent | 连续 + 单模型多率 + **逐图自适应**（与我们 §3.8 直接对应） | **中高** |
| **ALIT** | 2024，arXiv [2411.02393](https://arxiv.org/abs/2411.02393) | 编码侧 recurrent：每轮新增 token、残差更新旧 token，按重建阈值 halt；IN-100 训练 | ✅（32–256） | halt 准则 | 可选（连续/离散消融） | 连续 + 单模型多预算 + **SAVOIAS 人类复杂度分析**（与我们的分析轴重合） | **中高**（论文已引） |
| **PLONQ** | 2021，arXiv [2102.02913](https://arxiv.org/abs/2102.02913) | 单一 bitstream 渐进压缩：**nested quantization**（粗量化网格嵌套于细网格）+ latent ordering，任意截断点可解码 | ✅ | 按比特截断 | ✅ + 熵编码 | "单模型 + 任意截断点"的**思想祖先**（度量是 PSNR/BD-rate） | 中（谱系） |
| **Toderici et al.** | ICLR 2016，arXiv [1511.06085](https://arxiv.org/abs/1511.06085) | LSTM **循环**渐进压缩：每步 8 bits，8–16 步越传越清晰 | ✅ | 按步截断比特 | ✅（8 bits/步） | "按步数换质量"的直系祖先；论文已引 | 中（谱系） |
> 说明：上表"量化"列指 token 是否离散码本；本文**连续 token、无码本、无 VAE/flow 底座**，这一点在表中是少数派，但**不构成"能力"上的新颖性**（只构成"设置"上的差异）。

---

## 2. 逐篇卡片（含"别人怎么实验"）

### 2.1 FlexTok（ICML 2025）
- 机制：三阶段——Stage 0 VAE（SDXL VAE，8× 下采样）→ Stage 1 tokenizer（2D latent 重采样为 1D 有序离散 token，nested dropout）→ Stage 2 rectified flow 解码器；任意前缀可解码。
- 实验：
  - 数据/分辨率：**ImageNet-1k 256×256**（另有 DFN-2B 做文本条件）；重建在 **ImageNet-1k 完整验证集**的 256×256 crop 上评。
  - 指标：**rFID（主）+ MAE + DreamSim**；附录补 PSNR/SSIM/LPIPS；生成用 gFID/IS/CLIPScore/sFID/precision/recall。
  - 预算点：$K_{\text{keep}}\in\{1,2,4,8,16,\dots,256\}$，**每个点都报 rFID/MAE/DreamSim**。Table 8（d18-d28）实测：1 token rFID 4.01 / PSNR 9.35 dB；2 tokens 3.09；4 tokens 2.43；8 tokens 1.90；16 tokens 1.61；32 tokens **1.45**；64 tokens 1.37；128 tokens 1.20；256 tokens 1.08。**bytes 口径**：码本 64k ⇒ 每 token 2 bytes（即 32 tokens = 64 bytes）。
  - 生成：类条件 1.33B AR Transformer，8–128 token 区间 gFID<2；文本条件用 DFN-2B。
  - 消融：2D vs 1D tokenizer、noise schedule、REPA、register token vs nested dropout、nested dropout 长度分布（pow2 / unifpow2）、VAE 通道数、flow 采样步数（25 步后饱和）、AR 模型规模（1–8 token 时 gFID 与 AR 规模无关，>128 token 时强相关）。
  - 关键发现（对我们有用）：**rFID 在 token 数增加时改善、MAE 在低 token 数下继续变差**——分布级与像素级指标不同向，这正是我们在 §4.2 用 rFID 为主、PSNR/L1 为辅的理由。

### 2.2 NesTok（2026-09）
- 机制：基于 One-D-Piece 的 ViT tokenizer（ViT-B 编码 + ViT-L 解码、4,096 码本、latent 32）；提出 **nested self-alignment / cross-length training**：联合优化各长度重建，用全长序列指导短序列。
- 实验：ImageNet-1k 256×256；rFID 用 MAR 的评测代码、gFID 用 ADM 的 TF 套件（50k 生成图）；数字：**rFID 0.98 @256 token**（其 variable-length 1D baseline 1.93；One-D-Piece 1.08 @641M；NesTok 390M）；**gFID 1.46**（LlamaGen AR，无额外扩散组件）；逐预算对比 32/64/128/256，32 token 时 rFID 3.49。
- 消融（Table 2）：定长 1D（rFID 最好但不支持变长）→ 加 nested dropout（支持变长但重建变差）→ 加 cross-length training（恢复）。**"nested dropout 会伤重建"是这条线公认的痛点**，NesTok 的解法是"全长指导短序列"——**与我们的逐步监督同向**，但它只对齐重建、没有"读出侧"这一步。
- 相关工作里把方法按解码器分两类：**ViT 解码器**（One-D-Piece、SpectralAR、ReTok）vs **Diffusion/Flow 解码器**（FlexTok、D-AR、CaTok）；并指出 FlexTok "32 token gFID 最好，但增加 token 反而掉"。这句话对我们是**现成的对照点**：我们的单调下降 rFID 曲线（89.78→19.97）说明"增加 token 仍有收益"，与 FlexTok 的生成侧非单调形成对照（但我们是像素重建，不能直接驳）。

### 2.3 CaTok（2026-03）——与本文机制最接近
- 机制：1D 因果 tokenizer + **MeanFlow** 解码器；"**selecting tokens over time intervals and binding them to the MeanFlow objective**"——把 token 序列按区间切开、每个区间绑定一个 flow 时间步；另提案 REPA-A 正则（对齐视觉基础模型特征）。
- 实验：ImageNet 重建 **FID 0.75 / PSNR 22.53 / SSIM 0.674**（更少 epoch）；AR 生成与主流可比；支持单步生成与多步采样。
- 对我们的意义：**"解码时间步 ↔ token 区间"不是我们首创**（虽然它用 flow 时间步、我们用 Transformer 读出步，且它仍要 VQ 底座）。写作时必须与它划清：**我们是"同一 Transformer 解码器的逐步读出"，不引入 flow/扩散时间维，且 token 连续**。

### 2.4 DetailFlow（2025-05）
- 机制：next-detail prediction——用**逐级退化的图像**监督 token 序列，使序列天然 coarse-to-fine；配套 parallel inference + self-correction。
- 实验：ImageNet-1k 256；2.96 gFID @128 token（VAR/FlexVAR 需 680 token）。
- 对我们的意义："由粗到细"的**监督构造**已有成熟做法；我们的"第 t 步只看到前 2t 个 token 的监督"是同一族的另一种实现（训练信号来自截断输入，而非退化图像目标）。

### 2.5 One-D-Piece（2025）
- 机制：Tail Token Drop 正则（训练时随机截尾），把信息压到头部；质量可控压缩；"same tokenizer, variable length"。
- 实验：重建质量对比 **JPEG / WebP**（同字节数下感知质量更好）→ 说明这条线**确实把"字节数"当预算轴**；下游任务：分类、检索等；后续工作（NesTok）把它作为变长 1D 基线。
- 对我们的意义：**"预算 = 字节数"的对照方式**值得借鉴：我们现在用"连续值个数"，审稿人会问 bf16/量化后多少字节。

### 2.6 ElasticTok（ICLR 2025）
- 机制：训练时对每帧的 token 编码**随机丢尾部**，推理时按**重建阈值**动态分配 token 数（"more complex data can leverage more tokens"）。
- 实验：图像 + 视频；按 MSE 阈值分配；定性展示复杂内容用更多 token。
- 对我们的意义：**"按重建质量阈值逐图定预算"已被 ElasticTok 做过**，且我们的 §3.8/§4 自适应实验若写成"首次"会被打；我们论文目前的写法（"自适应否证 + 只主张可截断"）是安全的。

### 2.7 ChannelTok（2026-06）
- 机制：不是空间 token 而是**通道 token**；训练时尾部通道随机丢弃 → 推理保留前 k 个通道即可变长。
- 实验：ImageNet；**rFID 2.92**、**159 M** 参数、解码比次优快 8.6×；多预算质量一致；与"重参数/重解码器"的同类比效率。
- 对我们的意义：说明"尾部丢弃 + 前缀即预算"的设计空间很宽；也说明**效率（解码速度/参数量）是这条线的硬指标**——我们 481 M、16 步串行解码，效率上不占优，必须在论文里明确"不主张效率最优"。

### 2.8 SpectralAR（2025-06）/ Semanticist（ICCV'25）
- SpectralAR：图像 → **嵌套谱 token**（低频到高频），粗到细 AR；ImageNet-1k 重建 + 生成，3.02 gFID @64 token / 310 M。
- Semanticist：把 **PCA 式"解释方差递减"** 写进 token 结构（1D 因果序列，逐 token 贡献互不重叠的信息），配扩散解码器；声称 SOTA 重建 + 可解释性 + 更少 token 训练/推理。
- 对我们的意义：**两篇都说明"有序/嵌套/按重要性排序"的 token 是主流设计**；Semanticist 更是**直接支持我们"PCA 是很强的同信息量基线"这件事**——我们的 PCA 对照不是自曝其短，而是与文献一致的诚实设定；但写作上要把它放进"相关方法"的谱系里（"逐块 PCA"是 Semanticist 的线性极端版）。

### 2.9 Matryoshka 系列（ICLR'25 / ICML'22 / ICLR'14）
- M3：把视觉内容表示为**嵌套 token 集合**，推理时按实例选粒度；发现 COCO 类基准约 **9 个 token** 就接近 576 token 的准确率。
- MRL：维度嵌套（低维前缀可独立用）；nested dropout：随机保留前缀学有序表示。
- 对我们的意义：我们论文已引；可再强调"嵌套结构在**表示学习**与 **LMM 投影器**两侧都已被验证"，我们的贡献只是"在**解码读出**侧实现"，避免被说成"嵌套思想新"。

### 2.10 综述仓库（MLLM-Token-Compression）评估

- **出处与范围**：`Towards Efficient Multimodal Large Language Models: A Survey on Token Compression`（Linli Yao 等，TechRxiv v1.0，2025-12；**无 arXiv 版本**——arXiv 上相近题目的 2507.20198 是 Kele Shao 等的另一篇，勿混）。仓库 = 约 370 条论文的**标签索引** + 同名 30 页 PDF（taxonomy 在 PDF 的 Fig.2 / Fig.6，README 只有表格）。
- **taxonomy（PDF §3，按"压在哪"）**：Vision Encoder（Inside：Visual Token Dropping / Merging / Multi-Scale；Outside：Purely-Vision / Text-guided）→ Projector（Transformation：Pooling / Pixel Shuffle / Convolution；Query-based：Q-Former 及其变体 / Cross-Attention；Importance-Driven）→ LLM（Prefill：Importance / Learnable / Merging / Fusion；Decoding：KV-cache 压缩）→ Multi-Module（Collaborative / **Progressive**）。第二轴（§4，"怎么选"）：时序增强（Fixed / Dynamic / Hybrid）、纯视觉 vs 文本引导、合并 vs 丢弃、Plug-in vs Retrain、训练 vs 推理。
- **关键事实（对我们最有用）**：
  1. **边界明确排除我们这条线**：原文"Here we focus **exclusively on methods applied in multimodal LLMs**, and **do not review token compression for pure vision tasks**"；§2.1 又写偏离 (vision encoder → projector → LLM) 三段式架构的范式"beyond the scope"。**TiTok / FlexTok / VQGAN / RQ-VAE / LlamaGen 在仓库与 PDF 中 0 命中**。
  2. **没有重建类指标**：全文 rFID / FID / PSNR / SSIM / LPIPS 均 **0 次**；评测协议（§5.2）是 Accuracy / GPT-Score 为主，效率用 token retention、FLOPs、latency、memory。
  3. **没有"单模型多预算"这一列**：最接近的是仓库标签 `Ratio: Fix / Dynamic`（我统计 README：Dynamic 151 条 vs Fix 121 条）与 §7.2 的开放挑战"task- and content-aware compression … dynamically determines the degree and manner of token reduction"（引 M3：多数基准 **9 token** 够用，OCR/文档要 **144–576**）。
  4. **没有数值对比表**：Table 1 只有 Method / Date / Venue / Modality / Position / TQ / Retrain-Plugin；README 的 benchmark 套件（宣称 26 个 benchmark）仍写 "Coming Soon"。
- **仓库里与我们动机最近的条目**（都属理解侧，不与本文抢 rFID 赛道）：
  | 论文 | 出处 | 相近点 |
  |---|---|---|
  | **VideoFlexTok** | 2026-04，[2604.12887](https://arxiv.org/abs/2604.12887) | 唯一 `Adaptive_Tokenization` 标签：变长 + 粗到细 + flow 解码器，任意 token 数可重建 |
  | **PVC: Progressive Visual Token Compression** | 2024-12，[2412.09613](https://arxiv.org/abs/2412.09613) | "渐进式"压缩，统一图像/视频、共享 token 预算（但"渐进"指推理流程多阶段，不是 anytime 码流） |
  | **Video-XL-Pro（ReCoT）** | 2025-03，[2503.18478](https://arxiv.org/abs/2503.18478) | **用重建监督**做极长视频 token 压缩（重建是手段，目标是理解效率） |
  | **MQT（Matryoshka Query Transformer）** | 2024-05，[2405.19315](https://arxiv.org/abs/2405.19315) | 训练时随机取前 $m<M$ 个 query token → 单模型多预算（嵌套思想的 LMM 版本） |
  | **VTC（Visual Token Coding）** | 2026-08，[2608.28008](https://arxiv.org/abs/2608.28008) | 含 **Dynamic Token Allocation (DyTA)**，按帧复杂度分配预算 |
  | **VisionWeave** | 2026-10，[2610.07987](https://arxiv.org/abs/2610.07987) | "弹性视觉表示"（内容自适应粒度）作为 MLLM 原生能力 |
- **借鉴方式**：引言/§2 若要提"MLLM 侧也在追求可变预算"，可引该综述 + VideoFlexTok；但**不能**用它证明"图像 tokenizer 的可变长是新问题"——它恰好说明这是一个跨越理解/生成两侧的成熟方向。


### 2.11 CAT（Continuous Adaptive Tokenizer，2025，arXiv [2501.03120](https://arxiv.org/abs/2501.03120)）
- 机制：**嵌套 VAE**——单个 VAE 在中间层保留 skip，使同一模型支持 8×/16×/32× 三种压缩率；逐图压缩率由一个**外部打分器**（图像 caption + LLM 按"感知复杂度"打分）选择。
- 实验：512×512、约 380 M 图（Shutterstock）；rFID / LPIPS / PSNR 在 **COCO / ImageNet / CelebA / ChartQA** 四个集上评；187 M 参数；下游 DiT 生成；消融压缩率分布（4,7）vs（2,8）与 latent 通道数；与 JPEG/MSE/LPIPS 的"复杂度预测能力"比（Pearson r 0.55 对 0.31/0.36/0.23）。
- 对我们的意义：**"连续 + 单模型多率"我们不是第一个**（这是比 FlexTok 更容易被忽略的威胁）；但它靠**外部打分器 + 嵌套 VAE**，不是"同一条码按读出步截断"，也不是每图自适应 halt 的端到端方案。我们 §3.8 的"自适应预算"论证与它的复杂度打分器**正面重叠**，必须在 related work 里点名区分（"它用外部预测器选率，我们用同一码的读出步数覆盖多预算"）。

### 2.12 谱系祖先：PLONQ（2021）与 Toderici 等（ICLR 2016）
- **PLONQ**（arXiv [2102.02913](https://arxiv.org/abs/2102.02913)）：单一 bitstream 的渐进压缩——**nested quantization**（细网格的量化格点包含粗网格）+ latent 排序，接收方在任意字节处截断都能解码出相应质量；在 Kodak/Tecnick/JPEG-AI 测试集上与 BPG444 比 BD-rate，baseline 含"多模型变码率""非渐进单模型变码率""SPIHT"。
- **Toderici 等**（arXiv [1511.06085](https://arxiv.org/abs/1511.06085)）：LSTM **循环**渐进压缩，每步 8 bit、8–16 步，早期步给全局结构、后期步补细节；32×32 thumbnail、10 万图，按 SSIM@64B/128B 与 JPEG/JPEG2000/WebP 比。
- 对我们的意义：这两篇证明"**单模型 + 按步/按字节截断 + 由粗到细**"在**压缩**领域已有二十年谱系（我们论文已引 Toderici）；审稿人若问"这不就是渐进压缩吗"，正解是：**对象不同**（我们压的是 token 数量/表示长度，不是比特流）、**底座不同**（无量化/熵编码）、**评估不同**（rFID/PSNR 而非 BD-rate），但**思想上不是新的**——所以贡献必须落在"视觉 token 接口上的读出侧实现 + 受控分析"，不能落在"提出可截断"。

---

## 3. "别人怎么实验"的协议对照（供我们改稿参考）

| 维度 | FlexTok | NesTok | CaTok | ChannelTok | One-D-Piece | ElasticTok | **本文** |
|---|---|---|---|---|---|---|---|
| 数据/分辨率 | IN-1k 256（+DFN-2B） | IN-1k 256 | IN-1k 256 | IN-1k 256 | IN-1k 256 | 图像+视频 | **IN-100 224**（受控、小规模） |
| 重建留出集 | IN-1k **完整 val** | IN-1k val | IN-1k val | IN-1k val | IN-1k val | 图像/视频 | **IN-100 val 5,000** |
| 主指标 | **rFID** + MAE + DreamSim（附 PSNR/SSIM/LPIPS） | **rFID** + gFID + IS | **FID** + PSNR + SSIM | **rFID** + 效率 | 重建质量 vs JPEG/WebP | 重建阈值/MSE | **rFID** + PSNR/SSIM/L1 |
| 预算点 | 1,2,4,8,…,256 | 32,64,128,256 | 多档 | 多档 | 多码率 | 动态 | **2,4,…,32（16 档）** |
| 字节口径 | 报 **bytes**（2 B/token） | 未强调 | 未强调 | 报参数量/速度 | **按字节比 JPEG/WebP** | — | **未报**（连续值个数） |
| 下游 AR 生成 | ✅ 1.33B AR，gFID<2 @8–128 | ✅ LlamaGen，gFID 1.46 | ✅ | ✅ 变长 AR | 分类/检索等 | 视频生成 | **❌ 无** |
| 消融重点 | 2D/1D、nested dropout vs register、flow 步数、VAE 通道、AR 规模 | 定长 vs nested vs cross-length | 因果性/REPA-A | 通道 vs 空间、速度 | Tail drop 比例 | 丢弃比例/阈值 | **容量、复杂度、PCA、读出计划** |
| 参数量 | 数百 M–1B+ | 390 M（tokenizer） | — | **159 M** | 641 M | — | **481.5 M** |
| **CAT（对照）** | 512²、380M 图 | COCO/IN/CelebA/ChartQA | rFID/LPIPS/PSNR | 8/16/32× 三档 | 未强调 | ✅ DiT | — | 187 M |
| **ALIT（对照）** | IN-100 训练 | IN-100/COCO/WIT | rFID + linear probe | 逐轮 32→256（halt） | 未强调 | ✅ 有（编码侧循环） | ✅ SAVOIAS 复杂度 | 52.8–409 M |

**共性结论**：这条线的"标配四件套" = ① rFID 随预算曲线；② 字节口径；③ 至少一个定长/同预算基线；④ 下游 AR 生成。我们**满足 ①，缺 ②③④**。

---

## 4. 我们的实验可怎么补（按性价比排序）

1. **补字节口径（低成本）**：现在写"32 token × 1248 维 = 39,936 个连续值"；再加一句 bf16/fp16 下 ≈ 79.9 KB/图，并说明"与 VQ 类 tokenizer 的 2 B/token 不可比，本文不做真实比特率主张"——把审稿人最容易问的问题先答掉。
2. **补定长同预算基线（中低成本，需训练）**：在**同一架构/同一数据**上训一个"定长 32-token、单步读出"的模型（我们其实已有阶段 1 的 $T=1$ 单步模型 rFID 6.13，但它是同一码的另一种读出，不是独立定长基线）；更省的做法是引 NesTok Table 2 的结论（nested dropout 会伤重建 → 用 cross-length/逐步监督恢复）并在正文说明我们没有独立定长基线是**受控规模下的取舍**。
3. **补 nested-dropout 基线（中成本，1 个模型）**：把我们的 16 步监督换成"随机长度前缀 + 单次解码"的 nested dropout 训练，比较 rFID–预算曲线。这是与 FlexTok/NesTok 最直接的机制对照，也最能说明"读出侧逐步监督"是否比"码侧随机截断"更好——如果没差别，反而应老实写"两种实现等价"。
4. **补逐图自适应预算实验（低成本，已有数据）**：用每图逐步 L1（`per_image_joined.csv`）做"给定误差阈值 → 每图选步数"的分配曲线，报平均 token 数 vs 平均 L1（ElasticTok/M3 的协议）。我们已有全部数据，只差画图与一段分析。
5. **补 LPIPS / DreamSim（低成本，需装库或离线）**：FlexTok 强调"分布级（rFID）与像素级（MAE/PSNR）不同向"，我们已有 rFID/PSNR/L1，补 LPIPS 能对齐这条线的报告习惯。
6. **下游 AR 生成（高成本，但最有说服力）**：在 token 序列上训一个小 AR（如 100–300 M）报 gFID 随预算变化。若时间不允许，至少在 §5 局限里写明"未做下游生成，因此不主张生成侧的可截断收益"——这正是 FlexTok 的核心卖点，我们回避它比含糊带过更安全。
7. **补"每个预算单独训一个定长模型"的对照（中高成本，最硬）**：FlexTok 的对照是"参数量/训练步数/数据完全对齐的 2D grid + flow decoder"，TiTok/MRL 也都是"每个预算单独训一个"。我们若只有"同一模型的逐步读出"，审稿人会问"是不是逐步监督拖累了 32-token 上限（19.97 对 PCA 2.12）"。至少可以训 4 个定长小模型（4/8/16/32 token，单步读出、其余设置不变）做"专训 vs 共享"的分歧点——工作量约等于阶段 1 的 4 次短训，但能回答最关键的公平性问题。
8. **补"token 轴"的 PCA 截断基线（低成本，直击最致命的审稿意见）**：现在我们的 PCA 是**逐 patch 空间基**（196×r）；建议再加一条：把 32×1248 的 token 矩阵做 PCA（全数据共享基），按方差取前 $m$ 个 token（$m=4/8/16/32$）再解码——即"只保留贡献最大的 token"。若这条曲线明显优于我们的逐步读出，说明我们的读出顺序不如数据驱动排序；若相近，则正好证明"按步读出的顺序≈PCA 顺序"。**这是回应"nested dropout ≡ PCA"质疑的唯一实验级证据**。
9. **补"逐步边际贡献"分析（低成本，已有数据）**：用每步的 L1/PSNR 增量画"第 t 步带来的增益"柱状图，并统计"哪一步之后增益 < 0.1 dB"——这是对"越靠后是残差/细化"这一说法的直接量化，也是 One-D-Piece "替换第 i 个 token 看重建变化"的对应物。
10. **补下游探针（低成本替代方案）**：用 One-D-Piece 的 LPA（linear probing accuracy）协议，在 4/8/16/32 token 的码上各训一个线性分类器（ImageNet-100 标签），报"截断后语义还剩多少"。这比训 AR 便宜得多，却能部分回应"生成侧可用性"。

---

## 5. 写作/引用建议（可直接落地）

**建议新增引用（按重要性）**：
1. FlexTok（已引 ✅）——补一句"它在生成侧发现 token 数增加后收益递减/反转，我们的重建侧是单调的"，作为不比较数值的理由。
2. NesTok（2026-09，arXiv 2609.36756）——"cross-length training 与本文逐步监督同向；但它靠码侧嵌套，我们靠读出步"。
3. CaTok（2026-03，arXiv 2603.06449）——"把解码时间步绑定到 token 区间已有先例（MeanFlow 时间维）；本文是纯 Transformer 的读出步，不引入 flow 时间维，且 token 连续"。
4. One-D-Piece（2025，arXiv 2501.10064）+ ElasticTok（已引 ✅）+ ChannelTok（2026-06，arXiv 2606.04461）——"尾部丢弃/前缀保留"的码侧实现。
5. DetailFlow（2025-05，arXiv 2505.21473）——"由粗到细的监督构造已有 next-detail 预测"。
6. SpectralAR（2025-06，arXiv 2506.10962）、Semanticist（ICCV'25，arXiv 2503.08685）——"有序/嵌套谱/PCA 式排序是 tokenizer 的主流结构"，同时**给我们的 PCA 基线一个文献根据**。
7. VideoFlexTok（2026-04，arXiv 2604.12887）——若讨论视频/多模态扩展。
8. **CAT（2025，arXiv 2501.03120）**——"连续 + 单模型多压缩率 + 逐图自适应选率"已有先例；必须点名区分（它靠外部打分器选 8/16/32×，我们靠同一条码的读出步覆盖多预算）。
9. **PLONQ（2021，arXiv 2102.02913）+ Toderici 等（ICLR 2016，arXiv 1511.06085，已引 ✅）**——"单模型 + 任意截断点 + 由粗到细"在压缩领域的谱系祖先；用于把我们的定位写成"该谱系在视觉 token 接口上的一个实例"，而不是"提出可截断"。
10. **nested dropout（ICML 2014，arXiv 1402.0915，已引 ✅）**——需要补一句"在半线性自编码器下与 PCA 等价"，并接上我们 §4.2 的 PCA 讨论（见下）。

**§2 可加一段（草稿，接在"可变长度分词与循环分配"之后）**：
> 近两年的变长 1D tokenizer 把"单模型多预算"做成了标配：FlexTok 用 nested dropout 学有序序列、以 rectified-flow 解码器保证任意前缀可解；One-D-Piece 用 Tail Token Drop 把信息压到头部，按字节数对标 JPEG/WebP；ElasticTok 训练时随机丢尾部 token，推理时按重建阈值逐图分配；NesTok 进一步指出仅做 nested dropout 会损害短前缀的重建，提出 cross-length training 用全长序列指导短序列；CaTok 则把解码时间步与 token 区间绑定。这些工作与本文共享"码本身有序、前缀即可用"的前提，差别在可变性发生在哪一侧：它们让**码**变长、由训练目标决定顺序，本文让**读出**变长、由解码步数决定顺序；本文也不使用量化码本、VAE 或 flow 解码器，因此只与同协议的逐块 PCA 做数值比较，不与上述工作的 rFID 并列。

**§5 局限可加一句**：
> 本文未与 FlexTok、NesTok、CaTok 等变长 tokenizer 做同协议数值比较（它们的码本、底座、分辨率与留出集均不同），也未训练下游 AR 模型验证生成侧的可截断收益；"单模型多预算"这一能力本身已被上述工作验证，本文只主张其在**读出侧**的实现方式与相应分析。

**§4.2 或 §5 可加一段"与 PCA 的关系"（草稿）**：
> "有序前缀码"与 PCA 有已知的理论联系：nested dropout 在半线性自编码器下的解与 PCA 的子空间一致。本文的 PCA 对照因此可以读作"线性、逐图最优"的极端：它每图独立拟合、每个 patch 保留 $r$ 个系数，序列长度不降；本文的读出前缀是**单模型共享**的非线性码，长度由读出步数直接给出。两者在"前面承载主成分、后面承载残差"这一点上同源，差别在"是否逐图重训、是否跨图共享、是否能在推理时按步停止"。这一对照同时也是本文对"嵌套排序只是 PCA 的重新发现"这一质疑的回应：若排序只是线性 PCA 的重排，逐块 PCA 在像素指标上的优势就应当只体现在"每 patch 更多系数"，而无法解释读出前缀在多预算下的单调 rFID 曲线（89.78→19.97）。

**若审稿人问"与 FlexTok/NesTok 有何不同"**，一句话回答：
> 它们改变码（长度/顺序/量化），我们改变读出（同一 32-token 连续码按解码步截断）；它们的"可变"来自训练时的随机截断（nested/tail dropout），我们的"可变"来自 BPTT 逐步读出的均匀窗口；代价是我们只能按步长（2 token）截断、粒度更粗，收益是我们不需要码本/flow 解码器、且能给出容量与复杂度的受控分析。

---

## 6. 来源 URL

- 综述仓库：<https://github.com/yaolinli/MLLM-Token-Compression>（README raw 抓取，370 条，`Adaptive_Tokenization` 1 条）
- 综述论文：<https://www.techrxiv.org/doi/full/10.36227/techrxiv.176823010.07236701/v1>
- FlexTok：<https://arxiv.org/abs/2502.13967>；PMLR：<https://proceedings.mlr.press/v267/bachmann25a.html>
- NesTok：<https://arxiv.org/abs/2609.36756>
- CaTok：<https://arxiv.org/abs/2603.06449>
- DetailFlow：<https://arxiv.org/abs/2505.21473>
- One-D-Piece：<https://arxiv.org/abs/2501.10064>
- ElasticTok：<https://arxiv.org/abs/2410.08368>
- ChannelTok：<https://arxiv.org/abs/2606.04461>
- SpectralAR：<https://arxiv.org/abs/2506.10962>
- Semanticist：<https://arxiv.org/abs/2503.08685>
- VideoFlexTok：<https://arxiv.org/abs/2604.12887>
- CAT：<https://arxiv.org/abs/2501.03120>
- PLONQ：<https://arxiv.org/abs/2102.02913>；Toderici et al.：<https://arxiv.org/abs/1511.06085>
- nested dropout：<https://arxiv.org/abs/1402.0915>（ICML 2014）
- Matryoshka MM：<https://arxiv.org/abs/2405.17430>；MRL：<https://arxiv.org/abs/2205.13147>
- 本文对照协议来源：`doc/2026-10-26/REPORT_rfid_m240_pca.md`、`doc/2026-10-26/REPORT_ood_savoias_scale.md`

---

## 7. 专项核查："循环重建"这个思想有没有雷同？（决定实验去留）

### 7.1 先把"循环重建"拆成 4 个可分离要素

| 要素 | 本文的实现 | 文献里是否有先例 |
|---|---|---|
| (a) **逐步增长的输入前缀**（每步多看 2 个 token，每步都重建） | 第 $t$ 步读前 $2t$ 个 token、输出整图 | **有，且很密**：FlexTok（nested dropout）、One-D-Piece（Tail Token Drop）、NesTok（cross-length training）、DetailFlow（逐级退化监督）、ALIT（逐轮新增 latent）——"多长度都重建"本身不新 |
| (b) **画布 carry**（$\hat{Y}_t = \hat{Y}_{t-1} + \Delta_t$，同一个解码器带上一步输出） | `Y = query_base + Y_{t-1}`（裸加，BPTT） | **有直系祖先**：Toderici 等 2015/2016 的 RNN 编解码 + **additive / scaled-additive reconstruction**（每轮解码残差、累加到上一轮重建；一次训练、任意码率）；PLONQ（2021）的"从粗到细渐进细化 latent + 任意截断点"。**"循环 + 残差累加 + 渐进重建"在压缩领域是 2015–2016 年的标准配方** |
| (c) **跨步 BPTT**（carry 不 detach） | 端到端反传 T 步 | **有内部证据**：本仓库 2026-09-16 的 detach vs BPTT 对照（576-token/24 步配置）——像素 L1 **23.72 → 16.42（−30.8%）**，逐步曲线由"平"转"降"，显存/步速不变（单 seed）。说明"循环的梯度通路"确有作用，但**不构成对外新颖性**（Toderici 的 RNN 本来就是端到端） |
| (d) **步数 ↔ 预算**（$K=2T$，按步截断即合法档位） | 16 步 × 2 token = 2…32 | **有近亲**：ALIT 逐轮增量 + 重建阈值 halt；CaTok 把 **flow 时间步绑定到 token 区间**；PLONQ 的 latent 重要性排序支持任意截断点。**"步数就是预算轴"不是我们首创** |

**逐项结论**：四个要素**各自都有先例**；我们没找到的是**"四要素的这一个具体组合"**——固定 $K=32$ 连续码（无量化/无 VAE/无 flow）+ 解码器侧画布 carry + 均匀 2-token 读窗口 + BPTT + 按步截断成 2…32 的嵌套档，且从零训练、以 rFID 评估。这是一个**组合级/设置级**的差异，不是思想级的新颖性。

### 7.2 与最接近的 4 个工作逐条对照

| | 循环在哪 | 解码器 | 递进载体 | 量化 | 变长/截断 | 与本文的差别 |
|---|---|---|---|---|---|---|
| **Toderici 2015/2016**（[1511.06085](https://arxiv.org/abs/1511.06085) / [1608.05148](https://arxiv.org/abs/1608.05148)） | 编解码器都是 RNN，**逐次迭代** | RNN 解码，**加法式重建** | 比特流前缀（熵编码） | 有（二值化/熵编码） | 任意码率（单模型） | 对象是**比特流**不是 token；有量化/熵编码；无 token 接口、无 rFID 口径 |
| **ALIT**（[2411.02393](https://arxiv.org/abs/2411.02393)） | **编码器**（每轮新增 latent、旧 latent 残差更新） | **并行、无状态**（本仓库 `ANALYSIS_alit_vs_assocvtc.md` 已核对源码） | 写侧（可读键 32→256） | 有（共享 1D codebook） | halt 准则选轮数 | **镜像关系**：它循环在写侧、解码器并行；我们循环在读侧、编码器单次。它每轮还要重编码整图，我们码不动 |
| **CaTok**（[2603.06449](https://arxiv.org/abs/2603.06449)） | 解码器的 **MeanFlow 时间步** | flow 解码器 | 时间步 ↔ token 区间 | 有（VQ 底座） | 区间截断 | 机制最接近，但它用 flow 时间维 + 量化；我们用纯 Transformer 读出步 + 连续码 |
| **NesTok / FlexTok**（[2609.36756](https://arxiv.org/abs/2609.36756) / [2502.13967](https://arxiv.org/abs/2502.13967)） | 无循环（每档独立解码） | ViT / flow | 码侧嵌套（前缀） | 有 | 任意前缀 | 它们是"码侧可变"，我们是"读出侧可变"；它们没有 carry 与 BPTT |

### 7.3 判定：实验能不能继续？

**能继续，但"循环重建"不能再作为卖点，只能作为"被检验的设计选择"。** 三条依据：

1. **没有完全雷同的已发表工作**（组合级差异），所以不存在"重复劳动"式的致命雷同；
2. **但思想层已被覆盖**（Toderici 的加法式循环重建、ALIT 的循环重建 + halt、CaTok 的步↔区间、多长度深监督），继续把"提出了循环重建"写成贡献，审稿人一定会打回；
3. **我们自己的数据目前对"循环"也不利**：单步 $T=1$ 的 rFID 6.13 明显好于 16 步循环的 19.97（虽然训练预算不对齐：45 epoch vs 6 epoch），若不做**同预算对照**，循环的收益无法被证明——这才是这条线"能不能继续"的真正问题，而不是文献雷同。

### 7.4 决定性实验（做完才能决定主线去留）

统一条件：同一个 32-token 连续码、同一编码器/解码器参数量、同数据（IN-100 224）、同训练预算、同一评测（rFID + PSNR/L1，逐档 2…32）：

| 臂 | 设计 | 已有？ | 作用 |
|---|---|---|---|
| **A 现行** | 画布 carry + BPTT，$T=16$ | ✅ 已有（rFID 19.97 / PSNR 27.94） | 现行方案 |
| **B 无 carry 独立档** | 同权重共享，第 $t$ 步只吃前缀、**不带** $\hat{Y}_{t-1}$，同 16 步同预算训练 | ❌ **必须补** | 分离"多前缀深监督"与"循环 carry"的贡献——这是判定循环是否必要的唯一干净对照 |
| **C 单步全码** | $T=1$ 一次读满 32 token | ✅ 有阶段 1（rFID 6.13），但 45 epoch 从零，预算不对齐 | 需要"6 epoch 同预算"版本才能公平比 |
| **D carry 但 detach** | 保留 carry、切断跨步梯度 | ✅ 旧 576-token 配置有（L1 23.72→16.42 表明 carry 梯度重要），新 240M/K=32 **未做** | 证明收益来自"梯度通路"而非"多步结构" |

**判据（建议先写死，避免事后解释）**：
- 若 **A ≈ B**（逐档 rFID 相对差 < 5% 或 PSNR 差 < 0.2 dB）⇒ **循环没有独立价值** ⇒ 主线应改为"多前缀深监督 + 读出侧截断"，循环只作为实现细节；此时 T=24 之类的长循环训练应停，把算力转去补 B/C 对照与下游探针。
- 若 **A 明显优于 B**（例如 32-token 档 rFID 降 >10%、或低档收益更大）⇒ 循环可作为**贡献点之一**（"一次训练、一个码、按读出步覆盖多预算"），并用 D 臂 + 旧 detach 证据解释机制。
- 若 **B 优于 A**（可能！多前缀独立解码避免"串联草稿链"误差累积）⇒ 那是一个更有意思的负结果，论文主线改成"为什么循环读出在多预算下不如独立多档"，同样可发表，但不能叫"循环读出压缩"。

**成本**（按本仓库既有配方估，需实测）：B/D 臂可在阶段 1 权重上按阶段 2 配方（6 epoch、bs 64、11,406 步）重训——63M 档约 1.5–2 h/臂（PLAN 里 B1 估 ~1.6 h），240M 档约 6–8 h/臂。建议**先在 63M/119M 档跑 B/D**（便宜、机制问题与容量弱相关），有信号再上 240M。C 臂的"同预算版"约 1–2 h。

### 7.5 写作层面的最小改动（若继续）

1. 把"循环重建"从贡献句里拿掉，改写成"我们把**变长能力放在读出侧**，并检验循环 carry 是否必要"；新颖性主张收窄为"读出侧可截断 + 无量化 + 受控分析"。
2. §2 补 Toderici（已引，补 additive/scaled-additive reconstruction 一句）、PLONQ、ALIT（已引，补镜像关系）、CaTok、NesTok/FlexTok/One-D-Piece/DetailFlow（多长度深监督）。
3. 仓库已有的 `ANALYSIS_alit_vs_assocvtc.md` 的"镜像论证"（ALIT 循环在编码器、解码器并行无状态；我们反之）可以直接搬进 §2/§3，这是目前**最有力、且已被源码核验**的差异化表述。
4. §5 局限加两句：未与 Toderici 式渐进编码器做比特率级比较；未做"无 carry 独立档"对照前，不主张循环本身带来增益。
