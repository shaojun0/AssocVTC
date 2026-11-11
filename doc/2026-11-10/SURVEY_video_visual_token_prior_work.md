# 视频视觉 Token 编码/压缩：我们的想法 vs. 先验工作（撞车检查）

> 调研对象：把「图像 → 32 token → 4:1 merge → 8 档预算前缀（k=1..8）」的 VLM 视觉编码器无缝切到视频，用①时序冗余、②按重建精度平台阈值自适应每帧 token 数、③第 t 帧在 t−1 帧解码器状态上重建、④同一套权重图像=单帧视频。
> 日期：2026-11-10。方法：arXiv / CVF / NeurIPS / ICLR 官方页面逐条核实；**只写能打开原文的**，打不开或没核实的明确标注。

---

## 1. 一句话结论

**最接近 ElasticTok（arXiv 2410.08368）**——它已经同时做了「①用前帧做条件 + ②每帧变长 token + ③按重建阈值在推理时定长度」，**逐条命中我们的第 1/2/3 点**；在"进了 VLM"这一侧最接近 **AdaCodec（arXiv 2606.02569，NeurIPS 2026）**——它在 video MLLM 里显式做 I 帧/P 帧预测编码，并**用预测代价阈值决定何时插 I 帧**，命中我们的第 1/3 点。

**雷同度：中高（与已发表工作重叠约 60–70%）。** 四个子点里没有一个是全新的；但**这个特定组合**——「已经训好的 VLM + 离散 8 档预算投影器 + **嵌套前缀一次解码**就能拿到全部档位的重建误差 + 按**边际增益**阈值选档 + 上一帧**解码器状态**作条件」——**未找到直接同款**。

别人没做的（可辩护的窄缝）：
- **(a)** 把「重建误差平台」判据**直接接到 VLM 的预算档位选择器上**（现有工作都在 tokenizer 阶段：ElasticTok/AdapTok/InfoTok；或在 MLLM 侧但按语义聚类复杂度：DiVT）。
- **(b)** 跨帧复用的是**解码器状态**，不是像素域运动补偿（AdaCodec）、也不是特征层 KV cache（ElasticTok/PVC）。
- **(c)** 用**相对边际增益** ΔAcc<ε 而非绝对重建误差阈值（ElasticTok/ALIT 都是绝对阈值），且**不为选档多跑前向**（ElasticTok 要二分搜索 ≈ log 次解码，AdapTok 要额外训 scorer + 解 ILP）。

⚠️ 三个必须先自我修正的硬伤，见 §5.2（尤其：**重建需要解码器，与"不引入新大模块"直接冲突**；以及 **AdapTok 已公开发文批评"固定阈值"方案**）。

---

## 2. 最接近的 5 个先验

### 2.1 ElasticTok：Adaptive Tokenization for Image and Video（arXiv 2024-10，v2 2025-02）🔥 最像

- 链接：https://arxiv.org/abs/2410.08368 ｜ HTML：https://arxiv.org/html/2410.08368v2 ｜ 项目页：http://largeworldmodel.github.io/elastictok
- 作者：Wilson Yan, Vlad Mnih, Aleksandra Faust, Matei Zaharia, Pieter Abbeel, Hao Liu（UC Berkeley / Google DeepMind / CMU）。**会议未核实，按 arXiv 引**。
- 核心机制：训练时对每个 block **随机采样保留 token 数 ℓ~U{Mmin..Mmax}，把尾部 token 掩掉**（tail-drop），编码器额外接收这个 mask；多 block 时用 **block-causal mask**，并把前序 block 的 **KV cache** 传下去，于是**每一帧都是在前面帧的基础上编码**。推理提供两种模式：**指定长度**、或**指定"目标重建阈值"**（target reconstruction threshold，如 MSE=0.003），后者用 Full/Binned/Binary Search 或训一个回归网络来找"满足阈值的最短长度"。
- 原文两句话最关键：
  - "During inference, ElasticTok can dynamically allocate tokens when needed – **more complex data can leverage more tokens**, while simpler data only needs a few."
  - 图 5 说明："**Different visual inputs will saturate reconstruction quality at a different number of tokens**" —— 这正是我们第 2 点的前提（重建精度存在平台，且平台位置随内容变）。
- 与我们逐条对比：
  1. **时序冗余假设**：✅ 完全相同（"conditions on prior frames"，多 block 因果 + KV cache）。
  2. **按重建误差阈值截断每帧预算**：✅ **机制几乎相同**（阈值 + 搜索最短满足长度 + 每帧独立定长）。差别仅在判据形式：它是**绝对**重建误差阈值，我们要用**边际增益**；它在 tokenizer 的 1D latent 前缀上截断，我们在 4:1 merge 后的 8 档预算前缀上截断。
  3. **第 t 帧在 t−1 帧解码器状态上重建**：🟡 **效果等价、实现不同**。它是"因果注意力 + KV cache 复用"，不是"复用解码器隐状态/参数"；也没有"P 帧残差"的显式形式。
  4. **图像=单帧视频、同一套权重**：✅ 完全相同（原文："images are considered as 1 frame videos"，训练时把图重复 4 次凑 block）。但它**没有**"不引入新模块"这条卖点——它本身就是重新训的 tokenizer（200M 参数 ViT 编解码）。
- 一句话：**我们的 1/2/3/4 在 ElasticTok 里都能找到对应句，只是它不在 VLM 里、也不是 8 档预算**。这是审稿人最可能甩给我们的第一篇。

### 2.2 AdapTok：Adaptive & Temporally Causal Video Tokenization in 1D Latent Space（CVPR 2026）🔥 最像的"改进版"，且已批评过阈值法

- 链接：CVF https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html ｜ arXiv：https://arxiv.org/abs/2505.17011 ｜ 代码：https://github.com/VisionXLab/AdapTok
- 作者：Yan Li, Changyao Tian, Renqiu Xia, ... Hongsheng Li, Jifeng Dai, Hao Li, Xue Yang（SJTU / MMLab CUHK / OpenGVLab）。CVPR 2026, pp. 16163-16172。
- 核心机制：block-causal 编解码 + **block-wise tail-drop** 训练变长；**block causal scorer** 一次前向**预测"用不同 token 数时该 block 的重建质量"**（用 perceptual loss 当 GT，MSE 回归）；推理时 **IPAL：解整数线性规划**，在**给定全局总预算**下把 token 分给各 block（min Σ 预测分数，s.t. 每样本选一个长度、总量 = B·N_b）。
- **它对我们最致命的一句**（§2.2 原文）："ElasticTok applies tail dropping ... and supports adaptive inference by selecting token counts via a fixed threshold. **However, ... relying on a fixed threshold prevents token allocation under a specified global budget.**" —— 也就是说**"按阈值截断"这个做法已经被点名批评过**（不能保证总预算/会跨样本不均衡）。我们必须正面回应，不能假装它是新的。
- 与我们逐条对比：
  1. 时序冗余：✅（block-causal，"preceding frames 独立于后续帧"）。
  2. 按重建误差定每帧预算：✅ **但它是"预测"重建质量再全局分配**，我们是"直接测"重建质量再阈值截断。它是**全局预算约束**下的最优分配，我们是**每帧独立的最小充分预算**——这是最实质的区别，也是我们可以反打它的地方（见 §5.1）。
  3. 第 t 帧在 t−1 帧状态上：🟡 因果注意力实现，不是解码器状态复用。
  4. 图像=单帧视频：🟡 它主要做视频重建/生成，**没有**"统一图像与视频、图像零损失"的卖点。

### 2.3 ALIT：Adaptive Length Image Tokenization via Recurrent Allocation（ICLR 2025）🔥 图像侧的"阈值"祖先

- 链接：https://arxiv.org/abs/2411.02393 ｜ ICLR 2025 proceedings PDF：https://proceedings.iclr.cc/paper_files/paper/2025/file/534ec8dfc0312d96e1c1b357304fadc2-Paper-Conference.pdf ｜ 代码：https://github.com/ShivamDuggal4/adaptive-length-tokenizer
- 作者：Shivam Duggal, Phillip Isola, Antonio Torralba, William T. Freeman（MIT）。
- 核心机制：encoder-decoder **循环 rollout**，每轮把 2D token 蒸馏进 1D latent，并**逐轮新增 token**（32→256），单一模型覆盖变长；"token 数对齐图像熵/熟悉度/下游需求"，报告 per-image 的 rate–distortion / Pareto 曲线。
- 与我们逐条对比：
  1. 时序冗余：❌ 纯图像，无跨帧。
  2. 按重建误差截断预算：✅ **图像侧的同一思想**（用重建质量曲线选长度；本仓库此前的核查笔记 `assocvtc-novelty-check.md` 记录其判据为"重建损失<阈值"，本次未逐页复核该句，引用前请回原文 §4 确认）。
  3. 跨帧状态复用：❌。
  4. 图像=视频：❌（它是图像 tokenizer，不接 VLM）。
- 意义：**"用重建质量决定传多少 token"这件事在图像侧 2024-11 就已经发表**。我们如果在论文里把它当新意，会被直接打回。

### 2.4 AdaCodec：A Predictive Visual Code for Video MLLMs（NeurIPS 2026）🔥 VLM 里的 P 帧

- 链接：https://arxiv.org/abs/2606.02569 ｜ HTML：https://arxiv.org/html/2606.02569v1 ｜ 项目页：https://HaowenHou.github.io/AdaCodec-Page/ ｜ NeurIPS 2026 poster：https://neurips.cc/virtual/2026/loc/sydney/poster/148821
- 作者：Haowen Hou, Zhen Huang, ... Nan Duan, Jiaqi Wang（SJTU / 上海创智 / JD）。基于 Qwen3-VL-8B。
- 核心机制：把视频 MLLM 的视觉接口改成**预测码**——按 **adaptive GOP** 切分，每个 GOP 一个 **I 帧**（走完整 ViT，全量 token）+ 若干 **P 帧**（走 **P-tokenizer**：输入是 5 通道 `[残差 r_t, 运动矢量 M_t]`，输出 **N_P ≪ N_I** 个 token）。**I 帧的插入位置由阈值 γ 决定**：逐帧算预测代价 `ℓ_t = Σ_b SAD(块, 参考块) + λ‖d‖₁`，`ℓ_t > γ` 就开新 GOP（即"预测不准 → 花全量 token"）。Stage-1 用冻结教师特征对齐训 P-tokenizer，Stage-2 冻结视觉侧只训 LLM。
- 与我们逐条对比：
  1. 时序冗余假设：✅ 完全相同，且它把"何时假设不成立"处理得比我们细（**scene change 用 pcost 阈值自动切 I 帧**；我们方案里只说"假设非剪辑"，这是我们的弱点）。
  2. 按误差定预算：🟡 **有阈值，但阈值作用在"预测代价 → I/P 二值决策"上**，不是"每帧连续预算"。P 帧的 token 数是固定的 N_P，**没有**"再加一个 token 收益 < ε 就停"的平台判据。这是**我们唯一比它细的地方**。
  3. 第 t 帧在 t−1 帧基础上重建：✅ **在 VLM 里做到了**，而且是显式 P 帧（运动补偿 + 残差）。差别：它靠**像素域运动搜索**（引入新 tokenizer 模块），我们主张**复用解码器状态**（不引入运动估计）。
  4. 图像=单帧视频：🟡 未强调统一；它的 I 帧路径等于单帧，但没有"图像零损失"的实验叙事。
- **这是"我们第 3 点已经被人做进 VLM"的直接证据**，而且发在 NeurIPS 2026。任何"我们是第一个在 VLM 里做预测编码"的说法都不能写。

### 2.5 PVC：Progressive Visual Token Compression（CVPR 2025）🔥 "图像=静帧视频"已被做过

- 链接：https://arxiv.org/abs/2412.09613 ｜ HTML：https://arxiv.org/html/2412.09613v1 ｜ 代码：https://github.com/OpenGVLab/PVC
- 作者：Chenyu Yang, Xuan Dong, Xizhou Zhu, ... Jifeng Dai（清华 / OpenGVLab / SenseTime）。基于 InternVL2。
- 核心机制：**把每个图像重复 4 次变成"静态视频"**，所有输入统一成视频；ViT 最后 L 层插入**因果时序注意力**（每个 patch 只 attend 到前帧同位置）+ **AdaLN（注入 timestep）**，使"当前帧只编码前帧没提出的新信息"；压缩模块是 **PixelShuffle(16×) + AdaLN + 共享 MLP**，默认 **64 token/帧**。
- 与我们逐条对比：
  1. 时序冗余：✅ 完全相同（原文明写"adjacent frames are often similar and naturally have temporal redundancy... progressive compression"）。
  2. 按重建误差定预算：❌ **完全没有**——固定 64 token/帧，无重建目标（端到端为 VLM 训）。
  3. 跨帧状态复用：🟡 因果时序注意力 = 特征层"只补新信息"，**不是**解码器状态，也没有像素重建。
  4. 图像=单帧视频：✅ **一模一样的主张，2024-12 就发了**。差别：PVC 必须**改编码器结构**（加时序注意力 + AdaLN + 16× PixelShuffle），我们主张**零新模块**。
- 结论：**我们的第 4 点（"图像=单帧视频、不引入新大模块"）不能单独当卖点**，PVC 已经占了"统一"这半句；能主张的只有"统一且不改结构"。

### 2.6 补充：还应该引但不必当"最接近"的

| 工作 | 年份/出处 | 为什么相关 |
|---|---|---|
| **InfoTok** https://arxiv.org/abs/2512.16975 | ICLR 2026（Oral） | 用信息论（ELBO）证"固定码率次优"，做**自适应视频 token 数**；省 20% token、2.3× 压缩。是"自适应预算"最强的理论版对手 |
| **DiVT: A More Word-like Image Tokenization for MLLMs** https://arxiv.org/abs/2605.17954 | CVPR 2026 | **在 MLLM 侧**按图像复杂度自适应 token 预算，且"不改视觉编码器、不改 LLM"。直接对标我们第 4 点的"零模块" |
| **DynTok** https://arxiv.org/abs/2506.03990 | arXiv 2025 | 视频 token 分组+组内合并，降到 44.4%；低信息区压更狠。是"时间维 merge"基线 |
| **Matryoshka Multimodal Models** https://arxiv.org/abs/2405.17430 | ICLR 2025 | **嵌套 token，一次前向拿到多个预算**——我们"避免 8× 前向"的机制原型 |
| **FlexTok** https://arxiv.org/abs/2502.13967 | ICML 2025 | 1D 变长 token，任意前缀可解码（1..256），nested dropout |
| **CoPE-VideoLM** https://arxiv.org/abs/2602.13191 ｜ **ReMoRa** https://arxiv.org/abs/2602.16412 | 2026 / CVPR 2026 | 直接吃标准码流的 codec primitives（GOP、运动矢量）喂 MLLM。是"P 帧进 VLM"的另一条工程路线（不改 codec，只学怎么消费） |
| **Video-XL** https://arxiv.org/abs/2409.14485 ｜ **VideoChat-Flash** https://arxiv.org/abs/2501.00574 | CVPR 2025 / 2025 | 长视频主流省法：视觉上下文 latent 摘要、层次化压缩（不是每帧预算） |

---

## 3. 「按重建误差阈值截断每帧预算」这个机制，谁做过？

### 3.1 直接做过的（按相似度排）

| 工作 | 判据 | 作用对象 | 与我们差在哪 |
|---|---|---|---|
| **ElasticTok**（2410.08368） | **绝对**重建阈值（如 MSE 0.003），二分/全搜索找最短满足长度 | **每帧**（block）的 1D latent 前缀 | 最像。差：不在 VLM 预算档位上；阈值是绝对的；搜索要多次解码 |
| **AdapTok**（2505.17011） | **预测**的重建质量（perceptual loss）+ **全局预算** ILP | 每 block | 差：它是"预测后全局分配"，我们是"实测后逐帧截断"；且它明确批评固定阈值不能控总预算 |
| **ALIT**（2411.02393） | 重建质量曲线 / 阈值（图像） | 每图 | 差：纯图像、无跨帧 |
| **InfoTok**（2512.16975） | 信息量（ELBO）而非重建误差 | 每段 | 差：判据是信息论量，不是"平台" |
| **DiVT**（2605.17954） | 语义聚类复杂度（非重建） | 每图，MLLM 侧 | 差：判据不是重建误差 |
| **RVQ 变码率线**（见 3.2） | 码本深度 / 量化器个数 | 同一帧的**码本层数** | 差：**同一帧的深度** vs 我们的**跨帧 + 预算档位** |

### 3.2 RVQ 提前终止：最像的"技术动作"，但不是同一件事

- **SoundStream**（arXiv 2107.03312；TASLP 2021, DOI 10.1109/TASLP.2021.3129994）提出 **quantizer dropout**：训练时随机只用前 n 个 RVQ 级，于是**一套权重支持 3k–18kbps 多档码率**——这就是"按需截断码本深度"的原始做法，和我们的"8 档前缀"在**形式上同构**。
- **EnCodec / DAC** 同样用"改 num_quantizers 即改码率"做变码率（工程上是标准操作）。
- **VRVQ: Variable Bitrate Residual Vector Quantization for Audio Compression**（NeurIPS 2024，slides：https://neurips.cc/media/neurips-2024/Slides/98231.pdf）专门做变码率 RVQ。
- 还有一篇**负面结果**值得看：*Early Stopping in Residual Quantization: An Empirical Negative Result*（TechRxiv）——说明"按误差提前停 RVQ"**不一定有效**，这类负面证据审稿人可能会拿来打我们。
- **关键差别（必须写进论文）**：
  - RVQ 提前终止发生在**同一帧的码本深度轴**上，rate 是真实比特，且有熵编码/码率失真理论支撑；
  - 我们发生在**跨帧的预算档位轴**上，rate 只是"token 数"，**没有熵模型 ⇒ 不是真实比特**（这是 ALIT/ElasticTok 同类工作的共同软肋，审稿人一定会问 bpp）。
  - 换句话说：**动作像 RVQ，但语义是"率失真工作点选择"，不是"量化深度选择"**。写成"我们把 RVQ 的 early-termination 搬到跨帧预算"会被认为牵强；写成"我们把码率-失真工作点选择从码本深度轴搬到跨帧 token 轴"更准确。

### 3.3 结论

**"按重建误差阈值截断每帧 token 预算"没有被漏掉，ElasticTok 已经做了，而且 AdapTok 已经公开批评过它的缺陷。** 我们能主张的只有两点：
1. 判据换成**相对边际增益**（`(L(t)−L(t+1))/L(t) < ε`），对"平台"更鲁棒，且不需要预先知道绝对误差量级；
2. 判据作用在**已训好的 VLM 的 8 档预算前缀**上（**一次解码即可拿到全部档位的误差**，因为 8 档是嵌套前缀），**不需要 ElasticTok 那样的 log 次搜索、也不需要 AdapTok 那样额外训 scorer**。

第 2 点是真优势，但它是**工程效率优势，不是新机制**——论文里要老实这么写。

---

## 4. 「跨帧解码器状态复用 / 预测编码」：谁做过，和 P 帧什么关系，进 VLM 的先例

### 4.1 经典预测编码线（我们是在重复 70 年的老思想）

- **H.264/HEVC/AV1 的 I/P/B 帧**：I 帧帧内编码，P 帧用**运动补偿 + 残差**相对前向参考帧编码，B 帧双向。**"第 2 帧在第 1 帧基础上重建"就是 P 帧的定义**，1950s DPCM 起就是了。
- **DVC / DCVC 系列（深度视频压缩）**：把运动估计、运动补偿、残差编解码、熵模型全部换成网络。DCVC（NeurIPS 2021，https://papers.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf）的关键是**条件熵模型**：用前帧的**重建**当条件，编当前帧的 latent。后续 DCVC-TCM/DCVC-FM/DCVC-HEM，以及 ICCV 2025 的 MH-LVC（多假设时序预测）都在这一支。综述/开源实现见 OpenDCVCs（https://arxiv.org/abs/2508.04491）。
- **Recurrent autoencoder for video compression（Toderici 线）**：CVPR 2017（arXiv 1608.05148）共享参数循环编解码 + BPTT + **每次迭代都有损失** + "任意迭代数 = 一条合法低码率流"。**这是"循环重建 + 每步损失 + 可截断"的结构祖先**。
- **与我们第 3 点的关系**：我们说的是"第 t 帧的解码在 t−1 帧解码器状态之上"。这在编码理论里就是 **P 帧 + 条件解码器**；在深度压缩里就是 **conditional decoder / conditional entropy model**。**思想层面 100% 已有**，没有任何新颖性可主张。

### 4.2 进 VLM 的先例（有，而且不止一个）

| 工作 | 怎么"用上一帧" | 层级 |
|---|---|---|
| **AdaCodec**（2606.02569, NeurIPS 2026） | 显式 I/P 帧；P 帧输入 = 像素域**运动矢量 + 残差**，5 通道进 P-tokenizer；pcost 阈值切 GOP | **视觉接口层**（最接近我们想要的） |
| **PVC**（2412.09613, CVPR 2025） | ViT 内**因果时序注意力**，"当前帧只编码前帧未提出的新信息" + AdaLN 区分 timestep | 编码器内部 |
| **ElasticTok**（2410.08368） | block-causal + **KV cache 传递**，"conditions on prior frames" | tokenizer 内部 |
| **CoPE-VideoLM**（2602.13191）/ **ReMoRa**（2602.16412, CVPR 2026） | 不改 codec，直接消费标准码流的 GOP/运动矢量 | 输入接口层 |
| **AdaCodec 提到的 EMA** | 用 I 帧 + 运动矢量构建 GOP 级表示 | 输入接口层 |

**未找到直接工作**：把"上一帧**解码器隐状态**"（而不是像素、不是运动矢量、不是 KV cache）作为第 t 帧重建/编码的条件，并把它接到 VLM 的 token 预算上。这一条是干净的缝，但**它是否真比"因果注意力 + KV cache"更好，必须有 ablation 才能说**——否则审稿人会说"你只是换了个说法实现 PVC/ElasticTok"。

### 4.3 一个必须承认的事实

"预测编码"这个词现在在 VLM 圈是**热词**：AdaCodec 的 intro 直接把 "predictive coding" 与生物视觉的 prediction error 挂钩，并引用 H.264/HEVC。**我们再讲一遍同样的故事，会被认为是 AdaCodec 的弱化版**（它是完整 P 帧系统，我们只是"复用状态"）。所以叙事必须避开"我们是预测编码"，转到"**我们是预算决策**"。

---

## 5. 差异化与审稿风险

### 5.1 可以写进论文的 3 条 claim（按可辩护强度排序）

**Claim 1（最强，工程向）："一次解码，全档位可选"——把 8 档预算从 8 次前向降为 1 次。**
现有做法要么**多预算重跑**（每档一次前向，我们的 8× 问题），要么**搜索**（ElasticTok 二分 ≈ log₂8=3 次解码 + 每次都要过编码器），要么**额外训一个 scorer**（AdapTok 的 block causal scorer + ILP）。我们的 8 档是同一串 8 个 token 的**嵌套前缀**（4:1 merge 后 k=1..8），所以**一次解码就能拿到 8 个前缀各自的误差**，取"边际增益 < ε 的最小 k"。这是**结构性优势**，且可测（TTFT / 解码次数 / FLOPs）。
→ 这条要配一张图：x 轴 k=1..8，y 轴重建误差，8 条曲线各自平台位置不同（对应 ElasticTok 图 5 的现象），标注"每条曲线来自同一次解码"。

**Claim 2（中等）：判据从"绝对误差阈值"改成"相对边际增益"，对平台更鲁棒，且天然带"够用就停"的语义。**
ElasticTok/ALIT 用绝对阈值（MSE 0.003 这种），**阈值量级依赖数据归一化和分辨率**，换个数据集就得重调；AdapTok 已经指出固定阈值无法满足全局预算。边际增益 `Δ = (L(k)−L(k+1))/L(k) < ε` 是**无量纲**的，且直接对应"平台"的定义。
→ 但要诚实：这只是**判据的细化**，不是新机制；而且 AdapTok 的批评（不能控总预算）对**边际增益同样成立**——必须有"每帧阈值 + 全片预算上限"的两级设计，否则这一条会被直接打回。

**Claim 3（中等偏弱，但别人确实没做）：图像与视频共用**同一套权重、同一个预算投影器、同一个阈值**，且不引入运动估计/时序注意力等新模块。**
PVC 统一了"图像=静帧视频"但**改了编码器结构**（时序注意力 + AdaLN + 16× PixelShuffle）；AdaCodec 引入了**新的 P-tokenizer**；ElasticTok 是**重训的 tokenizer**。我们主张"零新模块、图像指标不掉"。
→ 这条必须配"图像侧零退化"的硬实验（DocVQA/OCR 这类细节敏感任务），否则等于没做。

### 5.2 审稿人最可能打的 4 个点 + 回应

**① "这不就是 P 帧 / 残差量化吗？"（最可能、最致命）**
回应要点：
- P 帧优化的是**给人看的像素重建 + 比特流**；我们的"重建"只是**训练脚手架 + 预算判据**，产物是喂 LLM 的语义 token，**不做像素级传输**（我们甚至可以不传残差）。
- 我们**没有**运动估计/运动补偿模块，也没有熵编码；AdaCodec 有前者（SAD 搜索 + 六边形搜索），SoundStream 有后者。
- 但**必须承认**：条件结构（用前帧预测当前帧，只传"预测解释不了的部分"）与 P 帧同构。**诚实的写法是"借用了预测编码的条件结构，但目标函数与产物不同"**，而不是声称"我们发现了一个新结构"。

**② "ElasticTok 已经做了阈值 + 每帧变长 + 前帧条件；AdapTok 还指出固定阈值不行。你的新意在哪？"（第二致命）**
回应要点：
- 位置不同：它们在 **tokenizer**，我们在**已训好的 VLM 的预算投影器**（8 档离散、4:1 merge 之后）。这是"部署侧"而非"预训练侧"，代价（要不要重训编码器）完全不同。
- 效率不同：它们要搜索/要 scorer，我们**一次解码**（Claim 1）。
- **但**：如果我们的实验只是"在 ElasticTok 的思路上换个位置"，创新度会被判为 incremental。**必须做与 ElasticTok/AdapTok 的正面比较**（同预算下的重建曲线 + 下游 QA + 解码次数），这是硬要求。

**③ "重建精度和 VLM 下游性能有什么关系？"（方法论层面最致命）**
- 这是**我们方案的根本假设**：重建误差平台 ⇒ 语义够用。但 VLM 要的是**判别性语义**，不是像素保真。一个"重建已经很平"的帧，可能恰好缺了 OCR 需要的那点高频；反过来"重建还很差"的帧可能对 QA 毫无信息量。
- 已有反例方向：`压缩密度 ↑ ⇒ 幻觉 ↓` 只在膝点前成立（见本仓库此前核查的 [2602.02539] 相变结论）。所以"重建够了就停"很可能**停在语义不够的点上**。
- **必须补的实验**：同一帧上同时测"重建误差曲线"与"下游 QA/OCR 准确率曲线"，证明两者平台位置**一致或至少单调相关**。**这个实验不做，整篇论文的地基是空的。**
- 建议叙事改成"**找膝点（knee-seeking）**"而不是"省 token"：既解释了为什么不是越少越好，也把重建判据降级为"膝点的代理指标"。

**④ "你省了 token，但多了解码器——真的省算力吗？"（工程层面最容易被抓）**
- **我们的方案描述里没有解码器**（编码器→预算投影器→LLM），但"循环重建"必须有一个**能出像素的解码器**。这与"不引入新的大模块"**直接冲突**。
- 三条出路，必须在论文里选一条并明说：
  (a) 复用训练 tokenizer 时用的解码器，**只在训练/校准阶段用**，推理时用一个小 MLP 预测"边际增益"（等于 AdapTok 的 scorer，回到被批评的路线）；
  (b) 推理时**真的跑解码器**，但只跑 1 次（Claim 1 的嵌套前缀正好支持），并**把解码 FLOPs 计入总账**；
  (c) 把重建换成**特征空间重建**（对 LLM 输入 embedding 或对教师特征做回归），避免像素解码器——但这会削弱"重建精度"的可解释性。
- 无论哪条，**"不引入新大模块"这句话目前是站不住的**，要么改措辞（"不引入新的大**编码**模块"），要么补 (a)。

**⑤ 附带的两个小雷**
- **"假设视频帧连续、非剪辑"**：AdaCodec 用 pcost 阈值自动切 I 帧正是为了解决这个；ElasticTok 也观察到 scene change 处 token 用量飙升。我们如果只"假设"，会在真实 benchmark（VideoMME/MLVU 里有大量剪辑）上掉点。**至少要加一个 scene-cut 检测的 fallback**。
- **"8 档只有 8 个离散点"**：4:1 merge 后 32→8，k=1..8 只有 8 个操作点，**R-D 曲线的分辨率极低**，"平台"可能只落在 2–3 个点上，够不够判"边际增益"是实证问题。先做零成本验证：拿现有 eval 的 per-k 重建误差，看 8 个点是否单调、平台是否可辨。**如果不单调，整个方案不成立，越早验证越好。**

### 5.3 一句话怎么定位自己

> "我们不是提出新的视频 tokenizer，也不是第一个做预测编码的 VLM；我们把**预算决策**从'多档重跑/额外打分器'变成'**一次解码、按边际增益在嵌套前缀上选档**'，并证明**图像与视频可以共用同一套权重与同一套阈值**。"

这句话能站住，但它是一个**效率/系统论文**的定位，不是**新机制**的定位。如果要投机制向的会，需要补 §5.2③ 那个实验，把"重建平台 = 语义膝点"变成实证结论。

---

## 6. 参考链接

**最接近的 5 篇**
1. ElasticTok: Adaptive Tokenization for Image and Video — https://arxiv.org/abs/2410.08368 ｜ https://arxiv.org/html/2410.08368v2
2. AdapTok: Learning Adaptive and Temporally Causal Video Tokenization in a 1D Latent Space (CVPR 2026) — https://arxiv.org/abs/2505.17011 ｜ https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html ｜ https://github.com/VisionXLab/AdapTok
3. ALIT: Adaptive Length Image Tokenization via Recurrent Allocation (ICLR 2025) — https://arxiv.org/abs/2411.02393 ｜ https://proceedings.iclr.cc/paper_files/paper/2025/file/534ec8dfc0312d96e1c1b357304fadc2-Paper-Conference.pdf ｜ https://github.com/ShivamDuggal4/adaptive-length-tokenizer
4. AdaCodec: A Predictive Visual Code for Video MLLMs (NeurIPS 2026) — https://arxiv.org/abs/2606.02569 ｜ https://arxiv.org/html/2606.02569v1 ｜ https://neurips.cc/virtual/2026/loc/sydney/poster/148821
5. PVC: Progressive Visual Token Compression (CVPR 2025) — https://arxiv.org/abs/2412.09613 ｜ https://arxiv.org/html/2412.09613v1 ｜ https://github.com/OpenGVLab/PVC

**自适应预算 / 变长 token**
6. InfoTok (ICLR 2026) — https://arxiv.org/abs/2512.16975
7. DiVT: A More Word-like Image Tokenization for MLLMs (CVPR 2026) — https://arxiv.org/abs/2605.17954 ｜ https://openaccess.thecvf.com/content/CVPR2026/html/Lee_A_More_Word-like_Image_Tokenization_for_MLLMs_CVPR_2026_paper.html
8. Matryoshka Multimodal Models (ICLR 2025) — https://arxiv.org/abs/2405.17430
9. FlexTok (ICML 2025) — https://arxiv.org/abs/2502.13967
10. Toderici et al., Full Resolution Image Compression with Recurrent Neural Networks (CVPR 2017) — https://arxiv.org/abs/1608.05148

**视频 token 压缩 / 长视频**
11. DynTok — https://arxiv.org/abs/2506.03990
12. Video-XL (CVPR 2025) — https://arxiv.org/abs/2409.14485
13. VideoChat-Flash — https://arxiv.org/abs/2501.00574
14. Frame-Voyager (ICLR 2025) — https://proceedings.iclr.cc/paper_files/paper/2025/hash/d18d208fa9c333483e5724ade7beff0f-Abstract-Conference.html

**预测编码 / 视频压缩**
15. DCVC: Deep Contextual Video Compression (NeurIPS 2021) — https://papers.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf
16. OpenDCVCs（DCVC 系列开源实现与评测） — https://arxiv.org/abs/2508.04491
17. MH-LVC: Multi-Hypothesis Temporal Prediction for Learned Conditional Residual Video Coding (ICCV 2025) — https://openaccess.thecvf.com/content/ICCV2025/papers/Phung_MH-LVC_Multi-Hypothesis_Temporal_Prediction_for_Learned_Conditional_Residual_Video_Coding_ICCV_2025_paper.pdf
18. CoPE-VideoLM — https://arxiv.org/abs/2602.13191 ｜ https://microsoft.github.io/CoPE/
19. ReMoRa (CVPR 2026) — https://arxiv.org/abs/2602.16412

**RVQ 变码率 / 提前终止**
20. SoundStream (TASLP 2021，quantizer dropout) — https://arxiv.org/abs/2107.03312 ｜ DOI 10.1109/TASLP.2021.3129994
21. VRVQ: Variable Bitrate Residual Vector Quantization for Audio Compression (NeurIPS 2024) — https://neurips.cc/media/neurips-2024/Slides/98231.pdf
22. Early Stopping in Residual Quantization: An Empirical Negative Result (TechRxiv) — https://www.techrxiv.org/users/978315/articles/1345192

**视频 tokenizer（背景，不是最接近）**
23. MAGVIT-v2 / Language Model Beats Diffusion — https://arxiv.org/abs/2310.05737
24. Cosmos Tokenizer — https://research.nvidia.com/labs/cosmos-lab/cosmos-tokenizer/
25. VidTok — https://arxiv.org/abs/2412.13061 ｜ https://github.com/microsoft/vidtok
26. VideoGPT — https://arxiv.org/abs/2104.10157

---

## 附：核实状态

- **已打开官方页面确认标题/作者/编号**：ElasticTok(2410.08368)、AdapTok(2505.17011 + CVF)、ALIT(2411.02393 + ICLR25 PDF)、AdaCodec(2606.02569 + NeurIPS26 poster)、PVC(2412.09613)、InfoTok(2512.16975)、DiVT(2605.17954 + CVF)、DynTok(2506.03990)、Matryoshka(2405.17430)、DCVC(NeurIPS21 PDF)、Frame-Voyager(ICLR25)。
- **只看到摘要/片段、未逐页核实**：VRVQ（仅 slides）、FlexTok（经本仓库旧笔记转引，未本次复核）、Video-XL/VideoChat-Flash/CoPE-VideoLM/ReMoRa（仅搜索片段与摘要级）、MAGVIT-v2/Cosmos/VidTok/VideoGPT（背景引用，未逐篇读）。
- **未找到直接工作**：(i) 在已训好 VLM 的**离散预算档位**上用**边际增益阈值**逐帧选档；(ii) 复用**上一帧解码器状态**（非运动矢量、非 KV cache）作为下一帧重建条件；(iii) 同时满足"图像=单帧视频 + 零新模块 + 重建判据"的完整组合。
- **明确风险**：`assocvtc-novelty-check.md` 中关于 ALIT"用重建损失<阈值当判据"的一句，本次**未回原文复核**，引用前请确认。
