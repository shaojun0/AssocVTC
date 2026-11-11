# 视频中间态（StateGen）改造设计：砍掉软拷贝 × KV 常驻初始化向量

> 日期: 2026-11-11 ｜ 性质: 决策 + 实现规格（handoff 给下一个会话执行）
> 执行位置: `/home/linaro/dsh/video_recon_probe`（实验代码仓，非本仓）；本文只作决策与规格存档
> 结论分级: **[数据]** = 实测/落盘统计支撑；**[推断]** = 推理，附可证伪点
> 前置阅读: `REPORT_中间态生成基底_小规模测试_20261110.md`、`REPORT_逐步收紧预算分配_文献与离线回放_20261111.md`（同目录在实验仓）

---

## 0. 决策速览（TL;DR）

1. **[数据] 砍掉"最近 ≤16 帧软拷贝"路径（`Δ_copy`）。** 它在 G2 checkpoint 上已退化为"52% 抄上一帧 + 35% 不抄 + 其余模糊平均"，period 处质量 3.9%/1.4%（低于均匀基线 6–11%），既视感完全没实现；且"抄上一帧"在 revisit 片段上是 −1.08/−1.85 dB 的负收益。
2. **[数据] 保留"抄"这个归纳偏置的价值判断**：probe 显示抄匹配帧 +3.33/+3.65 dB、抄上一帧 +2.6~3.4 dB（pan/shake/cut）。所以问题在"选哪一帧"这条通道坏了，不是"抄"这个想法坏了。
3. **新结构（本文件要实现的）**：交叉注意力常驻一个可学习的**初始化 KV**——
   - 无历史：`KV = 初始化向量`，`Q = n 个初始化向量`（= state tokens）；
   - 有历史：`KV = 初始化向量 + 历史向量组`，`Q = 上一帧中间态输出`；
   - `Δ` 只经由 patch 级模板注意力从 `KV` 读出，`base_out` 保持零初始化。
4. **[数据] 训练预算保持随机**（`--budget-mode random`），不再做"探索期→贪婪期"；λ/影子价格分配器暂缓（现工作点分配器上界仅 +0.05~0.09 dB，见预算分配报告）。
5. **图像一致性策略**：默认 `--cold-delta off`——无历史帧仍然跑注意力（KV 里有初始化向量）但**不把 Δ 注入 `Y_0`**，从而保住"图像=单帧视频逐位一致"（`max|Δ|=0` 仍可断言）。要试"冷帧也吃先验"再开 `--cold-delta on`。

---

## 1. 现状（被砍掉的实现，逐行）

文件 `video_recon_probe/model_state.py`，`StateGen`：

| 位置 | 内容 |
|---|---|
| `:92` | `init_state`：可学习 `(1, tokens=8, 384)`，仅在 `s_prev is None` 时作为 Q |
| `:93` | `age_pos`：可学习 `(max_frames=64, 384)`，帧龄位置编码，`age_pos[0]`=上一帧 |
| `:96-98` | `layers`：`StateLayer`×depth(1)，自注意力 + 交叉注意力(历史) + FFN |
| `:99-103` | `base_queries (196,384)` + `base_cross` + `base_out`（零初始化） |
| `:107-113` | `lag_gate Linear(384,16)`（bias 初始 `[2,-2,...]`）、`null_gate`（零初始化） |
| `:114-119` | `cue_q/cue_k`（identity 初始化）、`cue_scale=8` |
| `:121-123` | `gain_gate`（零初始化）⇒ `gain=1+softplus(·)` 初始 1.693 |
| `:152-159` | `ctx = cat([hist_j + age_pos[n-1-j]])` → LN/Linear → `StateLayer(s_prev, ctx)` |
| `:161-178` | **软拷贝**：`usable=min(n,16)`；`scores=lag_gate(smean)[:,:usable].flip(-1) + cue_scale·cos(cue_q(cue_f), cue_k(cue_j))`；`w=softmax([null, scores])`；`Δ_copy = gain·Σ_j w_j R_j` |
| `:179-185` | `Δ = Δ_copy + base_out(base_cross(base_queries + s.mean(1), ctx))` |
| `:145-151` | `hist` 为空（首帧/场景切换/15% 冷帧）⇒ `Δ=None`，`Y_0 = query_base + query_pos`（与图像模型逐位一致） |

`hist` 里放的是每帧一个向量：`U = recon_vectors(sts[n-1]) = norm_out(carry_n − query_base)`（`train_state.py:185`）；`n` 为预算规则选出的步数。`cue/cue_hist = z_cls`，由 `train_state.py:144-147`、`eval_state.py:143-160/225-231` 传入。

---

## 2. 砍掉软拷贝的实测证据（G2 ckpt，`analysis/eval_state_g2/results.json` 的 `attn`，964 帧）

**[数据] 软拷贝 softmax 权重（去掉 null 后归一化，按帧龄）**

| 模式 | age0 | age1 | age2 | age3 | age4 | age5 | period 处质量 |
|---|---|---|---|---|---|---|---|
| static | 0.491 | 0.085 | 0.030 | 0.111 | 0.033 | 0.120 | — |
| fast | 0.716 | 0.103 | 0.023 | 0.087 | 0.017 | 0.056 | — |
| cut | 0.642 | 0.138 | 0.026 | 0.120 | 0.025 | 0.077 | — |
| revisit8 | 0.414 | 0.103 | 0.008 | 0.107 | 0.012 | 0.144 | **0.039** |
| revisit12 | 0.413 | 0.132 | 0.014 | 0.125 | 0.022 | 0.111 | **0.014** |
| real | 0.492 | 0.052 | 0.012 | 0.059 | 0.013 | 0.054 | — |

- `top_age = 0`（上一帧）占 **929/964 = 96.4%**；`null_w = 0.352 ± 0.150`；归一化熵 0.627；top1 份额 0.565。
- period 处质量 **低于均匀基线**（K=8~16 ⇒ 均匀 1/K = 6–12%），说明内容检索通道没有把质量挪到 `t−period`。
- **[数据] 未解释异常**：奇数帧龄（1,3,5）≈0.09–0.10，偶数帧龄（2,4）≈0.02–0.03；该奇偶效应与窗口长度奇偶无关（`K%2 × a%2` 四格都成立），real 视频上同样存在 ⇒ 不是 `flip` 对齐伪影，来自 cue 相似度/输入的 2 帧周期结构。**实现前可花 10 分钟复查一次**（但已不影响砍掉的决定）。
- **[数据] 另一条路径也不选择**：`base_cross` 的帧级注意力质量近乎均匀（age0 0.219 → age15 0.063，归一化熵 0.94–1.0），等于"对全部历史做平均"。

**[数据] 机制本身的价值（probe，冷启动基线逐帧）**

| 抄什么 | static | slow | mid | fast | shake | cut | revisit8 | revisit12 |
|---|---|---|---|---|---|---|---|---|
| 抄上一帧 R | +2.97 | +3.20 | +3.18 | +2.57 | +3.28 | +3.41 | **−1.08** | **−1.85** |
| 抄匹配帧（period） | — | — | — | — | — | — | **+3.33** | **+3.65** |
| z_cls 检索 top-1 命中率 | — | — | — | — | — | — | **0/48** | **0/24** |

最优拷贝幅度 a≈1.4–2（与 `gain` 初始 1.693 一致）⇒ 机制表达力没问题，**坏在"选帧"通道**。

---

## 3. 目标结构（本文件要实现的设计）

保留 `StateLayer`（状态递推）与 `base_cross`（patch 级模板读出），**删除软拷贝**，并让交叉注意力始终带一个可学习的"初始化 KV"：

```
K/V 组成（每帧）:
    ctx = kv_init                       # (B, KV0, d)  可学习，无帧龄编码
    if hist:  ctx = cat([kv_init, hist_ctx])   # hist_ctx = Σ_j (R_j + age_pos[n-1-j])，仅最近 max_lag 帧
Q 组成:
    s_q = init_state (B,8,d)             # 无历史（s_prev is None）
        = s_prev (B,8,d)                 # 有历史 = 上一帧中间态输出
状态更新:
    s = StateLayer(s_q, ctx) → norm
Δ 读出:
    Δ = base_out( base_cross( base_queries(196) + s.mean(1), ctx ) )   # base_out 零初始化
Y_0 = query_base + query_pos + Δ        # 无历史且 --cold-delta off ⇒ Δ 强制 None
```

对应"你原话"的映射：

| 原话 | 代码语义 |
|---|---|
| 无历史时 KV 输入是一个初始化向量 | `ctx = kv_init`（可学习；`KV0` 建议 1，可配 4） |
| 无历史时 Q 输入是 n 个初始化向量 | `Q = init_state`，`n = tokens = 8`（**与帧数无关**；注意别把 n 理解成帧数） |
| 有历史时 KV 输入是初始化向量 + 历史向量组 | `ctx = cat([kv_init, hist_ctx])` |
| 有历史时 Q 输入是上一步中间态输出 | `Q = s_prev` |

---

## 4. 实现规格（按文件）

### 4.1 `model_state.py`

**删除**
- `lag_gate`、`null_gate`、`cue_q`、`cue_k`、`cue_scale`、`gain_gate` 的定义与初始化（`:107-126`）
- `step()` 中 `:161-178` 的软拷贝计算；`cue`/`cue_hist` 形参可保留但标记 deprecated（减少改 `train_state.py/eval_state.py` 的摩擦），内部不再使用
- `last_frame_w`/`last_frame_ages`/`last_gain` 三个属性（或保留名字改语义，见 4.3）

**新增**
```python
self.kv_tokens = int(kv_tokens)                     # 默认 1
self.kv_init = nn.Parameter(torch.randn(1, self.kv_tokens, dec_dim) * 0.02)
```
- `kv_init` **不加帧龄编码**；`age_pos` 只作用于历史部分（保持 `age_pos[0]`=上一帧的语义）。
- `StateGen.__init__` / `state_cfg_from_base` / `build_state_model` 增加 `kv_tokens` 参数（默认 1），`cfg` 里记录。

**step() 新逻辑（伪代码）**
```python
def step(self, s_prev, hist, cue=None, cue_hist=None, return_attn=False):
    B = ...
    init_ctx = self.kv_init.expand(B, -1, -1)                     # (B, KV0, d)
    if hist:
        n = len(hist)
        hctx = torch.cat([h + self.age_pos[n-1-i].view(1,1,-1)
                          for i, h in enumerate(hist)], dim=1)
        hctx = self.hist_proj(self.hist_norm(hctx))
        ctx = torch.cat([init_ctx, hctx], dim=1)
    else:
        ctx = init_ctx
    s = self.init_state.expand(B,-1,-1) if s_prev is None else s_prev
    for layer in self.layers:
        s = layer(s, ctx)
    s = self.norm(s)
    if not hist and self.cold_delta == "off":      # 默认：保图像逐位一致
        return s, None, None
    q = self.base_queries.unsqueeze(0).expand(B,-1,-1) + s.mean(1, keepdim=True)
    if return_attn:
        d_attn, attn = self.base_cross(q, ctx, return_attn=True)
        return s, self.base_out(d_attn), attn
    return s, self.base_out(self.base_cross(q, ctx)), None
```
- `cold_delta` 由构造参数/全局开关传入（`"off"|"on"`），默认 `"off"`。
- 无历史且 `cold_delta="on"` 时：`Δ` 会注入 `Y_0` ⇒ 训练后**图像路径不再与图像模型逐位一致**，这是有意的 A/B 选项；`smoke` 里要分别断言两种情况。

### 4.2 `train_state.py`
- `model.state_gen.step(s, hist, cue=cues[:,t], cue_hist=cue_hist)` → `model.state_gen.step(s, hist)`；`cue_hist` 的维护可保留但不再需要（建议一并删掉，减少无意义显存/时间）。
- **预算保持随机**：`--budget-mode random` 为默认，`n = randint(1, T+1)` per-sample；深监督仍为 `--supervise all`。
- 冷帧比例 `--cold-ratio 0.15` 不变（保证无历史路径被充分训练）。

### 4.3 `eval_state.py`
- 删掉 `cue`/`cue_hist` 的构造与传参（`:143-160`、`:225-231`）。
- `attn` 统计改造：保留 `base_cross` 的帧级质量 `mass`/`ages`/`top_age`/`mass_at_period`（`:176-186`），**删除** `sel_ages`/`sel_mass`/`null_w`；新增两项：
  - `kv_mass`：`base_cross` 注意力里分给 `kv_init` 那 `KV0` 个 key 的质量（= 新的"不抄/背景"门，可解释性指标）；
  - `hist_mass_total = 1 − kv_mass`。
- 断言：`--cold-delta off` 时 `state_full` 的第 0 帧 `base is None`；`image_parity` 检查保持。

### 4.4 `smoke_state.py`
新增/更新单测：
1. 无历史：`step(None, [])` 返回 `delta is None`（`cold_delta=off`），且 `s.shape=(B,8,d)`；
2. 无历史 + `cold_delta=on`：`delta` 非 None、形状 `(B,196,d)`；
3. 有历史：`ctx` 长度 = `KV0 + n*196`（用一个 `n=3` 的小 case 断言）；
4. 初始化不变性：`base_out` 零初始化 ⇒ 有历史首帧 `delta ≡ 0`（`max|·|=0`）；
5. ckpt 兼容：旧 state ckpt 用 `strict=False` 加载，缺 `kv_init`、多 `lag_gate` 等 key 时不报错（并打印被忽略的 key 数）；
6. 图像 parity：在 `cold_delta=off` 下，单帧输入的输出与 base 模型逐位一致（`max|Δ|=0`）。

### 4.5 checkpoint 兼容
- 旧 ckpt（E/F/G1/G2，`run_state_*/last.pt`）缺 `state_gen.kv_init`、多 `state_gen.{lag_gate,null_gate,cue_q,cue_k,cue_scale,gain_gate}.*`。
- 提供一个小工具（或加载处加参数）：剥离旧 key + `strict=False`，供只做评测对比用；**新训练一律从 base 图像 ckpt 起**（`build_state_model` 原本就是严格加载 base、随机初始化 state_gen，不受影响）。

---

## 5. 验收标准（下个会话按这个判"成了/没成"）

**功能性**
- `python3 smoke_state.py` 全过（含 4.4 的 6 条）。
- `--cold-delta off` 下图像 parity `max|Δ|=0` 复现。

**效果（1500 步，`--budget-mode random`，其余超参与 G2 对齐）**
1. `state_full` 不低于 G2 的 19.058（允许 −0.15 dB 回退，超出则记失败）；
2. **既视感指标**（本设计的主 KPI）：`mass_at_period` 在 revisit8/revisit12 上 **> 0.15**（现状 0.039/0.014；均匀基线 0.06–0.11），且 `top_age` 不再 96% 集中在上一次；
3. `state_fixed_n1` 不低于 G2 的 14.747；
4. 同 token 曲线（`make_compare_fig.py`）不劣于 G2；
5. `kv_mass` 报告出来：若长期 > 0.7 视为"初始化向量退化成 null token"，需要调 `KV0`/温度/加正则（见 §6）。

**可解释性/诊断**
- 记录每帧 `kv_mass`、`hist_mass`、`top_age` 直方图、period 质量随训练步的变化（训练日志每 100 步采样一次 val）。

---

## 6. 风险与备选

| 风险 | 现象 | 备选 |
|---|---|---|
| `kv_init` 吸成 null token | `kv_mass→1`、历史注意力饥饿 | 减小 `KV0`；给 `kv_mass` 加目标正则（如在无历史帧才允许高）；或对 `kv` logits 加偏置 |
| 砍掉软拷贝后丢掉 +2.6~3.4 dB 的"抄上一帧"先验 | 训练初期 `state_*` 掉 1–3 dB（类似 v1/v2） | 用 `state_lag` 的帧龄先验先恢复短程（把 `age_pos` 之外再加一个可学 lag bias，纯 logits，不涉及拷贝）；或短程阶段保留"抄上一帧"作为初始化、再退火到注意力 |
| 冷帧不开 Δ ⇒ 冷帧状态未被 Δ 监督 | 冷帧 `s` 只被后续帧间接训练 | 先跑 `--cold-delta on` 的 A/B；若 ImageNet 单图指标掉太多则回退 `off` |
| 注意力在 `n*196` 个 key 上摊平（历史 patch 级竞争） | `mass` 仍近均匀 | 在 KV 侧对每帧做 patch 池化得到 1 个帧级 key（帧级 K/V + patch 级模板分开），或分层：先帧级选帧，再 patch 级读出 |
| `max_lag=16` 截断周期 > 16 | 长周期既视感仍不可见 | 提到 32/64 并测显存；或仅在慢路径上放开、拷贝路径已删故无额外开销 |

**若验收 2 失败（period 质量仍 < 0.1）**：说明瓶颈不在"聚合结构"而在"检索信号"。届时执行另一条线（见 `REPORT_中间态生成基底_小规模测试_20261110.md` §5）：传 4–8 bit 参考帧索引 / 专用检索 token / 像素级匹配，而不是继续堆注意力。

---

## 7. 复现与数据位置

- 实验代码仓: `/home/linaro/dsh/video_recon_probe`（`model_state.py`/`train_state.py`/`eval_state.py`/`smoke_state.py`）
- 现有评测结果: `analysis/eval_state_{a,e,f,g1,g2}/`（`results.json` 含 `attn` 统计）；图: `analysis/plots_state_*`
- 训练脚本: `run_state_e.sh`/`run_state_f.sh`（MUSA）；`a800_run_state_g{1,2}.sh`、`a800_run_eval_state.sh`（A800，已关机）
- 数据: MUSA 机 `/root/autodl-tmp/video_recon/data/{stills,bench,bench2,videos}`；bench=36 seq×16f，bench2=18 seq×16f（revisit8/revisit12/osc8），real=4 段 2fps
- 评测: `python3 eval_state.py --ckpt <last.pt> --bench <dirs> --out analysis/eval_state_<tag>`；本文件所有数字来自 `eval_state_g2`
- 报告: 同实验仓 `REPORT_中间态生成基底_小规模测试_20261110.md`（§2 诊断、§4 原因、§5 下一步）、`REPORT_逐步收紧预算分配_文献与离线回放_20261111.md`

## 8. 相关决定（本轮一并记录）
1. 训练预算**保持随机**（`--budget-mode random`）；"探索期→贪婪期"不再做。
2. λ 影子价格 / token 桶分配器：文献与离线回放已做完（上界 +0.05~0.09 dB@16 tok），暂缓实现，不作论文卖点。
3. 软拷贝（`Δ_copy`）整套删除，不再保留开关。
