> **集成状态**：已合入 `feat/vlm8b-v1`（tip `0340f40`，2026-11-06）。复现/验证脚本与日志在 `/home/linaro/dsh/dsfix/`（region-9 机器 `/root/dsfix/` 同名一份）。Lead 已独立复跑 4 项验证（2 卡修复路径 / guard 负例 / 旧代码 IndexError 负例 / repro selftest）全部通过。

# vlm8b `--freeze-encoder-epochs 1` DeepSpeed 修复报告

> 任务：让 E1 冻结 encoder / E2 起解冻续训在 DeepSpeed 下**真正可用**（不再被
> `IndexError: list index out of range` 挡住，也不是绕过）。
> 机器：region-9（2×A100-PCIE-40GB，torch 2.12.1+cu130，deepspeed 0.19.9，
> transformers 5.22.1）。基于 `feat/vlm8b-v1`（tip `47b98f1`）。
> 日期：2026-11-06。

---

## 0. 结论（TL;DR）

**根因**：不是「冻结时机选错」，而是 **DeepSpeed 0.19.9 在 engine 初始化时就把
`requires_grad=False` 的参数从优化器/ZeRO 分区里剔除**（`is_optimized_parameter()`），
而旧代码在 `Trainer` 创建**之前**就 `model.encoder.freeze()`：

- **stage 1/2**：全冻结的 encoder param group 被过滤成空组后，构造器仍按
  `bit16_groups[i][0]` 取 dtype → `IndexError: list index out of range`
  （`deepspeed/runtime/zero/stage_1_and_2.py:442`）；
- **stage 3**：不报错，但该组永远不进 DS 优化器 → **E2 解冻后 encoder 静默永不更新**
  （比崩溃更危险）。

**修复**：删掉 `train.py main()` 里 Trainer 创建前的 `model.encoder.freeze()`，
把 E1 冻结交给 **`EncoderFreezeCallback.on_train_begin`**（= engine 建好之后）；
并在 `VLM8BudgetTrainer.train()` 入口（= DS engine init 之前）加**前置 guard**，
把「提前冻结」从 stage1/2 的难懂 IndexError / stage3 的静默失效变成显式 RuntimeError。
此时参数仍在 ZeRO 分区里：E1 无梯度 ⇒ AdamW 整步跳过（权重逐元素不变），E2 解冻后
梯度出现 ⇒ 正常更新。**不需要改 DeepSpeed 源码**，patch 只动 `train.py`。
stage 1/2/3 全部验证通过；`--freeze-encoder-epochs 0`（run2/run3 的用法）行为零变化。

**可直接复制**（与原 CLI 完全一致，`--freeze-encoder-epochs 1` 现在真的可用）：

```bash
torchrun --nproc_per_node=2 train.py \
    --freeze-encoder-epochs 1 \
    --output-dir /root/autodl-tmp/vlm8b/output/run1 \
    --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl
```

---

## 1. 复现（真 traceback）

真实权重不在新机器，按同样的**结构性**条件用极小随机模型复现（脚本
`repro_ds_freeze.py`，6 个 param group：lm/proj/enc × decay/no-decay，bf16，
AdamW(0.9,0.95)，grad accum，`ds_zero2.json`；encoder 两组在 engine init 前
`requires_grad=False`，与旧 `train.py main()` 顺序一致）：

```bash
cd /root/dsfix/project
torchrun --nproc_per_node=1 repro_ds_freeze.py --stage 2 --freeze-when before
```

```text
File ".../deepspeed/runtime/zero/stage_1_and_2.py", line 442, in __init__
    param_dtype = self.bit16_groups[i][0].dtype
IndexError: list index out of range
```

真实项目代码路径（`VLM8Budget` + 真 `StandardAttentionEncoder` + 项目 Trainer）下
用旧写法也稳定复现（2 卡）：`logs/verify_regression_2gpu.log`，帧同上。

一键回归自检（4 个变体，各跑独立进程）:

```bash
/root/miniconda3/bin/python repro_ds_freeze.py --selftest
# 结果：ALL PASS（logs/repro_selftest.log）
```

| 变体（stage / 冻结时机） | 结果 | E1 encoder Δw | E2 encoder Δw |
|---|---|---|---|
| 2 / engine init 前（旧代码） | **IndexError @ stage_1_and_2.py:442** | — | — |
| 1 / engine init 前（旧代码） | **IndexError @ stage_1_and_2.py:442** | — | — |
| 3 / engine init 前（旧代码） | 不崩，但**静默失效** | 0.0 | **0.0** ← 解冻后不更新 |
| 2 / on_train_begin（修复） | OK | **0.0** | 7.63e-6 |
| 3 / on_train_begin（修复） | OK | **0.0** | 7.63e-6 |

---

## 2. 根因（DS 源码 + 行号）

**① 0.19.9 的过滤函数** `deepspeed/runtime/utils.py:822`：

```python
def is_optimized_parameter(param):
    """Whether an optimizer wrapper should own ``param``: trainable, with at least one element.

    Frozen parameters have always been left out of the optimizer groups. ...
    """
    numel = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
    return param.requires_grad and numel > 0
```

**② stage 1/2：先过滤、后仍按 `[0]` 取 dtype** `stage_1_and_2.py:423-442`：

```python
for i, param_group in enumerate(self.optimizer.param_groups):
    trainable_parameters = []
    for param in param_group['params']:
        if is_optimized_parameter(param):          # ← 冻结参数在这里被剔除
            param.grad_accum = None
            param.param_idx_in_group = len(trainable_parameters)
            trainable_parameters.append(param)
    self.bit16_groups.append(trainable_parameters)
    ...
    param_dtype = self.bit16_groups[i][0].dtype    # ← 442 行：空组 → IndexError
```

**③ 为什么「init 前冻结」即使不崩也救不回来**：同一构造函数随后把 underlying
optimizer 的每组 `params` 换成**已分区的 fp32 flat buffer**（`stage_1_and_2.py:584`）：

```python
# Set local optimizer to have flat params of its own partition.
param_group['params'] = [self.single_partition_of_fp32_groups[i]]
```

分区/平坦化在 init 时一次性完成，之后无法把参数加回 ZeRO 分区 ⇒ 初始化时被冻结的
参数永久脱离 optimizer（E2 解冻也不会更新）。这解释了 stage 3 的「静默失效」。

**④ stage 3 同样的过滤，但不崩** `stage3.py:727-741`：

```python
def _get_trainable_parameter_groups(self):
    for source_param_group_id, param_group in enumerate(self.optimizer.param_groups):
        trainable_params_by_group = collections.OrderedDict()
        for param in param_group[PARAMS_KEY]:
            if not is_optimized_parameter(param):   # ← 冻结参数剔除
                continue
            ...
        if len(trainable_params_by_group) == 0:     # ← 空组直接跳过（所以不崩）
            continue
```

⇒ 两种失败模式：**stage 1/2 崩（好事，暴露问题）**；
**stage 3 静默不更新（坏事，跑完才发现 E2 encoder 没训）**。

---

## 3. 候选方案对比

| # | 方案 | 正确性 | 改动面 | 结论 |
|---|---|---|---|---|
| 1 | patch DS：空组时跳过 `bit16_groups[i][0]` | ❌ 只消掉崩溃；init 前冻结的参数已被排除在 ZeRO 分区外，E2 仍不更新（stage3 同理）。还会让「静默失效」出现在 stage1/2 | 改 site-packages，不可随项目复用 | **否决** |
| 2 | 不冻结，E1 把 encoder 组 `lr=0, wd=0` | ✅ E1 权重逐元素不变（AdamW 下 lr=0 全跳过；但 lr 被 HF scheduler 每步重写，需 callback 覆盖） | 小，但要故意避开 scheduler；且**梯度照算** | 可行 fallback；语义上 encoder 仍在图里、grad-clip/Adam 状态受影响，不如方案 4 干净 |
| 3 | 两阶段：E1 真冻结（不进 optimizer）→ 存盘 → 重启 → E2 加载继续 | ✅ 语义最干净，且**真省 E1 显存** | 大：需重启进程、LR scheduler 按 global step 偏移、ckpt 续训/数据顺序、CLI 封装成一条命令；多一类失败模式 | 备选（要省显存才值，见 §6） |
| **4** | **冻结挪到 `on_train_begin`（engine init 之后）+ `Trainer.train()` 前置 guard** | ✅ E1 逐元素不变；E2 正常更新；stage 1/2/3 全过 | **删 2 行 + 注释 + train() guard**，不改 DS、不改 ds_json、不改 CLI | **采用** |

方案 4 也是三者中**唯一**能保持单进程、单命令、单 scheduler、无重启的，且顺带把
「stage 3 静默失效」这类隐患用防御断言变成显式报错。

---

## 4. 最终修复（`ds_freeze_fix.patch`，可直接 `git apply`）

只改 `train.py`（126 行 diff，实际逻辑改动 = 删除 2 行 + 新增 `Trainer.train()`
前置 guard + 注释）：已确认 `git apply --check` 对 `feat/vlm8b-v1`（`47b98f1`，
`train.py` sha256 与本地快照一致）通过。

1. **删除** `main()` 里 Trainer 创建前的：
   ```python
   if args.freeze_encoder_epochs > 0:
       model.encoder.freeze()
   ```
   位置换成详细注释（说明为什么不能在此冻结）；`main()` 把
   `freeze_encoder_epochs=args.freeze_encoder_epochs` 传给 `VLM8BudgetTrainer`。
2. **`VLM8BudgetTrainer.train()` 前置 guard**（真正的 fail-fast 检查点）：
   ```python
   def train(self, *args, **kwargs):
       if self.freeze_encoder_epochs > 0:
           frozen = [name for name, param in self.model.encoder.named_parameters()
                     if not param.requires_grad]
           if frozen:
               raise RuntimeError("encoder 在 Trainer 创建前就被冻结了……"
                                  "请改由 EncoderFreezeCallback.on_train_begin 执行")
       return super().train(*args, **kwargs)
   ```
   这是「提前冻结」**唯一会出问题的路径**，且此刻 `self.model` 还不是 DS engine，
   `requires_grad` 就是 engine init 会看到的状态 ⇒ 检查 100% 命中，并且发生在
   stage1/2 崩溃 / stage3 静默之前。负例日志证明它真的执行
   （`logs/verify_guard_runtimeerror_2gpu.log`）。
3. `EncoderFreezeCallback` docstring 写清「只允许在 engine init 之后切换冻结」并
   指向 `train()` guard。**删除了初版里基于 callback `optimizer` kwarg 的检查**：
   transformers 5.22.1 中 `CallbackHandler.optimizer` 从未被赋值（`trainer.py` 里
   没有 `callback_handler.optimizer = ...`），callback 收到的 `optimizer` 恒为
   `None`（且 stage1/2 在 `on_train_begin` 之前就崩），那段检查是**死代码**，已移除。
4. `build_param_groups` 旧注释（「encoder 组仍留在 optimizer 里…ZeRO-3 engine 初始化后
   无法再加参数」）替换为完整的 DeepSpeed 生命周期冻结协议说明（含 DS 源码行号）。
5. `--freeze-encoder-epochs` help 补充「冻结在 engine 初始化之后生效」。

**没有**修改 DeepSpeed site-packages ⇒ 无需 `ds_site_packages.patch`。

---

## 5. 验证数据

### 5.1 真实项目代码路径（`verify_fix_tiny.py`，2×A100，ZeRO-2）

构成：`VLM8Budget` + 真 `StandardAttentionEncoder(dim=1744, depth=1, patch=16, heads=16,
num_specials=32)` + 极小随机 `Qwen3_5ForConditionalGeneration`（hidden=2560=LM_HIDDEN，
vocab 64，1 层，随机权重，删 visual tower，bf16）+ **项目自己的**
`build_param_groups` / `VLM8BudgetTrainer` / `EncoderFreezeCallback` + `ds_zero2.json`；
2 卡 × micro 2 img × accum 2 = **20 optimizer steps/epoch × 2 epoch = 40 步**，真正跨过
epoch 边界触发 `on_epoch_begin` 解冻。

```bash
cd /root/dsfix/project
torchrun --nproc_per_node=2 verify_fix_tiny.py            # EXIT=0, ALL ASSERTIONS PASSED
torchrun --nproc_per_node=2 verify_fix_tiny.py --save     # 再加 epoch ckpt + final save 路径
torchrun --nproc_per_node=2 verify_fix_tiny.py --freeze-when before            # 负例①：新 guard → RuntimeError
torchrun --nproc_per_node=2 verify_fix_tiny.py --freeze-when before --no-guard # 负例②：旧代码 → DS IndexError
```

| 指标（2 卡 ZeRO-2） | E1 | E2（解冻后） | 断言 |
|---|---|---|---|
| steps / epoch | 20 | 20 | ✅ 跨 epoch、解冻后继续 ≥5 步 |
| encoder `requires_grad` | **False**（全冻结） | **True** | ✅ |
| encoder 权重 vs 初始 | **bit-exact 相同**（`torch.equal`），Δw=**0.0** | Δw=**1.53e-5** | ✅ E1 不变 / E2 更新 |
| projector Δw | 1.46e-3 | 4.88e-4 | ✅ 每轮更新 |
| LM Δw | 2.44e-4 | 2.44e-4 | ✅ 每轮更新 |
| peak 显存/卡 | 1.90 GB | 1.86 GB | — |
| 每 epoch 用时 | 3.31 s | 2.41 s | — |
| epoch ckpt + `save_model(final)` | checkpoint-40 + final 均正常 | — | ✅（`--save`） |
| 0 参数被提前冻结 | guard 在 engine init 前通过（不误报） | — | ✅（同上，EXIT=0） |

日志：`logs/verify_fix_2gpu.log`、`logs/verify_fix_2gpu_save.log`。

**负例（真实项目代码路径，2 卡，均 EXIT=0 且打印 `NEGATIVE-TEST PASS`）**：
- `logs/verify_guard_runtimeerror_2gpu.log`：`--freeze-when before`（模拟旧代码在
  Trainer 前冻结）→ 新 guard 在 DS engine init **之前**抛
  `RuntimeError: encoder 在 Trainer 创建前就被冻结了（19 个参数…）`；
- `logs/verify_oldbug_indexerror_2gpu.log`：`--freeze-when before --no-guard`
  （关闭 guard，等价修复前代码）→ 仍复现 DS 原生
  `IndexError @ stage_1_and_2.py:442`。两条都留，证明 guard 真的执行、且它拦下的
  正是原来那个 IndexError。

### 5.2 ZeRO-3（尽力而为，也过了）

同脚本 `--deepspeed ds_zero3.json`，1 卡与 2 卡均 EXIT=0：
E1 encoder bit-exact、Δw=0；E2 Δw=1.53e-5；projector/LM 每轮更新。
日志：`logs/verify_fix_z3_1gpu.log`、`logs/verify_fix_z3_2gpu.log`。
注：历史上 `ZeRO-3 + torch 2.12 + 4B 真模型` 第 1 步 backward NCCL 死锁是**另一件事**
（本报告未处理，tiny 随机模型复现不出来）；生产仍建议 ZeRO-2。

### 5.3 生产尺寸 encoder 的显存/速度 trade-off（2 卡 ZeRO-2）

把验证模型换成**生产同结构同尺寸**的 encoder（`dim=1744, depth=13` → **476.2M** 参数，
= 真实 encoder 的 480M），LM 仍是极小随机（4B 权重不在本机），跑 20 steps/epoch：

| E1 配置 | E1 peak alloc /卡 | E1 20 步用时 | E1 是否更新 | 说明 |
|---|---|---|---|---|
| 修复路径（callback 冻结） | **8.95–8.97 GB** | 5.35 s | 否（bit-exact） | 本 patch |
| 不冻结（`--freeze-encoder-epochs 0` 等价） | 8.55 GB | 11.08 s | 是（Δw 1.2e-4） | 旧 run2/run3 的语义 |
| 两阶段 phase-1（encoder 完全不进 optimizer） | **4.34 GB** | 2.62 s | 否 | 备选方案 3 |

E2（解冻）三者一致：~8.5 GB/卡、10.3 s/20 步、encoder Δw>0。

解读：
- **本修复省的是 encoder backward**：E1 5.35s vs 不冻结 11.08s（此微缩模型里 encoder
  占 94% 参数，所以差异被放大；**真实 vlm8b 里 encoder 480M vs LM 4B ≈ 10%**，按
  FLOPs 摊薄后 E1 每步只省约 1–2% 时间，属「语义正确 + 少量效率」而非性能优化）；
- **本修复不省 optimizer 显存**：engine init 时 encoder 参数可训，DS 已为它建好
  fp32 master/Adam 分区（上表 8.95 vs 4.34 的差额 ≈ 4.6 GB/卡就是这部分）。E1 显存
  与不冻结基本同量级。
- **想真省 E1 显存只能两阶段**：476M encoder + 2 卡 ZeRO-2 下实测 **省 ~4.6 GB/卡**、
  E1 每 20 步 2.62s vs 5.35s（省掉「encoder 进 optimizer」的那部分 step）。
  代价：进程重启一次（~1–2 min）、LR scheduler 需要按 global step 偏移续上 cosine、
  E1 末尾要存盘 + E2 用 `--init-from` 加载（train.py 已有 `--init-from`/epoch ckpt，
  但仍需封装与验证），且多一类失败模式。**如果 E1 冻结的目的是显存而非「别动 encoder」，
  再上这个方案；否则不建议**。

---

## 6. 对现有实验（run2/run3）的影响 & 默认值建议

- run2/run3 都显式传 **`--freeze-encoder-epochs 0`**（Lead 确认）。本 patch 对
  `freeze_epochs == 0` **零影响**：被删的两行只在 `>0` 时执行；callback 在 0 时仍然是
  `on_train_begin → unfreeze()`（与旧代码一致，本来就没冻结过）；新增的 `train()` guard 只在
  `>0` 时运行。⇒ **正在跑的 run2/run3 无需任何操作，行为不变**；patch 只影响后续想开
  冻结的 run。
- **默认值未改**（仍是 `1`，与 `feat/vlm8b-v1` 相同，避免改变既有默认语义——
  注意：如果你以为默认是 0，实际代码是 1；run 脚本显式传 0 的话不受影响）。
  若希望「默认 0、要冻结才显式开」，改 `parse_args` 的 `default=0` 一行即可，
  本次未做等确认。
- 建议：**不改默认、不换 deepspeed 版本**（0.19.9 的 `is_optimized_parameter`
  行为就是本 bug 前提；换版本等于换坑）。

---

## 7. 交付物

本机共享盘 `/home/linaro/dsh/dsfix/`（region-9 上同名留一份 `/root/dsfix/`）：

| 文件 | 说明 |
|---|---|
| `ds_freeze_fix.patch` | 对 `feat/vlm8b-v1` 的 git diff，只改 `train.py`；`git apply --check` 通过 |
| `repro_ds_freeze.py` | 最小复现 + `--selftest` 回归自检（stage1/2/3 × 旧/新冻结时机） |
| `verify_fix_tiny.py` | 真实项目代码路径端到端验证（2 卡 2 epoch、bit-exact/Δw 断言、显存/速度；`--save` 验存盘；`--freeze-when before` 是反例；`--freeze-mode`/`--encoder-depth` 出 bench 数据） |
| `train.py` | 修复后的文件（= baseline + patch，便于直接看） |
| `logs/` | 复现/回归/验证/bench 全部日志（region-9 `/root/dsfix/logs/` 同步一份） |
| `REPORT.md` | 本报告 |

关键日志：
- `logs/repro_z2_before_1gpu.log`：原始 IndexError 复现
- `logs/repro_selftest.log`：5 个变体回归自检 ALL PASS
- `logs/verify_fix_2gpu.log` / `verify_fix_2gpu_save.log`：真实代码路径 2 卡验证（EXIT=0）
- `logs/verify_guard_runtimeerror_2gpu.log`：负例① 新 `train()` guard → RuntimeError（NEGATIVE-TEST PASS）
- `logs/verify_oldbug_indexerror_2gpu.log`：负例② 无 guard 旧代码 → DS IndexError（NEGATIVE-TEST PASS）
- `logs/verify_fix_z3_1gpu.log` / `verify_fix_z3_2gpu.log`：ZeRO-3
- `logs/bench_fixed_enc13.log` / `bench_none_enc13.log` / `bench_exclude_enc13.log`：显存/速度 trade-off

集成方式：

```bash
cd <AssocVTC 工作区>
git checkout feat/vlm8b-v1        # 47b98f1
git apply --check /path/to/ds_freeze_fix.patch
git apply /path/to/ds_freeze_fix.patch
git diff --stat                    # 只应有 train.py
```

## 8. 未覆盖 / 风险

1. 验证用的是**极小随机 LM**（4B 权重不在本机、且禁止从主训练机拷贝）：encoder 是
   生产同结构同尺寸（476M），Trainer/DS/param-group/callback 是项目真代码；但 4B LM
   的数值行为、显存总量未在本机复测。上机后建议先 `--smoke` 跑 2 步确认。
2. 本报告只解决 **stage1/2 + stage3 的冻结/解冻生命周期**；`ZeRO-3 真模型 backward
   NCCL 死锁`是独立问题（生产用 ZeRO-2，未处理）。
3. 防御断言依赖 DS 内部属性（`bit16_groups`/`fp16_groups`）；取不到时静默跳过（不误
   报），已在 stage2/stage3 实测不误触发。
4. `--freeze-encoder-epochs > 1` 走同一 callback（每 epoch 判 `state.epoch < N`），
   逻辑未变、未单独长跑验证（语义上同 N=1）。
