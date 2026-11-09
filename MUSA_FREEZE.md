# encoder 冻结机制 × 国产算力适配（MUSA / MTT S4000）

> 结论先行，机制在后。全部数据来自 2×MTT S4000（48GB/卡）、driver 5.2.0、
> torch 2.11.0.post1 + torch_musa 2.11.0.post1、deepspeed 0.19.9、transformers 5.13.0、
> torchrun 2 rank 的实测，不是「应该能跑」。

## 0. 三条结论

| 问题 | 结论 |
|---|---|
| 想**全程**冻住 encoder（最省、最稳） | `--freeze-encoder all`（或 `stem` / `last=N` / `blocks=0-8`）。冻结在 Trainer 之前生效，可训练参数才进 optimizer，**encoder 与 ckpt 逐位一致（Δ=0.0）**，ZeRO-2 验证通过 |
| 想 **E1 冻、E2 起解冻** | 只能用 `--freeze-encoder all --freeze-encoder-epochs 1`，且必须走 DeepSpeed。切换发生在 engine 建好之后。注意：解冻前那段不是「严格不动」，见 §2.4 的动量残留（生产 lr=5e-6 时低于 bf16 精度，实测不可见；lr 放大到 1e-2 时每步可动 ≤ lr） |
| 想用 **ZeRO-3** | 本仓 encoder 用不了：`cls_token`/`special_tokens`/`pos_embed` 是裸 Parameter，ZeRO-3 不为它们 all-gather，**不冻结也会崩**。代码已 fail-fast，请用默认的 `ds_zero2.json` |

国产卡上要跑起来，需要 4 个补丁（都收敛在 `musa.py` / 各文件注释里）：MCCL 通信后端、
fp64 范数探测、奇数 head_dim 的 SDPA、deepspeed 的 dist-info。详见 §3。

## 1. 怎么用

```bash
# 0) 自检（CPU，不加载 Qwen / DeepSpeed）
python3 smoke.py          # 原张量逻辑，19 项
python3 test_freeze.py         # 冻结语义，27 项
python3 test_freeze_compat.py  # 兼容性契约（state_dict/groups/加载/身份），18 项

# 1) 静态冻结：整段 encoder 冻死，只训 proj + LLM
torchrun --nproc_per_node=2 train.py \
    --freeze-encoder all --deepspeed ds_zero2.json ...

# 2) 部分冻结：只训最后 2 个 block + 末层 LayerNorm
torchrun --nproc_per_node=2 train.py --freeze-encoder last=2 ...

# 3) 冻 stem，训全部 block
--freeze-encoder stem
# 4) 冻前 9 个 block（0..8，含端点），其余可训
--freeze-encoder blocks=0-8
# 5) 组合
--freeze-encoder blocks=0-8,norm
# 6) E1 冻、E2 起解冻（仅 DeepSpeed）
--freeze-encoder all --freeze-encoder-epochs 1
```

规格语法：`none` / `all` / `stem` / `norm` / `blocks=a-b` / `blocks=a,b` / `last=N`，
逗号（或 `+`）组合。`blocks` 是 `encoder.net.blocks` 的下标，0 最靠输入。

## 2. 机制：为什么「设 requires_grad=False」远不够

### 2.0 兼容性契约（先看这一节）

冻结被设计成一个**派生状态**：它不新增模块、不新增 buffer、不改 `state_dict`、不改
默认路径。五条不变量由 `test_freeze_compat.py`（CPU，18 项）+ `musa_e2e.py` 逐条断言：

| 不变量 | 为什么重要 | 怎么保证 / 证据 |
|---|---|---|
| **C1 不改 state_dict**：任意冻结规格下参数名/形状完全一致 | 阶段2 ckpt 的 `strict=True` 加载、`save_pretrained`、`--init-from` 都不受影响 | 冻结只动 `requires_grad`；6 种规格键/形状差异 = 0 |
| **C2 默认路径零改动**：`none` 的 param groups 与旧实现逐组一致（顺序/lr/wd/参数集合） | 不冻结的 run 才能和历史 run 逐位对齐 | 与旧 `build_param_groups` 逐字复刻对照：`new=6 old=6`，逐组一致 |
| **C3 先冻后载**：已冻结模型仍能 `load_state_dict(strict=True)` | warm start / 续训时「先建模型再灌权重」的顺序很常见 | 冻结后灌入，`max|Δ|=0`，且灌完仍冻结 |
| **C4 参数身份稳定**：只改 `requires_grad`，不重建 Parameter | 重建会让 optimizer 持有旧引用 → 解冻后静默不更新 | 冻结/解冻前后 `id(p)` 与 `data_ptr()` 均不变 |
| **C5 冻结不影响保存**（两卡，真 Trainer+ZeRO-2） | 冻结参数被剔出 optimizer，但必须仍在 ckpt 里 | `save_model` 后 43/43 个 encoder 张量齐全、与引擎内**逐位一致**、键无缺失 |

另一条被这次实验抓到的生产问题：`trainer.save_model()` 只在 `args.should_save`
（world-zero）落盘，其余 rank 若立刻去读/退出会看到半成品。已加
`accelerator.wait_for_everyone()`（`train.py` 保存后）。

### 2.1 transformers 侧：三处会咬人

1. **`Trainer.create_optimizer` 默认按 `requires_grad` 过滤参数**，但本仓覆盖了
   `create_optimizer`（三组 lr），所以必须自己在建组时过滤——`freeze.build_param_groups()`
   就是这么做的，而且**丢弃空组**。
2. **`Trainer` 每个 epoch 调 `model.train()`**，会把冻结子模块里的 dropout 也切回
   train。`VLM8Budget.train()` 覆盖后调 `freeze.keep_frozen_eval()`，保证冻结段始终
   eval（否则同一份权重两种前向，冻结对比实验不可复现）。
3. **`post_init()` 在 transformers 5.13 会重初始化已加载的权重**：实测
   `VLM8Budget(...)` 构造完，encoder 与 ckpt 的 `max|Δw| = 0.121`（等于白训）。
   5.22.1 上没这个行为（上游注释据此写成「v5 只在 meta 上 post_init」），但本机是
   5.13。`VLM8Budget._post_init_keep_weights()` 现在把 `init_weights` 临时置空，
   只让 post_init 登记 `all_tied_weights_keys`。

### 2.2 DeepSpeed 侧：engine 初始化是「一次性契约」

DeepSpeed 0.19.9 在 engine 初始化时用
`is_optimized_parameter(param) = requires_grad and numel > 0`
（`deepspeed/runtime/utils.py:822`）决定哪些参数进 ZeRO 分区/优化器。由此得到
四条实测规则：

| 冻结时机 | ZeRO-1/2 | ZeRO-3 |
|---|---|---|
| **整组**在初始化前冻 | `IndexError: list index out of range`（`stage_1_and_2.py:442` 取 `bit16_groups[i][0].dtype`，组被剔空） | 初始化不报错，但该组永不进优化器：**解冻后静默不更新** |
| **部分**在初始化前冻（每组仍有可训参数） | 正常；冻结参数被剔出优化器（省 optimizer state），逐位不变 | 同上（但本仓 encoder 用不了 ZeRO-3，见 §3.5） |
| 初始化和**之后**再冻（`requires_grad=False`） | 冻结期不再产生新梯度；但**动量残留会继续推动参数**（§2.4） | 同左 |
| 用 `detach` 门控（不动 requires_grad） | 等价于上一行（同样有动量残留） | — |

**所以「静态冻结」要在 Trainer 之前做，「按 epoch 冻结」必须在 `on_train_begin`
（engine 建好之后）做**。`train.py` 两道 guard 把这两种错配变成可读报错：
`VLM8BudgetTrainer.train()`（提前冻 + 没开 DS）、`assert_zero3_freeze_compatible()`。

### 2.3 部分冻结：图必须穿过未冻层

`AssocVTCEncoder.forward` 只在**整段可训参数为 0** 时才 `no_grad` + `detach`；
部分冻结（`stem` / `blocks=a-b` / `last=N`）走正常前向，冻结层靠 `requires_grad=False`
自然截断，未冻的尾段照常反传。

### 2.4 运行时冻结的真相：ZeRO 会「惯性滑行」

ZeRO-1/2 把每组的参数拍平成一条 fp32 主权重、并给底层 AdamW 一条**稠密**梯度。
`requires_grad=False` 只是让新梯度为 0，**优化器里上一阶段的 `exp_avg/exp_avg_sq`
仍按 `(β1/√β2)^t ≈ 0.91^t` 衰减着推动参数**，每步量级 ≤ lr（另加 weight decay）。
2×S4000 + ZeRO-2 实测（真实 encoder + 真实 Trainer）：

| encoder lr | 冻结后 2 步的 `max|Δw|` | 说明 |
|---|---|---|
| 1e-2（故意放大） | 1.56e-2 | ≈ 1.5·lr，就是动量残留 |
| 5e-6（生产值） | 0（bf16 下不可见） | bf16 ulp ≈ 6.1e-5，远大于 lr |

这解释了上游 A800 上「E1 encoder bit-exact、Δw=0.0」的观测：不是数学上为 0，
而是 lr 太小、被 bf16 精度掩盖。**要严格不变，就用静态冻结**（参数根本不在
optimizer 里，实测 Δ=0.0）；分阶段冻结请接受「≤ lr/步、几十步内衰减完」的漂移。

另外，切换冻结状态后要清一次梯度（`EncoderFreezeCallback._clear_grads`）：不清会
用上一轮已累积的 stale grad 再推一步（实测 6.1e-5，等于一个 bf16 ulp）。

### 2.5 参数组

`freeze.build_param_groups()` 输出「只有可训练参数、且没有空组」的组。整段冻结
encoder 时组数从 6 降到 4（lm 2 组 + proj 2 组），ZeRO-2 才不会因为空组在初始化时
`IndexError`——这是 `--freeze-encoder all` 能直接跑起来的原因。

## 3. 国产适配（MUSA）清单

全部差异收敛在 `musa.py`，`train.py:main()` 第一行 `musa.bootstrap()`；代码里没有
`"cuda"` 硬编码。

1. **`torch.cuda` 是 MUSA 的别名**：`torch.cuda.is_available()=True`、
   `get_device_name(0)="MTT S4000"`、`torch.cuda.device_count()=2`，所以
   transformers / accelerate / Trainer 里写死的 `"cuda"` 大多不用改；但
   `torch.version.cuda is None`，别写 `if torch.version.cuda` 的分支。
2. **通信后端是 MCCL，不是 NCCL**：`is_nccl_available()=False`，
   `Backend.MCCL` 可用，`init_process_group("mccl")` 双卡 all_reduce 正常；gloo 虽在，
   但 `ProcessGroupGloo::allreduce: unsupported device type musa`。
   DeepSpeed 的 `CUDA_Accelerator` 把后端名写死成 `"nccl"`，`musa.patch_deepspeed()`
   在 engine 初始化前改成 `"mccl"`，否则直接失败。
3. **fp64 是「假支持」**：`is_fp64_supported()` 返回 True，但 `Tensor.norm()` 对
   Double 直接 `RuntimeError: Norm only supports Float/Half/BFloat16`。DeepSpeed 的
   `get_norm_dtype()` 正是靠这个探测决定梯度范数用 fp64 ⇒ **ZeRO-2/3 + bf16 会在
   `stage3.py:2579` 崩**（不冻结也一样）。patch 成 False 后退回 fp32 范数，正常。
4. **SDPA 只有 math 可靠，且奇数 head_dim 会硬崩**：本版枚举里有 flash/efficient 名字，
   但阶段2 编码器 `dim=1744 / heads=16 ⇒ head_dim=109`（奇数），MuDNN flash 直接
   `RuntimeError: MuDNNFlashSDPAFwd ... Flash Attention 2 Not Support HeadDim Odd`，
   **而且不回退**。`encoder._sdpa()` 对 `head_dim % 8 != 0` 显式走 `SDPBackend.MATH`
   （数值等价；CUDA 上也会因同样的限制回退到 math，无回归）。
5. **ZeRO-3 与本 encoder 不兼容**：`cls_token/special_tokens/pos_embed` 是裸
   Parameter，ZeRO-3 只在模块边界 all-gather ⇒ 前向拿到空分区
   （`size of tensor a (0) ... b (1744)`），`GatheredParameters` 也救不回来
   （pre/post hook 会重新分区）。`train.py` 对 `stage>=3` 直接 fail-fast，走 ZeRO-2。
6. **deepspeed 安装**：镜像里没有，`pip install deepspeed==0.19.9` 的 sdist 会在
   setup 阶段调 `$CUDA_HOME/bin/nvcc`（MUSA 的 CUDA_HOME 指向 `/usr/local/musa`，
   没有 nvcc）而失败。做法：解包后把 `deepspeed/` 拷进 site-packages，并补一份
   `deepspeed-0.19.9.dist-info/METADATA`——否则 transformers 的
   `is_deepspeed_available()`（importlib.metadata）认为没装，
   `TrainingArguments(deepspeed=...)` 直接 `ImportError`。DS 的 CUDA 扩展全部按需
   JIT，用 torch AdamW + `zero_allow_untested_optimizer` 时不会碰它们。
7. **bf16 可用**：`torch.cuda.is_bf16_supported()=True`，`ds_zero2.json` 的
   `"bf16": {"enabled": true}` 与 Trainer 的 `bf16=True` 直接可用（实测 loss 正常下降）。

## 4. 实测矩阵（2×MTT S4000，torchrun 2 rank）

### 4.1 DeepSpeed 生命周期（tiny model，直接 `deepspeed.initialize`）

| 用例 | 结果 |
|---|---|
| ZeRO-2，整组冻结后初始化 | `IndexError: list index out of range`（复现上游事故） |
| ZeRO-2，部分冻结后初始化 | 正常；冻结 block `Δw=0`，其余更新 |
| ZeRO-3，整组冻结后初始化 | 初始化不报错；解冻后**不更新**（静默） |
| ZeRO-2，初始化后冻结→解冻 | 冻结期 `Δw=0`（bf16 精度内），解冻后 `Δw=9.8e-4` |
| ZeRO-2，`detach` 门控 | 同上，不需要动 `requires_grad` |
| ZeRO-3 + bf16（未打 fp64 补丁） | `Norm only supports ... Double`（`stage3.py:2579`） |
| ZeRO-3 + bf16（打补丁后） | 正常 step |

### 4.2 真实训练栈（`musa_e2e.py`：真 `AssocVTCEncoder` + 真 `VLM8BudgetTrainer` + DS ZeRO-2）

| 断言 | 结果 |
|---|---|
| 静态 `all`：encoder 与 ckpt 逐位一致 | **Δ=0.0** ✓ |
| 静态 `all`：projector 仍更新 | Δ=1.4e-1 ✓ |
| 静态 `last=2`：`stem + blocks[0:1]` 与 ckpt 一致 | **Δ=0.0** ✓ |
| 静态 `last=2`：`blocks[1:]` 在更新 | Δ=2.0e-2 ✓ |
| engine 建好后冻结：只剩动量残留 | Δ=1.56e-2，lr=1e-2（≤3·lr）✓ |
| engine 建好后冻结：projector 仍更新 | Δ=2.2e-3 ✓ |
| 解冻后 encoder 恢复更新 | Δ=1.59e-2 ✓ |

复现：

```bash
torchrun --nproc_per_node=2 musa_e2e.py --stage 2   # 7 项全 PASS
torchrun --nproc_per_node=2 musa_e2e.py --stage 3   # 2 项 fail-fast PASS
```

## 5. 已知限制

- **ZeRO-3 不可用**（§3.5）。想用 ZeRO-3 需要把 encoder 的三个裸 Parameter 包成子模块
  ——那会改参数名，破坏阶段2 ckpt 的 `strict=True` 契约，需要同时迁移 ckpt。
- **分阶段冻结不是严格 bit-exact**（§2.4）；要严格不变请用静态冻结。
- **`transformers` 版本敏感**：本机 5.13 与 `requirements.txt` 的 5.22.1 行为不同
  （post_init 重初始化已在代码里兜住，但其它 v5 小版本差异未逐一验证）。
- 本机没有 Qwen3.5-4B 权重与阶段2 ckpt（`/root/qwen_models`、`exp448/B` 不存在），
  所以端到端用的是**真 encoder 结构 + 合成 ckpt + 同维微型 LM**；4B 全流程需要把
  两处路径指到真实资产后再跑一次 `train.py --smoke`。
