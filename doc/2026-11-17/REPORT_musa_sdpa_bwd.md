# MUSA SDPA 反向 bug：muDNN flash 在 `head_dim % 8 != 0` 时 dQ/dK 算错（2026-11-17）

## 0. 一句话

`torch_musa 2.11.0.post1` 在 MTT S4000（arch 220）上把 bf16/fp16 的
`F.scaled_dot_product_attention` 派发到 **muDNN flash kernel**（`RunFlash`/`RunFlashBwd`），
而该 kernel 的**反向**在 `head_dim % 8 != 0` 时给出**错误的 dQ/dK**（前向、dV 正常）：

- `head_dim=78`（本仓编码器 1248/16）、`head_dim=90`（解码器 1440/16）→ bf16 dQ 相对误差 **2.6e-2 / 4.2e-2**，fp16 **2.6e-2 / 4.2e-2**（**两种 dtype 一样错 ⇒ 不是精度问题，是 kernel 逻辑错**）；
- `head_dim=64/72/80/88/96/112/128`（8 的倍数）→ bf16 **4.8e-3**、fp16 **6.0e-4**（正常）；
- 同一 dtype 下把后端钉到 math → dQ 误差回到 **6.0e-3**（bf16）/ **7.4e-4**（fp16）⇒ fp16 下 flash 比 math 差 **35 倍**（fp16 对 8 对齐的 D=64 只差 6.0e-4）；
- 误差随 logits 量级单调放大，且**同一 head_dim 下 flash 始终比 math 差 4–16 倍**
  （D=78：qk 标准差 ×1/×2/×4/×8/×16 → dQ 误差 2.6e-2 / 0.21 / 0.67 / 1.41 / 2.90，
  math 同条件 6.0e-3 / 1.4e-2 / 5.0e-2 / 0.18 / 0.78；而 D=64 的 flash/math 比值 < 1），
  这与"训练到后期（logits 增大）才炸"完全吻合；
- 结论：之前报告里"bf16 flash/mem-efficient 反向坏"的**归因不准确**，真正的触发条件是 **head_dim 非 8 的倍数**，而 `SDPA_BACKEND=math` 之所以有效，是因为它**改变了 kernel**（flash→math），不是因为"绕开了 bf16"。

## 1. 环境（`/root/autodl-tmp/musa_bug/env.txt`，113 行，关键行）

```
GPU                 4x MTT S4000 48GiB, Driver 5.2.0-server, mthreads-gmi 2.4.2
MUSA SDK            5.2.0  (/usr/local/musa), muDNN 3.4.0 (banner "muDNN(v3400)")
compute capability  2.2  -> torch_musa arch 220
torch.__version__             = 2.11.0.post1
torch.version.musa            = 50200
torch.version.git_version     = 2b86b38358bd692be2b1fc5c37c0fdd6e12f8db5
torch_musa.__version__        = 2.11.0.post1+b431668
torch_musa.version.git_version= b43166890d500f2c8a28ba1e8fdaf18e8ae3d24d
python 3.10.12, torchvision = 空壳（无 metadata）
flash_sdp_enabled=1  mem_efficient_sdp_enabled=1  math_sdp_enabled=1
can_use_flash_attention(..)=False   ← 在 MUSA 上是**错的**，flash 实际在用
is_flash_attention_available()=False ← 同上，不可信
```

## 2. 最小复现（`/root/autodl-tmp/musa_bug/min_repro.py`，纯 torch，18 s）

```bash
MUSA_VISIBLE_DEVICES=0 python /root/autodl-tmp/musa_bug/min_repro.py            # 主对照
MUSA_VISIBLE_DEVICES=0 python /root/autodl-tmp/musa_bug/min_repro.py --sweep dim # head_dim 扫描
```

做法：随机 Q/K/V/G（G = 上游梯度），`loss=(out*G).sum()`，把 dQ/dK/dV 与**手写 fp64 CPU 参考**
（muDNN 无 fp64 SDPA，参考不能再用 muDNN）比较，打印相对 L2 误差与范数比。
默认用真实形状 `B2 H8 N213 head_dim78`，并给 `head_dim=64` 作对照。

实测输出（2026-11-17 21:40，4 张卡同时跑生产训练）：

```
=== B=2 H=8 N=213 D=78 MODEL SHAPE ===                    （ref = fp64 CPU）
case                    dQ        dK        dV       aQ    aK    aV   fwd_loss
default / bf16    2.614e-02 1.824e-02 3.757e-03   1.000 1.000 1.000  1.62e-02
math    / bf16    6.017e-03 5.983e-03 5.181e-03   1.000 1.000 1.000  1.16e-02
default / fp16    2.570e-02 1.772e-02 4.665e-04   1.000 1.000 1.000  2.08e-04
default / fp32    4.256e-07 4.243e-07 4.013e-07   1.000 1.000 1.000  1.63e-07

=== B=2 H=8 N=213 D=64 CONTROL (8-aligned) ===
default / bf16    4.832e-03 4.473e-03 3.803e-03   1.000 1.000 1.000  8.09e-03
math    / bf16    5.972e-03 5.978e-03 5.121e-03   1.000 1.000 1.000  1.16e-02
default / fp16    5.963e-04 5.528e-04 4.745e-04   1.000 1.000 1.000  9.66e-05
default / fp32    3.999e-07 3.977e-07 3.777e-07   1.000 1.000 1.000  2.96e-06
```

- 前向（fwd_loss）在两种 head_dim 下都是 bf16 该有的量级（1e-2/1e-4），**前向不是问题**；
- `dV` 也基本正常（3.8e-3），**只有 dQ/dK 错**；
- `aQ/aK ≈ 1.000` 说明错的是**方向**不是整体缩放（下游会因为方向错而炸，不是简单地"梯度变大"）。

## 3. 定位：逐变量固定后的结果

### 3.1 触发条件 = `head_dim % 8 != 0`（`--sweep dim`，B2 H8 N213 bf16/fp16，ref fp64）

| head_dim | flash bf16 dQ | flash fp16 dQ | 判定 |
|---|---|---|---|
| 56 | 6.1e-3 | 7.5e-4 | flash 被拒（<64）→ 实际走 math ✅ |
| 60 | 6.0e-3 | 7.5e-4 | 同上 ✅ |
| 64 | 4.8e-3 | 6.0e-4 | ✅ |
| 65 | RuntimeError | RuntimeError | 奇数维直接抛错（`Flash Attention 2 Not Support HeadDim Odd`），**不 fallback** |
| 66 | 4.8e-2 | 4.8e-2 | ❌ |
| 68 | 4.0e-2 | 4.0e-2 | ❌ |
| 72 | 4.8e-3 | 6.0e-4 | ✅ |
| 76 | 3.7e-2 | 3.6e-2 | ❌ |
| **78** | **2.6e-2** | **2.6e-2** | ❌ ← 本仓编码器 |
| 80 | 4.8e-3 | 6.0e-4 | ✅ |
| 84 | 3.7e-2 | 3.7e-2 | ❌ |
| 88 | 4.8e-3 | 6.0e-4 | ✅ |
| **90** | **4.2e-2** | **4.2e-2** | ❌ ← 本仓解码器 |
| 96 / 112 / 128 | 4.8e-3 | 6.0e-4 | ✅ |
| 100 | 3.1e-2 | 3.0e-2 | ❌ |

**判别标准很干净：`head_dim % 8 == 0` → 误差 = dtype 噪声；否则 → 误差比 dtype 噪声大 5–70 倍，
且 bf16 与 fp16 给出彼此相差仅 4.9e-3（= dtype 噪声）、却共同偏离 fp64 3e-2 的结果。**
⇒ 两个 dtype 共享同一个**系统性**错误 ⇒ 算法/索引层面的 bug，不是舍入。

### 3.2 后端三开关

- MUSA 的 `_fused_sdp_choice` 被 torch_musa 覆盖（`SDPBackendSelect.cpp` + `SDPUtils.h`）：
  **只可能返回 flash 或 math**；`mem_efficient` 永远选不到（单独只开 mem_eff 直接
  `RuntimeError: Invalid backend configuration for musa!`）。
- `enable_flash_sdp(False)` **一个开关就够**（math 默认是开的），实测即回到正确数值。
- `torch.backends.cuda.can_use_flash_attention()` 在 MUSA 上恒为 False（走的是 ATen 通用实现，
  不知道 MUSA 的覆盖），**不能用来判断实际后端**，别被它误导。

### 3.3 与 torch_musa 源码的对应（关键证据）

安装包里带 C++ 源码（`/usr/local/lib/python3.10/dist-packages/torch_musa/csrc/aten/ops/attention/mudnn/`）：

```cpp
// SDPUtils.h（已装版本 2.11.0.post1）
const bool fwd_only = (!input_requires_grad(params));
const bool is_head_dim_qkv_128 = (qk_head_dim <= 128 && v_head_dim <= 128);
bool enable = is_head_dim_qkv_128;
if (arch == 220) {
  if (fwd_only) { enable = enable || is_head_dim_qkv_160; }
  else          { enable = enable && (qk_head_dim >= 64 && v_head_dim >= 64); }   // ← 只查 >=64
}
... return check_tensor_dtype(params, {kHalf, kBFloat16}, true);                  // ← 只查 dtype
```

即：`head_dim=78/90` **全部通过检查** → 派发到 `MuDNNFlashSDPAFwd/Bwd`（`SDP.cpp`）→
`sdpa.RunFlash*()`（muDNN 闭源 kernel）。**没有任何 `%8` 对齐检查**。

上游 `main` 分支持有更新的 `SDPUtils.h`（我 diff 过 blob：本地 `97bf4a4…`，main `01fc717…`），
其中已经写下了这条约束，但**只在 dropout>0 时才生效**：

```cpp
  // mutlass tme require head dim % 8 == 0
  if ((enable && has_dropout) &&
      !(qk_head_dim % 8 == 0 && v_head_dim % 8 == 0)) {
    enable = false;
  }
```

⇒ **上游知道这条约束、但漏了 dropout==0 的训练路径；main 分支至今仍然会派发错**。
这就是补丁点（见 §5）。

### 3.4 其它变量（都不改变结论）

| 变量 | 测法 | 结果 |
|---|---|---|
| dtype | bf16 / fp16 / fp32 | 只有 bf16+fp16 走 flash；**两者错得一样**（2.6e-2），fp32 走 math 恒正确（4e-7） |
| 大数值激活 | `q,k` 标准差 ×1/2/4/8/16（logits 标准差 ≈ 平方） | D=78 flash dQ：2.6e-2 / 0.21 / 0.67 / 1.41 / 2.90，同条件 math：6.0e-3 / 1.4e-2 / 5.0e-2 / 0.18 / 0.78 ⇒ **flash/math 误差比 4.3 / 15.7 / 13.5 / 7.9 / 3.7**；D=64 的 flash/math 比 **0.8 / 0.6 / 0.55 / 0.47 / 0.58**（flash 反而更好）⇒ 比值把"softmax 饱和导致 fp64 参考本身病态"这一共同因素约掉了 |
| 序列长度 N | 17/64/128/196/213/256/384/512 | 每个 N 都错（0.10–0.11），与 N 无关 |
| head_dim | 56…128 | **唯一决定因素**（§3.1） |
| `is_causal` | False/True | True 更糟：D=78 → 0.18，D=64 → 4.4e-3 |
| 显式 `scale` | 默认 / 1√d / 0.05 | 与 bug 无关（缺陷在 kernel，不在 scale 传递） |
| `dropout_p>0` | 0.1 | 被 `check_musa_attention_input` 拒绝 → 自动走 math（这个 fallback 是对的） |
| 内存布局 | contiguous vs `reshape().transpose(1,2)`（本仓真实写法） | 两种都错（7.1e-2 / 2.7e-2），**错法随布局变化** ⇒ 进一步说明是索引/分块层面 |
| B/H | B1H1 / B2H8 / B4H16 | 都错 |
| `qk_head_dim != v_head_dim` | 78/64、64/78 | **flash 给出 nan / 4.0 级垃圾梯度，且 math 直接抛错**（RunMath 不支持非对称维）→ 另一个独立缺陷 |
| cudnn/allow_tf32 | 无 cudnn（`cudnn.is_available()=False`），`matmul.allow_tf32=False` | 与此 bug 无关 |

### 3.5 为什么"训练到 1900 步才炸"：链式放大（真实 bf16 autocast，`probe_chain.py`）

`Linear(624→1872) → 8×78 头 SDPA → Linear(624→624)`，与真实块同构，bf16 autocast vs fp32 vs fp64：

| qkv 权重 scale | bf16/fp32 qkv.w 梯度范数比 | 相对 fp64 误差 | **math 后端 bf16 的比/误差** |
|---|---|---|---|
| 1 | 1.00 | 0.00 | 1.00 / 0.00 |
| 4 | 1.03 | 0.03 | 1.00 / 0.00 |
| 6 | 1.12 | 0.12 | 1.00 / 0.00 |
| 8 | 1.24 | 0.24 | 1.00 / 0.00 |
| 10 | 1.37 | 0.37 | 1.01 / 0.01 |
| 12 | **1.50** | **0.50** | 0.98 / 0.02 |

这正好复刻了旧报告用 `tools/diag_musa_grads.py` 观测到的指标（健康 0.92–1.37 → 崩后 44×/231×）：
**激活越大错得越多，且误差不随 batch/层数平均掉（系统性偏差）**。训练后期 logits 变大
→ 单步更新方向偏 10%–50% → lr=2e-4 下几十步就进高曲率区 → 梯度爆炸/跳步。math 后端全程 1.00。
（旧报告里 44×/231× 是已经崩坏的权重上的极端值，本复现给出的是同一机制的**可控刻度**。）

## 4. 与旧结论的差异（如实报告）

| 旧结论 | 本次实测 | 说明 |
|---|---|---|
| "bf16 flash/mem-efficient 反向坏" | 坏的是 **head_dim%8≠0 时的 flash 反向**，与 bf16/fp16 无关（两者同错） | 归因需要更正；`SDPA_BACKEND=math` 依然有效，但理由是"换 kernel" |
| "mem_efficient 也坏" | MUSA **从不选 mem_efficient**，三个开关里只有 flash/math 有意义 | 该说法无对象 |
| "math 治好了" | ✅ 复现：D=78 bf16 dQ 2.6e-2 → 6.0e-3（= dtype 噪声） | 一致 |
| "前向逐位一致、只有反向坏" | 前向 loss 误差 1.6e-2（bf16 正常量级），dQ/dK 2.6e-2 且 **fp16 也错** | 前向确实不是问题，"逐位一致"多半是 4 位小数的打印巧合 |
| "224² E1 崩 / T1 math 不崩" | E1/T1 都是 enc 1248/16=78、dec 1440/16=90 → **都命中 flash bug**，与 T1 的 `SDPA_BACKEND=math` 解释自洽（但两者 K 不同：48 vs 32，属混杂因素） | 一致，需注意混杂 |

## 5. 修复 / 规避（按优先级）

| # | 方案 | 代价 | 适用 |
|---|---|---|---|
| **P0（现在就能用）** | 训练里 `torch.backends.cuda.enable_flash_sdp(False)`（math 默认开，**一个开关即可**）。已由 `train_linattn.py` 的 `SDPA_BACKEND=math` 实现 | **≈0**：N=213/D=78 微基准 math **0.71 ms** vs flash **2.46 ms**（fwd+bwd，B4 H16），math 反而更快；旧报告全步 +18% 的结论建议复测（那次是 448² 端到端） | 所有 MUSA 训练；对 CUDA 无影响（默认不干预） |
| **P1** | 若必须用 flash：把 `head_dim` 改成 8 的倍数（enc 1248→16 头=78 ⇒ 改 12 头=104 或 13 头=96 或 1280/16=80；dec 1440/16=90 ⇒ 96 或 80） | 改模型宽度，需要重跑超参 | 长训、想省显存时 |
| **P2** | 用 fp32 attention（`enable_flash_sdp(False)` 且输入不 autocast 到 bf16） | 显存×2、带宽×2 | 只在小 N、精度敏感场景 |
| **P3** | 升级 torch_musa：**不保证修好**。GitHub 最新 release 是 `2.11.0.post1`（2026-08-17，与本机同版），但 main 有 `2.11.0.post2`(#155) / `2.13.0.post1`(#159) 的更新；**main 的 `SDPUtils.h` 仍然只在 dropout>0 时检查 %8**，所以升级到 main 也还会错 | 需重装/重建，风险高（生产训练在跑） | 只在上游合并 P4 补丁后 |
| **P4（给上游的补丁方向）** | `torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h`，`check_musa_attention_input()`：把 `%8` 检查从 `has_dropout` 分支里放出来，**训练路径（`!fwd_only`）一律要求对齐**，不满足则返回 false → 自动 fallback 到 math | 一行；对 8 对齐的模型零性能损失 | 根治 |

建议补丁（相对上游 `main`，`torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h`）：

```diff
-  // mutlass tme require head dim % 8 == 0
-  if ((enable && has_dropout) &&
-      !(qk_head_dim % 8 == 0 && v_head_dim % 8 == 0)) {
-    enable = false;
-  }
+  // mutlass tme requires head dim % 8 == 0; the backward kernel silently
+  // returns wrong dQ/dK for unaligned head dims, so gate it on every path
+  // (forward is fine, backward is not -- but keep it simple and gate both).
+  if (enable && !(qk_head_dim % 8 == 0 && v_head_dim % 8 == 0)) {
+    enable = false;
+  }
```

另外两处顺带该修的：
1. `head_dim` 为奇数时 `check_musa_attention_input` 放行、kernel 抛
   `Flash Attention 2 Not Support HeadDim Odd` —— 应该在检查里拒掉、静默 fallback 到 math；
2. `qk_head_dim != v_head_dim` 时 flash 返回 nan/垃圾梯度、math 直接抛错 —— 前者的检查该补
   （上游 main 已加 `const bool sym = (qk_head_dim == v_head_dim);`，可用作参考）。

## 6. 上游渠道（含确切 URL）

| 渠道 | URL | 判断 |
|---|---|---|
| **torch_musa GitHub Issues**（首选） | https://github.com/MooreThreads/torch_musa/issues | **最可能被受理**。仓库活跃（523★、34 open、2026-11-10 仍在推），且已有一个**同机器同版本的 muDNN/flash-SDPA issue**：#162（`MuDNNFlashSDPABwd fails in autocast training`，其中第 6 条就是 flash SDPA 反向），维护者 `lijing-mt` 2026-11-13 回复"we are currently analyzing it" —— 直接接着 #162 提一条独立 issue 最好 |
| 同上，issue 格式 | 该仓库**没有** `.github/ISSUE_TEMPLATE`（API 返回 404） | 按 #162 已验证被接受的格式写：`[torch_musa][muDNN][SDPA] 标题` + 共享环境块 + Description/Error/Steps/Expected vs Actual/Impact/Workaround |
| torch_musa 文档 FAQ | https://docs.mthreads.com/torchmusa/torchmusa-doc-online/FAQ/ | 只适合"怎么用"，不接 bug |
| 摩尔线程开发者论坛 | https://developer.mthreads.com/forum/ （版块例：`/forum/d/32-torch-musa-qu-dong-pei-tao-zhi-nan`） | 中文提问可以，但**没有 issue 跟踪号**，修不修不可追踪；适合并行"催" |
| PyTorch 上游 | https://github.com/pytorch/pytorch/issues | **不适合**。这是 PrivateUse1 后端（torch_musa）自己的 `_fused_sdp_choice` 覆盖 + 闭源 muDNN kernel，PyTorch 侧代码没参与；按 PyTorch 的判定标准（"reproducible on a supported backend / in-tree code"）会被 turn down。除非发现在 `sdp_utils_cpp.h` 上游逻辑本身有错（本次没有） |
| AutoDL | 工单/客服（控制台内） | **不适合当 bug 渠道**：镜像只是打包，torch_musa/muDNN 是摩尔线程的；可以在工单里要求"提供 torch_musa 2.11.0.post2 / 2.13 镜像"作为**升级路径**，但根因不在 AutoDL |

**需要附的材料**（issue 里已包含）：`min_repro.py` 全文、`env.txt`、head_dim 扫描表、
D=78 的 fwd/bwd 分列误差、fp16==bf16 同错的关键论据、`SDPUtils.h` 的派发代码与 `%8` 注释、
以及"一键规避 = `enable_flash_sdp(False)`"。

## 7. 产物

```
/root/autodl-tmp/musa_bug/min_repro.py                       # 最小复现（18 s，纯 torch）
/root/autodl-tmp/musa_bug/env.txt                            # 环境版本快照（113 行）
/root/autodl-tmp/musa_bug/ISSUE_torch_musa_sdpa_bwd.md       # 英文 issue 草稿（可直接贴）
/root/autodl-tmp/musa_bug/REPORT_musa_sdpa_bwd.md            # 本报告
/root/autodl-tmp/musa_bug/probe_{dim,misc,cols,vdim,prec,scale2,chain}.py   # 逐变量探针
/root/autodl-tmp/musa_bug/issue_snippet.py                   # issue 里那段 40 行独立复现
```

所有测试均在 `MUSA_VISIBLE_DEVICES=0`、单进程、张量 ≤ 2×8×213×90（<10 MiB）、单次 <20 s 内完成；
未 kill/暂停任何生产进程，未写 `/root/autodl-tmp/{video_recon,tok448_v2,AssocVTC_v10}`，
未改动 `/root/AssocVTC_tok448`（只读代码 + 只读 args.json）。

## 7. 上游提交（2026-11-17）

- 渠道选择：`https://github.com/MooreThreads/torch_musa/issues`（仓库活跃；**无 issue 模板**；
  已有同机同版本 meta-issue **#162**，其中第 6 条即 `MuDNNFlashSDPABwd fails in autocast training`；
  维护者 `lijing-mt` 2026-11-13 回复 "we are currently analyzing it."）。
  PyTorch 上游（pytorch/pytorch）**不适合**：PrivateUse1 后端的 `_fused_sdp_choice` 覆盖 +
  闭源 muDNN kernel，PyTorch 侧代码未参与。AutoDL 仅适合作为"能否提供 post2/2.13 镜像"的升级诉求。
- 已提交评论：https://github.com/MooreThreads/torch_musa/issues/162#issuecomment-6950049064
  （2026-11-17 13:53:56Z，作者 `shaojun0`，5192 字符，含根因、head_dim 扫描表、最小复现、
  `SDPUtils.h` 一行补丁建议、端到端 +8% 规避代价）。
- 评论正文存档：`COMMENT_torch_musa_162.md`；可直接贴的完整 issue 草稿：`ISSUE_torch_musa_sdpa_bwd.md`。
- 后续：若维护者要求单独 issue，可用 `ISSUE_torch_musa_sdpa_bwd.md` 直接开（`gh issue create`）。
