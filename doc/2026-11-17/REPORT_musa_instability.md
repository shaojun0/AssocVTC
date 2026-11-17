> 🔴 **根因已于同日精确定位并更正**（详见 `REPORT_musa_sdpa_bwd.md`）：
> 真正的触发条件不是 "bf16"，而是 **muDNN flash kernel 的反向在 `head_dim % 8 != 0` 时算错 dQ/dK**
> （本仓 enc 1248/16=**78**、dec 1440/16=**90** 全部命中；bf16 与 fp16 错得一样 ⇒ 算法/索引 bug，
> 与 dtype 无关；前向与 dV 正常）。下文"bf16 SDPA 反向"的描述是当时的近似归因，正确结论以
> `REPORT_musa_sdpa_bwd.md` 为准；`SDPA_BACKEND=math` 依然是正确的规避手段（它换的是 kernel，
> 不是精度）。另外实测端到端代价是 **+8%**（1.019 → 1.098 s/step），不是下文写的 +18%。
> 上游渠道：https://github.com/MooreThreads/torch_musa/issues/162（同机同版本，维护者已回复"正在分析"）。

# MUSA 训练崩溃根因：bf16 SDPA 反向数值坏（2026-11-17）

## 0. 一句话

`connect.sha1`（4×MTT S4000 / torch_musa 2.11）上，`F.scaled_dot_product_attention`
的 **flash / mem-efficient 后端在 bf16 autocast 下反向数值是坏的**：同一权重、同一 batch，
前向 loss 与 fp32 逐位一致，但回传的编码器梯度可达 fp32 的 **30–230 倍**。到训练后期
（梯度尺度和激活变大）这种坏梯度被触发，一次更新就把权重打进高曲率区，随后
`--skip_grad` 大量跳步、val 一路下滑。

**规避**：训练时把 SDPA 后端钉到 math —— 新加环境变量 `SDPA_BACKEND=math`
（`train_linattn.py` 启动时设置 `torch.backends.cuda.enable_{flash,mem_efficient}_sdp(False)`
+ `enable_math_sdp(True)`）。224² CUDA 参照配置在 MUSA 上由「ep2 崩」变成完全复现 CUDA。

> ⚠️ 更正：`RUN_tok448_2stage.md` §2 曾把 448² 崩溃归因为「窄瓶颈信息压力」，
> **该归因是错的**——全宽（`zs_dim=0`）在 MUSA 上同样崩；两者都只是这个 SDPA bug 的表象。
> 信息论/曲率那套分析对「为什么高 lr 危险」仍然成立，但不是本次崩溃的主因。

## 1. 现象：3 次崩溃、2 次正常

| 实验 | 配置 | 结果 |
|---|---|---|
| `tok448_k16`（11:40） | 448²/K16/zs448（窄瓶颈），4 卡 DDP，bf16，lr 2e-4 | ep1 **13.31** → ep2 12.77 → ep3 12.32 dB；gn 51→254→603→1210；跳过 42% |
| `tok448_2stage/s1`（今日 14:46） | 448²/K16/**全宽** zs=0，同协议 | ep1 **13.20** → ep2 12.04 dB；step 1885 起 enc 梯度 3.3→7.5→31→…→5.3e4；skip 26%@step1900 |
| `diag/E1_224_ctl`（15:44） | **224² CUDA 参照配置**（K32/全宽，4 卡 DDP micro32） | ep1 **13.24** → ep2 **11.16** → ep3 10.77 dB；gn 454、26 skip —— **CUDA 上稳定的配置在 MUSA 上照样崩** |
| `diag/T1_224_math`（16:46） | 同 E1，但 `SDPA_BACKEND=math` | ep1 **13.27** → ep2 **13.97** → ep3 14.56 dB；max enc gn **5.85**、**0 skip** |
| `tok448_v2/s1`（16:56 起，进行中） | 448²/K16/全宽 + `SDPA_BACKEND=math` | step 3250 时 loss 0.46–0.51、gn 0.8–1.2、**0 skip**（旧实现在 ~1885 就爆） |

CUDA 参照（`doc/2026-10-24/artifacts/scale/train_logs/m240_t1.jsonl`，224²/K32，A800 单卡 bs128）：
ep1 13.26 → ep2 14.20 → ep3 16.87 dB，grad_norm p50 0.20 / p90 0.51 / max 9.33，0 skip。
⇒ T1 与参照逐项吻合（ep2 差 0.23 dB，来自 crop RNG/数据管线差异，量级正常）。

## 2. 证据链（`tools/diag_musa_grads.py`）

同一 ckpt、同一 batch，分别用 `torch.autocast(bf16)` 与 fp32 前反向，比较分组梯度范数：

**健康权重（step 950）**：bf16/fp32 = 0.92–1.37 ⇒ bf16 总体正常，不是什么"bf16 一律不行"。

**已崩权重（step 1900, `evidence/ckpt_step1900_diverged.pt`）**：
```
[b09] loss bf16=0.9772 fp32=0.9772 | gn bf16=290.16 fp32=6.542  ratio=44.4 | enc=290.055/6.446
[b11] loss bf16=1.0200 fp32=1.0201 | gn bf16=65.325 fp32=0.282  ratio=231.6| enc=65.078/0.209
[b07] loss bf16=0.8992 fp32=0.8991 | gn bf16=4.044  fp32=0.563  ratio=7.2  | enc=3.977/0.538
```
- 前向 loss 完全一致（bf16 前向没问题）；
- `dec_head` 梯度两边逐位相同（0.254/0.254）⇒ 像素头正常；
- 差异全部集中在 **编码器/注意力路径**（enc、dec_mem、dec_stack）；
- 换 math 后端后，同一批数据多数 batch 的 bf16/fp32 回到 **1.00**（坏权重上残留 23.9×/3.39×，
  属已被污染状态，不是新 bug）。

## 3. 排除项（都测了，均正常）

| 嫌疑 | 测法 | 结果 |
|---|---|---|
| AdamW 实现 | MUSA 上一步 AdamW（foreach/非 foreach）vs 手算公式 | 相对误差 3e-8 ✅ |
| 梯度裁剪 | 注入范数 9.83 的 grad 后 `clip_grad_norm_(1.0)` | 返回值=原范数、梯度被归一到 1.0 ✅ |
| autocast 策略 | 探针各算子 dtype/误差 | linear→bf16、layer_norm→fp32、softmax→fp32、SDPA→bf16；误差 2.4e-3/3.6e-3（正常）✅ |
| 数据管线（tvmini） | 与 torchvision 0.23 语义对齐（RRC 面积/长宽比、BICUBIC、/255、Normalize） | 分布等价 ✅ |
| 窄瓶颈（zs_dim=448） | 同一 MUSA 上跑全宽对照 | 全宽同样崩 ⇒ 不是瓶颈 ✅ |
| 分辨率（448²） | 224² 参照配置 | 224² 在 MUSA 也崩 ⇒ 不是分辨率 ✅ |

## 4. 修复与复现

```bash
# 修复：SDPA 后端钉 math（train_linattn.py 已支持）
SDPA_BACKEND=math torchrun --nproc_per_node=4 ... train_linattn.py <原参数>
```

- 代码：`train_linattn.py` 顶部 `_SDPA_BACKEND = os.environ.get("SDPA_BACKEND", "")`
  （`math|flash|mem_efficient`，空 = 不干预；CUDA 机器默认行为不变）。
- 代价：448² 实测 ~1.1 s/step（原 0.93，**慢 ~18%**，math attention 在 N=213 上仍便宜）。
- 注意：**训练与评测必须同一个后端**（本仓库其它 CUDA 结果不受影响，因为默认不干预）。

## 5. 后续

- 两阶段正式 run（`/root/autodl-tmp/tok448_v2`，`doc/2026-11-17/run_tok448_2stage_v2.sh`）：
  阶段 1 全宽 16×1248 45ep（≈13 h）+ 阶段 2 SVD→16×448 3ep（≈1.2 h），全程 `SDPA_BACKEND=math`；
- 崩掉的权重留档在 `/root/autodl-tmp/tok448_v2/evidence/`（step950 健康 / step1900 崩）；
- 训练完成后：ImageNet-100 val + SAVOIAS 逐图 dump → `tools/analyze_savoias_single.py` → 报告。
