# artifacts/alit 产物清单与口径边界（执行侧交付说明）

> 本目录 = **原始数字与产物**。"成立/否证/数据不足"三态判决由 alit-report 的
> `tools/alit_report.py` + PLAN §8 唯一裁定，见 `report_tables.md`。本文件不裁定判决。

## 1. 代码/脚本（本 agent write scope）

| 文件 | 作用 |
|---|---|
| `../../../../run_alit_a.sh`（仓库根） | E-A 零训练流水线：preflight→回归门(FAIL 硬停)→A0 四臂复现→A1 dump→A2/A4/A6→A3 三级 OOD→A5；缺依赖写 `SKIP_*.json` 后继续 |
| `../../../../run_alit_b.sh`（仓库根） | E-B 重训臂：臂表 + 断点续跑（status.json/ckpt_last.pt）+ 每臂硬超时 + 失败隔离 + 缺开关显式 SKIP + 每臂即时同进程评测 |
| `gate_diff.py` | 回归门：新代码默认路径 vs 发布版逐字段/逐位 diff |
| `zs_redundancy.py` | E-A5 第 1 项：`metrics_linattn.zs_stats` → z_s 冗余度 |
| `alit_a_numbers.py` | E-A 原始数字汇总（**不含判决**） |
| `collect_b_arms.py` | 从 `eval_*.json`+`stepwise_*.json` 重建 B 臂总表（PSNR / s7 / s8 / Δ₁% / 众数 / vs 基线） |
| `truncation_table.py` | E-B4 截断可用性表（**无需额外训练**：stepwise 曲线即截断曲线） |
| `probe_region_linattn.py` | E-B2 判据探针（README §4.1 三条）：`step_px_scale[t]`、`scale_ratio`、`E_px[t][r]`、对角/非对角比、对角 CV |
| `NOTES_run_pitfalls.md` | 运行坑（batch_size 默认 128、`--smoke --limit` 空转、grad-ckpt 显存事实）+ 口径边界 + B 阶段继续的决策留痕 |

## 2. 产物 json

| 文件 | 内容 |
|---|---|
| `baseline_four_arm_eval.json` | 四臂同进程 5000 张复现（26.6020/31.2186/25.3043/28.5823 dB，max\|Δ\|=0.0004） |
| `stepwise_{linattn_k64,stdattn_k64,linattn_k64_rec,stdattn_k64_rec}.json` | 发布版代码四臂逐步（E-A0 基准） |
| `stepwise_stdattn_k64_rec_published_code.json` | 回归门冻结基准 |
| `stepwise_gate_stdattn_k64_rec.json` + `gate_diff_stdattn_rec.json` | 回归门（**verdict=PASS, max\|Δ\|=0**） |
| `per_image_{stdattn_k64_rec,linattn_k64_rec,in100_stdattn_rec,coco_stdattn_rec,far_stdattn_rec}.json` | E-A1 每图每步（L1/PSNR/最优停步/类别/文件/复杂度代理） |
| `tsc_stdattn_rec.json`, `tsc_linattn_rec.json` | E-A1/A2/A4/A6 分析（含 `matched_equal_budget`、Spearman ρ+CI、Δ_t 分解） |
| `domain_ladder.json` | E-A3 三级域阶梯（iid/coco/ade20k） |
| `ood_{in100,coco,far}_stdattn_rec.json` | E-A3 各域逐步 |
| `zs_redundancy_stdattn_rec.json` | E-A5：K=64/D=448, eff_rank=47.08(0.736×K), cos_mean=0.2569, frac(cos>0.9)=0.0030 |
| `alit_a_numbers.json`, `alit_a_summary.json`, `alit_a_steps.tsv` | E-A 数字汇总/步骤 rc/耗时 |
| `eval_<arm>.json` | 每臂**同进程**全量 5000 张评测（含 stdattn_k64_rec 与 stdattn_k64 对照） |
| `stepwise_<arm>.json` | 每臂逐步（L1_255/PSNR/Δ_t/最优停步/Δ₁占比） |
| `b_arms_table.json` | B 臂总表（11 臂）+ Δs7/Δs8/众数 |
| `truncation_table.json` | E-B4 截断表（t=3/5/7/8 → 15/35/63/64 token）+ Δ vs 基线 |
| `probe_region_all.json` | E-B2 判据探针（9 臂；prog_curve 与 stepwise 互校 max\|Δ\|=3.5e-6） |
| `probe_region_tail_nockpt.json` | 追加 2 臂（b1_tail27, b1_mean_nockpt） |
| `probe_region_all11.json` | **11 臂合并版**（报告优先引用这份；`merged_from` 字段注明来源） |
| `task_fidelity_stdattn_rec{,_cls,_lstsq}.json` | E-C1 任务保真探针三臂（primary / use_cls / lstsq） |

## 3. 口径边界（报告必须连同数字一起写）

1. **同机复跑基线 = 28.4094 dB ≠ 历史 28.5823 dB（−0.173 dB）**。architecture 逐项相同（params 63,054,528、step 11406、epoch 6）；
   差异候选 = 单种子重跑漂移 + `dec_grad_ckpt` 开关（历史 args/cfg **无该键**，git 显示该开关是最近 commit `da95749` 才引入）。
   本报告**不宣称复现 28.582**；B 臂一切比较以 `b1_mean=28.4094` 为同机同协议基线（全部 `--dec_grad_ckpt 1`）。
2. **历史 rec 臂的 grad-ckpt 不可分离**：旧实例 8 臂表与本轮 B 臂在这一点上存在同一混淆，由 alit-report 标注。
3. **"末步"定义**：T=8 协议下模型交付输出 = **step8（t=64）**；step7（t=63）是次末档/最优停步档。本目录所有表同时给 s7/s8。
4. **Δ₁ 不可跨监督口径比较**：`Δ₁` 在监督目标改变时口径不同（dh1 降到 18.49%，dh2/dh3 升到 32.87%/33.72%，方向相反）。
   只能在同监督口径内比较（b1 系列、b3_grow 可读；dh 系列须单独标注）。
5. **README §4.1 判据 (1) 的 0.015 阈值**出自旧探针（`model_v2.py` + 576 patch/5 区 + 另一 PixelHead 输出空间），
   与本次 196 patch/8 区**不可假定同尺度** ⇒ 报告用**尺度无关比值**（`scale_ratio`、对角/非对角比），阈值可比性交 alit-report。
6. **E-C1 探针可分辨力受限**：各档 train_top1 0.68–0.77 而 val_top1 0.049–0.103（100 类 chance 0.01），严重过拟合；
   `indistinguishable` 应表述为"在该探针可分辨力下测不到差异"，不能表述为"自适应 = 固定"。
7. **未做项**：B4 专训档 K′=16/32/48（无法热启动、3 臂从零 ≈15 h 超预算 ⇒ 数据不足 + 成本理由）；
   `b1_increasing` 第二 seed（排期牺牲，效应 +0.448 dB ≫ 噪声阈值）；E-A5 注意力熵（PLAN 允许后置）。
8. **全部单种子（seed 0）**；任何 <0.10 px / <0.5 pp 的差异不作结论（PLAN §9 口径）。
9. **OOD 三级**：IID=IN-100 val(5000) / 中距=COCO2017 val2017(5000) / 远距=ADE20K val(**2000**)；
   ADE20K 原始分辨率不一，走与 IID 相同的 val transform（Resize256+CenterCrop224）。
10. **GPU 独占**：所有 GPU 任务串行执行；无并发。新实例 GPU 任务已全部结束（最终臂 = `b1_mean_nockpt`）。
