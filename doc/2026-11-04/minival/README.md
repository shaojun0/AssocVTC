# 多预算联合训练最小验证（2026-11-04，4080S 单卡）

> 回答的问题：AssocVTC 想要「所有 token 预算同时训练、不随机抽」（类 RNN→Transformer 的训练并行/推理串行），
> 是否可行、打包+掩码是否等价、无掩码会怎样、随机抽（MQT 式）是否够用。

## 一句话结论

**机制成立**：掩码打包 ≡ 多次独立前向（`max|Δ|=1.1e-5`）；3000 步收敛后联合训练不仅不交对齐税，反而
「负税」（B 的 K=32 L1 0.0521 vs 单预算 0.0911，−42.8%），且低预算是真训出来的（截断基线差 2.18×）；
无掩码则按预算切分、低预算崩溃且训练/部署形态不一致；随机抽方差大、同计算下不如联合。

## 目录

| 路径 | 内容 |
|---|---|
| `REPORT.md` | 完整报告（设计/结果/与 9-22、9-24、10-16 历史证据对应/诚实边界/A800 建议） |
| `code/minival.py` | Track 1 全部 4 臂 + 打包等价性检查 + 串行升级模拟（自包含纯 PyTorch） |
| `code/{rerun_seeds,long_run,c_diag,analyze}.py` | 补充 seed / 3000 步收敛 regime / 无掩码诊断 / 汇总 |
| `code/probe_real_model.py` | Track 2：真实 63M 模型（112²/patch16/N=49）single vs joint 探针 |
| `code/model_v2_linattn.py` | 真实模型只读副本（来源仓库主分支） |
| `results/results*.json` | 各 run 原始数字（1500/3000 步、补充 seed、无掩码诊断、Track 2） |
| `results/merged_results.json` / `analysis.md` | 合并数字与自动汇总表 |
| `logs/*.log` | 原始训练/评测日志（`git add -f` 入库，`.gitignore` 默认忽略 *.log） |

## 复现

见 `REPORT.md` §2.4；全部实验串行跑在单卡 RTX 4080S 32G 上，总计约 33 分钟 GPU 时间。

## 边界

玩具 0.98M / 32² 合成几何图，K 区间接近饱和（K=8≈K=32），无法复现真实 448 主线 K 单调性与对齐税量级；
收敛 regime 仅 1 seed；真实 `train_linattn.py` 因实例缺 transformers 等依赖未跑完整 joint-vs-single 对照。
