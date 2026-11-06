# 评测对齐混合集训（eval-aligned mixture）脚本

对应文档：`doc/2026-11-06/PLAN_eval_aligned_mixture.md`

| 脚本 | 作用 | 运行位置 |
|---|---|---|
| `build_img_eval_mix.py` | 图像类评测数据（DocVQA/TextVQA/ChartQA/MMStar/MathVista/MathVision/We-Math/MMMU）→ jsonl + 图片 | 主训练机 |
| `prepare_mix.py` | 合并 text/img/ImageNet 锚定 → `mix_train.jsonl`、`mix_heldout.jsonl`；`stats`/`check` 子命令 | 主训练机 |
| `rebalance_mix.py` | 在既有 `mix_train.jsonl` 上按 source 重采样 → `mix_train_bal.jsonl`（第一版，每图像任务 ≤2,000） | 主训练机 |
| `expand_docvqa.py` / `expand_docvqa_train_split.py` / `expand_chartqa_train.py` | 图像任务扩容（DocVQA validation 续扫 +49、DocVQA train 865、ChartQA train 4,500），带 heldout 同题/图片校验 | 主训练机 |
| `rebuild_mix2.py` | 第二版训练集：文本 82k + 图像 40.3k（大任务封顶 6,000、ImageNet 3,000）→ `mix_train_bal2.jsonl`，逐行校验 | 主训练机 |
| `preflight_mix.py` | CPU 全量预检：token ≤1024、图片存在、Dataset+Collator+expand_budget_rows 形状/移位/target 一致 | 主训练机 |
| `run_eval_mix.sh` | run3 启动（`mix_train_bal.jsonl`、epochs 按 DEADLINE 自动倒推、峰值 LR 减半、磁盘守卫、`DRY=1` 预演、结束后自动拉起评测链） | 主训练机 |
| `eval_after_run2.sh` | 等 run2 结束 → 三组视觉消融 + 通用 bench | 主训练机 |
| `eval_after_run3.sh` | 等 run3 结束 → mix_heldout 三组消融 + ImageNet-100 val + 通用 bench + 分任务表 | 主训练机 |
| `eval_mix_breakdown.py` | 按 source × budget 的准确率表（回答"预算机制在哪类任务上成立"） | 主训练机（GPU） |
| `make_report.py` | 把 eval.json/bench json 拼成报告 markdown 表格 | 主训练机 |

数据（远端）：`/root/autodl-tmp/vlm8b/data_eval/`，**训练用 `mix_train_bal2.jsonl` 122,259 行（文本 82,000 + 图像 40,259：大任务封顶 6,000、ImageNet-100 3,000、中小任务到数据上限）**；历史版本 `mix_train_bal.jsonl`（102,396）与 `mix_train.jsonl`（179,845）保留可回退；留出 `mix_heldout.jsonl` 15,702 行不变。
