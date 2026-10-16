# AssocVTC 训练指标可视化看板

把 **训练日志 + 探针 JSON + 全量推理 JSON** 三类数据源统一解析成 `metrics.json`，
再由两个服务呈现。当前做 **baseline / 旧 stack2x / 修复版 stack2x 三路同图对照**：

| 端口 | 服务 | 内容 | 公网地址 |
|---|---|---|---|
| **6006** | TensorBoard | scalars 事件文件（loss / grad_norm / lr / eval_recon / 探针），每个 run 一个子目录 | `https://u849831-me4w-bf7ae4cd.westb.seetacloud.com:8443` |
| **6008** | Streamlit | 交互看板（14 张 Plotly 图 + 三路对照表 + 数据源状态） | `https://uu849831-me4w-bf7ae4cd.westb.seetacloud.com:8443` |

> ⚠️ **公网地址注意大小写/字母个数**：两个域名都解析到同一 IP（`36.103.198.205`），
> 但映射到**不同的本地端口** —— 已实测确认：
> * 单 `u`：`u849831-me4w-bf7ae4cd...` → **本地 6006 = TensorBoard**
> * 双 `u`：`uu849831-me4w-bf7ae4cd...` → **本地 6008 = Streamlit**
>
> 两个域名都返回 HTTP 200，但内容不同，请按上表使用。

---

## 1. 数据源

看板目前做 **三路同图对照**：

| run key | 说明 | 颜色 |
|---|---|---|
| `stack2x_lr1e4_slice05` | **修复版**：`self.stack` 2x + 峰值 lr 1.5e-4 → **1.0e-4** | 🟢 `#2ca02c` |
| `stack2x_slice05` | 旧 run（lr 1.5e-4，后步坍缩），日志停在 step 2500 | 🔵 `#1f77b4` |
| `baseline_blockdiag_slice05` | 基线 blockdiag，final eval_recon **0.3318** | 🔴 `#d62728` |

| # | 数据 | 路径 | 当前状态 |
|---|---|---|---|
| 1 | **修复版**训练日志（**持续增长**） | `/root/train_logs/stack2x_lr1e4_slice05.log` | ✅ 训练中（约 1.96 s/it，8760 step） |
| 1 | 旧 run 训练日志 | `/root/train_logs/stack2x_slice05.log` | ⚠️ 已停止，最后 tqdm `2500/8760`（状态判定为 `stalled`） |
| 1 | 基线训练日志 | `/root/train_logs/blockdiag_slice05.log` | ✅ 完整（438 个 train 点 + 5 个 eval 点，final eval_recon **0.3318**） |
| 2 | 修复版终版探针 | `/root/autodl-tmp/sr-diffusion-v3-stack2x/output/probe/probe_stack2x_lr1e4_slice05.json` | ⏳ **待训练结束后生成** |
| 2 | 修复版 checkpoint 探针 | `.../output/probe/probe_stack2x_lr1e4_slice05_step<step>.json` | ⏳ 待生成（用于「随训练进程变化」图） |
| 2 | 旧 run 探针 | `.../output/probe/probe_stack2x_slice05_step2000.json` | ✅ 已加载（tag=`stack2x_ckpt2000`） |
| 2 | 基线探针 | `.../output/probe/probe_baseline_blockdiag_slice05.json` | ⏳ 该路径暂缺；已自动回落到 `/root/train_logs/probe_blockdiag.json` ✅ |
| 3 | 修复版全量推理 | `.../output/phase1_v2_stack2x_lr1e4_slice05/infer_test.json` | ⏳ 待生成 |
| 3 | 旧 run 全量推理 | `.../output/phase1_v2_stack2x_slice05/infer_test.json` | ⏳ 待生成 |
| 3 | 基线全量推理 | `.../output/baseline_blockdiag_slice05_infer_test.json` | ⏳ 待生成 |

### run 状态含义

`status` 由日志推断，**不会把停掉的 run 显示成"训练中"**：

| status | 判定 |
|---|---|
| `running` | 有训练点且日志在 `SRDASH_STALE_SECONDS`（默认 420s）内有新写入 |
| `stalled` | 有训练点但日志超过 420s 没更新、进度又没跑满（如旧 run 停在 2500/8760） |
| `finished` | 日志出现 `TRAIN_EXIT=0` / `[final]`，或 tqdm 已到 `8760/8760` |
| `missing` / `unknown` | 日志文件不存在 / 无法判定 |

### 日志行格式与 step 推导

每 `log_every=20` 步打印一行 Python dict，**没有 step 字段，只有 epoch**：

```
{'loss': '1.117', 'grad_norm': '0.6628', 'learning_rate': '4.523e-05', 'epoch': '0.3653'}
{'eval_loss': '0.3316', 'eval_recon': '0.3318', 'eval_runtime': '97.37', 'epoch': '40'}
```

本实验 `steps_per_epoch = 7009 // (16*2) = 219`，所以：

```
step = round(epoch * 219)
```

解析器会用日志里 tqdm 的 `N/8760` 做交叉校验（`meta.tqdm_step` / `meta.tqdm_total`）。
实测吻合：`epoch=3.653 → step=800`，tqdm 同行显示 `800/8760`；`epoch=3.744 → 820`，tqdm 显示 `820/8760`。✅

### 探针 JSON 字段

`step_px_scale`（长度 5，**就是"step1~5 的平均值"**：每步未累加增量的平均绝对值）、
`prog_curve_255`（5 步累积重建 L1）、`E_px`（5×5：step × region 的 0-255 L1）、
`E_nrm`、`z_s_within_std`、`z_s_block_cos`、`steps`、`tag`、`n_img`、`regions`、`blocks`、`ckpt`。

---

## 2. 快速开始（服务器上）

源码运行副本在 `/root/sr-metrics-dash/`。

```bash
cd /root/sr-metrics-dash

bash run_dash.sh start      # 一键：解析 + 写 TB 事件 + 起 6006 + 起 6008 + 起刷新循环
bash run_dash.sh status     # 进程 / 端口 / HTTP 自检
bash run_dash.sh refresh    # 只跑一次解析 + 写事件文件
bash run_dash.sh stop       # 停止三者
bash run_dash.sh restart
bash run_dash.sh logs       # tail 三个日志
bash run_dash.sh health     # 只做 HTTP 自检
```

### 从本地仓库部署到服务器

```bash
cd /home/linaro/dsh/AssocVTC/tools/metrics_dash
scp -P 38024 -o StrictHostKeyChecking=no \
    parse_metrics.py emit_tensorboard.py app_streamlit.py make_static_html.py run_dash.sh \
    root@connect.westb.seetacloud.com:/root/sr-metrics-dash/
ssh -p 38024 -o StrictHostKeyChecking=no root@connect.westb.seetacloud.com \
    'chmod 755 /root/sr-metrics-dash/*.py /root/sr-metrics-dash/*.sh && cd /root/sr-metrics-dash && bash run_dash.sh restart'
```

### 手动分步（等价于 `start`）

```bash
export PATH=/root/miniconda3/bin:$PATH

# 1) 解析（纯标准库，任意 python）
/root/dashvenv/bin/python /root/sr-metrics-dash/parse_metrics.py --out /root/sr-metrics-dash/data

# 2) 写 TensorBoard 事件文件（需要 torch，用 conda base 的 python）
/root/miniconda3/bin/python /root/sr-metrics-dash/emit_tensorboard.py \
    --metrics /root/sr-metrics-dash/data/metrics.json --logdir /root/tf-logs-metrics

# 3) TensorBoard（新 logdir，不碰 /root/tf-logs）
setsid nohup /root/miniconda3/bin/python -m tensorboard.main \
    --host 0.0.0.0 --port 6006 --logdir /root/tf-logs-metrics --reload_interval 15 \
    >> /root/train_logs/dash_tensorboard.log 2>&1 &

# 4) Streamlit 看板
setsid nohup /root/dashvenv/bin/streamlit run /root/sr-metrics-dash/app_streamlit.py \
    --server.address 0.0.0.0 --server.port 6008 --server.headless true \
    >> /root/train_logs/dash_streamlit.log 2>&1 &

# 5) 刷新循环
setsid nohup bash /root/sr-metrics-dash/run_dash.sh loop \
    >> /root/train_logs/dash_loop.log 2>&1 &
```

### 自检

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:6006   # → 200
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:6008   # → 200
```

---

## 3. 刷新机制

`run_dash.sh loop` 是一个后台 shell 循环，默认 **每 45 秒**（`REFRESH_SECONDS=45`）：

1. 重新跑 `parse_metrics.py` → **原子重写** `data/metrics.json` + 3 个 CSV
   （先写 `.tmp` 再 `os.replace`，看板不会读到半个文件）；
2. 重新跑 `emit_tensorboard.py` → 删除 `--logdir` 下旧的 `events.out.tfevents.*`
   再全量重写（**幂等**，日志增长不会产生重复点）；
3. TensorBoard 本身 `--reload_interval 15` 秒自动重载；
4. Streamlit 侧用 `st.fragment(run_every="30s")` 自动重跑渲染
   （旧版本 Streamlit 回落到 HTML `setTimeout(location.reload)`）。

日志增长 → 最长约 45 秒后图表自动更新，无需人工干预。

### 环境变量（改端口/间隔/路径）

`TB_PORT`(6006) `DASH_PORT`(6008) `REFRESH_SECONDS`(45) `DATA_DIR` `TF_LOGDIR`(/root/tf-logs-metrics)
`BASE_PY` `/root/miniconda3/bin/python` `VENV_PY`/`STREAMLIT_BIN`(`/root/dashvenv/...`)
`SRDASH_ROOT`（项目根，默认 `/root/autodl-tmp/sr-diffusion-v3-stack2x`）
`SRDASH_LOGDIR`（日志目录，默认 `/root/train_logs`）

---

## 4. 图表清单 & 当前数据状态

| # | 图 | 数据源 | 现在有数据？ |
|---|---|---|---|
| ① | train loss vs step（**三路同图**） | 训练日志 | ✅ 基线 438 点（完整）；旧 run 125 点；修复版增长中 |
| ② | 梯度平均变化 grad_norm vs step（对数轴可切） | 训练日志 | ✅ 三路都有 |
| ③ | learning_rate vs step（可看 1.5e-4 vs 1.0e-4 峰值差） | 训练日志 | ✅ 三路都有 |
| ④ | eval_recon vs step + 基线 0.3318 虚线 | 训练日志 eval 行 | ✅ 基线 5 点（0.486→0.4199→0.3525→0.3325→**0.3318**）；旧 run 1 点（step 2000）<br>⏳ 修复版 0 点（eval_every=2000，尚未到） |
| ⑤ | **step1~5 的平均值** `step_px_scale`（柱状，三路并排） | 探针 JSON | ✅ 基线 `1.0273→0.0335`；✅ 旧 run `0.1814→0.1814`（**完全坍缩**）<br>⏳ 修复版待生成 |
| ⑥ | `step_px_scale` 各采样步随训练进程变化 | 多 checkpoint 探针 | ✅ 旧 run 1 点（step 2000）<br>⏳ 需 ≥2 个探针才有曲线；修复版待生成 |
| ⑦ | `E_px` step × region 热力图（0-255 L1） | 探针 JSON | ✅ 基线 + 旧 run；⏳ 修复版待生成 |
| ⑧ | `E_nrm` 热力图（折叠面板） | 探针 JSON | 同上 |
| ⑨ | `prog_curve_255` 渐进曲线 | 探针 JSON | 同上 |
| ⑩ | `z_s_within_std` / 块内 cos | 探针 JSON | 同上 |
| ⑪ | 全量推理 `full_pixel_l1_255` / `step_pixel_l1_255` | infer JSON | ⏳ 三路都待生成 |
| ⑫ | 数据源状态表（OK / 缺失清单） | — | ✅ |
| ⑬ | **三路关键指标对照表**（loss / grad_norm / lr / eval_recon / px_scale / z_s） | 全部 | ✅ 见 Tab「损失/梯度/学习率」底部 |

TensorBoard 侧当前 tag 数：`baseline_blockdiag_slice05` **47**、`stack2x_slice05` **47**、
`stack2x_lr1e4_slice05` **4**（修复版探针未生成，故只有 train/*）。

---

## 5. 如何加上新 checkpoint 的探针数据

看板会**自动发现** `output/probe/` 与 `/root/train_logs/` 下的所有探针 JSON，
无需改代码、无需重启。要让某个探针出现在"随训练进程变化"图（⑥）上，必须能推断出它属于哪个 step。
按优先级：

1. **JSON 里带 `step` 字段**（最稳）：
   ```json
   { "step": 4000, "tag": "lr1e4_step4000", "step_px_scale": [...], ... }
   ```
   也认 `train_step` / `global_step` / `ckpt_step` / `steps_trained`。
2. **文件名带 step**：`probe_stack2x_lr1e4_slice05_step4000.json`、
   `probe_stack2x_lr1e4_slice05_ckpt6000.json`、`probe_stack2x_lr1e4_slice05_checkpoint-8000.json`
   （正则 `(?:step|ckpt|checkpoint|iter)[-_]?(\d+)`）。
3. **`ckpt` 字段路径带 step**：如 `"ckpt": "output/.../checkpoint-6000/model.pt"`。

都推断不出时，该探针仍会出现在图⑤（`step1~5` 柱状图）与所有探针表格里，只是不进图⑥。

### 落地流程

```bash
# 例：修复版 run 训练途中每 2000 步存了 checkpoint，对 checkpoint-4000 跑一次探针
# （用既有链路 probe_step_collapse.py 生成，注意：要用空闲 GPU 或 CPU，别抢训练资源）

# 方式 A：直接放进自动扫描目录，文件名带 step
cp probe_result.json \
   /root/autodl-tmp/sr-diffusion-v3-stack2x/output/probe/probe_stack2x_lr1e4_slice05_step4000.json

# 方式 B：放到 /root/train_logs 并命名成 probe_stack2x_lr1e4_slice05_step4000.json（同样会被扫到）

# 然后等下一个刷新周期（≤45s），或立刻手动刷一次：
cd /root/sr-metrics-dash && bash run_dash.sh refresh
```

> ### ⚠️ 关键：probe_globs 必须精确，否则三路会互相串数据
> 旧 run 的 key 是 `stack2x_slice05`，修复版是 `stack2x_lr1e4_slice05`，
> **前者是后者的子串**。如果 glob 写成 `probe_*stack2x*slice05*.json`，
> 旧 run 就会把修复版的探针也扫进来。当前用的是**精确前缀**：
>
> | run | probe_globs |
> |---|---|
> | `stack2x_lr1e4_slice05` | `probe_stack2x_lr1e4_slice05*.json` |
> | `stack2x_slice05` | `probe_stack2x_slice05*.json` |
> | `baseline_blockdiag_slice05` | `probe_*baseline*blockdiag*slice05*.json`、`probe_*blockdiag*.json` |
>
> 目录：`<PROJECT_ROOT>/output/probe/` 与 `/root/train_logs/`。
> 已用回归测试断言三路探针互不串（见 README 末尾「回归测试」）。

基线探针的"正式路径"`.../output/probe/probe_baseline_blockdiag_slice05.json` 一旦出现，
会自动优先于回落的 `/root/train_logs/probe_blockdiag.json`（`probe_files` 按顺序取第一个存在的）。

### 再加第 4 个 run 怎么做

在 `parse_metrics.py` 的 `RUNS` 字典里加一条即可（看板与 TB 会自动适配 N 路，无需改 UI）：

```python
"stack2x_lr5e5_slice05": {
    "label": "stack2x lr5e-5 slice05",       # 图例名
    "role": "exp",
    "color": "#ff7f0e",                       # 换一个不撞的颜色
    "logs": [os.path.join(TRAIN_LOG_DIR, "stack2x_lr5e5_slice05.log")],
    "probe_files": [os.path.join(PROBE_DIR, "probe_stack2x_lr5e5_slice05.json")],
    "probe_globs": [os.path.join(PROBE_DIR, "probe_stack2x_lr5e5_slice05*.json"),
                    os.path.join(TRAIN_LOG_DIR, "probe_stack2x_lr5e5_slice05*.json")],
    "infer_files": [os.path.join(PROJECT_ROOT, "output",
                                 "phase1_v2_stack2x_lr5e5_slice05", "infer_test.json")],
},
```

改完 `bash run_dash.sh restart`（刷新循环用的是长期运行的脚本副本，改了 `parse_metrics.py`
需要 restart 才会生效；只加/换探针 JSON 则不需要重启）。

---

## 6. 目录结构与文件

```
tools/metrics_dash/
├── parse_metrics.py        # 核心：logs + 探针 + infer JSON → metrics.json / 3 个 CSV（纯标准库）
├── emit_tensorboard.py     # 幂等重写 TensorBoard 事件文件（需 torch，用 conda base python）
├── app_streamlit.py        # Streamlit 交互看板（6008）
├── make_static_html.py     # 兜底：自包含静态 HTML（Plotly CDN），streamlit 装不上时用
├── run_dash.sh             # 一键 start/stop/status/refresh/loop/logs/health
├── .gitignore              # 忽略 data/ run/ *.pid
└── README.md
```

服务器运行副本：`/root/sr-metrics-dash/`（`data/` 为运行期产物）

**运行期产物**（服务器）：

```
/root/sr-metrics-dash/data/metrics.json        # 统一数据（看板读它）
/root/sr-metrics-dash/data/metrics_train.csv   # run, step, epoch, loss, grad_norm, learning_rate
/root/sr-metrics-dash/data/metrics_eval.csv    # run, step, epoch, eval_loss, eval_recon, eval_runtime
/root/sr-metrics-dash/data/metrics_probe.csv   # run, tag, step, ..., step_px_scale_s1..s5, prog_curve_s1..s5
/root/sr-metrics-dash/run/*.pid                # 进程 pid
/root/tf-logs-metrics/<run>/events.out.tfevents.*   # TensorBoard 事件
/root/train_logs/dash_tensorboard.log
/root/train_logs/dash_streamlit.log
/root/train_logs/dash_loop.log
```

---

## 7. 安全约束（已严格遵守）

* ❌ **不 kill / 不重启 / 不干扰训练进程**（`accelerate launch ... output/phase1_v2_stack2x_slice05`、
  `... phase1_v2_stack2x_lr1e4_slice05`）。
  所有脚本**只读**训练日志与 JSON，全部只跑 CPU，不碰 GPU。
* ❌ **不触碰 `/root/tf-logs` 与 6007 端口的既有 TensorBoard**；新事件写到
  `/root/tf-logs-metrics`。
* ✅ 端口 6006 / 6008 均监听 `0.0.0.0`。
* ✅ **没有往 conda base 环境装任何包**（避免升级 numpy/pillow 等把正在跑的 2 卡训练搞崩）。
  Streamlit + Plotly 装在独立 venv `/root/dashvenv`；
  写 TB 事件用的 `torch` 直接复用 conda base 里已有的版本。
* ✅ `run_dash.sh start` 在端口被占用时会**跳过启动**而不是抢端口。
* ✅ 改 run 列表只会新增 TB 子目录与一组曲线，**端口与公网 URL 不变**。

---

## 8. 回归测试

改动 `parse_metrics.py` 的 `RUNS` / glob 后，务必跑一次"探针隔离"回归
（旧 run 的 key `stack2x_slice05` 是修复版 `stack2x_lr1e4_slice05` 的**子串**，最容易串数据）：

```bash
# 造一个三路假数据集，断言 probe_history 互不串，并检查 status 判定
SRDASH_ROOT=/tmp/t/root SRDASH_LOGDIR=/tmp/t/logs \
  python parse_metrics.py --out /tmp/t/out
python - <<'PY'
import json
d = json.load(open("/tmp/t/out/metrics.json"))
for run, h in d["probe_history"].items():
    print(run, "->", [(e["tag"], e["step"]) for e in h])
assert [e["tag"] for e in d["probe_history"]["stack2x_slice05"]] == ["stack2x_ckpt2000"]
assert [e["tag"] for e in d["probe_history"]["stack2x_lr1e4_slice05"]] == ["lr1e4_ckpt2000"]
print("✅ 三路探针隔离正确")
PY
```

线上自检（每次改完重启后跑）：

```bash
curl -s -o /dev/null -w '6006=%{http_code}\n' http://127.0.0.1:6006
curl -s -o /dev/null -w '6008=%{http_code}\n' http://127.0.0.1:6008
# curl 200 不代表脚本没崩 —— 用 Streamlit 无头渲染确认 0 异常
/root/dashvenv/bin/python - <<'PY'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("/root/sr-metrics-dash/app_streamlit.py", default_timeout=180)
at.run()
print("exceptions:", [e.value for e in at.exception])
PY
```

---

## 9. 常见问题

**Q: 6008 打开是白屏 / 一直在加载？**
Streamlit 首次连接要初始化会话，刷新一次即可。若持续失败看
`/root/train_logs/dash_streamlit.log`。

**Q: TensorBoard 6006 显示 "No dashboards are active"？**
确认 `--logdir` 是 `/root/tf-logs-metrics`（不是 `/root/tf-logs`）。
刷新循环每 45s 会删掉旧事件文件重写，TensorBoard 可能在几秒内短暂读不到，稍等即可。

**Q: 某个 run 的 loss 曲线为什么只有几十个点？**
日志是每 20 step 一行，约 1.95 s/it，约 6.5 分钟才出一行。跑满 8760 步约有 438 行。

**Q: 图④ eval_recon 修复版是空的？**
`eval_every=2000`，第一次 eval 在 step 2000；训练到才会出现。

**Q: 旧 run 显示 `stalled` 是什么意思？**
它的日志停在 `2500/8760` 且超过 420s 没有新写入 —— 训练已不在跑。看板不会把它显示成
"训练中"。阈值可用 `SRDASH_STALE_SECONDS` 调整。

**Q: 想换端口 / 刷新间隔？**
`TB_PORT=6006 DASH_PORT=6008 REFRESH_SECONDS=30 bash run_dash.sh restart`
