# artifacts — 官方 ALIT 单卡复现（2026-10-17）

> 报告正文：[`../../REPORT_alit_official_repro.md`](../../REPORT_alit_official_repro.md)
> 上游代码：`ShivamDuggal4/adaptive-length-tokenizer` @ `8c0b4ec`（2025-02-11）
> 机器：AutoDL 单卡 RTX 4080 SUPER 32 GB（已关机）
> 数据：ImageNet-100 train 126,689（本实例无 val）

## 这是什么

把**官方 ALIT**（不是我们移植的版本）在本机跑通、实测吞吐、并把停机前的全部证据留档。
用途：给移植线一个**官方参考点**和**实测预算**，不改变主线判决。

## 快速判读

- **预算结论**（`bench_alit_tiny.txt` + `s1_progress.json`）：
  `alit_tiny`(52.8M 可训) / 256² / 8 rollout 全监督 = **0.478 s/iter、1.05 h/epoch（IN-100, bs16）**；
  200 epoch 单卡 **8.6 天**；16 GB 卡（bs8）**13.8 天**；只采 2 个迭代算梯度 **5.7 天**。
- **epoch-0 阶梯是平线**（`ladder_epoch0.json`：62.6±0.3/255）——这是 warmup 阶段的**启动态基线**，
  **不能**用来说明"ALIT 的阶梯不存在"；有意义的阶梯要 epoch ≥ 5（本次关机前未产出）。
- `vqgan_recon_check.png` 是替代 base tokenizer 的重建检查：能重建但偏糊（PSNR 20.2 dB），
  ⇒ **绝对 FID 不可与论文比**，只比阶梯形状。
- `patches.diff` + `taming_model/` 是让官方代码在本环境跑起来的**必要改动**（5 处，详见报告 §5.1）。

## 文件

| 文件 | 内容 |
|---|---|
| `REPRO_NOTES.md` | 服务器上原样的复现笔记（命令、patch 说明、论文附录基线） |
| `patches.diff` | 5 处兼容 patch（unified diff，基线 `8c0b4ec`） |
| `taming_model/model.py` | 官方 taming 版（整文件替换用）；`model.py.alit_vendored_bak` = 上游 Mage 版备份 |
| `s1_progress.json` | 训练配置 + 实测吞吐 + 已完成 epoch + ckpt 清单（机器可读） |
| `bench_alit_tiny.txt` | 8 档 batch/梯度吞吐与显存 |
| `train_s1_full.txt` | stage-1 训练日志（epoch 0 完整 + epoch 1 部分） |
| `driver_s1.txt` | 自动续跑 driver 日志（含断点恢复两次启动） |
| `env.txt` | GPU/驱动/torch/timm/PL 版本与机器规格 |
| `ladder_epoch0.json` / `ladder_epoch0_grid.png` | epoch-0 的 32→256 阶梯（启动态基线，勿当结论） |
| `vqgan_recon_check.png` | 替代 base tokenizer 重建检查（上=原图/中=[0,1]/下=[-1,1]） |
| `scripts/run_s1_30ep.sh` | 训练 driver（自动续跑、每 epoch 落 ckpt、`--save_freq`） |
| `scripts/bench_alit.py` | 吞吐/显存 bench |
| `scripts/eval_ladder.py` | 32→256 逐档 L1/PSNR + 拼图（分批防 OOM） |
| `scripts/ladder_watch.sh` | 后台轮询 checkpoint 自动评测 |

## 未入库

- checkpoint（1.33 GB/个）与 VQGAN 权重（958 MB）：留在实例数据盘 `/root/autodl-tmp/`，跨关机保留；
  按报告 §5.1 的命令可从 HF 重新拉取 base tokenizer，再续跑重建。
