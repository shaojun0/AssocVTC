# 外网 2×A800 实验已停止并关机（2026-10-14 14:44:24）

## 为什么停
用户在内网用 DCU 同步训练，不需要再消耗 AutoDL 外网余额。

## 停止时的状态（文件都在数据盘，关机后保留）
- 最后优化步：**4240 / 77402**（1 epoch 是 38701 步，所以约 0.11 epoch）
- 最后 loss 0.86921（20 步滑动均值），per_decoder [0.841,0.844,0.842,0.852]（四档同步）
- 步速 2.565 s/step，单卡峰值 58.93 GiB / 80 GiB
- **已保存 ckpt_2000.pt / ckpt_4000.pt（各 20.56 GB，共 39 GB）** — 这是 ×10 解码器
  在六源数据上真实训过的权重，可续训也可留作对照

## 本轮已经拿到、不因关机损失的结论
- 解码器参数量 **140.9M → 1409.3M（×10.001）**，全模型 445.3M → 1713.6M
  （depth 2 / width 3072 / heads 24 / mlp_ratio 5.12）
- 2 卡 A800：2.63–2.99 s/step、峰值 58.9 GiB/卡、单 ckpt 20.56 GB
- 代码已推 GitHub：b607273（重构+×10）、a649290（DCU 路径+check_platform）、
  c1b5a4e（数据布局 --verify）
- 报告：doc/2026-10-14/REPORT_multi_portable.md

## 要恢复训练（数据/索引/权重都还在，无需重建）
    cd /root/autodl-tmp/code/SR-Diffusion-v3
    EPOCHS=2 NUM_GPUS=2 LAUNCHER=accelerate OUTPUT_DIR=/root/autodl-tmp/output/multires_10x \
      setsid nohup bash run_multi_10x.sh > /root/autodl-tmp/logs/run_multi_10x.log 2>&1 &
（想从 ckpt_4000 续、保留优化器状态，用 --resume .../ckpt_4000.pt，不要带 --reset_step_on_resume）

## 数据资产（135 GB 占用，均在 /root/autodl-tmp）
- data/ 六源 259,641 索引 + index_cache.json（绝对路径！换挂载点要重建）
- models/dinov2-large/（1.16 GB，3 个文件）
- output/multires_10x/ckpt_{2000,4000}.pt（39 GB）

## 只想要数据不要 ckpt 省存储
    rm -f /root/autodl-tmp/output/multires_10x/ckpt_*.pt
