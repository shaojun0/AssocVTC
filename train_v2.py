"""
AssocVTC Phase 1 v2 — 训练（test 分支: 注意力机制改写后, 像素目标）
=================================================
⚠️ 项目目标（权威版 2026-08-28 修订 v2, 见 doc/2026-08-28/GOAL_compression_for_nlp.md）:
    本训练是 Phase 1 的**脚手架**——用像素重建代理任务练编码器的
    "token 压缩 + 联想"能力; 完成后冻结编码器接 Qwen 做 NLP 解码。
    **验收标准是 Phase 2 文字生成质量**; 像素重建 = **信息保持探针**
    （能还原像素 ⇒ z 携带整图信息, 重建质量决定 NLP 天花板）——不是
    "不追", 而是追"K 压缩下活信息保真"（纹理级清晰度属死信息才不追）。
架构（model_v2.py, register 式唯一路径）: DINOv2-large(参数不冻结) +
    register specials(K) 直接拼进输入序列 [cls; specials(K); patches(N)]
    (1+K+N token, 全双向) → OutputQueryDecoder（**顺序循环**: 每步只读自己那块
    z_s 切片, 平方采样计划）→ PixelHead → **像素 patch 预测**, 平权全覆盖 L1
    重建原始像素 pixel_values（非 DINO 特征）。无 TextDecoder、无 ReEncoder。
    register 数 K = num_specials 与 patch 数 N 解耦: 默认（--num_specials 0）
    由**最终生效采样步集**自动推导 K = min( max_{t∈steps}((⌊√t⌋+1)²−1), N )
    ——编码器 register 数 = 解码器实际读的范围, 不存在"花瓶 register"
    （见 model_v2.py derive_num_specials 与 doc/2026-09-02/
    DESIGN_v2_num_specials_from_max_steps.md）。全量默认（无
    decoder_steps/slice 切片）→ K=N（向后兼容）; 例: N=576 slice [4:9] →
    steps=[25,36,49,64,81] → K=99; 显式 steps=[64] → K=80。

2026-08-27 重大修复（用户发现）:
    监督目标从「DINO patch 特征」改为「原始像素 pixel_values」——
    之前特征目标退化（工地图特征空间近常数, 学质心即低 L1, 假收敛）;
    像素目标有真实空间结构, 强制模型保留空间信息。
    PixelHead: 每 patch 特征 (N,D) → (N, 14*14*3) 像素解码。

2026-08-27 训练口径调整（用户要求, 沿自上轮）:
    · 去掉加权体系: 全部采样步**平权**（loss = mean_t L1, 无 density/
      uniform/capability 权重）—— decoder 的 loss_weight 机制已整块删除;
    · 全 fp32 训练（**默认口径**）: 训练/评估不用 bf16/fp16 —— 公平性（之前
      bf16 算 fp32 存被质疑）; TrainingArguments 默认不设 bf16/fp16, 不套 autocast。
      ⚠️ 2026-09-22 起新增 **`--fp16`**（AMP: autocast + GradScaler, 权重仍
      fp32）与 **`--optim`**（如 `adamw_bnb_8bit` = bitsandbytes 8-bit AdamW）
      两个开关, **默认值 = 上面的历史口径逐位不变**; 高分辨率（896×504）上
      实测 fp32→fp16+8bit 把 448 ms/img 压到 102 ms/img（见 tools/run_896_*.sh）。
    · batch_size 默认提高到 16/卡（97GB 显存充裕）。

2026-09-02（K 与 N 解耦 + 修复 train/infer 与 model_v2.py 的接口脱节）:
    · 删除 ReEncoder 时代残留的 --reencoder_depth / --no_causal_specials /
      --register_specials 参数与 model.init_reencoder_from_dino() 调用——
      register 式是唯一路径, model_v2.py 已无这些接口（旧 train_v2.py 会
      直接 TypeError/AttributeError）。
    · --num_specials（默认 0 = 自动）: register/specials 数 K 与 patch 数 N
      解耦——K = 解码器实际读取的 z_s 范围, 由最终采样步集自动推导
      K = min( max_{t∈steps}((⌊√t⌋+1)²−1), N )（全量默认 → K=N 向后兼容;
      N=576 slice [4:9] → steps=[25,36,49,64,81] → K=99; 消除"花瓶
      register", 证据见 DESIGN doc）。显式 K>0 时模型断言 max(采样步) ≤ K
      （复现旧 checkpoint 权重形状时须显式传回旧 K, 如 K=N）。序列 =
      [cls; specials(K); patches(N)] = 1+K+N token, N 只管查询基行数/输出
      patch 数。
    · model_info.json 记录 num_specials（推理/可视化按它对齐权重形状——
      K 错了 checkpoint 形状就对不上, strict load 即崩）与 decoder_steps。

2026-09-15（顺序循环成为唯一解码路径; 循环 carry 截断梯度）:
    · 针对"后步输出≈0、整图由 step-1 一肩扛"的零增量自锁, 解码器从"一次并行
      算完 T 步 + 输出累加"改成**顺序循环**: step1 查询 = query_base, step t≥2
      查询 = query_base + 上一步输出; 每步只读自己那块 z_s（读窗口 = 该步平方块
      的 A 切片, 首步含 z_cls）, 历史输出经循环携带 ⇒ 后期步可基于"当前画到哪"
      做残差修正。**并行路径已整块删除**（跨步 tgt_mask / query_mask_mode /
      --recurrent* 一并消失; 复现并行时代产物请从 git 取回当时的 model_v2.py）。
    · 循环 carry detach（当时 decode 侧还有一层"累加 carry detach", 现已随累加
      口径删除）⇒ 每步 Y_t 恰从自己那一步的损失收 1 份梯度（步间无梯度回流）;
      代价 = 时间（步间顺序依赖, 无法并行算完 T 步）。
    · ⚠️ detach 的实测语义（2026-09-15 复核, 与 doc/2026-09-15/
      DESIGN_v2_recurrent.md §2.4 记的"默认 recurrent_detach=False = 整条循环
      反传 BPTT"不同）: carry 被硬编码 detach ⇒ ①∂L_t/∂Y_{t-1}=0,
      循环只有**前向**耦合, 没有任何损失项要求"上一步输出成为对下一步有用的
      草稿"; ②query_base 只从 step0 的损失收梯度（∂L_t/∂query_base=0, t≥1）。
      "后期步基于当前画布做残差修正"目前只有前向通路支撑; 要 BPTT 口径需改
      model_v2.py OutputQueryDecoder.forward 里那一行（已无开关）。

2026-09-15（损失口径 = 直接预测, 无累加）:
    · 训练 loss = `mean_t L1(PixelHead(Y_t), target)`: 每个采样步的输出 Y_t
      **各自直接**过 PixelHead 预测整图, 各步平权（全轨迹深监督）, **不做任何
      累加/集成**。`F_hat = Y_pix[:, -1]`（最后一步的直接预测）, 监控量
      `recon` = 它的 L1 = 训练 loss 的最后一项。
    · 旧口径（`--loss_mode {cumulative,final}` + `--loss_decouple` 切"cumsum
      累加结果"）已整块删除: 传这些开关直接 TypeError（不静默忽略）。

HF Trainer 风格（消除造轮子）:
    · 训练循环 / 梯度累积 / 调度器 / checkpoint / 分布式 → 全部交给
      transformers.Trainer + TrainingArguments（lr_scheduler_type="cosine"
      + warmup_ratio, save_strategy="steps", ddp_find_unused_parameters=
      False, report_to=[]）; 优化器 → Trainer 默认 AdamW。不再手写
      acc.no_sync / 手动 grad_accum / get_scheduler / acc.save_state。
    · 数据 → data_v2.py 的 ParquetImageDataset + V2Collator 直接作为
      train_dataset / eval_dataset / data_collator 传给 Trainer。
    · 模型 forward 返回 dict{"loss", ...}: Trainer.compute_loss 原生支持
      dict 输出取 "loss" 键（transformers 4.x / 5.x 均如此:
      loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0],
      dict 缺 "loss" 才报错）——无需子类化 Trainer。

数据（data_v2.py）: 原图 → 旋转(最优角) → 等比缩放 → 居中填充 1600:900
    (16:9) 画布 → 16:9 模型输入 (448×252)。轮廓不变形、内容面积最大化。

多卡: accelerate launch --multi_gpu --num_processes N
（Trainer 在 accelerate launch 下自动接管 DDP, 等价于 Trainer 自管 DDP;
全 fp32 无 autocast, 无需显式 .to(device)）

用法:
    accelerate launch --multi_gpu --num_processes 2 \
        train_v2.py --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --output_dir output/phase1_v2_pixelfp32
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from transformers import Dinov2Model, Trainer, TrainingArguments
from transformers import TrainerCallback
from transformers.trainer_utils import set_seed

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2, fixed_block_starts


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="train phase1 v2 (DINOv2-large unfrozen, OutputQueryDecoder)")
    p.add_argument("--data_dir", required=True, help="parquet 目录(train-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--output_dir", default="output/phase1_v2")
    p.add_argument("--model_input", default="448x252", help="16:9 模型输入 WxH (14 的倍数)")
    p.add_argument("--canvas", default="1600x900", help="旋转+缩放+填充目标画布")
    p.add_argument("--angle_step", type=float, default=0.5, help="最优旋转角网格步长(度)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=0, help="0 = epochs 决定")
    p.add_argument("--batch_size", type=int, default=16, help="每卡 batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    # ── 精度 / 优化器（2026-09-22 新增; 默认值与历史行为逐位一致）──
    p.add_argument("--fp16", action="store_true",
                   help="fp16 混合精度（AMP: autocast + GradScaler, 由 accelerate "
                        "接管; 模型权重与 final_model.pt 仍为 fp32）。默认关 = 纯 "
                        "fp32（历史基线口径）。实测 896×504/d4 上 448→102 ms/img")
    p.add_argument("--optim", default="",
                   help="优化器（转发给 TrainingArguments.optim）: **留空 = 不传, 用库默认**"
                        "（transformers 5.16.1 实测 = adamw_torch_fused, 与历史各臂一致）; "
                        "可显式指定 adamw_torch / adamw_torch_fused / adamw_bnb_8bit"
                        "（8-bit AdamW, 需 bitsandbytes）/ paged_adamw_8bit …")
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本(调试)")
    p.add_argument("--eval_limit", type=int, default=0, help="eval 只用前 N 条(冒烟)")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Trainer checkpoint 目录(如 output_dir/checkpoint-123)")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟: 不存 checkpoint, 但照常导出 final_model.pt")
    # ── 模型（register 式: DINOv2 + specials(K) + OutputQueryDecoder）──
    p.add_argument("--num_specials", type=int, default=0,
                   help="register/specials 数 K: 0=自动由最终采样步集推导 "
                        "(K = min(max_t((⌊√t⌋+1)²−1), N); 默认); >0=显式 K "
                        "(须 ≥ 最大采样步值; 复现旧 checkpoint 时显式传回旧 K)")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--decoder_depth", type=int, default=2,
                   help="OutputQueryDecoder 的 TransformerDecoder 层数")
    p.add_argument("--stack_dim", type=int, default=0,
                   help="OutputQueryDecoder 的 self.stack 的 d_model: 0=与模型 dim "
                        "相同（默认, 不加投影, 与历史实现逐位一致）; >0 且 ≠dim 时"
                        "在 self.stack 前后各加一层 Linear 投影(dim→stack_dim→dim), "
                        "FFN 宽度随 stack_dim 等比缩放(mlp_ratio 语义不变)。"
                        "nhead(--heads) 须整除 stack_dim")
    p.add_argument("--decoder_dropout", type=float, default=0.0,
                   help="self.stack(nn.TransformerDecoderLayer) 的 dropout, 默认 0.0")
    p.add_argument("--warm_steps", type=int, default=0,
                   help="A 相（单步全读热启动）的优化步数, 0=关（历史口径）。前 "
                        "warm_steps 个优化步把 decoder.steps 设成 [K]（**只跑 1 步、"
                        "一次读完全部 z_s + z_cls**）, 到点自动切回真采样步集跑多步"
                        "循环。动机（实测）: 多步循环的早期步只读到 2k+1 个 token, "
                        "其条件最优解接近'预测均值'; 从零直接训多步会停在糊区, 而单步"
                        "全读没有这个内部冲突、梯度明确指向锐。224² 实测 A 相要 "
                        "1500~2000 步才逃出平凡解（A=1000→3000 使 16 步曲线从平线变"
                        "单调降, 最优 PSNR +4.43 dB）, 见 "
                        "doc/2026-09-23/REPORT_224_A3000_verdict.md")
    # ── 模型（解码器 = 顺序循环, 2026-09-15 起唯一路径）──
    # 并行路径的 --recurrent* / --query_mask_mode 开关已随之删除。
    # 损失口径的唯一性: 每步 Y 各自直接预测像素（无累加）⇒ 无 --loss_* 开关
    # （旧 --loss_mode / --loss_decouple 已删除, 传进模型直接 TypeError）。
    p.add_argument("--slice_start", type=int, default=None,
                   help="可选挑选分块起点索引(如 4 ⇔ 计划[4:9]); 默认 None = 全部分块")
    p.add_argument("--slice_end", type=int, default=None,
                   help="可选挑选分块终点索引(如 9 ⇔ 计划[4:9]); 默认 None = 全部分块")
    # ── 模型（OutputQueryDecoder 采样计划）──
    p.add_argument("--decoder_steps", default=None,
                   help="解码器采样时刻列表(逗号分隔), 默认 square_block_starts(N) "
                        "(分块起点=平方数) 再按 slice 切片; K 自动由最终步集推导")
    # ── 模型（读窗口计划: square=平方块历史口径 / fixed=自然数步值+固定宽度块）──
    p.add_argument("--step_plan", default="square", choices=["square", "fixed"],
                   help="解码器读窗口计划: square=步值 k²（历史唯一口径, 默认）; "
                        "fixed=步值自然数 1..⌈N/block⌉、每步固定读 block 个 z_s "
                        "（2026-09-23 res-sweep 的 fixw4 口径, K=N; 见 "
                        "doc/2026-09-23/REPORT_fixw4_plan.md）")
    p.add_argument("--block", type=int, default=0,
                   help="--step_plan fixed 时每步读的 z_s 个数（如 4）; square 忽略")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Eval 指标 — Trainer 的 prediction_step 对无 labels 模型直接 forward,
# 把输出 dict 按插入序转成值的元组（多 batch 后是拼接数组的元组; 4.x 与
# 5.x 行为一致）。注意: HF 会把 "loss" 从 logits 里剔除（只剩 recon /
# F_hat）, 所以本函数实际只统计得到 recon; eval_loss 由 Trainer 自己的
# compute_loss 路径报告（见下面的 SRPhase1V2Trainer）, 不在这里。
# ═══════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, dict):
        items = preds
    elif isinstance(preds, (tuple, list)):
        # prediction_step 无 labels 路径: logits = 除 loss/ignore_keys 外所有
        # 输出（dict 插入序）= (recon, F_hat)。
        items = dict(zip(("recon", "F_hat"), preds))
    else:
        items = {}
    metrics = {}
    # ("loss", "recon") 里 "loss" 因上面所述恒为 None（保留只是防御性写法,
    # 若将来 HF 不再剔除 loss 键也能取到）。
    for k in ("loss", "recon"):
        v = items.get(k)
        if v is not None:
            v = np.asarray(v)
            if v.size > 0:
                metrics[k] = float(v.mean())
    return metrics


# ═══════════════════════════════════════════════════════════════
# Trainer 子类（官方扩展点, 不算造轮子）: 本模型无 labels、无 config,
# Trainer 默认 can_return_loss=False → eval 时 loss_without_labels=False
# → eval_loss 为 None。置 True 让 eval 也走 compute_loss 路径, 正常报告
# eval_loss（模型 forward 返回 dict 含 "loss", compute_loss 原生支持）。
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True   # 无 labels 模型也允许 eval 算 loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """eval 时忽略巨型逐采样步输出（Y_pix B×25×576×588 累积即 OOM）。

        渐进曲线由 infer_v2_test.py 单独测（fp32, 逐步度量）; 训练/eval 只
        需要 loss/recon/F_hat —— ignore Y_pix/target_pix 后每个预测只有
        recon+F_hat(~22MB/batch), 全量 3004 张累积 ≈4GB, 无 OOM 风险。
        """
        if ignore_keys is None:
            ignore_keys = ["Y_pix", "target_pix"]
        return super().prediction_step(model, inputs, prediction_loss_only,
                                       ignore_keys=ignore_keys)


class WarmStartSwitchCallback(TrainerCallback):
    """A 相（单步 + 全读热启动）→ B 相（真采样步集）的自动切换（--warm_steps）。

    用户口径 2026-09-23: 多步循环的早期步只读 2k+1 个 token, 条件最优解接近
    "预测均值"; 从零直接训多步会停在该平凡解（曲线平）。A 相是**单步、一次读完全部
    z_s + z_cls**, 信息全给、不能靠输出常量交差, 梯度明确指向"锐"。224²/旧 224 臂
    实测: A=1000（未逃逸）时 B 相 5000 步只降 3.3 L1; A=3000（已逃逸）时降 19.7 L1,
    曲线由平变单调降。见 doc/2026-09-23/REPORT_224_A3000_verdict.md。

    实现: 只改 `decoder.steps`（前 warm_steps 个优化步 = [K] 单步全读）, 到点切回。
    持有点是**未包装的裸模型**（DDP 下 ddp.module 即该对象）⇒ 多卡同样生效。
    """

    def __init__(self, raw_model, full_steps, warm_steps, single_steps):
        super().__init__()
        self.raw = raw_model
        self.full_steps = list(full_steps)
        self.warm_steps = int(warm_steps)
        self.single_steps = list(single_steps)
        self.switched = False

    def on_step_begin(self, args, state, control, **kwargs):
        if self.switched or state.global_step < self.warm_steps:
            return
        self.raw.decoder.steps = list(self.full_steps)
        self.switched = True
        print(f"[warm] A 相结束（{self.warm_steps} 步单步全读）→ B 相切回 "
              f"{len(self.full_steps)} 步真轨迹 {self.full_steps}", flush=True)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    W, H = (int(v) for v in args.model_input.lower().split("x"))
    CW, CH = (int(v) for v in args.canvas.lower().split("x"))
    assert W % 14 == 0 and H % 14 == 0, "DINOv2-large patch=14, 输入须为 14 的倍数"
    num_patches = (W // 14) * (H // 14)

    steps = None
    if args.decoder_steps:
        steps = [int(s) for s in args.decoder_steps.split(",") if s.strip()]
        # 采样时刻 t ∈ [0, N]（t ≤ K ≤ N 由模型构造时最终校验——显式 K 过小
        # 会给出清晰报错; 这里只做快速失败, 免得到 DINO 加载完才报）
        assert steps and all(0 <= s <= num_patches for s in steps), \
            f"decoder_steps 越界: {steps} (N={num_patches})"

    # fixed 计划: 步集 = 自然数 1..⌈N/block⌉（K=N, 见 model_v2.fixed_block_starts）,
    # 再按 slice 切片; square 计划仍交给模型内部 select_steps（square_block_starts
    # + slice）。注意 select_steps 只认 square 基表, 故 fixed 必须在此显式算好传入。
    if steps is None and args.step_plan == "fixed":
        assert args.block >= 1, "--step_plan fixed 必须给 --block ≥1"
        base = fixed_block_starts(num_patches, args.block)
        lo = 0 if args.slice_start is None else int(args.slice_start)
        hi = len(base) if args.slice_end is None else int(args.slice_end)
        assert 0 <= lo < hi <= len(base), \
            f"slice 越界: [{lo}:{hi}] of {len(base)} 个 fixed 步"
        steps = base[lo:hi]

    # ── 数据（纯重建模式, tokenizer=None; data_v2.py 已提供）──
    train_files = sorted(glob.glob(os.path.join(args.data_dir, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert train_files, f"无 train-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(CW, CH), angle_step=args.angle_step)
    train_ds = ParquetImageDataset(train_files, limit=args.limit)
    eval_ds = None
    if test_files:
        eval_ds = ParquetImageDataset(test_files, limit=args.eval_limit)
    else:
        print("[warn] 无 test-*.parquet, 跳过 eval")

    # ── 模型: DINOv2-large 不冻结 + register specials(K) → OutputQueryDecoder ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    # 权重带 mask_token(use_mask_token=True) 但本任务不传 bool_masked_pos,
    # 该参数从不参与前向 → DDP 报"未用参数"。移除并关掉 flag。
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token

    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size,
                       heads=args.heads, mlp_ratio=args.mlp_ratio,
                       decoder_steps=steps,
                       decoder_depth=args.decoder_depth,
                       skip_steps=args.slice_start,
                       max_steps=args.slice_end,
                       num_specials=(args.num_specials or None),
                       stack_dim=args.stack_dim,
                       decoder_dropout=args.decoder_dropout,
                       step_plan=args.step_plan, block=args.block)

    K = model.num_specials
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _steps_brief = (str(model.decoder.steps) if len(model.decoder.steps) <= 12
                    else f"{model.decoder.steps[:4]}...+{model.decoder.steps[-2:]}"
                         f" (共 {len(model.decoder.steps)})")
    print(f"[model] 可训练参数 {n_train / 1e6:.1f}M (含 DINOv2-large, 不冻结); "
          f"输入 {W}x{H}, patches={num_patches}, specials={K} "
          f"(序列 {1 + K + num_patches} token), decoder 采样 "
          f"{len(model.decoder.steps)} 步 {_steps_brief}")
    if args.num_specials:
        print(f"[model] num_specials 显式 = {K}（须 ≥ max 采样步; "
              f"复现旧 checkpoint 用, 如 K=N={num_patches}）")
    else:
        print(f"[model] num_specials 自动推导 = {K} "
              f"({'fixed 计划 ⇒ K=N' if args.step_plan == 'fixed' else 'K = derive_num_specials(N, 最终采样步集)'}, 无花瓶 register)")
    print(f"[model] 读窗口计划: step_plan={args.step_plan}"
          + (f" block={args.block}" if args.step_plan == "fixed" else "")
          + f" | slice_start={args.slice_start} slice_end={args.slice_end}"
          f" | |T|={len(model.decoder.steps)}")
    print(f"[model] self.stack: d_model={model.decoder.stack_dim} "
          f"(模型 dim={dino.config.hidden_size}), heads={args.heads}, "
          f"depth={args.decoder_depth}, dropout={args.decoder_dropout}"
          f"{'  ← 加宽: 前后 Linear 投影' if model.decoder.stack_dim != dino.config.hidden_size else '  (未加宽)'}")
    _carry = "BPTT（不 detach, 循环 carry 反传; 2026-09-23 起唯一路径）"
    print(f"[model] 解码器: 顺序循环（唯一路径）| 每步只读自己那块 z_s 切片, "
          f"上一步输出以 **{_carry}** 喂回当下一步查询 ⇒ wall-clock "
          f"长于旧并行路径（以时间换跨步信息流）")
    print("[model] 损失: 直接预测 mean_t L1(PixelHead(Y_t), target) "
          "（每个采样步各自直接预测整图, 无累加/集成; 各步平权深监督）"
          " | F_hat = 最后一步的直接预测, recon = 其 L1 (= loss 最后一项)")
    print(f"[train] 精度: {'fp16 混合精度（autocast+GradScaler, 权重 fp32）' if args.fp16 else '纯 fp32（不套 autocast）'}"
          f" | 优化器: {args.optim or '库默认（不传 optim）'}")

    # ── A 相（单步 + 全读热启动, --warm_steps）: 见该 arg 的 help ──
    full_steps = list(model.decoder.steps)
    single_steps = [model.num_specials]     # 首步 lo=0 ⇒ 一次读完 z_s 全部(+z_cls)
    if args.warm_steps < 0:
        raise SystemExit("--warm_steps 不能为负")
    if args.warm_steps > 0:
        model.decoder.steps = list(single_steps)
        print(f"[warm] A 相开: 前 {args.warm_steps} 个优化步 decoder.steps="
              f"{single_steps}（单步 + 全读）; 到点切回 {len(full_steps)} 步真轨迹")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── Trainer: 训练循环/梯度累积/调度/checkpoint/分布式全部交给它 ──
    n_proc = int(os.environ.get("WORLD_SIZE", "1"))      # accelerate/DDP 进程数
    micro_per_epoch = max(1, len(train_ds) // (args.batch_size * n_proc))
    # Trainer 的**优化步** = 微批步 // grad_accum（warmup/cosine 调度器都按优化步数走,
    # 见 TrainingArguments.warmup_steps 语义）; grad_accum=1 时与历史逐位一致。
    if args.max_steps > 0:
        total_steps = int(args.max_steps)                      # 冒烟: 直接给优化步
    else:
        total_steps = max(1, micro_per_epoch * args.epochs // args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,   # 0 = epochs 决定
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # eval 预测(含 Y_pix B×25×576×588)分批移到 CPU, 避免全量累积 GPU OOM
        eval_accumulation_steps=2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=args.grad_clip,
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        logging_steps=args.log_every,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_every,
        save_strategy="no" if args.smoke else "steps",   # 冒烟不存 checkpoint
        save_steps=args.save_every,
        seed=args.seed,
        report_to=[],                  # 不上报 wandb / tensorboard
        remove_unused_columns=False,   # 原始 dict 样本交给 V2Collator
        ddp_find_unused_parameters=False,
        # fp16 混合精度（默认关 = 历史口径「全 fp32, 不套 autocast」）:
        # fp16=True ⇒ accelerate mixed_precision="fp16" ⇒ autocast + GradScaler,
        # 权重仍 fp32（final_model.pt 不变）; eval 默认 fp16_full_eval=False 走 fp32。
        fp16=args.fp16,
    )
    # --optim 留空 ⇒ **不传**（保持 TrainingArguments 的库默认, 本环境 =
    # adamw_torch_fused = 历史各臂实际用的那个）; 显式给了才覆盖。
    if args.optim:
        training_args.optim = args.optim

    trainer = SRPhase1V2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=coll,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )
    if args.warm_steps > 0:
        trainer.add_callback(WarmStartSwitchCallback(
            model, full_steps, args.warm_steps, single_steps))

    n_proc = trainer.accelerator.num_processes
    trainer.accelerator.print(
        f"[train] {len(train_ds)} 样本 | 每卡 bs={args.batch_size} "
        f"x {n_proc} 卡 | grad_accum={args.grad_accum} "
        f"⇒ 全局 batch {args.batch_size * n_proc * args.grad_accum} | "
        f"~{micro_per_epoch} 微批/epoch（优化步/epoch ≈ "
        f"{max(1, micro_per_epoch // args.grad_accum)}）"
        f" x {args.epochs} epoch = {total_steps} 优化步 "
        f"| warmup {warmup_steps} 步 "
        f"| 实际优化器={training_args.optim} fp16={training_args.fp16} "
        f"(真实优化步按 Trainer 口径, 与 grad_accum 有关)")

    if args.resume:
        trainer.accelerator.print(f"[resume] 从 {args.resume} 恢复 (Trainer checkpoint)")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── 收尾: 仅主进程导出推理权重（全 fp32 训练, 权重天然 fp32）──
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        raw = trainer.accelerator.unwrap_model(trainer.model)   # DDP 包装下取回裸模型
        sd = raw.state_dict()
        final = os.path.join(args.output_dir, "final_model.pt")
        torch.save(sd, final)
        info = {"input_size": [W, H], "canvas": [CW, CH],
                "num_patches": num_patches,
                "num_specials": model.num_specials,
                "dim": dino.config.hidden_size,
                "heads": args.heads, "mlp_ratio": args.mlp_ratio,
                "decoder_depth": args.decoder_depth,
                "stack_dim": int(raw.decoder.stack_dim),
                "decoder_dropout": args.decoder_dropout,
                "slice_start": args.slice_start, "slice_end": args.slice_end,
                "step_plan": args.step_plan, "block": args.block,
                "decoder_steps": raw.decoder.steps,
                "warm_steps": int(args.warm_steps),
                "loss": "mean_t L1(PixelHead(Y_t), target)（直接预测, 无累加）",
                "target": "pixel_values (归一化空间, PixelHead 解码)",
                "dino_dir": args.dino_dir, "dtype": "fp32"}
        with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"[final] {final} 已保存 (fp32, 含 DINO 权重)")


if __name__ == "__main__":
    main()
