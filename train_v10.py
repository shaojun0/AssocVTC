"""v10 训练脚本：冻结图像分词器（E + R），只训「中间态 + 整片段循环读出」。

训练器口径与 `train_v2.py` 一致：**不手写训练循环**。
训练循环 / 梯度累积 / cosine 调度 / checkpoint / 分布式 / DeepSpeed
全部交给 `transformers.Trainer + TrainingArguments`；本文件只保留：
  * `ClipDataset` + 一个把 tensor 包成 `{"x": ...}` 的 collator；
  * `compute_metrics`（latent L1 / PSNR）；
  * `V10Trainer`（官方扩展点）：无 labels 模型打开 `can_return_loss`，
    eval 时忽略巨型 pred/target 输出，只保留标量指标。

单卡::

    python train_v10.py \
        --data_root /root/autodl-tmp/video_recon/data3 \
        --tok_ckpt  /root/autodl-tmp/tok448/ckpt_best.pt \
        --out       /root/autodl-tmp/v10_wholeclip \
        --batch 2 --accum_steps 2 --lr 2e-4 --max_steps 20000

DeepSpeed（2 卡，PyTorch 启动器）::

    torchrun --nproc_per_node=2 train_v10.py \
        --data_root /root/autodl-tmp/video_recon/data3 \
        --tok_ckpt  /root/autodl-tmp/tok448/ckpt_best.pt \
        --out       /root/autodl-tmp/v10_wholeclip \
        --deepspeed ds_zero2_v10.json \
        --batch 2 --accum_steps 2 --lr 2e-4 --max_steps 20000

也可以使用 `accelerate launch --multi_gpu --num_processes 2 ...`。
DeepSpeed 的 stage / bf16 / offload 全部在 `ds_zero2_v10.json` 里改。

注意：DeepSpeed bf16 会把**冻结分词器也转成 bf16**（DeepSpeed 的模型级 cast）。
若要和 fp32 分词器基线做严格 PSNR 对照，把配置里的 `bf16.enabled` 设为 `false`
并加 `--amp 0`；普通单卡路径的 `--amp 1` 是 torch.autocast，权重仍是 fp32。

关键口径（详见 doc/2026-11-16/DESIGN_v10_video_whole_clip.md）
------------------------------------------------------------
* 冻结分词器：E 与 R 都是同一个 `SRPhase1LinAttn` ckpt（`--tok_ckpt`）；
  Trainer 的默认优化器只会拿到 `requires_grad=True` 的 middle / readout，
  冻结分词器不会进 optimizer。
* 宽度：`--chunk_tokens 4 × --rate_dim 448 = 1792` 是唯一宽度，`--mid_dim /`
  `--dec_dim` 留 0 自动取它；填别的值会被 assert 拒绝（那是纯填充冗余）。
* 率 / 目标：率 = 每帧前 `--rate_tokens` 个 special token（默认 16 → 4×1792），
  目标 = 全部 `--target_tokens` + z_cls（默认 32+1 → 8×1792 + 1 行）。
  率 < 目标是故意的；`--target_tokens` 必须等于分词器 num_specials，
  否则冻结 R 的 `render` 会 assert 失败。
* 损失：逐步平权 latent L1（`--step_weights mean|lin|last`），深监督全部 30 步。
* 评测：`--psnr_eval 1` 时，`V10Trainer.evaluate` 临时给裸模型设
  `eval_pixel=True`，`VideoV10.forward` 就额外返回 `psnr` / `psnr_tok_full`；
  `prediction_step` 会丢掉 `pix/pred/target/z`，避免 eval 把整段特征堆到 CPU。
* 断点续训：`--resume <Trainer checkpoint 目录>`；不要传旧的单个 `.pt`。
* MUSA：仓库里有 `musa.py` 时自动 `musa.bootstrap()`，在 Trainer 初始化
  DeepSpeed 前修正 MCCL/fp64 探测；CUDA/CPU 环境静默跳过。

自检：``python train_v10.py --smoke``（小数据集/小步数，本地 CPU 也能跑）
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from data_v10 import ClipDataset
from model_v10 import V10Config, VideoV10


def bootstrap_musa() -> None:
    """有 MUSA 适配层时先打补丁；普通 CUDA/CPU 环境静默跳过。"""
    try:
        from musa import bootstrap
    except ImportError:
        return
    bootstrap()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v10 全片段循环读出训练（HF Trainer）")
    # 数据
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp/video_recon/data3")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--img", type=int, default=448)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条（调试）")
    p.add_argument("--num_workers", type=int, default=8)
    # 分词器
    p.add_argument("--tok_ckpt", type=str, required=True)
    # 结构
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--rate_tokens", type=int, default=16)
    p.add_argument("--rate_dim", type=int, default=448)
    p.add_argument("--target_tokens", type=int, default=32)
    p.add_argument("--chunk_tokens", type=int, default=4,
                   help="一个 1792 向量拼几个 token（宽度 = chunk_tokens × rate_dim）")
    p.add_argument("--mid_dim", type=int, default=0,
                   help="0 = 自动取 chunk_tokens×rate_dim；填别的宽度会被 assert 拒绝")
    p.add_argument("--mid_depth", type=int, default=6)
    p.add_argument("--mid_heads", type=int, default=16)
    p.add_argument("--mid_ratio", type=float, default=4.0)
    p.add_argument("--dec_dim", type=int, default=0,
                   help="0 = 自动取 chunk_tokens×rate_dim")
    p.add_argument("--dec_depth", type=int, default=4)
    p.add_argument("--dec_heads", type=int, default=16)
    p.add_argument("--dec_ratio", type=float, default=2.0)
    p.add_argument("--step_gate", type=int, default=0,
                   help="1 = 零初始化 γ 门控；0（默认）= w≡1，carry 等权相加（同 v2）")
    p.add_argument("--step_weights", type=str, default="mean")
    # 优化（全部转发给 TrainingArguments）
    p.add_argument("--batch", type=int, default=2, help="每卡 micro-batch")
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--amp", type=int, default=1, help="bf16（CPU 自动关闭）")
    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default="",
                   help="DeepSpeed JSON；留空=普通单卡/accelerate 路径")
    p.add_argument("--local_rank", type=int,
                   default=int(os.environ.get("LOCAL_RANK", "-1")))
    # 日志/评测/保存
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=2,
                   help="验证样本数（per_device_eval_batch_size=1）")
    p.add_argument("--psnr_eval", type=int, default=1)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--resume", type=str, default="",
                   help="Trainer checkpoint 目录（如 out/checkpoint-2000）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true",
                   help="用随机张量代替真实片段（验通路用，不落盘数据）")
    return p.parse_args()


class SyntheticClips(torch.utils.data.Dataset):
    """随机片段：只用来验证训练通路/显存（真实训练必须用 --data_root）。"""

    def __init__(self, n: int, frames: int, img: int):
        self.n, self.frames, self.img = int(n), int(frames), int(img)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(1000 + int(i))
        return torch.rand(self.frames, 3, self.img, self.img, generator=g)


def build_cfg(a: argparse.Namespace) -> V10Config:
    return V10Config(
        tok_ckpt=a.tok_ckpt, frames=a.frames, group=a.group,
        rate_tokens=a.rate_tokens, rate_dim=a.rate_dim,
        target_tokens=a.target_tokens, chunk_tokens=a.chunk_tokens,
        mid_dim=a.mid_dim, mid_depth=a.mid_depth, mid_heads=a.mid_heads,
        mid_ratio=a.mid_ratio, dec_dim=a.dec_dim, dec_depth=a.dec_depth,
        dec_heads=a.dec_heads, dec_ratio=a.dec_ratio,
        step_gate=bool(a.step_gate), step_weights=a.step_weights)


def collate_clips(batch):
    """Dataset 返回 (F,3,H,W)；Trainer 的模型签名需要 `x=`。"""
    return {"x": torch.stack(batch, dim=0)}


def compute_metrics(eval_pred):
    """eval 输出只剩标量：l1（以及可选的 psnr / psnr_tok_full）。

    Trainer 会把 dict 输出按插入序转成 tuple；`V10Trainer.prediction_step`
    已把 pred/target/z/pix/step_l1 丢掉，所以这里不需要处理大张量。
    """
    preds = eval_pred.predictions
    names = ("l1", "psnr", "psnr_tok_full")
    if isinstance(preds, dict):
        items = preds
    elif isinstance(preds, (tuple, list)):
        items = dict(zip(names, preds))
    else:
        items = {"l1": preds}

    out = {}
    for key, name in (("l1", "latent_l1"),
                      ("psnr", "psnr"),
                      ("psnr_tok_full", "psnr_tok_full")):
        value = items.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size:
            out[name] = float(arr.mean())
    return out


def _is_musa() -> bool:
    return hasattr(torch, "musa") and bool(torch.musa.is_available())


class V10Trainer(Trainer):
    """官方扩展点：无 labels 的 eval loss、丢掉巨型输出、MUSA RNG 适配。"""

    def __init__(self, *args, psnr_eval: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True
        self.psnr_eval = bool(psnr_eval)

    def evaluate(self, *args, **kwargs):
        raw = self.accelerator.unwrap_model(self.model)
        raw.eval_pixel = self.psnr_eval
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            raw.eval_pixel = False

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        # forward 还会返回 pred/target/z/pix；eval 只需要几个标量。
        ignore_keys = list(ignore_keys or []) + [
            "step_l1", "pred", "z", "target", "latents", "pix"]
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    # ── MUSA RNG 兼容 ──────────────────────────────────────────
    # transformers 5.22 的 _save_rng_state 先 if torch.cuda.is_available()
    # 调 torch.cuda.random，再处理 MUSA；但 torch_musa 没有 cuda.random。
    # 正常 CUDA 继续走父类，MUSA 用 torch.musa.get/set_rng_state_all。
    def _save_rng_state(self, output_dir: str) -> None:
        if not _is_musa():
            return super()._save_rng_state(output_dir)
        from transformers.training_args import ParallelMode
        states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            states["musa"] = torch.musa.get_rng_state_all()
        else:
            states["musa"] = torch.musa.get_rng_state()
        os.makedirs(output_dir, exist_ok=True)
        name = ("rng_state.pth" if self.args.world_size <= 1
                else f"rng_state_{self.args.process_index}.pth")
        torch.save(states, os.path.join(output_dir, name))

    def _load_rng_state(self, checkpoint: str | None) -> None:
        if not _is_musa():
            return super()._load_rng_state(checkpoint)
        if checkpoint is None:
            return
        name = ("rng_state.pth" if self.args.world_size <= 1
                else f"rng_state_{self.args.process_index}.pth")
        path = os.path.join(checkpoint, name)
        if not os.path.isfile(path):
            return
        states = torch.load(path, map_location="cpu", weights_only=False)
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.random.set_rng_state(states["cpu"])
        if "musa" in states:
            if self.args.world_size > 1:
                torch.musa.set_rng_state_all(states["musa"])
            else:
                torch.musa.set_rng_state(states["musa"])


def main() -> None:
    bootstrap_musa()
    a = parse_args()
    set_seed(a.seed)

    if a.deepspeed and not os.path.isfile(a.deepspeed):
        raise FileNotFoundError(f"DeepSpeed 配置不存在: {a.deepspeed}")

    if a.smoke:
        a.max_steps = min(a.max_steps, 4)
        a.batch = min(a.batch, 1)
        a.num_workers = 0
        a.val_every = 2
        a.val_batches = 1
        a.psnr_eval = 0
        a.save_total_limit = 1

    os.makedirs(a.out, exist_ok=True)
    cfg = build_cfg(a)

    # ── 数据
    if a.synthetic:
        tr = SyntheticClips(32, a.frames, a.img)
        va = SyntheticClips(4, a.frames, a.img)
    else:
        tr = ClipDataset(a.data_root, "train", frames=a.frames, img=a.img,
                         limit=a.limit)
        va = ClipDataset(a.data_root, "val", frames=a.frames, img=a.img,
                         limit=max(1, a.val_batches))
    eval_enabled = a.val_batches > 0

    # ── 模型（分词器在 FrozenTokenizer 里已冻结）
    model = VideoV10(cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    warmup_steps = int(a.warmup_frac * a.max_steps)

    eval_steps = max(1, a.val_every)
    save_steps = max(1, a.save_every)
    if eval_enabled and save_steps % eval_steps:
        # load_best_model_at_end 要求 save_steps 是 eval_steps 的整数倍。
        save_steps = eval_steps * ((save_steps + eval_steps - 1) // eval_steps)
    load_best = eval_enabled and not a.smoke

    training_args = TrainingArguments(
        output_dir=a.out,
        max_steps=a.max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.accum_steps,
        # MUSA 的 torch.optim.AdamW(fused=True) 不可用；用标准非 fused 实现，
        # 也与改 Trainer 前手写 AdamW 的口径一致。
        optim="adamw_torch",
        learning_rate=a.lr,
        weight_decay=a.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=a.grad_clip,
        logging_steps=a.log_every,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=eval_steps,
        save_strategy="no" if a.smoke else "steps",
        save_steps=save_steps,
        save_total_limit=a.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_latent_l1" if load_best else None,
        greater_is_better=False,
        dataloader_num_workers=a.num_workers,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        bf16=bool(a.amp) and torch.cuda.is_available(),
        fp16=False,
        seed=a.seed,
        data_seed=a.seed,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        deepspeed=a.deepspeed or None,
        save_only_model=False,
    )

    trainer = V10Trainer(
        model=model,
        args=training_args,
        train_dataset=tr,
        eval_dataset=va if eval_enabled else None,
        data_collator=collate_clips,
        compute_metrics=compute_metrics if eval_enabled else None,
        psnr_eval=bool(a.psnr_eval),
    )

    trainer.accelerator.print(
        f"[v10] trainable={n_train/1e6:.2f}M "
        f"{ {k: round(v/1e6, 2) for k, v in model.trainable_param_groups().items()} } "
        f"values/frame={cfg.values_per_frame()} "
        f"batch={a.batch}×{a.accum_steps}×{trainer.accelerator.num_processes} "
        f"steps={a.max_steps} eval={eval_enabled} psnr={bool(a.psnr_eval)} "
        f"deepspeed={a.deepspeed or 'off'}")
    if trainer.accelerator.is_main_process:
        with open(os.path.join(a.out, "run_args.json"), "w",
                  encoding="utf-8") as f:
            json.dump(dict(vars(a), cfg=cfg.to_dict()), f,
                      ensure_ascii=False, indent=2)

    if a.resume:
        trainer.accelerator.print(f"[resume] {a.resume}")
    train_result = trainer.train(resume_from_checkpoint=a.resume or None)

    # ── 收尾：Trainer 负责 gather / 主进程落盘
    # 模型现在是 PreTrainedModel，trainer.save_model 走 save_pretrained：
    # final/ 里有 model.safetensors + config.json（含 auto_map）+ 模型代码，
    # 可以直接 AutoModel.from_pretrained(final_dir, trust_remote_code=True)。
    trainer.accelerator.wait_for_everyone()
    final_dir = os.path.join(a.out, "final")
    trainer.save_model(final_dir)
    if trainer.accelerator.is_main_process:
        trainer.save_state()
        trainer.save_metrics("train", train_result.metrics)
        print(f"[done] best={trainer.state.best_metric} out={final_dir}", flush=True)
        print(f"[load] AutoModel.from_pretrained({final_dir!r}, trust_remote_code=True)",
              flush=True)


if __name__ == "__main__":
    main()
