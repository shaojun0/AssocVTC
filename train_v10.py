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
import contextlib
import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
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
    p.add_argument("--latent_cache", type=str, default="",
                   help="build_latent_cache.py 产出的目录；给了就直接读 (z_cls,z_s)，"
                        "训练时不再解码 mp4 / 跑冻结编码器（eval 仍用原始 mp4 算 PSNR）")
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
    p.add_argument("--step_gate", type=int, default=1)
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


class CachedLatents(torch.utils.data.Dataset):
    """读 `build_latent_cache.py` 的 shard（冻结编码器已离线算好）。

    布局：全局样本 i 落在 shard `i % world` 的第 `i // world` 行，行内
    (frames, 1+K, dim) float16（= z_cls 一行 + K 个 special）。缓存的这个量
    就是分词器能给出的全部（rate 与 target 都只用它），训练时不必再解 mp4、
    也不必再跑 487M 冻结编码器 —— 单样本省 ~30% 时间。
    """

    def __init__(self, cache_dir: str, limit: int = 0):
        self.dir = str(cache_dir)
        with open(os.path.join(self.dir, "meta.json"), "r") as f:
            meta = json.load(f)
        self.meta = meta
        self.n = int(meta["n"])
        self.world = int(meta["world"])
        self.frames = int(meta["frames"])
        self.rows = int(meta["rows"])
        self.dim = int(meta["dim"])
        if limit and int(limit) > 0:
            self.n = min(self.n, int(limit))
        self.mm = []
        for r in range(self.world):
            p = os.path.join(self.dir, f"rank{r}.npy")
            self.mm.append(np.load(p, mmap_mode="r"))
        for r in range(self.world):
            need = max(0, (self.n - r + self.world - 1) // self.world)
            assert self.mm[r].shape[0] >= need, (r, self.mm[r].shape, need)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        i = int(i)
        r = i % self.world
        row = i // self.world
        arr = self.mm[r][row]                              # (F,1+K,dim) fp16
        t = torch.from_numpy(np.array(arr)).float()        # 拷贝出 memmap
        return {"z_cls": t[:, :1], "z_s": t[:, 1:]}


def collate_cached(batch):
    return {"z_cls": torch.stack([b["z_cls"] for b in batch], dim=0),
            "z_s": torch.stack([b["z_s"] for b in batch], dim=0)}


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


def set_sdpa_backend() -> str:
    """选 SDPA 后端；MUSA 默认 math（flash 反向在 head_dim%8≠0 时数值错误）。

    torch_musa 的 muDNN flash 反向对 head_dim 不是 8 的倍数时算出错梯度
    （本分词器 enc 1248/16=78、dec 1440/16=90 都命中，见
    doc/2026-11-17/REPORT_musa_sdpa_bwd.md）。冻结编码器在这里也要前向，
    所以整个进程统一关掉 flash。CUDA 上不设时保持默认（flash 更快）。
    """
    want = os.environ.get("SDPA_BACKEND", "").strip().lower()
    if not want and _is_musa():
        want = "math"
    if not want:
        return ""
    for name, flag in (("flash", torch.backends.cuda.enable_flash_sdp),
                       ("mem_efficient", torch.backends.cuda.enable_mem_efficient_sdp),
                       ("math", torch.backends.cuda.enable_math_sdp)):
        try:
            flag(name == want)
        except Exception:                                  # 旧版 torch 没这些开关
            pass
    if _is_musa():
        print(f"[musa] SDPA_BACKEND={want}", flush=True)
    return want


def _patch_ddp_musa() -> None:
    """类级修掉 MUSA 上 DDP 的 c10d logger timer 崩溃（HF Trainer 内部建 DDP）。

    与 train_linattn.py 的实例级补丁同源：torch 2.11 + torch_musa 的
    `DDP._pre_forward` 在第二次 forward 调 `logger.set_runtime_stats_and_log()`
    → C++ `reducer_->timer_` 空指针 assert。这里逐行照抄 torch 2.11 的实现、
    只删掉那一行 logger 调用，其余（lazy init / bucket / 输入搬移 / cast）不变。
    """
    if not _is_musa():
        return
    try:
        import types
        from torch.distributed.algorithms.join import Join
        from torch.nn.parallel.distributed import (
            DistributedDataParallel as _DDP, _to_kwargs, _cast_forward_inputs)
    except Exception as exc:                               # torch < 2.11 直接跳过
        print(f"[musa] DDP patch 跳过: {type(exc).__name__}", flush=True)
        return
    if getattr(_DDP, "_dsh_musa_patched", False):
        return

    def _pre_forward(self, *inputs, **kwargs):
        if self._use_python_reducer:
            return inputs, kwargs
        if not self._lazy_init_ran and not torch.compiler.is_compiling():
            self._lazy_init()
        if self._delay_all_reduce_all_params:
            return inputs, kwargs
        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            # 原实现: self.logger.set_runtime_stats_and_log()  ← MUSA timer 崩
            self.reducer.prepare_for_forward()
        work = Join.notify_join_context(self)
        if work:
            self.reducer._set_forward_pass_work_handle(
                work, self._divide_by_initial_world_size)
        if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
            self._has_rebuilt_buckets = True
        if self._check_sync_bufs_pre_fwd():
            self._sync_buffers()
        if self._join_config.enable:
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)
        if self.device_ids:
            moved_inputs, moved_kwargs = _to_kwargs(
                inputs, kwargs,
                torch.device(self.device_type, self.device_ids[0]),
                self.use_side_stream_for_tensor_copies)
            args, kwargs = moved_inputs[0], moved_kwargs[0]
            if self.mixed_precision is not None:
                args, kwargs = _cast_forward_inputs(
                    self.mixed_precision.param_dtype, *args, **kwargs)
            return args, kwargs
        if self.mixed_precision is not None:
            inputs, kwargs = _cast_forward_inputs(
                self.mixed_precision.param_dtype, *inputs, **kwargs)
        return inputs, kwargs

    _DDP._pre_forward = _pre_forward
    _DDP._dsh_musa_patched = True
    print("[musa] DDP._pre_forward 已打补丁（跳过 c10d logger timer）", flush=True)


class V10EvalCallback(TrainerCallback):
    """独立评测：直接调 `VideoV10.evaluate()`（latent L1 + 像素 PSNR）。

    为什么不用 Trainer 的 eval/compute_metrics：本模型 forward 返回 dict，
    transformers 5.13 的 `prediction_step` 在无 labels 时会把 dict 折成 tuple，
    实测 `compute_metrics` 拿不到 `l1/psnr` 字段（eval 只剩 runtime 指标），
    `metric_for_best_model` 直接 KeyError。这里自己评测，指标直接 trainer.log，
    并按 PSNR 在 <out>/best_trainable.pt 存 middle/readout（小、可单独加载）。
    """

    def __init__(self, data_root: str, frames: int, img: int, n_clips: int,
                 every: int, out_dir: str, psnr: bool = True):
        self.every = max(1, int(every))
        self.n_clips = max(1, int(n_clips))
        self.out_dir = str(out_dir)
        self.psnr = bool(psnr)
        self._ds = None
        self._spec = (str(data_root), int(frames), int(img))
        self.best = None
        self.best_step = 0

    def _ensure_ds(self):
        if self._ds is None:
            from data_v10 import ClipDataset
            self._ds = ClipDataset(self._spec[0], "val", frames=self._spec[1],
                                   img=self._spec[2], limit=self.n_clips)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step % self.every:
            return
        trainer = self.trainer
        raw = trainer.accelerator.unwrap_model(trainer.model)
        was_training = raw.training
        raw.eval()
        self._ensure_ds()
        acc, n = {}, 0
        dev = trainer.accelerator.device
        amp = bool(args.bf16 or args.fp16)
        with torch.no_grad():
            for i in range(len(self._ds)):
                x = self._ds[i].unsqueeze(0).to(dev)
                ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp
                       else contextlib.nullcontext())
                with ctx:
                    out = raw.evaluate(x, pixel_psnr=self.psnr)
                for k, v in out.items():
                    acc[k] = acc.get(k, 0.0) + float(v)
                n += 1
        raw.train(was_training)
        metrics = {f"eval_{k}": v / max(n, 1) for k, v in acc.items()}
        metrics["eval_clips"] = n
        trainer.log(metrics)
        if self.psnr and "eval_psnr" in metrics and \
                (self.best is None or metrics["eval_psnr"] > self.best):
            self.best = metrics["eval_psnr"]
            self.best_step = int(state.global_step)
            os.makedirs(self.out_dir, exist_ok=True)
            torch.save(dict(
                middle=raw.middle.state_dict(), readout=raw.readout.state_dict(),
                step=self.best_step, metrics=metrics, cfg=asdict(raw.cfg),
                tok_ckpt=raw.cfg.tok_ckpt), os.path.join(self.out_dir,
                                                        "best_trainable.pt"))


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
    set_sdpa_backend()
    _patch_ddp_musa()
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
        train_collate = collate_clips
    else:
        if a.latent_cache:
            tr = CachedLatents(a.latent_cache, limit=a.limit)
            train_collate = collate_cached
        else:
            tr = ClipDataset(a.data_root, "train", frames=a.frames, img=a.img,
                             limit=a.limit)
            train_collate = collate_clips
    # 评测走 V10EvalCallback（见下方注释），不占用 Trainer 的 eval 数据通路
    eval_enabled = a.val_batches > 0

    # ── 模型（分词器在 FrozenTokenizer 里已冻结）
    model = VideoV10(cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    warmup_steps = int(a.warmup_frac * a.max_steps)

    save_steps = max(1, a.save_every)

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
        eval_strategy="no",
        save_strategy="no" if a.smoke else "steps",
        save_steps=save_steps,
        save_total_limit=a.save_total_limit,
        load_best_model_at_end=False,
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

    eval_cb = None
    if eval_enabled:
        eval_cb = V10EvalCallback(
            data_root=a.data_root, frames=a.frames, img=a.img,
            n_clips=a.val_batches, every=a.val_every, out_dir=a.out,
            psnr=bool(a.psnr_eval))
    trainer = V10Trainer(
        model=model,
        args=training_args,
        train_dataset=tr,
        eval_dataset=None,
        data_collator=train_collate,
        compute_metrics=None,
        callbacks=[eval_cb] if eval_cb is not None else None,
        psnr_eval=bool(a.psnr_eval),
    )
    if eval_cb is not None:
        eval_cb.trainer = trainer

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
            json.dump(dict(vars(a), cfg=asdict(cfg)), f,
                      ensure_ascii=False, indent=2)

    if a.resume:
        trainer.accelerator.print(f"[resume] {a.resume}")
    train_result = trainer.train(resume_from_checkpoint=a.resume or None)

    # ── 收尾：Trainer 负责 gather / 主进程落盘
    trainer.accelerator.wait_for_everyone()
    final_dir = os.path.join(a.out, "final")
    trainer.save_model(final_dir)
    if trainer.accelerator.is_main_process:
        trainer.save_state()
        trainer.save_metrics("train", train_result.metrics)
        if eval_cb is not None:
            print(f"[done] best_psnr={eval_cb.best} @step={eval_cb.best_step} "
                  f"out={final_dir}", flush=True)
        else:
            print(f"[done] out={final_dir}", flush=True)


if __name__ == "__main__":
    main()
