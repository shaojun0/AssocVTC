"""
vlm8b 训练入口 — HF Trainer + DeepSpeed ZeRO-2/3（CUDA / 国产 MUSA 同一份代码）
============================================================================

协议（设计 §5）: 3 epoch（encoder 默认不冻结；encoder 5e-6 / LLM 1e-5 / proj 1e-4），
cosine + warmup 3%。每步 B 张图 × V 行（k=1..8），行 k = `[vis[:, :k]; text]`，
loss = Σ_k w_k·CE_k。

**encoder 冻结**（`freeze.py` 实现，`MUSA_FREEZE.md` 讲机制）::

    --freeze-encoder all                # 整个 encoder 冻死（只训 proj + LLM）
    --freeze-encoder stem               # 只训 blocks + norm
    --freeze-encoder last=2             # 只训最后 2 个 block + norm
    --freeze-encoder blocks=0-8         # 冻前 9 个 block
    --freeze-encoder-epochs 1           # E1 冻、E2 起解冻（仅 DeepSpeed 合法）

静态冻结（不配 `--freeze-encoder-epochs`）在 Trainer 之前生效：可训练参数才进
optimizer，整组冻结不会崩 DeepSpeed（空组被丢弃）。
按 epoch 冻结/解冻必须在 DeepSpeed engine 初始化**之后**切换 `requires_grad`
（`EncoderFreezeCallback.on_train_begin`），否则 stage 2 初始化 `IndexError`、
stage 3 解冻后静默不更新——`VLM8BudgetTrainer.train()` 有 fail-fast guard。

**国产适配**：`musa.bootstrap()` 在 main 第一行，负责设备探测、MCCL 后端、
DeepSpeed 的 fp64 范数补丁（见 `musa.py` 顶部三条硬事实）；代码本身不出现
`"cuda"` 硬编码（除 torchrun 启动命令）。

**Trainer 内不做 eval**：`forward` 不返回 logits，`prediction_step` 无法工作 ⇒
`eval_strategy="no"` 且不传 eval_dataset；评测统一走 `eval.py`。

启动（remote, 2 卡）::

    torchrun --nproc_per_node=2 train.py \
        --output-dir /root/autodl-tmp/vlm8b/output/run1 \
        --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl

    # 冻结 encoder 的实验臂（E1 冻、E2 起解冻）
    torchrun --nproc_per_node=2 train.py --freeze-encoder all --freeze-encoder-epochs 1 ...
"""
import argparse
import glob
import json
import math
import os

from torch.optim import AdamW
from transformers import (AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

import musa
from data import EightBudgetCollator, SelfDistillSFTDataset, build_image_transform
from freeze import (FreezeReport, build_param_groups as _build_param_groups,
                    freeze_encoder, unfreeze_all)
from model import VLM8Budget, VLM8BudgetConfig

# ═══════════════════════════════════════════════════════════════
# 默认路径 / 超参（remote 实测位置；全部可用 CLI 覆盖）
# ═══════════════════════════════════════════════════════════════

DEFAULT_ENCODER_CKPT = "/root/autodl-tmp/exp448/B/ckpt_best.pt"
DEFAULT_QWEN_DIR = "/root/qwen_models/Qwen3.5-4B"
DEFAULT_IMAGE_ROOT = "/root/autodl-tmp/datasets/imagenet100"
DEFAULT_DATA_DIR = "/root/autodl-tmp/exp_qwen/in100_selfdistill/sft"
DEFAULT_TRAIN_JSONL = os.path.join(DEFAULT_DATA_DIR, "train_nothink_all.jsonl")
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/vlm8b/output/run1"
# ZeRO-2 是默认：ZeRO-3 在 torch 2.12 + deepspeed 0.19.9 上第 1 步 backward 就
# collective 死锁（同一 seq 号两侧 op 类型不同）；ZeRO-2 显存也够（~35GB/卡）。
# 想试 ZeRO-3 传 `--deepspeed ds_zero3.json`。
DEFAULT_DEEPSPEED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "ds_zero2.json")
DEFAULT_BUDGET_SET = "1,2,3,4,5,6,7,8"

EPOCHS, MICRO_BATCH, GRAD_ACCUM = 3, 2, 8        # 2 img/GPU ×8 accum ×2 GPU
LR_LLM, LR_PROJ, LR_ENCODER = 1e-5, 1e-4, 5e-6
WD_LLM, WD_OTHER, WARMUP_RATIO = 0.1, 0.05, 0.03
GRAD_CLIP, LOGGING_STEPS, SAVE_STEPS, SEED = 1.0, 20, 500, 42


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    p = argparse.ArgumentParser(
        description="vlm8b: AssocVTC encoder + 8-budget projector + Qwen3.5-4B "
                    "full finetune (DeepSpeed ZeRO-3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--encoder-ckpt", default=DEFAULT_ENCODER_CKPT,
                   help="阶段2 ckpt（只读 encoder.* 权重）")
    p.add_argument("--qwen-dir", default=DEFAULT_QWEN_DIR)
    p.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--deepspeed", default=DEFAULT_DEEPSPEED,
                   help="DeepSpeed json；传空串可关闭（单卡调试）")
    p.add_argument("--epochs", type=float, default=EPOCHS)
    p.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH,
                   help="每 GPU 的**图片**数（行数 = 图片数 × 预算数）")
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr-llm", type=float, default=LR_LLM)
    p.add_argument("--lr-proj", type=float, default=LR_PROJ)
    p.add_argument("--lr-encoder", type=float, default=LR_ENCODER)
    p.add_argument("--wd-llm", type=float, default=WD_LLM)
    p.add_argument("--wd-other", type=float, default=WD_OTHER)
    p.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--freeze-encoder", default="none",
                   help="encoder 冻结规格：none/all/stem/blocks=0-8/blocks=0,1/last=2/"
                        "norm，逗号组合；语义见 freeze.py")
    p.add_argument("--freeze-encoder-epochs", type=int, default=0,
                   help="前 N 个 epoch 冻结 encoder（0=不按 epoch 冻结）；"
                        ">0 时需 DeepSpeed，冻结在 engine 建好后生效、E(N+1) 自动解冻")
    p.add_argument("--budget-set", default=DEFAULT_BUDGET_SET,
                   help="预算集合，逗号分隔，如 1,2,4,8")
    p.add_argument("--loss-weighting", default="equal", choices=["equal", "inv-k"])
    p.add_argument("--smoke", action="store_true", help="冒烟：2 step + 不存盘 + 每步日志")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--save-total-limit", type=int, default=2,
                   help="保留的 epoch ckpt 数（磁盘紧时设 1）")
    p.add_argument("--init-from", default="",
                   help="从已有训练产物目录（含 model.safetensors）继续训练")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="开 gradient checkpointing（省显存，但 ZeRO-3 下有卡死风险）")
    return p.parse_args()


def parse_budget_set(raw: str) -> list:
    """`"1,2,4,8"` → `[1, 2, 4, 8]`（升序去重）。"""
    return sorted({int(part) for part in str(raw).split(",") if part.strip()})


def deepspeed_stage(path: str) -> int:
    """读 deepspeed json 的 ZeRO stage（文件缺失/未配 `zero_optimization` → 0）。"""
    if not path:
        return 0
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        return int(cfg.get("zero_optimization", {}).get("stage", 0) or 0)
    except (OSError, ValueError, TypeError):
        return 0


def assert_zero3_freeze_compatible(stage: int, freeze_active: bool) -> None:
    """ZeRO-3 与 AssocVTCEncoder 不兼容，提前给可执行的报错。

    根因在 `model.AssocVTCEncoder` 的文档里：`cls_token` / `special_tokens` /
    `pos_embed` 是挂在自己身上的**裸 Parameter**（不是子模块参数）。ZeRO-3 只在
    模块边界 all-gather，裸参数永远是空分区 ⇒ 前向直接
    `size of tensor a (0) must match ... b (1744)`（2×MTT S4000 + deepspeed 0.19.9
    实测，**不冻结也会崩**；`GatheredParameters` 也救不回来，DS 的 pre/post hook
    会重新分区）。冻结还额外叠加「engine 初始化后不能整组剔空」的问题。
    → 本仓在有 encoder 的场景只走 ZeRO-1/2（默认 `ds_zero2.json`）。
    """
    if stage >= 3:
        raise RuntimeError(
            "ZeRO-3 与 AssocVTCEncoder 不兼容（不冻结也会崩）：encoder 的 cls_token/"
            "special_tokens/pos_embed 是裸 Parameter，ZeRO-3 不会为它们 all-gather，"
            "前向会拿到空分区（size of tensor a (0) vs b (1744)）。"
            + ("此外 ZeRO-3 下冻结 encoder 还会踩 engine 初始化剔空整组的问题。"
               if freeze_active else "")
            + "请改用 --deepspeed ds_zero2.json（本仓默认，2×MTT S4000 已验证）。")


# ═══════════════════════════════════════════════════════════════
# 构建：param groups
# ═══════════════════════════════════════════════════════════════

def load_init_weights(model: VLM8Budget, checkpoint: str) -> None:
    """从已有产物目录灌入权重（用于续训；safetensors 单文件或分片均可）。"""
    from safetensors.torch import load_file
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as handle:
            names = sorted(set(json.load(handle)["weight_map"].values()))
        paths = [os.path.join(checkpoint, name) for name in names]
    else:
        paths = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    assert paths, f"--init-from 目录里没有 safetensors: {checkpoint}"
    state = {}
    for path in paths:
        state.update(load_file(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"lm.lm_head.weight"}, (
        f"续训权重不匹配: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    print(f"[vlm8b] 续训权重已载入 {checkpoint}（{len(paths)} 文件，"
          f"missing={list(missing)}）")


def build_param_groups(model: VLM8Budget, lr_llm: float, lr_proj: float,
                       lr_encoder: float, wd_llm: float, wd_other: float) -> list:
    """三组参数（LLM / projector / encoder）各自 lr + wd；bias/LayerNorm 不衰减。

    实现在 `freeze.build_param_groups()`：**只收 requires_grad=True 的参数，并丢弃
    空组**。这不是洁癖——DeepSpeed 在 engine 初始化时按 `is_optimized_parameter()`
    （requires_grad and numel>0）过滤参数，若某个 group 被整组剔空，stage 1/2 会在
    `stage_1_and_2.py` 取 `bit16_groups[i][0].dtype` 时 `IndexError`（上游 vlm8b
    首跑就死在这里）。整段冻结 encoder 时这里会自然产出「只有 lm + proj」的组。
    """
    return _build_param_groups(
        ((model.lm, lr_llm, wd_llm),
         (model.projector, lr_proj, wd_other),
         (model.encoder, lr_encoder, wd_other)))


# ═══════════════════════════════════════════════════════════════
# Trainer + callbacks
# ═══════════════════════════════════════════════════════════════

class VLM8BudgetTrainer(Trainer):
    """用 init 时建好的 param groups 构造 AdamW（β=0.9/0.95, eps=1e-8）。

    `freeze_encoder_epochs > 0` 时，冻结/解冻由 `EncoderFreezeCallback` 在 engine
    建好之后执行；本类只在 `train()` 进 engine 之前做「冻结时机」fail-fast。
    """

    freeze_encoder_epochs = 0

    def __init__(self, *args, param_groups=None, freeze_encoder_epochs=0, **kwargs):
        self._param_groups = param_groups
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Trainer 默认只有单组 lr；这里换成三组。"""
        if self.optimizer is None:
            self.optimizer = AdamW(self._param_groups, betas=(0.9, 0.95), eps=1e-8)
        return self.optimizer

    def train(self, *args, **kwargs):
        """进 DeepSpeed engine 之前检查冻结时机（此时 self.model 还是普通 nn.Module）。

        两种错误配置必须在这里拦住，否则报错完全指不到根因：

        1. **按 epoch 冻结却提前冻了**：DS 0.19.9 在 engine 初始化时把冻结参数剔出
           ZeRO 分区 → stage 1/2 `IndexError`；stage 3 不崩，但该组永不进优化器，
           解冻后静默不更新。
        2. **按 epoch 冻结却没开 DeepSpeed**：DDP 在 wrap 时就把参数分桶，之后改
           `requires_grad` 会触发 reducer 的「unused parameter」错，且同样无法解冻。

        静态冻结（`--freeze-encoder X` 且 epochs=0）不在此列：参数一开始就不在
        optimizer 里，也不需要解冻。
        """
        if self.freeze_encoder_epochs > 0:
            if self.args.deepspeed is None:
                raise RuntimeError(
                    "--freeze-encoder-epochs 只支持 DeepSpeed 路径（--deepspeed ds_zero2.json）；"
                    "单卡/纯 DDP 下 DDP 在 wrap 时已分桶，无法事后解冻。"
                    "要么用静态冻结（去掉 --freeze-encoder-epochs），要么开 DeepSpeed。")
            frozen = [name for name, param in self.model.encoder.named_parameters()
                      if not param.requires_grad]
            if frozen:
                raise RuntimeError(
                    f"encoder 在 Trainer 创建前就被冻结了（{len(frozen)} 个参数，如 {frozen[:3]}）。"
                    "DeepSpeed 只在 engine 初始化时收集未冻结参数：stage1/2 会 "
                    "IndexError，stage3 则解冻后静默不更新。请把冻结交给 "
                    "EncoderFreezeCallback.on_train_begin（engine 建好之后）。")
        return super().train(*args, **kwargs)


class EncoderFreezeCallback(TrainerCallback):
    """按 epoch 冻结/解冻 encoder；只在 engine 建好之后切换 requires_grad。

    时序契约（transformers 5.x）：`Trainer.train()` → `_prepare_for_training()`
    （DeepSpeed engine 初始化）→ `on_train_begin` → `on_epoch_begin`。所以在
    `on_train_begin` 里冻结是安全的：参数已经在优化器里，冻结只是「没有梯度」，
    AdamW 对 `grad is None` 的参数整步跳过，E(N+1) 解冻后照常更新。
    """

    def __init__(self, encoder, spec: str, freeze_epochs: int):
        self.encoder = encoder
        self.spec = str(spec)
        self.freeze_epochs = int(freeze_epochs)
        self._frozen = False

    @staticmethod
    def _clear_grads(kwargs) -> None:
        """冻结状态切换后必须清一次梯度。

        DeepSpeed ZeRO-1/2 把梯度累积在扁平分区里（`averaged_gradients`），
        `requires_grad=False` 只是让下一步**不再产生**新梯度，**不会**清掉上一轮已
        累积的那份。不清就会在切换后的第一步用「上一轮的 stale grad」再更新一次
        ——实测 2×MTT S4000 + ZeRO-2 上冻结后的 encoder 被推动 6.1e-5（一个 bf16 ulp）。
        """
        model = kwargs.get("model")
        if model is not None and hasattr(model, "zero_grad"):
            try:
                model.zero_grad(set_to_none=True)
            except TypeError:
                model.zero_grad()

    def _apply(self, spec: str, **kwargs) -> None:
        report: FreezeReport = freeze_encoder(self.encoder, spec)
        self._frozen = report.active
        self._clear_grads(kwargs)
        print(f"[freeze] {report.summary()}")

    def on_train_begin(self, args, state, control, **kwargs):
        if self.freeze_epochs > 0:
            self._apply(self.spec, **kwargs)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.freeze_epochs <= 0:
            return
        epoch = float(state.epoch or 0.0)
        if epoch >= self.freeze_epochs and self._frozen:
            print(f"[freeze] epoch {epoch:g} ≥ {self.freeze_epochs} → 解冻 encoder")
            report = unfreeze_all(self.encoder)
            self._frozen = report.active
            self._clear_grads(kwargs)
            print(f"[freeze] {report.summary()}")


class BudgetCELogCallback(TrainerCallback):
    """每个 logging step 把 `model.last_budget_ce` 追加到 JSONL。"""

    def __init__(self, model: VLM8Budget, path: str):
        self.model = model
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or self.model.last_budget_ce is None:
            return
        record = {"step": int(state.global_step),
                  "epoch": None if state.epoch is None else round(float(state.epoch), 4),
                  **self.model.last_budget_ce}
        self.model.last_budget_ce = None
        for key in ("loss", "learning_rate", "grad_norm"):
            if logs and logs.get(key) is not None:
                record[key] = float(logs[key])
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """构建模型/数据/Trainer，跑 3 epoch 并保存最终 HF 权重。"""
    # 国产适配第一步：探测设备 + 打 DeepSpeed 补丁（必须在 engine 初始化前）。
    musa.bootstrap()
    args = parse_args()
    if args.smoke:
        print("[vlm8b] smoke mode: max_steps=2, save_strategy=no")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    budgets = parse_budget_set(args.budget_set)
    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SelfDistillSFTDataset(
        args.train_jsonl, tokenizer, args.image_root, build_image_transform(train=True))
    print(f"[vlm8b] dataset={len(train_dataset)} "
          f"no_image={sum(1 for r in train_dataset.records if not r['has_image'])}")

    # transformers 5.x 的 TrainingArguments 没有 warmup_ratio（只有 warmup_steps），
    # 这里按 数据集/等效批 估总步数后换算；smoke 不设 warmup。
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    steps_per_epoch = math.ceil(
        len(train_dataset) / (args.micro_batch_size * args.grad_accum * world_size))
    total_steps = 2 if args.smoke else int(math.ceil(steps_per_epoch * args.epochs))
    warmup_steps = 0 if args.smoke else max(1, int(round(total_steps * args.warmup_ratio)))
    print(f"[vlm8b] dataset={len(train_dataset)} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps} warmup_steps={warmup_steps}")

    model = VLM8Budget(VLM8BudgetConfig(
        encoder_ckpt=args.encoder_ckpt, qwen_dir=args.qwen_dir,
        budget_ids=budgets,
        loss_weighting=args.loss_weighting,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id),
        grad_ckpt=args.grad_ckpt)
    assert not args.grad_ckpt, (
        "--grad-ckpt 与两段式前向不兼容：梯度检查点会强制 use_cache=False，"
        "prompt 的 KV cache 就没了；算力已经省在 prompt 只前向一次上，别再开检查点")
    if args.init_from:
        load_init_weights(model, args.init_from)

    # 静态冻结：必须在建 param_groups / Trainer 之前（组里就不会出现冻结参数）。
    # 按 epoch 冻结（freeze_encoder_epochs>0）则相反：现在不能冻，交给 callback。
    freeze_report = None
    if args.freeze_encoder_epochs <= 0:
        freeze_report = model.freeze_encoder(args.freeze_encoder)
        print(f"[freeze] {freeze_report.summary()}")
    else:
        print(f"[freeze] 按 epoch 冻结：E1..E{args.freeze_encoder_epochs} 冻结 "
              f"({args.freeze_encoder})，之后自动解冻（engine 建好后生效）")

    param_groups = build_param_groups(
        model, args.lr_llm, args.lr_proj, args.lr_encoder,
        args.wd_llm, args.wd_other)
    assert_zero3_freeze_compatible(
        deepspeed_stage(args.deepspeed) if args.deepspeed else 0,
        freeze_active=(args.freeze_encoder_epochs > 0
                       or (freeze_report is not None and freeze_report.active)))
    print(f"[vlm8b] param_groups={len(param_groups)} "
          f"({', '.join(str(len(g['params'])) for g in param_groups)} 张量)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr_llm,           # 组 lr 由 param_groups 覆盖
        weight_decay=args.wd_llm,
        max_grad_norm=GRAD_CLIP,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=1 if args.smoke else LOGGING_STEPS,
        save_strategy="no" if args.smoke else "epoch",
        save_steps=SAVE_STEPS,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",                 # forward 不返回 logits ⇒ 不用 prediction_step
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=[],
        deepspeed=args.deepspeed or None,
        max_steps=2 if args.smoke else -1,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
        save_only_model=True,               # 中途不存 optimizer
    )

    trainer = VLM8BudgetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EightBudgetCollator(tokenizer),
        param_groups=param_groups,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        callbacks=[
            EncoderFreezeCallback(
                model.encoder, args.freeze_encoder, args.freeze_encoder_epochs),
            BudgetCELogCallback(
                model, os.path.join(args.output_dir, "per_budget_ce.jsonl")),
        ],
    )

    print(f"[vlm8b] budgets={budgets} loss_weighting={args.loss_weighting} "
          f"epochs={args.epochs} "
          f"micro={args.micro_batch_size}×{args.grad_accum} "
          f"lr={args.lr_llm:g}/{args.lr_proj:g}/{args.lr_encoder:g}")

    trainer.train()
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)           # 只有 world-zero 真正落盘（args.should_save）
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()   # 其余 rank 等写完，避免抢读半成品
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(args.output_dir, "run_args.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2)
        print(f"[vlm8b] final model saved to {final_dir}")


if __name__ == "__main__":
    main()
