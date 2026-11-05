"""
vlm8b 训练入口 — HF Trainer + DeepSpeed ZeRO-3（2×A800）
=========================================================

协议（`doc/2026-11-05/DESIGN_8budget_vlm_finetune.md` §5）::

    E1（epoch 1）    encoder 冻结（requires_grad=False / eval）
                    只训 Qwen + projector
    E2–E3（epoch 2+） 全解冻：encoder lr 5e-6 / LLM 1e-5 / proj 1e-4
                    所有组都是 cosine → 0，warmup 3%，grad_clip 1.0

    每步：B 张图 × 8 行（k=1..8），行 k = [vis[:, :k]; text]，
          loss = Σ_k w_k·CE_k（w 等权或 ∝1/k）。

本文件包含::

    1. 默认路径 / 超参常量
    2. argparse
    3. tokenizer / 数据集 / param groups 构建
    4. VLM8BudgetTrainer（不同 lr 的分组 optimizer）
    5. EncoderFreezeCallback（epoch 边界冻结/解冻）
    6. BudgetCELogCallback（每 N 步把 per-budget CE 写 JSONL）
    7. main()

启动（remote, 2 卡）::

    torchrun --nproc_per_node=2 vlm8b/train.py \
        --output-dir /root/autodl-tmp/vlm8b/output/run1 \
        --train-jsonl .../train_nothink_all.jsonl \
        --val-jsonl   .../val_nothink_all.jsonl

先用 `--smoke`（= `--max-steps 2`）确认能跑通再长跑。

**Trainer 内不做 eval**（`forward` 不返回 `logits`，`prediction_step` 无法工作）：
`eval_strategy` 恒为 `"no"`、不传 `eval_dataset`；每 epoch 的 ckpt 由
`--save-strategy epoch` 存下，训练结束后用 `eval.py` 评测（README §4.4 / §11）。
"""

import json
import os
from typing import Dict, List, Optional

from torch.optim import AdamW
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:                                            # script / package 两种运行方式
    from .model import VLM8Budget, VLM8BudgetConfig
    from .data import (
        EightBudgetCollator,
        SelfDistillSFTDataset,
        build_image_transform,
    )
except ImportError:                             # pragma: no cover - 脚本入口
    from model import VLM8Budget, VLM8BudgetConfig
    from data import (
        EightBudgetCollator,
        SelfDistillSFTDataset,
        build_image_transform,
    )


# ═══════════════════════════════════════════════════════════════
# 默认路径（remote 实测位置；全部可用 CLI 覆盖）
# ═══════════════════════════════════════════════════════════════

DEFAULT_ENCODER_CKPT = "/root/autodl-tmp/exp448/B/ckpt_best.pt"
DEFAULT_QWEN_DIR = "/root/qwen_models/Qwen3.5-4B"
DEFAULT_IMAGE_ROOT = "/root/autodl-tmp/datasets/imagenet100"
DEFAULT_DATA_DIR = "/root/autodl-tmp/exp_qwen/in100_selfdistill/sft"
DEFAULT_TRAIN_JSONL = os.path.join(DEFAULT_DATA_DIR, "train_nothink_all.jsonl")
DEFAULT_VAL_JSONL = os.path.join(DEFAULT_DATA_DIR, "val_nothink_all.jsonl")
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/vlm8b/output/run1"
DEFAULT_DEEPSPEED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ds_zero3.json")

#: 默认超参（设计 §5）。
DEFAULT_EPOCHS = 3
DEFAULT_MICRO_BATCH = 2                  # 2 img/GPU = 16 行
DEFAULT_GRAD_ACCUM = 8                   # ×2 GPU ⇒ global 32 img/step = 256 行
DEFAULT_LR_LLM = 1e-5
DEFAULT_LR_PROJ = 1e-4
DEFAULT_LR_ENCODER = 5e-6
DEFAULT_WD_LLM = 0.1
DEFAULT_WD_OTHER = 0.05
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_LOGGING_STEPS = 20
DEFAULT_BUDGET_SET = "1,2,3,4,5,6,7,8"
DEFAULT_SEED = 42
DEFAULT_MAX_TEXT_LENGTH = 1024

#: Trainer 内**不做** eval（P1#1）。原因：`VLM8Budget.forward()` 只返回
#: loss / per-budget CE，**不返回 `logits`**，HF `prediction_step` 取不到 logits
#: （要么崩、要么产出无意义指标）。评测统一走 `eval.py`；训练中每 epoch 的 ckpt
#: 由 `save_strategy="epoch"` 存下，训练结束后再跑 eval。
#: `smoke.py::check_trainer_no_eval` 用微型替身跑 1 步 `Trainer.train()` 验证
#: 这条默认路径不碰 `prediction_step`。
TRAINER_EVAL_STRATEGY: str = "no"


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    """解析训练参数（所有设计签字点都有对应 CLI，见 README §6）。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="vlm8b: AssocVTC encoder + 8-budget projector + Qwen3.5-4B "
                    "full finetune (DeepSpeed ZeRO-3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 路径 ──
    paths = parser.add_argument_group("paths")
    paths.add_argument("--encoder-ckpt", default=DEFAULT_ENCODER_CKPT,
                       help="阶段2 ckpt（只读 encoder.* 权重）")
    paths.add_argument("--qwen-dir", default=DEFAULT_QWEN_DIR,
                       help="Qwen3.5-4B 目录")
    paths.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT,
                       help="IN-100 图片根目录")
    paths.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    paths.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL,
                       help="【deprecated，忽略】Trainer 内不做 eval，本参数不读；"
                            "只保留给旧启动脚本，评测用 eval.py --jsonl")
    paths.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    paths.add_argument("--deepspeed", default=DEFAULT_DEEPSPEED,
                       help="DeepSpeed json；传空串可关闭（单卡调试）")
    paths.add_argument("--assoctvc-repo", default="",
                       help="AssocVTC 代码目录（默认 = vlm8b/ 的上一级）")

    # ── 数据 ──
    data = parser.add_argument_group("data")
    data.add_argument("--messages-key", default="messages_short",
                      choices=["messages_short", "messages"])
    data.add_argument("--train-transform", default="rrc",
                      choices=["rrc", "deterministic"],
                      help="训练图变换：rrc=RandomResizedCrop+HFlip（默认）；"
                           "deterministic=val 口径 Resize+CenterCrop")
    data.add_argument("--max-text-length", type=int,
                      default=DEFAULT_MAX_TEXT_LENGTH)
    data.add_argument("--num-workers", type=int, default=4)
    data.add_argument("--train-limit", type=int, default=0,
                      help="只用前 N 条训练（0=全部；smoke 用）")
    data.add_argument("--eval-limit", type=int, default=0,
                      help="【deprecated，忽略】原为 Trainer 每 epoch 快评取前 N 条 "
                           "val；forward 不返回 logits，Trainer 内 eval 已关闭。"
                           "评测请用 eval.py --limit")

    # ── 训练 ──
    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    train.add_argument("--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH,
                       help="每 GPU 的**图片**数（行数 = 图片数 × 预算数）")
    train.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    train.add_argument("--lr-llm", type=float, default=DEFAULT_LR_LLM)
    train.add_argument("--lr-proj", type=float, default=DEFAULT_LR_PROJ)
    train.add_argument("--lr-encoder", type=float, default=DEFAULT_LR_ENCODER)
    train.add_argument("--wd-llm", type=float, default=DEFAULT_WD_LLM)
    train.add_argument("--wd-other", type=float, default=DEFAULT_WD_OTHER)
    train.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    train.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    train.add_argument("--freeze-encoder-epochs", type=int, default=1,
                       help="前 N 个 epoch 冻结 encoder（E1=1；0=全程解冻）")
    train.add_argument("--budget-set", default=DEFAULT_BUDGET_SET,
                       help="训练/记录的预算集合，逗号分隔，如 1,2,4,8")
    train.add_argument("--loss-weighting", default="equal",
                       choices=["equal", "inv-k"])
    train.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    train.add_argument("--save-strategy", default="epoch",
                       choices=["epoch", "steps", "no"])
    train.add_argument("--save-steps", type=int, default=500)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--max-steps", type=int, default=-1,
                       help=">0 时覆盖 epochs（smoke 用）")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--disable-gradient-checkpointing", action="store_true")
    train.add_argument("--resume-from-checkpoint", default=None,
                       help="Trainer checkpoint 目录（只存了模型，无 optimizer 状态）")
    train.add_argument("--smoke", action="store_true",
                       help="冒烟：--max-steps 2 + 不存盘 + 每步日志")

    return parser.parse_args()


def parse_budget_set(raw: str) -> List[int]:
    """`"1,2,4,8"` → [1,2,4,8]（升序去重）。"""
    ks = sorted({int(part) for part in str(raw).split(",") if part.strip()})
    if not ks:
        raise ValueError(f"--budget-set 为空: {raw!r}")
    return ks


# ═══════════════════════════════════════════════════════════════
# 构建：tokenizer / dataset / param groups
# ═══════════════════════════════════════════════════════════════

def build_tokenizer(qwen_dir: str):
    """加载 Qwen tokenizer；`pad_token` 缺失时用 eos（right-padding）。"""
    tokenizer = AutoTokenizer.from_pretrained(
        qwen_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataset(
    jsonl_path: str,
    tokenizer,
    image_root: str,
    train: bool,
    messages_key: str,
    max_text_length: int,
    limit: int = 0,
    augment: bool = True,
) -> Dataset:
    """构造 SFT dataset（train/val 只差图像变换）；`limit>0` 时取前 N 条。"""
    dataset = SelfDistillSFTDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        image_root=image_root,
        transform=build_image_transform(train=train, augment=augment),
        messages_key=messages_key,
        max_text_length=max_text_length,
    )
    if limit and limit > 0:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    return dataset


def build_param_groups(
    model: VLM8Budget,
    lr_llm: float,
    lr_proj: float,
    lr_encoder: float,
    wd_llm: float,
    wd_other: float,
) -> List[Dict[str, object]]:
    """E1/E2 共用的三组参数（LLM / projector / encoder），各自 lr + wd。

    每组再按"是否衰减"拆成两小组：``ndim >= 2``（Linear/Embedding）才加 wd，
    bias / LayerNorm 不衰减（Qwen 预训练口径）。

    ⚠️ E1 冻结 encoder 时，encoder 组 **仍留在 optimizer 里**（requires_grad=False
    ⇒ 无梯度、不更新）。原因是 ZeRO-3 在 engine 初始化后无法把新参数加进
    optimizer；若 E1 直接排除，E2 解冻后 encoder 永远拿不到更新。详见 README §7
    的"实现裁决"。
    """
    if model.encoder is None or model.lm is None:
        raise RuntimeError("build_param_groups 需要已加载 encoder 与 lm")

    groups: List[Dict[str, object]] = []
    for name, module, lr, wd in (
        ("llm", model.lm, lr_llm, wd_llm),
        ("projector", model.projector, lr_proj, wd_other),
        ("encoder", model.encoder, lr_encoder, wd_other),
    ):
        params = list(module.parameters())
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        groups.append({"params": decay, "lr": lr, "weight_decay": wd})
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        print(f"[vlm8b] param group {name:9s}: lr={lr:g} wd={wd:g} "
              f"params={sum(p.numel() for p in params):,}")
    return groups


# ═══════════════════════════════════════════════════════════════
# Trainer —— 分组 lr（HF 默认只有一个 lr）
# ═══════════════════════════════════════════════════════════════

class VLM8BudgetTrainer(Trainer):
    """把 init 时建好的 param groups 交给 AdamW（β=0.9/0.95, eps=1e-8）。

    HF Trainer 默认按 `args.learning_rate` 建单组；这里用 `param_groups`
    覆盖，scheduler 仍然是 Trainer 的 cosine + warmup（按组 lr 各自缩放）。
    """

    def __init__(self, *args, param_groups: Optional[List[Dict[str, object]]] = None,
                 **kwargs):
        self._param_groups = param_groups
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """构造 AdamW；DeepSpeed 初始化时也会调用本方法。"""
        if self.optimizer is None:
            if self._param_groups is None:
                raise RuntimeError("VLM8BudgetTrainer 需要 param_groups")
            self.optimizer = AdamW(
                self._param_groups, betas=(0.9, 0.95), eps=1e-8)
        return self.optimizer


# ═══════════════════════════════════════════════════════════════
# Callback 1 —— epoch 边界冻结/解冻 encoder
# ═══════════════════════════════════════════════════════════════

class EncoderFreezeCallback(TrainerCallback):
    """E1 冻结 / E2–E3 解冻，并在状态真正切换时打一行日志。"""

    def __init__(self, encoder, freeze_epochs: int):
        self.encoder = encoder
        self.freeze_epochs = int(freeze_epochs)
        self._state: Optional[bool] = None

    def _apply(self, freeze: bool, epoch: Optional[float]) -> None:
        """幂等地切换冻结状态。"""
        if self._state is freeze:
            return
        if freeze:
            self.encoder.freeze()
        else:
            self.encoder.unfreeze()
        self._state = freeze
        epoch_text = "begin" if epoch is None else f"epoch {int(epoch) + 1}"
        phase = "E1 (encoder FROZEN)" if freeze else "E2+ (encoder TRAINABLE)"
        print(f"[vlm8b] {epoch_text}: {phase}")

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时先落到 E1 的状态（freeze_epochs=0 则直接全解冻）。"""
        self._apply(self.freeze_epochs > 0, state.epoch)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个 epoch 开始时判断是否跨过冻结边界。"""
        epoch = int(state.epoch or 0)
        self._apply(epoch < self.freeze_epochs, state.epoch)


# ═══════════════════════════════════════════════════════════════
# Callback 2 —— per-budget CE 写 JSONL（主进程）
# ═══════════════════════════════════════════════════════════════

class BudgetCELogCallback(TrainerCallback):
    """每个 logging step 从 model 取走累积的 per-budget CE，追加到 JSONL。"""

    def __init__(self, model: VLM8Budget, path: str):
        self.model = model
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """写入 ``{"step","epoch","loss","per_budget_ce":{k:ce}}``。"""
        if not state.is_world_process_zero:
            return
        stats = self.model.pop_budget_ce_stats()
        if stats is None:
            return
        budget_ks = list(stats.get("budget_ks") or [])
        per_budget = list(stats.get("per_budget_ce") or [])
        record = {
            "step": int(state.global_step),
            "epoch": None if state.epoch is None else round(float(state.epoch), 4),
            "n_micro": int(stats.get("n_micro", 0)),
            "micro_loss": float(stats.get("loss", 0.0)),
            "per_budget_ce": {
                str(k): float(v) for k, v in zip(budget_ks, per_budget)},
        }
        if per_budget:
            record["ce_mean"] = float(sum(per_budget) / len(per_budget))
        if logs:
            for key in ("loss", "learning_rate", "grad_norm"):
                if key in logs and logs[key] is not None:
                    record[key] = float(logs[key])
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """构建模型/数据/Trainer，跑 E1–E3 并保存最终 HF 权重。"""
    args = parse_args()
    if args.smoke:
        if args.max_steps <= 0:
            args.max_steps = 2
        args.save_strategy = "no"
        args.logging_steps = 1
        print("[vlm8b] smoke mode: max_steps=2, save_strategy=no")
    if args.eval_limit > 0:
        print("[vlm8b] warning: --eval-limit 已废弃且被忽略"
              "（forward 不返回 logits，Trainer 内不做 eval）；"
              "请在训练结束后用 eval.py --limit 评测")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    budget_ids = parse_budget_set(args.budget_set)
    tokenizer = build_tokenizer(args.qwen_dir)
    print(f"[vlm8b] tokenizer: pad_token_id={tokenizer.pad_token_id} "
          f"eos_token_id={tokenizer.eos_token_id}")

    train_dataset = build_dataset(
        args.train_jsonl, tokenizer, args.image_root, train=True,
        messages_key=args.messages_key, max_text_length=args.max_text_length,
        limit=args.train_limit, augment=args.train_transform == "rrc")
    # Trainer 内不 eval（P1#1）：不构造 val 数据集、不传 eval_dataset，
    # 因此 HF 永远不会走 prediction_step（它需要 forward 返回 logits）。
    print(f"[vlm8b] train={len(train_dataset)}  "
          f"eval=0 (Trainer 内不 eval；评测走 eval.py)")

    config = VLM8BudgetConfig(
        encoder_ckpt=args.encoder_ckpt,
        qwen_dir=args.qwen_dir,
        assoctvc_repo=args.assoctvc_repo,
        budget_ids=tuple(budget_ids),
        loss_weighting=args.loss_weighting,
        freeze_encoder=args.freeze_encoder_epochs > 0,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        auto_load_modules=True,
    )
    model = VLM8Budget(config)
    model.set_loss_weighting(args.loss_weighting)
    print(f"[vlm8b] encoder params={model.encoder.num_parameters():,}  "
          f"llm params={sum(p.numel() for p in model.lm.parameters()):,}  "
          f"projector params={sum(p.numel() for p in model.projector.parameters()):,}")

    param_groups = build_param_groups(
        model, lr_llm=args.lr_llm, lr_proj=args.lr_proj,
        lr_encoder=args.lr_encoder, wd_llm=args.wd_llm,
        wd_other=args.wd_other)

    collator = EightBudgetCollator(tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr_llm,          # 组 lr 由 param_groups 覆盖
        weight_decay=args.wd_llm,
        max_grad_norm=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # ↓ Trainer 内不做 eval（P1#1）：forward 不返回 logits，
        #   prediction_step 无法工作 ⇒ 固定 "no"，评测交给 eval.py。
        eval_strategy=TRAINER_EVAL_STRATEGY,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,       # model.py 里手动对 Qwen 开
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,        # collator 输出含 gt_name
        report_to=[],
        deepspeed=args.deepspeed if args.deepspeed else None,
        max_steps=args.max_steps,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
        save_only_model=True,               # 设计 §5：中途不存 optimizer
    )

    trainer = VLM8BudgetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        param_groups=param_groups,
        callbacks=[
            EncoderFreezeCallback(model.encoder, args.freeze_encoder_epochs),
            BudgetCELogCallback(
                model, os.path.join(args.output_dir, "per_budget_ce.jsonl")),
        ],
    )

    print(f"\n{'=' * 62}\n"
          f"vlm8b training\n"
          f"  budgets: {budget_ids}  loss_weighting={args.loss_weighting}\n"
          f"  epochs: {args.epochs}  freeze_encoder_epochs="
          f"{args.freeze_encoder_epochs}\n"
          f"  micro: {args.micro_batch_size} img/GPU × {args.grad_accum} accum\n"
          f"  lr: llm={args.lr_llm:g} proj={args.lr_proj:g} "
          f"enc={args.lr_encoder:g}  warmup={args.warmup_ratio:.0%}\n"
          f"  output: {args.output_dir}\n{'=' * 62}\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)           # ZeRO-3 下由 Trainer gather 后保存
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(args.output_dir, "run_args.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2)
        print(f"[vlm8b] final model saved to {final_dir}")


if __name__ == "__main__":
    main()
