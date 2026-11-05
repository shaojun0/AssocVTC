"""
vlm8b 训练入口 — HF Trainer + DeepSpeed ZeRO-3
==============================================

协议（设计 §5）: E1 encoder 冻结（requires_grad=False / eval），只训 LLM + projector；
E2–E3 全解冻（encoder 5e-6 / LLM 1e-5 / proj 1e-4），cosine + warmup 3%。
每步 B 张图 × V 行（k=1..8），行 k = `[vis[:, :k]; text]`，loss = Σ_k w_k·CE_k。

**Trainer 内不做 eval**：`forward` 不返回 logits，`prediction_step` 无法工作 ⇒
`eval_strategy="no"` 且不传 eval_dataset；评测统一走 `eval.py`。

启动（remote, 2 卡）::

    torchrun --nproc_per_node=2 train.py \
        --output-dir /root/autodl-tmp/vlm8b/output/run1 \
        --train-jsonl /root/autodl-tmp/exp_qwen/in100_selfdistill/sft/train_nothink_all.jsonl
"""
import argparse
import json
import math
import os

from torch.optim import AdamW
from transformers import (AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from data import EightBudgetCollator, SelfDistillSFTDataset, build_image_transform
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
DEFAULT_DEEPSPEED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "ds_zero3.json")
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
    p.add_argument("--freeze-encoder-epochs", type=int, default=1,
                   help="前 N 个 epoch 冻结 encoder（E1=1；0=全程解冻）")
    p.add_argument("--budget-set", default=DEFAULT_BUDGET_SET,
                   help="预算集合，逗号分隔，如 1,2,4,8")
    p.add_argument("--loss-weighting", default="equal", choices=["equal", "inv-k"])
    p.add_argument("--smoke", action="store_true", help="冒烟：2 step + 不存盘 + 每步日志")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--grad-ckpt", action="store_true",
                   help="开 gradient checkpointing（省显存，但 ZeRO-3 下有卡死风险）")
    return p.parse_args()


def parse_budget_set(raw: str) -> list:
    """`"1,2,4,8"` → `[1, 2, 4, 8]`（升序去重）。"""
    return sorted({int(part) for part in str(raw).split(",") if part.strip()})


# ═══════════════════════════════════════════════════════════════
# 构建：param groups
# ═══════════════════════════════════════════════════════════════

def build_param_groups(model: VLM8Budget, lr_llm: float, lr_proj: float,
                       lr_encoder: float, wd_llm: float, wd_other: float) -> list:
    """三组参数（LLM / projector / encoder）各自 lr + wd；bias/LayerNorm 不衰减。

    E1 冻结期 encoder 组仍留在 optimizer 里（requires_grad=False ⇒ 无梯度）：
    ZeRO-3 engine 初始化后无法再加参数，否则 E2 解冻后 encoder 永远拿不到更新。
    """
    groups = []
    for module, lr, wd in ((model.lm, lr_llm, wd_llm),
                           (model.projector, lr_proj, wd_other),
                           (model.encoder, lr_encoder, wd_other)):
        params = list(module.parameters())
        groups.append({"params": [p for p in params if p.ndim >= 2],
                       "lr": lr, "weight_decay": wd})
        groups.append({"params": [p for p in params if p.ndim < 2],
                       "lr": lr, "weight_decay": 0.0})
    return groups


# ═══════════════════════════════════════════════════════════════
# Trainer + callbacks
# ═══════════════════════════════════════════════════════════════

class VLM8BudgetTrainer(Trainer):
    """用 init 时建好的 param groups 构造 AdamW（β=0.9/0.95, eps=1e-8）。"""

    def __init__(self, *args, param_groups=None, **kwargs):
        self._param_groups = param_groups
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Trainer 默认只有单组 lr；这里换成三组。"""
        if self.optimizer is None:
            self.optimizer = AdamW(self._param_groups, betas=(0.9, 0.95), eps=1e-8)
        return self.optimizer


class EncoderFreezeCallback(TrainerCallback):
    """E1 冻结 / E2–E3 解冻 encoder（状态真的切换时才打日志）。"""

    def __init__(self, encoder, freeze_epochs: int):
        self.encoder = encoder
        self.freeze_epochs = int(freeze_epochs)
        self._frozen = None

    def _apply(self, freeze: bool) -> None:
        if self._frozen is freeze:
            return
        (self.encoder.freeze if freeze else self.encoder.unfreeze)()
        self._frozen = freeze
        print(f"[vlm8b] encoder {'FROZEN (E1)' if freeze else 'TRAINABLE (E2+)'}")

    def on_train_begin(self, args, state, control, **kwargs):
        self._apply(self.freeze_epochs > 0)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._apply(int(state.epoch or 0) < self.freeze_epochs)


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
    """构建模型/数据/Trainer，跑 E1–E3 并保存最终 HF 权重。"""
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
    if args.freeze_encoder_epochs > 0:
        model.encoder.freeze()

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
        save_total_limit=2,
        eval_strategy="no",                 # forward 不返回 logits ⇒ 不用 prediction_step
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=[],
        deepspeed=args.deepspeed or None,
        max_steps=2 if args.smoke else -1,
        seed=SEED,
        data_seed=SEED,
        ddp_find_unused_parameters=False,
        save_only_model=True,               # 中途不存 optimizer
    )

    trainer = VLM8BudgetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EightBudgetCollator(tokenizer),
        param_groups=build_param_groups(
            model, args.lr_llm, args.lr_proj, args.lr_encoder,
            args.wd_llm, args.wd_other),
        callbacks=[
            EncoderFreezeCallback(model.encoder, args.freeze_encoder_epochs),
            BudgetCELogCallback(
                model, os.path.join(args.output_dir, "per_budget_ce.jsonl")),
        ],
    )

    print(f"[vlm8b] budgets={budgets} loss_weighting={args.loss_weighting} "
          f"epochs={args.epochs} freeze={args.freeze_encoder_epochs} "
          f"micro={args.micro_batch_size}×{args.grad_accum} "
          f"lr={args.lr_llm:g}/{args.lr_proj:g}/{args.lr_encoder:g}")

    trainer.train()
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
