"""
MUSA 端到端：真实 Trainer + DeepSpeed + 本仓冻结代码（2×MTT S4000）
=================================================================

跑的是**生产代码路径**，只把 Qwen3.5-4B 换成一个同维（2560）的微型替身：

- `AssocVTCEncoder` 从合成 ckpt 恢复真 `StandardAttentionEncoder`
  (dim=1744/depth=13/patch=32/448², 31.8M)，**不是**假模块；
- `VLM8BudgetTrainer` + `build_param_groups` + `freeze_encoder` 全部走本仓实现；
- DeepSpeed 用 `ds_zero2.json` / `ds_zero3.json`，`musa.bootstrap()` 打补丁。

四个断言（挂了就说明冻结在真实训练栈上不成立）：

  A. 静态 `all` 冻结：encoder 逐位不变，projector/LM 有更新
  B. 静态 `last=2`：blocks[0:11] 不变，blocks[11]/[12]/norm 有更新
  C. engine 建好后冻结：2 步内 encoder 不变
  D. 同一次运行里解冻：接下来 2 步 encoder 开始更新（E1 冻 / E2 解的关键路径）

用法：
    torchrun --nproc_per_node=2 musa_e2e.py --stage 2
    torchrun --nproc_per_node=2 musa_e2e.py --stage 3
"""
import argparse
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, default_data_collator

import musa

musa.bootstrap()

from freeze import freeze_encoder  # noqa: E402
from model import (ENC_DIM, IMAGE_SIZE, NUM_SPECIALS, PATCH_SIZE, VLM8Budget,
                   VLM8BudgetConfig)  # noqa: E402
from train import VLM8BudgetTrainer, build_param_groups  # noqa: E402

DIM, HEADS, DEPTH = ENC_DIM, 16, 3      # depth 只用契约允许的 3 层，够测 blocks 冻结
LM_HIDDEN, VOCAB, TEXT_LEN = 2560, 128, 24
ENC_LR, PROJ_LR, LM_LR = 1e-2, 1e-3, 1e-4


# ═══════════════════════════════════════════════════════════════
# 夹具：真 encoder + 微型 LM
# ═══════════════════════════════════════════════════════════════

def make_encoder_ckpt(path: str) -> str:
    """合成阶段2 ckpt（`cfg` 契约 + `encoder.*` 权重），让真实加载路径跑起来。

    **必须确定性**：每个 rank 都会调用本函数，随机种子不同会让各 rank 拿到不同的
    encoder 权重（DeepSpeed 启动时会从 rank0 广播，rank1 的基线随之漂移），
    冻结是否真的成立就无法判断了。
    """
    from encoder import StandardAttentionEncoder
    state = torch.get_rng_state()
    torch.manual_seed(1234)
    try:
        net = StandardAttentionEncoder(dim=DIM, depth=DEPTH, heads=HEADS, patch=PATCH_SIZE,
                                       img_size=IMAGE_SIZE, num_specials=NUM_SPECIALS)
    finally:
        torch.set_rng_state(state)
    cfg = {"enc_dim": DIM, "enc_depth": DEPTH, "enc_heads": HEADS, "patch": PATCH_SIZE,
           "img_size": IMAGE_SIZE, "num_specials": NUM_SPECIALS}
    torch.save({"cfg": cfg,
                "model": {f"encoder.{k}": v for k, v in net.state_dict().items()}},
               path)
    return path


class _Backbone(nn.Module):
    """一维 causal 自注意力骨架（只为给 Trainer 提供 hidden）。"""

    def __init__(self, dim=LM_HIDDEN, heads=8):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads

    def forward(self, inputs_embeds, attention_mask=None, use_cache=False):
        b, n, d = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = inputs_embeds.new_ones((b, n), dtype=torch.long)
        q, k, v = self.qkv(inputs_embeds).reshape(
            b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5
        causal = torch.ones(n, n, dtype=torch.bool, device=inputs_embeds.device).tril()
        scores = scores.masked_fill(~(causal & attention_mask.bool()[:, None, None, :]), -1e9)
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=inputs_embeds + self.proj(out))


class _TinyLM(nn.Module):
    """`get_input_embeddings` / `model` / `lm_head` 三件套替身（同 2560 维）。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, LM_HIDDEN)
        self.model = _Backbone()
        self.lm_head = nn.Linear(LM_HIDDEN, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.embed


class _Rows(Dataset):
    def __init__(self, n=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            ids = torch.randint(1, VOCAB, (TEXT_LEN,), generator=g)
            labels = torch.full((TEXT_LEN,), -100, dtype=torch.long)
            labels[-4:] = ids[-4:]
            self.items.append({
                "pixel_values": torch.randn(3, IMAGE_SIZE, IMAGE_SIZE, generator=g),
                "input_ids": ids,
                "attention_mask": torch.ones(TEXT_LEN, dtype=torch.long),
                "labels": labels,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ═══════════════════════════════════════════════════════════════
# 断言工具
# ═══════════════════════════════════════════════════════════════

def snapshot(model: VLM8Budget) -> dict:
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()}


def diff(model: VLM8Budget, before: dict) -> dict:
    out = {}
    for name, p in model.named_parameters():
        a = p.detach().float().cpu()
        out[name] = float((a - before[name]).abs().max())
    return out


def drift_vs_ckpt(model: VLM8Budget, ckpt: str, prefixes=None):
    """encoder 参数与 ckpt 的 bf16 参考值比较 → (最大偏差, 参数名)。

    训练精度是 bf16，所以参考值也先转 bf16；没被动过的参数应当**逐位相等**（0.0）。
    """
    state = torch.load(ckpt, map_location="cpu")["model"]
    worst, worst_name = 0.0, ""
    for name, p in model.named_parameters():
        if not name.startswith("encoder.net."):
            continue
        if prefixes and not any(name.startswith(x) for x in prefixes):
            continue
        key = "encoder." + name[len("encoder.net."):]
        ref = state[key].bfloat16().float()
        d = float((p.detach().float().cpu() - ref).abs().max())
        if d > worst:
            worst, worst_name = d, name
    return worst, worst_name


class Result:
    def __init__(self):
        self.ok = self.bad = 0

    def check(self, name, cond, detail=""):
        self.ok, self.bad = self.ok + bool(cond), self.bad + (not cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)

    def summary(self):
        print(f"\nE2E RESULT: {'PASS' if not self.bad else 'FAIL'} "
              f"(passed={self.ok}, failed={self.bad})", flush=True)
        return 0 if not self.bad else 1


def build_trainer(ckpt: str, out: str, ds_json: str, spec: str, steps: int = 2):
    model = VLM8Budget(VLM8BudgetConfig(
        encoder_ckpt=ckpt, qwen_dir="", budget_ids=list(range(1, 9)),
        pad_token_id=0, eos_token_id=0), encoder=None, lm=_TinyLM())
    report = freeze_encoder(model.encoder, spec)
    groups = build_param_groups(model, LM_LR, PROJ_LR, ENC_LR, 0.1, 0.05)
    args = TrainingArguments(
        output_dir=out, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, max_steps=steps,
        learning_rate=1e-4, lr_scheduler_type="constant", warmup_steps=0,
        bf16=True, logging_steps=1,
        save_strategy="no", eval_strategy="no", report_to=[],
        remove_unused_columns=False, dataloader_num_workers=0,
        deepspeed=ds_json, ddp_find_unused_parameters=False,
        save_only_model=True, seed=0,
    )
    trainer = VLM8BudgetTrainer(model=model, args=args, train_dataset=_Rows(),
                                data_collator=default_data_collator,
                                param_groups=groups)
    return model, trainer, report, groups


def manual_steps(trainer, steps: int, tag: str = "") -> None:
    """绕过 Trainer loop 直接推 engine：验证「训练中途改冻结状态」的路径。

    不用 `trainer.get_train_dataloader()`：它的 loader 已被 accelerate 包装/消费过，
    二次迭代在部分版本上拿不到 batch（这里实测为空循环）。直接建一个干净 loader。
    """
    engine = trainer.model_wrapped if hasattr(trainer, "model_wrapped") else trainer.model
    loader = DataLoader(trainer.train_dataset,
                        batch_size=trainer.args.per_device_train_batch_size,
                        collate_fn=default_data_collator)
    start = int(getattr(engine, "global_steps", 0))
    for i, batch in enumerate(loader):
        if i >= steps:
            break
        inputs = trainer._prepare_inputs(batch)
        loss = trainer.compute_loss(engine, inputs)
        engine.backward(loss)
        engine.step()
        engine.zero_grad()
        print(f"[{tag}] manual step {i} loss={float(loss.detach()):.4f}", flush=True)
    print(f"[{tag}] global_steps {start} -> {int(getattr(engine, 'global_steps', 0))}",
          flush=True)
    engine.train()


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2, choices=[2, 3])
    ap.add_argument("--steps", type=int, default=2)
    a = ap.parse_args()

    ds_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"ds_zero{a.stage}.json")
    tmp = tempfile.mkdtemp(prefix="vlm8b_e2e_")
    ckpt = make_encoder_ckpt(os.path.join(tmp, "enc.pt"))
    res = Result()
    print(f"===== stage={a.stage} ds={ds_json} ckpt={ckpt}", flush=True)

    # ---- ZeRO-3：与本 encoder 的裸 Parameter 不兼容 → 必须 fail-fast ----
    if a.stage >= 3:
        from train import assert_zero3_freeze_compatible
        for active in (False, True):
            try:
                assert_zero3_freeze_compatible(3, active)
                res.check(f"stage3(freeze={active}) → fail-fast", False, "没有抛错")
            except RuntimeError as exc:
                res.check(f"stage3(freeze={active}) → fail-fast（提示改用 ZeRO-2）", True,
                          str(exc)[:50])
        return res.summary()

    # ---- A: 静态全冻 ----
    model, trainer, report, groups = build_trainer(ckpt, os.path.join(tmp, "a"), ds_json,
                                                   "all", a.steps)
    print(f"[A] {report.summary()}", flush=True)
    before = snapshot(model)
    trainer.train()

    # 训练精度是 bf16 ⇒ 与 ckpt 的 bf16 参考值比；没冻住的参数必然偏
    enc, worst_name = drift_vs_ckpt(model, ckpt)
    proj = diff(model, before)["projector.token_proj.weight"]
    res.check(f"stage{a.stage} A: encoder 全冻 → 与 ckpt 逐位一致", enc == 0.0,
              f"maxΔ={enc:.2e} @{worst_name}")
    res.check(f"stage{a.stage} A: projector 仍在更新", proj > 0.0, f"maxΔ={proj:.2e}")

    # ---- B: 静态 last=2 ----
    model, trainer, report, groups = build_trainer(ckpt, os.path.join(tmp, "b"), ds_json,
                                                   "last=2", a.steps)
    print(f"[B] {report.summary()}", flush=True)
    before = snapshot(model)
    trainer.train()
    keep_from = DEPTH - 2                      # last=2 ⇒ 只有最后 2 个 block + norm 可训
    early, early_name = drift_vs_ckpt(
        model, ckpt, [f"encoder.net.blocks.{i}." for i in range(keep_from)]
        + ["encoder.net.patch_embed.", "encoder.net.pos_embed", "encoder.net.cls_token",
           "encoder.net.special_tokens"])
    late = max(diff(model, before)[
        f"encoder.net.blocks.{i}.attn.qkv.weight"] for i in range(keep_from, DEPTH))
    res.check(f"stage{a.stage} B: blocks[0:{keep_from}]+stem 冻结 → 与 ckpt 逐位一致",
              early == 0.0, f"maxΔ={early:.2e} @{early_name}")
    res.check(f"stage{a.stage} B: blocks[{keep_from}:] 在更新", late > 0.0,
              f"maxΔ={late:.2e}")

    # ---- C/D: engine 建好后冻结 → 解冻 ----
    model, trainer, report, groups = build_trainer(ckpt, os.path.join(tmp, "c"), ds_json,
                                                   "none", a.steps)
    trainer.train()
    freeze_encoder(model.encoder, "all")
    # ZeRO 的扁平梯度分区不会因 requires_grad=False 自动清零：切换冻结后必须清一次，
    # 否则下一步会用上一轮的 stale grad 再推一次（真实训练里由 EncoderFreezeCallback 做）。
    trainer.model_wrapped.zero_grad()
    before = snapshot(model)
    manual_steps(trainer, a.steps, tag="C")
    d = diff(model, before)
    enc = max(v for k, v in d.items() if k.startswith("encoder."))
    proj = d["projector.token_proj.weight"]
    # 运行时冻结**做不到**逐位不变：ZeRO-2 把每组参数拍平成一条 fp32 主权重，底层
    # AdamW 看到的是稠密 grad，冻结只让新梯度为 0，**上一阶段的 exp_avg/exp_avg_sq
    # 仍按 (β1/√β2)^t ≈ 0.91^t 衰减着推动参数**，每步 ≤ lr（权重衰减另算）。
    # 生产 lr=5e-6 时这个量级被 bf16 精度（ulp≈6e-5）完全掩盖，所以上游 A800 上
    # 看到的是 Δw=0.0；这里故意把 encoder lr 放大到 1e-2 才暴露出来。
    res.check(f"stage{a.stage} C: engine 建好后冻结 → 只剩动量残留（≤ 3·lr）",
              enc <= 3 * ENC_LR, f"maxΔ={enc:.2e} lr={ENC_LR:g}")
    res.check(f"stage{a.stage} C: 冻结期间 projector 仍更新", proj > 0.0, f"maxΔ={proj:.2e}")

    freeze_encoder(model.encoder, "none")
    trainer.model_wrapped.zero_grad()
    before = snapshot(model)
    manual_steps(trainer, a.steps, tag="D")
    d = diff(model, before)
    enc = max(v for k, v in d.items() if k.startswith("encoder."))
    res.check(f"stage{a.stage} D: 解冻后 encoder 恢复更新", enc > 2.5e-4, f"maxΔ={enc:.2e}")
    return res.summary()


if __name__ == "__main__":
    sys.exit(main())
