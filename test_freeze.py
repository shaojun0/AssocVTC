"""
冻结机制自检（纯 CPU，不加载 Qwen / DeepSpeed）
==============================================

覆盖 `freeze.py` 的语义契约——这些断言挂了，冻结实验的结果就不可信：

1. 规格解析：none/all/stem/blocks=a-b/last=N/组合，越界必须报错
2. 整段冻结：无参数可训练、forward 走 no_grad（输出不需梯度）
3. 部分冻结：冻结 block 的 grad 为 None；未冻 block + norm 有 grad
4. 优化器步：冻结参数逐位不变，可训练参数确实变小
5. param groups：不含冻结参数、无空组、全冻时报可读错误（DS 的 IndexError 前身）
6. `model.train()` 之后冻结子模块仍是 eval（Trainer 每 epoch 会覆盖）
7. 幂等：改规格再冻结不会残留上一次的 requires_grad

用法：`python3 test_freeze.py`（0=全 PASS）
"""
import sys

import torch
import torch.nn as nn

import freeze as F
from encoder import StandardAttentionEncoder

DIM, DEPTH, HEADS, PATCH, IMG, NSPEC = 32, 3, 2, 8, 16, 4


class Checker:
    def __init__(self):
        self.passed = self.failed = 0

    def check(self, name, cond, detail=""):
        self.passed, self.failed = self.passed + bool(cond), self.failed + (not cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def close(self, name, a, b, tol=0.0):
        delta = float((a.detach().float() - b.detach().float()).abs().max())
        self.check(name, delta <= tol, f"max|Δ|={delta:.2e} tol={tol:g}")

    def raises(self, name, fn, exc=Exception):
        try:
            fn()
        except exc as e:
            self.check(name, True, f"{type(e).__name__}: {str(e)[:60]}")
        except Exception as e:                     # 抛错类型不对
            self.check(name, False, f"期望 {exc.__name__}，得到 {type(e).__name__}")
        else:
            self.check(name, False, "没有抛错")

    def summary(self):
        print(f"\nRESULT: {'PASS' if not self.failed else 'FAIL'}  "
              f"(passed={self.passed}, failed={self.failed})")
        return 0 if not self.failed else 1


def enc(depth: int = DEPTH) -> StandardAttentionEncoder:
    return StandardAttentionEncoder(dim=DIM, depth=depth, heads=HEADS, patch=PATCH,
                                    img_size=IMG, num_specials=NSPEC)


def grads(net: nn.Module) -> dict:
    return {name: None if p.grad is None else float(p.grad.abs().max())
            for name, p in net.named_parameters()}


def backward(net: nn.Module, depth: int = DEPTH) -> None:
    """对 z_s 造一个二次 loss 反传（不依赖 LM）。"""
    net.zero_grad(set_to_none=True)
    x = torch.randn(2, 3, IMG, IMG)
    z_s = net(x)[1]
    (z_s.float() ** 2).mean().backward()


# ═══════════════════════════════════════════════════════════════

def test_parse(c: Checker) -> None:
    p = F.parse_spec("none", DEPTH)
    c.check("none → 不冻结", not p.active)
    p = F.parse_spec("all", DEPTH)
    c.check("all → 全冻", p.all_ and p.active)
    p = F.parse_spec("blocks=0-1", DEPTH)
    c.check("blocks=0-1 → {0,1}", p.blocks == {0, 1}, str(sorted(p.blocks)))
    p = F.parse_spec("blocks=0,2", DEPTH)
    c.check("blocks=0,2 → {0,2}", p.blocks == {0, 2})
    p = F.parse_spec("last=1", DEPTH)
    c.check("last=1 → 冻 stem + blocks[0:2]，norm 可训",
            p.stem and p.blocks == {0, 1} and not p.norm, str(sorted(p.blocks)))
    p = F.parse_spec("stem,norm", DEPTH)
    c.check("组合 stem,norm", p.stem and p.norm)
    c.raises("blocks 越界报错", lambda: F.parse_spec("blocks=0-9", DEPTH), ValueError)
    c.raises("未知 token 报错", lambda: F.parse_spec("half", DEPTH), ValueError)
    c.raises("last 越界报错", lambda: F.parse_spec("last=9", DEPTH), ValueError)


def test_freeze_all(c: Checker) -> None:
    net = enc().eval()
    report = F.freeze_encoder(net, "all")
    total = sum(p.numel() for p in net.parameters())
    c.check("all: 全部参数被冻", report.frozen_numel == total and report.trainable_numel == 0,
            report.summary())
    with torch.no_grad():
        z_s = net(torch.randn(1, 3, IMG, IMG))[1]
    c.check("冻结后前向输出不需要梯度（外层 no_grad 下）", not z_s.requires_grad)


def test_partial(c: Checker) -> None:
    net = enc().train()
    F.freeze_encoder(net, "blocks=0-0")
    backward(net)
    g = grads(net)
    frozen = [k for k in g if k.startswith("blocks.0.")]
    trainable_blocks = [k for k in g if k.startswith("blocks.1.") or k.startswith("blocks.2.")]
    c.check("blocks=0-0: block0 的 grad 全为 None", all(g[k] is None for k in frozen),
            f"{len(frozen)} 个张量")
    c.check("blocks=0-0: block1/2 有 grad", all(g[k] is not None for k in trainable_blocks))
    c.check("blocks=0-0: stem 仍有 grad",
            all(g[k] is not None for k in g if k.startswith(("patch_embed", "pos_embed"))))


def test_last(c: Checker) -> None:
    net = enc().train()
    F.freeze_encoder(net, "last=1")
    backward(net)
    g = grads(net)
    c.check("last=1: stem 冻结（pos_embed.grad is None）", g["pos_embed"] is None)
    c.check("last=1: 前 2 个 block 冻结",
            g["blocks.0.attn.qkv.weight"] is None and g["blocks.1.attn.qkv.weight"] is None)
    c.check("last=1: 末 block + norm 可训",
            g["blocks.2.attn.qkv.weight"] is not None and g["norm.weight"] is not None)


def test_optimizer_step(c: Checker) -> None:
    net = enc().train()
    F.freeze_encoder(net, "blocks=0-1")
    params = {k: v.detach().clone() for k, v in net.named_parameters()}
    groups = F.build_param_groups([(net, 1e-2, 0.0)])
    opt = torch.optim.AdamW(groups, lr=1e-2)
    backward(net)
    opt.step()
    moved = {k: float((v.detach() - params[k]).abs().max()) for k, v in net.named_parameters()}
    frozen = [k for k in moved if k.startswith(("blocks.0.", "blocks.1."))]
    c.check("一步 AdamW 后：冻结 block 逐位不变",
            all(moved[k] == 0.0 for k in frozen), f"{len(frozen)} 个张量")
    c.check("一步 AdamW 后：末 block / norm / stem 有更新",
            moved["blocks.2.attn.qkv.weight"] > 0 and moved["norm.weight"] > 0
            and moved["patch_embed.weight"] > 0,
            f"Δ={moved['blocks.2.attn.qkv.weight']:.2e}")


def test_param_groups(c: Checker) -> None:
    net = enc()
    F.freeze_encoder(net, "all")
    try:
        F.build_param_groups([(net, 1e-3, 0.0)])
        c.check("全冻 + 只有 encoder 组 → 报错", False)
    except RuntimeError as e:
        c.check("全冻 + 只有 encoder 组 → 报错", True, str(e)[:40])

    net = enc()
    F.freeze_encoder(net, "blocks=0-1")
    groups = F.build_param_groups([(net, 1e-3, 0.0)])
    in_groups = {id(p) for g in groups for p in g["params"]}
    frozen_ids = {id(p) for p in net.parameters() if not p.requires_grad}
    c.check("param groups 不含任何冻结参数", not (in_groups & frozen_ids),
            f"{len(frozen_ids)} 个冻结")
    c.check("param groups 无空组", all(len(g["params"]) > 0 for g in groups),
            f"{len(groups)} 组")


def test_model_train_mode(c: Checker) -> None:
    """Trainer 会调 model.train()，冻结段必须留在 eval。"""
    from model import VLM8Budget, VLM8BudgetConfig
    from smoke import _FakeEncoder, _TinyLM

    model = VLM8Budget(VLM8BudgetConfig(encoder_ckpt="", qwen_dir=""),
                       encoder=_FakeEncoder(), lm=_TinyLM())
    report = model.freeze_encoder("all")
    model.train()
    c.check("freeze(all) 后 model.train() 仍把 encoder 留在 eval",
            model.encoder.training is False, report.summary())
    groups = __import__("train").build_param_groups(model, 1e-5, 1e-4, 5e-6, 0.1, 0.05)
    names = {id(p) for g in groups for p in g["params"]}
    enc_ids = {id(p) for p in model.encoder.parameters()}
    c.check("整段冻结时 param groups 不含 encoder（DeepSpeed 不会见到空组）",
            not (names & enc_ids), f"{len(groups)} 组")
    model.freeze_encoder("none")
    model.train()
    c.check("解冻 + train() → encoder 恢复 train", model.encoder.training is True)


def test_idempotent(c: Checker) -> None:
    net = enc()
    F.freeze_encoder(net, "all")
    F.freeze_encoder(net, "last=1")
    g = grads(net)
    del g
    c.check("先 all 再 last=1：norm 可训（不会残留 all）",
            net.norm.weight.requires_grad and not net.pos_embed.requires_grad)
    F.freeze_encoder(net, "none")
    c.check("切回 none：全部参数恢复可训",
            all(p.requires_grad for p in net.parameters()))


def main() -> int:
    print(f"vlm8b freeze selftest  torch={torch.__version__}")
    c = Checker()
    test_parse(c)
    test_freeze_all(c)
    test_partial(c)
    test_last(c)
    test_optimizer_step(c)
    test_param_groups(c)
    test_model_train_mode(c)
    test_idempotent(c)
    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
