"""
冻结的**兼容性**契约（CPU，可反复跑）
====================================

冻结机制好不好，不看「冻住了没有」，而看它有没有破坏既有约定。本文件把四条不变量
写成断言——任何一条挂了，冻结实验的结果就没法和历史 run / 别的实验臂比较：

  C1  **不改 state_dict**：任何冻结规格下参数名、形状完全一致
      ⇒ 阶段2 ckpt 的 `strict=True` 加载、`save_pretrained`、`--init-from` 全不受影响。
  C2  **默认路径零改动**：`--freeze-encoder none` 产出的 param groups 与旧实现逐组
      （lr / wd / 参数集合 / 顺序）完全一致 ⇒ 不冻结的 run 与历史 run 可逐位对齐。
  C3  **先冻后载**：已冻结的模型仍能 `load_state_dict`（warm start / 续训），
      且参数对象不被替换（`load_state_dict` 是 copy_，不是重建）。
  C4  **参数身份稳定**：冻结/解冻只改 `requires_grad`，不重建 Parameter
      （否则 optimizer 持有旧引用，解冻后静默不更新——上游事故的另一种形态）。

跑法：`python3 test_freeze_compat.py`（0=全 PASS）
"""
import sys

import torch
import torch.nn as nn

import freeze as F
from encoder import StandardAttentionEncoder
from model import VLM8Budget, VLM8BudgetConfig

DIM, DEPTH, HEADS, PATCH, IMG, NSPEC = 32, 3, 2, 8, 16, 4
SPECS = ["none", "all", "stem", "norm", "last=2", "blocks=0-1", "blocks=0-1,norm"]


class Checker:
    def __init__(self):
        self.ok = self.bad = 0

    def check(self, name, cond, detail=""):
        self.ok, self.bad = self.ok + bool(cond), self.bad + (not cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def summary(self):
        print(f"\nRESULT: {'PASS' if not self.bad else 'FAIL'}  "
              f"(passed={self.ok}, failed={self.bad})")
        return 0 if not self.bad else 1


class Wrapped(nn.Module):
    """模拟 `AssocVTCEncoder` 的 `.net` 包一层 + 一个下游头（检验子模块边界）。"""

    def __init__(self):
        super().__init__()
        self.net = StandardAttentionEncoder(dim=DIM, depth=DEPTH, heads=HEADS, patch=PATCH,
                                            img_size=IMG, num_specials=NSPEC)
        self.head = nn.Linear(DIM, 4)

    def forward(self, x):
        return self.head(self.net(x)[1].mean(dim=1))


# ═══════════════════════════════════════════════════════════════

def c1_state_dict_keys(c: Checker) -> None:
    """任何冻结规格都不改参数名/形状。"""
    base = None
    for spec in SPECS:
        model = Wrapped()
        F.freeze_encoder(model, spec)
        sd = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        if base is None:
            base, ref_spec = sd, spec
            c.check("C1 state_dict 非空且含 encoder+head", len(sd) > 10, f"{len(sd)} 项")
            continue
        c.check(f"C1 {spec} 与 {ref_spec} 的 state_dict 键/形状完全一致",
                sd == base, f"差异={len(set(sd.items()) ^ set(base.items()))}")

    # VLM8Budget 层（真实 wrapper 命名空间）也要一致
    from smoke import _FakeEncoder, _TinyLM
    keys = None
    for spec in ("none", "all"):
        model = VLM8Budget(VLM8BudgetConfig(encoder_ckpt="", qwen_dir=""),
                           encoder=_FakeEncoder(), lm=_TinyLM())
        model.freeze_encoder(spec)
        cur = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        if keys is None:
            keys = cur
        else:
            c.check(f"C1 VLM8Budget({spec}) state_dict 与 none 一致", cur == keys,
                    f"{len(cur)} 项")


def legacy_param_groups(model) -> list:
    """旧实现（commit f5c0028 的 build_param_groups）逐字复刻，作对照基线。"""
    groups = []
    for module, lr, wd in ((model.lm, 1e-5, 0.1),
                           (model.projector, 1e-4, 0.05),
                           (model.encoder, 5e-6, 0.05)):
        params = list(module.parameters())
        groups.append({"params": [p for p in params if p.ndim >= 2],
                       "lr": lr, "weight_decay": wd})
        groups.append({"params": [p for p in params if p.ndim < 2],
                       "lr": lr, "weight_decay": 0.0})
    return groups


def c2_default_path_identical(c: Checker) -> None:
    """`none` 时新旧 param groups 必须逐组一致（含顺序与 lr/wd）。"""
    from smoke import _FakeEncoder, _TinyLM
    model = VLM8Budget(VLM8BudgetConfig(encoder_ckpt="", qwen_dir=""),
                       encoder=_FakeEncoder(), lm=_TinyLM())
    model.freeze_encoder("none")
    new = F.build_param_groups(((model.lm, 1e-5, 0.1),
                                (model.projector, 1e-4, 0.05),
                                (model.encoder, 5e-6, 0.05)))
    old = legacy_param_groups(model)
    same = len(new) == len(old)
    detail = f"new={len(new)} old={len(old)}"
    if same:
        for i, (a, b) in enumerate(zip(new, old)):
            if (a["lr"], a["weight_decay"]) != (b["lr"], b["weight_decay"]):
                same = False
                detail = f"group{i} lr/wd 不同"
                break
            if [id(p) for p in a["params"]] != [id(p) for p in b["params"]]:
                same = False
                detail = f"group{i} 参数集合/顺序不同"
                break
    c.check("C2 --freeze-encoder none 与旧实现 param groups 逐组一致", same, detail)

    # 整段冻结：空组被丢弃，且剩下的组仍是旧结构的子集
    model.freeze_encoder("all")
    frozen = F.build_param_groups(((model.lm, 1e-5, 0.1),
                                   (model.projector, 1e-4, 0.05),
                                   (model.encoder, 5e-6, 0.05)))
    c.check("C2 整段冻结 → 组数 6→4（encoder 两组被丢弃，而不是空组）",
            len(frozen) == 4 and all(len(g["params"]) > 0 for g in frozen),
            f"{[len(g['params']) for g in frozen]}")


def c3_freeze_then_load(c: Checker) -> None:
    """先冻结再灌权重（warm start / 续训）必须可用，且真正生效。"""
    net = Wrapped()
    F.freeze_encoder(net, "all")
    src = Wrapped().state_dict()
    src["net.pos_embed"] = src["net.pos_embed"] + 0.5          # 造一个可辨识的差异
    missing, unexpected = net.load_state_dict(src, strict=True)
    delta = float((net.net.pos_embed.detach() - src["net.pos_embed"]).abs().max())
    c.check("C3 已冻结模型仍能 load_state_dict（strict=True）",
            not missing and not unexpected and delta == 0.0,
            f"max|Δ|={delta:.1e}")
    c.check("C3 灌入后参数仍是冻结状态", not net.net.pos_embed.requires_grad)


def c4_param_identity(c: Checker) -> None:
    """冻结/解冻只改 requires_grad，不重建 Parameter（optimizer 引用不能失效）。"""
    net = Wrapped()
    ids = {name: id(p) for name, p in net.named_parameters()}
    ptrs = {name: p.data_ptr() for name, p in net.named_parameters()}
    F.freeze_encoder(net, "all")
    mid_ids = {name: id(p) for name, p in net.named_parameters()}
    F.freeze_encoder(net, "none")
    after = {name: id(p) for name, p in net.named_parameters()}
    c.check("C4 冻结前后 Parameter 对象 id 不变", ids == mid_ids, "all")
    c.check("C4 解冻后 Parameter 对象 id 不变", ids == after, "none")
    c.check("C4 data_ptr 不变（没有重新分配存储）",
            ptrs == {name: p.data_ptr() for name, p in net.named_parameters()})


def c5_trained_weights_survive(c: Checker) -> None:
    """冻结后做一次 step：冻结参数不变、state_dict 可原样保存。"""
    net = Wrapped()
    F.freeze_encoder(net, "last=1")
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    opt = torch.optim.AdamW(F.build_param_groups([(net, 1e-2, 0.0)]), lr=1e-2)
    net(torch.randn(1, 3, IMG, IMG)).pow(2).mean().backward()
    opt.step()
    moved = {k: float((v.detach() - before[k]).abs().max())
             for k, v in net.state_dict().items()}
    frozen = [k for k in moved if k.startswith(("net.blocks.0.", "net.blocks.1.",
                                                "net.patch_embed.", "net.pos_embed"))]
    c.check("C5 一步 AdamW：冻结部分逐位不变", all(moved[k] == 0.0 for k in frozen),
            f"{len(frozen)} 个张量")
    c.check("C5 可训练部分在动", moved["head.weight"] > 0.0, f"Δ={moved['head.weight']:.1e}")
    # 保存/加载 round-trip（冻结不影响序列化）
    buf = {k: v.detach().clone() for k, v in net.state_dict().items()}
    fresh = Wrapped()
    fresh.load_state_dict(buf, strict=True)
    c.check("C5 state_dict round-trip 后冻结权重逐位一致",
            float((fresh.net.pos_embed.detach() - buf["net.pos_embed"]).abs().max()) == 0.0)


def main() -> int:
    print(f"vlm8b freeze compat  torch={torch.__version__}")
    c = Checker()
    c1_state_dict_keys(c)
    c2_default_path_identical(c)
    c3_freeze_then_load(c)
    c4_param_identity(c)
    c5_trained_weights_survive(c)
    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
