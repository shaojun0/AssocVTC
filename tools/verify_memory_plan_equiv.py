#!/usr/bin/env python3
"""对照验证：`memory_plan` 重构（HEAD）与重构前实现（git 198aaa3）逐位等价。

为什么需要：`WholeClipDecoder.forward` 从布尔 `full_read` 改成 v2 同构的
`memory_plan`（每步读平坦 memory 的连续窗口）时，一个容易踩的坑是「把
`step_emb` 加到默认 plan 的 memory 上」——cross-attn 用同一个 memory 当 K 和 V，
key 侧的常数偏移会被 softmax 消掉，但 **V 侧会把它原样带进输出**，于是默认
（阶段 2 / 逐档评测）路径的输出会漂移。本脚本用旧代码做 bit-exact 对照：

    默认 plan、n_steps=1、n_steps=2、旧 full_read vs 新 full_memory_plan、
    显式默认 plan，以及默认路径的**梯度**（BPTT，两步）

用法（CPU 即可，需要仓库 git 历史里有 198aaa3）::

    python3 tools/verify_memory_plan_equiv.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_REF = "198aaa3"          # 重构前（full_read 布尔）的最后一个代码版本


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    tmp = tempfile.mkdtemp()
    old_path = os.path.join(tmp, "model_v10_old.py")
    src = subprocess.check_output(
        ["git", "-C", REPO, "show", f"{OLD_REF}:model_v10.py"])
    open(old_path, "wb").write(src)
    sys.path.insert(0, REPO)
    old = _load(old_path, "model_v10_old")
    new = _load(os.path.join(REPO, "model_v10.py"), "model_v10_new")

    tok_cfg = dict(img_size=32, patch=16, num_specials=4, enc_dim=32, enc_depth=1,
                   enc_heads=2, enc_mlp_ratio=4.0, enc_attn="std",
                   dec_dim=32, dec_depth=1, dec_heads=2, dec_mlp_ratio=4.0,
                   pixel_hidden=32, zs_dim=8, dec_recurrent=False,
                   dec_steps="1", dec_step_plan="square")
    kw = dict(tok_ckpt="", tok_cfg=tok_cfg, frames=8, group=4, rate_tokens=4,
              rate_dim=8, target_tokens=4, chunk_tokens=4, mid_depth=2,
              mid_heads=2, mid_ratio=2.0, mid_dim=32, dec_dim=32, dec_depth=2,
              dec_heads=2, dec_ratio=2.0)

    # ── 1) decoder 级：同权重，5 条路径逐位比
    torch.manual_seed(0)
    od = old.WholeClipDecoder(old.V10Config(**kw), code_dim=8, num_specials=4).eval()
    nd = new.WholeClipDecoder(new.V10Config(**kw), code_dim=8, num_specials=4).eval()
    nd.load_state_dict(od.state_dict(), strict=True)
    B, S, P, d = 2, kw["frames"] // kw["group"], (kw["rate_tokens"] * kw["group"]) // kw["chunk_tokens"], 32
    z = torch.randn(B, S, P, d, generator=torch.Generator().manual_seed(7))

    def cmp(tag, a, b):
        ok = (torch.equal(a["pred"], b["pred"])
              and torch.equal(a["preds"], b["preds"]))
        md = (a["preds"] - b["preds"]).abs().max().item()
        print(f"[decoder] {tag:42s} bit_exact={ok}  max|Δ|={md:.3e}")
        return ok

    all_ok = True
    with torch.no_grad():
        all_ok &= cmp("默认 plan（全 S 步）", od(z, return_step_preds=True),
                      nd(z, return_step_preds=True))
        all_ok &= cmp("n_steps=1", od(z, n_steps=1, return_step_preds=True),
                      nd(z, n_steps=1, return_step_preds=True))
        all_ok &= cmp("n_steps=2", od(z, n_steps=2, return_step_preds=True),
                      nd(z, n_steps=2, return_step_preds=True))
        all_ok &= cmp("旧 full_read vs 新 full_memory_plan()",
                      od(z, n_steps=1, full_read=True, return_step_preds=True),
                      nd(z, memory_plan=nd.full_memory_plan(),
                         return_step_preds=True))
        all_ok &= cmp("显式传入 default_memory_plan(2)", od(z, n_steps=2,
                                                           return_step_preds=True),
                      nd(z, memory_plan=nd.default_memory_plan(2),
                         return_step_preds=True))

    # ── 2) 默认路径梯度逐位（BPTT 两步）
    def grads(mod):
        mod.zero_grad(set_to_none=True)
        mod(z, return_step_preds=True)["preds"].pow(2).mean().backward()
        return {n: p.grad.detach().clone() for n, p in mod.named_parameters()
                if p.grad is not None}

    go, gn = grads(od), grads(nd)
    gok = (set(go) == set(gn)) and all(torch.equal(go[k], gn[k]) for k in go)
    print(f"[decoder] 默认路径梯度                       bit_exact={gok}")
    all_ok &= gok

    # ── 3) VideoV10 级（随机分词器，同 seed）：默认 / n_steps / 旧全读
    torch.manual_seed(0)
    om = old.VideoV10(old.V10Config(**kw)).eval()
    torch.manual_seed(0)
    nm = new.VideoV10(new.V10Config(**kw)).eval()
    nm.load_state_dict(om.state_dict(), strict=True)
    Fc, Kc, c = kw["frames"], 4, 8
    g2 = torch.Generator().manual_seed(3)
    zc = torch.randn(2, Fc, 1, c, generator=g2)
    zs = torch.randn(2, Fc, Kc, c, generator=g2)

    with torch.no_grad():
        a = om(x=None, z_cls=zc, z_s=zs)
        b = nm(x=None, z_cls=zc, z_s=zs)
        v_ok = torch.equal(a["pred"], b["pred"])
        print(f"[VideoV10] 默认路径                          bit_exact={v_ok}")
        all_ok &= v_ok
        a = om(x=None, z_cls=zc, z_s=zs, n_steps=1, full_read=True)
        b = nm(x=None, z_cls=zc, z_s=zs,
               memory_plan=nm.readout.full_memory_plan())
        v2_ok = torch.equal(a["pred"], b["pred"])
        print(f"[VideoV10] 旧全读 vs 新 full plan            bit_exact={v2_ok}")
        all_ok &= v2_ok

    # ── 4) 训练态切换
    nm.train()
    s_default = tuple(nm(x=None, z_cls=zc, z_s=zs)["step_l1"].shape)
    nm.train_memory_plan = nm.readout.full_memory_plan()
    s_full = tuple(nm(x=None, z_cls=zc, z_s=zs)["step_l1"].shape)
    nm.eval()
    s_eval = tuple(nm(x=None, z_cls=zc, z_s=zs)["step_l1"].shape)
    sw = (s_default == (nm.cfg.n_steps,) and s_full == (1,)
          and s_eval == (nm.cfg.n_steps,))
    print(f"[switch] train default={s_default} train full={s_full} "
          f"eval={s_eval}  ok={sw}")
    all_ok &= sw
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
