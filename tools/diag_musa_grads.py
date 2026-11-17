"""MUSA 数值诊断: 同一批数据/权重下 bf16-autocast 与 fp32 的梯度对比 + 分组范数。

用法（远端 /root/AssocVTC_tok448）:
    MUSA_VISIBLE_DEVICES=0 python tools/diag_musa_grads.py --ckpt <ckpt> --n 24 --bs 16 [--math 1]

动机（2026-11-17）: 224² 的 CUDA 参照配置（K=32, lr 2e-4）在 CUDA 上 45ep 稳定
(max gn 9.3, 0 skip)，同一配置在 MUSA 上 ep2 起梯度爆到 1e2~1e4。本脚本检查
「同一权重/同一 batch，bf16-autocast 前反向 vs fp32 前反向」的梯度差异，
以及分组（enc / zs / dec_mem / dec_stack / dec_head）范数，用来定位是否某个
MUSA kernel 的反向在 bf16 下产生了异常梯度。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v2_linattn import LinAttnConfig, SRPhase1LinAttn  # noqa: E402
import data_linattn as D  # noqa: E402


def groups(model):
    g = {"enc": [], "zs": [], "dec_mem": [], "dec_st": [], "dec_head": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder"):
            g["enc"].append(p)
        elif name.startswith(("zs_proj", "zcls_proj")):
            g["zs"].append(p)
        elif name.startswith(("decoder.mem_proj", "decoder.mem_norm")):
            g["dec_mem"].append(p)
        elif name.startswith("decoder.pixel_head"):
            g["dec_head"].append(p)
        else:
            g["dec_st"].append(p)
    return g


def norms(model):
    out = {}
    for k, ps in groups(model).items():
        s = sum(float(p.grad.detach().pow(2).sum()) for p in ps if p.grad is not None)
        out[k] = s ** 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--math", type=int, default=0, help="1: 关闭 flash/mem-efficient SDPA 走 math")
    a = ap.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    annot = LinAttnConfig.__annotations__
    cfg = LinAttnConfig(**{k: v for k, v in ck["cfg"].items() if k in annot})
    model = SRPhase1LinAttn(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.train()
    if a.math:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[diag] SDPA backends: math only")

    tf = D.build_transform(cfg.img_size, train=True)
    items = D.scan_imagenet100("/root/autodl-tmp/datasets/imagenet100")
    tr, _va = D.split_index(items, val_per_class=50)
    ds = D.ImageNet100(tr, tf, img_size=cfg.img_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True,
                                     num_workers=4, drop_last=True)
    print("[diag] ckpt step={} epoch={} img={} patch={} K={}".format(
        ck.get("step"), ck.get("epoch"), cfg.img_size, cfg.patch, cfg.num_specials))

    worst = []
    for i, (x, _) in enumerate(dl):
        if i >= a.n:
            break
        x = x.to(dev)
        res = {}
        for tag, amp in (("bf16", True), ("fp32", False)):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(x)
                loss = out["loss"]
            loss.backward()
            res[tag] = (norms(model), float(loss))
        b, f = res["bf16"], res["fp32"]
        tot_b = sum(v * v for v in b[0].values()) ** 0.5
        tot_f = sum(v * v for v in f[0].values()) ** 0.5
        line = ("[b{:02d}] loss bf16={:.4f} fp32={:.4f} | gn bf16={:.3f} fp32={:.3f} "
                "ratio={:.2f} | ").format(i, b[1], f[1], tot_b, tot_f,
                                          tot_b / max(tot_f, 1e-12))
        line += " ".join("{}={:.3f}/{:.3f}".format(k, b[0][k], f[0][k]) for k in b[0])
        print(line, flush=True)
        worst.append((max(tot_b / max(tot_f, 1e-12), tot_f / max(tot_b, 1e-12)),
                      i, tot_b, tot_f))
        if i >= a.n - 1:
            break
    worst.sort(reverse=True)
    print("[diag] worst ratio batches:",
          [(round(w[0], 2), w[1], round(w[2], 2), round(w[3], 2)) for w in worst[:5]])


if __name__ == "__main__":
    main()
