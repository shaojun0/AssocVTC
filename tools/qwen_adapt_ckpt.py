#!/usr/bin/env python3
"""把 AssocVTC 240M 档（t16 B 相）ckpt 的 decoder 权重适配成 Qwen token 探针的 init。

- 只取 ckpt['model'] 里 `decoder.*`（mem_proj 是 1248→1440，形状不符 ⇒ 丢弃，保持随机）
- mem_pos: ckpt 是 (1, 33, 1440)（z_cls + K=32）→ 探针 K=49 ⇒ (1, 50, 1440)：
  第 0 行（z_cls）直拷，1..32 行按线性插值铺成 1..49 行
"""
import json
import torch
import torch.nn.functional as F

SRC = "/root/autodl-tmp/qwen_vtc/probe/ckpt_m240_t16.pt"
DST = "/root/autodl-tmp/qwen_vtc/probe/init_m240_decoder.pt"

ck = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ck["model"] if "model" in ck else ck
out, dropped = {}, []
for k, v in sd.items():
    if not k.startswith("decoder."):
        continue
    kk = k[len("decoder."):]
    if kk.startswith("mem_proj"):
        dropped.append((k, tuple(v.shape)))
        continue
    if kk == "mem_pos":
        src = v[0]                                   # (33, 1440)
        cls_row, zs = src[:1], src[1:]               # (1,1440), (32,1440)
        zs_i = F.interpolate(zs.t().unsqueeze(0), size=49, mode="linear",
                             align_corners=False)[0].t()          # (49,1440)
        v = torch.cat([cls_row, zs_i], 0).unsqueeze(0)           # (1,50,1440)
    out["decoder." + kk] = v.clone()
info = {"src": SRC, "ckpt_step": ck.get("step"), "ckpt_best_psnr": ck.get("best_psnr"),
        "n_keys": len(out), "dropped": dropped,
        "keys": sorted(out.keys())[:6] + ["..."]}
torch.save({"model": out, "info": info}, DST)
print(json.dumps({k: info[k] for k in ("ckpt_step", "ckpt_best_psnr", "n_keys", "dropped")}, indent=1))
print("saved", DST)
