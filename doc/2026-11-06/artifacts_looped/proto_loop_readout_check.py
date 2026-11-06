"""循环读出（V1）缩小版自检 — 对应 doc/2026-11-06/DESIGN_looped_readout_vlm8b.md §4.9。
验证：形状/读窗口、前缀因果性、BPTT vs detach 的信用分配差异。
用法：python3 proto_loop_readout_check.py（CPU 即可，torch>=2.0）
"""
import torch, torch.nn as nn, math

def step_windows(num_specials, steps, block):
    out=[]
    for i,t in enumerate(steps):
        k=int(t); hi=min(k*block,num_specials); lo=0 if i==0 else (k-1)*block+1
        out.append((lo,hi))
    return out

class LoopVisualReadout(nn.Module):
    def __init__(s, enc_dim=1744, dim=256, depth=2, heads=8, mlp=4.0,
                 n_tokens=8, n_specials=32, lm_hidden=64, block=4):
        super().__init__()
        s.mem_proj = nn.Linear(enc_dim, dim) if enc_dim!=dim else nn.Identity()
        s.mem_norm = nn.LayerNorm(dim)
        s.mem_pos  = nn.Parameter(torch.zeros(1,1+n_specials,dim))
        s.q_bank   = nn.Parameter(torch.randn(n_tokens,dim)*0.02)
        s.stack    = nn.TransformerDecoder(nn.TransformerDecoderLayer(
                         dim,heads,int(dim*mlp),batch_first=True,norm_first=True), num_layers=depth)
        s.norm_out = nn.LayerNorm(dim); s.out_proj = nn.Linear(dim,lm_hidden)
        s.windows  = step_windows(n_specials, list(range(1,n_tokens+1)), block)
    def forward(s, z_cls, z_s, detach_carry=False):
        mem = s.mem_norm(s.mem_proj(torch.cat([z_cls,z_s],1))) + s.mem_pos
        outs,Y=[],None
        for t,(lo,hi) in enumerate(s.windows):
            q=s.q_bank[t].view(1,1,-1).expand(mem.size(0),-1,-1)
            q_in=q if Y is None else q+Y
            Y=s.stack(q_in, mem[:,lo:hi+1])
            outs.append(s.out_proj(s.norm_out(Y)))
            Y=Y.detach() if detach_carry else Y
        return torch.cat(outs,1)

torch.manual_seed(0)
B,K,D,H=2,32,1744,64
m=LoopVisualReadout(lm_hidden=H); m.eval()
zc=torch.randn(B,1,D); zs=torch.randn(B,K,D)
vis=m(zc,zs); print("shape", tuple(vis.shape), "windows", m.windows)

# 1) 前缀因果性
with torch.no_grad():
    v0=vis.clone()
    zs2=zs.clone(); zs2[:,4:8]+=5.0        # 步2 的窗口（z_s[4..7]）
    v1=m(zc,zs2)
    print("perturb z_s[4:8] -> |d vis|", [round(float((v1[:,i]-v0[:,i]).abs().max()),4) for i in range(8)])
    zs3=zs.clone(); zs3[:,20]+=5.0         # 步6 的窗口（z_s[20..23]）
    v2=m(zc,zs3)
    print("perturb z_s[20]  -> |d vis|", [round(float((v2[:,i]-v0[:,i]).abs().max()),4) for i in range(8)])

# 2) BPTT：末 token 的 loss 能回流到 step-1 的 query 与共享栈
m.train(); m.zero_grad()
vis=m(zc,zs); loss=vis[:,7].pow(2).mean(); loss.backward()
print("BPTT  grad q_bank[0] norm =", round(float(m.q_bank.grad[0].norm()),6),
      "| grad stack.L0 self_attn =", round(float(m.stack.layers[0].self_attn.in_proj_weight.grad.norm()),6))
# 3) detach 对照
m.zero_grad(); vis_d=m(zc,zs,detach_carry=True); vis_d[:,7].pow(2).mean().backward()
print("detach grad q_bank[0] norm =", round(float(m.q_bank.grad[0].norm()),6),
      "(应为 0；说明 BPTT 才给早步信用分配)")
