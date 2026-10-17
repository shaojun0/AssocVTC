import time, torch, sys
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import adaptive_tokenizers

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'alit_tiny'
model = adaptive_tokenizers.__dict__[MODEL](
    base_tokenizer_args={"id": "vqgan", "is_requires_grad": False},
    quantize_latent=True, factorize_latent=True,
    train_stage="latent_distillation_pretrain").cuda()
base = model.base_tokenizer
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"[{MODEL}] trainable={trainable/1e6:.1f}M total={total/1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = NativeScaler()

def bench(bs, iters=6, grad_iters=-1):
    x = torch.randn(bs, 3, 256, 256).cuda().clamp(-1, 1)
    for _ in range(2):
        with torch.cuda.amp.autocast():
            loss, _ = model(x, sample_grad_iters=grad_iters)
        scaler(loss, opt, clip_grad=3.0, parameters=model.parameters(), update_grad=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        with torch.cuda.amp.autocast():
            loss, _ = model(x, sample_grad_iters=grad_iters)
        scaler(loss, opt, clip_grad=3.0, parameters=model.parameters(), update_grad=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"bs={bs:>2} grad_iters={grad_iters:>2}: {dt*1000:6.0f} ms/step | {bs/dt:6.1f} img/s | peak {mem:5.1f} GiB", flush=True)
    del x; torch.cuda.empty_cache()

for bs in [4, 8, 12, 16, 20, 24]:
    try:
        bench(bs)
    except RuntimeError as e:
        print(f"bs={bs}: OOM/err {str(e)[:80]}", flush=True); torch.cuda.empty_cache(); break
# 只对部分迭代算 loss 的省法
bench(8, grad_iters=2)
bench(16, grad_iters=2)
