#!/usr/bin/env python3
"""check_platform.py — 一条命令判定「这台机器能不能跑本仓库的训练」。

为什么需要它：`srdiff_runtime.py` 把设备差异收敛成了一处代码路径，但**代码路径
就位 ≠ 这台机器能跑**。未实测平台（海光 DCU / 昇腾 NPU / Intel XPU）上真正会卡人的
是环境层的事实：ROCm/DTK 版本、bf16 有没有、SDPA 能不能用、RCCL/HCCL 通信能不能起、
显存统计 API 在不在。本脚本把这些逐项**真跑一遍**并以 PASS/FAIL/WARN 列表输出，
不需要训练数据、不需要权重、不写任何文件。

用法::

    python tools/check_platform.py                 # 自动探测
    python tools/check_platform.py --device_type dcu
    python tools/check_platform.py --dino_dir /root/autodl-tmp/models/dinov2-large  # 加测模型加载
    python tools/check_platform.py --json out.json

判据：所有 **FAIL** 都必须解决才能开训；WARN 是「能跑但有取舍」（例如 bf16 不可用
⇒ 必须 `--precision fp32`）。退出码 = FAIL 数量（0 = 全绿）。
"""
import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))

RESULTS = []


def check(name, fn, level="FAIL"):
    """level=FAIL：失败即计入退出码；level=WARN：失败只提示。"""
    try:
        detail = fn()
        RESULTS.append((name, "PASS", detail or "", level))
    except Exception as e:
        RESULTS.append((name, level, f"{type(e).__name__}: {e}".split("\n")[0][:200], level))
    return RESULTS[-1][1] == "PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device_type", default=os.environ.get("SRDIFF_DEVICE", "auto"))
    ap.add_argument("--dino_dir", default=os.environ.get("SRDIFF_DINO_DIR", ""))
    ap.add_argument("--json", default="")
    ap.add_argument("--skip_dist", action="store_true", help="跳过分布式后端自检")
    args = ap.parse_args()

    from srdiff_runtime import Runtime, is_rocm, detect_device_type

    print("=" * 78)
    print("check_platform — SR-Diffusion-v3 平台自检")
    print("=" * 78)

    # 1) 解释器 / torch
    check("import torch", lambda: __import__("torch").__version__)
    import torch

    # 2) 设备探测（本仓库的唯一入口）
    rt = Runtime(args.device_type)
    info = rt.describe()
    print(f"\n[探测] requested={info['requested']} → device_type={info['device_type']} "
          f"| torch={info['torch']} | rocm_hip={info['rocm_hip']} | cuda={info['cuda']}")
    print(f"[探测] device={info['device_name']} count={info['device_count']} "
          f"backend={info['dist_backend']} attn(auto)={info['attn_implementation_auto']}")
    if info["is_dcu_alias"] and not is_rocm():
        RESULTS.append(("dcu alias 与实际平台一致", "WARN",
                        "指定了 dcu 但 torch.version.hip 为空（装的不是 DCU 版 torch？）", "WARN"))

    def _alias():
        assert detect_device_type("dcu") == "cuda", "dcu 未解析成 cuda"
        return "dcu → cuda ✓"

    check("dcu 别名解析", _alias, level="WARN")

    # 3) 加速器可用性 + 显存 / 名字 API（srdiff_runtime 依赖这几个）
    def _accel():
        if rt.device_type == "cpu":
            return "cpu（只能冒烟）"
        assert rt.device_count() >= 1, "device_count()=0"
        return f"{rt.device_name()} × {rt.device_count()}"

    check("加速器可见", _accel)

    def _mem():
        if rt.device_type == "cpu":
            return "n/a"
        rt.reset_peak_memory()
        d = rt.device()
        x = torch.zeros(8 * 1024 * 1024, device=d)          # 32 MB
        peak = rt.peak_memory_gib()
        del x
        rt.empty_cache()
        assert peak > 0, f"peak_memory_gib()={peak}（显存统计 API 不可用）"
        return f"peak_memory_gib()={peak:.3f} GiB ✓"

    check("显存统计 API", _mem)

    def _sync():
        rt.synchronize()
        return "ok"

    check("synchronize()", _sync, level="WARN")

    # 4) bf16 / fp16（决定 --precision 能不能用 bf16）
    def _bf16():
        if rt.device_type == "cpu":
            return "cpu：不支持（--precision 会被强制 fp32）"
        d = rt.device()
        a = torch.randn(512, 512, device=d, dtype=torch.bfloat16)
        b = (a @ a).float().abs().mean().item()
        assert b == b, "bf16 结果为 NaN"
        note = "" if rt.supports_bf16() else "  ⚠️ supports_bf16()=False"
        return f"bf16 矩阵乘 OK (mean|.|={b:.3f}){note}"

    check("bf16 计算", _bf16, level="WARN")

    def _amp():
        if rt.device_type == "cpu":
            return "cpu：amp_context 退化为空上下文"
        with rt.amp_context("bf16"):
            y = torch.randn(64, 64, device=rt.device()) @ torch.randn(64, 64, device=rt.device())
        assert y.dtype == torch.bfloat16, f"autocast 未生效，dtype={y.dtype}"
        return f"torch.autocast({rt.device_type!r}, bf16) 生效 ✓"

    check("设备无关 autocast", _amp, level="WARN")

    # 5) SDPA / eager 注意力（DINOv2 会走这条路径）
    def _sdpa():
        d = rt.device()
        q = torch.randn(2, 4, 128, 64, device=d)
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        assert out.shape == q.shape
        extra = ""
        if rt.device_type == "cuda":
            # 注意：sdpa_kernel() 上下文只会「优先」选后端，后端不可用时不报错，
            # 所以别用它判断可用性 —— 直接问后端的可用性谓词。
            try:
                fa = bool(torch.backends.cuda.is_flash_attention_available())
                extra = (f"（flash_attn 后端可用={fa}"
                         + ("；DCU/ROCm 上通常为 False，SDPA 会退 math，属预期，"
                            "必要时用 --attn_implementation eager）" if rt.is_rocm else "")
                         + "）")
            except Exception:
                pass
        return f"SDPA forward OK{extra}"

    check("SDPA 注意力", _sdpa)

    # 6) 梯度检查点（编码器/解码器都开了 use_reentrant=False）
    def _ckpt():
        lin = torch.nn.Linear(64, 64).to(rt.device())
        x = torch.randn(4, 64, device=rt.device(), requires_grad=True)
        y = torch.utils.checkpoint.checkpoint(lin, x, use_reentrant=False)
        y.sum().backward()
        assert x.grad is not None
        return "torch.utils.checkpoint(use_reentrant=False) 前反向 OK ✓"

    check("梯度检查点", _ckpt)

    # 7) 分布式后端（nccl / hccl / ccl / gloo）
    if not args.skip_dist:
        def _dist():
            import torch.distributed as dist
            be = rt.dist_backend()
            if not dist.is_available():
                raise RuntimeError("torch.distributed 不可用")
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29577")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            if rt.device_type != "cpu":
                rt.set_device(0)
            dist.init_process_group(be, rank=0, world_size=1)
            t = torch.ones(4, device=rt.device()) * 2
            dist.all_reduce(t)
            dist.destroy_process_group()
            assert float(t[0]) == 2.0
            return f"init_process_group({be!r}) + all_reduce OK ✓"

        check(f"分布式后端 {rt.dist_backend()}", _dist)

    # 8) 模型加载（可选：给了 dino_dir 才测，验证 DINOv2 + 注意力实现真能建起来）
    if args.dino_dir:
        def _dino():
            from transformers.models.dinov2 import Dinov2Model
            impl = rt.attn_implementation("auto")
            m = Dinov2Model.from_pretrained(args.dino_dir, local_files_only=True,
                                            attn_implementation=impl)
            m = m.to(rt.device()).eval()
            x = torch.randn(1, 3, 224, 224, device=rt.device())
            with torch.no_grad(), rt.amp_context("bf16"):
                o = m(x).last_hidden_state
            n = sum(p.numel() for p in m.parameters()) / 1e6
            return (f"Dinov2Model(attn={impl}) 前向 OK, hidden={tuple(o.shape)}, "
                    f"{n:.1f} M 参数")

        check(f"DINOv2 前向 (attn={rt.attn_implementation('auto')})", _dino)

    # ── 汇总 ──
    print("\n" + "=" * 78)
    print(f"{'检查项':<34}{'结果':<7}说明")
    print("-" * 78)
    fails = 0
    for name, status, detail, level in RESULTS:
        if status != "PASS" and level == "FAIL":
            fails += 1
        print(f"{name:<34}{status:<7}{detail}")
    print("-" * 78)

    if fails == 0:
        print("结论：全部硬性检查通过 ✅")
        if rt.device_type == "cpu":
            print("      ⚠️ 但这是 CPU 机器 —— 可以跑自检，不能训练 DINOv2-large。")
    else:
        print(f"结论：{fails} 项硬性检查 FAIL ❌ —— 先解决再开训。")
    if not args.dino_dir:
        print("提示：加 --dino_dir <权重目录> 可加测 DINOv2 前向；"
              "L3 端到端冒烟见 python train_multi.py --smoke --limit 32 --max_steps 3")
    print("=" * 78)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"runtime": info,
                       "results": [{"name": n, "status": s, "detail": d}
                                   for n, s, d, _ in RESULTS]}, f,
                      indent=2, ensure_ascii=False)
        print(f"[json] {args.json}")
    return fails


if __name__ == "__main__":
    try:
        sys.exit(min(main(), 120))
    except Exception:
        traceback.print_exc()
        sys.exit(121)
