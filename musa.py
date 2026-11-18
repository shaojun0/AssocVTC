"""
国产算力适配层：摩尔线程 MUSA（MTT S4000）
=========================================

**目标**：同一份训练代码在 CUDA 与 MUSA 上都能跑，不靠「改源码 + 记命令」，而是
把差异收敛到这个文件的 `bootstrap()` 里。调用时机：`train.py:main()` 第一行。

本机实测（torch_musa 2.11 / 2×MTT S4000 / deepspeed 0.19.9）得到三条硬事实：

1. **`torch.cuda` 是 MUSA 的别名**：`torch.cuda.is_available()` 为 True、
   `get_device_name(0) == "MTT S4000"`，所以 transformers / accelerate / Trainer
   里写死的 `"cuda"` 大多不用改；但 `torch.version.cuda is None`，任何
   `if torch.version.cuda` 的分支都要小心。
2. **集合通信后端是 MCCL，不是 NCCL**：`torch.distributed.is_nccl_available()` 为
   False，`Backend.MCCL` 可用，`init_process_group("mccl")` 双卡 all_reduce 正常。
   gloo 虽在，但不支持 musa 张量（`ProcessGroupGloo::allreduce: unsupported
   device type musa`）。DeepSpeed 的 `CUDA_Accelerator` 把后端名写死成 `"nccl"`，
   必须在 engine 初始化前改成 `"mccl"`，否则 `init_process_group` 直接失败。
3. **SDPA 只有 math 后端**：本版 torch_musa 的 `SDPBackend` 里没有 FLASH /
   EFFICIENT / CUDNN，`F.scaled_dot_product_attention` 会走 math 实现。功能正确，
   但 448²/p32（229 token）这种小序列本来也吃不满 flash 的收益；要用 flash 得换
   MUSA 自带的 kernel 库（如 vllm-musa / flash-attn 的 MUSA 分支），不在本仓范围。
4. **fp64 是「假支持」**：`torch.cuda.get_device_properties` 与
   `is_fp64_supported()` 都报 True，但 `Tensor.norm()` 对 Double 直接
   `RuntimeError: Norm only supports Float/Half/BFloat16 input`。DeepSpeed 的
   `get_norm_dtype()` 正是靠 `is_fp64_supported()` 决定用 fp64 累加梯度范数
   （为了数值精度），于是 **ZeRO-3 + bf16 在 MUSA 上不开冻结也会崩**。把该探测
   改成 False 后，DS 退回 fp32 范数，ZeRO-2/3 全部正常（实测见 `MUSA_FREEZE.md`）。
"""
import os
from typing import Dict, List, Optional

import torch

_QUIET = os.environ.get("VLM8B_QUIET", "0") == "1"


# ═══════════════════════════════════════════════════════════════
# 设备探测
# ═══════════════════════════════════════════════════════════════

def is_musa() -> bool:
    """是否是摩尔线程 MUSA 后端（torch_musa 已加载且设备可用）。"""
    return hasattr(torch, "musa") and bool(torch.musa.is_available())


def device_type() -> str:
    """`"musa"` / `"cuda"` / `"cpu"`——给 `torch.autocast(device_type=...)` 用。"""
    if is_musa():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_name(index: int = 0) -> str:
    """`"musa:0"` / `"cuda:0"` / `"cpu"`。"""
    kind = device_type()
    return "cpu" if kind == "cpu" else f"{kind}:{index}"


def get_device(index: int = 0) -> torch.device:
    return torch.device(device_name(index))


def accelerator_name() -> str:
    """人类可读的加速卡名（日志用）。"""
    if device_type() == "cpu":
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return device_type()


def is_bf16_supported() -> bool:
    """MUSA 的 bf16 由 `torch.cuda.is_bf16_supported()` 一起报（实测 True）。"""
    if device_type() == "cpu":
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def empty_cache() -> None:
    if device_type() != "cpu":
        torch.cuda.empty_cache()


def synchronize() -> None:
    if device_type() != "cpu":
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════
# 分布式：MCCL vs NCCL
# ═══════════════════════════════════════════════════════════════

def comm_backend() -> str:
    """本机可用的集合通信后端名。MUSA 上是 `mccl`（gloo 不支持 musa 张量）。"""
    for name in ("mccl", "nccl", "gloo"):
        try:
            if torch.distributed.is_backend_available(name):
                return name
        except Exception:
            continue
    return "gloo"


def sdpa_backends() -> List[str]:
    """`torch.nn.attention.SDPBackend` 里实际存在的后端（名字小写）。

    注意：MUSA 这版枚举**不支持迭代**（`list(SDPBackend)` 为空），只能按属性名探测；
    实测只有 `MATH`，所以 `F.scaled_dot_product_attention` 走 math 实现。
    """
    try:
        enum = torch.nn.attention.SDPBackend
        return sorted(n.lower() for n in dir(enum) if n.isupper())
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
# DeepSpeed 补丁
# ═══════════════════════════════════════════════════════════════

def patch_deepspeed(verbose: bool = True) -> Dict[str, object]:
    """在 `deepspeed.initialize()` 之前修正两处「CUDA 假设」。

    幂等；deepspeed 没装时返回 `{"deepspeed": "not installed"}`。
    """
    try:
        from deepspeed.accelerator import get_accelerator
    except Exception as exc:                     # 没装 DS：单卡 torch 路径不需要
        return {"deepspeed": f"not installed ({type(exc).__name__})"}

    acc = get_accelerator()
    applied: Dict[str, object] = {}

    want = comm_backend()
    try:
        current = acc.communication_backend_name()
    except Exception:
        current = "?"
    if want in ("mccl", "nccl") and current != want:
        acc._communication_backend_name = want
        applied["comm_backend"] = f"{current} -> {want}"

    # fp64 假设：MUSA 报支持但 norm 不支持 → DS 的梯度范数会崩
    if is_musa():
        try:
            if acc.is_fp64_supported():
                acc.is_fp64_supported = lambda: False
                applied["fp64_norm"] = "double -> float32 (MUSA norm 不支持 fp64)"
        except Exception:
            pass

    if applied and verbose and not _QUIET:
        print(f"[musa] DeepSpeed 补丁: {applied}")
    return applied


def report() -> Dict[str, object]:
    """一次性打印/返回环境事实，便于写实验记录。"""
    info: Dict[str, object] = {
        "torch": torch.__version__,
        "device_type": device_type(),
        "accelerator": accelerator_name(),
        "device_count": torch.cuda.device_count() if device_type() != "cpu" else 0,
        "bf16": is_bf16_supported(),
        "sdpa_backends": sdpa_backends(),
        "comm_backend": comm_backend(),
        "torch.version.cuda": torch.version.cuda,
    }
    if is_musa():
        info["torch_musa"] = getattr(torch, "musa", None) and \
            __import__("torch_musa").__version__
    return info


def bootstrap(verbose: bool = True) -> Dict[str, object]:
    """训练入口调用：打一行环境、打 DS 补丁、返回 report。"""
    info = report()
    patch_deepspeed(verbose=verbose)
    if verbose and not _QUIET:
        print(f"[musa] {info['accelerator']} ×{info['device_count']}  "
              f"device_type={info['device_type']}  bf16={info['bf16']}  "
              f"comm={info['comm_backend']}  sdpa={info['sdpa_backends']}")
    return info
