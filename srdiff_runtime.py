"""srdiff_runtime — 设备 / 分布式后端无关的运行时层（2026-10-14）

为什么有这一层（用户 2026-10-14 拍板）
------------------------------------
旧的 `train_multi.py` 把设备写死成 CUDA：`torch.cuda.set_device` /
`device="cuda"` / `dist.init_process_group("nccl")` / `torch.autocast("cuda")` /
`torch.cuda.max_memory_allocated`。换到**海光 DCU（ROCm/HIP）**或
**昇腾 NPU（CANN/torch_npu）** 上，代码必须改。本模块把所有平台差异收敛到一处，
训练/推理脚本只调用这里的函数 ⇒ 换卡不改代码。

平台对照
--------
============  ========  ==========  ============  ==================================
平台          设备串    分布式后端  显存 API      备注
============  ========  ==========  ============  ==================================
NVIDIA GPU    cuda      nccl        torch.cuda    —
海光 DCU      cuda      nccl        torch.cuda    ROCm 版 torch 里 `torch.cuda` 即 HIP；
                                                  `is_rocm()` 返回 True（可据此换 attn 实现）
昇腾 NPU      npu       hccl        torch.npu     需先 `import torch_npu`（本模块自动）
Intel XPU     xpu       ccl         torch.xpu     需 intel_extension_for_pytorch
Apple MPS     mps       gloo        —             只能单卡冒烟
CPU           cpu       gloo        —             只能冒烟（DINOv2-large 训不动）
============  ========  ==========  ============  ==================================

用法::

    from srdiff_runtime import Runtime
    rt = Runtime(args.device_type)          # "auto" 自动探测
    print(rt.describe())
    device = rt.device(index=0)             # torch.device
    ctx = rt.amp_context("bf16")            # 设备无关 autocast
    print(rt.peak_memory_gib(), rt.dist_backend(), rt.attn_implementation("sdpa"))

注意：**可见卡的环境变量必须在 `import torch` 之前设置**（torch 初始化后改无效）。
本模块提供 `set_visible_devices()`，请在 import torch 之前调用。
"""
from __future__ import annotations

import contextlib
import os
from typing import Optional

DEVICE_TYPES = ("cuda", "npu", "xpu", "mps", "cpu", "dcu")

# "dcu" 是**别名**：海光 DCU 用的是 ROCm/HIP 版 torch，设备串与后端都是 `cuda`/`nccl`，
# 单独给一个名字只为让调用方/日志能显式区分「我跑在 DCU 上」，并据此取更保守的默认值。
_ALIASES = {"dcu": "cuda"}

_VISIBLE_ENV = {
    "cuda": "CUDA_VISIBLE_DEVICES",       # NVIDIA 与 DCU(ROCm) 都用这个
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
    "mps": None,
    "cpu": None,
}

_DIST_BACKEND = {"cuda": "nccl", "npu": "hccl", "xpu": "ccl",
                 "mps": "gloo", "cpu": "gloo"}


# ── 可见设备（必须在 import torch 之前）────────────────────────────────────
def set_visible_devices(devices, device_type: str = "cuda"):
    """设置可见卡（`devices` = "0,1" / [0,1] / None）。返回用到的环境变量名。"""
    var = _VISIBLE_ENV.get(device_type)
    if var is None or devices is None:
        return None
    val = devices if isinstance(devices, str) else ",".join(str(d) for d in devices)
    os.environ[var] = val
    if device_type == "cuda":                      # DCU 的 ROCm 原生变量，一并设上
        os.environ.setdefault("HIP_VISIBLE_DEVICES", val)
    return var


# ── 探测 ────────────────────────────────────────────────────────────────────
def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def npu_available() -> bool:
    if not _has("torch_npu"):
        return False
    import torch
    return bool(getattr(torch, "npu", None)) and torch.npu.is_available() \
        and torch.npu.device_count() > 0


def cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def xpu_available() -> bool:
    import torch
    return bool(getattr(torch, "xpu", None)) and torch.xpu.is_available()


def mps_available() -> bool:
    import torch
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def detect_device_type(prefer: Optional[str] = "auto") -> str:
    """auto 时按 npu → cuda(DCU/ROCm 也在内) → xpu → mps → cpu 顺序探测。

    别名（如 "dcu"）在这里就解析成真实设备串（"cuda"），保证任何调用方拿到的是
    torch 能直接用的名字；原始请求名由 `Runtime.requested` 保留供日志/分支使用。
    """
    if prefer in _ALIASES:
        prefer = _ALIASES[prefer]
    if prefer and prefer != "auto":
        if prefer not in DEVICE_TYPES:
            raise ValueError(f"未知 device_type={prefer}，可选 {DEVICE_TYPES} 或 auto")
        return prefer
    for dt, fn in (("npu", npu_available), ("cuda", cuda_available),
                   ("xpu", xpu_available), ("mps", mps_available)):
        try:
            if fn():
                return dt
        except Exception:
            continue
    return "cpu"


def is_rocm() -> bool:
    """DCU / AMD ROCm 版 torch：`torch.version.hip` 非空。"""
    try:
        import torch
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


# ── Runtime ─────────────────────────────────────────────────────────────────
class Runtime:
    """设备无关运行时。构造时自动 import torch_npu（若在 NPU 上）。"""

    def __init__(self, device_type: str = "auto", index: int = 0):
        requested = device_type
        if requested in _ALIASES:                     # dcu → cuda（ROCm/HIP）
            device_type = _ALIASES[requested]
        self.device_type = detect_device_type(device_type)
        if self.device_type == "npu":
            import torch_npu  # noqa: F401  （导入即注册 torch.npu）
        import torch
        self.torch = torch
        self.index = int(index)
        self.requested = requested
        # ROCm/HIP（海光 DCU、AMD GPU）：torch.version.hip 非空，设备串仍是 "cuda"
        self.is_rocm = bool(getattr(torch.version, "hip", None))
        if requested == "dcu" and not self.is_rocm:
            print("[warn] --device_type dcu 但本机 torch 不是 ROCm/HIP 版"
                  f"（torch.version.hip={getattr(torch.version, 'hip', None)}）"
                  "，按 device_type=cuda 继续；请确认装的是 DCU 版 torch/dtkit")

    # 设备
    def device(self, index: Optional[int] = None):
        i = self.index if index is None else int(index)
        if self.device_type == "cpu":
            return self.torch.device("cpu")
        return self.torch.device(self.device_type, i)

    @property
    def module(self):
        """底层加速器模块（torch.cuda / torch.npu / torch.xpu / None）。"""
        return getattr(self.torch, self.device_type, None) if self.device_type != "cpu" else None

    def device_count(self) -> int:
        m = self.module
        if m is None:
            return 1
        try:
            return m.device_count()
        except Exception:
            return 1

    def dist_backend(self) -> str:
        return _DIST_BACKEND[self.device_type]

    def set_device(self, index: Optional[int] = None):
        m = self.module
        if m is not None and hasattr(m, "set_device"):
            m.set_device(self.index if index is None else int(index))

    def is_accelerator(self) -> bool:
        return self.device_type in ("cuda", "npu", "xpu", "mps")

    # 精度
    def amp_dtype(self, precision: str):
        t = self.torch
        if precision in ("fp32", "no", "none"):
            return None
        if precision == "bf16":
            return t.bfloat16
        if precision == "fp16":
            return t.float16
        raise ValueError(f"未知 precision={precision}（bf16/fp16/fp32）")

    def amp_context(self, precision: str):
        """设备无关 autocast；fp32 时是空上下文。"""
        dt = self.amp_dtype(precision)
        if dt is None or self.device_type == "cpu":
            return contextlib.nullcontext()
        return self.torch.autocast(device_type=self.device_type, dtype=dt)

    def mixed_precision_flag(self, precision: str) -> str:
        """给 accelerate.Accelerator(mixed_precision=...) 的值。"""
        return {"fp32": "no", "none": "no", "no": "no"}.get(precision, precision)

    def supports_bf16(self) -> bool:
        if self.device_type == "cpu":
            return False
        try:
            return bool(self.torch.cuda.is_bf16_supported()) if self.device_type == "cuda" \
                else True
        except Exception:
            return True

    # 注意力实现
    def attn_implementation(self, requested: str = "auto") -> str:
        """按平台给注意力实现。`auto` 的规则（保守优先「先跑通」）：

        · NPU / XPU → `eager`（torch_npu / IPEX 的 SDPA 融合算子版本差异大）；
        · **ROCm/HIP（海光 DCU）→ `eager`**。DCU 的 gfx 目标（gfx906/gfx928 等）
          通常没有 CK flash-attn 后端，SDPA 会退化成 math 实现 —— 能跑但不快，
          且个别 dtkit 版本上会直接报错；未实测平台先取最保守的路径，
          确认无误后可用 `--attn_implementation sdpa` 打开；
        · NVIDIA CUDA → `sdpa`（transformers 默认，实测可用）。

        `flash_attention_2` 需要 flash-attn，DCU/NPU 上没有官方后端 ⇒ 只能显式传。
        """
        if requested and requested != "auto":
            return requested
        if self.device_type in ("npu", "xpu") or self.is_rocm:
            return "eager"
        return "sdpa"

    # 显存
    def peak_memory_gib(self, index: Optional[int] = None) -> float:
        m = self.module
        if m is None or not hasattr(m, "max_memory_allocated"):
            return 0.0
        try:
            return float(m.max_memory_allocated(self.index if index is None else index)) / 2 ** 30
        except Exception:
            return 0.0

    def reset_peak_memory(self, index: Optional[int] = None):
        m = self.module
        if m is not None and hasattr(m, "reset_peak_memory_stats"):
            with contextlib.suppress(Exception):
                m.reset_peak_memory_stats(self.index if index is None else index)

    def memory_allocated_gib(self, index: Optional[int] = None) -> float:
        m = self.module
        if m is None or not hasattr(m, "memory_allocated"):
            return 0.0
        with contextlib.suppress(Exception):
            return float(m.memory_allocated(self.index if index is None else index)) / 2 ** 30
        return 0.0

    def empty_cache(self):
        m = self.module
        if m is not None and hasattr(m, "empty_cache"):
            with contextlib.suppress(Exception):
                m.empty_cache()

    def synchronize(self, index: Optional[int] = None):
        m = self.module
        if m is None or not hasattr(m, "synchronize"):
            return
        with contextlib.suppress(Exception):
            m.synchronize(self.index if index is None else index)

    # 信息
    def device_name(self, index: Optional[int] = None) -> str:
        i = self.index if index is None else int(index)
        m = self.module
        try:
            if m is not None and hasattr(m, "get_device_name"):
                return str(m.get_device_name(i))
        except Exception:
            pass
        import platform
        return platform.processor() or self.device_type

    def describe(self) -> dict:
        import platform
        t = self.torch
        return {
            "requested": self.requested,
            "device_type": self.device_type,
            "device_name": self.device_name(),
            "device_count": self.device_count(),
            "rocm_hip": getattr(t.version, "hip", None),
            "is_dcu_alias": self.requested == "dcu",
            "cuda": getattr(t.version, "cuda", None),
            "torch": t.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dist_backend": self.dist_backend(),
            "pin_memory": self.device_type in ("cuda", "xpu"),
            "attn_implementation_auto": self.attn_implementation("auto"),
        }

    def __repr__(self):
        d = self.describe()
        return (f"Runtime({d['device_type']}, {d['device_name']}, "
                f"n={d['device_count']}, backend={d['dist_backend']})")


def summarize_env() -> dict:
    """一行命令打印环境（`python srdiff_runtime.py`）。"""
    rt = Runtime("auto")
    d = rt.describe()
    d["visual"] = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get(
        "ASCEND_RT_VISIBLE_DEVICES")
    if rt.torch.cuda.is_available() and rt.device_type == "cuda":
        d["devices"] = [rt.torch.cuda.get_device_name(i)
                        for i in range(rt.torch.cuda.device_count())]
    return d


if __name__ == "__main__":
    import json
    print(json.dumps(summarize_env(), indent=2, ensure_ascii=False))
