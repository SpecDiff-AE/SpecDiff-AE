"""ROCm/HIP capability detection for DiffSpec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AmdGpuInfo:
    """Small, JSON-friendly snapshot of the active AMD GPU runtime."""

    available: bool
    is_rocm: bool
    device_index: int | None = None
    name: str | None = None
    gcn_arch_name: str | None = None
    total_memory: int | None = None
    l2_cache_size: int | None = None
    multiprocessor_count: int | None = None
    warp_size: int | None = None
    hip_version: str | None = None
    torch_version: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "is_rocm": self.is_rocm,
            "device_index": self.device_index,
            "name": self.name,
            "gcn_arch_name": self.gcn_arch_name,
            "total_memory": self.total_memory,
            "l2_cache_size": self.l2_cache_size,
            "multiprocessor_count": self.multiprocessor_count,
            "warp_size": self.warp_size,
            "hip_version": self.hip_version,
            "torch_version": self.torch_version,
            "reason": self.reason,
        }


def is_rocm_pytorch() -> bool:
    """Return True when this PyTorch build targets ROCm/HIP."""

    return bool(getattr(torch.version, "hip", None))


def _read_property(props: Any, *names: str) -> Any:
    for name in names:
        if hasattr(props, name):
            return getattr(props, name)
    return None


def get_amd_gpu_info(device: int | torch.device | None = None) -> AmdGpuInfo:
    """Return active-device information without requiring an AMD machine."""

    if not is_rocm_pytorch():
        return AmdGpuInfo(
            available=False,
            is_rocm=False,
            hip_version=None,
            torch_version=torch.__version__,
            reason="pytorch_build_is_not_rocm",
        )

    if not torch.cuda.is_available():
        return AmdGpuInfo(
            available=False,
            is_rocm=True,
            hip_version=torch.version.hip,
            torch_version=torch.__version__,
            reason="rocm_pytorch_without_visible_gpu",
        )

    if device is None:
        device_index = torch.cuda.current_device()
    else:
        device_index = torch.device(device).index if not isinstance(device, int) else device
        if device_index is None:
            device_index = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device_index)
    return AmdGpuInfo(
        available=True,
        is_rocm=True,
        device_index=int(device_index),
        name=getattr(props, "name", None),
        gcn_arch_name=_read_property(props, "gcnArchName", "gcn_arch_name"),
        total_memory=getattr(props, "total_memory", None),
        l2_cache_size=_read_property(props, "l2_cache_size", "l2CacheSize"),
        multiprocessor_count=_read_property(
            props,
            "multi_processor_count",
            "multiprocessor_count",
        ),
        warp_size=getattr(props, "warp_size", None),
        hip_version=torch.version.hip,
        torch_version=torch.__version__,
    )
