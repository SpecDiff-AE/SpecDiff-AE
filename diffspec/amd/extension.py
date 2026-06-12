"""Optional HIP runtime extension loader.

This module keeps extension loading lazy. Importing :mod:`diffspec.amd` should
work on CUDA-only and CPU-only machines; the HIP extension is compiled only
when the caller explicitly asks for an AMD residency hint on a ROCm PyTorch
build.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from .capabilities import is_rocm_pytorch


@dataclass(frozen=True)
class HipRuntimeExtensionStatus:
    """Structured status for the optional ROCm runtime extension."""

    available: bool
    attempted: bool
    reason: str | None = None
    error: str | None = None
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "attempted": self.attempted,
            "reason": self.reason,
            "error": self.error,
            "sources": list(self.sources),
        }


_LAST_STATUS = HipRuntimeExtensionStatus(
    available=False,
    attempted=False,
    reason="not_requested",
)


@lru_cache(maxsize=1)
def load_hip_runtime_extension(verbose: bool = False) -> Any | None:
    """Compile and load the tiny HIP runtime helper extension if possible."""

    global _LAST_STATUS
    if not is_rocm_pytorch():
        _LAST_STATUS = HipRuntimeExtensionStatus(
            available=False,
            attempted=False,
            reason="pytorch_build_is_not_rocm",
        )
        return None

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:
        _LAST_STATUS = HipRuntimeExtensionStatus(
            available=False,
            attempted=False,
            reason="torch_cpp_extension_unavailable",
            error=repr(exc),
        )
        return None

    root = Path(__file__).resolve().parent
    sources = (
        root / "hip_runtime.cpp",
        root / "hip_kernels.hip",
    )
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        _LAST_STATUS = HipRuntimeExtensionStatus(
            available=False,
            attempted=False,
            reason="extension_source_missing",
            error=", ".join(missing),
            sources=tuple(str(source) for source in sources),
        )
        return None

    try:
        module = load(
            name="diffspec_amd_hip_runtime",
            sources=[str(source) for source in sources],
            verbose=verbose,
            with_cuda=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
        _LAST_STATUS = HipRuntimeExtensionStatus(
            available=True,
            attempted=True,
            sources=tuple(str(source) for source in sources),
        )
        return module
    except Exception as exc:
        _LAST_STATUS = HipRuntimeExtensionStatus(
            available=False,
            attempted=True,
            reason="extension_build_or_load_failed",
            error=repr(exc),
            sources=tuple(str(source) for source in sources),
        )
        return None


def get_hip_runtime_extension_status() -> HipRuntimeExtensionStatus:
    """Return the most recent extension-load status."""

    return _LAST_STATUS
