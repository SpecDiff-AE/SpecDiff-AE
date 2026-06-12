"""AMD ROCm helpers for DiffSpec memory-residency experiments.

The package intentionally uses AMD/HIP vocabulary.  CUDA APW maps to HIP's
access-policy-window stream attribute where available; Hopper TMA maps to a
portable contiguous-arena plus LDS tile-staging path.
"""

from .capabilities import AmdGpuInfo, get_amd_gpu_info, is_rocm_pytorch
from .extension import HipRuntimeExtensionStatus, get_hip_runtime_extension_status
from .residency import AmdResidencyConfig, AmdResidencyStatus, apply_residency_hint
from .staging import AmdArenaConfig, AmdChunkArena

__all__ = [
    "AmdGpuInfo",
    "AmdResidencyConfig",
    "AmdResidencyStatus",
    "HipRuntimeExtensionStatus",
    "AmdArenaConfig",
    "AmdChunkArena",
    "apply_residency_hint",
    "get_amd_gpu_info",
    "get_hip_runtime_extension_status",
    "is_rocm_pytorch",
]
