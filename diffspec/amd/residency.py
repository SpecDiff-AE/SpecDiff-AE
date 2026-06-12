"""AMD/HIP cache-residency hints for DiffSpec Chunk Arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .capabilities import get_amd_gpu_info, is_rocm_pytorch
from .extension import get_hip_runtime_extension_status, load_hip_runtime_extension


@dataclass(frozen=True)
class AmdResidencyConfig:
    """Configuration for HIP access-policy-window hints."""

    enable: bool = True
    hit_ratio: float = 0.85
    max_window_bytes: int | None = None
    try_runtime_extension: bool = True
    warn_on_fallback: bool = False


@dataclass(frozen=True)
class AmdResidencyStatus:
    """Outcome of applying a cache-residency hint."""

    requested: bool
    applied: bool
    backend: str
    requested_bytes: int
    window_bytes: int
    hit_ratio: float
    reason: str | None = None
    device: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "applied": self.applied,
            "backend": self.backend,
            "requested_bytes": self.requested_bytes,
            "window_bytes": self.window_bytes,
            "hit_ratio": self.hit_ratio,
            "reason": self.reason,
            "device": self.device,
        }


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _stream_handle(stream: torch.cuda.Stream | None, device: torch.device) -> int:
    if stream is None:
        stream = torch.cuda.current_stream(device)
    return int(stream.cuda_stream)


def apply_residency_hint(
    tensor: torch.Tensor,
    config: AmdResidencyConfig | None = None,
    stream: torch.cuda.Stream | None = None,
) -> AmdResidencyStatus:
    """Apply a HIP APW-like residency hint to ``tensor`` when possible.

    HIP exposes an access-policy-window stream attribute. PyTorch does not
    expose that call directly, so this function lazily uses the tiny optional
    extension in :mod:`diffspec.amd.hip_runtime` when a ROCm build is present.
    On non-ROCm machines it returns a structured no-op status.
    """

    config = config or AmdResidencyConfig()
    requested_bytes = _tensor_nbytes(tensor)
    hit_ratio = max(0.0, min(float(config.hit_ratio), 1.0))
    device_info = get_amd_gpu_info(tensor.device if tensor.device.type == "cuda" else None)
    device_dict = device_info.to_dict()

    if not config.enable:
        return AmdResidencyStatus(
            requested=False,
            applied=False,
            backend="disabled",
            requested_bytes=requested_bytes,
            window_bytes=0,
            hit_ratio=hit_ratio,
            reason="config_disabled",
            device=device_dict,
        )

    if tensor.device.type != "cuda":
        return AmdResidencyStatus(
            requested=True,
            applied=False,
            backend="none",
            requested_bytes=requested_bytes,
            window_bytes=0,
            hit_ratio=hit_ratio,
            reason="tensor_is_not_on_gpu",
            device=device_dict,
        )

    if not is_rocm_pytorch():
        return AmdResidencyStatus(
            requested=True,
            applied=False,
            backend="none",
            requested_bytes=requested_bytes,
            window_bytes=0,
            hit_ratio=hit_ratio,
            reason="pytorch_build_is_not_rocm",
            device=device_dict,
        )

    window_bytes = requested_bytes
    if config.max_window_bytes is not None:
        window_bytes = min(window_bytes, max(int(config.max_window_bytes), 0))

    if window_bytes <= 0:
        return AmdResidencyStatus(
            requested=True,
            applied=False,
            backend="none",
            requested_bytes=requested_bytes,
            window_bytes=0,
            hit_ratio=hit_ratio,
            reason="empty_window",
            device=device_dict,
        )

    if not config.try_runtime_extension:
        return AmdResidencyStatus(
            requested=True,
            applied=False,
            backend="metadata_only",
            requested_bytes=requested_bytes,
            window_bytes=window_bytes,
            hit_ratio=hit_ratio,
            reason="runtime_extension_disabled",
            device=device_dict,
        )

    ext = load_hip_runtime_extension()
    if ext is None:
        return AmdResidencyStatus(
            requested=True,
            applied=False,
            backend="metadata_only",
            requested_bytes=requested_bytes,
            window_bytes=window_bytes,
            hit_ratio=hit_ratio,
            reason=(
                "hip_runtime_extension_unavailable:"
                f"{get_hip_runtime_extension_status().reason}"
            ),
            device=device_dict,
        )

    result = ext.set_access_policy_window(
        _stream_handle(stream, tensor.device),
        int(tensor.data_ptr()),
        int(window_bytes),
        float(hit_ratio),
    )
    applied = bool(result.get("applied", False))
    return AmdResidencyStatus(
        requested=True,
        applied=applied,
        backend="hip_access_policy_window",
        requested_bytes=requested_bytes,
        window_bytes=int(result.get("window_bytes", window_bytes if applied else 0)),
        hit_ratio=hit_ratio,
        reason=result.get("reason"),
        device=device_dict,
    )


def clear_residency_hint(stream: torch.cuda.Stream | None = None) -> dict[str, Any]:
    """Clear the HIP access-policy-window on the current stream if available."""

    if not is_rocm_pytorch() or not torch.cuda.is_available():
        return {"applied": False, "reason": "rocm_gpu_unavailable"}
    ext = load_hip_runtime_extension()
    if ext is None:
        return {"applied": False, "reason": "hip_runtime_extension_unavailable"}
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return dict(ext.clear_access_policy_window(_stream_handle(stream, device)))
