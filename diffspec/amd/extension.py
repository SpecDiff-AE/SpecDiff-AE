"""Optional HIP runtime extension loader.

This module keeps extension loading lazy. Importing :mod:`diffspec.amd` should
work on CUDA-only and CPU-only machines; the HIP extension is compiled only
when the caller explicitly asks for an AMD residency hint on a ROCm PyTorch
build.
"""

from __future__ import annotations

import os
import shutil
import sys
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


def _hip_extension_compile_flags() -> tuple[list[str], list[str]]:
    """Return host/device flags needed by the ROCm extension.

    PyTorch compiles ``.cpp`` sources with the host C++ compiler even for a
    ROCm extension.  The explicit project macro keeps the host binding on the
    HIP runtime path without depending on compiler-private HIP macros.
    """

    common = ["-O3", "-DDIFFSPEC_WITH_HIP=1"]
    return common, common


def _ensure_python_bin_on_path() -> None:
    """Make console scripts from the active Python environment discoverable."""

    candidate_bins = [
        str(Path(sys.executable).parent),
        str(Path(sys.prefix) / "bin"),
    ]
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    prepend = [path for path in candidate_bins if path and path not in path_parts]
    if prepend:
        os.environ["PATH"] = os.pathsep.join([*prepend, *path_parts])


def _copy_if_changed(src: Path, dst: Path) -> None:
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    shutil.copy2(src, dst)


def _prepare_extension_sources(sources: tuple[Path, ...]) -> tuple[Path, tuple[str, ...]]:
    """Copy HIP extension sources away from the repo before torch hipifies them."""

    build_root = Path(
        os.environ.get(
            "DIFFSPEC_HIP_EXTENSION_BUILD_DIR",
            Path.home() / ".cache" / "diffspec" / "hip_runtime_extension",
        )
    )
    source_root = build_root / "src"
    source_root.mkdir(parents=True, exist_ok=True)
    staged_sources: list[str] = []
    for source in sources:
        staged = source_root / source.name
        _copy_if_changed(source, staged)
        staged_sources.append(str(staged))
    return build_root, tuple(staged_sources)


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

    _ensure_python_bin_on_path()
    build_root, staged_sources = _prepare_extension_sources(sources)
    build_root.mkdir(parents=True, exist_ok=True)

    try:
        module = load(
            name="diffspec_amd_hip_runtime",
            sources=list(staged_sources),
            build_directory=str(build_root),
            verbose=verbose,
            with_cuda=True,
            extra_cflags=_hip_extension_compile_flags()[0],
            extra_cuda_cflags=_hip_extension_compile_flags()[1],
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
