"""AMD-friendly contiguous arena and LDS-staging abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .capabilities import is_rocm_pytorch
from .extension import get_hip_runtime_extension_status, load_hip_runtime_extension
from .residency import AmdResidencyConfig, AmdResidencyStatus, apply_residency_hint


@dataclass
class AmdArenaConfig:
    """Configuration for AMD Chunk Arena staging."""

    max_chunks: int = 64
    chunk_size: int = 64
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    batch_size: int = 1
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    enable_residency_hint: bool = False
    residency: AmdResidencyConfig | None = None
    enable_hip_compact: bool = True


class AmdChunkArena:
    """Contiguous KV working-set arena for ROCm/HIP execution.

    This is the AMD-side counterpart to the CUDA Chunk Arena/APW/TMA
    experiments:

    * APW-like behavior is requested through HIP access-policy-window hints.
    * TMA-like behavior is expressed as physically contiguous arena staging,
      with the HIP microbenchmark using LDS tile reuse for the inner loop.
    """

    def __init__(self, config: AmdArenaConfig):
        self.backend_name = "amd_rocm"
        self.config = config
        self.max_chunks = int(config.max_chunks)
        self.chunk_size = int(config.chunk_size)
        self.num_layers = int(config.num_layers)
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.head_dim)
        self.batch_size = int(config.batch_size)
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        self.arena_kv = torch.zeros(
            self.num_layers,
            2,
            self.batch_size,
            self.num_heads,
            self.max_chunks * self.chunk_size,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.chunk_to_offset: dict[int, int] = {}
        self.active_chunks: list[int] = []
        self.active_spans: list[tuple[int, int, int]] = []
        self.active_len = 0
        self.free_offsets: list[int] = list(range(self.max_chunks))
        self.last_residency_status: AmdResidencyStatus | None = None
        self.last_compact_status: dict | None = None
        self.compact_backend = "torch_copy"
        self.total_updates = 0
        self.total_copied_tokens = 0

    def update(
        self,
        chunks: Iterable[int | tuple[int, int, int]],
        full_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Update resident chunks and return the active contiguous view."""

        self.total_updates += 1
        spans = self._normalize_spans(chunks)

        # The model path needs an exact, contiguous working KV layout.  Copy the
        # selected spans into the arena prefix in order; the HIP microbenchmark
        # isolates the hardware cost of LDS/APW staging.
        self.chunk_to_offset.clear()
        self.active_chunks = [chunk_id for chunk_id, _, _ in spans]
        self.active_spans = spans
        self.active_len = self._compute_active_len(spans, full_kv_cache)
        self.free_offsets = list(range(len(spans), self.max_chunks))

        if full_kv_cache is not None:
            for offset, (chunk_id, _, _) in enumerate(spans):
                self.chunk_to_offset[chunk_id] = offset
            copied = self._stage_spans(spans, full_kv_cache)
            self.total_copied_tokens += copied

        view = self.working_view()
        if self.config.enable_residency_hint:
            self.last_residency_status = apply_residency_hint(
                view,
                self.config.residency or AmdResidencyConfig(),
            )
        return view

    def _normalize_spans(
        self,
        chunks: Iterable[int | tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        spans: list[tuple[int, int, int]] = []
        for raw in list(chunks)[: self.max_chunks]:
            if isinstance(raw, tuple):
                if len(raw) != 3:
                    raise ValueError("Chunk span tuples must be (chunk_id, start, end)")
                chunk_id, start, end = raw
            else:
                chunk_id = int(raw)
                start = chunk_id * self.chunk_size
                end = start + self.chunk_size
            start = max(int(start), 0)
            end = max(int(end), start)
            spans.append((int(chunk_id), start, end))
        spans.sort(key=lambda item: (item[1], item[2], item[0]))
        return spans

    def _compute_active_len(
        self,
        spans: list[tuple[int, int, int]],
        full_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> int:
        if full_kv_cache is None or len(full_kv_cache) == 0:
            return sum(max(end - start, 0) for _, start, end in spans)

        key0, value0 = full_kv_cache[0]
        if key0.dim() == 3:
            key0 = key0.unsqueeze(0)
        if value0.dim() == 3:
            value0 = value0.unsqueeze(0)
        seq_len = min(int(key0.size(2)), int(value0.size(2)))
        active_len = 0
        for _, start, end in spans:
            actual_end = min(int(end), seq_len)
            active_len += max(actual_end - int(start), 0)
        return active_len

    def _span_tensor(
        self,
        spans: list[tuple[int, int, int]],
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        normalized: list[tuple[int, int]] = []
        active_len = 0
        for _, start, end in spans:
            actual_start = min(max(int(start), 0), seq_len)
            actual_end = min(max(int(end), actual_start), seq_len)
            normalized.append((actual_start, actual_end))
            active_len += max(actual_end - actual_start, 0)
        span_tensor = torch.tensor(normalized, dtype=torch.long, device=device)
        return span_tensor, int(active_len)

    def _stage_spans(
        self,
        spans: list[tuple[int, int, int]],
        full_kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        if not spans:
            self.last_compact_status = {"backend": "none", "reason": "empty_spans"}
            return 0

        key0, value0 = full_kv_cache[0]
        if key0.dim() == 3:
            key0 = key0.unsqueeze(0)
        if value0.dim() == 3:
            value0 = value0.unsqueeze(0)
        seq_len = min(int(key0.size(2)), int(value0.size(2)))
        span_tensor, active_len = self._span_tensor(spans, seq_len, self.device)
        if active_len <= 0:
            self.last_compact_status = {"backend": "none", "reason": "empty_active_len"}
            return 0

        if self._try_hip_compact(full_kv_cache, span_tensor, active_len):
            return active_len

        cursor = 0
        for _, start, end in spans:
            copied = self._copy_span(start, end, cursor, full_kv_cache)
            cursor += copied
        self.compact_backend = "torch_copy"
        if self.last_compact_status is None:
            self.last_compact_status = {"backend": "torch_copy", "applied": True}
        return active_len

    def _try_hip_compact(
        self,
        full_kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        span_tensor: torch.Tensor,
        active_len: int,
    ) -> bool:
        if (
            not self.config.enable_hip_compact
            or self.device.type != "cuda"
            or not is_rocm_pytorch()
        ):
            self.last_compact_status = {
                "backend": "torch_copy",
                "applied": False,
                "reason": "hip_compact_not_applicable",
            }
            return False

        ext = load_hip_runtime_extension()
        if ext is None:
            self.last_compact_status = {
                "backend": "torch_copy",
                "applied": False,
                "reason": "hip_runtime_extension_unavailable",
                "extension": get_hip_runtime_extension_status().to_dict(),
            }
            return False

        statuses = []
        try:
            for layer_idx in range(min(self.num_layers, len(full_kv_cache))):
                key, value = full_kv_cache[layer_idx]
                if key.dim() == 3:
                    key = key.unsqueeze(0)
                if value.dim() == 3:
                    value = value.unsqueeze(0)
                self._validate_source_pair(key, value)
                dst_key = self.arena_kv[layer_idx, 0, :, :, :active_len, :]
                dst_value = self.arena_kv[layer_idx, 1, :, :, :active_len, :]
                statuses.append(
                    dict(ext.compact_kv_spans(key.contiguous(), dst_key, span_tensor, active_len))
                )
                statuses.append(
                    dict(ext.compact_kv_spans(value.contiguous(), dst_value, span_tensor, active_len))
                )
        except Exception as exc:
            self.last_compact_status = {
                "backend": "torch_copy",
                "applied": False,
                "reason": "hip_compact_failed",
                "error": repr(exc),
                "extension": get_hip_runtime_extension_status().to_dict(),
            }
            return False

        self.compact_backend = "hip_compact_kv_spans"
        self.last_compact_status = {
            "backend": self.compact_backend,
            "applied": True,
            "active_len": int(active_len),
            "calls": len(statuses),
            "extension": get_hip_runtime_extension_status().to_dict(),
        }
        return True

    def _validate_source_pair(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.size(1) != self.num_heads or value.size(1) != self.num_heads:
            raise RuntimeError(
                "AmdChunkArena head mismatch: "
                f"arena={self.num_heads}, key={key.size(1)}, value={value.size(1)}"
            )
        if key.size(0) != self.batch_size or value.size(0) != self.batch_size:
            raise RuntimeError(
                "AmdChunkArena batch mismatch: "
                f"arena={self.batch_size}, key={key.size(0)}, value={value.size(0)}"
            )
        if key.size(3) != self.head_dim or value.size(3) != self.head_dim:
            raise RuntimeError(
                "AmdChunkArena head_dim mismatch: "
                f"arena={self.head_dim}, key={key.size(3)}, value={value.size(3)}"
            )

    def _copy_span(
        self,
        start: int,
        end: int,
        slot_start: int,
        full_kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        copied_tokens = 0

        for layer_idx in range(min(self.num_layers, len(full_kv_cache))):
            key, value = full_kv_cache[layer_idx]
            if key.dim() == 3:
                key = key.unsqueeze(0)
            if value.dim() == 3:
                value = value.unsqueeze(0)
            self._validate_source_pair(key, value)

            actual_end = min(end, key.size(2), value.size(2))
            actual_len = max(actual_end - start, 0)
            if actual_len == 0:
                continue
            slot_end = slot_start + actual_len
            self.arena_kv[layer_idx, 0, :, :, slot_start:slot_end, :].copy_(
                key[:, :, start:actual_end, :],
                non_blocking=True,
            )
            self.arena_kv[layer_idx, 1, :, :, slot_start:slot_end, :].copy_(
                value[:, :, start:actual_end, :],
                non_blocking=True,
            )
            copied_tokens = max(copied_tokens, actual_len)
        return copied_tokens

    def working_view(self) -> torch.Tensor:
        return self.arena_kv[:, :, :, :, : self.active_len, :]

    def as_kv_list(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return the active arena prefix as a per-layer KV list."""

        view = self.working_view()
        return [(view[layer_idx, 0], view[layer_idx, 1]) for layer_idx in range(view.size(0))]

    def get_statistics(self) -> dict:
        active_bytes = self.working_view().numel() * self.working_view().element_size()
        arena_bytes = self.arena_kv.numel() * self.arena_kv.element_size()
        return {
            "backend": "amd_rocm",
            "total_updates": self.total_updates,
            "active_chunks": len(self.active_chunks),
            "active_spans": [list(span) for span in self.active_spans],
            "max_chunks": self.max_chunks,
            "utilization": len(self.active_chunks) / max(self.max_chunks, 1),
            "active_bytes": int(active_bytes),
            "arena_bytes": int(arena_bytes),
            "total_copied_tokens": int(self.total_copied_tokens),
            "compact_backend": self.compact_backend,
            "compact_status": self.last_compact_status,
            "residency": (
                self.last_residency_status.to_dict()
                if self.last_residency_status is not None
                else None
            ),
        }

    def reset(self) -> None:
        self.chunk_to_offset.clear()
        self.active_chunks.clear()
        self.active_spans.clear()
        self.active_len = 0
        self.free_offsets = list(range(self.max_chunks))
        self.arena_kv.zero_()
        self.last_residency_status = None
        self.last_compact_status = None
        self.compact_backend = "torch_copy"
