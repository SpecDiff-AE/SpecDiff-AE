"""Salience-aware chunk encoding for retrieval-side draft KV compression."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SalienceAwareChunkEncoder(nn.Module):
    """Encode a token chunk into a proxy key/value pair.

    The encoder computes salience weights from key/value magnitudes, builds a
    weighted centroid, optionally preserves boundary tokens, and optionally
    projects the resulting proxy back to the original head dimension.
    """

    def __init__(
        self,
        head_dim: int,
        enable_boundary_preservation: bool = True,
        enable_learned_projection: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.enable_boundary_preservation = enable_boundary_preservation
        self.enable_learned_projection = enable_learned_projection

        input_dim = 3 * head_dim if enable_boundary_preservation else head_dim
        if enable_learned_projection:
            self.key_projection = nn.Linear(input_dim, head_dim, bias=False, device=device, dtype=dtype)
            self.value_projection = nn.Linear(input_dim, head_dim, bias=False, device=device, dtype=dtype)
            self._initialize_projection(device=device, dtype=dtype)
        else:
            self.key_projection = None
            self.value_projection = None

    def _initialize_projection(
        self,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> None:
        """Initialize projections close to centroid passthrough behavior."""

        assert self.key_projection is not None
        assert self.value_projection is not None
        eye = torch.eye(self.head_dim, device=device, dtype=dtype)
        with torch.no_grad():
            self.key_projection.weight.zero_()
            self.value_projection.weight.zero_()
            self.key_projection.weight[:, : self.head_dim].copy_(eye)
            self.value_projection.weight[:, : self.head_dim].copy_(eye)
            if self.enable_boundary_preservation:
                self.key_projection.weight[:, self.head_dim :].normal_(0, 0.02)
                self.value_projection.weight[:, self.head_dim :].normal_(0, 0.02)

    @torch.no_grad()
    def encode_chunk(
        self,
        chunk_keys: torch.Tensor,
        chunk_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one chunk with shape ``[chunk_len, head_dim]``."""

        chunk_len = chunk_keys.shape[0]
        if chunk_len == 0:
            return (
                torch.zeros(self.head_dim, device=chunk_keys.device, dtype=chunk_keys.dtype),
                torch.zeros(self.head_dim, device=chunk_values.device, dtype=chunk_values.dtype),
            )

        if chunk_len == 1:
            key_proxy = chunk_keys[0]
            value_proxy = chunk_values[0]
        else:
            key_norms = torch.norm(chunk_keys, dim=-1, p=2) ** 2
            value_norms = torch.norm(chunk_values, dim=-1, p=2) ** 2
            salience_scores = key_norms + value_norms
            salience_sum = salience_scores.sum()
            if salience_sum < 1e-8:
                salience = torch.ones_like(salience_scores) / chunk_len
            else:
                salience = salience_scores / salience_sum

            key_proxy = (salience.unsqueeze(-1) * chunk_keys).sum(dim=0)
            value_proxy = (salience.unsqueeze(-1) * chunk_values).sum(dim=0)

        if self.enable_boundary_preservation:
            key_proxy = torch.cat([key_proxy, chunk_keys[0], chunk_keys[-1]])
            value_proxy = torch.cat([value_proxy, chunk_values[0], chunk_values[-1]])

        if self.enable_learned_projection and self.key_projection is not None:
            assert self.value_projection is not None
            return self.key_projection(key_proxy), self.value_projection(value_proxy)

        if self.enable_boundary_preservation:
            key_parts = key_proxy.view(3, self.head_dim)
            value_parts = value_proxy.view(3, self.head_dim)
            return key_parts.mean(dim=0), value_parts.mean(dim=0)

        return key_proxy, value_proxy

    @torch.no_grad()
    def encode_chunks_batch(
        self,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        chunks: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode many chunks from full KV tensors.

        Args:
            full_keys: Tensor with shape ``[num_heads, seq_len, head_dim]``.
            full_values: Tensor with shape ``[num_heads, seq_len, head_dim]``.
            chunks: ``(chunk_id, start, end)`` metadata entries.
        """

        num_heads = full_keys.shape[0]
        num_chunks = len(chunks)
        proxy_keys = torch.zeros(
            num_heads,
            num_chunks,
            self.head_dim,
            device=full_keys.device,
            dtype=full_keys.dtype,
        )
        proxy_values = torch.zeros_like(proxy_keys)

        for head_idx in range(num_heads):
            for chunk_idx, (_, start, end) in enumerate(chunks):
                proxy_key, proxy_value = self.encode_chunk(
                    full_keys[head_idx, start:end, :],
                    full_values[head_idx, start:end, :],
                )
                proxy_keys[head_idx, chunk_idx] = proxy_key
                proxy_values[head_idx, chunk_idx] = proxy_value

        return proxy_keys, proxy_values

    def forward(
        self,
        chunk_keys: torch.Tensor,
        chunk_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode either one chunk or a batch of chunks."""

        if chunk_keys.dim() == 2:
            return self.encode_chunk(chunk_keys, chunk_values)
        if chunk_keys.dim() == 3:
            key_proxies = []
            value_proxies = []
            for batch_idx in range(chunk_keys.shape[0]):
                key_proxy, value_proxy = self.encode_chunk(
                    chunk_keys[batch_idx],
                    chunk_values[batch_idx],
                )
                key_proxies.append(key_proxy)
                value_proxies.append(value_proxy)
            return torch.stack(key_proxies), torch.stack(value_proxies)
        raise ValueError(f"Unexpected chunk tensor shape: {tuple(chunk_keys.shape)}")
