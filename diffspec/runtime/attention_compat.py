"""FlashAttention imports with a small PyTorch fallback.

The production path still uses flash-attn when it is installed.  The fallback
keeps tests and smoke benchmarks runnable in environments where the compiled
extension is unavailable.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _torch_flash_sdp_enabled() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled())
    except Exception:
        return False


PYTORCH_FLASH_ATTN_AVAILABLE = _torch_flash_sdp_enabled()


def sdpa_backend_name() -> str:
    if FLASH_ATTN_AVAILABLE:
        return "flash_attn"
    if PYTORCH_FLASH_ATTN_AVAILABLE:
        return "torch_sdpa_flash"
    return "torch_sdpa"


def _sdpa_flash_attention(q, k, v, *, attn_mask=None, dropout_p=0.0, is_causal=False):
    if PYTORCH_FLASH_ATTN_AVAILABLE and q.is_cuda and attn_mask is None:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
        except Exception:
            pass
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except Exception:
    FLASH_ATTN_AVAILABLE = False

    def _to_bhsd(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)

    def _to_bshd(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)

    def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(k_len - q_len, k_len, device=device).view(q_len, 1)
        k_pos = torch.arange(k_len, device=device).view(1, k_len)
        return k_pos <= q_pos

    def flash_attn_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        causal: bool = False,
        window_size: Optional[Tuple[int, int]] = None,
        **_: object,
    ) -> torch.Tensor:
        q_bhsd = _to_bhsd(q)
        k_bhsd = _to_bhsd(k)
        v_bhsd = _to_bhsd(v)

        attn_mask = None
        if causal and q.size(1) != k.size(1):
            attn_mask = _causal_mask(q.size(1), k.size(1), q.device)
            causal = False

        out = _sdpa_flash_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=causal,
        )
        return _to_bshd(out)

    def flash_attn_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        causal: bool = False,
        cache_seqlens: Optional[int] = None,
        return_softmax_lse: bool = False,
        **_: object,
    ):
        if cache_seqlens is not None:
            k = k_cache[:, : int(cache_seqlens), :, :]
            v = v_cache[:, : int(cache_seqlens), :, :]
        else:
            k = k_cache
            v = v_cache

        out = flash_attn_func(q, k, v, causal=causal)
        if not return_softmax_lse:
            return out

        scores = torch.matmul(
            _to_bhsd(q).float(),
            _to_bhsd(k).float().transpose(-2, -1),
        ) / math.sqrt(q.size(-1))
        if causal:
            mask = _causal_mask(q.size(1), k.size(1), q.device)
            scores = scores.masked_fill(~mask.view(1, 1, q.size(1), k.size(1)), float("-inf"))
        # Match flash-attn's softmax_lse layout: [batch, heads, seqlen_q].
        lse = torch.logsumexp(scores, dim=-1)
        return out, lse
