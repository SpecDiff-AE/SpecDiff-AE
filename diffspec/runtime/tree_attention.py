"""Tree-attention runtime kernels used by DiffSpec target adapters."""

from __future__ import annotations
import math
import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    class _TritonStub:
        constexpr = object()

        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def cdiv(*_args, **_kwargs):
            raise RuntimeError("Triton is not available")

    triton = _TritonStub()
    tl = _TritonStub()
    TRITON_AVAILABLE = False


__all__ = [
    "TRITON_AVAILABLE",
    "prefix_attention_forward",
    "tree_attention_forward",
    "tree_attention_forward_naive",
    "compiled_tree_attention_reference",
]


def tree_attention_forward_naive(query_states, key_states, value_states, tree_mask, 
                  cache_lens, prefix_lse, bsz, q_len, num_heads, hidden_dim):
    tree_mask = tree_mask[:, :, -q_len:, -q_len:]
    tree_mask = (tree_mask == 0).to(torch.int8) # convert to 1 and 0

    softmax_scale = 1. / math.sqrt(hidden_dim)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.permute(0, 2, 3, 1)
    value_states = value_states.transpose(1, 2)
    
    attn_score = torch.matmul(query_states, key_states) * softmax_scale

    attn_score = attn_score.to(torch.float32)
    attn_score_tree_mask = tree_mask.expand(-1, num_heads, -1, -1)
    attn_score = attn_score.masked_fill(attn_score_tree_mask == 0, -float('inf'))
    attn_weight = torch.softmax(attn_score, dim=-1).to(query_states.dtype)
    current_out = torch.matmul(attn_weight, value_states).permute(0, 2, 1, 3)
    current_lse = attn_score.logsumexp(dim=-1, keepdim=True).transpose(1, 2)

    prefix_lse = prefix_lse.reshape(bsz, num_heads, q_len, -1).transpose(1, 2)

    weight = torch.nn.functional.sigmoid(prefix_lse - current_lse).to(query_states.dtype)
    return current_out, weight


@torch.compile()
def compiled_tree_attention_reference(query_states, key_states, value_states, tree_mask, 
                  cache_lens, prefix_lse, bsz, q_len, num_heads, hidden_dim):
    tree_mask = tree_mask[:, :, -q_len:, -q_len:]
    tree_mask = (tree_mask == 0).to(torch.int8) # convert to 1 and 0

    softmax_scale = 1. / math.sqrt(hidden_dim)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.permute(0, 2, 3, 1)
    value_states = value_states.transpose(1, 2)
    
    attn_score = torch.matmul(query_states, key_states) * softmax_scale

    attn_score = attn_score.to(torch.float32)
    attn_score_tree_mask = tree_mask.expand(-1, num_heads, -1, -1)
    attn_score = attn_score.masked_fill(attn_score_tree_mask == 0, -float('inf'))
    attn_weight = torch.softmax(attn_score, dim=-1).to(query_states.dtype)
    current_out = torch.matmul(attn_weight, value_states).permute(0, 2, 1, 3)
    current_lse = attn_score.logsumexp(dim=-1, keepdim=True).transpose(1, 2)

    prefix_lse = prefix_lse.reshape(bsz, num_heads, q_len, -1).transpose(1, 2)

    weight = torch.nn.functional.sigmoid(prefix_lse - current_lse).to(query_states.dtype)
    return current_out, weight

# =========================
# Triton kernel
# =========================
# Computes O and LSE for attention over the "current" block (M x N) using an arbitrary
# boolean/int8 mask (1=allowed, 0=blocked). Q, K, V are provided with arbitrary strides.
# Online softmax is used to avoid materializing the score matrix and to minimize memory.
#
# Shapes (per call):
#   Q: [B, M, H, D]
#   K: [B, N, H, D]
#   V: [B, N, H, D]
#   MASK: [B, M, N] (int8/bool) with 1 for allowed positions, 0 for disallowed
# Outputs:
#   O: [B, M, H, D]  (same dtype as inputs)
#   LSE: [B, M, H]   (float32)
#
# Grid:
#   pid_bh = B * H   (program_id(0))
#   pid_m  = ceil_div(M, BLOCK_M) (program_id(1))
#
# Notes for H100 (SM90):
#   - Use num_warps=4/8 depending on D.
#   - Use num_stages=4 for better pipelining on Hopper.

@triton.jit
def _attn_prefix_lse_kernel(
    Q_ptr, K_ptr, LSE_ptr,
    B, H, M, N, D,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_lb, stride_lh, stride_lm,  # lse strides: [B, H, M]
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    h = pid_bh % H
    b = pid_bh // H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + b * stride_qb + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    offs_n_init = tl.arange(0, BLOCK_N)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + offs_n_init
        k_ptrs = K_ptr + b * stride_kb + offs_n[:, None] * stride_kn + h * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(offs_n[None, :] < N, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, block_max)
        scores_exp = tl.exp(scores - m_i_new[:, None])
        l_i = l_i * tl.exp(m_i - m_i_new) + tl.sum(scores_exp, axis=1)
        m_i = m_i_new

    lse = tl.log(tl.maximum(l_i, 1e-20)) + m_i
    l_ptrs = LSE_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lm
    tl.store(l_ptrs, lse, mask=offs_m < M)


@triton.jit
def _attn_prefix_block_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    B, H, M, N, D,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_lb, stride_lh, stride_lm,  # lse strides: [B, H, M]
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    h = pid_bh % H
    b = pid_bh // H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + b * stride_qb + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    offs_n_init = tl.arange(0, BLOCK_N)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + offs_n_init

        k_ptrs = K_ptr + b * stride_kb + offs_n[:, None] * stride_kn + h * stride_kh + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + b * stride_vb + offs_n[:, None] * stride_vn + h * stride_vh + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.).to(tl.float32)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(offs_n[None, :] < N, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, block_max)
        scores_exp = tl.exp(scores - m_i_new[:, None])
        l_i_new = l_i * tl.exp(m_i - m_i_new) + tl.sum(scores_exp, axis=1)

        alpha = tl.where(l_i_new > 0, (l_i * tl.exp(m_i - m_i_new)) / l_i_new, 0.0)
        acc = acc * alpha[:, None] + tl.dot(scores_exp / tl.maximum(l_i_new[:, None], 1e-20), v)

        m_i = m_i_new
        l_i = l_i_new

    o_ptrs = O_ptr + b * stride_ob + offs_m[:, None] * stride_om + h * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))

    lse = tl.log(tl.maximum(l_i, 1e-20)) + m_i
    l_ptrs = LSE_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lm
    tl.store(l_ptrs, lse, mask=offs_m < M)


@triton.jit
def _attn_tree_block_kernel(
    Q_ptr, K_ptr, V_ptr, MASK_ptr, O_ptr, LSE_ptr,
    B, H, M, N, D,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_mb, stride_mm, stride_mn,  # mask strides: [B, M, N]
    stride_ob, stride_om, stride_oh, stride_od,
    stride_lb, stride_lm, stride_lh,  # lse strides: [B, M, H]
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    # derive b, h from flattened pid_bh
    h = pid_bh % H
    b = pid_bh // H

    # row indices (queries)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Q tile [BLOCK_M, D]
    q_ptrs = Q_ptr + b*stride_qb + offs_m[:, None]*stride_qm + h*stride_qh + offs_d[None, :]*stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.).to(tl.float32)

    # online softmax stats
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], tl.float32)                # running sum exp
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)       # accumulator for O

    offs_n_init = tl.arange(0, BLOCK_N)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + offs_n_init

        # K, V tiles [BLOCK_N, D]
        k_ptrs = K_ptr + b*stride_kb + offs_n[:, None]*stride_kn + h*stride_kh + offs_d[None, :]*stride_kd
        v_ptrs = V_ptr + b*stride_vb + offs_n[:, None]*stride_vn + h*stride_vh + offs_d[None, :]*stride_vd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.).to(tl.float32)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.).to(tl.float32)

        # scores = Q @ K^T -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)) * scale

        # load mask block [BLOCK_M, BLOCK_N] for (b, m_block, n_block)
        m_ptrs = MASK_ptr + b*stride_mb + offs_m[:, None]*stride_mm + offs_n[None, :]*stride_mn
        m_mask = tl.load(m_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0).to(tl.int1)
        scores = tl.where(m_mask, scores, -float("inf"))

        # online softmax update
        block_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, block_max)
        scores_exp = tl.exp(scores - m_i_new[:, None])
        l_i_new = l_i * tl.exp(m_i - m_i_new) + tl.sum(scores_exp, axis=1)

        # acc update
        alpha = tl.where(l_i_new > 0, (l_i * tl.exp(m_i - m_i_new)) / l_i_new, 0.0)
        acc = acc * alpha[:, None] + tl.dot(scores_exp / tl.maximum(l_i_new[:, None], 1e-20), v)

        m_i = m_i_new
        l_i = l_i_new

    # write O as [B, M, H, D]
    o = acc.to(tl.float32)
    o_ptrs = O_ptr + b*stride_ob + offs_m[:, None]*stride_om + h*stride_oh + offs_d[None, :]*stride_od
    tl.store(o_ptrs, o, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))

    # write LSE = log(l_i) + m_i  as [B, M, H]
    lse = tl.log(tl.maximum(l_i, 1e-20)) + m_i
    l_ptrs = LSE_ptr + b*stride_lb + offs_m*stride_lm + h*stride_lh
    tl.store(l_ptrs, lse, mask=offs_m < M)


def _get_num_warps(D: int) -> int:
    # Heuristic tuned for H100
    if D >= 192:
        return 8
    elif D >= 96:
        return 4
    else:
        return 2


def _get_block_sizes(D: int):
    # Favor square-ish tiles; adjust for larger D to keep register pressure manageable
    if D >= 192:
        return 64, 64, min(128, D)
    elif D >= 128:
        return 64, 64, min(128, D)
    else:
        return 64, 64, min(64, D)

def _tree_attn_triton(q, k, v, mask, scale):
    """
    q: [B, M, H, D]
    k: [B, N, H, D]
    v: [B, N, H, D]
    mask: [B, M, N] int8/bool (1 allowed, 0 blocked)
    Returns:
      o: [B, M, H, D] dtype same as q
      lse: [B, M, H] float32
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and mask.is_cuda
    B, M, H, D = q.shape
    Bk, N, Hk, Dk = k.shape
    assert B == Bk and H == Hk and D == Dk, f"Incompatible shapes: q={q.shape}, k={k.shape}, v={v.shape}"
    assert mask.shape == (B, M, N), f"mask must be [B, M, N], got {mask.shape}"
    # allocate outputs
    o = torch.empty_like(q)
    lse = torch.empty((B, M, H), dtype=torch.float32, device=q.device)

    # set up launch grid
    BLOCK_M, BLOCK_N, BLOCK_D = _get_block_sizes(D)
    num_warps = _get_num_warps(D)
    num_stages = 4

    grid = (B * H, triton.cdiv(M, BLOCK_M))

    _attn_tree_block_kernel[grid](
        q, k, v, mask, o, lse,
        B, H, M, N, D,
        *q.stride(), *k.stride(), *v.stride(),
        *mask.stride(),
        *o.stride(),
        *lse.stride(),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o, lse


def _prefix_attn_triton(q, k, v, scale):
    """
    q: [B, M, H, D]
    k/v: [B, N, H, D]
    Returns:
      o: [B, M, H, D]
      lse: [B, H, M], matching flash-attn's return_softmax_lse layout.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    B, M, H, D = q.shape
    Bk, N, Hk, Dk = k.shape
    assert B == Bk and H == Hk and D == Dk, f"Incompatible shapes: q={q.shape}, k={k.shape}, v={v.shape}"

    o = torch.empty_like(q)
    lse = torch.empty((B, H, M), dtype=torch.float32, device=q.device)

    BLOCK_M, BLOCK_N, BLOCK_D = _get_block_sizes(D)
    num_warps = _get_num_warps(D)
    grid = (B * H, triton.cdiv(M, BLOCK_M))

    _attn_prefix_block_kernel[grid](
        q, k, v, o, lse,
        B, H, M, N, D,
        *q.stride(), *k.stride(), *v.stride(),
        *o.stride(),
        *lse.stride(),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=4,
    )
    return o, lse


def _prefix_lse_triton(q, k, scale):
    """
    q: [B, M, H, D]
    k: [B, N, H, D]
    Returns LSE as [B, H, M], matching flash-attn's return_softmax_lse layout.
    """
    assert q.is_cuda and k.is_cuda
    B, M, H, D = q.shape
    Bk, N, Hk, Dk = k.shape
    assert B == Bk and H == Hk and D == Dk, f"Incompatible shapes: q={q.shape}, k={k.shape}"

    lse = torch.empty((B, H, M), dtype=torch.float32, device=q.device)
    BLOCK_M, BLOCK_N, BLOCK_D = _get_block_sizes(D)
    num_warps = _get_num_warps(D)
    grid = (B * H, triton.cdiv(M, BLOCK_M))

    _attn_prefix_lse_kernel[grid](
        q, k, lse,
        B, H, M, N, D,
        *q.stride(), *k.stride(), *lse.stride(),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=4,
    )
    return lse


def prefix_attention_forward(query_states, key_states, value_states, bsz, q_len, num_heads, hidden_dim):
    """
    FlashAttention-compatible prefix attention fallback with LSE.

    This is used only when flash-attn is not installed. It avoids materializing
    the [B, H, Q, K] score tensor that the PyTorch fallback needs for LSE.
    """
    softmax_scale = 1.0 / math.sqrt(hidden_dim)
    use_triton_prefix = os.environ.get("DIFFSPEC_USE_TRITON_PREFIX", "1") != "0"
    if use_triton_prefix and TRITON_AVAILABLE and query_states.is_cuda:
        use_flash_output = os.environ.get("DIFFSPEC_PREFIX_FLASH_OUTPUT", "1") != "0"
        if use_flash_output:
            try:
                from .attention_compat import flash_attn_func

                out = flash_attn_func(query_states, key_states, value_states, causal=False)
                lse = _prefix_lse_triton(query_states, key_states, softmax_scale)
                return out, lse
            except Exception:
                pass
        try:
            return _prefix_attn_triton(query_states, key_states, value_states, softmax_scale)
        except Exception:
            pass

    q = query_states.transpose(1, 2).float()
    k = key_states.transpose(1, 2).float()
    v = value_states.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    weights = torch.softmax(scores, dim=-1).to(value_states.dtype)
    out = torch.matmul(weights, v).transpose(1, 2)
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def _maybe_to_int8_mask(tree_mask, bsz, num_heads, q_len):
    # tree_mask expected shape (bsz, 1, total_len, total_len) or (bsz, 1, q_len, q_len) already
    # We select the lower-right q_len x q_len block and convert to {0,1} int8 with 1 meaning "keep"
    if tree_mask.dim() == 3:
        mask_block = tree_mask[:, -q_len:, -q_len:]
    else:
        mask_block = tree_mask[:, :, -q_len:, -q_len:]
    if mask_block.dtype in (torch.bool, torch.int8, torch.uint8):
        mask_block = mask_block.to(torch.int8)
    else:
        mask_block = (mask_block == 0).to(torch.int8)
    return mask_block.view(bsz, q_len, q_len)

def tree_attention_forward(query_states, key_states, value_states, tree_mask,
                         cache_lens, prefix_lse, bsz, q_len, num_heads, hidden_dim):
    """
    High-performance attention on the "current" (tree-masked) block using Triton on H100.

    Args:
        query_states: [B, M, H, D] (q_flash)
        key_states:   [B, M, H, D] (post-cache segment)
        value_states: [B, M, H, D]
        tree_mask:    [B, 1, total_len, total_len]; only the last MxM block is used
        cache_lens:   (unused here; kept for signature compatibility)
        prefix_lse:   [B, M, H] or [B, M, H, 1] from flash_attn_with_kvcache(return_softmax_lse=True)
        bsz, q_len, num_heads, hidden_dim: ints

    Returns:
        current_out: [B, M, H, D]  (same dtype as inputs)
        weight:      [B, M, H, 1] (same dtype as inputs) = sigmoid(prefix_lse - current_lse)
    """
    B, M, H, D = bsz, q_len, num_heads, hidden_dim
    assert query_states.shape[:3] == (B, M, H) and query_states.shape[3] == D, \
        f"query_states shape {query_states.shape} != {(B, M, H, D)}"
    assert key_states.shape == (B, M, H, D) and value_states.shape == (B, M, H, D), \
        "key/value must be post-cache block"

    # Prepare mask [B, M, N] with N=M here
    mask_block = _maybe_to_int8_mask(tree_mask, B, H, M)

    # scale
    softmax_scale = 1.0 / math.sqrt(D)

    # Call Triton kernel; fallback to PyTorch if anything goes wrong.
    if TRITON_AVAILABLE and query_states.is_cuda:
        try:
            current_out, current_lse = _tree_attn_triton(
                query_states, key_states, value_states, mask_block, softmax_scale
            )
        except Exception:
            current_out, weight = tree_attention_forward_naive(
                query_states,
                key_states,
                value_states,
                tree_mask,
                cache_lens,
                prefix_lse,
                bsz,
                q_len,
                num_heads,
                hidden_dim,
            )
            return current_out, weight
    else:
        current_out, weight = tree_attention_forward_naive(
            query_states,
            key_states,
            value_states,
            tree_mask,
            cache_lens,
            prefix_lse,
            bsz,
            q_len,
            num_heads,
            hidden_dim,
        )
        return current_out, weight
    

    # Normalize prefix_lse shape to [B, M, H]
    # if prefix_lse.dim() == 4 and prefix_lse.size(-1) == 1:
    #     prefix_lse_ = prefix_lse.squeeze(-1)
    # else:
    #     prefix_lse_ = prefix_lse
    # prefix_lse_ = prefix_lse_.view(B, M, H).to(torch.float32)
    # Match the reference layout: reshape to [B, H, M, 1], then transpose to [B, M, H].
    prefix_lse_ = prefix_lse.reshape(B, num_heads, q_len, -1).transpose(1, 2).squeeze(-1).to(torch.float32)


    # weight = sigmoid(prefix_lse - current_lse) -> [B, M, H, 1]
    weight = torch.sigmoid(prefix_lse_ - current_lse).to(query_states.dtype).unsqueeze(-1)

    return current_out, weight
