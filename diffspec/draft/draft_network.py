# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
import csv
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import time
import json
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN

from ..runtime.draft_tree import DraftTree



from termcolor import colored

from ..runtime.attention_compat import flash_attn_func


# [MODIFIED] Import from transformer library
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers import LlamaConfig

logger = logging.get_logger(__name__)

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # Assume cos, sin shape: (1, 1, seq_len, head_dim)
    # and x is shape: (batch, num_heads, seq_len, head_dim)
    # Flatten the batch dimension of position_ids: shape (seq_len,)
    idx = position_ids.squeeze(0)
    # Index along the sequence length dimension (dim=2)
    cos_indexed = torch.index_select(cos, dim=2, index=idx)  # shape: (1, 1, seq_len, head_dim)
    sin_indexed = torch.index_select(sin, dim=2, index=idx)
    
    # Optionally expand to match x's number of heads if needed.
    # For example, if x has shape (B, H, seq_len, head_dim) and cos_indexed is (1, 1, seq_len, head_dim),
    # then:
    cos_indexed = cos_indexed.expand(x.size(0), x.size(1), -1, -1)
    sin_indexed = sin_indexed.expand(x.size(0), x.size(1), -1, -1)
    
    x_embed = (x * cos_indexed) + (rotate_half(x) * sin_indexed)
    return x_embed


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# Find dim range bounds based on rotations
def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case

def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class LlamaYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn(device)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(x.dtype), persistent=False)
        return (
            # Match the 4D layout expected by `apply_rotary_pos_emb_single`
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0),
        )

    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mscale = float(_yarn_get_mscale(self.scale) * self.attn_factor) # Get n-d magnitude scaling corrected for interpolation



class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )




class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        
        self.scaling = self.head_dim**-0.5


        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            rope_cfg = self.config.rope_scaling if isinstance(self.config.rope_scaling, dict) else {}
            # Be tolerant of configs that omit `type` (some YaRN exports only include factor/max positions)
            scaling_type = rope_cfg.get("type")
            if scaling_type is None:
                # Heuristic: presence of YaRN-specific keys or rope_type flag -> treat as YaRN
                if rope_cfg.get("rope_type", "").lower() == "yarn" or any(
                    k in rope_cfg for k in ["original_max_position_embeddings", "beta_fast", "beta_slow", "attn_factor", "extrapolation_factor"]
                ):
                    scaling_type = "yarn"
                else:
                    scaling_type = "linear"
            scaling_factor = rope_cfg.get("factor", rope_cfg.get("scaling_factor", 1.0))
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "yarn":
                original_max_position_embeddings = rope_cfg.get("original_max_position_embeddings", self.max_position_embeddings)
                beta_fast = rope_cfg.get("beta_fast", 32)
                beta_slow = rope_cfg.get("beta_slow", 1)
                attn_factor = rope_cfg.get("attn_factor", 1)
                extrapolation_factor = rope_cfg.get("extrapolation_factor", 1)
                finetuned = rope_cfg.get("finetuned", False)
                base_theta = getattr(self.config, "rope_theta", 10000)
                self.rotary_emb = LlamaYaRNScaledRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=base_theta,
                    scale=scaling_factor,
                    original_max_position_embeddings=original_max_position_embeddings,
                    extrapolation_factor=extrapolation_factor,
                    attn_factor=attn_factor,
                    beta_fast=beta_fast,
                    beta_slow=beta_slow,
                    finetuned=finetuned,
                    device=self.q_proj.weight.device,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_position_ids: bool = False,
        init: bool = False,
        draft_use_flash_prefill = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Prefill
        if init:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            kv_seq_len = key_states.shape[-2]
            
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            
            
            # Concatenate and return BEFORE rotating
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            query_position_ids = torch.arange(
                        kv_seq_len - q_len,  # past_length
                        kv_seq_len,           # past_length + q_len
                        device=key_states.device
                    ).unsqueeze(0)
            key_position_ids = torch.arange(0, kv_seq_len, device=key_states.device).unsqueeze(0)

            past_key_position_ids = key_position_ids
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if draft_use_flash_prefill:
                query_states = query_states.transpose(1,2)
                key_states = key_states.transpose(1,2)
                value_states = value_states.transpose(1,2)

                attn_output = flash_attn_func(query_states, key_states, value_states, 
                                              window_size=(512,-1),
                                              causal=True)
                
                attn_output = attn_output.contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)
                
            else:
                if query_states.device.type == "cuda" and attention_mask is not None:
                    query_states = query_states.contiguous()
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()

                is_causal = True if attention_mask is None and q_len > 1 else False

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask.to(dtype=query_states.dtype),
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
                
                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)

        # Init forward / tree attention
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                cache_len = past_key_value[0].shape[-2]
                kv_seq_len += cache_len
                

            # Concatenate and return BEFORE rotating
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # for prefill / init forward
            if position_ids is None:
                query_position_ids = torch.arange(
                            kv_seq_len - q_len,  # past_length
                            kv_seq_len,           # past_length + q_len
                            device=key_states.device
                        ).unsqueeze(0)
                key_position_ids = torch.arange(0, kv_seq_len, device=key_states.device).unsqueeze(0)

            # for tree drafting 
            else:
                # [MODIFIED] Enhanced position IDs handling for tree drafting
                query_position_ids = position_ids.unsqueeze(0) if position_ids.dim() == 1 else position_ids
                
                # Build key_position_ids correctly
                if past_key_position_ids is not None:
                    # past_key_position_ids should contain positions for ALL tokens in cache
                    past_kv_len = past_key_position_ids.shape[1]
                    if past_kv_len + query_position_ids.shape[1] == kv_seq_len:
                        # Normal case: concatenate past and current positions
                        key_position_ids = torch.cat([past_key_position_ids, query_position_ids], dim=1)
                    else:
                        # Mismatch case: rebuild key_position_ids
                        print(f"Warning: Position IDs length mismatch. Past: {past_kv_len}, Query: {query_position_ids.shape[1]}, Total KV: {kv_seq_len}")
                        # Generate consecutive position IDs starting from 0
                        key_position_ids = torch.arange(0, kv_seq_len, device=key_states.device).unsqueeze(0)
                        # Update the query part with provided position_ids if they make sense
                        if query_position_ids.max() < kv_seq_len:
                            key_position_ids[0, -query_position_ids.shape[1]:] = query_position_ids.squeeze(0)
                else:
                    # No past position IDs: generate from scratch
                    key_position_ids = torch.arange(0, kv_seq_len, device=key_states.device).unsqueeze(0)
                    # Use provided position_ids for the query part if reasonable
                    if query_position_ids.max() < kv_seq_len and query_position_ids.shape[1] <= kv_seq_len:
                        start_pos = kv_seq_len - query_position_ids.shape[1]
                        key_position_ids[0, start_pos:] = query_position_ids.squeeze(0)

            past_key_position_ids = key_position_ids
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                # [MODIFIED] Defensive fix for attention mask size mismatch
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(f"Attention mask has incorrect size: got {attention_mask.size()}, but expected {(bsz, 1, q_len, kv_seq_len)}. Slicing to match.")
 

                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, past_key_position_ids


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.last = last
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_position_ids=None,
        init: bool = False,
        draft_use_flash_prefill = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # if self.index != 0:
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        
        # print(colored("input_emb max", "red"), input_emb.shape)
        # print(colored("hidden_states max", "red"), hidden_states.shape)
       
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        # Self Attention
        # start_time = time.time()
        hidden_states, self_attn_weights, present_key_value, past_key_position_ids = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            past_key_position_ids=past_key_position_ids,
            init=init,
            draft_use_flash_prefill=draft_use_flash_prefill
        )
        hidden_states = residual + hidden_states
        # end_time = time.time()
        # print("dfraft attention time is", end_time - start_time)
        # Fully Connected
        # start_time = time.time()
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # end_time = time.time()
        # print("dfraft FFN time is", end_time - start_time)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, past_key_position_ids

KV = Tuple[torch.Tensor, torch.Tensor]        # (K, V)
FullKV = List[KV]                              # per-layer list of (K,V)
WorkingKV = List[KV]

class DraftNetwork(nn.Module):
    def __init__(self,config,load_emb=False,path=None,bias=True):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head=nn.Linear(config.hidden_size,config.draft_vocab_size,bias=False)
       
        if load_emb and not hasattr(config, "target_hidden_size"):
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

  
        self.hidden_size = config.hidden_size
        self.midlayer = LlamaDecoderLayeremb(config)
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
        self.norm=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # [MODIFIED] Initialize d2t and t2d mappings
        d2t=torch.zeros((config.draft_vocab_size),dtype=torch.long)
        t2d=torch.zeros((config.vocab_size),dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        self.past_key_position_ids = None
        
        # Optional semantic redundancy penalty for retrieval scoring.
        self.gamma_sem = 0

        # Cached semantic chunk representations and previous selection.
        self.chunk_reps = None
        self.prev_selected_chunk_ids = []
        # Tensorized chunk descriptors on the draft device.
        self._chunks_se = None          # [N, 2] -> (start, end), long, device=self.device
        self._chunks_starts = None      # [N]
        self._chunks_ends = None        # [N]

        # Scratch buffers for interval-union index construction.
        self._diff_buf = None           # [full_cache_budget+1] int32, device=self.device
        self._ones_buf = None           # [max_chunks] int32 constant +1
        self._minus_ones_buf = None     # [max_chunks] int32 constant -1

        # Retrieval-overlap tracing.
        self.chunk_overlap_history = []
        self.chunk_overlap_steps = []
        self._chunk_overlap_event_idx = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                #inputs_embeds.dtype,
                torch.float32, # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        tree_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        init: bool = False,
        draft_use_flash_prefill=False
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
        
        if past_key_values is not None:
            # Each layer stores a (key_states, value_states) tuple with shape
            # [batch_size, num_heads, seq_len, head_dim].
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            
            
        if tree_attention_mask is None and init and draft_use_flash_prefill:
            attention_mask = None
        elif tree_attention_mask is None:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
            )
        else:
            attention_mask=tree_attention_mask

        inputs_embeds=inputs_embeds.to(hidden_states.dtype)
        
        # print(inputs_embeds.shape, hidden_states.shape)
        if hidden_states.shape[-1]!=inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None


        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs, past_key_position_ids = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True,
            init=init,
            draft_use_flash_prefill=draft_use_flash_prefill,
            past_key_position_ids=self.past_key_position_ids,

        )
        self.past_key_position_ids = past_key_position_ids
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        hidden_states = layer_outputs[0]
        
        
        
        if use_cache:
            return hidden_states,next_decoder_cache

        return hidden_states

    @torch.no_grad()
    def generate(self,hidden_states,input_ids,head,max_length=4,use_cache=False):
        return_input_ids=copy.deepcopy(input_ids[0].tolist())
        input_ids=input_ids[:,1:]

        #input_ids=input_ids.to(hidden_states.device)
        if use_cache:
            past_key_values=None
            for i in range(max_length):
                if past_key_values!=None:
                    out_hidden,past_key_values = self(out_hidden[:, -1:], input_ids=torch.tensor([[token]]).to(input_ids.device),past_key_values=past_key_values,use_cache=True)
                else:
                    out_hidden, past_key_values = self(hidden_states, input_ids=input_ids,use_cache=True)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                #input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                return_input_ids.append(token.item())
                if token == 2:
                    break
                #hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
        else:
            for i in range(max_length):
                out_hidden=self(hidden_states,input_ids=input_ids)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                return_input_ids.append(token.item())
                input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                if token==2:
                    break
                hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)

        return return_input_ids

    @torch.no_grad()
    def repeat_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0].repeat(numr,1,1,1),i[1].repeat(numr,1,1,1)))
        return tuple(newkv)

    @torch.no_grad()
    def reduce_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0][:numr],i[1][:numr]))
        return tuple(newkv)


    def reset_kv(self):
        self.draft_stable_kv=None
        


    # def process_tree_mask(self, tree_attention_mask, init_len):
    #     # [MODIFIED] Enhanced tree mask processing with better shape handling
    #     device = tree_attention_mask.device
    #     dtype = torch.float32
        
    #     # tree_attention_mask shape: (tree_size, tree_size)
    #     tree_size = tree_attention_mask.size(0)
        
    #     # Create attention mask for past KV cache (init_len tokens)
    #     # Past tokens can attend to all past tokens (causal mask)
    #     past_mask = torch.zeros((tree_size, init_len), dtype=dtype, device=device)
        
    #     # Convert tree mask: 0 -> -inf (cannot attend), 1 -> 0 (can attend)
    #     tree_mask = torch.where(tree_attention_mask == 0, 
    #                            torch.finfo(dtype).min, 
    #                            torch.tensor(0.0, dtype=dtype, device=device))
        
    #     # Combine past and tree masks: [tree_size, init_len + tree_size]
    #     full_mask = torch.cat([past_mask, tree_mask], dim=-1)
        
    #     # Add batch and head dimensions: [1, 1, tree_size, init_len + tree_size]
    #     full_mask = full_mask.unsqueeze(0).unsqueeze(0)
        
    #     return full_mask
    
    def process_tree_mask(self,tree_attention_mask,init_len):
        attention_mask=torch.full((tree_attention_mask.size(0), init_len), 0, device=tree_attention_mask.device)
        tree_mask = torch.where(tree_attention_mask == 0, torch.finfo(torch.float32).min, 0)
        attention_mask=torch.cat([attention_mask,tree_mask],dim=-1)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    @torch.no_grad()
    def build_speculative_tree(self, hidden_states, input_ids, head, nodes, threshold=0.5, max_depth=10):
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        len_posi = input_ids.shape[1]
      
        
        # Initial forward with draft model
        if hasattr(self, "draft_stable_kv") and self.draft_stable_kv is not None:

            if self.use_retrieval_cache:
                full_kv_len = self.total_seq_len
                out_hidden, past_key_values = self(
                    hidden_states, 
                    input_ids=input_ids[:, full_kv_len:],
                    past_key_values=self.draft_stable_kv, 
                    use_cache=True,
                    draft_use_flash_prefill = self.draft_use_flash_prefill
                    )
            else:
                kv_len = self.draft_stable_kv[0][0].shape[2]
                out_hidden, past_key_values = self(
                    hidden_states, 
                    input_ids=input_ids[:, kv_len:],
                    past_key_values=self.draft_stable_kv, 
                    use_cache=True,
                    draft_use_flash_prefill = self.draft_use_flash_prefill
                    )
        # Prefill draft model
        else:
 
            out_hidden, past_key_values = self(hidden_states, 
                                            input_ids=input_ids, 
                                            use_cache=True,
                                            
                                            init=True,
                                            
                                            draft_use_flash_prefill=self.draft_use_flash_prefill)
        if self.use_retrieval_cache:
            newly_appended_len = input_ids.shape[-1] - self.total_seq_len
            self.update_full_draft_cache(past_key_values, tokens_appended=newly_appended_len)
            self.draft_stable_kv = self.refresh_working_cache(top_k_chunks=self.retrieve_top_k)
        else:
            self.draft_stable_kv = past_key_values
        
        past_key_values=self.draft_stable_kv
        
        # new total length of kv cache after initial forward
   
        init_len=past_key_values[0][0].size(2)

        if self.use_retrieval_cache:
            target_model_pos_diff = len_posi - (init_len - 1) 

        last_headout = self.lm_head(self.norm(out_hidden[:, -1]))
        last_hidden = out_hidden[:, -1]
        # print(colored(f"DEBUG last_hidden.shape={last_hidden.shape}", 'magenta'))
        # if not self.diff_device:
        #     last_headout = head(last_hidden)
        # else:
        #     if hasattr(self, "layer_device"):
        #         last_headout = head(last_hidden)
        #         last_headout = last_headout.to(self.layer_device)
        #     else:
        #         last_headout = F.linear(last_hidden, self.headweight)

        # print(colored(f"DEBUG after head last_headout.shape={last_headout.shape}", 'magenta'))
        # print(colored(f"DEBUG new_headout={self.lm_head(self.norm(out_hidden[:, -1])).shape}", 'magenta'))
        hazard_tracker = (
            getattr(self, "hazard_tracker", None)
            if getattr(self, "enable_hazard_profile", False)
            else None
        )
        tree = DraftTree(nodes, hidden_states.device, threshold, max_depth, hazard_tracker=hazard_tracker)
        logits = last_headout.unsqueeze(0)
        # print(colored(f"DEBUG logits.shape={logits.shape}", 'magenta'))
        step = 0
        # start = time.perf_counter()
        # torch.cuda.synchronize()
        while True:
            tree_output = tree.update(
                torch.softmax(logits.to(hidden_states.device), dim = -1, dtype = torch.float32)
            )
            # tree_output = tree.update(
            #     self.logsoftmax(logits.to(hidden_states.device))
            # )

            # [MODIFIED] Apply d2t mapping for vocabulary space conversion
            # The Tree returns token indices in draft model's vocabulary space.
            # When draft_vocab_size != vocab_size, we need to map them to target vocabulary space.
            draft_input_ids = tree_output["input_ids"]
            if self.config.vocab_size == self.config.draft_vocab_size:
                input_ids = draft_input_ids.unsqueeze(0)
            else:
                # Map from draft vocabulary space to target vocabulary space
                input_ids = (draft_input_ids + self.d2t[draft_input_ids]).unsqueeze(0)

            if self.use_retrieval_cache:
                # [MODIFIED] Enhanced position calculation for retrieval cache
                # tree_output["position_ids"] contains relative positions within tree
                # Need to add offset to make them absolute positions
                position_ids = tree_output["position_ids"] + (init_len - 1)
            else:
                # [MODIFIED] Enhanced position calculation for normal cache
                # For normal cache, use the original length + tree positions
                position_ids = tree_output["position_ids"] + len_posi

            if tree_output["is_final"]:
                break
            tree_attention_mask_with_kv=self.process_tree_mask(tree_output["attention_mask"],init_len)

            if step==0:
                hidden_states=last_hidden.repeat(1,nodes,1)
            else:
                hidden_states=out_hidden[:,tree_output["parent_last"],:]

            # tree attention with draft model (pass last hidden states)
            out_hidden, past_key_values = self(hidden_states, 
                                               input_ids=input_ids,
                                               tree_attention_mask=tree_attention_mask_with_kv,
                                               past_key_values=past_key_values,
                                            #    position_ids=position_ids-1, 
                                               position_ids=position_ids - 1,
                                               use_cache=True,
                                               draft_use_flash_prefill = self.draft_use_flash_prefill
                                               )

            # if not self.diff_device:
            #     last_headout = head(out_hidden[0])
            # else:
            #     if hasattr(self, "layer_device"):
            #         last_headout = head(out_hidden[0])
            #         last_headout = last_headout.to(self.layer_device)
            #     else:
            #         last_headout = F.linear(out_hidden[0], self.headweight)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            logits = last_headout.unsqueeze(0)
            step += 1
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start
        # print(f'build_speculative_tree took {elapsed} seconds')
        if self.use_retrieval_cache:
            position_ids += target_model_pos_diff

        # [MODIFIED] Apply d2t mapping to final output input_ids 
        # Ensure the final returned tokens are in target vocabulary space
        final_draft_input_ids = tree_output["input_ids"]
        if self.config.vocab_size == self.config.draft_vocab_size:
            final_input_ids = final_draft_input_ids.unsqueeze(0)
        else:
            # Map from draft vocabulary space to target vocabulary space
            final_input_ids = (final_draft_input_ids + self.d2t[final_draft_input_ids]).unsqueeze(0)

        return final_input_ids, position_ids, tree_output["attention_mask"], tree_output["parent_last"]

    @torch.no_grad()
    def _compute_chunk_rep(self, start: int, end: int) -> torch.Tensor:
        """
        Build a unit-norm chunk representation from the last-layer key cache.

        Invalid or empty chunks return a zero vector.
        """
        if end <= start:
            return torch.zeros(self.full_draft_kv[-1][0].size(-1), device='cpu')
        if getattr(self, "enable_salience_encoding", False) and getattr(self, "chunk_encoder", None) is not None:
            full_K = self.full_draft_kv[-1][0][0]  # [H, L, Dh]
            full_V = self.full_draft_kv[-1][1][0]  # [H, L, Dh]
            if end - start <= 0:
                return torch.zeros(full_K.size(-1), device='cpu')
            proxy_k, _ = self.chunk_encoder.encode_chunks_batch(
                full_K,
                full_V,
                [(0, start, end)]
            )
            rep = proxy_k[:, 0, :].mean(dim=0).float()
        else:
            full_K = self.full_draft_kv[-1][0]                      # K: [1, H, L, Dh]
            rep = full_K[0, :, start:end, :].mean(dim=(0, 1)).float()  # [Dh]
        rep = rep / (rep.norm(p=2) + 1e-6)
        return rep.cpu()

    def _ensure_chunk_reps_init(self):
        """Rebuild semantic chunk representations after chunk metadata changes."""
        if getattr(self, "chunks", None) is None:
            self.chunk_reps = None
            return
        if (self.chunk_reps is None) or (len(self.chunk_reps) != len(self.chunks)):
            self.chunk_reps = []
            for (_, s, e) in self.chunks:
                self.chunk_reps.append(self._compute_chunk_rep(int(s), int(e)))

    def _update_last_chunk_rep(self):
        """Refresh the representation of the most recently extended chunk."""
        if not self.chunks or self.chunk_reps is None:
            return
        _, s, e = self.chunks[-1]
        self.chunk_reps[-1] = self._compute_chunk_rep(int(s), int(e))
    
    
    
    
    def update_full_draft_cache(self, new_kv: List[Tuple[torch.Tensor, torch.Tensor]], tokens_appended: int):
        """
        Update the full draft KV cache with the new tokens.
        new_kv is the returned KV from the forward pass (a working-cache view).
        tokens_appended is the number of new tokens processed in this forward pass.
        
        The full cache is preallocated with size self.full_cache_budget, and
        self.total_seq_len tracks the current number of tokens stored.
        This function copies the last tokens_appended tokens from new_kv (from the working view)
        into the full cache.
        """
        # Check that we don't exceed the allocated budget.
        if self.total_seq_len + tokens_appended > self.full_cache_budget:
            raise RuntimeError(
                f"Full cache budget exceeded: total_seq_len {self.total_seq_len} + new {tokens_appended} > {self.full_cache_budget}"
            )

        # Precompute destination slice indices.
        dest_start = self.total_seq_len
        dest_end = dest_start + tokens_appended
        device = self.midlayer.self_attn.q_proj.weight.device

        # For each layer in the new KV, copy the last tokens_appended tokens into the full cache.
        for i, (new_K, new_V) in enumerate(new_kv):
            full_K, full_V = self.full_draft_kv[i]
            # Ensure new_K and new_V are on the correct device.
            new_K = new_K.to(device, non_blocking=True)
            new_V = new_V.to(device, non_blocking=True)
            # Copy the last tokens_appended tokens from new_K/new_V into the full cache.
            full_K[:, :, dest_start:dest_end, :].copy_(new_K[:, :, -tokens_appended:, :])
            full_V[:, :, dest_start:dest_end, :].copy_(new_V[:, :, -tokens_appended:, :])
        
        # Update the total sequence length.
        self.total_seq_len = dest_end

        if getattr(self, "enable_paged_kv", False) and getattr(self, "paged_kv", None) is not None:
            seq_id = getattr(self, "paged_kv_seq_id", 0)
            for layer_idx, (new_K, new_V) in enumerate(new_kv):
                for offset in range(tokens_appended):
                    token_idx = dest_start + offset
                    k = new_K[0, :, -tokens_appended + offset, :]
                    v = new_V[0, :, -tokens_appended + offset, :]
                    self.paged_kv.write_kv(seq_id, layer_idx, token_idx, k, v)


    def update_working_cache_from_full(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build the working cache (self.draft_stable_kv) by indexing into the full cache.
        The working cache is defined to be the concatenation of:
        - the first sink_size tokens (the "sink" region), and 
        - the last recent_size tokens (the "recent" region)
        If total_seq_len is less than sink_size+recent_size, simply use all tokens.
        Also updates self.evicted to be the number of tokens that are outside the working window.
        """
        working_kv = []
        if getattr(self, "enable_paged_kv", False) and getattr(self, "paged_kv", None) is not None:
            if self.total_seq_len <= self.sink_size + self.recent_size:
                indices = torch.arange(0, self.total_seq_len, device=self.device, dtype=torch.long)
            else:
                sink = torch.arange(0, self.sink_size, device=self.device, dtype=torch.long)
                recent = torch.arange(self.total_seq_len - self.recent_size, self.total_seq_len,
                                      device=self.device, dtype=torch.long)
                indices = torch.cat([sink, recent], dim=0)
            working_kv = self._build_working_kv_from_indices(indices)
        else:
            for (full_K, full_V) in self.full_draft_kv:
                if self.total_seq_len <= self.sink_size + self.recent_size:
                    working_K_layer = full_K[:, :, :self.total_seq_len, :].clone()
                    working_V_layer = full_V[:, :, :self.total_seq_len, :].clone()
                    # print(colored(f'Using cache regions: {0}~{self.total_seq_len-1}','magenta'))
                else:
                    sink_part_K = full_K[:, :, :self.sink_size, :].clone()
                    sink_part_V = full_V[:, :, :self.sink_size, :].clone()
                    recent_part_K = full_K[:, :, self.total_seq_len - self.recent_size:self.total_seq_len, :].clone()
                    recent_part_V = full_V[:, :, self.total_seq_len - self.recent_size:self.total_seq_len, :].clone()
                    working_K_layer = torch.cat([sink_part_K, recent_part_K], dim=2)
                    working_V_layer = torch.cat([sink_part_V, recent_part_V], dim=2)
                    # print(colored(f'Using cache regions: {0}~{self.sink_size-1}, {self.total_seq_len - self.recent_size}~{self.total_seq_len-1}','magenta'))
                working_kv.append((working_K_layer, working_V_layer))
        self.evicted = max(self.total_seq_len - (self.sink_size + self.recent_size), 0)
        working_kv_len = working_kv[0][0].shape[2]
        # print(colored(f'Working KV length: {working_kv_len}','red'))
        
        # truncate Draft model's past_key_position_ids:
         # this is when the prefill chunk size is smaller than the working cache size (only right after prefill)
        self.past_key_position_ids = (
            torch.cat([self.past_key_position_ids,
                    torch.arange(self.past_key_position_ids.shape[1], working_kv_len, device=self.past_key_position_ids.device).unsqueeze(0)],
                    dim=1)
            if self.past_key_position_ids.shape[1] < working_kv_len
            else self.past_key_position_ids
        )[:, :working_kv_len]

        self.recent_start = max(0, self.total_seq_len - self.recent_size)
        self.recent_end = self.total_seq_len-1
        return working_kv

    def _build_working_kv_from_indices(self, indices: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Build the working KV cache from selected token indices."""
        if getattr(self, "enable_paged_kv", False) and getattr(self, "paged_kv", None) is not None:
            seq_id = getattr(self, "paged_kv_seq_id", 0)
            index_list = indices.detach().cpu().tolist()
            working_kv = []
            for layer_idx in range(len(self.full_draft_kv)):
                k_tokens = []
                v_tokens = []
                for token_idx in index_list:
                    k, v = self.paged_kv.read_kv(seq_id, layer_idx, int(token_idx))
                    k_tokens.append(k)
                    v_tokens.append(v)
                if k_tokens:
                    k_layer = torch.stack(k_tokens, dim=1).unsqueeze(0)  # [1, H, L, D]
                    v_layer = torch.stack(v_tokens, dim=1).unsqueeze(0)
                else:
                    head_dim = self.full_draft_kv[layer_idx][0].size(-1)
                    num_heads = self.full_draft_kv[layer_idx][0].size(1)
                    k_layer = torch.zeros(1, num_heads, 0, head_dim, device=self.device)
                    v_layer = torch.zeros(1, num_heads, 0, head_dim, device=self.device)
                working_kv.append((k_layer, v_layer))
            return working_kv

        working_kv = []
        for (full_K, full_V) in self.full_draft_kv:
            working_K_layer = full_K.index_select(dim=2, index=indices)
            working_V_layer = full_V.index_select(dim=2, index=indices)
            working_kv.append((working_K_layer, working_V_layer))
        return working_kv

    def refresh_working_cache(self, top_k_chunks: int = 15):
        """
        Convenience function that first updates the chunk metadata (if new tokens were appended)
        and then updates the working cache (self.draft_stable_kv) based on retrieval.
        
        It assumes that self.attn_scores_final is already set (e.g., computed from the last accepted query)
        and that self.total_seq_len has been updated by update_full_draft_cache.
        """
        is_updated_chunks = self.update_chunks()
        working_kv = self._select_working_cache_from_chunks(
            top_k_chunks=top_k_chunks,
            do_retrieval=self.retrieval_condition,
            is_updated_chunks=is_updated_chunks,
        )
        return working_kv

    def update_chunks(self):
        """
        Called after new tokens have been appended to the full KV cache.
        self.total_seq_len has been updated externally (by update_full_draft_cache).
        This function updates self.chunks to reflect the new total length.
        
        It does so by:
        1) Filling the last chunk (if not already full) with some of the new tokens.
        2) Creating new chunk(s) (each of size self.retrieval_chunk_size, except possibly the last one)
            for any remaining new tokens.
        """
        # Calculate how many new tokens were appended.
        new_tokens = self.total_seq_len - self.seq_len_total_old
        if new_tokens <= 0:
            return  # No new tokens; nothing to do.

        # If there are no chunks yet, create them from scratch.
        if not hasattr(self, "chunks") or self.chunks is None or len(self.chunks) == 0:
            self.prepare_chunks()
            if getattr(self, "gamma_sem", 0.0) > 0:
                self._ensure_chunk_reps_init()
            return True

        # Get the last chunk's info.
        last_chunk_idx, last_start, last_end = self.chunks[-1]
        last_chunk_size = last_end - last_start
        remaining_new_tokens = new_tokens

        # 1) If the last chunk is not full, fill it up as much as possible.
        capacity = self.retrieval_chunk_size - last_chunk_size
        if capacity > 0:
            tokens_to_add = min(capacity, remaining_new_tokens)
            # Update the last chunk's end index.
            self.chunks[-1] = (last_chunk_idx, last_start, last_end + tokens_to_add)
            remaining_new_tokens -= tokens_to_add
    
        
        
        
        
        
        
        
        
        
        
        
        
        current_start = self.total_seq_len - remaining_new_tokens
        while remaining_new_tokens > 0:
            tokens_in_chunk = min(self.retrieval_chunk_size, remaining_new_tokens)
            new_chunk = (self.chunks[-1][0] + 1, current_start, current_start + tokens_in_chunk)
            self.chunks.append(new_chunk)
            current_start += tokens_in_chunk
            remaining_new_tokens -= tokens_in_chunk

        # Update the stored old full length.
        self.seq_len_total_old = self.total_seq_len
        self.num_chunks = len(self.chunks)


        # Rebuild tensorized chunk descriptors after metadata changes.
        self._chunks_se = torch.tensor(
            [[start, end] for (_, start, end) in self.chunks],
            dtype=torch.long, device=self.device
        )
        self._chunks_starts = self._chunks_se[:, 0]
        self._chunks_ends   = self._chunks_se[:, 1]

        if self.num_chunks > self.num_chunks_old:
            self.num_chunks_old = self.num_chunks
            self._update_last_chunk_rep()
            return True # new chunk was added => update working cache
        return False

    def prepare_chunks(self):
        """
        Called once (right after prefill) to split the full cache (of length self.total_seq_len)
        into consecutive chunks of fixed size (self.retrieval_chunk_size). Each chunk is represented as a tuple:
        (chunk_idx, start, end) where end - start <= self.retrieval_chunk_size.
        """
        self.chunks = []
        current_start = 0
        chunk_idx = 0
        while current_start < self.total_seq_len:
            end_pos = min(current_start + self.retrieval_chunk_size, self.total_seq_len)
            self.chunks.append((chunk_idx, current_start, end_pos))
            chunk_idx += 1
            current_start = end_pos
        # Save the current full length so that later we know how many new tokens were appended.
        self.seq_len_total_old = self.total_seq_len
        self.num_chunks = len(self.chunks)
        self.num_chunks_old = self.num_chunks
        # Materialize the Python chunk list as resident tensors.
        self._chunks_se = torch.tensor(
            [[start, end] for (_, start, end) in self.chunks],
            dtype=torch.long, device=self.device
        )
        self._chunks_starts = self._chunks_se[:, 0]
        self._chunks_ends   = self._chunks_se[:, 1]

        # Scratch buffers and constant update vectors.
        buf_len = int(self.full_cache_budget) + 1
        if (self._diff_buf is None) or (self._diff_buf.numel() < buf_len):
            self._diff_buf = torch.zeros(buf_len, dtype=torch.int32, device=self.device)

        
        
        max_possible_indices = max(getattr(self, 'retrieve_top_k', 64) + 4, 64)
        if (self._ones_buf is None) or (self._ones_buf.numel() < max_possible_indices):
            self._ones_buf       = torch.ones(max_possible_indices, dtype=torch.int32, device=self.device)
            self._minus_ones_buf = -torch.ones(max_possible_indices, dtype=torch.int32, device=self.device)

    @torch.no_grad()
    def _select_working_cache_from_chunks(
        self,
        top_k_chunks: int = 15,
        do_retrieval: bool = False,
        is_updated_chunks: bool = False
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select chunks, build the interval union, and gather the working KV cache.

        Retrieval mode scores chunks with attention prefix sums and an optional
        semantic redundancy penalty. Non-retrieval mode extends the previously
        selected chunk set.
        """
        assert self.full_draft_kv is not None, "full_draft_kv has not been initialized"
        L = int(self.total_seq_len)

        # Initialize selected chunks on the first refresh.
        if not hasattr(self, "selected_chunks") or len(self.selected_chunks) == 0:
            num_init = min(self.retrieve_top_k, len(self.chunks))
            self.selected_chunks = self.chunks[-num_init:]

        # Infer the device used for retrieval scoring.
        def _infer_dev():
            if getattr(self, "attn_scores_final", None) is not None:
                return self.attn_scores_final.device
            return self.full_draft_kv[0][0].device
        dev = _infer_dev()

        # Keep chunk boundaries on the scoring device.
        if getattr(self, "_chunks_starts", None) is None or getattr(self, "_chunks_ends", None) is None:
            se = torch.tensor([[s, e] for (_, s, e) in self.chunks], dtype=torch.long, device=dev)
            self._chunks_starts, self._chunks_ends = se[:, 0], se[:, 1]
        else:
            self._chunks_starts = self._chunks_starts.to(dev)
            self._chunks_ends   = self._chunks_ends.to(dev)
        starts: torch.Tensor = self._chunks_starts  # [N]
        ends:   torch.Tensor = self._chunks_ends    # [N]
        N = int(starts.numel())

        # Ensure scratch buffers are on the scoring device and large enough.
        need_len = L + 1
        if (getattr(self, "_diff_buf", None) is None) or (self._diff_buf.device != dev) or (self._diff_buf.numel() < need_len):
            self._diff_buf = torch.zeros(need_len, dtype=torch.int32, device=dev)
        max_possible_indices = max(getattr(self, 'retrieve_top_k', 64) + 4, 64)
        if (getattr(self, "_ones_buf", None) is None) or (self._ones_buf.device != dev) or (self._ones_buf.numel() < max_possible_indices):
            self._ones_buf       = torch.ones (max_possible_indices, dtype=torch.int32, device=dev)
            self._minus_ones_buf = -torch.ones(max_possible_indices, dtype=torch.int32, device=dev)

        retrieved_indices: torch.Tensor

        if do_retrieval:
            attn = None
            if getattr(self, "attn_scores_final", None) is not None:
                attn = self.attn_scores_final

            if attn is None:
                do_retrieval = False
            else:
                attn = attn.to(dev)
                L_attn = int(attn.numel())
                if L_attn == 0:
                    do_retrieval = False
                else:
                    cum = torch.cumsum(attn, dim=0)  # [L_attn]

                    lower = torch.zeros_like(starts, dtype=cum.dtype, device=dev)
                    mask_pos = starts > 0
                    if bool(mask_pos.any()):
                        lower_idx = (starts[mask_pos] - 1).clamp_(min=0, max=L_attn - 1)
                        lower[mask_pos] = cum.index_select(0, lower_idx)

                    ends_m1 = (ends - 1).clamp_(min=0, max=L_attn - 1)
                    upper = cum.index_select(0, ends_m1)

                    chunk_sums  = upper - lower
                    lengths     = (ends - starts).clamp_min(1).to(chunk_sums.dtype)
                    chunk_means = chunk_sums / lengths

                    score = chunk_means
                    if getattr(self, "gamma_sem", 0.0) > 0:
                        if hasattr(self, "_ensure_chunk_reps_init"):
                            self._ensure_chunk_reps_init()
                        if getattr(self, "chunk_reps", None) is not None:
                            reps_all = torch.stack(self.chunk_reps, dim=0).to(score.dtype).to(dev)  # [N,d]
                            if getattr(self, "prev_selected_chunk_ids", None) and len(self.prev_selected_chunk_ids) > 0:
                                sel_ids = torch.tensor(self.prev_selected_chunk_ids, device=dev, dtype=torch.long)
                                sel = reps_all.index_select(0, sel_ids)                             # [m,d]
                                sem_red = (reps_all @ sel.t()).amax(dim=1).clamp_min(0.0)           # [N]
                                score = score - self.gamma_sem * sem_red.to(score.dtype)

                    k = min(int(top_k_chunks), N) if N > 0 else 0
                    if k <= 0:
                        return self.update_working_cache_from_full()
                    selected_indices = torch.topk(score, k=k, dim=0).indices
                    selected_indices, _ = torch.sort(selected_indices)
                    new_chunk_ids = selected_indices.tolist()
                    self._record_chunk_overlap_ratio(self.prev_selected_chunk_ids, new_chunk_ids)
                    self.prev_selected_chunk_ids = new_chunk_ids
                    self.selected_chunks = [self.chunks[i] for i in self.prev_selected_chunk_ids]

                    diff = self._diff_buf[: L + 1]
                    diff.zero_()
                    sel_st = starts.index_select(0, selected_indices).clamp_(min=0, max=L)  # ∈[0,L]
                    sel_en = ends.index_select(0, selected_indices).clamp_(min=0, max=L)    # ∈[0,L]
                    diff.index_add_(0, sel_st, self._ones_buf[: sel_st.numel()])
                    diff.index_add_(0, sel_en, self._minus_ones_buf[: sel_en.numel()])
                    mask_u = torch.cumsum(diff[:L], dim=0) > 0
                    retrieved_indices = torch.nonzero(mask_u, as_tuple=False).squeeze(1).to(torch.long)

                    # Clear one-shot retrieval inputs.
                    self.retrieval_condition = False
                    self.attn_scores = None
                    self.attn_scores_final = None

        if not do_retrieval:
            if is_updated_chunks and len(self.chunks) > 0:
                new_chunk = self.chunks[-1]
                new_id = new_chunk[0]
                if new_id not in {cid for cid, _, _ in self.selected_chunks}:
                    self.selected_chunks.append(new_chunk)

            if len(self.selected_chunks) > 0 and len(self.chunks) > 0:
                if self.selected_chunks[-1][0] == self.chunks[-1][0]:
                    cid, s, _ = self.selected_chunks[-1]
                    self.selected_chunks[-1] = (cid, s, self.chunks[-1][2])

            if len(self.selected_chunks) == 0:
                return self.update_working_cache_from_full()

            sel_st = torch.tensor([s for (_, s, e) in self.selected_chunks], dtype=torch.long, device=dev).clamp_(min=0, max=L)
            sel_en = torch.tensor([e for (_, s, e) in self.selected_chunks], dtype=torch.long, device=dev).clamp_(min=0, max=L)

            diff = self._diff_buf[: L + 1]
            diff.zero_()
            diff.index_add_(0, sel_st, self._ones_buf[: sel_st.numel()])
            diff.index_add_(0, sel_en, self._minus_ones_buf[: sel_en.numel()])
            mask_u = torch.cumsum(diff[:L], dim=0) > 0
            retrieved_indices = torch.nonzero(mask_u, as_tuple=False).squeeze(1).to(torch.long)

        if retrieved_indices.numel() == 0:
            return self.update_working_cache_from_full()

        chunk_arena = getattr(self, "chunk_arena", None)
        if getattr(chunk_arena, "backend_name", None) == "amd_rocm":
            working_view = chunk_arena.update(self.selected_chunks, self.full_draft_kv)
            working_kv = chunk_arena.as_kv_list()
            self.draft_stable_kv = working_kv

            working_len = int(working_view.size(4))
            self.evicted = self.total_seq_len - working_len
            past_ids = self.past_key_position_ids
            T_now = past_ids.shape[1]
            if T_now < working_len:
                extra = torch.arange(T_now, working_len, device=past_ids.device).unsqueeze(0)
                self.past_key_position_ids = torch.cat([past_ids, extra], dim=1)
            else:
                self.past_key_position_ids = past_ids[:, :working_len]

            if getattr(self, "retrieval_verbose", False) and hasattr(self, "print_retrieved_chunks"):
                if do_retrieval:
                    self.print_retrieved_chunks(order="id")
            return working_kv

        working_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for (full_K, full_V) in self.full_draft_kv:
            working_K_layer = full_K.index_select(dim=2, index=retrieved_indices).contiguous()
            working_V_layer = full_V.index_select(dim=2, index=retrieved_indices).contiguous()
            working_kv.append((working_K_layer, working_V_layer))
        self.draft_stable_kv = working_kv

        self.evicted = self.total_seq_len - retrieved_indices.numel()
        past_ids = self.past_key_position_ids  # [1, T]
        T_now = past_ids.shape[1]
        T_need = retrieved_indices.numel()
        if T_now < T_need:
            extra = torch.arange(T_now, T_need, device=past_ids.device).unsqueeze(0)
            self.past_key_position_ids = torch.cat([past_ids, extra], dim=1)
        else:
            self.past_key_position_ids = past_ids[:, :T_need]

        if getattr(self, "retrieval_verbose", False) and hasattr(self, "print_retrieved_chunks"):
            if do_retrieval:
                self.print_retrieved_chunks(order="id")

        return working_kv


    def print_retrieved_chunks(self, order="id"):
        if order == "score":
            chunks_list = self.selected_chunks
            msg = "\nRetrieved chunk IDs (descending attn score): "
        elif order == "id":
            chunks_list = sorted(self.selected_chunks, key=lambda x: x[0])
            msg = "\nRetrieved chunk IDs: "
        else:
            print(colored(f"Unknown 'order' option: {order}. Choose 'score' or 'id'.", 'red'))
            return

        chunk_ids_str = ", ".join(str(chunk[0]) for chunk in chunks_list)

        print(colored(msg + chunk_ids_str + '\n', 'yellow'))

    def _record_chunk_overlap_ratio(self, prev_chunk_ids, new_chunk_ids):
        """
        Record how much consecutive retrieval selections overlap.

        The ratio is the share of previously selected chunks that remain in the
        current retrieval result.
        """
        if not prev_chunk_ids:
            return

        prev_set = set(prev_chunk_ids)
        if not prev_set:
            return

        current_set = set(new_chunk_ids)
        overlap_ratio = len(prev_set & current_set) / len(prev_set)

        self._chunk_overlap_event_idx += 1
        self.chunk_overlap_steps.append(self._chunk_overlap_event_idx)
        self.chunk_overlap_history.append(overlap_ratio)

    def plot_chunk_overlap_history(self, save_path=None, show=False):
        """
        Plot retrieval-overlap history.

        Args:
            save_path (str, optional): Output path for the plot.
            show (bool): Whether to call plt.show(); defaults to False.
        """
        if not self.chunk_overlap_history:
            print(colored("No retrieval overlap data recorded yet. Trigger retrieval at least twice to collect data.", "yellow"))
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required to plot the overlap history. Please install it via `pip install matplotlib`.") from exc

        steps = self.chunk_overlap_steps or list(range(1, len(self.chunk_overlap_history) + 1))
        overlap_avg = sum(self.chunk_overlap_history) / len(self.chunk_overlap_history)

        plt.figure(figsize=(8, 4))
        plt.plot(steps, self.chunk_overlap_history, marker="o", linewidth=1.8, label="Overlap")
        plt.axhline(overlap_avg, color="red", linestyle="--", linewidth=1.2,
                    label=f"Average = {overlap_avg:.3f}")
        plt.xlabel("Retrieval step")
        plt.ylabel("Chunk overlap rate")
        plt.ylim(0.0, 1.05)
        plt.title("Chunk Selection Overlap Between Consecutive Retrievals")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="best")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def save_chunk_overlap_history(self, save_path):
        """
        Save retrieval-overlap history as a CSV file.
        """
        if not self.chunk_overlap_history:
            print(colored("No retrieval overlap data to save.", "yellow"))
            return

        steps = self.chunk_overlap_steps or list(range(1, len(self.chunk_overlap_history) + 1))
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "overlap_rate"])
            writer.writerows(zip(steps, self.chunk_overlap_history))

        print(colored(f"Chunk-overlap data saved to {save_path}", "green"))

    def reset_chunk_overlap_history(self):
        """Clear recorded retrieval-overlap statistics."""
        self.chunk_overlap_history = []
        self.chunk_overlap_steps = []
        self._chunk_overlap_event_idx = 0
