"""
Chunk Arena for physically contiguous working-set storage.

Implements the DiffSpec Chunk Arena concept from Section 3.1. The arena
pre-allocates a contiguous KV buffer for the current top-k chunks, stages
only changed chunks during updates, and exposes optional APW hooks for L2
residency experiments.
"""

import torch
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChunkArenaConfig:
    """Configuration for :class:`ChunkArena`."""
    max_chunks: int = 64
    chunk_size: int = 64
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    enable_apw: bool = False
    apw_hit_ratio: float = 0.5


class ChunkArena:
    """Manage a contiguous KV arena for the active chunk working set.

    The arena maintains a ``chunk_id -> arena_offset`` mapping and updates
    only the chunks that enter or leave the top-k working set. This converts
    sparse logical chunk selection into physically contiguous memory access.
    
    Args:
        config: Arena configuration.
    """
    
    def __init__(self, config: ChunkArenaConfig):
        self.config = config
        self.max_chunks = config.max_chunks
        self.chunk_size = config.chunk_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.device = config.device
        self.dtype = config.dtype
        
        # Pre-allocate the contiguous arena buffer.
        # Shape: [num_layers, 2, 1, num_heads, max_chunks*chunk_size, head_dim]
        self.arena_kv = torch.zeros(
            config.num_layers,
            2,  # K and V
            1,  # batch
            config.num_heads,
            config.max_chunks * config.chunk_size,
            config.head_dim,
            dtype=config.dtype,
            device=config.device
        )
        
        # chunk_id -> arena_offset
        self.chunk_to_offset: Dict[int, int] = {}
        
        # Current resident chunks in arena order.
        self.active_chunks: List[int] = []
        
        # Free arena slots.
        self.free_offsets: List[int] = list(range(config.max_chunks))
        
        # Running statistics.
        self.total_updates = 0
        self.total_evictions = 0
        self.total_additions = 0
        self.hit_count = 0
        self.miss_count = 0
        
        # APW support. The Python class exposes hooks; a CUDA extension is
        # required to set the actual access-policy window.
        self.enable_apw = config.enable_apw
        self.apw_hit_ratio = config.apw_hit_ratio
        self.apw_enabled_flag = False
    
    def update_arena(
        self, 
        new_top_k_chunks: List[int],
        full_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """Differentially update the arena working set.
        
        Args:
            new_top_k_chunks: Top-k chunk ids ordered by importance.
            full_kv_cache: Optional full-history KV cache. Each layer is a
                ``(K, V)`` tuple with shape
                ``[batch, num_heads, seq_len, head_dim]``.
        
        Returns:
            Contiguous arena view with shape
            ``[layers, 2, 1, heads, active_len, dim]``.
        """
        self.total_updates += 1
        
        new_top_k_chunks = new_top_k_chunks[:self.max_chunks]
        
        old_set = set(self.active_chunks)
        new_set = set(new_top_k_chunks)
        
        # Differential update
        to_evict = old_set - new_set
        to_add = new_set - old_set
        
        self.hit_count += len(old_set & new_set)
        self.miss_count += len(to_add)
        
        # Step 1: Evict chunks
        for chunk_id in to_evict:
            if chunk_id in self.chunk_to_offset:
                offset = self.chunk_to_offset.pop(chunk_id)
                self.free_offsets.append(offset)
                self.active_chunks.remove(chunk_id)
                self.total_evictions += 1
        
        # Step 2: Add new chunks
        if full_kv_cache is not None:
            for chunk_id in to_add:
                if not self.free_offsets:
                    if self.active_chunks:
                        oldest_chunk = self.active_chunks.pop(0)
                        offset = self.chunk_to_offset.pop(oldest_chunk)
                        self.free_offsets.append(offset)
                    else:
                        raise RuntimeError("Arena overflow and no chunks to evict")
                
                offset = self.free_offsets.pop(0)
                self.chunk_to_offset[chunk_id] = offset
                self.active_chunks.append(chunk_id)
                self.total_additions += 1
                
                # Copy chunk KV from full cache to arena
                chunk_start = chunk_id * self.chunk_size
                chunk_end = chunk_start + self.chunk_size
                
                for layer_idx in range(self.num_layers):
                    if layer_idx < len(full_kv_cache):
                        k_cache, v_cache = full_kv_cache[layer_idx]
                        if k_cache.dim() == 3:
                            k_cache = k_cache.unsqueeze(0)
                        if v_cache.dim() == 3:
                            v_cache = v_cache.unsqueeze(0)
                        if k_cache.size(1) != self.num_heads:
                            raise RuntimeError(
                                f"ChunkArena head mismatch: arena_heads={self.num_heads}, cache_heads={k_cache.size(1)}"
                            )
                        
                        actual_end = min(chunk_end, k_cache.size(2))
                        actual_len = actual_end - chunk_start
                        
                        if actual_len > 0:
                            # Copy K
                            self.arena_kv[
                                layer_idx, 0, :, :, 
                                offset*self.chunk_size:offset*self.chunk_size+actual_len, 
                                :
                            ] = k_cache[:, :, chunk_start:actual_end, :]
                            
                            # Copy V
                            self.arena_kv[
                                layer_idx, 1, :, :,
                                offset*self.chunk_size:offset*self.chunk_size+actual_len,
                                :
                            ] = v_cache[:, :, chunk_start:actual_end, :]
        
        # The current implementation returns the active prefix directly. A
        # later compaction pass can reorder slots to exactly match top-k order.
        active_len = len(self.active_chunks) * self.chunk_size
        return self.arena_kv[:, :, :, :, :active_len, :]
    
    def get_chunk_kv(
        self, 
        chunk_id: int,
        layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return K/V tensors for a resident chunk.
        
        Returns:
            ``(K, V)`` or ``None`` if the chunk is not resident. Each tensor
            has shape ``[1, num_heads, chunk_size, head_dim]``.
        """
        if chunk_id not in self.chunk_to_offset:
            return None
        
        offset = self.chunk_to_offset[chunk_id]
        start = offset * self.chunk_size
        end = start + self.chunk_size
        
        k = self.arena_kv[layer_idx, 0, :, :, start:end, :]
        v = self.arena_kv[layer_idx, 1, :, :, start:end, :]
        
        return k, v
    
    def get_working_cache(self) -> torch.Tensor:
        """Return the contiguous active working-cache view.
        
        Returns:
            Tensor with shape ``[layers, 2, 1, heads, active_len, dim]``.
        """
        active_len = len(self.active_chunks) * self.chunk_size
        return self.arena_kv[:, :, :, :, :active_len, :]
    
    def get_chunk_mapping(self) -> Dict[int, int]:
        """Return a copy of the current chunk-to-offset mapping."""
        return self.chunk_to_offset.copy()
    
    def enable_l2_residency(self, stream=None):
        """Enable CUDA L2 residency hints through an access-policy window.
        
        Args:
            stream: Optional CUDA stream. The current Python implementation
                keeps this hook for the CUDA extension path.
        """
        if not self.enable_apw:
            return
        
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available; skipping APW setup.", RuntimeWarning, stacklevel=2)
            return
        
        try:
            base_ptr = self.arena_kv.data_ptr()
            num_bytes = self.arena_kv.numel() * self.arena_kv.element_size()
            _ = base_ptr
            
            # The actual APW setup requires a C++/CUDA extension. PyTorch does
            # not expose cudaStreamSetAttribute directly.
            """
            cudaStreamAttrValue stream_attr;
            stream_attr.accessPolicyWindow.base_ptr = (void*)base_ptr;
            stream_attr.accessPolicyWindow.num_bytes = num_bytes;
            stream_attr.accessPolicyWindow.hitRatio = self.apw_hit_ratio;
            stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
            """
            
            self.apw_enabled_flag = True
            warnings.warn(
                f"APW hook marked enabled for a {num_bytes/(1024**2):.1f} MB arena; "
                "install the CUDA extension to apply the stream attribute.",
                RuntimeWarning,
                stacklevel=2,
            )
            
        except Exception as e:
            warnings.warn(f"Failed to enable APW: {e}", RuntimeWarning, stacklevel=2)
    
    def disable_l2_residency(self, stream=None):
        """Disable the APW residency marker."""
        if not self.apw_enabled_flag:
            return
        
        try:
            # cudaCtxResetPersistingL2Cache() or set num_bytes=0
            self.apw_enabled_flag = False
        except Exception as e:
            warnings.warn(f"Failed to disable APW: {e}", RuntimeWarning, stacklevel=2)
    
    def get_statistics(self) -> Dict:
        """Return arena update and residency statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'total_updates': self.total_updates,
            'total_evictions': self.total_evictions,
            'total_additions': self.total_additions,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'active_chunks': len(self.active_chunks),
            'max_chunks': self.max_chunks,
            'utilization': len(self.active_chunks) / self.max_chunks,
            'arena_size_mb': (
                self.arena_kv.numel() * self.arena_kv.element_size() / (1024**2)
            ),
            'apw_enabled': self.apw_enabled_flag
        }
    
    def reset(self):
        """Reset arena contents while preserving aggregate counters."""
        self.chunk_to_offset.clear()
        self.active_chunks.clear()
        self.free_offsets = list(range(self.max_chunks))
        self.arena_kv.zero_()
        
    def compact(self):
        """Compact active chunks into contiguous arena slots."""
        if len(self.active_chunks) == 0:
            return
        
        new_chunk_to_offset = {}
        new_arena = torch.zeros_like(self.arena_kv)
        
        for new_offset, chunk_id in enumerate(self.active_chunks):
            old_offset = self.chunk_to_offset[chunk_id]
            new_chunk_to_offset[chunk_id] = new_offset
            
            # Copy the resident chunk into its compacted slot.
            old_start = old_offset * self.chunk_size
            old_end = old_start + self.chunk_size
            new_start = new_offset * self.chunk_size
            new_end = new_start + self.chunk_size
            
            new_arena[:, :, :, :, new_start:new_end, :] = \
                self.arena_kv[:, :, :, :, old_start:old_end, :]
        
        self.arena_kv = new_arena
        self.chunk_to_offset = new_chunk_to_offset
        self.free_offsets = list(range(len(self.active_chunks), self.max_chunks))


class ChunkArenaWithAPW(ChunkArena):
    """Chunk Arena variant reserved for CUDA-extension APW integration."""
    
    def __init__(self, config: ChunkArenaConfig):
        super().__init__(config)
        self.apw_stream = None
        
        if self.enable_apw:
            self._init_cuda_extension()
    
    def _init_cuda_extension(self):
        """Initialize the CUDA extension when it is available."""
        try:
            # import chunk_arena_cuda
            # self.cuda_ext = chunk_arena_cuda
            pass
        except ImportError:
            warnings.warn("CUDA extension is not available; APW disabled.", RuntimeWarning, stacklevel=2)
            self.enable_apw = False
    
    def set_l2_cache_policy(
        self, 
        hit_ratio: float = 0.5,
        persisting_size_mb: Optional[float] = None
    ):
        """Set the L2 cache policy through the CUDA extension.
        
        Args:
            hit_ratio: Target APW hit ratio.
            persisting_size_mb: Optional L2 persisting-window size in MB.
        """
        if not self.enable_apw:
            return
        
        if persisting_size_mb is None:
            persisting_size_mb = self.get_statistics()['arena_size_mb']
        
        # self.cuda_ext.set_l2_policy(
        #     self.arena_kv.data_ptr(),
        #     int(persisting_size_mb * 1024 * 1024),
        #     hit_ratio
        # )
