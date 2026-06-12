"""
Paged KV Cache with Copy-on-Write

Implements the DiffSpec paged KV substrate from Section 3.2:
fixed-size pages, logical-to-physical block mapping, copy-on-write prefix
sharing, and reference-counted reclamation.
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PagedKVConfig:
    """Configuration for :class:`PagedKVCache`."""
    page_size: int = 16
    max_pages: int = 1024
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    enable_cow: bool = True


class PagedKVCache:
    """Paged KV cache manager with optional copy-on-write sharing.

    The cache splits KV state into fixed-size pages, maps each sequence's
    logical pages to physical pages, and shares prefix pages across forked
    sequences until a write requires copy-on-write isolation.
    
    Args:
        config: Cache configuration.
    """
    
    def __init__(self, config: PagedKVConfig):
        self.config = config
        self.page_size = config.page_size
        self.max_pages = config.max_pages
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.device = config.device
        self.dtype = config.dtype
        if str(self.device).startswith("cpu") and self.dtype == torch.float16:
            self.dtype = torch.float32
            self.config.dtype = self.dtype
        self.enable_cow = config.enable_cow
        
        # Pre-allocate the physical page pool.
        # Shape: [num_layers, 2, max_pages, num_heads, page_size, head_dim]
        #        [layers,    K/V, pages,     heads,     tokens,    dim]
        self.page_pool = torch.zeros(
            config.num_layers,
            2,  # K and V
            config.max_pages,
            config.num_heads,
            config.page_size,
            config.head_dim,
            dtype=self.dtype,
            device=config.device
        )
        
        # Free physical page ids.
        self.free_pages = set(range(config.max_pages))
        
        # Page reference counts for COW.
        self.page_ref_count = torch.zeros(config.max_pages, dtype=torch.int32, device='cpu')
        
        # Logical-to-physical mappings: seq_id -> {logical_page: physical_page}.
        self.page_tables = {}
        
        # Allocation statistics.
        self.total_allocated = 0
        self.total_freed = 0
        self.cow_saves = 0
        
    def allocate_sequence(self, seq_id: int, num_tokens: int) -> Dict[int, int]:
        """Allocate pages for a new sequence.
        
        Args:
            seq_id: Sequence id.
            num_tokens: Number of tokens to reserve.
        
        Returns:
            Mapping from logical page index to physical page index.
        """
        num_pages = (num_tokens + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < num_pages:
            raise RuntimeError(
                f"Out of memory: need {num_pages} pages, "
                f"only {len(self.free_pages)} available"
            )
        
        page_table = {}
        for logical_idx in range(num_pages):
            physical_idx = self.free_pages.pop()
            page_table[logical_idx] = physical_idx
            self.page_ref_count[physical_idx] = 1
            self.total_allocated += 1
        
        self.page_tables[seq_id] = page_table
        
        return page_table
    
    def fork_sequence(
        self, 
        parent_seq_id: int, 
        child_seq_id: int,
        shared_prefix_len: int
    ):
        """Fork a child sequence that shares prefix pages with its parent.
        
        Args:
            parent_seq_id: Parent sequence id.
            child_seq_id: Child sequence id.
            shared_prefix_len: Shared prefix length in tokens.
        """
        if not self.enable_cow:
            self.copy_sequence(parent_seq_id, child_seq_id)
            return
        
        if parent_seq_id not in self.page_tables:
            raise ValueError(f"Parent sequence {parent_seq_id} not found")
        
        parent_table = self.page_tables[parent_seq_id]
        shared_pages = shared_prefix_len // self.page_size
        
        child_table = {}
        
        for logical_idx in range(shared_pages):
            if logical_idx in parent_table:
                physical_idx = parent_table[logical_idx]
                child_table[logical_idx] = physical_idx
                self.page_ref_count[physical_idx] += 1
                self.cow_saves += 1
        
        self.page_tables[child_seq_id] = child_table
    
    def copy_sequence(self, src_seq_id: int, dst_seq_id: int):
        """Copy an entire sequence without sharing pages."""
        if src_seq_id not in self.page_tables:
            raise ValueError(f"Source sequence {src_seq_id} not found")
        
        src_table = self.page_tables[src_seq_id]
        num_pages = len(src_table)
        
        dst_table = {}
        for logical_idx, src_physical_idx in src_table.items():
            if len(self.free_pages) == 0:
                raise RuntimeError("Out of memory")
            
            dst_physical_idx = self.free_pages.pop()
            dst_table[logical_idx] = dst_physical_idx
            self.page_ref_count[dst_physical_idx] = 1
            
            self.page_pool[:, :, dst_physical_idx] = \
                self.page_pool[:, :, src_physical_idx].clone()
            
            self.total_allocated += 1
        
        self.page_tables[dst_seq_id] = dst_table
    
    def write_kv(
        self, 
        seq_id: int, 
        layer_idx: int,
        token_idx: int,
        k: torch.Tensor,
        v: torch.Tensor
    ):
        """Write one token's K/V tensors.
        
        Args:
            seq_id: Sequence id.
            layer_idx: Layer index.
            token_idx: Logical token position.
            k: Key tensor [num_heads, head_dim]
            v: Value tensor [num_heads, head_dim]
        """
        if seq_id not in self.page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated")
        
        page_table = self.page_tables[seq_id]
        logical_page = token_idx // self.page_size
        offset = token_idx % self.page_size

        if k.shape[0] != self.num_heads or k.shape[1] != self.head_dim:
            raise RuntimeError(
                f"PagedKV k/v shape mismatch: expected ({self.num_heads}, {self.head_dim}), got {tuple(k.shape)}"
            )
        
        if logical_page not in page_table:
            if len(self.free_pages) == 0:
                raise RuntimeError("Out of memory")
            physical_page = self.free_pages.pop()
            page_table[logical_page] = physical_page
            self.page_ref_count[physical_page] = 1
            self.total_allocated += 1
        else:
            physical_page = page_table[logical_page]
            
            if self.enable_cow and self.page_ref_count[physical_page] > 1:
                if len(self.free_pages) == 0:
                    raise RuntimeError("Out of memory")
                
                new_physical_page = self.free_pages.pop()
                self.page_pool[:, :, new_physical_page] = \
                    self.page_pool[:, :, physical_page].clone()
                
                self.page_ref_count[physical_page] -= 1
                self.page_ref_count[new_physical_page] = 1
                
                page_table[logical_page] = new_physical_page
                physical_page = new_physical_page
                self.total_allocated += 1
        
        self.page_pool[layer_idx, 0, physical_page, :, offset, :] = k
        self.page_pool[layer_idx, 1, physical_page, :, offset, :] = v
    
    def read_kv(
        self, 
        seq_id: int, 
        layer_idx: int,
        token_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read one token's K/V tensors.
        
        Returns:
            Key and value tensors with shape ``[num_heads, head_dim]``.
        """
        if seq_id not in self.page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated")
        
        page_table = self.page_tables[seq_id]
        logical_page = token_idx // self.page_size
        offset = token_idx % self.page_size
        
        if logical_page not in page_table:
            return (
                torch.zeros(self.num_heads, self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
            )
        
        physical_page = page_table[logical_page]
        k = self.page_pool[layer_idx, 0, physical_page, :, offset, :]
        v = self.page_pool[layer_idx, 1, physical_page, :, offset, :]
        
        return k, v
    
    def get_contiguous_kv(
        self, 
        seq_id: int, 
        layer_idx: int,
        start_token: int = 0,
        end_token: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather a contiguous K/V range for attention.
        
        Args:
            seq_id: Sequence id.
            layer_idx: Layer index.
            start_token: Inclusive start token.
            end_token: Exclusive end token. When omitted, reads to the last
                allocated logical page.
        
        Returns:
            Key and value tensors with shape ``[num_heads, seq_len, head_dim]``.
        """
        if seq_id not in self.page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated")
        
        page_table = self.page_tables[seq_id]
        
        if end_token is None:
            max_logical_page = max(page_table.keys())
            end_token = (max_logical_page + 1) * self.page_size
        
        seq_len = end_token - start_token
        
        k_out = torch.zeros(
            self.num_heads, seq_len, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        v_out = torch.zeros(
            self.num_heads, seq_len, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        
        current_pos = 0
        for token_idx in range(start_token, end_token):
            logical_page = token_idx // self.page_size
            offset = token_idx % self.page_size
            
            if logical_page in page_table:
                physical_page = page_table[logical_page]
                k_out[:, current_pos, :] = \
                    self.page_pool[layer_idx, 0, physical_page, :, offset, :]
                v_out[:, current_pos, :] = \
                    self.page_pool[layer_idx, 1, physical_page, :, offset, :]
            
            current_pos += 1
        
        return k_out, v_out
    
    def free_sequence(self, seq_id: int):
        """Release all pages owned or referenced by a sequence."""
        if seq_id not in self.page_tables:
            return
        
        page_table = self.page_tables[seq_id]
        
        for physical_page in page_table.values():
            self.page_ref_count[physical_page] -= 1
            if self.page_ref_count[physical_page] == 0:
                self.free_pages.add(physical_page)
                self.total_freed += 1
        
        del self.page_tables[seq_id]
    
    def get_physical_blocks(self, seq_id: int) -> List[int]:
        """Return the physical block list used by kernels."""
        if seq_id not in self.page_tables:
            return []
        
        page_table = self.page_tables[seq_id]
        max_logical = max(page_table.keys()) if page_table else -1
        
        blocks = []
        for logical_idx in range(max_logical + 1):
            if logical_idx in page_table:
                blocks.append(page_table[logical_idx])
            else:
                blocks.append(-1)
        
        return blocks
    
    def get_statistics(self) -> Dict:
        """Return cache allocation and sharing statistics."""
        total_pages = self.max_pages
        used_pages = total_pages - len(self.free_pages)
        
        shared_pages = int((self.page_ref_count > 1).sum().item())
        
        return {
            'total_pages': total_pages,
            'used_pages': used_pages,
            'free_pages': len(self.free_pages),
            'utilization': used_pages / total_pages,
            'total_allocated': self.total_allocated,
            'total_freed': self.total_freed,
            'cow_saves': self.cow_saves,
            'shared_pages': shared_pages,
            'num_sequences': len(self.page_tables),
            'memory_mb': (
                self.page_pool.numel() * self.page_pool.element_size() / (1024**2)
            )
        }
    
    def defragment(self):
        """Move active pages toward a contiguous physical range."""
        used_pages = set()
        for page_table in self.page_tables.values():
            used_pages.update(page_table.values())
        
        used_pages = sorted(used_pages)
        
        if not used_pages or used_pages[-1] - used_pages[0] + 1 == len(used_pages):
            return
        
        remap = {}
        for new_idx, old_idx in enumerate(used_pages):
            if new_idx != old_idx:
                remap[old_idx] = new_idx
        
        if not remap:
            return
        
        temp_pool = self.page_pool.clone()
        for old_idx, new_idx in remap.items():
            self.page_pool[:, :, new_idx] = temp_pool[:, :, old_idx]
        
        for page_table in self.page_tables.values():
            for logical_idx, physical_idx in page_table.items():
                if physical_idx in remap:
                    page_table[logical_idx] = remap[physical_idx]
        
        new_ref_count = torch.zeros_like(self.page_ref_count)
        for old_idx, new_idx in remap.items():
            new_ref_count[new_idx] = self.page_ref_count[old_idx]
        for idx in used_pages:
            if idx not in remap:
                new_ref_count[idx] = self.page_ref_count[idx]
        self.page_ref_count = new_ref_count
        
        self.free_pages = set(range(self.max_pages)) - set(
            p for table in self.page_tables.values() for p in table.values()
        )
