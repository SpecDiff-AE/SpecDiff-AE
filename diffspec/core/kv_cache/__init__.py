"""
KV-cache management components.

Exports paged KV-cache storage and Chunk Arena helpers used by the
DiffSpec runtime.
"""

from .paged_kv import PagedKVCache, PagedKVConfig
from .chunk_arena import ChunkArena, ChunkArenaConfig

__all__ = [
    'PagedKVCache',
    'PagedKVConfig',
    'ChunkArena',
    'ChunkArenaConfig',
]
