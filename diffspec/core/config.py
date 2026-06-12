"""Configuration objects for DiffSpec runtime components."""

from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class DiffSpecConfig:
    """Runtime configuration for DiffSpec components."""
    
    # ========== Salience-Aware Encoding (§2.1) ==========
    enable_salience_encoding: bool = False
    """Enable salience-aware chunk encoding."""
    
    chunk_size: int = 32
    """Chunk size."""
    
    top_k_chunks: int = 32
    """Number of chunks selected by retrieval."""
    
    # ========== Rejection-Hazard Profile (§2.2) ==========
    enable_hazard_profile: bool = True
    """Enable hazard-guided dynamic tree construction."""
    
    hazard_window_size: int = 50
    """Sliding-window length used by the hazard estimator."""
    
    hazard_alpha: float = 0.2
    """Exponential moving average factor for hazard updates."""
    
    tree_node_budget: int = 48
    """Total draft-tree node budget."""
    
    max_tree_depth: int = 6
    """Maximum draft-tree depth."""
    
    # ========== Chunk Arena (§3.1) ==========
    enable_chunk_arena: bool = False
    """Enable Chunk Arena storage."""
    
    max_arena_chunks: int = 64
    """Maximum number of chunks stored in the arena."""
    
    enable_apw_residency: bool = False
    """Enable access-policy-window L2 residency hints."""
    
    apw_hit_ratio: float = 0.5
    """Target hit ratio for the APW window."""
    
    # ========== Bundle Scheduling (§3.2) ==========
    enable_bundle_scheduling: bool = False
    """Enable bundle scheduling."""
    
    bundle_size: int = 3
    """Fixed bundle size."""
    
    max_bundles: int = 100
    """Maximum number of bundles."""
    
    # ========== Paged KV (§3.2) ==========
    enable_paged_kv: bool = False
    """Enable paged KV-cache storage."""
    
    page_size: int = 16
    """KV-cache page size."""
    
    max_pages: int = 1024
    """Maximum number of KV-cache pages."""
    
    enable_cow: bool = True
    """Enable copy-on-write for paged KV cache."""
    
    # ========== Fused Kernels (§3.2) ==========
    enable_fused_kernel: bool = False
    """Enable fused tree verification kernels."""
    
    enable_triton: bool = True
    """Enable Triton kernels where available."""
    
    # ========== Model Configuration ==========
    num_layers: int = 32
    """Number of model layers."""
    
    num_heads: int = 32
    """Number of attention heads."""
    
    head_dim: int = 128
    """Per-head hidden dimension."""
    
    vocab_size: int = 32000
    """Vocabulary size."""
    
    # ========== Runtime Configuration ==========
    device: str = "cuda"
    """Runtime device."""
    
    dtype: str = "float16"
    """Runtime dtype: float16, bfloat16, or float32."""
    
    # ========== Profiling & Debug ==========
    enable_profiling: bool = False
    """Enable performance profiling."""
    
    verbose: bool = False
    """Enable verbose runtime logging."""
    
    def __post_init__(self):
        """Validate the configuration."""
        assert self.chunk_size > 0, "chunk_size must be positive"
        assert self.top_k_chunks <= self.max_arena_chunks, \
            "top_k_chunks must not exceed max_arena_chunks"
        assert 0 < self.hazard_alpha < 1, "hazard_alpha must be in (0, 1)"
        assert self.bundle_size > 0, "bundle_size must be positive"
        assert self.page_size > 0, "page_size must be positive"
        
        if self.enable_apw_residency and self.device != "cuda":
            warnings.warn(
                "APW/L2 residency requires a GPU device exposed through PyTorch's "
                "`cuda` namespace. This includes ROCm PyTorch on AMD GPUs; "
                "disabling enable_apw_residency for non-GPU device.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.enable_apw_residency = False
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'DiffSpecConfig':
        """Create a configuration from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Return the configuration as a plain dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def get_default(cls) -> 'DiffSpecConfig':
        """Return the default configuration."""
        return cls()
    
    @classmethod
    def get_minimal(cls) -> 'DiffSpecConfig':
        """Return a minimal configuration with optional optimizations disabled."""
        return cls(
            enable_salience_encoding=False,
            enable_hazard_profile=False,
            enable_chunk_arena=False,
            enable_apw_residency=False,
            enable_bundle_scheduling=False,
            enable_paged_kv=False,
            enable_fused_kernel=False,
            enable_triton=False
        )
    
    @classmethod
    def get_performance(cls) -> 'DiffSpecConfig':
        """Return the recommended high-throughput configuration.

        The stable path is relevance-aware retrieval plus hazard-guided tree
        construction. Arena, bundle, fused-kernel, and paged-KV features remain
        available for controlled experiments, but are not enabled by default.
        """
        return cls(
            enable_salience_encoding=False,
            enable_hazard_profile=True,
            enable_chunk_arena=False,
            enable_apw_residency=False,
            enable_bundle_scheduling=False,
            enable_paged_kv=False,
            enable_fused_kernel=False,
            enable_triton=True,
            chunk_size=32,
            top_k_chunks=32,
            tree_node_budget=16,
            max_tree_depth=4,
            bundle_size=3
        )
    
    @classmethod
    def get_memory_efficient(cls) -> 'DiffSpecConfig':
        """Return a memory-conscious configuration."""
        return cls(
            enable_salience_encoding=False,
            enable_hazard_profile=True,
            enable_chunk_arena=False,
            enable_apw_residency=False,
            enable_bundle_scheduling=False,
            enable_paged_kv=False,
            enable_fused_kernel=False,
            enable_triton=True,
            chunk_size=32,
            top_k_chunks=32,
            max_arena_chunks=32,
            tree_node_budget=16,
            max_tree_depth=4,
            page_size=16,
            max_pages=512
        )
    
    def summary(self) -> str:
        """Return a human-readable configuration summary."""
        lines = ["DiffSpec Configuration:"]
        lines.append("=" * 60)
        
        lines.append("Accuracy Optimizations:")
        lines.append(f"  Salience Encoding: {'enabled' if self.enable_salience_encoding else 'disabled'}")
        lines.append(f"  Hazard Profile: {'enabled' if self.enable_hazard_profile else 'disabled'}")
        if self.enable_hazard_profile:
            lines.append(f"    - Window size: {self.hazard_window_size}")
            lines.append(f"    - Node budget: {self.tree_node_budget}")
        
        lines.append("\nPerformance Optimizations:")
        lines.append(f"  Chunk Arena: {'enabled' if self.enable_chunk_arena else 'disabled'}")
        if self.enable_chunk_arena:
            lines.append(f"    - Max chunks: {self.max_arena_chunks}")
            lines.append(f"    - APW residency: {'enabled' if self.enable_apw_residency else 'disabled'}")
        lines.append(f"  Bundle Scheduling: {'enabled' if self.enable_bundle_scheduling else 'disabled'}")
        if self.enable_bundle_scheduling:
            lines.append(f"    - Bundle size: {self.bundle_size}")
        
        lines.append("\nMemory Optimizations:")
        lines.append(f"  Paged KV: {'enabled' if self.enable_paged_kv else 'disabled'}")
        if self.enable_paged_kv:
            lines.append(f"    - Page size: {self.page_size}")
            lines.append(f"    - Max pages: {self.max_pages}")
            lines.append(f"    - COW: {'enabled' if self.enable_cow else 'disabled'}")
        
        lines.append("\nKernel Optimizations:")
        lines.append(f"  Fused Kernel: {'enabled' if self.enable_fused_kernel else 'disabled'}")
        lines.append(f"  Triton: {'enabled' if self.enable_triton else 'disabled'}")
        
        # Runtime
        lines.append("\nRuntime:")
        lines.append(f"  Device: {self.device}")
        lines.append(f"  Dtype: {self.dtype}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


DEFAULT_CONFIG = DiffSpecConfig.get_default()
MINIMAL_CONFIG = DiffSpecConfig.get_minimal()
PERFORMANCE_CONFIG = DiffSpecConfig.get_performance()
MEMORY_EFFICIENT_CONFIG = DiffSpecConfig.get_memory_efficient()
