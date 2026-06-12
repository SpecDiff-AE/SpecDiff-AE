"""High-level orchestration interface for DiffSpec components."""

import torch
from typing import Optional, Dict, List, Tuple
from .config import DiffSpecConfig
from .hazard_profile import HazardProfileTracker
from .bundle_scheduler import BundleScheduler, TreeDescriptorBuilder
from .kv_cache import PagedKVCache, PagedKVConfig, ChunkArena, ChunkArenaConfig
from ..runtime.bundle_verifier import FusedBundleVerifier
from ..amd import AmdArenaConfig, AmdChunkArena, AmdResidencyConfig, is_rocm_pytorch


class DiffSpecEngine:
    """DiffSpec inference engine.

    The engine wires together optional optimization components behind a compact
    API used by examples and unit tests.

    Example:
        >>> config = DiffSpecConfig.get_performance()
        >>> engine = DiffSpecEngine(config)
        >>> engine.initialize_session(input_ids)
        >>> # Draft phase
        >>> draft_tokens = engine.draft(...)
        >>> # Verify phase
        >>> accepted_tokens = engine.verify(draft_tokens, ...)
    """
    
    def __init__(self, config: DiffSpecConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self._init_components()
        
        self.stats = {
            'total_drafts': 0,
            'total_verifications': 0,
            'total_accepted': 0,
            'total_rejected': 0
        }
    
    def _init_components(self):
        """Initialize enabled optimization components."""
        
        if self.config.enable_hazard_profile:
            self.hazard_tracker = HazardProfileTracker(
                max_depth=self.config.max_tree_depth,
                window_size=self.config.hazard_window_size,
                alpha=self.config.hazard_alpha,
                device=self.config.device
            )
        else:
            self.hazard_tracker = None
        
        if self.config.enable_bundle_scheduling:
            self.bundle_scheduler = BundleScheduler(
                bundle_size=self.config.bundle_size,
                max_bundles=self.config.max_bundles,
                device=self.config.device
            )
        else:
            self.bundle_scheduler = None
        
        if self.config.enable_paged_kv:
            paged_config = PagedKVConfig(
                page_size=self.config.page_size,
                max_pages=self.config.max_pages,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                device=self.config.device,
                enable_cow=self.config.enable_cow
            )
            self.paged_kv = PagedKVCache(paged_config)
        else:
            self.paged_kv = None
        
        if self.config.enable_chunk_arena:
            if is_rocm_pytorch() and self.device.type == "cuda":
                arena_config = AmdArenaConfig(
                    max_chunks=self.config.max_arena_chunks,
                    chunk_size=self.config.chunk_size,
                    num_layers=self.config.num_layers,
                    num_heads=self.config.num_heads,
                    head_dim=self.config.head_dim,
                    device=self.config.device,
                    enable_residency_hint=self.config.enable_apw_residency,
                    residency=AmdResidencyConfig(
                        enable=self.config.enable_apw_residency,
                        hit_ratio=self.config.apw_hit_ratio,
                    ),
                )
                self.chunk_arena = AmdChunkArena(arena_config)
            else:
                arena_config = ChunkArenaConfig(
                    max_chunks=self.config.max_arena_chunks,
                    chunk_size=self.config.chunk_size,
                    num_layers=self.config.num_layers,
                    num_heads=self.config.num_heads,
                    head_dim=self.config.head_dim,
                    device=self.config.device,
                    enable_apw=self.config.enable_apw_residency,
                    apw_hit_ratio=self.config.apw_hit_ratio
                )
                self.chunk_arena = ChunkArena(arena_config)
        else:
            self.chunk_arena = None
        
        if self.config.enable_fused_kernel:
            self.bundle_verifier = FusedBundleVerifier(
                bundle_size=self.config.bundle_size,
                enable_triton=self.config.enable_triton,
                device=self.config.device
            )
        else:
            self.bundle_verifier = None
    
    def initialize_session(self, input_ids: torch.Tensor):
        """Initialize per-request component state."""
        if self.hazard_tracker is not None:
            self.hazard_tracker.reset()
        
        if self.paged_kv is not None:
            seq_len = input_ids.size(1)
            self.paged_kv.allocate_sequence(seq_id=0, num_tokens=seq_len)
        
        if self.chunk_arena is not None:
            self.chunk_arena.reset()
        
        if self.config.verbose:
            print(f"[DiffSpec] Session initialized with {input_ids.size(1)} tokens")
    
    def get_tree_config(self) -> Dict:
        """Return the current dynamic tree configuration."""
        if self.hazard_tracker is not None:
            return self.hazard_tracker.get_dynamic_tree_config(
                total_nodes=self.config.tree_node_budget,
                num_peaks=2
            )
        else:
            return {
                'layer_budgets': [
                    self.config.tree_node_budget // self.config.max_tree_depth
                ] * self.config.max_tree_depth,
                'total_nodes': self.config.tree_node_budget
            }
    
    def update_rejection_profile(self, first_reject_depth: int):
        """Update the hazard profile with one selected-path stop depth."""
        if self.hazard_tracker is not None:
            self.hazard_tracker.update(first_reject_depth)
            
            if self.config.verbose:
                stats = self.hazard_tracker.get_statistics()
                print(f"[Hazard] Reject depth: {first_reject_depth}, "
                      f"Avg: {stats['avg_reject_depth']:.2f}")
    
    def schedule_verification_bundles(
        self,
        tree_descriptor: Dict
    ) -> List[Dict]:
        """Build verification bundles for a draft tree descriptor."""
        if self.bundle_scheduler is not None:
            return self.bundle_scheduler.schedule_bundles(
                tree_descriptor,
                return_tensors=True
            )
        else:
            return [{
                'leaves': tree_descriptor['leaf_ids'],
                'br_anc': 0,
                'shared_prefix': [],
                'prefix_length': 0,
                'bundle_id': 0
            }]
    
    def verify_bundle(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        masks: List[torch.Tensor],
        shared_prefix_len: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify a scheduled bundle and return attention output plus LSE."""
        if self.bundle_verifier is not None:
            return self.bundle_verifier.verify_bundle(
                q, k, v, masks, shared_prefix_len
            )
        else:
            # Fallback to standard attention
            raise NotImplementedError("Standard attention fallback not implemented")
    
    def get_statistics(self) -> Dict:
        """Return statistics collected by the engine and enabled components."""
        stats = {
            'engine': self.stats.copy(),
            'config': {
                'hazard_enabled': self.config.enable_hazard_profile,
                'bundle_enabled': self.config.enable_bundle_scheduling,
                'paged_kv_enabled': self.config.enable_paged_kv,
                'arena_enabled': self.config.enable_chunk_arena,
            }
        }
        
        if self.hazard_tracker is not None:
            stats['hazard_profile'] = self.hazard_tracker.get_statistics()
        
        if self.bundle_scheduler is not None:
            stats['bundle_scheduler'] = self.bundle_scheduler.get_statistics()
        
        if self.paged_kv is not None:
            stats['paged_kv'] = self.paged_kv.get_statistics()
        
        if self.chunk_arena is not None:
            stats['chunk_arena'] = self.chunk_arena.get_statistics()
        
        if self.bundle_verifier is not None:
            stats['bundle_verifier'] = self.bundle_verifier.get_statistics()
        
        return stats
    
    def print_statistics(self):
        """Print a human-readable statistics report."""
        print("\n" + "=" * 70)
        print("DiffSpec Engine Statistics")
        print("=" * 70)
        
        stats = self.get_statistics()
        
        # Engine stats
        print("\nEngine:")
        for key, value in stats['engine'].items():
            print(f"  {key}: {value}")
        
        # Component stats
        if 'hazard_profile' in stats:
            print("\nHazard Profile:")
            for key, value in stats['hazard_profile'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        if 'bundle_scheduler' in stats:
            print("\nBundle Scheduler:")
            for key, value in stats['bundle_scheduler'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        if 'paged_kv' in stats:
            print("\nPaged KV Cache:")
            for key, value in stats['paged_kv'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2%}" if 'rate' in key or 'utilization' in key 
                          else f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        if 'chunk_arena' in stats:
            print("\nChunk Arena:")
            for key, value in stats['chunk_arena'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2%}" if 'rate' in key or 'utilization' in key
                          else f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        print("=" * 70 + "\n")
    
    def reset(self):
        """Reset component state and engine counters."""
        if self.hazard_tracker is not None:
            self.hazard_tracker.reset()
        
        if self.chunk_arena is not None:
            self.chunk_arena.reset()
        
        self.stats = {
            'total_drafts': 0,
            'total_verifications': 0,
            'total_accepted': 0,
            'total_rejected': 0
        }
        
        if self.config.verbose:
            print("[DiffSpec] Engine reset")
