"""
Basic DiffSpec usage examples.
"""

import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffspec.core.config import DiffSpecConfig
from diffspec.core.diffspec_engine import DiffSpecEngine


def example_basic():
    """Run a minimal engine session with synthetic input."""
    print("=" * 70)
    print("Example 1: basic engine usage")
    print("=" * 70 + "\n")
    
    random.seed(0)
    torch.manual_seed(0)

    config = DiffSpecConfig.get_performance()
    config.device = "cpu"
    config.dtype = "float32"
    print(config.summary())
    
    engine = DiffSpecEngine(config)
    
    input_ids = torch.randint(0, 1000, (1, 50))
    engine.initialize_session(input_ids)
    
    for step in range(10):
        tree_config = engine.get_tree_config()
        _ = tree_config
        
        first_reject_depth = random.randint(0, 7)
        engine.update_rejection_profile(first_reject_depth)
        
        engine.stats['total_drafts'] += 1
        engine.stats['total_verifications'] += 1
        engine.stats['total_accepted'] += random.randint(5, 10)
    
    engine.print_statistics()


def example_custom_config():
    """Create an engine from an explicit configuration."""
    print("\n" + "=" * 70)
    print("Example 2: custom configuration")
    print("=" * 70 + "\n")
    
    config = DiffSpecConfig(
        enable_salience_encoding=False,
        enable_hazard_profile=True,
        enable_chunk_arena=True,
        enable_bundle_scheduling=False,
        enable_paged_kv=False,
        enable_fused_kernel=True,
        enable_triton=False,
        chunk_size=32,
        tree_node_budget=30,
        max_arena_chunks=32,
        device="cpu",
        verbose=True,
    )
    
    print(config.summary())
    
    engine = DiffSpecEngine(config)
    _ = engine
    print("\nEngine created successfully.")


def example_hazard_profile():
    """Use the hazard tracker directly."""
    print("\n" + "=" * 70)
    print("Example 3: standalone hazard profile")
    print("=" * 70 + "\n")
    
    from diffspec.core.hazard_profile import HazardProfileTracker
    
    tracker = HazardProfileTracker(
        max_depth=8,
        window_size=20,
        device='cpu'
    )
    
    random.seed(0)
    print("Simulating a rejection pattern...")
    for _ in range(30):
        depth = random.choices([2, 3, 1, 4, 5], weights=[0.4, 0.3, 0.1, 0.1, 0.1])[0]
        tracker.update(depth)
    
    print("\n" + tracker.visualize_profile())
    
    stats = tracker.get_statistics()
    print("\nStatistics:")
    print(f"  Average rejection depth: {stats['avg_reject_depth']:.2f}")
    print(f"  Max-hazard depth: {stats['max_hazard_depth']}")
    print(f"  Max-hazard probability: {stats['max_hazard_prob']:.2%}")
    
    budgets = tracker.allocate_budget(total_nodes=50, num_peaks=2)
    print(f"\nPer-depth branch budget: {budgets}")


def example_bundle_scheduler():
    """Use the bundle scheduler directly."""
    print("\n" + "=" * 70)
    print("Example 4: standalone bundle scheduler")
    print("=" * 70 + "\n")
    
    from diffspec.core.bundle_scheduler import BundleScheduler
    
    scheduler = BundleScheduler(bundle_size=3, device='cpu')
    
    # Tree: 0 -> [1, 2] -> [3, 4, 5, 6, 7, 8].
    tree_descriptor = {
        'leaf_ids': [3, 4, 5, 6, 7, 8],
        'parent_map': {
            0: -1,
            1: 0, 2: 0,
            3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2
        },
        'depth_map': {
            0: 0,
            1: 1, 2: 1,
            3: 2, 4: 2, 5: 2,
            6: 2, 7: 2, 8: 2
        }
    }
    
    bundles = scheduler.schedule_bundles(tree_descriptor, return_tensors=False)
    
    print(f"Generated {len(bundles)} bundles:\n")
    for bundle in bundles:
        print(f"Bundle {bundle['bundle_id']}:")
        print(f"  Leaves: {bundle['leaves']}")
        print(f"  Branching ancestor: {bundle['br_anc']}")
        print(f"  Shared prefix length: {bundle['prefix_length']}")
        print(f"  Shared prefix: {bundle['shared_prefix']}\n")
    
    stats = scheduler.get_statistics()
    print("Statistics:")
    print(f"  Scheduled leaves: {stats['total_scheduled_leaves']}")
    print(f"  Bundles created: {stats['total_bundles_created']}")
    print(f"  Average bundle size: {stats['avg_bundle_size']:.2f}")


def example_comparison():
    """Compare built-in DiffSpec configuration presets."""
    print("\n" + "=" * 70)
    print("Example 5: configuration presets")
    print("=" * 70 + "\n")
    
    configs = [
        ("Minimal", DiffSpecConfig.get_minimal()),
        ("Default", DiffSpecConfig.get_default()),
        ("Performance", DiffSpecConfig.get_performance()),
        ("Memory efficient", DiffSpecConfig.get_memory_efficient())
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        print("-" * 60)
        print(f"  Hazard Profile: {format_enabled(config.enable_hazard_profile)}")
        print(f"  Chunk Arena: {format_enabled(config.enable_chunk_arena)}")
        print(f"  Bundle Scheduling: {format_enabled(config.enable_bundle_scheduling)}")
        print(f"  Paged KV: {format_enabled(config.enable_paged_kv)}")
        print(f"  Fused Kernel: {format_enabled(config.enable_fused_kernel)}")
        print(f"  Node budget: {config.tree_node_budget}")
        print(f"  Chunk size: {config.chunk_size}")


def format_enabled(value: bool) -> str:
    """Return a compact ASCII status label."""
    return "enabled" if value else "disabled"


if __name__ == '__main__':
    example_basic()
    example_custom_config()
    example_hazard_profile()
    example_bundle_scheduler()
    example_comparison()
    
    print("\n" + "=" * 70)
    print("All examples finished.")
    print("=" * 70)
