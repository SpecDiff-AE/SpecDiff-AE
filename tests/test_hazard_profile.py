import pytest
import torch

from diffspec.core.hazard_profile import HazardProfileTracker


def test_basic_statistics_track_rejections():
    tracker = HazardProfileTracker(max_depth=8, window_size=10, device="cpu")

    for depth in [2, 3, 2, 4, 2, 5, 3, 2, 2, 3]:
        tracker.update(depth)

    stats = tracker.get_statistics()
    assert stats["total_rejections"] == 10
    assert 2.0 <= stats["avg_reject_depth"] <= 3.5


def test_budget_allocation_prioritizes_hazard_peaks():
    tracker = HazardProfileTracker(max_depth=8, window_size=20, device="cpu")

    for _ in range(15):
        tracker.update(2)
    for _ in range(10):
        tracker.update(3)
    for _ in range(5):
        tracker.update(5)

    budgets = tracker.allocate_budget(total_nodes=50, num_peaks=2)

    assert budgets[2] > budgets[0]
    assert budgets[3] > budgets[0]
    assert sum(budgets) <= 55


def test_visualization_returns_ascii_profile():
    tracker = HazardProfileTracker(max_depth=6, window_size=20, device="cpu")

    for _ in range(20):
        tracker.update(2)
    for _ in range(10):
        tracker.update(3)
    for _ in range(5):
        tracker.update(1)

    profile = tracker.visualize_profile()

    assert isinstance(profile, str)
    assert "depth" in profile.lower()


def test_dynamic_tree_config_matches_depth_count():
    tracker = HazardProfileTracker(max_depth=8, window_size=30, device="cpu")

    for depth in [2] * 10 + [4] * 8 + [3] * 5:
        tracker.update(depth)

    config = tracker.get_dynamic_tree_config(total_nodes=50, num_peaks=2)

    assert len(config["layer_budgets"]) == 8
    assert config["total_nodes"] <= 55


def test_state_roundtrip_preserves_tracker_values():
    tracker = HazardProfileTracker(max_depth=8, window_size=20, device="cpu")
    for depth in [2, 3, 2, 4, 3, 2]:
        tracker.update(depth)

    restored = HazardProfileTracker(max_depth=8, window_size=20, device="cpu")
    restored.load_state(tracker.save_state())

    assert restored.total_rejections == tracker.total_rejections
    assert len(restored.reject_history) == len(tracker.reject_history)
    assert torch.allclose(restored.hazard_profile, tracker.hazard_profile)


def test_cuda_tensors_when_cuda_is_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    tracker = HazardProfileTracker(max_depth=8, window_size=20, device="cuda")
    for depth in [2, 3, 2, 4]:
        tracker.update(depth)

    assert tracker.hazard_profile.device.type == "cuda"
    assert tracker.depth_counts.device.type == "cuda"
