import torch

from diffspec.core.bundle_scheduler import BundleScheduler, TreeDescriptorBuilder


def test_basic_scheduling_returns_bundle_descriptors():
    scheduler = BundleScheduler(bundle_size=3, device="cpu")
    tree_descriptor = {
        "leaf_ids": [3, 4, 5, 6],
        "parent_map": {0: -1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2},
        "depth_map": {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2},
    }

    bundles = scheduler.schedule_bundles(tree_descriptor, return_tensors=False)

    assert len(bundles) >= 1
    assert all("leaves" in bundle for bundle in bundles)
    assert all("br_anc" in bundle for bundle in bundles)


def test_branching_ancestor_is_shared_by_sibling_leaves():
    scheduler = BundleScheduler(bundle_size=3, device="cpu")
    parent_map = {
        0: -1,
        1: 0,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 3,
        7: 4,
        8: 5,
    }

    assert scheduler._find_branching_ancestor(6, parent_map) == 2
    assert scheduler._find_branching_ancestor(7, parent_map) == 2
    assert scheduler._find_branching_ancestor(8, parent_map) == 2


def test_path_extraction_includes_ancestor_and_leaf():
    scheduler = BundleScheduler(bundle_size=3, device="cpu")
    parent_map = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3}

    path = scheduler._get_path_to_ancestor(4, 1, parent_map)

    assert path == [1, 2, 3, 4]


def test_tensor_conversion_returns_tensor_fields():
    scheduler = BundleScheduler(bundle_size=2, device="cpu")
    tree_descriptor = {
        "leaf_ids": [2, 3, 4, 5],
        "parent_map": {0: -1, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1},
        "depth_map": {idx: 1 if idx > 0 else 0 for idx in range(6)},
    }

    bundles = scheduler.schedule_bundles(tree_descriptor, return_tensors=True)

    assert len(bundles) >= 1
    assert all(isinstance(bundle["leaves"], torch.Tensor) for bundle in bundles)
    assert all(isinstance(bundle["shared_prefix"], torch.Tensor) for bundle in bundles)


def test_tree_descriptor_builder_from_parent_tensor():
    parent = torch.tensor([0, 0, 0, 1, 1, 2, 2])

    descriptor = TreeDescriptorBuilder.from_parent_tensor(parent, num_nodes=8)

    assert len(descriptor["leaf_ids"]) > 0
    assert len(descriptor["parent_map"]) == 8


def test_scheduler_statistics_accumulate_across_calls():
    scheduler = BundleScheduler(bundle_size=3, device="cpu")

    for _ in range(3):
        tree_descriptor = {
            "leaf_ids": [3, 4, 5, 6, 7, 8],
            "parent_map": {
                0: -1,
                1: 0,
                2: 0,
                3: 1,
                4: 1,
                5: 2,
                6: 2,
                7: 1,
                8: 2,
            },
            "depth_map": {idx: 1 if 0 < idx < 3 else 2 for idx in range(9)},
        }
        scheduler.schedule_bundles(tree_descriptor, return_tensors=False)

    stats = scheduler.get_statistics()

    assert stats["total_scheduled_leaves"] == 18
    assert stats["total_bundles_created"] > 0


def test_attention_mask_creation_uses_requested_sequence_length():
    scheduler = BundleScheduler(bundle_size=2, device="cpu")
    tree_descriptor = {
        "leaf_ids": [3, 4, 5, 6],
        "parent_map": {0: -1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2},
        "depth_map": {idx: min(idx, 2) for idx in range(7)},
    }

    bundles = scheduler.schedule_bundles(tree_descriptor, return_tensors=True)
    masks = scheduler.create_bundle_attention_masks(bundles, max_seq_len=10)

    assert len(masks) == len(bundles)
    assert all(mask.shape[1] == 10 for mask in masks)
    assert all(mask.dtype == torch.int8 for mask in masks)
