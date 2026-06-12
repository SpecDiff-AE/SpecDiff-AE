import torch

from diffspec.core.kv_cache import PagedKVCache, PagedKVConfig


def make_config(**overrides):
    values = {
        "page_size": 16,
        "max_pages": 100,
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 32,
        "device": "cpu",
    }
    values.update(overrides)
    return PagedKVConfig(**values)


def test_allocate_and_free_sequence_updates_page_usage():
    cache = PagedKVCache(make_config(num_layers=4, num_heads=8, head_dim=64))

    page_table = cache.allocate_sequence(seq_id=0, num_tokens=50)
    assert len(page_table) == 4

    cache.free_sequence(0)
    assert cache.get_statistics()["used_pages"] == 0


def test_read_write_roundtrip_for_one_token():
    cache = PagedKVCache(make_config(page_size=8, max_pages=50))
    cache.allocate_sequence(seq_id=0, num_tokens=20)
    key = torch.randn(4, 32)
    value = torch.randn(4, 32)

    cache.write_kv(seq_id=0, layer_idx=0, token_idx=5, k=key, v=value)
    key_read, value_read = cache.read_kv(seq_id=0, layer_idx=0, token_idx=5)

    assert torch.allclose(key, key_read, atol=1e-5)
    assert torch.allclose(value, value_read, atol=1e-5)


def test_copy_on_write_keeps_parent_page_unchanged():
    cache = PagedKVCache(make_config(enable_cow=True))
    cache.allocate_sequence(seq_id=0, num_tokens=50)
    original_key = torch.randn(4, 32)
    original_value = torch.randn(4, 32)
    cache.write_kv(seq_id=0, layer_idx=0, token_idx=5, k=original_key, v=original_value)

    cache.fork_sequence(parent_seq_id=0, child_seq_id=1, shared_prefix_len=32)
    stats = cache.get_statistics()
    assert stats["cow_saves"] > 0

    child_key = torch.randn(4, 32)
    child_value = torch.randn(4, 32)
    cache.write_kv(seq_id=1, layer_idx=0, token_idx=5, k=child_key, v=child_value)
    parent_key, _ = cache.read_kv(seq_id=0, layer_idx=0, token_idx=5)

    assert torch.allclose(parent_key, original_key, atol=1e-5)
    assert not torch.allclose(parent_key, child_key)


def test_contiguous_kv_reads_requested_window():
    cache = PagedKVCache(make_config(page_size=8, max_pages=50))
    cache.allocate_sequence(seq_id=0, num_tokens=20)

    for token_idx in range(15):
        key = torch.ones(4, 32) * token_idx
        value = torch.ones(4, 32) * token_idx
        cache.write_kv(seq_id=0, layer_idx=0, token_idx=token_idx, k=key, v=value)

    keys, values = cache.get_contiguous_kv(seq_id=0, layer_idx=0, start_token=5, end_token=10)

    assert keys.shape == (4, 5, 32)
    assert values.shape == (4, 5, 32)
    assert keys[0, 0, 0] == 5.0


def test_physical_block_list_matches_page_table():
    cache = PagedKVCache(make_config(max_pages=50))
    page_table = cache.allocate_sequence(seq_id=0, num_tokens=50)

    blocks = cache.get_physical_blocks(seq_id=0)

    assert len(blocks) == len(page_table)
    assert all(block >= 0 for block in blocks)


def test_defragment_preserves_live_sequences():
    cache = PagedKVCache(make_config(max_pages=100))
    for seq_id in range(5):
        cache.allocate_sequence(seq_id=seq_id, num_tokens=30)

    cache.free_sequence(1)
    cache.free_sequence(3)
    cache.defragment()

    blocks = cache.get_physical_blocks(seq_id=0)
    assert blocks
    assert all(block >= 0 for block in blocks)
