import torch

from diffspec.core.chunk_encoding import SalienceAwareChunkEncoder


def test_single_chunk_encoding_returns_proxy_vectors():
    head_dim = 16
    encoder = SalienceAwareChunkEncoder(head_dim=head_dim)

    chunk_keys = torch.randn(8, head_dim)
    chunk_values = torch.randn(8, head_dim)

    key_proxy, value_proxy = encoder.encode_chunk(chunk_keys, chunk_values)

    assert key_proxy.shape == (head_dim,)
    assert value_proxy.shape == (head_dim,)
    assert key_proxy.abs().sum() > 0
    assert value_proxy.abs().sum() > 0


def test_batch_chunk_encoding_preserves_head_and_chunk_axes():
    head_dim = 16
    num_heads = 4
    chunk_size = 8
    num_chunks = 3
    encoder = SalienceAwareChunkEncoder(head_dim=head_dim)

    full_keys = torch.randn(num_heads, chunk_size * num_chunks, head_dim)
    full_values = torch.randn(num_heads, chunk_size * num_chunks, head_dim)
    chunks = [(idx, idx * chunk_size, (idx + 1) * chunk_size) for idx in range(num_chunks)]

    proxy_keys, proxy_values = encoder.encode_chunks_batch(full_keys, full_values, chunks)

    assert proxy_keys.shape == (num_heads, num_chunks, head_dim)
    assert proxy_values.shape == (num_heads, num_chunks, head_dim)


def test_empty_chunk_returns_zero_proxies():
    head_dim = 8
    encoder = SalienceAwareChunkEncoder(head_dim=head_dim)

    empty_keys = torch.empty(0, head_dim)
    empty_values = torch.empty(0, head_dim)

    key_proxy, value_proxy = encoder.encode_chunk(empty_keys, empty_values)

    assert torch.equal(key_proxy, torch.zeros(head_dim))
    assert torch.equal(value_proxy, torch.zeros(head_dim))
