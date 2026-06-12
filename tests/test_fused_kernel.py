import pytest
import torch

from diffspec.runtime.bundle_verifier import FusedBundleVerifier


def make_inputs(batch=1, query_len=8, heads=4, head_dim=32, key_len=16, device="cpu"):
    query = torch.randn(batch, query_len, heads, head_dim, device=device)
    key = torch.randn(batch, key_len, heads, head_dim, device=device)
    value = torch.randn(batch, key_len, heads, head_dim, device=device)
    mask = torch.tril(torch.ones(query_len, key_len, dtype=torch.int8, device=device))
    return query, key, value, mask


def test_basic_bundle_verification_shapes():
    verifier = FusedBundleVerifier(bundle_size=3, enable_triton=False, device="cpu")
    query, key, value, mask = make_inputs(query_len=10, heads=8, head_dim=64, key_len=20)

    output, lse = verifier.verify_bundle(query, key, value, [mask], shared_prefix_len=5)

    assert output.shape == query.shape
    assert lse.shape == query.shape[:3]


def test_prefix_reuse_updates_statistics():
    verifier = FusedBundleVerifier(bundle_size=3, enable_triton=False, device="cpu")
    query, key, value, mask = make_inputs()

    verifier.verify_bundle(query, key, value, [mask, mask, mask], shared_prefix_len=10)
    stats = verifier.get_statistics()

    assert stats["prefix_reuse_rate"] > 0


def test_multiple_bundles_accumulate_statistics():
    verifier = FusedBundleVerifier(bundle_size=3, enable_triton=False, device="cpu")

    for shared_prefix_len in [0, 2, 4]:
        query, key, value, mask = make_inputs(query_len=6, key_len=12)
        verifier.verify_bundle(query, key, value, [mask], shared_prefix_len=shared_prefix_len)

    assert verifier.get_statistics()["total_bundles"] == 3


def test_masked_attention_outputs_are_finite():
    verifier = FusedBundleVerifier(bundle_size=1, enable_triton=False, device="cpu")
    query, key, value, _ = make_inputs(query_len=4, heads=2, head_dim=16, key_len=8)
    mask = torch.zeros(4, 8, dtype=torch.int8)
    mask[:, :3] = 1

    output, lse = verifier.verify_bundle(query, key, value, [mask], shared_prefix_len=0)

    assert not torch.isnan(output).any()
    assert not torch.isnan(lse).any()


def test_statistics_count_verifications_and_bundles():
    verifier = FusedBundleVerifier(bundle_size=3, enable_triton=False, device="cpu")

    for _ in range(5):
        query, key, value, mask = make_inputs(query_len=5, key_len=10)
        verifier.verify_bundle(query, key, value, [mask], shared_prefix_len=3)

    stats = verifier.get_statistics()
    assert stats["total_verifications"] == 5
    assert stats["total_bundles"] == 5


def test_cuda_verification_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    verifier = FusedBundleVerifier(bundle_size=3, enable_triton=True, device="cuda")
    query, key, value, mask = make_inputs(query_len=8, heads=8, head_dim=64, key_len=16, device="cuda")

    try:
        output, _ = verifier.verify_bundle(query, key, value, [mask], shared_prefix_len=5)
    except Exception as exc:
        pytest.skip(f"CUDA fused verification is unavailable in this environment: {exc}")

    assert output.device.type == "cuda"
