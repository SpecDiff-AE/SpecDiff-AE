# DiffSpec AMD ROCm Backend

This directory contains the AMD/HIP memory-residency and staging path used by
DiffSpec when the runtime is ROCm PyTorch. It is intentionally separated from
the NVIDIA path: CUDA/NVIDIA runs continue to use the existing
`diffspec.core.kv_cache.ChunkArena` implementation.

## Mechanism Mapping

| NVIDIA mechanism | AMD ROCm path | Implementation |
|---|---|---|
| APW / persisting L2 access window | HIP access-policy-window stream attribute | `hip_runtime.cpp` calls `hipStreamSetAttribute(..., hipStreamAttributeAccessPolicyWindow, ...)`; Python exposes it through `apply_residency_hint`. |
| TMA-style bulk working-set reuse | Contiguous KV arena plus AMD LDS tile staging | `AmdChunkArena` compacts selected KV spans into an active arena with the optional HIP compact kernel; `kv_access_profile.hip` compares direct arena reads with 16-byte vectorized LDS tile staging and reuse. |
| Sparse KV gather baseline | HIP sparse gather kernel | `baseline_sparse` reads selected tokens from full KV to measure the cost avoided by arena staging. |

HIP exposes an APW-like stream attribute, but it does not expose CUDA Hopper TMA
or CUDA cooperative-groups `memcpy_async`. For AMD we therefore claim an
equivalent memory-system optimization goal, not a hardware-identical TMA API:
the active KV working set is made arena-local, optionally marked with HIP APW,
and the AMD profiling path repeatedly consumes it from LDS tiles. The strict
profiler gate requires `arena_apw_tma` to report `lds_tile_staging`,
positive `lds_tile_bytes`, and a 16-byte vectorized LDS staging path.

## Runtime Integration

The AMD path is automatically selected only when `torch.version.hip` is present.
On ROCm PyTorch, `torch.cuda` remains the public PyTorch device namespace, so
the device string is still usually `"cuda"`.

DiffSpec uses `AmdChunkArena` in two places:

- `diffspec.draft.DiffSpecModel`: when retrieval cache and the `chunk_arena`
  plugin are enabled under ROCm PyTorch.
- `diffspec.core.DiffSpecEngine`: when `enable_chunk_arena=True` under ROCm
  PyTorch.

The hot path is `DraftNetwork._select_working_cache_from_chunks()`: selected
retrieval chunks are staged by `AmdChunkArena.update(...)`, converted back to
per-layer `(K, V)` tensors by `as_kv_list()`, and assigned to
`draft_stable_kv`. This means the AMD arena affects the actual DiffSpec decode
working cache, not only a standalone benchmark.


