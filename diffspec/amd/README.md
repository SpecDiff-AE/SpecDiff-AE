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

Environment knobs:

```bash
export DIFFSPEC_AMD_RESIDENCY_HINT=1
export DIFFSPEC_AMD_APW_HIT_RATIO=0.85
export DIFFSPEC_AMD_APW_MAX_BYTES=268435456
```

## Validation

Run this first on an AMD ROCm host:

```bash
python diffspec/amd/validate_rocm_backend.py \
  --strict \
  --enable-residency \
  --output results/amd_chunk_arena/validator.json
```

The validator checks:

- PyTorch is a ROCm build and an AMD GPU is visible.
- `hipcc` is available.
- The optional HIP runtime extension can request an APW-like residency hint.
- `AmdChunkArena` stages exact selected KV spans and returns the same tensors
  that DiffSpec will feed into the draft model.
- The HIP runtime extension can compact selected KV spans for the real
  working-cache path instead of falling back to torch-copy staging.
- `DraftNetwork._select_working_cache_from_chunks()` consumes `AmdChunkArena`
  in the real DiffSpec working-cache path and assigns the staged tensors to
  `draft_stable_kv`.

The JSON output contains a `checks.diffspec_working_cache_path` section. A
passing AMD run should show `"backend": "amd_rocm"`, the expected
`working_len`, and non-empty `arena_statistics.active_spans`.
With `--strict`, the validator also requires `hipcc`, real HIP
access-policy-window application, and HIP compact-kernel staging. If APW or
the compact kernel falls back to metadata/torch-copy status, the command exits
non-zero and records the reason in
`checks.strict_requirements.failures`.

On non-ROCm hosts, use `--allow-no-rocm` only for CI/import checks:

```bash
python diffspec/amd/validate_rocm_backend.py --allow-no-rocm
```

That mode must not be used as AMD performance evidence.

## HIP Microbenchmark

Build and run:

```bash
python diffspec/amd/hip/run_hip_profiles.py \
  --full-tokens 131072 \
  --active-chunks 128 \
  --chunk-size 64 \
  --heads 8 \
  --head-dim 64 \
  --passes 4
```

Outputs are written to `results/amd_chunk_arena/`:

- `raw_results.json`: event timings for all modes.
- `report.md`: Markdown summary table.
- `profiler_tools.json`: detected ROCm profiling tools.
- `profiler_commands.md`: reviewer-facing ROCm Compute Profiler commands.
- `*.log`: stdout/stderr from each mode.
- `compile.log`: `hipcc` build output.

The four measured modes are:

- `baseline_sparse`: sparse reads from the full KV buffer.
- `arena`: reads from a contiguous active Chunk Arena.
- `arena_apw`: `arena` plus HIP access-policy-window residency hint.
- `arena_apw_tma`: `arena_apw` plus LDS tile staging and reuse. This is the
  AMD TMA analogue; `tma_analogue_backend` must be `lds_tile_staging` and
  `lds_vector_width_bytes` must be at least `16`.

To collect native hardware counters on an AMD host:

```bash
python diffspec/amd/hip/run_hip_profiles.py \
  --strict \
  --run-rocprof-compute \
  --full-tokens 131072 \
  --active-chunks 128 \
  --chunk-size 64 \
  --heads 8 \
  --head-dim 64 \
  --passes 4
```

Reviewer-facing metrics to report from ROCm Compute Profiler:

| Claim | Native metrics |
|---|---|
| APW improves cache residency | `L2 Cache Hit Rate`, `L2 Cache BW`, `L2-Fabric Read BW` |
| Arena/APW reduce memory pressure | `HBM Read Traffic`, achieved HBM bandwidth |
| LDS staging provides TMA-like reuse | `Theoretical LDS Bandwidth`, `LDS Bank Conflicts/Access`, reduced HBM/L2 reads for `arena_apw_tma` |
| Throughput is not an occupancy artifact | `Wavefront Occupancy`, `Active CUs` |

The profiler runner writes `strict_requirements.json`. In strict mode, the
file must show `"passed": true`; otherwise the run should not be used as AMD
APW/LDS evidence. In strict mode, APW failures include the HIP runtime error in
`apw_error`, and LDS/TMA-analogue failures are reported as
`lds_tile_staging_not_reported`, `lds_tile_bytes_not_positive`, or
`vectorized_lds_staging_not_reported`.

## NVIDIA Safety

The AMD backend is gated by `torch.version.hip`. On NVIDIA CUDA builds,
`is_rocm_pytorch()` is false, so DiffSpec keeps using the existing CUDA/NVIDIA
code path. The unit tests cover this fallback behavior.
