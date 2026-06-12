# Nsight Chunk Arena Profiling

This directory contains a standalone CUDA microbenchmark for profiling the
DiffSpec Chunk Arena idea with Nsight Compute.

## Configurations

- `baseline_sparse`: sparse KV gather from a large full KV buffer into a compact output.
- `arena`: read/write from a physically contiguous Chunk Arena working set.
- `arena_apw`: same arena path with CUDA Access Policy Window configured for L2 residency.
- `arena_apw_tma`: arena path with APW plus Hopper async bulk global-to-shared copy through `cuda::memcpy_async`.

The Python model path currently stages Chunk Arena data, but APW/TMA are not
wired into `diffspec/draft/diffspec_model.py`. This harness isolates the hardware
mechanisms so Nsight Compute can report L2/HBM/bandwidth/occupancy evidence.

## Run

```bash
python benchmarks/nsight_chunk_arena/run_ncu_profiles.py
```

Outputs are written to `results/nsight_chunk_arena/`:

- `summary.csv`: machine-readable table.
- `report.md`: Markdown report with the headline comparison table.
- `*.ncu.csv`: raw Nsight Compute CSV for each mode.
- `raw_results.json`: CUDA event timings and parsed Nsight metrics.
- `tma_sass_check.txt`: SASS evidence that the APW+TMA path compiled to a
  Hopper bulk-copy instruction.

The report includes both raw Nsight counters and derived advantage metrics:

- HBM/logical byte ratio and logical/HBM reuse factor.
- HBM bytes per active decode token.
- L2 device-fill volume and L2 reuse per HBM fill.
- L1TEX request traffic and hit rate.
- DRAM pressure reduction against sparse KV gather.
- Full-KV to active-Chunk-Arena working-set compression and L2/APW fit.

Useful smaller smoke test:

```bash
python benchmarks/nsight_chunk_arena/run_ncu_profiles.py \
  --full-tokens 4096 \
  --active-chunks 8 \
  --chunk-size 32 \
  --heads 4 \
  --head-dim 32 \
  --passes 2 \
  --warmup 1 \
  --timing-iters 2
```

If Nsight Compute counter permissions are restricted, the runner still writes
CUDA event timings and records the Nsight error in the report.

On machines where `/proc/driver/nvidia/params` has
`RmProfilingAdminOnly: 1`, run Nsight Compute inside a privileged NVIDIA Docker
container with a newer NCU install:

```bash
python benchmarks/nsight_chunk_arena/run_ncu_profiles.py \
  --docker-ncu \
  --docker-ncu-root "$HOME/.cache/ncu2025" \
  --out-dir results/nsight_chunk_arena
```

The `--docker-ncu-root` directory must contain an `ncu` executable, for example
a copy of `/usr/local/cuda-12.8/nsight-compute-2025.1.0`.
