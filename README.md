# DiffSpec: Accelerating Long Sequence Generation with Differential Speculative Decoding

![Task](https://img.shields.io/badge/Task-LLM_Inference_Acceleration-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

**DiffSpec** is a speculative decoding framework for long-context LLM
inference. By introducing **Salience-Aware Chunk Encoding**, **Hazard Profiling**, and optimized **KV Cache Management**, DiffSpec significantly improves draft acceptance rates and reduces verification overhead compared to standard EAGLE and other speculative decoding methods.


## 🚀 Features

- **Hazard-Guided Tree Construction**: Tracks selected-path stop depths and
  adapts branching budget toward difficult speculative depths.
- **Salience-Aware Chunk Encoding**: Encodes long-context chunks for
  retrieval-aware working-set selection.
- **Chunk Arena and Paged KV**: Provides memory-management primitives for
  physically contiguous chunk staging and copy-on-write KV-page sharing.
- **Bundle Scheduler**: Groups draft-tree leaves by branching ancestor for
  prefix-coherent target verification.

## 🛠️ Installation

### Prerequisites

- Python 3.10 or newer
- PyTorch with CUDA support
- Transformers
- Safetensors
- NVIDIA GPU for model benchmarks; H100/A100 class GPUs are recommended
- FlashAttention, optional but recommended
- Triton, optional for the optimized tree-attention path
- Nsight Compute, required only for profiling scripts

Model downloads use the domestic Hugging Face by default:

For fully offline or cached runs, set local model paths:

```bash
export DIFFSPEC_BASE_MODEL=/path/to/Llama-3.1-8B-Instruct
export DIFFSPEC_DRAFT_MODEL=/path/to/EAGLE3-LLaMA3.1-Instruct-8B
```

## 📂 Project Structure

```text
diffspec_clean/
├── diffspec/
│   ├── amd/                  # AMD GPU support
│   ├── core/                 # Policies, config, chunk encoding, scheduler, KV helpers
│   ├── draft/                # DiffSpecModel, draft network, verification loop
│   ├── runtime/              # Target adapters, attention kernels, runtime backends
│   └── defaults.py           # Shared default model identifiers and mirror settings
├── benchmarks/
│   ├── paired_inference_benchmark.py
│   ├── branch_policy_analysis.py
│   ├── hazard_stability_analysis.py
│   ├── collect_ncu_inference.py
│   ├── attention_kernel_profile.py
│   └── nsight_chunk_arena/
├── docs/
│   └── reproduction_llama31_8b.md
├── examples/
│   └── basic_usage.py
├── scripts/
│   └── reproduce_llama31_8b.sh
├── tests/
└── run_tests.sh
```


## ⚡Quick Start

### 1. Run Core Examples

```bash
python examples/basic_usage.py
```

This exercises the configuration object, hazard profile tracker, bundle
scheduler, and the high-level `DiffSpecEngine` with synthetic CPU inputs.

### 2. Run Unit Tests

```bash
./run_tests.sh unit
```

### 3. Run the Benchmark

```bash
./scripts/reproduce_llama31_8b.sh --preset smoke --gpu-index 0
```

The smoke preset runs one GovReport 50K-context case with a short tuning pass
and a short final paired Auto-vs-DiffSpec evaluation. Results are written to:

```text
results/llama31_8b_repro/<preset>_<timestamp>/
```

## Reproduction

Run the full Llama-3.1-8B reproduction protocol:

```bash
./scripts/reproduce_llama31_8b.sh --preset full --gpu-index 0
```

The full preset:

- Uses a Llama-3.1-8B target model and an EAGLE3 Llama-3.1-8B draft model.
- Tunes DiffSpec over the declared branch-policy/config grid.
- Selects the fastest DiffSpec configuration per dataset/context.
- Re-runs paired Auto vs DiffSpec with the selected configuration.
- Reports decode-only throughput/speedup and end-to-end throughput/speedup
  separately.

For protocol details, see
[docs/reproduction_llama31_8b.md](docs/reproduction_llama31_8b.md).

### Custom Benchmark

```bash
./scripts/reproduce_llama31_8b.sh --preset custom \
  --data-files "govreport/govreport_16K.jsonl" \
  --context-targets "50000" \
  --candidate-configs "64:0.08:8" \
  --policies "online,full,uniform4" \
  --tune-max-new-tokens 128 \
  --eval-max-new-tokens 512 \
  --gpu-index 0
```

### Paired Inference Benchmark

```bash
python benchmarks/paired_inference_benchmark.py \
  --base_model /path/to/base_model \
  --draft_model /path/to/draft_model \
  --data_files govreport/govreport_16K.jsonl \
  --context_targets 50000 \
  --max_new_tokens 128
```

## Profiling

### End-to-End Nsight Collection

```bash
python benchmarks/collect_ncu_inference.py --help
```

This harness collects Nsight Compute metrics for Auto and DiffSpec runs,
including L2 hit rate, HBM traffic, L1TEX traffic, and kernel occupancy.


## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

