# Llama-3.1-8B Reproduction

This repository provides a one-command harness for the Llama-3.1-8B DiffSpec
results.

## Quick Smoke Test

```bash
./scripts/reproduce_llama31_8b.sh --preset smoke --gpu-index 0
```

The smoke preset uses one GovReport 50K-target case, a short 128-token tuning
run, and a short 128-token final Auto-vs-DiffSpec evaluation. This keeps the
run small while exercising the long-context DiffSpec path and decode-only
speedup check.

## Full Reproduction

```bash
./scripts/reproduce_llama31_8b.sh --preset full --gpu-index 0
```

The full preset:

- Uses only the Llama-3.1-8B target model and the EAGLE3 Llama-3.1-8B draft
  model.
- Tunes DiffSpec over the declared branch-policy/config grid.
- Selects the fastest DiffSpec configuration per dataset/context.
- Re-runs paired Auto vs DiffSpec with the selected configuration.
- Reports both decode-only throughput/speedup and end-to-end generation
  throughput/speedup.
- Writes `FINAL_REPORT.md`, `tuning_summary.csv`, `selected_configs.json`, and
  `final_summary.csv`.

Default model identifiers:

```text
base:  meta-llama/Llama-3.1-8B-Instruct
draft: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```

Override these with `--base-model` and `--draft-model`, or set
`DIFFSPEC_BASE_MODEL` and `DIFFSPEC_DRAFT_MODEL`, if your models are stored in
a local Hugging Face cache. The shell harness sets
`HF_ENDPOINT=https://hf-mirror.com` by default for domestic mirror downloads.

## What "Best" Means

The harness reports the best DiffSpec setting **within the declared tuning
grid**, not a mathematical global optimum. The default full grid is:

```text
configs: 64:0.08:8,96:0.08:12,128:0.04:16
policies: online,full,uniform4,shallow,deep
retrieval_min_context: 48000
```

The 48K retrieval gate is intentional: the bundled PG19 50K target materializes
as 48,749 prompt tokens, so a 50K gate would disable hazard profiling for that
case.

You can expand the grid:

```bash
./scripts/reproduce_llama31_8b.sh --preset full \
  --candidate-configs "64:0.08:8,96:0.08:12,128:0.04:16,160:0.02:20" \
  --policies "online,full,uniform4,shallow,deep"
```

## Custom Run

```bash
./scripts/reproduce_llama31_8b.sh --preset custom \
  --data-files "govreport/govreport_16K.jsonl" \
  --context-targets "50000" \
  --candidate-configs "64:0.08:8" \
  --policies "online,full,uniform4" \
  --tune-max-new-tokens 128 \
  --eval-max-new-tokens 512
```

## Output

By default, outputs are written to:

```text
results/llama31_8b_repro/<preset>_<timestamp>/
```

Important files:

| File | Meaning |
|---|---|
| `FINAL_REPORT.md` | Human-readable reproduction report. |
| `tuning_summary.csv` | Every DiffSpec tuning row. |
| `selected_configs.json` | Fastest tuned config per dataset/context. |
| `final_summary.csv` | Final paired Auto-vs-DiffSpec table, including decode-only and end-to-end speedups. |
| `eval/*/paired_inference_benchmark.json` | Raw final benchmark outputs. |
