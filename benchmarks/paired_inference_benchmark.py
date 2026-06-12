#!/usr/bin/env python3
"""Robust paired benchmark for Auto(AR) vs DiffSpec.

The output JSON is intentionally sample-centric: each case stores the input
text, both generated outputs, throughput, accept length, and paired speedup.
Plots are generated from the same JSON payload.
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import torch
from termcolor import colored
from tqdm import tqdm

from benchmarks.common import load_text_records, parse_plugin_mask, prepare_input_ids, prompt_token_count
from diffspec.draft.diffspec_model import DiffSpecModel
from diffspec.defaults import DEFAULT_BASE_MODEL, DEFAULT_DRAFT_MODEL
from diffspec.runtime.attention_compat import (
    FLASH_ATTN_AVAILABLE,
    PYTORCH_FLASH_ATTN_AVAILABLE,
    sdpa_backend_name,
)
from diffspec.runtime.tree_attention import TRITON_AVAILABLE

def fit_text_to_target(tokenizer, text: str, target_tokens: int, tolerance: float) -> Dict:
    """Trim a raw input so the final chat prompt is near target_tokens without overshooting."""
    count = prompt_token_count(tokenizer, text)
    if count <= target_tokens:
        return {"text": text, "input_token_count": count}

    raw_ids = tokenizer(text, add_special_tokens=False).input_ids
    overhead = max(count - len(raw_ids), 0)
    keep = max(target_tokens - overhead, 32)
    best_text = tokenizer.decode(
        raw_ids[:keep],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    best_count = prompt_token_count(tokenizer, best_text)

    # A short binary refinement keeps contexts close without exceeding the target.
    low, high = 32, min(len(raw_ids), max(keep * 2, keep + 1))
    for _ in range(8):
        mid = (low + high) // 2
        candidate = tokenizer.decode(
            raw_ids[:mid],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        candidate_count = prompt_token_count(tokenizer, candidate)
        if (
            candidate_count <= target_tokens
            and abs(candidate_count - target_tokens) < abs(best_count - target_tokens)
        ):
            best_text, best_count = candidate, candidate_count
        if candidate_count > target_tokens:
            high = mid - 1
        else:
            low = mid + 1

    return {"text": best_text, "input_token_count": best_count}


def build_context_cases(
    tokenizer,
    data_dir: Path,
    data_files: List[str],
    context_targets: List[int],
    samples_per_context: int,
    tolerance: float,
    max_source_records: int,
) -> List[Dict]:
    cases = []
    separator = "\n\n--- next source document ---\n\n"

    for data_file in data_files:
        path = data_dir / data_file
        records = load_text_records(path, max_records=max_source_records)
        if not records:
            raise RuntimeError(f"No usable text records found in {path}")

        dataset_name = data_file.replace(os.sep, "/")
        for target in context_targets:
            for sample_id in range(samples_per_context):
                start = sample_id % len(records)
                parts = []
                source_indices = []
                current_count = 0

                for offset in range(len(records)):
                    rec = records[(start + offset) % len(records)]
                    parts.append(rec["text"])
                    source_indices.append(rec["source_index"])
                    candidate = separator.join(parts)
                    current_count = prompt_token_count(tokenizer, candidate)
                    if current_count >= int(target * (1.0 - tolerance)):
                        break

                fitted = fit_text_to_target(tokenizer, separator.join(parts), target, tolerance)
                cases.append(
                    {
                        "case_id": (
                            f"{Path(data_file).stem}_target{target}_sample{sample_id}"
                        ),
                        "dataset": dataset_name,
                        "source_indices": source_indices,
                        "context_target_tokens": target,
                        "input_token_count": int(fitted["input_token_count"]),
                        "input_text": fitted["text"],
                    }
                )
                print(
                    colored(
                        f"Built {cases[-1]['case_id']}: "
                        f"{cases[-1]['input_token_count']} prompt tokens",
                        "white",
                    )
                )

    return cases


def run_autoregressive(model, input_ids: torch.Tensor, max_new_tokens: int) -> Dict:
    return model.autoregressive_generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        output_result_line=False,
        verbose=False,
        return_generated_text=True,
        use_flash_prefill=True,
    )


def run_diffspec_generation(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    args,
) -> Dict:
    hybrid_tree_flag = 0 if args.disable_hybrid_tree_attn else 1
    return model.diffspec_generate(
        input_ids,
        temperature=0,
        max_new_tokens=max_new_tokens,
        nodes=args.nodes,
        threshold=args.threshold,
        max_depth=args.max_depth,
        output_result_line=False,
        verbose=False,
        use_diffspec=[1, 1, hybrid_tree_flag, 1, 1],
        diffspec_plugins=args.plugin_mask,
        retrieval_chunk_size=args.chunk_size,
        retrieve_top_k=args.top_k,
        retrieve_every_n_steps=args.retrieve_every_n,
        retrieval_min_context=args.retrieval_min_context,
        retrieval_verbose=False,
        return_generated_text=True,
    )


def measure_generation(method: str, model, input_ids: torch.Tensor, args) -> Dict:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    if method == "Auto":
        result = run_autoregressive(model, input_ids, args.max_new_tokens)
        result["avg_accept_length"] = 1.0
    elif method == "DiffSpec":
        result = run_diffspec_generation(model, input_ids, args.max_new_tokens, args)
    else:
        raise ValueError(f"Unknown method: {method}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result["wall_time"] = time.perf_counter() - wall_start
    result["tokens_per_sec"] = (
        result["total_generated"] / result["inference_time"]
        if result.get("inference_time", 0) > 0
        else 0.0
    )
    return result


def maybe_warmup(model, tokenizer, case: Dict, method: str, device: torch.device, args):
    if args.warmup_runs <= 0:
        return
    input_ids = prepare_input_ids(tokenizer, case["input_text"], device)
    warmup_args = argparse.Namespace(**vars(args))
    warmup_args.max_new_tokens = min(args.max_new_tokens, args.warmup_max_new_tokens)
    for _ in range(args.warmup_runs):
        _ = measure_generation(method, model, input_ids, warmup_args)


def numeric_stats(values: Iterable[float]) -> Dict:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
            "p05": None,
        }
    sorted_values = sorted(values)
    p05_idx = max(0, int(0.05 * (len(sorted_values) - 1)))
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
        "p05": round(sorted_values[p05_idx], 4),
    }


def speedup_field_for(metric: str) -> str:
    return "decode_speedup_vs_auto" if metric == "decode" else "speedup_vs_auto"


def speedup_label_for(metric: str) -> str:
    return "Decode-only speedup" if metric == "decode" else "End-to-end speedup"


def summarize_group(cases: List[Dict], min_speedup: float, target_metric: str) -> Dict:
    auto_tps = []
    diffspec_tps = []
    auto_decode_tps = []
    diffspec_decode_tps = []
    speedups = []
    decode_speedups = []
    accept_lengths = []
    input_lengths = []
    passed = 0
    target_field = speedup_field_for(target_metric)

    for case in cases:
        paired_runs = case.get("paired_runs", [])
        if not paired_runs:
            auto = case["runs"].get("Auto")
            diffspec = case["runs"].get("DiffSpec")
            if auto and diffspec:
                paired_runs = [
                    {
                        "Auto": auto,
                        "DiffSpec": diffspec,
                        "speedup_vs_auto": case.get("speedup_vs_auto"),
                        "decode_speedup_vs_auto": case.get("decode_speedup_vs_auto"),
                        "input_token_count": case["input_token_count"],
                    }
                ]
        for pair in paired_runs:
            auto = pair.get("Auto")
            diffspec = pair.get("DiffSpec")
            if not auto or not diffspec:
                continue
            auto_tps.append(auto["tokens_per_sec"])
            diffspec_tps.append(diffspec["tokens_per_sec"])
            if auto.get("decode_tokens_per_sec") is not None:
                auto_decode_tps.append(auto["decode_tokens_per_sec"])
            if diffspec.get("decode_tokens_per_sec") is not None:
                diffspec_decode_tps.append(diffspec["decode_tokens_per_sec"])
            accept_lengths.append(diffspec["avg_accept_length"])
            input_lengths.append(pair.get("input_token_count", case["input_token_count"]))
            speedup = pair.get("speedup_vs_auto")
            if speedup is not None:
                speedups.append(speedup)
            decode_speedup = pair.get("decode_speedup_vs_auto")
            if decode_speedup is not None:
                decode_speedups.append(decode_speedup)
            target_speedup = pair.get(target_field)
            if target_speedup is None:
                target_speedup = speedup
            if target_speedup is not None and target_speedup >= min_speedup:
                passed += 1

    num_pairs = max(len(speedups), len(decode_speedups))

    return {
        "num_pairs": num_pairs,
        "input_tokens": numeric_stats(input_lengths),
        "auto_tokens_per_sec": numeric_stats(auto_tps),
        "diffspec_tokens_per_sec": numeric_stats(diffspec_tps),
        "auto_decode_tokens_per_sec": numeric_stats(auto_decode_tps),
        "diffspec_decode_tokens_per_sec": numeric_stats(diffspec_decode_tps),
        "diffspec_avg_accept_length": numeric_stats(accept_lengths),
        "speedup_vs_auto": numeric_stats(speedups),
        "decode_speedup_vs_auto": numeric_stats(decode_speedups),
        "target_speedup": min_speedup,
        "target_metric": target_metric,
        "target_pass_count": passed,
        "target_pass_rate": round(passed / num_pairs, 4) if num_pairs else None,
    }


def build_summary(cases: List[Dict], min_speedup: float, target_metric: str) -> Dict:
    summary = {"overall": summarize_group(cases, min_speedup, target_metric)}

    by_context = {}
    for case in cases:
        by_context.setdefault(str(case["context_target_tokens"]), []).append(case)
    summary["by_context_target"] = {
        key: summarize_group(group, min_speedup, target_metric)
        for key, group in sorted(by_context.items(), key=lambda item: int(item[0]))
    }

    by_dataset = {}
    for case in cases:
        by_dataset.setdefault(case["dataset"], []).append(case)
    summary["by_dataset"] = {
        key: summarize_group(group, min_speedup, target_metric)
        for key, group in sorted(by_dataset.items())
    }

    return summary


def plot_results(
    cases: List[Dict],
    summary: Dict,
    output_dir: Path,
    min_speedup: float,
    target_metric: str,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    speedup_field = speedup_field_for(target_metric)
    speedup_label = speedup_label_for(target_metric)

    contexts = sorted({case["context_target_tokens"] for case in cases})

    def context_values(method: str, field: str, context: int) -> List[float]:
        vals = []
        for case in cases:
            if case["context_target_tokens"] != context:
                continue
            paired_runs = case.get("paired_runs", [])
            if paired_runs:
                for pair in paired_runs:
                    run = pair.get(method)
                    if run and field in run:
                        vals.append(float(run[field]))
            else:
                run = case["runs"].get(method)
                if run and field in run:
                    vals.append(float(run[field]))
        return vals

    # Throughput by context.
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.36
    x = list(range(len(contexts)))
    for offset, method, color in [(-width / 2, "Auto", "#4c78a8"), (width / 2, "DiffSpec", "#f58518")]:
        means = []
        stds = []
        for ctx in contexts:
            stats = numeric_stats(context_values(method, "tokens_per_sec", ctx))
            means.append(stats["mean"] or 0.0)
            stds.append(stats["std"] or 0.0)
        ax.bar([i + offset for i in x], means, width, yerr=stds, capsize=4, label=method, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ctx // 1000}K" for ctx in contexts])
    ax.set_xlabel("Target context")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Auto vs DiffSpec Throughput")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "throughput_by_context.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    plot_paths.append(str(path))

    # Speedup by context.
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, ctx in enumerate(contexts):
        vals = [
            float(pair[speedup_field])
            for case in cases
            if case["context_target_tokens"] == ctx
            for pair in case.get("paired_runs", [])
            if pair.get(speedup_field) is not None
        ]
        ax.scatter([idx] * len(vals), vals, color="#54a24b", alpha=0.75)
        stats = numeric_stats(vals)
        if stats["mean"] is not None:
            ax.errorbar(
                idx,
                stats["mean"],
                yerr=stats["std"] or 0.0,
                color="#1b5e20",
                fmt="o",
                capsize=5,
                markersize=7,
            )
    ax.axhline(min_speedup, color="#d62728", linestyle="--", linewidth=1.5, label=f"{min_speedup:.1f}x target")
    ax.set_xticks(list(range(len(contexts))))
    ax.set_xticklabels([f"{ctx // 1000}K" for ctx in contexts])
    ax.set_xlabel("Target context")
    ax.set_ylabel(f"DiffSpec {speedup_label.lower()} vs Auto")
    ax.set_title(f"Paired {speedup_label} Stability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "speedup_by_context.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    plot_paths.append(str(path))

    # Accept length by context.
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [context_values("DiffSpec", "avg_accept_length", ctx) for ctx in contexts]
    ax.boxplot(data, tick_labels=[f"{ctx // 1000}K" for ctx in contexts], showmeans=True)
    ax.set_xlabel("Target context")
    ax.set_ylabel("Average accept length")
    ax.set_title("DiffSpec Accept Length Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "accept_length_by_context.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    plot_paths.append(str(path))

    # Dataset speedup summary.
    datasets = sorted({case["dataset"] for case in cases})
    fig, ax = plt.subplots(figsize=(10, 5))
    means = []
    stds = []
    for dataset in datasets:
        vals = [
            float(pair[speedup_field])
            for case in cases
            if case["dataset"] == dataset
            for pair in case.get("paired_runs", [])
            if pair.get(speedup_field) is not None
        ]
        stats = numeric_stats(vals)
        means.append(stats["mean"] or 0.0)
        stds.append(stats["std"] or 0.0)
    ax.bar(datasets, means, yerr=stds, capsize=4, color="#72b7b2")
    ax.axhline(min_speedup, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_ylabel(f"DiffSpec {speedup_label.lower()} vs Auto")
    ax.set_title(f"{speedup_label} by Dataset")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "speedup_by_dataset.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    plot_paths.append(str(path))

    return plot_paths


def collect_environment(device: torch.device) -> Dict:
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "flash_attn_available": bool(FLASH_ATTN_AVAILABLE),
        "torch_flash_sdp_available": bool(PYTORCH_FLASH_ATTN_AVAILABLE),
        "attention_backend": sdpa_backend_name(),
        "triton_available": bool(TRITON_AVAILABLE),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        env.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            }
        )
    return env


def main():
    parser = argparse.ArgumentParser(description="Robust paired Auto vs DiffSpec benchmark")
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
    )
    parser.add_argument(
        "--draft_model",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
    )
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "diffspec" / "data"))
    parser.add_argument(
        "--data_files",
        nargs="+",
        default=["govreport/govreport_16K.jsonl", "pg-19/pg19_16K.jsonl"],
    )
    parser.add_argument("--context_targets", type=int, nargs="+", default=[16000, 24000, 32000])
    parser.add_argument("--samples_per_context", type=int, default=2)
    parser.add_argument("--repeat_runs", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--context_tolerance", type=float, default=0.08)
    parser.add_argument("--max_source_records", type=int, default=16)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--warmup_max_new_tokens", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--retrieve_every_n", type=int, default=4)
    parser.add_argument("--retrieval_min_context", type=int, default=50000)
    parser.add_argument("--plugin_mask", type=str, default="0,1,0,0,0,0")
    parser.add_argument(
        "--disable_hybrid_tree_attn",
        action="store_true",
        help="Disable the flash/prefix + Triton hybrid tree verification path.",
    )
    parser.add_argument("--min_speedup_vs_auto", type=float, default=4.8)
    parser.add_argument(
        "--speedup_target_metric",
        "--speedup-target-metric",
        choices=["decode", "end_to_end"],
        default="decode",
        help="Metric used for target pass/fail checks.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--enforce_all_pairs",
        action="store_true",
        help="Exit with failure if any paired DiffSpec run is below the speedup target.",
    )
    parser.add_argument(
        "--enforce_context_means",
        action="store_true",
        help="Exit with failure if any context mean is below the speedup target.",
    )
    args = parser.parse_args()
    args.plugin_mask = parse_plugin_mask(args.plugin_mask)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or (REPO_ROOT / "results" / "paired_inference_benchmark" / timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json or (output_dir / "paired_inference_benchmark.json"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda:0")

    print(colored("Loading model...", "cyan"))
    model = DiffSpecModel.from_pretrained(
        base_model_path=args.base_model,
        draft_model_path=args.draft_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    tokenizer = model.tokenizer

    print(colored("Building context cases...", "cyan"))
    cases = build_context_cases(
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir),
        data_files=args.data_files,
        context_targets=args.context_targets,
        samples_per_context=args.samples_per_context,
        tolerance=args.context_tolerance,
        max_source_records=args.max_source_records,
    )

    print(colored("Running paired benchmark...", "cyan"))
    for case in tqdm(cases, desc="cases"):
        case["runs"] = {}
        input_ids = prepare_input_ids(tokenizer, case["input_text"], device)
        case["input_token_count"] = int(input_ids.shape[1])

        for repeat_idx in range(args.repeat_runs):
            pair_id = f"{case['case_id']}_repeat{repeat_idx}"
            pair = {
                "case_id": pair_id,
                "repeat": repeat_idx,
                "input_token_count": case["input_token_count"],
            }
            if repeat_idx == 0:
                maybe_warmup(model, tokenizer, case, "Auto", device, args)
                maybe_warmup(model, tokenizer, case, "DiffSpec", device, args)

            auto = measure_generation("Auto", model, input_ids, args)
            diffspec = measure_generation("DiffSpec", model, input_ids, args)
            speedup = (
                diffspec["tokens_per_sec"] / auto["tokens_per_sec"]
                if auto["tokens_per_sec"] > 0
                else None
            )
            decode_speedup = (
                diffspec["decode_tokens_per_sec"] / auto["decode_tokens_per_sec"]
                if auto.get("decode_tokens_per_sec", 0) > 0
                else None
            )

            pair["Auto"] = auto
            pair["DiffSpec"] = diffspec
            pair["speedup_vs_auto"] = round(speedup, 4) if speedup is not None else None
            pair["decode_speedup_vs_auto"] = round(decode_speedup, 4) if decode_speedup is not None else None
            case.setdefault("paired_runs", []).append(pair)

        # For summary and simple inspection, also expose repeat-mean runs.
        auto_runs = [pair["Auto"] for pair in case["paired_runs"]]
        ds_runs = [pair["DiffSpec"] for pair in case["paired_runs"]]
        case["runs"]["Auto"] = {
            "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in auto_runs),
            "decode_tokens_per_sec": statistics.mean(r["decode_tokens_per_sec"] for r in auto_runs),
            "avg_accept_length": 1.0,
            "total_generated_mean": statistics.mean(r["total_generated"] for r in auto_runs),
            "inference_time_mean": statistics.mean(r["inference_time"] for r in auto_runs),
            "output_text": auto_runs[-1].get("generated_text", ""),
            "generated_token_ids": auto_runs[-1].get("generated_token_ids", []),
        }
        case["runs"]["DiffSpec"] = {
            "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in ds_runs),
            "decode_tokens_per_sec": statistics.mean(r["decode_tokens_per_sec"] for r in ds_runs),
            "avg_accept_length": statistics.mean(r["avg_accept_length"] for r in ds_runs),
            "total_generated_mean": statistics.mean(r["total_generated"] for r in ds_runs),
            "inference_time_mean": statistics.mean(r["inference_time"] for r in ds_runs),
            "output_text": ds_runs[-1].get("generated_text", ""),
            "generated_token_ids": ds_runs[-1].get("generated_token_ids", []),
            "accept_length_list": ds_runs[-1].get("accept_length_list", []),
            "runtime_policy": ds_runs[-1].get("runtime_policy", {}),
        }
        case["speedup_vs_auto"] = round(
            case["runs"]["DiffSpec"]["tokens_per_sec"] / case["runs"]["Auto"]["tokens_per_sec"],
            4,
        )
        case["decode_speedup_vs_auto"] = round(
            case["runs"]["DiffSpec"]["decode_tokens_per_sec"] / case["runs"]["Auto"]["decode_tokens_per_sec"],
            4,
        )
        speedup_field = speedup_field_for(args.speedup_target_metric)
        case["meets_speedup_target"] = case[speedup_field] >= args.min_speedup_vs_auto

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = build_summary(cases, args.min_speedup_vs_auto, args.speedup_target_metric)
    plots = plot_results(cases, summary, output_dir, args.min_speedup_vs_auto, args.speedup_target_metric)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            **vars(args),
            "output_dir": str(output_dir),
            "output_json": str(output_json),
        },
        "environment": collect_environment(device),
        "summary": summary,
        "cases": cases,
        "plots": plots,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(colored(f"Saved JSON: {output_json}", "green"))
    for path in plots:
        print(colored(f"Saved plot: {path}", "green"))

    overall = summary["overall"]
    speed_field = speedup_field_for(args.speedup_target_metric)
    speed_stats = overall[speed_field]
    print(
        colored(
            f"Overall DiffSpec {speedup_label_for(args.speedup_target_metric).lower()} vs Auto: "
            f"mean={speed_stats['mean']}x, min={speed_stats['min']}x, "
            f"pass_rate={overall['target_pass_rate']}",
            "cyan",
        )
    )

    failures = []
    if args.enforce_all_pairs:
        for case in cases:
            for pair in case.get("paired_runs", []):
                speedup = pair.get(speed_field)
                if speedup is None or speedup < args.min_speedup_vs_auto:
                    failures.append(pair["case_id"])
    if args.enforce_context_means:
        for ctx, stats in summary["by_context_target"].items():
            mean_speedup = stats[speed_field]["mean"]
            if mean_speedup is None or mean_speedup < args.min_speedup_vs_auto:
                failures.append(f"context_{ctx}_mean")
    if failures:
        raise RuntimeError(
            "Speedup target not met for: "
            + ", ".join(failures[:12])
            + (" ..." if len(failures) > 12 else "")
        )


if __name__ == "__main__":
    main()
