#!/usr/bin/env python3
"""Compare online hazard-guided branching with static branching policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("DIFFSPEC_PROFILE", "1")
os.environ["DIFFSPEC_HAZARD_TRACE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from termcolor import colored

from benchmarks.common import format_value as fmt, parse_plugin_mask
from benchmarks.paired_inference_benchmark import build_context_cases, collect_environment, prepare_input_ids
from diffspec.defaults import DEFAULT_BASE_MODEL, DEFAULT_DRAFT_MODEL
from diffspec.draft.diffspec_model import DiffSpecModel


@contextmanager
def branch_policy(policy: str):
    old = os.environ.get("DIFFSPEC_BRANCH_POLICY")
    os.environ["DIFFSPEC_BRANCH_POLICY"] = policy
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("DIFFSPEC_BRANCH_POLICY", None)
        else:
            os.environ["DIFFSPEC_BRANCH_POLICY"] = old


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def js_divergence(p: list[float], q: list[float]) -> float:
    eps = 1.0e-12
    p = [max(float(x), eps) for x in p]
    q = [max(float(x), eps) for x in q]
    ps = sum(p)
    qs = sum(q)
    p = [x / ps for x in p]
    q = [x / qs for x in q]
    m = [(a + b) * 0.5 for a, b in zip(p, q)]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(a, b))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def histogram(values: list[int], labels: list[int]) -> tuple[list[int], list[float]]:
    index = {label: i for i, label in enumerate(labels)}
    counts = [0 for _ in labels]
    for value in values:
        if value in index:
            counts[index[value]] += 1
    total = sum(counts)
    probs = [count / total if total else 0.0 for count in counts]
    return counts, probs


def mode_label(counts: list[int], labels: list[int]) -> tuple[int, float]:
    total = sum(counts)
    if total == 0:
        return 0, 0.0
    pos = max(range(len(counts)), key=lambda i: counts[i])
    return labels[pos], counts[pos] / total


def quantile(sorted_values: list[int], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def window_profile(records: list[dict[str, Any]], accept_lengths: list[int], max_depth: int, window: int) -> list[dict[str, Any]]:
    labels = list(range(max_depth))
    rows = []
    prev_probs = None
    for start in range(0, len(records), window):
        chunk = records[start:start + window]
        accepts = accept_lengths[start:start + len(chunk)]
        if not chunk:
            continue
        values = [int(row["profile_depth"]) for row in chunk]
        counts, probs = histogram(values, labels)
        mode, share = mode_label(counts, labels)
        row = {
            "window_index": len(rows),
            "start_step": start + 1,
            "end_step": start + len(chunk),
            "steps": len(chunk),
            "profile_mode": mode,
            "profile_mode_share": share,
            "profile_entropy": entropy(probs),
            "profile_jsd_vs_prev": js_divergence(prev_probs, probs) if prev_probs is not None else None,
            "accept_mean": statistics.mean(accepts) if accepts else None,
            "accept_min": min(accepts) if accepts else None,
            "accept_max": max(accepts) if accepts else None,
        }
        rows.append(row)
        prev_probs = probs
    return rows


def summarize_run(
    policy: str,
    case: dict[str, Any],
    result: dict[str, Any],
    max_depth: int,
    window_size: int,
) -> dict[str, Any]:
    stats = result.get("hazard_stats") or {}
    trace = stats.get("trace_records") or []
    accepts = [int(x) for x in result.get("accept_length_list", [])]
    sorted_accepts = sorted(accepts)
    profile = stats.get("final_hazard_profile") or (
        trace[-1].get("hazard_profile") if trace else [1.0 / max_depth] * max_depth
    )
    profile = [float(x) for x in profile]
    prof_entropy = entropy(profile)
    labels = list(range(max_depth))
    counts, empirical_probs = histogram([int(row["profile_depth"]) for row in trace], labels)
    empirical_entropy = entropy(empirical_probs)
    top_depth, top_share = mode_label(counts, labels)
    tail = window_profile(trace, accepts, max_depth, window_size)[-3:]
    tail_jsds = [row["profile_jsd_vs_prev"] for row in tail if row["profile_jsd_vs_prev"] is not None]
    tail_modes = [row["profile_mode"] for row in tail]
    uniform_entropy = math.log2(max_depth)
    return {
        "policy": policy,
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "context_target": case["context_target_tokens"],
        "input_tokens": case["input_token_count"],
        "generated_tokens": result.get("total_generated"),
        "iterations": len(trace),
        "decode_tokens_per_sec": result.get("decode_tokens_per_sec"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "avg_accept_length": result.get("avg_accept_length"),
        "accept_mean_exact": statistics.mean(accepts) if accepts else None,
        "accept_std": statistics.pstdev(accepts) if len(accepts) > 1 else 0.0,
        "accept_median": statistics.median(accepts) if accepts else None,
        "accept_p25": quantile(sorted_accepts, 0.25),
        "accept_p75": quantile(sorted_accepts, 0.75),
        "accept_min": min(accepts) if accepts else None,
        "accept_max": max(accepts) if accepts else None,
        "accept_full_or_late_share": (
            sum(1 for value in accepts if value >= max_depth) / len(accepts)
            if accepts else None
        ),
        "final_top_depth": int(stats.get("max_hazard_depth", max(range(len(profile)), key=lambda i: profile[i]))),
        "final_top_prob": float(stats.get("max_hazard_prob", max(profile))),
        "final_entropy": prof_entropy,
        "entropy_norm": prof_entropy / uniform_entropy if uniform_entropy > 0 else None,
        "kl_to_uniform_bits": uniform_entropy - prof_entropy,
        "l1_to_uniform": sum(abs(p - 1.0 / max_depth) for p in profile),
        "empirical_top_depth": top_depth,
        "empirical_top_share": top_share,
        "empirical_entropy": empirical_entropy,
        "tail_mode_stable": bool(tail_modes and len(set(tail_modes)) == 1),
        "tail_jsd_mean": statistics.mean(tail_jsds) if tail_jsds else None,
        "profile": profile,
        "empirical_counts": counts,
        "accept_counts": histogram(accepts, list(range(max_depth + 1)))[0],
        "runtime_policy": result.get("runtime_policy", {}),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_keys = [
        "policy",
        "case_id",
        "dataset",
        "context_target",
        "input_tokens",
        "generated_tokens",
        "iterations",
        "decode_tokens_per_sec",
        "tokens_per_sec",
        "avg_accept_length",
        "accept_mean_exact",
        "accept_std",
        "accept_median",
        "accept_p25",
        "accept_p75",
        "accept_min",
        "accept_max",
        "accept_full_or_late_share",
        "final_top_depth",
        "final_top_prob",
        "final_entropy",
        "entropy_norm",
        "kl_to_uniform_bits",
        "l1_to_uniform",
        "empirical_top_depth",
        "empirical_top_share",
        "empirical_entropy",
        "tail_mode_stable",
        "tail_jsd_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    policies = sorted({row["policy"] for row in rows})
    for policy in policies:
        group = [row for row in rows if row["policy"] == policy]
        out.append({
            "policy": policy,
            "runs": len(group),
            "decode_tps_mean": statistics.mean(row["decode_tokens_per_sec"] for row in group),
            "decode_tps_std": statistics.pstdev(row["decode_tokens_per_sec"] for row in group) if len(group) > 1 else 0.0,
            "accept_mean": statistics.mean(row["avg_accept_length"] for row in group),
            "accept_std": statistics.pstdev(row["avg_accept_length"] for row in group) if len(group) > 1 else 0.0,
            "accept_median_mean": statistics.mean(row["accept_median"] for row in group if row["accept_median"] is not None),
            "accept_iqr_mean": statistics.mean(
                (row["accept_p75"] - row["accept_p25"])
                for row in group
                if row["accept_p75"] is not None and row["accept_p25"] is not None
            ),
            "full_or_late_mean": statistics.mean(
                row["accept_full_or_late_share"]
                for row in group
                if row["accept_full_or_late_share"] is not None
            ),
            "top_prob_mean": statistics.mean(row["final_top_prob"] for row in group),
            "entropy_norm_mean": statistics.mean(row["entropy_norm"] for row in group),
            "kl_uniform_mean": statistics.mean(row["kl_to_uniform_bits"] for row in group),
            "tail_jsd_mean": (
                statistics.mean(row["tail_jsd_mean"] for row in group if row["tail_jsd_mean"] is not None)
                if any(row["tail_jsd_mean"] is not None for row in group)
                else None
            ),
            "stable_tail_rate": sum(1 for row in group if row["tail_mode_stable"]) / len(group),
            "common_top_depth": max(set(row["final_top_depth"] for row in group), key=[row["final_top_depth"] for row in group].count),
        })
    return out


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    agg = aggregate(rows)
    baseline = next((row for row in agg if row["policy"] == "full"), None)
    lines = [
        "# Online Hazard Estimation vs Static Branching Policy",
        "",
        "This report addresses whether online hazard estimation learns a non-uniform level-wise failure profile, or whether failures become effectively uniform across depths.",
        "",
        "## Setup",
        "",
        f"- policies: {', '.join(args.policies)}",
        f"- context_targets: {', '.join(map(str, args.context_targets))}",
        f"- max_new_tokens: {args.max_new_tokens}",
        f"- nodes / threshold / max_depth: {args.nodes} / {args.threshold} / {args.max_depth}",
        f"- window_size: {args.window_size}",
        "",
        "## Policy Aggregate",
        "",
        "| Policy | Runs | Decode tok/s | Avg accept | Median accept | IQR | Full/late % | Top depth | Top prob | Normalized entropy | KL to uniform | Tail JSD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        speed = ""
        if baseline and baseline["decode_tps_mean"] > 0:
            speed = f" ({row['decode_tps_mean'] / baseline['decode_tps_mean']:.2f}x full)"
        lines.append(
            f"| {row['policy']} | {row['runs']} | {fmt(row['decode_tps_mean'], 2)}{speed} | "
            f"{fmt(row['accept_mean'], 2)} | {fmt(row['accept_median_mean'], 2)} | "
            f"{fmt(row['accept_iqr_mean'], 2)} | {fmt(100 * row['full_or_late_mean'], 1)} | "
            f"{row['common_top_depth']} | {fmt(row['top_prob_mean'], 3)} | "
            f"{fmt(row['entropy_norm_mean'], 3)} | {fmt(row['kl_uniform_mean'], 3)} | "
            f"{fmt(row['tail_jsd_mean'], 3)} |"
        )
    lines.extend([
        "",
        "## Per-Run Profile Shape",
        "",
        "| Policy | Dataset | Context | Iter | Decode tok/s | Avg accept | Median | IQR | Full/late % | Final top | Entropy norm | KL to uniform | Empirical top | Tail JSD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['policy']} | {Path(row['dataset']).parts[0]} | {row['input_tokens']} | {row['iterations']} | "
            f"{fmt(row['decode_tokens_per_sec'], 2)} | {fmt(row['avg_accept_length'], 2)} | "
            f"{fmt(row['accept_median'], 2)} | {fmt(row['accept_p75'] - row['accept_p25'] if row['accept_p75'] is not None and row['accept_p25'] is not None else None, 2)} | "
            f"{fmt(100 * row['accept_full_or_late_share'] if row['accept_full_or_late_share'] is not None else None, 1)} | "
            f"d{row['final_top_depth']} p={fmt(row['final_top_prob'], 3)} | "
            f"{fmt(row['entropy_norm'], 3)} | {fmt(row['kl_to_uniform_bits'], 3)} | "
            f"d{row['empirical_top_depth']} {fmt(100 * row['empirical_top_share'], 1)}% | "
            f"{fmt(row['tail_jsd_mean'], 3)} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A uniform failure profile over 8 depths would have normalized entropy near 1.0 and KL-to-uniform near 0. The measured online profiles are far from that regime when the top probability is high and normalized entropy is well below 1.0.",
        "",
        "The profile is not a universal property of language alone: it is conditioned on the current prompt, draft model, tree policy, and recent accepted path. Online estimation is useful because it tracks this local non-uniformity instead of assuming one static branching shape for every region of generation.",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    run_records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    partial: bool = False,
) -> None:
    stem = "hazard_policy.partial" if partial else "hazard_policy"
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        } | {"plugin_mask": args.plugin_mask},
        "environment": collect_environment(device),
        "runs": run_records,
        "summary": rows,
        "aggregate": aggregate(rows) if rows else [],
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(output, indent=2))
    write_summary_csv(output_dir / "summary.csv", rows)
    write_report(output_dir / "report.md", rows, args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "diffspec" / "data"))
    parser.add_argument("--data-files", nargs="+", default=["govreport/govreport_16K.jsonl"])
    parser.add_argument("--context-targets", type=int, nargs="+", default=[50000, 128000])
    parser.add_argument("--samples-per-context", type=int, default=1)
    parser.add_argument("--max-source-records", type=int, default=16)
    parser.add_argument("--context-tolerance", type=float, default=0.08)
    parser.add_argument("--policies", nargs="+", default=["online", "full", "uniform4", "shallow", "deep"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--retrieve-every-n", type=int, default=4)
    parser.add_argument("--retrieval-min-context", type=int, default=50000)
    parser.add_argument("--plugin-mask", default="0,1,0,0,0,0")
    parser.add_argument("--disable-hybrid-tree-attn", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "hazard_policy")
    args = parser.parse_args()
    args.plugin_mask = parse_plugin_mask(args.plugin_mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.gpu_index)
    device = torch.device(f"cuda:{args.gpu_index}")

    print(colored("Loading model...", "cyan"))
    model = DiffSpecModel.from_pretrained(
        base_model_path=args.base_model,
        draft_model_path=args.draft_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    tokenizer = model.tokenizer

    print(colored("Building cases...", "cyan"))
    cases = build_context_cases(
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir),
        data_files=args.data_files,
        context_targets=args.context_targets,
        samples_per_context=args.samples_per_context,
        tolerance=args.context_tolerance,
        max_source_records=args.max_source_records,
    )

    all_rows = []
    run_records = []
    hybrid_tree_flag = 0 if args.disable_hybrid_tree_attn else 1
    for case in cases:
        input_ids = prepare_input_ids(tokenizer, case["input_text"], device)
        case["input_token_count"] = int(input_ids.shape[1])
        for policy in args.policies:
            print(colored(f"Running {policy} on {case['case_id']} ({case['input_token_count']} tokens)", "cyan"))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with branch_policy(policy):
                result = model.diffspec_generate(
                    input_ids,
                    temperature=0,
                    max_new_tokens=args.max_new_tokens,
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
                    return_generated_text=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            result["outer_wall_time"] = time.perf_counter() - start
            summary = summarize_run(policy, case, result, args.max_depth, args.window_size)
            all_rows.append(summary)
            run_records.append({
                "policy": policy,
                "case": {key: value for key, value in case.items() if key != "input_text"},
                "result": result,
                "summary": summary,
            })
            write_outputs(args.output_dir, args, device, run_records, all_rows, partial=True)
            print(
                colored(
                    f"  decode={summary['decode_tokens_per_sec']:.2f} tok/s, "
                    f"accept={summary['avg_accept_length']:.2f}, "
                    f"top=d{summary['final_top_depth']} p={summary['final_top_prob']:.3f}, "
                    f"Hnorm={summary['entropy_norm']:.3f}",
                    "white",
                )
            )
            torch.cuda.empty_cache()

    write_outputs(args.output_dir, args, device, run_records, all_rows, partial=False)
    print(colored(f"Wrote {args.output_dir}", "green"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
