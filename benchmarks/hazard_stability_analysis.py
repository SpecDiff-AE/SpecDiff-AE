#!/usr/bin/env python3
"""Profile whether DiffSpec's hazard failure mode stabilizes during real generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
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

from benchmarks.common import format_value as fmt
from benchmarks.common import parse_plugin_mask, safe_div, sanitize_json as sanitize
from benchmarks.paired_inference_benchmark import (
    build_context_cases,
    collect_environment,
    prepare_input_ids,
    measure_generation,
)
from diffspec.defaults import DEFAULT_BASE_MODEL, DEFAULT_DRAFT_MODEL
from diffspec.draft.diffspec_model import DiffSpecModel


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
    idx = {label: i for i, label in enumerate(labels)}
    counts = [0 for _ in labels]
    for value in values:
        if value in idx:
            counts[idx[value]] += 1
    total = sum(counts)
    probs = [safe_div(count, total) or 0.0 for count in counts]
    return counts, probs


def mode_label(counts: list[int], labels: list[int]) -> tuple[int, float]:
    total = sum(counts)
    if not counts or total == 0:
        return 0, 0.0
    i = max(range(len(counts)), key=lambda j: counts[j])
    return labels[i], counts[i] / total


def windowed(records: list[dict[str, Any]], accept_lengths: list[int], max_depth: int, window: int) -> list[dict[str, Any]]:
    raw_labels = [-1] + list(range(max_depth))
    prof_labels = list(range(max_depth))
    rows = []
    prev_raw_probs = None
    prev_prof_probs = None
    for start in range(0, len(records), window):
        chunk = records[start:start + window]
        accepts = accept_lengths[start:start + len(chunk)]
        if not chunk:
            continue
        raw_values = [int(r["raw_first_reject_depth"]) for r in chunk]
        prof_values = [int(r["profile_depth"]) for r in chunk]
        raw_counts, raw_probs = histogram(raw_values, raw_labels)
        prof_counts, prof_probs = histogram(prof_values, prof_labels)
        raw_mode, raw_mode_share = mode_label(raw_counts, raw_labels)
        prof_mode, prof_mode_share = mode_label(prof_counts, prof_labels)
        full_accept_count = sum(1 for r in chunk if r.get("full_accept"))
        row = {
            "window_index": len(rows),
            "start_step": start + 1,
            "end_step": start + len(chunk),
            "steps": len(chunk),
            "raw_mode": raw_mode,
            "raw_mode_share": raw_mode_share,
            "full_accept_share": full_accept_count / len(chunk),
            "profile_mode": prof_mode,
            "profile_mode_share": prof_mode_share,
            "raw_entropy": entropy(raw_probs),
            "profile_entropy": entropy(prof_probs),
            "raw_jsd_vs_prev": js_divergence(prev_raw_probs, raw_probs) if prev_raw_probs is not None else None,
            "profile_jsd_vs_prev": js_divergence(prev_prof_probs, prof_probs) if prev_prof_probs is not None else None,
            "accept_mean": statistics.mean(accepts) if accepts else None,
            "accept_std": statistics.pstdev(accepts) if len(accepts) > 1 else 0.0,
            "accept_min": min(accepts) if accepts else None,
            "accept_max": max(accepts) if accepts else None,
            "accept_mode": max(set(accepts), key=accepts.count) if accepts else None,
            "final_tracker_top_depth": chunk[-1].get("max_hazard_depth"),
            "final_tracker_top_prob": chunk[-1].get("max_hazard_prob"),
            "final_tracker_entropy": chunk[-1].get("hazard_entropy"),
        }
        rows.append(row)
        prev_raw_probs = raw_probs
        prev_prof_probs = prof_probs
    return rows


def write_window_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, output: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    result = output["result"]
    analysis = output["analysis"]
    lines = [
        "# Hazard Profile Stability",
        "",
        "This report checks whether the observed first-reject failure mode stabilizes during a real DiffSpec generation.",
        "",
        "## Workload",
        "",
        f"- dataset: {output['case']['dataset']}",
        f"- input_tokens: {output['case']['input_token_count']}",
        f"- max_new_tokens: {output['config']['max_new_tokens']}",
        f"- generated_tokens: {result.get('total_generated')}",
        f"- iterations: {analysis['iterations']}",
        f"- window_size: {analysis['window_size']}",
        f"- nodes/threshold/max_depth: {output['config']['nodes']} / {output['config']['threshold']} / {output['config']['max_depth']}",
        f"- retrieval_enabled: {result.get('runtime_policy', {}).get('retrieval_enabled')}",
        f"- tree_backend: {result.get('runtime_policy', {}).get('tree_attention_backend')}",
        "",
        "## Verdict",
        "",
        f"- Tracker final top depth: {analysis['final_tracker_top_depth']} with probability {fmt(analysis['final_tracker_top_prob'], 4)}.",
        f"- Tracker final entropy: {fmt(analysis['final_tracker_entropy'], 4)} bits.",
        f"- Window profile mode stable in tail: {analysis['tail_profile_mode_stable']}.",
        f"- Tail profile JSD mean: {fmt(analysis['tail_profile_jsd_mean'], 4)}.",
        f"- Tail accept-length mean range: {fmt(analysis['tail_accept_mean_min'], 2)} to {fmt(analysis['tail_accept_mean_max'], 2)}.",
        "",
        analysis["interpretation"],
        "",
        "## Window Summary",
        "",
        "| Window | Steps | Raw mode | Full accept % | Profile mode | Profile JSD | Tracker top p | Accept mean | Accept range |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['window_index']} | {row['steps']} | {row['raw_mode']} ({fmt(100*row['raw_mode_share'], 1)}%) | "
            f"{fmt(100*row['full_accept_share'], 1)} | {row['profile_mode']} ({fmt(100*row['profile_mode_share'], 1)}%) | "
            f"{fmt(row['profile_jsd_vs_prev'], 4)} | {fmt(row['final_tracker_top_prob'], 4)} | "
            f"{fmt(row['accept_mean'], 2)} | {fmt(row['accept_min'], 0)}-{fmt(row['accept_max'], 0)} |"
        )
    lines.extend(
        [
            "",
            "## Important Implementation Note",
            "",
            "The current tracker maps full-accept events (`raw_first_reject_depth = -1`) into the deepest profile bucket (`max_depth - 1`). Therefore a stable peak at the deepest bucket means the verifier mostly reaches the end of the tree or rejects late; it does not distinguish full acceptance from a true depth-7 first reject.",
            "",
            "The traced metric is the earliest rejected node anywhere in the candidate tree. In a wide tree, shallow off-path branch failures can dominate this metric even when the selected verification path accepts many tokens. For selected-path failure stability, compare this report with an accept-length-derived stop-depth trace.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def analyze(result: dict[str, Any], max_depth: int, window: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hazard_stats = result.get("hazard_stats") or {}
    trace = hazard_stats.get("trace_records") or []
    accept_lengths = result.get("accept_length_list") or []
    rows = windowed(trace, [int(x) for x in accept_lengths], max_depth, window)
    tail = rows[-min(3, len(rows)):] if rows else []
    tail_modes = [row["profile_mode"] for row in tail]
    tail_profile_jsds = [row["profile_jsd_vs_prev"] for row in tail if row["profile_jsd_vs_prev"] is not None]
    tail_accept_means = [row["accept_mean"] for row in tail if row["accept_mean"] is not None]
    final = trace[-1] if trace else {}
    stable = bool(tail_modes and len(set(tail_modes)) == 1)
    tail_jsd_mean = statistics.mean(tail_profile_jsds) if tail_profile_jsds else None
    if stable and final.get("max_hazard_prob", 0) >= 0.9 and (tail_jsd_mean is None or tail_jsd_mean <= 0.05):
        interpretation = (
            "The hazard profile itself converges strongly: the EMA profile collapses to one dominant bucket in the tail windows. "
            "However, this should be read as a stable late-failure/full-accept bucket, not as every iteration having identical accept length."
        )
    elif stable and final.get("max_hazard_prob", 0) >= 0.75 and (tail_jsd_mean is None or tail_jsd_mean <= 0.05):
        interpretation = (
            "The hazard profile has a stable dominant bucket in the tail windows, but it does not collapse to a single failure mode. "
            "Accept lengths remain variable, so this is only weak stability of the current tracker metric."
        )
    else:
        interpretation = (
            "The hazard profile does not fully converge under this workload; the dominant bucket or window distribution remains variable."
        )
    return {
        "iterations": len(trace),
        "window_size": window,
        "final_tracker_top_depth": final.get("max_hazard_depth"),
        "final_tracker_top_prob": final.get("max_hazard_prob"),
        "final_tracker_entropy": final.get("hazard_entropy"),
        "tail_profile_mode_stable": stable,
        "tail_profile_jsd_mean": tail_jsd_mean,
        "tail_accept_mean_min": min(tail_accept_means) if tail_accept_means else None,
        "tail_accept_mean_max": max(tail_accept_means) if tail_accept_means else None,
        "interpretation": interpretation,
    }, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "diffspec" / "data"))
    parser.add_argument("--data-files", nargs="+", default=["govreport/govreport_16K.jsonl"])
    parser.add_argument("--context-target", type=int, default=50000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--context-tolerance", type=float, default=0.08)
    parser.add_argument("--max-source-records", type=int, default=16)
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
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "hazard_stability")
    args = parser.parse_args()
    args.plugin_mask = parse_plugin_mask(args.plugin_mask)
    args.warmup_runs = 0
    args.warmup_max_new_tokens = 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu_index}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(colored("Loading model...", "cyan"), flush=True)
    model = DiffSpecModel.from_pretrained(
        base_model_path=args.base_model,
        draft_model_path=args.draft_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    tokenizer = model.tokenizer

    print(colored("Building case...", "cyan"), flush=True)
    cases = build_context_cases(
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir),
        data_files=args.data_files,
        context_targets=[args.context_target],
        samples_per_context=1,
        tolerance=args.context_tolerance,
        max_source_records=args.max_source_records,
    )
    case = cases[0]
    input_ids = prepare_input_ids(tokenizer, case["input_text"], device)
    case["input_token_count"] = int(input_ids.shape[1])

    print(colored("Running DiffSpec with hazard tracing...", "cyan"), flush=True)
    start = time.perf_counter()
    result = measure_generation("DiffSpec", model, input_ids, args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    outer_wall = time.perf_counter() - start
    result = sanitize(result)
    result.pop("generated_text", None)
    result.pop("generated_token_ids", None)
    result["outer_wall_time"] = outer_wall

    analysis, rows = analyze(result, args.max_depth, args.window_size)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "context_target": args.context_target,
            "max_new_tokens": args.max_new_tokens,
            "nodes": args.nodes,
            "threshold": args.threshold,
            "max_depth": args.max_depth,
            "chunk_size": args.chunk_size,
            "top_k": args.top_k,
            "retrieve_every_n": args.retrieve_every_n,
            "retrieval_min_context": args.retrieval_min_context,
            "plugin_mask": args.plugin_mask,
            "disable_hybrid_tree_attn": args.disable_hybrid_tree_attn,
        },
        "environment": collect_environment(device),
        "case": {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "input_token_count": case["input_token_count"],
            "context_target_tokens": case["context_target_tokens"],
        },
        "result": result,
        "analysis": analysis,
        "windows": rows,
    }
    json_path = args.output_dir / "hazard_stability.json"
    csv_path = args.output_dir / "windows.csv"
    report_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(sanitize(output), indent=2, ensure_ascii=False) + "\n")
    write_window_csv(csv_path, rows)
    write_report(report_path, sanitize(output), rows)
    print(colored(f"Wrote {json_path}", "green"))
    print(colored(f"Wrote {csv_path}", "green"))
    print(colored(f"Wrote {report_path}", "green"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
