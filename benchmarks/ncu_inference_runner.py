#!/usr/bin/env python3
"""Run one real inference method under Nsight Compute.

The model exposes DIFFSPEC_NCU_PROFILE_PHASE hooks around decode loops. This
runner keeps warmup outside that range, then prints one RESULT_JSON payload for
the outer NCU driver to parse.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("DIFFSPEC_PROFILE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from termcolor import colored

from benchmarks.common import parse_plugin_mask, sanitize_json as sanitize
from benchmarks.paired_inference_benchmark import (
    build_context_cases,
    collect_environment,
    prepare_input_ids,
    measure_generation,
)
from diffspec.defaults import DEFAULT_BASE_MODEL, DEFAULT_DRAFT_MODEL
from diffspec.draft.diffspec_model import DiffSpecModel


def warmup(method: str, model: DiffSpecModel, input_ids: torch.Tensor, args: argparse.Namespace) -> None:
    if args.warmup_runs <= 0:
        return
    old_phase = os.environ.pop("DIFFSPEC_NCU_PROFILE_PHASE", None)
    warm_args = argparse.Namespace(**vars(args))
    warm_args.max_new_tokens = min(args.warmup_max_new_tokens, args.max_new_tokens)
    try:
        for _ in range(args.warmup_runs):
            _ = measure_generation(method, model, input_ids, warm_args)
    finally:
        if old_phase is not None:
            os.environ["DIFFSPEC_NCU_PROFILE_PHASE"] = old_phase
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def run_measured(method: str, model: DiffSpecModel, input_ids: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(input_ids.device)
        before_alloc = torch.cuda.memory_allocated(input_ids.device)
        before_reserved = torch.cuda.memory_reserved(input_ids.device)
    else:
        before_alloc = before_reserved = 0

    if args.profile_phase:
        os.environ["DIFFSPEC_NCU_PROFILE_PHASE"] = args.profile_phase

    outer_start = time.perf_counter()
    result = measure_generation(method, model, input_ids, args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    outer_wall = time.perf_counter() - outer_start

    if torch.cuda.is_available():
        after_alloc = torch.cuda.memory_allocated(input_ids.device)
        after_reserved = torch.cuda.memory_reserved(input_ids.device)
        peak_alloc = torch.cuda.max_memory_allocated(input_ids.device)
        peak_reserved = torch.cuda.max_memory_reserved(input_ids.device)
    else:
        after_alloc = after_reserved = peak_alloc = peak_reserved = 0

    out = sanitize(result)
    out.pop("generated_text", None)
    out.pop("generated_token_ids", None)
    out["outer_wall_time"] = outer_wall
    out["cuda_memory"] = {
        "before_allocated_mb": before_alloc / (1024.0 * 1024.0),
        "after_allocated_mb": after_alloc / (1024.0 * 1024.0),
        "peak_allocated_mb": peak_alloc / (1024.0 * 1024.0),
        "before_reserved_mb": before_reserved / (1024.0 * 1024.0),
        "after_reserved_mb": after_reserved / (1024.0 * 1024.0),
        "peak_reserved_mb": peak_reserved / (1024.0 * 1024.0),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["Auto", "DiffSpec"], required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "diffspec" / "data"))
    parser.add_argument("--data-files", nargs="+", default=["govreport/govreport_16K.jsonl"])
    parser.add_argument("--context-target", type=int, default=16000)
    parser.add_argument("--samples-per-context", type=int, default=1)
    parser.add_argument("--max-source-records", type=int, default=16)
    parser.add_argument("--context-tolerance", type=float, default=0.08)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--warmup-max-new-tokens", type=int, default=4)
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
    parser.add_argument(
        "--profile-phase",
        default=None,
        help="Set DIFFSPEC_NCU_PROFILE_PHASE for the measured run, e.g. auto_decode or diffspec_decode.",
    )
    args = parser.parse_args()
    args.plugin_mask = parse_plugin_mask(args.plugin_mask)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCU real inference profiling")
    device = torch.device(f"cuda:{args.gpu_index}")

    print(colored(f"Loading model for {args.method}...", "cyan"), flush=True)
    model = DiffSpecModel.from_pretrained(
        base_model_path=args.base_model,
        draft_model_path=args.draft_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    tokenizer = model.tokenizer

    print(colored("Building real inference case...", "cyan"), flush=True)
    cases = build_context_cases(
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir),
        data_files=args.data_files,
        context_targets=[args.context_target],
        samples_per_context=args.samples_per_context,
        tolerance=args.context_tolerance,
        max_source_records=args.max_source_records,
    )
    case = cases[0]
    input_ids = prepare_input_ids(tokenizer, case["input_text"], device)
    case["input_token_count"] = int(input_ids.shape[1])

    warmup(args.method, model, input_ids, args)
    print(colored(f"Measuring {args.method} real inference...", "cyan"), flush=True)
    result = run_measured(args.method, model, input_ids, args)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "method": args.method,
        "profile_phase": args.profile_phase,
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
            "context_target_tokens": case["context_target_tokens"],
            "input_token_count": case["input_token_count"],
            "input_text_chars": len(case.get("input_text", "")),
        },
        "result": result,
    }
    print("RESULT_JSON " + json.dumps(sanitize(payload), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
