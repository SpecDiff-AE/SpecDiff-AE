#!/usr/bin/env python3
"""Profile real Auto vs DiffSpec inference runs.

This script intentionally uses the same DiffSpec inference entry points as
benchmarks/paired_inference_benchmark.py. It adds runtime instrumentation around the
real generation path instead of profiling an isolated KV microbenchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
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

from benchmarks.common import (
    format_value as fmt,
    mean,
    parse_plugin_mask,
    percentile,
    safe_div,
    sanitize_json as sanitize,
)
from benchmarks.paired_inference_benchmark import (
    build_context_cases,
    collect_environment,
    prepare_input_ids,
    measure_generation,
)
from diffspec.defaults import DEFAULT_BASE_MODEL, DEFAULT_DRAFT_MODEL
from diffspec.draft.diffspec_model import DiffSpecModel


class NvidiaSmiSampler:
    def __init__(self, gpu_index: int, interval_ms: int):
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.samples: list[dict[str, float | str]] = []
        self._stop = threading.Event()

    def start(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            str(self.gpu_index),
            "-lms",
            str(self.interval_ms),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            sample = {
                "timestamp": parts[0],
                "gpu_index": _parse_float(parts[1]),
                "gpu_util_pct": _parse_float(parts[2]),
                "mem_util_pct": _parse_float(parts[3]),
                "memory_used_mb": _parse_float(parts[4]),
                "power_w": _parse_float(parts[5]),
            }
            self.samples.append(sample)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=3)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        gpu = [float(s["gpu_util_pct"]) for s in self.samples if s.get("gpu_util_pct") is not None]
        mem = [float(s["mem_util_pct"]) for s in self.samples if s.get("mem_util_pct") is not None]
        mem_used = [float(s["memory_used_mb"]) for s in self.samples if s.get("memory_used_mb") is not None]
        power = [float(s["power_w"]) for s in self.samples if s.get("power_w") is not None]
        return {
            "sample_count": len(self.samples),
            "gpu_util_avg_pct": mean(gpu),
            "gpu_util_p95_pct": percentile(gpu, 95),
            "gpu_util_max_pct": max(gpu) if gpu else None,
            "mem_util_avg_pct": mean(mem),
            "mem_util_p95_pct": percentile(mem, 95),
            "mem_util_max_pct": max(mem) if mem else None,
            "memory_used_avg_mb": mean(mem_used),
            "memory_used_max_mb": max(mem_used) if mem_used else None,
            "power_avg_w": mean(power),
            "power_max_w": max(power) if power else None,
        }


def _parse_float(text: str) -> float | None:
    try:
        cleaned = text.strip().replace(" W", "").replace(" MiB", "")
        if cleaned in {"", "[N/A]", "N/A"}:
            return None
        return float(cleaned)
    except ValueError:
        return None


def warmup(method: str, model: DiffSpecModel, input_ids: torch.Tensor, args: argparse.Namespace) -> None:
    warm_args = argparse.Namespace(**vars(args))
    warm_args.max_new_tokens = min(args.warmup_max_new_tokens, args.max_new_tokens)
    for _ in range(args.warmup_runs):
        _ = measure_generation(method, model, input_ids, warm_args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def run_one(method: str, model: DiffSpecModel, input_ids: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(input_ids.device)
        before_alloc = torch.cuda.memory_allocated(input_ids.device)
        before_reserved = torch.cuda.memory_reserved(input_ids.device)
    else:
        before_alloc = before_reserved = 0

    sampler = NvidiaSmiSampler(args.gpu_index, args.smi_interval_ms)
    sampler.start()
    outer_start = time.perf_counter()
    result = measure_generation(method, model, input_ids, args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    outer_wall = time.perf_counter() - outer_start
    smi_summary = sampler.stop()

    if torch.cuda.is_available():
        after_alloc = torch.cuda.memory_allocated(input_ids.device)
        after_reserved = torch.cuda.memory_reserved(input_ids.device)
        peak_alloc = torch.cuda.max_memory_allocated(input_ids.device)
        peak_reserved = torch.cuda.max_memory_reserved(input_ids.device)
    else:
        after_alloc = after_reserved = peak_alloc = peak_reserved = 0

    out = sanitize(result)
    out["outer_wall_time"] = outer_wall
    out["cuda_memory"] = {
        "before_allocated_mb": before_alloc / (1024.0 * 1024.0),
        "after_allocated_mb": after_alloc / (1024.0 * 1024.0),
        "peak_allocated_mb": peak_alloc / (1024.0 * 1024.0),
        "before_reserved_mb": before_reserved / (1024.0 * 1024.0),
        "after_reserved_mb": after_reserved / (1024.0 * 1024.0),
        "peak_reserved_mb": peak_reserved / (1024.0 * 1024.0),
    }
    out["nvidia_smi"] = smi_summary

    avg_power = smi_summary.get("power_avg_w")
    total_generated = out.get("total_generated")
    if avg_power is not None:
        out["estimated_energy_j"] = float(avg_power) * outer_wall
        out["estimated_energy_j_per_token"] = safe_div(out["estimated_energy_j"], total_generated)
    return out


def derive(auto: dict[str, Any], diffspec: dict[str, Any]) -> dict[str, Any]:
    ds_profile = diffspec.get("profile_stats") or {}
    iterations = ds_profile.get("iterations")
    ds_generated = diffspec.get("total_generated")
    auto_generated = auto.get("total_generated")
    return {
        "speedup_tokens_per_sec": safe_div(diffspec.get("tokens_per_sec"), auto.get("tokens_per_sec")),
        "speedup_decode_tokens_per_sec": safe_div(
            diffspec.get("decode_tokens_per_sec"), auto.get("decode_tokens_per_sec")
        ),
        "wall_time_reduction_pct": (
            100.0 * (float(auto["outer_wall_time"]) - float(diffspec["outer_wall_time"])) / float(auto["outer_wall_time"])
            if auto.get("outer_wall_time") else None
        ),
        "diffspec_iterations": iterations,
        "diffspec_tokens_per_tree_verify": safe_div(ds_generated, iterations),
        "target_decode_call_reduction_pct": (
            100.0 * (1.0 - safe_div(iterations, ds_generated))
            if iterations is not None and ds_generated else None
        ),
        "avg_accept_length": diffspec.get("avg_accept_length"),
        "auto_decode_time_s": auto.get("decode_time"),
        "diffspec_decode_time_s": diffspec.get("decode_time"),
        "decode_time_reduction_pct": (
            100.0 * (float(auto["decode_time"]) - float(diffspec["decode_time"])) / float(auto["decode_time"])
            if auto.get("decode_time") else None
        ),
        "prefill_time_s": auto.get("prefill_time"),
        "diffspec_initial_setup_s": ds_profile.get("initial_setup_time"),
        "diffspec_target_tree_decode_s": ds_profile.get("target_tree_decode_time"),
        "diffspec_verify_next_draft_s": ds_profile.get("verify_and_next_draft_time"),
        "gpu_peak_allocated_mb_delta": (
            float(diffspec["cuda_memory"]["peak_allocated_mb"]) - float(auto["cuda_memory"]["peak_allocated_mb"])
            if auto.get("cuda_memory") and diffspec.get("cuda_memory") else None
        ),
        "gpu_smi_memory_used_mb_delta": (
            float(diffspec["nvidia_smi"]["memory_used_max_mb"]) - float(auto["nvidia_smi"]["memory_used_max_mb"])
            if auto.get("nvidia_smi", {}).get("memory_used_max_mb") is not None
            and diffspec.get("nvidia_smi", {}).get("memory_used_max_mb") is not None else None
        ),
        "energy_per_token_reduction_pct": (
            100.0
            * (
                float(auto["estimated_energy_j_per_token"])
                - float(diffspec["estimated_energy_j_per_token"])
            )
            / float(auto["estimated_energy_j_per_token"])
            if auto.get("estimated_energy_j_per_token") else None
        ),
        "generated_token_delta": (
            int(ds_generated) - int(auto_generated)
            if ds_generated is not None and auto_generated is not None else None
        ),
    }


def write_csv(path: Path, auto: dict[str, Any], diffspec: dict[str, Any], derived: dict[str, Any]) -> None:
    rows = []
    for name, result in [("Auto", auto), ("DiffSpec", diffspec)]:
        rows.append(
            {
                "method": name,
                "tokens_per_sec": result.get("tokens_per_sec"),
                "decode_tokens_per_sec": result.get("decode_tokens_per_sec"),
                "total_generated": result.get("total_generated"),
                "inference_time_s": result.get("inference_time"),
                "outer_wall_time_s": result.get("outer_wall_time"),
                "decode_time_s": result.get("decode_time"),
                "prefill_or_setup_s": result.get("prefill_time")
                if name == "Auto"
                else (result.get("profile_stats") or {}).get("initial_setup_time"),
                "avg_accept_length": result.get("avg_accept_length", 1.0),
                "gpu_peak_allocated_mb": result.get("cuda_memory", {}).get("peak_allocated_mb"),
                "gpu_peak_reserved_mb": result.get("cuda_memory", {}).get("peak_reserved_mb"),
                "smi_gpu_util_avg_pct": result.get("nvidia_smi", {}).get("gpu_util_avg_pct"),
                "smi_mem_util_avg_pct": result.get("nvidia_smi", {}).get("mem_util_avg_pct"),
                "smi_memory_used_max_mb": result.get("nvidia_smi", {}).get("memory_used_max_mb"),
                "estimated_energy_j_per_token": result.get("estimated_energy_j_per_token"),
            }
        )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    derived_path = path.with_name(path.stem + "_derived.csv")
    with derived_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in derived.items():
            writer.writerow({"metric": key, "value": value})


def write_report(
    path: Path,
    output: dict[str, Any],
    auto: dict[str, Any],
    diffspec: dict[str, Any],
    derived: dict[str, Any],
) -> None:
    config = output["config"]
    env = output["environment"]
    ds_profile = diffspec.get("profile_stats") or {}
    runtime_policy = diffspec.get("runtime_policy") or {}
    lines = [
        "# Real Inference Profile: Auto vs DiffSpec",
        "",
        "This report measures the actual model inference path, not an isolated KV microbenchmark.",
        "",
        "## Workload",
        "",
        f"- dataset: {', '.join(config['data_files'])}",
        f"- context_target_tokens: {config['context_target']}",
        f"- measured_input_tokens: {output['case']['input_token_count']}",
        f"- max_new_tokens: {config['max_new_tokens']}",
        f"- nodes/threshold/max_depth: {config['nodes']} / {config['threshold']} / {config['max_depth']}",
        f"- plugin_mask: {config['plugin_mask']}",
        f"- GPU: {env.get('gpu_name')} ({env.get('gpu_total_memory_gb')} GiB)",
        f"- attention_backend: {env.get('attention_backend')}",
        "",
        "## Headline Evidence",
        "",
        (
            f"- DiffSpec throughput is {fmt(derived.get('speedup_tokens_per_sec'), 2)}x Auto "
            f"({fmt(auto.get('tokens_per_sec'), 2)} -> {fmt(diffspec.get('tokens_per_sec'), 2)} tok/s)."
        ),
        (
            f"- Decode-only throughput is {fmt(derived.get('speedup_decode_tokens_per_sec'), 2)}x Auto "
            f"({fmt(auto.get('decode_tokens_per_sec'), 2)} -> {fmt(diffspec.get('decode_tokens_per_sec'), 2)} tok/s)."
        ),
        (
            f"- DiffSpec emits {fmt(derived.get('diffspec_tokens_per_tree_verify'), 2)} tokens per target tree verification, "
            f"cutting target decode calls per output token by {fmt(derived.get('target_decode_call_reduction_pct'), 2)}%."
        ),
        (
            f"- Estimated energy per token is reduced by {fmt(derived.get('energy_per_token_reduction_pct'), 2)}% "
            "from the sampled nvidia-smi power data."
        ),
        "",
        "## End-to-End Metrics",
        "",
        "| Method | Tokens/s | Decode tokens/s | Generated | Inference s | Decode s | Prefill/setup s | Avg accept | Peak alloc MB | SMI GPU util avg % | SMI mem util avg % | Energy J/token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _method_row("Auto", auto),
        _method_row("DiffSpec", diffspec),
        "",
        "## Algorithmic Efficiency",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Speedup vs Auto | {fmt(derived.get('speedup_tokens_per_sec'), 3)}x |",
        f"| Decode speedup vs Auto | {fmt(derived.get('speedup_decode_tokens_per_sec'), 3)}x |",
        f"| Wall time reduction | {fmt(derived.get('wall_time_reduction_pct'), 2)}% |",
        f"| Decode time reduction | {fmt(derived.get('decode_time_reduction_pct'), 2)}% |",
        f"| DiffSpec iterations | {fmt(derived.get('diffspec_iterations'), 0)} |",
        f"| Tokens per target tree verification | {fmt(derived.get('diffspec_tokens_per_tree_verify'), 2)} |",
        f"| Target decode call reduction per output token | {fmt(derived.get('target_decode_call_reduction_pct'), 2)}% |",
        f"| Avg accept length | {fmt(derived.get('avg_accept_length'), 2)} |",
        "",
        "## Real GPU Sampling",
        "",
        "| Method | Samples | GPU util avg % | GPU util p95 % | Mem util avg % | Mem util p95 % | Max used MB | Avg power W |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _smi_row("Auto", auto),
        _smi_row("DiffSpec", diffspec),
        "",
        "## DiffSpec Internal Timing",
        "",
        "| Component | Seconds | Share of DiffSpec decode |",
        "|---|---:|---:|",
        _component_row("initial_setup", ds_profile.get("initial_setup_time"), diffspec.get("decode_time")),
        _component_row("target_tree_decode", ds_profile.get("target_tree_decode_time"), diffspec.get("decode_time")),
        _component_row("verify_and_next_draft", ds_profile.get("verify_and_next_draft_time"), diffspec.get("decode_time")),
        "",
        "## Runtime Policy",
        "",
        "```json",
        json.dumps(runtime_policy, indent=2, sort_keys=True),
        "```",
        "",
        "## Instrumentation Changelog",
        "",
        "| File | Change type | What was added/modified |",
        "|---|---|---|",
        "| `benchmarks/inference_profile.py` | created | Real inference profiling harness for Auto vs DiffSpec |",
    ]
    path.write_text("\n".join(lines) + "\n")


def _method_row(name: str, result: dict[str, Any]) -> str:
    setup = result.get("prefill_time")
    if setup is None:
        setup = (result.get("profile_stats") or {}).get("initial_setup_time")
    return (
        f"| {name} | {fmt(result.get('tokens_per_sec'), 2)} | "
        f"{fmt(result.get('decode_tokens_per_sec'), 2)} | "
        f"{fmt(result.get('total_generated'), 0)} | "
        f"{fmt(result.get('inference_time'), 2)} | "
        f"{fmt(result.get('decode_time'), 2)} | "
        f"{fmt(setup, 2)} | "
        f"{fmt(result.get('avg_accept_length', 1.0), 2)} | "
        f"{fmt(result.get('cuda_memory', {}).get('peak_allocated_mb'), 1)} | "
        f"{fmt(result.get('nvidia_smi', {}).get('gpu_util_avg_pct'), 1)} | "
        f"{fmt(result.get('nvidia_smi', {}).get('mem_util_avg_pct'), 1)} | "
        f"{fmt(result.get('estimated_energy_j_per_token'), 2)} |"
    )


def _smi_row(name: str, result: dict[str, Any]) -> str:
    smi = result.get("nvidia_smi") or {}
    return (
        f"| {name} | {fmt(smi.get('sample_count'), 0)} | "
        f"{fmt(smi.get('gpu_util_avg_pct'), 1)} | "
        f"{fmt(smi.get('gpu_util_p95_pct'), 1)} | "
        f"{fmt(smi.get('mem_util_avg_pct'), 1)} | "
        f"{fmt(smi.get('mem_util_p95_pct'), 1)} | "
        f"{fmt(smi.get('memory_used_max_mb'), 0)} | "
        f"{fmt(smi.get('power_avg_w'), 1)} |"
    )


def _component_row(name: str, seconds: float | None, decode_time: float | None) -> str:
    share = 100.0 * safe_div(seconds, decode_time) if seconds is not None and decode_time else None
    return f"| {name} | {fmt(seconds, 3)} | {fmt(share, 2)}% |"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "diffspec" / "data"))
    parser.add_argument("--data-files", nargs="+", default=["govreport/govreport_16K.jsonl"])
    parser.add_argument("--context-target", type=int, default=128000)
    parser.add_argument("--samples-per-context", type=int, default=1)
    parser.add_argument("--max-source-records", type=int, default=16)
    parser.add_argument("--context-tolerance", type=float, default=0.08)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-max-new-tokens", type=int, default=16)
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
    parser.add_argument("--smi-interval-ms", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "real_inference_profile")
    parser.add_argument("--store-generated-text", action="store_true")
    args = parser.parse_args()
    args.plugin_mask = parse_plugin_mask(args.plugin_mask)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real inference profiling")
    device = torch.device(f"cuda:{args.gpu_index}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(colored("Loading model...", "cyan"))
    model = DiffSpecModel.from_pretrained(
        base_model_path=args.base_model,
        draft_model_path=args.draft_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    tokenizer = model.tokenizer

    print(colored("Building real inference case...", "cyan"))
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

    print(colored("Warming up Auto...", "yellow"))
    warmup("Auto", model, input_ids, args)
    print(colored("Measuring Auto real inference...", "cyan"))
    auto = run_one("Auto", model, input_ids, args)

    print(colored("Warming up DiffSpec...", "yellow"))
    warmup("DiffSpec", model, input_ids, args)
    print(colored("Measuring DiffSpec real inference...", "cyan"))
    diffspec = run_one("DiffSpec", model, input_ids, args)

    if not args.store_generated_text:
        for result in (auto, diffspec):
            result.pop("generated_text", None)
            result.pop("generated_token_ids", None)

    derived = derive(auto, diffspec)
    case_public = {
        key: value
        for key, value in case.items()
        if key not in {"input_text", "runs", "paired_runs"}
    }
    case_public["input_text_chars"] = len(case.get("input_text", ""))

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": args.base_model,
            "draft_model": args.draft_model,
            "data_dir": args.data_dir,
            "data_files": args.data_files,
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
        "case": case_public,
        "Auto": auto,
        "DiffSpec": diffspec,
        "derived": derived,
    }

    json_path = args.output_dir / "real_inference_profile.json"
    csv_path = args.output_dir / "real_inference_profile.csv"
    md_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(sanitize(output), ensure_ascii=False, indent=2) + "\n")
    write_csv(csv_path, auto, diffspec, derived)
    write_report(md_path, sanitize(output), auto, diffspec, derived)
    print(colored(f"Wrote {json_path}", "green"))
    print(colored(f"Wrote {csv_path}", "green"))
    print(colored(f"Wrote {md_path}", "green"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
