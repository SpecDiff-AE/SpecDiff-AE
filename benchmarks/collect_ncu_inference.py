#!/usr/bin/env python3
"""Collect decode-only Nsight Compute counters for real Auto/DiffSpec inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results" / "real_inference_ncu"
CONTAINER_DRAFT_ENV = "/opt/diffspec/draft-env"
CONTAINER_BASE_MODEL = "/models/base"
CONTAINER_DRAFT_MODEL = "/models/draft"
DEFAULT_NCU_ROOT = os.environ.get("DIFFSPEC_NCU_ROOT", str(Path.home() / ".cache" / "ncu2025"))
DEFAULT_DRAFT_ENV_CACHE = os.environ.get(
    "DIFFSPEC_DRAFT_ENV_CACHE",
    str(Path.home() / ".cache" / "diffspec" / "envs" / "draft"),
)
DEFAULT_BASE_MODEL_CACHE = os.environ.get(
    "DIFFSPEC_BASE_MODEL_CACHE",
    str(Path.home() / ".cache" / "diffspec" / "models" / "base"),
)
DEFAULT_DRAFT_MODEL_CACHE = os.environ.get(
    "DIFFSPEC_DRAFT_MODEL_CACHE",
    str(Path.home() / ".cache" / "diffspec" / "models" / "draft"),
)

METRIC_CANDIDATES = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    "sm__maximum_warps_per_active_cycle_pct",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors.sum",
    "lts__d_sectors_fill_device.sum",
    "l1tex__t_bytes.sum",
    "l1tex__t_bytes_lookup_hit.sum",
    "l1tex__t_bytes_lookup_miss.sum",
]


def run(cmd: list[str], cwd: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc


def docker_cmd(args: argparse.Namespace, inner: list[str]) -> list[str]:
    mounts = [
        (REPO_ROOT, "/work", False),
        (Path(args.ncu_root), "/opt/ncu", True),
        (Path(args.draft_env_cache), CONTAINER_DRAFT_ENV, True),
        (Path(args.base_model_cache), CONTAINER_BASE_MODEL, True),
        (Path(args.draft_model_cache), CONTAINER_DRAFT_MODEL, True),
    ]
    cmd = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--gpus",
        "all",
        "--privileged",
        "--cap-add=SYS_ADMIN",
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        "--pid=host",
        "--ipc=host",
        "-e",
        "HOME=/tmp",
        "-e",
        "HF_ENDPOINT=https://hf-mirror.com",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "DIFFSPEC_PROFILE=1",
    ]
    for host, target, readonly in mounts:
        suffix = ":ro" if readonly else ""
        cmd.extend(["-v", f"{host.resolve()}:{target}{suffix}"])
    shell_cmd = " ".join(shlex.quote(part) for part in inner)
    cmd.extend(["-w", "/work", args.docker_image, "bash", "-lc", shell_cmd])
    return cmd


def check_inputs(args: argparse.Namespace) -> None:
    required = {
        "ncu": Path(args.ncu_root) / "ncu",
        "draft env": Path(args.draft_env_cache) / "bin" / "python",
        "base model cache": Path(args.base_model_cache),
        "draft model cache": Path(args.draft_model_cache),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))


def query_metrics(args: argparse.Namespace, out_dir: Path) -> set[str]:
    inner = ["/opt/ncu/ncu", "--query-metrics", "--query-metrics-mode", "all", "--chips", args.chip]
    proc = run(docker_cmd(args, inner), REPO_ROOT)
    text = proc.stdout + proc.stderr
    (out_dir / "available_metrics.raw.txt").write_text(text)
    metrics: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Chip "):
            continue
        first = stripped.split(None, 1)[0]
        if "__" in first or first.startswith("gpu__"):
            metrics.add(first)
    if not metrics:
        return set(METRIC_CANDIDATES)
    availability = [
        f"{metric}: {'yes' if metric in metrics else 'no'}"
        for metric in METRIC_CANDIDATES
    ]
    (out_dir / "available_metrics.txt").write_text("\n".join(availability) + "\n")
    return metrics


def select_metrics(available: set[str]) -> list[str]:
    selected = [metric for metric in METRIC_CANDIDATES if metric in available]
    for required in ("gpu__time_duration.sum", "dram__bytes_read.sum"):
        if required not in selected:
            selected.append(required)
    seen: set[str] = set()
    out: list[str] = []
    for metric in selected:
        if metric not in seen:
            seen.add(metric)
            out.append(metric)
    return out


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan"}:
        return None
    text = text.rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def _is_metric_name(name: str) -> bool:
    return "__" in name or name.startswith("gpu__") or name.startswith("dram__")


def parse_ncu_rows(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    long_header: list[str] | None = None
    wide_header: list[str] | None = None
    grouped: dict[tuple[str, ...], dict[str, float]] = {}

    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if "Metric Name" in row and "Metric Value" in row:
            long_header = row
            wide_header = None
            continue
        if long_header is not None and len(row) == len(long_header):
            rec = dict(zip(long_header, row))
            name = rec.get("Metric Name")
            value = parse_float(rec.get("Metric Value"))
            if name and value is not None:
                key_cols = [
                    "ID",
                    "Process ID",
                    "Kernel Name",
                    "Context",
                    "Stream",
                    "Section Name",
                ]
                key = tuple(rec.get(col, "") for col in key_cols if col in rec)
                grouped.setdefault(key, {})[name] = value
            continue
        if any(_is_metric_name(col) for col in row):
            wide_header = row
            long_header = None
            continue
        if wide_header is None or len(row) != len(wide_header):
            continue
        metrics: dict[str, float] = {}
        for name, value_text in zip(wide_header, row):
            if not _is_metric_name(name):
                continue
            value = parse_float(value_text)
            if value is not None:
                metrics[name] = value
        if metrics:
            rows.append(metrics)

    if grouped:
        rows.extend(grouped.values())
    return rows


def parse_result_json(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if "RESULT_JSON" not in line:
            continue
        _, payload = line.split("RESULT_JSON", 1)
        try:
            return json.loads(payload.strip())
        except json.JSONDecodeError:
            match = re.search(r"RESULT_JSON\s+(\{.*\})", line)
            if match:
                return json.loads(match.group(1))
    return {}


def sum_metric(rows: list[dict[str, float]], name: str) -> float | None:
    vals = [row[name] for row in rows if name in row]
    return sum(vals) if vals else None


def weighted_avg_metric(rows: list[dict[str, float]], name: str) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    plain: list[float] = []
    for row in rows:
        value = row.get(name)
        if value is None:
            continue
        plain.append(value)
        weight = row.get("gpu__time_duration.sum")
        if weight is not None and weight > 0:
            weighted_sum += value * weight
            weight_sum += weight
    if weight_sum > 0:
        return weighted_sum / weight_sum
    return sum(plain) / len(plain) if plain else None


def safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None:
        return None
    den_f = float(den)
    if den_f == 0.0:
        return None
    return float(num) / den_f


def bytes_to_mib(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def summarize_ncu(method: str, stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    rows = parse_ncu_rows(stdout)
    program = parse_result_json(stdout + "\n" + stderr)
    result = program.get("result") or {}
    duration_ns = sum_metric(rows, "gpu__time_duration.sum")
    read_bytes = sum_metric(rows, "dram__bytes_read.sum")
    write_bytes = sum_metric(rows, "dram__bytes_write.sum")
    total_dram_bytes = (read_bytes or 0.0) + (write_bytes or 0.0) if read_bytes is not None or write_bytes is not None else None
    duration_s = duration_ns / 1.0e9 if duration_ns is not None else None

    l2_source = "unavailable"
    l2_hit = None
    l2_hit_sectors = sum_metric(rows, "lts__t_sectors_lookup_hit.sum")
    l2_total_sectors = sum_metric(rows, "lts__t_sectors.sum")
    l2_fill_sectors = sum_metric(rows, "lts__d_sectors_fill_device.sum")
    if l2_hit_sectors is not None and l2_total_sectors and l2_total_sectors > 0:
        l2_hit = 100.0 * l2_hit_sectors / l2_total_sectors
        l2_source = "lts__t_sectors_lookup_hit / lts__t_sectors"
    else:
        l2_hit = weighted_avg_metric(rows, "lts__t_sector_hit_rate.pct")
        if l2_hit is not None:
            l2_source = "time-weighted lts__t_sector_hit_rate.pct"
    if l2_hit is None and l2_fill_sectors is not None and l2_total_sectors and l2_total_sectors > 0:
        l2_hit = 100.0 * (1.0 - l2_fill_sectors / l2_total_sectors)
        l2_source = "1 - lts__d_sectors_fill_device / lts__t_sectors"
    if l2_hit is not None:
        l2_hit = max(0.0, min(100.0, l2_hit))

    l1tex_bytes = sum_metric(rows, "l1tex__t_bytes.sum")
    l1tex_hit_bytes = sum_metric(rows, "l1tex__t_bytes_lookup_hit.sum")
    l1tex_miss_bytes = sum_metric(rows, "l1tex__t_bytes_lookup_miss.sum")
    l1tex_hit_rate = safe_div(100.0 * l1tex_hit_bytes, (l1tex_hit_bytes or 0.0) + (l1tex_miss_bytes or 0.0))

    errors = []
    for line in (stdout + "\n" + stderr).splitlines():
        if (
            "ERR_NVGPUCTRPERM" in line
            or "No kernels were profiled" in line
            or "Failed to prepare kernel" in line
            or "Failed to profile" in line
        ):
            errors.append(line.strip())

    total_generated = result.get("total_generated")
    decode_time = result.get("decode_time")
    kernel_time_ms = duration_ns / 1.0e6 if duration_ns is not None else None
    return {
        "method": method,
        "ncu_error": returncode != 0,
        "ncu_returncode": returncode,
        "ncu_error_summary": "\n".join(errors),
        "profiled_kernel_count": len(rows),
        "profiled_kernel_time_ms": kernel_time_ms,
        "program_total_generated": total_generated,
        "program_tokens_per_sec": result.get("tokens_per_sec"),
        "program_decode_tokens_per_sec": result.get("decode_tokens_per_sec"),
        "program_inference_time_s": result.get("inference_time"),
        "program_decode_time_s": decode_time,
        "program_outer_wall_time_s": result.get("outer_wall_time"),
        "program_avg_accept_length": result.get("avg_accept_length", 1.0 if method == "Auto" else None),
        "program_prefill_or_setup_s": result.get("prefill_time")
        if method == "Auto"
        else (result.get("profile_stats") or {}).get("initial_setup_time"),
        "diffspec_iterations": (result.get("profile_stats") or {}).get("iterations"),
        "diffspec_tokens_per_tree_verify": safe_div(total_generated, (result.get("profile_stats") or {}).get("iterations")),
        "l2_hit_rate_pct": l2_hit,
        "l2_hit_rate_source": l2_source,
        "l2_hit_sectors_m": l2_hit_sectors / 1.0e6 if l2_hit_sectors is not None else None,
        "l2_total_sectors_m": l2_total_sectors / 1.0e6 if l2_total_sectors is not None else None,
        "l2_device_fill_sectors_m": l2_fill_sectors / 1.0e6 if l2_fill_sectors is not None else None,
        "hbm_read_mb": bytes_to_mib(read_bytes),
        "hbm_write_mb": bytes_to_mib(write_bytes),
        "hbm_total_mb": bytes_to_mib(total_dram_bytes),
        "achieved_hbm_read_bw_gbps": safe_div(read_bytes, duration_s * 1.0e9 if duration_s else None),
        "achieved_hbm_write_bw_gbps": safe_div(write_bytes, duration_s * 1.0e9 if duration_s else None),
        "achieved_hbm_total_bw_gbps": safe_div(total_dram_bytes, duration_s * 1.0e9 if duration_s else None),
        "dram_throughput_pct": weighted_avg_metric(rows, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
        or weighted_avg_metric(rows, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        "sm_utilization_pct": weighted_avg_metric(rows, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "kernel_occupancy_pct": weighted_avg_metric(rows, "sm__warps_active.avg.pct_of_peak_sustained_active")
        or weighted_avg_metric(rows, "sm__warps_active.avg.pct_of_peak_sustained_elapsed")
        or weighted_avg_metric(rows, "sm__maximum_warps_per_active_cycle_pct"),
        "l1tex_traffic_mb": bytes_to_mib(l1tex_bytes),
        "l1tex_hit_rate_pct": l1tex_hit_rate,
        "profiled_kernel_ms_per_generated_token": safe_div(kernel_time_ms, total_generated),
        "hbm_read_mb_per_generated_token": safe_div(bytes_to_mib(read_bytes), total_generated),
        "case_input_token_count": (program.get("case") or {}).get("input_token_count"),
        "profile_phase": program.get("profile_phase"),
        "attention_backend": (program.get("environment") or {}).get("attention_backend"),
        "runtime_policy": result.get("runtime_policy"),
    }


def inference_args(args: argparse.Namespace, method: str) -> list[str]:
    phase = "auto_decode" if method == "Auto" else "diffspec_decode"
    return [
        f"{CONTAINER_DRAFT_ENV}/bin/python",
        "/work/benchmarks/ncu_inference_runner.py",
        "--method",
        method,
        "--base-model",
        CONTAINER_BASE_MODEL,
        "--draft-model",
        CONTAINER_DRAFT_MODEL,
        "--data-dir",
        "/work/diffspec/data",
        "--data-files",
        args.data_file,
        "--context-target",
        str(args.context_target),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--warmup-max-new-tokens",
        str(args.warmup_max_new_tokens),
        "--nodes",
        str(args.nodes),
        "--threshold",
        str(args.threshold),
        "--max-depth",
        str(args.max_depth),
        "--chunk-size",
        str(args.chunk_size),
        "--top-k",
        str(args.top_k),
        "--retrieve-every-n",
        str(args.retrieve_every_n),
        "--retrieval-min-context",
        str(args.retrieval_min_context),
        "--plugin-mask",
        args.plugin_mask,
        "--profile-phase",
        phase,
    ]


def run_ncu(method: str, metrics: list[str], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    raw_path = out_dir / f"{method.lower()}.decode.ncu.csv"
    err_path = out_dir / f"{method.lower()}.decode.ncu.stderr.txt"
    inner = [
        "/opt/ncu/ncu",
        "--csv",
        "--page",
        "raw",
        "--profile-from-start",
        "off",
        "--target-processes",
        "all",
        "--cache-control",
        args.cache_control,
        "--replay-mode",
        args.replay_mode,
        "--kernel-name-base",
        "function",
    ]
    if args.launch_count > 0:
        inner.extend(["--launch-count", str(args.launch_count)])
    inner.extend(["--metrics", ",".join(metrics)])
    inner.extend(inference_args(args, method))
    proc = run(docker_cmd(args, inner), REPO_ROOT)
    raw_path.write_text(proc.stdout)
    err_path.write_text(proc.stderr)
    summary = summarize_ncu(method, proc.stdout, proc.stderr, proc.returncode)
    summary["raw_ncu_csv"] = str(raw_path)
    summary["raw_ncu_stderr"] = str(err_path)
    return summary


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    return str(value)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def annotate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {row["method"]: row for row in rows}
    auto = by_method.get("Auto")
    ds = by_method.get("DiffSpec")
    if not auto or not ds:
        return {}
    return {
        "decode_speedup_vs_auto": safe_div(ds.get("program_decode_tokens_per_sec"), auto.get("program_decode_tokens_per_sec")),
        "end_to_end_speedup_vs_auto": safe_div(ds.get("program_tokens_per_sec"), auto.get("program_tokens_per_sec")),
        "profiled_kernel_time_reduction_pct": (
            100.0
            * (float(auto["profiled_kernel_time_ms"]) - float(ds["profiled_kernel_time_ms"]))
            / float(auto["profiled_kernel_time_ms"])
            if auto.get("profiled_kernel_time_ms") else None
        ),
        "hbm_read_reduction_pct": (
            100.0 * (float(auto["hbm_read_mb"]) - float(ds["hbm_read_mb"])) / float(auto["hbm_read_mb"])
            if auto.get("hbm_read_mb") else None
        ),
        "hbm_read_per_token_reduction_pct": (
            100.0
            * (
                float(auto["hbm_read_mb_per_generated_token"])
                - float(ds["hbm_read_mb_per_generated_token"])
            )
            / float(auto["hbm_read_mb_per_generated_token"])
            if auto.get("hbm_read_mb_per_generated_token") else None
        ),
        "l2_hit_delta_pp": (
            float(ds["l2_hit_rate_pct"]) - float(auto["l2_hit_rate_pct"])
            if auto.get("l2_hit_rate_pct") is not None and ds.get("l2_hit_rate_pct") is not None else None
        ),
        "dram_throughput_delta_pp": (
            float(ds["dram_throughput_pct"]) - float(auto["dram_throughput_pct"])
            if auto.get("dram_throughput_pct") is not None and ds.get("dram_throughput_pct") is not None else None
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "ncu_error",
        "profile_phase",
        "case_input_token_count",
        "program_total_generated",
        "program_tokens_per_sec",
        "program_decode_tokens_per_sec",
        "program_inference_time_s",
        "program_decode_time_s",
        "program_prefill_or_setup_s",
        "program_avg_accept_length",
        "diffspec_iterations",
        "diffspec_tokens_per_tree_verify",
        "profiled_kernel_count",
        "profiled_kernel_time_ms",
        "profiled_kernel_ms_per_generated_token",
        "l2_hit_rate_pct",
        "l2_hit_rate_source",
        "l2_hit_sectors_m",
        "l2_total_sectors_m",
        "l2_device_fill_sectors_m",
        "hbm_read_mb",
        "hbm_write_mb",
        "hbm_total_mb",
        "hbm_read_mb_per_generated_token",
        "achieved_hbm_read_bw_gbps",
        "achieved_hbm_write_bw_gbps",
        "achieved_hbm_total_bw_gbps",
        "dram_throughput_pct",
        "sm_utilization_pct",
        "kernel_occupancy_pct",
        "l1tex_traffic_mb",
        "l1tex_hit_rate_pct",
        "ncu_returncode",
        "ncu_error_summary",
        "raw_ncu_csv",
        "raw_ncu_stderr",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], derived: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Real Inference Nsight Compute Profile",
        "",
        "This profile uses the real DiffSpecModel inference path. The Nsight Compute collection range is decode-only via DIFFSPEC_NCU_PROFILE_PHASE hooks.",
        "",
        "## Workload",
        "",
        f"- data_file: {args.data_file}",
        f"- context_target_tokens: {args.context_target}",
        f"- max_new_tokens: {args.max_new_tokens}",
        f"- NCU launch_count: {args.launch_count if args.launch_count > 0 else 'all'}",
        f"- replay_mode/cache_control: {args.replay_mode} / {args.cache_control}",
        f"- nodes/threshold/max_depth: {args.nodes} / {args.threshold} / {args.max_depth}",
        f"- plugin_mask: {args.plugin_mask}",
        "",
            "## Program Timing Under NCU Replay",
            "",
            "Nsight Compute kernel replay perturbs wall time heavily. Use these timings only to identify the profiled workload shape; use a non-NCU run for throughput claims.",
            "",
            "| Method | Input tokens | Generated | Replay decode tok/s | Replay end-to-end tok/s | Replay decode s | Prefill/setup s | Avg accept | Iterations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {fmt(row.get('case_input_token_count'), 0)} | "
            f"{fmt(row.get('program_total_generated'), 0)} | "
            f"{fmt(row.get('program_decode_tokens_per_sec'), 2)} | "
            f"{fmt(row.get('program_tokens_per_sec'), 2)} | "
            f"{fmt(row.get('program_decode_time_s'), 3)} | "
            f"{fmt(row.get('program_prefill_or_setup_s'), 3)} | "
            f"{fmt(row.get('program_avg_accept_length'), 2)} | "
            f"{fmt(row.get('diffspec_iterations'), 0)} |"
        )
    lines.extend(
        [
            "",
            f"- Decode speedup vs Auto: {fmt(derived.get('decode_speedup_vs_auto'), 3)}x",
            f"- End-to-end speedup vs Auto: {fmt(derived.get('end_to_end_speedup_vs_auto'), 3)}x",
            "",
            "## Decode-Only NCU Counters",
            "",
            "| Method | Kernels | Kernel ms | L2 hit % | HBM read MB | HBM write MB | HBM total BW GB/s | DRAM % peak | SM util % | Occupancy % | L1TEX MB | L1TEX hit % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['method']} | {fmt(row.get('profiled_kernel_count'), 0)} | "
            f"{fmt(row.get('profiled_kernel_time_ms'), 3)} | "
            f"{fmt(row.get('l2_hit_rate_pct'), 2)} | "
            f"{fmt(row.get('hbm_read_mb'), 2)} | "
            f"{fmt(row.get('hbm_write_mb'), 2)} | "
            f"{fmt(row.get('achieved_hbm_total_bw_gbps'), 2)} | "
            f"{fmt(row.get('dram_throughput_pct'), 2)} | "
            f"{fmt(row.get('sm_utilization_pct'), 2)} | "
            f"{fmt(row.get('kernel_occupancy_pct'), 2)} | "
            f"{fmt(row.get('l1tex_traffic_mb'), 2)} | "
            f"{fmt(row.get('l1tex_hit_rate_pct'), 2)} |"
        )
    lines.extend(
        [
            "",
            "## DiffSpec vs Auto NCU Delta",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Profiled kernel time reduction | {fmt(derived.get('profiled_kernel_time_reduction_pct'), 2)}% |",
            f"| HBM read reduction | {fmt(derived.get('hbm_read_reduction_pct'), 2)}% |",
            f"| HBM read/token reduction | {fmt(derived.get('hbm_read_per_token_reduction_pct'), 2)}% |",
            f"| L2 hit-rate delta | {fmt(derived.get('l2_hit_delta_pp'), 2)} pp |",
            f"| DRAM throughput delta | {fmt(derived.get('dram_throughput_delta_pp'), 2)} pp |",
            "",
            "## Raw Artifacts",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row['method']}: `{row.get('raw_ncu_csv')}`, `{row.get('raw_ncu_stderr')}`")
    lines.extend(
        [
            "",
            "## Instrumentation Changelog",
            "",
            "| File | Change type | What was added/modified |",
            "|---|---|---|",
            "| `diffspec/draft/diffspec_model.py` | modified | Decode-only CUDA profiler hooks controlled by `DIFFSPEC_NCU_PROFILE_PHASE` |",
            "| `benchmarks/ncu_inference_runner.py` | created | Single-method real inference runner for NCU |",
            "| `benchmarks/collect_ncu_inference.py` | created | Docker/NCU collection, parsing, CSV and Markdown reporting |",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ncu-root", default=DEFAULT_NCU_ROOT)
    parser.add_argument("--draft-env-cache", default=DEFAULT_DRAFT_ENV_CACHE)
    parser.add_argument("--base-model-cache", default=DEFAULT_BASE_MODEL_CACHE)
    parser.add_argument("--draft-model-cache", default=DEFAULT_DRAFT_MODEL_CACHE)
    parser.add_argument("--docker-image", default="ubuntu:24.04")
    parser.add_argument("--chip", default="h100")
    parser.add_argument("--data-file", default="govreport/govreport_16K.jsonl")
    parser.add_argument("--context-target", type=int, default=16000)
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
    parser.add_argument("--cache-control", default="none", choices=["none", "all"])
    parser.add_argument("--replay-mode", default="kernel", choices=["kernel", "application"])
    parser.add_argument("--launch-count", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_inputs(args)
    available = query_metrics(args, args.output_dir)
    metrics = select_metrics(available)
    (args.output_dir / "selected_metrics.txt").write_text("\n".join(metrics) + "\n")

    rows = []
    for method in ("Auto", "DiffSpec"):
        rows.append(run_ncu(method, metrics, args, args.output_dir))

    derived = annotate(rows)
    output = json_safe({"config": vars(args), "selected_metrics": metrics, "rows": rows, "derived": derived})
    json_path = args.output_dir / "real_inference_ncu.json"
    csv_path = args.output_dir / "summary.csv"
    report_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    write_csv(csv_path, rows)
    write_report(report_path, rows, derived, args)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
