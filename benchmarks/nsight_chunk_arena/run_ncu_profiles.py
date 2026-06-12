#!/usr/bin/env python3
"""Run Nsight Compute profiles for DiffSpec KV access microbenchmarks."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "benchmarks" / "nsight_chunk_arena" / "kv_access_profile.cu"
DEFAULT_OUT = REPO_ROOT / "results" / "nsight_chunk_arena"

MODES = [
    ("baseline_sparse", "baseline sparse KV gather"),
    ("arena", "Chunk Arena without APW"),
    ("arena_apw", "Chunk Arena with APW"),
    ("arena_apw_tma", "Chunk Arena with APW+TMA"),
]

METRIC_CANDIDATES = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
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


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
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


def find_tool(name: str, preferred: str | None = None) -> str:
    if preferred:
        path = Path(preferred)
        if path.exists():
            return str(path)
    proc = subprocess.run(["bash", "-lc", f"command -v {name}"], text=True, stdout=subprocess.PIPE)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    raise SystemExit(f"Cannot find required tool: {name}")


def to_container_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "/work/" + str(resolved.relative_to(REPO_ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(resolved)


def docker_ncu_cmd(args: argparse.Namespace, inner: list[str]) -> list[str]:
    if not args.docker_ncu_root:
        raise SystemExit("--docker-ncu requires --docker-ncu-root")
    root = Path(args.docker_ncu_root).resolve()
    if not (root / "ncu").exists():
        raise SystemExit(f"Cannot find ncu under --docker-ncu-root: {root}")
    shell_cmd = " ".join(shlex.quote(part) for part in inner)
    return [
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
        "-v",
        f"{REPO_ROOT}:/work",
        "-v",
        f"{root}:/opt/ncu:ro",
        "-w",
        "/work",
        args.docker_image,
        "bash",
        "-lc",
        shell_cmd,
    ]


def compile_binary(args: argparse.Namespace, out_dir: Path) -> tuple[Path, bool, str]:
    binary = out_dir / "kv_access_profile"
    nvcc = find_tool("nvcc", args.nvcc)
    common = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-lineinfo",
        f"-arch={args.arch}",
        "--expt-relaxed-constexpr",
        "-o",
        str(binary),
        str(SRC),
    ]
    proc = run(common, REPO_ROOT, check=False)
    compile_log = proc.stdout + proc.stderr
    if proc.returncode == 0:
        return binary, True, compile_log

    fallback = common[:-3] + ["-DDIFFSPEC_DISABLE_TMA_ASYNC"] + common[-3:]
    proc2 = run(fallback, REPO_ROOT, check=False)
    compile_log += "\n--- fallback without TMA async ---\n" + proc2.stdout + proc2.stderr
    if proc2.returncode != 0:
        sys.stderr.write(compile_log)
        raise SystemExit(proc2.returncode)
    return binary, False, compile_log


def collect_tma_sass_evidence(binary: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Record whether the APW+TMA kernel contains Hopper bulk-copy SASS."""
    candidates: list[str] = []
    nvcc_path = Path(args.nvcc)
    if nvcc_path.exists():
        candidates.append(str(nvcc_path.with_name("cuobjdump")))
    try:
        candidates.append(find_tool("cuobjdump"))
    except SystemExit:
        pass

    cuobjdump = next((path for path in candidates if Path(path).exists()), None)
    if not cuobjdump:
        text = "cuobjdump not found; SASS evidence not collected.\n"
        (out_dir / "tma_sass_check.txt").write_text(text)
        return {"available": False, "bulk_copy_instruction": False, "lines": []}

    proc = run(
        [cuobjdump, "--dump-sass", "--print-code", str(binary)],
        REPO_ROOT,
        check=False,
    )
    if proc.returncode != 0:
        proc = run([cuobjdump, "--dump-sass", str(binary)], REPO_ROOT, check=False)
    text = proc.stdout + proc.stderr
    interesting: list[str] = []
    patterns = re.compile(
        r"diffspec_chunk_arena_apw_tma_kernel|UBLKCP|CP\.ASYNC|TMA|MBARRIER|SYNCS",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        if patterns.search(line):
            interesting.append(line.rstrip())
        if len(interesting) >= 120:
            break

    has_kernel = any("diffspec_chunk_arena_apw_tma_kernel" in line for line in interesting)
    has_bulk_copy = any("UBLKCP" in line.upper() or "CP.ASYNC" in line.upper() for line in interesting)
    out = [
        f"cuobjdump: {cuobjdump}",
        f"apw_tma_kernel_symbol: {'yes' if has_kernel else 'no'}",
        f"hopper_bulk_copy_instruction: {'yes' if has_bulk_copy else 'no'}",
        "",
        "Matched SASS lines:",
        *interesting,
    ]
    (out_dir / "tma_sass_check.txt").write_text("\n".join(out) + "\n")
    return {
        "available": proc.returncode == 0,
        "bulk_copy_instruction": has_bulk_copy,
        "lines": interesting,
    }


def query_metrics(args: argparse.Namespace, out_dir: Path) -> set[str]:
    if args.docker_ncu:
        proc = run(
            docker_ncu_cmd(
                args,
                ["/opt/ncu/ncu", "--query-metrics", "--query-metrics-mode", "all", "--chips", args.chip],
            ),
            REPO_ROOT,
            check=False,
        )
    else:
        ncu = find_tool("ncu", args.ncu)
        proc = run(
            [ncu, "--query-metrics", "--query-metrics-mode", "all", "--chips", args.chip],
            REPO_ROOT,
            check=False,
        )
    text = proc.stdout + proc.stderr
    metrics: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Chip "):
            continue
        first = stripped.split(None, 1)[0]
        if "__" in first or first.startswith("gpu__"):
            metrics.add(first)
    availability = [
        f"{metric}: {'yes' if metric in metrics else 'no'}"
        for metric in METRIC_CANDIDATES
    ]
    (out_dir / "available_metrics.txt").write_text("\n".join(availability) + "\n")
    return metrics


def select_metrics(available: set[str]) -> list[str]:
    selected = [m for m in METRIC_CANDIDATES if m in available]
    if "gpu__time_duration.sum" not in selected:
        selected.append("gpu__time_duration.sum")
    if "dram__bytes_read.sum" not in selected:
        selected.append("dram__bytes_read.sum")
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


def parse_ncu_csv(text: str) -> dict[str, float]:
    metrics: dict[str, list[float]] = {}
    long_header: list[str] | None = None
    wide_header: list[str] | None = None
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if "Metric Name" in row and "Metric Value" in row:
            long_header = row
            continue
        if long_header is not None and len(row) == len(long_header):
            rec = dict(zip(long_header, row))
            name = rec.get("Metric Name")
            value = parse_float(rec.get("Metric Value"))
            if name and value is not None:
                metrics.setdefault(name, []).append(value)
            continue
        if wide_header is None and any("__" in col for col in row):
            wide_header = row
            continue
        if wide_header is None or len(row) != len(wide_header):
            continue
        for name, value_text in zip(wide_header, row):
            if "__" not in name:
                continue
            value = parse_float(value_text)
            if value is not None:
                metrics.setdefault(name, []).append(value)
    return {name: sum(vals) / len(vals) for name, vals in metrics.items() if vals}


def parse_result_json(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if "RESULT_JSON" not in line:
            continue
        _, payload = line.split("RESULT_JSON", 1)
        return json.loads(payload.strip())
    return {}


def run_plain(binary: Path, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [str(binary), "--mode", mode] + benchmark_args(args)
    proc = run(cmd, REPO_ROOT, check=True)
    return parse_result_json(proc.stdout + proc.stderr)


def benchmark_args(args: argparse.Namespace) -> list[str]:
    return [
        "--full-tokens",
        str(args.full_tokens),
        "--active-chunks",
        str(args.active_chunks),
        "--chunk-size",
        str(args.chunk_size),
        "--heads",
        str(args.heads),
        "--head-dim",
        str(args.head_dim),
        "--passes",
        str(args.passes),
        "--warmup",
        str(args.warmup),
        "--timing-iters",
        str(args.timing_iters),
        "--tma-tile-bytes",
        str(args.tma_tile_bytes),
        "--apw-hit-ratio",
        str(args.apw_hit_ratio),
        "--seed",
        str(args.seed),
    ]


def run_ncu(binary: Path, mode: str, metrics: list[str], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    raw_path = out_dir / f"{mode}.ncu.csv"
    err_path = out_dir / f"{mode}.ncu.stderr.txt"
    ncu_exe = "/opt/ncu/ncu" if args.docker_ncu else find_tool("ncu", args.ncu)
    target_binary = to_container_path(binary) if args.docker_ncu else str(binary)
    inner = [
        ncu_exe,
        "--csv",
        "--page",
        "raw",
        "--profile-from-start",
        "off",
        "--cache-control",
        args.cache_control,
        "--replay-mode",
        "kernel",
        "--kernel-name-base",
        "function",
        "-k",
        "regex:diffspec_.*_kernel",
        "--launch-count",
        "1",
        "--metrics",
        ",".join(metrics),
        target_binary,
        "--mode",
        mode,
    ] + benchmark_args(args)
    cmd = docker_ncu_cmd(args, inner) if args.docker_ncu else inner
    proc = run(cmd, REPO_ROOT, check=False)
    raw_path.write_text(proc.stdout)
    err_path.write_text(proc.stderr)
    if proc.returncode != 0:
        error_lines = []
        for line in (proc.stdout + "\n" + proc.stderr).splitlines():
            if (
                "ERR_NVGPUCTRPERM" in line
                or "No kernels were profiled" in line
                or "Failed to prepare kernel" in line
                or "Failed to profile" in line
            ):
                error_lines.append(line.strip())
        return {
            "mode": mode,
            "ncu_error": True,
            "ncu_returncode": proc.returncode,
            "ncu_stderr": proc.stderr.strip(),
            "ncu_error_summary": "\n".join(error_lines),
            "program": parse_result_json(proc.stdout + proc.stderr),
            "metrics": {},
        }
    return {
        "mode": mode,
        "ncu_error": False,
        "program": parse_result_json(proc.stdout + proc.stderr),
        "metrics": parse_ncu_csv(proc.stdout),
    }


def pick(metrics: dict[str, float], names: list[str]) -> float | None:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


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


def summarize_one(raw: dict[str, Any], plain: dict[str, Any], label: str) -> dict[str, Any]:
    metrics = raw.get("metrics", {})
    program = raw.get("program") or plain or {}
    duration_ns = pick(metrics, ["gpu__time_duration.sum"])
    dram_read = pick(metrics, ["dram__bytes_read.sum"])
    dram_pct = pick(
        metrics,
        [
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    )
    sm_util = pick(metrics, ["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
    occupancy = pick(
        metrics,
        [
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            "sm__maximum_warps_per_active_cycle_pct",
        ],
    )
    l2_hit_sectors = pick(metrics, ["lts__t_sectors_lookup_hit.sum"])
    l2_total_sectors = pick(metrics, ["lts__t_sectors.sum"])
    l2_device_fill_sectors = pick(metrics, ["lts__d_sectors_fill_device.sum"])
    l1tex_bytes = pick(metrics, ["l1tex__t_bytes.sum"])
    l1tex_hit_bytes = pick(metrics, ["l1tex__t_bytes_lookup_hit.sum"])
    l1tex_miss_bytes = pick(metrics, ["l1tex__t_bytes_lookup_miss.sum"])
    l2_cache_size = pick(metrics, ["device__attribute_l2_cache_size"])
    max_persisting_l2 = pick(metrics, ["device__attribute_max_persisting_l2_cache_size"])

    l2_source = "unavailable"
    l2_hit = pick(metrics, ["lts__t_sector_hit_rate.pct"])
    if l2_hit is not None:
        l2_source = "lts__t_sector_hit_rate.pct"
    if l2_hit is None:
        hit = pick(metrics, ["lts__t_sectors_lookup_hit.sum"])
        total = pick(metrics, ["lts__t_sectors.sum"])
        if hit is not None and total and total > 0:
            l2_hit = 100.0 * hit / total
            l2_source = "lts__t_sectors_lookup_hit / lts__t_sectors"
    if l2_hit is None:
        total = pick(metrics, ["lts__t_sectors.sum"])
        fill = pick(metrics, ["lts__d_sectors_fill_device.sum"])
        if total and fill is not None and total > 0:
            l2_hit = max(0.0, min(100.0, 100.0 * (1.0 - fill / total)))
            l2_source = "1 - lts__d_sectors_fill_device / lts__t_sectors"
    if l2_hit is None:
        hit = pick(metrics, ["l1tex__t_bytes_lookup_hit.sum"])
        miss = pick(metrics, ["l1tex__t_bytes_lookup_miss.sum"])
        if hit is not None and miss is not None and hit + miss > 0:
            l2_hit = 100.0 * hit / (hit + miss)
            l2_source = "fallback L1TEX lookup hit rate"
    if l2_hit is not None:
        if l2_hit < 0.0 or l2_hit > 100.0:
            l2_source += " (bounded to 0-100%)"
        l2_hit = max(0.0, min(100.0, l2_hit))

    if dram_read is not None and duration_ns and duration_ns > 0:
        dram_bw = dram_read / (duration_ns / 1.0e9) / 1.0e9
    else:
        dram_bw = None

    mode = raw["mode"]
    tma_ms = plain.get("elapsed_ms") if mode == "arena_apw_tma" else None
    logical_read_bytes = program.get("logical_read_bytes")
    active_bytes = program.get("active_bytes")
    full_bytes = program.get("full_bytes")
    active_tokens = program.get("active_tokens")
    apw_window_bytes = program.get("apw_window_bytes") or 0
    active_set_pct_of_l2 = safe_div(100.0 * active_bytes, l2_cache_size)
    l1tex_hit_rate = None
    if l1tex_hit_bytes is not None and l1tex_miss_bytes is not None:
        l1tex_hit_rate = safe_div(100.0 * l1tex_hit_bytes, l1tex_hit_bytes + l1tex_miss_bytes)
    return {
        "mode": mode,
        "label": label,
        "ncu_error": bool(raw.get("ncu_error")),
        "elapsed_ms_event": plain.get("elapsed_ms") or program.get("elapsed_ms"),
        "elapsed_ms_ncu": duration_ns / 1.0e6 if duration_ns is not None else None,
        "logical_read_gbps_event": plain.get("logical_read_gbps") or program.get("logical_read_gbps"),
        "logical_read_mb": bytes_to_mib(logical_read_bytes),
        "active_arena_mb": bytes_to_mib(active_bytes),
        "full_kv_mb": bytes_to_mib(full_bytes),
        "working_set_compression_x": safe_div(full_bytes, active_bytes),
        "active_set_pct_of_full": safe_div(100.0 * active_bytes, full_bytes),
        "active_set_pct_of_l2": active_set_pct_of_l2,
        "l2_headroom_pct": 100.0 - active_set_pct_of_l2 if active_set_pct_of_l2 is not None else None,
        "active_set_pct_of_persisting_l2": safe_div(100.0 * active_bytes, max_persisting_l2),
        "apw_window_active_coverage_pct": safe_div(100.0 * apw_window_bytes, active_bytes),
        "l2_hit_rate_pct": l2_hit,
        "l2_hit_rate_source": l2_source,
        "l2_hit_sectors_m": l2_hit_sectors / 1.0e6 if l2_hit_sectors is not None else None,
        "l2_total_sectors_m": l2_total_sectors / 1.0e6 if l2_total_sectors is not None else None,
        "l2_device_fill_sectors_m": l2_device_fill_sectors / 1.0e6 if l2_device_fill_sectors is not None else None,
        "l2_device_fill_mb": bytes_to_mib(l2_device_fill_sectors * 32.0 if l2_device_fill_sectors is not None else None),
        "l2_reuse_per_hbm_fill": safe_div(l2_hit_sectors, l2_device_fill_sectors),
        "hbm_read_mb": bytes_to_mib(dram_read),
        "hbm_bytes_per_logical_byte": safe_div(dram_read, logical_read_bytes),
        "logical_bytes_per_hbm_byte": safe_div(logical_read_bytes, dram_read),
        "hbm_bytes_per_active_token": safe_div(dram_read, active_tokens),
        "achieved_memory_bw_gbps": dram_bw,
        "dram_throughput_pct": dram_pct,
        "l1tex_traffic_mb": bytes_to_mib(l1tex_bytes),
        "l1tex_hit_rate_pct": l1tex_hit_rate,
        "l1tex_miss_mb": bytes_to_mib(l1tex_miss_bytes),
        "kernel_occupancy_pct": occupancy,
        "sm_utilization_pct": sm_util,
        "tma_copy_time_ms": tma_ms,
        "apw_enabled": program.get("apw_enabled"),
        "apw_window_mb": (program.get("apw_window_bytes") or 0) / (1024.0 * 1024.0),
        "tma_async_compiled": program.get("tma_async_compiled"),
        "ncu_stderr": raw.get("ncu_stderr", ""),
        "ncu_error_summary": raw.get("ncu_error_summary", ""),
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def annotate_vs_baseline(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    baseline = rows[0]

    def ratio(base_key: str, row: dict[str, Any]) -> float | None:
        base = baseline.get(base_key)
        value = row.get(base_key)
        if base is None or value is None or value == 0:
            return None
        return float(base) / float(value)

    def reduction(base_key: str, row: dict[str, Any]) -> float | None:
        base = baseline.get(base_key)
        value = row.get(base_key)
        if base is None or value is None or base == 0:
            return None
        return 100.0 * (float(base) - float(value)) / float(base)

    def delta(base_key: str, row: dict[str, Any]) -> float | None:
        base = baseline.get(base_key)
        value = row.get(base_key)
        if base is None or value is None:
            return None
        return float(value) - float(base)

    for row in rows:
        row["event_speedup_vs_baseline"] = ratio("elapsed_ms_event", row)
        row["ncu_speedup_vs_baseline"] = ratio("elapsed_ms_ncu", row)
        row["hbm_read_reduction_vs_baseline_pct"] = reduction("hbm_read_mb", row)
        row["hbm_per_logical_reduction_vs_baseline_pct"] = reduction("hbm_bytes_per_logical_byte", row)
        row["logical_per_hbm_gain_vs_baseline"] = ratio("logical_bytes_per_hbm_byte", row)
        if baseline.get("logical_bytes_per_hbm_byte") and row.get("logical_bytes_per_hbm_byte"):
            row["logical_per_hbm_gain_vs_baseline"] = (
                float(row["logical_bytes_per_hbm_byte"]) / float(baseline["logical_bytes_per_hbm_byte"])
            )
        row["hbm_per_token_reduction_vs_baseline_pct"] = reduction("hbm_bytes_per_active_token", row)
        row["l2_device_fill_reduction_vs_baseline_pct"] = reduction("l2_device_fill_mb", row)
        row["l2_reuse_gain_vs_baseline"] = ratio("l2_reuse_per_hbm_fill", row)
        if baseline.get("l2_reuse_per_hbm_fill") and row.get("l2_reuse_per_hbm_fill"):
            row["l2_reuse_gain_vs_baseline"] = (
                float(row["l2_reuse_per_hbm_fill"]) / float(baseline["l2_reuse_per_hbm_fill"])
            )
        row["l1tex_traffic_reduction_vs_baseline_pct"] = reduction("l1tex_traffic_mb", row)
        row["l1tex_miss_reduction_vs_baseline_pct"] = reduction("l1tex_miss_mb", row)
        row["dram_pressure_reduction_vs_baseline_pp"] = (
            float(baseline["dram_throughput_pct"]) - float(row["dram_throughput_pct"])
            if baseline.get("dram_throughput_pct") is not None and row.get("dram_throughput_pct") is not None
            else None
        )
        row["l2_hit_delta_vs_baseline_pp"] = delta("l2_hit_rate_pct", row)
        row["occupancy_delta_vs_baseline_pp"] = delta("kernel_occupancy_pct", row)
        row["sm_util_delta_vs_baseline_pp"] = delta("sm_utilization_pct", row)


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "mode",
        "label",
        "elapsed_ms_event",
        "elapsed_ms_ncu",
        "logical_read_gbps_event",
        "logical_read_mb",
        "active_arena_mb",
        "full_kv_mb",
        "working_set_compression_x",
        "active_set_pct_of_full",
        "active_set_pct_of_l2",
        "l2_headroom_pct",
        "active_set_pct_of_persisting_l2",
        "apw_window_active_coverage_pct",
        "l2_hit_rate_pct",
        "l2_hit_sectors_m",
        "l2_total_sectors_m",
        "l2_device_fill_sectors_m",
        "l2_device_fill_mb",
        "l2_reuse_per_hbm_fill",
        "hbm_read_mb",
        "hbm_bytes_per_logical_byte",
        "logical_bytes_per_hbm_byte",
        "hbm_bytes_per_active_token",
        "achieved_memory_bw_gbps",
        "dram_throughput_pct",
        "l1tex_traffic_mb",
        "l1tex_hit_rate_pct",
        "l1tex_miss_mb",
        "kernel_occupancy_pct",
        "sm_utilization_pct",
        "tma_copy_time_ms",
        "apw_enabled",
        "apw_window_mb",
        "tma_async_compiled",
        "l2_hit_rate_source",
        "event_speedup_vs_baseline",
        "ncu_speedup_vs_baseline",
        "hbm_read_reduction_vs_baseline_pct",
        "hbm_per_logical_reduction_vs_baseline_pct",
        "logical_per_hbm_gain_vs_baseline",
        "hbm_per_token_reduction_vs_baseline_pct",
        "l2_device_fill_reduction_vs_baseline_pct",
        "l2_reuse_gain_vs_baseline",
        "l1tex_traffic_reduction_vs_baseline_pct",
        "l1tex_miss_reduction_vs_baseline_pct",
        "dram_pressure_reduction_vs_baseline_pp",
        "l2_hit_delta_vs_baseline_pp",
        "occupancy_delta_vs_baseline_pp",
        "sm_util_delta_vs_baseline_pp",
        "ncu_error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_report(
    rows: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
    metrics: list[str],
    compile_log: str,
    sass_evidence: dict[str, Any],
) -> None:
    table_header = (
        "| Config | Event time ms | NCU kernel ms | Logical read GB/s | L2 hit rate % | "
        "HBM read MB | Achieved BW GB/s | DRAM % peak | Occupancy % | SM util % | TMA copy ms | APW | NCU error |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    )
    table_rows = []
    for row in rows:
        table_rows.append(
            "| {label} | {event} | {ncu} | {gbps} | {l2} | {hbm} | {bw} | {dram} | {occ} | {sm} | {tma} | {apw} | {err} |".format(
                label=row["label"],
                event=fmt(row["elapsed_ms_event"], 3),
                ncu=fmt(row["elapsed_ms_ncu"], 3),
                gbps=fmt(row["logical_read_gbps_event"], 2),
                l2=fmt(row["l2_hit_rate_pct"], 2),
                hbm=fmt(row["hbm_read_mb"], 2),
                bw=fmt(row["achieved_memory_bw_gbps"], 2),
                dram=fmt(row["dram_throughput_pct"], 2),
                occ=fmt(row["kernel_occupancy_pct"], 2),
                sm=fmt(row["sm_utilization_pct"], 2),
                tma=fmt(row["tma_copy_time_ms"], 3),
                apw="yes" if row.get("apw_enabled") else "no",
                err="NCU error" if row.get("ncu_error") else "",
            )
        )

    advantage_header = (
        "| Config | Event speedup | NCU speedup | HBM read reduction % | "
        "L2 hit delta pp | Occupancy delta pp | SM util delta pp |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    advantage_rows = []
    for row in rows:
        advantage_rows.append(
            "| {label} | {event} | {ncu} | {hbm} | {l2} | {occ} | {sm} |".format(
                label=row["label"],
                event=fmt(row.get("event_speedup_vs_baseline"), 2),
                ncu=fmt(row.get("ncu_speedup_vs_baseline"), 2),
                hbm=fmt(row.get("hbm_read_reduction_vs_baseline_pct"), 2),
                l2=fmt(row.get("l2_hit_delta_vs_baseline_pp"), 2),
                occ=fmt(row.get("occupancy_delta_vs_baseline_pp"), 2),
                sm=fmt(row.get("sm_util_delta_vs_baseline_pp"), 2),
            )
        )

    efficiency_header = (
        "| Config | HBM/logical B | Logical/HBM reuse | HBM B/token | "
        "L2 fill MB | L2 reuse/fill | L1TEX traffic MB | L1TEX hit % | DRAM pressure % |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    efficiency_rows = []
    for row in rows:
        efficiency_rows.append(
            "| {label} | {hbm_per_logical} | {logical_per_hbm} | {hbm_per_token} | "
            "{l2_fill} | {l2_reuse} | {l1tex} | {l1hit} | {dram} |".format(
                label=row["label"],
                hbm_per_logical=fmt(row.get("hbm_bytes_per_logical_byte"), 3),
                logical_per_hbm=fmt(row.get("logical_bytes_per_hbm_byte"), 2),
                hbm_per_token=fmt(row.get("hbm_bytes_per_active_token"), 1),
                l2_fill=fmt(row.get("l2_device_fill_mb"), 2),
                l2_reuse=fmt(row.get("l2_reuse_per_hbm_fill"), 2),
                l1tex=fmt(row.get("l1tex_traffic_mb"), 2),
                l1hit=fmt(row.get("l1tex_hit_rate_pct"), 2),
                dram=fmt(row.get("dram_throughput_pct"), 2),
            )
        )

    extra_advantage_header = (
        "| Config | Logical/HBM gain | HBM/logical reduction % | HBM B/token reduction % | "
        "L2 fill reduction % | L2 reuse gain | L1TEX traffic reduction % | DRAM pressure reduction pp |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    extra_advantage_rows = []
    for row in rows:
        extra_advantage_rows.append(
            "| {label} | {logical_gain} | {hbm_logical_red} | {token_red} | "
            "{l2_fill_red} | {l2_reuse_gain} | {l1_red} | {dram_red} |".format(
                label=row["label"],
                logical_gain=fmt(row.get("logical_per_hbm_gain_vs_baseline"), 2),
                hbm_logical_red=fmt(row.get("hbm_per_logical_reduction_vs_baseline_pct"), 2),
                token_red=fmt(row.get("hbm_per_token_reduction_vs_baseline_pct"), 2),
                l2_fill_red=fmt(row.get("l2_device_fill_reduction_vs_baseline_pct"), 2),
                l2_reuse_gain=fmt(row.get("l2_reuse_gain_vs_baseline"), 2),
                l1_red=fmt(row.get("l1tex_traffic_reduction_vs_baseline_pct"), 2),
                dram_red=fmt(row.get("dram_pressure_reduction_vs_baseline_pp"), 2),
            )
        )

    working_set_header = (
        "| Config | Full KV MB | Active set MB | Active/full % | Full/active | "
        "Active/L2 % | L2 headroom % | APW window/active % |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    working_set_rows = []
    for row in rows:
        working_set_rows.append(
            "| {label} | {full} | {active} | {active_pct} | {compression} | "
            "{l2_fit} | {l2_headroom} | {apw_fit} |".format(
                label=row["label"],
                full=fmt(row.get("full_kv_mb"), 2),
                active=fmt(row.get("active_arena_mb"), 2),
                active_pct=fmt(row.get("active_set_pct_of_full"), 2),
                compression=fmt(row.get("working_set_compression_x"), 2),
                l2_fit=fmt(row.get("active_set_pct_of_l2"), 2),
                l2_headroom=fmt(row.get("l2_headroom_pct"), 2),
                apw_fit=fmt(row.get("apw_window_active_coverage_pct"), 2),
            )
        )

    by_mode = {row["mode"]: row for row in rows}
    direct_arena = by_mode.get("arena", {})
    apw_tma = by_mode.get("arena_apw_tma", {})
    headline = [
        "## Headline Evidence",
        "",
        (
            "- Direct Chunk Arena is the apples-to-apples sparse-gather replacement: "
            f"{fmt(direct_arena.get('ncu_speedup_vs_baseline'), 2)}x NCU kernel speedup, "
            f"{fmt(direct_arena.get('hbm_read_reduction_vs_baseline_pct'), 2)}% less HBM read, "
            f"{fmt(direct_arena.get('logical_per_hbm_gain_vs_baseline'), 2)}x more logical KV work per HBM byte, and "
            f"{fmt(direct_arena.get('occupancy_delta_vs_baseline_pp'), 2)} pp higher occupancy."
        ),
        (
            "- APW+TMA stages contiguous arena tiles with Hopper async bulk copy: "
            f"{fmt(apw_tma.get('ncu_speedup_vs_baseline'), 2)}x NCU kernel speedup, "
            f"{fmt(apw_tma.get('logical_per_hbm_gain_vs_baseline'), 2)}x more logical KV work per HBM byte, and "
            f"{fmt(apw_tma.get('l1tex_traffic_reduction_vs_baseline_pct'), 2)}% less L1TEX traffic."
        ),
        (
            "- The active decode working set is "
            f"{fmt(direct_arena.get('active_arena_mb'), 2)} MiB out of "
            f"{fmt(direct_arena.get('full_kv_mb'), 2)} MiB full KV "
            f"({fmt(direct_arena.get('working_set_compression_x'), 2)}x smaller), occupying only "
            f"{fmt(direct_arena.get('active_set_pct_of_l2'), 2)}% of L2."
        ),
        "",
    ]

    notes = [
        "# DiffSpec Chunk Arena Nsight Compute Profile",
        "",
        "This report profiles a standalone KV-access microbenchmark that isolates the paper idea of converting sparse logical KV access into a physically contiguous Chunk Arena.",
        "",
        "## Workload",
        "",
        f"- full_tokens: {args.full_tokens}",
        f"- active_chunks: {args.active_chunks}",
        f"- chunk_size: {args.chunk_size}",
        f"- active_tokens: {args.active_chunks * args.chunk_size}",
        f"- heads: {args.heads}",
        f"- head_dim: {args.head_dim}",
        f"- passes per target kernel: {args.passes}",
        f"- cache-control: {args.cache_control}",
        f"- Nsight runner: {'dockerized ' + str(args.docker_ncu_root) if args.docker_ncu else 'host ncu'}",
        "",
        "## Summary",
        "",
        table_header + "\n".join(table_rows),
        "",
        *headline,
        "## Advantage Vs Baseline",
        "",
        advantage_header + "\n".join(advantage_rows),
        "",
        "## Data-Movement Efficiency",
        "",
        efficiency_header + "\n".join(efficiency_rows),
        "",
        "## Extra Advantage Vs Baseline",
        "",
        extra_advantage_header + "\n".join(extra_advantage_rows),
        "",
        "## Working-Set Fit",
        "",
        working_set_header + "\n".join(working_set_rows),
        "",
        "## Metric Notes",
        "",
        "- Event time is measured by CUDA events outside Nsight Compute and is useful as a sanity check.",
        "- NCU kernel time, HBM read bytes, bandwidth, occupancy, and SM utilization are read from Nsight Compute CSV when counter access succeeds.",
        "- L2 hit rate uses the exact L2 hit-rate metric if available; otherwise the report records the derived source in `summary.csv`; display values are bounded to 0-100%.",
        "- `Logical/HBM reuse` is logical KV read bytes divided by measured HBM read bytes; higher means each byte fetched from HBM produces more useful decode KV work.",
        "- `L2 reuse/fill` is L2 hit sectors divided by device-fill sectors; higher means fewer HBM fills serve more L2 hits.",
        "- `DRAM pressure reduction` is baseline DRAM throughput percentage minus the current config's percentage; positive values mean lower HBM pressure for the same logical decode workload.",
        "- The `arena_apw_tma` path uses Hopper async bulk global-to-shared copy through `cuda::memcpy_async` to stage each arena tile once and reuse it from shared memory across passes; it reports the measured target-kernel time as TMA copy time.",
        "- TMA can reduce repeated L2/L1TEX traffic by moving reuse into shared memory. In that case, L2 hit rate percentage alone may fall even when kernel time and logical work per HBM byte improve, because fewer repeated L2 lookups are issued.",
        "- Current Python DiffSpec code has Chunk Arena staging but APW/TMA are not wired into the model path; this harness profiles the hardware mechanism separately.",
        "- In this microbenchmark, `Chunk Arena without APW` is the direct physically-contiguous replacement for sparse KV gather. The APW and async-copy variants isolate additional hardware mechanisms and can trade latency for L2 residency/occupancy.",
        "",
        "## APW+TMA SASS Check",
        "",
        f"- APW+TMA kernel symbol found: {'yes' if any('diffspec_chunk_arena_apw_tma_kernel' in line for line in sass_evidence.get('lines', [])) else 'no'}",
        f"- Hopper bulk-copy instruction found: {'yes' if sass_evidence.get('bulk_copy_instruction') else 'no'}",
        "- Full matched lines are in `tma_sass_check.txt`.",
        "",
    ]
    if any(row.get("ncu_error") for row in rows):
        nvidia_params = ""
        params_path = Path("/proc/driver/nvidia/params")
        if params_path.exists():
            for line in params_path.read_text(errors="ignore").splitlines():
                if "RmProfilingAdminOnly" in line:
                    nvidia_params = line.strip()
                    break
        unique_errors = sorted(
            {row.get("ncu_error_summary", "") for row in rows if row.get("ncu_error_summary")}
        )
        notes.extend(
            [
                "## Nsight Counter Status",
                "",
                "Nsight Compute launched and attached to the benchmark process, but this machine blocks non-admin access to NVIDIA hardware performance counters.",
                "",
                f"- driver parameter: `{nvidia_params or 'not readable'}`",
                "- consequence: L2 hit rate, HBM read bytes, achieved bandwidth, occupancy, and SM utilization are reported as `n/a` until an administrator enables counter access.",
                "- collected evidence: CUDA event timing, APW configuration status, generated Nsight error logs, and raw command lines.",
                "",
                "Relevant Nsight messages:",
                "",
                "```",
                "\n".join(unique_errors)[:4000],
                "```",
                "",
            ]
        )
    notes.extend(
        [
        "## Metrics Requested",
        "",
        "```",
        ",".join(metrics),
        "```",
        "",
        "## Instrumentation Changelog",
        "",
        "| File | Change type | What was added/modified | Line(s) |",
        "|------|-------------|-------------------------|---------|",
        "| `benchmarks/nsight_chunk_arena/kv_access_profile.cu` | created | CUDA KV sparse gather / Chunk Arena / APW / async bulk-copy microbenchmark | - |",
        "| `benchmarks/nsight_chunk_arena/run_ncu_profiles.py` | created | Build, run Nsight Compute, parse CSV, generate report | - |",
        "| `benchmarks/nsight_chunk_arena/README.md` | created | Usage notes for the profiling harness | - |",
        "",
        "## Compile Log",
        "",
        "```",
        compile_log.strip()[-4000:],
        "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(notes) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nvcc", default="/usr/local/cuda-12.8/bin/nvcc")
    parser.add_argument("--ncu", default=None)
    parser.add_argument("--arch", default="sm_90a")
    parser.add_argument("--chip", default="gh100")
    parser.add_argument("--full-tokens", type=int, default=131072)
    parser.add_argument("--active-chunks", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--timing-iters", type=int, default=30)
    parser.add_argument("--tma-tile-bytes", type=int, default=16384)
    parser.add_argument("--apw-hit-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--cache-control", choices=["none", "all"], default="none")
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument("--docker-ncu", action="store_true", help="Run Nsight Compute inside a privileged NVIDIA Docker container.")
    parser.add_argument(
        "--docker-ncu-root",
        default=os.environ.get("DIFFSPEC_NCU_ROOT", str(Path.home() / ".cache" / "ncu2025")),
        help="Host path to an Nsight Compute directory containing ncu.",
    )
    parser.add_argument("--docker-image", default="ubuntu:24.04")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    binary, tma_compiled, compile_log = compile_binary(args, out_dir)
    (out_dir / "compile.log").write_text(compile_log)
    sass_evidence = collect_tma_sass_evidence(binary, out_dir, args)

    ncu_metrics: list[str] = []
    if not args.skip_ncu:
        available = query_metrics(args, out_dir)
        ncu_metrics = select_metrics(available)
        (out_dir / "selected_metrics.txt").write_text("\n".join(ncu_metrics) + "\n")

    rows: list[dict[str, Any]] = []
    raw_json: dict[str, Any] = {
        "tma_compiled_initially": tma_compiled,
        "modes": {},
    }
    for mode, label in MODES:
        plain = run_plain(binary, mode, args)
        if args.skip_ncu:
            raw = {"mode": mode, "program": plain, "metrics": {}, "ncu_error": False}
        else:
            raw = run_ncu(binary, mode, ncu_metrics, args, out_dir)
        raw_json["modes"][mode] = {"plain": plain, "ncu": raw}
        rows.append(summarize_one(raw, plain, label))

    annotate_vs_baseline(rows)
    (out_dir / "raw_results.json").write_text(json.dumps(raw_json, indent=2, sort_keys=True))
    write_summary_csv(rows, out_dir / "summary.csv")
    write_report(rows, out_dir, args, ncu_metrics, compile_log, sass_evidence)
    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
