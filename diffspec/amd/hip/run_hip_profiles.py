#!/usr/bin/env python3
"""Build and run AMD ROCm Chunk Arena residency/staging microbenchmarks."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


SRC = Path(__file__).with_name("kv_access_profile.hip")
DEFAULT_OUT = Path("results") / "amd_chunk_arena"
MODES = [
    ("baseline_sparse", "Sparse full-KV gather"),
    ("arena", "Contiguous Chunk Arena"),
    ("arena_apw", "Chunk Arena + HIP access-policy window"),
    ("arena_apw_tma", "Chunk Arena + HIP APW + LDS tile staging (TMA analogue)"),
]
PROFILER_TOOLS = ["rocprof-compute", "rocprofv3", "rocprof", "omniperf"]


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def compile_binary(args: argparse.Namespace, out_dir: Path) -> Path:
    hipcc = shutil.which(args.hipcc) or args.hipcc
    binary = out_dir / "kv_access_profile"
    cmd = [
        hipcc,
        "-O3",
        "--std=c++17",
        str(SRC),
        "-o",
        str(binary),
    ]
    proc = run(cmd)
    (out_dir / "compile.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"hipcc failed, see {out_dir / 'compile.log'}")
    return binary


def build_mode_cmd(binary: Path, mode: str, args: argparse.Namespace) -> list[str]:
    return [
        str(binary),
        "--mode",
        mode,
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
        "--kv-parts",
        str(args.kv_parts),
        "--passes",
        str(args.passes),
        "--warmup",
        str(args.warmup),
        "--timing-iters",
        str(args.timing_iters),
        "--block-threads",
        str(args.block_threads),
        "--lds-tile-bytes",
        str(args.lds_tile_bytes),
        "--apw-hit-ratio",
        str(args.apw_hit_ratio),
    ]


def run_mode(binary: Path, mode: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    cmd = build_mode_cmd(binary, mode, args)
    proc = run(cmd)
    (out_dir / f"{mode}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"mode": mode, "error": proc.stderr.strip() or proc.stdout.strip()}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"mode": mode, "error": f"failed to parse JSON: {exc}", "stdout": proc.stdout}


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def collect_profiler_tools() -> dict[str, dict[str, Any]]:
    tools: dict[str, dict[str, Any]] = {}
    for tool in PROFILER_TOOLS:
        path = shutil.which(tool)
        tools[tool] = {"available": path is not None, "path": path}
        if path is not None:
            version_arg = "-v" if tool == "rocprof-compute" else "--version"
            proc = run([path, version_arg])
            tools[tool]["version"] = (proc.stdout + proc.stderr).splitlines()[:5]
    return tools


def command_line(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def write_profiler_commands(binary: Path, args: argparse.Namespace, out_dir: Path) -> None:
    lines = [
        "# AMD ROCm Profiler Commands",
        "",
        "Run these on an AMD ROCm host after the HIP binary is built. The same",
        "four modes are profiled so the reviewer can compare sparse full-KV,",
        "contiguous arena, APW, and APW+LDS staging.",
        "",
        "## Native Metric Blocks",
        "",
        "- `-b 2`: GPU speed-of-light, including occupancy and cache bandwidth.",
        "- `-b 3`: memory chart across LDS/L1/L2/HBM.",
        "- `-b 4`: empirical hierarchical roofline.",
        "- `-b 17`: L2 cache, including hit rate, L2 bandwidth, L2-fabric bandwidth, and HBM traffic.",
        "",
        "## Collection",
        "",
    ]
    for mode, description in MODES:
        cmd = [
            "rocprof-compute",
            "profile",
            "--name",
            f"diffspec_amd_{mode}",
            "--no-roof",
            "--",
            *build_mode_cmd(binary, mode, args),
        ]
        lines.extend(
            [
                f"### {mode}",
                "",
                f"{description}.",
                "",
                "```bash",
                command_line(cmd),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Analysis",
            "",
            "Replace `<workload-dir>` with the directory emitted by `rocprof-compute profile`.",
            "",
            "```bash",
            "rocprof-compute analyze -p <workload-dir> -b 2 3 4 17 -n per_kernel",
            "```",
            "",
            "Reviewer-facing columns to extract from ROCm Compute Profiler:",
            "",
            "| Family | Native metric names to report | Why it matters |",
            "|---|---|---|",
            "| L2 locality | `L2 Cache Hit Rate`, `L2 Cache BW`, `L2-Fabric Read BW` | APW should increase L2 reuse and reduce fabric/HBM pressure. |",
            "| HBM pressure | `HBM Read Traffic`, `HBM Write and Atomic Traffic`, achieved HBM bandwidth | Arena staging should reduce physical memory traffic for the same logical KV reuse. |",
            "| LDS/TMA-like staging | `Theoretical LDS Bandwidth`, `LDS Bank Conflicts/Access` | The APW+LDS mode should shift repeated reads from HBM/L2 into LDS. |",
            "| Occupancy | `Wavefront Occupancy`, `Active CUs` | Confirms the staging path does not win by starving the GPU. |",
            "",
        ]
    )
    (out_dir / "profiler_commands.md").write_text("\n".join(lines))


def run_rocprof_compute(binary: Path, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    profiler = shutil.which("rocprof-compute")
    if profiler is None:
        return {"requested": True, "available": False, "reason": "rocprof-compute_not_found"}

    results: list[dict[str, Any]] = []
    for mode, _ in MODES:
        cmd = [
            profiler,
            "profile",
            "--name",
            f"diffspec_amd_{mode}",
            "--no-roof",
            "--",
            *build_mode_cmd(binary, mode, args),
        ]
        proc = run(cmd, cwd=out_dir)
        (out_dir / f"{mode}.rocprof_compute.log").write_text(proc.stdout + proc.stderr)
        results.append(
            {
                "mode": mode,
                "returncode": proc.returncode,
                "command": command_line(cmd),
                "log": str(out_dir / f"{mode}.rocprof_compute.log"),
            }
        )
    return {
        "requested": True,
        "available": True,
        "workloads_dir": str(out_dir / "workloads"),
        "modes": results,
    }


def evaluate_run_requirements(
    rows: list[dict[str, Any]],
    profile_result: dict[str, Any] | None,
    *,
    require_apw: bool,
    require_rocprof_compute: bool,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    rows_by_mode = {row.get("mode"): row for row in rows}

    if require_apw:
        for mode in ("arena_apw", "arena_apw_tma"):
            row = rows_by_mode.get(mode)
            if row is None:
                failures.append({"check": mode, "reason": "missing_result"})
            elif row.get("error"):
                failures.append({"check": mode, "reason": str(row.get("error"))})
            elif row.get("apw_enabled") is not True:
                failures.append({"check": mode, "reason": "hip_apw_not_enabled"})
        tma_row = rows_by_mode.get("arena_apw_tma")
        if tma_row is None:
            failures.append({"check": "arena_apw_tma", "reason": "missing_result"})
        elif tma_row.get("tma_analogue_backend") != "lds_tile_staging":
            failures.append(
                {
                    "check": "arena_apw_tma",
                    "reason": "lds_tile_staging_not_reported",
                }
            )

    if require_rocprof_compute:
        if profile_result is None:
            failures.append(
                {"check": "rocprof_compute", "reason": "profiler_not_requested"}
            )
        elif not profile_result.get("available", False):
            failures.append(
                {
                    "check": "rocprof_compute",
                    "reason": str(profile_result.get("reason") or "profiler_unavailable"),
                }
            )
        else:
            for item in profile_result.get("modes", []):
                if item.get("returncode") != 0:
                    failures.append(
                        {
                            "check": f"rocprof_compute:{item.get('mode')}",
                            "reason": f"returncode={item.get('returncode')}",
                        }
                    )

    return {
        "passed": len(failures) == 0,
        "required": {
            "hip_apw_enabled": bool(require_apw),
            "rocprof_compute_success": bool(require_rocprof_compute),
        },
        "failures": failures,
    }


def write_report(rows: list[dict[str, Any]], out_dir: Path, profiler_tools: dict[str, Any]) -> None:
    base = next((row for row in rows if row.get("mode") == "baseline_sparse"), None)
    base_ms = float(base.get("elapsed_ms", 0.0)) if base and not base.get("error") else 0.0
    lines = [
        "# AMD ROCm Chunk Arena Profile",
        "",
        "| Mode | Description | Event ms | Speedup vs sparse | Logical GB/s | APW | Error |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    descriptions = dict(MODES)
    for row in rows:
        elapsed = row.get("elapsed_ms")
        logical_bytes = row.get("logical_bytes")
        gbps = None
        if elapsed and logical_bytes:
            gbps = float(logical_bytes) / (float(elapsed) * 1.0e-3) / 1.0e9
        speedup = base_ms / float(elapsed) if base_ms and elapsed else None
        lines.append(
            f"| {row.get('mode')} | {descriptions.get(row.get('mode'), '')} | "
            f"{fmt(elapsed, 3)} | {fmt(speedup, 2)}x | {fmt(gbps, 2)} | "
            f"{fmt(row.get('apw_enabled'))} | {row.get('error', '')} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `arena_apw` uses HIP's access-policy-window stream attribute when the ROCm runtime exposes it.",
            "- `arena_apw_tma` is the AMD analogue of the CUDA APW+TMA experiment: contiguous arena data is staged into LDS tiles and reused there.",
            "- HIP does not expose CUDA cooperative-groups `memcpy_async`; this benchmark therefore reports an LDS staging path, not a hardware-identical Hopper TMA path.",
            "",
            "Profiler tools detected:",
            "",
            "| Tool | Available | Path |",
            "|---|---|---|",
        ]
    )
    for tool, info in profiler_tools.items():
        lines.append(f"| {tool} | {fmt(info.get('available'))} | {info.get('path') or ''} |")
    lines.extend(
        [
            "",
            "Use `profiler_commands.md` or rerun with `--run-rocprof-compute` on an AMD host to collect reviewer-facing L2/HBM/LDS/occupancy counters.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipcc", default="hipcc")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--full-tokens", type=int, default=131072)
    parser.add_argument("--active-chunks", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-parts", type=int, default=2)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--timing-iters", type=int, default=30)
    parser.add_argument("--block-threads", type=int, default=256)
    parser.add_argument("--lds-tile-bytes", type=int, default=16384)
    parser.add_argument("--apw-hit-ratio", type=float, default=0.85)
    parser.add_argument(
        "--run-rocprof-compute",
        action="store_true",
        help="Run ROCm Compute Profiler for each mode after event timing.",
    )
    parser.add_argument(
        "--require-apw",
        action="store_true",
        help="Fail if APW modes do not report hipStreamSetAttribute success.",
    )
    parser.add_argument(
        "--require-rocprof-compute",
        action="store_true",
        help="Fail if ROCm Compute Profiler is unavailable or any profile run fails.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reviewer mode: require APW; if rocprof-compute is requested, require it to pass.",
    )
    args = parser.parse_args()
    require_apw = args.require_apw or args.strict
    require_rocprof_compute = args.require_rocprof_compute or (
        args.strict and args.run_rocprof_compute
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    binary = compile_binary(args, args.out_dir)
    rows = [run_mode(binary, mode, args, args.out_dir) for mode, _ in MODES]
    (args.out_dir / "raw_results.json").write_text(json.dumps(rows, indent=2))
    profiler_tools = collect_profiler_tools()
    (args.out_dir / "profiler_tools.json").write_text(json.dumps(profiler_tools, indent=2))
    write_profiler_commands(binary, args, args.out_dir)
    profile_result = None
    if args.run_rocprof_compute:
        profile_result = run_rocprof_compute(binary, args, args.out_dir)
        (args.out_dir / "rocprof_compute.json").write_text(json.dumps(profile_result, indent=2))
    requirements = evaluate_run_requirements(
        rows,
        profile_result,
        require_apw=require_apw,
        require_rocprof_compute=require_rocprof_compute,
    )
    (args.out_dir / "strict_requirements.json").write_text(
        json.dumps(requirements, indent=2)
    )
    write_report(rows, args.out_dir, profiler_tools)
    print(f"Wrote {args.out_dir}")
    return 0 if requirements["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
