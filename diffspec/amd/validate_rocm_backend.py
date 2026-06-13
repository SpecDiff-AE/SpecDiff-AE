#!/usr/bin/env python3
"""Validate that the AMD ROCm backend is usable by DiffSpec."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if os.environ.get("MKL_THREADING_LAYER", "").upper() != "GNU":
    os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from diffspec.amd import (  # noqa: E402
    AmdArenaConfig,
    AmdChunkArena,
    AmdResidencyConfig,
    apply_residency_hint,
    get_amd_gpu_info,
    get_hip_runtime_extension_status,
    is_rocm_pytorch,
)
from diffspec.core.config import DiffSpecConfig  # noqa: E402
from diffspec.core.diffspec_engine import DiffSpecEngine  # noqa: E402
from diffspec.draft.draft_network import DraftNetwork  # noqa: E402


def check_hipcc() -> dict[str, Any]:
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        return {"available": False, "path": None, "version": None}
    proc = subprocess.run(
        [hipcc, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "available": proc.returncode == 0,
        "path": hipcc,
        "version": proc.stdout.splitlines()[:5],
    }


def validate_arena(device: torch.device, enable_residency: bool) -> dict[str, Any]:
    config = AmdArenaConfig(
        max_chunks=4,
        chunk_size=4,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        device=str(device),
        dtype=torch.float32,
        enable_residency_hint=enable_residency,
        residency=AmdResidencyConfig(enable=enable_residency, hit_ratio=0.85),
    )
    arena = AmdChunkArena(config)
    full_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(config.num_layers):
        key = (
            torch.arange(
                config.batch_size * config.num_heads * 24 * config.head_dim,
                dtype=torch.float32,
                device=device,
            )
            .view(config.batch_size, config.num_heads, 24, config.head_dim)
            + layer_idx * 10000
        )
        value = key + 1000
        full_kv.append((key, value))

    spans = [(0, 0, 4), (2, 8, 11), (4, 16, 20)]
    view = arena.update(spans, full_kv)
    working_kv = arena.as_kv_list()
    expected_k = torch.cat(
        [full_kv[0][0][:, :, start:end, :] for _, start, end in spans],
        dim=2,
    )
    expected_v = torch.cat(
        [full_kv[0][1][:, :, start:end, :] for _, start, end in spans],
        dim=2,
    )
    torch.testing.assert_close(working_kv[0][0], expected_k)
    torch.testing.assert_close(working_kv[0][1], expected_v)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "passed": True,
        "view_shape": list(view.shape),
        "working_len": int(view.size(4)),
        "statistics": arena.get_statistics(),
    }


class _DraftNetworkProbe:
    """Minimal object for exercising DraftNetwork's working-cache path."""

    pass


def _make_probe(device: torch.device, enable_residency: bool) -> tuple[_DraftNetworkProbe, list[tuple[torch.Tensor, torch.Tensor]]]:
    config = AmdArenaConfig(
        max_chunks=4,
        chunk_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        device=str(device),
        dtype=torch.float32,
        enable_residency_hint=enable_residency,
        residency=AmdResidencyConfig(enable=enable_residency, hit_ratio=0.85),
    )
    key = torch.arange(
        config.batch_size * config.num_heads * 18 * config.head_dim,
        dtype=torch.float32,
        device=device,
    ).view(config.batch_size, config.num_heads, 18, config.head_dim)
    value = key + 1000

    probe = _DraftNetworkProbe()
    probe.full_draft_kv = [(key, value)]
    probe.total_seq_len = 18
    probe.retrieve_top_k = 3
    probe.retrieval_chunk_size = 4
    probe.chunks = [(0, 0, 3), (1, 3, 8), (2, 8, 10), (3, 10, 16), (4, 16, 18)]
    probe.selected_chunks = [(0, 0, 3), (2, 8, 10), (4, 16, 18)]
    probe.prev_selected_chunk_ids = []
    probe.attn_scores_final = None
    probe.attn_scores = None
    probe.retrieval_condition = False
    probe.gamma_sem = 0.0
    probe._chunks_starts = None
    probe._chunks_ends = None
    probe._diff_buf = None
    probe._ones_buf = None
    probe._minus_ones_buf = None
    probe.device = device
    probe.chunk_arena = AmdChunkArena(config)
    probe.past_key_position_ids = torch.arange(18, device=device).unsqueeze(0)
    probe.retrieval_verbose = False
    return probe, [(key, value)]


def validate_diffspec_working_cache_path(
    device: torch.device,
    enable_residency: bool,
) -> dict[str, Any]:
    """Exercise the actual DraftNetwork chunk-selection integration path."""

    probe, full_kv = _make_probe(device, enable_residency)
    working_kv = DraftNetwork._select_working_cache_from_chunks(
        probe,
        top_k_chunks=3,
        do_retrieval=False,
        is_updated_chunks=False,
    )
    spans = probe.selected_chunks
    expected_k = torch.cat(
        [full_kv[0][0][:, :, start:end, :] for _, start, end in spans],
        dim=2,
    )
    expected_v = torch.cat(
        [full_kv[0][1][:, :, start:end, :] for _, start, end in spans],
        dim=2,
    )
    torch.testing.assert_close(working_kv[0][0], expected_k)
    torch.testing.assert_close(working_kv[0][1], expected_v)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "passed": True,
        "method": "DraftNetwork._select_working_cache_from_chunks",
        "backend": getattr(probe.chunk_arena, "backend_name", None),
        "working_len": int(working_kv[0][0].size(2)),
        "draft_stable_kv_len": int(probe.draft_stable_kv[0][0].size(2)),
        "evicted": int(probe.evicted),
        "arena_statistics": probe.chunk_arena.get_statistics(),
    }


def validate_diffspec_engine_path(
    device: torch.device,
    enable_residency: bool,
) -> dict[str, Any]:
    """Exercise DiffSpecEngine's ROCm component selection and arena update."""

    config_device = "cuda" if device.type == "cuda" else str(device)
    config = DiffSpecConfig(
        enable_hazard_profile=False,
        enable_chunk_arena=True,
        enable_apw_residency=enable_residency,
        top_k_chunks=2,
        max_arena_chunks=2,
        chunk_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        device=config_device,
    )
    engine = DiffSpecEngine(config)
    if not isinstance(engine.chunk_arena, AmdChunkArena):
        return {
            "passed": False,
            "backend": type(engine.chunk_arena).__name__,
            "reason": "engine_did_not_select_amd_chunk_arena",
        }

    key = torch.arange(
        1 * config.num_heads * 16 * config.head_dim,
        dtype=torch.float16,
        device=device,
    ).view(1, config.num_heads, 16, config.head_dim)
    value = key + 1000
    spans = [(0, 0, 4), (2, 8, 12)]
    view = engine.chunk_arena.update(spans, [(key, value)])
    working_kv = engine.chunk_arena.as_kv_list()
    expected_k = torch.cat(
        [key[:, :, start:end, :] for _, start, end in spans],
        dim=2,
    )
    expected_v = torch.cat(
        [value[:, :, start:end, :] for _, start, end in spans],
        dim=2,
    )
    torch.testing.assert_close(working_kv[0][0], expected_k)
    torch.testing.assert_close(working_kv[0][1], expected_v)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    stats = engine.chunk_arena.get_statistics()
    return {
        "passed": True,
        "method": "DiffSpecEngine._init_components+AmdChunkArena.update",
        "backend": getattr(engine.chunk_arena, "backend_name", None),
        "view_shape": list(view.shape),
        "working_len": int(working_kv[0][0].size(2)),
        "arena_statistics": stats,
    }


def validate_secondary_device_compact_path(enable_residency: bool) -> dict[str, Any]:
    """Exercise HIP compact on a non-current GPU when one is visible."""

    if torch.cuda.device_count() < 2:
        return {
            "passed": True,
            "skipped": True,
            "reason": "fewer_than_two_visible_gpus",
        }

    device = torch.device("cuda:1")
    config = AmdArenaConfig(
        max_chunks=2,
        chunk_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        device=str(device),
        dtype=torch.float16,
        enable_residency_hint=enable_residency,
        residency=AmdResidencyConfig(enable=enable_residency, hit_ratio=0.85),
    )
    arena = AmdChunkArena(config)
    key = torch.arange(
        config.batch_size * config.num_heads * 16 * config.head_dim,
        dtype=torch.float16,
        device=device,
    ).view(config.batch_size, config.num_heads, 16, config.head_dim)
    value = key + 1000
    spans = [(0, 0, 4), (2, 8, 12)]
    view = arena.update(spans, [(key, value)])
    working_kv = arena.as_kv_list()
    expected_k = torch.cat(
        [key[:, :, start:end, :] for _, start, end in spans],
        dim=2,
    )
    expected_v = torch.cat(
        [value[:, :, start:end, :] for _, start, end in spans],
        dim=2,
    )
    torch.testing.assert_close(working_kv[0][0], expected_k)
    torch.testing.assert_close(working_kv[0][1], expected_v)
    torch.cuda.synchronize(device)

    return {
        "passed": True,
        "device": str(device),
        "view_shape": list(view.shape),
        "working_len": int(working_kv[0][0].size(2)),
        "arena_statistics": arena.get_statistics(),
    }


def evaluate_strict_requirements(
    output: dict[str, Any],
    *,
    require_hipcc: bool,
    require_residency: bool,
    require_hip_compact: bool,
) -> dict[str, Any]:
    """Return reviewer-facing pass/fail gates for AMD hardware evidence."""

    failures: list[dict[str, str]] = []
    if require_hipcc and not output.get("hipcc", {}).get("available", False):
        failures.append(
            {
                "check": "hipcc",
                "reason": "hipcc_not_available",
            }
        )

    if require_residency:
        status_paths = [
            ("residency_hint", output["checks"].get("residency_hint")),
            (
                "arena_staging.statistics.residency",
                output["checks"]
                .get("arena_staging", {})
                .get("statistics", {})
                .get("residency"),
            ),
            (
                "diffspec_working_cache_path.arena_statistics.residency",
                output["checks"]
                .get("diffspec_working_cache_path", {})
                .get("arena_statistics", {})
                .get("residency"),
            ),
            (
                "diffspec_engine_path.arena_statistics.residency",
                output["checks"]
                .get("diffspec_engine_path", {})
                .get("arena_statistics", {})
                .get("residency"),
            ),
        ]
        secondary_device = output["checks"].get("secondary_device_compact_path", {})
        if not secondary_device.get("skipped", False):
            status_paths.append(
                (
                    "secondary_device_compact_path.arena_statistics.residency",
                    secondary_device
                    .get("arena_statistics", {})
                    .get("residency"),
                )
            )
        for name, status in status_paths:
            if not status or not status.get("applied", False):
                failures.append(
                    {
                        "check": name,
                        "reason": (
                            "missing_status"
                            if not status
                            else str(status.get("reason") or "apw_not_applied")
                        ),
                    }
                )

    if require_hip_compact:
        compact_paths = [
            (
                "arena_staging.statistics.compact_status",
                output["checks"]
                .get("arena_staging", {})
                .get("statistics", {})
                .get("compact_status"),
            ),
            (
                "diffspec_working_cache_path.arena_statistics.compact_status",
                output["checks"]
                .get("diffspec_working_cache_path", {})
                .get("arena_statistics", {})
                .get("compact_status"),
            ),
            (
                "diffspec_engine_path.arena_statistics.compact_status",
                output["checks"]
                .get("diffspec_engine_path", {})
                .get("arena_statistics", {})
                .get("compact_status"),
            ),
        ]
        secondary_device = output["checks"].get("secondary_device_compact_path", {})
        if not secondary_device.get("skipped", False):
            compact_paths.append(
                (
                    "secondary_device_compact_path.arena_statistics.compact_status",
                    secondary_device
                    .get("arena_statistics", {})
                    .get("compact_status"),
                )
            )
        for name, status in compact_paths:
            if not status or not status.get("applied", False):
                failures.append(
                    {
                        "check": name,
                        "reason": (
                            "missing_status"
                            if not status
                            else str(status.get("reason") or "hip_compact_not_applied")
                        ),
                    }
                )

    return {
        "passed": len(failures) == 0,
        "required": {
            "hipcc": bool(require_hipcc),
            "apw_residency_applied": bool(require_residency),
            "hip_compact_kernel_applied": bool(require_hip_compact),
        },
        "failures": failures,
    }


def write_output(output: dict[str, Any], output_path: Path | None) -> None:
    text = json.dumps(output, indent=2)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-no-rocm", action="store_true")
    parser.add_argument("--enable-residency", action="store_true")
    parser.add_argument(
        "--require-residency",
        action="store_true",
        help="Fail if HIP access-policy-window residency is not actually applied.",
    )
    parser.add_argument(
        "--require-hipcc",
        action="store_true",
        help="Fail if hipcc is not available on the AMD validation host.",
    )
    parser.add_argument(
        "--require-hip-compact",
        action="store_true",
        help="Fail if AmdChunkArena does not use the HIP compact/staging kernel.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reviewer mode: require hipcc and real HIP APW application.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    enable_residency = args.enable_residency or args.require_residency or args.strict
    require_residency = args.require_residency or args.strict
    require_hipcc = args.require_hipcc or args.strict
    require_hip_compact = args.require_hip_compact or args.strict

    info = get_amd_gpu_info()
    output: dict[str, Any] = {
        "rocm_pytorch": is_rocm_pytorch(),
        "device": info.to_dict(),
        "hipcc": check_hipcc(),
        "hip_runtime_extension": get_hip_runtime_extension_status().to_dict(),
        "checks": {},
    }

    if not is_rocm_pytorch() or not torch.cuda.is_available():
        output["checks"]["rocm_gpu_required"] = {
            "passed": False,
            "reason": info.reason or "rocm_gpu_unavailable",
        }
        write_output(output, args.output)
        return 0 if args.allow_no_rocm else 2

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    tensor = torch.empty(1024, device=device)
    output["checks"]["residency_hint"] = apply_residency_hint(
        tensor,
        AmdResidencyConfig(enable=enable_residency, hit_ratio=0.85),
    ).to_dict()
    output["checks"]["arena_staging"] = validate_arena(device, enable_residency)
    output["checks"]["diffspec_working_cache_path"] = validate_diffspec_working_cache_path(
        device,
        enable_residency,
    )
    output["checks"]["diffspec_engine_path"] = validate_diffspec_engine_path(
        device,
        enable_residency,
    )
    output["checks"]["secondary_device_compact_path"] = validate_secondary_device_compact_path(
        enable_residency,
    )
    output["checks"]["strict_requirements"] = evaluate_strict_requirements(
        output,
        require_hipcc=require_hipcc,
        require_residency=require_residency,
        require_hip_compact=require_hip_compact,
    )
    output["hip_runtime_extension"] = get_hip_runtime_extension_status().to_dict()

    write_output(output, args.output)
    return 0 if output["checks"]["strict_requirements"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
