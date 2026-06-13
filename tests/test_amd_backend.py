import json
import subprocess
import sys
from pathlib import Path

import torch

from diffspec.amd import (
    AmdArenaConfig,
    AmdChunkArena,
    AmdResidencyConfig,
    apply_residency_hint,
    get_amd_gpu_info,
    get_hip_runtime_extension_status,
)
from diffspec.amd.extension import _hip_extension_compile_flags
from diffspec.core.config import DiffSpecConfig
from diffspec.core.diffspec_engine import DiffSpecEngine
from diffspec.core.kv_cache.chunk_arena import ChunkArena
from diffspec.amd.validate_rocm_backend import (
    evaluate_strict_requirements,
    validate_diffspec_working_cache_path,
)
from diffspec.amd.hip.run_hip_profiles import evaluate_run_requirements


def test_amd_gpu_info_is_json_friendly():
    info = get_amd_gpu_info()
    payload = info.to_dict()

    assert "available" in payload
    assert "is_rocm" in payload
    assert "torch_version" in payload


def test_residency_hint_on_cpu_is_structured_noop():
    tensor = torch.zeros(16)
    status = apply_residency_hint(tensor, AmdResidencyConfig(enable=True))

    assert status.requested is True
    assert status.applied is False
    assert status.reason == "tensor_is_not_on_gpu"
    assert status.requested_bytes == tensor.numel() * tensor.element_size()


def test_amd_chunk_arena_stages_cpu_kv_cache():
    config = AmdArenaConfig(
        max_chunks=4,
        chunk_size=2,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
    )
    arena = AmdChunkArena(config)
    full_kv = []
    for layer in range(2):
        key = torch.arange(1 * 2 * 8 * 4, dtype=torch.float32).view(1, 2, 8, 4) + layer
        value = key + 1000
        full_kv.append((key, value))

    view = arena.update([0, 2], full_kv)

    assert view.shape == (2, 2, 1, 2, 4, 4)
    assert arena.active_chunks == [0, 2]
    assert torch.equal(view[0, 0, :, :, :2, :], full_kv[0][0][:, :, :2, :])
    assert torch.equal(view[0, 1, :, :, :2, :], full_kv[0][1][:, :, :2, :])
    stats = arena.get_statistics()
    assert stats["backend"] == "amd_rocm"
    assert stats["active_chunks"] == 2
    assert stats["total_updates"] == 1
    assert stats["compact_backend"] == "torch_copy"
    assert stats["compact_status"]["reason"] == "hip_compact_not_applicable"


def test_amd_chunk_arena_stages_exact_partial_spans():
    config = AmdArenaConfig(
        max_chunks=4,
        chunk_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
    )
    arena = AmdChunkArena(config)
    key = torch.arange(1 * 2 * 18 * 4, dtype=torch.float32).view(1, 2, 18, 4)
    value = key + 1000
    spans = [(0, 0, 3), (2, 8, 10), (4, 16, 20)]

    view = arena.update(spans, [(key, value)])
    working_kv = arena.as_kv_list()
    expected_key = torch.cat(
        [key[:, :, 0:3, :], key[:, :, 8:10, :], key[:, :, 16:18, :]],
        dim=2,
    )
    expected_value = torch.cat(
        [value[:, :, 0:3, :], value[:, :, 8:10, :], value[:, :, 16:18, :]],
        dim=2,
    )

    assert view.shape == (1, 2, 1, 2, 7, 4)
    assert torch.equal(working_kv[0][0], expected_key)
    assert torch.equal(working_kv[0][1], expected_value)
    assert arena.get_statistics()["active_spans"] == [
        [0, 0, 3],
        [2, 8, 10],
        [4, 16, 20],
    ]


def test_amd_chunk_arena_eviction_reuses_capacity():
    config = AmdArenaConfig(
        max_chunks=1,
        chunk_size=2,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    arena = AmdChunkArena(config)
    key = torch.arange(1 * 1 * 8 * 2, dtype=torch.float32).view(1, 1, 8, 2)
    full_kv = [(key, key + 10)]

    arena.update([0], full_kv)
    arena.update([2], full_kv)

    assert arena.active_chunks == [2]
    assert arena.get_statistics()["active_chunks"] == 1


def test_diffspec_engine_keeps_standard_arena_on_non_rocm_runtime():
    config = DiffSpecConfig(
        enable_chunk_arena=True,
        top_k_chunks=2,
        max_arena_chunks=2,
        chunk_size=2,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        device="cpu",
    )
    engine = DiffSpecEngine(config)

    assert isinstance(engine.chunk_arena, ChunkArena)
    assert not isinstance(engine.chunk_arena, AmdChunkArena)


def test_amd_backend_exercises_diffspec_working_cache_path_on_cpu():
    result = validate_diffspec_working_cache_path(
        torch.device("cpu"),
        enable_residency=False,
    )

    assert result["passed"] is True
    assert result["backend"] == "amd_rocm"
    assert result["method"] == "DraftNetwork._select_working_cache_from_chunks"
    assert result["working_len"] == 7
    assert result["draft_stable_kv_len"] == 7
    assert result["evicted"] == 11
    assert result["arena_statistics"]["active_spans"] == [
        [0, 0, 3],
        [2, 8, 10],
        [4, 16, 18],
    ]
    assert result["arena_statistics"]["compact_backend"] == "torch_copy"


def test_hip_runtime_extension_status_is_structured_without_rocm():
    status = get_hip_runtime_extension_status().to_dict()

    assert "available" in status
    assert "attempted" in status
    assert "reason" in status


def test_hip_runtime_extension_uses_explicit_hip_compile_macro():
    host_flags, device_flags = _hip_extension_compile_flags()

    assert "-DDIFFSPEC_WITH_HIP=1" in host_flags
    assert "-DDIFFSPEC_WITH_HIP=1" in device_flags


def test_validate_rocm_backend_allow_no_rocm_is_structured():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "diffspec/amd/validate_rocm_backend.py",
            "--allow-no-rocm",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert "rocm_pytorch" in payload
    assert "device" in payload
    assert "hipcc" in payload
    assert "checks" in payload


def test_validator_strict_requirements_fail_when_apw_is_not_applied():
    output = {
        "hipcc": {"available": False},
        "checks": {
            "residency_hint": {"applied": False, "reason": "extension_unavailable"},
            "arena_staging": {
                "statistics": {"residency": {"applied": True, "reason": None}}
            },
            "diffspec_working_cache_path": {
                "arena_statistics": {
                    "residency": {"applied": False, "reason": "apw_not_applied"}
                }
            },
        },
    }

    result = evaluate_strict_requirements(
        output,
        require_hipcc=True,
        require_residency=True,
        require_hip_compact=True,
    )

    assert result["passed"] is False
    failed_checks = {failure["check"] for failure in result["failures"]}
    assert "hipcc" in failed_checks
    assert "residency_hint" in failed_checks
    assert "diffspec_working_cache_path.arena_statistics.residency" in failed_checks
    assert "diffspec_engine_path.arena_statistics.residency" in failed_checks
    assert "arena_staging.statistics.compact_status" in failed_checks


def test_validator_strict_requirements_include_engine_and_skip_secondary_when_requested():
    applied_residency = {"applied": True, "reason": None}
    applied_compact = {"applied": True, "reason": None}
    output = {
        "hipcc": {"available": True},
        "checks": {
            "residency_hint": applied_residency,
            "arena_staging": {
                "statistics": {
                    "residency": applied_residency,
                    "compact_status": applied_compact,
                }
            },
            "diffspec_working_cache_path": {
                "arena_statistics": {
                    "residency": applied_residency,
                    "compact_status": applied_compact,
                }
            },
            "diffspec_engine_path": {
                "arena_statistics": {
                    "residency": {"applied": False, "reason": "engine_apw_not_applied"},
                    "compact_status": applied_compact,
                }
            },
            "secondary_device_compact_path": {
                "passed": True,
                "skipped": True,
                "reason": "fewer_than_two_visible_gpus",
            },
        },
    }

    result = evaluate_strict_requirements(
        output,
        require_hipcc=True,
        require_residency=True,
        require_hip_compact=True,
    )

    assert result["passed"] is False
    assert result["failures"] == [
        {
            "check": "diffspec_engine_path.arena_statistics.residency",
            "reason": "engine_apw_not_applied",
        }
    ]


def test_hip_profile_requirements_fail_when_apw_modes_do_not_enable_apw():
    rows = [
        {"mode": "baseline_sparse", "elapsed_ms": 3.0},
        {"mode": "arena", "elapsed_ms": 2.0},
        {"mode": "arena_apw", "elapsed_ms": 1.5, "apw_enabled": False},
        {
            "mode": "arena_apw_tma",
            "elapsed_ms": 1.0,
            "apw_enabled": True,
            "tma_analogue_backend": "lds_tile_staging",
            "lds_tile_bytes": 16384,
            "lds_vector_width_bytes": 16,
            "amd_feature_path": "hip_access_policy_window+lds_tile_staging_b128",
        },
    ]

    result = evaluate_run_requirements(
        rows,
        profile_result=None,
        require_apw=True,
        require_lds_staging=True,
        require_rocprof_compute=False,
    )

    assert result["passed"] is False
    assert result["failures"] == [
        {"check": "arena_apw", "reason": "hip_apw_not_enabled"}
    ]


def test_hip_profile_requirements_fail_when_tma_mode_does_not_stage_lds():
    rows = [
        {"mode": "baseline_sparse", "elapsed_ms": 3.0},
        {"mode": "arena", "elapsed_ms": 2.0},
        {"mode": "arena_apw", "elapsed_ms": 1.5, "apw_enabled": True},
        {
            "mode": "arena_apw_tma",
            "elapsed_ms": 1.0,
            "apw_enabled": True,
            "tma_analogue_backend": "none",
            "lds_tile_bytes": 0,
            "lds_vector_width_bytes": 0,
            "amd_feature_path": "none",
        },
    ]

    result = evaluate_run_requirements(
        rows,
        profile_result=None,
        require_apw=True,
        require_lds_staging=True,
        require_rocprof_compute=False,
    )

    assert result["passed"] is False
    assert result["failures"] == [
        {"check": "arena_apw_tma", "reason": "lds_tile_staging_not_reported"}
    ]


def test_hip_profile_requirements_fail_when_lds_vector_path_is_missing():
    rows = [
        {"mode": "baseline_sparse", "elapsed_ms": 3.0},
        {"mode": "arena", "elapsed_ms": 2.0},
        {"mode": "arena_apw", "elapsed_ms": 1.5, "apw_enabled": True},
        {
            "mode": "arena_apw_tma",
            "elapsed_ms": 1.0,
            "apw_enabled": True,
            "tma_analogue_backend": "lds_tile_staging",
            "lds_tile_bytes": 16384,
            "lds_vector_width_bytes": 4,
            "amd_feature_path": "hip_access_policy_window+lds_tile_staging_b128",
        },
    ]

    result = evaluate_run_requirements(
        rows,
        profile_result=None,
        require_apw=True,
        require_lds_staging=True,
        require_rocprof_compute=False,
    )

    assert result["passed"] is False
    assert result["failures"] == [
        {"check": "arena_apw_tma", "reason": "vectorized_lds_staging_not_reported"}
    ]
