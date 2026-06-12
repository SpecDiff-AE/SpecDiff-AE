#!/usr/bin/env python3
"""Microbenchmarks for DiffSpec flash/prefix/tree attention kernels."""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from diffspec.runtime.attention_compat import (
    FLASH_ATTN_AVAILABLE,
    PYTORCH_FLASH_ATTN_AVAILABLE,
    flash_attn_func,
    sdpa_backend_name,
)
from diffspec.runtime.tree_attention import (
    TRITON_AVAILABLE,
    prefix_attention_forward,
    tree_attention_forward,
    tree_attention_forward_naive,
)


def cuda_time(fn, warmup: int, repeat: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "repeat": repeat,
    }


def make_tree_mask(batch: int, q_len: int, device: torch.device) -> torch.Tensor:
    keep = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=device))
    mask = torch.full((batch, 1, q_len, q_len), torch.finfo(torch.float32).min, device=device)
    mask[:, :, keep] = 0.0
    return mask


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-5)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def bench_prefix(args, device: torch.device) -> dict:
    q = torch.randn(args.batch, args.q_len, args.heads, args.head_dim, device=device, dtype=torch.float16)
    k = torch.randn(args.batch, args.prefix_len, args.heads, args.head_dim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)

    def run_diffspec_prefix():
        return prefix_attention_forward(q, k, v, args.batch, args.q_len, args.heads, args.head_dim)

    def run_flash_func():
        return flash_attn_func(q, k, v, causal=False)

    prefix_o, prefix_lse = run_diffspec_prefix()
    flash_o = run_flash_func()
    return {
        "shape": {
            "batch": args.batch,
            "q_len": args.q_len,
            "prefix_len": args.prefix_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
        },
        "diffspec_prefix_ms": cuda_time(run_diffspec_prefix, args.warmup, args.repeat),
        "flash_func_ms": cuda_time(run_flash_func, args.warmup, args.repeat),
        "output_error_vs_flash_func": compare_tensors(prefix_o, flash_o),
        "lse_shape": list(prefix_lse.shape),
    }


def bench_tree(args, device: torch.device) -> dict:
    q = torch.randn(args.batch, args.q_len, args.heads, args.head_dim, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    prefix_lse = torch.randn(args.batch, args.heads, args.q_len, device=device, dtype=torch.float32)
    mask = make_tree_mask(args.batch, args.q_len, device)

    def run_triton_tree():
        return tree_attention_forward(
            q,
            k,
            v,
            mask,
            args.prefix_len,
            prefix_lse,
            args.batch,
            args.q_len,
            args.heads,
            args.head_dim,
        )

    def run_torch_tree():
        return tree_attention_forward_naive(
            q,
            k,
            v,
            mask,
            args.prefix_len,
            prefix_lse,
            args.batch,
            args.q_len,
            args.heads,
            args.head_dim,
        )

    triton_o, triton_w = run_triton_tree()
    torch_o, torch_w = run_torch_tree()
    return {
        "shape": {
            "batch": args.batch,
            "q_len": args.q_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
        },
        "triton_tree_ms": cuda_time(run_triton_tree, args.warmup, args.repeat),
        "torch_tree_ms": cuda_time(run_torch_tree, args.warmup, args.repeat),
        "output_error_vs_torch": compare_tensors(triton_o, torch_o),
        "weight_error_vs_torch": compare_tensors(triton_w, torch_w),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile DiffSpec attention kernels")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q_len", type=int, default=48)
    parser.add_argument("--prefix_len", type=int, default=32000)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    result = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "external_flash_attn_available": bool(FLASH_ATTN_AVAILABLE),
            "torch_flash_sdp_available": bool(PYTORCH_FLASH_ATTN_AVAILABLE),
            "attention_backend": sdpa_backend_name(),
            "triton_available": bool(TRITON_AVAILABLE),
        },
        "prefix": bench_prefix(args, device),
        "tree": bench_tree(args, device),
    }

    output = Path(args.output or REPO_ROOT / "results" / "attention_kernel_profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
