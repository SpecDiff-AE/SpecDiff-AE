#!/usr/bin/env python3
"""Sweep DiffSpec tree parameters against the autoregressive baseline."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.common import load_texts, prepare_input_ids  # noqa: E402
from diffspec.draft.diffspec_model import DiffSpecModel  # noqa: E402
from diffspec.defaults import DEFAULT_BASE_MODEL, DEFAULT_DRAFT_MODEL  # noqa: E402


def resolve_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda:0"), torch.float16
    return torch.device("cpu"), torch.float32


def run_autoregressive(model, input_ids: torch.Tensor, max_new_tokens: int):
    return model.autoregressive_generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        output_result_line=False,
        verbose=False,
        use_flash_prefill=True,
    )


def parse_configs(configs: str):
    out = []
    for item in configs.split(","):
        item = item.strip()
        if not item:
            continue
        nodes, threshold, depth = item.split(":")
        out.append((int(nodes), float(threshold), int(depth)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--draft_model", type=str, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(REPO_ROOT, "diffspec", "data", "govreport", "govreport_16K.jsonl"))
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--retrieval_min_context", type=int, default=48000)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument(
        "--configs",
        type=str,
        default="128:0.02:20,160:0.02:20,192:0.02:20,224:0.02:20,256:0.02:20,320:0.02:20,"
                "192:0.01:24,224:0.01:24,256:0.01:24,192:0.04:16,224:0.04:16",
    )
    args = parser.parse_args()

    device, dtype = resolve_device()
    model = DiffSpecModel.from_pretrained(
        base_model_path=args.base_model,
        draft_model_path=args.draft_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
    ).eval()
    text = load_texts(Path(args.data_path), 1)[0]
    input_ids = prepare_input_ids(model.tokenizer, text, device)
    ar = run_autoregressive(model, input_ids, args.max_new_tokens)
    ar_tps = ar["tokens_per_sec"]
    print({"baseline": "autoregressive", "tokens_per_sec": ar_tps, "input_len": int(input_ids.shape[1])})

    hybrid_flag = 1 if args.hybrid else 0
    for nodes, threshold, depth in parse_configs(args.configs):
        torch.cuda.empty_cache()
        result = model.diffspec_generate(
            input_ids,
            temperature=0,
            max_new_tokens=args.max_new_tokens,
            nodes=nodes,
            threshold=threshold,
            max_depth=depth,
            output_result_line=False,
            verbose=False,
            use_diffspec=[1, 1, hybrid_flag, 1, 1],
            diffspec_plugins=[0, 0, 0, 0, 0, 0],
            retrieval_chunk_size=32,
            retrieve_top_k=32,
            retrieve_every_n_steps=4,
            retrieval_min_context=args.retrieval_min_context,
        )
        print({
            "nodes": nodes,
            "threshold": threshold,
            "max_depth": depth,
            "hybrid": bool(hybrid_flag),
            "tokens_per_sec": result["tokens_per_sec"],
            "speedup_vs_ar": round(result["tokens_per_sec"] / ar_tps, 3) if ar_tps else 0.0,
            "avg_accept_length": result.get("avg_accept_length"),
            "accept_length_list": result.get("accept_length_list"),
        })


if __name__ == "__main__":
    main()
