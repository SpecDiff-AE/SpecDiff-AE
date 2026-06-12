#!/usr/bin/env python3
"""
Plot DiffSpec plugin ablation results.
"""

import argparse
import json
import os
from datetime import datetime


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_output_path(input_path: str, output_path: str) -> str:
    if output_path:
        return output_path
    base_dir = os.path.dirname(os.path.abspath(input_path))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"diffspec_ablation_plot_{ts}.png")


def plot_results(results: dict, output_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install via `pip install matplotlib`.") from exc

    experiments = results.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in results. Run compare_vs_ar.py with --ablation.")

    labels = [exp["name"] for exp in experiments]
    speedup = [exp["summary"]["speedup_vs_ar"] for exp in experiments]
    accept_len = [exp["summary"]["avg_accept_length"] for exp in experiments]
    tps = [exp["summary"]["avg_method_tokens_per_sec"] for exp in experiments]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(labels, speedup, color="#4C78A8")
    axes[0].set_title("Speedup vs AR")
    axes[0].set_ylabel("x")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(labels, accept_len, color="#F58518")
    axes[1].set_title("Avg Accept Length")
    axes[1].set_ylabel("tokens")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(labels, tps, color="#54A24B")
    axes[2].set_title("Tokens / Sec")
    axes[2].set_ylabel("tps")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot DiffSpec plugin ablation results")
    parser.add_argument("--input", type=str, required=True, help="Ablation JSON from compare_vs_ar.py")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    results = load_results(args.input)
    output_path = resolve_output_path(args.input, args.output)
    plot_results(results, output_path)
    print(f"[plot] Saved to {output_path}")


if __name__ == "__main__":
    main()
