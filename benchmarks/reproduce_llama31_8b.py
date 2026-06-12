#!/usr/bin/env python3
"""One-click Llama-3.1-8B DiffSpec reproduction harness.

The harness has two phases:
1. Tune DiffSpec over an explicit branch-policy/config grid.
2. Re-run Auto vs. DiffSpec using the fastest tuned config per dataset/context.

"Optimal" in the generated report always means best within this declared grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import format_value as fmt, mean
from diffspec.defaults import DEFAULT_BASE_MODEL, DEFAULT_DRAFT_MODEL, require_existing_local_path


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in parse_csv_list(text)]


def parse_configs(text: str) -> list[dict[str, Any]]:
    configs = []
    for raw in parse_csv_list(text):
        try:
            nodes, threshold, depth = raw.split(":")
        except ValueError as exc:
            raise ValueError(f"Invalid config {raw!r}; expected nodes:threshold:max_depth") from exc
        configs.append({
            "name": f"n{int(nodes)}_t{threshold.replace('.', 'p')}_d{int(depth)}",
            "nodes": int(nodes),
            "threshold": float(threshold),
            "max_depth": int(depth),
            "raw": raw,
        })
    if not configs:
        raise ValueError("At least one candidate config is required")
    return configs


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(cmd: list[str], *, env: dict[str, str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[run] {shell_join(cmd)}")
    print(f"[log] {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shell_join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {shell_join(cmd)}")


def ensure_inputs(args: argparse.Namespace) -> None:
    missing = []
    for value in [args.base_model, args.draft_model]:
        error = require_existing_local_path(value)
        if error:
            missing.append(error)
    data_dir = Path(args.data_dir)
    for rel in args.data_files:
        if not (data_dir / rel).exists():
            missing.append(str(data_dir / rel))
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing required model/data paths:\n  - {joined}")


def make_env(args: argparse.Namespace, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    if extra:
        env.update(extra)
    return env


def run_tuning(args: argparse.Namespace, configs: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    tune_rows: list[dict[str, Any]] = []
    for config in configs:
        config_dir = out_dir / "tune" / config["name"]
        cmd = [
            sys.executable,
            "benchmarks/branch_policy_analysis.py",
            "--output-dir",
            str(config_dir),
            "--base-model",
            args.base_model,
            "--draft-model",
            args.draft_model,
            "--data-dir",
            args.data_dir,
            "--data-files",
            *args.data_files,
            "--context-targets",
            *[str(x) for x in args.context_targets],
            "--samples-per-context",
            str(args.samples_per_context),
            "--max-new-tokens",
            str(args.tune_max_new_tokens),
            "--policies",
            *args.policies,
            "--nodes",
            str(config["nodes"]),
            "--threshold",
            str(config["threshold"]),
            "--max-depth",
            str(config["max_depth"]),
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
            "--gpu-index",
            "0",
        ]
        if args.disable_hybrid_tree_attn:
            cmd.append("--disable-hybrid-tree-attn")
        summary_path = config_dir / "summary.csv"
        if args.reuse_existing and summary_path.exists():
            print(f"[reuse] {summary_path}")
        else:
            run_command(cmd, env=make_env(args), cwd=REPO_ROOT, log_path=config_dir / "run.log")
        with summary_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row.update({
                    "config_name": config["name"],
                    "nodes": config["nodes"],
                    "threshold": config["threshold"],
                    "max_depth": config["max_depth"],
                    "tune_dir": str(config_dir),
                })
                tune_rows.append(row)
    if not tune_rows:
        raise RuntimeError("No tuning rows were produced")
    return tune_rows


def select_best(tune_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in tune_rows:
        key = (row["dataset"], str(row["context_target"]))
        score = float(row["decode_tokens_per_sec"])
        if key not in best or score > float(best[key]["decode_tokens_per_sec"]):
            best[key] = row
    return best


def write_tuning_outputs(out_dir: Path, tune_rows: list[dict[str, Any]], best: dict[tuple[str, str], dict[str, Any]]) -> None:
    tune_csv = out_dir / "tuning_summary.csv"
    fieldnames = [
        "dataset",
        "context_target",
        "policy",
        "config_name",
        "nodes",
        "threshold",
        "max_depth",
        "decode_tokens_per_sec",
        "avg_accept_length",
        "accept_median",
        "final_top_depth",
        "final_top_prob",
        "entropy_norm",
        "tune_dir",
    ]
    with tune_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tune_rows)
    selected = {
        f"{dataset}::{context}": row
        for (dataset, context), row in sorted(best.items())
    }
    (out_dir / "selected_configs.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def eval_one(args: argparse.Namespace, out_dir: Path, selected: dict[str, Any]) -> Path:
    dataset = selected["dataset"]
    context = str(selected["context_target"])
    policy = selected["policy"]
    eval_id = (
        dataset.replace("/", "_").replace(".jsonl", "")
        + f"_ctx{context}_{policy}_{selected['config_name']}"
    )
    eval_dir = out_dir / "eval" / eval_id
    eval_json = eval_dir / "paired_inference_benchmark.json"
    cmd = [
        sys.executable,
        "benchmarks/paired_inference_benchmark.py",
        "--base_model",
        args.base_model,
        "--draft_model",
        args.draft_model,
        "--data_dir",
        args.data_dir,
        "--data_files",
        dataset,
        "--context_targets",
        context,
        "--samples_per_context",
        str(args.samples_per_context),
        "--repeat_runs",
        str(args.repeat_runs),
        "--max_new_tokens",
        str(args.eval_max_new_tokens),
        "--context_tolerance",
        str(args.context_tolerance),
        "--max_source_records",
        str(args.max_source_records),
        "--warmup_runs",
        str(args.warmup_runs),
        "--warmup_max_new_tokens",
        str(args.warmup_max_new_tokens),
        "--nodes",
        str(selected["nodes"]),
        "--threshold",
        str(selected["threshold"]),
        "--max_depth",
        str(selected["max_depth"]),
        "--chunk_size",
        str(args.chunk_size),
        "--top_k",
        str(args.top_k),
        "--retrieve_every_n",
        str(args.retrieve_every_n),
        "--retrieval_min_context",
        str(args.retrieval_min_context),
        "--plugin_mask",
        args.plugin_mask,
        "--min_speedup_vs_auto",
        str(args.min_speedup_vs_auto),
        "--speedup_target_metric",
        "decode",
        "--output_dir",
        str(eval_dir),
        "--output_json",
        str(eval_json),
    ]
    if args.disable_hybrid_tree_attn:
        cmd.append("--disable_hybrid_tree_attn")
    if args.reuse_existing and eval_json.exists():
        print(f"[reuse] {eval_json}")
    else:
        env = make_env(args, {"DIFFSPEC_BRANCH_POLICY": policy})
        run_command(cmd, env=env, cwd=REPO_ROOT, log_path=eval_dir / "run.log")
    return eval_json


def summarize_eval(eval_jsons: list[Path], best: dict[tuple[str, str], dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in eval_jsons:
        data = json.loads(path.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            key = (case["dataset"], str(case["context_target_tokens"]))
            selected = best.get(key)
            for pair in case.get("paired_runs", []):
                auto = pair["Auto"]
                diff = pair["DiffSpec"]
                auto_decode_tps = auto.get("decode_tokens_per_sec")
                diff_decode_tps = diff.get("decode_tokens_per_sec")
                decode_speedup = (
                    diff_decode_tps / auto_decode_tps
                    if auto_decode_tps and diff_decode_tps
                    else None
                )
                rows.append({
                    "dataset": case["dataset"],
                    "context_target": case["context_target_tokens"],
                    "input_tokens": case["input_token_count"],
                    "repeat": pair["repeat"],
                    "policy": selected["policy"] if selected else diff.get("runtime_policy", {}).get("branch_policy"),
                    "nodes": selected["nodes"] if selected else data["config"].get("nodes"),
                    "threshold": selected["threshold"] if selected else data["config"].get("threshold"),
                    "max_depth": selected["max_depth"] if selected else data["config"].get("max_depth"),
                    "auto_tps": auto.get("tokens_per_sec"),
                    "diffspec_tps": diff.get("tokens_per_sec"),
                    "speedup_vs_auto": pair.get("speedup_vs_auto"),
                    "auto_decode_tps": auto_decode_tps,
                    "diffspec_decode_tps": diff_decode_tps,
                    "decode_speedup_vs_auto": round(decode_speedup, 4) if decode_speedup is not None else None,
                    "avg_accept_length": diff.get("avg_accept_length"),
                    "auto_generated": auto.get("total_generated"),
                    "diffspec_generated": diff.get("total_generated"),
                    "eval_json": str(path),
                })
    write_eval_csv(out_dir / "final_summary.csv", rows)
    write_report(out_dir / "FINAL_REPORT.md", rows, best, args)
    return rows


def write_eval_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "context_target",
        "input_tokens",
        "repeat",
        "policy",
        "nodes",
        "threshold",
        "max_depth",
        "auto_tps",
        "diffspec_tps",
        "speedup_vs_auto",
        "auto_decode_tps",
        "diffspec_decode_tps",
        "decode_speedup_vs_auto",
        "avg_accept_length",
        "auto_generated",
        "diffspec_generated",
        "eval_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    best: dict[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], str(row["context_target"])), []).append(row)
    lines = [
        "# Llama-3.1-8B DiffSpec Reproduction Report",
        "",
        "This report is generated by `benchmarks/reproduce_llama31_8b.py`.",
        "`Best` means fastest DiffSpec configuration within the declared tuning grid.",
        "",
        "## Declared Search Space",
        "",
        f"- candidate configs: `{args.candidate_configs}`",
        f"- branch policies: `{', '.join(args.policies)}`",
        f"- tune tokens: {args.tune_max_new_tokens}",
        f"- final eval tokens: {args.eval_max_new_tokens}",
        f"- datasets: `{', '.join(args.data_files)}`",
        f"- context targets: `{', '.join(str(x) for x in args.context_targets)}`",
        f"- retrieval min context: {args.retrieval_min_context}",
        "",
        "## Selected Configs",
        "",
        "| Dataset | Context | Policy | Nodes | Threshold | Max depth | Tune tok/s | Tune avg accept |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (dataset, context), row in sorted(best.items()):
        lines.append(
            f"| {dataset} | {context} | {row['policy']} | {row['nodes']} | "
            f"{row['threshold']} | {row['max_depth']} | "
            f"{fmt(float(row['decode_tokens_per_sec']), 2)} | "
            f"{fmt(float(row['avg_accept_length']), 2)} |"
        )
    lines.extend([
        "",
        "## Final Decode-Only Auto vs DiffSpec",
        "",
        "| Dataset | Context | Policy | Auto decode tok/s | DiffSpec decode tok/s | Decode speedup | Avg accept |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for key, group in sorted(grouped.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {group[0]['policy']} | "
            f"{fmt(mean([r['auto_decode_tps'] for r in group]), 2)} | "
            f"{fmt(mean([r['diffspec_decode_tps'] for r in group]), 2)} | "
            f"{fmt(mean([r['decode_speedup_vs_auto'] for r in group]), 2)}x | "
            f"{fmt(mean([r['avg_accept_length'] for r in group]), 2)} |"
        )
    lines.extend([
        "",
        "## Final End-to-End Generation",
        "",
        "| Dataset | Context | Policy | Auto tok/s | DiffSpec tok/s | End-to-end speedup | Avg accept |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for key, group in sorted(grouped.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {group[0]['policy']} | "
            f"{fmt(mean([r['auto_tps'] for r in group]), 2)} | "
            f"{fmt(mean([r['diffspec_tps'] for r in group]), 2)} | "
            f"{fmt(mean([r['speedup_vs_auto'] for r in group]), 2)}x | "
            f"{fmt(mean([r['avg_accept_length'] for r in group]), 2)} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `tuning_summary.csv`: all tuning rows.",
        "- `selected_configs.json`: fastest tuned config per dataset/context.",
        "- `final_summary.csv`: final paired Auto-vs-DiffSpec rows.",
        "- `eval/*/paired_inference_benchmark.json`: raw paired benchmark outputs.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "smoke":
        if args.data_files_arg is None:
            args.data_files = ["govreport/govreport_16K.jsonl"]
        if args.context_targets_arg is None:
            args.context_targets = [50000]
        if args.candidate_configs_arg is None:
            args.candidate_configs = "64:0.08:8"
        if args.policies_arg is None:
            args.policies = ["online", "full"]
        args.tune_max_new_tokens = args.tune_max_new_tokens or 128
        args.eval_max_new_tokens = args.eval_max_new_tokens or 128
        args.samples_per_context = args.samples_per_context or 1
        args.repeat_runs = args.repeat_runs or 1
    elif args.preset == "full":
        if args.data_files_arg is None:
            args.data_files = ["govreport/govreport_16K.jsonl", "pg-19/pg19_16K.jsonl"]
        if args.context_targets_arg is None:
            args.context_targets = [50000, 128000]
        if args.candidate_configs_arg is None:
            args.candidate_configs = "64:0.08:8,96:0.08:12,128:0.04:16"
        if args.policies_arg is None:
            args.policies = ["online", "full", "uniform4", "shallow", "deep"]
        args.tune_max_new_tokens = args.tune_max_new_tokens or 512
        args.eval_max_new_tokens = args.eval_max_new_tokens or 4096
        args.samples_per_context = args.samples_per_context or 1
        args.repeat_runs = args.repeat_runs or 1
    else:
        args.tune_max_new_tokens = args.tune_max_new_tokens or 128
        args.eval_max_new_tokens = args.eval_max_new_tokens or 512
        args.samples_per_context = args.samples_per_context or 1
        args.repeat_runs = args.repeat_runs or 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "full", "custom"], default="smoke")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "diffspec" / "data"))
    parser.add_argument("--data-files", dest="data_files_arg", default=None)
    parser.add_argument("--context-targets", dest="context_targets_arg", default=None)
    parser.add_argument("--candidate-configs", dest="candidate_configs_arg", default=None)
    parser.add_argument("--policies", dest="policies_arg", default=None)
    parser.add_argument("--tune-max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--samples-per-context", type=int, default=None)
    parser.add_argument("--repeat-runs", type=int, default=None)
    parser.add_argument("--context-tolerance", type=float, default=0.08)
    parser.add_argument("--max-source-records", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-max-new-tokens", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--retrieve-every-n", type=int, default=4)
    parser.add_argument("--retrieval-min-context", type=int, default=48000)
    parser.add_argument("--plugin-mask", default="0,1,0,0,0,0")
    parser.add_argument("--disable-hybrid-tree-attn", action="store_true")
    parser.add_argument("--min-speedup-vs-auto", type=float, default=4.8)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    args.data_files = parse_csv_list(args.data_files_arg) if args.data_files_arg else []
    args.context_targets = parse_int_list(args.context_targets_arg) if args.context_targets_arg else []
    args.candidate_configs = args.candidate_configs_arg or ""
    args.policies = parse_csv_list(args.policies_arg) if args.policies_arg else []
    apply_preset(args)
    configs = parse_configs(args.candidate_configs)

    ensure_inputs(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or (REPO_ROOT / "results" / "llama31_8b_repro" / f"{args.preset}_{timestamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "args": vars(args),
        "configs": configs,
        "python": sys.executable,
    }, indent=2), encoding="utf-8")

    tune_rows = run_tuning(args, configs, out_dir)
    best = select_best(tune_rows)
    write_tuning_outputs(out_dir, tune_rows, best)

    eval_jsons = []
    for selected in best.values():
        eval_jsons.append(eval_one(args, out_dir, selected))
    final_rows = summarize_eval(eval_jsons, best, out_dir, args)

    print("\n[done]")
    print(f"Report: {out_dir / 'FINAL_REPORT.md'}")
    print(f"Final summary: {out_dir / 'final_summary.csv'}")
    print(f"Rows: {len(final_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
