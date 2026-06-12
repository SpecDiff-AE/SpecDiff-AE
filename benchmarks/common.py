"""Shared utilities for DiffSpec benchmark scripts."""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


SYSTEM_PROMPT = "You are a helpful assistant."


def configure_runtime(*, profile: bool = False, hazard_trace: bool = False) -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if profile:
        os.environ.setdefault("DIFFSPEC_PROFILE", "1")
    if hazard_trace:
        os.environ["DIFFSPEC_HAZARD_TRACE"] = "1"


def add_repo_root_to_path(file: str, *, parents_up: int = 1) -> Path:
    root = Path(file).resolve().parents[parents_up]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def parse_plugin_mask(mask: str) -> list[int]:
    parts = [part.strip() for part in mask.split(",") if part.strip()]
    if len(parts) != 6:
        raise ValueError("--plugin-mask must contain six comma-separated flags")
    return [1 if int(part) else 0 for part in parts]


def load_text_records(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if max_records is not None and len(records) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("input")
            if text:
                records.append({"source_index": line_no, "text": text})
    return records


def load_texts(path: Path, max_records: int | None = None) -> list[str]:
    return [record["text"] for record in load_text_records(path, max_records)]


def build_chat_prompt(tokenizer, input_text: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def prompt_token_count(tokenizer, input_text: str) -> int:
    prompt = build_chat_prompt(tokenizer, input_text)
    return len(tokenizer([prompt], add_special_tokens=False).input_ids[0])


def prepare_input_ids(tokenizer, input_text: str, device: torch.device) -> torch.Tensor:
    prompt = build_chat_prompt(tokenizer, input_text)
    encoded = tokenizer([prompt], add_special_tokens=False).input_ids
    return torch.as_tensor(encoded, device=device)


def mean(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.mean(numeric) if numeric else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None or float(den) == 0.0:
        return None
    return float(num) / float(den)


def format_value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): sanitize_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
