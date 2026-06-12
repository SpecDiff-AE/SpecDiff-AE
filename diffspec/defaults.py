"""Shared defaults for public DiffSpec entry points."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_BASE_MODEL = os.environ.get(
    "DIFFSPEC_BASE_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct",
)
DEFAULT_DRAFT_MODEL = os.environ.get(
    "DIFFSPEC_DRAFT_MODEL",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
)
DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def looks_like_local_path(value: str | os.PathLike[str]) -> bool:
    """Return True for filesystem-looking model identifiers."""

    text = os.fspath(value)
    return (
        text.startswith(("/", "./", "../", "~"))
        or Path(os.path.expanduser(text)).exists()
    )


def require_existing_local_path(value: str | os.PathLike[str]) -> str | None:
    """Return an error string only when a local-looking path is missing."""

    text = os.fspath(value)
    if looks_like_local_path(text) and not Path(os.path.expanduser(text)).exists():
        return text
    return None
