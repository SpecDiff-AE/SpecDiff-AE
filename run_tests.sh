#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONDONTWRITEBYTECODE=1

if [[ "${DIFFSPEC_SKIP_CONDA:-0}" != "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  else
    echo "conda was not found. Set DIFFSPEC_SKIP_CONDA=1 to use the current Python environment." >&2
    exit 1
  fi
  env_name="${DIFFSPEC_CONDA_ENV:-diffspec}"
  if [[ -z "${DIFFSPEC_CONDA_ENV:-}" ]] && ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    env_name="eagle"
  fi
  conda activate "$env_name"
fi

case "${1:-unit}" in
  unit)
    python -m pytest tests -q
    ;;
  smoke)
    shift || true
    ./scripts/reproduce_llama31_8b.sh --preset smoke "$@"
    ;;
  full)
    shift || true
    ./scripts/reproduce_llama31_8b.sh --preset full "$@"
    ;;
  *)
    echo "Usage: $0 [unit|smoke|full] [extra reproduction args]" >&2
    exit 2
    ;;
esac
