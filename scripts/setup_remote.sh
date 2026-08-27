#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-uv}"

cd "${REPO_ROOT}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "uv is missing and curl is unavailable. Install uv, then rerun this script." >&2
    exit 2
  fi
  echo "uv was not found; installing it with the official installer."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv installation completed but '${UV_BIN}' is still not on PATH." >&2
  exit 2
fi

export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
"${UV_BIN}" sync --all-extras --python "${PYTHON_BIN}"
"${UV_BIN}" run python -m unittest discover -s tests -v
"${UV_BIN}" run python -m mats_experiments.validate_config \
  --config configs/exp1_synthetic_smoke.yaml
"${UV_BIN}" run python -m mats_experiments.validate_config \
  --config configs/mvp_16h_qwen05b_gsm8k.yaml --skip-data
"${UV_BIN}" run python -m mats_experiments.validate_config \
  --config configs/gsm8k_grpo_viability.yaml --skip-data

"${UV_BIN}" run python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to launch GPU experiments.")
PY

echo "Remote environment is ready at ${VENV_DIR}"
