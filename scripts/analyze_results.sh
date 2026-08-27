#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY INPUT_GLOB [INPUT_GLOB ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIRECTORY="$1"
shift
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"
cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/setup_remote.sh first." >&2
  exit 2
fi

"${PYTHON_BIN}" -m mats_experiments.analysis \
  --output-dir "${OUTPUT_DIRECTORY}" \
  --inputs "$@"
