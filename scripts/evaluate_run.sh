#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 CONFIG RUN_DIRECTORY" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="$1"
RUN_DIRECTORY="$2"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/setup_remote.sh first." >&2
  exit 2
fi
CHECKPOINT_ROOT="${RUN_DIRECTORY}/checkpoints"
if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi

echo "[eval] persistent adapter workers=${EVAL_WORKERS} run=${RUN_DIRECTORY}"
"${PYTHON_BIN}" -m mats_experiments.evaluate_many \
  --config "${CONFIG}" \
  --run-directory "${RUN_DIRECTORY}" \
  --workers "${EVAL_WORKERS}"
