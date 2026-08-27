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

mapfile -t CHECKPOINTS < <(
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -exec test -f '{}/adapter_config.json' ';' -print | sort -V
)
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No PEFT checkpoints containing adapter_config.json found under ${CHECKPOINT_ROOT}" >&2
  exit 2
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
  checkpoint_name="$(basename -- "${checkpoint}")"
  output="${RUN_DIRECTORY}/evaluation_${checkpoint_name}.json"
  echo "[eval] ${checkpoint} -> ${output}"
  "${PYTHON_BIN}" -m mats_experiments.evaluate \
    --config "${CONFIG}" --checkpoint "${checkpoint}" --output "${output}"
done
