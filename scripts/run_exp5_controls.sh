#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 ANCHORED_CONFIG KL_CONFIG" >&2
  echo "Override independent seeds with SEEDS='2 3 4'." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ANCHORED_CONFIG="$1"
KL_CONFIG="$2"
SEEDS_TEXT="${SEEDS:-2 3 4}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/setup_remote.sh first." >&2
  exit 2
fi
read -r -a SEED_ARRAY <<< "${SEEDS_TEXT//,/ }"
for seed in "${SEED_ARRAY[@]}"; do
  echo "[E5] anchored SFT seed=${seed}"
  "${PYTHON_BIN}" -m mats_experiments.train_anchored_sft \
    --config "${ANCHORED_CONFIG}" --seed "${seed}"
  echo "[E5] explicit SFT+KL seed=${seed}"
  "${PYTHON_BIN}" -m mats_experiments.train_sft_kl \
    --config "${KL_CONFIG}" --seed "${seed}"
done
