#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${1:-configs/exp1_qwen_gsm8k.yaml}}"
SEEDS_TEXT="${SEEDS:-1 2 3}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/setup_remote.sh first." >&2
  exit 2
fi
read -r -a SEED_ARRAY <<< "${SEEDS_TEXT//,/ }"
if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "No seeds supplied through SEEDS." >&2
  exit 2
fi

for seed in "${SEED_ARRAY[@]}"; do
  echo "[E1] SFT seed=${seed} config=${CONFIG}"
  "${PYTHON_BIN}" -m mats_experiments.train_sft --config "${CONFIG}" --seed "${seed}"
  echo "[E1] GRPO seed=${seed} config=${CONFIG}"
  "${PYTHON_BIN}" -m mats_experiments.train_grpo --config "${CONFIG}" --seed "${seed}"
done

echo "Experiment 1 training finished. Evaluate every saved checkpoint to build Pareto fronts."
