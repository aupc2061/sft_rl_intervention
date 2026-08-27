#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-configs/gsm8k_grpo_viability.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/setup_remote.sh first." >&2
  exit 2
fi
if [[ -z "${WANDB_PROJECT:-}" ]]; then
  echo "WANDB_PROJECT is required for the viability run." >&2
  echo "Run 'uv run wandb login', then export WANDB_PROJECT=sft-rl-intervention." >&2
  exit 2
fi

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gsm8k-qwen05b-grpo-viability-seed1}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
export WANDB_WATCH="${WANDB_WATCH:-false}"
export WANDB_TAGS="${WANDB_TAGS:-viability,gsm8k,qwen2.5-0.5b,grpo}"

RUN_ROOT="$("${PYTHON_BIN}" -c 'import sys; from pathlib import Path; from mats_experiments.config import load_config; c=load_config(sys.argv[1]); print(Path(c.experiment.output_root)/c.experiment.name)' "${CONFIG}")"
REPORT_DIR="${RUN_ROOT}/viability_report"
mkdir -p "${REPORT_DIR}"

echo "[1/3] Base-policy group-signal gate: 64 prompts x 4 generations"
if ! "${PYTHON_BIN}" -m mats_experiments.grpo_viability \
  --config "${CONFIG}" \
  --prompts 64 \
  --policy-label base \
  --output "${REPORT_DIR}/base_viability.json" \
  --completions-output "${REPORT_DIR}/base_completions.jsonl"; then
  echo "GSM8K failed the predeclared base-policy viability gate." >&2
  echo "Do not launch the full matrix. Inspect ${REPORT_DIR}/base_viability.json." >&2
  exit 3
fi

echo "[2/3] Four-step GRPO health pilot with per-step W&B logging"
"${PYTHON_BIN}" -m mats_experiments.train_grpo --config "${CONFIG}" --seed 1

GRPO_RUN="$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'grpo-seed1-*' \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "${GRPO_RUN}" || ! -f "${GRPO_RUN}/checkpoints/final/adapter_config.json" ]]; then
  echo "Could not locate the completed viability GRPO checkpoint under ${RUN_ROOT}." >&2
  exit 2
fi

echo "[3/3] Post-pilot rollout diagnostic on the same prompt pool and seeds"
"${PYTHON_BIN}" -m mats_experiments.grpo_viability \
  --config "${CONFIG}" \
  --checkpoint "${GRPO_RUN}/checkpoints/final" \
  --prompts 64 \
  --policy-label post-grpo \
  --diagnostic-only \
  --output "${REPORT_DIR}/post_grpo_viability.json" \
  --completions-output "${REPORT_DIR}/post_grpo_completions.jsonl"

printf 'GRPO_RUN=%q\n' "${GRPO_RUN}" > "${REPORT_DIR}/latest_viability_run.env"
echo "Viability run complete."
echo "Base gate: ${REPORT_DIR}/base_viability.json"
echo "Post-GRPO diagnostic: ${REPORT_DIR}/post_grpo_viability.json"
echo "Trainer state: ${GRPO_RUN}/checkpoints/trainer_state.json"
