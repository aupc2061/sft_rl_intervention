#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "Usage: $0 CONFIG SFT_RUN GRPO_RUN [REPORT_DIR]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="$1"
SFT_RUN="$2"
RL_RUN="$3"
REPORT_DIR="${4:-$(dirname -- "${SFT_RUN}")/mvp_report}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"
MIN_RHO_GAP="${MIN_RHO_GAP:-0.02}"
MAX_ACCURACY_GAP="${MAX_ACCURACY_GAP:-0.10}"
MIN_KL_GAP="${MIN_KL_GAP:-0.0}"

cd "${REPO_ROOT}"

echo "[1/4] Evaluate SFT checkpoints"
bash "${SCRIPT_DIR}/evaluate_run.sh" "${CONFIG}" "${SFT_RUN}"

echo "[2/4] Evaluate GRPO checkpoints"
bash "${SCRIPT_DIR}/evaluate_run.sh" "${CONFIG}" "${RL_RUN}"

echo "[3/4] Match checkpoints and extract traces"
"${PYTHON_BIN}" -m mats_experiments.mvp_analysis \
  --sft-run "${SFT_RUN}" \
  --rl-run "${RL_RUN}" \
  --output-dir "${REPORT_DIR}" \
  --max-accuracy-gap "${MAX_ACCURACY_GAP}" \
  --min-kl-gap "${MIN_KL_GAP}" \
  --match-only

MATCHED_PAIR="${REPORT_DIR}/matched_pair.json"
SFT_CHECKPOINT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sft_checkpoint_path"])' "${MATCHED_PAIR}")"
RL_CHECKPOINT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rl_checkpoint_path"])' "${MATCHED_PAIR}")"

bash "${SCRIPT_DIR}/extract_traces.sh" \
  "${CONFIG}" "${SFT_CHECKPOINT}" "${SFT_RUN}/artifacts/traces.pt"
bash "${SCRIPT_DIR}/extract_traces.sh" \
  "${CONFIG}" "${RL_CHECKPOINT}" "${RL_RUN}/artifacts/traces.pt"

echo "[4/4] Create Figures 1-2 and apply E1/E3 gates"
"${PYTHON_BIN}" -m mats_experiments.mvp_analysis \
  --sft-run "${SFT_RUN}" \
  --rl-run "${RL_RUN}" \
  --output-dir "${REPORT_DIR}" \
  --min-rho-gap "${MIN_RHO_GAP}" \
  --max-accuracy-gap "${MAX_ACCURACY_GAP}" \
  --min-kl-gap "${MIN_KL_GAP}"

RUN_ROOT="$(dirname -- "${SFT_RUN}")"
{
  printf 'CONFIG=%q\n' "${CONFIG}"
  printf 'SFT_RUN=%q\n' "${SFT_RUN}"
  printf 'RL_RUN=%q\n' "${RL_RUN}"
  printf 'REPORT_DIR=%q\n' "${REPORT_DIR}"
} > "${RUN_ROOT}/latest_runs.env"

echo "Resumed E1-E2 complete. Inspect ${REPORT_DIR}/selected_layer.json before E3."
