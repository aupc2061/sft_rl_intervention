#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-configs/mvp_16h_qwen05b_gsm8k.yaml}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/exp1_synthetic_smoke.yaml}"
SEED="${SEED:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
MIN_RHO_GAP="${MIN_RHO_GAP:-0.02}"
MAX_ACCURACY_GAP="${MAX_ACCURACY_GAP:-0.10}"
MIN_KL_GAP="${MIN_KL_GAP:-0.0}"
REQUIRE_VIABILITY="${REQUIRE_VIABILITY:-1}"
VIABILITY_RESULT="${VIABILITY_RESULT:-outputs/gsm8k_grpo_viability_qwen05b/viability_report/base_viability.json}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/setup_remote.sh first." >&2
  exit 2
fi
if [[ "${REQUIRE_VIABILITY}" == "1" ]]; then
  if [[ ! -f "${VIABILITY_RESULT}" ]]; then
    echo "Missing required GSM8K viability result: ${VIABILITY_RESULT}" >&2
    echo "Run scripts/run_gsm8k_viability.sh before the full matrix." >&2
    exit 2
  fi
  if ! "${PYTHON_BIN}" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["suitable_for_grpo"] else 1)' "${VIABILITY_RESULT}"; then
    echo "The GSM8K viability gate failed; refusing to launch the full matrix." >&2
    exit 3
  fi
fi

latest_run() {
  local root="$1"
  local pattern="$2"
  find "${root}" -mindepth 1 -maxdepth 1 -type d -name "${pattern}" \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo "[smoke] Pipeline smoke test"
  "${PYTHON_BIN}" -m mats_experiments.train_sft --config "${SMOKE_CONFIG}" --seed "${SEED}"
  "${PYTHON_BIN}" -m mats_experiments.train_grpo --config "${SMOKE_CONFIG}" --seed "${SEED}"
fi

RUN_ROOT="$("${PYTHON_BIN}" -c 'import sys; from pathlib import Path; from mats_experiments.config import load_config; c=load_config(sys.argv[1]); print(Path(c.experiment.output_root)/c.experiment.name)' "${CONFIG}")"
mkdir -p "${RUN_ROOT}"

echo "[1/6] Train one SFT run"
"${PYTHON_BIN}" -m mats_experiments.train_sft --config "${CONFIG}" --seed "${SEED}"
SFT_RUN="$(latest_run "${RUN_ROOT}" "sft-seed${SEED}-*")"
if [[ -z "${SFT_RUN}" ]]; then
  echo "Could not locate the completed SFT run under ${RUN_ROOT}" >&2
  exit 2
fi

echo "[2/6] Train one GRPO run"
"${PYTHON_BIN}" -m mats_experiments.train_grpo --config "${CONFIG}" --seed "${SEED}"
RL_RUN="$(latest_run "${RUN_ROOT}" "grpo-seed${SEED}-*")"
if [[ -z "${RL_RUN}" ]]; then
  echo "Could not locate the completed GRPO run under ${RUN_ROOT}" >&2
  exit 2
fi

echo "[3/6] Evaluate all saved checkpoints"
bash "${SCRIPT_DIR}/evaluate_run.sh" "${CONFIG}" "${SFT_RUN}"
bash "${SCRIPT_DIR}/evaluate_run.sh" "${CONFIG}" "${RL_RUN}"

REPORT_DIR="${RUN_ROOT}/mvp_report"
echo "[4/6] Match SFT and RL checkpoints by task accuracy"
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

echo "[5/6] Extract all-layer traces from matched-performance checkpoints"
bash "${SCRIPT_DIR}/extract_traces.sh" \
  "${CONFIG}" "${SFT_CHECKPOINT}" "${SFT_RUN}/artifacts/traces.pt"
bash "${SCRIPT_DIR}/extract_traces.sh" \
  "${CONFIG}" "${RL_CHECKPOINT}" "${RL_RUN}/artifacts/traces.pt"

echo "[6/6] Create Figures 1-2 and apply the E1 plus discovery-only E3 gates"
"${PYTHON_BIN}" -m mats_experiments.mvp_analysis \
  --sft-run "${SFT_RUN}" \
  --rl-run "${RL_RUN}" \
  --output-dir "${REPORT_DIR}" \
  --min-rho-gap "${MIN_RHO_GAP}" \
  --max-accuracy-gap "${MAX_ACCURACY_GAP}" \
  --min-kl-gap "${MIN_KL_GAP}"

{
  printf 'CONFIG=%q\n' "${CONFIG}"
  printf 'SFT_RUN=%q\n' "${SFT_RUN}"
  printf 'RL_RUN=%q\n' "${RL_RUN}"
  printf 'REPORT_DIR=%q\n' "${REPORT_DIR}"
} > "${RUN_ROOT}/latest_runs.env"

echo "E1-E2 complete. Runs are recorded in ${RUN_ROOT}/latest_runs.env"
echo "Inspect ${REPORT_DIR}/selected_layer.json before launching E3."
