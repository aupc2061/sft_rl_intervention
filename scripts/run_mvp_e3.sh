#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "Usage: $0 CONFIG SFT_RUN RL_RUN [REPORT_DIR]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="$1"
SFT_RUN="$2"
RL_RUN="$3"
REPORT_DIR="${4:-$(dirname -- "${SFT_RUN}")/mvp_report}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/setup_remote.sh first." >&2
  exit 2
fi

SELECTION="${REPORT_DIR}/selected_layer.json"
if [[ ! -f "${SELECTION}" ]]; then
  echo "Missing ${SELECTION}; run scripts/run_mvp_e1_e2.sh first." >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passes_e3_gate"] else 1)' "${SELECTION}"; then
  echo "E3 gate did not pass. Stop here and report the trace null result."
  exit 0
fi

LAYER="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["layer"])' "${SELECTION}")"
SFT_TRACE="${SFT_RUN}/artifacts/traces.pt"
MATCHED_PAIR="${REPORT_DIR}/matched_pair.json"
RL_CHECKPOINT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rl_checkpoint_path"])' "${MATCHED_PAIR}")"
SFT_CHECKPOINT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sft_checkpoint_path"])' "${MATCHED_PAIR}")"
INTERVENTIONS="${REPORT_DIR}/interventions"
mkdir -p "${INTERVENTIONS}/semantic" "${INTERVENTIONS}/random_artifacts"

echo "[E3] SFT trace, layer=${LAYER}, four scales"
SFT_CHECKPOINT="${SFT_CHECKPOINT}" SCALES="-1 0 0.5 1" bash "${SCRIPT_DIR}/run_intervention_grid.sh" \
  "${CONFIG}" "${RL_CHECKPOINT}" "${SFT_TRACE}" "${LAYER}" add "${INTERVENTIONS}/semantic"

for random_seed in 101 102 103; do
  artifact="${INTERVENTIONS}/random_artifacts/random_${random_seed}.pt"
  output="${INTERVENTIONS}/random_${random_seed}"
  "${PYTHON_BIN}" -m mats_experiments.random_directions \
    --source "${SFT_TRACE}" --output "${artifact}" --seed "${random_seed}"
  # Scale zero is identical for every direction and was already measured in the semantic grid.
  SFT_CHECKPOINT="${SFT_CHECKPOINT}" SCALES="-1 0.5 1" bash "${SCRIPT_DIR}/run_intervention_grid.sh" \
    "${CONFIG}" "${RL_CHECKPOINT}" "${artifact}" "${LAYER}" add "${output}"
done

echo "[E3] Create Figure 3"
"${PYTHON_BIN}" -m mats_experiments.mvp_analysis \
  --sft-run "${SFT_RUN}" \
  --rl-run "${RL_RUN}" \
  --output-dir "${REPORT_DIR}" \
  --interventions "${INTERVENTIONS}" \
  --min-rho-gap "$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["minimum_discovery_gap"])' "${SELECTION}")" \
  --max-accuracy-gap "$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_accuracy_gap"])' "${MATCHED_PAIR}")" \
  --min-kl-gap "$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["minimum_sft_minus_rl_kl"])' "${MATCHED_PAIR}")"

echo "E3 complete. Final figures are in ${REPORT_DIR}."
