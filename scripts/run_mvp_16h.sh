#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-configs/mvp_16h_qwen05b_gsm8k.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/run_mvp_e1_e2.sh" "${CONFIG}"

RUN_ROOT="$("${PYTHON_BIN}" -c 'import sys; from pathlib import Path; from mats_experiments.config import load_config; c=load_config(sys.argv[1]); print(Path(c.experiment.output_root)/c.experiment.name)' "${CONFIG}")"
# This file is produced by the immediately preceding trusted local runner.
# shellcheck disable=SC1090
source "${RUN_ROOT}/latest_runs.env"

bash "${SCRIPT_DIR}/run_mvp_e3.sh" \
  "${CONFIG}" "${SFT_RUN}" "${RL_RUN}" "${REPORT_DIR}"
