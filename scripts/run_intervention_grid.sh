#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 6 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT DIRECTION_ARTIFACT LAYERS [OPERATION] [OUTPUT_DIR]" >&2
  echo "LAYERS may be comma- or space-separated. Override scales with SCALES." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="$1"
CHECKPOINT="$2"
DIRECTION_ARTIFACT="$3"
LAYERS_TEXT="$4"
OPERATION="${5:-add}"
OUTPUT_DIR="${6:-outputs/interventions}"
SCALES_TEXT="${SCALES:--1 -0.5 0 0.5 1}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/setup_remote.sh first." >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
SFT_ARGS=()
if [[ -n "${SFT_CHECKPOINT}" ]]; then
  SFT_ARGS=(--sft-checkpoint "${SFT_CHECKPOINT}")
fi
read -r -a LAYERS <<< "${LAYERS_TEXT//,/ }"
read -r -a SCALES_ARRAY <<< "${SCALES_TEXT//,/ }"

for layer in "${LAYERS[@]}"; do
  for scale in "${SCALES_ARRAY[@]}"; do
    safe_scale="${scale//-/neg}"
    safe_scale="${safe_scale//./p}"
    output="${OUTPUT_DIR}/layer${layer}_${OPERATION}_${safe_scale}.json"
    echo "[intervene] layer=${layer} operation=${OPERATION} scale=${scale}"
    "${PYTHON_BIN}" -m mats_experiments.intervene \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      "${SFT_ARGS[@]}" \
      --direction-artifact "${DIRECTION_ARTIFACT}" \
      --layer "${layer}" \
      --operation "${OPERATION}" \
      --scale "${scale}" \
      --output "${output}"
  done
done
