#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-configs/mvp_16h_qwen05b_gsm8k.yaml}"
REPORT_DIR="${2:-outputs/mvp_16h_qwen05b_gsm8k/mvp_report}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR:-${REPO_ROOT}/.venv}/bin/python}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${REPO_ROOT}/outputs/mvp_16h_qwen05b_gsm8k/sft-seed1-20260827T152137Z-360f21a0a7/checkpoints/checkpoint-12}"
RL_CHECKPOINT="${RL_CHECKPOINT:-${REPO_ROOT}/outputs/mvp_16h_qwen05b_gsm8k/grpo-seed1-20260827T152202Z-5153bd3355/checkpoints/checkpoint-20}"
DIRECTION_ARTIFACT="${DIRECTION_ARTIFACT:-${REPO_ROOT}/outputs/mvp_16h_qwen05b_gsm8k/sft-seed1-20260827T152137Z-360f21a0a7/artifacts/traces.pt}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m mats_experiments.hellaswag_eval \
  --config "${CONFIG}" \
  --sft-checkpoint "${SFT_CHECKPOINT}" \
  --rl-checkpoint "${RL_CHECKPOINT}" \
  --direction-artifact "${DIRECTION_ARTIFACT}" \
  --layer 10 \
  --output "${REPORT_DIR}/hellaswag_retention.json" \
  --limit "${HELLASWAG_LIMIT:-200}" \
  --batch-size "${HELLASWAG_BATCH_SIZE:-4}" \
  --device "${HELLASWAG_DEVICE:-cpu}"
