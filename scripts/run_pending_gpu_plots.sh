#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
ROOT="${REPO_ROOT}/outputs/mvp_16h_qwen05b_gsm8k"
REPORT="${ROOT}/mvp_report"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m mats_experiments.pending_gpu_plots \
  --config configs/mvp_16h_qwen05b_gsm8k.yaml \
  --sft-checkpoint "${ROOT}/sft-seed1-20260827T152137Z-360f21a0a7/checkpoints/checkpoint-12" \
  --rl-checkpoint "${ROOT}/grpo-seed1-20260827T152202Z-5153bd3355/checkpoints/checkpoint-20" \
  --direction-artifact "${ROOT}/sft-seed1-20260827T152137Z-360f21a0a7/artifacts/traces.pt" \
  --e4-trajectory-cache "${REPORT}/e4_interventions/full/trajectory_cache/rl.pt" \
  --e4-metrics "${REPORT}/e4_metrics.json" \
  --output-dir "${REPORT}/pending_gpu_plots" \
  "$@"
