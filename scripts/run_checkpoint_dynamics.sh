#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; RUNS="${ROOT}/outputs/mvp_16h_qwen05b_gsm8k"
exec "${ROOT}/.venv/bin/python" -m mats_experiments.checkpoint_dynamics --config "${ROOT}/configs/mvp_16h_qwen05b_gsm8k.yaml" --sft-run "${RUNS}/sft-seed1-20260827T152137Z-360f21a0a7" --grpo-run "${RUNS}/grpo-seed1-20260827T152202Z-5153bd3355" --output-dir "${RUNS}/mvp_report/pending_gpu_plots" --workers "${P12_WORKERS:-3}"
