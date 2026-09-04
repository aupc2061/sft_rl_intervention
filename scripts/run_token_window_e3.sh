#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; RUNS="${ROOT}/outputs/mvp_16h_qwen05b_gsm8k"; OUT="${RUNS}/mvp_report/pending_gpu_plots"
exec "${ROOT}/.venv/bin/python" -m mats_experiments.token_window_e3 --config "${ROOT}/configs/mvp_16h_qwen05b_gsm8k.yaml" --sft-checkpoint "${RUNS}/sft-seed1-20260827T152137Z-360f21a0a7/checkpoints/checkpoint-12" --rl-checkpoint "${RUNS}/grpo-seed1-20260827T152202Z-5153bd3355/checkpoints/checkpoint-20" --raw-p0 "${OUT}/p0_raw_arrays.pt" --raw-p12 "${OUT}/p1_p2_raw_arrays.pt" --output-dir "${OUT}"
