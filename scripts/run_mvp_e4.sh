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
MATCHED_PAIR="${REPORT_DIR}/matched_pair.json"
E3_METRICS="${REPORT_DIR}/e3_metrics.json"
for required in "${SELECTION}" "${MATCHED_PAIR}" "${E3_METRICS}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required completed E1-E3 artifact: ${required}" >&2
    exit 2
  fi
done

LAYER="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["layer"])' "${SELECTION}")"
SFT_CHECKPOINT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sft_checkpoint_path"])' "${MATCHED_PAIR}")"
RL_CHECKPOINT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rl_checkpoint_path"])' "${MATCHED_PAIR}")"
SFT_TRACE="${SFT_RUN}/artifacts/traces.pt"
RANDOM_ROOT="${REPORT_DIR}/interventions/random_artifacts"
E4_ROOT="${REPORT_DIR}/e4_interventions"

for required in "${SFT_TRACE}" \
  "${RANDOM_ROOT}/random_101.pt" \
  "${RANDOM_ROOT}/random_102.pt" \
  "${RANDOM_ROOT}/random_103.pt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing E4 direction artifact: ${required}" >&2
    exit 2
  fi
done

# A100-SXM4 defaults. The evaluator batches cells inside one process, sharing fixed trajectories,
# source logits, base activations, model weights, and KV caches instead of duplicating models across
# one process per cell. Override any value after an OOM or after checking nvidia-smi utilization.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
E4_TRAJECTORY_BATCH_SIZE="${E4_TRAJECTORY_BATCH_SIZE:-16}"
E4_KL_BATCH_SIZE="${E4_KL_BATCH_SIZE:-4}"
E4_CELL_CHUNK_SIZE="${E4_CELL_CHUNK_SIZE:-4}"
E4_GENERATION_BATCH_SIZE="${E4_GENERATION_BATCH_SIZE:-64}"

COMMON_ARGS=(
  --config "${CONFIG}"
  --sft-checkpoint "${SFT_CHECKPOINT}"
  --rl-checkpoint "${RL_CHECKPOINT}"
  --semantic-artifact "${SFT_TRACE}"
  --random-artifact "101=${RANDOM_ROOT}/random_101.pt"
  --random-artifact "102=${RANDOM_ROOT}/random_102.pt"
  --random-artifact "103=${RANDOM_ROOT}/random_103.pt"
  --layer "${LAYER}"
  --trajectory-batch-size "${E4_TRAJECTORY_BATCH_SIZE}"
  --kl-batch-size "${E4_KL_BATCH_SIZE}"
  --cell-chunk-size "${E4_CELL_CHUNK_SIZE}"
  --generation-batch-size "${E4_GENERATION_BATCH_SIZE}"
)

if [[ "${E4_SKIP_SMOKE:-0}" != "1" && ! -f "${E4_ROOT}/smoke/e4_raw_summary.json" ]]; then
  echo "[E4] exact-pairing smoke test (2 cells, 2 examples, 16 generated tokens)"
  "${PYTHON_BIN}" -m mats_experiments.e4 \
    "${COMMON_ARGS[@]}" \
    --output-dir "${E4_ROOT}/smoke" \
    --smoke
fi

echo "[E4] layer=${LAYER}, cells=17, KL batch=${E4_KL_BATCH_SIZE}, cell chunk=${E4_CELL_CHUNK_SIZE}, generation batch=${E4_GENERATION_BATCH_SIZE}"
"${PYTHON_BIN}" -m mats_experiments.e4 \
  "${COMMON_ARGS[@]}" \
  --output-dir "${E4_ROOT}/full"

echo "[E4] paired-bootstrap analysis and Figure 4"
"${PYTHON_BIN}" -m mats_experiments.e4_analysis \
  --interventions "${E4_ROOT}/full" \
  --output-dir "${REPORT_DIR}" \
  --bootstrap-samples 2000 \
  --seed 1

echo "E4 complete: ${REPORT_DIR}/e4_metrics.json and ${REPORT_DIR}/figure4_necessity.png"
