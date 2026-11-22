#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-${VENV_DIR:-$repo_root/.venv}/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Missing Python environment. Run bash scripts/setup_remote.sh first." >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
exec "$python_bin" -m mats_experiments.grpo_learning_curve "$@"
