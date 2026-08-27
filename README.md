# SFT versus RL global-bias experiments

This repository implements the minimum viable causal program:

1. establish an SFT/RL performance-KL-retention gap;
2. extract paired global activation differences on unrelated probes;
3. test SFT-direction sufficiency in RL and necessity in SFT;
4. train SFT while anchoring a pilot-derived direction to the base model.

For the current 16-hour work sample, only the observation, trace extraction, and one sufficiency
intervention are active. The deadline-scoped plan is [docs/mvp_16h.md](docs/mvp_16h.md); the larger
research matrix is deferred follow-up work.

The complete frozen specification, including the exact E1-E3 estimands and interpretation rules,
is [docs/implementation_summary.md](docs/implementation_summary.md).

Every run writes a self-contained directory under `outputs/<experiment>/<run_id>/` containing
the resolved configuration, environment manifest, JSONL metrics, model artifacts, and plots.

## Scientific gates

- Do not begin activation analysis unless Experiment 1 shows a meaningful SFT/RL gap.
- Select layers on the discovery probe split and report results on the confirmation split.
- In the full follow-up, match random controls by induced policy KL. The deadline MVP uses three
  norm-matched directions at frozen scales and reports induced KL as a limitation-aware control.
- Derive training-time anchor directions from an independent pilot seed.
- Inference-time old-task degradation is called *functional degradation*, not parameter forgetting.

## Installation

```powershell
uv sync --all-extras
uv run python -m unittest discover -s tests -v
```

The local workspace used to create this scaffold had neither a GPU nor the ML dependencies, so
the real training commands are prepared but were not launched here.

## First runs

Validate configuration and offline dataset construction:

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m mats_experiments.validate_config --config configs/exp1_synthetic_smoke.yaml
uv run python -m mats_experiments.validate_config --config configs/mvp_16h_qwen05b_gsm8k.yaml --skip-data
```

Then launch the two gatekeeper training methods on a CUDA machine:

```powershell
uv run mats-train-sft --config configs/mvp_16h_qwen05b_gsm8k.yaml --seed 1
uv run mats-train-grpo --config configs/mvp_16h_qwen05b_gsm8k.yaml --seed 1
```

Evaluate every saved checkpoint, not just the final endpoint:

```powershell
.\scripts\evaluate_run.ps1 -Config configs/mvp_16h_qwen05b_gsm8k.yaml -RunDirectory outputs/mvp_16h_qwen05b_gsm8k/<run-id>
```

See [docs/mvp_16h.md](docs/mvp_16h.md) for the active matrix, stopping rules, timeline, resource
estimate, and analysis plan.

## Remote Linux GPU workflow

```bash
chmod +x scripts/*.sh
./scripts/setup_remote.sh

# Required dataset/reward-signal gate before the full matrix.
uv run wandb login
export WANDB_PROJECT=sft-rl-intervention
./scripts/run_gsm8k_viability.sh configs/gsm8k_grpo_viability.yaml

# Complete workflow: smoke, one SFT, one RL, traces, then conditional E3.
./scripts/run_mvp_16h.sh configs/mvp_16h_qwen05b_gsm8k.yaml
```

`setup_remote.sh` installs `uv` with its official installer when necessary, then runs
`uv sync --all-extras`. Every subsequent Bash runner automatically uses `.venv/bin/python`, so no
manual activation step is required. Set `UV_PROJECT_ENVIRONMENT` or `VENV_DIR` if the environment
must live somewhere else.

See [docs/gsm8k_viability.md](docs/gsm8k_viability.md) for the predeclared reward-signal thresholds
and W&B metrics. When `WANDB_PROJECT` is set, all Trainer-based SFT and GRPO runs log to W&B;
otherwise they retain local-only logging.

To inspect the trace result before spending compute on steering, run the two phases separately:

```bash
./scripts/run_mvp_e1_e2.sh configs/mvp_16h_qwen05b_gsm8k.yaml
cat outputs/mvp_16h_qwen05b_gsm8k/mvp_report/selected_layer.json

source outputs/mvp_16h_qwen05b_gsm8k/latest_runs.env
./scripts/run_mvp_e3.sh "$CONFIG" "$SFT_RUN" "$RL_RUN" "$REPORT_DIR"
```

Set `CUDA_VISIBLE_DEVICES` before a command to bind it to a specific GPU. Multiple independent
seeds may be launched in separate processes only when each process has its own GPU. The 16-hour MVP
uses exactly one seed and must not be expanded during the sprint.
