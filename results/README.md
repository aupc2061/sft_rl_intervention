# Result artifact index

This file maps the repository's headline claims to machine-readable artifacts. Paths are relative
to the repository root.

## Headline results

The compact, machine-readable record of every number reported in the root README is
[`headline_results.json`](headline_results.json).

## Core E1–E4 artifacts

| Claim or output | Artifact |
|---|---|
| Accuracy-matched SFT/GRPO checkpoints and forward KL from base | [`headline_results.json`](headline_results.json), field `e1` |
| Discovery-selected layer and held-out globality statistics | [`headline_results.json`](headline_results.json), field `e2` |
| SFT all-layer activation trace | [`traces.pt`](../outputs/mvp_16h_qwen05b_gsm8k/sft-seed1-20260827T152137Z-360f21a0a7/artifacts/traces.pt) |
| GRPO all-layer activation trace | [`traces.pt`](../outputs/mvp_16h_qwen05b_gsm8k/grpo-seed1-20260827T152202Z-5153bd3355/artifacts/traces.pt) |
| E3 semantic and random-control metrics | [`e3_metrics.json`](../outputs/mvp_16h_qwen05b_gsm8k/mvp_report/e3_metrics.json) |
| E4 paired-restoration and random-control metrics | [`e4_metrics.json`](../outputs/mvp_16h_qwen05b_gsm8k/mvp_report/e4_metrics.json) |

## E5 off-task diagnostics

| Benchmark | Artifact | Status |
|---|---|---|
| WinoGrande, 200 examples | [`headline_results.json`](headline_results.json), field `e5.winogrande_accuracy` | No measurable SFT deficit; strong option-1 bias |
| HellaSwag, full validation split | [`headline_results.json`](headline_results.json), field `e5.hellaswag_accuracy` | No measurable SFT deficit; strong answer-position bias |

These benchmarks do not establish forgetting or recovery in this experiment.

## Additional diagnostics

The files under
[`pending_gpu_plots/`](../outputs/mvp_16h_qwen05b_gsm8k/mvp_report/pending_gpu_plots/)
contain the CSV/JSON sources and rendered plots for:

- confirmation-prompt alignment and explained fraction;
- checkpoint dynamics;
- token-window sensitivity;
- trace stability across discovery subset sizes;
- prompt-level causal effects;
- token- and prompt-level projection energy.

## Matched checkpoints

- SFT: `outputs/mvp_16h_qwen05b_gsm8k/sft-seed1-20260827T152137Z-360f21a0a7/checkpoints/checkpoint-12`
- GRPO: `outputs/mvp_16h_qwen05b_gsm8k/grpo-seed1-20260827T152202Z-5153bd3355/checkpoints/checkpoint-20`

The exact selection rule and recorded task/KL values are summarized in `headline_results.json`;
downstream E2–E4 analyses use this frozen pair.
