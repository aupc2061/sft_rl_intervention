# Complete frozen implementation summary

## Research question

Does SFT leave a stronger globally consistent activation trace than online RL, and does adding the
matched-SFT trace to an accuracy-matched RL model move its output distribution toward matched SFT
more strongly than norm-matched random directions?

This is a one-model, one-seed exploratory MATS work sample. It tests a causal mechanism in one
controlled setting; it does not establish generality.

## Fixed components

| Component | Frozen value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Adaptation | LoRA rank 8, alpha 16; q/k/v/o projections |
| Seed | 1 |
| New task | GSM8K |
| Train set | 256 prompts |
| Task test | 48 held-out prompts |
| Unrelated probe set | 128 synthetic non-math passages |
| Cheap old-task set | 32 held-out addition/subtraction prompts |
| SFT rate | `1e-4` |
| GRPO rate | `1e-6` |
| Training | 1 epoch, batch 4, gradient accumulation 4 |
| Generation | 96 completion tokens, 4 GRPO generations |
| E1 KL prompts | 16 |
| E3 scales | `{-1, 0, 0.5, 1}` |
| Random controls | 3, seeds 101-103 |

The objectives necessarily use different frozen learning rates, but share the base checkpoint,
data, LoRA parameterization, seed, epoch count, and evaluation protocol. There is no sweep.

## Experiment matrix

| Run ID | Factor | Value | Fixed configuration | Predeclared interpretation |
|---|---|---|---|---|
| smoke-sft | objective | SFT | tiny synthetic setup | pipeline viability only |
| smoke-rl | objective | GRPO | same tiny setup | pipeline viability only |
| mvp-sft | objective | SFT | fixed setup above | one checkpoint trajectory |
| mvp-rl | objective | GRPO | fixed setup above | one checkpoint trajectory |
| e3-semantic | direction | raw matched-SFT trace | matched RL, selected layer, four scales | specific movement toward SFT |
| e3-random-101 | direction | norm-matched Gaussian | same RL/layer/scales | generic disruption control |
| e3-random-102 | direction | norm-matched Gaussian | same RL/layer/scales | generic disruption control |
| e3-random-103 | direction | norm-matched Gaussian | same RL/layer/scales | generic disruption control |

There are two training runs plus 13 intervention evaluations: four semantic cells and three
nonzero cells for each random direction. Random scale zero reuses the identical unmodified-RL cell.

## E1: establish and match the behavioral phenomenon

Every saved SFT and GRPO checkpoint is evaluated on the same held-out data. The code plots task
accuracy against forward policy KL from the base model.

Among all SFT/RL checkpoint pairs whose absolute task-accuracy difference is at most
`MAX_ACCURACY_GAP=0.10`, choose the pair maximizing the lower of their two accuracies. Accuracy gap
breaks ties. Call this pair `(M_SFT*, M_RL*)`.

E1 passes only when:

```text
abs(A_SFT* - A_RL*) <= 0.10
and
KL(pi_0 || pi_SFT*) - KL(pi_0 || pi_RL*) > 0.
```

If E1 fails, the matched pair and Figure 1 are still reportable, but E3 is blocked.

## E2: compare global activation traces

Only the E1-matched checkpoints are used:

```text
d_l^m(x) = h_l^m*(x) - h_l^0(x)
delta_l^m = E_x d_l^m(x),  m in {SFT, RL}
rho_l^m = ||E_x d_l^m(x)||^2 / E_x ||d_l^m(x)||^2.
```

The 128 unrelated probes are split deterministically in half. Layer selection uses only

```text
argmax_l rho_l^SFT(discovery) - rho_l^RL(discovery).
```

Confirmation rho, split-half cosine, and mean pairwise cosine are reported but cannot affect layer
selection. E3 additionally requires a discovery rho gap of at least `MIN_RHO_GAP=0.02`.

## E3: one sufficiency intervention

At the single selected layer, the matched RL residual stream is changed at every token position:

```text
h_l^RL* <- h_l^RL* + alpha * delta_l^SFT.
```

The saved direction is not normalized and is not scaled to residual RMS. Consequently `alpha=1`
adds exactly one raw SFT mean-shift vector. For every random direction:

```text
||r_i||_2 = ||delta_l^SFT||_2
h_l^RL* <- h_l^RL* + alpha * r_i.
```

### Primary E3 estimand

```text
D_SFT(alpha) = E_x KL(pi_SFT*(.|x) || pi_intervened(.|x))
Delta_toward_SFT(alpha) = D_SFT(0) - D_SFT(alpha).
```

Positive values mean movement toward SFT. This is estimated as token-mean full-vocabulary KL on
trajectories sampled from matched SFT over the fixed 16-prompt KL subset. Prompt-indexed seeds do
not depend on alpha or direction, giving every cell the same SFT source sampling protocol.

### Secondary E3 estimands

```text
Delta_KL_base = KL(pi_0 || pi_intervened) - KL(pi_0 || pi_RL*)
Delta_A_task  = A_task(pi_intervened) - A_task(pi_RL*)
Delta_B_domain = B_domain(pi_intervened) - B_domain(pi_RL*).
```

Old-task accuracy, reverse KL, and intervention-induced KL are retained as diagnostic outputs but
are not used to define success. Domain intrusion is explicitly secondary.

## Gates and stopping behavior

```text
E1 fails -> report behavioral null; E3 does not run.
E1 passes, E2 gap < 0.02 -> report trace null; E3 does not run.
Both pass -> run exactly one layer, four scales, and three random controls.
```

No extra seed, layer, scale, task, learning-rate sweep, E4 necessity experiment, anchored SFT,
SFT+KL, SAE, CKA, or circuit analysis is permitted during this sprint.

## Outputs

| Artifact | Contents |
|---|---|
| `figure1_task_vs_kl.png` | all SFT/RL checkpoints: task accuracy versus forward KL |
| `matched_pair.json` | selected checkpoint paths, accuracies, KLs, and E1 gate |
| `figure2_layer_vs_rho.png` | confirmation rho across layers, with discovery-selected layer |
| `selected_layer.json` | discovery statistics and combined E3 gate |
| `figure3_steering.png` | primary toward-SFT metric plus three secondary metrics |
| `e3_metrics.json` | raw and derived values for every semantic/random cell |
| intervention JSON files | task records, probe generations, KL values, direction norms, metadata |

All run directories also contain resolved configuration, environment manifest, checkpoints,
metrics, and status files.

## Resource estimate

| Stage | Target GPU time |
|---|---:|
| Setup and smoke | 0.5-0.75 h |
| SFT plus GRPO | 2-5 h |
| Checkpoint evaluation | 1-2 h |
| Trace extraction and Figures 1-2 | 0.5-1 h |
| E3 including matched-SFT KL | 1-2.5 h |
| Buffer | 1-2 h |
| Total | approximately 5-11 GPU-hours |

E3 loads the 0.5B matched SFT and RL models together. This is appropriate for an ordinary modern
GPU, but it is the peak-memory stage. Storage should remain below roughly 20 GB. No paid API is
used.

## Remote execution

```bash
chmod +x scripts/*.sh
./scripts/setup_remote.sh
./scripts/run_mvp_16h.sh configs/mvp_16h_qwen05b_gsm8k.yaml
```

The setup script installs `uv` if needed and runs `uv sync --all-extras`; later scripts use the uv
environment's Python directly. To inspect the gate before E3:

```bash
./scripts/run_mvp_e1_e2.sh configs/mvp_16h_qwen05b_gsm8k.yaml
cat outputs/mvp_16h_qwen05b_gsm8k/mvp_report/matched_pair.json
cat outputs/mvp_16h_qwen05b_gsm8k/mvp_report/selected_layer.json

source outputs/mvp_16h_qwen05b_gsm8k/latest_runs.env
./scripts/run_mvp_e3.sh "$CONFIG" "$SFT_RUN" "$RL_RUN" "$REPORT_DIR"
```

## Interpretation rules

- Strongest positive evidence: the semantic direction yields positive `Delta_toward_SFT` larger
  than all three random controls at comparable alpha, without destroying task accuracy.
- Partial evidence: the semantic direction moves toward SFT but random directions do too, or task
  accuracy falls sharply. This suggests generic disruption rather than a specific mediator.
- Trace null: E1 passes but the discovery rho gap fails. The KL phenomenon is not explained by this
  global mean-shift statistic in this setup.
- Behavioral null: no accuracy-matched SFT-greater-than-RL KL pair exists.
- A one-seed result is a case study. Prompt bootstrap intervals do not substitute for independent
  training seeds.
