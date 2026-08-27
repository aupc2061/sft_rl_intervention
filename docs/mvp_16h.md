# Sixteen-hour MVP

## Claim under test

> Does SFT leave a stronger globally consistent activation trace than online RL, and does adding
> this trace to an accuracy-matched RL model move its output distribution toward the matched SFT
> model more strongly than norm-matched random directions?

This is an exploratory one-seed work sample. It cannot establish robustness across seeds, models,
or task domains.

## Frozen matrix

| Run ID | Factor | Value | Everything else fixed | Expected outcome |
|---|---|---|---|---|
| smoke-sft | objective | SFT | 0.5B, LoRA, synthetic data, seed 1 | pipeline completes |
| smoke-rl | objective | GRPO | same smoke configuration | pipeline completes |
| mvp-sft | objective | SFT | 0.5B, GSM8K, LoRA rank 8, seed 1 | higher KL at comparable accuracy |
| mvp-rl | objective | GRPO | same model/data/LoRA/seed | lower KL at comparable accuracy |
| e3-semantic | direction | matched-SFT trace | E1-matched RL checkpoint, one layer, four scales | positive toward-SFT KL reduction |
| e3-random-{1..3} | direction | random norm-matched | same RL model/layer/scales | generic-disruption baseline |

No hyperparameter sweep, E5 training intervention, SFT+KL, CKA, circuits, SAE, or second task is
allowed during this sprint.

The optimizer rates are frozen rather than swept: `1e-4` for LoRA SFT and `1e-6` for GRPO. Model,
data, trainable LoRA modules, seed, and nominal epoch count remain controlled across objectives.

## Gate and stopping rules

1. Among checkpoint pairs within `MAX_ACCURACY_GAP` (default `0.10`), choose the pair with the
   highest shared task accuracy; if none qualify, report the closest pair. E1 passes only when the
   tolerance is met and SFT's forward KL is larger than RL's. A clean null is reportable and blocks
   E3.
   Formally,
   `(M_SFT*, M_RL*) = highest-shared-accuracy checkpoint pair satisfying MAX_ACCURACY_GAP`.
   All later traces and interventions use this pair rather than the final checkpoints.
2. Select exactly one layer using the **discovery split** score
   `rho_discovery(SFT) - rho_discovery(RL)`.
3. Proceed to E3 only if E1 passes and the trace gap is at least `MIN_RHO_GAP` (default `0.02`).
4. Confirmation `rho` and split-half cosine are reported but never used for layer selection.
5. E3 uses four scales `{-1, 0, 0.5, 1}` and three random directions. Do not add layers or controls.
   Trace extraction uses all 128 probes; each steering cell uses a frozen 32-probe subset to keep
   the causal grid inside the deadline. The trace pools the final five non-padding prompt positions, which are
   causally conditioned on the user content, and computes model-minus-base differences in FP32.
6. If any stage exceeds its time box, preserve completed outputs and move to analysis/write-up.

## Frozen estimands

On unrelated probes, define the matched-checkpoint mean shifts

```text
delta_l^SFT = E_x [h_l^SFT*(x) - h_l^0(x)]
delta_l^RL  = E_x [h_l^RL*(x)  - h_l^0(x)].
```

E2 selects one layer using only the discovery split of the probe set. E3 then applies the raw
matched-SFT shift at every residual-stream token position:

```text
h_l^RL* <- h_l^RL* + alpha * delta_l^SFT,
alpha in {-1, 0, 0.5, 1}.
```

Thus `alpha=1` means one complete SFT-sized mean activation shift. Each random control `r_i` is
constructed with `||r_i||_2 = ||delta_l^SFT||_2` and receives the identical alpha grid.

The primary E3 quantity is

```text
D_SFT(alpha) = E_x KL(pi_SFT*(.|x) || pi_RL*+alpha*delta(.|x))
Delta_toward_SFT(alpha) = D_SFT(0) - D_SFT(alpha).
```

Positive `Delta_toward_SFT` means that the intervened RL output distribution moved toward matched
SFT. The implementation uses a Monte Carlo token-level KL on a fixed 16-prompt task-test subset,
with fixed-seed trajectories sampled from matched SFT. The same source-trajectory protocol is used
for every semantic and random cell.

Secondary E3 quantities are

```text
Delta_KL_base(alpha) = KL(pi_0 || pi_intervention) - KL(pi_0 || pi_RL*)
Delta_A_task(alpha)  = A_task(pi_intervention) - A_task(pi_RL*)
Delta_B_domain(alpha)= B_domain(pi_intervention) - B_domain(pi_RL*).
```

Domain intrusion is secondary and is not the definition of SFT-likeness.

## Time and compute budget

| Wall-clock block | Work | Target |
|---|---|---:|
| 0:00-0:45 | uv setup, CUDA check, smoke SFT/RL | 0.75 h |
| 0:45-5:30 | one SFT and one GRPO run | 4.75 h |
| 5:30-7:30 | checkpoint accuracy/KL evaluation | 2.0 h |
| 7:30-8:15 | SFT/RL activation extraction | 0.75 h |
| 8:15-8:45 | Figures 1-2 and E3 gate | 0.5 h |
| 8:45-11:15 | semantic + three random steering grids, if gated | 2.5 h |
| 11:15-12:00 | Figure 3 and checks | 0.75 h |
| 12:00-16:00 | write-up, rerun one failed cell only | 4.0 h |

Estimated GPU use is approximately 5-10 GPU-hours on a modern datacenter GPU, with substantial
hardware dependence. Storage should remain under roughly 20 GB because only LoRA checkpoints and
small result artifacts are saved. No paid API is required.

## Required outputs

1. `figure1_task_vs_kl.png`: SFT and RL checkpoints, task accuracy versus forward KL.
2. `figure2_layer_vs_rho.png`: confirmation globality ratio for SFT and RL across layers.
3. `figure3_steering.png`: toward-SFT KL reduction, base KL, task accuracy, and secondary domain
   intrusion for the semantic direction and three random controls.
4. `selected_layer.json`: discovery-only layer decision and whether E3 passed the gate.
5. `matched_pair.json`: the accuracy-matched checkpoints and E1 gate decision.
6. `e3_metrics.json`: raw and baseline-subtracted E3 estimands for every displayed cell.
7. A short limitations section explicitly stating the one-seed, one-model nature of the evidence.

## Analysis

- Show every checkpoint and every random control; do not report only the best point.
- Use prompt bootstrap intervals where already available, but do not call prompt count independent
  experimental replication.
- Treat seed 1 as a case study rather than a population estimate.
- For E3, use `Delta_toward_SFT` as the primary outcome and subtract the common scale-zero RL
  baseline for secondary curves.
- Distinguish inference-time functional degradation from parameter forgetting.
