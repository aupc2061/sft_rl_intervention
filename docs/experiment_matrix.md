# Full post-application research matrix

> **Do not run this matrix during the 16-hour sprint.** The active deadline-scoped plan is
> [mvp_16h.md](mvp_16h.md). This document is retained only as the follow-up research program.

## Primary estimands

- Experiment 1: forward KL at matched new-task accuracy; old-task retention is secondary.
- Experiment 2: globality ratio `rho = ||E[d]||^2 / E[||d||^2]` on held-out unrelated probes.
- Experiments 3-4: change in a predeclared domain-intrusion score at matched induced KL.
- Experiment 5: change in forward KL and retention caused by pilot-derived anchoring, conditional on
  the intervention measurably reducing the candidate trace.

## Training matrix

| Run family | Method | Changed factor | Fixed configuration | Expected outcome |
|---|---|---|---|---|
| E1-SFT | SFT | Objective=fixed demonstrations | Base, task, LoRA, seed set | Higher KL at matched accuracy |
| E1-RL | GRPO | Objective=online binary reward | Base, task, LoRA, seed set | Lower KL at matched accuracy |
| E5-ANCHOR | Anchored SFT | Anchor beta | Pilot direction, SFT data and LoRA | Dose-dependent trace/KL reduction |
| E5-RANDOM | Anchored SFT | Direction=random | Norm/layer/beta matched | Controls generic regularization |
| E5-KL | SFT+KL | Explicit KL coefficient | SFT data and LoRA | Conservative-SFT positive control |

Seeds are `{1, 2, 3}`. Training strength is represented by saved checkpoints, avoiding separate
retraining solely to obtain performance-matched points.

## Inference intervention matrix

After discovery-layer screening, retain at most three layers and evaluate on confirmation probes.

| Target | Direction | Operation | Scales |
|---|---|---|---|
| Base | SFT trace | add | `-1,-0.5,0,0.5,1` |
| RL | SFT trace | add | `-1,-0.5,0,0.5,1` |
| RL | RL trace | add | `-1,-0.5,0,0.5,1` |
| RL | contrast trace | add | `-1,-0.5,0,0.5,1` |
| RL | random controls (8) | add | calibrated to induced KL |
| SFT | SFT/contrast trace | subtract mean shift | `0,0.25,0.5,0.75,1` |
| SFT | SFT/contrast trace | paired base restoration | `0,0.25,0.5,0.75,1` |

## Stopping rules

1. Stop after E1 for a task if the SFT and RL Pareto fronts show no meaningful KL separation.
2. Stop trace interpretation if held-out `rho`, cross-prompt alignment, and split-half direction
   cosine do not exceed random baselines.
3. Do not claim specificity if semantic and random directions have comparable effects after induced
   KL matching.
4. Do not launch anchored SFT unless a pilot direction transfers to an independent inference seed.

## Analysis plan

- Bootstrap prompts/examples (10,000 resamples) for 95% confidence intervals.
- Treat seed as the replication unit; plot every seed rather than only aggregate bars.
- Compare Pareto frontiers rather than single hand-picked endpoints.
- Fit retention as a function of forward KL, method, and their interaction.
- Correct layer-wise discovery tests using Benjamini-Hochberg; confirmation layers are tested once.
- Failed runs remain in the manifest and are excluded only using predeclared infrastructure criteria.

Primary plots:

1. new-task accuracy versus forward KL;
2. new-task accuracy versus old-task retention;
3. forward KL versus retention;
4. layer versus trace globality/readability;
5. intervention scale versus outcomes, with induced KL shown alongside scale.

## Resource estimate

For Qwen2.5-3B LoRA with short GSM8K generations, rough planning numbers are:

- E1: 2 methods x 3 seeds = 6 runs, approximately 60-150 accelerator-hours;
- activation extraction/evaluation: approximately 10-30 accelerator-hours;
- anchored-SFT pilot: 5 beta values + random + KL control, 3 seeds would be 21 runs and should only
  be launched after sequential elimination; approximately 100-300 accelerator-hours;
- adapters and metrics: roughly 20-60 GB; full checkpoints can exceed 300 GB.

These are budgeting estimates, not measured throughput. Run the 0.5B smoke configuration first and
record actual tokens/second before scheduling the 3B matrix. No external API is required.
