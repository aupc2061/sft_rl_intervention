# GSM8K GRPO learning curve with Qwen2.5-1.5B

This run asks whether GRPO improves the untrained Qwen2.5-1.5B-Instruct policy on GSM8K.
It uses 4,096 training questions (16× the original subset) and a fixed, seeded sample of
512 questions from the official test split. This is a readiness check for later intervention
studies. The new configuration changes multiple factors relative to the 0.5B run.

| Stage | Training | Evaluation |
|---|---|---|
| Pilot | 12 optimizer steps on the 4,096-question pool | Per-step rollout logs, greedy exact-answer accuracy at step 0 and 12 |
| Full | One epoch from the untrained policy, about 1,024 optimizer steps on one GPU | Greedy exact-answer accuracy at step 0, every 128 steps, and the final checkpoint |

Both stages use seed 1, rank-8 LoRA, BF16, eight completions per prompt, temperature 0.8,
and a 768-token completion cap. The full epoch budgets 32,768 sampled completions plus
4,608 greedy generations for the checkpoint curve if it reaches the expected 1,024 steps.
The training-time estimate is about 85× the pilot's training time, with evaluation and
checkpoint I/O on top. This estimate is best refined using actual step times after warm-up.
By default the logs stay local. W&B is optional.

## Run on Linux with one BF16 CUDA GPU

```bash
bash scripts/setup_remote.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gsm8k_qwen15b_grpo.sh --pilot
```

For W&B logging, run `uv run wandb login` and set `WANDB_PROJECT` before launching.
After reviewing the pilot, pass its printed run directory to the full run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gsm8k_qwen15b_grpo.sh --pilot-run /path/to/pilot/run
```

Both stages write `artifacts/grpo_learning_curve.json`, per-example predictions, and
`plots/grpo_learning_curve.png` under their individual run directories. To regenerate the
curve after training completes, use `--evaluate-run /path/to/run` with the same config.

## Read the results

- Check the pilot's mixed-reward group fraction and length clipping. The initial operational
  gate is at least 15% mixed groups and at most 25% clipped completions, averaged across
  pilot steps. Some uniform-reward groups are expected with binary rewards. The gate is
  a check for usable rollout signal.
- Use the final checkpoint's paired accuracy change vs. step 0 as the primary result. A
  positive change with a paired 95% bootstrap interval above zero is evidence of improvement
  on this sample. Intermediate checkpoints show whether gains persist or are isolated.
- Trainer reward is measured on changing training prompts. Inspect it alongside the mixed-group
  fraction and clipping. A rising reward curve alone cannot establish held-out improvement.
- The training reward and primary evaluation require an explicit numeric `Answer:` line or a
  numeric boxed answer on the last nonempty line. A number appearing only in the reasoning
  gets zero credit. Valid answers phrased only in prose can also receive zero credit under this
  format. The report includes recognized final-answer rate and legacy last-number accuracy on
  the same held-out completions so format compliance can be separated from answer accuracy.
- The held-out curve covers one seed. Bootstrap intervals measure example-level uncertainty
  and do not account for seed-to-seed variation.
