# GSM8K GRPO viability gate

This gate answers only whether the frozen Qwen2.5-0.5B, four-generation setup supplies usable
binary-reward groups for GRPO. It cannot be selected based on the later KL or activation results.

## Fixed pilot

| Factor | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Dataset | first 64 seeded GSM8K training-pool examples |
| Generations per prompt | 4 |
| Sampling temperature | 0.8 |
| Maximum completion | 96 tokens |
| Reward | exact final numeric answer, binary |
| Short training | 64 prompts, one epoch, four optimizer steps |
| Seed | 1 |

The base rollout is the actual dataset gate. The short training run is a health diagnostic showing
whether reward and optimization metrics remain finite and whether reward behavior changes at all.

## Predeclared base-policy gates

| Metric | Pass rule | Failure meaning |
|---|---:|---|
| Parsed-answer rate | at least 0.80 | answer format/reward parser mismatch |
| Mixed-group fraction | at least 0.15 | too few groups have nonzero GRPO advantage |
| All-zero-group fraction | at most 0.80 | reward starvation |
| All-one-group fraction | at most 0.80 | task saturation |
| Truncated-completion fraction | at most 0.25 | 96-token limit is too short |

For binary reward and four generations, a mixed group has between one and three correct samples.
All-zero and all-one groups have zero within-group reward variance.

## W&B logging

When `WANDB_PROJECT` is set, every Trainer run uses `report_to="wandb"` and a descriptive run name.
The viability runner creates one W&B group containing:

1. `base-rollout-viability`: gate metrics and a completion table;
2. the four-step GRPO Trainer run: per-step TRL optimization, reward, completion, and custom parser
   metrics;
3. `post-grpo-rollout-viability`: the same rollout diagnostic after the pilot adapter.

Current TRL versions report reward mean/std, the fraction of zero-std reward groups, completion
length/clipping, entropy, and optimization statistics. The reward function also requests logging of
parsed-answer rate and exact numeric accuracy when supported by the installed TRL version.

## Run

```bash
uv run wandb login
export WANDB_PROJECT=sft-rl-intervention
export WANDB_ENTITY=YOUR_WANDB_ENTITY  # omit if your default entity is correct

mkdir -p logs
set -o pipefail
bash scripts/run_gsm8k_viability.sh configs/gsm8k_grpo_viability.yaml \
  2>&1 | tee logs/gsm8k_viability.log
```

Expected local outputs:

```text
outputs/gsm8k_grpo_viability_qwen05b/viability_report/base_viability.json
outputs/gsm8k_grpo_viability_qwen05b/viability_report/base_completions.jsonl
outputs/gsm8k_grpo_viability_qwen05b/viability_report/post_grpo_viability.json
outputs/gsm8k_grpo_viability_qwen05b/viability_report/post_grpo_completions.jsonl
```

If the base gate fails, the script exits before training and the full E1-E3 runner refuses to run.
Do not relax thresholds after seeing the result. A dataset change should preserve the model,
generation count, reward-verification quality, and all downstream train/test/probe separation.

If the base gate passes, inspect W&B for finite loss/gradient values, nonzero reward variation,
completion clipping, and any reward change over the four pilot steps. The full matrix may then be
launched with the original frozen GSM8K configuration.
