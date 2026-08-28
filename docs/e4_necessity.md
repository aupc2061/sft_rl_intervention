# E4: exact paired-base necessity intervention

## Frozen question

Does restoring the matched SFT model's fine-tuning-induced residual component along the selected
SFT trace toward the paired base activation move its output distribution toward matched RL more
than restoring along random orientations?

E4 changes only the restoration orientation and strength. It reuses:

- matched SFT checkpoint 12 and matched GRPO checkpoint 20 from E1;
- discovery-selected layer 10 from E2;
- the matched-SFT trace and random seeds 101-103 used in E3;
- seed 1 and the existing task, old-task, probe, and KL splits.

## Intervention

At every token position and the same layer hook used to extract the trace:

```text
h_sft(x_1:t) <- h_sft(x_1:t)
                - beta P_direction(h_sft(x_1:t) - h_base(x_1:t)).
```

The base and SFT activations are computed from identical token IDs, causal prefixes, attention
masks, position convention, and layer hook. This is not restoration to a global base mean.

For fixed-trajectory KL, RL first samples the completion. RL, base, and SFT are then teacher-forced
on the identical cached sequence. For generation metrics, base and SFT maintain separate KV caches
but consume the same evolving intervened-SFT prefix at every decoding step.

## Experiment matrix

| Run family | Factor | Values | Fixed configuration | Expected outcome |
|---|---|---:|---|---|
| E4-semantic | restoration strength | `0,.25,.5,.75,1` | SFT trace, matched SFT, layer 10 | dose-dependent movement toward RL |
| E4-random-101 | restoration strength | `.25,.5,.75,1` | random orientation 101, same target/layer | generic projection control |
| E4-random-102 | restoration strength | `.25,.5,.75,1` | random orientation 102, same target/layer | generic projection control |
| E4-random-103 | restoration strength | `.25,.5,.75,1` | random orientation 103, same target/layer | generic projection control |

There are 17 unique cells. The zero-strength SFT policy is direction-independent and is reused as
the virtual zero cell for every random curve.

Random-vector norm is not a factor: `P_(c r) = P_r` for every nonzero scalar `c`. Existing E3
random artifacts are reused for their orientations; their stored norms are bookkeeping only.

## Estimands and analysis

The primary metric uses fixed RL-sampled trajectories:

```text
D_RL(beta) = E_x KL(matched RL(.|x) || restored SFT_beta(.|x))
Delta_toward_RL(beta) = D_RL(0) - D_RL(beta).
```

Positive `Delta_toward_RL` means movement toward RL. Prompt-paired bootstrap intervals are computed
for each cell. At a shared beta, semantic specificity is:

```text
Delta_toward_RL(semantic) - Delta_toward_RL(random).
```

The analysis reports whether its 95% paired-bootstrap interval excludes zero. Secondary metrics
are forward KL from base, new-task accuracy, old-task accuracy, intervention-induced KL on the
fixed RL prefixes, and domain intrusion. Domain intrusion remains diagnostic rather than defining
success.

## A100-SXM4 execution design

E4 uses one matched-SFT/base PEFT model and one matched-RL model. The SFT model is alternated between
adapter-disabled base passes and adapter-enabled SFT passes, with independent KV-cache streams.
This avoids loading a third model.

Cells are packed into GPU batches rather than evaluated by one process per cell. Within each KL
batch, the evaluator reuses the fixed trajectories, reference logits, and base activations across
multiple directions and beta values. Free generation also mixes cells in the batch and performs
synchronized base/SFT decoding. Trajectories are cached on disk, results are written atomically,
TF32 and BF16 remain enabled, and batch sizes are environment-overridable.

Default A100 40 GB settings:

```text
E4_TRAJECTORY_BATCH_SIZE=16
E4_KL_BATCH_SIZE=4
E4_CELL_CHUNK_SIZE=4
E4_GENERATION_BATCH_SIZE=64
```

Expected cost is approximately 1.5-2.5 times E3 wall time, with no training and little additional
storage. The exact paired base pass approximately doubles SFT evaluation compute, while batching
and cross-cell reuse recover much of that overhead. There are no API costs.

## Execution

```bash
chmod +x scripts/*.sh
source outputs/mvp_16h_qwen05b_gsm8k/latest_runs.env
./scripts/run_mvp_e4.sh "$CONFIG" "$SFT_RUN" "$RL_RUN" "$REPORT_DIR"
```

The script first runs a two-cell, two-example smoke unless `E4_SKIP_SMOKE=1`. It then writes:

- `mvp_report/e4_interventions/full/`: 17 self-contained cell files and trajectory caches;
- `mvp_report/e4_metrics.json`: derived metrics and paired-bootstrap specificity comparisons;
- `mvp_report/figure4_necessity.png`: primary and secondary dose-response plots.

If the defaults OOM, reduce `E4_CELL_CHUNK_SIZE` first, then `E4_KL_BATCH_SIZE`. If decoding memory
is the problem, reduce `E4_GENERATION_BATCH_SIZE`. These changes affect throughput, not estimands.

## Interpretation

- Strong partial-necessity evidence: positive semantic `Delta_toward_RL`, larger than every random
  orientation at the same beta, without a large new-task accuracy loss.
- Functional but nonselective: semantic restoration moves toward RL but random restorations do too.
- Destructive ablation: movement toward RL is accompanied by substantial task-accuracy loss.
- Necessity null: semantic restoration does not move SFT toward RL.

Even a positive result supports only partial necessity or mediation in this one-model, one-seed
setting. It does not establish that the trace explains the complete SFT/RL difference.
