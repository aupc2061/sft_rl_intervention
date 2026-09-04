# Pending GPU plot experiments

_Deferred analyses for the SFT--RL activation-intervention writeup, recorded 3 September 2026_

---

## 🎯 Decision summary

The current E2--E4 results already support the narrow claim that the SFT mean-shift direction is stable and causally captures part of the SFT--RL functional difference. The tasks below are **not required to preserve that result**. They matter because they would strengthen the mechanistic interpretation, especially the currently missing prompt-level link from activation geometry to behavioral effect.

If GPU time is limited, run tasks P0.1 and P0.2 first. Stop after those unless their outputs are informative.

## ✍️ Pending tasks

| Priority | Task | Importance | Status |
| --- | --- | --- | --- |
| P0.1 | E2/E3 prompt-level trace expression versus causal effect | High | Complete |
| P0.2 | E4 projection energy versus causal movement | High | Complete |
| P1.1 | Confirmation-prompt alignment and explained-fraction distributions | Medium-high | Complete |
| P1.2 | Clean direct-load E1 checkpoint dynamics | Medium | Complete |
| P2.1 | Trace stability versus probe-set size | Medium-low | Complete |
| P2.2 | Token-position extraction ablation | Optional | Complete |

### P0.1: Connect E2 geometry to E3 causal behavior

**Question:** Do prompts that express more of the SFT trace undergo a larger causal change when that trace is added to RL?

- Reuse the frozen 16 E3 primary prompts and trajectories
- Run base and matched SFT on each exact E3 prefix
- At layer 10, retain each prompt-level displacement (d_i^{\mathrm{SFT}})
- Compute both:
  - cosine alignment, \(\cos(d_i^{\mathrm{SFT}},\delta^{\mathrm{SFT}})\)
  - projection coefficient, \(\langle d_i^{\mathrm{SFT}},\delta\rangle / \|\delta\|^2\)
- Relate each quantity to the already saved \(\Delta_{\mathrm{toward\,SFT},i}(\alpha=1)\)
- Report Pearson and Spearman correlations, raw scatter, and a leave-one-out sensitivity check

**Why this is important:** This is the strongest missing mechanistic bridge. E2 currently identifies a global direction and E3 shows a causal aggregate effect, but the existing artifacts cannot show that prompt-to-prompt variation in the representation predicts prompt-to-prompt causal variation.

**Interpretation rule:** A null association would not invalidate the aggregate causal effect. It would mean that this prompt-level activation summary does not explain effect heterogeneity.

### P0.2: Measure E4 projection energy and relate it to causal movement

**Question:** Is the restored component geometrically small but functionally influential, or simply a large part of the SFT--base activation change?

- Reuse the frozen RL trajectories from E4
- Teacher-force base and matched SFT on the exact same prefix at every evaluated token
- Retain the layer-10 residual \(v_{i,t}=h^{\mathrm{SFT}}_{i,t}-h^0_{i,t}\)
- Compute:
  \[
  q_{i,t}=\frac{\|P_\delta v_{i,t}\|^2}{\|v_{i,t}\|^2}
  \]
- Plot token-level and prompt-aggregated distributions
- Relate prompt-mean projection energy and coefficient to the saved \(\Delta_{\mathrm{toward\,RL},i}(\beta=1)\)
- Include token-position summaries to show whether a few positions dominate

**Why this is important:** This determines whether E4 demonstrates high causal leverage from a small activation component or removal of a geometrically dominant component. Those are materially different mechanisms.

### P1.1: Show the confirmation-prompt alignment distribution

**Question:** Are E2's high globality statistics broad across prompts or driven by a minority of examples?

- Recompute and retain all 64 confirmation-prompt displacement vectors for SFT and RL
- Plot paired strip/violin distributions of \(\cos(d_i,\delta)\)
- Plot the per-prompt explained fraction \(\|P_\delta d_i\|^2/\|d_i\|^2\)
- Pair SFT and RL by prompt and bootstrap the paired mean difference
- Preserve direction magnitude as a separate statistic; do not conflate alignment with norm

**Why this is important:** It makes the scalar E2 globality result auditable and intuitive. It is less important than P0 because it strengthens characterization rather than directly linking representation to behavior.

### P1.2: Re-evaluate E1 checkpoint dynamics by direct loading

**Question:** Is the selected SFT/RL pair representative of the training trajectories rather than an isolated checkpoint coincidence?

- Evaluate every retained SFT checkpoint by loading its adapter into a fresh base-model process
- Do the same for retained GRPO checkpoints
- Verify unloaded base logits before each adapter evaluation
- Record GSM8K accuracy and forward KL from base on the same fixed examples and prefixes
- Plot accuracy versus optimizer step, forward KL versus optimizer step, and the accuracy--KL frontier
- Highlight the pair selected under the frozen matching rule

**Why this is important:** A clean trajectory would strengthen the central setup, but the corrected selected pair is already sufficient for E2--E4. This becomes high priority only if the final application prominently claims a general training-dynamics or Pareto-frontier pattern.

### P2.1: Trace stability versus probe-set size

**Question:** How many unrelated prompts are required to estimate the direction reliably?

- Save individual discovery-prompt displacements
- For \(n\in\{8,16,32,64\}\), draw repeated subsets using a fixed seed schedule
- Compare each subset direction with a disjoint confirmation reference direction
- Plot mean cosine with bootstrap intervals for SFT and RL
- Never compare a subset direction with a full direction containing the same examples without labeling the overlap

**Why this is less important:** The existing split-half cosines are already approximately 0.999 for SFT and 0.998 for RL. This plot improves robustness reporting but is unlikely to change the main conclusion.

### P2.2: Token-position extraction ablation

**Question:** Does the result depend on averaging the final five non-padding prompt positions?

- Recompute directions using last 1, 3, 5, and 10 prompt tokens
- Compare direction norm, confirmation globality, and cross-window cosine
- Only rerun E3 at \(\alpha=1\) for windows that produce meaningfully distinct directions
- Keep model, prompts, layer, checkpoint pair, and trajectories fixed

**Why this is optional:** It validates a design choice but adds little if all extraction windows produce nearly identical directions. It should not displace the prompt-level geometry-to-behavior analyses.

## 📊 Required outputs

Each GPU task should write raw quantities before plotting:

- Per-example or per-token arrays in a safe tensor format
- A JSON summary with source checkpoints, layer, prompt IDs, token positions, and formulas
- A CSV containing the plotted rows
- A deterministic plotting script
- PNG and vector PDF versions of each accepted figure

Do not save only aggregate means again; the missing individual-level arrays are the reason these plots cannot be produced from the present artifacts.

## ⚙️ Execution order

1. Add a single inference pass that retains layer-10 residuals for the frozen E3 prompts and E4 trajectories
2. Produce P0.1 and P0.2 from that shared cache
3. Decide whether the results strengthen the mechanism
4. If useful, extend the same cache format to the 64 E2 confirmation prompts for P1.1
5. Run P1.2 only if the writeup needs a clean training-dynamics claim
6. Treat P2.1 and P2.2 as optional robustness work

## 📋 Completion criteria

- [x] P0.1 raw activation projections saved and plotted
- [x] P0.1 leave-one-out correlation sensitivity reported
- [x] P0.2 token-level projection fractions saved and plotted
- [x] P0.2 prompt-level projection versus causal effect plotted
- [x] P1.1 paired SFT/RL distributions plotted with paired intervals
- [x] P1.2 all displayed checkpoints verified through fresh direct loading
- [x] P2.1 subset sampling uses a disjoint reference
- [x] P2.2 changes only the token-extraction window
- [x] Every accepted figure has an exact source artifact in `outputs/mvp_16h_qwen05b_gsm8k/mvp_report/pending_gpu_plots/`; claims and caveats remain subject to writeup review
