"""GPU-backed prompt-level analyses listed in docs/PENDING_GPU_PLOTS.md.

The command deliberately writes the reusable individual-level cache before making
figures.  P0.1 reconstructs the frozen E3 trajectories from their declared seeds;
P0.2 consumes the already-frozen E4 RL trajectory cache.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .activations import _mean_selected_tokens
from .config import load_config
from .data import build_dataset
from .e4 import _collate_trajectories, _model_device, _sequence_kl
from .hf_backend import (
    adapter_enabled,
    encode_generation_prompt,
    encode_generation_prompts,
    load_adapter_model,
    load_tokenizer,
    require_training_stack,
    set_seed,
)
from .interventions import ResidualIntervention


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _correlations(x, y) -> dict[str, float]:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(x, y)[0, 1])
    # No-tie assumption is inappropriate here, so use average ranks from pandas.
    import pandas as pd

    spearman = float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def _loo_correlations(x, y) -> list[dict[str, float]]:
    return [
        {"omitted_index": i, **_correlations(x[:i] + x[i + 1 :], y[:i] + y[i + 1 :])}
        for i in range(len(x))
    ]


def _layer_vectors(model, encoded, layer: int, *, adapter: bool):
    import torch

    with torch.inference_mode(), adapter_enabled(model, adapter):
        hidden = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[layer + 1]
    return hidden.detach().float().cpu()


def _e3_effects(cfg, sft_model, rl_model, tokenizer, prompts, direction, layer: int):
    """Reproduce E3's per-prompt SFT||RL KL reduction at alpha=1."""
    import torch
    import torch.nn.functional as functional

    intervention = ResidualIntervention(layer, direction, "add", 1.0)
    effects = []
    trajectories = []
    with intervention.install(rl_model):
        for index, prompt in enumerate(prompts):
            encoded = encode_generation_prompt(tokenizer, prompt, return_tensors="pt").to(
                _model_device(sft_model)
            )
            set_seed(cfg.experiment.seed * 1_000_000 + 10_000 + index)
            with torch.inference_mode(), adapter_enabled(sft_model, True):
                sequence = sft_model.generate(
                    **encoded,
                    max_new_tokens=cfg.training.max_completion_length,
                    do_sample=True,
                    temperature=cfg.evaluation.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )
                source_logits = sft_model(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False).logits
            sequence_rl = sequence.to(_model_device(rl_model))
            mask = torch.zeros((1, sequence.shape[1] - 1), dtype=torch.bool, device=sequence_rl.device)
            mask[:, encoded["input_ids"].shape[1] - 1 :] = True
            source_logp = functional.log_softmax(source_logits[:, :-1].float(), dim=-1).to(sequence_rl.device)
            with torch.inference_mode(), intervention.disabled(), adapter_enabled(rl_model, True):
                baseline = rl_model(input_ids=sequence_rl, attention_mask=torch.ones_like(sequence_rl), use_cache=False).logits
            with torch.inference_mode(), adapter_enabled(rl_model, True):
                steered = rl_model(input_ids=sequence_rl, attention_mask=torch.ones_like(sequence_rl), use_cache=False).logits
            baseline_kl = float(_sequence_kl(source_logp, baseline, mask)[0].cpu())
            steered_kl = float(_sequence_kl(source_logp, steered, mask)[0].cpu())
            effects.append(baseline_kl - steered_kl)
            trajectories.append({"input_ids": sequence[0].cpu().tolist(), "completion_start": encoded["input_ids"].shape[1]})
            print(f"[P0.1] reconstructed E3 trajectory {index + 1}/{len(prompts)}", flush=True)
    return effects, trajectories


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict[str, Any]:
    require_training_stack()
    import numpy as np
    import torch
    import torch.nn.functional as functional

    cfg = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(args.direction_artifact, map_location="cpu", weights_only=True)
    direction = artifact["directions"][args.layer].detach().float()
    direction_norm2 = float(direction.square().sum())
    tokenizer = load_tokenizer(cfg, padding_side="right")
    trace_cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype=cfg.traces.dtype))
    trace_sft_model = load_adapter_model(trace_cfg, args.sft_checkpoint)
    sft_model = load_adapter_model(cfg, args.sft_checkpoint)
    rl_model = load_adapter_model(cfg, args.rl_checkpoint)
    bundle = build_dataset(cfg.data, cfg.experiment.seed)
    examples = bundle.task_test[: cfg.evaluation.kl_samples]
    prompts = [row.prompt for row in examples]

    # P0.1 prompt vectors use the exact configured final-five prompt-token summary.
    encoded = encode_generation_prompts(
        tokenizer, prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=cfg.training.max_length,
    ).to(_model_device(trace_sft_model))
    base_prompt = _layer_vectors(trace_sft_model, encoded, args.layer, adapter=False)
    sft_prompt = _layer_vectors(trace_sft_model, encoded, args.layer, adapter=True)
    base_mean = _mean_selected_tokens(base_prompt, encoded["attention_mask"].cpu(), cfg.traces.token_positions)
    sft_mean = _mean_selected_tokens(sft_prompt, encoded["attention_mask"].cpu(), cfg.traces.token_positions)
    displacement = sft_mean - base_mean
    cosine = functional.cosine_similarity(displacement, direction.unsqueeze(0), dim=-1).numpy()
    coefficient = (displacement @ direction / direction_norm2).numpy()
    effects, e3_trajectories = _e3_effects(
        cfg, sft_model, rl_model, tokenizer, prompts, direction, args.layer
    )
    p01_rows = [
        {"prompt_index": i, "example_id": examples[i].example_id, "cosine_alignment": float(cosine[i]),
         "projection_coefficient": float(coefficient[i]), "delta_toward_sft_alpha1": float(effects[i])}
        for i in range(len(examples))
    ]
    _save_csv(output / "p01_prompt_geometry.csv", p01_rows)
    p01_summary = {
        "cosine_vs_effect": {**_correlations(cosine.tolist(), effects), "leave_one_out": _loo_correlations(cosine.tolist(), effects)},
        "coefficient_vs_effect": {**_correlations(coefficient.tolist(), effects), "leave_one_out": _loo_correlations(coefficient.tolist(), effects)},
    }

    # P0.2 retains every evaluated completion-token residual from frozen E4 RL trajectories.
    cache = torch.load(args.e4_trajectory_cache, map_location="cpu", weights_only=True)
    records = cache["records"]
    token_rows = []
    prompt_rows = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        ids, attention, prediction = _collate_trajectories(batch, tokenizer.pad_token_id, _model_device(trace_sft_model))
        base_hidden = _layer_vectors(trace_sft_model, {"input_ids": ids, "attention_mask": attention}, args.layer, adapter=False)
        sft_hidden = _layer_vectors(trace_sft_model, {"input_ids": ids, "attention_mask": attention}, args.layer, adapter=True)
        residual = sft_hidden - base_hidden
        # prediction[:, t-1] evaluates token t; measure the residual at that same token position.
        for local, record in enumerate(batch):
            prompt_index = start + local
            token_indices = (torch.nonzero(prediction[local], as_tuple=False).flatten() + 1).cpu()
            values = residual[local].index_select(0, token_indices)
            dots = values @ direction
            fractions = dots.square() / (direction_norm2 * values.square().sum(dim=-1).clamp_min(1e-20))
            coefficients = dots / direction_norm2
            for relative, (token_index, q, coef) in enumerate(zip(token_indices, fractions, coefficients, strict=True)):
                token_rows.append({"prompt_index": prompt_index, "token_position": relative, "absolute_token_index": int(token_index),
                                   "projection_energy_fraction": float(q), "projection_coefficient": float(coef)})
            prompt_rows.append({"prompt_index": prompt_index, "mean_projection_energy_fraction": float(fractions.mean()),
                                "mean_projection_coefficient": float(coefficients.mean())})
        print(f"[P0.2] extracted E4 residuals {min(start + args.batch_size, len(records))}/{len(records)}", flush=True)
    e4_metrics = json.loads(Path(args.e4_metrics).read_text(encoding="utf-8"))
    semantic = next(row for row in e4_metrics["rows"] if row["label"] == "semantic" and float(row["beta"]) == 1.0)
    causal = semantic["delta_toward_rl_values"]
    for row, effect in zip(prompt_rows, causal, strict=True):
        row["delta_toward_rl_beta1"] = float(effect)
    _save_csv(output / "p02_token_projection_energy.csv", token_rows)
    _save_csv(output / "p02_prompt_projection_energy.csv", prompt_rows)
    q = [row["mean_projection_energy_fraction"] for row in prompt_rows]
    c = [row["mean_projection_coefficient"] for row in prompt_rows]
    p02_summary = {"energy_vs_effect": _correlations(q, causal), "coefficient_vs_effect": _correlations(c, causal)}

    torch.save({"direction": direction, "p01_displacements": displacement, "p01_e3_trajectories": e3_trajectories,
                "p01_effects": torch.tensor(effects), "p02_token_rows": token_rows,
                "metadata": {"layer": args.layer, "token_positions": list(cfg.traces.token_positions),
                             "sft_checkpoint": args.sft_checkpoint, "rl_checkpoint": args.rl_checkpoint,
                             "direction_artifact": args.direction_artifact, "e4_trajectory_cache": args.e4_trajectory_cache}},
               output / "p0_raw_arrays.pt")
    summary = {"layer": args.layer, "formulas": {"p01_coefficient": "<d_i,delta>/||delta||^2",
                "p02_energy": "||P_delta v_it||^2/||v_it||^2"}, "p01": p01_summary, "p02": p02_summary,
               "counts": {"p01_prompts": len(p01_rows), "p02_prompts": len(prompt_rows), "p02_tokens": len(token_rows)}}
    _atomic_json(output / "p0_summary.json", summary)
    _plot(output, p01_rows, token_rows, prompt_rows)
    return summary


def _plot(output: Path, p01, tokens, prompts) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, label in ((axes[0], "cosine_alignment", "Cosine alignment"),
                             (axes[1], "projection_coefficient", "Projection coefficient")):
        ax.scatter([r[field] for r in p01], [r["delta_toward_sft_alpha1"] for r in p01])
        ax.set(xlabel=label, ylabel="Delta toward SFT (alpha=1)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"p01_trace_vs_causal_effect.{ext}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist([r["projection_energy_fraction"] for r in tokens], bins=40)
    axes[0].set(xlabel="Token projection-energy fraction", ylabel="Count")
    axes[1].hist([r["mean_projection_energy_fraction"] for r in prompts], bins=12)
    axes[1].set(xlabel="Prompt-mean projection energy", ylabel="Count")
    axes[2].scatter([r["mean_projection_energy_fraction"] for r in prompts], [r["delta_toward_rl_beta1"] for r in prompts])
    axes[2].set(xlabel="Prompt-mean projection energy", ylabel="Delta toward RL (beta=1)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"p02_projection_energy.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P0.1/P0.2 pending GPU plot analyses")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--direction-artifact", required=True)
    parser.add_argument("--e4-trajectory-cache", required=True)
    parser.add_argument("--e4-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
