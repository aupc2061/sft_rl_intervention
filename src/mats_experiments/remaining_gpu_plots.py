"""Run P1.1, P2.1, and P2.2 from the pending GPU plots checklist."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

from .config import load_config
from .data import build_dataset
from .e4 import _model_device
from .hf_backend import adapter_enabled, encode_generation_prompts, load_adapter_model, load_tokenizer


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vectors(hidden, mask, window):
    import torch

    rows = []
    positions = tuple(range(-window, 0))
    for row in range(hidden.shape[0]):
        indices = torch.nonzero(mask[row], as_tuple=False).flatten()
        selected = torch.stack([indices[p] for p in positions if -len(indices) <= p])
        rows.append(hidden[row].index_select(0, selected).mean(0))
    return torch.stack(rows)


def _extract(model, tokenizer, prompts, layer, batch_size):
    import torch

    base, tuned, masks = [], [], []
    for start in range(0, len(prompts), batch_size):
        encoded = encode_generation_prompts(
            tokenizer, prompts[start : start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(_model_device(model))
        with torch.inference_mode(), adapter_enabled(model, False):
            b = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[layer + 1]
        with torch.inference_mode(), adapter_enabled(model, True):
            t = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[layer + 1]
        # Store unpadded sequences so batching cannot affect token indexing later.
        for i in range(len(b)):
            valid = encoded["attention_mask"][i].bool()
            base.append(b[i, valid].detach().float().cpu())
            tuned.append(t[i, valid].detach().float().cpu())
            masks.append(torch.ones(int(valid.sum()), dtype=torch.long))
        print(f"[remaining] extracted {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return base, tuned, masks


def _window_matrix(base, tuned, window):
    import torch

    return torch.stack([(t[-window:] - b[-window:]).mean(0) for b, t in zip(base, tuned, strict=True)])


def _bootstrap_mean(values, seed=1, samples=2000):
    import numpy as np

    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(samples)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as F

    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg, padding_side="right")
    probes = build_dataset(cfg.data, cfg.experiment.seed).probe
    prompts = [x.prompt for x in probes]
    ids = [x.example_id for x in probes]
    trace_cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype=cfg.traces.dtype))
    sft_model = load_adapter_model(trace_cfg, args.sft_checkpoint)
    sft_base, sft_tuned, _ = _extract(sft_model, tokenizer, prompts, args.layer, args.batch_size)
    del sft_model
    torch.cuda.empty_cache()
    rl_model = load_adapter_model(trace_cfg, args.rl_checkpoint)
    rl_base, rl_tuned, _ = _extract(rl_model, tokenizer, prompts, args.layer, args.batch_size)
    del rl_model
    torch.cuda.empty_cache()

    boundary = int(len(probes) * cfg.traces.discovery_fraction)
    sft = _window_matrix(sft_base, sft_tuned, 5)
    rl = _window_matrix(rl_base, rl_tuned, 5)
    sft_direction = sft[:boundary].mean(0)
    rl_direction = rl[:boundary].mean(0)

    # P1.1: frozen discovery direction evaluated only on disjoint confirmation prompts.
    rows = []
    for i in range(boundary, len(probes)):
        item = {"prompt_index": i, "example_id": ids[i]}
        for label, matrix, direction in (("sft", sft, sft_direction), ("rl", rl, rl_direction)):
            d = matrix[i]
            item[f"{label}_cosine_alignment"] = float(F.cosine_similarity(d[None], direction[None]))
            item[f"{label}_explained_fraction"] = float((d @ direction).square() / (d.square().sum() * direction.square().sum()).clamp_min(1e-20))
            item[f"{label}_displacement_norm"] = float(torch.linalg.vector_norm(d))
        rows.append(item)
    _write_csv(out / "p11_confirmation_distributions.csv", rows)
    p11 = {}
    for metric in ("cosine_alignment", "explained_fraction"):
        differences = [r[f"sft_{metric}"] - r[f"rl_{metric}"] for r in rows]
        p11[metric] = {"paired_mean_difference_sft_minus_rl": float(np.mean(differences)),
                       "paired_bootstrap_ci95": _bootstrap_mean(differences)}

    # P2.1: subsets come exclusively from discovery; reference is exclusively confirmation.
    rng = np.random.default_rng(args.subset_seed)
    stability_rows = []
    for label, matrix in (("sft", sft), ("rl", rl)):
        reference = matrix[boundary:].mean(0)
        for n in (8, 16, 32, 64):
            for repeat in range(args.subset_repeats):
                indices = rng.choice(boundary, n, replace=False)
                estimate = matrix[indices].mean(0)
                stability_rows.append({"model": label, "subset_size": n, "repeat": repeat,
                                       "cosine_to_disjoint_confirmation": float(F.cosine_similarity(estimate[None], reference[None]))})
    _write_csv(out / "p21_trace_stability.csv", stability_rows)

    # P2.2: change only final-token window, holding prompts/layer/checkpoints fixed.
    window_rows = []
    window_matrices = {"sft": {}, "rl": {}}
    for label, base, tuned in (("sft", sft_base, sft_tuned), ("rl", rl_base, rl_tuned)):
        for window in (1, 3, 5, 10):
            matrix = _window_matrix(base, tuned, window)
            window_matrices[label][window] = matrix
        reference_direction = window_matrices[label][5][:boundary].mean(0)
        for window, matrix in window_matrices[label].items():
            direction = matrix[:boundary].mean(0)
            confirmation = matrix[boundary:]
            rho = direction.square().sum() / confirmation.square().sum(1).mean()
            window_rows.append({"model": label, "token_window": window,
                                "direction_norm": float(torch.linalg.vector_norm(direction)),
                                "confirmation_globality": float(rho),
                                "cosine_to_window5": float(F.cosine_similarity(direction[None], reference_direction[None]))})
    _write_csv(out / "p22_token_window_ablation.csv", window_rows)
    rerun = [r for r in window_rows if r["token_window"] != 5 and r["cosine_to_window5"] < args.distinct_cosine]

    torch.save({"example_ids": ids, "boundary": boundary, "sft_window_matrices": window_matrices["sft"],
                "rl_window_matrices": window_matrices["rl"], "metadata": {"layer": args.layer,
                "sft_checkpoint": args.sft_checkpoint, "rl_checkpoint": args.rl_checkpoint}}, out / "p1_p2_raw_arrays.pt")
    summary = {"p11": p11, "p21": {"subset_seed": args.subset_seed, "repeats": args.subset_repeats,
               "reference": "disjoint 64-prompt confirmation mean"},
               "p22": {"distinct_cosine_threshold": args.distinct_cosine, "e3_rerun_required": rerun},
               "counts": {"discovery": boundary, "confirmation": len(probes) - boundary}}
    (out / "p1_p2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plots(out, rows, stability_rows, window_rows)
    return summary


def _plots(out, p11, p21, p22):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in ((axes[0], "cosine_alignment", "Alignment"), (axes[1], "explained_fraction", "Explained fraction")):
        values = [[r[f"sft_{metric}"] for r in p11], [r[f"rl_{metric}"] for r in p11]]
        ax.violinplot(values, showmeans=True)
        for a, b in zip(*values, strict=True):
            ax.plot((1, 2), (a, b), color="0.75", alpha=.25, linewidth=.5)
        ax.set(xticks=(1, 2), xticklabels=("SFT", "RL"), title=title)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(out / f"p11_confirmation_distributions.{ext}", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in ("sft", "rl"):
        xs, means, lows, highs = [], [], [], []
        for n in (8, 16, 32, 64):
            v=np.array([r["cosine_to_disjoint_confirmation"] for r in p21 if r["model"]==label and r["subset_size"]==n])
            xs.append(n); means.append(v.mean()); lows.append(np.quantile(v,.025)); highs.append(np.quantile(v,.975))
        lower = np.maximum(0.0, np.array(means) - np.array(lows))
        upper = np.maximum(0.0, np.array(highs) - np.array(means))
        ax.errorbar(xs, means, yerr=[lower, upper], marker="o", label=label.upper())
    ax.set(xlabel="Discovery subset size", ylabel="Cosine to disjoint confirmation direction"); ax.legend(); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(out / f"p21_trace_stability.{ext}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label in ("sft", "rl"):
        group=sorted((r for r in p22 if r["model"]==label), key=lambda r:r["token_window"])
        for ax, metric, ylabel in zip(axes, ("direction_norm","confirmation_globality","cosine_to_window5"), ("Direction norm","Confirmation globality","Cosine to window 5")):
            ax.plot([r["token_window"] for r in group], [r[metric] for r in group], marker="o", label=label.upper()); ax.set(xlabel="Final prompt tokens", ylabel=ylabel)
    axes[0].legend(); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(out / f"p22_token_window_ablation.{ext}", dpi=200)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--sft-checkpoint", required=True); p.add_argument("--rl-checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--layer", type=int, default=10); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--subset-seed", type=int, default=20260903); p.add_argument("--subset-repeats", type=int, default=1000)
    p.add_argument("--distinct-cosine", type=float, default=.99)
    print(json.dumps(run(p.parse_args()), indent=2))


if __name__ == "__main__": main()
