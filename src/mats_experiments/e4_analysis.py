from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .numerics import bootstrap_interval


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive_e4_row(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    paired = [
        base - observed
        for base, observed in zip(
            baseline["rl_to_intervention_kl_values"],
            row["rl_to_intervention_kl_values"],
            strict=True,
        )
    ]
    return {
        "label": row["label"],
        "beta": float(row["beta"]),
        "delta_toward_rl": statistics.mean(paired),
        "delta_toward_rl_values": paired,
        "delta_base_kl": row["forward_kl"] - baseline["forward_kl"],
        "delta_task_accuracy": row["task_accuracy"] - baseline["task_accuracy"],
        "delta_old_accuracy": row["old_accuracy"] - baseline["old_accuracy"],
        "delta_domain_intrusion": row["domain_intrusion"] - baseline["domain_intrusion"],
        "rl_to_intervention_kl": row["rl_to_intervention_kl"],
        "base_to_intervention_kl": row["forward_kl"],
        "induced_kl_on_rl_prefixes": row["induced_kl_on_rl_prefixes"],
        "task_accuracy": row["task_accuracy"],
        "old_accuracy": row["old_accuracy"],
        "domain_intrusion": row["domain_intrusion"],
    }


def analyze_e4(interventions, output_directory, *, bootstrap_samples: int, seed: int):
    root = Path(interventions)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.rglob("layer*_restore_paired_*.json")):
        payload = _load(path)
        if "rl_to_intervention_kl_values" in payload:
            rows.append(payload)
    if not rows:
        raise ValueError(f"No E4 intervention cells found under {root}")
    baseline = next(
        (row for row in rows if row["label"] == "semantic" and float(row["beta"]) == 0.0),
        None,
    )
    if baseline is None:
        raise ValueError("Missing semantic beta=0 E4 baseline")

    labels = sorted({row["label"] for row in rows})
    existing = {(row["label"], float(row["beta"])) for row in rows}
    for label in labels:
        if (label, 0.0) not in existing:
            rows.append({**baseline, "label": label, "beta": 0.0})
    derived = [_derive_e4_row(row, baseline) for row in rows]
    for row in derived:
        low, high = bootstrap_interval(
            row["delta_toward_rl_values"],
            samples=bootstrap_samples,
            seed=seed,
        )
        row["delta_toward_rl_ci95"] = [low, high]

    semantic_by_beta = {
        row["beta"]: row for row in derived if row["label"] == "semantic"
    }
    specificity = []
    for row in derived:
        if not row["label"].startswith("random_") or row["beta"] == 0.0:
            continue
        semantic = semantic_by_beta[row["beta"]]
        # (baseline-semantic) - (baseline-random) = random KL - semantic KL.
        paired = [
            semantic_value - random_value
            for random_value, semantic_value in zip(
                row["delta_toward_rl_values"],
                semantic["delta_toward_rl_values"],
                strict=True,
            )
        ]
        low, high = bootstrap_interval(paired, samples=bootstrap_samples, seed=seed)
        specificity.append(
            {
                "beta": row["beta"],
                "random_label": row["label"],
                "semantic_minus_random_delta_toward_rl": statistics.mean(paired),
                "ci95": [low, high],
                "semantic_beats_random": low > 0.0,
            }
        )

    metrics = {
        "primary_metric": "delta_toward_rl",
        "definition": "D_RL(0) - D_RL(beta); positive means movement toward matched RL",
        "primary_kl": "KL(matched RL || paired-restored SFT) on fixed RL-sampled trajectories",
        "operation": "exact same-prefix paired-base component restoration",
        "random_controls": "orientation controls; projection is invariant to vector norm and sign",
        "rows": sorted(derived, key=lambda row: (row["label"], row["beta"])),
        "semantic_vs_random": specificity,
    }
    (output / "e4_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies with: uv sync --all-extras") from exc

    specs = (
        ("delta_toward_rl", "Toward-RL KL reduction (primary)"),
        ("delta_base_kl", "Delta KL(base || intervention)"),
        ("delta_task_accuracy", "Delta new-task accuracy"),
        ("delta_old_accuracy", "Delta old-task accuracy"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flatten()
    random_labels = sorted(label for label in labels if label.startswith("random_"))
    semantic_rows = sorted(
        (row for row in derived if row["label"] == "semantic"), key=lambda row: row["beta"]
    )
    for axis, (metric, ylabel) in zip(axes, specs):
        for label in random_labels:
            group = sorted(
                (row for row in derived if row["label"] == label), key=lambda row: row["beta"]
            )
            axis.plot(
                [row["beta"] for row in group],
                [row[metric] for row in group],
                color="0.75",
                alpha=0.8,
                linewidth=1,
            )
        if random_labels:
            betas = sorted({row["beta"] for row in derived})
            random_means = [
                statistics.mean(
                    row[metric]
                    for row in derived
                    if row["label"] in random_labels and row["beta"] == beta
                )
                for beta in betas
            ]
            axis.plot(betas, random_means, color="0.35", linestyle="--", label="Random mean")
        axis.plot(
            [row["beta"] for row in semantic_rows],
            [row[metric] for row in semantic_rows],
            marker="o",
            color="#e7298a",
            linewidth=2,
            label="SFT trace",
        )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xlabel("Paired-restoration strength beta")
        axis.set_ylabel(ylabel)
    axes[0].legend()
    figure.suptitle("Figure 4: SFT-trace necessity versus random orientations")
    figure.tight_layout()
    figure.savefig(output / "figure4_necessity.png", dpi=200)
    plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot the E4 necessity experiment")
    parser.add_argument("--interventions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    metrics = analyze_e4(
        args.interventions,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output_dir) / "e4_metrics.json"),
                "rows": len(metrics["rows"]),
                "specificity_comparisons": len(metrics["semantic_vs_random"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
