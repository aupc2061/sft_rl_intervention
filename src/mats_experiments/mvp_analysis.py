from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_step(path: Path) -> float:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return float(match.group(1)) if match else float("inf")


def _evaluation_points(run: Path, method: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run.glob("evaluation_*.json"), key=_checkpoint_step):
        payload = _load(path)
        rows.append(
            {
                "method": method,
                "checkpoint": path.stem.removeprefix("evaluation_"),
                "checkpoint_path": str(run / "checkpoints" / path.stem.removeprefix("evaluation_")),
                "step": _checkpoint_step(path),
                "task_accuracy": payload["task_accuracy"]["mean"],
                "forward_kl": payload["forward_kl"]["mean"],
            }
        )
    if not rows:
        raise ValueError(f"No evaluation_*.json files found in {run}")
    finite = [row["step"] for row in rows if row["step"] != float("inf")]
    final_step = max(finite, default=0.0) + 1.0
    for row in rows:
        if row["step"] == float("inf"):
            row["step"] = final_step
    return rows


def _matched_pair(
    sft_rows: list[dict[str, Any]],
    rl_rows: list[dict[str, Any]],
    max_accuracy_gap: float,
    min_kl_gap: float,
) -> dict[str, Any]:
    pairs = []
    for sft in sft_rows:
        for rl in rl_rows:
            pairs.append(
                {
                    "sft_checkpoint": sft["checkpoint"],
                    "sft_checkpoint_path": sft["checkpoint_path"],
                    "rl_checkpoint": rl["checkpoint"],
                    "rl_checkpoint_path": rl["checkpoint_path"],
                    "sft_task_accuracy": sft["task_accuracy"],
                    "rl_task_accuracy": rl["task_accuracy"],
                    "accuracy_gap": abs(sft["task_accuracy"] - rl["task_accuracy"]),
                    "sft_forward_kl": sft["forward_kl"],
                    "rl_forward_kl": rl["forward_kl"],
                    "sft_minus_rl_forward_kl": sft["forward_kl"] - rl["forward_kl"],
                }
            )
    eligible = [row for row in pairs if row["accuracy_gap"] <= max_accuracy_gap]
    if eligible:
        # Within the predeclared matching tolerance, use the strongest shared-capability pair.
        selected = max(
            eligible,
            key=lambda row: (
                min(row["sft_task_accuracy"], row["rl_task_accuracy"]),
                -row["accuracy_gap"],
            ),
        )
    else:
        selected = min(pairs, key=lambda row: row["accuracy_gap"])
    selected.update(
        {
            "max_accuracy_gap": max_accuracy_gap,
            "minimum_sft_minus_rl_kl": min_kl_gap,
            "passes_e1_gate": selected["accuracy_gap"] <= max_accuracy_gap
            and selected["sft_minus_rl_forward_kl"] > min_kl_gap,
            "selection_rule": (
                "maximum shared task accuracy among pairs within tolerance; otherwise minimum gap"
            ),
        }
    )
    return selected


def _trace_summary(run: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((run / "artifacts").glob("*.summary.json"))
    if not candidates:
        raise ValueError(f"No trace summary found under {run / 'artifacts'}")
    return candidates[-1], _load(candidates[-1])


def _select_layer(
    sft: dict[str, Any],
    rl: dict[str, Any],
    minimum_gap: float,
    passes_e1_gate: bool = True,
) -> dict[str, Any]:
    shared = sorted(set(sft["layers"]) & set(rl["layers"]), key=int)
    if not shared:
        raise ValueError("SFT and RL trace summaries share no layers")
    candidates = []
    for layer in shared:
        sft_stats, rl_stats = sft["layers"][layer], rl["layers"][layer]
        candidates.append(
            {
                "layer": int(layer),
                "discovery_gap": sft_stats["rho_discovery"] - rl_stats["rho_discovery"],
                "sft_rho_discovery": sft_stats["rho_discovery"],
                "rl_rho_discovery": rl_stats["rho_discovery"],
                "sft_rho_confirmation": sft_stats["rho_confirmation"],
                "rl_rho_confirmation": rl_stats["rho_confirmation"],
                "sft_split_half_cosine": sft_stats["split_half_cosine"],
            }
        )
    selected = max(candidates, key=lambda row: row["discovery_gap"])
    return {
        **selected,
        "minimum_discovery_gap": minimum_gap,
        "passes_trace_gate": selected["discovery_gap"] >= minimum_gap,
        "passes_e1_gate": passes_e1_gate,
        "passes_e3_gate": passes_e1_gate and selected["discovery_gap"] >= minimum_gap,
        "selection_rule": "argmax_layer rho_discovery(SFT)-rho_discovery(RL)",
        "confirmation_metrics_used_for_selection": False,
    }


def _plot_figure1(rows, output: Path, plt) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    colors = {"SFT": "#d95f02", "RL": "#1b9e77"}
    for method in ("SFT", "RL"):
        group = sorted((row for row in rows if row["method"] == method), key=lambda row: row["step"])
        axis.plot(
            [row["task_accuracy"] for row in group],
            [row["forward_kl"] for row in group],
            marker="o",
            label=method,
            color=colors[method],
        )
        for row in group:
            axis.annotate(row["checkpoint"], (row["task_accuracy"], row["forward_kl"]), fontsize=7)
    axis.set_xlabel("New-task accuracy")
    axis.set_ylabel("Forward KL from base")
    axis.set_title("Figure 1: Learning versus distributional shift")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figure1_task_vs_kl.png", dpi=200)
    plt.close(figure)


def _plot_figure2(sft, rl, selected, output: Path, plt) -> None:
    layers = sorted(set(sft["layers"]) & set(rl["layers"]), key=int)
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(
        [int(layer) for layer in layers],
        [sft["layers"][layer]["rho_confirmation"] for layer in layers],
        marker="o",
        markersize=3,
        label="SFT",
        color="#d95f02",
    )
    axis.plot(
        [int(layer) for layer in layers],
        [rl["layers"][layer]["rho_confirmation"] for layer in layers],
        marker="o",
        markersize=3,
        label="RL",
        color="#1b9e77",
    )
    axis.axvline(selected["layer"], color="black", linestyle="--", alpha=0.5, label="Selected on discovery")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Confirmation globality ratio (rho)")
    axis.set_title("Figure 2: Global activation trace across layers")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figure2_layer_vs_rho.png", dpi=200)
    plt.close(figure)


def _direction_label(path: Path, payload: dict[str, Any]) -> str:
    kind = payload.get("direction_metadata", {}).get("kind", "")
    if kind == "random_control" or "random" in path.parts:
        seed = payload.get("direction_metadata", {}).get("seed", path.parent.name)
        return f"random-{seed}"
    return "SFT trace"


def _derive_e3_row(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row["label"],
        "scale": float(row["scale"]),
        "direction_l2_norm": row.get("direction_l2_norm"),
        "delta_toward_sft": baseline["sft_to_intervention_kl"]
        - row["sft_to_intervention_kl"],
        "delta_base_kl": row["forward_kl"] - baseline["forward_kl"],
        "delta_task_accuracy": row["task_accuracy"] - baseline["task_accuracy"],
        "delta_domain_intrusion": row["domain_intrusion"] - baseline["domain_intrusion"],
        "sft_to_intervention_kl": row["sft_to_intervention_kl"],
        "base_to_intervention_kl": row["forward_kl"],
        "task_accuracy": row["task_accuracy"],
        "domain_intrusion": row["domain_intrusion"],
    }


def _plot_figure3(root: Path, output: Path, plt) -> bool:
    paths = sorted(root.rglob("*.json"))
    rows = []
    for path in paths:
        payload = _load(path)
        if (
            "scale" not in payload
            or "forward_kl" not in payload
            or payload.get("sft_to_intervention_kl") is None
        ):
            continue
        rows.append({"label": _direction_label(path, payload), **payload})
    if not rows:
        return False

    baselines = {}
    for row in rows:
        if float(row["scale"]) == 0.0:
            baselines[row["label"]] = row
    # A zero-scale intervention is direction-independent. Reuse the semantic zero cell for random
    # controls so the 16-hour runner does not spend GPU time recomputing identical baselines.
    if "SFT trace" in baselines:
        for label in {row["label"] for row in rows}:
            baselines.setdefault(label, baselines["SFT trace"])
    # Add virtual zero cells for controls. They represent the same unmodified RL policy and make
    # every curve visibly pass through the origin without rerunning an identical GPU evaluation.
    for label in {row["label"] for row in rows}:
        if label in baselines and not any(row["label"] == label and float(row["scale"]) == 0.0 for row in rows):
            rows.append({**baselines[label], "label": label, "scale": 0.0})
    derived_rows = []
    for row in rows:
        derived_rows.append(_derive_e3_row(row, baselines[row["label"]]))
    (output / "e3_metrics.json").write_text(
        json.dumps(
            {
                "primary_metric": "delta_toward_sft",
                "definition": "D_SFT(0) - D_SFT(alpha); positive means movement toward matched SFT",
                "steering": "hidden <- hidden + alpha * raw_direction",
                "rows": derived_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metric_specs = (
        ("delta_toward_sft", "Toward-SFT KL reduction (primary)"),
        ("delta_base_kl", "Delta KL(base || intervention)"),
        ("delta_task_accuracy", "Delta new-task accuracy"),
        ("delta_domain_intrusion", "Delta domain intrusion (secondary)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flatten()
    random_labels = sorted(label for label in {row["label"] for row in rows} if label.startswith("random-"))
    semantic = sorted(
        (row for row in derived_rows if row["label"] == "SFT trace"),
        key=lambda row: row["scale"],
    )
    for axis, (metric, ylabel) in zip(axes, metric_specs):
        for label in random_labels:
            group = sorted(
                (row for row in derived_rows if row["label"] == label),
                key=lambda row: row["scale"],
            )
            axis.plot(
                [row["scale"] for row in group],
                [row[metric] for row in group],
                color="0.75",
                alpha=0.8,
                linewidth=1,
            )
        if random_labels:
            scales = sorted({row["scale"] for row in derived_rows})
            means = []
            for scale in scales:
                values = [
                    row[metric]
                    for row in derived_rows
                    if row["label"] in random_labels
                    and row["scale"] == scale
                ]
                means.append(statistics.mean(values) if values else float("nan"))
            axis.plot(scales, means, color="0.35", linestyle="--", label="Random mean")
        if semantic:
            axis.plot(
                [row["scale"] for row in semantic],
                [row[metric] for row in semantic],
                marker="o",
                color="#7570b3",
                linewidth=2,
                label="SFT trace",
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xlabel("Steering scale")
        axis.set_ylabel(ylabel)
    axes[0].legend()
    figure.suptitle("Figure 3: Raw SFT-trace steering versus norm-matched random controls")
    figure.tight_layout()
    figure.savefig(output / "figure3_steering.png", dpi=200)
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the three figures for the 16-hour MVP")
    parser.add_argument("--sft-run", required=True)
    parser.add_argument("--rl-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interventions")
    parser.add_argument("--min-rho-gap", type=float, default=0.02)
    parser.add_argument("--max-accuracy-gap", type=float, default=0.10)
    parser.add_argument("--min-kl-gap", type=float, default=0.0)
    parser.add_argument("--match-only", action="store_true")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Install analysis dependencies with: uv sync --all-extras') from exc

    sft_run, rl_run = Path(args.sft_run), Path(args.rl_run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sft_evaluations = _evaluation_points(sft_run, "SFT")
    rl_evaluations = _evaluation_points(rl_run, "RL")
    evaluations = sft_evaluations + rl_evaluations
    _plot_figure1(evaluations, output, plt)
    matched = _matched_pair(
        sft_evaluations,
        rl_evaluations,
        args.max_accuracy_gap,
        args.min_kl_gap,
    )
    (output / "matched_pair.json").write_text(json.dumps(matched, indent=2), encoding="utf-8")
    if args.match_only:
        print(json.dumps(matched, indent=2))
        return
    sft_path, sft_trace = _trace_summary(sft_run)
    rl_path, rl_trace = _trace_summary(rl_run)
    selected = _select_layer(sft_trace, rl_trace, args.min_rho_gap, matched["passes_e1_gate"])
    selected.update({"sft_trace_summary": str(sft_path), "rl_trace_summary": str(rl_path)})
    (output / "selected_layer.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    _plot_figure2(sft_trace, rl_trace, selected, output, plt)
    figure3 = _plot_figure3(Path(args.interventions), output, plt) if args.interventions else False
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "selected_layer": selected["layer"],
                "discovery_gap": selected["discovery_gap"],
                "passes_e3_gate": selected["passes_e3_gate"],
                "passes_e1_gate": matched["passes_e1_gate"],
                "figure3_created": figure3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
