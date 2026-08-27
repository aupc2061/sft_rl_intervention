from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def _flatten_evaluation(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "kind": "evaluation",
        "task_accuracy": payload["task_accuracy"]["mean"],
        "old_accuracy": payload["old_retention"]["mean"],
        "forward_kl": payload["forward_kl"]["mean"],
        "reverse_kl": payload["reverse_kl"]["mean"],
    }


def _flatten_intervention(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "kind": "intervention",
        **{
            key: payload.get(key)
            for key in (
                "layer",
                "operation",
                "scale",
                "task_accuracy",
                "old_accuracy",
                "forward_kl",
                "reverse_kl",
                "induced_kl",
                "domain_intrusion",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate result JSON and create primary plots")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies with: uv sync --all-extras") from exc

    paths = [Path(path) for pattern in args.inputs for path in glob.glob(pattern, recursive=True)]
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_flatten_evaluation(path) if "old_retention" in payload else _flatten_intervention(path))
    if not rows:
        raise ValueError("No matching result JSON files")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "aggregate.csv", index=False)

    for x, y, name in (
        ("task_accuracy", "forward_kl", "task_vs_forward_kl"),
        ("task_accuracy", "old_accuracy", "task_vs_retention"),
        ("forward_kl", "old_accuracy", "forward_kl_vs_retention"),
    ):
        subset = frame.dropna(subset=[x, y])
        if subset.empty:
            continue
        figure, axis = plt.subplots(figsize=(6, 4))
        for kind, group in subset.groupby("kind"):
            axis.scatter(group[x], group[y], label=kind, alpha=0.8)
        axis.set_xlabel(x.replace("_", " "))
        axis.set_ylabel(y.replace("_", " "))
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=180)
        plt.close(figure)
    print(json.dumps({"rows": len(frame), "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
