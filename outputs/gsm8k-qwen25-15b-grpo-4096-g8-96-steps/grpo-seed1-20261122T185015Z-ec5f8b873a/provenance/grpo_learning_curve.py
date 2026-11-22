"""Run a GRPO health pilot and evaluate held-out learning trajectories."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean

from .config import load_config
from .data import build_dataset
from .evaluate import accuracy_records
from .hf_backend import load_adapter_model, load_tokenizer, set_seed
from .numerics import bootstrap_interval
from .results import RunDirectory
from .rewards import exact_numeric_reward, extract_final_numeric_answer, final_numeric_reward
from .train_grpo import train


REWARD_SCORING = "explicit-final-line-v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def training_curve(run: RunDirectory):
    state = read_json(run.root / "checkpoints" / "trainer_state.json")
    logs = [
        row for row in state["log_history"] if "frac_reward_zero_std" in row
    ]
    if not logs:
        raise RuntimeError("Trainer did not log frac_reward_zero_std. Check the installed TRL version.")
    rows = [
        {
            "step": row["step"],
            "reward": row["reward"],
            "mixed_group_fraction": 1 - row["frac_reward_zero_std"],
            "clipped_fraction": row.get("completions/clipped_ratio"),
            "final_answer_rate": row.get("parsed_answer_rate"),
        }
        for row in logs
    ]
    return state["global_step"], rows


def evaluate_curve(cfg, run: RunDirectory, *, pilot: bool, training_seconds=None, pilot_baseline=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_step, training = training_curve(run)
    checkpoint_root = run.root / "checkpoints"
    policies = [(0, checkpoint_root / "final")]
    if not pilot and pilot_baseline is None:
        policies.extend(sorted(
            (int(path.name.removeprefix("checkpoint-")), path)
            for path in checkpoint_root.glob("checkpoint-*")
            if path.name.removeprefix("checkpoint-").isdigit()
            and int(path.name.removeprefix("checkpoint-")) < final_step
            and (path / "adapter_config.json").is_file()
        ))
    policies.append((final_step, checkpoint_root / "final"))

    examples = build_dataset(cfg.data, cfg.experiment.seed, final_answer_line=True, task_only=True).task_test
    tokenizer = load_tokenizer(cfg)
    points = []
    base = None
    for step, checkpoint in policies:
        print(f"Held-out GSM8K: step {step}, {len(examples)} examples", flush=True)
        if step == 0 and pilot_baseline is not None:
            records = read_json(pilot_baseline)["records"]
            for record, example in zip(records, examples, strict=True):
                if (record["example_id"], record["prompt"], record["answer"]) != (
                    example.example_id, example.prompt, example.answer
                ):
                    raise ValueError("Pilot baseline and held-out examples differ")
        else:
            set_seed(cfg.experiment.seed)
            model = load_adapter_model(cfg, checkpoint)
            records = accuracy_records(
                model, tokenizer, examples, cfg, adapter=step != 0, score=final_numeric_reward
            )
            del model
        for record in records:
            completion = record["completion"]
            record["has_final_answer"] = extract_final_numeric_answer(completion) is not None
            record["legacy_correct"] = exact_numeric_reward(completion, record["answer"])
        if base is None:
            base = records
        differences = []
        for record, reference in zip(records, base, strict=True):
            if record["example_id"] != reference["example_id"]:
                raise ValueError("Held-out example order changed between checkpoints")
            differences.append(record["correct"] - reference["correct"])
        point = {
            "step": step,
            "policy": "untrained" if step == 0 else checkpoint.name,
            "accuracy": mean(row["correct"] for row in records),
            "legacy_accuracy": mean(row["legacy_correct"] for row in records),
            "final_answer_rate": mean(row["has_final_answer"] for row in records),
            "delta_from_base": mean(differences),
            "paired_delta_ci95": bootstrap_interval(
                differences, samples=cfg.evaluation.bootstrap_samples, seed=cfg.experiment.seed
            ),
        }
        points.append(point)
        run.write_json(f"artifacts/heldout_step_{step}.json", {"summary": point, "records": records})

    mean_mixed = mean(row["mixed_group_fraction"] for row in training)
    clipped = [row["clipped_fraction"] for row in training if row["clipped_fraction"] is not None]
    mean_clipped = mean(clipped) if len(clipped) == len(training) else None
    final_answers = [row["final_answer_rate"] for row in training if row["final_answer_rate"] is not None]
    mean_final_answer_rate = mean(final_answers) if len(final_answers) == len(training) else None
    # Operational pilot gate. Accuracy on a 12-step pilot is diagnostic only.
    passes = mean_mixed >= 0.15 and mean_clipped is not None and mean_clipped <= 0.25
    report = {
        "config_fingerprint": cfg.fingerprint(),
        "reward_scoring": REWARD_SCORING,
        "pilot": pilot,
        "final_step": final_step,
        "training_seconds": training_seconds,
        "heldout_examples": len(examples),
        "reused_pilot_baseline": str(pilot_baseline) if pilot_baseline is not None else None,
        "mean_mixed_group_fraction": mean_mixed,
        "mean_clipped_fraction": mean_clipped,
        "mean_training_final_answer_rate": mean_final_answer_rate,
        "passes_rollout_gate": passes,
        "training": training,
        "heldout": points,
    }
    run.write_json("artifacts/grpo_learning_curve.json", report)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    axes[0].plot([point["step"] for point in points], [point["accuracy"] for point in points], "o-")
    axes[0].axhline(points[0]["accuracy"], linestyle="--", color="gray")
    axes[0].set_ylabel("Held-out GSM8K accuracy")
    for field, label in (
        ("reward", "Mean training reward"),
        ("mixed_group_fraction", "Mixed-reward groups"),
        ("clipped_fraction", "Length-clipped completions"),
        ("final_answer_rate", "Recognized final answers"),
    ):
        values = [row for row in training if row[field] is not None]
        if values:
            # Average up to 32 logged steps to make the full-run trend legible.
            windows = [values[i:i + 32] for i in range(0, len(values), 32)] if not pilot else [
                [row] for row in values
            ]
            axes[1].plot(
                [mean(row["step"] for row in window) for window in windows],
                [mean(row[field] for row in window) for window in windows],
                label=label,
            )
    axes[1].legend()
    axes[1].set(xlabel="Optimizer step", ylabel="Fraction", ylim=(-0.03, 1.03))
    fig.savefig(run.root / "plots" / "grpo_learning_curve.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in report.items() if key != "training"}, indent=2))
    print(f"Run directory: {run.root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gsm8k_qwen15b_grpo.yaml")
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--pilot", action="store_true")
    phase.add_argument("--pilot-run", type=Path, help="Passed pilot directory for full training")
    phase.add_argument("--evaluate-run", type=Path, help="Evaluate an existing completed run")
    parser.add_argument("--max-steps", type=int, help="Cap a follow-up run and evaluate only its endpoint")
    args = parser.parse_args()
    if args.max_steps is not None and (not args.pilot_run or args.max_steps < 1):
        parser.error("--max-steps requires --pilot-run and a positive step count.")
    cfg = load_config(args.config)

    import torch

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        parser.error("Expose one CUDA GPU with BF16 support using CUDA_VISIBLE_DEVICES.")
    pilot_cfg = load_config(args.config, {"experiment.name": cfg.experiment.name + "-pilot"})
    if args.evaluate_run:
        run = RunDirectory(args.evaluate_run.resolve())
        metadata = read_json(run.root / "artifacts" / "learning_run.json")
        if (
            metadata["config_fingerprint"] != cfg.fingerprint()
            or metadata.get("reward_scoring") != REWARD_SCORING
        ):
            parser.error("Evaluation config or reward scorer differs from the completed run.")
        if read_json(run.root / "status.json")["status"] != "completed":
            parser.error("Evaluation requires a completed run.")
        baseline = metadata.get("pilot_baseline")
        evaluate_curve(
            cfg, run, pilot=metadata["pilot"], training_seconds=metadata["training_seconds"],
            pilot_baseline=Path(baseline) if baseline else None,
        )
        return
    if args.pilot_run:
        pilot = read_json(args.pilot_run / "artifacts" / "grpo_learning_curve.json")
        if (
            not pilot["pilot"]
            or not pilot["passes_rollout_gate"]
            or pilot["config_fingerprint"] != cfg.fingerprint()
            or pilot.get("reward_scoring") != REWARD_SCORING
        ):
            parser.error("Full training requires a passing pilot with the same configuration.")

    pilot_baseline = None
    run_cfg = pilot_cfg if args.pilot else cfg
    stage = "pilot" if args.pilot else "full"
    if args.max_steps is not None:
        stage = f"{args.max_steps}-steps"
        pilot_baseline = args.pilot_run.resolve() / "artifacts" / "heldout_step_0.json"
        if not pilot_baseline.is_file():
            parser.error("A capped follow-up requires the passing pilot's step-zero evaluation.")
        run_cfg = load_config(args.config, {"experiment.name": f"{cfg.experiment.name}-{stage}"})
    os.environ.setdefault("WANDB_RUN_GROUP", cfg.experiment.name)
    os.environ.setdefault("WANDB_TAGS", "gsm8k,qwen2.5-1.5b,grpo," + stage)
    # Use a separate output namespace so pilot checkpoints cannot be mistaken for the full run.
    started = time.monotonic()
    run = train(
        run_cfg,
        max_steps=12 if args.pilot else (args.max_steps or -1),
        strict_final_answer=True,
        task_only=True,
    )
    seconds = time.monotonic() - started
    run.write_json("artifacts/learning_run.json", {
        "pilot": args.pilot,
        "config_fingerprint": cfg.fingerprint(),
        "reward_scoring": REWARD_SCORING,
        "training_seconds": seconds,
        "max_steps": args.max_steps,
        "pilot_baseline": str(pilot_baseline) if pilot_baseline is not None else None,
    })
    print(f"Training completed: {run.root} ({seconds:.0f} seconds)", flush=True)
    evaluate_curve(cfg, run, pilot=args.pilot, training_seconds=seconds, pilot_baseline=pilot_baseline)


if __name__ == "__main__":
    main()
