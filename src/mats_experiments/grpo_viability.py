from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from .config import load_config
from .data import build_dataset
from .hf_backend import (
    encode_generation_prompt,
    generation_stop_token_ids,
    load_adapter_model,
    model_init_kwargs,
    load_tokenizer,
    require_training_stack,
    set_seed,
)
from .rewards import exact_numeric_reward, extract_numeric_answer


def summarize_groups(
    groups: list[dict[str, Any]],
    *,
    min_parsed_rate: float,
    min_mixed_group_fraction: float,
    max_all_zero_fraction: float,
    max_all_one_fraction: float,
    max_truncated_fraction: float,
) -> dict[str, Any]:
    completions = [row for group in groups for row in group["generations"]]
    group_reward_sums = [sum(row["reward"] for row in group["generations"]) for group in groups]
    group_sizes = [len(group["generations"]) for group in groups]
    mixed = [0 < total < size for total, size in zip(group_reward_sums, group_sizes)]
    all_zero = [total == 0 for total in group_reward_sums]
    all_one = [total == size for total, size in zip(group_reward_sums, group_sizes)]
    parsed_rate = statistics.mean(float(row["parsed"]) for row in completions)
    reward_mean = statistics.mean(float(row["reward"]) for row in completions)
    truncated_fraction = statistics.mean(float(row["truncated"]) for row in completions)
    mixed_fraction = statistics.mean(float(value) for value in mixed)
    all_zero_fraction = statistics.mean(float(value) for value in all_zero)
    all_one_fraction = statistics.mean(float(value) for value in all_one)
    reward_stds = [
        statistics.pstdev(float(row["reward"]) for row in group["generations"])
        for group in groups
    ]
    checks = {
        "parsed_rate": parsed_rate >= min_parsed_rate,
        "mixed_group_fraction": mixed_fraction >= min_mixed_group_fraction,
        "all_zero_group_fraction": all_zero_fraction <= max_all_zero_fraction,
        "all_one_group_fraction": all_one_fraction <= max_all_one_fraction,
        "truncated_fraction": truncated_fraction <= max_truncated_fraction,
    }
    return {
        "suitable_for_grpo": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_parsed_rate": min_parsed_rate,
            "min_mixed_group_fraction": min_mixed_group_fraction,
            "max_all_zero_group_fraction": max_all_zero_fraction,
            "max_all_one_group_fraction": max_all_one_fraction,
            "max_truncated_fraction": max_truncated_fraction,
        },
        "metrics": {
            "prompt_groups": len(groups),
            "generations": len(completions),
            "generations_per_group": group_sizes[0] if len(set(group_sizes)) == 1 else group_sizes,
            "parsed_rate": parsed_rate,
            "reward_mean": reward_mean,
            "pass_at_group": statistics.mean(float(total > 0) for total in group_reward_sums),
            "mixed_group_fraction": mixed_fraction,
            "zero_reward_std_group_fraction": all_zero_fraction + all_one_fraction,
            "all_zero_group_fraction": all_zero_fraction,
            "all_one_group_fraction": all_one_fraction,
            "mean_within_group_reward_std": statistics.mean(reward_stds),
            "truncated_fraction": truncated_fraction,
            "mean_completion_tokens": statistics.mean(
                float(row["completion_tokens"]) for row in completions
            ),
        },
    }


def _load_policy(cfg, checkpoint: str | None):
    if checkpoint:
        return load_adapter_model(cfg, checkpoint)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        device_map="auto",
        **model_init_kwargs(cfg),
    )
    model.eval()
    return model


def collect_groups(cfg, checkpoint: str | None, prompt_count: int) -> list[dict[str, Any]]:
    require_training_stack()
    import torch

    model = _load_policy(cfg, checkpoint)
    tokenizer = load_tokenizer(cfg, padding_side="left")
    stop_token_ids = generation_stop_token_ids(model, tokenizer)
    device = next(model.parameters()).device
    examples = build_dataset(cfg.data, cfg.experiment.seed).train[:prompt_count]
    groups: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        encoded = encode_generation_prompt(
            tokenizer,
            example.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.training.max_length,
        ).to(device)
        set_seed(cfg.experiment.seed * 1_000_000 + index)
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                do_sample=True,
                temperature=cfg.evaluation.temperature,
                max_new_tokens=cfg.training.max_completion_length,
                num_return_sequences=cfg.training.num_generations,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_tokens = encoded["input_ids"].shape[1]
        generations = []
        for output in outputs:
            generated = output[prompt_tokens:]
            stop_positions = [
                position
                for position, token_id in enumerate(generated.tolist())
                if token_id in stop_token_ids
            ]
            ended_with_eos = bool(stop_positions)
            completion_length = stop_positions[0] + 1 if stop_positions else len(generated)
            completion_tokens = generated[:completion_length]
            text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            generations.append(
                {
                    "text": text,
                    "predicted_answer": extract_numeric_answer(text),
                    "parsed": extract_numeric_answer(text) is not None,
                    "reward": exact_numeric_reward(text, example.answer),
                    "completion_tokens": int(completion_length),
                    "truncated": bool(
                        completion_length >= cfg.training.max_completion_length
                        and not ended_with_eos
                    ),
                }
            )
        groups.append(
            {
                "example_id": example.example_id,
                "prompt": example.prompt,
                "answer": example.answer,
                "generations": generations,
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == len(examples):
            print(f"Generated {index + 1}/{len(examples)} prompt groups", flush=True)
    if not groups:
        raise ValueError("The viability prompt set is empty")
    return groups


def _log_wandb(cfg, policy_label: str, summary: dict[str, Any], groups) -> None:
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return
    import wandb

    run = wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY") or None,
        group=os.environ.get("WANDB_RUN_GROUP") or "gsm8k-grpo-viability",
        name=f"{cfg.experiment.name}-{policy_label}-rollout-viability",
        job_type="dataset-viability",
        config={**cfg.as_dict(), "policy_label": policy_label, **summary["thresholds"]},
    )
    run.log({f"viability/{key}": value for key, value in summary["metrics"].items() if isinstance(value, (int, float))})
    table = wandb.Table(
        columns=[
            "example_id",
            "generation_index",
            "answer",
            "predicted_answer",
            "reward",
            "parsed",
            "completion_tokens",
            "truncated",
            "completion",
        ]
    )
    for group in groups:
        for generation_index, row in enumerate(group["generations"]):
            table.add_data(
                group["example_id"],
                generation_index,
                group["answer"],
                row["predicted_answer"],
                row["reward"],
                row["parsed"],
                row["completion_tokens"],
                row["truncated"],
                row["text"],
            )
    run.log({"viability/completions": table})
    run.summary["suitable_for_grpo"] = summary["suitable_for_grpo"]
    for key, value in summary["checks"].items():
        run.summary[f"gate/{key}"] = value
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a task supplies usable GRPO groups")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--completions-output")
    parser.add_argument("--policy-label", default="base")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--min-parsed-rate", type=float, default=0.80)
    parser.add_argument("--min-mixed-group-fraction", type=float, default=0.15)
    parser.add_argument("--max-all-zero-fraction", type=float, default=0.80)
    parser.add_argument("--max-all-one-fraction", type=float, default=0.80)
    parser.add_argument("--max-truncated-fraction", type=float, default=0.25)
    args = parser.parse_args()
    cfg = load_config(args.config)
    groups = collect_groups(cfg, args.checkpoint, args.prompts)
    summary = summarize_groups(
        groups,
        min_parsed_rate=args.min_parsed_rate,
        min_mixed_group_fraction=args.min_mixed_group_fraction,
        max_all_zero_fraction=args.max_all_zero_fraction,
        max_all_one_fraction=args.max_all_one_fraction,
        max_truncated_fraction=args.max_truncated_fraction,
    )
    summary.update(
        {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "policy_label": args.policy_label,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    completions_output = Path(args.completions_output) if args.completions_output else output.with_suffix(".completions.jsonl")
    with completions_output.open("w", encoding="utf-8") as stream:
        for group in groups:
            stream.write(json.dumps(group) + "\n")
    _log_wandb(cfg, args.policy_label, summary, groups)
    print(json.dumps(summary, indent=2))
    if not args.diagnostic_only and not summary["suitable_for_grpo"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
