from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .config import load_config
from .data import Example, build_dataset
from .hf_backend import (
    adapter_enabled,
    encode_generation_prompt,
    load_adapter_model,
    load_tokenizer,
    require_training_stack,
)
from .numerics import bootstrap_interval
from .rewards import exact_numeric_reward


def _model_device(model):
    return next(model.parameters()).device


def generate_completion(model, tokenizer, prompt: str, cfg, *, adapter: bool, sample: bool) -> str:
    import torch

    encoded = encode_generation_prompt(tokenizer, prompt, return_tensors="pt").to(_model_device(model))
    generation_kwargs = {
        "max_new_tokens": cfg.evaluation.max_new_tokens,
        "do_sample": sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if sample:
        generation_kwargs["temperature"] = cfg.evaluation.temperature
    with torch.no_grad(), adapter_enabled(model, adapter):
        output = model.generate(**encoded, **generation_kwargs)
    generated = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def accuracy_records(model, tokenizer, examples: Iterable[Example], cfg, *, adapter: bool):
    records: list[dict[str, Any]] = []
    for example in examples:
        completion = generate_completion(model, tokenizer, example.prompt, cfg, adapter=adapter, sample=False)
        reward = exact_numeric_reward(completion, example.answer)
        records.append(
            {
                "example_id": example.example_id,
                "prompt": example.prompt,
                "completion": completion,
                "answer": example.answer,
                "correct": reward,
            }
        )
    return records


def _trajectory_kl(model, tokenizer, prompt: str, cfg, source_adapter: bool, direction: str) -> float:
    """Token-mean KL on trajectories sampled from the first distribution in the KL."""
    import torch
    import torch.nn.functional as functional

    encoded = encode_generation_prompt(tokenizer, prompt, return_tensors="pt").to(_model_device(model))
    with torch.no_grad(), adapter_enabled(model, source_adapter):
        sequence = model.generate(
            **encoded,
            max_new_tokens=cfg.training.max_completion_length,
            do_sample=True,
            temperature=cfg.evaluation.temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    outputs: dict[bool, Any] = {}
    for enabled in (False, True):
        with torch.no_grad(), adapter_enabled(model, enabled):
            outputs[enabled] = model(input_ids=sequence, attention_mask=torch.ones_like(sequence)).logits
    start = encoded["input_ids"].shape[1] - 1
    stop = sequence.shape[1] - 1
    if stop <= start:
        return 0.0
    base_logp = functional.log_softmax(outputs[False][:, start:stop, :].float(), dim=-1)
    ft_logp = functional.log_softmax(outputs[True][:, start:stop, :].float(), dim=-1)
    if direction == "forward":
        source_logp, target_logp = base_logp, ft_logp
    elif direction == "reverse":
        source_logp, target_logp = ft_logp, base_logp
    else:
        raise ValueError(f"Unknown KL direction: {direction}")
    source_prob = source_logp.exp()
    kl = (source_prob * (source_logp - target_logp)).sum(dim=-1)
    return float(kl.mean().item())


def evaluate_checkpoint(cfg, checkpoint: str | Path) -> dict[str, Any]:
    require_training_stack()
    model = load_adapter_model(cfg, checkpoint)
    tokenizer = load_tokenizer(cfg, padding_side="left")
    bundle = build_dataset(cfg.data, cfg.experiment.seed)
    task_records = accuracy_records(model, tokenizer, bundle.task_test, cfg, adapter=True)
    old_records = accuracy_records(model, tokenizer, bundle.old, cfg, adapter=True)
    base_old_records = accuracy_records(model, tokenizer, bundle.old, cfg, adapter=False)
    kl_prompts = bundle.task_test[: cfg.evaluation.kl_samples]
    forward = [
        _trajectory_kl(model, tokenizer, example.prompt, cfg, source_adapter=False, direction="forward")
        for example in kl_prompts
    ]
    reverse = [
        _trajectory_kl(model, tokenizer, example.prompt, cfg, source_adapter=True, direction="reverse")
        for example in kl_prompts
    ]

    def summarize(records):
        values = [float(record["correct"]) for record in records]
        low, high = bootstrap_interval(
            values,
            samples=cfg.evaluation.bootstrap_samples,
            seed=cfg.experiment.seed,
        )
        return {"mean": sum(values) / len(values), "ci95": [low, high], "n": len(values)}

    return {
        "task_accuracy": summarize(task_records),
        "old_retention": summarize(old_records),
        "base_old_accuracy": summarize(base_old_records),
        "forward_kl": {"mean": sum(forward) / len(forward), "values": forward},
        "reverse_kl": {"mean": sum(reverse) / len(reverse), "values": reverse},
        "task_records": task_records,
        "old_records": old_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate performance, retention, and policy KL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    payload = evaluate_checkpoint(cfg, args.checkpoint)
    output = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **{key: payload[key] for key in payload if key.endswith("kl")}}, indent=2))


if __name__ == "__main__":
    main()
