from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

from .config import load_config
from .data import build_dataset
from .evaluate import accuracy_records, generate_completions
from .hf_backend import (
    adapter_enabled,
    encode_generation_prompt,
    load_adapter_model,
    load_tokenizer,
    require_training_stack,
    set_seed,
)
from .interventions import ResidualIntervention
from .rewards import arithmetic_domain_intrusion


@contextlib.contextmanager
def _policy_state(model, intervention, policy: str):
    with contextlib.ExitStack() as stack:
        if policy == "base":
            stack.enter_context(intervention.disabled())
            stack.enter_context(adapter_enabled(model, False))
        elif policy == "unmodified":
            stack.enter_context(intervention.disabled())
            stack.enter_context(adapter_enabled(model, True))
        elif policy == "intervened":
            stack.enter_context(adapter_enabled(model, True))
        else:
            raise ValueError(f"Unknown policy state: {policy}")
        yield


def _policy_kl(
    model,
    tokenizer,
    intervention,
    prompt,
    cfg,
    source: str,
    target: str,
    sample_seed: int,
) -> float:
    import torch
    import torch.nn.functional as functional

    device = next(model.parameters()).device
    encoded = encode_generation_prompt(tokenizer, prompt, return_tensors="pt").to(device)
    set_seed(sample_seed)
    with torch.no_grad(), _policy_state(model, intervention, source):
        sequence = model.generate(
            **encoded,
            max_new_tokens=cfg.training.max_completion_length,
            do_sample=True,
            temperature=cfg.evaluation.temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    def log_probs(policy):
        with torch.no_grad(), _policy_state(model, intervention, policy):
            logits = model(
                input_ids=sequence,
                attention_mask=torch.ones_like(sequence),
                use_cache=False,
            ).logits
        start = encoded["input_ids"].shape[1] - 1
        stop = sequence.shape[1] - 1
        return functional.log_softmax(logits[:, start:stop].float(), dim=-1)

    source_logp = log_probs(source)
    target_logp = log_probs(target)
    if source_logp.shape[1] == 0:
        return 0.0
    return float((source_logp.exp() * (source_logp - target_logp)).sum(dim=-1).mean().item())


def _sft_to_intervention_kl(
    sft_model,
    intervened_model,
    tokenizer,
    prompt,
    cfg,
    sample_seed: int,
) -> float:
    """MC token-level KL(SFT* || intervened RL*) on an SFT-sampled trajectory."""
    import torch
    import torch.nn.functional as functional

    source_device = next(sft_model.parameters()).device
    target_device = next(intervened_model.parameters()).device
    encoded = encode_generation_prompt(tokenizer, prompt, return_tensors="pt").to(source_device)
    set_seed(sample_seed)
    with torch.no_grad():
        sequence = sft_model.generate(
            **encoded,
            max_new_tokens=cfg.training.max_completion_length,
            do_sample=True,
            temperature=cfg.evaluation.temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
        source_logits = sft_model(
            input_ids=sequence,
            attention_mask=torch.ones_like(sequence),
            use_cache=False,
        ).logits
        target_sequence = sequence.to(target_device)
        target_logits = intervened_model(
            input_ids=target_sequence,
            attention_mask=torch.ones_like(target_sequence),
            use_cache=False,
        ).logits
    start = encoded["input_ids"].shape[1] - 1
    stop = sequence.shape[1] - 1
    if stop <= start:
        return 0.0
    source_logp = functional.log_softmax(source_logits[:, start:stop].float(), dim=-1)
    target_logp = functional.log_softmax(target_logits[:, start:stop].float(), dim=-1)
    if source_logp.device != target_logp.device:
        source_logp = source_logp.to(target_logp.device)
    return float((source_logp.exp() * (source_logp - target_logp)).sum(dim=-1).mean().item())


def run_intervention(
    cfg,
    checkpoint,
    direction_artifact,
    layer,
    operation,
    scale,
    *,
    sft_checkpoint=None,
):
    require_training_stack()
    import torch

    artifact = torch.load(direction_artifact, map_location="cpu", weights_only=True)
    direction = artifact["directions"][layer]
    base_mean = artifact.get("base_means", {}).get(layer)
    model = load_adapter_model(cfg, checkpoint)
    sft_model = load_adapter_model(cfg, sft_checkpoint) if sft_checkpoint else None
    tokenizer = load_tokenizer(cfg, padding_side="left")
    bundle = build_dataset(cfg.data, cfg.experiment.seed)
    intervention = ResidualIntervention(layer, direction, operation, scale, base_mean=base_mean)
    with intervention.install(model):
        task = accuracy_records(model, tokenizer, bundle.task_test, cfg, adapter=True)
        old = accuracy_records(model, tokenizer, bundle.old, cfg, adapter=True)
        intervention_probes = bundle.probe[: cfg.evaluation.intervention_probe_samples]
        probe_completions = generate_completions(
            model,
            tokenizer,
            [example.prompt for example in intervention_probes],
            cfg,
            adapter=True,
            sample=False,
        )
        probes = [
            {
                "example_id": example.example_id,
                "completion": completion,
            }
            for example, completion in zip(intervention_probes, probe_completions, strict=True)
        ]
        kl_examples = bundle.task_test[: cfg.evaluation.kl_samples]
        # These seeds depend only on the prompt index, not direction or alpha. Every E3 cell
        # therefore uses the same SFT-sampled trajectories for the primary estimand.
        sft_to_intervention = (
            [
                _sft_to_intervention_kl(
                    sft_model,
                    model,
                    tokenizer,
                    row.prompt,
                    cfg,
                    cfg.experiment.seed * 1_000_000 + 10_000 + index,
                )
                for index, row in enumerate(kl_examples)
            ]
            if sft_model is not None
            else []
        )
        forward_kl = [
            _policy_kl(
                model,
                tokenizer,
                intervention,
                row.prompt,
                cfg,
                "base",
                "intervened",
                cfg.experiment.seed * 1_000_000 + 20_000 + index,
            )
            for index, row in enumerate(kl_examples)
        ]
        reverse_kl = [
            _policy_kl(
                model,
                tokenizer,
                intervention,
                row.prompt,
                cfg,
                "intervened",
                "base",
                cfg.experiment.seed * 1_000_000 + 30_000 + index,
            )
            for index, row in enumerate(kl_examples)
        ]
        induced_kl = [
            _policy_kl(
                model,
                tokenizer,
                intervention,
                row.prompt,
                cfg,
                "unmodified",
                "intervened",
                cfg.experiment.seed * 1_000_000 + 40_000 + index,
            )
            for index, row in enumerate(kl_examples)
        ]
    intrusion = [arithmetic_domain_intrusion(row["completion"]) for row in probes]
    return {
        "layer": layer,
        "operation": operation,
        "scale": scale,
        "steering_definition": "hidden <- hidden + scale * raw_direction",
        "direction_l2_norm": float(torch.linalg.vector_norm(direction.float()).item()),
        "direction_metadata": artifact.get("metadata", {}),
        "matched_sft_checkpoint": str(sft_checkpoint) if sft_checkpoint else None,
        "task_accuracy": sum(row["correct"] for row in task) / len(task),
        "old_accuracy": sum(row["correct"] for row in old) / len(old),
        "forward_kl": sum(forward_kl) / len(forward_kl),
        "reverse_kl": sum(reverse_kl) / len(reverse_kl),
        "induced_kl": sum(induced_kl) / len(induced_kl),
        "sft_to_intervention_kl": (
            sum(sft_to_intervention) / len(sft_to_intervention)
            if sft_to_intervention
            else None
        ),
        "sft_to_intervention_kl_values": sft_to_intervention,
        "domain_intrusion": sum(intrusion) / len(intrusion),
        "sample_counts": {
            "task": len(task),
            "old": len(old),
            "probe": len(probes),
            "kl": len(kl_examples),
        },
        "task_records": task,
        "old_records": old,
        "probe_generations": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an inference-time residual intervention")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--direction-artifact", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--operation", choices=("add", "subtract_mean", "restore_base_mean"), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_intervention(
        cfg,
        args.checkpoint,
        args.direction_artifact,
        args.layer,
        args.operation,
        args.scale,
        sft_checkpoint=args.sft_checkpoint,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "task_accuracy",
                    "old_accuracy",
                    "forward_kl",
                    "reverse_kl",
                    "induced_kl",
                    "sft_to_intervention_kl",
                    "domain_intrusion",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
