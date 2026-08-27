from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .data import build_dataset
from .hf_backend import (
    adapter_enabled,
    encode_generation_prompts,
    load_adapter_model,
    load_tokenizer,
    require_training_stack,
)


def _device(model):
    return next(model.parameters()).device


def _mean_selected_tokens(hidden, attention_mask, positions: tuple[int, ...]):
    import torch

    rows = []
    for row in range(hidden.shape[0]):
        token_indices = torch.nonzero(attention_mask[row], as_tuple=False).flatten()
        valid = [position for position in positions if -len(token_indices) <= position < len(token_indices)]
        if not valid:
            valid = [-1]
        selected = torch.stack([token_indices[position] for position in valid])
        rows.append(hidden[row].index_select(0, selected).mean(dim=0))
    return torch.stack(rows)


def extract_trace_statistics(cfg, checkpoint: str | Path, output: str | Path) -> dict[str, Any]:
    require_training_stack()
    import torch
    import torch.nn.functional as functional

    trace_cfg = dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, dtype=cfg.traces.dtype)
    )
    model = load_adapter_model(trace_cfg, checkpoint)
    tokenizer = load_tokenizer(cfg, padding_side="right")
    probes = build_dataset(cfg.data, cfg.experiment.seed).probe
    per_layer_differences: list[list[Any]] | None = None
    per_layer_base: list[list[Any]] | None = None

    batch_size = cfg.evaluation.batch_size
    if batch_size < 1:
        raise ValueError("Trace extraction batch size must be positive")
    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        encoded = encode_generation_prompts(
            tokenizer,
            [example.prompt for example in batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=cfg.training.max_length,
        ).to(_device(model))
        with torch.no_grad(), adapter_enabled(model, False):
            base = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states
        with torch.no_grad(), adapter_enabled(model, True):
            finetuned = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states
        # hidden_states[0] is the embedding output; layer i maps to hidden_states[i + 1].
        if per_layer_differences is None:
            per_layer_differences = [[] for _ in range(len(base) - 1)]
            per_layer_base = [[] for _ in range(len(base) - 1)]
        for layer in range(len(base) - 1):
            base_vector = _mean_selected_tokens(
                base[layer + 1], encoded["attention_mask"], cfg.traces.token_positions
            )
            ft_vector = _mean_selected_tokens(
                finetuned[layer + 1], encoded["attention_mask"], cfg.traces.token_positions
            )
            per_layer_base[layer].extend(base_vector.detach().float().cpu().unbind())
            per_layer_differences[layer].extend(
                (ft_vector - base_vector).detach().float().cpu().unbind()
            )

    if per_layer_differences is None or per_layer_base is None:
        raise ValueError("Probe dataset is empty")

    boundary = max(1, min(len(probes) - 1, int(len(probes) * cfg.traces.discovery_fraction)))
    payload: dict[str, Any] = {"layers": {}, "directions": {}, "base_means": {}}
    for layer, rows in enumerate(per_layer_differences):
        matrix = torch.stack(rows)
        base_matrix = torch.stack(per_layer_base[layer])
        discovery, confirmation = matrix[:boundary], matrix[boundary:]
        direction = discovery.mean(dim=0)
        denominator = discovery.pow(2).sum(dim=-1).mean()
        rho_discovery = direction.pow(2).sum() / denominator if denominator > 0 else torch.tensor(0.0)
        confirmation_mean = confirmation.mean(dim=0)
        confirmation_denominator = confirmation.pow(2).sum(dim=-1).mean()
        rho_confirmation = (
            confirmation_mean.pow(2).sum() / confirmation_denominator
            if confirmation_denominator > 0
            else torch.tensor(0.0)
        )
        split_cosine = functional.cosine_similarity(
            direction.unsqueeze(0), confirmation_mean.unsqueeze(0), dim=-1
        ).item()
        normalized_rows = functional.normalize(confirmation, dim=-1)
        gram = normalized_rows @ normalized_rows.T
        count = gram.shape[0]
        pairwise = (
            (gram.sum() - gram.diag().sum()) / (count * (count - 1)) if count > 1 else torch.tensor(0.0)
        )
        payload["directions"][layer] = direction
        payload["base_means"][layer] = base_matrix[:boundary].mean(dim=0)
        payload["layers"][str(layer)] = {
            "rho_discovery": float(rho_discovery.item()),
            "rho_confirmation": float(rho_confirmation.item()),
            "split_half_cosine": float(split_cosine),
            "mean_pairwise_cosine_confirmation": float(pairwise.item()),
            "direction_norm": float(torch.linalg.vector_norm(direction).item()),
            "n_discovery": len(discovery),
            "n_confirmation": len(confirmation),
        }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "directions": payload.pop("directions"),
            "base_means": payload.pop("base_means"),
            "metadata": {
                "kind": "model_minus_base_trace",
                "checkpoint": str(checkpoint),
                "token_positions": cfg.traces.token_positions,
                "dtype": cfg.traces.dtype,
                "discovery_examples": boundary,
            },
        },
        output,
    )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"artifact": str(output), "summary": str(summary_path), **payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract held-out global activation traces")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = extract_trace_statistics(load_config(args.config), args.checkpoint, args.output)
    print(json.dumps({"artifact": result["artifact"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
