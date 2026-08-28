from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_config
from .data import build_dataset
from .hf_backend import (
    adapter_enabled,
    encode_generation_prompts,
    generation_stop_token_ids,
    load_adapter_model,
    load_tokenizer,
    require_training_stack,
    set_seed,
)
from .interventions import PairedBaseRestoration
from .rewards import arithmetic_domain_intrusion, exact_numeric_reward


@dataclass(frozen=True)
class E4Cell:
    label: str
    artifact: str
    beta: float


def build_e4_cells(semantic_artifact, random_artifacts) -> list[E4Cell]:
    """Return the frozen 17-cell necessity matrix; zero is shared by orientation controls."""
    cells = [
        E4Cell("semantic", str(semantic_artifact), beta)
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    for label, artifact in random_artifacts:
        cells.extend(
            E4Cell(f"random_{label}", str(artifact), beta)
            for beta in (0.25, 0.5, 0.75, 1.0)
        )
    return cells


def _safe_beta(beta: float) -> str:
    return f"{beta:g}".replace("-", "neg").replace(".", "p")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _configure_inference() -> None:
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _model_device(model):
    return next(model.parameters()).device


def _trim_at_stop(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    for index, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            return token_ids[: index + 1]
    return token_ids


def _sample_trajectories(
    model,
    tokenizer,
    prompts: list[str],
    cfg,
    *,
    adapter: bool,
    seed_offset: int,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Sample once in GPU batches, then store unpadded sequences for every E4 cell."""
    import torch

    records: list[dict[str, Any]] = []
    stop_ids = generation_stop_token_ids(model, tokenizer)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": cfg.evaluation.temperature,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    with torch.inference_mode(), adapter_enabled(model, adapter):
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = encode_generation_prompts(
                tokenizer,
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.training.max_length,
            ).to(_model_device(model))
            # A batch-indexed seed is deterministic while retaining batched A100 generation.
            set_seed(cfg.experiment.seed * 1_000_000 + seed_offset + start)
            output = model.generate(**encoded, **generation_kwargs)
            prompt_width = encoded["input_ids"].shape[1]
            for row in range(len(batch)):
                prompt_ids = encoded["input_ids"][row][encoded["attention_mask"][row].bool()].tolist()
                completion_ids = _trim_at_stop(output[row, prompt_width:].tolist(), stop_ids)
                records.append(
                    {
                        "input_ids": prompt_ids + completion_ids,
                        "completion_start": len(prompt_ids),
                    }
                )
    return records


def _load_or_sample_trajectories(
    path: Path,
    model,
    tokenizer,
    prompts: list[str],
    cfg,
    *,
    adapter: bool,
    policy_label: str,
    seed_offset: int,
    batch_size: int,
    max_new_tokens: int,
):
    import torch

    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("policy_label") != policy_label or len(payload.get("records", [])) != len(prompts):
            raise ValueError(f"Stale or incompatible trajectory cache: {path}")
        print(f"[E4] reusing {policy_label} trajectory cache {path}", flush=True)
        return payload["records"]
    records = _sample_trajectories(
        model,
        tokenizer,
        prompts,
        cfg,
        adapter=adapter,
        seed_offset=seed_offset,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "policy_label": policy_label,
            "seed": cfg.experiment.seed,
            "prompt_count": len(prompts),
            "max_new_tokens": max_new_tokens,
            "records": records,
        },
        temporary,
    )
    os.replace(temporary, path)
    return records


def _collate_trajectories(records, pad_token_id: int, device):
    import torch

    width = max(len(row["input_ids"]) for row in records)
    input_ids = torch.full((len(records), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    prediction_mask = torch.zeros((len(records), width - 1), dtype=torch.bool, device=device)
    for index, row in enumerate(records):
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[index, : len(ids)] = ids
        attention_mask[index, : len(ids)] = 1
        # Logit position j predicts token j+1. Include generated completion tokens only.
        start = max(0, int(row["completion_start"]) - 1)
        stop = max(start, len(ids) - 1)
        prediction_mask[index, start:stop] = True
    return input_ids, attention_mask, prediction_mask


def _sequence_kl(source_logp, target_logits, prediction_mask, token_chunk_size: int = 64):
    import torch
    import torch.nn.functional as functional

    # Full-vocabulary float32 log-softmax over an entire 768-token batch can transiently occupy
    # most of a 40 GB card. Chunking only the token dimension is mathematically identical and
    # keeps enough memory free to batch several intervention cells through the model together.
    token_values = []
    target_logits = target_logits[:, :-1]
    for start in range(0, target_logits.shape[1], token_chunk_size):
        stop = min(target_logits.shape[1], start + token_chunk_size)
        source_chunk = source_logp[:, start:stop]
        target_chunk = functional.log_softmax(target_logits[:, start:stop].float(), dim=-1)
        token_values.append(
            (source_chunk.exp() * (source_chunk - target_chunk)).sum(dim=-1)
        )
    token_kl = torch.cat(token_values, dim=1)
    mask = prediction_mask.to(token_kl.dtype)
    denominator = mask.sum(dim=-1).clamp_min(1)
    return (token_kl * mask).sum(dim=-1) / denominator


def _teacher_forced_grid(
    trajectories,
    *,
    reference_model,
    reference_adapter: bool,
    paired_model,
    tokenizer,
    restorer,
    directions,
    cells,
    batch_size: int,
    cell_chunk_size: int,
    include_induced_kl: bool,
    cache_directory: Path | None = None,
):
    """Evaluate all cells while reusing source logits and exact paired base activations."""
    import torch
    import torch.nn.functional as functional

    device = _model_device(paired_model)
    values = [[] for _ in cells]
    induced = [[] for _ in cells] if include_induced_kl else None
    with torch.inference_mode():
        for start in range(0, len(trajectories), batch_size):
            records = trajectories[start : start + batch_size]
            cache_path = cache_directory / f"batch_{start:05d}.json" if cache_directory else None
            if cache_path is not None and cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                signature = [(cell.label, cell.beta) for cell in cells]
                if (
                    cached.get("cell_signature") != [list(item) for item in signature]
                    or cached.get("record_count") != len(records)
                ):
                    raise ValueError(f"Stale E4 KL batch cache: {cache_path}")
                for cell_index, batch_values in enumerate(cached["values"]):
                    values[cell_index].extend(batch_values)
                if induced is not None:
                    for cell_index, batch_values in enumerate(cached["induced"]):
                        induced[cell_index].extend(batch_values)
                print(f"[E4] reusing KL batch cache {cache_path}", flush=True)
                continue
            batch_values = [[] for _ in cells]
            batch_induced = [[] for _ in cells] if include_induced_kl else None
            input_ids, attention_mask, prediction_mask = _collate_trajectories(
                records, tokenizer.pad_token_id, device
            )
            with restorer.capture_base(), adapter_enabled(paired_model, False):
                base_output = paired_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            base_hidden = restorer.base_hidden
            if reference_model is paired_model and not reference_adapter:
                source_logits = base_output.logits
            else:
                with adapter_enabled(reference_model, reference_adapter):
                    source_logits = reference_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
            source_logp = functional.log_softmax(source_logits[:, :-1].float(), dim=-1)
            del source_logits, base_output

            baseline_logp = None
            if include_induced_kl:
                with adapter_enabled(paired_model, True):
                    baseline_logits = paired_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
                baseline_logp = functional.log_softmax(baseline_logits[:, :-1].float(), dim=-1)
                del baseline_logits

            batch_count = input_ids.shape[0]
            for cell_start in range(0, len(cells), cell_chunk_size):
                cell_stop = min(len(cells), cell_start + cell_chunk_size)
                chunk_directions = torch.stack(directions[cell_start:cell_stop]).to(device)
                chunk_betas = torch.tensor(
                    [cell.beta for cell in cells[cell_start:cell_stop]],
                    device=device,
                    dtype=base_hidden.dtype,
                )
                chunk_count = cell_stop - cell_start
                expanded_ids = input_ids.repeat(chunk_count, 1)
                expanded_attention = attention_mask.repeat(chunk_count, 1)
                expanded_base = base_hidden.repeat(chunk_count, 1, 1)
                expanded_directions = chunk_directions.repeat_interleave(batch_count, dim=0)
                expanded_betas = chunk_betas.repeat_interleave(batch_count)
                with restorer.restore(expanded_base, expanded_directions, expanded_betas), adapter_enabled(
                    paired_model, True
                ):
                    target_logits = paired_model(
                        input_ids=expanded_ids,
                        attention_mask=expanded_attention,
                        use_cache=False,
                    ).logits
                target_logits = target_logits.reshape(
                    chunk_count, batch_count, target_logits.shape[1], target_logits.shape[2]
                )
                for local_index in range(chunk_count):
                    cell_index = cell_start + local_index
                    cell_values = _sequence_kl(
                        source_logp,
                        target_logits[local_index],
                        prediction_mask,
                    )
                    cell_values_list = [float(value) for value in cell_values.cpu()]
                    values[cell_index].extend(cell_values_list)
                    batch_values[cell_index].extend(cell_values_list)
                    if induced is not None and baseline_logp is not None:
                        induced_values = _sequence_kl(
                            baseline_logp,
                            target_logits[local_index],
                            prediction_mask,
                        )
                        induced_list = [float(value) for value in induced_values.cpu()]
                        induced[cell_index].extend(induced_list)
                        batch_induced[cell_index].extend(induced_list)
                del target_logits
            if cache_path is not None:
                _atomic_json(
                    cache_path,
                    {
                        "cell_signature": [[cell.label, cell.beta] for cell in cells],
                        "record_count": len(records),
                        "values": batch_values,
                        "induced": batch_induced,
                    },
                )
    return values, induced


def _paired_greedy_generate(
    model,
    tokenizer,
    prompts: list[str],
    directions,
    betas,
    cfg,
    restorer,
    *,
    max_new_tokens: int,
) -> list[str]:
    """Greedy decoding with synchronized base/SFT KV caches on every evolving prefix."""
    import torch

    device = _model_device(model)
    encoded = encode_generation_prompts(
        tokenizer,
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.training.max_length,
    ).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    directions = directions.to(device)
    betas = betas.to(device)
    stop_ids = generation_stop_token_ids(model, tokenizer)
    stop_tensor = torch.tensor(sorted(stop_ids), dtype=torch.long, device=device)
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in prompts]

    with torch.inference_mode():
        with restorer.capture_base(), adapter_enabled(model, False):
            base_output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        base_hidden = restorer.base_hidden
        base_past = base_output.past_key_values
        del base_output
        with restorer.restore(base_hidden, directions, betas), adapter_enabled(model, True):
            sft_output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = sft_output.logits[:, -1]
        sft_past = sft_output.past_key_values
        del sft_output

        for step in range(max_new_tokens):
            was_active = ~finished
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(was_active, next_token, torch.full_like(next_token, tokenizer.pad_token_id))
            for row, active in enumerate(was_active.tolist()):
                if active:
                    generated[row].append(int(next_token[row].item()))
            finished = finished | (was_active & torch.isin(next_token, stop_tensor))
            if bool(finished.all()) or step + 1 == max_new_tokens:
                break

            attention_mask = torch.cat(
                [attention_mask, was_active.to(attention_mask.dtype)[:, None]], dim=1
            )
            step_ids = next_token[:, None]
            with restorer.capture_base(), adapter_enabled(model, False):
                base_output = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=base_past,
                    use_cache=True,
                )
            base_hidden = restorer.base_hidden
            base_past = base_output.past_key_values
            del base_output
            with restorer.restore(base_hidden, directions, betas), adapter_enabled(model, True):
                sft_output = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=sft_past,
                    use_cache=True,
                )
            logits = sft_output.logits[:, -1]
            sft_past = sft_output.past_key_values
            del sft_output
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def _generation_grid(
    paired_model,
    tokenizer,
    restorer,
    directions,
    cells,
    categorized_examples,
    cfg,
    *,
    batch_size: int,
    max_new_tokens: int,
    cache_directory: Path | None = None,
):
    import torch

    rows = [
        (cell_index, category, example)
        for cell_index in range(len(cells))
        for category, examples in categorized_examples.items()
        for example in examples
    ]
    results = [dict(task=[], old=[], probe=[]) for _ in cells]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        row_signature = [
            [cells[cell_index].label, cells[cell_index].beta, category, example.example_id]
            for cell_index, category, example in batch
        ]
        cache_path = cache_directory / f"batch_{start:05d}.json" if cache_directory else None
        if cache_path is not None and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("row_signature") != row_signature:
                raise ValueError(f"Stale E4 generation batch cache: {cache_path}")
            completions = cached["completions"]
            print(f"[E4] reusing generation batch cache {cache_path}", flush=True)
        else:
            prompts = [example.prompt for _, _, example in batch]
            batch_directions = torch.stack([directions[cell_index] for cell_index, _, _ in batch])
            batch_betas = torch.tensor(
                [cells[cell_index].beta for cell_index, _, _ in batch], dtype=batch_directions.dtype
            )
            completions = _paired_greedy_generate(
                paired_model,
                tokenizer,
                prompts,
                batch_directions,
                batch_betas,
                cfg,
                restorer,
                max_new_tokens=max_new_tokens,
            )
            if cache_path is not None:
                _atomic_json(cache_path, {"row_signature": row_signature, "completions": completions})
        for (cell_index, category, example), completion in zip(batch, completions, strict=True):
            record = {"example_id": example.example_id, "completion": completion}
            if category in {"task", "old"}:
                record.update(
                    {
                        "prompt": example.prompt,
                        "answer": example.answer,
                        "correct": exact_numeric_reward(completion, example.answer),
                    }
                )
            results[cell_index][category].append(record)
        print(f"[E4] generated {min(start + len(batch), len(rows))}/{len(rows)} rows", flush=True)
    return results


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_e4(
    config_path,
    sft_checkpoint,
    rl_checkpoint,
    semantic_artifact,
    random_artifacts,
    layer: int,
    output_directory,
    *,
    trajectory_batch_size: int,
    kl_batch_size: int,
    cell_chunk_size: int,
    generation_batch_size: int,
    smoke: bool = False,
):
    require_training_stack()
    import torch

    if {label for label, _ in random_artifacts} != {"101", "102", "103"} or len(random_artifacts) != 3:
        raise ValueError("Frozen E4 requires exactly random orientation controls 101, 102, and 103")
    for name, value in (
        ("trajectory_batch_size", trajectory_batch_size),
        ("kl_batch_size", kl_batch_size),
        ("cell_chunk_size", cell_chunk_size),
        ("generation_batch_size", generation_batch_size),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    _configure_inference()
    cfg = load_config(config_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "e4_raw_summary.json").is_file():
        print(f"[E4] completed result already exists at {output}; skipping", flush=True)
        return
    cells = build_e4_cells(semantic_artifact, random_artifacts)
    if smoke:
        cells = cells[:2]

    artifact_cache = {}
    directions = []
    metadata = []
    for cell in cells:
        artifact = artifact_cache.get(cell.artifact)
        if artifact is None:
            artifact = torch.load(cell.artifact, map_location="cpu", weights_only=True)
            artifact_cache[cell.artifact] = artifact
        directions.append(artifact["directions"][layer].float())
        metadata.append(artifact.get("metadata", {}))

    print("[E4] loading matched SFT/base and RL policies", flush=True)
    paired_model = load_adapter_model(cfg, sft_checkpoint)
    rl_model = load_adapter_model(cfg, rl_checkpoint)
    tokenizer = load_tokenizer(cfg, padding_side="left")
    bundle = build_dataset(cfg.data, cfg.experiment.seed)
    kl_examples = list(bundle.task_test[: (2 if smoke else cfg.evaluation.kl_samples)])
    max_new_tokens = 16 if smoke else cfg.training.max_completion_length
    cache = output / "trajectory_cache"
    rl_trajectories = _load_or_sample_trajectories(
        cache / "rl.pt",
        rl_model,
        tokenizer,
        [row.prompt for row in kl_examples],
        cfg,
        adapter=True,
        policy_label="matched_rl",
        seed_offset=50_000,
        batch_size=trajectory_batch_size,
        max_new_tokens=max_new_tokens,
    )
    base_trajectories = _load_or_sample_trajectories(
        cache / "base.pt",
        paired_model,
        tokenizer,
        [row.prompt for row in kl_examples],
        cfg,
        adapter=False,
        policy_label="base",
        seed_offset=60_000,
        batch_size=trajectory_batch_size,
        max_new_tokens=max_new_tokens,
    )

    restorer = PairedBaseRestoration(layer)
    with restorer.install(paired_model):
        print("[E4] primary KL: matched RL || paired-restored SFT", flush=True)
        toward_rl_values, induced_values = _teacher_forced_grid(
            rl_trajectories,
            reference_model=rl_model,
            reference_adapter=True,
            paired_model=paired_model,
            tokenizer=tokenizer,
            restorer=restorer,
            directions=directions,
            cells=cells,
            batch_size=kl_batch_size,
            cell_chunk_size=cell_chunk_size,
            include_induced_kl=True,
            cache_directory=output / "stage_cache" / "primary_kl",
        )
        # Matched RL is no longer needed after the primary reference logits are complete.
        del rl_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[E4] forward KL: base || paired-restored SFT", flush=True)
        forward_values, _ = _teacher_forced_grid(
            base_trajectories,
            reference_model=paired_model,
            reference_adapter=False,
            paired_model=paired_model,
            tokenizer=tokenizer,
            restorer=restorer,
            directions=directions,
            cells=cells,
            batch_size=kl_batch_size,
            cell_chunk_size=cell_chunk_size,
            include_induced_kl=False,
            cache_directory=output / "stage_cache" / "forward_kl",
        )
        categorized = {
            "task": list(bundle.task_test[: (2 if smoke else len(bundle.task_test))]),
            "old": list(bundle.old[: (2 if smoke else len(bundle.old))]),
            "probe": list(bundle.probe[: (2 if smoke else cfg.evaluation.intervention_probe_samples)]),
        }
        print("[E4] exact paired-prefix greedy generation across all cells", flush=True)
        generations = _generation_grid(
            paired_model,
            tokenizer,
            restorer,
            directions,
            cells,
            categorized,
            cfg,
            batch_size=generation_batch_size,
            max_new_tokens=max_new_tokens,
            cache_directory=output / "stage_cache" / "generation",
        )

    if induced_values is None:
        raise RuntimeError("E4 induced-KL diagnostics were not computed")
    if cells[0].beta != 0.0 or _mean(induced_values[0]) > 1e-6:
        raise RuntimeError("E4 beta=0 invariance check failed")

    raw_rows = []
    for index, cell in enumerate(cells):
        task = generations[index]["task"]
        old = generations[index]["old"]
        probes = generations[index]["probe"]
        intrusion = [arithmetic_domain_intrusion(row["completion"]) for row in probes]
        payload = {
            "label": cell.label,
            "layer": layer,
            "operation": "restore_paired_base",
            "beta": cell.beta,
            "restoration_definition": (
                "h_sft(x_1:t) <- h_sft(x_1:t) - beta * "
                "P_direction(h_sft(x_1:t) - h_base(x_1:t))"
            ),
            "pairing": "same token IDs, causal prefixes, masks, position convention, and hook point",
            "direction_control": "orientation only; projector is invariant to direction norm and sign",
            "direction_l2_norm_bookkeeping": float(torch.linalg.vector_norm(directions[index]).item()),
            "direction_metadata": metadata[index],
            "matched_sft_checkpoint": str(sft_checkpoint),
            "matched_rl_checkpoint": str(rl_checkpoint),
            "rl_to_intervention_kl": _mean(toward_rl_values[index]),
            "rl_to_intervention_kl_values": toward_rl_values[index],
            "forward_kl": _mean(forward_values[index]),
            "forward_kl_values": forward_values[index],
            "induced_kl_on_rl_prefixes": _mean(induced_values[index]),
            "induced_kl_on_rl_prefixes_values": induced_values[index],
            "task_accuracy": _mean(float(row["correct"]) for row in task),
            "old_accuracy": _mean(float(row["correct"]) for row in old),
            "domain_intrusion": _mean(intrusion),
            "sample_counts": {
                "task": len(task),
                "old": len(old),
                "probe": len(probes),
                "kl": len(toward_rl_values[index]),
            },
            "task_records": task,
            "old_records": old,
            "probe_generations": probes,
        }
        destination = output / cell.label / f"layer{layer}_restore_paired_{_safe_beta(cell.beta)}.json"
        _atomic_json(destination, payload)
        raw_rows.append({"output": str(destination), **{key: payload[key] for key in (
            "label", "beta", "rl_to_intervention_kl", "forward_kl", "induced_kl_on_rl_prefixes",
            "task_accuracy", "old_accuracy", "domain_intrusion"
        )}})
    _atomic_json(
        output / "e4_raw_summary.json",
        {
            "matrix": "5 semantic betas plus 3 random orientations x 4 nonzero betas",
            "cell_count": len(cells),
            "layer": layer,
            "smoke": smoke,
            "rows": raw_rows,
        },
    )
    print(json.dumps({"output": str(output), "cells": len(cells), "smoke": smoke}, indent=2))


def _random_artifact(value):
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from error
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact paired-base E4 necessity interventions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--semantic-artifact", required=True)
    parser.add_argument("--random-artifact", action="append", type=_random_artifact, default=[])
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectory-batch-size", type=int, default=16)
    parser.add_argument("--kl-batch-size", type=int, default=2)
    parser.add_argument("--cell-chunk-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_e4(
        args.config,
        args.sft_checkpoint,
        args.rl_checkpoint,
        args.semantic_artifact,
        args.random_artifact,
        args.layer,
        args.output_dir,
        trajectory_batch_size=args.trajectory_batch_size,
        kl_batch_size=args.kl_batch_size,
        cell_chunk_size=args.cell_chunk_size,
        generation_batch_size=args.generation_batch_size,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
