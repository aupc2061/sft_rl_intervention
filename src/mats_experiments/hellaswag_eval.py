"""Likelihood-based HellaSwag retention evaluation for base, SFT, RL, and E4 states.

This is deliberately an evaluation-only experiment.  It scores four answer labels under the
model's chat template, avoiding brittle free-generation parsing.  E4 evaluations use the same
paired-prefix base restoration operation as the original necessity experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_config
from .e4 import _model_device
from .hf_backend import adapter_enabled, load_adapter_model, load_tokenizer, require_training_stack
from .interventions import PairedBaseRestoration
from .numerics import bootstrap_interval


@dataclass(frozen=True)
class HellaSwagExample:
    example_id: str
    context: str
    endings: tuple[str, str, str, str]
    answer: int


@dataclass(frozen=True)
class EvaluationCell:
    label: str
    kind: str
    beta: float | None = None
    direction_path: str | None = None


def format_hellaswag_prompt(example: HellaSwagExample) -> str:
    """Render a four-choice HellaSwag prompt with a fixed label contract."""
    options = "\n".join(f"Option {index}: {ending}" for index, ending in enumerate(example.endings, start=1))
    return (
        "Choose the most plausible continuation of the context. Reply with only 1, 2, 3, or 4.\n\n"
        f"Context: {example.context}\n"
        f"{options}\n"
        "Answer:"
    )


def load_hellaswag(limit: int, seed: int, *, split: str = "validation") -> list[HellaSwagExample]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HellaSwag evaluation requires datasets; install with uv sync --all-extras") from exc

    rows = list(load_dataset("hellaswag", split=split))
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = rows[:limit]
    if len(selected) < limit:
        raise ValueError(f"Requested {limit} HellaSwag examples but split {split!r} has only {len(selected)}")
    return [
        HellaSwagExample(
            example_id=f"hellaswag-{split}-{index:05d}",
            context=str(row["ctx"]),
            endings=tuple(str(ending) for ending in row["endings"]),
            answer=int(row["label"]) + 1,
        )
        for index, row in enumerate(selected)
    ]


def _chat_tokens(tokenizer, prompt: str, answer: str | None):
    """Tokenize a user prompt, optionally followed by a fixed assistant answer label."""
    messages = [{"role": "user", "content": prompt}]
    if answer is None:
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
    )


def build_choice_records(tokenizer, examples: Iterable[HellaSwagExample]) -> list[dict[str, Any]]:
    """Return four tokenized candidate continuations per example with an exact prefix contract."""
    records: list[dict[str, Any]] = []
    for example in examples:
        prompt = format_hellaswag_prompt(example)
        prompt_ids = list(_chat_tokens(tokenizer, prompt, None))
        for label in ("1", "2", "3", "4"):
            full_ids = list(_chat_tokens(tokenizer, prompt, label))
            if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
                raise RuntimeError(
                    "Chat-template tokenization does not preserve the generation-prefix boundary; "
                    "cannot score answer labels safely."
                )
            records.append(
                {
                    "example_id": example.example_id,
                    "gold": example.answer,
                    "candidate": int(label),
                    "input_ids": full_ids,
                    "answer_start": len(prompt_ids),
                    "prompt": prompt,
                }
            )
    return records


def _collate_choice_records(records: list[dict[str, Any]], pad_token_id: int, device):
    import torch

    width = max(len(row["input_ids"]) for row in records)
    input_ids = torch.full((len(records), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    choice_mask = torch.zeros((len(records), width - 1), dtype=torch.bool, device=device)
    for index, row in enumerate(records):
        tokens = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[index, -len(tokens) :] = tokens
        attention_mask[index, -len(tokens) :] = 1
        # Logit j predicts token j+1. The answer starts at unpadded token ``answer_start``.
        left_pad = width - len(tokens)
        start = left_pad + int(row["answer_start"]) - 1
        stop = width - 1
        choice_mask[index, start:stop] = True
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids, choice_mask


def _choice_log_likelihood(logits, input_ids, choice_mask) -> list[float]:
    import torch.nn.functional as functional

    log_probs = functional.log_softmax(logits[:, :-1].float(), dim=-1)
    target = input_ids[:, 1:]
    token_scores = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return (token_scores * choice_mask.to(token_scores.dtype)).sum(dim=-1).tolist()


def _score_records(model, records, tokenizer, *, batch_size: int, restorer=None, direction=None, beta=None):
    import torch

    values: list[float] = []
    device = _model_device(model)
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            input_ids, attention_mask, position_ids, choice_mask = _collate_choice_records(
                batch, tokenizer.pad_token_id, device
            )
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": False,
            }
            if restorer is None:
                output = model(**kwargs)
            else:
                with restorer.capture_base(), adapter_enabled(model, False):
                    model(**kwargs)
                with restorer.restore(restorer.base_hidden, direction, beta), adapter_enabled(model, True):
                    output = model(**kwargs)
            values.extend(_choice_log_likelihood(output.logits, input_ids, choice_mask))
            completed = start + len(batch)
            if completed == len(records) or completed % (batch_size * 100) == 0:
                print(f"[HellaSwag] scored {completed}/{len(records)} candidate labels", flush=True)
    return values


def summarize_choices(records: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    if len(records) != len(scores) or len(records) % 4:
        raise ValueError("Expected exactly four scored candidates per example")
    examples: list[dict[str, Any]] = []
    correct_values: list[float] = []
    margins: list[float] = []
    for index in range(0, len(records), 4):
        candidates = records[index : index + 4]
        candidate_scores = scores[index : index + 4]
        if len({candidate["example_id"] for candidate in candidates}) != 1:
            raise ValueError("Candidate records are not grouped by example")
        ranked = sorted(zip(candidate_scores, candidates), key=lambda item: item[0], reverse=True)
        predicted = ranked[0][1]["candidate"]
        gold = candidates[0]["gold"]
        correct = float(predicted == gold)
        correct_values.append(correct)
        margins.append(ranked[0][0] - ranked[1][0])
        examples.append(
            {
                "example_id": candidates[0]["example_id"],
                "gold": gold,
                "prediction": predicted,
                "correct": correct,
                "logprob_options": candidate_scores,
            }
        )
    low, high = bootstrap_interval(correct_values, samples=2_000, seed=1)
    return {
        "accuracy": sum(correct_values) / len(correct_values),
        "accuracy_ci95": [low, high],
        "mean_top_two_logprob_margin": sum(margins) / len(margins),
        "examples": examples,
    }


def _load_direction(path: str | Path, layer: int):
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    try:
        return artifact["directions"][layer].float(), artifact.get("metadata", {})
    except KeyError as exc:
        raise KeyError(f"Direction artifact {path} has no layer {layer}") from exc


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_model(cfg, checkpoint: str, device: str):
    if device == "auto":
        return load_adapter_model(cfg, checkpoint, device_map="auto")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of auto, cpu, cuda")
    if device == "cuda":
        return load_adapter_model(cfg, checkpoint, device_map="auto")

    # BF16 is needlessly slow on many desktop CPUs; float32 is the conservative CPU path.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model


def run_hellaswag(
    *,
    config_path: str,
    sft_checkpoint: str,
    rl_checkpoint: str,
    direction_artifact: str,
    layer: int,
    output_path: str,
    limit: int,
    split: str,
    batch_size: int,
    device: str,
    e4_betas: tuple[float, ...],
) -> dict[str, Any]:
    require_training_stack()
    cfg = load_config(config_path)
    examples = load_hellaswag(limit, cfg.experiment.seed, split=split)
    tokenizer = load_tokenizer(cfg, padding_side="left")
    records = build_choice_records(tokenizer, examples)
    direction, metadata = _load_direction(direction_artifact, layer)
    cells = [EvaluationCell("base", "base"), EvaluationCell("sft", "sft")]
    cells.extend(EvaluationCell(f"e4_semantic_beta_{beta:g}", "e4", beta, direction_artifact) for beta in e4_betas)
    cells.append(EvaluationCell("rl", "rl"))
    output = Path(output_path)
    if output.is_file():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")

    results: list[dict[str, Any]] = []
    sft_model = _load_model(cfg, sft_checkpoint, device)
    try:
        for cell in cells:
            if cell.kind == "rl":
                continue
            if cell.kind == "base":
                with adapter_enabled(sft_model, False):
                    scores = _score_records(sft_model, records, tokenizer, batch_size=batch_size)
            elif cell.kind == "sft":
                with adapter_enabled(sft_model, True):
                    scores = _score_records(sft_model, records, tokenizer, batch_size=batch_size)
            else:
                restorer = PairedBaseRestoration(layer)
                with restorer.install(sft_model):
                    scores = _score_records(
                        sft_model,
                        records,
                        tokenizer,
                        batch_size=batch_size,
                        restorer=restorer,
                        direction=direction,
                        beta=cell.beta,
                    )
            results.append({"label": cell.label, "kind": cell.kind, "beta": cell.beta, **summarize_choices(records, scores)})
    finally:
        del sft_model

    rl_model = _load_model(cfg, rl_checkpoint, device)
    try:
        with adapter_enabled(rl_model, True):
            scores = _score_records(rl_model, records, tokenizer, batch_size=batch_size)
        results.append({"label": "rl", "kind": "rl", "beta": None, **summarize_choices(records, scores)})
    finally:
        del rl_model

    summary_rows = [
        {key: row[key] for key in ("label", "kind", "beta", "accuracy", "accuracy_ci95", "mean_top_two_logprob_margin")}
        for row in results
    ]
    payload = {
        "benchmark": "hellaswag",
        "split": split,
        "limit": limit,
        "seed": cfg.experiment.seed,
        "scoring": "zero-shot chat-template likelihood of assistant answer labels 1 versus 2",
        "sample_unit": "HellaSwag examples; bootstrap intervals resample examples",
        "sft_checkpoint": sft_checkpoint,
        "rl_checkpoint": rl_checkpoint,
        "direction_artifact": direction_artifact,
        "layer": layer,
        "direction_metadata": metadata,
        "device": device,
        "batch_size": batch_size,
        "summary": summary_rows,
        "results": results,
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base/SFT/RL/E4 on HellaSwag by four-choice answer-label likelihood")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--rl-checkpoint", required=True)
    parser.add_argument("--direction-artifact", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--e4-beta", action="append", type=float, default=None)
    args = parser.parse_args()
    payload = run_hellaswag(
        config_path=args.config,
        sft_checkpoint=args.sft_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        direction_artifact=args.direction_artifact,
        layer=args.layer,
        output_path=args.output,
        limit=args.limit,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        e4_betas=tuple(args.e4_beta if args.e4_beta is not None else (0.5, 1.0)),
    )
    print(json.dumps({"output": args.output, "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
