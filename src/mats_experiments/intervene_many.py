from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .data import build_dataset
from .hf_backend import load_adapter_model, load_tokenizer, require_training_stack
from .intervene import run_intervention


@dataclass(frozen=True)
class InterventionCell:
    label: str
    artifact: str
    scale: float
    output: str


def _safe_scale(scale: float) -> str:
    return f"{scale:g}".replace("-", "neg").replace(".", "p")


def build_cells(output_directory, semantic_artifact, random_artifacts, layer, operation="add"):
    root = Path(output_directory)
    specifications = [
        ("semantic", str(semantic_artifact), (-1.0, 0.0, 0.5, 1.0)),
        *((f"random_{label}", artifact, (-1.0, 0.5, 1.0)) for label, artifact in random_artifacts),
    ]
    return [
        InterventionCell(label, artifact, scale, str(root / label / f"layer{layer}_{operation}_{_safe_scale(scale)}.json"))
        for label, artifact, scales in specifications
        for scale in scales
    ]


def partition_cells(cells, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    partitions = [[] for _ in range(min(workers, len(cells)))]
    for index, cell in enumerate(cells):
        partitions[index % len(partitions)].append(cell)
    return partitions


def _write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _worker(worker_id, config_path, checkpoint, sft_checkpoint, layer, operation, cells):
    require_training_stack()
    import torch

    cfg = load_config(config_path)
    model = load_adapter_model(cfg, checkpoint)
    sft_model = load_adapter_model(cfg, sft_checkpoint)
    tokenizer = load_tokenizer(cfg, padding_side="left")
    bundle = build_dataset(cfg.data, cfg.experiment.seed)
    artifacts = {}
    sequence_cache = {}
    for cell in cells:
        output = Path(cell.output)
        if output.is_file():
            print(f"[E3 worker {worker_id}] skipping existing {output}", flush=True)
            continue
        artifact = artifacts.get(cell.artifact)
        if artifact is None:
            artifact = torch.load(cell.artifact, map_location="cpu", weights_only=True)
            artifacts[cell.artifact] = artifact
        print(f"[E3 worker {worker_id}] {cell.label} scale={cell.scale:g} -> {output}", flush=True)
        payload = run_intervention(
            cfg, checkpoint, cell.artifact, layer, operation, cell.scale,
            sft_checkpoint=sft_checkpoint, model=model, sft_model=sft_model,
            tokenizer=tokenizer, bundle=bundle, artifact=artifact,
            sequence_cache=sequence_cache,
        )
        _write_json_atomic(output, payload)
        print(json.dumps({"worker": worker_id, "output": str(output), "task_accuracy": payload["task_accuracy"], "sft_to_intervention_kl": payload["sft_to_intervention_kl"]}), flush=True)


def run_grid(config_path, checkpoint, sft_checkpoint, semantic_artifact, random_artifacts, layer, operation, output_directory, workers):
    cells = [cell for cell in build_cells(output_directory, semantic_artifact, random_artifacts, layer, operation) if not Path(cell.output).is_file()]
    if not cells:
        print("[E3] all intervention cells already exist", flush=True)
        return
    partitions = partition_cells(cells, workers)
    context = mp.get_context("spawn")
    processes = [context.Process(target=_worker, args=(worker_id, config_path, checkpoint, sft_checkpoint, layer, operation, partition), name=f"intervention-evaluator-{worker_id}") for worker_id, partition in enumerate(partitions)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode != 0 for process in processes):
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError("At least one intervention worker failed")


def _random_artifact(value):
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from error
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, path


def main():
    parser = argparse.ArgumentParser(description="Run E3 cells with persistent GPU workers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--semantic-artifact", required=True)
    parser.add_argument("--random-artifact", action="append", type=_random_artifact, default=[])
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--operation", default="add", choices=("add", "subtract_mean", "restore_base_mean"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run_grid(args.config, args.checkpoint, args.sft_checkpoint, args.semantic_artifact, args.random_artifact, args.layer, args.operation, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
