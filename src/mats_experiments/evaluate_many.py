from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable

from .config import load_config
from .data import build_dataset
from .evaluate import evaluate_model
from .hf_backend import load_adapter_model, load_tokenizer, require_training_stack


def discover_unique_checkpoints(run_directory: str | Path) -> list[Path]:
    checkpoint_root = Path(run_directory) / "checkpoints"
    checkpoints = sorted(
        (
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and (path / "adapter_config.json").is_file()
        ),
        key=_checkpoint_sort_key,
    )
    if not checkpoints:
        raise ValueError(f"No PEFT checkpoints found under {checkpoint_root}")
    unique: list[Path] = []
    fingerprints: set[str] = set()
    for checkpoint in checkpoints:
        fingerprint = _adapter_fingerprint(checkpoint)
        if fingerprint in fingerprints:
            print(f"[eval] skipping duplicate adapter {checkpoint}", flush=True)
            continue
        fingerprints.add(fingerprint)
        unique.append(checkpoint)
    return unique


def partition_checkpoints(checkpoints: Iterable[Path], workers: int) -> list[list[Path]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    partitions = [[] for _ in range(workers)]
    for index, checkpoint in enumerate(checkpoints):
        partitions[index % workers].append(checkpoint)
    return [partition for partition in partitions if partition]


def _checkpoint_sort_key(path: Path) -> tuple[int, int | str]:
    name = path.name
    if name.startswith("checkpoint-"):
        try:
            return (0, int(name.removeprefix("checkpoint-")))
        except ValueError:
            pass
    return (1, name)


def _adapter_fingerprint(checkpoint: Path) -> str:
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        artifact = checkpoint / filename
        if artifact.is_file():
            digest = hashlib.sha256()
            with artifact.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    raise ValueError(f"Adapter weights are missing from {checkpoint}")


def _write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _worker(worker_id: int, config_path: str, run_directory: str, checkpoints: list[str]) -> None:
    require_training_stack()
    cfg = load_config(config_path)
    paths = [Path(checkpoint) for checkpoint in checkpoints]
    model = load_adapter_model(cfg, paths[0])
    adapter_names = {paths[0]: "default"}
    for index, checkpoint in enumerate(paths[1:], start=1):
        name = f"worker{worker_id}_checkpoint{index}"
        model.load_adapter(checkpoint, adapter_name=name, is_trainable=False)
        adapter_names[checkpoint] = name
    tokenizer = load_tokenizer(cfg, padding_side="left")
    bundle = build_dataset(cfg.data, cfg.experiment.seed)
    run_root = Path(run_directory)
    for checkpoint in paths:
        model.set_adapter(adapter_names[checkpoint])
        output = run_root / f"evaluation_{checkpoint.name}.json"
        print(f"[eval worker {worker_id}] {checkpoint} -> {output}", flush=True)
        payload = evaluate_model(cfg, model, tokenizer, bundle)
        _write_json_atomic(output, payload)
        print(
            json.dumps(
                {
                    "worker": worker_id,
                    "output": str(output),
                    "task_accuracy": payload["task_accuracy"]["mean"],
                    "forward_kl": payload["forward_kl"]["mean"],
                    "reverse_kl": payload["reverse_kl"]["mean"],
                }
            ),
            flush=True,
        )


def evaluate_run(config_path: str, run_directory: str, workers: int = 2) -> None:
    checkpoints = discover_unique_checkpoints(run_directory)
    partitions = partition_checkpoints(checkpoints, min(workers, len(checkpoints)))
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(index, config_path, run_directory, [str(path) for path in partition]),
            name=f"checkpoint-evaluator-{index}",
        )
        for index, partition in enumerate(partitions)
    ]
    for process in processes:
        process.start()
    failed = False
    for process in processes:
        process.join()
        failed = failed or process.exitcode != 0
    if failed:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError("At least one checkpoint evaluation worker failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints with persistent GPU workers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    evaluate_run(args.config, args.run_directory, args.workers)


if __name__ == "__main__":
    main()
