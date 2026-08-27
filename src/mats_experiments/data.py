from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from .config import DataConfig


@dataclass(frozen=True)
class Example:
    prompt: str
    completion: str
    answer: str
    split: str
    example_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "answer": self.answer,
            "split": self.split,
            "example_id": self.example_id,
        }


@dataclass(frozen=True)
class DatasetBundle:
    train: tuple[Example, ...]
    task_test: tuple[Example, ...]
    probe: tuple[Example, ...]
    old: tuple[Example, ...]


def build_dataset(cfg: DataConfig, seed: int) -> DatasetBundle:
    if cfg.kind == "synthetic_arithmetic":
        return build_synthetic_arithmetic(cfg, seed)
    if cfg.kind == "gsm8k":
        return build_gsm8k(cfg, seed)
    raise ValueError(f"Unsupported dataset kind: {cfg.kind}")


def build_synthetic_arithmetic(cfg: DataConfig, seed: int) -> DatasetBundle:
    rng = random.Random(seed)
    seen: set[tuple[str, int, int]] = set()

    def generate(count: int, split: str, operations: tuple[str, ...]) -> tuple[Example, ...]:
        rows: list[Example] = []
        while len(rows) < count:
            operation = rng.choice(operations)
            a = rng.randint(1, cfg.max_operand)
            b = rng.randint(1, cfg.max_operand)
            key = (operation, a, b)
            if key in seen:
                continue
            seen.add(key)
            if operation == "multiply":
                symbol, answer = "*", a * b
            elif operation == "add":
                symbol, answer = "+", a + b
            else:
                if b > a:
                    a, b = b, a
                symbol, answer = "-", a - b
            prompt = (
                "Solve the arithmetic problem. Give a short explanation and end with "
                f"'Answer: <number>'.\nProblem: {a} {symbol} {b}"
            )
            completion = f"Compute {a} {symbol} {b}. Answer: {answer}"
            rows.append(Example(prompt, completion, str(answer), split, f"{split}-{len(rows):05d}"))
        return tuple(rows)

    train = generate(cfg.train_size, "train", ("multiply",))
    task_test = generate(cfg.task_test_size, "task_test", ("multiply",))
    old = generate(cfg.old_size, "old", ("add", "subtract"))
    probe = tuple(
        Example(text, "", "", "probe", f"probe-{index:05d}")
        for index, text in enumerate(_unrelated_probe_texts(cfg.probe_size, rng))
    )
    return DatasetBundle(train, task_test, probe, old)


def build_gsm8k(cfg: DataConfig, seed: int) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for data.kind=gsm8k") from exc

    dataset = load_dataset("openai/gsm8k", "main")
    rng = random.Random(seed)

    def convert(rows: Iterable[dict[str, Any]], split: str, count: int) -> tuple[Example, ...]:
        rows = list(rows)
        rng.shuffle(rows)
        result: list[Example] = []
        for index, row in enumerate(rows[:count]):
            answer = str(row["answer"]).split("####")[-1].strip().replace(",", "")
            prompt = (
                "Solve the following problem. Give a concise derivation and end with "
                f"'Answer: <number>'.\nProblem: {row['question']}"
            )
            completion = str(row["answer"]).replace("####", "Answer:")
            result.append(Example(prompt, completion, answer, split, f"{split}-{index:05d}"))
        return tuple(result)

    train_rows = list(dataset["train"])
    test_rows = list(dataset["test"])
    train = convert(train_rows, "train", cfg.train_size)
    task_test = convert(test_rows, "task_test", cfg.task_test_size)
    old = build_synthetic_arithmetic(cfg, seed + 1).old
    probe_rng = random.Random(seed + 2)
    probe = tuple(
        Example(text, "", "", "probe", f"probe-{index:05d}")
        for index, text in enumerate(_unrelated_probe_texts(cfg.probe_size, probe_rng))
    )
    return DatasetBundle(train, task_test, probe, old)


def to_hf_dataset(examples: Iterable[Example], conversational: bool = False):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for Hugging Face training") from exc
    if conversational:
        rows = [
            {
                **example.as_dict(),
                "prompt": [{"role": "user", "content": example.prompt}],
            }
            for example in examples
        ]
    else:
        rows = [example.as_dict() for example in examples]
    return Dataset.from_list(rows)


def _unrelated_probe_texts(count: int, rng: random.Random) -> list[str]:
    subjects = (
        "a coastal wetland",
        "Renaissance painting",
        "public transport planning",
        "bread fermentation",
        "a chamber music rehearsal",
        "cloud formation",
        "urban tree cover",
        "the history of paper",
    )
    verbs = ("describes", "compares", "summarizes", "questions", "illustrates")
    endings = (
        "without reaching a final conclusion.",
        "using observations gathered over several years.",
        "for a general audience with no specialist background.",
        "while noting several practical tradeoffs.",
    )
    return [
        f"This short passage {rng.choice(verbs)} {rng.choice(subjects)} {rng.choice(endings)}"
        for _ in range(count)
    ]

