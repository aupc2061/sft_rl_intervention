from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_root: str = "outputs"
    seed: int = 1


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    dtype: str = "bfloat16"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    attention_implementation: str | None = None


@dataclass(frozen=True)
class DataConfig:
    kind: str = "synthetic_arithmetic"
    train_size: int = 128
    task_test_size: int = 64
    probe_size: int = 64
    old_size: int = 64
    max_operand: int = 30


@dataclass(frozen=True)
class TrainingConfig:
    method: str = "sft"
    learning_rate: float = 1e-4
    sft_learning_rate: float | None = None
    grpo_learning_rate: float | None = None
    epochs: float = 1.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    sft_batch_size: int | None = None
    sft_gradient_accumulation_steps: int | None = None
    gradient_checkpointing: bool = True
    tf32: bool | None = None
    max_length: int = 512
    max_completion_length: int = 128
    num_generations: int = 4
    save_steps: int = 50
    logging_steps: int = 5
    anchor_direction: str | None = None
    anchor_layer: int | None = None
    anchor_beta: float = 0.0
    kl_coefficient: float = 0.0


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 2
    max_new_tokens: int = 128
    kl_samples: int = 64
    intervention_probe_samples: int = 32
    temperature: float = 0.8
    bootstrap_samples: int = 10_000


@dataclass(frozen=True)
class TraceConfig:
    dtype: str = "float32"
    token_positions: tuple[int, ...] = (-5, -4, -3, -2, -1)
    discovery_fraction: float = 0.5
    top_k_layers: int = 3
    intervention_scales: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass(frozen=True)
class ProjectConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    traces: TraceConfig = field(default_factory=TraceConfig)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def validate(self) -> None:
        if self.training.method not in {"sft", "grpo", "anchored_sft", "sft_kl"}:
            raise ValueError(f"Unsupported training method: {self.training.method}")
        if self.data.kind not in {"synthetic_arithmetic", "gsm8k"}:
            raise ValueError(f"Unsupported dataset kind: {self.data.kind}")
        if not 0 < self.traces.discovery_fraction < 1:
            raise ValueError("traces.discovery_fraction must lie strictly between 0 and 1")
        if self.traces.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported traces.dtype: {self.traces.dtype}")
        if self.training.method == "anchored_sft":
            if not self.training.anchor_direction or self.training.anchor_layer is None:
                raise ValueError("anchored_sft requires anchor_direction and anchor_layer")
            if not 0 <= self.training.anchor_beta <= 1:
                raise ValueError("anchor_beta must be in [0, 1]")
        if self.training.kl_coefficient < 0:
            raise ValueError("kl_coefficient must be non-negative")
        for name in ("learning_rate", "sft_learning_rate", "grpo_learning_rate"):
            value = getattr(self.training, name)
            if value is not None and value <= 0:
                raise ValueError(f"training.{name} must be positive")
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "sft_batch_size",
            "sft_gradient_accumulation_steps",
        ):
            value = getattr(self.training, name)
            if value is not None and value <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.training.num_generations < 2 and self.training.method == "grpo":
            raise ValueError("GRPO requires at least two generations per prompt")
        for name in ("train_size", "task_test_size", "probe_size", "old_size"):
            if getattr(self.data, name) <= 0:
                raise ValueError(f"data.{name} must be positive")
        if self.evaluation.intervention_probe_samples <= 0:
            raise ValueError("evaluation.intervention_probe_samples must be positive")


def _construct(cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    for field_info in dataclasses.fields(cls):
        if field_info.name in values and field_info.name in {
            "target_modules",
            "token_positions",
            "intervention_scales",
        }:
            values[field_info.name] = tuple(values[field_info.name])
    return cls(**values)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ProjectConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load experiment configs") from exc

    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    if overrides:
        _apply_overrides(raw, overrides)
    cfg = ProjectConfig(
        experiment=_construct(ExperimentConfig, raw.get("experiment")),
        model=_construct(ModelConfig, raw.get("model")),
        data=_construct(DataConfig, raw.get("data")),
        training=_construct(TrainingConfig, raw.get("training")),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation")),
        traces=_construct(TraceConfig, raw.get("traces")),
    )
    cfg.validate()
    return cfg


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor = raw
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
