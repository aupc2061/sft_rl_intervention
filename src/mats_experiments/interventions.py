from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator

from .hf_backend import transformer_layers


def normalize_direction(direction):
    import torch

    norm = torch.linalg.vector_norm(direction.float())
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Intervention direction must have finite non-zero norm")
    return direction / norm.to(direction.dtype)


def add_direction(hidden, direction, scale: float):
    """Add ``scale`` copies of the raw saved mean-shift vector to every token position."""
    direction = direction.to(device=hidden.device, dtype=hidden.dtype)
    return hidden + scale * direction


def subtract_mean_shift(hidden, direction, scale: float):
    direction = direction.to(device=hidden.device, dtype=hidden.dtype)
    return hidden - scale * direction


def restore_base_component(finetuned, base, direction, beta: float = 1.0):
    """Replace only the fine-tuning-induced deviation along direction with the base component."""
    unit = normalize_direction(direction).to(device=finetuned.device, dtype=finetuned.dtype)
    deviation = finetuned - base.to(device=finetuned.device, dtype=finetuned.dtype)
    coefficient = (deviation * unit).sum(dim=-1, keepdim=True)
    return finetuned - beta * coefficient * unit


def restore_base_mean_component(hidden, base_mean, direction, beta: float = 1.0):
    """Restore a direction toward its base-model mean without erasing content variation orthogonal to it."""
    unit = normalize_direction(direction).to(device=hidden.device, dtype=hidden.dtype)
    mean = base_mean.to(device=hidden.device, dtype=hidden.dtype)
    coefficient = ((hidden - mean) * unit).sum(dim=-1, keepdim=True)
    return hidden - beta * coefficient * unit


def _replace_hidden(output: Any, replacement):
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    return replacement


@dataclass
class ResidualIntervention:
    layer: int
    direction: Any
    operation: str
    scale: float
    base_mean: Any | None = None
    active: bool = True

    def apply(self, hidden):
        if self.operation == "add":
            return add_direction(hidden, self.direction, self.scale)
        if self.operation == "subtract_mean":
            return subtract_mean_shift(hidden, self.direction, self.scale)
        if self.operation == "restore_base_mean":
            if self.base_mean is None:
                raise ValueError("restore_base_mean requires a saved base_mean")
            return restore_base_mean_component(hidden, self.base_mean, self.direction, self.scale)
        raise ValueError(f"Unsupported intervention operation: {self.operation}")

    @contextlib.contextmanager
    def install(self, model) -> Iterator["ResidualIntervention"]:
        layers = transformer_layers(model)
        if not 0 <= self.layer < len(layers):
            raise IndexError(f"Layer {self.layer} outside [0, {len(layers)})")

        def hook(_module, _inputs, output):
            if not self.active:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            return _replace_hidden(output, self.apply(hidden))

        handle = layers[self.layer].register_forward_hook(hook)
        try:
            yield self
        finally:
            handle.remove()

    @contextlib.contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = previous
