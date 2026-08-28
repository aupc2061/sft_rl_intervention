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


def restore_paired_base_component(finetuned, base, direction, beta=1.0):
    """Restore the fine-tuned residual toward the paired base residual along ``direction``.

    ``finetuned`` and ``base`` must describe the same token prefixes at the same hook point.
    ``direction`` may be one vector for the whole batch or one vector per batch row. Direction
    magnitude and sign do not affect the projector; only its one-dimensional subspace matters.
    """
    import torch

    if finetuned.shape != base.shape:
        raise ValueError(
            f"Paired activations must have identical shapes, got {finetuned.shape} and {base.shape}"
        )
    direction = direction.to(device=finetuned.device, dtype=finetuned.dtype)
    if direction.ndim == 1:
        unit = normalize_direction(direction)
    elif direction.ndim == 2:
        norms = torch.linalg.vector_norm(direction.float(), dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Every intervention direction must have finite non-zero norm")
        unit = direction / norms.to(direction.dtype)
        unit = unit[:, None, :]
    else:
        raise ValueError("Direction must have shape [hidden] or [batch, hidden]")

    if torch.is_tensor(beta):
        beta = beta.to(device=finetuned.device, dtype=finetuned.dtype)
        if beta.ndim == 1:
            beta = beta[:, None, None]
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


class PairedBaseRestoration:
    """Coordinate exact base/SFT passes through one PEFT model at one decoder layer.

    A caller first runs the model with adapters disabled inside :meth:`capture_base`, then runs
    the same token prefixes with adapters enabled inside :meth:`restore`. Keeping the two KV-cache
    streams outside this object makes the same mechanism work for teacher forcing and decoding.
    """

    def __init__(self, layer: int):
        self.layer = layer
        self.mode = "disabled"
        self.base_hidden = None
        self.direction = None
        self.beta = None

    @contextlib.contextmanager
    def capture_base(self):
        previous = self.mode
        self.base_hidden = None
        self.mode = "capture"
        try:
            yield self
        finally:
            self.mode = previous

    @contextlib.contextmanager
    def restore(self, base_hidden, direction, beta):
        previous = (self.mode, self.base_hidden, self.direction, self.beta)
        self.mode = "restore"
        self.base_hidden = base_hidden
        self.direction = direction
        self.beta = beta
        try:
            yield self
        finally:
            self.mode, self.base_hidden, self.direction, self.beta = previous

    @contextlib.contextmanager
    def install(self, model) -> Iterator["PairedBaseRestoration"]:
        layers = transformer_layers(model)
        if not 0 <= self.layer < len(layers):
            raise IndexError(f"Layer {self.layer} outside [0, {len(layers)})")

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.mode == "capture":
                self.base_hidden = hidden.detach()
                return output
            if self.mode == "restore":
                if self.base_hidden is None or self.direction is None or self.beta is None:
                    raise RuntimeError("Paired restoration is missing its base activation or control")
                replacement = restore_paired_base_component(
                    hidden, self.base_hidden, self.direction, self.beta
                )
                return _replace_hidden(output, replacement)
            return output

        handle = layers[self.layer].register_forward_hook(hook)
        try:
            yield self
        finally:
            handle.remove()
