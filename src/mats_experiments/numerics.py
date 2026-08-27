from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Sequence


def kl_divergence(p: Sequence[float], q: Sequence[float], epsilon: float = 1e-12) -> float:
    if len(p) != len(q) or not p:
        raise ValueError("p and q must be non-empty and have equal length")
    p_total, q_total = sum(p), sum(q)
    if p_total <= 0 or q_total <= 0:
        raise ValueError("Distributions must have positive mass")
    value = 0.0
    for p_i, q_i in zip(p, q):
        p_norm = max(p_i / p_total, epsilon)
        q_norm = max(q_i / q_total, epsilon)
        value += p_norm * math.log(p_norm / q_norm)
    return value


def vector_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("At least one vector is required")
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("Vectors must be non-empty and have equal width")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def squared_norm(vector: Sequence[float]) -> float:
    return sum(value * value for value in vector)


def globality_ratio(differences: Sequence[Sequence[float]]) -> float:
    """Fraction of mean squared representation change explained by a global offset."""
    mean = vector_mean(differences)
    denominator = sum(squared_norm(row) for row in differences) / len(differences)
    return squared_norm(mean) / denominator if denominator > 0 else 0.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal width")
    denominator = math.sqrt(squared_norm(a) * squared_norm(b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


def mean_pairwise_cosine(vectors: Sequence[Sequence[float]]) -> float:
    if len(vectors) < 2:
        return 0.0
    values = [
        cosine_similarity(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    return sum(values) / len(values)


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sequence")
    statistic = statistic or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    estimates = sorted(
        statistic([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    low = estimates[max(0, int(tail * samples))]
    high = estimates[min(samples - 1, int((1.0 - tail) * samples) - 1)]
    return low, high

