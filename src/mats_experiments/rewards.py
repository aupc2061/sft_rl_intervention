from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


_ANSWER_PATTERN = re.compile(r"(?i)answer\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_BOXED_PATTERN = re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}")
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_numeric_answer(text: str) -> str | None:
    for pattern in (_ANSWER_PATTERN, _BOXED_PATTERN):
        matches = pattern.findall(text)
        if matches:
            return normalize_number(matches[-1])
    matches = _NUMBER_PATTERN.findall(text)
    return normalize_number(matches[-1]) if matches else None


def normalize_number(value: str) -> str | None:
    try:
        number = Decimal(value.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def exact_numeric_reward(completion: str, answer: str) -> float:
    predicted = extract_numeric_answer(completion)
    expected = normalize_number(answer)
    return float(predicted is not None and expected is not None and predicted == expected)


def arithmetic_domain_intrusion(text: str) -> float:
    """Predeclared cheap proxy for arithmetic-task intrusion into unrelated generations."""
    signals = (
        bool(_ANSWER_PATTERN.search(text)),
        "calculate" in text.lower(),
        "compute" in text.lower(),
        bool(re.search(r"\d\s*[+*=/\-]\s*\d", text)),
    )
    return sum(float(signal) for signal in signals) / len(signals)


def grpo_numeric_reward(
    completions: Iterable[Any], answer: Iterable[str], **_: Any
) -> list[float]:
    texts = [_completion_text(completion) for completion in completions]
    return [exact_numeric_reward(text, target) for text, target in zip(texts, answer)]


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)
