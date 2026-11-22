from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


_ANSWER_PATTERN = re.compile(r"(?i)answer\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_BOXED_PATTERN = re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}")
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_FINAL_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_FINAL_LINE_PATTERN = re.compile(
    rf"(?:\s*(?:final\s+)?answer\s*:\s*\$?\s*({_FINAL_NUMBER})\s*[.!]?\s*"
    rf"|\s*(?:(?:final\s+)?answer\s*:\s*)?\\boxed\{{\s*({_FINAL_NUMBER})\s*\}}\s*[.!]?\s*)",
    re.IGNORECASE,
)


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


def extract_final_numeric_answer(text: str) -> str | None:
    """Read a labeled number or boxed number on the last nonempty line."""
    lines = text.strip().splitlines()
    if not lines:
        return None
    line = lines[-1].strip().replace("**", "").replace("__", "").strip()
    for opening, closing in (("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"), ("$", "$")):
        if line.startswith(opening) and line.endswith(closing) and len(line) > len(opening) + len(closing):
            line = line[len(opening):-len(closing)].strip()
            break
    match = _FINAL_LINE_PATTERN.fullmatch(line)
    return normalize_number(match[1] or match[2]) if match else None


def final_numeric_reward(completion: str, answer: str) -> float:
    predicted = extract_final_numeric_answer(completion)
    return float(predicted is not None and predicted == normalize_number(answer))


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
    completions: Iterable[Any],
    answer: Iterable[str],
    log_extra=None,
    log_metric=None,
    **_: Any,
) -> list[float]:
    return _score_grpo(completions, answer, extract_numeric_answer, log_extra, log_metric)


def grpo_final_numeric_reward(
    completions: Iterable[Any],
    answer: Iterable[str],
    log_extra=None,
    log_metric=None,
    **_: Any,
) -> list[float]:
    return _score_grpo(completions, answer, extract_final_numeric_answer, log_extra, log_metric)


def _score_grpo(completions, answer, extract, log_extra, log_metric) -> list[float]:
    texts = [_completion_text(completion) for completion in completions]
    targets = list(answer)
    if len(texts) != len(targets):
        raise ValueError("GRPO received different numbers of completions and gold answers")
    predictions = [extract(text) for text in texts]
    rewards = [
        float(prediction is not None and prediction == normalize_number(target))
        for prediction, target in zip(predictions, targets)
    ]
    if callable(log_extra):
        log_extra("gold_answer", targets)
        log_extra("parsed_answer", [prediction or "[unparsed]" for prediction in predictions])
    if callable(log_metric) and texts:
        log_metric("parsed_answer_rate", sum(value is not None for value in predictions) / len(texts))
        log_metric("exact_numeric_accuracy", sum(rewards) / len(rewards))
    return rewards


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)
