from __future__ import annotations

import math
from collections.abc import Sequence

from kaggle_researcher.eda.metrics.gini import _binary_auc


def sklearn_available() -> bool:
    return False


def binary_roc_auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    return _binary_auc(_as_float_list(y_true), _as_float_list(y_score))


def logloss(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    eps: float = 1e-15,
) -> float:
    labels = _as_float_list(y_true)
    probabilities = [min(max(value, eps), 1 - eps) for value in _as_float_list(y_prob)]
    if len(labels) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same length")
    if not labels:
        raise ValueError("y_true and y_prob must not be empty")
    loss = sum(
        label * math.log(probability) + (1 - label) * math.log(1 - probability)
        for label, probability in zip(labels, probabilities, strict=True)
    )
    return float(-loss / len(labels))


def accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    correct = sum(1 for expected, actual in zip(y_true, y_pred, strict=True) if expected == actual)
    return correct / len(y_true)


def f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    true_positive = sum(
        1
        for expected, actual in zip(y_true, y_pred, strict=True)
        if int(expected) == 1 and int(actual) == 1
    )
    false_positive = sum(
        1
        for expected, actual in zip(y_true, y_pred, strict=True)
        if int(expected) == 0 and int(actual) == 1
    )
    false_negative = sum(
        1
        for expected, actual in zip(y_true, y_pred, strict=True)
        if int(expected) == 1 and int(actual) == 0
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2 * true_positive) / denominator


def _as_float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


__all__ = [
    "accuracy",
    "binary_roc_auc",
    "f1",
    "logloss",
    "sklearn_available",
]
