from __future__ import annotations

import math
from collections.abc import Sequence

from kaggle_researcher.eda.metrics.gini import _binary_auc

try:  # pragma: no cover - availability depends on the local environment.
    from sklearn.metrics import accuracy_score as _sk_accuracy_score
    from sklearn.metrics import f1_score as _sk_f1_score
    from sklearn.metrics import log_loss as _sk_log_loss
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score
except Exception:  # pragma: no cover
    _sk_accuracy_score = None
    _sk_f1_score = None
    _sk_log_loss = None
    _sk_roc_auc_score = None


def sklearn_available() -> bool:
    return all(
        helper is not None
        for helper in (
            _sk_accuracy_score,
            _sk_f1_score,
            _sk_log_loss,
            _sk_roc_auc_score,
        )
    )


def binary_roc_auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    if _sk_roc_auc_score is not None:
        return float(_sk_roc_auc_score(y_true, y_score))
    return _binary_auc(_as_float_list(y_true), _as_float_list(y_score))


def logloss(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    eps: float = 1e-15,
) -> float:
    if _sk_log_loss is not None:
        return float(_sk_log_loss(y_true, y_prob))

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
    if _sk_accuracy_score is not None:
        return float(_sk_accuracy_score(y_true, y_pred))
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    correct = sum(1 for expected, actual in zip(y_true, y_pred, strict=True) if expected == actual)
    return correct / len(y_true)


def f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if _sk_f1_score is not None:
        return float(_sk_f1_score(y_true, y_pred))
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
