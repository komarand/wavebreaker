from __future__ import annotations

import math
from collections.abc import Sequence


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    labels, predictions = _aligned_float_lists(y_true, y_pred)
    return sum(
        (label - prediction) ** 2
        for label, prediction in zip(labels, predictions)
    ) / len(labels)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    labels, predictions = _aligned_float_lists(y_true, y_pred)
    return sum(
        abs(label - prediction)
        for label, prediction in zip(labels, predictions)
    ) / len(labels)


def rmsle(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    labels, predictions = _aligned_float_lists(y_true, y_pred)
    if any(value < 0 for value in labels) or any(value < 0 for value in predictions):
        raise ValueError("rmsle requires non-negative y_true and y_pred values")
    return math.sqrt(
        sum(
            (math.log1p(label) - math.log1p(prediction)) ** 2
            for label, prediction in zip(labels, predictions)
        )
        / len(labels)
    )


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    labels, predictions = _aligned_float_lists(y_true, y_pred)
    non_zero_pairs = [
        (label, prediction)
        for label, prediction in zip(labels, predictions)
        if label != 0
    ]
    if not non_zero_pairs:
        raise ValueError("mape is undefined when all y_true values are zero")
    return sum(
        abs((label - prediction) / label)
        for label, prediction in non_zero_pairs
    ) / len(non_zero_pairs)


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    labels, predictions = _aligned_float_lists(y_true, y_pred)
    terms = []
    for label, prediction in zip(labels, predictions):
        denominator = abs(label) + abs(prediction)
        terms.append(0.0 if denominator == 0 else 2 * abs(prediction - label) / denominator)
    return sum(terms) / len(terms)


def r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    labels, predictions = _aligned_float_lists(y_true, y_pred)
    mean_label = sum(labels) / len(labels)
    total_sum_squares = sum((label - mean_label) ** 2 for label in labels)
    if total_sum_squares == 0:
        return 0.0
    residual_sum_squares = sum(
        (label - prediction) ** 2
        for label, prediction in zip(labels, predictions)
    )
    return 1 - residual_sum_squares / total_sum_squares


def _aligned_float_lists(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> tuple[list[float], list[float]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return [float(value) for value in y_true], [float(value) for value in y_pred]


__all__ = ["mae", "mape", "mse", "r2", "rmse", "rmsle", "smape"]
