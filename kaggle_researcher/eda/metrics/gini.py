from __future__ import annotations

from collections.abc import Sequence


def gini_from_predictions(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return the normalized Gini coefficient implied by prediction ranks."""

    return normalized_gini(y_true, y_pred)


def normalized_gini(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute binary normalized Gini as ``2 * ROC_AUC - 1``.

    Tied predictions receive average ranks. If a sample has a single class, the
    ranking is undefined and this helper returns 0.0 instead of failing.
    """

    labels = _as_float_list(y_true, "y_true")
    predictions = _as_float_list(y_pred, "y_pred")
    auc = _binary_auc(labels, predictions)
    return float(2 * auc - 1)


def _binary_auc(labels: Sequence[float], predictions: Sequence[float]) -> float:
    labels = _as_float_list(labels, "labels")
    predictions = _as_float_list(predictions, "predictions")
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    if not labels:
        raise ValueError("labels and predictions must not be empty")

    positives = sum(1 for value in labels if value > 0)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    ranks = _average_ranks(predictions)
    positive_rank_sum = sum(rank for label, rank in zip(labels, ranks, strict=True) if label > 0)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_ranks(values: list[float]) -> list[float]:
    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed_values):
        next_position = position + 1
        while (
            next_position < len(indexed_values)
            and indexed_values[next_position][1] == indexed_values[position][1]
        ):
            next_position += 1

        average_rank = (position + 1 + next_position) / 2
        for indexed_position in range(position, next_position):
            original_index = indexed_values[indexed_position][0]
            ranks[original_index] = average_rank
        position = next_position
    return ranks


def _as_float_list(values: Sequence[float], argument_name: str) -> list[float]:
    try:
        return [float(value) for value in values]
    except TypeError as exc:
        raise TypeError(f"{argument_name} must be a sequence of numbers") from exc


__all__ = ["gini_from_predictions", "normalized_gini"]
