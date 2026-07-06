from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from kaggle_researcher.eda.metrics.gini import normalized_gini


def weekly_gini(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    week: Sequence[Any],
) -> list[dict[str, Any]]:
    """Compute normalized Gini for each week/period."""

    if not (len(y_true) == len(y_pred) == len(week)):
        raise ValueError("y_true, y_pred, and week must have the same length")
    if len(y_true) == 0:
        raise ValueError("y_true, y_pred, and week must not be empty")

    grouped: dict[Any, dict[str, list[Any]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for label, prediction, period in zip(y_true, y_pred, week, strict=True):
        grouped[period]["y_true"].append(label)
        grouped[period]["y_pred"].append(prediction)

    components: list[dict[str, Any]] = []
    for period in sorted(grouped, key=_natural_sort_key):
        period_true = grouped[period]["y_true"]
        period_pred = grouped[period]["y_pred"]
        positives = sum(1 for value in period_true if float(value) > 0)
        components.append(
            {
                "week": period,
                "n_rows": len(period_true),
                "n_positive": positives,
                "gini": normalized_gini(period_true, period_pred),
            }
        )
    return components


def simple_gini_stability_score(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    week: Sequence[Any],
) -> dict[str, Any]:
    """Return a compact stability summary from weekly Gini values."""

    components = weekly_gini(y_true, y_pred, week)
    values = [float(item["gini"]) for item in components]
    average_gini = sum(values) / len(values)
    trend_penalty = abs(_linear_slope(values)) if len(values) > 1 else 0.0
    residual_std_penalty = _residual_std(values) if len(values) > 2 else 0.0
    score = average_gini - trend_penalty - residual_std_penalty

    return {
        "score": score,
        "average_gini": average_gini,
        "weekly_gini": components,
        "trend_penalty": trend_penalty,
        "residual_std_penalty": residual_std_penalty,
    }


def _linear_slope(values: list[float]) -> float:
    n_values = len(values)
    x_mean = (n_values - 1) / 2
    y_mean = sum(values) / n_values
    denominator = sum((index - x_mean) ** 2 for index in range(n_values))
    if denominator == 0:
        return 0.0
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    return numerator / denominator


def _residual_std(values: list[float]) -> float:
    n_values = len(values)
    slope = _linear_slope(values)
    intercept = (sum(values) / n_values) - slope * ((n_values - 1) / 2)
    residuals = [value - (intercept + slope * index) for index, value in enumerate(values)]
    variance = sum(residual**2 for residual in residuals) / n_values
    return variance**0.5


def _natural_sort_key(value: Any) -> tuple[str, Any]:
    if isinstance(value, (int, float)):
        return ("number", value)
    text = str(value)
    try:
        return ("number", float(text))
    except ValueError:
        return ("text", text)


__all__ = ["simple_gini_stability_score", "weekly_gini"]
