from __future__ import annotations

import math
import statistics
from typing import Any

from kaggle_researcher.facts.models import CvLbPair, NotebookFacts


def build_cv_lb_pairs(notebooks: list[NotebookFacts]) -> list[CvLbPair]:
    pairs: list[CvLbPair] = []
    for notebook in notebooks:
        if notebook.public_score is None:
            continue
        declared_cv = _select_declared_cv(notebook.declared_cv)
        public_score = _finite_float(notebook.public_score)
        if declared_cv is None or public_score is None:
            continue
        pairs.append(
            CvLbPair(
                notebook_ref=notebook.ref,
                declared_cv=declared_cv,
                public_score=public_score,
                lineage_cluster_id=notebook.lineage_cluster_id,
            )
        )
    return pairs


def summarize_cv_lb(pairs: list[CvLbPair]) -> dict[str, int | float | None]:
    count = len(pairs)
    gaps = [pair.declared_cv - pair.public_score for pair in pairs]
    return {
        "count": count,
        "mean_gap": statistics.fmean(gaps) if gaps else None,
        "median_gap": statistics.median(gaps) if gaps else None,
        "spearman": (
            _spearman_correlation(
                [pair.declared_cv for pair in pairs],
                [pair.public_score for pair in pairs],
            )
            if count >= 3
            else None
        ),
        "distinct_lineage_clusters": len(
            {pair.lineage_cluster_id for pair in pairs}
        ),
    }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _select_declared_cv(values: list[Any]) -> float | None:
    numeric_values = [
        number for value in values if (number := _finite_float(value)) is not None
    ]
    if not numeric_values:
        return None
    if len(numeric_values) <= 3:
        return numeric_values[0]
    # Long lists are usually fold-like scores; the median limits outlier influence.
    return statistics.median(numeric_values)


def _spearman_correlation(
    left: list[float],
    right: list[float],
) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths")
    if len(left) < 2:
        return None
    return _pearson_correlation(_average_ranks(left), _average_ranks(right))


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Pearson inputs must have equal lengths")
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_offsets = [value - left_mean for value in left]
    right_offsets = [value - right_mean for value in right]
    left_squared = sum(value * value for value in left_offsets)
    right_squared = sum(value * value for value in right_offsets)
    if left_squared == 0 or right_squared == 0:
        return None
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left_offsets, right_offsets)
    )
    correlation = numerator / math.sqrt(left_squared * right_squared)
    return max(-1.0, min(1.0, correlation))
