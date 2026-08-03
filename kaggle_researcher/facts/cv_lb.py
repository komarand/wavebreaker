from __future__ import annotations

import math
import statistics
from typing import Any

from kaggle_researcher.facts.models import CvLbDiagnostics, CvLbPair, NotebookFacts


def build_cv_lb_pairs(
    notebooks: list[NotebookFacts],
    competition_metric_name: str | None = None,
) -> list[CvLbPair]:
    pairs: list[CvLbPair] = []
    for notebook in notebooks:
        if notebook.public_score is None:
            continue
        declared_cv = _select_declared_cv(notebook.declared_cv)
        public_score = _finite_float(notebook.public_score)
        if declared_cv is None or public_score is None:
            continue
        notebook_metric_name = _notebook_cv_metric(notebook)
        if not _metrics_are_comparable(
            notebook_metric_name,
            competition_metric_name,
        ):
            continue
        pairs.append(
            CvLbPair(
                notebook_ref=notebook.ref,
                declared_cv=declared_cv,
                public_score=public_score,
                lineage_cluster_id=notebook.lineage_cluster_id,
                metric_name=notebook_metric_name,
            )
        )
    return pairs


def diagnose_cv_lb(
    notebooks: list[NotebookFacts],
    pairs: list[CvLbPair],
) -> CvLbDiagnostics:
    with_public_score = sum(notebook.public_score is not None for notebook in notebooks)
    with_declared_cv = sum(bool(notebook.declared_cv) for notebook in notebooks)
    with_both = sum(
        notebook.public_score is not None and bool(notebook.declared_cv)
        for notebook in notebooks
    )
    rejected = max(0, with_both - len(pairs))

    zero_pairs_reason = None
    if not pairs:
        if not notebooks:
            zero_pairs_reason = "No notebooks were collected."
        elif with_public_score == 0 and with_declared_cv == 0:
            zero_pairs_reason = (
                "Kaggle kernel objects expose no public scores and no notebooks "
                "contain declared CV observations."
            )
        elif with_public_score == 0:
            zero_pairs_reason = "Kaggle kernel objects expose no public scores."
        elif with_declared_cv == 0:
            zero_pairs_reason = "No notebooks contain declared CV observations."
        elif rejected:
            zero_pairs_reason = "CV and leaderboard metrics are not comparable."
        else:
            zero_pairs_reason = "No finite comparable CV/LB values were available."

    return CvLbDiagnostics(
        notebooks_total=len(notebooks),
        notebooks_with_public_score=with_public_score,
        notebooks_with_declared_cv=with_declared_cv,
        notebooks_with_both=with_both,
        comparable_pairs=len(pairs),
        rejected_non_comparable_pairs=rejected,
        zero_pairs_reason=zero_pairs_reason,
    )


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


def _notebook_cv_metric(notebook: NotebookFacts) -> str | None:
    observation_metrics = {
        metric
        for observation in notebook.declared_cv_observations
        if (metric := _canonical_metric_name(observation.metric_name)) is not None
    }
    if len(observation_metrics) == 1:
        return next(iter(observation_metrics))

    code_metrics = {
        metric
        for observation in notebook.metrics
        if (metric := _canonical_metric_name(observation.name)) is not None
    }
    return next(iter(code_metrics)) if len(code_metrics) == 1 else None


def _metrics_are_comparable(
    notebook_metric_name: str | None,
    competition_metric_name: str | None,
) -> bool:
    notebook_metric = _canonical_metric_name(notebook_metric_name)
    competition_metric = _canonical_metric_name(competition_metric_name)
    return (
        notebook_metric is None
        or competition_metric is None
        or notebook_metric == competition_metric
    )


def _canonical_metric_name(value: str | None) -> str | None:
    if not value:
        return None
    normalized = "".join(character for character in value.lower() if character.isalnum())
    if normalized in {"map", "meanaverageprecision", "averageprecisionscore"}:
        return "mAP"
    if normalized in {"top1", "rank1"}:
        return "rank-1"
    if normalized in {"accuracy", "accuracyscore"}:
        return "accuracy"
    if normalized in {"rocauc", "rocaucscore", "auc"}:
        return "roc_auc"
    if normalized in {"logloss"}:
        return "log_loss"
    if normalized in {"f1", "f1score"}:
        return "f1"
    return value.strip().lower() or None


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
        for left_value, right_value in zip(left_offsets, right_offsets, strict=False)
    )
    correlation = numerator / math.sqrt(left_squared * right_squared)
    return max(-1.0, min(1.0, correlation))
