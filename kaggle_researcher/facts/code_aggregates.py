from __future__ import annotations

import math
import statistics
from collections import Counter
from decimal import Decimal
from typing import Literal

from kaggle_researcher.facts.models import (
    CodeAggregates,
    CodeFamilyUsage,
    CodeObservation,
    KwargDistribution,
    ModelCombination,
    NotebookFacts,
    OptimizationDirection,
)

CODE_AGGREGATE_LIMIT = 25
KWARGS_DISTRIBUTION_LIMIT = 6
MODEL_COMBINATION_LIMIT = 10

_CodeField = Literal["models", "splitters", "metrics", "feature_ops"]
_ParsedKwarg = float | bool | str


def compute_code_aggregates(
    notebooks: list[NotebookFacts],
    *,
    optimization_direction: OptimizationDirection | None = None,
) -> CodeAggregates | None:
    if not notebooks:
        return None

    total_clusters = len({notebook.lineage_cluster_id for notebook in notebooks})
    return CodeAggregates(
        total_clusters=total_clusters,
        total_notebooks=len(notebooks),
        models=_family_usage(
            notebooks,
            "models",
            total_clusters,
            optimization_direction=optimization_direction,
        ),
        splitters=_family_usage(notebooks, "splitters", total_clusters),
        metrics=_family_usage(notebooks, "metrics", total_clusters),
        feature_ops=_family_usage(notebooks, "feature_ops", total_clusters),
        model_combinations=_model_combinations(notebooks),
    )


def _family_usage(
    notebooks: list[NotebookFacts],
    field_name: _CodeField,
    total_clusters: int,
    *,
    optimization_direction: OptimizationDirection | None = None,
) -> list[CodeFamilyUsage]:
    clusters_by_name: dict[str, set[str]] = {}
    notebooks_by_name: dict[str, set[int]] = {}
    public_scores_by_name: dict[str, list[float]] = {}
    kwargs_by_name: dict[
        str,
        dict[str, dict[str, list[_ParsedKwarg]]],
    ] = {}

    for notebook_index, notebook in enumerate(notebooks):
        observations: list[CodeObservation] = getattr(notebook, field_name)
        names_in_notebook = {observation.name for observation in observations}
        for name in names_in_notebook:
            clusters_by_name.setdefault(name, set()).add(notebook.lineage_cluster_id)
            notebooks_by_name.setdefault(name, set()).add(notebook_index)
            score = _finite_float(notebook.public_score)
            if score is not None:
                public_scores_by_name.setdefault(name, []).append(score)

        if field_name == "models":
            for observation in observations:
                for key, raw_value in observation.kwargs.items():
                    parsed = _parse_kwarg(raw_value)
                    if parsed is None:
                        continue
                    kwargs_by_name.setdefault(observation.name, {}).setdefault(
                        key,
                        {},
                    ).setdefault(notebook.lineage_cluster_id, []).append(parsed)

    ordered_names = sorted(
        clusters_by_name,
        key=lambda name: (-len(clusters_by_name[name]), name),
    )[:CODE_AGGREGATE_LIMIT]
    usages: list[CodeFamilyUsage] = []
    for name in ordered_names:
        scores = public_scores_by_name.get(name, [])
        usages.append(
            CodeFamilyUsage(
                name=name,
                cluster_count=len(clusters_by_name[name]),
                notebook_count=len(notebooks_by_name[name]),
                cluster_share=round(len(clusters_by_name[name]) / total_clusters, 4),
                kwargs_distribution=(
                    _kwargs_distribution(kwargs_by_name.get(name, {}))
                    if field_name == "models"
                    else []
                ),
                best_public_score=(
                    min(scores)
                    if scores and optimization_direction == "minimize"
                    else max(scores)
                    if scores
                    else None
                ),
            )
        )
    return usages


def _kwargs_distribution(
    values_by_key: dict[str, dict[str, list[_ParsedKwarg]]],
) -> list[KwargDistribution]:
    candidates: list[KwargDistribution] = []
    for key, values_by_cluster in values_by_key.items():
        cluster_values = [
            representative
            for values in values_by_cluster.values()
            if (representative := _representative_value(values)) is not None
        ]
        if len(cluster_values) < 2:
            continue
        candidates.append(_distribution_for_key(key, cluster_values))

    candidates.sort(key=lambda item: (-item.cluster_count, item.key))
    return candidates[:KWARGS_DISTRIBUTION_LIMIT]


def _distribution_for_key(
    key: str,
    cluster_values: list[_ParsedKwarg],
) -> KwargDistribution:
    distinct_values = len({_kwarg_identity(value) for value in cluster_values})
    if all(_is_numeric(value) for value in cluster_values):
        numeric_values = [float(value) for value in cluster_values]
        is_integer = all(value.is_integer() for value in numeric_values)
        median = (
            float(statistics.median_low(numeric_values))
            if is_integer
            else float(statistics.median(numeric_values))
        )
        return KwargDistribution(
            key=key,
            cluster_count=len(cluster_values),
            median=_format_number(median, force_integer=is_integer),
            minimum=_format_number(min(numeric_values), force_integer=is_integer),
            maximum=_format_number(max(numeric_values), force_integer=is_integer),
            distinct_values=distinct_values,
            is_integer=is_integer,
        )

    representative = _mode(cluster_values)
    return KwargDistribution(
        key=key,
        cluster_count=len(cluster_values),
        median=_format_kwarg(representative),
        distinct_values=distinct_values,
        is_integer=False,
    )


def _representative_value(values: list[_ParsedKwarg]) -> _ParsedKwarg | None:
    if not values:
        return None
    if all(_is_numeric(value) for value in values):
        numeric_values = [float(value) for value in values]
        if all(value.is_integer() for value in numeric_values):
            return float(statistics.median_low(numeric_values))
        return float(statistics.median(numeric_values))
    return _mode(values)


def _parse_kwarg(raw_value: str) -> _ParsedKwarg | None:
    stripped = raw_value.strip()
    normalized = stripped.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    try:
        value = float(stripped)
    except (TypeError, ValueError):
        return stripped if stripped and stripped != "<expr>" else None
    return value if math.isfinite(value) else None


def _format_kwarg(value: _ParsedKwarg) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _format_number(value, force_integer=value.is_integer())
    return value


def _format_number(value: float, *, force_integer: bool) -> str:
    if force_integer:
        return str(int(value))
    if value == 0:
        return "0"
    formatted = format(Decimal(format(value, ".12g")), "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _is_numeric(value: _ParsedKwarg) -> bool:
    return isinstance(value, float)


def _kwarg_identity(value: _ParsedKwarg) -> tuple[str, str]:
    return type(value).__name__, _format_kwarg(value)


def _mode(values: list[_ParsedKwarg]) -> _ParsedKwarg:
    counts = Counter(_kwarg_identity(value) for value in values)
    winner = min(counts, key=lambda identity: (-counts[identity], identity))
    return next(value for value in values if _kwarg_identity(value) == winner)


def _model_combinations(notebooks: list[NotebookFacts]) -> list[ModelCombination]:
    names_by_cluster: dict[str, set[str]] = {}
    for notebook in notebooks:
        names_by_cluster.setdefault(notebook.lineage_cluster_id, set()).update(
            observation.name for observation in notebook.models
        )
    counts = Counter(
        tuple(sorted(names))
        for names in names_by_cluster.values()
        if len(names) >= 2
    )
    combinations = [
        ModelCombination(names=list(names), cluster_count=count)
        for names, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count >= 2
    ]
    return combinations[:MODEL_COMBINATION_LIMIT]


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None
