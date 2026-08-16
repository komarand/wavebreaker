from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Literal

from kaggle_researcher.facts.models import (
    CodeAggregates,
    CodeFamilyUsage,
    CodeObservation,
    ModelCombination,
    NotebookFacts,
    OptimizationDirection,
)

CODE_AGGREGATE_LIMIT = 25
TYPICAL_KWARGS_LIMIT = 6
MODEL_COMBINATION_LIMIT = 10

_CodeField = Literal["models", "splitters", "metrics", "feature_ops"]
_ParsedKwarg = float | bool


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
                typical_kwargs=(
                    _typical_kwargs(kwargs_by_name.get(name, {}))
                    if field_name == "models"
                    else {}
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


def _typical_kwargs(
    values_by_key: dict[str, dict[str, list[_ParsedKwarg]]],
) -> dict[str, str]:
    candidates: list[tuple[str, int, _ParsedKwarg]] = []
    for key, values_by_cluster in values_by_key.items():
        cluster_values = [
            representative
            for values in values_by_cluster.values()
            if (representative := _representative_value(values)) is not None
        ]
        if len(cluster_values) < 2:
            continue
        representative = _representative_value(cluster_values)
        if representative is not None:
            candidates.append((key, len(cluster_values), representative))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    return {
        key: _format_kwarg(value)
        for key, _, value in candidates[:TYPICAL_KWARGS_LIMIT]
    }


def _representative_value(values: list[_ParsedKwarg]) -> _ParsedKwarg | None:
    if not values:
        return None
    if all(isinstance(value, bool) for value in values):
        counts = Counter(values)
        return sorted(counts, key=lambda value: (-counts[value], str(value)))[0]
    if all(isinstance(value, float) for value in values):
        return statistics.median(values)
    return None


def _parse_kwarg(raw_value: str) -> _ParsedKwarg | None:
    normalized = raw_value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _format_kwarg(value: _ParsedKwarg) -> str:
    if isinstance(value, bool):
        return str(value)
    return format(value, ".12g")


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
