from __future__ import annotations

from collections import defaultdict
from typing import Any

from kaggle_researcher.eda.validation.split_helpers import (
    _column_values,
    _is_missing,
    _ratio,
)


def summarize_group_counts(
    df: Any,
    group_col: str,
    target_col: str | None = None,
) -> dict[str, Any]:
    groups = _column_values(df, group_col)
    targets = _column_values(df, target_col) if target_col is not None else None
    grouped: dict[Any, list[Any]] = defaultdict(list)
    for index, group in enumerate(groups):
        if _is_missing(group):
            continue
        grouped[group].append(targets[index] if targets is not None else None)

    group_sizes = [len(values) for values in grouped.values()]
    result: dict[str, Any] = {
        "group_col": group_col,
        "n_rows": sum(group_sizes),
        "n_groups": len(grouped),
        "min_group_size": min(group_sizes) if group_sizes else 0,
        "max_group_size": max(group_sizes) if group_sizes else 0,
        "mean_group_size": (sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0,
    }
    if target_col is not None:
        target_means = []
        for group, values in grouped.items():
            numeric_targets = [float(value) for value in values if not _is_missing(value)]
            target_means.append(
                {
                    "group": group,
                    "n_rows": len(values),
                    "target_mean": (
                        sum(numeric_targets) / len(numeric_targets)
                        if numeric_targets
                        else None
                    ),
                }
            )
        result["target_by_group"] = target_means
    return result


def assess_group_split_feasibility(
    df: Any,
    group_col: str,
    target_col: str | None = None,
) -> dict[str, Any]:
    summary = summarize_group_counts(df, group_col, target_col=target_col)
    n_groups = int(summary["n_groups"])
    feasible = n_groups >= 2
    recommended_n_splits = min(5, n_groups) if feasible else 0
    warnings: list[str] = []
    if not feasible:
        warnings.append("At least two groups are required for group-based validation.")
    if feasible and n_groups < 5:
        warnings.append("Few groups are available; use fewer folds or a holdout split.")
    return {
        "group_col": group_col,
        "feasible": feasible,
        "reason": None if feasible else "Too few groups for group split.",
        "n_groups": n_groups,
        "recommended_n_splits": recommended_n_splits,
        "summary": summary,
        "warnings": warnings,
    }


def detect_group_leakage_risk(
    train_df: Any,
    test_df: Any,
    group_col: str,
) -> dict[str, Any]:
    train_groups = set(_non_missing(_column_values(train_df, group_col)))
    test_groups = set(_non_missing(_column_values(test_df, group_col)))
    overlap = sorted(train_groups & test_groups, key=lambda value: str(value))
    return {
        "group_col": group_col,
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "overlap_groups": overlap,
        "n_overlap_groups": len(overlap),
        "overlap_pct_of_test": _ratio(len(overlap), len(test_groups)),
        "leakage_risk": "high" if overlap else "low",
    }


def _non_missing(values: list[Any]) -> list[Any]:
    return [value for value in values if not _is_missing(value)]


__all__ = [
    "assess_group_split_feasibility",
    "detect_group_leakage_risk",
    "summarize_group_counts",
]
