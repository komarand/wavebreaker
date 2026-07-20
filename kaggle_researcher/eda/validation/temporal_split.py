from __future__ import annotations

from typing import Any

from kaggle_researcher.eda.validation.split_helpers import (
    _column_values,
    _is_missing,
    _natural_sort_key,
    _ratio,
)


def infer_periods(df: Any, time_col: str) -> list[Any]:
    values = [
        value
        for value in _column_values(df, time_col)
        if not _is_missing(value)
    ]
    return sorted(set(values), key=_natural_sort_key)


def build_latest_period_holdout(
    periods: list[Any],
    holdout_period_count: int = 4,
) -> dict[str, Any]:
    if holdout_period_count <= 0:
        raise ValueError("holdout_period_count must be positive")
    ordered_periods = sorted(set(periods), key=_natural_sort_key)
    if len(ordered_periods) <= holdout_period_count:
        return {
            "method": "temporal_holdout",
            "feasible": False,
            "reason": "Too few periods to leave a non-empty train set.",
            "n_periods": len(ordered_periods),
            "train_periods": [],
            "holdout_periods": ordered_periods,
        }
    return {
        "method": "temporal_holdout",
        "feasible": True,
        "n_periods": len(ordered_periods),
        "train_periods": ordered_periods[:-holdout_period_count],
        "holdout_periods": ordered_periods[-holdout_period_count:],
    }


def build_expanding_window_folds(
    periods: list[Any],
    n_folds: int = 5,
    min_train_periods: int = 3,
) -> list[dict[str, Any]]:
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if min_train_periods <= 0:
        raise ValueError("min_train_periods must be positive")
    ordered_periods = sorted(set(periods), key=_natural_sort_key)
    if len(ordered_periods) <= min_train_periods:
        return [
            {
                "method": "expanding_window",
                "feasible": False,
                "reason": "Too few periods to build an expanding window split.",
                "n_periods": len(ordered_periods),
                "min_train_periods": min_train_periods,
            }
        ]

    validation_periods = ordered_periods[min_train_periods:]
    validation_periods = validation_periods[-n_folds:]
    folds: list[dict[str, Any]] = []
    for fold_index, validation_period in enumerate(validation_periods, start=1):
        validation_position = ordered_periods.index(validation_period)
        folds.append(
            {
                "method": "expanding_window",
                "feasible": True,
                "fold": fold_index,
                "train_periods": ordered_periods[:validation_position],
                "validation_periods": [validation_period],
            }
        )
    return folds


def summarize_period_counts(
    df: Any,
    time_col: str,
    target_col: str | None = None,
) -> list[dict[str, Any]]:
    periods = _column_values(df, time_col)
    targets = _column_values(df, target_col) if target_col is not None else None
    grouped: dict[Any, list[Any]] = {}
    for index, period in enumerate(periods):
        if _is_missing(period):
            continue
        grouped.setdefault(period, []).append(targets[index] if targets is not None else None)
    total = sum(len(values) for values in grouped.values())
    rows: list[dict[str, Any]] = []
    for period in sorted(grouped, key=_natural_sort_key):
        values = grouped[period]
        row: dict[str, Any] = {
            "period": period,
            "n_rows": len(values),
            "pct": _ratio(len(values), total),
        }
        if target_col is not None:
            numeric_targets = [
                float(value)
                for value in values
                if not _is_missing(value)
            ]
            row["target_mean"] = (
                sum(numeric_targets) / len(numeric_targets)
                if numeric_targets
                else None
            )
        rows.append(row)
    return rows


__all__ = [
    "build_expanding_window_folds",
    "build_latest_period_holdout",
    "infer_periods",
    "summarize_period_counts",
]
