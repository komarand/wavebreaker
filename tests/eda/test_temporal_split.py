from __future__ import annotations

import polars as pl
import pytest

from kaggle_researcher.eda.validation.temporal_split import (
    build_expanding_window_folds,
    build_latest_period_holdout,
    infer_periods,
    summarize_period_counts,
)


def test_infer_periods_returns_unique_naturally_sorted_periods() -> None:
    df = pl.DataFrame({"week": [3, 1, 2, 2, None]})

    assert infer_periods(df, "week") == [1, 2, 3]


def test_latest_period_holdout_uses_latest_periods() -> None:
    result = build_latest_period_holdout([1, 2, 3, 4, 5], holdout_period_count=2)

    assert result["feasible"] is True
    assert result["train_periods"] == [1, 2, 3]
    assert result["holdout_periods"] == [4, 5]


def test_latest_period_holdout_is_infeasible_with_too_few_periods() -> None:
    result = build_latest_period_holdout([1, 2], holdout_period_count=2)

    assert result["feasible"] is False
    assert result["reason"] == "Too few periods to leave a non-empty train set."


def test_expanding_window_folds_are_deterministic() -> None:
    folds = build_expanding_window_folds([1, 2, 3, 4, 5], n_folds=2, min_train_periods=2)

    assert folds == [
        {
            "method": "expanding_window",
            "feasible": True,
            "fold": 1,
            "train_periods": [1, 2, 3],
            "validation_periods": [4],
        },
        {
            "method": "expanding_window",
            "feasible": True,
            "fold": 2,
            "train_periods": [1, 2, 3, 4],
            "validation_periods": [5],
        },
    ]


def test_expanding_window_is_infeasible_with_too_few_periods() -> None:
    folds = build_expanding_window_folds([1, 2], n_folds=3, min_train_periods=2)

    assert folds[0]["feasible"] is False
    assert folds[0]["reason"] == "Too few periods to build an expanding window split."


def test_summarize_period_counts_includes_target_mean() -> None:
    df = pl.DataFrame({"week": [1, 1, 2], "target": [0, 1, 1]})

    summary = summarize_period_counts(df, "week", target_col="target")

    assert summary == [
        {"period": 1, "n_rows": 2, "pct": pytest.approx(2 / 3), "target_mean": 0.5},
        {"period": 2, "n_rows": 1, "pct": pytest.approx(1 / 3), "target_mean": 1.0},
    ]
