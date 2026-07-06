from __future__ import annotations

import polars as pl
import pytest

from kaggle_researcher.eda.schemas import ColumnProfile, InferredSchema, TableProfile
from kaggle_researcher.eda.validation.split_helpers import (
    infer_candidate_group_columns,
    infer_candidate_time_columns,
    infer_class_balance,
    infer_regression_target_stats,
    summarize_column_distribution,
)


def test_infer_class_balance_for_binary_target() -> None:
    df = pl.DataFrame({"target": [0, 1, 1, 0, 1]})

    balance = infer_class_balance(df, "target")

    assert balance["n_rows"] == 5
    assert balance["n_classes"] == 2
    assert balance["positive_rate"] == pytest.approx(0.6)


def test_infer_regression_target_stats() -> None:
    df = pl.DataFrame({"target": [1.0, 2.0, 3.0]})

    stats = infer_regression_target_stats(df, "target")

    assert stats["mean"] == pytest.approx(2.0)
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["q50"] == 2.0


def test_candidate_group_and_time_columns_come_from_schema_and_profiles() -> None:
    schema = InferredSchema(
        global_roles={"candidate_join_keys": ["customer_id"]},
        candidate_time_columns=["week_num"],
        confidence="high",
    )
    profile = TableProfile(
        table_name="train",
        path="train.csv",
        n_rows=3,
        n_cols=3,
        columns=[
            ColumnProfile(name="query_id", dtype="String"),
            ColumnProfile(name="event_date", dtype="String", date_min="2024-01-01"),
            ColumnProfile(name="amount", dtype="Float64"),
        ],
    )

    group_candidates = infer_candidate_group_columns(schema, [profile])
    time_candidates = infer_candidate_time_columns(schema, [profile])

    assert [candidate["name"] for candidate in group_candidates] == ["customer_id", "query_id"]
    assert [candidate["name"] for candidate in time_candidates] == ["week_num", "event_date"]


def test_summarize_column_distribution_with_target_mean() -> None:
    df = pl.DataFrame({"segment": ["a", "a", "b"], "target": [0, 1, 1]})

    summary = summarize_column_distribution(df, "segment", target_col="target")

    assert summary == [
        {"value": "a", "n_rows": 2, "pct": pytest.approx(2 / 3), "target_mean": 0.5},
        {"value": "b", "n_rows": 1, "pct": pytest.approx(1 / 3), "target_mean": 1.0},
    ]
