from __future__ import annotations

import polars as pl

from kaggle_researcher.eda.validation.group_split import (
    assess_group_split_feasibility,
    detect_group_leakage_risk,
    summarize_group_counts,
)


def test_summarize_group_counts_reports_group_sizes_and_target_means() -> None:
    df = pl.DataFrame({"user_id": ["u1", "u1", "u2"], "target": [0, 1, 1]})

    summary = summarize_group_counts(df, "user_id", target_col="target")

    assert summary["n_rows"] == 3
    assert summary["n_groups"] == 2
    assert summary["min_group_size"] == 1
    assert summary["max_group_size"] == 2
    assert summary["target_by_group"][0]["target_mean"] == 0.5


def test_group_split_feasibility_requires_at_least_two_groups() -> None:
    infeasible = assess_group_split_feasibility(
        pl.DataFrame({"user_id": ["u1", "u1"]}),
        "user_id",
    )
    feasible = assess_group_split_feasibility(
        pl.DataFrame({"user_id": ["u1", "u2", "u3"]}),
        "user_id",
    )

    assert infeasible["feasible"] is False
    assert infeasible["reason"] == "Too few groups for group split."
    assert feasible["feasible"] is True
    assert feasible["recommended_n_splits"] == 3


def test_detect_group_leakage_risk_reports_train_test_overlap() -> None:
    train_df = pl.DataFrame({"user_id": ["u1", "u2", "u3"]})
    test_df = pl.DataFrame({"user_id": ["u3", "u4"]})

    result = detect_group_leakage_risk(train_df, test_df, "user_id")

    assert result["overlap_groups"] == ["u3"]
    assert result["n_overlap_groups"] == 1
    assert result["overlap_pct_of_test"] == 0.5
    assert result["leakage_risk"] == "high"
