from __future__ import annotations

from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.schemas import (
    ColumnProfile,
    EdaTaskPlan,
    InferredSchema,
    TableProfile,
)


def _schema(
    *,
    target_column: str | None = "target",
    prediction_column: str | None = "score",
    time_columns: list[str] | None = None,
    group_columns: list[str] | None = None,
) -> InferredSchema:
    return InferredSchema(
        target_column=target_column,
        prediction_column=prediction_column,
        candidate_time_columns=time_columns or [],
        candidate_group_columns=group_columns or [],
        confidence="high",
    )


def _profile_with_columns(*columns: str) -> TableProfile:
    return TableProfile(
        table_name="train",
        path="train.csv",
        n_rows=10,
        n_cols=len(columns),
        columns=[ColumnProfile(name=column, dtype="Int64") for column in columns],
    )


def _task_plan(metric_name: str, task_type: str = "binary_classification") -> EdaTaskPlan:
    return EdaTaskPlan(
        competition_id="generic_competition",
        task_type=task_type,
        metric={"name": metric_name},
    )


def test_auc_metric_evidence_sets_generic_rank_classification_fields() -> None:
    evidence = analyze_metric(_task_plan("roc_auc"), _schema(), [])

    assert evidence.metric_name == "auc"
    assert evidence.normalized_metric_name == "auc"
    assert evidence.task_type == "binary_classification"
    assert evidence.metric_family == "rank_classification"
    assert evidence.rank_based is True
    assert evidence.requires_probabilities is True
    assert evidence.requires_time is False
    assert evidence.requires_time_or_groups is False
    assert evidence.prediction_output_type == "probability"
    assert evidence.required_columns["prediction"] == "score"
    assert evidence.warnings == []


def test_logloss_evidence_requires_probabilities_and_calibration() -> None:
    evidence = analyze_metric(_task_plan("log_loss"), _schema(prediction_column=None), [])

    assert evidence.metric_name == "logloss"
    assert evidence.metric_family == "probabilistic_classification"
    assert evidence.requires_probabilities is True
    assert evidence.requires_calibration is True
    assert evidence.required_columns["prediction"] == "probability"


def test_f1_evidence_requires_threshold_search() -> None:
    evidence = analyze_metric(_task_plan("f1"), _schema(), [])

    assert evidence.metric_family == "threshold_classification"
    assert evidence.requires_threshold is True
    assert evidence.threshold_search_needed is True
    assert evidence.prediction_output_type == "label"


def test_ranking_metric_evidence_requires_query_groups() -> None:
    evidence = analyze_metric(
        _task_plan("ndcg", task_type="ranking"),
        _schema(group_columns=["query_id"]),
        [_profile_with_columns("query_id", "item_id")],
    )

    assert evidence.metric_family == "ranking"
    assert evidence.requires_query_groups is True
    assert evidence.requires_time_or_groups is True
    assert evidence.required_columns["group"] == "query_id"
    assert evidence.local_metric_available is False
    assert evidence.prediction_output_type == "ranked_score"


def test_gini_stability_evidence_requires_time_without_promoting_validation_policy() -> None:
    evidence = analyze_metric(
        _task_plan("gini_stability"),
        _schema(time_columns=["period"]),
        [_profile_with_columns("period", "target", "score")],
    )

    assert evidence.metric_name == "gini_stability"
    assert evidence.metric_family == "temporal_stability"
    assert evidence.base_metric == "gini"
    assert evidence.requires_time is True
    assert evidence.requires_time_or_groups is True
    assert evidence.required_columns["time"] == "period"
    assert set(evidence.components) == {
        "weekly_gini",
        "trend_penalty",
        "residual_std_penalty",
    }
    assert evidence.warnings == []


def test_gini_stability_warns_when_time_column_is_missing() -> None:
    evidence = analyze_metric(_task_plan("gini_stability"), _schema(), [])

    assert evidence.required_columns["time"] is None
    assert "gini_stability requires a time/period column, but none was found." in evidence.warnings


def test_unknown_metric_is_handled_gracefully() -> None:
    evidence = analyze_metric(_task_plan("mystery_metric"), _schema(), [])

    assert evidence.metric_name == "mystery_metric"
    assert evidence.metric_family == "unknown"
    assert evidence.local_metric_available is False
    assert evidence.needs_custom_implementation is True
    assert evidence.requires_probabilities is False
    assert evidence.warnings == [
        "Metric 'mystery_metric' requires manual metric implementation or verification."
    ]
