from __future__ import annotations

from kaggle_researcher.eda.metrics import MetricFamily, MetricSpec, TaskType, infer_metric_spec
from kaggle_researcher.eda.schemas import InferredSchema
from kaggle_researcher.eda.validation import select_validation_policy


def _schema(
    *,
    time_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    group_columns: list[str] | None = None,
    join_keys: list[str] | None = None,
) -> InferredSchema:
    return InferredSchema(
        global_roles={"candidate_join_keys": join_keys or []},
        candidate_time_columns=time_columns or [],
        candidate_date_columns=date_columns or [],
        candidate_group_columns=group_columns or [],
        confidence="high",
    )


def _assert_policy_shape(decision: dict) -> None:
    assert set(decision) == {
        "primary_validation",
        "diagnostic_validations",
        "rejected_validations",
        "confidence",
        "evidence_refs",
        "warnings",
        "limitations",
        "reasoning_summary",
    }


def test_iid_binary_classification_selects_stratified_kfold() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("auc", "binary_classification"),
        inferred_schema=_schema(),
    )

    _assert_policy_shape(decision)
    assert decision["primary_validation"]["method"] == "stratified_kfold"
    assert decision["diagnostic_validations"] == []
    assert decision["rejected_validations"] == []
    assert decision["confidence"] == "high"


def test_iid_regression_selects_kfold() -> None:
    decision = select_validation_policy(
        task_type=TaskType.REGRESSION,
        metric_spec=infer_metric_spec("rmse", "regression"),
        inferred_schema=_schema(),
    )

    assert decision["primary_validation"]["method"] == "kfold"
    assert decision["confidence"] == "high"


def test_time_column_alone_is_diagnostic_not_primary_temporal_cv() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("auc", "binary_classification"),
        inferred_schema=_schema(time_columns=["week"], date_columns=["event_date"]),
    )

    assert decision["primary_validation"]["method"] == "stratified_kfold"
    assert decision["confidence"] == "medium"
    assert decision["diagnostic_validations"][0]["method"] == "temporal_holdout"
    assert decision["diagnostic_validations"][0]["split_column"] == "week"
    assert decision["rejected_validations"][0]["method"] == "temporal_holdout_as_default"


def test_group_entity_risk_selects_stratified_group_kfold_for_classification() -> None:
    decision = select_validation_policy(
        task_type="binary_classification",
        metric_spec=infer_metric_spec("logloss", "binary_classification"),
        inferred_schema=_schema(group_columns=["customer_id"]),
    )

    assert decision["primary_validation"]["method"] == "stratified_group_kfold"
    assert decision["primary_validation"]["group_column"] == "customer_id"


def test_group_entity_risk_selects_group_kfold_for_regression() -> None:
    decision = select_validation_policy(
        task_type="regression",
        metric_spec=infer_metric_spec("mae", "regression"),
        inferred_schema=_schema(group_columns=["user_id"]),
    )

    assert decision["primary_validation"]["method"] == "group_kfold"
    assert decision["primary_validation"]["group_column"] == "user_id"


def test_ranking_query_task_selects_ranking_group_cv() -> None:
    decision = select_validation_policy(
        task_type=TaskType.RANKING,
        metric_spec=infer_metric_spec("ndcg", "ranking"),
        inferred_schema=_schema(group_columns=["session_id", "query_id"]),
    )

    assert decision["primary_validation"]["method"] == "ranking_group_cv"
    assert decision["primary_validation"]["group_column"] == "query_id"
    assert decision["confidence"] == "high"


def test_temporal_stability_metric_selects_temporal_policy_when_time_column_exists() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("gini_stability", "binary_classification"),
        inferred_schema=_schema(time_columns=["week"]),
    )

    assert decision["primary_validation"]["method"] == "temporal_holdout"
    assert decision["primary_validation"]["split_column"] == "week"
    assert decision["confidence"] == "high"
    assert "metric_evidence.requires_time" in decision["evidence_refs"]


def test_forecasting_task_selects_temporal_policy_without_stability_metric() -> None:
    decision = select_validation_policy(
        task_type=TaskType.FORECASTING_TABULAR,
        metric_spec=MetricSpec(
            name="rmse",
            family=MetricFamily.REGRESSION_ERROR,
            task_types=[TaskType.FORECASTING_TABULAR],
            greater_is_better=False,
            supports_local_eval=True,
        ),
        inferred_schema=_schema(date_columns=["date"]),
    )

    assert decision["primary_validation"]["method"] == "temporal_holdout"
    assert decision["primary_validation"]["split_column"] == "date"


def test_temporal_required_without_time_column_returns_custom_required() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("gini_stability", "binary_classification"),
        inferred_schema=_schema(),
    )

    assert decision["primary_validation"]["method"] == "custom_required"
    assert decision["warnings"] == [
        "Temporal validation is required, but no time/date column was inferred."
    ]


def test_unknown_metric_selects_custom_required() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("mystery_metric", "binary_classification"),
        inferred_schema=_schema(),
    )

    assert decision["primary_validation"]["method"] == "custom_required"
    assert decision["confidence"] == "low"
    assert decision["warnings"] == [
        "Metric requires custom validation review before choosing folds."
    ]
