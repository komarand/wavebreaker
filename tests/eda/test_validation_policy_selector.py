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


def test_iid_binary_classification_selects_stratified_kfold() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("auc", "binary_classification"),
        inferred_schema=_schema(),
    )

    assert decision.primary_validation["method"] == "StratifiedKFold"
    assert decision.diagnostic_validations == []
    assert decision.rejected_validations == []
    assert decision.confidence == "high"


def test_iid_regression_selects_kfold() -> None:
    decision = select_validation_policy(
        task_type=TaskType.REGRESSION,
        metric_spec=infer_metric_spec("rmse", "regression"),
        inferred_schema=_schema(),
    )

    assert decision.primary_validation["method"] == "KFold"
    assert decision.confidence == "high"


def test_time_column_alone_is_diagnostic_not_primary_temporal_cv() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("auc", "binary_classification"),
        inferred_schema=_schema(time_columns=["WEEK_NUM"], date_columns=["date_decision"]),
    )

    assert decision.primary_validation["method"] == "StratifiedKFold"
    assert decision.confidence == "medium"
    assert decision.diagnostic_validations == [
        {
            "method": "temporal_holdout_diagnostic",
            "split_column": "WEEK_NUM",
            "reason": "Time-like columns exist, but they are diagnostic unless metric/task/scout/drift evidence requires temporal validation.",
        }
    ]
    assert decision.rejected_validations == [
        {
            "method": "primary_temporal_cv_from_time_column_only",
            "reason": "A time column alone is not sufficient evidence for primary temporal CV.",
        }
    ]
    assert "inferred_schema.candidate_time_columns" in decision.evidence_refs


def test_group_entity_risk_selects_stratified_group_kfold_for_classification() -> None:
    decision = select_validation_policy(
        task_type="binary_classification",
        metric_spec=infer_metric_spec("logloss", "binary_classification"),
        inferred_schema=_schema(group_columns=["customer_id"]),
    )

    assert decision.primary_validation["method"] == "StratifiedGroupKFold"
    assert decision.primary_validation["group_column"] == "customer_id"
    assert decision.confidence == "high"


def test_group_entity_risk_selects_group_kfold_for_regression() -> None:
    decision = select_validation_policy(
        task_type="regression",
        metric_spec=infer_metric_spec("mae", "regression"),
        inferred_schema=_schema(group_columns=["user_id"]),
    )

    assert decision.primary_validation["method"] == "GroupKFold"
    assert decision.primary_validation["group_column"] == "user_id"


def test_ranking_query_task_selects_group_kfold_by_query_id() -> None:
    decision = select_validation_policy(
        task_type=TaskType.RANKING,
        metric_spec=infer_metric_spec("ndcg", "ranking"),
        inferred_schema=_schema(group_columns=["session_id", "query_id"]),
    )

    assert decision.primary_validation["method"] == "GroupKFold"
    assert decision.primary_validation["group_column"] == "query_id"
    assert decision.confidence == "high"


def test_temporal_stability_metric_selects_temporal_policy_when_time_column_exists() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("gini_stability", "binary_classification"),
        inferred_schema=_schema(time_columns=["WEEK_NUM"]),
    )

    assert decision.primary_validation["method"] == "temporal_holdout_and_expanding_cv"
    assert decision.primary_validation["split_column"] == "WEEK_NUM"
    assert decision.confidence == "high"
    assert "metric_evidence.requires_time_or_groups" in decision.evidence_refs


def test_forecasting_task_selects_temporal_policy_without_stability_metric() -> None:
    decision = select_validation_policy(
        task_type=TaskType.TIME_SERIES,
        metric_spec=MetricSpec(
            name="rmse",
            family=MetricFamily.REGRESSION,
            task_types=(TaskType.TIME_SERIES,),
            greater_is_better=False,
            local_metric_available=True,
        ),
        inferred_schema=_schema(date_columns=["date"]),
    )

    assert decision.primary_validation["method"] == "temporal_holdout_and_expanding_cv"
    assert decision.primary_validation["split_column"] == "date"


def test_scout_temporal_hypothesis_can_promote_temporal_policy() -> None:
    decision = select_validation_policy(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric_spec=infer_metric_spec("auc", "binary_classification"),
        inferred_schema=_schema(time_columns=["WEEK_NUM"]),
        scout_hypotheses=[
            {
                "hypothesis_id": "val_001",
                "claim": "Use out-of-time validation because labels drift over periods.",
            }
        ],
    )

    assert decision.primary_validation["method"] == "temporal_holdout_and_expanding_cv"
    assert "research_hypotheses.validation" in decision.evidence_refs


def test_drift_evidence_can_promote_temporal_policy() -> None:
    decision = select_validation_policy(
        task_type=TaskType.REGRESSION,
        metric_spec=infer_metric_spec("rmse", "regression"),
        inferred_schema=_schema(time_columns=["month"]),
        drift_evidence={"temporal_drift": {"severity": "high", "finding": "period shift"}},
    )

    assert decision.primary_validation["method"] == "temporal_holdout_and_expanding_cv"
    assert "drift_evidence" in decision.evidence_refs
