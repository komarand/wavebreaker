from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.validation_analyzer import analyze_validation
from kaggle_researcher.eda.schemas import (
    ColumnProfile,
    EdaTaskPlan,
    InferredSchema,
    MetricEvidence,
    TableProfile,
)


FIXTURE_DIR = Path("tests/fixtures/eda/home_credit_tiny")


def test_home_credit_fixture_with_gini_stability_recommends_temporal_validation() -> None:
    reader = DatasetReader(FIXTURE_DIR)
    schema = InferredSchema(
        target_column="target",
        primary_id_column="case_id",
        prediction_column="score",
        train_base_table="train_base.csv",
        test_base_table="test_base.csv",
        candidate_time_columns=["WEEK_NUM"],
        candidate_date_columns=["date_decision"],
        confidence="high",
    )
    profiles = [
        TableProfile(
            table_name="train_base",
            path="train_base.csv",
            n_rows=8,
            n_cols=9,
            columns=[
                ColumnProfile(name="WEEK_NUM", dtype="Int64"),
                ColumnProfile(name="date_decision", dtype="String", date_min="2024-01-03"),
                ColumnProfile(name="target", dtype="Int64"),
            ],
        )
    ]
    task_plan = EdaTaskPlan(
        competition_id="home_credit_tiny",
        task_type="binary_classification",
        metric={"name": "gini_stability"},
    )
    metric_evidence = analyze_metric(task_plan, schema, profiles)

    evidence = analyze_validation(schema, profiles, metric_evidence, reader)

    assert evidence.primary_validation["method"] == "temporal_holdout"
    assert evidence.primary_validation["split_column"] == "WEEK_NUM"
    assert evidence.target_available is True
    assert evidence.class_balance["positive_rate"] == 0.5
    assert evidence.target_by_period
    assert evidence.oot_holdout["method"] == "temporal_holdout"


def test_binary_iid_with_date_column_recommends_stratified_kfold(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "id,event_date,target\n"
        "1,2024-01-01,0\n"
        "2,2024-01-02,1\n"
        "3,2024-01-03,0\n",
        encoding="utf-8",
    )
    reader = DatasetReader(tmp_path)
    schema = InferredSchema(
        target_column="target",
        primary_id_column="id",
        train_base_table="train_base.csv",
        candidate_date_columns=["event_date"],
        confidence="high",
    )
    metric_evidence = MetricEvidence(
        metric_name="auc",
        task_type="binary_classification",
        metric_family="rank_classification",
        requires_probabilities=True,
    )

    evidence = analyze_validation(schema, [], metric_evidence, reader)

    assert evidence.primary_validation["method"] == "stratified_kfold"
    assert evidence.diagnostic_validations[0]["method"] == "temporal_holdout"
    assert evidence.rejected_validations[0]["method"] == "temporal_holdout_as_default"
    assert evidence.class_balance["n_classes"] == 2


def test_regression_fixture_recommends_kfold(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "id,target,feature\n1,10.0,3\n2,12.0,4\n3,14.0,5\n",
        encoding="utf-8",
    )
    reader = DatasetReader(tmp_path)
    schema = InferredSchema(
        target_column="target",
        primary_id_column="id",
        train_base_table="train_base.csv",
        confidence="high",
    )
    metric_evidence = MetricEvidence(
        metric_name="rmse",
        task_type="regression",
        metric_family="regression_error",
    )

    evidence = analyze_validation(schema, [], metric_evidence, reader)

    assert evidence.primary_validation["method"] == "kfold"
    assert evidence.target_summary["mean"] == 12.0
    assert evidence.class_balance == {}


def test_grouped_fixture_recommends_stratified_group_kfold(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "row_id,customer_group,target\n1,g1,0\n2,g1,1\n3,g2,0\n4,g3,1\n",
        encoding="utf-8",
    )
    reader = DatasetReader(tmp_path)
    schema = InferredSchema(
        target_column="target",
        primary_id_column="row_id",
        train_base_table="train_base.csv",
        candidate_group_columns=["customer_group"],
        confidence="high",
    )
    metric_evidence = MetricEvidence(
        metric_name="logloss",
        task_type="binary_classification",
        metric_family="probabilistic_classification",
        requires_probabilities=True,
    )

    evidence = analyze_validation(schema, [], metric_evidence, reader)

    assert evidence.primary_validation["method"] == "stratified_group_kfold"
    assert evidence.primary_validation["group_column"] == "customer_group"
    assert evidence.target_by_group


def test_ranking_fixture_recommends_query_aware_validation(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "query_id,item_id,target\nq1,i1,1\nq1,i2,0\nq2,i3,1\n",
        encoding="utf-8",
    )
    reader = DatasetReader(tmp_path)
    schema = InferredSchema(
        target_column="target",
        train_base_table="train_base.csv",
        candidate_group_columns=["query_id"],
        confidence="high",
    )
    metric_evidence = MetricEvidence(
        metric_name="ndcg",
        task_type="ranking",
        metric_family="ranking",
        requires_query_groups=True,
    )

    evidence = analyze_validation(schema, [], metric_evidence, reader)

    assert evidence.primary_validation["method"] == "ranking_group_cv"
    assert evidence.primary_validation["group_column"] == "query_id"
    assert evidence.query_columns[0]["name"] == "query_id"


def test_temporal_metric_without_time_column_returns_warning_and_limitation(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "id,target\n1,0\n2,1\n",
        encoding="utf-8",
    )
    reader = DatasetReader(tmp_path)
    schema = InferredSchema(
        target_column="target",
        primary_id_column="id",
        train_base_table="train_base.csv",
        confidence="medium",
    )
    metric_evidence = MetricEvidence(
        metric_name="gini_stability",
        task_type="binary_classification",
        metric_family="temporal_stability",
        requires_time=True,
    )

    evidence = analyze_validation(schema, [], metric_evidence, reader)

    assert evidence.primary_validation["method"] == "custom_required"
    assert evidence.confidence == "low"
    assert evidence.warnings
    assert evidence.limitations
