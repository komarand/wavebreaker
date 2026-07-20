from __future__ import annotations

import json
from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.baseline_runner import run_baseline
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.modules.validation_analyzer import analyze_validation
from kaggle_researcher.eda.schemas import (
    EdaTaskPlan,
    InferredSchema,
    LeakageCheckResult,
    MetricEvidence,
    ValidationEvidence,
)


FIXTURE_ROOT = Path("tests/fixtures/eda")


def test_binary_classification_fixture_baseline_runs(tmp_path: Path) -> None:
    schema, validation, metric, reader = _fixture_inputs("iid_binary_tiny")

    evidence = run_baseline(schema, validation, metric, [], reader, tmp_path)

    assert evidence["status"] == "completed"
    assert evidence["task_type"] == "binary_classification"
    assert evidence["validation_policy"]["method"] == "stratified_kfold"
    assert evidence["metric_name"] == "auc"
    assert evidence["metric_value"] is not None
    assert evidence["fold_results"]
    assert "target" not in evidence["feature_columns"]
    assert "row_id" not in evidence["feature_columns"]
    assert "event_date" not in evidence["feature_columns"]
    policy = evidence["preprocessing_policy"]
    assert policy["policy_version"] == "1.0"
    assert policy["fit_scope"] == "inside_cv_folds"
    assert policy["leakage_safe"] is True
    assert policy["safety_checks"]["fits_preprocessing_inside_folds"] is True
    assert policy["safety_checks"]["fits_preprocessing_on_full_train_before_cv"] is False
    assert policy["imputation"]["numeric"] == "median"
    assert policy["imputation"]["categorical"] == "most_frequent"
    assert policy["categorical_encoding"]["method"] == "one_hot"
    assert policy["categorical_encoding"]["handle_unknown"] == "ignore"
    assert policy["high_cardinality_handling"]["target_encoding"] == "disabled"
    assert {"target", "row_id", "event_date"} <= set(policy["excluded_columns"])
    reasons = {item["column"]: item["reason"] for item in evidence["excluded_column_details"]}
    assert reasons["target"] == "target_column"
    assert reasons["row_id"] == "primary_id"
    assert reasons["event_date"] == "time_column"
    assert policy["validation_split_policy"]["method"] == "stratified_kfold"
    assert Path(evidence["artifacts"]["oof_predictions"]).is_file()


def test_regression_fixture_baseline_runs_with_kfold(tmp_path: Path) -> None:
    schema, validation, metric, reader = _fixture_inputs("regression_tiny")

    evidence = run_baseline(schema, validation, metric, [], reader, tmp_path)

    assert evidence["status"] == "completed"
    assert evidence["task_type"] == "regression"
    assert evidence["validation_policy"]["method"] == "kfold"
    assert evidence["metric_name"] == "rmse"
    assert evidence["metric_value"] is not None
    assert "target" not in evidence["feature_columns"]
    assert "id" not in evidence["feature_columns"]


def test_ranking_and_survival_return_skipped_not_failure(tmp_path: Path) -> None:
    reader = DatasetReader(FIXTURE_ROOT / "ranking_tiny")
    schema = InferredSchema(
        target_column="target",
        primary_id_column="item_id",
        train_base_table="train_base.csv",
        confidence="high",
    )

    ranking = run_baseline(
        schema,
        ValidationEvidence(primary_validation={"method": "ranking_group_cv"}),
        MetricEvidence(
            metric_name="ndcg",
            task_type="ranking",
            metric_family="ranking",
            local_metric_available=False,
        ),
        [],
        reader,
        tmp_path,
    )
    survival = run_baseline(
        schema,
        ValidationEvidence(primary_validation={"method": "kfold"}),
        MetricEvidence(
            metric_name="concordance_index",
            task_type="survival",
            metric_family="survival",
            local_metric_available=False,
        ),
        [],
        reader,
        tmp_path,
    )

    assert ranking["status"] == "skipped"
    assert "not supported" in ranking["reason"]
    assert survival["status"] == "skipped"
    assert "not supported" in survival["reason"]


def test_feature_selection_excludes_target_id_group_and_critical_leakage_columns(
    tmp_path: Path,
) -> None:
    (tmp_path / "train_base.csv").write_text(
        "id,customer_id,target,feature_num,leaky_feature,event_date\n"
        "1,c1,0,10,0,2024-01-01\n"
        "2,c1,1,20,1,2024-01-02\n"
        "3,c2,0,30,0,2024-01-03\n"
        "4,c2,1,40,1,2024-01-04\n",
        encoding="utf-8",
    )
    reader = DatasetReader(tmp_path)
    schema = InferredSchema(
        target_column="target",
        primary_id_column="id",
        train_base_table="train_base.csv",
        confidence="high",
    )
    validation = ValidationEvidence(
        group_columns=[{"name": "customer_id"}],
        primary_validation={"method": "group_kfold", "group_column": "customer_id"},
    )
    metric = MetricEvidence(
        metric_name="accuracy",
        task_type="binary_classification",
        metric_family="threshold_classification",
        requires_threshold=True,
        local_metric_available=True,
    )
    leakage = [
        LeakageCheckResult(
            check_id="target_like_columns",
            status="failed",
            severity="critical",
            finding="Critical leakage-like feature.",
            evidence={"columns": [{"column": "leaky_feature"}]},
        )
    ]

    evidence = run_baseline(schema, validation, metric, leakage, reader, tmp_path / "baseline")

    assert "feature_num" in evidence["feature_columns"]
    assert "target" not in evidence["feature_columns"]
    assert "id" not in evidence["feature_columns"]
    assert "customer_id" not in evidence["feature_columns"]
    assert "leaky_feature" not in evidence["feature_columns"]
    assert "event_date" not in evidence["feature_columns"]
    assert {"target", "id", "customer_id", "leaky_feature", "event_date"} <= set(
        evidence["excluded_columns"]
    )
    reasons = {item["column"]: item["reason"] for item in evidence["excluded_column_details"]}
    assert reasons["target"] == "target_column"
    assert reasons["id"] == "primary_id"
    assert reasons["customer_id"] == "group_column"
    assert reasons["leaky_feature"] == "leakage_risk"
    assert reasons["event_date"] == "time_column"
    assert evidence["validation_policy"]["method"] == "group_kfold"


def test_preprocessing_policy_describes_numeric_categorical_high_card_and_text(
    tmp_path: Path,
) -> None:
    reader, schema, validation, metric = _custom_binary_fixture(tmp_path)

    evidence = run_baseline(schema, validation, metric, [], reader, tmp_path / "baseline")

    assert evidence["status"] == "completed"
    policy = evidence["preprocessing_policy"]
    assert "num_missing" in policy["numeric"]["columns"]
    assert policy["numeric"]["imputation"]["strategy"] == "median"
    assert policy["numeric"]["imputation"]["fit_scope"] == "inside_cv_folds"
    assert policy["numeric"]["scaling"]["enabled"] is True
    assert policy["numeric"]["scaling"]["strategy"] == "standard_scaler"
    assert "category" in policy["categorical"]["columns"]
    assert policy["categorical"]["missing_value_handling"]["strategy"] == "most_frequent"
    assert policy["categorical"]["encoding"]["strategy"] == "one_hot"
    assert policy["categorical"]["encoding"]["handle_unknown"] == "ignore"
    assert policy["categorical"]["target_encoding_used"] is False
    assert "rare_code" in policy["high_cardinality"]["columns"]
    assert policy["high_cardinality"]["strategy"] == "included_with_caution"
    assert policy["high_cardinality"]["target_encoding_used"] is False
    assert policy["high_cardinality"]["warnings"]
    assert "description" in policy["text_like"]["columns"]
    assert policy["text_like"]["strategy"] == "treated_as_categorical"
    assert policy["text_like"]["warnings"]


def test_fold_safe_policy_handles_unseen_validation_categories(tmp_path: Path) -> None:
    reader, schema, validation, metric = _custom_binary_fixture(tmp_path)

    evidence = run_baseline(schema, validation, metric, [], reader, tmp_path / "baseline")

    assert evidence["status"] == "completed"
    assert len(evidence["fold_results"]) >= 2
    safety = evidence["preprocessing_policy"]["safety_checks"]
    assert safety["fits_preprocessing_inside_folds"] is True
    assert safety["uses_target_encoding"] is False
    assert safety["uses_test_labels"] is False


def test_skipped_baseline_does_not_claim_preprocessing_policy(tmp_path: Path) -> None:
    reader = DatasetReader(FIXTURE_ROOT / "ranking_tiny")
    schema = InferredSchema(
        target_column="target",
        primary_id_column="item_id",
        train_base_table="train_base.csv",
        confidence="high",
    )

    evidence = run_baseline(
        schema,
        ValidationEvidence(primary_validation={"method": "ranking_group_cv"}),
        MetricEvidence(
            metric_name="ndcg",
            task_type="ranking",
            metric_family="ranking",
            local_metric_available=False,
        ),
        [],
        reader,
        tmp_path,
    )

    assert evidence["status"] == "skipped"
    assert "metric_value" not in evidence
    assert "preprocessing_policy" not in evidence


def _fixture_inputs(
    fixture_name: str,
) -> tuple[InferredSchema, ValidationEvidence, MetricEvidence, DatasetReader]:
    fixture_dir = FIXTURE_ROOT / fixture_name
    inventory = build_file_inventory(fixture_dir)
    reader = DatasetReader(fixture_dir)
    schema = infer_schema(inventory, reader)
    profiles = profile_tables(inventory, schema, reader)
    task_plan = EdaTaskPlan(
        **json.loads((fixture_dir / "eda_task_plan.json").read_text(encoding="utf-8"))
    )
    metric = analyze_metric(task_plan, schema, profiles)
    validation = analyze_validation(schema, profiles, metric, reader)
    return schema, validation, metric, reader


def _custom_binary_fixture(
    tmp_path: Path,
) -> tuple[DatasetReader, InferredSchema, ValidationEvidence, MetricEvidence]:
    (tmp_path / "train_base.csv").write_text(
        "id,target,num_missing,category,rare_code,description,event_date\n"
        "1,0,1.0,A,code_001,\"short product note with several words\",2024-01-01\n"
        "2,1,,B,code_002,\"long customer comment with enough tokens\",2024-01-02\n"
        "3,0,3.0,A,code_003,\"another descriptive text field value\",2024-01-03\n"
        "4,1,4.0,,code_004,\"message body contains plain text tokens\",2024-01-04\n"
        "5,0,5.0,C,code_005,\"review text with multiple natural words\",2024-01-05\n"
        "6,1,,D,code_006,\"free form description for this row\",2024-01-06\n"
        "7,0,7.0,E,code_007,\"human readable text column example\",2024-01-07\n"
        "8,1,8.0,F,code_008,\"another free text looking value\",2024-01-08\n",
        encoding="utf-8",
    )
    return (
        DatasetReader(tmp_path),
        InferredSchema(
            target_column="target",
            primary_id_column="id",
            train_base_table="train_base.csv",
            confidence="high",
        ),
        ValidationEvidence(primary_validation={"method": "stratified_kfold"}),
        MetricEvidence(
            metric_name="accuracy",
            task_type="binary_classification",
            metric_family="threshold_classification",
            requires_threshold=True,
            local_metric_available=True,
        ),
    )
