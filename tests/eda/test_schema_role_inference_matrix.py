from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.leakage_checker import check_leakage
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.modules.validation_analyzer import analyze_validation
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig, EdaTaskPlan


def test_binary_classification_basic_roles_make_downstream_checks_testable(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", "row_id,target,num_feature,cat_feature\n1,0,10,a\n2,1,20,b\n3,0,30,a\n4,1,40,b\n")
    _write_csv(tmp_path / "test.csv", "row_id,num_feature,cat_feature\n5,15,a\n6,25,b\n")
    _write_csv(tmp_path / "sample_submission.csv", "row_id,target\n5,0\n6,0\n")

    payload = _analyze(tmp_path, task_type="binary_classification", metric_name="roc_auc")

    assert payload["schema"].target_column == "target"
    assert payload["schema"].primary_id_column == "row_id"
    assert payload["schema"].sample_submission_table == "sample_submission.csv"
    assert payload["validation"].primary_validation["method"] == "stratified_kfold"
    assert _leakage(payload, "target_in_test").status == "passed"
    assert _leakage(payload, "id_overlap").status == "passed"


def test_binary_classification_nonstandard_submission_template_inferred_by_structure(tmp_path: Path) -> None:
    _write_csv(tmp_path / "training_data.csv", "object_id,is_clicked,x1,x2\n1,0,0.1,a\n2,1,0.8,b\n3,0,0.3,a\n4,1,0.7,c\n")
    _write_csv(tmp_path / "testing_data.csv", "object_id,x1,x2\n5,0.2,a\n6,0.6,b\n")
    _write_csv(tmp_path / "submission_template.csv", "object_id,is_clicked\n5,0\n6,0\n")

    schema = _analyze(tmp_path, task_type="binary_classification", metric_name="log_loss")["schema"]

    assert schema.target_column == "is_clicked"
    assert schema.primary_id_column == "object_id"
    assert schema.sample_submission_table == "submission_template.csv"


def test_regression_basic_uses_task_type_for_numeric_target(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", "id,price,size,city\n1,120000,10,a\n2,180000,20,b\n3,160000,15,a\n4,220000,30,c\n")
    _write_csv(tmp_path / "test.csv", "id,size,city\n5,12,a\n6,22,b\n")
    _write_csv(tmp_path / "sample_submission.csv", "id,price\n5,0\n6,0\n")

    schema = _analyze(tmp_path, task_type="regression", metric_name="rmse")["schema"]

    assert schema.target_column == "price"
    assert schema.primary_id_column == "id"
    assert schema.global_roles["target_column_confidence"] in {"medium", "high"}


def test_multiclass_basic_allows_multiple_prediction_columns(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", "row_id,label,num_feature\n1,a,0.1\n2,b,0.2\n3,c,0.3\n4,a,0.4\n")
    _write_csv(tmp_path / "test.csv", "row_id,num_feature\n5,0.5\n6,0.6\n")
    _write_csv(tmp_path / "sample_submission.csv", "row_id,class_a,class_b,class_c\n5,0.3,0.3,0.4\n6,0.2,0.5,0.3\n")

    schema = _analyze(tmp_path, task_type="multiclass_classification", metric_name="log_loss")["schema"]

    assert schema.target_column == "label"
    assert schema.primary_id_column == "row_id"
    assert schema.global_roles["prediction_columns"] == ["class_a", "class_b", "class_c"]


def test_grouped_task_distinguishes_primary_row_id_from_group_key(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", "row_id,user_id,target,num_feature\n1,u1,0,10\n2,u1,1,20\n3,u2,0,30\n4,u2,1,40\n")
    _write_csv(tmp_path / "test.csv", "row_id,user_id,num_feature\n5,u1,15\n6,u3,25\n")
    _write_csv(tmp_path / "sample_submission.csv", "row_id,target\n5,0\n6,0\n")

    schema = _analyze(tmp_path, task_type="binary_classification", metric_name="roc_auc")["schema"]

    assert schema.primary_id_column == "row_id"
    assert "user_id" in schema.candidate_group_columns


def test_ambiguous_target_is_not_forced_and_candidates_are_reported(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", "row_id,label,outcome,num_feature\n1,0,1,10\n2,1,0,20\n3,0,1,30\n4,1,0,40\n")
    _write_csv(tmp_path / "test.csv", "row_id,num_feature\n5,15\n6,25\n")
    _write_csv(tmp_path / "sample_submission.csv", "row_id,prediction\n5,0\n6,0\n")

    schema = _analyze(tmp_path, task_type="binary_classification", metric_name="roc_auc")["schema"]

    assert schema.target_column is None
    assert {item["name"] for item in schema.global_roles["target_column_candidates"]} >= {"label", "outcome"}
    assert any("Ambiguous target column candidates" in warning for warning in schema.warnings)


def test_sample_submission_with_target_like_column_is_not_classified_as_train(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", "row_id,target,num_feature\n1,0,10\n2,1,20\n3,0,30\n4,1,40\n")
    _write_csv(tmp_path / "test.csv", "row_id,num_feature\n5,15\n6,25\n")
    _write_csv(tmp_path / "sample_submission.csv", "row_id,target\n5,0\n6,0\n")

    schema = _analyze(tmp_path, task_type="binary_classification", metric_name="roc_auc")["schema"]
    tables = {table.path: table for table in schema.tables}

    assert tables["sample_submission.csv"].role == "submission"
    assert schema.target_column == "target"
    assert schema.sample_submission_table == "sample_submission.csv"


def test_empty_dataset_run_fails_clearly(tmp_path: Path) -> None:
    data_dir = tmp_path / "empty_data"
    data_dir.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_json(inputs / "research_hypotheses.json", {"competition_id": "empty_dataset", "hypotheses": []})
    _write_json(
        inputs / "eda_task_plan.json",
        {"competition_id": "empty_dataset", "task_type": "binary_classification", "metric": {"name": "roc_auc"}},
    )

    with pytest.raises(ValueError, match="No supported tabular data files"):
        asyncio.run(
            run_eda(
                EdaRunConfig(
                    competition_id="empty_dataset",
                    hypotheses_path=inputs / "research_hypotheses.json",
                    task_plan_path=inputs / "eda_task_plan.json",
                    local_dataset_path=data_dir,
                    output_dir=tmp_path / "runs",
                    download_dataset=False,
                )
            )
        )


def test_real_like_common_kaggle_patterns_are_inferred_generically(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", "PassengerId,Survived,Pclass,Fare,Embarked\n1,0,3,7.25,S\n2,1,1,71.28,C\n3,1,3,7.92,S\n4,0,1,53.10,S\n")
    _write_csv(tmp_path / "test.csv", "PassengerId,Pclass,Fare,Embarked\n5,3,8.05,Q\n6,1,26.55,S\n")
    _write_csv(tmp_path / "gender_submission.csv", "PassengerId,Survived\n5,0\n6,1\n")

    schema = _analyze(tmp_path, task_type="binary_classification", metric_name="accuracy")["schema"]

    assert schema.primary_id_column == "PassengerId"
    assert schema.target_column == "Survived"
    assert schema.sample_submission_table == "gender_submission.csv"


def _analyze(tmp_path: Path, *, task_type: str, metric_name: str) -> dict[str, Any]:
    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader, task_type_hint=task_type, metric_hint=metric_name)
    profiles = profile_tables(inventory, schema, reader, sample_rows=1000)
    metric = analyze_metric(
        EdaTaskPlan(competition_id="synthetic", task_type=task_type, metric={"name": metric_name}),
        schema,
        profiles,
    )
    validation = analyze_validation(schema, profiles, metric, reader)
    leakage = check_leakage(schema, validation, reader)
    return {"schema": schema, "profiles": profiles, "metric": metric, "validation": validation, "leakage": leakage}


def _leakage(payload: dict[str, Any], check_id: str) -> Any:
    return next(item for item in payload["leakage"] if item.check_id == check_id)


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
