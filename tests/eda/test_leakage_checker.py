from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.leakage_checker import check_leakage
from kaggle_researcher.eda.schemas import InferredSchema, ValidationEvidence


FIXTURE_DIR = Path("tests/fixtures/eda/home_credit_tiny")


def _result_by_id(results: list, check_id: str):
    return next(result for result in results if result.check_id == check_id)


def test_default_fixture_passes_id_overlap_check() -> None:
    schema = InferredSchema(
        target_column="target",
        primary_id_column="case_id",
        prediction_column="score",
        train_base_table="train_base.csv",
        test_base_table="test_base.csv",
        sample_submission_table="sample_submission.csv",
        confidence="high",
    )
    validation = ValidationEvidence(primary_validation={"method": "stratified_kfold"})
    reader = DatasetReader(FIXTURE_DIR)

    results = check_leakage(schema, validation, reader)

    id_overlap = _result_by_id(results, "id_overlap")
    assert id_overlap.status == "passed"
    assert id_overlap.severity == "low"


def test_overlapping_train_test_id_is_detected(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "id,target,value\n1,0,10\n2,1,20\n",
        encoding="utf-8",
    )
    (tmp_path / "test_base.csv").write_text(
        "id,value\n2,25\n3,30\n",
        encoding="utf-8",
    )
    schema = InferredSchema(
        target_column="target",
        primary_id_column="id",
        train_base_table="train_base.csv",
        test_base_table="test_base.csv",
        confidence="high",
    )

    results = check_leakage(schema, ValidationEvidence(), DatasetReader(tmp_path))

    id_overlap = _result_by_id(results, "id_overlap")
    assert id_overlap.status == "failed"
    assert id_overlap.severity == "high"
    assert id_overlap.evidence["overlap_examples"] == [2]


def test_target_column_present_in_test_is_critical(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    (tmp_path / "test_base.csv").write_text("id,target\n3,0\n", encoding="utf-8")
    schema = InferredSchema(
        target_column="target",
        primary_id_column="id",
        train_base_table="train_base.csv",
        test_base_table="test_base.csv",
        confidence="high",
    )

    results = check_leakage(schema, ValidationEvidence(), DatasetReader(tmp_path))

    target_check = _result_by_id(results, "target_in_test")
    assert target_check.status == "failed"
    assert target_check.severity == "critical"


def test_group_overlap_is_reported_with_contextual_severity(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "row_id,user_id,target\n1,u1,0\n2,u2,1\n",
        encoding="utf-8",
    )
    (tmp_path / "test_base.csv").write_text(
        "row_id,user_id\n3,u2\n4,u3\n",
        encoding="utf-8",
    )
    schema = InferredSchema(
        target_column="target",
        primary_id_column="row_id",
        train_base_table="train_base.csv",
        test_base_table="test_base.csv",
        confidence="high",
    )
    validation = ValidationEvidence(
        group_columns=[{"name": "user_id"}],
        primary_validation={"method": "stratified_group_kfold"},
    )

    results = check_leakage(schema, validation, DatasetReader(tmp_path))

    group_overlap = _result_by_id(results, "group_overlap")
    assert group_overlap.status == "warning"
    assert group_overlap.severity == "medium"
    assert group_overlap.evidence["overlap_groups"] == ["u2"]


def test_missing_id_returns_not_testable(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text("target,value\n0,10\n1,20\n", encoding="utf-8")
    (tmp_path / "test_base.csv").write_text("value\n30\n", encoding="utf-8")
    schema = InferredSchema(
        target_column="target",
        train_base_table="train_base.csv",
        test_base_table="test_base.csv",
        confidence="medium",
    )

    results = check_leakage(schema, ValidationEvidence(), DatasetReader(tmp_path))

    id_overlap = _result_by_id(results, "id_overlap")
    assert id_overlap.status == "not_testable"
    assert id_overlap.severity == "low"
