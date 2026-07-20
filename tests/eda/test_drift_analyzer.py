from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.drift_analyzer import analyze_drift
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.presets import HOME_CREDIT_CRMS_PRESET
from kaggle_researcher.eda.schemas import InferredSchema, ValidationEvidence


HOME_CREDIT_FIXTURE = Path("tests/fixtures/eda/home_credit_tiny")


def test_fixture_with_time_column_returns_target_drift() -> None:
    inventory = build_file_inventory(HOME_CREDIT_FIXTURE, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(HOME_CREDIT_FIXTURE)
    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)
    validation = ValidationEvidence(time_columns=[{"name": "WEEK_NUM"}])

    evidence = analyze_drift(schema, validation, reader)

    temporal = evidence["temporal_drift"]
    assert temporal["status"] == "computed"
    assert temporal["time_column"] == "WEEK_NUM"
    assert temporal["row_count_by_period"]
    assert temporal["target_drift"]["status"] == "computed"
    assert temporal["target_drift"]["target_column"] == "target"
    assert temporal["target_drift"]["by_period"]


def test_fixture_without_time_column_skips_temporal_drift_with_limitation(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "id,target,value\n"
        "1,0,10\n"
        "2,1,20\n",
        encoding="utf-8",
    )
    (tmp_path / "test_base.csv").write_text(
        "id,value\n"
        "3,30\n",
        encoding="utf-8",
    )
    schema = InferredSchema(
        target_column="target",
        primary_id_column="id",
        train_base_table="train_base.csv",
        test_base_table="test_base.csv",
        confidence="high",
    )

    evidence = analyze_drift(schema, ValidationEvidence(), DatasetReader(tmp_path))

    assert evidence["temporal_drift"]["status"] == "skipped"
    assert any("No time column" in limitation for limitation in evidence["limitations"])


def test_shifted_train_test_fixture_produces_high_drift_severity(tmp_path: Path) -> None:
    _write_shifted_fixture(tmp_path)
    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)

    evidence = analyze_drift(schema, ValidationEvidence(), reader)
    amount_psi = _column(evidence["numeric_psi"], "amount")
    region_shift = _column(evidence["categorical_shift"], "region")

    assert evidence["severity"] == "high"
    assert amount_psi["severity"] == "high"
    assert amount_psi["psi"] >= 0.25
    assert region_shift["severity"] in {"medium", "high"}


def test_target_id_and_group_columns_are_excluded_from_adversarial_features(tmp_path: Path) -> None:
    _write_shifted_fixture(tmp_path)
    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)
    validation = ValidationEvidence(group_columns=[{"name": "customer_id"}])

    evidence = analyze_drift(schema, validation, reader)
    adversarial = evidence["adversarial_validation"]

    assert "target" in adversarial["excluded_columns"]
    assert "id" in adversarial["excluded_columns"]
    assert "customer_id" in adversarial["excluded_columns"]
    assert "target" not in adversarial["feature_columns"]
    assert "id" not in adversarial["feature_columns"]
    assert "customer_id" not in adversarial["feature_columns"]


def _write_shifted_fixture(dataset_path: Path) -> None:
    (dataset_path / "train_base.csv").write_text(
        "id,customer_id,target,amount,region,stable_feature\n"
        "1,c1,0,10,north,1\n"
        "2,c2,1,11,north,1\n"
        "3,c3,0,12,north,2\n"
        "4,c4,1,13,south,2\n"
        "5,c5,0,14,south,3\n"
        "6,c6,1,15,south,3\n",
        encoding="utf-8",
    )
    (dataset_path / "test_base.csv").write_text(
        "id,customer_id,amount,region,stable_feature\n"
        "7,c7,100,west,1\n"
        "8,c8,101,west,1\n"
        "9,c9,102,west,2\n"
        "10,c10,103,west,2\n"
        "11,c11,104,west,3\n"
        "12,c12,105,west,3\n",
        encoding="utf-8",
    )


def _column(section: dict[str, Any], column_name: str) -> dict[str, Any]:
    return next(item for item in section["columns"] if item["column"] == column_name)
