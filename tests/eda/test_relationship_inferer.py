from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.relationship_inferer import infer_relationships
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.presets import HOME_CREDIT_CRMS_PRESET


HOME_CREDIT_FIXTURE = Path("tests/fixtures/eda/home_credit_tiny")


def test_home_credit_fixture_detects_case_id_relationship() -> None:
    inventory = build_file_inventory(HOME_CREDIT_FIXTURE, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(HOME_CREDIT_FIXTURE)
    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)

    evidence = infer_relationships(schema, inventory, reader)
    relationship = _relationship(evidence, "train_static_0.csv")

    assert evidence["base_table"] == "train_base.csv"
    assert evidence["base_id_column"] == "case_id"
    assert relationship["selected_join_key"] == "case_id"
    assert relationship["relationship_type"] in {"one_to_one", "one_to_many"}
    assert relationship["coverage_left_to_right"] > 0
    assert relationship["orphan_rate_right"] == 0


def test_generic_customer_id_relationship_detects_one_to_many(tmp_path: Path) -> None:
    _write_generic_customer_fixture(tmp_path)
    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)

    evidence = infer_relationships(schema, inventory, reader)
    relationship = _relationship(evidence, "train_orders.csv")

    assert evidence["base_table"] == "train_base.csv"
    assert evidence["base_id_column"] == "customer_id"
    assert relationship["selected_join_key"] == "customer_id"
    assert relationship["relationship_type"] == "one_to_many"
    assert relationship["requires_aggregation"] is True
    assert relationship["row_multiplication_risk"] in {"medium", "high"}
    assert relationship["avg_rows_per_left"] > 1
    assert relationship["max_rows_per_left"] == 2
    assert any("aggregate" in warning for warning in relationship["warnings"])


def test_generic_order_id_relationship_is_not_case_id_specific(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "order_id,target,total\n"
        "o1,0,10\n"
        "o2,1,20\n"
        "o3,0,30\n",
        encoding="utf-8",
    )
    (tmp_path / "test_base.csv").write_text(
        "order_id,total\n"
        "o4,40\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text(
        "order_id,prediction\n"
        "o4,0.5\n",
        encoding="utf-8",
    )
    (tmp_path / "train_order_features.csv").write_text(
        "order_id,coupon_used\n"
        "o1,1\n"
        "o2,0\n"
        "o3,1\n",
        encoding="utf-8",
    )

    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)

    evidence = infer_relationships(schema, inventory, reader)
    relationship = _relationship(evidence, "train_order_features.csv")

    assert evidence["base_id_column"] == "order_id"
    assert relationship["selected_join_key"] == "order_id"
    assert relationship["relationship_type"] == "one_to_one"
    assert relationship["row_multiplication_risk"] == "low"


def test_missing_join_key_returns_unknown_with_warning(tmp_path: Path) -> None:
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
    (tmp_path / "sample_submission.csv").write_text(
        "id,prediction\n"
        "3,0.5\n",
        encoding="utf-8",
    )
    (tmp_path / "train_extra.csv").write_text(
        "feature_name,feature_value\n"
        "a,1\n"
        "b,2\n",
        encoding="utf-8",
    )

    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)

    evidence = infer_relationships(schema, inventory, reader)
    relationship = _relationship(evidence, "train_extra.csv")

    assert relationship["selected_join_key"] is None
    assert relationship["relationship_type"] == "unknown"
    assert relationship["row_multiplication_risk"] == "unknown"
    assert relationship["confidence"] == "low"
    assert relationship["warnings"]


def _write_generic_customer_fixture(dataset_path: Path) -> None:
    (dataset_path / "train_base.csv").write_text(
        "customer_id,target,signup_date\n"
        "c1,0,2024-01-01\n"
        "c2,1,2024-01-02\n"
        "c3,0,2024-01-03\n",
        encoding="utf-8",
    )
    (dataset_path / "test_base.csv").write_text(
        "customer_id,signup_date\n"
        "c4,2024-01-04\n",
        encoding="utf-8",
    )
    (dataset_path / "sample_submission.csv").write_text(
        "customer_id,prediction\n"
        "c4,0.5\n",
        encoding="utf-8",
    )
    (dataset_path / "train_orders.csv").write_text(
        "customer_id,order_id,order_date,amount\n"
        "c1,o1,2024-02-01,10\n"
        "c1,o2,2024-02-02,11\n"
        "c2,o3,2024-02-03,12\n"
        "c3,o4,2024-02-04,13\n",
        encoding="utf-8",
    )


def _relationship(evidence: dict[str, Any], table_path: str) -> dict[str, Any]:
    return next(item for item in evidence["relationships"] if item["table"] == table_path)
