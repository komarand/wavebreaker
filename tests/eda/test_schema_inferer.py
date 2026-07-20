from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.presets import HOME_CREDIT_CRMS_PRESET


FIXTURE_DIR = Path("tests/fixtures/eda/home_credit_tiny")


def test_fixture_schema_identifies_global_roles() -> None:
    inventory = build_file_inventory(FIXTURE_DIR, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(FIXTURE_DIR)

    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)

    assert schema.target_column == "target"
    assert schema.primary_id_column == "case_id"
    assert schema.prediction_column == "score"
    assert schema.train_base_table == "train_base.csv"
    assert schema.test_base_table == "test_base.csv"
    assert schema.sample_submission_table == "sample_submission.csv"
    assert schema.candidate_time_columns == ["WEEK_NUM"]
    assert schema.candidate_date_columns == ["date_decision"]
    assert "case_id" in schema.global_roles["candidate_join_keys"]
    assert schema.confidence == "high"
    assert schema.warnings == []


def test_fixture_table_and_column_roles_are_inferred() -> None:
    inventory = build_file_inventory(FIXTURE_DIR, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(FIXTURE_DIR)

    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)
    tables_by_path = {table.path: table for table in schema.tables}
    train_roles = {
        column_role.name: column_role.role
        for column_role in tables_by_path["train_base.csv"].column_roles
    }
    test_roles = {
        column_role.name: column_role.role
        for column_role in tables_by_path["test_base.csv"].column_roles
    }
    submission_roles = {
        column_role.name: column_role.role
        for column_role in tables_by_path["sample_submission.csv"].column_roles
    }

    assert tables_by_path["train_base.csv"].role == "train"
    assert tables_by_path["train_base.csv"].table_type == "base"
    assert tables_by_path["test_base.csv"].role == "test"
    assert tables_by_path["test_base.csv"].table_type == "base"
    assert tables_by_path["train_static_0.csv"].table_type == "depth_0"
    assert train_roles["target"] == "target"
    assert train_roles["case_id"] == "primary_id"
    assert train_roles["WEEK_NUM"] == "time"
    assert train_roles["date_decision"] == "date"
    assert "target" not in test_roles
    assert submission_roles["score"] == "prediction"


def test_target_is_not_inferred_from_test_table(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "case_id,WEEK_NUM,date_decision,income\n1,1,2024-01-01,100\n",
        encoding="utf-8",
    )
    (tmp_path / "test_base.csv").write_text(
        "case_id,WEEK_NUM,date_decision,target\n2,2,2024-01-08,0\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text("case_id,score\n2,0.1\n", encoding="utf-8")

    inventory = build_file_inventory(tmp_path, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)

    test_table = next(table for table in schema.tables if table.path == "test_base.csv")
    test_target_role = next(role for role in test_table.column_roles if role.name == "target")

    assert schema.target_column is None
    assert test_target_role.role == "unknown"
    assert "Target column could not be inferred from train tables." in schema.warnings
    assert schema.confidence == "medium"


def test_generic_schema_inference_works_without_preset(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "id,event_date,target,value\n1,2024-01-01,0,10\n2,2024-01-02,1,20\n",
        encoding="utf-8",
    )
    (tmp_path / "test_base.csv").write_text(
        "id,event_date,value\n3,2024-01-03,30\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text(
        "id,prediction\n3,0.5\n",
        encoding="utf-8",
    )

    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)

    assert schema.target_column == "target"
    assert schema.primary_id_column == "id"
    assert schema.prediction_column == "prediction"
    assert schema.candidate_date_columns == ["event_date"]
    assert schema.confidence == "high"
    assert schema.warnings == []


def test_generic_schema_inference_uses_cross_table_id_and_prediction_evidence_without_preset() -> None:
    inventory = build_file_inventory(FIXTURE_DIR)
    reader = DatasetReader(FIXTURE_DIR)

    schema = infer_schema(inventory, reader)
    tables_by_path = {table.path: table for table in schema.tables}
    train_roles = {
        column_role.name: column_role.role
        for column_role in tables_by_path["train_base.csv"].column_roles
    }
    submission_roles = {
        column_role.name: column_role.role
        for column_role in tables_by_path["sample_submission.csv"].column_roles
    }

    assert schema.primary_id_column == "case_id"
    assert schema.prediction_column == "score"
    assert schema.candidate_time_columns == ["WEEK_NUM"]
    assert train_roles["case_id"] == "primary_id"
    assert train_roles["WEEK_NUM"] == "time"
    assert submission_roles["score"] == "prediction"


def test_missing_train_or_id_produces_degraded_confidence_and_warnings(tmp_path: Path) -> None:
    (tmp_path / "test_base.csv").write_text(
        "row_key,WEEK_NUM,date_decision\n2,2,2024-01-08\n",
        encoding="utf-8",
    )

    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)

    assert schema.train_base_table is None
    assert schema.primary_id_column is None
    assert schema.confidence == "low"
    assert "Train base table could not be inferred." in schema.warnings
    assert "Target column could not be inferred from train tables." in schema.warnings
    assert "Primary id column could not be inferred." in schema.warnings
