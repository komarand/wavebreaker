from __future__ import annotations

from pathlib import Path

import polars as pl

from kaggle_researcher.eda.modules.file_inventory import build_file_inventory


FIXTURE_DIR = Path("tests/fixtures/eda/home_credit_tiny")


def test_fixture_train_test_and_sample_submission_are_classified() -> None:
    inventory = build_file_inventory(FIXTURE_DIR)
    by_name = {file.name: file for file in inventory.files}

    assert by_name["train_base.csv"].role_hint == "train"
    assert by_name["train_base.csv"].table_hint == "base"
    assert by_name["test_base.csv"].role_hint == "test"
    assert by_name["test_base.csv"].table_hint == "base"
    assert by_name["sample_submission.csv"].role_hint == "sample_submission"
    assert by_name["sample_submission.csv"].table_hint == "submission"
    assert by_name["train_static_0.csv"].role_hint == "train"
    assert by_name["train_static_0.csv"].table_hint == "depth_0"
    assert by_name["test_static_0.csv"].role_hint == "test"
    assert by_name["test_static_0.csv"].table_hint == "depth_0"

    assert "train_base.csv" in inventory.train_files
    assert "test_base.csv" in inventory.test_files
    assert inventory.sample_submission_files == ["sample_submission.csv"]
    assert all(file.can_read for file in inventory.files if file.extension == ".csv")


def test_detected_formats_counts_fixture_files() -> None:
    inventory = build_file_inventory(FIXTURE_DIR)

    assert inventory.detected_formats[".csv"] == 5
    assert inventory.detected_formats[".json"] == 2


def test_unreadable_or_unsupported_file_is_recorded_without_crashing(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text("case_id,target\n1,0\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a data table", encoding="utf-8")

    inventory = build_file_inventory(tmp_path)
    by_name = {file.name: file for file in inventory.files}

    assert by_name["train_base.csv"].can_read is True
    assert by_name["notes.txt"].can_read is False
    assert "Unsupported dataset file extension" in by_name["notes.txt"].read_error
    assert inventory.suspicious_files == [
        {
            "path": "notes.txt",
            "reason": "Unsupported dataset file extension: .txt",
        }
    ]
    assert inventory.warnings == ["notes.txt: Unsupported dataset file extension: .txt"]


def test_duplicate_csv_parquet_logical_pair_is_detected(tmp_path: Path) -> None:
    frame = pl.DataFrame({"case_id": [1, 2], "target": [0, 1]})
    frame.write_csv(tmp_path / "train_base.csv")
    frame.write_parquet(tmp_path / "train_base.parquet")

    inventory = build_file_inventory(tmp_path)

    assert inventory.detected_formats[".csv"] == 1
    assert inventory.detected_formats[".parquet"] == 1
    assert inventory.duplicate_format_pairs == [
        {
            "logical_table": "train_base",
            "paths": ["train_base.csv", "train_base.parquet"],
            "formats": [".csv", ".parquet"],
        }
    ]


def test_missing_train_test_pairs_are_detected(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text("case_id,target\n1,0\n", encoding="utf-8")
    (tmp_path / "test_static_0.csv").write_text("case_id,status\n2,active\n", encoding="utf-8")

    inventory = build_file_inventory(tmp_path)

    assert inventory.missing_train_test_pairs == [
        {"logical_table": "base", "missing": "test"},
        {"logical_table": "static_0", "missing": "train"},
    ]


def test_zip_files_are_listed_but_not_schema_read(tmp_path: Path) -> None:
    (tmp_path / "competition.zip").write_bytes(b"not actually opened in inventory")

    inventory = build_file_inventory(tmp_path)
    only_file = inventory.files[0]

    assert only_file.extension == ".zip"
    assert only_file.can_read is True
    assert only_file.read_error is None
    assert inventory.detected_formats == {".zip": 1}
