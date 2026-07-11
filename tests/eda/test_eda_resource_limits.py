from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.drift_analyzer import analyze_drift
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.schemas import InferredSchema, ValidationEvidence


def test_dataset_reader_exposes_lightweight_file_size_info(tmp_path: Path) -> None:
    _write_binary_fixture(tmp_path, rows=4)
    reader = DatasetReader(tmp_path)

    info = reader.file_info("train_base.csv")

    assert info["path"] == "train_base.csv"
    assert info["extension"] == ".csv"
    assert info["size_bytes"] == reader.file_size_bytes("train_base.csv")
    assert info["size_bytes"] > 0


def test_low_profile_caps_force_sampling_with_warnings(tmp_path: Path) -> None:
    _write_binary_fixture(tmp_path, rows=10)
    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)

    profiles = profile_tables(
        inventory,
        schema,
        reader,
        sample_rows=5,
        max_full_scan_rows=100,
        max_table_bytes=1,
        max_column_cardinality_scan_rows=3,
    )
    train_profile = next(profile for profile in profiles if profile.path == "train_base.csv")

    assert train_profile.sampled is True
    assert train_profile.sample_rows == 3
    assert any("EDA_MAX_TABLE_BYTES" in warning for warning in train_profile.warnings)
    assert any("EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS" in warning for warning in train_profile.warnings)


def test_drift_caps_train_test_rows_and_reports_sampling(tmp_path: Path) -> None:
    _write_binary_fixture(tmp_path, rows=8)
    reader = DatasetReader(tmp_path)
    schema = InferredSchema(
        target_column="target",
        primary_id_column="id",
        train_base_table="train_base.csv",
        test_base_table="test_base.csv",
        confidence="high",
    )

    evidence = analyze_drift(schema, ValidationEvidence(), reader, max_rows=2)

    assert evidence["sampled"] is True
    assert evidence["sample_rows"] == 2
    assert evidence["adversarial_validation"]["sampled"] is True
    assert evidence["adversarial_validation"]["sample_rows"] == 2
    assert any("EDA_MAX_ADVERSARIAL_ROWS=2" in warning for warning in evidence["warnings"])
    assert any("bounded sample" in limitation for limitation in evidence["limitations"])


def _write_binary_fixture(dataset_path: Path, *, rows: int) -> None:
    train_lines = ["id,target,amount,region"]
    test_lines = ["id,amount,region"]
    for idx in range(rows):
        target = idx % 2
        region = "north" if idx % 2 == 0 else "south"
        train_lines.append(f"{idx},{target},{idx * 1.5},{region}")
        test_lines.append(f"{idx + rows},{idx * 2.5},{region}")
    (dataset_path / "train_base.csv").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (dataset_path / "test_base.csv").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
