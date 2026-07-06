from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.presets import HOME_CREDIT_CRMS_PRESET


FIXTURE_DIR = Path("tests/fixtures/eda/home_credit_tiny")


def test_fixture_profiles_include_all_readable_tables() -> None:
    inventory = build_file_inventory(FIXTURE_DIR, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(FIXTURE_DIR)
    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)

    profiles = profile_tables(inventory, schema, reader)

    readable_paths = {
        file.path
        for file in inventory.files
        if file.can_read and file.extension in {".csv", ".parquet", ".json", ".jsonl"}
    }
    assert {profile.path for profile in profiles} == readable_paths
    assert all(profile.n_rows is not None for profile in profiles)
    assert all(profile.n_cols > 0 for profile in profiles)


def test_target_column_profile_has_two_unique_values() -> None:
    inventory = build_file_inventory(FIXTURE_DIR, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(FIXTURE_DIR)
    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)

    profiles = profile_tables(inventory, schema, reader)
    train_profile = next(profile for profile in profiles if profile.path == "train_base.csv")
    target_profile = next(column for column in train_profile.columns if column.name == "target")

    assert train_profile.n_rows == 8
    assert target_profile.n_unique == 2
    assert target_profile.min == 0
    assert target_profile.max == 1


def test_mostly_missing_and_high_cardinality_columns_are_detected() -> None:
    inventory = build_file_inventory(FIXTURE_DIR, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(FIXTURE_DIR)
    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)

    profiles = profile_tables(inventory, schema, reader)
    train_profile = next(profile for profile in profiles if profile.path == "train_base.csv")

    assert "mostly_missing_note" in train_profile.mostly_missing_columns
    assert "applicant_code" in train_profile.high_cardinality_columns


def test_constant_columns_are_detected(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text(
        "id,target,constant_flag,mostly_missing\n"
        "1,0,yes,\n"
        "2,1,yes,\n"
        "3,0,yes,\n"
        "4,1,yes,\n"
        "5,0,yes,only_value\n",
        encoding="utf-8",
    )
    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader)

    profiles = profile_tables(inventory, schema, reader)
    train_profile = profiles[0]

    assert "constant_flag" in train_profile.constant_columns
    assert "mostly_missing" in train_profile.mostly_missing_columns
    assert "mostly_missing" not in train_profile.constant_columns


def test_sampled_flag_is_true_when_full_scan_limit_is_below_row_count() -> None:
    inventory = build_file_inventory(FIXTURE_DIR, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(FIXTURE_DIR)
    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)

    profiles = profile_tables(
        inventory,
        schema,
        reader,
        sample_rows=3,
        max_full_scan_rows=1,
    )
    train_profile = next(profile for profile in profiles if profile.path == "train_base.csv")

    assert train_profile.sampled is True
    assert train_profile.sample_rows == 3
    assert train_profile.n_rows == 8
