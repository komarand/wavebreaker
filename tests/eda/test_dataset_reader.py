from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError


def _write_tiny_tables(dataset_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "case_id": [1, 2, 3],
            "target": [0, 1, 0],
            "amount": [100.0, 250.5, 300.0],
        }
    )
    frame.write_csv(dataset_path / "train.csv")
    frame.write_parquet(dataset_path / "train.parquet")
    (dataset_path / "records.json").write_text(
        '[{"case_id":1,"target":0},{"case_id":2,"target":1}]',
        encoding="utf-8",
    )
    (dataset_path / "records.jsonl").write_text(
        '{"case_id":1,"target":0}\n{"case_id":2,"target":1}\n',
        encoding="utf-8",
    )


def test_csv_schema_head_and_count_work(tmp_path: Path) -> None:
    _write_tiny_tables(tmp_path)
    reader = DatasetReader(tmp_path)

    schema = reader.read_schema("train.csv")
    head = reader.file_head("train.csv", n_rows=2)
    count = reader.count_rows("train.csv")

    assert {column["name"] for column in schema} == {"case_id", "target", "amount"}
    assert isinstance(head, pl.DataFrame)
    assert head.shape == (2, 3)
    assert count == 3


def test_parquet_schema_head_and_count_work(tmp_path: Path) -> None:
    _write_tiny_tables(tmp_path)
    reader = DatasetReader(tmp_path)

    schema = reader.read_schema("train.parquet")
    head = reader.file_head("train.parquet", n_rows=1)
    count = reader.count_rows("train.parquet")

    assert {column["name"] for column in schema} == {"case_id", "target", "amount"}
    assert head.shape == (1, 3)
    assert count == 3


def test_json_and_jsonl_are_supported_for_bounded_reads(tmp_path: Path) -> None:
    _write_tiny_tables(tmp_path)
    reader = DatasetReader(tmp_path)

    json_head = reader.file_head("records.json", n_rows=1)
    jsonl_head = reader.file_head("records.jsonl", n_rows=1)

    assert json_head.shape == (1, 2)
    assert jsonl_head.shape == (1, 2)
    assert reader.count_rows("records.json") == 2
    assert reader.count_rows("records.jsonl") == 2


def test_sample_table_returns_polars_dataframe(tmp_path: Path) -> None:
    _write_tiny_tables(tmp_path)
    reader = DatasetReader(tmp_path)

    sampled = reader.sample_table("train.csv", n_rows=2, seed=123)

    assert isinstance(sampled, pl.DataFrame)
    assert sampled.height == 2


def test_read_columns_returns_only_requested_columns(tmp_path: Path) -> None:
    _write_tiny_tables(tmp_path)
    reader = DatasetReader(tmp_path)

    frame = reader.read_columns("train.csv", columns=["case_id", "target"], n_rows=2)

    assert frame.columns == ["case_id", "target"]
    assert frame.shape == (2, 2)


def test_resolve_path_rejects_unsupported_missing_and_escape_paths(tmp_path: Path) -> None:
    _write_tiny_tables(tmp_path)
    (tmp_path / "notes.txt").write_text("not tabular", encoding="utf-8")
    reader = DatasetReader(tmp_path)

    with pytest.raises(ReaderError, match="Unsupported dataset file extension"):
        reader.resolve_path("notes.txt")

    with pytest.raises(ReaderError, match="does not exist"):
        reader.resolve_path("missing.csv")

    with pytest.raises(ReaderError, match="escapes dataset directory"):
        reader.resolve_path("../outside.csv")


def test_invalid_arguments_raise_reader_error(tmp_path: Path) -> None:
    _write_tiny_tables(tmp_path)
    reader = DatasetReader(tmp_path)

    with pytest.raises(ReaderError, match="n_rows must be a positive integer"):
        reader.file_head("train.csv", n_rows=0)

    with pytest.raises(ReaderError, match="columns must contain"):
        reader.read_columns("train.csv", columns=[])
