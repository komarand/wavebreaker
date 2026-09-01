from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kaggle_researcher.facts.dataset_shape as dataset_shape_module
from kaggle_researcher.facts.dataset_shape import read_dataset_shape
from kaggle_researcher.facts.kaggle_api import KaggleRequestPolicy
from kaggle_researcher.facts.models import FileInfo, FileManifest


class FakeApi:
    def __init__(self, contents: dict[str, str], error: Exception | None = None) -> None:
        self.contents = contents
        self.error = error
        self.download_calls: list[str] = []
        self.download_directories: list[Path] = []

    def competition_download_file(
        self,
        competition: str,
        file_name: str,
        *,
        path: str,
        quiet: bool,
    ) -> None:
        assert competition == "example"
        assert quiet is True
        self.download_calls.append(file_name)
        destination = Path(path)
        self.download_directories.append(destination)
        if self.error is not None:
            raise self.error
        (destination / Path(file_name).name).write_text(
            self.contents[file_name],
            encoding="utf-8",
        )


@pytest.fixture(autouse=True)
def request_policy_without_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dataset_shape_module,
        "_DATASET_REQUEST_POLICY",
        KaggleRequestPolicy(base_delay_seconds=0, min_interval_seconds=0),
    )


def test_reads_rows_column_shapes_and_target_from_column_difference() -> None:
    manifest = _manifest(
        FileInfo(name="train.csv", size_bytes=100, role_hint="train"),
        FileInfo(name="test.csv", size_bytes=80, role_hint="test"),
    )
    api = FakeApi(
        {
            "train.csv": (
                "id,feature,flag,Survived\n"
                "1,1.5,true,0\n"
                "2,,false,1\n"
                "3,2.5,true,1\n"
                "4,99.0,false,0\n"
            ),
            "test.csv": "id,feature,flag\n5,1.1,true\n6,2.2,false\n",
        }
    )

    shape = read_dataset_shape(
        "example",
        manifest,
        max_bytes=1000,
        sample_rows=3,
        api=api,
    )

    assert shape is not None
    assert shape.status == "read"
    assert shape.coverage == "full_file"
    assert shape.train_rows == 4
    assert shape.test_rows == 2
    assert shape.train_test_row_ratio == 2.0
    assert shape.sampled_rows == 3
    assert [column.name for column in shape.columns] == [
        "id",
        "feature",
        "flag",
        "Survived",
    ]
    assert [column.inferred_type for column in shape.columns] == [
        "integer",
        "float",
        "boolean",
        "integer",
    ]
    assert shape.columns[1].null_share_in_sample == 0.3333
    assert shape.target is not None
    assert shape.target.column == "Survived"
    assert shape.target.class_counts_in_sample == {"0": 1, "1": 2}
    assert shape.target.is_binary_in_sample is True
    assert api.download_calls == ["train.csv", "test.csv"]
    assert all(not path.exists() for path in api.download_directories)


def test_oversized_file_is_not_downloaded_and_partial_status_is_honest() -> None:
    manifest = _manifest(
        FileInfo(name="train.csv", size_bytes=101, role_hint="train"),
        FileInfo(name="test.csv", size_bytes=20, role_hint="test"),
    )
    api = FakeApi({"test.csv": "id\n1\n2\n"})

    shape = read_dataset_shape(
        "example",
        manifest,
        max_bytes=100,
        sample_rows=10,
        api=api,
    )

    assert shape is not None
    assert shape.status == "partial"
    assert shape.coverage == "header_only"
    assert shape.train_rows is None
    assert shape.test_rows == 2
    assert shape.columns == []
    assert api.download_calls == ["test.csv"]
    assert "exceeds the 100-byte dataset read limit" in " ".join(shape.limitations)


def test_unsupported_formats_are_unavailable_without_download() -> None:
    manifest = _manifest(
        FileInfo(name="train.parquet", size_bytes=10, role_hint="train"),
        FileInfo(name="test.zip", size_bytes=10, role_hint="test"),
    )
    api = FakeApi({})

    shape = read_dataset_shape("example", manifest, 100, 10, api=api)

    assert shape is not None
    assert shape.status == "unavailable"
    assert shape.coverage == "none"
    assert "parquet" in " ".join(shape.limitations)
    assert "zip" in " ".join(shape.limitations)
    assert api.download_calls == []


def test_target_is_not_guessed_when_column_difference_is_ambiguous() -> None:
    manifest = _manifest(
        FileInfo(name="train.tsv", size_bytes=20, role_hint="train"),
        FileInfo(name="test.tsv", size_bytes=20, role_hint="test"),
    )
    api = FakeApi(
        {
            "train.tsv": "id\tlabel_a\tlabel_b\n1\t0\t1\n",
            "test.tsv": "id\n2\n",
        }
    )

    shape = read_dataset_shape("example", manifest, 100, 10, api=api)

    assert shape is not None
    assert shape.target is None
    assert "more than one train column" in " ".join(shape.limitations)


def test_sample_values_are_bounded_and_full_file_only_counts_rows() -> None:
    long_value = "x" * 40
    manifest = _manifest(
        FileInfo(name="train.csv", size_bytes=100, role_hint="train"),
        FileInfo(name="test.csv", size_bytes=100, role_hint="test"),
    )
    rows = "\n".join(f"{index},{long_value}{index},0" for index in range(10))
    api = FakeApi(
        {
            "train.csv": f"id,text,target\n{rows}\n",
            "test.csv": "id,text\n10,value\n",
        }
    )

    shape = read_dataset_shape("example", manifest, 1000, 6, api=api)

    assert shape is not None
    assert shape.train_rows == 10
    assert shape.sampled_rows == 6
    assert len(shape.columns[1].sample_values) == 5
    assert all(len(value) <= 32 for value in shape.columns[1].sample_values)


def test_download_error_is_nonfatal_and_403_is_recorded() -> None:
    class ForbiddenError(RuntimeError):
        status = 403

    manifest = _manifest(
        FileInfo(name="train.csv", size_bytes=10, role_hint="train"),
        FileInfo(name="test.csv", size_bytes=10, role_hint="test"),
    )
    api = FakeApi({}, error=ForbiddenError("rules not accepted"))

    shape = read_dataset_shape("example", manifest, 100, 10, api=api)

    assert shape is not None
    assert shape.status == "unavailable"
    assert shape.coverage == "none"
    assert "403" in " ".join(shape.limitations)


def _manifest(*files: FileInfo) -> FileManifest:
    return FileManifest(
        files=list(files),
        train_test_bytes_ratio=None,
        sample_submission_columns=[],
        sample_submission_source="unavailable",
        limitations=[],
    )
