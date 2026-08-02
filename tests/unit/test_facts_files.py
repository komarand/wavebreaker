from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Callable

import pytest

import kaggle_researcher.facts.files as files_module
from kaggle_researcher.facts.files import classify_role, fetch_file_manifest


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "facts" / "file_list.json"


class ForbiddenError(RuntimeError):
    status = 403


class FakeKaggleApi:
    list_response: Any = None
    list_error: Exception | None = None
    download_impl: Callable[[str, str, Path], None] | None = None
    instances: list["FakeKaggleApi"] = []

    def __init__(self) -> None:
        self.authenticated = False
        self.list_calls: list[str] = []
        self.download_calls: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def authenticate(self) -> None:
        self.authenticated = True

    def competition_list_files(self, competition: str) -> Any:
        self.list_calls.append(competition)
        if self.__class__.list_error is not None:
            raise self.__class__.list_error
        return self.__class__.list_response

    def competition_download_file(
        self,
        competition: str,
        file_name: str,
        *,
        path: str,
        quiet: bool,
    ) -> None:
        self.download_calls.append(
            {
                "competition": competition,
                "file_name": file_name,
                "path": path,
                "quiet": quiet,
            }
        )
        if self.__class__.download_impl is not None:
            self.__class__.download_impl(competition, file_name, Path(path))


@pytest.fixture(autouse=True)
def fake_kaggle_api(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeKaggleApi.list_response = {"files": []}
    FakeKaggleApi.list_error = None
    FakeKaggleApi.download_impl = None
    FakeKaggleApi.instances = []
    monkeypatch.setattr(files_module, "KaggleApi", FakeKaggleApi)


def _load_file_list() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("train.csv", "train"),
        ("nested/Train_features.parquet", "train"),
        (r"nested\test_data.csv", "test"),
        ("prefix_sample_submission_v2.csv", "submission"),
        ("metadata.json", "auxiliary"),
        ("train_sample_submission.csv", "train"),
    ],
)
def test_classify_role_uses_lowercased_basename(name: str, expected: str) -> None:
    assert classify_role(name) == expected


def test_full_fixture_maps_files_ratio_and_api_columns_without_download() -> None:
    FakeKaggleApi.list_response = _load_file_list()

    manifest = fetch_file_manifest("example-competition", max_sample_sub_bytes=5_000_000)

    assert [file.name for file in manifest.files] == [
        "train.csv",
        "nested/test.csv",
        "sample_submission.csv",
        "metadata.json",
    ]
    assert [file.size_bytes for file in manifest.files] == [1200, 400, 80, 50]
    assert [file.role_hint for file in manifest.files] == [
        "train",
        "test",
        "submission",
        "auxiliary",
    ]
    assert manifest.train_test_size_ratio == 3.0
    assert manifest.sample_submission_columns == ["id", "target"]
    assert manifest.sample_submission_source == "api"
    assert manifest.limitations == []

    api = FakeKaggleApi.instances[0]
    assert api.authenticated is True
    assert api.list_calls == ["example-competition"]
    assert api.download_calls == []


def test_download_reads_sample_header_and_never_downloads_train_or_test() -> None:
    fixture = _load_file_list()
    for file_object in fixture["files"]:
        file_object.pop("columns", None)
    FakeKaggleApi.list_response = fixture
    temp_paths: list[Path] = []

    def write_sample(_: str, file_name: str, path: Path) -> None:
        temp_paths.append(path)
        (path / file_name).write_text(
            "id,target\n1,0.25\n2,0.75\n",
            encoding="utf-8",
        )

    FakeKaggleApi.download_impl = write_sample

    manifest = fetch_file_manifest("example-competition", max_sample_sub_bytes=100)

    assert manifest.sample_submission_columns == ["id", "target"]
    assert manifest.sample_submission_source == "header_download"
    assert manifest.limitations == []
    assert [call["file_name"] for call in FakeKaggleApi.instances[0].download_calls] == [
        "sample_submission.csv"
    ]
    assert temp_paths and not temp_paths[0].exists()


def test_zip_sample_header_is_read_without_extracting_archive() -> None:
    FakeKaggleApi.list_response = {
        "files": [
            {
                "name": "sample_submission.csv.zip",
                "totalBytes": 90,
            }
        ]
    }

    def write_zip(_: str, file_name: str, path: Path) -> None:
        with zipfile.ZipFile(path / file_name, "w") as archive:
            archive.writestr("sample_submission.csv", "row_id,prediction\n1,0\n")

    FakeKaggleApi.download_impl = write_zip

    manifest = fetch_file_manifest("example-competition", max_sample_sub_bytes=100)

    assert manifest.sample_submission_columns == ["row_id", "prediction"]
    assert manifest.sample_submission_source == "header_download"


def test_ratio_is_none_when_train_or_test_is_missing() -> None:
    FakeKaggleApi.list_response = {
        "files": [
            {"name": "train.csv", "totalBytes": 100},
            {"name": "sample_submission.csv", "totalBytes": 5, "columns": ["target"]},
        ]
    }

    manifest = fetch_file_manifest("example-competition", max_sample_sub_bytes=100)

    assert manifest.train_test_size_ratio is None


def test_sample_at_download_limit_is_not_downloaded() -> None:
    FakeKaggleApi.list_response = {
        "files": [
            {"name": "sample_submission.csv", "totalBytes": 100},
        ]
    }

    manifest = fetch_file_manifest("example-competition", max_sample_sub_bytes=100)

    assert manifest.sample_submission_columns == []
    assert manifest.sample_submission_source == "unavailable"
    assert "is not below the 100-byte limit" in manifest.limitations[0]
    assert FakeKaggleApi.instances[0].download_calls == []


def test_listing_403_is_non_fatal_and_recorded_as_limitation() -> None:
    FakeKaggleApi.list_error = ForbiddenError("rules not accepted")

    manifest = fetch_file_manifest("restricted-competition", max_sample_sub_bytes=100)

    assert manifest.files == []
    assert manifest.train_test_size_ratio is None
    assert manifest.sample_submission_columns == []
    assert manifest.sample_submission_source == "unavailable"
    assert len(manifest.limitations) == 1
    assert "403" in manifest.limitations[0]


def test_sample_download_403_is_non_fatal_and_never_falls_back_to_other_files() -> None:
    FakeKaggleApi.list_response = {
        "files": [
            {"name": "train.csv", "totalBytes": 10},
            {"name": "test.csv", "totalBytes": 10},
            {"name": "sample_submission.csv", "totalBytes": 5},
        ]
    }

    def deny_download(_: str, __: str, ___: Path) -> None:
        raise ForbiddenError("rules not accepted")

    FakeKaggleApi.download_impl = deny_download

    manifest = fetch_file_manifest("restricted-competition", max_sample_sub_bytes=100)

    assert manifest.sample_submission_source == "unavailable"
    assert manifest.sample_submission_columns == []
    assert len(manifest.limitations) == 1
    assert "403" in manifest.limitations[0]
    assert [call["file_name"] for call in FakeKaggleApi.instances[0].download_calls] == [
        "sample_submission.csv"
    ]
