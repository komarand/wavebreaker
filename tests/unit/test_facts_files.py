from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kaggle_researcher.facts.files as files_module
from kaggle_researcher.config import DEFAULT_MAX_SAMPLE_SUB_BYTES
from kaggle_researcher.facts.files import (
    _find_downloaded_file,
    classify_role,
    fetch_file_manifest,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "facts" / "file_list.json"


class ForbiddenError(RuntimeError):
    status = 403


class FakeKaggleApi:
    list_response: Any = None
    list_error: Exception | None = None
    download_impl: Callable[[str, str, Path], None] | None = None
    instances: list[FakeKaggleApi] = []

    def __init__(self) -> None:
        self.authenticated = False
        self.list_calls: list[str] = []
        self.download_calls: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def authenticate(self) -> None:
        self.authenticated = True

    def competition_list_files(self, competition: str, **kwargs: Any) -> Any:
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

    def create_api() -> FakeKaggleApi:
        api = FakeKaggleApi()
        api.authenticate()
        return api

    monkeypatch.setattr(files_module, "create_kaggle_api", create_api)
    monkeypatch.setattr(
        files_module,
        "_FILE_REQUEST_POLICY",
        files_module.KaggleRequestPolicy(
            base_delay_seconds=0,
            min_interval_seconds=0,
        ),
    )


def _load_file_list() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_file_listing_retries_transient_rate_limit() -> None:
    class RateLimited(RuntimeError):
        status = 429

    class TransientApi:
        def __init__(self) -> None:
            self.calls = 0

        def competition_list_files(self, competition: str) -> dict[str, Any]:
            self.calls += 1
            if self.calls < 3:
                raise RateLimited("rate limited")
            return {"files": []}

    api = TransientApi()

    manifest = fetch_file_manifest("example-competition", 100, api=api)

    assert manifest.files == []
    assert api.calls == 3


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("train.csv", "train"),
        ("nested/Train_features.parquet", "train"),
        (r"nested\test_data.csv", "test"),
        ("prefix_sample_submission_v2.csv", "submission"),
        ("sample-submission-v2.csv", "submission"),
        ("submission.csv", "submission"),
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
    assert manifest.sample_submission_status == "api"
    assert manifest.limitations == []

    api = FakeKaggleApi.instances[0]
    assert api.authenticated is True
    assert api.list_calls == ["example-competition"]
    assert api.download_calls == []


def test_kaggle_2_file_response_paginates_with_next_page_token() -> None:
    class PaginatedApi:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def competition_list_files(
            self,
            competition: str,
            *,
            page_token: str | None = None,
        ) -> Any:
            assert competition == "example-competition"
            self.calls.append(page_token)
            if page_token is None:
                return SimpleNamespace(
                    files=[{"name": "train.csv", "totalBytes": 100}],
                    next_page_token="page-2",
                )
            return SimpleNamespace(
                files=[
                    {"name": "test.csv", "totalBytes": 50},
                    {
                        "name": "sample_submission.csv",
                        "totalBytes": 5,
                        "columns": ["id", "target"],
                    },
                ],
                next_page_token=None,
            )

    api = PaginatedApi()

    manifest = fetch_file_manifest(
        "example-competition",
        max_sample_sub_bytes=100,
        api=api,
    )

    assert api.calls == [None, "page-2"]
    assert [file.name for file in manifest.files] == [
        "train.csv",
        "test.csv",
        "sample_submission.csv",
    ]
    assert manifest.train_test_size_ratio == 2.0


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
    assert manifest.sample_submission_source == "full_download"
    assert manifest.sample_submission_status == "full_download"
    assert manifest.limitations == []
    assert [call["file_name"] for call in FakeKaggleApi.instances[0].download_calls] == [
        "sample_submission.csv"
    ]
    assert temp_paths and not temp_paths[0].exists()


def test_default_limit_downloads_multi_megabyte_sample_submission() -> None:
    FakeKaggleApi.list_response = {
        "files": [
            {
                "name": "sample_submission.csv",
                "totalBytes": 7_703_870,
            }
        ]
    }

    def write_sample(_: str, file_name: str, path: Path) -> None:
        (path / file_name).write_text("id,target\n1,0.5\n", encoding="utf-8")

    FakeKaggleApi.download_impl = write_sample

    manifest = fetch_file_manifest(
        "playground",
        max_sample_sub_bytes=DEFAULT_MAX_SAMPLE_SUB_BYTES,
    )

    assert DEFAULT_MAX_SAMPLE_SUB_BYTES == 50_000_000
    assert manifest.sample_submission_status == "full_download"
    assert manifest.sample_submission_columns == ["id", "target"]


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
    assert manifest.sample_submission_source == "full_download"
    assert manifest.sample_submission_status == "full_download"


def test_submission_csv_without_sample_prefix_is_downloaded() -> None:
    FakeKaggleApi.list_response = {"files": [{"name": "submission.csv", "totalBytes": 20}]}

    def write_sample(_: str, file_name: str, path: Path) -> None:
        (path / file_name).write_text("id,target\n1,0.5\n", encoding="utf-8")

    FakeKaggleApi.download_impl = write_sample

    manifest = fetch_file_manifest("playground", max_sample_sub_bytes=100)

    assert manifest.sample_submission_columns == ["id", "target"]
    assert manifest.sample_submission_status == "full_download"
    assert manifest.sample_submission_source == "full_download"


def test_submission_candidate_priority_precedes_file_size() -> None:
    FakeKaggleApi.list_response = {
        "files": [
            {"name": "submission.csv", "totalBytes": 1},
            {"name": "sample-submission.csv", "totalBytes": 2},
            {"name": "sample_submission.csv", "totalBytes": 30},
        ]
    }

    def write_sample(_: str, file_name: str, path: Path) -> None:
        (path / file_name).write_text("row_id,prediction\n", encoding="utf-8")

    FakeKaggleApi.download_impl = write_sample

    manifest = fetch_file_manifest("example", max_sample_sub_bytes=100)

    assert manifest.sample_submission_status == "full_download"
    assert FakeKaggleApi.instances[0].download_calls[0]["file_name"] == ("sample_submission.csv")


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
    assert manifest.sample_submission_status == "size_over_limit"
    assert "is not below the 100-byte limit" in manifest.limitations[0]
    assert FakeKaggleApi.instances[0].download_calls == []


def test_listing_403_is_non_fatal_and_recorded_as_limitation() -> None:
    FakeKaggleApi.list_error = ForbiddenError("rules not accepted")

    manifest = fetch_file_manifest("restricted-competition", max_sample_sub_bytes=100)

    assert manifest.files == []
    assert manifest.train_test_size_ratio is None
    assert manifest.sample_submission_columns == []
    assert manifest.sample_submission_source == "unavailable"
    assert manifest.sample_submission_status == "download_forbidden"
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
    assert manifest.sample_submission_status == "download_forbidden"
    assert manifest.sample_submission_columns == []
    assert len(manifest.limitations) == 1
    assert "403" in manifest.limitations[0]
    assert [call["file_name"] for call in FakeKaggleApi.instances[0].download_calls] == [
        "sample_submission.csv"
    ]


def test_missing_submission_file_has_structured_status() -> None:
    FakeKaggleApi.list_response = {"files": [{"name": "train.csv", "totalBytes": 10}]}

    manifest = fetch_file_manifest("example", max_sample_sub_bytes=100)

    assert manifest.sample_submission_status == "file_not_found"
    assert manifest.sample_submission_source == "unavailable"


def test_unknown_submission_size_has_structured_status() -> None:
    FakeKaggleApi.list_response = {"files": [{"name": "submission.csv"}]}

    manifest = fetch_file_manifest("example", max_sample_sub_bytes=100)

    assert manifest.sample_submission_status == "size_unknown"
    assert manifest.sample_submission_source == "unavailable"


def test_non_forbidden_download_failure_is_non_fatal_and_structured() -> None:
    FakeKaggleApi.list_response = {"files": [{"name": "submission.csv", "totalBytes": 10}]}

    def fail_download(_: str, __: str, ___: Path) -> None:
        raise RuntimeError("temporary failure")

    FakeKaggleApi.download_impl = fail_download

    manifest = fetch_file_manifest("example", max_sample_sub_bytes=100)

    assert manifest.sample_submission_status == "download_failed"
    assert manifest.sample_submission_source == "unavailable"
    assert "RuntimeError" in manifest.limitations[0]


def test_downloaded_file_with_empty_header_is_structured() -> None:
    FakeKaggleApi.list_response = {"files": [{"name": "submission.csv", "totalBytes": 10}]}

    def write_empty(_: str, file_name: str, path: Path) -> None:
        (path / file_name).write_text("", encoding="utf-8")

    FakeKaggleApi.download_impl = write_empty

    manifest = fetch_file_manifest("example", max_sample_sub_bytes=100)

    assert manifest.sample_submission_status == "header_unreadable"
    assert manifest.sample_submission_source == "unavailable"


def test_find_downloaded_file_returns_exact_expected_file(tmp_path: Path) -> None:
    expected = tmp_path / "sample_submission.csv"
    expected.write_text("id,target\n", encoding="utf-8")
    (tmp_path / "unrelated.csv").write_text("wrong\n", encoding="utf-8")

    assert _find_downloaded_file(tmp_path, "sample_submission.csv") == expected


@pytest.mark.parametrize("found_names", [["wrong.csv"], ["a.csv", "b.csv"]])
def test_find_downloaded_file_fails_closed_on_name_mismatch(
    tmp_path: Path,
    found_names: list[str],
) -> None:
    for name in found_names:
        (tmp_path / name).write_text("wrong\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        _find_downloaded_file(tmp_path, "sample_submission.csv")

    message = str(exc_info.value)
    assert "sample_submission.csv" in message
    for name in found_names:
        assert name in message
