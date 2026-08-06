from __future__ import annotations

import csv
import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from kaggle_researcher.facts.kaggle_api import (
    GLOBAL_KAGGLE_POLICY,
    create_kaggle_api,
    is_forbidden,
    unpack_list_response,
)
from kaggle_researcher.facts.kaggle_api import (
    KaggleRequestPolicy as KaggleRequestPolicy,
)
from kaggle_researcher.facts.models import FileInfo, FileManifest

_FILE_REQUEST_POLICY = GLOBAL_KAGGLE_POLICY


class _SampleHeaderUnreadable(RuntimeError):
    """Raised after a successful download when the sample header cannot be read."""


def classify_role(name: str) -> str:
    basename = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if basename.startswith("train"):
        return "train"
    if basename.startswith("test"):
        return "test"
    if _submission_name_priority(basename) is not None:
        return "submission"
    return "auxiliary"


def fetch_file_manifest(
    slug: str,
    max_sample_sub_bytes: int,
    api: Any | None = None,
) -> FileManifest:
    if api is None:
        api = create_kaggle_api()
    limitations: list[str] = []

    try:
        raw_files = _list_competition_files(api, slug)
    except Exception as exc:
        if not is_forbidden(exc):
            raise
        limitations.append(
            "Kaggle API returned 403 while listing files; competition rules may not be accepted."
        )
        return FileManifest(
            files=[],
            train_test_size_ratio=None,
            sample_submission_columns=[],
            sample_submission_source=_sample_submission_source("download_forbidden"),
            sample_submission_status="download_forbidden",
            limitations=limitations,
        )

    file_records: list[tuple[FileInfo, Any]] = []
    for raw_file in raw_files:
        name = _get_file_value(raw_file, "name", "fileName", "file_name", "filename", "ref")
        if not isinstance(name, str) or not name:
            limitations.append("A file entry without a readable name was omitted.")
            continue
        size_bytes = _parse_size_bytes(
            _get_file_value(
                raw_file,
                "totalBytes",
                "total_bytes",
                "sizeBytes",
                "size_bytes",
                "size",
            )
        )
        file_info = FileInfo(
            name=name,
            size_bytes=size_bytes,
            role_hint=classify_role(name),
        )
        file_records.append((file_info, raw_file))

    files = [file_info for file_info, _ in file_records]
    ratio = _train_test_size_ratio(files)
    sample_candidates = [
        (file_info, raw_file)
        for file_info, raw_file in file_records
        if file_info.role_hint == "submission"
    ]
    sample_record = min(sample_candidates, key=_sample_record_sort_key, default=None)

    sample_columns: list[str] = []
    sample_status = "file_not_found"
    if sample_record is None:
        limitations.append("No sample_submission file was listed by the Kaggle API.")
    else:
        sample_file, raw_sample_file = sample_record
        sample_columns = _extract_columns(
            _get_file_value(
                raw_sample_file,
                "columns",
                "columnNames",
                "column_names",
            )
        )
        if sample_columns:
            sample_status = "api"
        elif sample_file.size_bytes is None:
            sample_status = "size_unknown"
            limitations.append(f"Cannot download {sample_file.name}: its size is unavailable.")
        elif sample_file.size_bytes >= max_sample_sub_bytes:
            sample_status = "size_over_limit"
            limitations.append(
                f"Cannot download {sample_file.name}: size {sample_file.size_bytes} bytes "
                f"is not below the {max_sample_sub_bytes}-byte limit."
            )
        else:
            try:
                sample_columns = _download_sample_header(api, slug, sample_file.name)
            except _SampleHeaderUnreadable:
                sample_status = "header_unreadable"
                limitations.append(
                    f"The downloaded {sample_file.name} did not contain a readable header."
                )
            except Exception as exc:
                if is_forbidden(exc):
                    sample_status = "download_forbidden"
                    limitations.append(
                        "Kaggle API returned 403 while downloading sample_submission; "
                        "competition rules may not be accepted."
                    )
                else:
                    sample_status = "download_failed"
                    limitations.append(
                        f"Kaggle API failed to download {sample_file.name} "
                        f"({type(exc).__name__})."
                    )
            else:
                if sample_columns:
                    sample_status = "full_download"
                else:
                    sample_status = "header_unreadable"
                    limitations.append(
                        f"The downloaded {sample_file.name} did not contain a readable header."
                    )

    return FileManifest(
        files=files,
        train_test_size_ratio=ratio,
        sample_submission_columns=sample_columns,
        sample_submission_source=_sample_submission_source(sample_status),
        sample_submission_status=sample_status,
        limitations=limitations,
    )


def _submission_name_priority(name: str) -> int | None:
    basename = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if basename == "sample_submission.csv":
        return 0
    if "sample_submission" in basename or "sample-submission" in basename:
        return 1
    if "submission" in basename and Path(basename).suffix == ".csv":
        return 2
    return None


def _sample_record_sort_key(record: tuple[FileInfo, Any]) -> tuple[int, bool, int, str]:
    file_info, _ = record
    priority = _submission_name_priority(file_info.name)
    if priority is None:
        raise ValueError("sample record does not have a submission filename")
    return (
        priority,
        file_info.size_bytes is None,
        file_info.size_bytes or 0,
        file_info.name.casefold(),
    )


def _sample_submission_source(status: str) -> str:
    if status == "api":
        return "api"
    if status == "full_download":
        return "full_download"
    return "unavailable"


def _list_competition_files(api: Any, slug: str) -> list[Any]:
    files: list[Any] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()

    while True:
        if page_token is None:
            response = _FILE_REQUEST_POLICY.call(lambda: api.competition_list_files(slug))
        else:
            response = _FILE_REQUEST_POLICY.call(
                lambda token=page_token: api.competition_list_files(
                    slug,
                    page_token=token,
                )
            )
        unpacked = unpack_list_response(response, "files")
        files.extend(unpacked.items)

        next_page_token = unpacked.next_page_token
        if not next_page_token or next_page_token in seen_tokens or not unpacked.items:
            break
        seen_tokens.add(next_page_token)
        page_token = next_page_token

    return files


def _train_test_size_ratio(files: list[FileInfo]) -> float | None:
    train_files = [file for file in files if file.role_hint == "train"]
    test_files = [file for file in files if file.role_hint == "test"]
    if not train_files or not test_files:
        return None
    if any(file.size_bytes is None for file in [*train_files, *test_files]):
        return None

    train_size = sum(file.size_bytes or 0 for file in train_files)
    test_size = sum(file.size_bytes or 0 for file in test_files)
    if test_size == 0:
        return None
    return train_size / test_size


def _download_sample_header(api: Any, slug: str, file_name: str) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="kaggle_sample_submission_") as temp_dir:
        _FILE_REQUEST_POLICY.call(
            lambda: api.competition_download_file(
                slug,
                file_name,
                path=temp_dir,
                quiet=True,
            )
        )
        try:
            downloaded_path = _find_downloaded_file(Path(temp_dir), file_name)
            return _read_header(downloaded_path)
        except Exception as exc:
            raise _SampleHeaderUnreadable(str(exc)) from exc


def _find_downloaded_file(temp_dir: Path, requested_name: str) -> Path:
    downloaded_files = sorted(path for path in temp_dir.rglob("*") if path.is_file())
    if not downloaded_files:
        raise RuntimeError(f"Kaggle download for {requested_name} did not produce a file.")

    requested_basename = requested_name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for path in downloaded_files:
        if path.name.lower() == requested_basename:
            return path

    found_names = [str(path.relative_to(temp_dir)) for path in downloaded_files]
    raise RuntimeError(
        f"Kaggle download for {requested_name!r} did not produce the expected filename. "
        f"Found: {found_names}"
    )


def _read_header(path: Path) -> list[str]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = sorted(member for member in archive.namelist() if not member.endswith("/"))
            if not members:
                return []
            with archive.open(members[0]) as binary_stream:
                with io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="") as stream:
                    return _read_csv_header(stream)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return _read_csv_header(stream)


def _read_csv_header(stream: Any) -> list[str]:
    try:
        return next(csv.reader(stream))
    except StopIteration:
        return []


def _extract_columns(raw_columns: Any) -> list[str]:
    if raw_columns is None:
        return []
    if isinstance(raw_columns, str):
        return [raw_columns] if raw_columns else []
    if not isinstance(raw_columns, list | tuple):
        return []

    columns: list[str] = []
    for raw_column in raw_columns:
        if isinstance(raw_column, str):
            name = raw_column
        else:
            name = _get_file_value(
                raw_column,
                "name",
                "columnName",
                "column_name",
            )
        if isinstance(name, str) and name:
            columns.append(name)
    return columns


def _parse_size_bytes(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        size = int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return size if size >= 0 else None


def _get_file_value(file_object: Any, *names: str) -> Any:
    if file_object is None:
        return None
    if isinstance(file_object, dict):
        normalized_values = {_normalize_key(str(key)): value for key, value in file_object.items()}
        for name in names:
            value = normalized_values.get(_normalize_key(name))
            if value is not None and value != "":
                return value
        return None

    for name in names:
        try:
            value = getattr(file_object, name)
        except Exception:
            continue
        if value is not None and value != "":
            return value

    normalized_names = {_normalize_key(name) for name in names}
    for attribute_name in dir(file_object):
        if _normalize_key(attribute_name) not in normalized_names:
            continue
        try:
            value = getattr(file_object, attribute_name)
        except Exception:
            continue
        if value is not None and value != "":
            return value
    return None


def _normalize_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())
