from __future__ import annotations

import csv
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from kaggle_researcher.facts.files import _find_downloaded_file
from kaggle_researcher.facts.kaggle_api import (
    GLOBAL_KAGGLE_POLICY,
    create_kaggle_api,
    is_forbidden,
)
from kaggle_researcher.facts.models import (
    ColumnShape,
    DatasetShape,
    FileInfo,
    FileManifest,
    TargetShape,
)

_DATASET_REQUEST_POLICY = GLOBAL_KAGGLE_POLICY
_SUPPORTED_SUFFIXES = {".csv", ".tsv"}
_SAMPLE_VALUE_LIMIT = 5
_SAMPLE_VALUE_LENGTH = 32


@dataclass(frozen=True, slots=True)
class _ReadTable:
    columns: list[str]
    row_count: int
    sample: list[dict[str, str]]


def read_dataset_shape(
    slug: str,
    manifest: FileManifest,
    max_bytes: int,
    sample_rows: int,
    api: Any | None = None,
) -> DatasetShape | None:
    """Read bounded CSV/TSV structure without profiling the full dataset."""
    if max_bytes <= 0 or sample_rows <= 0:
        return _unavailable("Dataset read limits must be positive integers.")

    train_file, train_limitation = _select_file(manifest, "train")
    test_file, test_limitation = _select_file(manifest, "test")
    limitations = [
        limitation
        for limitation in (train_limitation, test_limitation)
        if limitation is not None
    ]
    if train_file is None and test_file is None:
        return _unavailable(*limitations)

    readable: dict[str, FileInfo] = {}
    for role, file_info in (("train", train_file), ("test", test_file)):
        if file_info is None:
            continue
        if file_info.size_bytes is None:
            limitations.append(
                f"Cannot read {file_info.name}: its size is unavailable, so the byte "
                "limit cannot be enforced."
            )
        elif file_info.size_bytes > max_bytes:
            limitations.append(
                f"Did not download {file_info.name}: size {file_info.size_bytes} bytes exceeds "
                f"the {max_bytes}-byte dataset read limit."
            )
        else:
            readable[role] = file_info

    if not readable:
        return _unavailable(*limitations)

    if api is None:
        try:
            api = create_kaggle_api()
        except Exception as exc:
            return _unavailable(
                *limitations,
                f"Kaggle API setup failed while reading dataset shape ({type(exc).__name__}).",
            )

    tables: dict[str, _ReadTable] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="kaggle_dataset_shape_") as temp_dir:
            destination = Path(temp_dir)
            for role in ("train", "test"):
                file_info = readable.get(role)
                if file_info is None:
                    continue
                _DATASET_REQUEST_POLICY.call(
                    lambda item=file_info: api.competition_download_file(
                        slug,
                        item.name,
                        path=temp_dir,
                        quiet=True,
                    )
                )
                downloaded_path = _find_downloaded_file(destination, file_info.name)
                tables[role] = _read_table(
                    downloaded_path,
                    delimiter="\t" if Path(file_info.name).suffix.lower() == ".tsv" else ",",
                    sample_rows=sample_rows if role == "train" else 0,
                )
    except Exception as exc:
        if is_forbidden(exc):
            reason = (
                "Kaggle API returned 403 while downloading train/test data; "
                "competition rules may not be accepted."
            )
        else:
            reason = f"Dataset shape read failed ({type(exc).__name__}: {exc})."
        return _unavailable(*limitations, reason)

    train = tables.get("train")
    test = tables.get("test")
    columns = _column_shapes(train) if train is not None else []
    target = None
    if train is not None and test is not None:
        target_names = [name for name in train.columns if name not in set(test.columns)]
        if len(target_names) == 1:
            target = _target_shape(target_names[0], train.sample)
        elif not target_names:
            limitations.append(
                "Target is unavailable: no train column is absent from test columns."
            )
        else:
            limitations.append(
                "Target is unavailable: more than one train column is absent from test "
                f"columns ({', '.join(target_names)})."
            )
    else:
        limitations.append(
            "Target is unavailable because both train and test headers were not read."
        )

    train_rows = train.row_count if train is not None else None
    test_rows = test.row_count if test is not None else None
    if train is not None and train.row_count > len(train.sample):
        limitations.append(
            f"Column and target statistics use only the first {len(train.sample)} train "
            "rows; full-file processing was limited to row counting."
        )
    ratio = (
        round(train_rows / test_rows, 6)
        if train_rows is not None and test_rows not in (None, 0)
        else None
    )
    both_read = train is not None and test is not None
    coverage = (
        "full_file"
        if both_read
        else "sampled"
        if train is not None and train.sample
        else "header_only"
    )
    return DatasetShape(
        status="read" if both_read else "partial",
        train_rows=train_rows,
        test_rows=test_rows,
        train_test_row_ratio=ratio,
        sampled_rows=len(train.sample) if train is not None else 0,
        columns=columns,
        target=target,
        coverage=coverage,
        limitations=limitations,
    )


def _select_file(
    manifest: FileManifest,
    role: Literal["train", "test"],
) -> tuple[FileInfo | None, str | None]:
    role_files = [file for file in manifest.files if file.role_hint == role]
    supported = [
        file for file in role_files if Path(file.name).suffix.lower() in _SUPPORTED_SUFFIXES
    ]
    if len(supported) == 1:
        return supported[0], None
    if len(supported) > 1:
        return None, f"Dataset shape does not guess among multiple {role} CSV/TSV files."
    if role_files:
        formats = ", ".join(
            sorted(
                {
                    Path(file.name).suffix.lower() or "no extension"
                    for file in role_files
                }
            )
        )
        return None, f"Dataset shape cannot read {role} format(s): {formats}."
    return None, f"No {role} file was identified in the competition manifest."


def _read_table(path: Path, *, delimiter: str, sample_rows: int) -> _ReadTable:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path.name} has no header") from exc
        if not columns or any(not column for column in columns):
            raise ValueError(f"{path.name} has an unreadable header")

        row_count = 0
        sample: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            row_count += 1
            if len(sample) < sample_rows:
                sample.append(
                    {
                        column: row[index].strip() if index < len(row) else ""
                        for index, column in enumerate(columns)
                    }
                )
    return _ReadTable(columns=columns, row_count=row_count, sample=sample)


def _column_shapes(table: _ReadTable) -> list[ColumnShape]:
    return [
        _column_shape(column, [row.get(column, "") for row in table.sample])
        for column in table.columns
    ]


def _column_shape(name: str, values: list[str]) -> ColumnShape:
    nonempty = [value for value in values if value != ""]
    distinct = list(dict.fromkeys(nonempty))
    null_share = round((len(values) - len(nonempty)) / len(values), 4) if values else None
    return ColumnShape(
        name=name,
        inferred_type=_infer_type(nonempty),
        distinct_in_sample=len(set(nonempty)),
        null_share_in_sample=null_share,
        sample_values=[
            value[:_SAMPLE_VALUE_LENGTH] for value in distinct[:_SAMPLE_VALUE_LIMIT]
        ],
    )


def _infer_type(values: list[str]) -> Literal[
    "integer", "float", "string", "boolean", "unknown"
]:
    if not values:
        return "unknown"
    if all(_parses_int(value) for value in values):
        return "integer"
    if all(_parses_float(value) for value in values):
        return "float"
    if all(value.casefold() in {"true", "false"} for value in values):
        return "boolean"
    return "string"


def _parses_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _parses_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _target_shape(column: str, sample: list[dict[str, str]]) -> TargetShape:
    values = [row.get(column, "") for row in sample]
    counts = Counter(value for value in values if value != "")
    distinct = len(counts)
    return TargetShape(
        column=column,
        distinct_in_sample=distinct,
        class_counts_in_sample=dict(counts) if distinct <= 20 else None,
        is_binary_in_sample=(distinct == 2) if distinct else None,
    )


def _unavailable(*limitations: str) -> DatasetShape:
    return DatasetShape(
        status="unavailable",
        train_rows=None,
        test_rows=None,
        train_test_row_ratio=None,
        sampled_rows=0,
        columns=[],
        target=None,
        coverage="none",
        limitations=[item for item in limitations if item],
    )
