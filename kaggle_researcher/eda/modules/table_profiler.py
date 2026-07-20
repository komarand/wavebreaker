from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.schemas import (
    ColumnProfile,
    FileInventoryResult,
    InferredSchema,
    TableProfile,
)


READABLE_TABULAR_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}
MOSTLY_MISSING_THRESHOLD = 0.8
HIGH_CARDINALITY_RATIO_THRESHOLD = 0.9
TOP_VALUE_LIMIT = 5
DATE_PARSE_SUCCESS_THRESHOLD = 0.8
DATE_PARSE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%m/%d/%Y")


def profile_tables(
    file_inventory: FileInventoryResult,
    inferred_schema: InferredSchema,
    reader: DatasetReader,
    sample_rows: int = 200_000,
    max_full_scan_rows: int = 2_000_000,
    max_table_bytes: int | None = None,
    max_column_cardinality_scan_rows: int | None = None,
) -> list[TableProfile]:
    """Build safe profiles for readable tabular files."""

    del inferred_schema
    if sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer")
    if max_full_scan_rows < 0:
        raise ValueError("max_full_scan_rows must be non-negative")
    if max_table_bytes is not None and max_table_bytes <= 0:
        raise ValueError("max_table_bytes must be a positive integer")
    if (
        max_column_cardinality_scan_rows is not None
        and max_column_cardinality_scan_rows <= 0
    ):
        raise ValueError("max_column_cardinality_scan_rows must be a positive integer")

    profiles: list[TableProfile] = []
    for dataset_file in file_inventory.files:
        if not dataset_file.can_read or dataset_file.extension not in READABLE_TABULAR_EXTENSIONS:
            continue

        warnings: list[str] = []
        n_rows = _safe_count_rows(reader, dataset_file.path, warnings)
        try:
            schema = reader.read_schema(dataset_file.path)
        except ReaderError as exc:
            profiles.append(
                TableProfile(
                    table_name=Path(dataset_file.name).stem,
                    path=dataset_file.path,
                    n_rows=n_rows,
                    n_cols=0,
                    warnings=[*warnings, str(exc)],
                )
            )
            continue

        columns = [column["name"] for column in schema]
        sample_reasons = _sample_reasons(
            reader=reader,
            relative_path=dataset_file.path,
            n_rows=n_rows,
            max_full_scan_rows=max_full_scan_rows,
            max_table_bytes=max_table_bytes,
            max_column_cardinality_scan_rows=max_column_cardinality_scan_rows,
            warnings=warnings,
        )
        sampled = bool(sample_reasons)
        effective_sample_rows = _effective_sample_rows(
            sample_rows,
            max_column_cardinality_scan_rows=max_column_cardinality_scan_rows,
        )
        frame = _read_profile_frame(
            reader,
            dataset_file.path,
            columns,
            sampled=sampled,
            sample_rows=effective_sample_rows,
            warnings=warnings,
        )
        if sampled:
            warnings.extend(sample_reasons)
            warnings.append(f"Profile is based on a bounded sample of {frame.height} rows.")

        column_profiles = [_profile_column(frame[column]) for column in frame.columns]
        mostly_missing_columns = [
            column.name for column in column_profiles if column.is_mostly_missing
        ]
        high_cardinality_columns = [
            column.name for column in column_profiles if column.is_high_cardinality
        ]
        constant_columns = [column.name for column in column_profiles if column.is_constant]

        profiles.append(
            TableProfile(
                table_name=Path(dataset_file.name).stem,
                path=dataset_file.path,
                n_rows=n_rows,
                n_cols=len(columns),
                sampled=sampled,
                sample_rows=frame.height if sampled else None,
                columns=column_profiles,
                mostly_missing_columns=mostly_missing_columns,
                high_cardinality_columns=high_cardinality_columns,
                constant_columns=constant_columns,
                warnings=warnings,
            )
        )

    return profiles


def _sample_reasons(
    *,
    reader: DatasetReader,
    relative_path: str,
    n_rows: int | None,
    max_full_scan_rows: int,
    max_table_bytes: int | None,
    max_column_cardinality_scan_rows: int | None,
    warnings: list[str],
) -> list[str]:
    reasons: list[str] = []
    if n_rows is None:
        reasons.append("Profile sampled because row count is unavailable.")
    elif n_rows > max_full_scan_rows:
        reasons.append(
            f"Profile sampled because row count {n_rows} exceeds "
            f"EDA_MAX_PROFILE_ROWS_FULL_SCAN={max_full_scan_rows}."
        )
    if (
        max_column_cardinality_scan_rows is not None
        and n_rows is not None
        and n_rows > max_column_cardinality_scan_rows
    ):
        reasons.append(
            f"Profile sampled because row count {n_rows} exceeds "
            "EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS="
            f"{max_column_cardinality_scan_rows}."
        )
    if max_table_bytes is not None:
        try:
            size_bytes = reader.file_size_bytes(relative_path)
        except ReaderError as exc:
            warnings.append(f"Could not read file size for profiling: {exc}")
        else:
            if size_bytes > max_table_bytes:
                reasons.append(
                    f"Profile sampled because file size {size_bytes} bytes exceeds "
                    f"EDA_MAX_TABLE_BYTES={max_table_bytes}."
                )
    return reasons


def _effective_sample_rows(
    sample_rows: int,
    *,
    max_column_cardinality_scan_rows: int | None,
) -> int:
    if max_column_cardinality_scan_rows is None:
        return sample_rows
    return min(sample_rows, max_column_cardinality_scan_rows)


def _safe_count_rows(
    reader: DatasetReader,
    relative_path: str,
    warnings: list[str],
) -> int | None:
    try:
        return reader.count_rows(relative_path)
    except ReaderError as exc:
        warnings.append(f"Could not count rows: {exc}")
        return None


def _read_profile_frame(
    reader: DatasetReader,
    relative_path: str,
    columns: list[str],
    *,
    sampled: bool,
    sample_rows: int,
    warnings: list[str],
) -> pl.DataFrame:
    try:
        if sampled:
            return reader.sample_table(relative_path, n_rows=sample_rows)
        return reader.read_columns(relative_path, columns=columns)
    except ReaderError as exc:
        warnings.append(f"Could not read profile rows: {exc}")
        return pl.DataFrame({column: [] for column in columns})


def _profile_column(series: pl.Series) -> ColumnProfile:
    row_count = len(series)
    missing_mask = _missing_mask(series)
    missing_count = int(missing_mask.sum())
    non_missing_count = row_count - missing_count
    non_missing = series.filter(~missing_mask)
    n_unique = int(non_missing.n_unique()) if non_missing_count else 0
    missing_pct = _ratio(missing_count, row_count)
    unique_ratio = _ratio(n_unique, non_missing_count)
    is_mostly_missing = missing_pct >= MOSTLY_MISSING_THRESHOLD if row_count else False
    is_constant = row_count > 0 and missing_count == 0 and n_unique == 1
    is_high_cardinality = (
        non_missing_count > 0
        and n_unique > 1
        and unique_ratio >= HIGH_CARDINALITY_RATIO_THRESHOLD
    )

    profile_data: dict[str, Any] = {
        "name": series.name,
        "dtype": str(series.dtype),
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "n_unique": n_unique,
        "unique_ratio": unique_ratio,
        "top_values": _top_values(non_missing),
        "date_min": None,
        "date_max": None,
        "is_constant": is_constant,
        "is_mostly_missing": is_mostly_missing,
        "is_high_cardinality": is_high_cardinality,
    }

    if _is_numeric(series):
        profile_data.update(_numeric_stats(non_missing))

    date_range = _date_range(non_missing, non_missing_count=non_missing_count)
    if date_range is not None:
        profile_data["date_min"] = date_range[0]
        profile_data["date_max"] = date_range[1]

    return ColumnProfile(**profile_data)


def _numeric_stats(series: pl.Series) -> dict[str, float | int | None]:
    if len(series) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "q01": None,
            "q05": None,
            "q50": None,
            "q95": None,
            "q99": None,
        }
    return {
        "mean": _to_builtin_number(series.mean()),
        "std": _to_builtin_number(series.std()),
        "min": _to_builtin_number(series.min()),
        "max": _to_builtin_number(series.max()),
        "q01": _to_builtin_number(series.quantile(0.01)),
        "q05": _to_builtin_number(series.quantile(0.05)),
        "q50": _to_builtin_number(series.quantile(0.50)),
        "q95": _to_builtin_number(series.quantile(0.95)),
        "q99": _to_builtin_number(series.quantile(0.99)),
    }


def _top_values(series: pl.Series) -> list[dict[str, Any]]:
    if len(series) == 0:
        return []
    counts = (
        series.value_counts(sort=True)
        .head(TOP_VALUE_LIMIT)
        .rename({"count": "n"})
        .to_dicts()
    )
    values: list[dict[str, Any]] = []
    for item in counts:
        value = item.get(series.name)
        count = item.get("n", 0)
        values.append({"value": _normalise_value(value), "count": int(count)})
    return values


def _missing_mask(series: pl.Series) -> pl.Series:
    mask = series.is_null()
    if _is_string(series):
        mask = mask | (series.str.strip_chars() == "")
    if _is_numeric(series):
        mask = mask | series.is_nan().fill_null(False)
    return mask


def _date_range(
    series: pl.Series,
    *,
    non_missing_count: int,
) -> tuple[str, str] | None:
    if non_missing_count == 0:
        return None
    if series.dtype in {pl.Date, pl.Datetime}:
        return str(series.min()), str(series.max())
    if not _is_string(series):
        return None

    for date_format in DATE_PARSE_FORMATS:
        try:
            parsed = series.str.strptime(pl.Date, format=date_format, strict=False).drop_nulls()
        except pl.exceptions.PolarsError:
            continue
        if len(parsed) == 0:
            continue
        parse_ratio = len(parsed) / non_missing_count
        if parse_ratio >= DATE_PARSE_SUCCESS_THRESHOLD:
            return str(parsed.min()), str(parsed.max())
    return None


def _is_numeric(series: pl.Series) -> bool:
    return series.dtype.is_numeric()


def _is_string(series: pl.Series) -> bool:
    return series.dtype == pl.String or series.dtype == pl.Utf8


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _to_builtin_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


__all__ = ["profile_tables"]
