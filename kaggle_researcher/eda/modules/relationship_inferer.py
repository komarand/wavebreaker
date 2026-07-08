from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.schemas import FileInventoryResult, InferredSchema, TableSchema


READABLE_TABULAR_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}
DEFAULT_MAX_CHECK_ROWS = 200_000
ID_ENTITY_TOKENS = (
    "case",
    "customer",
    "client",
    "user",
    "order",
    "query",
    "group",
    "entity",
    "session",
    "account",
    "patient",
)
GROUP_TOKENS = (
    "group",
    "query",
    "customer",
    "client",
    "user",
    "entity",
    "session",
    "account",
    "patient",
)
DATE_TOKENS = (
    "date",
    "time",
    "timestamp",
    "dt",
    "week",
    "month",
    "year",
    "period",
    "cutoff",
)


def infer_relationships(
    inferred_schema: InferredSchema,
    file_inventory: FileInventoryResult,
    reader: DatasetReader,
    max_check_rows: int = DEFAULT_MAX_CHECK_ROWS,
) -> dict[str, Any]:
    """Infer join relationships between the train base table and secondary tables."""

    if max_check_rows <= 0:
        raise ValueError("max_check_rows must be a positive integer")

    warnings: list[str] = []
    table_schemas = {table.path: table for table in inferred_schema.tables}
    base_table = _find_base_table(inferred_schema, file_inventory, table_schemas)
    if base_table is None:
        return {
            "base_table": None,
            "base_id_column": None,
            "relationships": [],
            "warnings": ["Base train table could not be inferred."],
        }

    base_columns, base_column_dtypes = _schema_columns(reader, base_table, table_schemas, warnings)
    base_id_column = _find_base_id_column(
        inferred_schema,
        base_table=base_table,
        base_columns=base_columns,
        table_schemas=table_schemas,
    )
    if base_id_column is None:
        warnings.append(f"{base_table}: base id column could not be inferred.")

    relationships: list[dict[str, Any]] = []
    for dataset_file in file_inventory.files:
        if not _is_secondary_train_test_table(dataset_file, base_table=base_table):
            continue

        table_path = dataset_file.path
        table_schema = table_schemas.get(table_path)
        table_columns, table_column_dtypes = _schema_columns(
            reader,
            table_path,
            table_schemas,
            warnings,
        )
        candidate_join_keys = _candidate_join_keys(
            inferred_schema,
            base_columns=base_columns,
            base_id_column=base_id_column,
            table_columns=table_columns,
            table_schema=table_schema,
        )
        selected_join_key = candidate_join_keys[0] if candidate_join_keys else None
        candidate_group_keys = _candidate_group_keys(table_columns)
        candidate_date_cutoff_columns = _candidate_date_columns(
            table_columns,
            column_dtypes=table_column_dtypes,
            table_schema=table_schema,
        )

        if selected_join_key is None:
            relationships.append(
                {
                    "table": table_path,
                    "role": dataset_file.role_hint,
                    "table_hint": dataset_file.table_hint,
                    "candidate_join_keys": candidate_join_keys,
                    "selected_join_key": None,
                    "relationship_type": "unknown",
                    "coverage_left_to_right": None,
                    "orphan_rate_right": None,
                    "avg_rows_per_left": None,
                    "max_rows_per_left": None,
                    "row_multiplication_risk": "unknown",
                    "requires_aggregation": False,
                    "candidate_group_keys": candidate_group_keys,
                    "candidate_date_cutoff_columns": candidate_date_cutoff_columns,
                    "confidence": "low",
                    "sampled": False,
                    "sample_rows": None,
                    "warnings": [
                        "No shared id-like join key was found between base and secondary table."
                    ],
                }
            )
            continue

        relationships.append(
            _relationship_stats(
                reader,
                base_table=base_table,
                table_path=table_path,
                join_key=selected_join_key,
                role=dataset_file.role_hint,
                table_hint=dataset_file.table_hint,
                candidate_join_keys=candidate_join_keys,
                candidate_group_keys=candidate_group_keys,
                candidate_date_cutoff_columns=candidate_date_cutoff_columns,
                max_check_rows=max_check_rows,
            )
        )

    return {
        "base_table": base_table,
        "base_id_column": base_id_column,
        "relationships": relationships,
        "warnings": warnings,
    }


def _find_base_table(
    inferred_schema: InferredSchema,
    file_inventory: FileInventoryResult,
    table_schemas: dict[str, TableSchema],
) -> str | None:
    if inferred_schema.train_base_table:
        return inferred_schema.train_base_table

    for table in table_schemas.values():
        if table.role == "train" and table.table_type == "base":
            return table.path

    for dataset_file in file_inventory.files:
        if (
            dataset_file.can_read
            and dataset_file.extension in READABLE_TABULAR_EXTENSIONS
            and dataset_file.role_hint == "train"
            and dataset_file.table_hint == "base"
        ):
            return dataset_file.path

    train_files = [
        file
        for file in file_inventory.files
        if file.can_read
        and file.extension in READABLE_TABULAR_EXTENSIONS
        and file.role_hint == "train"
    ]
    if len(train_files) == 1:
        return train_files[0].path
    return None


def _find_base_id_column(
    inferred_schema: InferredSchema,
    *,
    base_table: str,
    base_columns: list[str],
    table_schemas: dict[str, TableSchema],
) -> str | None:
    if inferred_schema.primary_id_column in base_columns:
        return inferred_schema.primary_id_column

    base_schema = table_schemas.get(base_table)
    if base_schema is not None:
        for key in base_schema.candidate_join_keys:
            if key in base_columns and _is_id_like(key):
                return key

    global_join_keys = inferred_schema.global_roles.get("candidate_join_keys", [])
    for key in global_join_keys:
        if key in base_columns and _is_id_like(str(key)):
            return str(key)

    for column in base_columns:
        if _is_primary_id_like(column):
            return column
    for column in base_columns:
        if _is_id_like(column):
            return column
    return None


def _is_secondary_train_test_table(dataset_file: Any, *, base_table: str) -> bool:
    if dataset_file.path == base_table:
        return False
    if not dataset_file.can_read or dataset_file.extension not in READABLE_TABULAR_EXTENSIONS:
        return False
    if dataset_file.role_hint not in {"train", "test"}:
        return False
    return dataset_file.table_hint != "base"


def _schema_columns(
    reader: DatasetReader,
    table_path: str,
    table_schemas: dict[str, TableSchema],
    warnings: list[str],
) -> tuple[list[str], dict[str, str]]:
    table_schema = table_schemas.get(table_path)
    if table_schema is not None and table_schema.columns:
        return (
            [str(column["name"]) for column in table_schema.columns],
            {str(column["name"]): str(column.get("dtype", "")) for column in table_schema.columns},
        )

    try:
        columns = reader.read_schema(table_path)
    except ReaderError as exc:
        warnings.append(f"{table_path}: {exc}")
        return [], {}
    return (
        [str(column["name"]) for column in columns],
        {str(column["name"]): str(column.get("dtype", "")) for column in columns},
    )


def _candidate_join_keys(
    inferred_schema: InferredSchema,
    *,
    base_columns: list[str],
    base_id_column: str | None,
    table_columns: list[str],
    table_schema: TableSchema | None,
) -> list[str]:
    base_by_norm = {_normalize(column): column for column in base_columns}
    table_by_norm = {_normalize(column): column for column in table_columns}
    shared_norms = set(base_by_norm) & set(table_by_norm)
    if not shared_norms:
        return []

    hinted_names = set()
    for value in inferred_schema.global_roles.get("candidate_join_keys", []):
        hinted_names.add(_normalize(str(value)))
    if table_schema is not None:
        hinted_names.update(_normalize(key) for key in table_schema.candidate_join_keys)
    if base_id_column is not None:
        hinted_names.add(_normalize(base_id_column))

    candidates: list[str] = []
    if base_id_column is not None and _normalize(base_id_column) in shared_norms:
        candidates.append(table_by_norm[_normalize(base_id_column)])

    for norm_name in sorted(shared_norms):
        if norm_name in hinted_names or _is_id_like(norm_name):
            candidates.append(table_by_norm[norm_name])

    return _unique(candidates)


def _relationship_stats(
    reader: DatasetReader,
    *,
    base_table: str,
    table_path: str,
    join_key: str,
    role: str,
    table_hint: str,
    candidate_join_keys: list[str],
    candidate_group_keys: list[str],
    candidate_date_cutoff_columns: list[str],
    max_check_rows: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    base_row_count = _safe_count_rows(reader, base_table, warnings)
    right_row_count = _safe_count_rows(reader, table_path, warnings)
    sampled = (
        (base_row_count is not None and base_row_count > max_check_rows)
        or (right_row_count is not None and right_row_count > max_check_rows)
    )
    if sampled:
        warnings.append(
            f"Relationship checks are based on the first {max_check_rows} rows per table."
        )

    try:
        base_frame = reader.read_columns(base_table, [join_key], n_rows=max_check_rows)
        right_frame = reader.read_columns(table_path, [join_key], n_rows=max_check_rows)
    except ReaderError as exc:
        return _unknown_relationship(
            table_path=table_path,
            role=role,
            table_hint=table_hint,
            candidate_join_keys=candidate_join_keys,
            selected_join_key=join_key,
            candidate_group_keys=candidate_group_keys,
            candidate_date_cutoff_columns=candidate_date_cutoff_columns,
            warning=f"Could not read join key for relationship check: {exc}",
            sampled=sampled,
            sample_rows=max_check_rows if sampled else None,
        )

    base_values = _non_missing_values(base_frame[join_key])
    right_values = _non_missing_values(right_frame[join_key])
    if not base_values or not right_values:
        return _unknown_relationship(
            table_path=table_path,
            role=role,
            table_hint=table_hint,
            candidate_join_keys=candidate_join_keys,
            selected_join_key=join_key,
            candidate_group_keys=candidate_group_keys,
            candidate_date_cutoff_columns=candidate_date_cutoff_columns,
            warning="Join key has no non-missing values in one of the checked tables.",
            sampled=sampled,
            sample_rows=max_check_rows if sampled else None,
        )

    left_counts = Counter(base_values)
    right_counts = Counter(right_values)
    left_keys = set(left_counts)
    right_keys = set(right_counts)
    matching_left_keys = left_keys & right_keys
    matched_right_rows = sum(count for key, count in right_counts.items() if key in left_keys)

    coverage_left_to_right = _ratio(len(matching_left_keys), len(left_keys))
    orphan_rate_right = _ratio(len(right_values) - matched_right_rows, len(right_values))
    avg_rows_per_left = _ratio(matched_right_rows, len(left_keys))
    max_rows_per_left = max((right_counts.get(key, 0) for key in left_keys), default=0)

    relationship_type = _relationship_type(
        max_left=max(left_counts.values(), default=0),
        max_right=max(right_counts.values(), default=0),
    )
    risk = _row_multiplication_risk(
        relationship_type=relationship_type,
        avg_rows_per_left=avg_rows_per_left,
        max_rows_per_left=max_rows_per_left,
    )
    requires_aggregation = relationship_type in {"one_to_many", "many_to_many"}
    if requires_aggregation:
        warnings.append(
            "Direct join can multiply base rows; aggregate secondary rows before joining."
        )

    return {
        "table": table_path,
        "role": role,
        "table_hint": table_hint,
        "candidate_join_keys": candidate_join_keys,
        "selected_join_key": join_key,
        "relationship_type": relationship_type,
        "coverage_left_to_right": coverage_left_to_right,
        "orphan_rate_right": orphan_rate_right,
        "avg_rows_per_left": avg_rows_per_left,
        "max_rows_per_left": max_rows_per_left,
        "row_multiplication_risk": risk,
        "requires_aggregation": requires_aggregation,
        "candidate_group_keys": candidate_group_keys,
        "candidate_date_cutoff_columns": candidate_date_cutoff_columns,
        "confidence": _confidence(coverage_left_to_right, orphan_rate_right, sampled),
        "sampled": sampled,
        "sample_rows": max_check_rows if sampled else None,
        "warnings": warnings,
    }


def _unknown_relationship(
    *,
    table_path: str,
    role: str,
    table_hint: str,
    candidate_join_keys: list[str],
    selected_join_key: str | None,
    candidate_group_keys: list[str],
    candidate_date_cutoff_columns: list[str],
    warning: str,
    sampled: bool,
    sample_rows: int | None,
) -> dict[str, Any]:
    return {
        "table": table_path,
        "role": role,
        "table_hint": table_hint,
        "candidate_join_keys": candidate_join_keys,
        "selected_join_key": selected_join_key,
        "relationship_type": "unknown",
        "coverage_left_to_right": None,
        "orphan_rate_right": None,
        "avg_rows_per_left": None,
        "max_rows_per_left": None,
        "row_multiplication_risk": "unknown",
        "requires_aggregation": False,
        "candidate_group_keys": candidate_group_keys,
        "candidate_date_cutoff_columns": candidate_date_cutoff_columns,
        "confidence": "low",
        "sampled": sampled,
        "sample_rows": sample_rows,
        "warnings": [warning],
    }


def _safe_count_rows(
    reader: DatasetReader,
    table_path: str,
    warnings: list[str],
) -> int | None:
    try:
        return reader.count_rows(table_path)
    except ReaderError as exc:
        warnings.append(f"{table_path}: could not count rows: {exc}")
        return None


def _relationship_type(*, max_left: int, max_right: int) -> str:
    if max_left <= 0 or max_right <= 0:
        return "unknown"
    if max_left <= 1 and max_right <= 1:
        return "one_to_one"
    if max_left <= 1 and max_right > 1:
        return "one_to_many"
    if max_left > 1 and max_right <= 1:
        return "many_to_one"
    return "many_to_many"


def _row_multiplication_risk(
    *,
    relationship_type: str,
    avg_rows_per_left: float,
    max_rows_per_left: int,
) -> str:
    if relationship_type == "unknown":
        return "unknown"
    if relationship_type in {"one_to_one", "many_to_one"}:
        return "low"
    if max_rows_per_left >= 5 or avg_rows_per_left >= 2.0:
        return "high"
    return "medium"


def _confidence(coverage_left_to_right: float, orphan_rate_right: float, sampled: bool) -> str:
    if coverage_left_to_right >= 0.8 and orphan_rate_right <= 0.2 and not sampled:
        return "high"
    if coverage_left_to_right > 0.0:
        return "medium"
    return "low"


def _candidate_group_keys(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if _is_id_like(column) and any(token in _normalize(column) for token in GROUP_TOKENS)
    ]


def _candidate_date_columns(
    columns: list[str],
    *,
    column_dtypes: dict[str, str],
    table_schema: TableSchema | None,
) -> list[str]:
    candidates: list[str] = []
    if table_schema is not None:
        candidates.extend(table_schema.candidate_date_columns)
        candidates.extend(table_schema.candidate_time_columns)

    for column in columns:
        dtype = column_dtypes.get(column, "").lower()
        normalized = _normalize(column)
        if "date" in dtype or "datetime" in dtype:
            candidates.append(column)
        elif any(token in normalized for token in DATE_TOKENS):
            candidates.append(column)
    return _unique(candidates)


def _non_missing_values(series: pl.Series) -> list[Any]:
    values: list[Any] = []
    for value in series.to_list():
        if value is None:
            continue
        if isinstance(value, float) and value != value:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        values.append(value)
    return values


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _is_primary_id_like(column_name: str) -> bool:
    normalized = _normalize(column_name)
    return normalized in {"id", "row_id", "record_id", "record_key"} or normalized.endswith("_id")


def _is_id_like(column_name: str) -> bool:
    normalized = _normalize(column_name)
    name_parts = set(_name_parts(normalized))
    return (
        normalized in {"id", "key", "row_id", "record_id", "record_key"}
        or normalized.endswith("_id")
        or normalized.endswith("_key")
        or (
            bool(name_parts & {"id", "key"})
            and bool(name_parts & set(ID_ENTITY_TOKENS))
        )
    )


def _normalize(value: str) -> str:
    return value.strip().lower()


def _name_parts(value: str) -> list[str]:
    return [part for part in value.replace("-", "_").split("_") if part]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


__all__ = ["infer_relationships"]
