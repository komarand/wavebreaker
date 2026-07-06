from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.presets import CompetitionPreset
from kaggle_researcher.eda.schemas import (
    ColumnRole,
    FileInventoryResult,
    InferredSchema,
    TableSchema,
)


READABLE_TABULAR_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}
ID_NAMES = {"id"}
TARGET_NAMES = {"target"}
PREDICTION_NAMES = {"prediction", "pred", "probability"}
GROUP_TOKENS = ("group", "fold", "customer", "client", "user")
TIME_TOKENS = ("period", "month", "year", "quarter")


def infer_schema(
    file_inventory: FileInventoryResult,
    reader: DatasetReader,
    preset: CompetitionPreset | None = None,
) -> InferredSchema:
    tables: list[TableSchema] = []
    warnings: list[str] = []

    for dataset_file in file_inventory.files:
        if not dataset_file.can_read or dataset_file.extension not in READABLE_TABULAR_EXTENSIONS:
            continue
        try:
            columns = reader.read_schema(dataset_file.path)
        except ReaderError as exc:
            warnings.append(f"{dataset_file.path}: {exc}")
            continue
        table_schema = _build_table_schema(dataset_file, columns, preset=preset)
        tables.append(table_schema)

    train_base_table = _find_table_path(tables, role="train", table_type="base")
    test_base_table = _find_table_path(tables, role="test", table_type="base")
    sample_submission_table = _find_table_path(tables, role="submission", table_type=None)

    target_column = _find_target_column(tables)
    primary_id_column = _find_primary_id_column(tables)
    prediction_column = _find_prediction_column(tables)

    candidate_time_columns = _unique_column_names(tables, role="time")
    candidate_date_columns = _unique_column_names(tables, role="date")
    candidate_group_columns = _unique_column_names(tables, role="group")
    candidate_join_keys = _unique_join_keys(tables)

    if train_base_table is None:
        warnings.append("Train base table could not be inferred.")
    if test_base_table is None:
        warnings.append("Test base table could not be inferred.")
    if target_column is None:
        warnings.append("Target column could not be inferred from train tables.")
    if primary_id_column is None:
        warnings.append("Primary id column could not be inferred.")

    confidence = _schema_confidence(
        train_base_table=train_base_table,
        test_base_table=test_base_table,
        target_column=target_column,
        primary_id_column=primary_id_column,
        warnings=warnings,
    )

    global_roles: dict[str, Any] = {
        "target_column": target_column,
        "primary_id_column": primary_id_column,
        "prediction_column": prediction_column,
        "train_base_table": train_base_table,
        "test_base_table": test_base_table,
        "sample_submission_table": sample_submission_table,
        "candidate_time_columns": candidate_time_columns,
        "candidate_date_columns": candidate_date_columns,
        "candidate_group_columns": candidate_group_columns,
        "candidate_join_keys": candidate_join_keys,
    }

    return InferredSchema(
        global_roles=global_roles,
        tables=tables,
        target_column=target_column,
        primary_id_column=primary_id_column,
        prediction_column=prediction_column,
        train_base_table=train_base_table,
        test_base_table=test_base_table,
        sample_submission_table=sample_submission_table,
        candidate_time_columns=candidate_time_columns,
        candidate_group_columns=candidate_group_columns,
        candidate_date_columns=candidate_date_columns,
        confidence=confidence,
        warnings=warnings,
    )


def _build_table_schema(
    dataset_file: Any,
    columns: list[dict[str, str]],
    *,
    preset: CompetitionPreset | None,
) -> TableSchema:
    table_role = _table_role(dataset_file.role_hint)
    table_type = _table_type(dataset_file.table_hint)
    column_roles = [
        _infer_column_role(column["name"], table_role=table_role, preset=preset)
        for column in columns
    ]
    candidate_join_keys = [
        role.name
        for role in column_roles
        if role.role in {"primary_id", "group"} or role.name.lower().endswith("_id")
    ]
    candidate_time_columns = [role.name for role in column_roles if role.role == "time"]
    candidate_date_columns = [role.name for role in column_roles if role.role == "date"]
    warnings: list[str] = []
    confidence = "high" if column_roles else "low"
    if not column_roles:
        warnings.append("No columns were available for schema inference.")

    return TableSchema(
        table_name=Path(dataset_file.name).stem,
        path=dataset_file.path,
        role=table_role,
        table_type=table_type,
        n_columns=len(columns),
        columns=columns,
        column_roles=column_roles,
        candidate_join_keys=candidate_join_keys,
        candidate_time_columns=candidate_time_columns,
        candidate_date_columns=candidate_date_columns,
        confidence=confidence,
        warnings=warnings,
    )


def _infer_column_role(
    column_name: str,
    table_role: str,
    preset: CompetitionPreset | None,
) -> ColumnRole:
    generic_role = _infer_generic_column_role(column_name, table_role=table_role)
    return _apply_preset_column_hint(
        generic_role,
        column_name=column_name,
        table_role=table_role,
        preset=preset,
    )


def _infer_generic_column_role(column_name: str, *, table_role: str) -> ColumnRole:
    normalized = column_name.lower()

    if normalized in TARGET_NAMES and table_role == "train":
        return ColumnRole(
            name=column_name,
            role="target",
            confidence="high",
            reason="Column matches generic target names and appears in a train table.",
        )
    if normalized in ID_NAMES:
        return ColumnRole(
            name=column_name,
            role="primary_id",
            confidence="high",
            reason="Column matches generic primary id names.",
        )
    if any(token in normalized for token in TIME_TOKENS):
        return ColumnRole(
            name=column_name,
            role="time",
            confidence="medium",
            reason="Column name contains a generic time-period signal.",
        )
    if "date" in normalized or "timestamp" in normalized or normalized.endswith("_dt"):
        return ColumnRole(
            name=column_name,
            role="date",
            confidence="high",
            reason="Column name contains date/timestamp signal.",
        )
    if table_role == "submission" and normalized in PREDICTION_NAMES:
        return ColumnRole(
            name=column_name,
            role="prediction",
            confidence="high",
            reason="Column matches generic prediction names in sample submission.",
        )
    if any(token in normalized for token in GROUP_TOKENS):
        return ColumnRole(
            name=column_name,
            role="group",
            confidence="medium",
            reason="Column name contains a group-like token.",
        )
    return ColumnRole(
        name=column_name,
        role="unknown",
        confidence="low",
        reason="No semantic role heuristic matched.",
    )


def _apply_preset_column_hint(
    generic_role: ColumnRole,
    *,
    column_name: str,
    table_role: str,
    preset: CompetitionPreset | None,
) -> ColumnRole:
    if preset is None or generic_role.role != "unknown":
        return generic_role

    normalized = column_name.lower()
    target_names = _preset_names(preset, "preferred_target_columns")
    id_names = _preset_names(preset, "preferred_id_columns")
    time_names = _preset_names(preset, "preferred_time_columns")
    prediction_names = _preset_names(preset, "preferred_prediction_columns")

    if normalized in target_names and table_role == "train":
        return ColumnRole(
            name=column_name,
            role="target",
            confidence="high",
            reason="Column matches preset target-column hints and appears in a train table.",
        )
    if normalized in id_names:
        return ColumnRole(
            name=column_name,
            role="primary_id",
            confidence="high",
            reason="Column matches preset primary id hints.",
        )
    if normalized in time_names:
        return ColumnRole(
            name=column_name,
            role="time",
            confidence="high",
            reason="Column matches preset time-column hints.",
        )
    if table_role == "submission" and normalized in prediction_names:
        return ColumnRole(
            name=column_name,
            role="prediction",
            confidence="high",
            reason="Column matches preset prediction-column hints in sample submission.",
        )
    return generic_role


def _table_role(role_hint: str) -> str:
    if role_hint == "train":
        return "train"
    if role_hint == "test":
        return "test"
    if role_hint == "sample_submission":
        return "submission"
    if role_hint == "metadata":
        return "metadata"
    return "unknown"


def _table_type(table_hint: str) -> str:
    if table_hint == "base":
        return "base"
    if table_hint in {"secondary", "depth_0", "depth_1", "depth_2"}:
        return table_hint
    return "unknown"


def _find_table_path(
    tables: list[TableSchema],
    role: str,
    table_type: str | None,
) -> str | None:
    for table in tables:
        if table.role != role:
            continue
        if table_type is None or table.table_type == table_type:
            return table.path
    return None


def _find_target_column(tables: list[TableSchema]) -> str | None:
    for table in tables:
        if table.role != "train":
            continue
        for column_role in table.column_roles:
            if column_role.role == "target":
                return column_role.name
    return None


def _find_prediction_column(tables: list[TableSchema]) -> str | None:
    for table in tables:
        if table.role != "submission":
            continue
        for column_role in table.column_roles:
            if column_role.role == "prediction":
                return column_role.name
    return None


def _find_primary_id_column(tables: list[TableSchema]) -> str | None:
    for table in tables:
        for column_role in table.column_roles:
            if column_role.role == "primary_id":
                return column_role.name
    return None


def _unique_column_names(
    tables: list[TableSchema],
    role: str,
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for table in tables:
        for column_role in table.column_roles:
            if column_role.role != role:
                continue
            if column_role.name not in seen:
                seen.add(column_role.name)
                result.append(column_role.name)
    return result


def _unique_join_keys(tables: list[TableSchema]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for table in tables:
        for key in table.candidate_join_keys:
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def _schema_confidence(
    *,
    train_base_table: str | None,
    test_base_table: str | None,
    target_column: str | None,
    primary_id_column: str | None,
    warnings: list[str],
) -> str:
    if all([train_base_table, test_base_table, target_column, primary_id_column]) and not warnings:
        return "high"
    if train_base_table and test_base_table and primary_id_column:
        return "medium"
    return "low"


def _preset_names(
    preset: CompetitionPreset,
    field_name: str,
) -> set[str]:
    return {str(name).lower() for name in getattr(preset, field_name)}


__all__ = ["infer_schema"]
