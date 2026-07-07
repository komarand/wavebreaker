from __future__ import annotations

from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.metrics import MetricFamily, TaskType, infer_metric_spec
from kaggle_researcher.eda.schemas import (
    InferredSchema,
    MetricEvidence,
    TableProfile,
    ValidationEvidence,
)
from kaggle_researcher.eda.validation import select_validation_policy
from kaggle_researcher.eda.validation.group_split import detect_group_leakage_risk
from kaggle_researcher.eda.validation.split_helpers import (
    infer_candidate_group_columns,
    infer_candidate_time_columns,
    infer_class_balance,
    infer_regression_target_stats,
    summarize_column_distribution,
)
from kaggle_researcher.eda.validation.temporal_split import (
    build_expanding_window_folds,
    build_latest_period_holdout,
    infer_periods,
    summarize_period_counts,
)


MAX_VALIDATION_ROWS = 200_000


def analyze_validation(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    metric_evidence: MetricEvidence,
    reader: DatasetReader,
) -> ValidationEvidence:
    warnings: list[str] = []
    limitations: list[str] = []

    train_table = inferred_schema.train_base_table
    test_table = inferred_schema.test_base_table
    target_col = inferred_schema.target_column
    id_col = inferred_schema.primary_id_column

    time_columns = infer_candidate_time_columns(inferred_schema, table_profiles)
    group_columns = infer_candidate_group_columns(inferred_schema, table_profiles)
    query_columns = [
        column for column in group_columns if "query" in str(column["name"]).lower()
    ]
    train_schema = _safe_schema(reader, train_table, warnings) if train_table else []
    test_schema = _safe_schema(reader, test_table, warnings) if test_table else []
    train_column_names = {column["name"] for column in train_schema}
    test_column_names = {column["name"] for column in test_schema}

    target_available = target_col is not None and target_col in train_column_names
    id_column_available = id_col is not None and id_col in train_column_names
    if train_table is None:
        warnings.append("Train base table is unavailable; validation evidence is limited.")
    if target_col is not None and not target_available:
        warnings.append(f"Target column '{target_col}' was not found in train base table.")
    if id_col is not None and not id_column_available:
        warnings.append(f"ID column '{id_col}' was not found in train base table.")

    selected_time_column = _first_available(time_columns, train_column_names)
    selected_group_column = _first_available(group_columns, train_column_names)

    train_frame = _read_train_frame(
        reader=reader,
        table=train_table,
        columns=_needed_train_columns(
            target_col=target_col if target_available else None,
            id_col=id_col if id_column_available else None,
            time_col=selected_time_column,
            group_col=selected_group_column,
        ),
        warnings=warnings,
        limitations=limitations,
    )

    class_balance: dict[str, Any] = {}
    target_summary: dict[str, Any] = {}
    target_by_period: list[dict[str, Any]] = []
    target_by_group: list[dict[str, Any]] = []
    oot_holdout: dict[str, Any] = {}
    temporal_folds: dict[str, Any] = {}
    test_time_relation: dict[str, Any] = {}
    test_group_relation: dict[str, Any] = {}
    validation_signals: dict[str, Any] = {}

    task_type = _task_type(metric_evidence)
    metric_family = _metric_family(metric_evidence)
    if train_frame is not None and target_available and target_col is not None:
        if _is_regression(task_type, metric_family):
            target_summary = infer_regression_target_stats(train_frame, target_col)
        else:
            class_balance = infer_class_balance(train_frame, target_col)

        if selected_time_column is not None:
            target_by_period = summarize_period_counts(
                train_frame,
                selected_time_column,
                target_col=target_col,
            )
        if selected_group_column is not None:
            target_by_group = summarize_column_distribution(
                train_frame,
                selected_group_column,
                target_col=target_col,
            )

    if train_frame is not None and selected_time_column is not None:
        periods = infer_periods(train_frame, selected_time_column)
        oot_holdout = build_latest_period_holdout(periods)
        temporal_folds = {
            "method": "expanding_window",
            "folds": build_expanding_window_folds(periods),
        }
        if test_table is not None and selected_time_column in test_column_names:
            test_frame = _safe_read_columns(
                reader,
                test_table,
                [selected_time_column],
                warnings,
                limitations,
            )
            if test_frame is not None:
                test_time_relation = _compare_time_ranges(
                    train_frame,
                    test_frame,
                    selected_time_column,
                )
                if test_time_relation.get("future_test") is True:
                    validation_signals["requires_temporal_validation"] = True

    if (
        test_table is not None
        and selected_group_column is not None
        and selected_group_column in test_column_names
        and train_frame is not None
    ):
        test_group_frame = _safe_read_columns(
            reader,
            test_table,
            [selected_group_column],
            warnings,
            limitations,
        )
        if test_group_frame is not None:
            test_group_relation = detect_group_leakage_risk(
                train_frame,
                test_group_frame,
                selected_group_column,
            )

    metric_spec = infer_metric_spec(metric_evidence.metric_name, task_type)
    policy = select_validation_policy(
        task_type=task_type,
        metric_spec=metric_spec,
        inferred_schema=inferred_schema,
        table_profiles=table_profiles,
        validation_signals=validation_signals,
    )

    warnings.extend(policy.get("warnings", []))
    limitations.extend(policy.get("limitations", []))

    return ValidationEvidence(
        target_available=target_available,
        id_column_available=id_column_available,
        target_column=target_col,
        id_column=id_col,
        time_columns=time_columns,
        group_columns=group_columns,
        query_columns=query_columns,
        class_balance=class_balance,
        target_summary=target_summary,
        target_by_period=target_by_period,
        target_by_group=target_by_group,
        test_time_relation=test_time_relation,
        test_group_relation=test_group_relation,
        oot_holdout=oot_holdout,
        temporal_folds=temporal_folds,
        primary_validation=policy["primary_validation"],
        diagnostic_validations=policy["diagnostic_validations"],
        rejected_validations=policy["rejected_validations"],
        recommended_validation=policy,
        confidence=policy["confidence"],
        evidence_refs=policy["evidence_refs"],
        reasoning_summary=policy["reasoning_summary"],
        warnings=_unique(warnings),
        limitations=_unique(limitations),
    )


def _safe_schema(
    reader: DatasetReader,
    table: str | None,
    warnings: list[str],
) -> list[dict[str, str]]:
    if table is None:
        return []
    try:
        return reader.read_schema(table)
    except ReaderError as exc:
        warnings.append(str(exc))
        return []


def _read_train_frame(
    *,
    reader: DatasetReader,
    table: str | None,
    columns: list[str],
    warnings: list[str],
    limitations: list[str],
) -> pl.DataFrame | None:
    if table is None or not columns:
        return None
    frame = _safe_read_columns(reader, table, columns, warnings, limitations)
    if frame is not None:
        limitations.append(
            f"Validation evidence uses at most {MAX_VALIDATION_ROWS} train rows."
        )
    return frame


def _safe_read_columns(
    reader: DatasetReader,
    table: str,
    columns: list[str],
    warnings: list[str],
    limitations: list[str],
) -> pl.DataFrame | None:
    if not columns:
        return None
    try:
        return reader.read_columns(table, columns=_unique(columns), n_rows=MAX_VALIDATION_ROWS)
    except ReaderError as exc:
        warnings.append(str(exc))
        limitations.append(f"Could not read validation columns from {table}.")
        return None


def _needed_train_columns(
    *,
    target_col: str | None,
    id_col: str | None,
    time_col: str | None,
    group_col: str | None,
) -> list[str]:
    return _unique([
        column
        for column in (target_col, id_col, time_col, group_col)
        if column is not None
    ])


def _first_available(candidates: list[dict[str, Any]], available_columns: set[str]) -> str | None:
    for candidate in candidates:
        name = str(candidate["name"])
        if name in available_columns:
            return name
    return None


def _task_type(metric_evidence: MetricEvidence) -> str:
    return metric_evidence.task_type or "unknown"


def _metric_family(metric_evidence: MetricEvidence) -> str:
    return metric_evidence.metric_family or "unknown"


def _is_regression(task_type: str, metric_family: str) -> bool:
    return (
        task_type == TaskType.REGRESSION.value
        or metric_family == MetricFamily.REGRESSION_ERROR.value
    )


def _compare_time_ranges(
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    time_col: str,
) -> dict[str, Any]:
    train_periods = infer_periods(train_frame, time_col)
    test_periods = infer_periods(test_frame, time_col)
    relation: dict[str, Any] = {
        "time_col": time_col,
        "train_min": train_periods[0] if train_periods else None,
        "train_max": train_periods[-1] if train_periods else None,
        "test_min": test_periods[0] if test_periods else None,
        "test_max": test_periods[-1] if test_periods else None,
        "future_test": False,
        "overlap_periods": [],
    }
    if not train_periods or not test_periods:
        return relation
    train_keys = {_sort_key(period) for period in train_periods}
    test_keys = {_sort_key(period) for period in test_periods}
    relation["overlap_periods"] = [
        period
        for period in test_periods
        if _sort_key(period) in train_keys
    ]
    relation["future_test"] = min(test_keys) > max(train_keys)
    return relation


def _sort_key(value: Any) -> tuple[str, Any]:
    if isinstance(value, (int, float)):
        return ("number", value)
    text = str(value)
    try:
        return ("number", float(text))
    except ValueError:
        return ("text", text)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["analyze_validation"]
