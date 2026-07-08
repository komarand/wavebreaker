from __future__ import annotations

from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.schemas import (
    InferredSchema,
    LeakageCheckResult,
    ValidationEvidence,
)


MAX_CHECK_ROWS = 200_000
TARGET_LIKE_TOKENS = ("target", "label", "outcome", "post_target", "future_target")


def check_leakage(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
) -> list[LeakageCheckResult]:
    train_table = inferred_schema.train_base_table
    test_table = inferred_schema.test_base_table
    sample_submission_table = inferred_schema.sample_submission_table
    train_schema = _safe_schema(reader, train_table)
    test_schema = _safe_schema(reader, test_table)
    sample_schema = _safe_schema(reader, sample_submission_table)

    results = [
        _check_id_overlap(inferred_schema, reader, train_schema, test_schema),
        _check_target_absent_from_test(inferred_schema, test_schema),
        _check_target_like_columns(inferred_schema, train_schema, test_schema),
        _check_sample_submission(inferred_schema, sample_schema),
        _check_duplicate_base_rows(inferred_schema, reader, train_schema, test_schema),
        _check_numeric_target_association(inferred_schema, reader, train_schema),
        _check_future_time_risk(validation_evidence),
        _check_group_overlap(
            inferred_schema,
            validation_evidence,
            reader,
            train_schema,
            test_schema,
        ),
        _check_ranking_query_overlap(validation_evidence),
    ]
    return results


def _check_id_overlap(
    schema: InferredSchema,
    reader: DatasetReader,
    train_schema: list[dict[str, str]],
    test_schema: list[dict[str, str]],
) -> LeakageCheckResult:
    id_col = schema.primary_id_column
    if (
        schema.train_base_table is None
        or schema.test_base_table is None
        or id_col is None
    ):
        return _result(
            "id_overlap",
            "not_testable",
            "low",
            "Train/test ID overlap could not be tested because required tables "
            "or ID column are missing.",
        )
    if id_col not in _schema_names(train_schema) or id_col not in _schema_names(test_schema):
        return _result(
            "id_overlap",
            "not_testable",
            "low",
            f"ID column '{id_col}' is not present in both train and test base tables.",
        )

    train_ids = _read_value_set(reader, schema.train_base_table, id_col)
    test_ids = _read_value_set(reader, schema.test_base_table, id_col)
    overlap = sorted(train_ids & test_ids, key=lambda value: str(value))
    if overlap:
        return _result(
            "id_overlap",
            "failed",
            "high",
            "Train and test base tables share ID values.",
            {"overlap_count": len(overlap), "overlap_examples": overlap[:10], "id_column": id_col},
        )
    return _result(
        "id_overlap",
        "passed",
        "low",
        "No train/test ID overlap was found within the checked rows.",
        {"id_column": id_col, "checked_rows_cap": MAX_CHECK_ROWS},
    )


def _check_target_absent_from_test(
    schema: InferredSchema,
    test_schema: list[dict[str, str]],
) -> LeakageCheckResult:
    target_col = schema.target_column
    if schema.test_base_table is None or target_col is None:
        return _result(
            "target_in_test",
            "not_testable",
            "low",
            "Target-in-test check requires a test base table and target column.",
        )
    if target_col in _schema_names(test_schema):
        return _result(
            "target_in_test",
            "failed",
            "critical",
            "Target column is present in the test base table.",
            {"target_column": target_col, "test_table": schema.test_base_table},
        )
    return _result(
        "target_in_test",
        "passed",
        "low",
        "Target column is absent from the test base table.",
        {"target_column": target_col, "test_table": schema.test_base_table},
    )


def _check_target_like_columns(
    schema: InferredSchema,
    train_schema: list[dict[str, str]],
    test_schema: list[dict[str, str]],
) -> LeakageCheckResult:
    target_col = schema.target_column
    flagged: list[dict[str, str]] = []
    for table_name, table_schema in (("train", train_schema), ("test", test_schema)):
        for column in table_schema:
            name = column["name"]
            if name == target_col:
                continue
            normalized = name.lower()
            if any(token in normalized for token in TARGET_LIKE_TOKENS):
                flagged.append({"table": table_name, "column": name})
    if flagged:
        return _result(
            "target_like_columns",
            "warning",
            "medium",
            "Target-like column names were found outside the inferred target column.",
            {"columns": flagged},
        )
    return _result(
        "target_like_columns",
        "passed",
        "low",
        "No target-like column names were found outside the inferred target column.",
    )


def _check_sample_submission(
    schema: InferredSchema,
    sample_schema: list[dict[str, str]],
) -> LeakageCheckResult:
    if schema.sample_submission_table is None:
        return _result(
            "sample_submission_structure",
            "not_testable",
            "low",
            "No sample submission table was inferred.",
        )
    column_names = _schema_names(sample_schema)
    id_ok = schema.primary_id_column is None or schema.primary_id_column in column_names
    prediction_ok = schema.prediction_column is None or schema.prediction_column in column_names
    if id_ok and prediction_ok and len(column_names) >= 2:
        return _result(
            "sample_submission_structure",
            "passed",
            "low",
            "Sample submission contains the expected ID/prediction structure.",
            {"columns": column_names},
        )
    return _result(
        "sample_submission_structure",
        "warning",
        "medium",
        "Sample submission structure does not match inferred ID/prediction columns.",
        {"columns": column_names},
    )


def _check_duplicate_base_rows(
    schema: InferredSchema,
    reader: DatasetReader,
    train_schema: list[dict[str, str]],
    test_schema: list[dict[str, str]],
) -> LeakageCheckResult:
    if schema.train_base_table is None or schema.test_base_table is None:
        return _result(
            "duplicate_base_rows",
            "not_testable",
            "low",
            "Duplicate row check requires train and test base tables.",
        )
    common_columns = sorted(_schema_names(train_schema) & _schema_names(test_schema))
    if not common_columns:
        return _result(
            "duplicate_base_rows",
            "not_testable",
            "low",
            "Train and test base tables have no common columns for duplicate checks.",
        )
    train_frame = _safe_read_columns(reader, schema.train_base_table, common_columns)
    test_frame = _safe_read_columns(reader, schema.test_base_table, common_columns)
    if train_frame is None or test_frame is None:
        return _result(
            "duplicate_base_rows",
            "not_testable",
            "low",
            "Could not read common train/test columns for duplicate row check.",
        )
    train_rows = {_row_tuple(row) for row in train_frame.iter_rows(named=True)}
    test_rows = {_row_tuple(row) for row in test_frame.iter_rows(named=True)}
    overlap = list(train_rows & test_rows)
    if overlap:
        return _result(
            "duplicate_base_rows",
            "failed",
            "high",
            "Duplicate rows were found across train and test base tables.",
            {"overlap_count": len(overlap), "checked_columns": common_columns},
        )
    return _result(
        "duplicate_base_rows",
        "passed",
        "low",
        "No duplicate rows were found across train and test base tables within cap.",
        {"checked_columns": common_columns, "checked_rows_cap": MAX_CHECK_ROWS},
    )


def _check_numeric_target_association(
    schema: InferredSchema,
    reader: DatasetReader,
    train_schema: list[dict[str, str]],
) -> LeakageCheckResult:
    target_col = schema.target_column
    excluded_columns = _excluded_feature_columns(schema, train_schema)
    if (
        schema.train_base_table is None
        or target_col is None
        or target_col not in _schema_names(train_schema)
    ):
        return _result(
            "numeric_target_association",
            "not_testable",
            "low",
            "Numeric target-association check requires a train target column.",
        )
    numeric_columns = [
        column["name"]
        for column in train_schema
        if column["name"] not in excluded_columns and _is_numeric_dtype(column["dtype"])
    ]
    if not numeric_columns:
        return _result(
            "numeric_target_association",
            "not_testable",
            "low",
            "No numeric feature columns were available for association checks.",
        )
    frame = _safe_read_columns(reader, schema.train_base_table, [target_col, *numeric_columns])
    if frame is None:
        return _result(
            "numeric_target_association",
            "not_testable",
            "low",
            "Could not read numeric columns for target-association check.",
        )
    flagged: list[dict[str, Any]] = []
    target_values = _float_values(frame[target_col].to_list())
    for column_name in numeric_columns:
        values = _float_values(frame[column_name].to_list())
        correlation = _abs_correlation(target_values, values)
        if correlation is not None and correlation >= 0.999:
            flagged.append({"column": column_name, "abs_correlation": correlation})
    if flagged:
        return _result(
            "numeric_target_association",
            "warning",
            "high",
            "Numeric columns have extremely high association with the target.",
            {"columns": flagged},
        )
    return _result(
        "numeric_target_association",
        "passed",
        "low",
        "No numeric columns showed extremely high target association within cap.",
        {"checked_columns": numeric_columns, "checked_rows_cap": MAX_CHECK_ROWS},
    )


def _check_future_time_risk(validation_evidence: ValidationEvidence) -> LeakageCheckResult:
    relation = validation_evidence.test_time_relation
    if not validation_evidence.time_columns and not relation:
        return _result(
            "future_time_risk",
            "not_testable",
            "low",
            "No time/date columns were available for future-information risk checks.",
        )
    if relation.get("future_test") is True:
        return _result(
            "future_time_risk",
            "warning",
            "medium",
            "Test periods appear to be after train periods; guard against "
            "future-information features.",
            relation,
        )
    return _result(
        "future_time_risk",
        "passed",
        "low",
        "Time/date presence alone is not leakage; no future-test relation was flagged.",
        relation,
    )


def _check_group_overlap(
    schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
    train_schema: list[dict[str, str]],
    test_schema: list[dict[str, str]],
) -> LeakageCheckResult:
    group_col = _first_group_column(validation_evidence)
    if group_col is None:
        return _result(
            "group_overlap",
            "not_testable",
            "low",
            "No group/entity column was inferred.",
        )
    if schema.train_base_table is None or schema.test_base_table is None:
        return _result(
            "group_overlap",
            "not_testable",
            "low",
            "Group overlap check requires train and test base tables.",
        )
    if group_col not in _schema_names(train_schema) or group_col not in _schema_names(test_schema):
        return _result(
            "group_overlap",
            "not_testable",
            "low",
            f"Group column '{group_col}' is not present in both train and test base tables.",
        )
    train_frame = _safe_read_columns(reader, schema.train_base_table, [group_col])
    test_frame = _safe_read_columns(reader, schema.test_base_table, [group_col])
    if train_frame is None or test_frame is None:
        return _result("group_overlap", "not_testable", "low", "Could not read group columns.")
    relation = _group_relation(train_frame, test_frame, group_col)
    if relation["n_overlap_groups"] == 0:
        return _result(
            "group_overlap",
            "passed",
            "low",
            "No train/test group overlap was found within cap.",
            relation,
        )
    severity = _group_overlap_severity(validation_evidence)
    return _result(
        "group_overlap",
        "warning",
        severity,
        "Train/test group overlap was found; severity depends on selected validation policy.",
        relation,
    )


def _check_ranking_query_overlap(validation_evidence: ValidationEvidence) -> LeakageCheckResult:
    if not validation_evidence.query_columns:
        return _result(
            "ranking_query_overlap",
            "not_testable",
            "low",
            "No query/ranking group column was inferred.",
        )
    relation = validation_evidence.test_group_relation
    if relation.get("n_overlap_groups", 0) > 0:
        return _result(
            "ranking_query_overlap",
            "warning",
            "high",
            "Query/group identifiers overlap between train and test in a ranking-like setup.",
            relation,
        )
    return _result(
        "ranking_query_overlap",
        "passed",
        "low",
        "No query/group overlap was reported for ranking-like validation evidence.",
        relation,
    )


def _safe_schema(reader: DatasetReader, table: str | None) -> list[dict[str, str]]:
    if table is None:
        return []
    try:
        return reader.read_schema(table)
    except ReaderError:
        return []


def _safe_read_columns(
    reader: DatasetReader,
    table: str,
    columns: list[str],
) -> pl.DataFrame | None:
    try:
        return reader.read_columns(table, columns=columns, n_rows=MAX_CHECK_ROWS)
    except ReaderError:
        return None


def _read_value_set(reader: DatasetReader, table: str, column: str) -> set[Any]:
    frame = _safe_read_columns(reader, table, [column])
    if frame is None:
        return set()
    return {value for value in frame[column].to_list() if value is not None and value != ""}


def _schema_names(schema: list[dict[str, str]]) -> set[str]:
    return {column["name"] for column in schema}


def _excluded_feature_columns(
    schema: InferredSchema,
    train_schema: list[dict[str, str]],
) -> set[str]:
    excluded = {schema.target_column, schema.primary_id_column}
    for column in train_schema:
        normalized = column["name"].lower()
        if any(
            token in normalized
            for token in ("id", "group", "session", "query", "customer", "client", "user")
        ):
            excluded.add(column["name"])
    return {column for column in excluded if column is not None}


def _row_tuple(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(row.items()))


def _is_numeric_dtype(dtype: str) -> bool:
    lowered = dtype.lower()
    return any(token in lowered for token in ("int", "float", "decimal"))


def _float_values(values: list[Any]) -> list[float | None]:
    result: list[float | None] = []
    for value in values:
        if value is None or value == "":
            result.append(None)
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(None)
    return result


def _abs_correlation(
    left_values: list[float | None],
    right_values: list[float | None],
) -> float | None:
    pairs = [
        (left, right)
        for left, right in zip(left_values, right_values, strict=True)
        if left is not None and right is not None
    ]
    if len(pairs) < 3:
        return None
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var == 0 or right_var == 0:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return abs(covariance / ((left_var * right_var) ** 0.5))


def _first_group_column(validation_evidence: ValidationEvidence) -> str | None:
    for candidate in validation_evidence.group_columns:
        return str(candidate["name"])
    return None


def _group_relation(
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    group_col: str,
) -> dict[str, Any]:
    train_groups = {
        value for value in train_frame[group_col].to_list() if value is not None and value != ""
    }
    test_groups = {
        value for value in test_frame[group_col].to_list() if value is not None and value != ""
    }
    overlap = sorted(train_groups & test_groups, key=lambda value: str(value))
    return {
        "group_col": group_col,
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "overlap_groups": overlap[:10],
        "n_overlap_groups": len(overlap),
        "overlap_pct_of_test": 0.0 if not test_groups else len(overlap) / len(test_groups),
    }


def _group_overlap_severity(validation_evidence: ValidationEvidence) -> str:
    method = validation_evidence.primary_validation.get("method")
    if method in {"group_kfold", "stratified_group_kfold", "ranking_group_cv"}:
        return "medium"
    return "high"


def _result(
    check_id: str,
    status: str,
    severity: str,
    finding: str,
    evidence: dict[str, Any] | None = None,
) -> LeakageCheckResult:
    return LeakageCheckResult(
        check_id=check_id,
        status=status,
        severity=severity,
        finding=finding,
        evidence=evidence or {},
    )


__all__ = ["check_leakage"]
