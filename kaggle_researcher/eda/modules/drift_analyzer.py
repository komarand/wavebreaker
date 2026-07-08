from __future__ import annotations

from collections import Counter
from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.schemas import InferredSchema, ValidationEvidence


NUMERIC_DTYPE_TOKENS = ("int", "float", "decimal")
CATEGORICAL_DTYPE_TOKENS = ("str", "utf8", "categorical", "bool")
GROUP_ID_TOKENS = ("id", "key", "group", "query", "customer", "client", "user", "session")
TOP_CATEGORICAL_VALUES = 20
PSI_EPSILON = 1e-6


def analyze_drift(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    reader: DatasetReader,
    max_rows: int = 500_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute generic diagnostic drift evidence for train/test base tables."""

    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")

    warnings: list[str] = []
    limitations: list[str] = []
    train_table = inferred_schema.train_base_table
    test_table = inferred_schema.test_base_table
    if train_table is None:
        return {
            "status": "not_testable",
            "severity": "unknown",
            "temporal_drift": {"status": "skipped"},
            "missingness_drift": {"status": "skipped", "columns": []},
            "numeric_psi": {"status": "skipped", "columns": []},
            "categorical_shift": {"status": "skipped", "columns": []},
            "adversarial_validation": {"status": "skipped"},
            "warnings": ["Train base table is unavailable; drift analysis cannot run."],
            "limitations": [],
        }

    train_schema = _safe_schema(reader, train_table, warnings)
    test_schema = _safe_schema(reader, test_table, warnings) if test_table else []
    train_names = _schema_names(train_schema)
    test_names = _schema_names(test_schema)
    shared_columns = sorted(train_names & test_names)

    time_column = _select_time_column(
        validation_evidence,
        inferred_schema,
        available_columns=train_names,
    )
    temporal_drift = _temporal_drift(
        reader=reader,
        train_table=train_table,
        train_schema=train_schema,
        time_column=time_column,
        target_column=inferred_schema.target_column,
        max_rows=max_rows,
        warnings=warnings,
        limitations=limitations,
    )

    if test_table is None:
        limitations.append("No test base table exists; train/test drift checks were skipped.")
        missingness_drift = {"status": "skipped", "columns": [], "severity": "unknown"}
        numeric_psi = {"status": "skipped", "columns": [], "severity": "unknown"}
        categorical_shift = {"status": "skipped", "columns": [], "severity": "unknown"}
        adversarial = {
            "status": "skipped",
            "reason": "No test base table exists.",
            "feature_columns": [],
            "excluded_columns": [],
        }
    elif not shared_columns:
        limitations.append("Train and test base tables have no shared columns.")
        missingness_drift = {"status": "skipped", "columns": [], "severity": "unknown"}
        numeric_psi = {"status": "skipped", "columns": [], "severity": "unknown"}
        categorical_shift = {"status": "skipped", "columns": [], "severity": "unknown"}
        adversarial = {
            "status": "skipped",
            "reason": "Train and test base tables have no shared columns.",
            "feature_columns": [],
            "excluded_columns": [],
        }
    else:
        train_frame, test_frame = _read_shared_frames(
            reader=reader,
            train_table=train_table,
            test_table=test_table,
            columns=shared_columns,
            max_rows=max_rows,
            warnings=warnings,
            limitations=limitations,
        )
        missingness_drift = _missingness_drift(train_frame, test_frame, shared_columns)
        numeric_psi = _numeric_psi(train_frame, test_frame, train_schema, test_schema)
        categorical_shift = _categorical_shift(train_frame, test_frame, train_schema, test_schema)
        adversarial = _adversarial_validation(
            train_frame,
            test_frame,
            inferred_schema,
            validation_evidence,
            random_seed=random_seed,
            warnings=warnings,
            limitations=limitations,
        )

    severity = _overall_severity(
        [
            temporal_drift.get("severity"),
            missingness_drift.get("severity"),
            numeric_psi.get("severity"),
            categorical_shift.get("severity"),
            adversarial.get("severity"),
        ]
    )
    return {
        "status": "completed",
        "severity": severity,
        "train_table": train_table,
        "test_table": test_table,
        "shared_columns": shared_columns,
        "temporal_drift": temporal_drift,
        "missingness_drift": missingness_drift,
        "numeric_psi": numeric_psi,
        "categorical_shift": categorical_shift,
        "adversarial_validation": adversarial,
        "warnings": _unique(warnings),
        "limitations": _unique(limitations),
    }


def _temporal_drift(
    *,
    reader: DatasetReader,
    train_table: str,
    train_schema: list[dict[str, str]],
    time_column: str | None,
    target_column: str | None,
    max_rows: int,
    warnings: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    if time_column is None:
        limitations.append("No time column exists; temporal drift diagnostics were skipped.")
        return {
            "status": "skipped",
            "reason": "No time column exists.",
            "row_count_by_period": [],
            "target_drift": {},
            "severity": "unknown",
        }

    columns = [time_column]
    if target_column is not None and target_column in _schema_names(train_schema):
        columns.append(target_column)
    frame = _safe_read_columns(reader, train_table, columns, max_rows, warnings, limitations)
    if frame is None or frame.is_empty():
        return {
            "status": "skipped",
            "reason": "Could not read temporal drift columns.",
            "row_count_by_period": [],
            "target_drift": {},
            "severity": "unknown",
        }

    row_count_by_period = _row_count_by_period(frame, time_column)
    target_drift: dict[str, Any] = {}
    severity = "low"
    if target_column is not None and target_column in frame.columns:
        target_by_period = _target_by_period(frame, time_column, target_column)
        if target_by_period:
            target_values = [
                item["target_mean"]
                for item in target_by_period
                if item["target_mean"] is not None
            ]
            max_delta = (
                round(max(target_values) - min(target_values), 6)
                if len(target_values) >= 2
                else 0.0
            )
            severity = _severity_from_abs_diff(max_delta, medium=0.1, high=0.25)
            target_drift = {
                "status": "computed",
                "time_column": time_column,
                "target_column": target_column,
                "by_period": target_by_period,
                "max_target_rate_delta": max_delta,
                "severity": severity,
            }
    else:
        limitations.append("Target drift by period skipped because target/time pair is unavailable.")
        target_drift = {
            "status": "skipped",
            "reason": "Target column is unavailable.",
        }

    return {
        "status": "computed",
        "time_column": time_column,
        "row_count_by_period": row_count_by_period,
        "target_drift": target_drift,
        "severity": severity,
    }


def _missingness_drift(
    train_frame: pl.DataFrame | None,
    test_frame: pl.DataFrame | None,
    shared_columns: list[str],
) -> dict[str, Any]:
    if train_frame is None or test_frame is None:
        return {"status": "skipped", "columns": [], "severity": "unknown"}

    results = []
    for column in shared_columns:
        train_missing = _missing_pct(train_frame[column])
        test_missing = _missing_pct(test_frame[column])
        abs_diff = round(abs(train_missing - test_missing), 6)
        results.append(
            {
                "column": column,
                "train_missing_pct": train_missing,
                "test_missing_pct": test_missing,
                "abs_diff": abs_diff,
                "severity": _severity_from_abs_diff(abs_diff, medium=0.1, high=0.3),
            }
        )
    return {
        "status": "computed",
        "columns": results,
        "severity": _overall_severity([item["severity"] for item in results]),
    }


def _numeric_psi(
    train_frame: pl.DataFrame | None,
    test_frame: pl.DataFrame | None,
    train_schema: list[dict[str, str]],
    test_schema: list[dict[str, str]],
) -> dict[str, Any]:
    if train_frame is None or test_frame is None:
        return {"status": "skipped", "columns": [], "severity": "unknown"}

    train_dtypes = _schema_dtypes(train_schema)
    test_dtypes = _schema_dtypes(test_schema)
    columns = [
        column
        for column in train_frame.columns
        if _is_numeric_dtype(train_dtypes.get(column, ""))
        and _is_numeric_dtype(test_dtypes.get(column, ""))
    ]
    results = []
    for column in columns:
        train_values = _float_values(train_frame[column])
        test_values = _float_values(test_frame[column])
        psi = _psi(train_values, test_values)
        if psi is None:
            continue
        results.append(
            {
                "column": column,
                "psi": psi,
                "severity": _severity_from_psi(psi),
            }
        )
    return {
        "status": "computed" if results else "skipped",
        "columns": results,
        "severity": _overall_severity([item["severity"] for item in results]),
    }


def _categorical_shift(
    train_frame: pl.DataFrame | None,
    test_frame: pl.DataFrame | None,
    train_schema: list[dict[str, str]],
    test_schema: list[dict[str, str]],
) -> dict[str, Any]:
    if train_frame is None or test_frame is None:
        return {"status": "skipped", "columns": [], "severity": "unknown"}

    train_dtypes = _schema_dtypes(train_schema)
    test_dtypes = _schema_dtypes(test_schema)
    columns = [
        column
        for column in train_frame.columns
        if _is_categorical_dtype(train_dtypes.get(column, ""))
        and _is_categorical_dtype(test_dtypes.get(column, ""))
    ]
    results = []
    for column in columns:
        distance = _total_variation_distance(
            _value_distribution(train_frame[column]),
            _value_distribution(test_frame[column]),
        )
        results.append(
            {
                "column": column,
                "total_variation_distance": distance,
                "severity": _severity_from_abs_diff(distance, medium=0.2, high=0.5),
            }
        )
    return {
        "status": "computed" if results else "skipped",
        "columns": results,
        "severity": _overall_severity([item["severity"] for item in results]),
    }


def _adversarial_validation(
    train_frame: pl.DataFrame | None,
    test_frame: pl.DataFrame | None,
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    *,
    random_seed: int,
    warnings: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    if train_frame is None or test_frame is None:
        return {"status": "skipped", "reason": "Train/test frames are unavailable."}

    feature_columns, excluded_columns = _adversarial_feature_columns(
        train_frame.columns,
        inferred_schema,
        validation_evidence,
    )
    base_result: dict[str, Any] = {
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
    }
    if not feature_columns:
        limitations.append("Adversarial validation skipped because no safe shared features exist.")
        return {
            "status": "skipped",
            "reason": "No safe shared features exist.",
            **base_result,
            "severity": "unknown",
        }

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception:
        limitations.append("sklearn is unavailable; adversarial validation was skipped.")
        return {
            "status": "skipped",
            "reason": "sklearn is unavailable.",
            **base_result,
            "severity": "unknown",
        }

    try:
        train_pd = train_frame.select(feature_columns).to_pandas()
        test_pd = test_frame.select(feature_columns).to_pandas()
        train_pd["__is_test"] = 0
        test_pd["__is_test"] = 1
        data = pl.concat(
            [pl.from_pandas(train_pd), pl.from_pandas(test_pd)],
            how="vertical_relaxed",
        ).to_pandas()
        y = data.pop("__is_test")
        numeric_features = [
            column for column in feature_columns if train_frame[column].dtype.is_numeric()
        ]
        categorical_features = [
            column for column in feature_columns if column not in numeric_features
        ]
        transformers = []
        if numeric_features:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                )
            )
        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical_features,
                )
            )
        model = Pipeline(
            [
                ("preprocess", ColumnTransformer(transformers)),
                (
                    "model",
                    LogisticRegression(max_iter=500, random_state=random_seed),
                ),
            ]
        )
        model.fit(data, y)
        scores = model.predict_proba(data)[:, 1]
        auc = round(float(roc_auc_score(y, scores)), 6)
        return {
            "status": "completed",
            "auc": auc,
            "top_features": _top_feature_scores(
                train_frame,
                test_frame,
                feature_columns,
            ),
            **base_result,
            "severity": _severity_from_auc(auc),
        }
    except Exception as exc:
        warnings.append(f"Adversarial validation failed: {exc}")
        return {
            "status": "skipped",
            "reason": "Adversarial validation failed.",
            **base_result,
            "severity": "unknown",
        }


def _adversarial_feature_columns(
    columns: list[str],
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
) -> tuple[list[str], list[str]]:
    explicit_exclusions = {
        inferred_schema.target_column,
        inferred_schema.primary_id_column,
        inferred_schema.prediction_column,
    }
    for candidate in [
        *validation_evidence.group_columns,
        *validation_evidence.query_columns,
    ]:
        name = candidate.get("name") if isinstance(candidate, dict) else None
        explicit_exclusions.add(str(name) if name is not None else None)

    excluded: list[str] = []
    features: list[str] = []
    for column in columns:
        normalized = column.lower()
        if column in explicit_exclusions or any(token in normalized for token in GROUP_ID_TOKENS):
            excluded.append(column)
        else:
            features.append(column)
    for column in explicit_exclusions:
        if column is not None and column not in excluded:
            excluded.append(column)
    return features, excluded


def _row_count_by_period(frame: pl.DataFrame, time_column: str) -> list[dict[str, Any]]:
    rows = (
        frame.group_by(time_column)
        .len()
        .sort(time_column)
        .rename({"len": "row_count"})
        .to_dicts()
    )
    return [
        {"period": _normalise_value(row[time_column]), "row_count": int(row["row_count"])}
        for row in rows
    ]


def _target_by_period(
    frame: pl.DataFrame,
    time_column: str,
    target_column: str,
) -> list[dict[str, Any]]:
    rows = (
        frame.group_by(time_column)
        .agg(
            [
                pl.len().alias("row_count"),
                pl.col(target_column).cast(pl.Float64, strict=False).mean().alias("target_mean"),
            ]
        )
        .sort(time_column)
        .to_dicts()
    )
    return [
        {
            "period": _normalise_value(row[time_column]),
            "row_count": int(row["row_count"]),
            "target_mean": _round_or_none(row["target_mean"]),
        }
        for row in rows
    ]


def _read_shared_frames(
    *,
    reader: DatasetReader,
    train_table: str,
    test_table: str,
    columns: list[str],
    max_rows: int,
    warnings: list[str],
    limitations: list[str],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    train_frame = _safe_read_columns(reader, train_table, columns, max_rows, warnings, limitations)
    test_frame = _safe_read_columns(reader, test_table, columns, max_rows, warnings, limitations)
    if train_frame is not None or test_frame is not None:
        limitations.append(f"Drift checks use at most {max_rows} rows per base table.")
    return train_frame, test_frame


def _safe_schema(
    reader: DatasetReader,
    table: str,
    warnings: list[str],
) -> list[dict[str, str]]:
    try:
        return reader.read_schema(table)
    except ReaderError as exc:
        warnings.append(str(exc))
        return []


def _safe_read_columns(
    reader: DatasetReader,
    table: str,
    columns: list[str],
    max_rows: int,
    warnings: list[str],
    limitations: list[str],
) -> pl.DataFrame | None:
    try:
        return reader.read_columns(table, columns=_unique(columns), n_rows=max_rows)
    except ReaderError as exc:
        warnings.append(str(exc))
        limitations.append(f"Could not read drift columns from {table}.")
        return None


def _select_time_column(
    validation_evidence: ValidationEvidence,
    inferred_schema: InferredSchema,
    *,
    available_columns: set[str],
) -> str | None:
    candidates: list[str] = []
    candidates.extend(str(item.get("name")) for item in validation_evidence.time_columns)
    candidates.extend(inferred_schema.candidate_time_columns)
    candidates.extend(inferred_schema.candidate_date_columns)
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    return None


def _psi(train_values: list[float], test_values: list[float]) -> float | None:
    if not train_values or not test_values:
        return None
    lower = min(train_values)
    upper = max(train_values)
    if lower == upper:
        return None
    n_bins = min(10, max(2, len(set(train_values))))
    width = (upper - lower) / n_bins
    train_counts = [0] * n_bins
    test_counts = [0] * n_bins
    for value in train_values:
        train_counts[_bin_index(value, lower, width, n_bins)] += 1
    for value in test_values:
        test_counts[_bin_index(value, lower, width, n_bins)] += 1
    train_total = sum(train_counts)
    test_total = sum(test_counts)
    psi = 0.0
    for train_count, test_count in zip(train_counts, test_counts, strict=True):
        train_pct = max(train_count / train_total, PSI_EPSILON)
        test_pct = max(test_count / test_total, PSI_EPSILON)
        psi += (test_pct - train_pct) * _safe_log(test_pct / train_pct)
    return round(float(psi), 6)


def _bin_index(value: float, lower: float, width: float, n_bins: int) -> int:
    if value <= lower:
        return 0
    index = int((value - lower) / width)
    return max(0, min(index, n_bins - 1))


def _value_distribution(series: pl.Series) -> dict[Any, float]:
    values = [_normalise_value(value) for value in series.to_list()]
    values = [value for value in values if value not in (None, "")]
    if not values:
        return {}
    counts = Counter(values).most_common(TOP_CATEGORICAL_VALUES)
    total = sum(count for _, count in counts)
    return {value: count / total for value, count in counts}


def _total_variation_distance(left: dict[Any, float], right: dict[Any, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return round(0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys), 6)


def _top_feature_scores(
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for column in feature_columns:
        if train_frame[column].dtype.is_numeric() and test_frame[column].dtype.is_numeric():
            train_values = _float_values(train_frame[column])
            test_values = _float_values(test_frame[column])
            if train_values and test_values:
                score = abs((sum(test_values) / len(test_values)) - (sum(train_values) / len(train_values)))
            else:
                score = 0.0
        else:
            score = _total_variation_distance(
                _value_distribution(train_frame[column]),
                _value_distribution(test_frame[column]),
            )
        scores.append({"feature": column, "shift_score": round(float(score), 6)})
    return sorted(scores, key=lambda item: item["shift_score"], reverse=True)[:10]


def _missing_pct(series: pl.Series) -> float:
    if len(series) == 0:
        return 0.0
    mask = series.is_null()
    if series.dtype in {pl.String, pl.Utf8}:
        mask = mask | (series.str.strip_chars() == "")
    if series.dtype.is_numeric():
        mask = mask | series.is_nan().fill_null(False)
    return round(float(mask.sum()) / len(series), 6)


def _float_values(series: pl.Series) -> list[float]:
    values = []
    for value in series.to_list():
        if value is None or value == "":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == numeric:
            values.append(numeric)
    return values


def _schema_names(schema: list[dict[str, str]]) -> set[str]:
    return {column["name"] for column in schema}


def _schema_dtypes(schema: list[dict[str, str]]) -> dict[str, str]:
    return {column["name"]: column["dtype"] for column in schema}


def _is_numeric_dtype(dtype: str) -> bool:
    lowered = dtype.lower()
    return any(token in lowered for token in NUMERIC_DTYPE_TOKENS)


def _is_categorical_dtype(dtype: str) -> bool:
    lowered = dtype.lower()
    return any(token in lowered for token in CATEGORICAL_DTYPE_TOKENS)


def _severity_from_abs_diff(value: float, *, medium: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _severity_from_psi(value: float) -> str:
    if value >= 0.25:
        return "high"
    if value >= 0.1:
        return "medium"
    return "low"


def _severity_from_auc(value: float) -> str:
    if value >= 0.8:
        return "high"
    if value >= 0.65:
        return "medium"
    return "low"


def _overall_severity(values: list[Any]) -> str:
    order = {"unknown": 0, None: 0, "low": 1, "medium": 2, "high": 3}
    highest = "unknown"
    for value in values:
        if order.get(value, 0) > order[highest]:
            highest = str(value)
    return highest


def _safe_log(value: float) -> float:
    import math

    return math.log(max(value, PSI_EPSILON))


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["analyze_drift"]
