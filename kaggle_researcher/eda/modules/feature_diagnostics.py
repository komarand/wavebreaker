from __future__ import annotations

from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.modules.column_policy import ColumnRolePolicy
from kaggle_researcher.eda.schemas import InferredSchema, MetricEvidence, TableProfile


NUMERIC_DTYPE_TOKENS = ("int", "float", "decimal")
CATEGORICAL_DTYPE_TOKENS = ("str", "utf8", "categorical", "bool")
TEXT_LENGTH_THRESHOLD = 20
HIGH_CARDINALITY_RATIO = 0.5
LOW_CARDINALITY_MAX = 20
RARE_CATEGORY_RATE_THRESHOLD = 0.2
HIGH_UNSEEN_RATE_THRESHOLD = 0.05
HIGH_MISSINGNESS_CAUTION = 0.2
SMALL_SAMPLE_ROWS = 30
MAX_DIAGNOSTIC_ROWS = 200_000


def diagnose_features(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    metric_evidence: MetricEvidence | dict[str, Any],
    drift_evidence: dict[str, Any] | None,
    reader: DatasetReader,
    max_rows: int = MAX_DIAGNOSTIC_ROWS,
) -> dict[str, Any]:
    """Build generic feature diagnostics from safe base-table columns."""

    policy = ColumnRolePolicy(inferred_schema, metric_evidence)
    base_profile = _base_profile(inferred_schema, table_profiles)
    if base_profile is None or inferred_schema.train_base_table is None:
        return {
            "status": "not_testable",
            "reason": "Train base profile is unavailable.",
            "numeric_feature_diagnostics": _empty_numeric(),
            "categorical_feature_diagnostics": _empty_categorical(),
            "missingness_diagnostics": _empty_missingness(),
            "text_feature_diagnostics": {"columns": [], "recommendations": []},
            "date_time_diagnostics": {"columns": [], "temporal_validation_signal": "not_testable"},
        }

    train_table = inferred_schema.train_base_table
    test_table = inferred_schema.test_base_table
    safe_columns = policy.safe_columns(
        [column.name for column in base_profile.columns],
        table=train_table,
        context="model_feature",
    )
    excluded_columns = policy.excluded_columns(
        [column.name for column in base_profile.columns],
        table=train_table,
        context="model_feature",
    )
    profiles_by_name = {column.name: column for column in base_profile.columns}
    numeric_columns = [
        column for column in safe_columns if _is_numeric_dtype(profiles_by_name[column].dtype)
    ]
    excluded_numeric_columns = [
        str(item["column"])
        for item in excluded_columns
        if isinstance(item, dict)
        and item.get("reason") == "primary_id"
        and item.get("column") in profiles_by_name
        and _is_numeric_dtype(profiles_by_name[str(item["column"])].dtype)
    ]
    categorical_columns = [
        column for column in safe_columns if _is_categorical_dtype(profiles_by_name[column].dtype)
    ]
    date_columns = [
        column.name
        for column in base_profile.columns
        if policy.is_time_or_date(column.name, train_table) or column.date_min is not None
    ]

    needed_columns = _unique([
        *safe_columns,
        *( [inferred_schema.target_column] if inferred_schema.target_column else [] ),
    ])
    train_frame = _safe_read(reader, train_table, needed_columns, max_rows)
    test_frame = (
        _safe_read(reader, test_table, safe_columns, max_rows)
        if test_table is not None and safe_columns
        else None
    )

    numeric = _numeric_diagnostics(
        numeric_columns,
        profiles_by_name,
        train_frame,
        inferred_schema.target_column,
        _as_dict(drift_evidence),
        excluded_numeric_columns,
    )
    categorical = _categorical_diagnostics(
        categorical_columns,
        profiles_by_name,
        train_frame,
        test_frame,
        inferred_schema.target_column,
    )
    missingness = _missingness_diagnostics(
        safe_columns,
        profiles_by_name,
        train_frame,
        test_frame,
        inferred_schema.target_column,
        _as_dict(drift_evidence),
    )
    text = _text_diagnostics(categorical_columns, profiles_by_name, train_frame)
    date_time = _date_time_diagnostics(
        date_columns,
        profiles_by_name,
        train_frame,
        test_frame,
        inferred_schema.target_column,
        metric_evidence,
    )
    return {
        "status": "completed",
        "train_table": train_table,
        "test_table": test_table,
        "safe_feature_columns": safe_columns,
        "excluded_columns": excluded_columns,
        "numeric_feature_diagnostics": numeric,
        "categorical_feature_diagnostics": categorical,
        "missingness_diagnostics": missingness,
        "text_feature_diagnostics": text,
        "date_time_diagnostics": date_time,
        "warnings": [],
        "limitations": [f"Feature diagnostics use at most {max_rows} rows per base table."],
    }


def _numeric_diagnostics(
    columns: list[str],
    profiles: dict[str, Any],
    train_frame: pl.DataFrame | None,
    target_column: str | None,
    drift: dict[str, Any],
    excluded_numeric_columns: list[str] | None = None,
) -> dict[str, Any]:
    shifted = {
        item.get("column"): item
        for item in _as_dict(drift.get("numeric_psi")).get("columns", [])
    }
    rows = []
    for column in columns:
        profile = profiles[column]
        numeric_kind = _feature_numeric_kind(profile, train_frame, column)
        skew_proxy = _skew_proxy(profile)
        outlier_applicable = numeric_kind == "continuous"
        outlier_rate = _outlier_rate(profile)
        rare_large_value_score = _rare_large_value_score(profile, train_frame, column)
        association = _numeric_target_association(train_frame, column, target_column)
        row = {
            "column": column,
            "feature_numeric_kind": numeric_kind,
            "feature_value_type": numeric_kind,
            "missing_pct": profile.missing_pct,
            "n_unique": profile.n_unique,
            "unique_ratio": profile.unique_ratio,
            "skew_proxy": skew_proxy,
            "outlier_rate": outlier_rate,
            "rare_large_value_score": rare_large_value_score,
            "outlier_diagnostic_method": (
                "quantile_tail_proxy"
                if outlier_applicable
                else "discrete_rare_large_value_proxy"
            ),
            "train_test_shift_score": _as_dict(shifted.get(column)).get("psi", 0.0),
            "target_association": association,
            "target_association_reliability": _numeric_target_association_reliability(
                numeric_kind,
                profile,
                train_frame.height if train_frame is not None else 0,
            ),
            "outlier_reliability": _numeric_outlier_reliability(numeric_kind),
            "shift_reliability": _numeric_shift_reliability(
                numeric_kind,
                train_frame.height if train_frame is not None else 0,
            ),
            "interpretation_warnings": _numeric_interpretation_warnings(numeric_kind),
        }
        rows.append(row)
    return {
        "columns": rows,
        "excluded_id_like": [
            {
                "column": column,
                "feature_numeric_kind": "id_like_excluded",
                "feature_value_type": "id_like_excluded",
                "outlier_rate": 0.0,
                "outlier_diagnostic_method": "not_applicable",
                "target_association_reliability": "not_applicable",
                "outlier_reliability": "not_applicable",
                "shift_reliability": "not_applicable",
                "interpretation_warnings": ["Column is excluded by role policy and is not an ordinary feature."],
            }
            for column in excluded_numeric_columns or []
        ],
        "top_predictive_candidates": _top_predictive(rows),
        "high_missingness": [row for row in rows if row["missing_pct"] >= 0.2],
        "outlier_heavy": [
            row
            for row in rows
            if row["feature_value_type"] == "continuous"
            and row["outlier_rate"] >= 0.05
            and row["outlier_reliability"] == "reliable"
        ],
        "rare_large_value_features": [
            row
            for row in rows
            if row["feature_value_type"] in {"binary", "ordinal_low_cardinality", "count_zero_inflated", "count"}
            and (row["outlier_rate"] >= 0.05 or row["rare_large_value_score"] >= 10.0)
        ],
        "shifted_features": [row for row in rows if row["train_test_shift_score"] >= 0.1],
        "low_information": [
            row
            for row in rows
            if row["n_unique"] <= 1
            or (
                row["unique_ratio"] <= 0.01
                and row["target_association"] < 0.05
                and row["feature_value_type"] == "continuous"
            )
        ],
    }


def _categorical_diagnostics(
    columns: list[str],
    profiles: dict[str, Any],
    train_frame: pl.DataFrame | None,
    test_frame: pl.DataFrame | None,
    target_column: str | None,
) -> dict[str, Any]:
    rows = []
    for column in columns:
        profile = profiles[column]
        unseen_rate = _unseen_category_rate(train_frame, test_frame, column)
        rare_rate = _rare_category_rate(train_frame, column)
        top_concentration = _top_category_concentration(profile)
        association = _categorical_target_association(train_frame, column, target_column)
        avg_len, avg_tokens, punctuation_ratio = _string_shape(train_frame, column)
        feature_value_type = _categorical_feature_value_type(
            profile,
            avg_len=avg_len,
            avg_tokens=avg_tokens,
            punctuation_ratio=punctuation_ratio,
        )
        cardinality = profile.n_unique or 0
        reliability = _target_association_reliability(
            cardinality=cardinality,
            unique_ratio=float(profile.unique_ratio or 0.0),
            n_rows=train_frame.height if train_frame is not None else 0,
            rare_category_rate=rare_rate,
            unseen_category_rate=unseen_rate,
            feature_value_type=feature_value_type,
        )
        encoding_reliability = _encoding_reliability(
            feature_value_type=feature_value_type,
            reliability=reliability,
            rare_category_rate=rare_rate,
            unseen_category_rate=unseen_rate,
            association=association,
        )
        row = {
            "column": column,
            "feature_value_type": feature_value_type,
            "cardinality": cardinality,
            "unique_ratio": profile.unique_ratio,
            "rare_category_rate": rare_rate,
            "unseen_category_rate": unseen_rate,
            "top_category_concentration": top_concentration,
            "missing_pct": profile.missing_pct,
            "target_association": association,
            "target_association_reliability": reliability,
            "encoding_reliability": encoding_reliability,
            "shift_reliability": _categorical_shift_reliability(unseen_rate, rare_rate, cardinality),
            "interpretation_warnings": _categorical_interpretation_warnings(
                feature_value_type=feature_value_type,
                reliability=reliability,
                unseen_category_rate=unseen_rate,
                rare_category_rate=rare_rate,
            ),
            "requires_fold_fitted_encoding": cardinality > LOW_CARDINALITY_MAX or association >= 0.2,
        }
        rows.append(row)
    return {
        "columns": rows,
        "low_cardinality_candidates": [row for row in rows if row["feature_value_type"] == "low_cardinality_categorical"],
        "high_cardinality_candidates": [
            row
            for row in rows
            if row["feature_value_type"] in {"high_cardinality_categorical", "code_like", "mixed_text_code"}
        ],
        "unseen_category_risks": [row for row in rows if row["unseen_category_rate"] >= HIGH_UNSEEN_RATE_THRESHOLD],
        "rare_category_heavy": [row for row in rows if row["rare_category_rate"] >= RARE_CATEGORY_RATE_THRESHOLD],
        "high_target_association_candidates": [
            row
            for row in rows
            if row["target_association"] >= 0.2
            and row["target_association_reliability"] == "reliable"
        ],
        "target_association_cautions": [
            row
            for row in rows
            if row["target_association_reliability"]
            in {"caution_high_cardinality", "caution_sparse_categories", "caution_high_unseen_rate", "caution_small_sample", "not_reliable"}
        ],
    }


def _missingness_diagnostics(
    columns: list[str],
    profiles: dict[str, Any],
    train_frame: pl.DataFrame | None,
    test_frame: pl.DataFrame | None,
    target_column: str | None,
    drift: dict[str, Any],
) -> dict[str, Any]:
    missing_shift = {
        item.get("column"): item
        for item in _as_dict(drift.get("missingness_drift")).get("columns", [])
    }
    rows = []
    for column in columns:
        profile = profiles[column]
        if not profile.missing_pct:
            continue
        association = _missingness_target_association(train_frame, column, target_column)
        shift = _as_dict(missing_shift.get(column)).get("abs_diff", 0.0)
        row = {
            "column": column,
            "missing_pct": profile.missing_pct,
            "target_association": association,
            "train_test_missingness_shift": shift,
        }
        rows.append(row)
    return {
        "columns": rows,
        "target_associated_missingness": [row for row in rows if row["target_association"] >= 0.1],
        "train_test_missingness_shift": [row for row in rows if row["train_test_missingness_shift"] >= 0.1],
        "recommended_indicators": [
            row for row in rows if row["missing_pct"] >= 0.05 or row["target_association"] >= 0.1
        ],
    }


def _text_diagnostics(
    columns: list[str],
    profiles: dict[str, Any],
    train_frame: pl.DataFrame | None,
) -> dict[str, Any]:
    rows = []
    for column in columns:
        profile = profiles[column]
        avg_len, avg_tokens, punctuation_ratio = _string_shape(train_frame, column)
        feature_value_type = _categorical_feature_value_type(
            profile,
            avg_len=avg_len,
            avg_tokens=avg_tokens,
            punctuation_ratio=punctuation_ratio,
        )
        if feature_value_type == "low_cardinality_categorical":
            continue
        rows.append(
            {
                "column": column,
                "classification": feature_value_type,
                "feature_value_type": feature_value_type,
                "avg_string_length": avg_len,
                "avg_token_count": avg_tokens,
                "punctuation_ratio": punctuation_ratio,
                "unique_ratio": profile.unique_ratio,
                "recommendations": _text_recommendations(feature_value_type),
            }
        )
    return {"columns": rows, "recommendations": _unique_flat(row["recommendations"] for row in rows)}


def _date_time_diagnostics(
    columns: list[str],
    profiles: dict[str, Any],
    train_frame: pl.DataFrame | None,
    test_frame: pl.DataFrame | None,
    target_column: str | None,
    metric_evidence: MetricEvidence | dict[str, Any],
) -> dict[str, Any]:
    metric = _as_dict(metric_evidence)
    rows = []
    for column in columns:
        profile = profiles.get(column)
        train_min = profile.date_min if profile is not None else None
        train_max = profile.date_max if profile is not None else None
        rows.append(
            {
                "column": column,
                "parse_success": "profile_detected" if train_min or train_max else "name_only",
                "train_min": train_min,
                "train_max": train_max,
                "test_range": _min_max(test_frame, column),
                "target_by_period_available": bool(target_column and train_frame is not None and column in train_frame.columns),
            }
        )
    if metric.get("requires_time"):
        signal = "required"
    elif rows:
        signal = "diagnostic_only"
    else:
        signal = "not_available"
    return {"columns": rows, "temporal_validation_signal": signal}


def _safe_read(
    reader: DatasetReader,
    table: str | None,
    columns: list[str],
    max_rows: int,
) -> pl.DataFrame | None:
    if table is None or not columns:
        return None
    try:
        schema = {column["name"] for column in reader.read_schema(table)}
        available_columns = [column for column in columns if column in schema]
        if not available_columns:
            return None
        return reader.read_columns(table, columns=available_columns, n_rows=max_rows)
    except ReaderError:
        return None


def _base_profile(inferred_schema: InferredSchema, table_profiles: list[TableProfile]) -> TableProfile | None:
    if inferred_schema.train_base_table:
        for profile in table_profiles:
            if profile.path == inferred_schema.train_base_table:
                return profile
    return table_profiles[0] if table_profiles else None


def _skew_proxy(profile: Any) -> float:
    if profile.q50 in (None, 0) or profile.max is None:
        return 0.0
    try:
        return round(abs(float(profile.max)) / max(abs(float(profile.q50)), 1e-9), 6)
    except (TypeError, ValueError):
        return 0.0


def _outlier_rate(profile: Any) -> float:
    if profile.q05 is None or profile.q95 is None or profile.min is None or profile.max is None:
        return 0.0
    try:
        width = max(float(profile.q95) - float(profile.q05), 1e-9)
        outside = max(0.0, float(profile.q05) - float(profile.min)) + max(0.0, float(profile.max) - float(profile.q95))
        return round(min(1.0, outside / (outside + width)), 6)
    except (TypeError, ValueError):
        return 0.0


def _rare_large_value_score(profile: Any, frame: pl.DataFrame | None, column: str) -> float:
    if frame is not None and column in frame.columns:
        values = sorted(_float_values(frame[column]))
        if values:
            median = values[len(values) // 2]
            denominator = median if median != 0 else 1.0
            return round(abs(max(values)) / max(abs(denominator), 1e-9), 6)
    if profile.max is None:
        return 0.0
    denominator = profile.q50 if profile.q50 not in (None, 0) else profile.q95
    if denominator is None:
        denominator = 1.0
    try:
        return round(abs(float(profile.max)) / max(abs(float(denominator)), 1e-9), 6)
    except (TypeError, ValueError):
        return 0.0


def _feature_numeric_kind(
    profile: Any,
    frame: pl.DataFrame | None,
    column: str,
) -> str:
    n_unique = int(profile.n_unique or 0)
    dtype = str(profile.dtype).lower()
    integer_like = "int" in dtype
    minimum = profile.min
    nonnegative = isinstance(minimum, (int, float)) and minimum >= 0
    zero_rate = _zero_rate(frame, column)
    unique_ratio = float(profile.unique_ratio or 0.0)
    if n_unique == 2:
        return "binary"
    if integer_like and nonnegative and zero_rate >= 0.2 and n_unique > 3:
        return "count_zero_inflated"
    if n_unique and n_unique <= LOW_CARDINALITY_MAX and unique_ratio <= 0.2:
        return "ordinal_low_cardinality"
    if integer_like and nonnegative:
        return "count"
    return "continuous"


def _zero_rate(frame: pl.DataFrame | None, column: str) -> float:
    if frame is None or column not in frame.columns or frame.height == 0:
        return 0.0
    values = frame[column].drop_nulls()
    if values.len() == 0:
        return 0.0
    try:
        return float((values == 0).sum()) / values.len()
    except (TypeError, ValueError):
        return 0.0


def _target_association_reliability(
    *,
    cardinality: int,
    unique_ratio: float,
    n_rows: int,
    rare_category_rate: float = 0.0,
    unseen_category_rate: float = 0.0,
    feature_value_type: str = "low_cardinality_categorical",
) -> str:
    average_rows_per_category = n_rows / cardinality if cardinality and n_rows else 0.0
    if unique_ratio >= 0.9 or average_rows_per_category < 2:
        return "not_reliable"
    if n_rows and n_rows < SMALL_SAMPLE_ROWS:
        return "caution_small_sample"
    if unseen_category_rate >= HIGH_UNSEEN_RATE_THRESHOLD:
        return "caution_high_unseen_rate"
    if rare_category_rate >= RARE_CATEGORY_RATE_THRESHOLD:
        return "caution_sparse_categories"
    if feature_value_type in {"text_like", "code_like", "mixed_text_code"}:
        return "caution_high_cardinality"
    if unique_ratio >= HIGH_CARDINALITY_RATIO:
        return "caution_high_cardinality"
    if cardinality <= LOW_CARDINALITY_MAX:
        return "reliable"
    return "caution_high_cardinality"


def _numeric_target_association_reliability(
    feature_value_type: str,
    profile: Any,
    n_rows: int,
) -> str:
    if n_rows == 0:
        return "not_applicable"
    if n_rows < SMALL_SAMPLE_ROWS:
        return "caution_small_sample"
    if float(profile.missing_pct or 0.0) >= HIGH_MISSINGNESS_CAUTION:
        return "caution_high_missingness"
    if feature_value_type in {"binary", "ordinal_low_cardinality"}:
        return "caution_low_cardinality"
    return "reliable"


def _numeric_outlier_reliability(feature_value_type: str) -> str:
    if feature_value_type == "continuous":
        return "reliable"
    if feature_value_type == "binary":
        return "not_applicable"
    if feature_value_type == "ordinal_low_cardinality":
        return "caution_low_cardinality"
    if feature_value_type == "count_zero_inflated":
        return "caution_zero_inflated"
    if feature_value_type == "count":
        return "caution_count_feature"
    return "not_applicable"


def _numeric_shift_reliability(feature_value_type: str, n_rows: int) -> str:
    if n_rows == 0:
        return "not_applicable"
    if n_rows < SMALL_SAMPLE_ROWS:
        return "caution_small_sample"
    if feature_value_type in {"binary", "ordinal_low_cardinality"}:
        return "caution_low_cardinality"
    return "reliable"


def _numeric_interpretation_warnings(feature_value_type: str) -> list[str]:
    if feature_value_type == "binary":
        return ["Binary numeric features are discrete; continuous outlier diagnostics are not applicable."]
    if feature_value_type == "ordinal_low_cardinality":
        return ["Low-cardinality numeric features should be interpreted as ordinal/discrete, not continuous."]
    if feature_value_type == "count_zero_inflated":
        return ["Zero-inflated count features can have rare large values; IQR-style outlier flags need caution."]
    if feature_value_type == "count":
        return ["Count features are discrete; rare large values are not the same as continuous outliers."]
    return []


def _categorical_feature_value_type(
    profile: Any,
    *,
    avg_len: float,
    avg_tokens: float,
    punctuation_ratio: float,
) -> str:
    unique_ratio = float(profile.unique_ratio or 0.0)
    cardinality = int(profile.n_unique or 0)
    if avg_tokens >= 3 or avg_len >= 35:
        if unique_ratio >= HIGH_CARDINALITY_RATIO and punctuation_ratio >= 0.05:
            return "mixed_text_code"
        return "text_like"
    if unique_ratio >= 0.8 and avg_len <= 24:
        return "code_like"
    if unique_ratio >= HIGH_CARDINALITY_RATIO or cardinality > LOW_CARDINALITY_MAX:
        return "high_cardinality_categorical"
    return "low_cardinality_categorical"


def _encoding_reliability(
    *,
    feature_value_type: str,
    reliability: str,
    rare_category_rate: float,
    unseen_category_rate: float,
    association: float,
) -> str:
    if association >= 0.2:
        return "high_leakage_risk_if_target_encoded"
    if feature_value_type in {"high_cardinality_categorical", "text_like", "code_like", "mixed_text_code"}:
        return "requires_fold_fitted_encoding"
    if rare_category_rate >= RARE_CATEGORY_RATE_THRESHOLD or unseen_category_rate >= HIGH_UNSEEN_RATE_THRESHOLD:
        return "requires_rare_handling"
    if reliability == "reliable":
        return "safe_basic_encoding"
    return "requires_rare_handling"


def _categorical_shift_reliability(unseen_category_rate: float, rare_category_rate: float, cardinality: int) -> str:
    if unseen_category_rate >= HIGH_UNSEEN_RATE_THRESHOLD:
        return "caution_high_unseen_rate"
    if rare_category_rate >= RARE_CATEGORY_RATE_THRESHOLD or cardinality > LOW_CARDINALITY_MAX:
        return "caution_sparse_categories"
    return "reliable"


def _categorical_interpretation_warnings(
    *,
    feature_value_type: str,
    reliability: str,
    unseen_category_rate: float,
    rare_category_rate: float,
) -> list[str]:
    warnings: list[str] = []
    if feature_value_type in {"high_cardinality_categorical", "code_like", "mixed_text_code"}:
        warnings.append("High-cardinality or code-like categories can produce unstable target rates.")
    if feature_value_type == "text_like":
        warnings.append("Text-like columns should be featurized as text/count signals before raw categorical encoding.")
    if reliability == "not_reliable":
        warnings.append("Target association is not reliable because categories are too sparse or near-unique.")
    if unseen_category_rate >= HIGH_UNSEEN_RATE_THRESHOLD:
        warnings.append("Many test categories are unseen in train; simple category lookup may not generalize.")
    if rare_category_rate >= RARE_CATEGORY_RATE_THRESHOLD:
        warnings.append("Rare categories require grouping or robust encoding.")
    return warnings


def _numeric_target_association(frame: pl.DataFrame | None, column: str, target: str | None) -> float:
    if frame is None or target is None or column not in frame.columns or target not in frame.columns:
        return 0.0
    return _abs_correlation(_float_values(frame[column]), _float_values(frame[target]))


def _categorical_target_association(frame: pl.DataFrame | None, column: str, target: str | None) -> float:
    if frame is None or target is None or column not in frame.columns or target not in frame.columns:
        return 0.0
    rows = frame.select([column, target]).drop_nulls()
    if rows.height < 3:
        return 0.0
    target_values = _float_values(rows[target])
    if not target_values:
        return 0.0
    global_mean = sum(target_values) / len(target_values)
    scores = []
    for group in rows.group_by(column).agg(pl.col(target).cast(pl.Float64, strict=False).mean().alias("target_mean")).to_dicts():
        value = group.get("target_mean")
        if value is not None:
            scores.append(abs(float(value) - global_mean))
    return round(max(scores) if scores else 0.0, 6)


def _missingness_target_association(frame: pl.DataFrame | None, column: str, target: str | None) -> float:
    if frame is None or target is None or column not in frame.columns or target not in frame.columns:
        return 0.0
    mask = frame[column].is_null().cast(pl.Int8)
    return _abs_correlation([float(value) for value in mask.to_list()], _float_values(frame[target]))


def _unseen_category_rate(train_frame: pl.DataFrame | None, test_frame: pl.DataFrame | None, column: str) -> float:
    if train_frame is None or test_frame is None or column not in train_frame.columns or column not in test_frame.columns:
        return 0.0
    train_values = _value_set(train_frame[column])
    test_values = _value_set(test_frame[column])
    if not test_values:
        return 0.0
    return round(len(test_values - train_values) / len(test_values), 6)


def _rare_category_rate(frame: pl.DataFrame | None, column: str) -> float:
    if frame is None or column not in frame.columns or frame.height == 0:
        return 0.0
    counts = frame[column].drop_nulls().value_counts().to_dicts()
    total = sum(int(row.get("count", 0)) for row in counts)
    if total == 0:
        return 0.0
    rare = sum(int(row.get("count", 0)) for row in counts if int(row.get("count", 0)) <= 2)
    return round(rare / total, 6)


def _top_category_concentration(profile: Any) -> float:
    total_known = sum(int(item.get("count", 0)) for item in profile.top_values)
    if not total_known or not profile.top_values:
        return 0.0
    return round(int(profile.top_values[0].get("count", 0)) / total_known, 6)


def _string_shape(frame: pl.DataFrame | None, column: str) -> tuple[float, float, float]:
    if frame is None or column not in frame.columns:
        return 0.0, 0.0, 0.0
    values = [str(value) for value in frame[column].drop_nulls().to_list()[:1000]]
    if not values:
        return 0.0, 0.0, 0.0
    avg_len = sum(len(value) for value in values) / len(values)
    avg_tokens = sum(len(value.split()) for value in values) / len(values)
    punctuation = sum(sum(not char.isalnum() and not char.isspace() for char in value) for value in values)
    total_chars = sum(len(value) for value in values) or 1
    return round(avg_len, 6), round(avg_tokens, 6), round(punctuation / total_chars, 6)


def _text_recommendations(classification: str) -> list[str]:
    if classification == "text_like":
        return ["Add length/count/token features before considering heavier NLP."]
    if classification == "code_like":
        return ["Use prefix/suffix/frequency diagnostics; avoid treating raw codes as stable ordinary categories."]
    if classification == "mixed_text_code":
        return ["Split text length/count signals from code-like prefix/suffix/frequency hypotheses."]
    if classification == "high_cardinality_categorical":
        return ["Use rare handling and fold-fitted encoders; validate category shift."]
    return ["Use fold-fitted encoders and rare-category handling."]


def _min_max(frame: pl.DataFrame | None, column: str) -> dict[str, Any]:
    if frame is None or column not in frame.columns:
        return {"min": None, "max": None}
    try:
        return {"min": _normalise(frame[column].min()), "max": _normalise(frame[column].max())}
    except Exception:
        return {"min": None, "max": None}


def _value_set(series: pl.Series) -> set[Any]:
    return {value for value in series.drop_nulls().to_list() if value != ""}


def _float_values(series: pl.Series) -> list[float]:
    values = []
    for value in series.to_list():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == numeric:
            values.append(numeric)
    return values


def _abs_correlation(left: list[float], right: list[float]) -> float:
    pairs = [(l, r) for l, r in zip(left, right, strict=False)]
    if len(pairs) < 3:
        return 0.0
    left_values = [pair[0] for pair in pairs]
    right_values = [pair[1] for pair in pairs]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_var = sum((value - left_mean) ** 2 for value in left_values)
    right_var = sum((value - right_mean) ** 2 for value in right_values)
    if left_var == 0 or right_var == 0:
        return 0.0
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    return round(abs(covariance / ((left_var * right_var) ** 0.5)), 6)


def _top(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: item.get(key) or 0, reverse=True)[:10]


def _top_predictive(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reliability_rank = {
        "reliable": 0,
        "caution_low_cardinality": 1,
        "caution_high_missingness": 2,
        "caution_small_sample": 3,
        "not_applicable": 4,
    }
    return sorted(
        rows,
        key=lambda item: (
            reliability_rank.get(str(item.get("target_association_reliability")), 9),
            -(float(item.get("target_association") or 0.0)),
            str(item.get("column") or ""),
        ),
    )[:10]


def _is_numeric_dtype(dtype: str) -> bool:
    return any(token in str(dtype).lower() for token in NUMERIC_DTYPE_TOKENS)


def _is_categorical_dtype(dtype: str) -> bool:
    return any(token in str(dtype).lower() for token in CATEGORICAL_DTYPE_TOKENS)


def _empty_numeric() -> dict[str, Any]:
    return {"columns": [], "excluded_id_like": [], "top_predictive_candidates": [], "high_missingness": [], "outlier_heavy": [], "rare_large_value_features": [], "shifted_features": [], "low_information": []}


def _empty_categorical() -> dict[str, Any]:
    return {"columns": [], "low_cardinality_candidates": [], "high_cardinality_candidates": [], "unseen_category_risks": [], "rare_category_heavy": [], "high_target_association_candidates": [], "target_association_cautions": []}


def _empty_missingness() -> dict[str, Any]:
    return {"columns": [], "target_associated_missingness": [], "train_test_missingness_shift": [], "recommended_indicators": []}


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def _normalise(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _unique_flat(values: Any) -> list[str]:
    result: list[str] = []
    for items in values:
        result.extend(items)
    return _unique(result)


__all__ = ["diagnose_features"]
