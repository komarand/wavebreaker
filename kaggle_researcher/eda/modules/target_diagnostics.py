from __future__ import annotations

from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.modules.column_policy import ColumnRolePolicy
from kaggle_researcher.eda.schemas import InferredSchema, MetricEvidence, TableProfile


MAX_TARGET_DIAGNOSTIC_ROWS = 200_000
MAX_FEATURE_COLUMNS = 12
MAX_MISSINGNESS_COLUMNS = 20
MAX_GROUP_ROWS = 12
MIN_RELIABLE_BIN_COUNT = 20
HIGH_CARDINALITY_RATIO = 0.5


def diagnose_target(
    inferred_schema: InferredSchema,
    metric_evidence: MetricEvidence | dict[str, Any],
    validation_evidence: dict[str, Any] | Any,
    feature_diagnostics: dict[str, Any] | None,
    table_profiles: list[TableProfile],
    reader: DatasetReader,
    max_rows: int = MAX_TARGET_DIAGNOSTIC_ROWS,
) -> dict[str, Any]:
    """Build generic, role-aware target diagnostics for the train base table."""

    metric = _as_dict(metric_evidence)
    validation = _as_dict(validation_evidence)
    warnings: list[str] = []
    limitations = [f"Target diagnostics use at most {max_rows} train rows."]
    target_column = inferred_schema.target_column
    train_table = inferred_schema.train_base_table
    task_type = str(metric.get("task_type") or "unknown")
    metric_name = str(metric.get("metric_name") or "unknown")

    if _unsupported_task_type(task_type, metric):
        return _not_testable(
            "Unsupported target diagnostic task type for this MVP.",
            target_column=target_column,
            task_type=task_type,
            metric_name=metric_name,
        )
    if not train_table:
        return _not_testable(
            "Train base table is unavailable.",
            target_column=target_column,
            task_type=task_type,
            metric_name=metric_name,
        )
    if not target_column:
        return _not_testable(
            "Target column is not inferred.",
            target_column=None,
            task_type=task_type,
            metric_name=metric_name,
        )

    base_profile = _base_profile(inferred_schema, table_profiles)
    if base_profile is None:
        return _not_testable(
            "Train base profile is unavailable.",
            target_column=target_column,
            task_type=task_type,
            metric_name=metric_name,
        )

    policy = ColumnRolePolicy(inferred_schema, metric_evidence)
    profiles = {column.name: column for column in base_profile.columns}
    safe_columns = policy.safe_columns(
        [column.name for column in base_profile.columns],
        table=train_table,
        context="model_feature",
    )
    safe_columns = [column for column in safe_columns if column != target_column]
    selected_feature_columns = _selected_feature_columns(safe_columns, profiles)
    group_columns = [
        column
        for column in inferred_schema.candidate_group_columns
        if column in profiles
    ][:2]
    profile_time_columns = [
        column.name
        for column in base_profile.columns
        if getattr(column, "date_min", None) is not None
    ]
    time_columns = [
        column
        for column in [
            *inferred_schema.candidate_time_columns,
            *inferred_schema.candidate_date_columns,
            *profile_time_columns,
        ]
        if column in profiles
    ][:2]
    needed_columns = _unique([
        target_column,
        *selected_feature_columns,
        *group_columns,
        *time_columns,
    ])
    frame = _safe_read(reader, train_table, needed_columns, max_rows, warnings, limitations)
    if frame is None or target_column not in frame.columns:
        return _not_testable(
            f"Target column '{target_column}' could not be read from train base table.",
            target_column=target_column,
            task_type=task_type,
            metric_name=metric_name,
            warnings=warnings,
            limitations=limitations,
        )

    target = frame[target_column]
    target_type = _target_type(task_type, metric, target)
    distribution = (
        _regression_distribution(target)
        if target_type == "regression"
        else _classification_distribution(target, target_type)
    )
    imbalance = _imbalance(distribution) if target_type in {"binary", "multiclass", "ranking"} else {}
    metric_implications = _metric_implications(metric, distribution, imbalance, target_type)
    validation_implications = _validation_implications(
        validation,
        distribution,
        imbalance,
        target_type,
        bool(time_columns),
        bool(group_columns),
    )
    target_by_feature = _target_by_feature(
        frame,
        target_column,
        selected_feature_columns,
        profiles,
        target_type,
    )
    target_by_missingness = _target_by_missingness(
        frame,
        target_column,
        safe_columns,
        profiles,
        target_type,
    )
    target_by_group = _target_by_group(frame, target_column, group_columns, target_type)
    target_by_time = _target_by_time(frame, target_column, time_columns, target_type)
    suspicious_patterns = _suspicious_patterns(
        distribution,
        imbalance,
        target_by_feature,
        target_column,
        target_type,
    )
    recommended_actions = _recommended_actions(
        distribution,
        imbalance,
        metric_implications,
        validation_implications,
        target_by_missingness,
        suspicious_patterns,
        target_type,
    )

    return {
        "status": "completed",
        "target_column": target_column,
        "task_type": task_type,
        "metric_name": metric_name,
        "distribution": distribution,
        "imbalance": imbalance,
        "metric_implications": metric_implications,
        "validation_implications": validation_implications,
        "target_by_feature": target_by_feature,
        "target_by_missingness": target_by_missingness,
        "target_by_group": target_by_group,
        "target_by_time": target_by_time,
        "suspicious_patterns": suspicious_patterns,
        "recommended_actions": recommended_actions,
        "limitations": _unique(limitations),
        "warnings": _unique(warnings),
    }


def _classification_distribution(target: pl.Series, target_type: str) -> dict[str, Any]:
    values = target.drop_nulls()
    counts = _value_counts(values)
    total = sum(row["count"] for row in counts)
    classes = [
        {
            "class": row["class"],
            "count": row["count"],
            "pct": _round(row["count"] / total if total else 0.0),
        }
        for row in counts
    ]
    result: dict[str, Any] = {
        "target_type": target_type,
        "n_rows": int(target.len()),
        "missing_target_count": int(target.null_count()),
        "n_classes": len(classes),
        "classes": classes,
    }
    if target_type == "binary" and classes:
        positive = classes[-1]
        result["positive_class"] = positive["class"]
        result["positive_rate"] = positive["pct"]
    if target_type in {"multiclass", "ranking"}:
        result["rare_classes"] = [
            item for item in classes if item["pct"] < 0.05 or item["count"] < 5
        ]
    return result


def _regression_distribution(target: pl.Series) -> dict[str, Any]:
    values = _numeric_values(target)
    missing = int(target.null_count())
    if not values:
        return {
            "target_type": "regression",
            "n_rows": int(target.len()),
            "missing_target_count": missing,
        }
    sorted_values = sorted(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance**0.5
    q01 = _quantile(sorted_values, 0.01)
    q05 = _quantile(sorted_values, 0.05)
    q50 = _quantile(sorted_values, 0.5)
    q95 = _quantile(sorted_values, 0.95)
    q99 = _quantile(sorted_values, 0.99)
    tail_width = max(q95 - q05, 1e-9)
    outlier_count = sum(1 for value in values if value < q05 - 1.5 * tail_width or value > q95 + 1.5 * tail_width)
    skew_proxy = abs((mean - q50) / max(std, 1e-9)) if std else 0.0
    max_to_median = abs(max(values)) / max(abs(q50), 1e-9) if q50 else abs(max(values))
    heavy_tail = outlier_count / len(values) >= 0.02 or max_to_median >= 10.0
    return {
        "target_type": "regression",
        "n_rows": int(target.len()),
        "non_null_count": len(values),
        "missing_target_count": missing,
        "mean": _round(mean),
        "std": _round(std),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "quantiles": {
            "q01": _round(q01),
            "q05": _round(q05),
            "q50": _round(q50),
            "q95": _round(q95),
            "q99": _round(q99),
        },
        "skew_proxy": _round(skew_proxy),
        "outlier_rate": _round(outlier_count / len(values)),
        "zero_rate": _round(sum(value == 0 for value in values) / len(values)),
        "negative_rate": _round(sum(value < 0 for value in values) / len(values)),
        "heavy_tail": heavy_tail,
        "target_transform_hints": _regression_transform_hints(heavy_tail, skew_proxy, outlier_count / len(values)),
    }


def _imbalance(distribution: dict[str, Any]) -> dict[str, Any]:
    classes = list(distribution.get("classes") or [])
    if not classes:
        return {
            "severity": "extreme",
            "recommended_handling": ["Inspect why no non-null target classes were found."],
        }
    minority = min(classes, key=lambda item: (item.get("pct", 0.0), item.get("class", "")))
    majority = max(classes, key=lambda item: (item.get("pct", 0.0), item.get("class", "")))
    minority_pct = float(minority.get("pct") or 0.0)
    severity = _imbalance_severity(minority_pct)
    handling = []
    if severity in {"moderate", "severe", "extreme"}:
        handling.extend([
            "Track minority-class performance in validation.",
            "Check per-fold class counts before trusting scores.",
        ])
    if len(classes) > 2:
        handling.append("Use stratified folds where compatible with group/time constraints.")
    return {
        "severity": severity,
        "minority_class": minority.get("class"),
        "minority_pct": _round(minority_pct),
        "majority_class": majority.get("class"),
        "majority_pct": _round(float(majority.get("pct") or 0.0)),
        "minority_majority_ratio": _round(minority_pct / max(float(majority.get("pct") or 0.0), 1e-9)),
        "rare_classes": distribution.get("rare_classes", []),
        "recommended_handling": handling,
    }


def _metric_implications(
    metric: dict[str, Any],
    distribution: dict[str, Any],
    imbalance: dict[str, Any],
    target_type: str,
) -> list[dict[str, Any]]:
    implications: list[dict[str, Any]] = []
    if metric.get("requires_threshold"):
        implications.append(_implication("P0", "threshold_tuning_required", "Metric requires thresholded predictions.", ["metric_evidence.requires_threshold"]))
    if metric.get("requires_probabilities"):
        implications.append(_implication("P0", "probability_output_required", "Metric requires probability or score outputs.", ["metric_evidence.requires_probabilities"]))
    if metric.get("rank_based"):
        implications.append(_implication("P0", "ranking_quality_matters", "Metric is rank-based; ordering may matter more than calibration.", ["metric_evidence.rank_based"]))
    metric_name = str(metric.get("metric_name") or "").lower()
    if target_type in {"binary", "multiclass"} and metric_name in {"accuracy", "categorization_accuracy"} and imbalance.get("severity") in {"moderate", "severe", "extreme"}:
        implications.append(_implication("P1", "accuracy_can_hide_minority_errors", "Accuracy may be insensitive to minority-class performance under imbalance.", ["target_diagnostics.imbalance", "metric_evidence.metric_name"]))
    if target_type == "regression" and distribution.get("heavy_tail"):
        implications.append(_implication("P1", "regression_metric_stability_check", "Heavy-tailed targets can make fold scores unstable.", ["target_diagnostics.distribution", "metric_evidence.metric_name"]))
    return implications


def _validation_implications(
    validation: dict[str, Any],
    distribution: dict[str, Any],
    imbalance: dict[str, Any],
    target_type: str,
    has_time_columns: bool,
    has_group_columns: bool,
) -> list[dict[str, Any]]:
    implications: list[dict[str, Any]] = []
    primary_method = str(_as_dict(validation.get("primary_validation")).get("method") or "").lower()
    if target_type in {"binary", "multiclass"} and not _group_or_time_method(primary_method):
        implications.append(_implication("P0", "stratification_required", "Classification target distribution should be preserved across folds.", ["target_diagnostics.distribution"]))
    if imbalance.get("severity") in {"severe", "extreme"}:
        implications.append(_implication("P0", "fold_class_count_checks_required", "Severe imbalance requires checking every fold has enough minority examples.", ["target_diagnostics.imbalance"]))
    elif imbalance.get("severity") == "moderate":
        implications.append(_implication("P1", "minority_fold_stability_check", "Moderate imbalance can still destabilize fold-level metrics.", ["target_diagnostics.imbalance"]))
    if distribution.get("rare_classes"):
        implications.append(_implication("P1", "rare_class_fold_coverage_check", "Rare classes should appear in every training and validation fold when possible.", ["target_diagnostics.distribution.rare_classes"]))
    if target_type == "regression" and distribution.get("heavy_tail"):
        implications.append(_implication("P1", "fold_score_stability_check", "Regression heavy tails and outliers can make fold scores unstable.", ["target_diagnostics.distribution"]))
    if has_time_columns and not _group_or_time_method(primary_method):
        implications.append(_implication("P2", "temporal_target_pattern_diagnostic", "Date/time target patterns are diagnostic evidence, not automatic primary temporal validation.", ["target_diagnostics.target_by_time"]))
    if has_group_columns and "group" not in primary_method:
        implications.append(_implication("P2", "group_target_pattern_diagnostic", "Group-level target variation should be checked without automatically overriding validation policy.", ["target_diagnostics.target_by_group"]))
    return implications


def _target_by_feature(
    frame: pl.DataFrame,
    target_column: str,
    columns: list[str],
    profiles: dict[str, Any],
    target_type: str,
) -> dict[str, Any]:
    numeric_rows = []
    categorical_rows = []
    for column in columns:
        profile = profiles.get(column)
        if profile is None or column not in frame.columns:
            continue
        if _is_numeric_dtype(profile.dtype):
            numeric_rows.append(_numeric_target_bins(frame, column, target_column, target_type))
        elif _is_categorical_dtype(profile.dtype):
            categorical_rows.append(_categorical_target_summary(frame, column, target_column, profile, target_type))
    return {
        "numeric_binned": _top_by_range(numeric_rows),
        "categorical": _top_by_range(categorical_rows),
    }


def _numeric_target_bins(
    frame: pl.DataFrame,
    column: str,
    target_column: str,
    target_type: str,
) -> dict[str, Any]:
    rows = frame.select([column, target_column]).drop_nulls()
    if rows.height < 3:
        return {"column": column, "bins": [], "target_rate_range": 0.0, "reliability": "caution_small_bins"}
    try:
        binned = rows.with_columns(
            pl.col(column).qcut(4, duplicates="drop").alias("_bin")
        )
    except Exception:
        binned = rows.with_columns(pl.col(column).cast(pl.Utf8, strict=False).alias("_bin"))
    value_name = "target_mean" if target_type == "regression" else "target_rate"
    grouped = (
        binned.group_by("_bin")
        .agg(
            pl.len().alias("count"),
            pl.col(target_column).cast(pl.Float64, strict=False).mean().alias(value_name),
            pl.col(target_column).cast(pl.Float64, strict=False).median().alias("target_median"),
        )
        .sort("_bin")
        .to_dicts()
    )
    values = [float(row.get(value_name) or 0.0) for row in grouped]
    min_count = min((int(row.get("count") or 0) for row in grouped), default=0)
    reliability = "reliable" if min_count >= MIN_RELIABLE_BIN_COUNT else "caution_small_bins"
    return {
        "column": column,
        "bins": [_normalise_row(row) for row in grouped[:6]],
        "target_rate_range": _round(max(values) - min(values) if values else 0.0),
        "target_mean_range": _round(max(values) - min(values) if values else 0.0),
        "min_bin_count": min_count,
        "reliability": reliability,
    }


def _categorical_target_summary(
    frame: pl.DataFrame,
    column: str,
    target_column: str,
    profile: Any,
    target_type: str,
) -> dict[str, Any]:
    rows = frame.select([column, target_column]).drop_nulls()
    if rows.height < 3:
        return {"column": column, "top_categories": [], "target_rate_range": 0.0, "reliability": "caution_sparse_categories"}
    value_name = "target_mean" if target_type == "regression" else "target_rate"
    grouped = (
        rows.group_by(column)
        .agg(
            pl.len().alias("count"),
            pl.col(target_column).cast(pl.Float64, strict=False).mean().alias(value_name),
            pl.col(target_column).cast(pl.Float64, strict=False).median().alias("target_median"),
        )
        .sort("count", descending=True)
        .head(8)
        .to_dicts()
    )
    values = [float(row.get(value_name) or 0.0) for row in grouped]
    cardinality = int(getattr(profile, "n_unique", 0) or 0)
    unique_ratio = float(getattr(profile, "unique_ratio", 0.0) or 0.0)
    min_count = min((int(row.get("count") or 0) for row in grouped), default=0)
    reliability = _categorical_reliability(cardinality, unique_ratio, min_count)
    return {
        "column": column,
        "cardinality": cardinality,
        "unique_ratio": _round(unique_ratio),
        "top_categories": [_normalise_row(row) for row in grouped],
        "target_rate_range": _round(max(values) - min(values) if values else 0.0),
        "target_mean_range": _round(max(values) - min(values) if values else 0.0),
        "min_category_count": min_count,
        "reliability": reliability,
    }


def _target_by_missingness(
    frame: pl.DataFrame,
    target_column: str,
    columns: list[str],
    profiles: dict[str, Any],
    target_type: str,
) -> list[dict[str, Any]]:
    rows = []
    for column in columns[:MAX_MISSINGNESS_COLUMNS]:
        profile = profiles.get(column)
        if profile is None or column not in frame.columns or not getattr(profile, "missing_pct", 0.0):
            continue
        missing = frame.filter(pl.col(column).is_null())
        present = frame.filter(pl.col(column).is_not_null())
        if missing.height == 0 or present.height == 0:
            continue
        if target_type == "regression":
            missing_value = _series_mean(missing[target_column])
            present_value = _series_mean(present[target_column])
            row = {
                "column": column,
                "missing_count": missing.height,
                "missing_pct": _round(missing.height / frame.height if frame.height else 0.0),
                "target_mean_missing": missing_value,
                "target_mean_present": present_value,
                "absolute_difference": _round(abs(missing_value - present_value)),
                "reliability": "reliable" if missing.height >= MIN_RELIABLE_BIN_COUNT else "caution_low_missing_count",
            }
        else:
            missing_value = _series_mean(missing[target_column])
            present_value = _series_mean(present[target_column])
            row = {
                "column": column,
                "missing_count": missing.height,
                "missing_pct": _round(missing.height / frame.height if frame.height else 0.0),
                "target_rate_missing": missing_value,
                "target_rate_present": present_value,
                "absolute_difference": _round(abs(missing_value - present_value)),
                "reliability": "reliable" if missing.height >= MIN_RELIABLE_BIN_COUNT else "caution_low_missing_count",
            }
        rows.append(row)
    return sorted(rows, key=lambda item: item["absolute_difference"], reverse=True)[:10]


def _target_by_group(
    frame: pl.DataFrame,
    target_column: str,
    group_columns: list[str],
    target_type: str,
) -> dict[str, Any]:
    if not group_columns:
        return {"status": "not_available", "reason": "No group columns were inferred.", "columns": []}
    columns = []
    for column in group_columns:
        if column not in frame.columns:
            continue
        value_name = "target_mean"
        grouped = (
            frame.select([column, target_column])
            .drop_nulls()
            .group_by(column)
            .agg(pl.len().alias("count"), pl.col(target_column).cast(pl.Float64, strict=False).mean().alias(value_name))
            .sort("count", descending=True)
            .head(MAX_GROUP_ROWS)
            .to_dicts()
        )
        values = [float(row.get(value_name) or 0.0) for row in grouped]
        columns.append({
            "column": column,
            "groups": [_normalise_row(row) for row in grouped],
            "target_range": _round(max(values) - min(values) if values else 0.0),
            "small_group_count": sum(1 for row in grouped if int(row.get("count") or 0) < 5),
            "target_type": target_type,
        })
    return {"status": "computed" if columns else "not_available", "columns": columns}


def _target_by_time(
    frame: pl.DataFrame,
    target_column: str,
    time_columns: list[str],
    target_type: str,
) -> dict[str, Any]:
    if not time_columns:
        return {"status": "not_available", "reason": "No time/date columns were inferred.", "columns": []}
    columns = []
    for column in time_columns:
        if column not in frame.columns:
            continue
        period = _period_expr(column)
        try:
            period_frame = frame.select([period.alias("_period"), pl.col(target_column)]).drop_nulls()
            grouped = (
                period_frame.group_by("_period")
                .agg(pl.len().alias("count"), pl.col(target_column).cast(pl.Float64, strict=False).mean().alias("target_mean"))
                .sort("_period")
                .to_dicts()
            )
        except Exception:
            grouped = []
        values = [float(row.get("target_mean") or 0.0) for row in grouped]
        shift = max(values) - min(values) if values else 0.0
        columns.append({
            "column": column,
            "periods": [_normalise_row(row) for row in grouped[:12]],
            "target_shift": _round(shift),
            "temporal_target_shift_severity": _shift_severity(shift),
            "target_type": target_type,
        })
    return {"status": "computed" if columns else "not_available", "columns": columns}


def _suspicious_patterns(
    distribution: dict[str, Any],
    imbalance: dict[str, Any],
    target_by_feature: dict[str, Any],
    target_column: str,
    target_type: str,
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    missing = int(distribution.get("missing_target_count") or 0)
    n_rows = int(distribution.get("n_rows") or 0)
    if missing:
        patterns.append(_pattern("medium" if missing / max(n_rows, 1) < 0.1 else "high", "missing_target_values", f"Target column {target_column} has missing values.", ["target_diagnostics.distribution"]))
    if target_type in {"binary", "multiclass", "ranking"}:
        if distribution.get("n_classes") in {0, 1}:
            patterns.append(_pattern("high", "constant_or_single_class_target", "Target has fewer than two non-null classes.", ["target_diagnostics.distribution"]))
        if imbalance.get("severity") in {"severe", "extreme"}:
            patterns.append(_pattern("medium", "very_rare_target_class", "At least one target class has very low support.", ["target_diagnostics.imbalance"]))
        for row in _as_dict(target_by_feature).get("categorical", []):
            if row.get("reliability") in {"caution_high_cardinality", "caution_sparse_categories"} and float(row.get("target_rate_range") or 0.0) >= 0.5:
                patterns.append(_pattern("low", "target_association_may_be_id_artifact", "A sparse or high-cardinality categorical appears target-associated; treat as leakage/id diagnostic, not proof.", ["target_diagnostics.target_by_feature.categorical"]))
                break
    if target_type == "regression":
        if distribution.get("heavy_tail"):
            patterns.append(_pattern("medium", "heavy_tail_or_extreme_outliers", "Regression target has heavy-tail or outlier evidence.", ["target_diagnostics.distribution"]))
        if distribution.get("std") == 0:
            patterns.append(_pattern("high", "constant_regression_target", "Regression target is constant or near-constant.", ["target_diagnostics.distribution"]))
        if float(distribution.get("zero_rate") or 0.0) >= 0.5:
            patterns.append(_pattern("low", "many_zero_targets", "Target has many zeros; check metric sensitivity before transforming.", ["target_diagnostics.distribution"]))
    return patterns


def _recommended_actions(
    distribution: dict[str, Any],
    imbalance: dict[str, Any],
    metric_implications: list[dict[str, Any]],
    validation_implications: list[dict[str, Any]],
    target_by_missingness: list[dict[str, Any]],
    suspicious_patterns: list[dict[str, Any]],
    target_type: str,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if target_type in {"binary", "multiclass"}:
        actions.append(_action("P0", "Preserve target distribution in validation folds.", "Classification target distribution must remain stable across folds.", ["target_diagnostics.distribution"], "medium", ["validation", "target"]))
    if imbalance.get("severity") in {"moderate", "severe", "extreme"}:
        actions.append(_action("P1", "Track minority-class performance in validation diagnostics.", "Target imbalance can make aggregate metrics misleading.", ["target_diagnostics.imbalance"], "medium", ["metric", "validation", "target"]))
    if any(item.get("implication") == "fold_class_count_checks_required" for item in validation_implications):
        actions.append(_action("P0", "Check per-fold class counts before trusting validation scores.", "Severe imbalance requires every fold to contain enough minority examples.", ["target_diagnostics.imbalance"], "high", ["validation", "target"]))
    if any(float(item.get("absolute_difference") or 0.0) >= 0.2 for item in target_by_missingness):
        actions.append(_action("P1", "Evaluate missingness indicators for columns whose missingness changes target behavior.", "Missingness is associated with target changes.", ["target_diagnostics.target_by_missingness"], "low", ["feature_engineering", "target"]))
    if target_type == "regression" and distribution.get("heavy_tail"):
        actions.append(_action("P1", "Test target transforms or robust losses as validation hypotheses.", "Regression target has heavy-tail or outlier evidence; transforms are hypotheses to validate.", ["target_diagnostics.distribution"], "medium", ["metric", "validation", "target"]))
    if suspicious_patterns:
        actions.append(_action("P1", "Audit suspicious target patterns before trusting feature gains.", "Target diagnostics found potential label artifacts or leakage-style signals.", ["target_diagnostics.suspicious_patterns"], "medium", ["leakage", "target"]))
    for implication in metric_implications:
        if implication.get("implication") == "threshold_tuning_required":
            actions.append(_action("P0", "Tune thresholds only inside validation folds.", "Metric evidence requires thresholded outputs.", ["metric_evidence.requires_threshold"], "medium", ["validation", "metric", "target"]))
    return _dedupe_actions(actions)


def _regression_transform_hints(heavy_tail: bool, skew_proxy: float, outlier_rate: float) -> list[dict[str, Any]]:
    hints = []
    if heavy_tail or skew_proxy >= 1.0:
        hints.append({
            "hint": "test_log_or_boxcox_like_transform",
            "why": "Target shape suggests a transform may help, but this is a validation hypothesis only.",
            "evidence_refs": ["target_diagnostics.distribution"],
        })
    if outlier_rate >= 0.02:
        hints.append({
            "hint": "test_robust_loss_or_clipping",
            "why": "Outlier-heavy targets can destabilize regression losses.",
            "evidence_refs": ["target_diagnostics.distribution"],
        })
    return hints


def _not_testable(
    reason: str,
    *,
    target_column: str | None,
    task_type: str,
    metric_name: str,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "not_testable",
        "target_column": target_column,
        "task_type": task_type,
        "metric_name": metric_name,
        "reason": reason,
        "distribution": {},
        "imbalance": {},
        "metric_implications": [],
        "validation_implications": [],
        "target_by_feature": {"numeric_binned": [], "categorical": []},
        "target_by_missingness": [],
        "target_by_group": {"status": "not_available", "columns": []},
        "target_by_time": {"status": "not_available", "columns": []},
        "suspicious_patterns": [],
        "recommended_actions": [],
        "limitations": _unique([*(limitations or []), reason]),
        "warnings": _unique(warnings or []),
    }


def _target_type(task_type: str, metric: dict[str, Any], target: pl.Series) -> str:
    lowered = task_type.lower()
    if "ranking" in lowered or metric.get("requires_query_groups"):
        return "ranking"
    if "regression" in lowered or metric.get("metric_family") == "regression_error":
        return "regression"
    unique_count = target.drop_nulls().n_unique()
    if unique_count == 2:
        return "binary"
    if unique_count > 2 and (not _is_numeric_series(target) or unique_count <= 50):
        return "multiclass"
    if _is_numeric_series(target):
        return "regression"
    return "multiclass"


def _selected_feature_columns(columns: list[str], profiles: dict[str, Any]) -> list[str]:
    scored = []
    for column in columns:
        profile = profiles.get(column)
        if profile is None:
            continue
        missing_pct = float(getattr(profile, "missing_pct", 0.0) or 0.0)
        unique_ratio = float(getattr(profile, "unique_ratio", 0.0) or 0.0)
        scored.append((missing_pct >= 0.01, -unique_ratio, column))
    return [column for _, _, column in sorted(scored, reverse=True)[:MAX_FEATURE_COLUMNS]]


def _safe_read(
    reader: DatasetReader,
    table: str,
    columns: list[str],
    max_rows: int,
    warnings: list[str],
    limitations: list[str],
) -> pl.DataFrame | None:
    try:
        schema = {column["name"] for column in reader.read_schema(table)}
        available = [column for column in columns if column in schema]
        if not available:
            return None
        return reader.read_columns(table, columns=available, n_rows=max_rows)
    except ReaderError as exc:
        warnings.append(str(exc))
        limitations.append(f"Could not read target diagnostic columns from {table}.")
        return None


def _value_counts(series: pl.Series) -> list[dict[str, Any]]:
    rows = series.value_counts(sort=True).to_dicts()
    count_key = "count"
    value_key = series.name
    return [
        {"class": str(row.get(value_key)), "count": int(row.get(count_key) or 0)}
        for row in rows
    ]


def _numeric_values(series: pl.Series) -> list[float]:
    values = []
    for value in series.to_list():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == numeric:
            values.append(numeric)
    return values


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * q)))
    return sorted_values[index]


def _series_mean(series: pl.Series) -> float:
    values = _numeric_values(series)
    return _round(sum(values) / len(values)) if values else 0.0


def _imbalance_severity(minority_pct: float) -> str:
    if minority_pct >= 0.35:
        return "none"
    if minority_pct >= 0.20:
        return "mild"
    if minority_pct >= 0.10:
        return "moderate"
    if minority_pct >= 0.01:
        return "severe"
    return "extreme"


def _categorical_reliability(cardinality: int, unique_ratio: float, min_count: int) -> str:
    if unique_ratio >= HIGH_CARDINALITY_RATIO:
        return "caution_high_cardinality"
    if min_count < MIN_RELIABLE_BIN_COUNT:
        return "caution_sparse_categories"
    return "reliable"


def _shift_severity(value: float) -> str:
    if value >= 0.30:
        return "high"
    if value >= 0.15:
        return "medium"
    if value > 0:
        return "low"
    return "none"


def _implication(priority: str, implication: str, why: str, evidence_refs: list[str]) -> dict[str, Any]:
    return {
        "implication": implication,
        "why": why,
        "evidence_refs": evidence_refs,
        "priority": priority,
    }


def _pattern(severity: str, pattern: str, why: str, evidence_refs: list[str]) -> dict[str, Any]:
    return {
        "severity": severity,
        "pattern": pattern,
        "why": why,
        "evidence_refs": evidence_refs,
        "limitations": ["Pattern is a lightweight diagnostic and should be verified before acting."],
    }


def _action(priority: str, action: str, why: str, evidence_refs: list[str], risk: str, applies_to: list[str]) -> dict[str, Any]:
    return {
        "priority": priority,
        "action": action,
        "why": why,
        "evidence_refs": evidence_refs,
        "risk": risk,
        "applies_to": applies_to,
        "source_categories": ["target_diagnostics"],
    }


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result = []
    for action in actions:
        key = (str(action.get("action")), tuple(action.get("evidence_refs") or []))
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _period_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False).str.slice(0, 7)


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _normalise(value) for key, value in row.items()}


def _normalise(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float):
        return _round(value)
    return value


def _top_by_range(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            float(item.get("target_rate_range") or item.get("target_mean_range") or 0.0),
            str(item.get("column") or ""),
        ),
        reverse=True,
    )[:8]


def _unsupported_task_type(task_type: str, metric: dict[str, Any]) -> bool:
    lowered = task_type.lower()
    metric_family = str(metric.get("metric_family") or "").lower()
    return any(token in lowered for token in ("survival", "time_to_event")) or "survival" in metric_family


def _group_or_time_method(method: str) -> bool:
    return any(token in method for token in ("group", "temporal", "time", "expanding", "oot"))


def _is_numeric_dtype(dtype: Any) -> bool:
    return any(token in str(dtype).lower() for token in ("int", "float", "decimal"))


def _is_categorical_dtype(dtype: Any) -> bool:
    return any(token in str(dtype).lower() for token in ("str", "utf8", "categorical", "bool"))


def _is_numeric_series(series: pl.Series) -> bool:
    return _is_numeric_dtype(series.dtype)


def _base_profile(inferred_schema: InferredSchema, table_profiles: list[TableProfile]) -> TableProfile | None:
    if inferred_schema.train_base_table:
        for profile in table_profiles:
            if profile.path == inferred_schema.train_base_table:
                return profile
    return table_profiles[0] if table_profiles else None


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _round(value: float) -> float:
    return round(float(value), 6)


__all__ = ["diagnose_target"]
