from __future__ import annotations

from typing import Any

from kaggle_researcher.eda.modules.column_policy import ColumnRolePolicy
from kaggle_researcher.eda.schemas import (
    InferredSchema,
    LeakageCheckResult,
    TableProfile,
)


FEATURE_FAMILIES = (
    "base_numeric_features",
    "base_categorical_features",
    "missingness_indicators",
    "date_features",
    "secondary_table_aggregations",
    "high_cardinality_encoding",
    "naive_target_encoding_or_woe",
    "oof_target_encoding_or_woe",
    "monotonic_or_binning_features",
    "ranking_group_features",
    "regression_target_transform",
)
NUMERIC_DTYPE_TOKENS = ("int", "float", "decimal")
CATEGORICAL_DTYPE_TOKENS = ("str", "utf8", "categorical", "bool")
GROUP_TOKENS = ("query", "group", "customer", "client", "user", "session")
DATE_TOKENS = ("date", "time", "timestamp", "week", "month", "period")


def probe_feature_families(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    relationship_evidence: dict,
    leakage_evidence: list[LeakageCheckResult],
    baseline_evidence: dict,
    metric_evidence: dict | None = None,
    validation_evidence: dict | None = None,
) -> list[dict[str, Any]]:
    """Assess feature-family potential from existing EDA evidence only."""

    metric = _as_dict(metric_evidence)
    base_profile = _base_profile(inferred_schema, table_profiles)
    secondary_profiles = _secondary_profiles(inferred_schema, table_profiles)
    excluded_columns = _excluded_columns(inferred_schema, leakage_evidence, metric)

    probes = [
        _base_numeric_features(base_profile, excluded_columns),
        _base_categorical_features(base_profile, excluded_columns),
        _missingness_indicators(base_profile, excluded_columns),
        _date_features(inferred_schema, base_profile),
        _secondary_table_aggregations(secondary_profiles, relationship_evidence),
        _high_cardinality_encoding(base_profile, excluded_columns),
        _naive_target_encoding_or_woe(base_profile, excluded_columns),
        _oof_target_encoding_or_woe(
            base_profile,
            excluded_columns,
            baseline_evidence,
            _as_dict(validation_evidence),
        ),
        _monotonic_or_binning_features(base_profile, excluded_columns, metric),
        _ranking_group_features(inferred_schema, table_profiles, metric),
        _regression_target_transform(inferred_schema, base_profile, metric),
    ]
    return [_normalise_probe(probe) for probe in probes]


def _base_numeric_features(
    base_profile: TableProfile | None,
    excluded_columns: set[str],
) -> dict[str, Any]:
    columns = _feature_columns(base_profile, excluded_columns, kind="numeric")
    if not columns:
        return _probe(
            "base_numeric_features",
            "not_testable",
            "low",
            {"numeric_columns": []},
            "No safe numeric base columns were found.",
        )
    return _probe(
        "base_numeric_features",
        "high_potential" if len(columns) >= 3 else "medium_potential",
        "low",
        {"numeric_columns": columns, "n_columns": len(columns)},
        "Use safe numeric base columns as the first feature block.",
    )


def _base_categorical_features(
    base_profile: TableProfile | None,
    excluded_columns: set[str],
) -> dict[str, Any]:
    columns = _feature_columns(base_profile, excluded_columns, kind="categorical")
    if not columns:
        return _probe(
            "base_categorical_features",
            "not_testable",
            "low",
            {"categorical_columns": []},
            "No safe categorical base columns were found.",
        )
    return _probe(
        "base_categorical_features",
        "medium_potential",
        "low",
        {"categorical_columns": columns, "n_columns": len(columns)},
        "Use fold-fitted categorical encoders; avoid target leakage.",
    )


def _missingness_indicators(
    base_profile: TableProfile | None,
    excluded_columns: set[str],
) -> dict[str, Any]:
    if base_profile is None:
        return _probe(
            "missingness_indicators",
            "not_testable",
            "low",
            {"columns": []},
            "Base table profile is unavailable.",
        )
    columns = [
        column.name
        for column in base_profile.columns
        if column.name not in excluded_columns and (column.missing_count or 0) > 0
    ]
    if not columns:
        return _probe(
            "missingness_indicators",
            "low_potential",
            "low",
            {"columns": []},
            "Missingness indicators are not a priority because no missing base columns were found.",
        )
    return _probe(
        "missingness_indicators",
        "medium_potential",
        "low",
        {"columns": columns, "n_columns": len(columns)},
        "Add simple missingness indicators for columns with observed missing values.",
    )


def _date_features(
    inferred_schema: InferredSchema,
    base_profile: TableProfile | None,
) -> dict[str, Any]:
    columns = _unique([
        *inferred_schema.candidate_time_columns,
        *inferred_schema.candidate_date_columns,
        *[
            column.name
            for column in (base_profile.columns if base_profile is not None else [])
            if column.date_min is not None
            or any(token in column.name.lower() for token in DATE_TOKENS)
        ],
    ])
    if not columns:
        return _probe(
            "date_features",
            "not_testable",
            "low",
            {"columns": []},
            "No time/date columns were found.",
        )
    return _probe(
        "date_features",
        "medium_potential",
        "medium",
        {"columns": columns},
        "Derive fold-safe date parts and recency features without peeking into future rows.",
    )


def _secondary_table_aggregations(
    secondary_profiles: list[TableProfile],
    relationship_evidence: dict,
) -> dict[str, Any]:
    secondary_tables = [profile.path for profile in secondary_profiles]
    relationships = _as_dict(relationship_evidence).get("relationships", [])
    if not secondary_tables:
        return _probe(
            "secondary_table_aggregations",
            "not_testable",
            "low",
            {"secondary_tables": []},
            "No secondary train/test tables were profiled.",
        )
    if not relationships:
        return _probe(
            "secondary_table_aggregations",
            "not_testable",
            "medium",
            {"secondary_tables": secondary_tables},
            "Run relationship inference before designing secondary-table aggregations.",
        )

    usable = [
        item
        for item in relationships
        if item.get("selected_join_key") and item.get("relationship_type") != "unknown"
    ]
    if not usable:
        return _probe(
            "secondary_table_aggregations",
            "not_testable",
            "medium",
            {"secondary_tables": secondary_tables, "relationships": relationships},
            "No usable join relationships were inferred for secondary aggregations.",
        )

    requires_aggregation = any(item.get("requires_aggregation") for item in usable)
    high_risk = any(item.get("row_multiplication_risk") == "high" for item in usable)
    return _probe(
        "secondary_table_aggregations",
        "high_potential" if requires_aggregation or high_risk else "medium_potential",
        "medium" if requires_aggregation else "low",
        {
            "secondary_tables": secondary_tables,
            "usable_relationships": [
                {
                    "table": item.get("table"),
                    "join_key": item.get("selected_join_key"),
                    "relationship_type": item.get("relationship_type"),
                    "requires_aggregation": item.get("requires_aggregation"),
                }
                for item in usable
            ],
        },
        "Aggregate secondary rows by the inferred join key before joining to the base table.",
    )


def _high_cardinality_encoding(
    base_profile: TableProfile | None,
    excluded_columns: set[str],
) -> dict[str, Any]:
    if base_profile is None:
        columns: list[str] = []
    else:
        columns = [
            column for column in base_profile.high_cardinality_columns if column not in excluded_columns
        ]
    if not columns:
        return _probe(
            "high_cardinality_encoding",
            "low_potential",
            "low",
            {"columns": []},
            "No high-cardinality safe base columns were detected.",
        )
    return _probe(
        "high_cardinality_encoding",
        "medium_potential",
        "medium",
        {"columns": columns},
        "Use leakage-aware encoders for high-cardinality columns, fitted inside validation folds.",
    )


def _naive_target_encoding_or_woe(
    base_profile: TableProfile | None,
    excluded_columns: set[str],
) -> dict[str, Any]:
    categorical_columns = _feature_columns(base_profile, excluded_columns, kind="categorical")
    if not categorical_columns:
        return _probe(
            "naive_target_encoding_or_woe",
            "not_testable",
            "high",
            {"categorical_columns": []},
            "No categorical columns were found for naive target encoding or WoE.",
        )
    return _probe(
        "naive_target_encoding_or_woe",
        "unsafe",
        "high",
        {"categorical_columns": categorical_columns, "safe_policy": False},
        "Do not use naive target encoding or WoE; global target statistics leak labels across validation folds.",
    )


def _oof_target_encoding_or_woe(
    base_profile: TableProfile | None,
    excluded_columns: set[str],
    baseline_evidence: dict,
    validation_evidence: dict,
) -> dict[str, Any]:
    categorical_columns = _feature_columns(base_profile, excluded_columns, kind="categorical")
    safe_oof_policy = _has_oof_safe_policy(baseline_evidence, validation_evidence)
    if not categorical_columns:
        return _probe(
            "oof_target_encoding_or_woe",
            "not_testable",
            "medium",
            {"categorical_columns": [], "safe_policy": False},
            "No categorical columns were found for out-of-fold target encoding or WoE.",
        )
    if safe_oof_policy:
        return _probe(
            "oof_target_encoding_or_woe",
            "medium_potential",
            "medium",
            {"categorical_columns": categorical_columns, "safe_policy": True},
            "Test only out-of-fold target encoding aligned to the selected validation policy.",
        )
    return _probe(
        "oof_target_encoding_or_woe",
        "not_testable",
        "high",
        {"categorical_columns": categorical_columns, "safe_policy": False},
        "A supported validation policy is required before testing fold-fitted target encoding.",
    )


def _monotonic_or_binning_features(
    base_profile: TableProfile | None,
    excluded_columns: set[str],
    metric: dict[str, Any],
) -> dict[str, Any]:
    numeric_columns = _feature_columns(base_profile, excluded_columns, kind="numeric")
    if not numeric_columns:
        return _probe(
            "monotonic_or_binning_features",
            "not_testable",
            "low",
            {"numeric_columns": []},
            "No numeric columns are available for binning or monotonic transforms.",
        )
    task_type = str(metric.get("task_type") or "")
    status = "medium_potential" if task_type in {"binary_classification", "regression"} else "low_potential"
    return _probe(
        "monotonic_or_binning_features",
        status,
        "low",
        {"numeric_columns": numeric_columns, "task_type": task_type or "unknown"},
        "Try simple bins or monotonic transforms only when validation confirms benefit.",
    )


def _ranking_group_features(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    metric: dict[str, Any],
) -> dict[str, Any]:
    task_type = str(metric.get("task_type") or "")
    metric_family = str(metric.get("metric_family") or "")
    ranking_like = task_type == "ranking" or metric_family == "ranking" or metric.get("requires_query_groups")
    if not ranking_like:
        return _probe(
            "ranking_group_features",
            "not_testable",
            "low",
            {"task_type": task_type or "unknown"},
            "Ranking group features are relevant only for ranking/query-group tasks.",
        )
    group_columns = _ranking_group_columns(inferred_schema, table_profiles)
    return _probe(
        "ranking_group_features",
        "high_potential" if group_columns else "not_testable",
        "medium" if group_columns else "low",
        {"group_columns": group_columns},
        (
            "Build query/group-level counts and within-group ranks using fold-safe transformations."
            if group_columns
            else "No query/group columns were found for ranking features."
        ),
    )


def _regression_target_transform(
    inferred_schema: InferredSchema,
    base_profile: TableProfile | None,
    metric: dict[str, Any],
) -> dict[str, Any]:
    task_type = str(metric.get("task_type") or "")
    metric_name = str(metric.get("metric_name") or "").lower()
    metric_family = str(metric.get("metric_family") or "")
    regression_like = task_type == "regression" or metric_family == "regression_error"
    if not regression_like:
        return _probe(
            "regression_target_transform",
            "not_testable",
            "low",
            {"task_type": task_type or "unknown"},
            "Target transforms are relevant only for regression metrics.",
        )
    target_profile = _target_profile(inferred_schema, base_profile)
    skew = _skew_proxy(target_profile)
    if target_profile is not None and skew >= 4.0:
        return _probe(
            "regression_target_transform",
            "high_potential" if metric_name in {"rmsle", "rmse", "mae"} else "medium_potential",
            "low",
            {
                "target_column": target_profile.name,
                "skew_proxy": skew,
                "metric_name": metric_name,
            },
            "Evaluate log/Box-Cox-style target transforms inside validation; invert predictions before scoring.",
        )
    return _probe(
        "regression_target_transform",
        "low_potential",
        "low",
        {
            "target_column": target_profile.name if target_profile is not None else None,
            "skew_proxy": skew,
            "metric_name": metric_name,
        },
        "Target transform is not a priority because target skew evidence is weak.",
    )


def _normalise_probe(probe: dict[str, Any]) -> dict[str, Any]:
    if probe["feature_family"] not in FEATURE_FAMILIES:
        raise ValueError(f"Unknown feature family: {probe['feature_family']}")
    return probe


def _probe(
    feature_family: str,
    status: str,
    leakage_risk: str,
    evidence: dict[str, Any],
    recommendation: str,
) -> dict[str, Any]:
    return {
        "feature_family": feature_family,
        "status": status,
        "leakage_risk": leakage_risk,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _base_profile(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
) -> TableProfile | None:
    if inferred_schema.train_base_table is not None:
        for profile in table_profiles:
            if profile.path == inferred_schema.train_base_table:
                return profile
    for profile in table_profiles:
        if profile.path.lower() in {"train.csv", "train_base.csv"}:
            return profile
    return table_profiles[0] if table_profiles else None


def _secondary_profiles(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
) -> list[TableProfile]:
    base_path = inferred_schema.train_base_table
    secondary = []
    for profile in table_profiles:
        path = profile.path.lower()
        if profile.path == base_path:
            continue
        if path.startswith(("train_", "test_")) and "base" not in path:
            secondary.append(profile)
    return secondary


def _feature_columns(
    base_profile: TableProfile | None,
    excluded_columns: set[str],
    *,
    kind: str,
) -> list[str]:
    if base_profile is None:
        return []
    columns = []
    for column in base_profile.columns:
        if column.name in excluded_columns:
            continue
        if kind == "numeric" and _is_numeric_dtype(column.dtype):
            columns.append(column.name)
        elif kind == "categorical" and _is_categorical_dtype(column.dtype):
            columns.append(column.name)
    return columns


def _excluded_columns(
    inferred_schema: InferredSchema,
    leakage_evidence: list[LeakageCheckResult],
    metric: dict[str, Any],
) -> set[str]:
    policy = ColumnRolePolicy(inferred_schema, metric)
    base_table = inferred_schema.train_base_table
    excluded = {
        column
        for column in (
            inferred_schema.target_column,
            inferred_schema.primary_id_column,
            inferred_schema.prediction_column,
        )
        if column is not None
    }
    for table in inferred_schema.tables:
        if base_table is not None and table.path != base_table:
            continue
        for role in table.column_roles:
            reason = policy.exclusion_reason(role.name, table=table.path, context="model_feature")
            if reason is not None:
                excluded.add(role.name)
    for item in leakage_evidence:
        payload = _as_dict(item)
        if payload.get("severity") != "critical":
            continue
        evidence = _as_dict(payload.get("evidence"))
        for key in ("target_column", "id_column", "column"):
            if evidence.get(key) is not None:
                excluded.add(str(evidence[key]))
        for column_info in evidence.get("columns", []):
            if isinstance(column_info, dict):
                for key in ("column", "name"):
                    if column_info.get(key) is not None:
                        excluded.add(str(column_info[key]))
            elif column_info is not None:
                excluded.add(str(column_info))
    return excluded


def _target_profile(
    inferred_schema: InferredSchema,
    base_profile: TableProfile | None,
) -> Any | None:
    if base_profile is None or inferred_schema.target_column is None:
        return None
    for column in base_profile.columns:
        if column.name == inferred_schema.target_column:
            return column
    return None


def _skew_proxy(column: Any | None) -> float:
    if column is None or column.q50 in (None, 0) or column.max is None:
        return 0.0
    try:
        return round(abs(float(column.max)) / max(abs(float(column.q50)), 1e-9), 6)
    except (TypeError, ValueError):
        return 0.0


def _has_oof_safe_policy(
    baseline_evidence: dict,
    validation_evidence: dict,
) -> bool:
    evidence = _as_dict(baseline_evidence)
    policy = _as_dict(validation_evidence.get("primary_validation"))
    if not policy:
        policy = _as_dict(evidence.get("validation_policy"))
    method = str(policy.get("method") or "").lower()
    return method in {
        "kfold",
        "stratified_kfold",
        "group_kfold",
        "stratified_group_kfold",
        "temporal_holdout",
        "temporal_cv",
        "expanding_window",
        "ranking_group_cv",
    }


def _ranking_group_columns(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
) -> list[str]:
    candidates = [
        *inferred_schema.candidate_group_columns,
        *[
            str(key)
            for key in inferred_schema.global_roles.get("candidate_join_keys", [])
            if any(token in str(key).lower() for token in GROUP_TOKENS)
        ],
    ]
    for profile in table_profiles:
        for column in profile.columns:
            if any(token in column.name.lower() for token in GROUP_TOKENS):
                candidates.append(column.name)
    return _unique(candidates)


def _is_numeric_dtype(dtype: str) -> bool:
    lowered = str(dtype).lower()
    return any(token in lowered for token in NUMERIC_DTYPE_TOKENS)


def _is_categorical_dtype(dtype: str) -> bool:
    lowered = str(dtype).lower()
    return any(token in lowered for token in CATEGORICAL_DTYPE_TOKENS)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["probe_feature_families"]
