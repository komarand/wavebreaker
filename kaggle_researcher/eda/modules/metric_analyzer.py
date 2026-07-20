from __future__ import annotations

from typing import Any

from kaggle_researcher.eda.metrics import (
    MetricFamily,
    TaskType,
    infer_metric_spec,
    normalize_metric_name,
)
from kaggle_researcher.eda.schemas import (
    EdaTaskPlan,
    InferredSchema,
    MetricEvidence,
    TableProfile,
)


def analyze_metric(
    task_plan: EdaTaskPlan,
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
) -> MetricEvidence:
    """Resolve metric requirements through the generic metric registry."""

    metric_name = _metric_name(task_plan)
    task_type = _task_type(task_plan.task_type)
    spec = infer_metric_spec(metric_name, task_type)
    warnings: list[str] = []
    components: dict[str, Any] = {}
    required_columns = _required_columns(spec, inferred_schema, table_profiles)

    if spec.needs_custom_implementation or spec.family in {
        MetricFamily.UNKNOWN,
        MetricFamily.CUSTOM,
    }:
        warnings.append(
            f"Metric '{spec.name}' requires manual metric implementation or verification."
        )

    base_metric: str | None = None
    requires_time_or_groups = bool(
        spec.requires_time
        or spec.requires_groups
        or spec.requires_query_groups
    )
    if spec.name == "gini_stability":
        base_metric = "gini"
        requires_time_or_groups = True
        components = {
            "weekly_gini": {
                "available": True,
                "helper": "kaggle_researcher.eda.metrics.gini_stability.weekly_gini",
            },
            "trend_penalty": {"available": True},
            "residual_std_penalty": {"available": True},
        }
        if required_columns.get("time") is None:
            warnings.append("gini_stability requires a time/period column, but none was found.")

    return MetricEvidence(
        metric_name=spec.name,
        normalized_metric_name=normalize_metric_name(spec.name),
        task_type=task_type.value,
        metric_family=spec.family.value,
        base_metric=base_metric,
        greater_is_better=spec.greater_is_better,
        requires_probabilities=spec.requires_probabilities,
        requires_threshold=spec.requires_threshold,
        requires_calibration=spec.requires_calibration,
        requires_groups=spec.requires_groups,
        requires_time=spec.requires_time,
        requires_query_groups=spec.requires_query_groups,
        rank_based=spec.rank_based,
        requires_time_or_groups=requires_time_or_groups,
        local_metric_available=spec.supports_local_eval,
        needs_custom_implementation=spec.needs_custom_implementation,
        threshold_search_needed=spec.requires_threshold,
        prediction_output_type=_prediction_output_type(spec),
        components=components,
        required_columns=required_columns,
        warnings=warnings,
    )


def _metric_name(task_plan: EdaTaskPlan) -> str | None:
    name = (task_plan.metric or {}).get("name")
    return str(name) if name is not None else None


def _task_type(task_type: str | None) -> TaskType:
    if task_type is None:
        return TaskType.UNKNOWN
    normalized = str(task_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "time_series":
        normalized = "forecasting_tabular"
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.UNKNOWN


def _prediction_output_type(spec: Any) -> str:
    if spec.requires_probabilities:
        return "probability"
    if spec.requires_threshold:
        return "label"
    if spec.family in {MetricFamily.REGRESSION_ERROR, MetricFamily.SURVIVAL}:
        return "score"
    if spec.family == MetricFamily.RANKING:
        return "ranked_score"
    return "unknown"


def _required_columns(
    spec: Any,
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
) -> dict[str, Any]:
    required: dict[str, Any] = {}
    if inferred_schema.target_column is not None:
        required["target"] = inferred_schema.target_column
    if spec.requires_probabilities or spec.requires_threshold or spec.family in {
        MetricFamily.REGRESSION_ERROR,
        MetricFamily.RANK_CLASSIFICATION,
        MetricFamily.RANKING,
        MetricFamily.SURVIVAL,
        MetricFamily.TEMPORAL_STABILITY,
    }:
        required["prediction"] = inferred_schema.prediction_column or _generic_prediction_name(spec)
    if spec.requires_groups or spec.requires_query_groups:
        required["group"] = _find_group_column(inferred_schema, table_profiles)
    if spec.requires_time:
        required["time"] = _find_time_column(inferred_schema, table_profiles)
    return required


def _generic_prediction_name(spec: Any) -> str:
    if spec.requires_probabilities:
        return "probability"
    if spec.requires_threshold:
        return "predicted_label"
    return "prediction"


def _find_time_column(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
) -> str | None:
    if inferred_schema.candidate_time_columns:
        return inferred_schema.candidate_time_columns[0]
    if inferred_schema.candidate_date_columns:
        return inferred_schema.candidate_date_columns[0]
    for column_name in _profile_column_names(table_profiles):
        normalized = column_name.strip().lower()
        if any(token in normalized for token in ("week", "period", "month", "date", "time")):
            return column_name
    return None


def _find_group_column(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
) -> str | None:
    if inferred_schema.candidate_group_columns:
        return inferred_schema.candidate_group_columns[0]
    for key in inferred_schema.global_roles.get("candidate_join_keys", []):
        normalized = str(key).lower()
        if any(token in normalized for token in ("query", "group", "session", "user")):
            return str(key)
    for column_name in _profile_column_names(table_profiles):
        normalized = column_name.strip().lower()
        if any(token in normalized for token in ("query", "group", "session")):
            return column_name
    return None


def _profile_column_names(table_profiles: list[TableProfile]) -> list[str]:
    return [
        column.name
        for profile in table_profiles
        for column in profile.columns
    ]


__all__ = ["analyze_metric"]
