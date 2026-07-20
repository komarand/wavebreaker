from __future__ import annotations

from typing import Any

from kaggle_researcher.eda.metrics import MetricFamily, MetricSpec, TaskType
from kaggle_researcher.eda.schemas import InferredSchema, TableProfile
from kaggle_researcher.eda.validation.split_helpers import (
    infer_candidate_group_columns,
    infer_candidate_time_columns,
)


class ValidationPolicySelector:
    def select(
        self,
        *,
        task_type: TaskType | str | None,
        metric_spec: MetricSpec,
        inferred_schema: InferredSchema,
        table_profiles: list[TableProfile] | None = None,
        validation_signals: dict[str, Any] | None = None,
        scout_hypotheses: list[Any] | None = None,
    ) -> dict[str, Any]:
        return select_validation_policy(
            task_type=task_type,
            metric_spec=metric_spec,
            inferred_schema=inferred_schema,
            table_profiles=table_profiles,
            validation_signals=validation_signals,
            scout_hypotheses=scout_hypotheses,
        )


def select_validation_policy(
    *,
    task_type: TaskType | str | None,
    metric_spec: MetricSpec,
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile] | None = None,
    validation_signals: dict[str, Any] | None = None,
    scout_hypotheses: list[Any] | None = None,
) -> dict[str, Any]:
    profiles = table_profiles or []
    signals = validation_signals or {}
    resolved_task_type = _coerce_task_type(task_type)
    time_candidates = infer_candidate_time_columns(inferred_schema, profiles)
    group_candidates = infer_candidate_group_columns(inferred_schema, profiles)
    time_column = time_candidates[0]["name"] if time_candidates else None
    group_column = _preferred_group_column(group_candidates)

    diagnostic_validations: list[dict[str, Any]] = []
    rejected_validations: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []

    if time_column is not None:
        evidence_refs.append("inferred_schema.candidate_time_columns")
        diagnostic_validations.append(
            {
                "method": "temporal_holdout",
                "split_column": time_column,
                "reason": (
                    "Time-like columns are diagnostic unless task, metric, "
                    "or signals require temporal validation."
                ),
            }
        )
        rejected_validations.append(
            {
                "method": "temporal_holdout_as_default",
                "reason": "A time column alone is not sufficient for primary temporal validation.",
            }
        )

    if _requires_custom_policy(metric_spec):
        warnings.append("Metric requires custom validation review before choosing folds.")
        return _decision(
            primary_validation={
                "method": "custom_required",
                "reason": "Unknown/custom metric needs manual validation design.",
            },
            diagnostic_validations=diagnostic_validations,
            rejected_validations=rejected_validations,
            confidence="low",
            evidence_refs=evidence_refs,
            warnings=warnings,
            limitations=limitations,
            reasoning_summary="Selected custom_required because the metric is unknown or custom.",
        )

    if _requires_temporal_policy(
        resolved_task_type,
        metric_spec,
        signals,
        scout_hypotheses,
    ):
        if time_column is None:
            warnings.append(
                "Temporal validation is required, but no time/date column was inferred."
            )
            limitation = "Temporal split cannot be constructed without a period/date column."
            return _decision(
                primary_validation={
                    "method": "custom_required",
                    "reason": "Temporal policy is required but no time column is available.",
                },
                diagnostic_validations=diagnostic_validations,
                rejected_validations=rejected_validations,
                confidence="low",
                evidence_refs=evidence_refs,
                warnings=warnings,
                limitations=[limitation],
                reasoning_summary=(
                    "Temporal signal exists, but the dataset lacks a usable time column."
                ),
            )
        evidence_refs.append("metric_evidence.requires_time")
        return _decision(
            primary_validation={
                "method": "temporal_holdout",
                "split_column": time_column,
                "reason": "Task, metric, or validation signals require temporal validation.",
            },
            diagnostic_validations=[
                {
                    "method": "expanding_window",
                    "split_column": time_column,
                    "reason": "Use as robustness check when enough periods exist.",
                },
                *diagnostic_validations,
            ],
            rejected_validations=rejected_validations,
            confidence="high",
            evidence_refs=_unique(evidence_refs),
            warnings=warnings,
            limitations=limitations,
            reasoning_summary=(
                "Selected temporal_holdout because temporal validation is explicitly required."
            ),
        )

    if resolved_task_type == TaskType.RANKING or metric_spec.requires_query_groups:
        if group_column is None:
            warnings.append("Ranking/query validation requires a query/group column.")
            return _decision(
                primary_validation={
                    "method": "custom_required",
                    "reason": "Ranking metric requires query groups, but none were inferred.",
                },
                diagnostic_validations=diagnostic_validations,
                rejected_validations=rejected_validations,
                confidence="low",
                evidence_refs=evidence_refs,
                warnings=warnings,
                limitations=["Ranking CV cannot be constructed without query groups."],
                reasoning_summary="Ranking metric/task requires grouped validation.",
            )
        evidence_refs.append("inferred_schema.candidate_group_columns")
        return _decision(
            primary_validation={
                "method": "ranking_group_cv",
                "group_column": group_column,
                "reason": "Ranking/query metrics must keep query groups intact.",
            },
            diagnostic_validations=diagnostic_validations,
            rejected_validations=rejected_validations,
            confidence="high",
            evidence_refs=_unique(evidence_refs),
            warnings=warnings,
            limitations=limitations,
            reasoning_summary=(
                "Selected ranking_group_cv because the task or metric is ranking-like."
            ),
        )

    if group_column is not None:
        method = (
            "stratified_group_kfold"
            if resolved_task_type in _CLASSIFICATION_TASK_TYPES
            else "group_kfold"
        )
        evidence_refs.append("inferred_schema.candidate_group_columns")
        return _decision(
            primary_validation={
                "method": method,
                "group_column": group_column,
                "reason": "Group/entity columns indicate leakage risk across ordinary folds.",
            },
            diagnostic_validations=diagnostic_validations,
            rejected_validations=rejected_validations,
            confidence="high",
            evidence_refs=_unique(evidence_refs),
            warnings=warnings,
            limitations=limitations,
            reasoning_summary=f"Selected {method} because group-like columns were inferred.",
        )

    if resolved_task_type in _CLASSIFICATION_TASK_TYPES:
        return _decision(
            primary_validation={
                "method": "stratified_kfold",
                "reason": "IID classification without group or temporal requirements.",
            },
            diagnostic_validations=diagnostic_validations,
            rejected_validations=rejected_validations,
            confidence="medium" if time_column else "high",
            evidence_refs=_unique(evidence_refs),
            warnings=warnings,
            limitations=limitations,
            reasoning_summary="Selected stratified_kfold for ordinary iid classification.",
        )

    if (
        resolved_task_type == TaskType.REGRESSION
        or metric_spec.family == MetricFamily.REGRESSION_ERROR
    ):
        return _decision(
            primary_validation={
                "method": "kfold",
                "reason": "IID regression without group or temporal requirements.",
            },
            diagnostic_validations=diagnostic_validations,
            rejected_validations=rejected_validations,
            confidence="medium" if time_column else "high",
            evidence_refs=_unique(evidence_refs),
            warnings=warnings,
            limitations=limitations,
            reasoning_summary="Selected kfold for ordinary iid regression.",
        )

    warnings.append("Task type is unknown; selected conservative KFold fallback.")
    return _decision(
        primary_validation={
            "method": "kfold",
            "reason": "Conservative fallback for unknown non-custom task type.",
        },
        diagnostic_validations=diagnostic_validations,
        rejected_validations=rejected_validations,
        confidence="low",
        evidence_refs=_unique(evidence_refs),
        warnings=warnings,
        limitations=limitations,
        reasoning_summary="Selected fallback kfold because no specialized policy matched.",
    )


def _decision(
    *,
    primary_validation: dict[str, Any],
    diagnostic_validations: list[dict[str, Any]],
    rejected_validations: list[dict[str, Any]],
    confidence: str,
    evidence_refs: list[str],
    warnings: list[str],
    limitations: list[str],
    reasoning_summary: str,
) -> dict[str, Any]:
    return {
        "primary_validation": primary_validation,
        "diagnostic_validations": diagnostic_validations,
        "rejected_validations": rejected_validations,
        "confidence": confidence,
        "evidence_refs": evidence_refs,
        "warnings": warnings,
        "limitations": limitations,
        "reasoning_summary": reasoning_summary,
    }


def _coerce_task_type(task_type: TaskType | str | None) -> TaskType:
    if isinstance(task_type, TaskType):
        return task_type
    if task_type is None:
        return TaskType.UNKNOWN
    normalized = str(task_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "time_series":
        normalized = "forecasting_tabular"
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.UNKNOWN


def _requires_custom_policy(metric_spec: MetricSpec) -> bool:
    return (
        metric_spec.needs_custom_implementation
        or metric_spec.family in {MetricFamily.UNKNOWN, MetricFamily.CUSTOM}
    )


def _requires_temporal_policy(
    task_type: TaskType,
    metric_spec: MetricSpec,
    validation_signals: dict[str, Any],
    scout_hypotheses: list[Any] | None,
) -> bool:
    if task_type == TaskType.FORECASTING_TABULAR:
        return True
    if metric_spec.requires_time or metric_spec.family == MetricFamily.TEMPORAL_STABILITY:
        return True
    if bool(validation_signals.get("requires_temporal_validation")):
        return True
    text = str(scout_hypotheses or "").lower()
    return any(term in text for term in ("out-of-time", "temporal validation", "rolling"))


def _preferred_group_column(candidates: list[dict[str, Any]]) -> str | None:
    if not candidates:
        return None
    for token in ("query", "session", "group"):
        for candidate in candidates:
            name = str(candidate["name"])
            if token in name.lower():
                return name
    return str(candidates[0]["name"])


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


_CLASSIFICATION_TASK_TYPES = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
}


__all__ = ["ValidationPolicySelector", "select_validation_policy"]
