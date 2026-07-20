from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaggle_researcher.eda.metrics import MetricFamily, MetricSpec, TaskType
from kaggle_researcher.eda.schemas import InferredSchema, TableProfile


@dataclass(frozen=True, slots=True)
class ValidationPolicyDecision:
    primary_validation: dict[str, Any]
    diagnostic_validations: list[dict[str, Any]] = field(default_factory=list)
    rejected_validations: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"
    evidence_refs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ValidationPolicySelector:
    def select(
        self,
        *,
        task_type: TaskType | str | None,
        metric_spec: MetricSpec,
        inferred_schema: InferredSchema,
        table_profiles: list[TableProfile] | None = None,
        scout_hypotheses: list[Any] | None = None,
        drift_evidence: dict[str, Any] | None = None,
    ) -> ValidationPolicyDecision:
        del table_profiles
        resolved_task_type = _coerce_task_type(task_type)
        evidence_refs: list[str] = []
        warnings: list[str] = []
        diagnostic_validations: list[dict[str, Any]] = []
        rejected_validations: list[dict[str, Any]] = []

        time_columns = list(inferred_schema.candidate_time_columns)
        date_columns = list(inferred_schema.candidate_date_columns)
        group_columns = _candidate_group_columns(inferred_schema)
        temporal_signal = _has_temporal_policy_signal(
            task_type=resolved_task_type,
            metric_spec=metric_spec,
            scout_hypotheses=scout_hypotheses or [],
            drift_evidence=drift_evidence or {},
        )

        if time_columns:
            evidence_refs.append("inferred_schema.candidate_time_columns")
            diagnostic_validations.append(
                {
                    "method": "temporal_holdout_diagnostic",
                    "split_column": time_columns[0],
                    "reason": "Time-like columns exist, but they are diagnostic unless metric/task/scout/drift evidence requires temporal validation.",
                }
            )
            rejected_validations.append(
                {
                    "method": "primary_temporal_cv_from_time_column_only",
                    "reason": "A time column alone is not sufficient evidence for primary temporal CV.",
                }
            )
        if date_columns:
            evidence_refs.append("inferred_schema.candidate_date_columns")

        if temporal_signal:
            split_column = _preferred_temporal_column(time_columns, date_columns)
            if split_column is None:
                warnings.append("Temporal/stability validation was selected, but no time/date column was inferred.")
                confidence = "medium"
            else:
                confidence = "high"
            primary = {
                "method": "temporal_holdout_and_expanding_cv",
                "split_column": split_column,
                "reason": "Task, metric, scout, or drift evidence indicates temporal/stability validation is required.",
            }
            evidence_refs.extend(_temporal_signal_refs(metric_spec, scout_hypotheses, drift_evidence))
            return ValidationPolicyDecision(
                primary_validation=primary,
                diagnostic_validations=diagnostic_validations,
                rejected_validations=rejected_validations,
                confidence=confidence,
                evidence_refs=_unique(evidence_refs),
                warnings=warnings,
            )

        if resolved_task_type == TaskType.RANKING:
            group_column = _preferred_query_group(group_columns)
            if group_column is None:
                warnings.append("Ranking task selected, but no query/group id column was inferred.")
                confidence = "medium"
            else:
                confidence = "high"
                evidence_refs.append("inferred_schema.candidate_group_columns")
            return ValidationPolicyDecision(
                primary_validation={
                    "method": "GroupKFold",
                    "group_column": group_column,
                    "reason": "Ranking/query tasks must keep query groups intact across folds.",
                },
                diagnostic_validations=diagnostic_validations,
                rejected_validations=rejected_validations,
                confidence=confidence,
                evidence_refs=_unique(evidence_refs),
                warnings=warnings,
            )

        if group_columns:
            evidence_refs.append("inferred_schema.candidate_group_columns")
            method = (
                "StratifiedGroupKFold"
                if resolved_task_type in {
                    TaskType.BINARY_CLASSIFICATION,
                    TaskType.MULTICLASS_CLASSIFICATION,
                    TaskType.MULTILABEL_CLASSIFICATION,
                }
                else "GroupKFold"
            )
            return ValidationPolicyDecision(
                primary_validation={
                    "method": method,
                    "group_column": group_columns[0],
                    "reason": "Group/entity columns suggest leakage risk if entities cross folds.",
                },
                diagnostic_validations=diagnostic_validations,
                rejected_validations=rejected_validations,
                confidence="high",
                evidence_refs=_unique(evidence_refs),
                warnings=warnings,
            )

        if resolved_task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        }:
            return ValidationPolicyDecision(
                primary_validation={
                    "method": "StratifiedKFold",
                    "reason": "IID classification without group/entity or temporal/stability evidence.",
                },
                diagnostic_validations=diagnostic_validations,
                rejected_validations=rejected_validations,
                confidence="medium" if time_columns else "high",
                evidence_refs=_unique(evidence_refs),
                warnings=warnings,
            )

        if resolved_task_type == TaskType.REGRESSION or metric_spec.family == MetricFamily.REGRESSION:
            return ValidationPolicyDecision(
                primary_validation={
                    "method": "KFold",
                    "reason": "IID regression without group/entity or temporal/stability evidence.",
                },
                diagnostic_validations=diagnostic_validations,
                rejected_validations=rejected_validations,
                confidence="medium" if time_columns else "high",
                evidence_refs=_unique(evidence_refs),
                warnings=warnings,
            )

        warnings.append("Task type is unknown; selected generic KFold as conservative fallback.")
        return ValidationPolicyDecision(
            primary_validation={
                "method": "KFold",
                "reason": "Fallback for unknown task type.",
            },
            diagnostic_validations=diagnostic_validations,
            rejected_validations=rejected_validations,
            confidence="low",
            evidence_refs=_unique(evidence_refs),
            warnings=warnings,
        )


def select_validation_policy(
    *,
    task_type: TaskType | str | None,
    metric_spec: MetricSpec,
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile] | None = None,
    scout_hypotheses: list[Any] | None = None,
    drift_evidence: dict[str, Any] | None = None,
) -> ValidationPolicyDecision:
    return ValidationPolicySelector().select(
        task_type=task_type,
        metric_spec=metric_spec,
        inferred_schema=inferred_schema,
        table_profiles=table_profiles,
        scout_hypotheses=scout_hypotheses,
        drift_evidence=drift_evidence,
    )


def _coerce_task_type(task_type: TaskType | str | None) -> TaskType:
    if isinstance(task_type, TaskType):
        return task_type
    if task_type is None:
        return TaskType.UNKNOWN
    normalized = str(task_type).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.UNKNOWN


def _candidate_group_columns(inferred_schema: InferredSchema) -> list[str]:
    candidates = list(inferred_schema.candidate_group_columns)
    for key in inferred_schema.global_roles.get("candidate_join_keys", []):
        normalized = str(key).lower()
        if any(token in normalized for token in ("query", "group", "customer", "user", "patient", "client")):
            candidates.append(str(key))
    return _unique(candidates)


def _has_temporal_policy_signal(
    *,
    task_type: TaskType,
    metric_spec: MetricSpec,
    scout_hypotheses: list[Any],
    drift_evidence: dict[str, Any],
) -> bool:
    if task_type == TaskType.TIME_SERIES:
        return True
    metric_name = metric_spec.name.lower()
    if "stability" in metric_name or "forecast" in metric_name:
        return True
    if _scout_requests_temporal_validation(scout_hypotheses):
        return True
    if _drift_requires_temporal_validation(drift_evidence):
        return True
    return False


def _scout_requests_temporal_validation(scout_hypotheses: list[Any]) -> bool:
    temporal_terms = (
        "temporal validation",
        "out-of-time",
        "out of time",
        "expanding",
        "rolling",
        "forecast",
        "stability",
    )
    for hypothesis in scout_hypotheses:
        if hasattr(hypothesis, "model_dump"):
            text = str(hypothesis.model_dump(mode="json")).lower()
        else:
            text = str(hypothesis).lower()
        if any(term in text for term in temporal_terms):
            return True
    return False


def _drift_requires_temporal_validation(drift_evidence: dict[str, Any]) -> bool:
    text = str(drift_evidence).lower()
    return any(
        term in text
        for term in (
            "temporal_drift",
            "time_drift",
            "period_drift",
            "medium",
            "high",
            "severe",
        )
    ) and any(term in text for term in ("drift", "shift"))


def _preferred_temporal_column(time_columns: list[str], date_columns: list[str]) -> str | None:
    if time_columns:
        return time_columns[0]
    if date_columns:
        return date_columns[0]
    return None


def _preferred_query_group(group_columns: list[str]) -> str | None:
    for column in group_columns:
        if "query" in column.lower():
            return column
    return group_columns[0] if group_columns else None


def _temporal_signal_refs(
    metric_spec: MetricSpec,
    scout_hypotheses: list[Any] | None,
    drift_evidence: dict[str, Any] | None,
) -> list[str]:
    refs: list[str] = []
    if "stability" in metric_spec.name.lower() or metric_spec.requires_groups_or_time:
        refs.append("metric_evidence.requires_time_or_groups")
    if scout_hypotheses:
        refs.append("research_hypotheses.validation")
    if drift_evidence:
        refs.append("drift_evidence")
    return refs


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "ValidationPolicyDecision",
    "ValidationPolicySelector",
    "select_validation_policy",
]
