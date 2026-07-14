from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, Literal, Mapping

from pydantic import Field, StringConstraints, ValidationInfo, model_validator

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.action_canonicalization import (
    canonicalize_final_strategy_actions,
)
from kaggle_researcher.contracts.ids import (
    EvidenceId,
    ExperimentId,
    HypothesisId,
    RiskId,
    SafetyConstraintId,
    SourceClaimId,
    ValidationRequirementId,
)
from kaggle_researcher.contracts.normalization import normalize_contract_payload

if TYPE_CHECKING:
    from kaggle_researcher.contracts.registries import ContractRegistries


Priority = Literal["P0", "P1", "P2", "P3"]
Confidence = Literal["low", "medium", "high"]
EvidenceOrigin = Literal[
    "EDA-confirmed", "EDA-inferred", "Source-supported", "Hypothesis-to-test",
    "Safety-warning", "Fallback-generated",
]
FinalValidationMethod = Literal[
    "stratified_kfold", "kfold", "group_kfold", "stratified_group_kfold",
    "temporal_holdout", "temporal_cv", "ranking_group_cv", "custom_required",
]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceRef = EvidenceId
REQUIRED_SECTION_IDS = [
    "executive_summary", "metric_and_validation", "dataset_facts_from_eda",
    "leakage_and_data_quality", "drift_and_leaderboard_risk", "baseline_findings",
    "feature_priorities", "modeling_plan", "experiments_queue", "what_not_to_do",
    "first_48_hours",
]
TEMPORAL_VALIDATION_METHODS = {"temporal_holdout", "temporal_cv"}
REPAIR_LIMITATION = (
    "Final strategy payload was repaired deterministically because the LLM "
    "omitted required linkage fields."
)
FALLBACK_LIMITATION = (
    "The LLM strategy remained invalid after deterministic repair, so this "
    "strategy was built from available Scout hypotheses and EDA evidence."
)


class FinalStrategyAction(ContractModel):
    action_id: NonEmptyString | None = None
    priority: Priority
    action: NonEmptyString
    reason: NonEmptyString
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    related_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    experiment_ids: list[ExperimentId] = Field(default_factory=list)
    source_claim: NonEmptyString | None = None
    source_refs: list[SourceClaimId] = Field(default_factory=list)
    eda_result_refs: list[EvidenceRef] = Field(default_factory=list)
    risk_ids: list[RiskId] = Field(default_factory=list)
    validation_requirement_ids: list[ValidationRequirementId] = Field(default_factory=list)
    safety_constraint_ids: list[SafetyConstraintId] = Field(default_factory=list)
    validation_strategy: FinalValidationMethod | None = None
    confidence: Confidence = "medium"
    evidence_origin: EvidenceOrigin = "Hypothesis-to-test"
    limitations: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: Any) -> Any:
        return normalize_contract_payload(value, cls.__name__)

    @model_validator(mode="after")
    def _require_strategy_links(self) -> "FinalStrategyAction":
        linked = list(dict.fromkeys([*self.related_hypothesis_ids, *self.hypothesis_ids]))
        object.__setattr__(self, "related_hypothesis_ids", linked)
        object.__setattr__(self, "hypothesis_ids", list(linked))
        if not self.evidence_refs:
            raise ValueError("FinalStrategyAction.evidence_refs must not be empty")
        if not self.related_hypothesis_ids:
            raise ValueError("FinalStrategyAction.related_hypothesis_ids must not be empty")
        return self


class FinalStrategySection(ContractModel):
    section_id: NonEmptyString
    title: NonEmptyString
    summary: NonEmptyString
    action_ids: list[NonEmptyString] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    related_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: Any) -> Any:
        return normalize_contract_payload(value, cls.__name__)

    @model_validator(mode="after")
    def _require_action_or_evidence(self) -> "FinalStrategySection":
        if not self.action_ids and not self.evidence_refs:
            raise ValueError("FinalStrategySection must include action_ids or evidence_refs")
        return self


class FinalStrategyResult(ContractModel):
    contract_family: Literal["final_strategy"] = "final_strategy"
    schema_version: Literal["1.0"] = "1.0"
    competition_id: NonEmptyString
    task_type: NonEmptyString | None = None
    metric: dict[str, Any] = Field(default_factory=dict)
    recommended_validation: FinalValidationMethod | None = None
    sections: list[FinalStrategySection] = Field(default_factory=list)
    actions: list[FinalStrategyAction] = Field(default_factory=list)
    source_to_hypothesis_links: list[dict[str, Any]] = Field(default_factory=list)
    hypothesis_to_eda_links: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[NonEmptyString] = Field(default_factory=list)
    models_used: dict[str, Any] = Field(default_factory=dict)
    reference_repairs: list[dict[str, str]] = Field(default_factory=list)
    acknowledged_risk_ids: list[RiskId] = Field(default_factory=list)
    selected_validation_requirement_ids: list[ValidationRequirementId] = Field(default_factory=list)
    enforced_safety_constraint_ids: list[SafetyConstraintId] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        if isinstance(value, cls) or info.field_name is not None:
            return value
        if isinstance(value, Mapping) and any(
            isinstance(action, FinalStrategyAction)
            for action in value.get("actions") or []
        ):
            return value
        normalized = normalize_contract_payload(value, cls.__name__)
        if not isinstance(normalized, Mapping):
            return normalized
        canonical, _ = canonicalize_final_strategy_actions(normalized)
        if not canonical.get("actions"):
            raise ValueError("FinalStrategyResult must include at least one action")
        return canonical


def migrate_legacy_final_strategy_payload(
    payload: Mapping[str, Any], *, registries: "ContractRegistries"
):
    """Move exact legacy generic references into their dedicated namespaces."""

    from kaggle_researcher.contracts.errors import AmbiguousReferenceError
    from kaggle_researcher.contracts.migration import MigrationResult

    migrated = deepcopy(dict(payload))
    changes: list[str] = []
    repairs = list(migrated.get("reference_repairs") or [])
    action_groups: list[tuple[str, list[Any]]] = [("actions", migrated.get("actions") or [])]
    for section_index, section in enumerate(migrated.get("sections") or []):
        if isinstance(section, Mapping):
            action_groups.append((
                f"sections[{section_index}].actions",
                list(section.get("actions") or []),
            ))
    field_for_namespace = {
        "risk": "risk_ids",
        "validation_requirement": "validation_requirement_ids",
        "safety_constraint": "safety_constraint_ids",
    }
    for group_path, actions in action_groups:
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            retained: list[Any] = []
            dedicated = {
                field: list(action.get(field) or []) for field in field_for_namespace.values()
            }
            for reference in action.get("evidence_refs") or []:
                try:
                    namespace = registries.namespace_for(str(reference))
                except AmbiguousReferenceError as exc:
                    raise AmbiguousReferenceError(
                        f"Legacy Final Strategy reference {reference!r} is ambiguous"
                    ) from exc
                target = field_for_namespace.get(namespace or "")
                if target is None:
                    retained.append(reference)
                    continue
                if reference not in dedicated[target]:
                    dedicated[target].append(reference)
                path = f"{group_path}[{index}].evidence_refs"
                changes.append(f"moved {reference!r} from {path} to {target}")
                repairs.append({
                    "field_path": path,
                    "original_id": str(reference),
                    "replacement_id": str(reference),
                })
            action["evidence_refs"] = list(dict.fromkeys(retained))
            for field, values in dedicated.items():
                action[field] = list(dict.fromkeys(values))
    migrated["reference_repairs"] = repairs
    return MigrationResult(
        migrated,
        str(payload.get("schema_version")) if payload.get("schema_version") else None,
        "1.0",
        bool(changes),
        changes,
        [],
    )


__all__ = [
    "Confidence", "EvidenceOrigin", "EvidenceRef", "FALLBACK_LIMITATION",
    "FinalStrategyAction", "FinalStrategyResult", "FinalStrategySection",
    "FinalValidationMethod", "Priority", "REPAIR_LIMITATION", "REQUIRED_SECTION_IDS",
    "TEMPORAL_VALIDATION_METHODS",
    "migrate_legacy_final_strategy_payload",
]
