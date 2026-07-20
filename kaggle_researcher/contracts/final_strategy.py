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
SynthesisStatus = Literal[
    "llm_success",
    "repaired_success",
    "degraded_fallback",
]
SelectionStatus = Literal[
    "llm_success", "repaired_success", "degraded_fallback", "failed",
]
RenderingStatus = Literal[
    "llm_success", "repaired_success", "deterministic_render", "failed",
]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceRef = EvidenceId
EvidenceBindingRole = Literal["primary", "supporting", "limitation"]
SectionAvailability = Literal["available", "limited", "not_available"]
First48HourWindow = Literal["0-4_hours", "4-12_hours", "12-24_hours", "24-48_hours"]
FitScope = Literal["per_row", "within_fold", "oof_only", "not_applicable"]
LeakageRisk = Literal["low", "medium", "high"]
ExperimentCost = Literal["low", "medium", "high"]
ExperimentStatus = Literal["required", "core", "backlog", "rejected", "blocked"]
FamilyStatus = Literal["planned", "backlog", "rejected"]
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


class EvidenceBinding(ContractModel):
    ref: EvidenceRef
    resolved_value_preview: str | int | float | bool | None | dict[str, Any] | list[Any]
    role: EvidenceBindingRole = "supporting"


class EvidenceCatalogEntry(ContractModel):
    ref: EvidenceRef
    resolved_value_preview: str | int | float | bool | None | dict[str, Any] | list[Any]
    value_type: NonEmptyString
    source_component: NonEmptyString
    specificity: Literal["root", "object", "item", "leaf"]
    available: bool = True
    warnings: list[NonEmptyString] = Field(default_factory=list)


class ExperimentArm(ContractModel):
    arm_id: NonEmptyString
    name: NonEmptyString
    exact_change: NonEmptyString
    generated_features: list[NonEmptyString] = Field(default_factory=list)
    fit_scope: FitScope
    leakage_risk: LeakageRisk
    dependencies: list[NonEmptyString] = Field(default_factory=list)


class FeatureExperimentFamily(ContractModel):
    family_id: NonEmptyString
    name: NonEmptyString
    priority: Priority
    input_columns: list[NonEmptyString]
    hypothesis: NonEmptyString
    baseline_arm: ExperimentArm
    candidate_arms: list[ExperimentArm]
    validation_strategy: FinalValidationMethod
    metric: NonEmptyString
    fit_scope: FitScope
    evidence_refs: list[EvidenceRef]
    motivating_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    risks: list[NonEmptyString]
    acceptance_rule: NonEmptyString
    estimated_cost: ExperimentCost
    status: FamilyStatus = "planned"

    @model_validator(mode="after")
    def _require_comparable_arms(self) -> "FeatureExperimentFamily":
        if not self.input_columns or not self.candidate_arms or not self.evidence_refs:
            raise ValueError("Feature experiment families require inputs, candidate arms, and evidence")
        arm_ids = [self.baseline_arm.arm_id, *(arm.arm_id for arm in self.candidate_arms)]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("Feature experiment family arm IDs must be unique")
        return self


class ExperimentBudget(ContractModel):
    max_core_experiments: int = Field(default=8, ge=1)
    max_first_24h_experiments: int = Field(default=4, ge=1)
    max_first_48h_experiments: int = Field(default=8, ge=1)
    max_high_cost_experiments: int = Field(default=2, ge=0)
    estimated_total_cost: float = Field(default=0.0, ge=0.0)
    budget_policy_version: NonEmptyString = "1.0"


class FinalStrategyQualityMetrics(ContractModel):
    model_config = {
        **ContractModel.model_config,
        "protected_namespaces": (),
    }
    action_count: int = 0
    feature_family_count: int = 0
    core_experiment_count: int = 0
    backlog_experiment_count: int = 0
    average_evidence_refs_per_action: float = 0.0
    max_evidence_refs_per_action: int = 0
    root_refs_for_specific_claims: int = 0
    unsupported_evidence_refs_removed: int = 0
    actions_exceeding_evidence_limits: int = 0
    unresolved_refs: int = 0
    actions_with_source_refs: int = 0
    actions_without_source_refs: int = 0
    hypothesis_role_counts: dict[str, int] = Field(default_factory=dict)
    duplicate_preview_bytes_avoided: int = 0
    model_self_comparisons_removed: int = 0
    first_48h_experiment_count: int = 0


class FeatureActionMetadata(ContractModel):
    """Executable, leakage-aware metadata for one derived-feature action."""

    input_columns: list[NonEmptyString]
    deterministic_transform: NonEmptyString
    fit_scope: FitScope
    leakage_risk: LeakageRisk
    validation_strategy: FinalValidationMethod
    expected_diagnostic: NonEmptyString

    @model_validator(mode="after")
    def _require_inputs(self) -> "FeatureActionMetadata":
        if not self.input_columns:
            raise ValueError("FeatureActionMetadata.input_columns must not be empty")
        object.__setattr__(self, "input_columns", sorted(set(self.input_columns)))
        return self


class FinalStrategyExperiment(ContractModel):
    """A deterministic, executable experiment compiled from validated evidence."""

    model_config = {
        **ContractModel.model_config,
        "protected_namespaces": (),
    }

    experiment_id: NonEmptyString
    priority: Priority
    name: NonEmptyString
    hypothesis: NonEmptyString
    change: NonEmptyString
    feature_inputs: list[NonEmptyString] = Field(default_factory=list)
    model_family: NonEmptyString
    validation_strategy: FinalValidationMethod
    success_metric: NonEmptyString
    acceptance_rule: NonEmptyString
    evidence_refs: list[EvidenceRef]
    primary_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    limitation_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    related_hypothesis_ids: list[HypothesisId]
    risks: list[NonEmptyString]
    fit_scope: FitScope
    baseline_canonical_family_id: NonEmptyString | None = None
    baseline_implementation_id: NonEmptyString | None = None
    candidate_canonical_family_id: NonEmptyString | None = None
    candidate_implementation_id: NonEmptyString | None = None
    status: ExperimentStatus = "core"
    estimated_cost: ExperimentCost = "low"
    dependencies: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_grounding(self) -> "FinalStrategyExperiment":
        if not self.evidence_refs:
            raise ValueError("FinalStrategyExperiment.evidence_refs must not be empty")
        if not self.related_hypothesis_ids:
            raise ValueError(
                "FinalStrategyExperiment.related_hypothesis_ids must not be empty"
            )
        if not self.risks:
            raise ValueError("FinalStrategyExperiment.risks must not be empty")
        object.__setattr__(self, "feature_inputs", sorted(set(self.feature_inputs)))
        object.__setattr__(self, "evidence_refs", list(dict.fromkeys(self.evidence_refs)))
        primary = [ref for ref in self.primary_evidence_refs if ref in self.evidence_refs]
        limitations = [ref for ref in self.limitation_evidence_refs if ref in self.evidence_refs]
        object.__setattr__(self, "primary_evidence_refs", primary or self.evidence_refs[:1])
        object.__setattr__(self, "limitation_evidence_refs", limitations)
        object.__setattr__(
            self,
            "related_hypothesis_ids",
            list(dict.fromkeys(self.related_hypothesis_ids)),
        )
        object.__setattr__(self, "risks", list(dict.fromkeys(self.risks)))
        return self


class FinalStrategyAction(ContractModel):
    action_id: NonEmptyString | None = None
    priority: Priority
    action: NonEmptyString
    reason: NonEmptyString
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    primary_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    limitation_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    evidence_overflow_reason: NonEmptyString | None = None
    # Accepted only as a legacy input adapter. Canonical JSON uses evidence_catalog.
    evidence_bindings: list[EvidenceBinding] = Field(default_factory=list, exclude=True)
    related_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    motivating_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    safety_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    validation_context_ids: list[HypothesisId] = Field(default_factory=list)
    rejected_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    hypothesis_role_overlap_reason: NonEmptyString | None = None
    experiment_ids: list[ExperimentId] = Field(default_factory=list)
    experiment_id: NonEmptyString | None = None
    hypothesis: NonEmptyString | None = None
    exact_change: NonEmptyString | None = None
    validation_policy: NonEmptyString | None = None
    success_criterion: NonEmptyString | None = None
    risk: NonEmptyString | None = None
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
    feature_metadata: FeatureActionMetadata | None = None
    action_kind: NonEmptyString = "general"
    dependencies: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: Any) -> Any:
        return normalize_contract_payload(value, cls.__name__)

    @model_validator(mode="after")
    def _require_strategy_links(self) -> "FinalStrategyAction":
        roles = [
            *self.motivating_hypothesis_ids,
            *self.safety_hypothesis_ids,
            *self.validation_context_ids,
            *self.rejected_hypothesis_ids,
        ]
        legacy = list(dict.fromkeys([*self.related_hypothesis_ids, *self.hypothesis_ids]))
        if not roles and legacy:
            object.__setattr__(self, "motivating_hypothesis_ids", list(legacy))
            roles = list(legacy)
        elif legacy:
            unassigned = [item for item in legacy if item not in roles]
            if unassigned:
                object.__setattr__(
                    self,
                    "motivating_hypothesis_ids",
                    list(dict.fromkeys([*self.motivating_hypothesis_ids, *unassigned])),
                )
                roles = [*roles, *unassigned]
        linked = list(dict.fromkeys(roles))
        object.__setattr__(self, "related_hypothesis_ids", linked)
        object.__setattr__(self, "hypothesis_ids", list(linked))
        if not self.evidence_refs:
            raise ValueError("FinalStrategyAction.evidence_refs must not be empty")
        if not self.related_hypothesis_ids:
            raise ValueError("FinalStrategyAction.related_hypothesis_ids must not be empty")
        primary = [ref for ref in self.primary_evidence_refs if ref in self.evidence_refs]
        limitations = [ref for ref in self.limitation_evidence_refs if ref in self.evidence_refs]
        object.__setattr__(self, "primary_evidence_refs", primary or self.evidence_refs[:1])
        object.__setattr__(self, "limitation_evidence_refs", limitations)
        role_memberships = [
            self.motivating_hypothesis_ids,
            self.safety_hypothesis_ids,
            self.validation_context_ids,
            self.rejected_hypothesis_ids,
        ]
        role_total = sum(len(set(values)) for values in role_memberships)
        if role_total != len(set().union(*(set(values) for values in role_memberships))):
            if not self.hypothesis_role_overlap_reason:
                raise ValueError("A hypothesis in multiple semantic roles requires justification")
        missing_eda_refs = set(self.eda_result_refs) - set(self.evidence_refs)
        if missing_eda_refs:
            raise ValueError(
                "FinalStrategyAction.eda_result_refs must be a subset of evidence_refs: "
                f"{sorted(missing_eda_refs)}"
            )
        binding_refs = [binding.ref for binding in self.evidence_bindings]
        if len(binding_refs) != len(set(binding_refs)):
            raise ValueError("FinalStrategyAction.evidence_bindings contains duplicate refs")
        unknown_binding_refs = set(binding_refs) - set(self.evidence_refs)
        if unknown_binding_refs:
            raise ValueError(
                "FinalStrategyAction.evidence_bindings must reference evidence_refs: "
                f"{sorted(unknown_binding_refs)}"
            )
        structured_experiment = (
            self.experiment_id,
            self.hypothesis,
            self.exact_change,
            self.validation_policy,
            self.success_criterion,
            self.risk,
        )
        if any(value is not None for value in structured_experiment) and not all(
            value is not None for value in structured_experiment
        ):
            raise ValueError(
                "Structured experiment actions require experiment_id, hypothesis, "
                "exact_change, validation_policy, success_criterion, and risk"
            )
        return self


class SourceToHypothesisLink(ContractModel):
    source_ref: SourceClaimId
    hypothesis_id: HypothesisId
    relationship: Literal["supports", "motivates", "contradicts"] = "supports"
    claim_summary: NonEmptyString | None = None
    confidence: Confidence = "medium"

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_claim_field(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "claim_summary" not in value:
            upgraded = dict(value)
            upgraded["claim_summary"] = upgraded.pop("source_claim", None)
            return upgraded
        return value


class HypothesisToEdaLink(ContractModel):
    hypothesis_id: HypothesisId
    eda_result_ref: EvidenceRef
    result_status: Literal[
        "confirmed",
        "partially_confirmed",
        "rejected",
        "not_testable",
        "skipped",
    ] = "confirmed"
    finding_summary: NonEmptyString = "Legacy EDA result link."
    confidence: Confidence = "medium"


class ActionProvenance(ContractModel):
    action_id: NonEmptyString
    source_refs: list[SourceClaimId] = Field(default_factory=list)
    hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    motivating_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    safety_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    validation_context_ids: list[HypothesisId] = Field(default_factory=list)
    eda_result_refs: list[EvidenceRef] = Field(default_factory=list)


class First48HourBlock(ContractModel):
    time_window: First48HourWindow
    summary: NonEmptyString
    action_ids: list[NonEmptyString] = Field(default_factory=list)
    experiment_ids: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_reference(self) -> "First48HourBlock":
        if not self.action_ids and not self.experiment_ids:
            raise ValueError("First48HourBlock must reference an action or experiment")
        return self


class FinalStrategySection(ContractModel):
    section_id: NonEmptyString
    title: NonEmptyString
    summary: NonEmptyString
    action_ids: list[NonEmptyString] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    related_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    source_refs: list[SourceClaimId] = Field(default_factory=list)
    eda_result_refs: list[EvidenceRef] = Field(default_factory=list)
    availability: SectionAvailability = "available"
    limitations: list[NonEmptyString] = Field(default_factory=list)
    time_blocks: list[First48HourBlock] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: Any) -> Any:
        return normalize_contract_payload(value, cls.__name__)

    @model_validator(mode="after")
    def _require_action_or_evidence(self) -> "FinalStrategySection":
        if (
            not self.action_ids
            and not self.evidence_refs
            and not self.limitations
            and self.availability == "available"
        ):
            raise ValueError("FinalStrategySection must include action_ids or evidence_refs")
        return self


class FinalStrategyResult(ContractModel):
    contract_family: Literal["final_strategy"] = "final_strategy"
    schema_version: Literal["1.0", "2.0"] = "2.0"
    competition_id: NonEmptyString
    synthesis_status: SynthesisStatus
    selection_status: SelectionStatus | None = None
    rendering_status: RenderingStatus | None = None
    llm_output_valid: bool
    repair_attempted: bool
    repair_succeeded: bool
    fallback_used: bool
    synthesis_diagnostics_path: str | None
    skeleton_id: str | None = None
    skeleton_hash: str | None = None
    selection_prompt_fingerprint: dict[str, Any] = Field(default_factory=dict)
    rendering_prompt_fingerprint: dict[str, Any] = Field(default_factory=dict)
    task_type: NonEmptyString | None = None
    metric: dict[str, Any] = Field(default_factory=dict)
    recommended_validation: FinalValidationMethod | None = None
    sections: list[FinalStrategySection] = Field(default_factory=list)
    actions: list[FinalStrategyAction] = Field(default_factory=list)
    experiments: list[FinalStrategyExperiment] = Field(default_factory=list)
    evidence_catalog: dict[str, EvidenceCatalogEntry] = Field(default_factory=dict)
    feature_experiment_families: list[FeatureExperimentFamily] = Field(default_factory=list)
    core_experiments: list[FinalStrategyExperiment] = Field(default_factory=list)
    experiment_backlog: list[FinalStrategyExperiment] = Field(default_factory=list)
    experiment_budget: ExperimentBudget = Field(default_factory=ExperimentBudget)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    quality_metrics: FinalStrategyQualityMetrics = Field(default_factory=FinalStrategyQualityMetrics)
    diagnostics_summary: dict[str, Any] = Field(default_factory=dict)
    source_to_hypothesis_links: list[SourceToHypothesisLink] = Field(default_factory=list)
    hypothesis_to_eda_links: list[HypothesisToEdaLink] = Field(default_factory=list)
    action_provenance: list[ActionProvenance] = Field(default_factory=list)
    limitations: list[NonEmptyString] = Field(default_factory=list)
    warnings: list[NonEmptyString] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def _validate_synthesis_status(self) -> "FinalStrategyResult":
        if self.selection_status is None:
            object.__setattr__(self, "selection_status", self.synthesis_status)
        if self.rendering_status is None:
            object.__setattr__(self, "rendering_status", "deterministic_render")
        if self.selection_status != self.synthesis_status:
            if self.skeleton_id is None:
                # Legacy artifacts predate the separate stage status. Their
                # compatibility alias is derived on read and cannot diverge.
                object.__setattr__(self, "selection_status", self.synthesis_status)
            else:
                raise ValueError(
                    "synthesis_status must describe and match strategic selection status"
                )
        expected = {
            "llm_success": {
                "llm_output_valid": True,
                "repair_attempted": False,
                "repair_succeeded": False,
                "fallback_used": False,
            },
            "repaired_success": {
                "llm_output_valid": False,
                "repair_attempted": True,
                "repair_succeeded": True,
                "fallback_used": False,
            },
            "degraded_fallback": {
                "llm_output_valid": False,
                "repair_succeeded": False,
                "fallback_used": True,
            },
        }[self.synthesis_status]
        contradictions = [
            f"{field_name} must be {expected_value!r}"
            for field_name, expected_value in expected.items()
            if getattr(self, field_name) is not expected_value
        ]
        if contradictions:
            raise ValueError(
                f"Contradictory {self.synthesis_status!r} synthesis state: "
                + "; ".join(contradictions)
            )
        if self.synthesis_status == "degraded_fallback" and not any(
            "fallback" in limitation.lower()
            or (
                "llm" in limitation.lower()
                and any(
                    marker in limitation.lower()
                    for marker in ("invalid", "did not satisfy", "remained invalid")
                )
            )
            for limitation in self.limitations
        ):
            raise ValueError(
                "degraded_fallback requires a human-readable fallback limitation"
            )
        if self.synthesis_status == "degraded_fallback":
            section_ids = [section.section_id for section in self.sections]
            if section_ids != REQUIRED_SECTION_IDS:
                raise ValueError(
                    "degraded_fallback sections must exist exactly once in canonical "
                    f"order: {REQUIRED_SECTION_IDS}"
                )
            executive = self.sections[0]
            if "degraded" not in executive.summary.lower():
                raise ValueError(
                    "degraded_fallback executive_summary must expose degraded status"
                )
            first_48 = self.sections[-1]
            windows = [block.time_window for block in first_48.time_blocks]
            expected_windows = [
                "0-4_hours", "4-12_hours", "12-24_hours", "24-48_hours",
            ]
            if windows != expected_windows:
                raise ValueError(
                    "degraded_fallback first_48_hours must contain the four canonical "
                    "ordered time blocks"
                )
        self._validate_provenance_shape()
        self._validate_first_48_hour_references()
        self._validate_compacted_contract()
        return self

    def _validate_compacted_contract(self) -> None:
        if self.schema_version != "2.0":
            return
        catalog_refs = set(self.evidence_catalog)
        if self.evidence_catalog:
            unresolved = sorted({
                ref
                for action in self.actions
                for ref in action.evidence_refs
                if ref not in catalog_refs
            })
            if unresolved:
                raise ValueError(f"Action evidence refs are absent from evidence_catalog: {unresolved}")
        core_ids = [item.experiment_id for item in self.core_experiments]
        backlog_ids = [item.experiment_id for item in self.experiment_backlog]
        overlap = set(core_ids) & set(backlog_ids)
        if overlap:
            raise ValueError(f"Core and backlog experiments overlap: {sorted(overlap)}")
        if len(core_ids) > self.experiment_budget.max_core_experiments:
            raise ValueError("Core experiment budget exceeded")
        high_cost = sum(item.estimated_cost == "high" for item in self.core_experiments)
        if high_cost > self.experiment_budget.max_high_cost_experiments:
            raise ValueError("High-cost experiment budget exceeded")
        for item in [*self.experiments, *self.core_experiments, *self.experiment_backlog]:
            if (
                item.baseline_canonical_family_id
                and item.candidate_canonical_family_id
                and item.baseline_canonical_family_id == item.candidate_canonical_family_id
            ):
                raise ValueError(
                    f"Experiment {item.experiment_id!r} compares one canonical model family with itself"
                )

    def _validate_first_48_hour_references(self) -> None:
        action_ids = {action.action_id for action in self.actions if action.action_id}
        experiment_ids = {
            action.experiment_id for action in self.actions if action.experiment_id
        }
        declared_experiment_ids = [item.experiment_id for item in self.experiments]
        if len(declared_experiment_ids) != len(set(declared_experiment_ids)):
            raise ValueError("experiments contains duplicate experiment_id values")
        experiment_ids.update(declared_experiment_ids)
        for section in self.sections:
            for block in section.time_blocks:
                unknown_actions = set(block.action_ids) - action_ids
                unknown_experiments = set(block.experiment_ids) - experiment_ids
                if unknown_actions or unknown_experiments:
                    raise ValueError(
                        f"Section {section.section_id!r} time block {block.time_window!r} "
                        "contains unknown references: "
                        f"actions={sorted(unknown_actions)}, "
                        f"experiments={sorted(unknown_experiments)}"
                    )

    def _validate_provenance_shape(self) -> None:
        source_link_keys = [
            (link.source_ref, link.hypothesis_id, link.relationship)
            for link in self.source_to_hypothesis_links
        ]
        if len(source_link_keys) != len(set(source_link_keys)):
            raise ValueError("source_to_hypothesis_links contains duplicate links")
        eda_link_keys = [
            (link.hypothesis_id, link.eda_result_ref)
            for link in self.hypothesis_to_eda_links
        ]
        if len(eda_link_keys) != len(set(eda_link_keys)):
            raise ValueError("hypothesis_to_eda_links contains duplicate links")

        action_map = {
            action.action_id: action for action in self.actions if action.action_id
        }
        if len(action_map) != len([action for action in self.actions if action.action_id]):
            raise ValueError("actions contains duplicate action_id values")
        expected_provenance = {
            action_id: ActionProvenance(
                action_id=action_id,
                source_refs=list(dict.fromkeys(action.source_refs)),
                hypothesis_ids=list(dict.fromkeys(action.hypothesis_ids)),
                motivating_hypothesis_ids=list(dict.fromkeys(action.motivating_hypothesis_ids)),
                safety_hypothesis_ids=list(dict.fromkeys(action.safety_hypothesis_ids)),
                validation_context_ids=list(dict.fromkeys(action.validation_context_ids)),
                eda_result_refs=list(dict.fromkeys(action.eda_result_refs)),
            )
            for action_id, action in action_map.items()
        }
        if not self.action_provenance or self.schema_version == "1.0":
            object.__setattr__(
                self,
                "action_provenance",
                list(expected_provenance.values()),
            )
        else:
            supplied = {item.action_id: item for item in self.action_provenance}
            if len(supplied) != len(self.action_provenance):
                raise ValueError("action_provenance contains duplicate action_id values")
            if supplied != expected_provenance:
                raise ValueError("action_provenance contradicts action-level provenance")

        for section in self.sections:
            if len(section.action_ids) != len(set(section.action_ids)):
                raise ValueError(
                    f"Section {section.section_id!r} contains duplicate action IDs"
                )
            unknown_actions = set(section.action_ids) - set(action_map)
            if unknown_actions:
                raise ValueError(
                    f"Section {section.section_id!r} references unknown actions: "
                    f"{sorted(unknown_actions)}"
                )
            if not section.action_ids:
                continue
            section_actions = [action_map[action_id] for action_id in section.action_ids]
            aggregates = {
                "evidence_refs": list(dict.fromkeys(
                    ref for action in section_actions for ref in action.evidence_refs
                )),
                "related_hypothesis_ids": list(dict.fromkeys(
                    ref for action in section_actions for ref in action.hypothesis_ids
                )),
                "source_refs": list(dict.fromkeys(
                    ref for action in section_actions for ref in action.source_refs
                )),
                "eda_result_refs": list(dict.fromkeys(
                    ref for action in section_actions for ref in action.eda_result_refs
                )),
            }
            for field_name, allowed_values in aggregates.items():
                current = list(getattr(section, field_name))
                if not current:
                    object.__setattr__(section, field_name, allowed_values)
                    continue
                unrelated = set(current) - set(allowed_values)
                if unrelated:
                    raise ValueError(
                        f"Section {section.section_id!r} {field_name} contains links "
                        f"not owned by its actions: {sorted(unrelated)}"
                    )


def upgrade_legacy_final_strategy_synthesis_status(
    payload: Mapping[str, Any],
    *,
    assumed_status: SynthesisStatus = "degraded_fallback",
) -> dict[str, Any]:
    """Upgrade pre-status artifacts before canonical validation.

    Missing status is treated conservatively as degraded unless a caller supplies
    authoritative provenance. No status is inferred from limitations text.
    """
    upgraded = deepcopy(dict(payload))
    if "synthesis_status" in upgraded:
        return upgraded
    state = {
        "llm_success": (True, False, False, False),
        "repaired_success": (False, True, True, False),
        "degraded_fallback": (False, False, False, True),
    }[assumed_status]
    upgraded.update({
        "synthesis_status": assumed_status,
        "llm_output_valid": state[0],
        "repair_attempted": state[1],
        "repair_succeeded": state[2],
        "fallback_used": state[3],
        "synthesis_diagnostics_path": None,
    })
    if assumed_status == "degraded_fallback":
        limitations = list(upgraded.get("limitations") or [])
        explanation = (
            "This legacy strategy predates authoritative synthesis-status metadata; "
            "it is treated conservatively as a degraded fallback."
        )
        if explanation not in limitations:
            limitations.append(explanation)
        upgraded["limitations"] = limitations
        canonical, _ = canonicalize_final_strategy_actions(upgraded)
        sections_by_id = {
            str(section.get("section_id") or ""): dict(section)
            for section in canonical.get("sections") or []
            if isinstance(section, Mapping) and section.get("section_id")
        }
        anchor_action_id = next(
            (
                str(action.get("action_id"))
                for action in canonical.get("actions") or []
                if isinstance(action, Mapping) and action.get("action_id")
            ),
            None,
        )
        ordered_sections: list[dict[str, Any]] = []
        for section_id in REQUIRED_SECTION_IDS:
            section = sections_by_id.get(section_id, {
                "section_id": section_id,
                "title": section_id.replace("_", " ").title(),
                "summary": (
                    "not_available: this section was absent from the legacy strategy artifact."
                ),
                "action_ids": [],
                "evidence_refs": ["final_synthesizer.repaired"],
                "availability": "not_available",
                "limitations": [
                    "The legacy artifact predates complete deterministic fallback sections."
                ],
            })
            if section_id == "executive_summary" and "degraded" not in str(
                section.get("summary") or ""
            ).lower():
                section["summary"] = (
                    "Degraded legacy fallback status. "
                    + str(section.get("summary") or "Legacy strategy summary unavailable.")
                )
            if section_id == "first_48_hours" and not section.get("time_blocks"):
                if not anchor_action_id:
                    raise ValueError(
                        "Cannot upgrade degraded legacy fallback without an action for first_48_hours references"
                    )
                section["time_blocks"] = [
                    {
                        "time_window": window,
                        "summary": "Continue the surviving legacy evidence-backed action.",
                        "action_ids": [anchor_action_id],
                        "experiment_ids": [],
                    }
                    for window in (
                        "0-4_hours", "4-12_hours", "12-24_hours", "24-48_hours",
                    )
                ]
            ordered_sections.append(section)
        canonical["sections"] = ordered_sections
        upgraded = canonical
    return upgraded


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


def upgrade_final_strategy_v1_to_v2(
    payload: Mapping[str, Any],
    *,
    evidence_pack: Mapping[str, Any],
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Read a v1 artifact and emit the compact v2 canonical representation.

    Legacy action-level evidence_bindings are accepted as input by the model but are
    intentionally excluded from v2 serialization; previews are rebuilt once in the
    top-level evidence_catalog.
    """

    from kaggle_researcher.reasoning.strategy_compaction import compact_final_strategy

    legacy = FinalStrategyResult.model_validate(payload)
    return compact_final_strategy(
        legacy,
        evidence_pack=evidence_pack,
        source_ids=source_ids or [],
    ).model_dump(mode="json")


__all__ = [
    "ActionProvenance", "Confidence", "EvidenceBinding", "EvidenceBindingRole",
    "EvidenceCatalogEntry", "EvidenceOrigin", "EvidenceRef", "ExperimentArm",
    "ExperimentBudget", "ExperimentCost", "ExperimentStatus", "FALLBACK_LIMITATION",
    "FamilyStatus", "FeatureActionMetadata", "FeatureExperimentFamily",
    "FinalStrategyExperiment", "FinalStrategyQualityMetrics", "FitScope", "LeakageRisk",
    "First48HourBlock", "First48HourWindow",
    "FinalStrategyAction", "FinalStrategyResult", "FinalStrategySection",
    "HypothesisToEdaLink",
    "FinalValidationMethod", "Priority", "REPAIR_LIMITATION", "REQUIRED_SECTION_IDS",
    "SourceToHypothesisLink",
    "SynthesisStatus",
    "SectionAvailability",
    "TEMPORAL_VALIDATION_METHODS",
    "migrate_legacy_final_strategy_payload",
    "upgrade_final_strategy_v1_to_v2",
    "upgrade_legacy_final_strategy_synthesis_status",
]
