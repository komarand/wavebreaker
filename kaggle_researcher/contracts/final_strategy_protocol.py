from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.final_strategy import (
    Confidence,
    ExperimentCost,
    FinalValidationMethod,
    FitScope,
    LeakageRisk,
    Priority,
)

SelectionStatus = Literal["llm_success", "repaired_success", "degraded_fallback", "failed"]
RenderingStatus = Literal["llm_success", "repaired_success", "deterministic_render", "failed"]


class PromptFingerprint(ContractModel):
    prompt_name: str
    prompt_version: str
    system_prompt_hash: str
    user_template_hash: str
    output_schema_version: str
    context_policy_version: str
    fingerprint: str


class ExperimentArmDraft(ContractModel):
    client_arm_key: str
    name: str
    exact_change: str
    generated_features: list[str] = Field(default_factory=list)
    fit_scope: FitScope
    leakage_risk: LeakageRisk
    dependencies: list[str] = Field(default_factory=list)


class SelectedActionDraft(ContractModel):
    client_action_key: str
    action_kind: str
    action: str
    priority: Priority
    confidence: Confidence
    reason: str
    primary_evidence_refs: list[str]
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    limitation_evidence_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    motivating_hypothesis_ids: list[str] = Field(default_factory=list)
    safety_hypothesis_ids: list[str] = Field(default_factory=list)
    validation_context_ids: list[str] = Field(default_factory=list)
    rejected_hypothesis_ids: list[str] = Field(default_factory=list)
    safety_constraint_ids: list[str] = Field(default_factory=list)
    validation_requirement_ids: list[str] = Field(default_factory=list)
    feature_metadata: dict[str, Any] | None = None
    dependencies: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_grounding(self) -> "SelectedActionDraft":
        if not self.primary_evidence_refs:
            raise ValueError("selected actions require a primary evidence ref")
        if not any((self.motivating_hypothesis_ids, self.safety_hypothesis_ids,
                    self.validation_context_ids, self.rejected_hypothesis_ids)):
            raise ValueError("selected actions require at least one hypothesis role")
        return self


class FeatureExperimentFamilyDraft(ContractModel):
    client_family_key: str
    name: str
    priority: Priority
    input_columns: list[str]
    hypothesis: str
    baseline_arm: ExperimentArmDraft
    candidate_arms: list[ExperimentArmDraft]
    validation_strategy: FinalValidationMethod
    metric_name: str
    primary_evidence_refs: list[str]
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    motivating_hypothesis_ids: list[str]
    safety_hypothesis_ids: list[str] = Field(default_factory=list)
    risks: list[str]
    acceptance_rule: str
    estimated_cost: ExperimentCost
    limitations: list[str] = Field(default_factory=list)


class CandidateExperimentDraft(ContractModel):
    model_config = {**ContractModel.model_config, "protected_namespaces": ()}

    client_experiment_key: str
    experiment_kind: str
    name: str
    priority: Priority
    confidence: Confidence
    hypothesis: str
    exact_change: str
    family_key: str | None = None
    model_family_id: str | None = None
    comparison_model_family_id: str | None = None
    validation_strategy: FinalValidationMethod
    metric_name: str
    acceptance_rule: str
    estimated_cost: ExperimentCost
    primary_evidence_refs: list[str]
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    motivating_hypothesis_ids: list[str]
    safety_hypothesis_ids: list[str] = Field(default_factory=list)
    validation_context_ids: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    risks: list[str]
    limitations: list[str] = Field(default_factory=list)


class SectionPlanDraft(ContractModel):
    section_id: str
    selected_action_keys: list[str] = Field(default_factory=list)
    selected_family_keys: list[str] = Field(default_factory=list)
    selected_experiment_keys: list[str] = Field(default_factory=list)
    summary_intent: str


class StrategySelectionDraft(ContractModel):
    schema_version: Literal["2.0"] = "2.0"
    selected_actions: list[SelectedActionDraft]
    feature_experiment_families: list[FeatureExperimentFamilyDraft] = Field(default_factory=list)
    candidate_experiments: list[CandidateExperimentDraft]
    proposed_core_experiment_ids: list[str]
    proposed_backlog_experiment_ids: list[str] = Field(default_factory=list)
    section_plan: list[SectionPlanDraft]
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_client_keys(self) -> "StrategySelectionDraft":
        groups = {
            "action": [item.client_action_key for item in self.selected_actions],
            "family": [item.client_family_key for item in self.feature_experiment_families],
            "experiment": [item.client_experiment_key for item in self.candidate_experiments],
        }
        for label, values in groups.items():
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} client keys")
        proposed = [*self.proposed_core_experiment_ids, *self.proposed_backlog_experiment_ids]
        if len(proposed) != len(set(proposed)):
            raise ValueError("core and backlog experiment proposals overlap or repeat")
        unknown = set(proposed) - set(groups["experiment"])
        if unknown:
            raise ValueError(f"proposed experiment keys are unknown: {sorted(unknown)}")
        return self


class StrategySkeleton(ContractModel):
    skeleton_schema_version: Literal["2.0"] = "2.0"
    skeleton_id: str
    skeleton_hash: str
    competition_id: str
    task_type: str
    metric: dict[str, Any]
    recommended_validation: FinalValidationMethod
    synthesis_selection_status: SelectionStatus
    evidence_catalog: dict[str, Any]
    source_to_hypothesis_links: list[dict[str, Any]]
    hypothesis_to_eda_links: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    action_provenance: list[dict[str, Any]]
    feature_experiment_families: list[dict[str, Any]]
    core_experiments: list[dict[str, Any]]
    experiment_backlog: list[dict[str, Any]]
    experiment_budget: dict[str, Any]
    dependency_graph: dict[str, list[str]]
    section_structure: list[dict[str, Any]]
    validation_requirement_ids: list[str]
    safety_constraint_ids: list[str]
    limitations: list[str]
    warnings: list[str]
    selection_prompt_fingerprint: PromptFingerprint
    client_key_map: dict[str, dict[str, str]] = Field(default_factory=dict)


class ActionWording(ContractModel):
    action_id: str
    display_action: str
    display_reason: str


class ExperimentWording(ContractModel):
    experiment_id: str
    display_name: str
    display_hypothesis: str
    display_exact_change: str
    display_acceptance_rule: str
    display_risk: str


class FamilyWording(ContractModel):
    family_id: str
    display_name: str
    display_hypothesis: str
    display_acceptance_rule: str
    display_risks: list[str]


class SectionSummaryWording(ContractModel):
    section_id: str
    summary: str


class StrategyRenderingDraft(ContractModel):
    schema_version: Literal["2.0"] = "2.0"
    skeleton_id: str
    skeleton_hash: str
    executive_summary: str
    action_wording: list[ActionWording]
    experiment_wording: list[ExperimentWording]
    family_wording: list[FamilyWording]
    section_summaries: list[SectionSummaryWording]
    limitation_wording: list[str] = Field(default_factory=list)
    uncertainty_summary: str


__all__ = [
    "ActionWording", "CandidateExperimentDraft", "ExperimentArmDraft",
    "ExperimentWording", "FamilyWording", "FeatureExperimentFamilyDraft",
    "PromptFingerprint", "RenderingStatus", "SectionPlanDraft",
    "SectionSummaryWording", "SelectedActionDraft", "SelectionStatus",
    "StrategyRenderingDraft", "StrategySelectionDraft", "StrategySkeleton",
]
