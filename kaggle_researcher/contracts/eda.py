from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import Field, model_validator

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.ids import (
    EdaTaskId,
    EvidenceId,
    HypothesisId,
    RiskId,
    SafetyConstraintId,
    ValidationRequirementId,
)
from kaggle_researcher.contracts.research import Priority
from kaggle_researcher.contracts.versions import CURRENT_SCHEMA_VERSION


EdaHypothesisStatus = Literal["confirmed", "partially_confirmed", "rejected", "not_testable", "skipped"]
RiskSeverity = Literal["info", "low", "medium", "high", "critical"]
RiskStatus = Literal["confirmed", "suspected", "mitigated_by_policy", "not_testable", "skipped", "resolved", "informational"]
RiskType = Literal[
    "schema", "metric", "validation", "leakage", "drift", "missingness",
    "high_cardinality", "target", "baseline", "relationship", "data_quality",
    "submission", "feature_engineering", "leaderboard", "notebook_source", "unsupported",
]


class EdaTask(ContractModel):
    task_id: EdaTaskId
    module: str = Field(min_length=1)
    priority: Priority
    blocking: bool = False
    related_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    dependencies: list[EdaTaskId] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class HypothesisIndexEntry(ContractModel):
    hypothesis_id: HypothesisId
    task_ids: list[EdaTaskId] = Field(default_factory=list)


class EdaTaskPlan(ContractModel):
    contract_family: Literal["eda_task_plan"] = "eda_task_plan"
    schema_version: Literal["1.0"] = CURRENT_SCHEMA_VERSION
    competition_id: str = Field(min_length=1)
    task_type: str | None = None
    metric: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    eda_tasks: list[EdaTask] = Field(default_factory=list)
    hypothesis_index: dict[HypothesisId, list[EdaTaskId]] = Field(default_factory=dict)
    recommended_module_sequence: list[str] = Field(default_factory=list)
    recommended_human_checklist: list[str] = Field(default_factory=list)
    blocking_tasks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_task_graph(self) -> "EdaTaskPlan":
        task_ids = [task.task_id for task in self.eda_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("eda_tasks contains duplicate task_id values")
        known = set(task_ids)
        graph: dict[EdaTaskId, list[EdaTaskId]] = defaultdict(list)
        for task in self.eda_tasks:
            if task.task_id in task.dependencies:
                raise ValueError(f"Task {task.task_id!r} depends on itself")
            unknown = set(task.dependencies) - known
            if unknown:
                raise ValueError(f"Task {task.task_id!r} has unknown dependencies: {sorted(unknown)}")
            graph[task.task_id].extend(task.dependencies)
        _assert_acyclic(graph)
        for hypothesis_id, indexed_tasks in self.hypothesis_index.items():
            unknown = set(indexed_tasks) - known
            if unknown:
                raise ValueError(
                    f"hypothesis_index.{hypothesis_id} references unknown tasks: {sorted(unknown)}"
                )
        return self


def _assert_acyclic(graph: dict[EdaTaskId, list[EdaTaskId]]) -> None:
    visiting: set[EdaTaskId] = set()
    visited: set[EdaTaskId] = set()

    def visit(node: EdaTaskId) -> None:
        if node in visiting:
            raise ValueError(f"EDA task dependency cycle contains {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph.get(node, []):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for task_id in graph:
        visit(task_id)


class HypothesisResult(ContractModel):
    hypothesis_id: HypothesisId
    category: str
    status: EdaHypothesisStatus
    confidence_after_eda: Literal["low", "medium", "high"]
    finding: str
    evidence_refs: list[EvidenceId] = Field(default_factory=list)
    impact_on_strategy: str
    limitations: list[str] = Field(default_factory=list)


class RecommendedNextAction(ContractModel):
    priority: Priority
    action: str
    why: str
    evidence_refs: list[EvidenceId] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"] | None = None
    applies_to: list[str] = Field(default_factory=list)
    source_categories: list[str] = Field(default_factory=list)


class EdaRisk(ContractModel):
    risk_id: RiskId
    risk_intent: str = ""
    risk_type: RiskType
    severity: RiskSeverity
    status: RiskStatus
    confidence: Literal["low", "medium", "high"]
    title: str
    finding: str
    impact: str
    mitigation: str | None = None
    applies_to: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceId] = Field(default_factory=list)
    related_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class SafetyConstraint(ContractModel):
    safety_constraint_id: SafetyConstraintId
    scope: str
    rule: str
    severity: Literal["advisory", "mandatory", "blocking"] = "mandatory"
    blocking: bool = True
    evidence_refs: list[EvidenceId] = Field(default_factory=list)
    reason: str
    evidence_origin: str = "dataset_measurement"


class ValidationRequirement(ContractModel):
    validation_requirement_id: ValidationRequirementId
    rule: str
    status: Literal["recommended", "required", "conditional"] = "required"
    mandatory: bool = True
    condition: str | None = None
    evidence_refs: list[EvidenceId] = Field(default_factory=list)
    reason: str
    evidence_origin: str = "statistical_diagnostic"


class EdaTestableHypothesis(ContractModel):
    hypothesis_id: HypothesisId
    scope: str
    statement: str
    trigger_finding: str
    why_unresolved: str
    evidence_refs: list[EvidenceId] = Field(default_factory=list)
    baseline_ref: str | None = None
    required_controls: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    priority_signal: Literal["important", "optional"]
    reliability: Literal["low", "medium", "high"]
    status: Literal["untested"] = "untested"
    evidence_origin: str = "reasoning_inference"


class EdaEvidencePack(ContractModel):
    contract_family: Literal["eda_evidence_pack"] = "eda_evidence_pack"
    schema_version: Literal["1.0"] = "1.0"
    competition_id: str
    created_at: str
    run_id: str
    dataset: dict[str, Any] = Field(default_factory=dict)
    file_inventory: dict[str, Any] = Field(default_factory=dict)
    inferred_schema: dict[str, Any] = Field(default_factory=dict)
    table_profiles: list[dict[str, Any]] = Field(default_factory=list)
    metric_evidence: dict[str, Any] = Field(default_factory=dict)
    validation_evidence: dict[str, Any] = Field(default_factory=dict)
    leakage_evidence: list[dict[str, Any]] = Field(default_factory=list)
    relationship_evidence: dict[str, Any] = Field(default_factory=dict)
    drift_evidence: dict[str, Any] = Field(default_factory=dict)
    baseline_evidence: dict[str, Any] = Field(default_factory=dict)
    baseline_ablation_evidence: dict[str, Any] = Field(default_factory=dict)
    feature_probe_evidence: list[dict[str, Any]] = Field(default_factory=list)
    feature_diagnostics: dict[str, Any] = Field(default_factory=dict)
    target_diagnostics: dict[str, Any] = Field(default_factory=dict)
    interaction_diagnostics: dict[str, Any] = Field(default_factory=dict)
    source_claim_validation: dict[str, Any] = Field(default_factory=dict)
    visual_diagnostics: dict[str, Any] = Field(default_factory=dict)
    slice_diagnostics: dict[str, Any] = Field(default_factory=dict)
    eda_risks: list[EdaRisk] = Field(default_factory=list)
    eda_implications: list[dict[str, Any]] = Field(default_factory=list)
    strategy_hints: list[dict[str, Any]] = Field(default_factory=list)
    safety_constraints: list[SafetyConstraint] = Field(default_factory=list)
    validation_requirements: list[ValidationRequirement] = Field(default_factory=list)
    testable_hypotheses: list[EdaTestableHypothesis] = Field(default_factory=list)
    experiment_candidates: list[dict[str, Any]] = Field(default_factory=list, json_schema_extra={"deprecated": True})
    module_classification: dict[str, str] = Field(default_factory=dict)
    stage_status: dict[str, str] = Field(default_factory=dict)
    evidence_origins: dict[str, str] = Field(default_factory=dict)
    deprecated_outputs: dict[str, dict[str, str]] = Field(default_factory=dict)
    baseline_purpose: str = "diagnostic_sanity_floor"
    ablation_purpose: str = "broad_feature_block_diagnostics"
    visual_diagnostics_role: str = "evidence_rendering"
    risk_scope: str = "eda_local"
    owner: str = "eda_engine"
    eligible_for_global_risk_synthesis: bool = True
    eda_risk_register: list[EdaRisk] = Field(default_factory=list, json_schema_extra={"deprecated": True})
    risk_summary: dict[str, Any] = Field(default_factory=dict)
    eda_strategy_hints: dict[str, list[dict[str, Any]]] = Field(default_factory=dict, json_schema_extra={"deprecated": True})
    notebook_static_analysis: dict[str, Any] = Field(default_factory=dict)
    hypothesis_results: list[HypothesisResult] = Field(default_factory=list)
    recommended_next_actions: list[RecommendedNextAction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)

__all__ = [
    "EdaEvidencePack", "EdaHypothesisStatus", "EdaRisk", "EdaTask", "EdaTaskPlan",
    "EdaTestableHypothesis", "HypothesisIndexEntry", "HypothesisResult", "RecommendedNextAction",
    "RiskSeverity", "RiskStatus", "RiskType", "SafetyConstraint",
    "ValidationRequirement",
]
