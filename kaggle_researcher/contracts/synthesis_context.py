from __future__ import annotations

from typing import Any, Iterable

from pydantic import Field

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.ids import (
    EvidenceId,
    ExperimentId,
    HypothesisId,
    RiskId,
    SafetyConstraintId,
    ValidationRequirementId,
)
from kaggle_researcher.contracts.registries import ContractRegistries
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.review import SkepticalReview
from kaggle_researcher.contracts.validation import (
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ValidationResult,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument


class ValidationArchitectContext(ContractModel):
    plan_data: PlanData
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    allowed_evidence_ids: list[EvidenceId] = Field(default_factory=list)


class ExperimentPlanningContext(ContractModel):
    validation: ValidationResult
    leakage: LeakageRiskResult
    metric: MetricResult
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    allowed_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    allowed_evidence_ids: list[EvidenceId] = Field(default_factory=list)
    safety_constraints: list[dict[str, Any]] = Field(default_factory=list)
    validation_requirements: list[dict[str, Any]] = Field(default_factory=list)
    baseline_summary: dict[str, Any] = Field(default_factory=dict)


class SkepticalReviewContext(ContractModel):
    experiments: ExperimentPlan
    allowed_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    allowed_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    allowed_evidence_refs: list[EvidenceId] = Field(default_factory=list)


class FinalSynthesisContext(ContractModel):
    competition_desc: str
    plan_data: PlanData
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    domain_patterns: list[dict[str, Any]] = Field(default_factory=list)
    research_hypotheses: ResearchHypotheses
    eda_evidence_pack: EdaEvidencePack
    eda_summary_text: str | None = None
    metric: MetricResult
    validation: ValidationResult
    leakage: LeakageRiskResult
    leaderboard: LeaderboardAuditResult
    experiment_plan: ExperimentPlan = Field(default_factory=ExperimentPlan)
    review: SkepticalReview | None = None
    approved_experiments: list[dict[str, Any]] = Field(default_factory=list)
    rejected_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    unresolved_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    allowed_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    allowed_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    allowed_evidence_refs: list[EvidenceId] = Field(default_factory=list)
    allowed_eda_result_refs: list[EvidenceId] = Field(default_factory=list)
    allowed_risk_ids: list[RiskId] = Field(default_factory=list)
    allowed_validation_requirement_ids: list[ValidationRequirementId] = Field(default_factory=list)
    allowed_safety_constraint_ids: list[SafetyConstraintId] = Field(default_factory=list)
    optional_stage_failure_messages: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def reference_prompt_payload(self) -> dict[str, Any]:
        return {
            "approved_experiments": self.approved_experiments,
            "rejected_experiment_ids": self.rejected_experiment_ids,
            "unresolved_hypotheses": self.unresolved_hypotheses,
            "allowed_experiment_ids": self.allowed_experiment_ids,
            "allowed_hypothesis_ids": self.allowed_hypothesis_ids,
            "allowed_evidence_refs": self.allowed_evidence_refs,
            "allowed_eda_result_refs": self.allowed_eda_result_refs,
            "allowed_risk_ids": self.allowed_risk_ids,
            "allowed_validation_requirement_ids": self.allowed_validation_requirement_ids,
            "allowed_safety_constraint_ids": self.allowed_safety_constraint_ids,
            "safety_constraints": [
                item.model_dump(mode="json") for item in self.eda_evidence_pack.safety_constraints
            ],
            "validation_requirements": [
                item.model_dump(mode="json") for item in self.eda_evidence_pack.validation_requirements
            ],
            "risks": [item.model_dump(mode="json") for item in self.eda_evidence_pack.eda_risks],
            "optional_stage_failure_messages": self.optional_stage_failure_messages,
            "limitations": self.limitations,
        }


def build_final_synthesis_context(
    *,
    competition_desc: str,
    research: Any,
    eda: Any,
    reasoning: Any,
    registries: ContractRegistries,
    eda_summary_text: str | None,
    optional_stage_failures: Iterable[Any] = (),
) -> FinalSynthesisContext:
    """Prepare the exact, validated Final Synthesizer input from stage results."""

    from kaggle_researcher.contracts.experiments import approved_experiment_summary

    evidence_ids = sorted({
        *registries.evidence.ids("eda_evidence"),
        *registries.evidence.ids("source_claim"),
        *registries.evidence.ids("synthetic_inference"),
    })
    failures = [getattr(value, "message", str(value)) for value in optional_stage_failures]
    return FinalSynthesisContext(
        competition_desc=competition_desc,
        plan_data=research.plan_data,
        retrieved_documents=list(research.retrieved_documents),
        domain_patterns=list(research.domain_patterns),
        research_hypotheses=research.hypotheses,
        eda_evidence_pack=eda.evidence_pack,
        eda_summary_text=eda_summary_text,
        metric=reasoning.metric,
        validation=reasoning.validation,
        leakage=reasoning.leakage,
        leaderboard=reasoning.leaderboard,
        experiment_plan=reasoning.experiments or ExperimentPlan(),
        review=reasoning.review,
        approved_experiments=approved_experiment_summary(registries.experiments),
        rejected_experiment_ids=sorted(registries.experiments.rejected_ids),
        unresolved_hypotheses=[
            item.model_dump(mode="json")
            for item in [
                *research.hypotheses.hypotheses,
                *eda.evidence_pack.testable_hypotheses,
            ]
        ],
        allowed_experiment_ids=sorted(registries.experiments.approved_ids),
        allowed_hypothesis_ids=sorted(registries.hypotheses.by_id),
        allowed_evidence_refs=evidence_ids,
        allowed_eda_result_refs=sorted(registries.evidence.ids("eda_evidence")),
        allowed_risk_ids=sorted(registries.risks.by_id),
        allowed_validation_requirement_ids=sorted(registries.validation_requirements.by_id),
        allowed_safety_constraint_ids=sorted(registries.safety_constraints.by_id),
        optional_stage_failure_messages=failures,
        limitations=[
            *research.hypotheses.scout_limitations,
            *eda.evidence_pack.limitations,
            *failures,
        ],
    )


__all__ = [
    "ExperimentPlanningContext", "FinalSynthesisContext",
    "SkepticalReviewContext", "ValidationArchitectContext",
    "build_final_synthesis_context",
]
